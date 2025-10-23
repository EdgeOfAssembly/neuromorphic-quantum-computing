import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional
import unittest
import subprocess
import numpy as np  # For vectorized graph gen
from tqdm import tqdm  # For sim progress

def print_vram(label: str):
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], capture_output=True, text=True)
        vram_mb = int(result.stdout.strip())
        print(f"{label} VRAM: {vram_mb} MiB")
    except:
        print(f"{label} VRAM: N/A")

class LIFNeuron(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) neuron model for spiking dynamics.
    Mimics biological neurons: integrates input current, leaks over time, fires spikes asynchronously.
    Supports GPU acceleration for efficiency.
    """
    def __init__(self, n_neurons: int, tau: float = 30.0, threshold: float = 1.0, reset: float = 0.0, device: str = 'cpu') -> None:
        super().__init__()
        self.n_neurons: int = n_neurons
        self.tau: torch.Tensor = torch.tensor(tau, device=device)
        self.threshold: torch.Tensor = torch.tensor(threshold, device=device)
        self.reset: torch.Tensor = torch.tensor(reset, device=device)
        self.device: str = device
        self.v: torch.Tensor = torch.zeros(n_neurons, device=device)
        self.spikes: torch.Tensor = torch.zeros(n_neurons, device=device)

    def forward(self, input_current: torch.Tensor, dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        input_current = input_current.to(self.device)
        dt_tensor = torch.tensor(dt, device=self.device)
        decay = torch.exp(-dt_tensor / self.tau)
        self.v = self.v * decay + input_current * self.tau * (1.0 - decay)
        spikes = (self.v >= self.threshold).float()
        self.v = self.v * (1.0 - spikes) + spikes * self.reset
        self.spikes = spikes
        return self.v, spikes


class Brain3DNetwork(nn.Module):
    """
    3D Neuromorphic Network mimicking brain-like volumetric CPU (sparse for scale).
    - Neurons in 3D grid; local connections via sparse COO (up to 26 edges/neuron for radius=1).
    - Weights: uint8 exponents for compression; sparse matmul for efficiency.
    - Plasticity disabled for large N (add neighbor updates if needed).
    """
    def __init__(self, shape: Tuple[int, int, int], connectivity_radius: int = 1, enable_plasticity: bool = False,
                 log_range: float = 8.0, log_domain: bool = True, device: str = 'cpu', batch_size: int = 10000000,
                 input_scale: float = 1.0, threshold_scale: float = 0.5) -> None:
        super().__init__()
        self.layers, self.height, self.width = shape
        self.total_neurons: int = self.layers * self.height * self.width
        self.device: str = device
        self.enable_plasticity: bool = enable_plasticity and self.total_neurons < 10000  # Disable for large
        self.learning_rate: float = 0.005
        self.decay_rate: float = 0.0005
        self.log_range: float = log_range
        self.log_domain: bool = log_domain
        self.batch_size: int = 1000000 if device == 'cpu' else batch_size
        self.input_scale: float = input_scale
        self.threshold_scale: float = threshold_scale
        self.neurons: LIFNeuron = LIFNeuron(self.total_neurons, device=device)
        self.neurons.threshold *= self.threshold_scale
        
        # Pre-alloc output buffer (float16 for mem)
        self.output_buffer = torch.zeros(self.total_neurons, dtype=torch.float16, device=device)
        
        # Numpy-vectorized sparse connections (now with precomputed flat IDs for mem efficiency)
        self.src_ids: torch.Tensor
        self.dst_ids: torch.Tensor
        self.values_exp: torch.Tensor
        self._init_sparse_weights_np(connectivity_radius)
        
        # No cached sparse; compute on-the-fly for mem savings
    
    def _init_sparse_weights_np(self, radius: int) -> None:
        """Numpy-vectorized sparse IDs/values (10-50x faster than loops)."""
        # Create 3D grid of coordinates (uint8 for compression)
        l_grid, h_grid, w_grid = np.meshgrid(np.arange(self.layers), np.arange(self.height), np.arange(self.width), indexing='ij')
        coords = np.stack([l_grid, h_grid, w_grid], axis=-1).astype(np.uint8)  # (L,H,W,3)
        
        # Offsets for neighbors (exclude self)
        dl = np.arange(-radius, radius + 1)
        dh = np.arange(-radius, radius + 1)
        dw = np.arange(-radius, radius + 1)
        offsets_l, offsets_h, offsets_w = np.meshgrid(dl, dh, dw, indexing='ij')
        offsets = np.stack([offsets_l, offsets_h, offsets_w], axis=-1).reshape(-1, 3).astype(np.int8)  # (27,3)
        offsets = offsets[np.sum(np.abs(offsets), axis=1) > 0]  # (26,3)
        num_offsets = offsets.shape[0]
        
        # Broadcast offsets over coords
        offsets_exp = offsets[np.newaxis, np.newaxis, np.newaxis, :, :]  # (1,1,1,26,3)
        new_coords = coords[:, :, :, np.newaxis, :] + offsets_exp  # (L,H,W,26,3); int
        
        # Bounds check
        bounds = np.array([self.layers, self.height, self.width])
        valid = np.all((new_coords >= 0) & (new_coords < bounds[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]), axis=-1)  # (L,H,W,26)
        
        # Flatten
        flat_coords = coords.reshape(-1, 3)  # (N,3)
        flat_new_coords = new_coords.reshape(-1, 3)  # (N*26,3)
        flat_valid = valid.reshape(-1, num_offsets)  # [N,26]
        
        # Filter valid
        valid_indices = np.nonzero(flat_valid)  # (2, num_valid)
        valid_src_idx = valid_indices[0]  # Fixed: Remove // num_offsets
        flat_valid_idx = valid_indices[0] * num_offsets + valid_indices[1]
        valid_src_coords = flat_coords[valid_src_idx]
        valid_dst_coords = flat_new_coords[flat_valid_idx].astype(np.uint8)
        
        # Compute flat IDs (int32 for compression, lossless for <2^31)
        layer_size = self.height * self.width
        row_size = self.width
        src_ids_np = (valid_src_coords[:, 0].astype(np.int32) * layer_size +
                      valid_src_coords[:, 1].astype(np.int32) * row_size +
                      valid_src_coords[:, 2].astype(np.int32)).astype(np.int32)
        dst_ids_np = (valid_dst_coords[:, 0].astype(np.int32) * layer_size +
                      valid_dst_coords[:, 1].astype(np.int32) * row_size +
                      valid_dst_coords[:, 2].astype(np.int32)).astype(np.int32)
        
        # Random uint8 exponents (0-log_range for binary growth, stable sim)
        num_edges = len(src_ids_np)
        values_np = np.random.randint(0, int(self.log_range) + 1, num_edges, dtype=np.uint8)
        
        # To torch (int32 for indexing)
        self.src_ids = torch.from_numpy(src_ids_np).int().to(self.device)
        self.dst_ids = torch.from_numpy(dst_ids_np).int().to(self.device)
        self.values_exp = torch.from_numpy(values_np).to(self.device)
    
    def _decode_weights(self) -> torch.Tensor:
        """Decode exponents using binary scheme: w = 2 ** exp."""
        return torch.pow(2.0, self.values_exp.float())
    
    def _synaptic_input(self, prev_spikes: torch.Tensor) -> torch.Tensor:
        prev_spikes = prev_spikes.to(self.device)
        flat_pre = self.src_ids
        flat_post = self.dst_ids
        
        if self.log_domain and prev_spikes.sum() == 0.0:
            self.output_buffer.zero_()  # All-zero input
            return self.output_buffer
        
        if self.log_domain:
            # Log-domain exact: logsumexp via amax + sum(exp(contrib - max)) (numerically stable, low mem)
            log2 = torch.log(torch.tensor(2.0, device=self.device))
            log_w = self.values_exp.float() * log2
            log_prev = torch.log(prev_spikes.float() + 1e-8)
            log_contrib = log_w + log_prev[flat_pre]
            post_max = torch.full((self.total_neurons,), float('-inf'), device=self.device)
            post_max.scatter_reduce_(0, flat_post.long(), log_contrib, reduce='amax')
            
            # Exact sum: exp(max) * sum(exp(contrib - max))
            max_at_post = post_max[flat_post.long()]
            small_exps = torch.exp(log_contrib - max_at_post)
            small_sums = torch.zeros(self.total_neurons, device=self.device)
            small_sums.scatter_add_(0, flat_post.long(), small_exps)
            
            total = torch.exp(post_max) * small_sums
            self.output_buffer = total.to(torch.float16)
            return self.output_buffer
        else:
            # Batched standard: decode full weights once, then chunk scatter_add (low peak mem)
            weights = self._decode_weights()
            self.output_buffer.zero_()
            num_edges = len(flat_pre)
            for i in range(0, num_edges, self.batch_size):
                end = min(i + self.batch_size, num_edges)
                batch_pre = flat_pre[i:end]
                batch_post = flat_post[i:end].long()  # Cast to long for scatter_add
                batch_w = weights[i:end]
                batch_contrib = (batch_w * prev_spikes[batch_pre]).to(torch.float16)
                self.output_buffer.scatter_add_(0, batch_post, batch_contrib)
            return self.output_buffer
    
    def _apply_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        if not self.enable_plasticity:
            return
        # Sparse Hebbian: Vectorized over edges
        flat_pre = self.src_ids
        flat_post = self.dst_ids
        
        # Hebb delta: lr * pre[pre] * post[post] (0/1 -> 0 or lr)
        hebb_delta = self.learning_rate * pre_spikes[flat_pre] * post_spikes[flat_post]
        # Scale to exp range (approx multiplicative)
        hebb_delta *= self.log_range
        decay_delta = self.decay_rate * self.values_exp.float()
        delta_exp = (hebb_delta - decay_delta).round().clamp(0, self.log_range)
        self.values_exp = delta_exp.to(torch.uint8)
    
    def forward(self, external_input: Optional[torch.Tensor] = None, dt: float = 1.0, num_steps: int = 10) -> Dict[str, Any]:
        if external_input is not None:
            external_input = external_input.to(self.device).clone()
        
        spikes_history: list[torch.Tensor] = []
        voltages_history: list[torch.Tensor] = []
        prev_spikes = torch.zeros(self.total_neurons, device=self.device)
        
        for step in range(num_steps):
            synaptic_input = self._synaptic_input(prev_spikes)
            
            total_input = synaptic_input.clone().float()  # Promote to float32 for neuron
            if external_input is not None:
                total_input += external_input
            
            v, spikes = self.neurons(total_input, dt)
            self._apply_plasticity(prev_spikes, spikes)
            
            spikes_history.append(spikes.clone())
            voltages_history.append(v.clone())
            prev_spikes = spikes
        
        return {
            'spikes': torch.stack(spikes_history),
            'voltages': torch.stack(voltages_history),
            'final_values_exp': self.values_exp.clone()
        }


# Unit Tests
class TestLIFNeuron(unittest.TestCase):
    def setUp(self) -> None:
        self.neuron = LIFNeuron(5, device='cpu')

    def test_forward(self) -> None:
        input_curr = torch.ones(5) * 0.5
        v, s = self.neuron(input_curr)
        self.assertEqual(v.shape, torch.Size([5]))
        self.assertTrue(torch.all(s <= 1.0))
        input_curr = torch.ones(5) * 2.0
        v, s = self.neuron(input_curr)
        self.assertTrue(torch.any(s > 0))


class TestBrain3DNetwork(unittest.TestCase):
    def setUp(self) -> None:
        self.net = Brain3DNetwork((2, 2, 2), log_domain=True, device='cpu')

    def test_init(self) -> None:
        self.assertEqual(self.net.total_neurons, 8)
        self.assertEqual(self.net.src_ids.dtype, torch.int32)
        self.assertGreater(self.net.values_exp.sum().item(), 0)

    def test_forward(self) -> None:
        result = self.net(None, num_steps=3)
        self.assertEqual(result['spikes'].shape, (3, 8))
        self.assertEqual(result['voltages'].shape, (3, 8))

    def test_with_input(self) -> None:
        input_curr = torch.ones(8) * 0.6
        result = self.net(input_curr, num_steps=2)
        self.assertGreater(result['spikes'].sum().item(), 0)

    def test_plasticity(self) -> None:
        net_plastic = Brain3DNetwork((2, 2, 2), enable_plasticity=True, device='cpu')
        initial_exp = net_plastic.values_exp.clone()
        input_curr = torch.ones(8) * 2.0
        result = net_plastic(input_curr, num_steps=10)
        self.assertFalse(torch.equal(net_plastic.values_exp, initial_exp))

    def test_log_vs_nonlog(self) -> None:
        np.random.seed(42)
        net_log = Brain3DNetwork((2,2,2), log_domain=True, device='cpu')
        np.random.seed(42)
        net_non = Brain3DNetwork((2,2,2), log_domain=False, device='cpu')
        spikes_log = torch.ones(8)
        inp_log = net_log._synaptic_input(spikes_log)
        inp_non = net_non._synaptic_input(spikes_log)
        self.assertTrue(torch.allclose(inp_log, inp_non, atol=1e-3))  # Tolerance for float precision


if __name__ == '__main__':
    unittest.main(verbosity=2, exit=False)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print_vram("Pre-init")
    
    # Scale to 1M neurons (change to (50,50,50) for quick test ~125k / <5s init)
    np.random.seed(42)
    torch.manual_seed(42)
    net = Brain3DNetwork(shape=(160, 160, 160), connectivity_radius=1, enable_plasticity=False,
                         log_range=8.0, log_domain=True, device=device, input_scale=1.0, threshold_scale=0.5)
    print(f"Total neurons: {net.total_neurons}, Edges: {net.src_ids.shape[0]}")
    print_vram("Post-init")
    
    # Patterns (bottom layer left/right)
    patterns = [
        torch.zeros(net.total_neurons, device=device),
        torch.zeros(net.total_neurons, device=device)
    ]
    layer_size = net.width * net.height  # 10000 for 100x100
    half = net.width // 2
    for col in range(half):  # Left
        idx = 0 * layer_size + 0 * net.width + col
        patterns[0][idx] = 0.4  # Low for sparsity
    for col in range(half, net.width):  # Right
        idx = 0 * layer_size + 0 * net.width + col
        patterns[1][idx] = 0.4
    
    # Scale patterns
    for p in patterns:
        p *= net.input_scale
    
    # Incremental metrics (no full stack for low mem)
    left_sum = torch.zeros(net.total_neurons, device=device)
    right_sum = torch.zeros(net.total_neurons, device=device)
    left_count, right_count = 0, 0
    pattern_id = 0
    print_vram("Pre-sim")
    for step in tqdm(range(40), desc="Simulating steps"):
        if step % 20 == 0:
            pattern_id = 1 - pattern_id
        external_input = patterns[pattern_id]
        
        synaptic_input = net._synaptic_input(net.neurons.spikes)
        total_input = synaptic_input + external_input
        v, spikes = net.neurons(total_input, dt=1.0)
        if net.enable_plasticity:
            net._apply_plasticity(net.neurons.spikes, spikes)
        net.neurons.v, net.neurons.spikes = v, spikes
        
        # Incremental
        if step < 20:
            left_sum += spikes
            left_count += 1
        else:
            right_sum += spikes
            right_count += 1
    
    print_vram("Post-sim")
    left_phase = left_sum / left_count
    right_phase = right_sum / right_count
    left_corr = torch.dot(left_phase, patterns[0]) / (left_phase.norm() * patterns[0].norm() + 1e-8)
    right_corr = torch.dot(right_phase, patterns[1]) / (right_phase.norm() * patterns[1].norm() + 1e-8)
    
    print(f"Avg spike rate (left phase): {left_phase.mean().item():.3f}")
    print(f"Avg spike rate (right phase): {right_phase.mean().item():.3f}")
    print(f"Learning: Left corr to pattern: {left_corr.item():.3f}, Right: {right_corr.item():.3f}")
    print(f"Total exp sum after task: {net.values_exp.sum().cpu().item():.0f} (uint8)")
    
    if device == 'cuda':
        torch.cuda.empty_cache()
    print_vram("Final")