import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
from typing import Tuple, Dict, Any, Optional, List
import subprocess
import numpy as np
from tqdm import tqdm
import argparse


def print_vram(label: str) -> None:
    """Print current GPU VRAM usage."""
    if not torch.cuda.is_available():
        print(f"{label} VRAM: N/A (CUDA not available)")
        return
    
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=5
        )
        vram_mb = int(result.stdout.strip())
        print(f"{label} VRAM: {vram_mb} MiB")
    except FileNotFoundError:
        print(f"{label} VRAM: N/A (nvidia-smi not found)")
    except subprocess.TimeoutExpired:
        print(f"{label} VRAM: N/A (nvidia-smi timeout)")
    except (subprocess.CalledProcessError, ValueError) as e:
        print(f"{label} VRAM: N/A (error)")


class LIFNeuron(nn.Module):
    """Leaky Integrate-and-Fire neuron with stateful voltage and spikes."""
    
    def __init__(self, n_neurons: int, tau: float = 30.0, threshold: float = 1.0,
                 reset: float = 0.0, device: str = 'cpu') -> None:
        super().__init__()
        self.n_neurons: int = n_neurons
        self.tau: torch.Tensor = torch.tensor(tau, device=device)
        self.threshold: torch.Tensor = torch.tensor(threshold, device=device)
        self.reset: torch.Tensor = torch.tensor(reset, device=device)
        self.device: str = device
        self.v: torch.Tensor = torch.zeros(n_neurons, device=device)
        self.spikes: torch.Tensor = torch.zeros(n_neurons, device=device)

    def forward(self, input_current: torch.Tensor, dt: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Integrate current, emit spikes, and update state."""
        input_current = input_current.to(self.device)
        dt_tensor = torch.tensor(dt, device=self.device)
        decay = torch.exp(-dt_tensor / self.tau)
        self.v = self.v * decay + input_current * self.tau * (1.0 - decay)  # FIXED: inputinput → input_current
        spikes = (self.v >= self.threshold).float()
        self.v = self.v * (1.0 - spikes) + spikes * self.reset
        self.spikes = spikes
        return self.v, spikes


class Brain3DNetwork(nn.Module):
    """3D sparse neuromorphic network with sampled Hebbian plasticity."""
    
    def __init__(self, shape: Tuple[int, int, int], connectivity_radius: int = 1,
                 enable_plasticity: bool = True, log_range: float = 8.0, log_domain: bool = True,
                 device: str = 'cpu', batch_size: int = 10000000, input_scale: float = 1.0,
                 threshold_scale: float = 0.1, use_strong_input: bool = False) -> None:
        super().__init__()
        self.layers, self.height, self.width = shape
        self.total_neurons: int = self.layers * self.height * self.width
        self.device: str = device
        self.enable_plasticity: bool = enable_plasticity
        self.use_strong_input: bool = use_strong_input
        self.log_range: float = log_range
        self.log_domain: bool = log_domain
        self.batch_size: int = 1000000 if device == 'cpu' else batch_size
        self.input_scale: float = input_scale
        self.threshold_scale: float = threshold_scale
        self.neurons: LIFNeuron = LIFNeuron(self.total_neurons, device=device)
        self.neurons.threshold *= self.threshold_scale

        # Auto-set plasticity sampling
        self.plasticity_sample_ratio: float = (
            0.10 if enable_plasticity and use_strong_input else
            0.01 if enable_plasticity else
            0.0
        )
        self.plasticity_lr: float = 0.1  # INCREASED for observable learning
        self.plasticity_decay: float = 0.0  # DISABLED decay for clearer strengthening

        self.output_buffer: torch.Tensor = torch.zeros(self.total_neurons, dtype=torch.float16, device=device)

        self.src_ids: torch.Tensor
        self.dst_ids: torch.Tensor
        self.values_exp: torch.Tensor
        self._init_sparse_weights_np(connectivity_radius)

    def _init_sparse_weights_np(self, radius: int) -> None:
        """Initialize 3D local connectivity using NumPy vectorization."""
        l_grid, h_grid, w_grid = np.meshgrid(
            np.arange(self.layers),
            np.arange(self.height),
            np.arange(self.width),
            indexing='ij'
        )
        coords: np.ndarray = np.stack([l_grid, h_grid, w_grid], axis=-1).astype(np.uint8)

        dl = np.arange(-radius, radius + 1)
        dh = np.arange(-radius, radius + 1)
        dw = np.arange(-radius, radius + 1)
        offsets_l, offsets_h, offsets_w = np.meshgrid(dl, dh, dw, indexing='ij')
        offsets: np.ndarray = np.stack([offsets_l, offsets_h, offsets_w], axis=-1).reshape(-1, 3).astype(np.int8)
        offsets = offsets[np.sum(np.abs(offsets), axis=1) > 0]
        num_offsets: int = offsets.shape[0]

        offsets_exp = offsets[np.newaxis, np.newaxis, np.newaxis, :, :]
        new_coords = coords[:, :, :, np.newaxis, :] + offsets_exp

        bounds = np.array([self.layers, self.height, self.width])
        valid = np.all((new_coords >= 0) & (new_coords < bounds[np.newaxis, np.newaxis, np.newaxis, np.newaxis, :]), axis=-1)

        flat_coords = coords.reshape(-1, 3)
        flat_new_coords = new_coords.reshape(-1, 3)
        flat_valid = valid.reshape(-1, num_offsets)

        valid_indices = np.nonzero(flat_valid)
        valid_src_idx = valid_indices[0]
        flat_valid_idx = valid_indices[0] * num_offsets + valid_indices[1]
        valid_src_coords = flat_coords[valid_src_idx]
        valid_dst_coords = flat_new_coords[flat_valid_idx].astype(np.uint8)

        layer_size = self.height * self.width
        row_size = self.width
        src_ids_np = (valid_src_coords[:, 0].astype(np.int32) * layer_size +
                      valid_src_coords[:, 1].astype(np.int32) * row_size +
                      valid_src_coords[:, 2].astype(np.int32)).astype(np.int32)
        dst_ids_np = (valid_dst_coords[:, 0].astype(np.int32) * layer_size +
                      valid_dst_coords[:, 1].astype(np.int32) * row_size +
                      valid_dst_coords[:, 2].astype(np.int32)).astype(np.int32)

        num_edges = len(src_ids_np)
        values_np = np.random.randint(0, int(self.log_range) + 1, num_edges, dtype=np.uint8)

        self.src_ids = torch.from_numpy(src_ids_np).int().to(self.device)
        self.dst_ids = torch.from_numpy(dst_ids_np).int().to(self.device)
        self.values_exp = torch.from_numpy(values_np).to(self.device)

    def _decode_weights(self) -> torch.Tensor:
        """Decode uint8 exponents to float weights: w = 2^exp."""
        return torch.pow(2.0, self.values_exp.float())

    def _synaptic_input(self, prev_spikes: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of presynaptic spikes."""
        prev_spikes = prev_spikes.to(self.device)
        flat_pre = self.src_ids
        flat_post = self.dst_ids

        if self.log_domain and prev_spikes.sum() == 0.0:
            self.output_buffer.zero_()
            return self.output_buffer

        if self.log_domain:
            log2 = torch.log(torch.tensor(2.0, device=self.device))
            log_w = self.values_exp.float() * log2
            log_prev = torch.log(prev_spikes.float() + 1e-8)
            log_contrib = log_w + log_prev[flat_pre]
            post_max = torch.full((self.total_neurons,), float('-inf'), device=self.device)
            post_max.scatter_reduce_(0, flat_post.long(), log_contrib, reduce='amax')
            max_at_post = post_max[flat_post.long()]
            small_exps = torch.exp(log_contrib - max_at_post)
            small_sums = torch.zeros(self.total_neurons, device=self.device)
            small_sums.scatter_add_(0, flat_post.long(), small_exps)
            total = torch.exp(post_max) * small_sums
            self.output_buffer = total.to(torch.float16)
            return self.output_buffer
        else:
            weights = self._decode_weights()
            self.output_buffer.zero_()
            num_edges = len(flat_pre)
            for i in range(0, num_edges, self.batch_size):
                end = min(i + self.batch_size, num_edges)
                batch_pre = flat_pre[i:end]
                batch_post = flat_post[i:end].long()
                batch_w = weights[i:end]
                batch_contrib = (batch_w * prev_spikes[batch_pre]).to(torch.float16)
                self.output_buffer.scatter_add_(0, batch_post, batch_contrib)
            return self.output_buffer

    def _apply_plasticity(self, pre_spikes: torch.Tensor, post_spikes: torch.Tensor) -> None:
        """Sampled Hebbian plasticity with decay."""
        if not self.enable_plasticity or self.plasticity_sample_ratio <= 0:
            return

        flat_pre = self.src_ids
        flat_post = self.dst_ids
        num_edges = len(flat_pre)

        sample_size = max(1, int(num_edges * self.plasticity_sample_ratio))
        sample_idx = torch.randperm(num_edges, device=self.device)[:sample_size]

        sampled_pre = flat_pre[sample_idx]
        sampled_post = flat_post[sample_idx]
        sampled_exp = self.values_exp[sample_idx].float()

        hebb_delta = self.plasticity_lr * pre_spikes[sampled_pre] * post_spikes[sampled_post]
        hebb_delta *= self.log_range
        decay_delta = self.plasticity_decay * sampled_exp

        new_exp = sampled_exp + hebb_delta - decay_delta
        new_exp = new_exp.clamp(0, self.log_range).round()

        self.values_exp.scatter_(0, sample_idx, new_exp.to(torch.uint8))

    def forward(self, external_input: Optional[torch.Tensor] = None, dt: float = 1.0,
                num_steps: int = 10) -> Dict[str, torch.Tensor]:
        """Run simulation for num_steps."""
        if external_input is not None:
            external_input = external_input.to(self.device).clone()

        spikes_history: List[torch.Tensor] = []
        voltages_history: List[torch.Tensor] = []
        prev_spikes = torch.zeros(self.total_neurons, device=self.device)

        for _ in range(num_steps):
            synaptic_input = self._synaptic_input(prev_spikes)
            total_input = synaptic_input.clone().float()
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


# === CLI & Main ===
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="brain3d.py — 3D Neuromorphic CPU")
    parser.add_argument('--grid', type=int, default=10, help='Cubic grid size N (NxNxN). Default: 10')
    parser.add_argument('--steps', type=int, default=200, help='Simulation steps (even; half per phase). Default: 200')
    parser.add_argument('--no-plasticity', action='store_true', help='Disable plasticity for max speed')
    parser.add_argument('--strong-input', action='store_true', help='Use strong input (4.0) on full half-sheets for learning')
    parser.add_argument('--3d-inputs', dest='volumetric_inputs', action='store_true', help='Use full 3D vertical sheets (all layers) for volumetric learning')
    args = parser.parse_args()

    grid_size: int = args.grid
    num_steps: int = args.steps
    enable_plasticity: bool = not args.no_plasticity
    use_strong_input: bool = args.strong_input
    use_3d_inputs: bool = args.volumetric_inputs

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print_vram("Pre-init")

    np.random.seed(42)
    torch.manual_seed(42)

    net = Brain3DNetwork(
        shape=(grid_size, grid_size, grid_size),
        connectivity_radius=1,
        enable_plasticity=enable_plasticity,
        log_range=8.0,
        log_domain=True,
        device=device,
        input_scale=1.0,
        threshold_scale=0.1,  # LOWERED
        use_strong_input=use_strong_input
    )

    print(f"Grid: {grid_size}³ → {net.total_neurons:,} neurons, {net.src_ids.shape[0]:,} edges")
    print(f"Plasticity: {'ON' if enable_plasticity else 'OFF'}")
    print(f"Input mode: {'STRONG (learning)' if use_strong_input else 'weak (sparse)'}")
    print(f"3D inputs: {'ON (volumetric sheets)' if use_3d_inputs else 'OFF (layer-0 only)'}")
    print(f"Sample ratio: {net.plasticity_sample_ratio:.1%}")
    print(f"Steps: {num_steps} (phases: {num_steps//2} each)")
    print_vram("Post-init")

    # === INPUT PATTERNS ===
    layer_size: int = grid_size * grid_size
    half: int = grid_size // 2
    patterns: List[torch.Tensor] = [torch.zeros(net.total_neurons, device=device) for _ in range(2)]

    input_val: float = 4.0 if use_strong_input else 0.4

    # Left pattern: cols 0 to half-1
    col_range_left = range(half)
    pattern_idx = 0
    if use_3d_inputs:
        for layer in range(grid_size):
            for row in range(grid_size):
                for col in col_range_left:
                    idx = layer * layer_size + row * grid_size + col
                    patterns[pattern_idx][idx] = input_val
    else:
        for row in range(grid_size):
            for col in col_range_left:
                idx = 0 * layer_size + row * grid_size + col
                patterns[pattern_idx][idx] = input_val

    # Right pattern: cols half to end
    col_range_right = range(half, grid_size)
    pattern_idx = 1
    if use_3d_inputs:
        for layer in range(grid_size):
            for row in range(grid_size):
                for col in col_range_right:
                    idx = layer * layer_size + row * grid_size + col
                    patterns[pattern_idx][idx] = input_val
    else:
        for row in range(grid_size):
            for col in col_range_right:
                idx = 0 * layer_size + row * grid_size + col
                patterns[pattern_idx][idx] = input_val

    for p in patterns:
        p *= net.input_scale

    # === SIMULATION: num_steps, half per pattern ===
    phase_steps = num_steps // 2
    left_sum = torch.zeros(net.total_neurons, device=device)
    right_sum = torch.zeros(net.total_neurons, device=device)
    left_count = right_count = 0
    pattern_id = 0
    initial_exp_sum = net.values_exp.sum().item()
    print_vram("Pre-sim")

    for step in tqdm(range(num_steps), desc="Simulating"):
        if step % phase_steps == 0:
            pattern_id = 1 - pattern_id
        external_input = patterns[pattern_id]

        synaptic_input = net._synaptic_input(net.neurons.spikes)
        total_input = synaptic_input + external_input
        v, spikes = net.neurons(total_input, dt=1.0)
        net._apply_plasticity(net.neurons.spikes, spikes)
        net.neurons.v = v
        net.neurons.spikes = spikes

        if pattern_id == 0:  # Left phase
            left_sum += spikes
            left_count += 1
        else:
            right_sum += spikes
            right_count += 1

    print_vram("Post-sim")
    left_phase = left_sum / left_count if left_count > 0 else torch.zeros_like(left_sum)
    right_phase = right_sum / right_count if right_count > 0 else torch.zeros_like(right_sum)
    left_corr = torch.dot(left_phase, patterns[0]) / (left_phase.norm() * patterns[0].norm() + 1e-8)
    right_corr = torch.dot(right_phase, patterns[1]) / (right_phase.norm() * patterns[1].norm() + 1e-8)

    print(f"Avg spike rate (left): {left_phase.mean():.3f}")
    print(f"Avg spike rate (right): {right_phase.mean():.3f}")
    print(f"Correlation → Left: {left_corr:.3f}, Right: {right_corr:.3f}")
    delta = net.values_exp.sum().item() - initial_exp_sum
    print(f"Weight change (Δexp): {delta:+,.0f}")

    if device == 'cuda':
        torch.cuda.empty_cache()
    print_vram("Final")