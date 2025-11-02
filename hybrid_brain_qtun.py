# -*- coding: utf-8 -*-
"""
hybrid_brain_qtun.py

Implements a hybrid of brain3d.py and qtun.py, creating a large-scale 3D
spiking neural network with quantum-inspired ternary uncertainty neurons.

Key Features:
- 3D grid of neurons with local connectivity.
- QTUN-inspired neuron model with three states: excited, inhibited, and superposed.
- Balanced Hebbian plasticity (STDP) with both potentiation and depression.
- Designed for scalability on consumer GPUs using PyTorch and sparse tensors.

Core Ideas Merged:
1. From brain3d.py: Scalable 3D SNN structure, sparse synaptic propagation.
2. From qtun.py: Ternary neuron state (qReLU concept) to model uncertainty.
3. New: A balanced and more stable STDP rule to prevent weight saturation.
"""
import torch
import torch.nn as nn
import numpy as np
import time

class QTUNeuron(nn.Module):
    """
    A Quantum-Inspired Ternary Uncertainty Neuron (QTUN) model.

    This neuron replaces the classic LIF neuron. It has three distinct operational
    regimes based on its membrane potential `v`:
    1. Inhibited: If v is below a negative threshold, output is 0.
    2. Superposed: If v is between the thresholds, output is a probabilistic value
       (sigmoid-scaled), representing uncertainty.
    3. Excited (Spike): If v exceeds the positive threshold, output is 1.

    This allows the network to represent and process uncertainty, inspired by the
    superposition state in qutrits.
    """
    def __init__(self, size, tau=20.0, rest_v=-65.0, excite_threshold=-50.0, inhibit_threshold=-70.0, reset_v=-65.0, device='cpu'):
        super().__init__()
        self.size = size
        self.tau = tau
        self.rest_v = rest_v
        self.excite_threshold = excite_threshold
        self.inhibit_threshold = inhibit_threshold
        self.reset_v = reset_v
        self.device = device

        self.v = torch.full(size, self.rest_v, device=self.device, dtype=torch.float32)

    def forward(self, I, dt=1.0):
        """
        Performs a single time step update of the neuron's state.
        """
        # Leaky integration
        decay = torch.exp(-dt / self.tau)
        self.v = self.v * decay + I * (1 - decay)

        # 1. Excited State (Spike)
        spikes = (self.v >= self.excite_threshold).float()

        # 2. Inhibited State
        inhibited = (self.v <= self.inhibit_threshold).float()

        # 3. Superposed State
        is_superposed = (1 - spikes) * (1 - inhibited)
        
        # Output is a mix of definite spikes and probabilistic superposed signals
        # Sigmoid scaling for the superposed state to keep it between 0 and 1
        midpoint = (self.excite_threshold + self.inhibit_threshold) / 2
        superposed_output = torch.sigmoid((self.v - midpoint) * 0.5) # Scaled to be smooth
        
        output_signal = spikes + is_superposed * superposed_output

        # Reset voltage for neurons that spiked
        self.v = self.v * (1 - spikes) + spikes * self.reset_v
        
        return output_signal, spikes


class Brain3DQTUNNetwork(nn.Module):
    """
    A 3D Spiking Neural Network using QTUNeurons and balanced STDP.
    """
    def __init__(self, grid_size=(16, 16, 16), connectivity_radius=1,
                 eta_ltp=0.01, eta_ltd=0.005, weight_decay=1e-5, device='cpu'):
        super().__init__()
        self.grid_size = grid_size
        self.num_neurons = int(np.prod(grid_size))
        self.connectivity_radius = connectivity_radius
        self.eta_ltp = eta_ltp
        self.eta_ltd = eta_ltd
        self.weight_decay = weight_decay
        self.device = device

        # Initialize QTUNeurons
        self.neurons = QTUNeuron(
            size=(self.num_neurons,),
            excite_threshold=-50.0,
            inhibit_threshold=-70.0,
            rest_v=-65.0,
            reset_v=-65.0,
            tau=20.0,
            device=device
        )

        # Initialize sparse synaptic weights
        self.synaptic_weights = self._initialize_sparse_weights()

    def _initialize_sparse_weights(self):
        """
        Creates a sparse weight matrix with local connectivity.
        """
        print("Initializing sparse synaptic connections...")
        start_time = time.time()

        indices = []
        values = []
        
        # Create a coordinate grid
        coords = np.array(np.unravel_index(np.arange(self.num_neurons), self.grid_size)).T
        
        for i in range(self.num_neurons):
            coord_i = coords[i]
            # Find neighbors within the Manhattan distance
            distances = np.sum(np.abs(coords - coord_i), axis=1)
            neighbors = np.where((distances > 0) & (distances <= self.connectivity_radius))[0]
            
            for j in neighbors:
                indices.append([i, j]) # Connection from j to i
                values.append(np.random.uniform(0.1, 0.5))

        if not indices:
             raise ValueError("No connections were created. Check grid size and connectivity radius.")

        indices_tensor = torch.tensor(indices, dtype=torch.long, device=self.device).t()
        values_tensor = torch.tensor(values, dtype=torch.float32, device=self.device)
        
        weights = torch.sparse_coo_tensor(indices_tensor, values_tensor, (self.num_neurons, self.num_neurons))
        weights = weights.coalesce()

        print(f"Created {weights._nnz()} connections in {time.time() - start_time:.2f}s.")
        return weights

    def _synaptic_input(self, prev_signals):
        """
        Calculate synaptic input current based on previous neuron outputs.
        """
        if prev_signals is None:
            return torch.zeros(self.num_neurons, device=self.device)
        
        # Sparse matrix multiplication for efficient synaptic propagation
        syn_input = torch.sparse.mm(self.synaptic_weights.t(), prev_signals.unsqueeze(1)).squeeze(1)
        return syn_input

    def _apply_plasticity(self, prev_signals, post_signals):
        """
        Apply balanced STDP rule.
        - Potentiation (LTP): If pre-synaptic fires and post-synaptic signal is strong.
        - Depression (LTD): If pre-synaptic fires and post-synaptic signal is weak.
        - Weight Decay: All weights decay slightly over time.
        """
        if prev_signals is None:
            return

        with torch.no_grad():
            indices = self.synaptic_weights.indices()
            values = self.synaptic_weights.values()
            
            # Get pre and post signals for each synapse
            pre_indices = indices[1]
            post_indices = indices[0]
            
            pre_s = prev_signals[pre_indices]
            post_s = post_signals[post_indices]
            
            # Balanced STDP rule
            # Potentiation: proportional to pre * post signals
            dw_ltp = self.eta_ltp * pre_s * post_s
            # Depression: proportional to pre * (1 - post) for binary post-spikes,
            # or pre * (max_signal - post) for graded signals. Here we use a simple form.
            dw_ltd = self.eta_ltd * pre_s * (1.0 - post_s)
            
            delta_w = dw_ltp - dw_ltd
            
            # Apply weight decay
            delta_w -= self.weight_decay * values
            
            new_values = values + delta_w
            new_values = torch.clamp(new_values, 0.0, 1.0) # Keep weights in [0, 1]
            
            self.synaptic_weights = torch.sparse_coo_tensor(indices, new_values, self.synaptic_weights.size()).coalesce()

    def forward(self, external_input, num_steps=10):
        """
        Run the simulation for a number of steps.
        """
        prev_signals = None
        outputs = []

        for step in range(num_steps):
            syn_I = self._synaptic_input(prev_signals)
            total_I = syn_I + external_input[step]
            
            # Get both the full output signal and binary spikes
            output_signals, spikes = self.neurons(total_I)
            
            # Plasticity uses the full signal to capture uncertainty
            self._apply_plasticity(prev_signals, output_signals)
            
            prev_signals = output_signals
            outputs.append(spikes) # Store binary spikes for analysis

        return torch.stack(outputs)