# -*- coding: utf-8 -*-
"""
test_hybrid_brain_qtun.py

Unit and integration tests for the hybrid_brain_qtun.py module.

Tests cover:
1. QTUNeuron state transitions (inhibited, superposed, excited).
2. Brain3DQTUNNetwork initialization and connectivity.
3. The balanced STDP plasticity rule (potentiation and depression).
4. An end-to-end simulation run to ensure functional integrity.
"""
import unittest
import torch
from hybrid_brain_qtun import QTUNeuron, Brain3DQTUNNetwork

class TestQTUNeuron(unittest.TestCase):
    """Tests for the QTUNeuron class."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.neuron = QTUNeuron(
            size=(3,),
            excite_threshold=-50.0,
            inhibit_threshold=-70.0,
            rest_v=-65.0,
            device=self.device
        )
        self.neuron.v = torch.tensor([-75.0, -60.0, -45.0], device=self.device) # Inhibited, Superposed, Excited

    def test_neuron_states(self):
        """Test that the neuron correctly identifies all three states."""
        I = torch.zeros(3, device=self.device)
        output_signal, spikes = self.neuron(I)

        # 1. Inhibited neuron should have 0 output and no spike
        self.assertAlmostEqual(output_signal[0].item(), 0.0, places=5)
        self.assertEqual(spikes[0].item(), 0.0)

        # 2. Superposed neuron should have a probabilistic output and no spike
        self.assertTrue(0 < output_signal[1].item() < 1)
        self.assertEqual(spikes[1].item(), 0.0)

        # 3. Excited neuron should have output of 1 and a spike
        self.assertAlmostEqual(output_signal[2].item(), 1.0, places=5)
        self.assertEqual(spikes[2].item(), 1.0)
        
        # Check voltage reset for the excited neuron
        self.assertAlmostEqual(self.neuron.v[2].item(), self.neuron.reset_v, places=5)


class TestBrain3DQTUNNetwork(unittest.TestCase):
    """Tests for the Brain3DQTUNNetwork class."""

    def setUp(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.grid_size = (4, 4, 4) # 64 neurons
        self.net = Brain3DQTUNNetwork(grid_size=self.grid_size, device=self.device)

    def test_initialization(self):
        """Test network initialization."""
        self.assertEqual(self.net.num_neurons, 64)
        self.assertTrue(self.net.synaptic_weights.is_sparse)
        self.assertGreater(self.net.synaptic_weights._nnz(), 0)

    def test_plasticity_rule(self):
        """Test the STDP rule for both potentiation and depression."""
        # Manually create a simple network with one connection: 1 -> 0
        self.net.synaptic_weights = torch.sparse_coo_tensor(
            torch.tensor([[0], [1]], device=self.device),
            torch.tensor([0.5], device=self.device),
            (self.net.num_neurons, self.net.num_neurons)
        ).coalesce()

        initial_weight = self.net.synaptic_weights.values()[0].item()

        # --- Test Case 1: Potentiation (LTP) ---
        # Pre-synaptic fires (1), Post-synaptic fires (0)
        pre_signals = torch.zeros(self.net.num_neurons, device=self.device)
        pre_signals[1] = 1.0 # Pre-synaptic neuron fires
        post_signals = torch.zeros(self.net.num_neurons, device=self.device)
        post_signals[0] = 1.0 # Post-synaptic neuron fires

        self.net._apply_plasticity(pre_signals, post_signals)
        weight_after_ltp = self.net.synaptic_weights.values()[0].item()
        self.assertGreater(weight_after_ltp, initial_weight)

        # --- Test Case 2: Depression (LTD) ---
        # Pre-synaptic fires (1), Post-synaptic is inhibited (0)
        pre_signals[1] = 1.0
        post_signals[0] = 0.0 # Post-synaptic is inhibited
        
        self.net._apply_plasticity(pre_signals, post_signals)
        weight_after_ltd = self.net.synaptic_weights.values()[0].item()
        self.assertLess(weight_after_ltd, weight_after_ltp)

    def test_end_to_end_simulation(self):
        """Test a full simulation run for a few steps."""
        num_steps = 5
        external_input = torch.zeros(num_steps, self.net.num_neurons, device=self.device)
        # Provide a small stimulus to a few neurons to kickstart activity
        external_input[0, :5] = -40.0 

        try:
            outputs = self.net(external_input, num_steps=num_steps)
            self.assertEqual(outputs.shape, (num_steps, self.net.num_neurons))
        except Exception as e:
            self.fail(f"End-to-end simulation failed with an exception: {e}")

if __name__ == '__main__':
    unittest.main()