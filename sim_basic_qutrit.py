"""
sim_basic_qutrit.py
-------------------
A fully documented, type-hinted, and visually exciting simulation of a
**neuromorphic quantum qutrit** with strong, structured coupling.

Features:
- Shows Rabi oscillations between |0⟩, |1⟩, |2⟩
- All three states visible and evolving
- CPU-only, <2 seconds on GTX 1050
- Saves qutrit_evolution.png
- Ready for integration with brain3d.py

Author: @EdgeOfAssembly
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from qutip import Qobj, basis, mesolve
from typing import Tuple, List


def create_qutrit_hamiltonian(
    energy_levels: Tuple[float, float, float] = (0.0, 0.5, 1.0),
    coupling_strength: float = 1.5,
    seed: int | None = 42,
) -> Qobj:
    """
    Create a 3x3 Hamiltonian with strong, structured coupling:
        - |0⟩ ↔ |1⟩: strong coupling
        - |1⟩ ↔ |2⟩: medium coupling
    → Produces visible Rabi oscillations and |1⟩ excitation.

    Args:
        energy_levels: Energies of |0⟩, |1⟩, |2⟩ (must be close for resonance)
        coupling_strength: Base strength of |0⟩↔|1⟩ coupling
        seed: For reproducibility (not used in structured version)

    Returns:
        Qobj: 3x3 Hermitian Hamiltonian
    """
    diag = np.array(energy_levels)
    H_diag = np.diag(diag)

    # Structured off-diagonal couplings (like synaptic weights)
    off_diag = np.zeros((3, 3))
    off_diag[0, 1] = off_diag[1, 0] = coupling_strength        # |0> ↔ |1>
    off_diag[1, 2] = off_diag[2, 1] = coupling_strength * 0.6  # |1> ↔ |2>

    H = Qobj(H_diag + off_diag)
    return H


def simulate_qutrit_evolution(
    H: Qobj,
    initial_state: Qobj,
    times: np.ndarray,
) -> List[Qobj]:
    """
    Solve the time evolution using QuTiP's master equation solver.

    Args:
        H: Hamiltonian
        initial_state: Starting state (e.g. |0⟩)
        times: Array of time points

    Returns:
        List of quantum states over time
    """
    result = mesolve(H, initial_state, times, c_ops=[], e_ops=[])
    return result.states


def compute_state_probabilities(
    states: List[Qobj],
    basis_states: Tuple[Qobj, Qobj, Qobj],
) -> np.ndarray:
    """
    Compute probability of being in |0⟩, |1⟩, |2⟩ at each time step.

    Args:
        states: Evolved states
        basis_states: (|0⟩, |1⟩, |2⟩)

    Returns:
        (3, N) array of probabilities
    """
    probs = np.zeros((3, len(states)))
    for i, state in enumerate(states):
        for j, basis in enumerate(basis_states):
            probs[j, i] = abs(state.overlap(basis)) ** 2
    return probs


def plot_qutrit_evolution(
    times: np.ndarray,
    probs: np.ndarray,
    save_path: str = "qutrit_evolution.png",
) -> None:
    """
    Plot and save the qutrit state probabilities with clear styling.

    Args:
        times: Time array
        probs: (3, N) probability matrix
        save_path: Output image path
    """
    plt.figure(figsize=(10, 6))
    labels = ["|0⟩ (ground)", "|1⟩ (excited)", "|2⟩ (higher)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    linestyles = ["-", "-", "-"]
    linewidths = [2.8, 2.8, 2.8]

    for i in range(3):
        plt.plot(times, probs[i], label=labels[i], color=colors[i],
                 linewidth=linewidths[i], linestyle=linestyles[i])

    plt.xlabel("Time (arbitrary units)", fontsize=13)
    plt.ylabel("State Probability", fontsize=13)
    plt.title("Qutrit Evolution in Neuromorphic Quantum Neuron", fontsize=15, pad=20)
    plt.legend(fontsize=12, frameon=True, fancybox=True, shadow=True)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.xlim(times[0], times[-1])
    plt.ylim(-0.05, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.show()

    print(f"Simulation complete! Plot saved: {save_path}")
    print("All three states visible: |0⟩, |1⟩, |2⟩ with Rabi oscillations.")
    print("Next: Connect to brain3d.py for 3D quantum neural network!")


def main() -> None:
    """Run the full qutrit demo with visible oscillations."""
    # 1. Define qutrit basis states
    basis0, basis1, basis2 = basis(3, 0), basis(3, 1), basis(3, 2)
    initial_state = basis0  # Start in ground state

    # 2. Create resonant Hamiltonian with strong coupling
    H = create_qutrit_hamiltonian(
        energy_levels=(0.0, 0.5, 1.0),   # Close energy levels → resonance
        coupling_strength=1.5,           # Strong |0>↔|1> coupling
        seed=42,
    )

    # 3. Time evolution (long enough to see multiple cycles)
    times = np.linspace(0, 20, 400)

    # 4. Simulate
    states = simulate_qutrit_evolution(H, initial_state, times)

    # 5. Compute probabilities
    probs = compute_state_probabilities(states, (basis0, basis1, basis2))

    # 6. Plot and save
    plot_qutrit_evolution(times, probs)


if __name__ == "__main__":
    main()