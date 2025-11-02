# Neuromorphic Quantum Computing
A 3D Brain-Inspired CPU with Schrödinger Qutrits & QTUN

[Python 3.9+]   [CUDA Required]   [Plasticity ON]

Requires NVIDIA GPU (CUDA) --- CPU fallback in development
Full 3D volumetric learning with synaptic plasticity

## Live Demo: 2+ Million Neuron 3D Brain

```bash
python3 brain3d.py --grid 128 --strong-input --3d-inputs
```

--------------------------------------------------------------------------------
[3D volumetric learning plot will be added]
--------------------------------------------------------------------------------

**Output:**
- 128³ grid → 2,097,152 neurons, 53.6M edges
- Plasticity ON (default)
- STRONG input on full 3D sheets
- Volumetric learning across all layers
- Weight change: +214M (learning confirmed)
- VRAM: ~3.4 GB peak

## Why This Project?

| Feature                        | Benefit                              |
|--------------------------------|--------------------------------------|
| 3D Neuromorphic CPU            | Mimics human volumetric processing   |
| Synaptic Plasticity            | Real-time learning (default ON)      |
| CUDA-Accelerated               | 128³ grid in 73 seconds              |
| Qutrits + QTUN                 | Quantum-enhanced control (in cartpole_a2c.py) |

## Quick Start (Full Experience)

```bash
# Requires: NVIDIA GPU + CUDA + PyTorch
python3 brain3d.py --grid 128 --strong-input --3d-inputs
```

For smaller test:
```bash
python3 brain3d.py --grid 64 --no-plasticity
```

## Key Programs

- **brain3d.py** --- 3D neuromorphic CPU with plasticity
- **cartpole_a2c.py** --- QTUN + A2C on CartPole (quantum control)
- **sim_basic_qutrit.py** --- Qutrit neuron demo (CPU-only)

## Roadmap

- [x] 128³ 3D brain with volumetric learning
- [x] Synaptic plasticity (default ON)
- [x] CUDA acceleration
- [ ] CPU fallback mode
- [ ] QTUN full integration
- [ ] Real-time visualization
- [ ] Research paper

## Contributing

See CONTRIBUTING.md

Good first issues:
- Add 3D spike visualization (brain3d.py)
- CPU-only mode (no CUDA)
- 1-page paper: "3D Neuromorphic Volumetric Learning"

## Hardware Requirements

| Component    | Required                      |
|--------------|-------------------------------|
| GPU          | NVIDIA with CUDA (4GB+ VRAM)  |
| RAM          | 16 GB+ recommended            |
| Storage      | 500 MB                        |

## Contributors

[You could be here!]

## License

This repository is dual-licensed.

For non-commercial use, this project is licensed under the GNU General Public License v3.0. Please see the [LICENSE](LICENSE) file for more details.

For commercial use, please contact the author, EdgeOfAssembly, at [haxbox2000@gmail.com](mailto:haxbox2000@gmail.com) to arrange a licensing agreement.

## Connect

[@EdgeOfAssembly](https://github.com/EdgeOfAssembly) | [Open an Issue](https://github.com/EdgeOfAssembly/neuromorphic-quantum-computing/issues)

---

Made with passion by [@EdgeOfAssembly](https://github.com/EdgeOfAssembly)  
2+ million neurons. 3D learning. CUDA-powered.