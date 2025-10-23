# Brain3D Benchmarking Guide

This document explains how to use the comprehensive benchmarking suite for the Brain3D neuromorphic network.

## Overview

The `benchmark_brain3d.py` script provides extensive performance analysis of the Brain3DNetwork implementation with and without learning (plasticity) enabled. It supports both CPU and GPU benchmarking.

## Features

- **Multiple Network Sizes**: Tests small (1K neurons) to huge (4M neurons) networks
- **Plasticity Comparison**: Benchmarks with and without learning enabled
- **Comprehensive Metrics**:
  - Initialization time and throughput
  - Simulation time and steps per second
  - Memory usage (GPU VRAM tracking when available)
  - Spike statistics and firing rates
  - Neuron update throughput
  - Plasticity overhead analysis
- **Detailed Reports**: Generates JSON files with all metrics
- **Summary Tables**: Easy-to-read performance comparisons

## Requirements

```bash
pip install torch numpy tqdm
```

## Usage

### Basic Usage

Simply run the benchmark script:

```bash
python3 benchmark_brain3d.py
```

This will:
1. Detect available hardware (CPU/GPU)
2. Run benchmarks on multiple network sizes
3. Compare performance with/without plasticity
4. Save results to a timestamped JSON file
5. Print summary tables

### Output

The script produces:
- Console output with detailed progress and results
- `benchmark_results_<device>_<timestamp>.json` - Complete metrics in JSON format

### Example Output

```
================================================================================
BENCHMARK SUMMARY TABLE
================================================================================
Shape           Neurons      Plasticity   Init(s)    Sim(s)     Step/s     Spike Rate
----------------------------------------------------------------------------------------------------
10x10x10        1,000        No           0.008      0.039      2587.32    0.9285
10x10x10        1,000        Yes          0.002      0.054      1851.63    0.9285
30x30x30        27,000       No           0.068      0.253      197.89     0.5901
50x50x50        125,000      No           0.271      0.795      37.76      0.1141
================================================================================
```

## Benchmark Configuration

The script tests the following network configurations:

### CPU Benchmarks
- **Small**: 10×10×10 (1K neurons), 100 steps
- **Medium**: 30×30×30 (27K neurons), 50 steps
- **Large**: 50×50×50 (125K neurons), 30 steps

### GPU Benchmarks (when available)
- **Very Large**: 100×100×100 (1M neurons), 20 steps
- **Huge**: 160×160×160 (4M neurons), 10 steps

## Performance Metrics

### Initialization Phase
- `init_time_seconds`: Time to create network and connections
- `neurons_per_second`: Neuron creation throughput
- `edges_per_second`: Synapse creation throughput
- `memory_delta_mb`: Memory allocated during initialization

### Simulation Phase
- `sim_time_seconds`: Total simulation time
- `time_per_step`: Average time per simulation step
- `steps_per_second`: Simulation throughput
- `neuron_updates_per_second`: Total neuron updates per second
- `spike_rate`: Average proportion of neurons spiking
- `avg_spikes_per_step`: Mean spike count per step

### Plasticity Analysis
- `plasticity_overhead_percent`: Performance overhead of learning
- `plasticity_slowdown_factor`: Speed reduction factor with learning

## Customization

To customize the benchmarks, edit the `test_configs` list in `benchmark_brain3d.py`:

```python
test_configs = [
    {'name': 'Custom', 'shape': (40, 40, 40), 'steps': 60},
    # Add more configurations...
]
```

## Understanding Results

### Initialization Performance
- **Higher is better**: More neurons/sec and edges/sec
- Initialization is mainly CPU-bound (graph construction)
- GPU doesn't significantly speed up initialization

### Simulation Performance
- **Higher is better**: More steps/sec and neuron updates/sec
- GPU can provide 10-100× speedup for large networks
- Plasticity typically adds 20-40% overhead

### Memory Usage
- Scales approximately linearly with neuron count
- Sparse connectivity keeps memory manageable
- GPU memory is the primary constraint for large networks

### Spike Statistics
- `spike_rate` indicates network activity level
- Too high (>0.9): Network may be over-excited
- Too low (<0.01): Network may be under-stimulated
- Optimal range typically 0.1-0.5 for biological realism

## Troubleshooting

### Out of Memory Errors
- Reduce network size in test_configs
- Reduce `num_steps` for large networks
- Ensure no other GPU processes are running

### Slow Performance
- CPU simulation is significantly slower than GPU
- For networks >100K neurons, GPU is recommended
- Reduce connectivity_radius to decrease edge count

### CUDA Not Available
- Ensure PyTorch with CUDA support is installed
- Verify NVIDIA drivers are up to date
- Check `torch.cuda.is_available()` returns True

## Benchmark Results Interpretation

### Good Performance Indicators
- Initialization: >100K neurons/sec
- Simulation (GPU): >100 steps/sec for 1M neurons
- Simulation (CPU): >10 steps/sec for 100K neurons
- Memory efficiency: <10 MB per 1K neurons

### Performance Comparison
Expected relative performance (approximate):
- Small networks (1K): Plasticity overhead ~20-30%
- Medium networks (27K): Plasticity overhead ~25-35%
- Large networks (125K+): Plasticity overhead ~30-40%

## Advanced Usage

### Running Specific Benchmarks

You can modify `benchmark_brain3d.py` to run custom benchmarks:

```python
from benchmark_brain3d import BenchmarkRunner

runner = BenchmarkRunner(device='cuda')

# Run a specific configuration
result = runner.run_full_benchmark(
    shape=(50, 50, 50),
    enable_plasticity=True,
    num_steps=100
)

# Run comparative analysis
comparison = runner.run_comparative_benchmark(
    shape=(30, 30, 30),
    num_steps=50
)

# Save results
runner.save_results('my_benchmark.json')
```

### Analyzing JSON Results

The JSON output contains detailed metrics for each benchmark:

```python
import json

with open('benchmark_results_cpu_20231023_120000.json') as f:
    data = json.load(f)

# Access results
for result in data['results']:
    print(f"Network: {result['shape']}")
    print(f"Simulation time: {result['sim_time_seconds']:.3f}s")
    print(f"Spike rate: {result['spike_rate']:.4f}")
```

## Contributing

To add new benchmark configurations or metrics:
1. Modify the `BenchmarkRunner` class in `benchmark_brain3d.py`
2. Add new metric calculations in the benchmark methods
3. Update the JSON output format if needed
4. Document new metrics in this README

## References

- Main implementation: `brain3d.py`
- Network architecture: See Brain3DNetwork class documentation
- LIF neuron model: See LIFNeuron class documentation
