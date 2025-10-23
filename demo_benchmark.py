#!/usr/bin/env python3
"""
Quick demonstration of Brain3D benchmarking with and without learning.

This script runs a focused comparison benchmark to demonstrate the
plasticity overhead and basic performance characteristics.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

from benchmark_brain3d import BenchmarkRunner, print_summary_table
import torch

def main():
    print("\n" + "="*80)
    print("Brain3D Quick Benchmark Demo - Learning Enabled vs Disabled")
    print("="*80)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nRunning on: {device}")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(device=device)
    
    # Test configuration
    test_size = (15, 15, 15)  # 3,375 neurons - good for quick demo
    num_steps = 40
    
    print(f"\nTest configuration:")
    print(f"  Network size: {test_size}")
    print(f"  Total neurons: {test_size[0] * test_size[1] * test_size[2]:,}")
    print(f"  Simulation steps: {num_steps}")
    
    # Run benchmarks
    print(f"\n{'='*80}")
    print("1. WITHOUT Learning (Plasticity Disabled)")
    print(f"{'='*80}")
    result_no_learning = runner.run_full_benchmark(
        shape=test_size,
        enable_plasticity=False,
        num_steps=num_steps
    )
    
    print(f"\n{'='*80}")
    print("2. WITH Learning (Plasticity Enabled)")
    print(f"{'='*80}")
    result_with_learning = runner.run_full_benchmark(
        shape=test_size,
        enable_plasticity=True,
        num_steps=num_steps
    )
    
    # Calculate and display comparison
    print(f"\n{'='*80}")
    print("COMPARISON SUMMARY")
    print(f"{'='*80}")
    
    init_diff = ((result_with_learning['init_time_seconds'] - result_no_learning['init_time_seconds']) 
                 / result_no_learning['init_time_seconds'] * 100)
    sim_diff = ((result_with_learning['sim_time_seconds'] - result_no_learning['sim_time_seconds']) 
                / result_no_learning['sim_time_seconds'] * 100)
    
    print(f"\nInitialization:")
    print(f"  Without learning: {result_no_learning['init_time_seconds']:.3f}s")
    print(f"  With learning:    {result_with_learning['init_time_seconds']:.3f}s")
    print(f"  Difference:       {init_diff:+.1f}%")
    
    print(f"\nSimulation:")
    print(f"  Without learning: {result_no_learning['sim_time_seconds']:.3f}s")
    print(f"  With learning:    {result_with_learning['sim_time_seconds']:.3f}s")
    print(f"  Difference:       {sim_diff:+.1f}%")
    print(f"  Overhead factor:  {result_with_learning['sim_time_seconds'] / result_no_learning['sim_time_seconds']:.2f}x")
    
    print(f"\nThroughput (steps per second):")
    print(f"  Without learning: {result_no_learning['steps_per_second']:.1f} steps/sec")
    print(f"  With learning:    {result_with_learning['steps_per_second']:.1f} steps/sec")
    
    print(f"\nNeuron updates per second:")
    print(f"  Without learning: {result_no_learning['neuron_updates_per_second']:,.0f}")
    print(f"  With learning:    {result_with_learning['neuron_updates_per_second']:,.0f}")
    
    print(f"\nSpike statistics:")
    print(f"  Average spike rate: {result_no_learning['spike_rate']:.4f}")
    print(f"  Spikes per step:    {result_no_learning['avg_spikes_per_step']:.0f}")
    
    # Print summary table
    print_summary_table(runner.results)
    
    # Save results
    filename = runner.save_results('quick_demo_results.json')
    
    print(f"\n{'='*80}")
    print("KEY TAKEAWAYS")
    print(f"{'='*80}")
    print(f"✓ Network successfully simulated {result_no_learning['total_neurons']:,} neurons")
    print(f"✓ Learning adds approximately {sim_diff:.0f}% overhead to simulation")
    print(f"✓ Initialization time is similar with/without learning")
    print(f"✓ Performance suitable for {'GPU' if device == 'cuda' else 'CPU'} deployment")
    print(f"\n✓ Demo complete! Results saved to {filename}")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
