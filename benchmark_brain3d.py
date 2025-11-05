#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for brain3d.py

This script performs extensive benchmarking of the Brain3DNetwork with and without
learning (plasticity) enabled, testing various network sizes and configurations.
Supports both CPU and GPU benchmarking when available.
"""

import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

# Import the brain3d module
from brain3d import Brain3DNetwork, print_vram


class BenchmarkRunner:
    """Runs comprehensive benchmarks on Brain3DNetwork"""
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results: List[Dict[str, Any]] = []
        
    def benchmark_initialization(self, shape: Tuple[int, int, int], 
                                connectivity_radius: int = 1,
                                enable_plasticity: bool = False,
                                log_domain: bool = True) -> Dict[str, Any]:
        """Benchmark network initialization time and memory"""
        print(f"\n{'='*80}")
        print(f"Benchmarking Initialization: shape={shape}, plasticity={enable_plasticity}, device={self.device}")
        print(f"{'='*80}")
        
        # Record initial memory
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            initial_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            initial_memory = 0
        
        # Time initialization
        start_time = time.time()
        np.random.seed(42)
        torch.manual_seed(42)
        
        net = Brain3DNetwork(
            shape=shape,
            connectivity_radius=connectivity_radius,
            enable_plasticity=enable_plasticity,
            log_range=8.0,
            log_domain=log_domain,
            device=self.device
        )
        
        init_time = time.time() - start_time
        
        # Record post-init memory
        if self.device == 'cuda' and torch.cuda.is_available():
            peak_memory = torch.cuda.max_memory_allocated() / (1024**2)  # MB
            allocated_memory = torch.cuda.memory_allocated() / (1024**2)  # MB
        else:
            peak_memory = 0
            allocated_memory = 0
        
        total_neurons = net.total_neurons
        total_edges = net.src_ids.shape[0]
        
        result = {
            'phase': 'initialization',
            'shape': shape,
            'total_neurons': total_neurons,
            'total_edges': total_edges,
            'connectivity_radius': connectivity_radius,
            'enable_plasticity': enable_plasticity,
            'log_domain': log_domain,
            'device': self.device,
            'init_time_seconds': init_time,
            'initial_memory_mb': initial_memory,
            'allocated_memory_mb': allocated_memory,
            'peak_memory_mb': peak_memory,
            'memory_delta_mb': allocated_memory - initial_memory,
            'neurons_per_second': total_neurons / init_time if init_time > 0 else 0,
            'edges_per_second': total_edges / init_time if init_time > 0 else 0
        }
        
        print(f"  Total neurons: {total_neurons:,}")
        print(f"  Total edges: {total_edges:,}")
        print(f"  Initialization time: {init_time:.3f}s")
        print(f"  Memory used: {result['memory_delta_mb']:.2f} MB")
        print(f"  Throughput: {result['neurons_per_second']:.0f} neurons/sec")
        
        return result, net
    
    def benchmark_simulation(self, net: Brain3DNetwork, num_steps: int = 50,
                           with_input: bool = True) -> Dict[str, Any]:
        """Benchmark simulation performance"""
        print(f"\nBenchmarking Simulation: steps={num_steps}, with_input={with_input}")
        
        # Prepare input if needed
        external_input = None
        if with_input:
            external_input = torch.zeros(net.total_neurons, device=self.device)
            # Stimulate bottom layer with some pattern
            layer_size = net.width * net.height
            for i in range(min(100, net.width)):
                idx = 0 * layer_size + 0 * net.width + i
                if idx < net.total_neurons:
                    external_input[idx] = 0.5
        
        # Record memory before simulation
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            pre_sim_memory = torch.cuda.memory_allocated() / (1024**2)
        else:
            pre_sim_memory = 0
        
        # Run simulation
        start_time = time.time()
        spike_counts = []
        
        prev_spikes = torch.zeros(net.total_neurons, device=self.device)
        for step in range(num_steps):
            synaptic_input = net._synaptic_input(prev_spikes)
            
            total_input = synaptic_input.float()
            if external_input is not None:
                total_input += external_input
            
            v, spikes = net.neurons(total_input, dt=1.0)
            
            if net.enable_plasticity:
                net._apply_plasticity(prev_spikes, spikes)
            
            net.neurons.v, net.neurons.spikes = v, spikes
            prev_spikes = spikes
            spike_counts.append(spikes.sum().item())
        
        sim_time = time.time() - start_time
        
        # Record memory after simulation
        if self.device == 'cuda' and torch.cuda.is_available():
            peak_sim_memory = torch.cuda.max_memory_allocated() / (1024**2)
            post_sim_memory = torch.cuda.memory_allocated() / (1024**2)
        else:
            peak_sim_memory = 0
            post_sim_memory = 0
        
        # Calculate statistics
        avg_spikes = np.mean(spike_counts)
        std_spikes = np.std(spike_counts)
        spike_rate = avg_spikes / net.total_neurons if net.total_neurons > 0 else 0
        
        result = {
            'phase': 'simulation',
            'num_steps': num_steps,
            'with_input': with_input,
            'enable_plasticity': net.enable_plasticity,
            'sim_time_seconds': sim_time,
            'time_per_step': sim_time / num_steps if num_steps > 0 else 0,
            'steps_per_second': num_steps / sim_time if sim_time > 0 else 0,
            'total_spikes': sum(spike_counts),
            'avg_spikes_per_step': avg_spikes,
            'std_spikes_per_step': std_spikes,
            'spike_rate': spike_rate,
            'pre_sim_memory_mb': pre_sim_memory,
            'post_sim_memory_mb': post_sim_memory,
            'peak_sim_memory_mb': peak_sim_memory,
            'sim_memory_delta_mb': post_sim_memory - pre_sim_memory,
            'neuron_updates_per_second': (net.total_neurons * num_steps) / sim_time if sim_time > 0 else 0
        }
        
        print(f"  Simulation time: {sim_time:.3f}s")
        print(f"  Time per step: {result['time_per_step']:.4f}s")
        print(f"  Steps per second: {result['steps_per_second']:.2f}")
        print(f"  Average spikes per step: {avg_spikes:.1f}")
        print(f"  Spike rate: {spike_rate:.4f}")
        print(f"  Neuron updates/sec: {result['neuron_updates_per_second']:.0f}")
        
        return result
    
    def run_full_benchmark(self, shape: Tuple[int, int, int],
                          connectivity_radius: int = 1,
                          enable_plasticity: bool = False,
                          num_steps: int = 50,
                          log_domain: bool = True) -> Dict[str, Any]:
        """Run a complete benchmark: initialization + simulation"""
        
        # Validate inputs
        if len(shape) != 3:
            raise ValueError(f"Shape must be a 3-tuple, got {shape}")
        if any(s <= 0 for s in shape):
            raise ValueError(f"Invalid shape: {shape}. All dimensions must be positive.")
        if num_steps <= 0:
            raise ValueError(f"Invalid num_steps: {num_steps}. Must be positive.")
        if connectivity_radius < 0:
            raise ValueError(f"Invalid connectivity_radius: {connectivity_radius}. Must be non-negative.")
        
        # Benchmark initialization
        init_result, net = self.benchmark_initialization(
            shape=shape,
            connectivity_radius=connectivity_radius,
            enable_plasticity=enable_plasticity,
            log_domain=log_domain
        )
        
        # Benchmark simulation
        sim_result = self.benchmark_simulation(net, num_steps=num_steps, with_input=True)
        
        # Combine results
        full_result = {
            **init_result,
            **sim_result,
            'total_time_seconds': init_result['init_time_seconds'] + sim_result['sim_time_seconds']
        }
        
        self.results.append(full_result)
        
        # Clean up
        del net
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return full_result
    
    def run_comparative_benchmark(self, shape: Tuple[int, int, int],
                                 num_steps: int = 50) -> Dict[str, Any]:
        """Compare performance with and without plasticity"""
        print(f"\n{'#'*80}")
        print(f"COMPARATIVE BENCHMARK: With vs Without Learning")
        print(f"Network shape: {shape}, Steps: {num_steps}")
        print(f"{'#'*80}")
        
        # Benchmark without plasticity
        print("\n>>> Running WITHOUT plasticity (learning disabled)")
        result_no_plasticity = self.run_full_benchmark(
            shape=shape,
            enable_plasticity=False,
            num_steps=num_steps
        )
        
        # Benchmark with plasticity
        print("\n>>> Running WITH plasticity (learning enabled)")
        result_with_plasticity = self.run_full_benchmark(
            shape=shape,
            enable_plasticity=True,
            num_steps=num_steps
        )
        
        # Calculate comparative metrics
        comparison = {
            'shape': shape,
            'num_steps': num_steps,
            'device': self.device,
            'without_plasticity': result_no_plasticity,
            'with_plasticity': result_with_plasticity,
            'plasticity_overhead_percent': (
                (result_with_plasticity['sim_time_seconds'] - result_no_plasticity['sim_time_seconds']) 
                / result_no_plasticity['sim_time_seconds'] * 100
                if result_no_plasticity['sim_time_seconds'] > 0 else 0
            ),
            'plasticity_slowdown_factor': (
                result_with_plasticity['sim_time_seconds'] / result_no_plasticity['sim_time_seconds']
                if result_no_plasticity['sim_time_seconds'] > 0 else 1
            )
        }
        
        print(f"\n{'='*80}")
        print(f"COMPARATIVE RESULTS")
        print(f"{'='*80}")
        print(f"Without plasticity: {result_no_plasticity['sim_time_seconds']:.3f}s")
        print(f"With plasticity: {result_with_plasticity['sim_time_seconds']:.3f}s")
        print(f"Overhead: {comparison['plasticity_overhead_percent']:.1f}%")
        print(f"Slowdown factor: {comparison['plasticity_slowdown_factor']:.2f}x")
        
        return comparison
    
    def save_results(self, filename: str = None):
        """Save benchmark results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'benchmark_results_{self.device}_{timestamp}.json'
        
        output = {
            'timestamp': datetime.now().isoformat(),
            'device': self.device,
            'pytorch_version': torch.__version__,
            'cuda_available': torch.cuda.is_available(),
            'results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n✓ Results saved to {filename}")
        return filename


def print_summary_table(results: List[Dict[str, Any]]):
    """Print a summary table of all benchmark results"""
    print(f"\n{'='*100}")
    print(f"BENCHMARK SUMMARY TABLE")
    print(f"{'='*100}")
    print(f"{'Shape':<15} {'Neurons':<12} {'Plasticity':<12} {'Init(s)':<10} {'Sim(s)':<10} {'Step/s':<10} {'Spike Rate':<12}")
    print(f"{'-'*100}")
    
    for r in results:
        shape_str = f"{r['shape'][0]}x{r['shape'][1]}x{r['shape'][2]}"
        neurons = r['total_neurons']
        plasticity = 'Yes' if r['enable_plasticity'] else 'No'
        init_time = r['init_time_seconds']
        sim_time = r['sim_time_seconds']
        steps_per_sec = r['steps_per_second']
        spike_rate = r.get('spike_rate', 0)
        
        print(f"{shape_str:<15} {neurons:<12,} {plasticity:<12} {init_time:<10.3f} {sim_time:<10.3f} {steps_per_sec:<10.2f} {spike_rate:<12.4f}")
    
    print(f"{'='*100}\n")


def main():
    """Main benchmarking routine"""
    print(f"\n{'#'*80}")
    print(f"Brain3D Network Comprehensive Benchmarking Suite")
    print(f"{'#'*80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("Note: CUDA not available, running on CPU")
    
    print_vram("Initial")
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(device=device)
    
    # Define test configurations
    # Small, medium, and large network sizes
    test_configs = [
        {'name': 'Small', 'shape': (10, 10, 10), 'steps': 100},  # 1K neurons
        {'name': 'Medium', 'shape': (30, 30, 30), 'steps': 50},  # 27K neurons
        {'name': 'Large', 'shape': (50, 50, 50), 'steps': 30},   # 125K neurons
    ]
    
    # If GPU is available, add even larger tests
    if device == 'cuda':
        test_configs.extend([
            {'name': 'Very Large', 'shape': (100, 100, 100), 'steps': 20},  # 1M neurons
            {'name': 'Huge', 'shape': (160, 160, 160), 'steps': 10},        # 4M neurons
        ])
    
    # Run benchmarks for each configuration
    for config in test_configs:
        print(f"\n{'#'*80}")
        print(f"Testing {config['name']} Network: {config['shape']}")
        print(f"{'#'*80}")
        
        try:
            # Test without plasticity
            print(f"\n>>> {config['name']} - WITHOUT Plasticity")
            runner.run_full_benchmark(
                shape=config['shape'],
                enable_plasticity=False,
                num_steps=config['steps']
            )
            
            # Test with plasticity (only for smaller networks to save time)
            if config['name'] in ['Small', 'Medium']:
                print(f"\n>>> {config['name']} - WITH Plasticity")
                runner.run_full_benchmark(
                    shape=config['shape'],
                    enable_plasticity=True,
                    num_steps=config['steps']
                )
        except Exception as e:
            print(f"ERROR in {config['name']} benchmark: {e}")
            import traceback
            traceback.print_exc()
    
    # Run comparative benchmark on a medium-sized network
    print(f"\n{'#'*80}")
    print(f"DETAILED COMPARATIVE ANALYSIS")
    print(f"{'#'*80}")
    
    try:
        runner.run_comparative_benchmark(shape=(20, 20, 20), num_steps=50)
    except Exception as e:
        print(f"ERROR in comparative benchmark: {e}")
        import traceback
        traceback.print_exc()
    
    # Print summary
    print_summary_table(runner.results)
    
    # Save results
    filename = runner.save_results()
    
    print(f"\n{'#'*80}")
    print(f"Benchmarking Complete!")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {filename}")
    print(f"{'#'*80}\n")
    
    print_vram("Final")
    
    return runner.results


if __name__ == '__main__':
    main()
