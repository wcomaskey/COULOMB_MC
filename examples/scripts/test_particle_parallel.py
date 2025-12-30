"""
Test pure particle-level parallelization using multiprocessing.

Since particles are independent, this should be embarrassingly parallel.
"""

import numpy as np
import time
from multiprocessing import Pool, cpu_count
from coulomb_mc.transport.engine import TransportEngine


def run_batch_particles(args):
    """
    Run transport for a batch of particles.
    
    Must be top-level function for pickling.
    """
    batch_size, seed = args
    
    # Set unique random seed for this batch
    np.random.seed(seed)
    
    # Create engine and beam
    engine = TransportEngine(material='water')
    beam = engine.create_beam('C-12', 400.0, n_particles=batch_size)
    
    # Transport
    stats = engine.transport(beam, max_depth=30.0, verbose=False)
    
    # Return dose deposit and stats
    return engine.dose_deposit.copy(), stats['n_steps']


def test_multiprocess_parallelization(n_particles=10000, n_processes=None):
    """
    Test particle-level parallelization using multiprocessing.
    """
    if n_processes is None:
        n_processes = cpu_count()
    
    print(f"Testing multiprocess particle parallelization")
    print(f"  Total particles: {n_particles:,}")
    print(f"  Processes: {n_processes}")
    print(f"  Particles per process: {n_particles // n_processes:,}")
    print()
    
    # Create batch arguments
    batch_size = n_particles // n_processes
    batch_args = [(batch_size, i * 1000) for i in range(n_processes)]
    
    print("Running parallel batches...")
    start = time.time()
    
    with Pool(processes=n_processes) as pool:
        results = pool.map(run_batch_particles, batch_args)
    
    elapsed = time.time() - start
    
    # Combine results
    engine = TransportEngine(material='water')
    combined_dose = np.zeros_like(engine.dose_deposit)
    total_steps = 0
    
    for dose, steps in results:
        combined_dose += dose
        total_steps += steps
    
    print(f"\nResults:")
    print(f"  Wall time: {elapsed:.2f} seconds")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Throughput: {n_particles/elapsed:.1f} particles/sec")
    print(f"  Step rate: {total_steps/elapsed:.1f} steps/sec")
    
    # Verify physics
    depth = (engine.dose_bins[:-1] + engine.dose_bins[1:]) / 2.0
    dose_normalized = 100.0 * combined_dose / np.max(combined_dose)
    peak_idx = np.argmax(dose_normalized)
    peak_depth = depth[peak_idx]
    print(f"  Bragg peak: {peak_depth:.2f} cm (expected: 26.4 cm)")
    
    return elapsed


def test_single_process(n_particles=10000):
    """
    Baseline: single process for comparison.
    """
    print(f"Testing single-process baseline")
    print(f"  Total particles: {n_particles:,}")
    print()
    
    np.random.seed(0)
    engine = TransportEngine(material='water')
    beam = engine.create_beam('C-12', 400.0, n_particles=n_particles)
    
    print("Running transport...")
    start = time.time()
    stats = engine.transport(beam, max_depth=30.0, verbose=False)
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Wall time: {elapsed:.2f} seconds")
    print(f"  Total steps: {stats['n_steps']:,}")
    print(f"  Throughput: {n_particles/elapsed:.1f} particles/sec")
    print(f"  Step rate: {stats['n_steps']/elapsed:.1f} steps/sec")
    
    # Verify physics
    depth, dose = engine.get_dose_depth(normalize=True)
    peak_idx = np.argmax(dose)
    peak_depth = depth[peak_idx]
    print(f"  Bragg peak: {peak_depth:.2f} cm (expected: 26.4 cm)")
    
    return elapsed


if __name__ == "__main__":
    print()
    print("=" * 70)
    print("PARTICLE-LEVEL PARALLELIZATION TEST")
    print("=" * 70)
    print()
    
    n_cpus = cpu_count()
    print(f"System has {n_cpus} CPUs available")
    print()
    
    # Test with different particle counts
    for n_particles in [1000, 10000]:
        print("\n" + "=" * 70)
        print(f"TEST: {n_particles:,} particles")
        print("=" * 70)
        print()
        
        # Single process baseline
        time_single = test_single_process(n_particles)
        
        print("\n")
        
        # Multi-process
        time_multi = test_multiprocess_parallelization(n_particles, n_processes=n_cpus)
        
        # Calculate speedup
        speedup = time_single / time_multi
        efficiency = speedup / n_cpus * 100
        
        print("\n" + "-" * 70)
        print(f"SPEEDUP ANALYSIS:")
        print(f"  Single process: {time_single:.2f}s")
        print(f"  {n_cpus} processes:  {time_multi:.2f}s")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Parallel efficiency: {efficiency:.1f}%")
        print("-" * 70)
    
    print("\n")
    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("Particle-level parallelization (multiprocessing) should show:")
    print("  - Near-linear speedup with number of CPUs")
    print("  - >80% parallel efficiency")
    print("  - No synchronization overhead (particles are independent)")
    print()
    print("If efficiency is low:")
    print("  - Check if transport is I/O bound (NIST data loading)")
    print("  - Check memory bandwidth limits")
    print("  - Consider GPU for 10^6+ particles")
    print()
