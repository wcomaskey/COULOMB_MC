"""
Profiling script to test different parallelization strategies.

Tests:
1. Serial execution (no parallelization)
2. Current implementation (step-level parallelization)
3. Particle-level parallelization
4. Batch-level parallelization

Goal: Find optimal CPU parallelization before GPU implementation.
"""

import numpy as np
import time
import cProfile
import pstats
from io import StringIO
from coulomb_mc.transport.engine import TransportEngine
from coulomb_mc.core.particle import ParticleArray
import numba


def profile_current_implementation(n_particles=10000):
    """Profile current step-level parallelization."""
    print("=" * 70)
    print("TEST 1: Current Implementation (Step-level parallelization)")
    print("=" * 70)
    print(f"Particles: {n_particles:,}")
    print()
    
    engine = TransportEngine(material='water')
    beam = engine.create_beam('C-12', 400.0, n_particles=n_particles)
    
    # Warm up JIT compilation
    print("Warming up JIT compilation...")
    test_beam = engine.create_beam('C-12', 400.0, n_particles=100)
    engine.reset_scoring()
    engine.transport(test_beam, max_depth=30.0, verbose=False)
    engine.reset_scoring()
    
    print("Running profiled transport...")
    start = time.time()
    
    # Profile with cProfile
    profiler = cProfile.Profile()
    profiler.enable()
    
    stats = engine.transport(beam, max_depth=30.0, verbose=False)
    
    profiler.disable()
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Steps: {stats['n_steps']:,}")
    print(f"  Particles/sec: {n_particles/elapsed:.1f}")
    print(f"  Steps/sec: {stats['n_steps']/elapsed:.1f}")
    
    # Show top time consumers
    print("\nTop 10 time consumers:")
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs()
    ps.sort_stats('cumulative')
    ps.print_stats(10)
    print(s.getvalue())
    
    return elapsed, stats


def profile_serial_execution(n_particles=10000):
    """Profile serial (non-parallel) execution for comparison."""
    print("=" * 70)
    print("TEST 2: Serial Execution (No parallelization)")
    print("=" * 70)
    print(f"Particles: {n_particles:,}")
    print("Note: Requires disabling Numba parallel=True")
    print("Skipping - would need code modification")
    print()


def test_particle_batch_parallelization(n_particles=10000, batch_size=1000):
    """
    Test particle-level parallelization using batches.
    
    Process particles in independent batches, then combine results.
    This is embarrassingly parallel - perfect for multiprocessing.
    """
    print("=" * 70)
    print("TEST 3: Particle Batch Parallelization")
    print("=" * 70)
    print(f"Total particles: {n_particles:,}")
    print(f"Batch size: {batch_size:,}")
    print(f"Number of batches: {n_particles // batch_size}")
    print()
    
    from multiprocessing import Pool, cpu_count
    n_cpus = cpu_count()
    print(f"Available CPUs: {n_cpus}")
    
    # Function to run one batch
    def run_batch(batch_id):
        engine = TransportEngine(material='water')
        beam = engine.create_beam('C-12', 400.0, n_particles=batch_size)
        stats = engine.transport(beam, max_depth=30.0, verbose=False)
        
        # Return dose deposit and stats
        return engine.dose_deposit, stats['n_steps']
    
    print(f"\nRunning {n_particles // batch_size} batches on {n_cpus} CPUs...")
    start = time.time()
    
    # Create batches
    n_batches = n_particles // batch_size
    batch_ids = range(n_batches)
    
    # Run in parallel
    with Pool(processes=n_cpus) as pool:
        results = pool.map(run_batch, batch_ids)
    
    # Combine results
    engine = TransportEngine(material='water')  # Just for binning info
    combined_dose = np.zeros_like(engine.dose_deposit)
    total_steps = 0
    
    for dose, steps in results:
        combined_dose += dose
        total_steps += steps
    
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Total time: {elapsed:.2f} seconds")
    print(f"  Total steps: {total_steps:,}")
    print(f"  Particles/sec: {n_particles/elapsed:.1f}")
    print(f"  Steps/sec: {total_steps/elapsed:.1f}")
    print(f"  Speedup vs single core: ~{n_cpus:.1f}x (theoretical)")
    
    # Check peak location
    depth = (engine.dose_bins[:-1] + engine.dose_bins[1:]) / 2.0
    dose_normalized = 100.0 * combined_dose / np.max(combined_dose)
    peak_idx = np.argmax(dose_normalized)
    peak_depth = depth[peak_idx]
    print(f"  Bragg peak: {peak_depth:.2f} cm")
    
    return elapsed, total_steps


def analyze_bottlenecks():
    """
    Analyze where time is spent in current implementation.
    """
    print("=" * 70)
    print("BOTTLENECK ANALYSIS")
    print("=" * 70)
    
    # Test different components individually
    from coulomb_mc.physics.stopping_power import StoppingPower
    from coulomb_mc.physics.scattering import highland_angle
    from coulomb_mc.transport.engine import calculate_adaptive_step_size
    
    sp = StoppingPower('water')
    n_calls = 10000
    
    # Test stopping power lookup
    print("\n1. Stopping Power Calculation:")
    start = time.time()
    for i in range(n_calls):
        sp.stopping_power_MeV_cm2_g(400.0, 12.0, 6.0)
    elapsed = time.time() - start
    print(f"   {n_calls:,} calls in {elapsed:.4f}s = {n_calls/elapsed:.0f} calls/sec")
    
    # Test range calculation
    print("\n2. Range Calculation:")
    start = time.time()
    for i in range(n_calls):
        sp.range_g_cm2(400.0, 12.0, 6.0)
    elapsed = time.time() - start
    print(f"   {n_calls:,} calls in {elapsed:.4f}s = {n_calls/elapsed:.0f} calls/sec")
    
    # Test scattering angle
    print("\n3. Highland Angle Calculation:")
    start = time.time()
    for i in range(n_calls):
        highland_angle(400.0, 6.0, 12.0, 0.01, 36.08)
    elapsed = time.time() - start
    print(f"   {n_calls:,} calls in {elapsed:.4f}s = {n_calls/elapsed:.0f} calls/sec")
    
    # Test adaptive step calculation
    print("\n4. Adaptive Step Size Calculation:")
    start = time.time()
    for i in range(n_calls):
        calculate_adaptive_step_size(400.0, 12.0, 6.0, 109.08, 27.0, 1.0, 0.01, 0.05)
    elapsed = time.time() - start
    print(f"   {n_calls:,} calls in {elapsed:.4f}s = {n_calls/elapsed:.0f} calls/sec")
    
    # Test transport step
    print("\n5. Transport Step (100 particles):")
    from coulomb_mc.transport.engine import transport_step_with_scattering_adaptive
    from coulomb_mc.core.particle import PARTICLE_DTYPE
    
    particles = np.zeros(100, dtype=PARTICLE_DTYPE)
    particles['alive'] = True
    particles['energy'] = 400.0
    particles['A'] = 12.0
    particles['Z'] = 6.0
    particles['weight'] = 1.0
    particles['direction'][:, 2] = 1.0
    
    step_sizes = np.full(100, 0.01)
    energy_loss = np.full(100, 0.1)
    theta_rms = np.full(100, 0.001)
    
    n_steps = 1000
    start = time.time()
    for i in range(n_steps):
        transport_step_with_scattering_adaptive(particles, step_sizes, energy_loss, theta_rms)
    elapsed = time.time() - start
    print(f"   {n_steps:,} steps x 100 particles in {elapsed:.4f}s")
    print(f"   = {n_steps*100/elapsed:.0f} particle-steps/sec")


def compare_parallel_strategies():
    """Compare different Numba parallel strategies."""
    print("=" * 70)
    print("NUMBA PARALLELIZATION COMPARISON")
    print("=" * 70)
    
    n = 10000
    
    # Test parallel vs serial in transport step
    from coulomb_mc.core.particle import PARTICLE_DTYPE
    import numba
    
    # Serial version
    @numba.njit(fastmath=True, cache=False)
    def transport_serial(particles, step_sizes, energy_loss, theta_rms):
        for i in range(len(particles)):
            if particles['alive'][i]:
                step_length = step_sizes[i]
                particles['position'][i, 2] += step_length * particles['direction'][i, 2]
                particles['energy'][i] -= energy_loss[i]
                if particles['energy'][i] < 0.1:
                    particles['alive'][i] = False
    
    # Parallel version
    @numba.njit(parallel=True, fastmath=True, cache=False)
    def transport_parallel(particles, step_sizes, energy_loss, theta_rms):
        for i in numba.prange(len(particles)):
            if particles['alive'][i]:
                step_length = step_sizes[i]
                particles['position'][i, 2] += step_length * particles['direction'][i, 2]
                particles['energy'][i] -= energy_loss[i]
                if particles['energy'][i] < 0.1:
                    particles['alive'][i] = False
    
    # Setup
    particles = np.zeros(n, dtype=PARTICLE_DTYPE)
    particles['alive'] = True
    particles['energy'] = 400.0
    particles['direction'][:, 2] = 1.0
    
    step_sizes = np.full(n, 0.01)
    energy_loss = np.full(n, 0.1)
    theta_rms = np.full(n, 0.001)
    
    # Warm up
    transport_serial(particles.copy(), step_sizes, energy_loss, theta_rms)
    transport_parallel(particles.copy(), step_sizes, energy_loss, theta_rms)
    
    # Benchmark serial
    print(f"\nSerial (no parallelization):")
    n_iters = 1000
    particles_test = particles.copy()
    start = time.time()
    for _ in range(n_iters):
        transport_serial(particles_test, step_sizes, energy_loss, theta_rms)
    elapsed_serial = time.time() - start
    print(f"  {n_iters} iterations x {n} particles: {elapsed_serial:.4f}s")
    print(f"  Rate: {n_iters*n/elapsed_serial:.0f} particle-steps/sec")
    
    # Benchmark parallel
    print(f"\nParallel (Numba prange):")
    particles_test = particles.copy()
    start = time.time()
    for _ in range(n_iters):
        transport_parallel(particles_test, step_sizes, energy_loss, theta_rms)
    elapsed_parallel = time.time() - start
    print(f"  {n_iters} iterations x {n} particles: {elapsed_parallel:.4f}s")
    print(f"  Rate: {n_iters*n/elapsed_parallel:.0f} particle-steps/sec")
    print(f"  Speedup: {elapsed_serial/elapsed_parallel:.2f}x")


if __name__ == "__main__":
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "PARALLELIZATION PROFILING SUITE" + " " * 22 + "║")
    print("╚" + "═" * 68 + "╝")
    print()
    
    # Run tests with small particle count first
    n_test = 1000
    n_full = 10000
    
    print("Running quick tests with 1,000 particles...\n")
    
    # Test 1: Current implementation
    time_current, stats_current = profile_current_implementation(n_test)
    
    # Test 2: Bottleneck analysis
    analyze_bottlenecks()
    
    # Test 3: Numba parallel comparison
    compare_parallel_strategies()
    
    # Test 4: Multiprocessing
    print("\n")
    time_batch, steps_batch = test_particle_batch_parallelization(n_particles=n_test, batch_size=250)
    
    # Summary
    print("\n")
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nFor {n_test} particles:")
    print(f"  Current implementation: {time_current:.2f}s ({n_test/time_current:.0f} particles/sec)")
    print(f"  Batch parallelization:  {time_batch:.2f}s ({n_test/time_batch:.0f} particles/sec)")
    print(f"  Speedup: {time_current/time_batch:.2f}x")
    
    print("\n")
    print("RECOMMENDATIONS:")
    print("  1. Check if Numba parallel=True is actually helping (compare speedup)")
    print("  2. Consider multiprocessing for large particle counts (10^6+)")
    print("  3. Profile shows where to focus GPU optimization")
    print()
