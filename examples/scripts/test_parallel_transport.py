"""
Test parallel transport implementation.

Validates:
1. Correctness - parallel produces same results as serial
2. Performance - measures speedup on multiple cores
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from coulomb_mc.transport.engine import TransportEngine

def test_parallel_correctness():
    """Test that parallel transport produces same dose distribution as serial."""
    print("="*70)
    print("TEST 1: Parallel Correctness")
    print("="*70)

    # Create engine
    engine = TransportEngine(material='water')

    # Small particle count for quick test
    n_particles = 1000
    energy = 400.0  # MeV/u
    particle_type = 'C-12'

    print(f"\nTransporting {n_particles} {particle_type} @ {energy} MeV/u")

    # Serial transport
    print("\n1. Serial transport...")
    beam_serial = engine.create_beam(particle_type, energy, n_particles=n_particles)
    engine.reset_scoring()
    start = time.time()
    stats_serial = engine.transport(beam_serial, max_depth=30.0, verbose=False)
    time_serial = time.time() - start
    depth_serial, dose_serial = engine.get_dose_depth(normalize=True)

    # Parallel transport (same initial conditions)
    print("2. Parallel transport...")
    beam_parallel = engine.create_beam(particle_type, energy, n_particles=n_particles)
    engine.reset_scoring()
    start = time.time()
    stats_parallel = engine.transport_parallel(beam_parallel, max_depth=30.0,
                                               n_processes=4, verbose=False)
    time_parallel = time.time() - start
    depth_parallel, dose_parallel = engine.get_dose_depth(normalize=True)

    print(f"\nResults:")
    print(f"  Serial:   {time_serial:.2f}s ({stats_serial['total_steps']:,} steps)")
    print(f"  Parallel: {time_parallel:.2f}s ({stats_parallel['total_steps']:,} steps)")
    print(f"  Speedup:  {time_serial/time_parallel:.2f}x")

    # Compare dose distributions
    max_diff = np.max(np.abs(dose_serial - dose_parallel))
    mean_diff = np.mean(np.abs(dose_serial - dose_parallel))

    print(f"\nDose distribution comparison:")
    print(f"  Max difference:  {max_diff:.3f}%")
    print(f"  Mean difference: {mean_diff:.3f}%")

    # Plot comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(depth_serial, dose_serial, 'b-', label='Serial', linewidth=2)
    plt.plot(depth_parallel, dose_parallel, 'r--', label='Parallel', linewidth=2, alpha=0.7)
    plt.xlabel('Depth (cm)')
    plt.ylabel('Relative Dose (%)')
    plt.title('Dose Comparison: Serial vs Parallel')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(depth_serial, dose_serial - dose_parallel, 'g-', linewidth=2)
    plt.xlabel('Depth (cm)')
    plt.ylabel('Difference (%)')
    plt.title('Difference: Serial - Parallel')
    plt.grid(True, alpha=0.3)
    plt.axhline(0, color='k', linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.savefig('parallel_validation.png', dpi=150)
    print(f"\nPlot saved: parallel_validation.png")

    # Statistical test (expect some noise due to randomness)
    if mean_diff < 2.0:  # Allow 2% mean difference due to Monte Carlo
        print("\nVALIDATION PASSED")
        return True
    else:
        print("\nVALIDATION FAILED - difference too large")
        return False

def test_parallel_scaling():
    """Test parallel scaling with different core counts."""
    print("\n" + "="*70)
    print("TEST 2: Parallel Scaling")
    print("="*70)

    engine = TransportEngine(material='water')
    n_particles = 10000
    energy = 400.0
    particle_type = 'C-12'

    print(f"\nTransporting {n_particles} {particle_type} @ {energy} MeV/u")

    # Serial baseline
    print("\nBaseline (serial)...")
    beam = engine.create_beam(particle_type, energy, n_particles=n_particles)
    engine.reset_scoring()
    start = time.time()
    stats = engine.transport(beam, max_depth=30.0, verbose=False)
    time_serial = time.time() - start
    print(f"  Time: {time_serial:.1f}s")
    print(f"  Rate: {n_particles/time_serial:.0f} particles/sec")

    # Test different core counts
    core_counts = [1, 2, 4, 8]
    times = []
    speedups = []
    efficiencies = []

    print(f"\nParallel scaling:")
    for n_cores in core_counts:
        beam = engine.create_beam(particle_type, energy, n_particles=n_particles)
        engine.reset_scoring()
        start = time.time()
        stats = engine.transport_parallel(beam, max_depth=30.0,
                                          n_processes=n_cores, verbose=False)
        elapsed = time.time() - start

        speedup = time_serial / elapsed
        efficiency = (speedup / n_cores) * 100

        times.append(elapsed)
        speedups.append(speedup)
        efficiencies.append(efficiency)

        print(f"  {n_cores} cores: {elapsed:.1f}s, {speedup:.2f}x speedup, {efficiency:.1f}% efficient")

    # Plot scaling
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(core_counts, speedups, 'bo-', markersize=8, linewidth=2, label='Actual')
    plt.plot(core_counts, core_counts, 'r--', linewidth=2, label='Ideal (linear)')
    plt.xlabel('Number of Cores')
    plt.ylabel('Speedup')
    plt.title('Parallel Scaling')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(core_counts, efficiencies, 'go-', markersize=8, linewidth=2)
    plt.axhline(100, color='r', linestyle='--', linewidth=2, alpha=0.5, label='100% efficient')
    plt.axhline(75, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='75% efficient')
    plt.xlabel('Number of Cores')
    plt.ylabel('Parallel Efficiency (%)')
    plt.title('Parallel Efficiency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 110)

    plt.tight_layout()
    plt.savefig('parallel_scaling.png', dpi=150)
    print(f"\nPlot saved: parallel_scaling.png")

    return speedups, efficiencies

def main():
    """Run all parallel transport tests."""
    print("\n" + "="*70)
    print("PARALLEL TRANSPORT VALIDATION")
    print("="*70)

    # Test 1: Correctness
    passed = test_parallel_correctness()

    if not passed:
        print("\nWARNING: Correctness test failed. Check implementation.")
        return

    # Test 2: Scaling
    speedups, efficiencies = test_parallel_scaling()

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Correctness: PASSED")
    print(f"Best speedup: {max(speedups):.2f}x on {[1,2,4,8][np.argmax(speedups)]} cores")
    print(f"Best efficiency: {max(efficiencies):.1f}%")

    if max(speedups) >= 6.0:
        print("\nEXCELLENT: Achieved target 6-8x speedup")
    elif max(speedups) >= 4.0:
        print("\nGOOD: Achieved 4-6x speedup (close to target)")
    else:
        print("\nNEEDS IMPROVEMENT: Speedup below target")

    print("\n" + "="*70)

if __name__ == '__main__':
    main()
