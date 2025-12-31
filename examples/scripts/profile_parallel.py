#!/usr/bin/env python3
"""
Profile parallel transport to identify bottlenecks.
"""
import numpy as np
import time
import cProfile
import pstats
from io import StringIO
from coulomb_mc.transport.engine import TransportEngine

def profile_serial_transport():
    """Profile serial transport in detail."""
    print("\n" + "="*70)
    print("PROFILING SERIAL TRANSPORT (VECTORIZED)")
    print("="*70)

    # Create engine first
    engine = TransportEngine(material='water')

    # Create beam with 100 particles
    beam = engine.create_beam('C-12', 400.0, n_particles=100)

    # Time serial transport with profiling
    print("\n1. Serial transport (100 particles):")

    profiler = cProfile.Profile()
    profiler.enable()
    start = time.time()

    stats = engine.transport(beam, max_depth=50.0, verbose=False)

    elapsed = time.time() - start
    profiler.disable()

    print(f"   Time: {elapsed:.6f}s")
    print(f"   Steps: {stats['n_steps']}")
    print(f"   Steps/particle: {stats['n_steps']/100:.1f}")

    # Show top time consumers
    print("\n   Top function calls:")
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    print(s.getvalue())

def profile_parallel_overhead():
    """Profile parallel transport overhead."""
    print("\n" + "="*70)
    print("PROFILING PARALLEL TRANSPORT OVERHEAD")
    print("="*70)

    engine = TransportEngine(material='water')
    beam = engine.create_beam('C-12', 400.0, n_particles=100)

    print("\n1. Serial transport (100 particles):")
    start = time.time()
    stats_serial = engine.transport(beam, max_depth=50.0, verbose=False)
    serial_time = time.time() - start
    print(f"   Time: {serial_time:.3f}s")
    print(f"   Steps: {stats_serial['n_steps']}")

    print("\n2. Parallel transport (100 particles, 4 cores):")
    engine2 = TransportEngine(material='water')
    beam2 = engine2.create_beam('C-12', 400.0, n_particles=100)

    start = time.time()
    stats_parallel = engine2.transport_parallel(beam2, max_depth=50.0,
                                                verbose=True, n_processes=4)
    parallel_time = time.time() - start

    print(f"\n   Total time: {parallel_time:.3f}s")
    print(f"   Steps: {stats_parallel['n_steps']}")
    print(f"   Speedup: {serial_time/parallel_time:.2f}x")
    print(f"   Efficiency: {(serial_time/parallel_time)/4*100:.1f}%")

def check_chunk_worker():
    """Check chunk-based worker function."""
    print("\n" + "="*70)
    print("CHECKING CHUNK-BASED WORKER")
    print("="*70)

    from coulomb_mc.transport.engine import _worker_engine, _init_worker, _transport_chunk_worker

    print("\n1. Testing worker initialization:")
    print("   Calling _init_worker...")
    start = time.time()
    _init_worker('water', 0.05)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.6f}s")

    # Note: _worker_engine will be None here because it's set in worker processes
    # This is expected behavior for multiprocessing globals
    print(f"   Note: Global engine in main process: {_worker_engine}")
    print(f"   (Worker engines are created in separate processes)")

    print("\n2. Testing chunk worker function:")
    # Create a small chunk of particles
    engine = TransportEngine(material='water')
    beam = engine.create_beam('C-12', 400.0, n_particles=10)

    work_item = {
        'particle_data': beam.particles.copy(),
        'max_depth': 50.0,
        'n_bins': 500,
        'dose_bins': np.linspace(0, 50, 501)
    }

    # Initialize worker in main process for testing
    _init_worker('water', 0.05)

    print("   Transporting 10-particle chunk using chunk worker...")
    start = time.time()
    result = _transport_chunk_worker(work_item)
    elapsed = time.time() - start

    print(f"   Time: {elapsed:.6f}s")
    print(f"   Steps: {result['n_steps']}")
    print(f"   Final states: {len(result['final_states'])} particles")
    print(f"   Dose array shape: {result['dose_deposit'].shape}")

def profile_stopping_power():
    """Profile stopping power calculations."""
    print("\n" + "="*70)
    print("PROFILING STOPPING POWER CALCULATIONS")
    print("="*70)

    from coulomb_mc.physics.stopping_power import StoppingPower

    print("\n1. StoppingPower initialization:")
    start = time.time()
    sp = StoppingPower(material='water')
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.6f}s")

    print("\n2. Single stopping power lookup:")
    energy = 400.0
    A = 12
    Z = 6

    # Warm up
    sp.stopping_power_MeV_cm2_g(energy, A, Z)

    # Time many lookups
    n_lookups = 10000
    start = time.time()
    for _ in range(n_lookups):
        sp.stopping_power_MeV_cm2_g(energy, A, Z)
    elapsed = time.time() - start

    print(f"   {n_lookups} lookups: {elapsed:.6f}s")
    print(f"   Per lookup: {elapsed/n_lookups*1e6:.2f} μs")

if __name__ == '__main__':
    profile_stopping_power()
    check_chunk_worker()
    profile_serial_transport()
    profile_parallel_overhead()
