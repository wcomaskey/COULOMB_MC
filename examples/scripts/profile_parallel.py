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

def profile_single_particle():
    """Profile single particle transport in detail."""
    print("\n" + "="*70)
    print("PROFILING SINGLE PARTICLE TRANSPORT")
    print("="*70)

    # Create engine first
    engine = TransportEngine(material='water')

    # Create single particle beam
    beam = engine.create_beam('C-12', 400.0, n_particles=1)

    # Time single particle transport with profiling
    print("\n1. Single particle transport:")

    profiler = cProfile.Profile()
    profiler.enable()
    start = time.time()

    result = engine._transport_single_particle(
        0,
        {
            'position': beam.particles['position'][0],
            'direction': beam.particles['direction'][0],
            'energy': beam.particles['energy'][0],
            'A': beam.particles['A'][0],
            'Z': beam.particles['Z'][0],
            'weight': beam.particles['weight'][0]
        },
        50.0
    )

    elapsed = time.time() - start
    profiler.disable()

    particle, dose_list = result
    print(f"   Time: {elapsed:.6f}s")
    print(f"   Steps: {len(dose_list)}")

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
    print(f"   Steps: {stats_parallel['total_steps']}")
    print(f"   Speedup: {serial_time/parallel_time:.2f}x")

def check_data_loading():
    """Check if NIST data is being loaded multiple times."""
    print("\n" + "="*70)
    print("CHECKING NIST DATA LOADING")
    print("="*70)

    from coulomb_mc.transport.engine import _worker_engine, _init_worker, _transport_particle_worker

    print("\n1. Testing worker initialization:")
    print("   Calling _init_worker...")
    start = time.time()
    _init_worker('water', 0.05)
    elapsed = time.time() - start
    print(f"   Time: {elapsed:.6f}s")
    print(f"   Global engine created: {_worker_engine is not None}")

    if _worker_engine is not None:
        print(f"   Engine material: {_worker_engine.material}")
        print(f"   Stopping power object: {_worker_engine.stopping_power}")

        # Check if stopping power data is loaded
        sp = _worker_engine.stopping_power
        print(f"   Data loaded: {hasattr(sp, 'energies_MeV_u')}")
        if hasattr(sp, 'energies_MeV_u'):
            print(f"   Energy points: {len(sp.energies_MeV_u) if sp.energies_MeV_u is not None else 'None'}")

    print("\n2. Testing worker function:")
    work_item = {
        'particle_id': 0,
        'position': np.array([0., 0., 0.]),
        'direction': np.array([0., 0., 1.]),
        'energy': 400.0,
        'A': 12,
        'Z': 6,
        'weight': 1.0,
        'max_depth': 50.0
    }

    print("   Transporting particle using global engine...")
    start = time.time()
    result = _transport_particle_worker(work_item)
    elapsed = time.time() - start
    particle, dose_list = result
    print(f"   Time: {elapsed:.6f}s")
    print(f"   Steps: {len(dose_list)}")

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
    check_data_loading()
    profile_single_particle()
    profile_parallel_overhead()
