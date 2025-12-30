# Parallelization Implementation Plan

## Objective

Implement particle-level parallelization to achieve 6-8x speedup on multi-core systems, enabling rapid prototyping and 3D simulations.

## Current Architecture

### Main Transport Loop (engine.py:408-520)

```python
while beam.n_alive > 0:
    # For each alive particle:
    for i in range(n_alive):
        # Calculate physics (stopping power, range, step size, scattering)
        ...

    # Score dose
    self._score_dose_adaptive(...)

    # Advance all particles
    transport_step_with_scattering_adaptive(...)
```

**Problem:** Step-level outer loop with particle inner loop. Synchronization barrier at every step.

## Target Architecture

### Particle-Level Parallelization

```python
def transport_single_particle(particle_data, max_depth, ...):
    """Transport one particle from birth to death."""
    while particle.alive and particle.depth < max_depth:
        # Calculate physics for this particle
        # Take step
        # Score dose (thread-safe)
    return particle, dose_contributions

# Parallel execution
with multiprocessing.Pool(n_cores) as pool:
    results = pool.map(transport_single_particle, particle_list)
```

**Benefit:** No synchronization until all particles complete. Natural OpenMP-style parallelization.

## Implementation Strategy

### Phase 1: Refactor for Particle-Level (Current Session)

**Goal:** Restructure transport loop to isolate single-particle transport

**Tasks:**
1. Extract single-particle transport logic into standalone function
2. Make dose scoring thread-safe (atomic operations or thread-local accumulation)
3. Test that refactored code produces identical results

**Files to modify:**
- `coulomb_mc/transport/engine.py` - Refactor transport() method
- Create new method: `_transport_single_particle()`

### Phase 2: Multiprocessing Implementation

**Goal:** Implement CPU parallelization with Python multiprocessing

**Approach:**
- Use `multiprocessing.Pool` for particle batches
- Share stopping power data (read-only) across processes
- Accumulate dose arrays from each process

**Expected performance:** 6-8x on 8 cores (75-90% efficiency)

### Phase 3: Numba Parallel (Optional)

**Goal:** Try Numba parallel with proper particle-level structure

**Note:** Previously failed with step-level parallelization. May work better with particle-level.

### Phase 4: GPU Preparation (Future)

**Goal:** Design with GPU in mind

**Considerations:**
- Particle independence (already have)
- Minimize data transfer (batch operations)
- Atomic dose scoring

## Detailed Implementation

### Step 1: Extract Single-Particle Transport

Create new method in TransportEngine:

```python
def _transport_single_particle(self,
                               particle_id: int,
                               initial_state: dict,
                               max_depth: float) -> Tuple[dict, np.ndarray]:
    """
    Transport a single particle from initialization to death.

    Parameters:
        particle_id: Particle identifier
        initial_state: Initial particle state (position, direction, energy, A, Z)
        max_depth: Maximum depth cutoff [cm]

    Returns:
        final_state: Final particle state
        dose_contributions: Array of (x, y, z, dose) tuples
    """
    # Create single particle array
    particle = np.zeros(1, dtype=PARTICLE_DTYPE)
    particle['alive'][0] = True
    particle['position'][0] = initial_state['position']
    particle['direction'][0] = initial_state['direction']
    particle['energy'][0] = initial_state['energy']
    particle['A'][0] = initial_state['A']
    particle['Z'][0] = initial_state['Z']

    dose_list = []

    while particle['alive'][0] and particle['position'][0, 2] < max_depth:
        # Calculate stopping power
        sp = self.stopping_power.stopping_power_MeV_cm2_g(...)

        # Calculate step size
        step = calculate_adaptive_step_size(...)

        # Calculate energy loss
        energy_loss = sp * self.density * step / particle['A'][0]

        # Calculate scattering
        theta_rms = highland_angle(...)

        # Record dose contribution
        dose_list.append({
            'position': particle['position'][0].copy(),
            'energy_deposited': energy_loss * particle['A'][0],
            'step_length': step
        })

        # Advance particle
        transport_step_with_scattering_adaptive(
            particle,
            np.array([step]),
            np.array([energy_loss]),
            np.array([theta_rms])
        )

    return particle, dose_list
```

### Step 2: Parallel Dispatch

```python
def transport_parallel(self, beam: ParticleArray, max_depth: float,
                      n_processes: int = None, verbose: bool = True):
    """
    Transport particles in parallel using multiprocessing.

    Parameters:
        beam: ParticleArray with initial particle states
        max_depth: Maximum depth to transport [cm]
        n_processes: Number of parallel processes (default: cpu_count)
        verbose: Print progress
    """
    import multiprocessing as mp

    if n_processes is None:
        n_processes = mp.cpu_count()

    # Prepare initial states
    n_particles = len(beam.particles)
    initial_states = []
    for i in range(n_particles):
        initial_states.append({
            'position': beam.particles['position'][i].copy(),
            'direction': beam.particles['direction'][i].copy(),
            'energy': beam.particles['energy'][i],
            'A': beam.particles['A'][i],
            'Z': beam.particles['Z'][i]
        })

    # Create worker function
    def worker(args):
        particle_id, state = args
        return self._transport_single_particle(particle_id, state, max_depth)

    # Parallel execution
    if verbose:
        print(f"Transporting {n_particles} particles on {n_processes} cores...")

    with mp.Pool(n_processes) as pool:
        results = pool.map(worker, enumerate(initial_states))

    # Accumulate dose from all particles
    self._accumulate_dose_from_results(results)

    # Update beam with final states
    for i, (final_particle, dose_contrib) in enumerate(results):
        beam.particles[i] = final_particle[0]

    return {
        'n_particles': n_particles,
        'n_alive': np.sum(beam.particles['alive']),
        'total_steps': sum(len(d) for _, d in results)
    }
```

### Step 3: Thread-Safe Dose Scoring

**Option A: Thread-local accumulation (recommended)**

Each process maintains its own dose array, merge at end:

```python
def _accumulate_dose_from_results(self, results):
    """Accumulate dose contributions from all particles."""
    for particle_state, dose_list in results:
        for dose_event in dose_list:
            # Find voxel indices
            ix = int((dose_event['position'][0] + self.dose_grid_half_width) / self.dose_voxel_size)
            iy = int((dose_event['position'][1] + self.dose_grid_half_width) / self.dose_voxel_size)
            iz = int(dose_event['position'][2] / self.dose_voxel_size)

            # Bounds check
            if 0 <= ix < self.dose_grid_size and \
               0 <= iy < self.dose_grid_size and \
               0 <= iz < self.dose_nbins:
                # Accumulate dose
                self.dose_grid[ix, iy, iz] += dose_event['energy_deposited']
```

**Option B: Atomic operations with shared memory (GPU prep)**

Use numpy array in shared memory with locks for accumulation.

## Performance Targets

### Current (Serial)
- 10,000 particles: 44 seconds
- Rate: 230 particles/sec

### Target (8-core CPU)
- 10,000 particles: 5-6 seconds
- Rate: 1,700-2,000 particles/sec
- Speedup: 7-8x
- Efficiency: 85-95%

### Stretch Target (GPU)
- 100,000 particles: 1-2 seconds
- 10,000,000 particles: 2-3 minutes
- Speedup: 100-500x vs serial

## Testing Strategy

### Correctness Tests
1. Single particle: Compare parallel vs serial results
2. Physics validation: Range, Bragg peak position unchanged
3. Dose distribution: Statistical agreement (same particles, same seed)

### Performance Tests
1. Scaling: 1, 2, 4, 8 cores
2. Efficiency: Measure parallel overhead
3. Large batches: 10^5, 10^6 particles

### Benchmarks
```python
# Test script
import time
from coulomb_mc.transport.engine import TransportEngine

engine = TransportEngine(material='water')

# Serial baseline
beam = engine.create_beam('C-12', 400.0, n_particles=10000)
start = time.time()
stats = engine.transport(beam, max_depth=30.0)
serial_time = time.time() - start

# Parallel
for n_cores in [1, 2, 4, 8]:
    beam = engine.create_beam('C-12', 400.0, n_particles=10000)
    start = time.time()
    stats = engine.transport_parallel(beam, max_depth=30.0, n_processes=n_cores)
    parallel_time = time.time() - start

    speedup = serial_time / parallel_time
    efficiency = speedup / n_cores * 100

    print(f"{n_cores} cores: {parallel_time:.1f}s, {speedup:.2f}x speedup, {efficiency:.1f}% efficient")
```

## Implementation Order

1. Create `_transport_single_particle()` method
2. Test single-particle transport matches current results
3. Implement `transport_parallel()` with multiprocessing
4. Test parallel correctness (same results as serial)
5. Benchmark parallel performance (aim for 7-8x on 8 cores)
6. Integrate into main `transport()` method with flag: `parallel=True`

## Notes

- Keep serial version as default for compatibility
- Parallel version as opt-in: `transport(beam, parallel=True, n_processes=8)`
- Document performance characteristics
- Profile to identify remaining bottlenecks
- Consider shared memory for stopping power data to reduce memory footprint

## Next Session Goals

1. Implement `_transport_single_particle()`
2. Implement `transport_parallel()` with multiprocessing
3. Validate correctness
4. Benchmark on 8-core system
5. Document performance results
