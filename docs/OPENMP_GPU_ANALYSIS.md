# OpenMP-Style Parallelization & GPU Strategy Analysis

## OpenMP for Python: Options

### Option 1: Numba with Threading (OpenMP-like)
```python
@numba.njit(parallel=True, fastmath=True)
def transport_batch(particles, ...):
    for i in numba.prange(len(particles)):  # OpenMP-like parallel loop
        # Each thread processes one particle completely
        while particle[i] is alive:
            step(particle[i])
```

**Issue:** Our current code parallelizes the STEP, not the PARTICLE
- Current: All particles take one step in parallel → bad
- Needed: Each particle runs full simulation in parallel → good

**Fix:** Restructure to particle-level outer loop

### Option 2: Python multiprocessing
```python
with multiprocessing.Pool(n_cpus) as pool:
    results = pool.map(transport_single_particle, particle_list)
```

**Pros:** True parallelism, no GIL
**Cons:** Process overhead, can't share memory easily

### Option 3: Numba.cuda (GPU)
```python
@numba.cuda.jit
def transport_kernel(particles, ...):
    i = cuda.grid(1)  # Thread index
    if i < len(particles):
        while particles[i].alive:
            step(particles[i])
```

**Perfect for GPU:** 1 thread = 1 particle, no synchronization needed

---

## Stopping Power Data Loading Analysis

### Current Implementation
```python
# In StoppingPower.__init__:
self.data = np.loadtxt(filename)  # Loads from disk every time
```

**Performance:**
- File I/O: ~10-50ms per load
- With multiprocessing (8 processes): 8 × 50ms = 400ms overhead
- Repeated for each batch!

### Optimization Strategies

#### Strategy 1: Preload to Shared Memory
```python
# Global shared data (loaded once)
NIST_DATA_CACHE = {}

def load_nist_data_once(material):
    if material not in NIST_DATA_CACHE:
        NIST_DATA_CACHE[material] = np.loadtxt(...)
    return NIST_DATA_CACHE[material]
```

**Speedup:** Eliminates redundant loads in multiprocessing

#### Strategy 2: Compile into Binary Array
```python
# Pre-convert ASCII → NumPy binary at install time
np.save('data/STOPRANGE_NIST_Proton.npy', data)

# Fast load (100x faster than loadtxt)
data = np.load('data/STOPRANGE_NIST_Proton.npy')
```

**Speedup:** 10-50ms → 0.5ms (100x faster)

#### Strategy 3: Embed in Module (for GPU)
```python
# Convert to constant arrays in code
PROTON_ENERGIES = np.array([0.001, 0.0015, ...])
PROTON_STOPPING_POWER = np.array([...])

# Or use @numba.cuda.const for GPU constant memory
```

**Speedup:** Zero load time, can use GPU constant memory

#### Strategy 4: GPU Texture Memory (Best for GPU)
```cuda
// CUDA texture memory for 2D interpolation
texture<float, 2, cudaReadModeElementType> tex_stopping_power;

__device__ float lookup_stopping_power(float energy, int Z) {
    // Hardware-accelerated 2D interpolation
    return tex2D(tex_stopping_power, energy, Z);
}
```

**Speedup:** Hardware interpolation, cached, very fast

---

## Recommended Implementation Plan

### Phase 1: Immediate (This Session)

**1. Convert NIST data to binary format**
```bash
# Create conversion script
python scripts/convert_nist_data.py
# Creates .npy files (100x faster loading)
```

**2. Restructure for particle-level parallelization**
```python
@numba.njit(fastmath=True)  # Remove parallel=True
def transport_single_particle(particle, material_data, max_depth):
    """Transport one particle from birth to death."""
    while particle.alive and particle.position[2] < max_depth:
        # Calculate step size
        # Apply physics
        # Update particle
    return particle

@numba.njit(parallel=True, fastmath=True)  
def transport_all_particles(particles, material_data, max_depth):
    """Transport all particles in parallel (OpenMP-style)."""
    for i in numba.prange(len(particles)):  # True particle parallelism
        particles[i] = transport_single_particle(particles[i], material_data, max_depth)
```

**Expected speedup:** 6-8x on 8 cores (75-100% efficiency)

### Phase 2: CPU Optimization (Week 3)

**3. Add multiprocessing fallback for huge simulations**
- Use when particles > 100,000
- Each process handles batch of particles
- Shared memory for NIST data

**4. Profile and optimize interpolation**
- Current: Linear interpolation
- Consider: Cubic spline (pre-computed)
- Or: Lookup table with fine grid

### Phase 3: GPU Implementation (Week 7-8)

**5. Port to CUDA**
```python
import numba.cuda as cuda

@cuda.jit
def transport_gpu_kernel(particles, stopping_power_texture, dose_grid, max_depth):
    """GPU kernel: 1 thread per particle."""
    idx = cuda.grid(1)
    if idx < particles.shape[0]:
        # Transport particle idx
        while particles[idx].alive and particles[idx].position[2] < max_depth:
            # Use texture memory for stopping power lookup
            sp = tex2D(stopping_power_texture, particles[idx].energy, particles[idx].Z)
            # Update particle
            # Atomic add to dose grid
        
def transport_gpu(particles, material, max_depth):
    """Wrapper for GPU transport."""
    # Allocate GPU memory
    d_particles = cuda.to_device(particles)
    
    # Configure kernel
    threads_per_block = 256
    blocks = (len(particles) + threads_per_block - 1) // threads_per_block
    
    # Launch kernel
    transport_gpu_kernel[blocks, threads_per_block](d_particles, ...)
    
    # Copy results back
    particles = d_particles.copy_to_host()
```

**Expected speedup:** 100-1000x vs single CPU core

---

## Detailed Implementation Steps

### Step 1: Convert NIST Data to Binary

Create `scripts/convert_nist_data.py`:
```python
import numpy as np
from pathlib import Path

data_dir = Path('data')
for dat_file in data_dir.glob('*.DAT'):
    print(f"Converting {dat_file.name}...")
    
    # Load ASCII data
    data = np.loadtxt(dat_file)
    
    # Save as binary
    npy_file = dat_file.with_suffix('.npy')
    np.save(npy_file, data)
    
    # Verify
    loaded = np.load(npy_file)
    assert np.allclose(data, loaded)
    print(f"  ✓ Saved to {npy_file.name}")
```

### Step 2: Update StoppingPower class

```python
class StoppingPower:
    def __init__(self, material='water'):
        # Try binary first, fall back to ASCII
        npy_file = data_path / f'{filename}.npy'
        dat_file = data_path / f'{filename}.DAT'
        
        if npy_file.exists():
            self.data = np.load(npy_file)  # 100x faster!
        elif dat_file.exists():
            self.data = np.loadtxt(dat_file)
            # Cache for next time
            np.save(npy_file, self.data)
        else:
            raise FileNotFoundError(...)
```

### Step 3: Restructure Transport Loop

Current (bad):
```python
for step in range(max_steps):  # Outer loop: steps
    for particle in numba.prange(particles):  # Inner loop: particles (parallel)
        update_one_step(particle)  # All particles synchronized!
```

New (good):
```python
for particle in numba.prange(particles):  # Outer loop: particles (parallel)
    while particle.alive:  # Inner loop: steps for this particle
        update_one_step(particle)  # Each particle independent!
```

This is true OpenMP-style: each thread processes one particle completely.

---

## GPU Compatibility Analysis

### What Works on GPU (CUDA):

✅ **Particle-level parallelization**
- 1 thread = 1 particle
- No synchronization between particles
- Perfect for GPU (embarrassingly parallel)

✅ **Stopping power lookup**
- Texture memory for 2D interpolation
- Constant memory for small tables
- Very fast on GPU

✅ **Random number generation**
- CUDA has cuRAND library
- Each thread has independent RNG state
- No conflicts

✅ **Atomic dose scoring**
- `atomicAdd()` for dose grid updates
- Race conditions acceptable (Monte Carlo)
- Or use shared memory reduction

### What Doesn't Work on GPU:

❌ **Python multiprocessing**
- GPU uses CUDA threads, not OS processes

❌ **File I/O during transport**
- Must load all data to GPU memory first

❌ **Dynamic memory allocation**
- Pre-allocate all arrays before kernel launch

### GPU Implementation Strategy:

1. **Pre-load everything to GPU:**
   ```python
   d_particles = cuda.to_device(particles)
   d_stopping_power = cuda.to_device(stopping_power_table)
   d_dose = cuda.to_device(dose_grid)
   ```

2. **Launch kernel with 1 thread per particle:**
   ```python
   threads = 256  # Typical GPU block size
   blocks = (n_particles + threads - 1) // threads
   transport_kernel[blocks, threads](d_particles, ...)
   ```

3. **Each thread runs independently:**
   ```cuda
   idx = cuda.grid(1)
   while particles[idx].alive:
       step(particles[idx])
   ```

4. **Copy results back:**
   ```python
   particles = d_particles.copy_to_host()
   dose = d_dose.copy_to_host()
   ```

---

## Performance Predictions

### Current State
- **10,000 particles:** 158s (63 particles/sec)
- **Bottleneck:** Wrong parallelization + slow data loading

### After Binary Data Loading
- **Expected:** 140s (71 particles/sec, +13% improvement)
- **Reason:** Eliminates 10-20ms file I/O overhead

### After Particle-Level Parallelization (OpenMP-style)
- **Expected:** 20s (500 particles/sec, 8x improvement)
- **Reason:** True parallelism, 8 cores × ~100% efficiency

### After GPU Implementation
- **Expected:** 0.1s (100,000 particles/sec, 1,500x improvement)
- **Reason:** Thousands of parallel threads

### For 10^7 Particles:

| Method | Time | Throughput |
|--------|------|------------|
| Current | 44 hours | 63/s |
| Binary data | 39 hours | 71/s |
| OpenMP-style | 5.5 hours | 500/s |
| GPU | 100 seconds | 100,000/s |

---

## Implementation Priority

### This Session (Next 2 hours):
1. ✅ Convert NIST data to binary (.npy)
2. ✅ Update StoppingPower to use binary
3. ✅ Restructure transport loop for particle-level parallelization
4. ✅ Test and verify physics accuracy
5. ✅ Benchmark speedup

### Week 3:
- Add batch processing for 10^6+ particles
- Optimize interpolation (spline or fine grid)
- Profile GPU requirements

### Week 7-8:
- Implement CUDA kernels
- Texture memory for stopping power
- Atomic dose scoring
- Benchmark vs CPU

---

*Analysis Date: 2025-12-29*
*Target: OpenMP-style particle parallelization + GPU readiness*
