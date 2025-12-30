# 3D Radiation Monte Carlo Development Plan

## Executive Summary

This plan outlines the development of a high-performance 3D radiation transport Monte Carlo program for heavy-ion therapy and radiation physics applications. The program will simulate charged particle transport through matter, including Coulomb scattering, energy loss, nuclear fragmentation, and energy straggling effects, with efficient CPU and GPU parallelization.

---

## 1. Current State Analysis

### 1.1 Existing Assets

**Python Implementation (`Coulomb_Multiple.py` and notebooks)**
- Basic Monte Carlo framework for multiple Coulomb scattering
- Molière scattering theory implementation
- NIST stopping power data integration
- 3D particle tracking with direction vectors
- Energy loss via LET (Linear Energy Transfer)
- Step-based transport algorithm
- Single particle trajectory output

**Fortran Implementation (`COULOMB_LET.f90`)**
- More complete physics modules:
  - `precision_mod`: Double precision management
  - `target_data`: Material composition handling (10 materials)
  - `let_module`: Stopping power calculations using NIST data
- Vector rotation using Rodrigues' formula
- Multiple Coulomb scattering routines
- OpenMP parallelization directives (incomplete)
- Material database support for 10 materials

**Data Resources**
- **NIST Stopping Power Data**:
  - `Pstar_NIST_Water.dat`: Proton stopping powers
  - `StopRange_NIST_Proton.dat`: Proton range tables
  - `StopRange_NIST_Helium.dat`: Helium range tables
- **Fragmentation Data**:
  - `qmsfrg_190_Dbase.dat`: Nuclear fragmentation database (10 MB)
  - `qmsfrg_190_TargFrag.dat`: Target fragmentation cross-sections
- **DPASS Integration**:
  - DPASS2106: Stopping power database program
  - Batch mode interface for automated queries
  - Database file: `DPASS_DB.dat` (6.8 MB)

**Analysis Modules**
- `Stopping_Power_Heavy-Ion11.py`: Heavy-ion stopping power models
  - Effective charge formulation
  - Weighted proton/helium model fitting
  - Energy-dependent weighting functions

### 1.2 Identified Gaps

**Physics**
- No energy straggling implementation
- Limited nuclear fragmentation modeling
- No secondary particle generation
- Simplified LET calculation (needs Bethe-Bloch)
- Missing charge-state dynamics for heavy ions

**Performance**
- No GPU implementation
- Incomplete OpenMP parallelization
- Inefficient file I/O (writing per step)
- No vectorization optimization
- Memory allocation not optimized

**Software Architecture**
- Mixed Python/Fortran without clear integration
- No build system
- No unit testing framework
- Limited error handling
- Hard-coded parameters

---

## 2. Development Roadmap

### Phase 1: Core Physics Engine (Months 1-3)

#### 1.1 Unified Physics Module Design
**Priority: Critical**

**Objectives:**
- Create modular physics engine in modern Fortran (2008+) or C++
- Implement all transport physics with validated algorithms
- Establish clear API for physics modules

**Tasks:**
1. **Material Database System**
   - Extend material support beyond 10 predefined materials
   - Support compound materials with arbitrary composition
   - Interface with NIST/ICRU databases
   - Implement density effect corrections

2. **Stopping Power Module**
   - Implement full Bethe-Bloch formula with shell corrections
   - Integrate DPASS database for proton/alpha stopping
   - Implement heavy-ion effective charge models (Barkas, Bloch)
   - Add charge-changing cross-sections
   - Validate against SRIM, ICRU reports

3. **Energy Straggling Module**
   - Implement Bohr straggling theory
   - Add Landau-Vavilov distributions for thin absorbers
   - Include restricted energy loss fluctuations
   - Correlation with range straggling

4. **Multiple Coulomb Scattering**
   - Refine Molière theory implementation
   - Add Highland approximation for speed
   - Implement Gaussian approximation for small angles
   - Add correlation between scattering and energy loss

5. **Nuclear Fragmentation Module**
   - Integrate QMSFRG database (already available)
   - Implement projectile fragmentation models:
     - Abrasion-ablation model
     - Statistical multifragmentation
   - Target fragmentation and recoils
   - Secondary particle generation and tracking
   - Fragment charge/mass distributions

**Deliverables:**
- Validated physics modules with unit tests
- Physics validation report against experimental data
- API documentation

---

### Phase 2: Geometry and Scoring (Month 2-3)

#### 2.1 Geometry Engine
**Priority: High**

**Objectives:**
- 3D voxelized geometry representation
- Efficient ray-tracing for particle transport
- Support for CT-based patient geometries

**Tasks:**
1. **Voxel Grid Implementation**
   - Regular Cartesian grid structure
   - Material indexing per voxel
   - Density variation support
   - Memory-efficient storage (compressed grids)

2. **Ray-Tracing Algorithm**
   - DDA (Digital Differential Analyzer) for voxel traversal
   - Boundary crossing detection
   - Adaptive step sizing near boundaries
   - Optimization for coherent rays (beam transport)

3. **Geometry I/O**
   - DICOM CT import capability
   - HU to material/density conversion
   - Simple geometric primitives (boxes, cylinders, spheres)
   - Output geometry visualization (VTK format)

#### 2.2 Scoring System
**Priority: High**

**Tasks:**
1. **Dose Deposition**
   - 3D dose grid (separate from geometry grid)
   - Track-length estimator for dose
   - Collision kerma approximation
   - Statistical uncertainty tracking (history-by-history)

2. **LET Scoring**
   - Dose-averaged LET (LETd)
   - Track-averaged LET (LETt)
   - LET spectra in voxels
   - Restricted LET calculations

3. **Fluence and Particle Spectra**
   - Energy-resolved particle fluence
   - Angular distributions
   - Charge/mass distributions for fragments

4. **Radiobiological Quantities**
   - RBE-weighted dose (with model flexibility)
   - Microdosimetric distributions (optional)

**Deliverables:**
- Geometry module with test cases
- Scoring system with statistical analysis
- I/O routines for visualization

---

### Phase 3: Parallelization Strategy (Month 3-5)

#### 3.1 CPU Parallelization
**Priority: Critical**

**Approach: History-based parallelism**

**Design:**
1. **OpenMP Thread-Level Parallelism**
   - Parallel over particle histories (embarrassingly parallel)
   - Thread-local scoring arrays
   - Reduction operation for final scoring
   - NUMA-aware memory allocation

2. **Vectorization (SIMD)**
   - Structure of Arrays (SoA) for particle data
   - Vectorized physics calculations where possible
   - Compiler intrinsics for critical sections
   - Auto-vectorization friendly loops

3. **Load Balancing**
   - Dynamic scheduling for variable-length histories
   - Chunk size optimization
   - Work stealing for imbalanced loads

**Implementation:**
```fortran
!$OMP PARALLEL DO PRIVATE(particle, local_dose) REDUCTION(+:global_dose) SCHEDULE(DYNAMIC)
do ihist = 1, nhistories
    call transport_particle(particle, geometry, physics, local_dose)
    global_dose = global_dose + local_dose
end do
!$OMP END PARALLEL DO
```

**Optimization Targets:**
- Cache-friendly data structures
- Minimize false sharing
- Reduce memory bandwidth requirements
- Profile-guided optimization

#### 3.2 GPU Parallelization
**Priority: High**

**Approach: Massive parallelism with careful memory management**

**Technology Options:**
1. **CUDA (NVIDIA)** - Best performance on NVIDIA GPUs
2. **HIP (AMD)** - Portable to AMD GPUs
3. **OpenMP 5.0+ offloading** - Portable but less mature
4. **SYCL** - Cross-platform, emerging standard

**Recommended: CUDA with abstraction layer for portability**

**Design:**
1. **Kernel Structure**
   - One thread per particle history
   - Warp-level coherence for similar particles
   - Shared memory for material data
   - Constant memory for physics tables

2. **Memory Management**
   - Coalesced global memory access
   - Texture memory for 3D geometry lookup
   - Shared memory for frequently accessed data
   - Page-locked host memory for transfers

3. **Random Number Generation**
   - cuRAND library for GPU RNG
   - Unique seed per thread
   - State stored in global memory

4. **Atomic Operations for Scoring**
   - Atomic adds for dose deposition
   - Consider binning strategies to reduce contention
   - Shared memory accumulation before global write

**Sample CUDA Kernel:**
```cuda
__global__ void transport_particles_gpu(
    ParticleState* particles,
    VoxelGeometry* geometry,
    PhysicsData* physics,
    float* dose_grid,
    int nparticles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= nparticles) return;

    ParticleState p = particles[tid];
    curandState rng_state = rng_states[tid];

    while (p.energy > cutoff_energy) {
        // Transport step
        float step_length = calculate_step(&p, geometry, physics);
        float energy_loss = calculate_energy_loss(&p, step_length, physics, &rng_state);

        // Score dose
        int voxel_idx = get_voxel_index(&p, geometry);
        atomicAdd(&dose_grid[voxel_idx], energy_loss);

        // Update particle state
        update_particle_state(&p, step_length, energy_loss, physics, &rng_state);
    }
}
```

**Optimization Strategy:**
- Occupancy optimization (threads per block)
- Minimize divergent branching
- Overlap computation with memory transfers
- Multi-GPU scaling with domain decomposition

#### 3.3 Hybrid CPU+GPU Approach
**Priority: Medium**

**Strategy:**
- CPU handles complex physics (fragmentation, rare events)
- GPU handles bulk transport (primary particles)
- Asynchronous execution with CUDA streams
- Dynamic load balancing between CPU/GPU

**Deliverables:**
- OpenMP parallelized CPU code
- CUDA GPU kernels
- Hybrid execution manager
- Performance benchmarking suite
- Scaling analysis (strong/weak scaling)

---

### Phase 4: Advanced Physics Implementation (Month 4-6)

#### 4.1 Energy Straggling
**Priority: High**

**Implementation:**
1. **Bohr Straggling** (Gaussian approximation)
   - Valid for thick absorbers
   - Fast sampling from normal distribution

2. **Landau-Vavilov Theory** (thin absorbers)
   - Asymmetric energy loss distribution
   - Tabulated Vavilov functions
   - Critical thickness determination

3. **Detailed Collision Sampling**
   - For very thin layers or high accuracy
   - δ-ray production with tracking
   - Restricted energy loss

**Validation:**
- Compare with analytical models
- Benchmark against GEANT4, FLUKA
- Experimental data comparison (Bragg peak width)

#### 4.2 Nuclear Fragmentation Enhancement
**Priority: High**

**Tasks:**
1. **QMSFRG Database Integration**
   - Parse binary database format
   - Interpolation in energy/atomic number
   - Cross-section lookup optimization
   - Validation against experimental data

2. **Fragment Generation**
   - Sample fragment charge/mass from distributions
   - Calculate fragment energy/direction
   - Implement angular distributions (longitudinal/transverse)
   - Account for energy-momentum conservation

3. **Secondary Particle Stacking**
   - Particle stack/queue management
   - Importance sampling for secondaries
   - Cutoff energies per particle type
   - Variance reduction techniques

4. **Neutron Production**
   - Simplified neutron transport (optional)
   - Dose contribution from neutrons
   - Or coupling with dedicated neutron code

**Physics Models:**
- Projectile fragmentation (primary beam)
- Target fragmentation (nuclear recoils)
- Coalescence model for light fragments
- Statistical decay of excited fragments

#### 4.3 Electromagnetic Processes
**Priority: Medium**

**Additional Processes:**
1. **δ-ray production** (energetic secondary electrons)
2. **Bremsstrahlung** (minor for ions, but present)
3. **Pair production** (at very high energies)
4. **Annihilation** (for positron-emitting fragments)

**Implementation:**
- Threshold energy for explicit simulation
- Condensed history for low-energy electrons
- Option to couple with electron/photon MC code

**Deliverables:**
- Complete straggling module with validation
- Fragmentation transport with QMSFRG
- Secondary particle tracking
- Physics validation report

---

### Phase 5: Software Architecture & I/O (Month 5-7)

#### 5.1 Modular Architecture
**Priority: Critical**

**Structure:**
```
COULOMB_MC/
├── src/
│   ├── core/
│   │   ├── particle.f90          # Particle state management
│   │   ├── geometry.f90          # Geometry engine
│   │   ├── random.f90            # RNG management
│   │   └── constants.f90         # Physical constants
│   ├── physics/
│   │   ├── stopping_power.f90    # Energy loss
│   │   ├── scattering.f90        # Coulomb scattering
│   │   ├── straggling.f90        # Energy straggling
│   │   ├── fragmentation.f90     # Nuclear reactions
│   │   └── cross_sections.f90    # Cross-section data
│   ├── transport/
│   │   ├── transport_cpu.f90     # CPU transport engine
│   │   ├── transport_gpu.cu      # GPU transport kernels
│   │   └── hybrid_manager.f90    # CPU/GPU coordination
│   ├── scoring/
│   │   ├── dose_scorer.f90       # Dose tallies
│   │   ├── let_scorer.f90        # LET tallies
│   │   └── spectrum_scorer.f90   # Particle spectra
│   └── io/
│       ├── input_parser.f90      # Input file parsing
│       ├── geometry_io.f90       # Geometry import
│       ├── output_writer.f90     # Results output
│       └── checkpoint.f90        # Simulation checkpointing
├── data/
│   ├── materials/                # Material database
│   ├── physics_tables/           # Pre-computed physics tables
│   └── cross_sections/           # QMSFRG and other data
├── tests/
│   ├── unit/                     # Unit tests per module
│   ├── integration/              # Integration tests
│   └── validation/               # Physics validation
├── benchmarks/                   # Performance benchmarks
├── examples/                     # Example input files
└── docs/                        # Documentation
```

#### 5.2 Input System
**Priority: High**

**Input File Format (JSON or Namelist):**
```json
{
  "simulation": {
    "name": "carbon_water_phantom",
    "nhistories": 1000000,
    "random_seed": 12345,
    "transport_mode": "hybrid",
    "n_threads": 16,
    "gpu_device": 0
  },
  "beam": {
    "particle": "C-12",
    "energy_MeV": 400.0,
    "energy_spread_MeV": 0.5,
    "position": [0.0, 0.0, -10.0],
    "direction": [0.0, 0.0, 1.0],
    "angular_spread_deg": 0.5,
    "beam_size_cm": 5.0
  },
  "geometry": {
    "type": "voxel",
    "file": "water_phantom.vox",
    "bounds": [[-15, 15], [-15, 15], [0, 30]],
    "voxel_size_mm": [2.5, 2.5, 2.5]
  },
  "physics": {
    "energy_cutoff_MeV": 0.1,
    "enable_fragmentation": true,
    "enable_straggling": true,
    "enable_delta_rays": false,
    "msc_algorithm": "moliere"
  },
  "scoring": {
    "dose_grid_mm": [2.5, 2.5, 2.5],
    "score_LET": true,
    "score_fluence": true,
    "score_fragments": true,
    "output_format": "hdf5"
  }
}
```

#### 5.3 Output System
**Priority: High**

**Output Formats:**
1. **HDF5** (recommended for large data)
   - Hierarchical data organization
   - Compression support
   - Parallel I/O capability
   - Metadata support

2. **VTK/VTU** (visualization)
   - Direct import to ParaView
   - 3D dose distributions
   - Vector fields (fluence directions)

3. **CSV/ASCII** (simple analysis)
   - Depth-dose profiles
   - LET distributions
   - Summary statistics

**Output Contents:**
- 3D dose distribution
- 3D LET distribution
- Particle fluence spectra
- Fragment yields
- Statistical uncertainties
- Simulation metadata

#### 5.4 Build System
**Priority: High**

**CMake Build System:**
```cmake
cmake_minimum_required(VERSION 3.18)
project(COULOMB_MC Fortran CUDA)

# Options
option(USE_OPENMP "Enable OpenMP" ON)
option(USE_CUDA "Enable CUDA GPU support" ON)
option(USE_HDF5 "Enable HDF5 output" ON)
option(BUILD_TESTS "Build unit tests" ON)

# Find dependencies
find_package(OpenMP REQUIRED)
find_package(HDF5 COMPONENTS Fortran REQUIRED)
if(USE_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# Compiler flags
set(CMAKE_Fortran_FLAGS "-O3 -march=native")
set(CMAKE_CUDA_FLAGS "-O3 -arch=sm_70")

# Add subdirectories
add_subdirectory(src)
add_subdirectory(tests)
```

**Deliverables:**
- Complete modular source tree
- Input/output system
- CMake build system
- User manual

---

### Phase 6: Validation & Optimization (Month 6-8)

#### 6.1 Physics Validation
**Priority: Critical**

**Validation Tests:**
1. **Stopping Power**
   - Compare with ICRU reports
   - Validate against SRIM calculations
   - Experimental data comparison (various ions/materials)

2. **Range**
   - Bragg peak position accuracy
   - Range straggling width
   - Experimental range measurements

3. **Multiple Scattering**
   - Angular distribution measurements
   - Lateral dose spread
   - Benchmark against Molière theory

4. **Fragmentation**
   - Fragment yields vs. experimental data
   - Energy spectra of fragments
   - Angular distributions
   - Compare with other codes (PHITS, FLUKA)

5. **Dose Distributions**
   - Depth-dose curves (pristine Bragg peaks)
   - Spread-out Bragg peaks (SOBP)
   - Lateral dose profiles
   - Fragment dose contribution
   - Comparison with clinical treatment planning systems

**Validation Database:**
- ICRU Report 73, 90 (heavy ions)
- GSI experimental data (carbon therapy)
- NIRS/HIMAC data (various ions)
- Published benchmark comparisons

#### 6.2 Performance Optimization
**Priority: High**

**Profiling:**
- CPU: Intel VTune, gprof, perf
- GPU: NVIDIA Nsight, nvprof
- Identify hotspots
- Memory bandwidth analysis

**Optimization Targets:**
1. **Physics Table Interpolation**
   - Pre-compute physics tables on fine grids
   - Optimized interpolation (linear, spline)
   - Table lookup vectorization

2. **Memory Access Patterns**
   - Cache-friendly data layouts
   - Prefetching for geometry access
   - Minimize allocations in loops

3. **Algorithmic Improvements**
   - Adaptive step sizing
   - Variance reduction techniques
   - Importance sampling

4. **GPU Optimizations**
   - Kernel fusion
   - Shared memory optimization
   - Warp divergence reduction
   - Multi-GPU scaling

**Performance Targets:**
- CPU: >100k particles/second/core
- GPU: >10M particles/second (RTX 4090 class)
- Linear scaling to 64+ CPU cores
- Multi-GPU efficiency >85%

#### 6.3 Verification & Testing
**Priority: Critical**

**Test Suite:**
1. **Unit Tests**
   - Each physics module
   - Geometry operations
   - I/O routines
   - ~80% code coverage target

2. **Integration Tests**
   - End-to-end simulation workflows
   - CPU vs. GPU result consistency
   - Parallel vs. serial consistency

3. **Regression Tests**
   - Automated testing on code changes
   - Reference results database
   - Continuous integration (GitHub Actions, GitLab CI)

**Tools:**
- pFUnit (Fortran unit testing)
- Google Test (C++ components)
- pytest (Python interface if developed)

**Deliverables:**
- Validation report with benchmark comparisons
- Performance analysis document
- Comprehensive test suite
- Optimization guide

---

### Phase 7: User Interface & Documentation (Month 7-9)

#### 7.1 Command-Line Interface
**Priority: High**

**Features:**
- Input file validation
- Progress reporting
- Real-time statistics
- Checkpoint/restart capability
- Error handling and logging

**Example Usage:**
```bash
# Basic run
coulomb_mc input.json -o output_dir/

# Specify number of threads
coulomb_mc input.json -t 32 -o output_dir/

# GPU execution
coulomb_mc input.json --gpu 0 -o output_dir/

# Restart from checkpoint
coulomb_mc input.json --restart checkpoint.h5 -o output_dir/
```

#### 7.2 Python Interface (Optional)
**Priority: Medium**

**Capabilities:**
- Input file generation
- Result analysis and visualization
- Batch job management
- Integration with Jupyter notebooks

**Example:**
```python
import coulomb_mc as cmc

# Create simulation
sim = cmc.Simulation()
sim.set_beam(particle='C-12', energy=400, spread=0.5)
sim.set_geometry('water_phantom.vox')
sim.set_physics(fragmentation=True, straggling=True)
sim.set_scoring(dose=True, let=True)

# Run
results = sim.run(nhistories=1e6, nthreads=16)

# Analyze
results.plot_depth_dose()
results.plot_let_distribution()
results.export_to_vtk('dose.vtu')
```

#### 7.3 Documentation
**Priority: Critical**

**Documentation Components:**
1. **User Manual**
   - Installation instructions
   - Quick start guide
   - Input file reference
   - Output format description
   - Example workflows

2. **Theory Manual**
   - Physics models description
   - Algorithm documentation
   - Validation results
   - Limitations and approximations

3. **Developer Guide**
   - Code architecture
   - API reference
   - Contribution guidelines
   - Coding standards

4. **Tutorial Collection**
   - Basic simulations
   - Advanced features
   - Performance tuning
   - Visualization examples

**Tools:**
- Sphinx or Doxygen for code documentation
- Markdown for user guides
- Jupyter notebooks for tutorials

**Deliverables:**
- Complete documentation suite
- CLI and Python interface
- Tutorial materials

---

## 3. Physics Improvements - Detailed Analysis

### 3.1 Energy Straggling Implementation

**Current State:** Not implemented

**Physics Background:**
Energy loss fluctuations arise from the statistical nature of individual collisions. Critical for:
- Accurate range uncertainty
- Bragg peak width
- Dose distribution tails

**Implementation Strategy:**

**For Thick Absorbers (many collisions):**
```fortran
subroutine sample_energy_loss_straggling(particle, step_length, mean_loss, actual_loss)
    ! Bohr straggling - Gaussian approximation
    real(dp) :: bohr_variance, sigma

    ! Calculate Bohr variance
    bohr_variance = calculate_bohr_variance(particle, material, step_length)
    sigma = sqrt(bohr_variance)

    ! Sample from Gaussian
    actual_loss = mean_loss + random_normal(0.0_dp, sigma)

    ! Ensure physical (positive energy loss)
    actual_loss = max(0.0_dp, min(actual_loss, particle%energy))
end subroutine
```

**For Thin Absorbers:**
```fortran
subroutine sample_landau_vavilov(particle, step_length, mean_loss, actual_loss)
    ! Landau-Vavilov distribution (asymmetric)
    real(dp) :: kappa, lambda

    ! Calculate Landau parameter
    kappa = calculate_kappa(particle, material, step_length)

    if (kappa > 10.0) then
        ! Use Gaussian (Bohr)
        call sample_energy_loss_straggling(...)
    else if (kappa > 0.01) then
        ! Use Vavilov distribution (tabulated)
        call sample_vavilov(kappa, lambda, delta_energy)
    else
        ! Use full Landau
        call sample_landau(lambda, delta_energy)
    end if

    actual_loss = mean_loss + delta_energy
end subroutine
```

**Validation:**
- Compare Bragg peak width with measurements
- Check range straggling vs. analytical formulas
- Benchmark against GEANT4

### 3.2 Nuclear Fragmentation - Advanced Implementation

**Current State:** Data files available, not integrated

**QMSFRG Database Structure:**
- Projectile-target combinations
- Energy-dependent cross-sections
- Fragment charge/mass distributions
- Angular distributions

**Implementation:**

```fortran
module fragmentation_module
    type :: FragmentationEvent
        integer :: Z_fragment, A_fragment
        real(dp) :: energy, theta, phi
    end type

    contains

    subroutine sample_fragmentation(particle, material, fragment_stack)
        real(dp) :: sigma_total, sigma_frag, rng

        ! Get total cross-section
        sigma_total = get_total_cross_section(particle%Z, particle%A, &
                                               material%Z, particle%energy)

        ! Mean free path
        mfp = 1.0 / (material%number_density * sigma_total)

        ! Sample interaction point
        distance = -log(random_uniform()) * mfp

        if (distance < step_length) then
            ! Fragmentation occurs
            call generate_fragments(particle, material, fragment_stack)
        end if
    end subroutine

    subroutine generate_fragments(particle, material, fragments)
        ! Use QMSFRG database
        call query_qmsfrg_database(particle%Z, particle%A, particle%energy, &
                                    fragment_distribution)

        ! Sample number of fragments
        n_fragments = sample_fragment_multiplicity(fragment_distribution)

        ! Sample each fragment
        do i = 1, n_fragments
            fragment%Z = sample_fragment_charge(fragment_distribution)
            fragment%A = sample_fragment_mass(fragment%Z, fragment_distribution)
            fragment%energy = sample_fragment_energy(fragment%Z, fragment%A, &
                                                     particle%energy)

            ! Angular distribution (forward-peaked)
            call sample_fragment_angle(fragment%energy, fragment%theta, fragment%phi)

            ! Add to stack
            call push_particle(fragment_stack, fragment)
        end do
    end subroutine
end module
```

**QMSFRG Database Access:**
```fortran
subroutine load_qmsfrg_database()
    ! Read binary database file
    open(unit=10, file='qmsfrg_190_Dbase.dat', form='unformatted', &
         access='stream', status='old')

    ! Parse header
    read(10) n_projectiles, n_targets, n_energies

    ! Read cross-section tables
    allocate(cross_sections(n_projectiles, n_targets, n_energies))
    read(10) cross_sections

    ! Read fragment distributions
    ! ... (parse complex data structure)

    close(10)
end subroutine

function interpolate_cross_section(Z_proj, A_proj, energy) result(sigma)
    ! Log-log interpolation in energy
    ! Bilinear in Z, A if needed
end function
```

**Fragment Transport:**
- Each fragment treated as new primary
- Recursive transport with stack management
- Track genealogy for analysis

### 3.3 Charge-State Dynamics

**Importance:** Heavy ions undergo charge-changing processes

**Implementation:**
```fortran
subroutine update_charge_state(particle, material, step_length)
    real(dp) :: prob_capture, prob_loss

    ! Calculate charge-changing probabilities
    prob_capture = 1.0 - exp(-sigma_capture * material%electron_density * step_length)
    prob_loss = 1.0 - exp(-sigma_loss * material%electron_density * step_length)

    if (random_uniform() < prob_capture) then
        particle%charge = particle%charge - 1
    else if (random_uniform() < prob_loss) then
        particle%charge = particle%charge + 1
    end if

    ! Update effective charge for stopping power
    particle%Z_eff = calculate_effective_charge(particle)
end subroutine
```

---

## 4. Parallelization - Detailed Strategy

### 4.1 CPU Parallelization - Best Practices

**Thread Safety:**
```fortran
module transport_engine
    use omp_lib
    implicit none

    ! Thread-local accumulators
    !$OMP THREADPRIVATE(local_dose, local_let)
    real(dp), allocatable :: local_dose(:,:,:)
    real(dp), allocatable :: local_let(:,:,:)

contains

    subroutine initialize_parallel_transport(n_threads)
        !$OMP PARALLEL
        ! Allocate thread-local arrays
        allocate(local_dose(nx, ny, nz))
        allocate(local_let(nx, ny, nz))
        local_dose = 0.0_dp
        local_let = 0.0_dp
        !$OMP END PARALLEL
    end subroutine

    subroutine transport_histories_parallel(n_histories)
        integer :: i

        !$OMP PARALLEL DO SCHEDULE(DYNAMIC, 1000) PRIVATE(i)
        do i = 1, n_histories
            call transport_single_particle(i)
        end do
        !$OMP END PARALLEL DO

        ! Reduction
        !$OMP PARALLEL
        !$OMP CRITICAL
        global_dose = global_dose + local_dose
        global_let = global_let + local_let
        !$OMP END CRITICAL
        !$OMP END PARALLEL
    end subroutine

    subroutine transport_single_particle(ihist)
        type(Particle) :: p

        ! Initialize particle from source
        call initialize_particle_from_beam(p, ihist)

        ! Transport loop
        do while (p%energy > cutoff_energy .and. p%is_alive)
            call transport_step(p, local_dose, local_let)
        end do
    end subroutine
end module
```

**Random Number Generation:**
- Each thread needs independent RNG stream
- Use MKL VSL or similar with stream-based RNG

```fortran
module random_module
    use mkl_vsl
    type(VSL_STREAM_STATE) :: stream
    !$OMP THREADPRIVATE(stream)

    subroutine init_random_parallel(base_seed)
        integer :: tid, seed
        !$OMP PARALLEL PRIVATE(tid, seed)
        tid = omp_get_thread_num()
        seed = base_seed + tid * 1000000
        call vslnewstream(stream, VSL_BRNG_MT19937, seed)
        !$OMP END PARALLEL
    end subroutine
end module
```

### 4.2 GPU Parallelization - Advanced Techniques

**Memory Hierarchy Usage:**

```cuda
// Constant memory for physics data (64 KB limit)
__constant__ MaterialData materials[100];
__constant__ PhysicsTable stopping_power_table;

// Texture memory for 3D geometry (optimized for spatial locality)
texture<uchar, 3, cudaReadModeElementType> geometry_tex;

// Shared memory for reduction operations
__shared__ float shared_dose[256];

__global__ void transport_kernel(
    ParticleState* particles,
    ScoringGrid* scoring,
    int n_particles
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_particles) return;

    // Local copy in registers
    ParticleState p = particles[tid];
    curandState rng = rng_states[tid];

    float local_dose_accumulator = 0.0f;

    // Transport loop
    while (p.energy > CUTOFF) {
        // Calculate step
        float step = compute_step_length(&p, &rng);

        // Energy loss
        float dE = compute_energy_loss(&p, step, &rng);
        local_dose_accumulator += dE;

        // Update position
        p.x += step * p.dx;
        p.y += step * p.dy;
        p.z += step * p.dz;

        // Check geometry using texture
        int mat_idx = tex3D(geometry_tex, p.x, p.y, p.z);

        // Scattering
        scatter_particle(&p, step, materials[mat_idx], &rng);

        // Update energy
        p.energy -= dE;
    }

    // Score dose using atomic operation
    int voxel_idx = get_voxel_index(p.x, p.y, p.z);
    atomicAdd(&scoring->dose[voxel_idx], local_dose_accumulator);
}
```

**Optimization: Reduce Atomic Contention**

```cuda
__global__ void transport_kernel_optimized(...) {
    __shared__ float shared_dose[DOSE_BINS];

    // Initialize shared memory
    if (threadIdx.x < DOSE_BINS) {
        shared_dose[threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // Transport and accumulate in shared memory
    // ... transport loop ...

    // Reduce within block
    atomicAdd(&shared_dose[local_bin], energy_deposit);

    __syncthreads();

    // Write to global memory (fewer atomic ops)
    if (threadIdx.x < DOSE_BINS) {
        atomicAdd(&global_dose[blockIdx.x * DOSE_BINS + threadIdx.x],
                  shared_dose[threadIdx.x]);
    }
}
```

**Multi-GPU Strategy:**

```cpp
void run_multi_gpu(int n_gpus, int total_histories) {
    std::vector<cudaStream_t> streams(n_gpus);

    // Divide work among GPUs
    int histories_per_gpu = total_histories / n_gpus;

    #pragma omp parallel for num_threads(n_gpus)
    for (int gpu = 0; gpu < n_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaStreamCreate(&streams[gpu]);

        // Allocate device memory
        ParticleState* d_particles;
        cudaMalloc(&d_particles, sizeof(ParticleState) * histories_per_gpu);

        // Launch kernel asynchronously
        int grid_size = (histories_per_gpu + 255) / 256;
        transport_kernel<<<grid_size, 256, 0, streams[gpu]>>>(
            d_particles, scoring_grids[gpu], histories_per_gpu
        );
    }

    // Synchronize and combine results
    for (int gpu = 0; gpu < n_gpus; gpu++) {
        cudaSetDevice(gpu);
        cudaStreamSynchronize(streams[gpu]);
        // Copy results and merge
    }
}
```

---

## 5. Implementation Timeline

### Month 1-2: Foundation
- [ ] Set up project repository and build system
- [ ] Implement core data structures (Particle, Geometry, Material)
- [ ] Basic stopping power module with NIST data
- [ ] Unit test framework setup

### Month 2-3: Core Physics
- [ ] Complete stopping power (Bethe-Bloch, effective charge)
- [ ] Molière scattering refinement
- [ ] Energy straggling (Bohr, Landau-Vavilov)
- [ ] Basic transport loop

### Month 3-4: Geometry & Scoring
- [ ] Voxelized geometry engine
- [ ] Ray-tracing algorithm
- [ ] Dose and LET scoring
- [ ] Validation with simple phantoms

### Month 4-5: CPU Parallelization
- [ ] OpenMP implementation
- [ ] Thread-local scoring
- [ ] RNG stream management
- [ ] Performance profiling and optimization

### Month 5-6: Fragmentation
- [ ] QMSFRG database integration
- [ ] Fragment generation algorithms
- [ ] Secondary particle stacking
- [ ] Validation against experimental data

### Month 6-7: GPU Implementation
- [ ] CUDA kernel development
- [ ] Memory optimization
- [ ] GPU-CPU result verification
- [ ] Performance benchmarking

### Month 7-8: Validation & Optimization
- [ ] Comprehensive physics validation
- [ ] Performance optimization (CPU & GPU)
- [ ] Multi-GPU scaling
- [ ] Regression test suite

### Month 8-9: Documentation & Release
- [ ] User manual and theory guide
- [ ] Code documentation
- [ ] Example library
- [ ] Initial release (v1.0)

---

## 6. Success Metrics

### Physics Accuracy
- Bragg peak position: < 1mm error
- Dose distribution: < 2% or 2mm (gamma analysis)
- LET accuracy: < 5% vs. analytical models
- Fragment yields: Within experimental uncertainties

### Performance
- **CPU**: > 100k particles/sec/core (single thread)
- **CPU (32 cores)**: > 3M particles/sec
- **GPU (A100)**: > 20M particles/sec
- **Multi-GPU (4x A100)**: > 70M particles/sec (>87% efficiency)
- Parallel efficiency: > 90% up to 64 cores

### Software Quality
- Code coverage: > 80%
- All validation tests passing
- Documentation completeness: > 95%
- No critical bugs in release

---

## 7. Risk Analysis & Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GPU memory limitations | High | Medium | Streaming particle batches, optimize memory layout |
| QMSFRG database integration complexity | Medium | High | Start with simpler models, incremental integration |
| Physics validation failures | High | Low | Continuous validation, expert consultation |
| Performance targets not met | Medium | Medium | Early prototyping, profiling, multiple optimization passes |
| Multi-GPU scaling issues | Low | Medium | Test on cloud platforms early, optimize communication |

### Schedule Risks

| Risk | Impact | Mitigation |
|------|--------|------------|
| Underestimated fragmentation complexity | Medium | Allocate buffer time, modular design allows deferral |
| GPU optimization harder than expected | Medium | CPU version fully functional first, GPU as enhancement |
| Validation takes longer | Low | Parallel validation during development |

---

## 8. Future Enhancements (Beyond v1.0)

### Advanced Physics
- Magnetic field deflection (for beam delivery systems)
- Delta-ray explicit transport
- Nuclear de-excitation (gamma production)
- Photonuclear reactions
- Coupling with electromagnetic shower code

### Performance
- AMD GPU support (ROCm/HIP)
- Intel GPU support (SYCL)
- ARM CPU optimization
- Distributed computing (MPI for clusters)

### Features
- Variance reduction techniques (splitting, Russian roulette)
- Importance sampling for rare events
- Adaptive mesh refinement for scoring
- Real-time visualization during simulation
- Machine learning-based physics surrogates

### Applications
- Treatment planning integration (DICOM-RT support)
- Radiobiological modeling (LEM, MKM)
- Shielding calculations (spacecraft, proton therapy vaults)
- Detector response simulation

---

## 9. Recommended Development Approach

### Phase 1 Priority: Establish Foundation
1. Start with **CPU-only** implementation in modern Fortran
2. Focus on **correct physics** before performance
3. Build **comprehensive test suite** from the start
4. Create **modular architecture** to allow parallel development

### Phase 2 Priority: Validate & Optimize CPU
1. Validate against **known benchmarks** (ICRU, experimental data)
2. Optimize **critical paths** with profiling
3. Achieve **target performance** on CPU
4. Document **physics models** thoroughly

### Phase 3 Priority: GPU Port
1. Port **validated CPU code** to GPU
2. Verify **bit-level consistency** between CPU and GPU
3. Optimize for **GPU architecture**
4. Implement **multi-GPU** if needed

### Development Tools
- **Version Control**: Git (GitHub/GitLab)
- **Build System**: CMake
- **Testing**: pFUnit, CTest
- **Profiling**: VTune (CPU), Nsight (GPU)
- **CI/CD**: GitHub Actions or GitLab CI
- **Documentation**: Doxygen + Sphinx

---

## 10. Conclusion

This development plan provides a structured path to creating a production-quality 3D radiation Monte Carlo code with:

- **Comprehensive Physics**: Stopping power, scattering, straggling, fragmentation
- **High Performance**: Optimized CPU and GPU implementations
- **Validated Results**: Extensive validation against experimental data
- **Maintainable Code**: Modular architecture, tested, documented
- **Scalable Design**: Multi-core CPU, multi-GPU support

The 9-month timeline is ambitious but achievable with focused effort. The modular design allows for incremental development and early testing of critical components.

**Next Steps:**
1. Review and refine this plan with stakeholders
2. Set up development environment and repository
3. Begin Phase 1 implementation (core physics modules)
4. Establish weekly progress milestones
5. Schedule validation checkpoints with domain experts

---

**Document Version:** 1.0
**Date:** 2025-12-28
**Author:** Development Plan for COULOMB_LET 3D Monte Carlo
