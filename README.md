# COULOMB_MC - Monte Carlo Radiation Transport Code

High-performance 3D Monte Carlo simulation for heavy-ion radiation transport in matter.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

COULOMB_MC is a Python-based Monte Carlo radiation transport code for simulating heavy-ion interactions in matter. It tracks individual particle trajectories through materials, accounting for:

- Energy loss (stopping power from NIST PSTAR/ASTAR data)
- Multiple Coulomb scattering (Highland/Molière theory)
- Range straggling (Bohr/Vavilov theory)
- Adaptive step sizing for accurate Bragg peak modeling
- 3D dose deposition in voxelized geometries

Originally developed in Fortran for radiation therapy and space radiation applications, this Python rewrite emphasizes performance, extensibility, and modern software engineering practices.

## Features

### Physics
- Heavy-ion transport (protons, helium, carbon, etc.)
- NIST PSTAR/ASTAR stopping power databases
- Multiple Coulomb scattering with validated angular distributions
- Adaptive stepping near Bragg peaks (<1% energy loss per step)
- Range straggling for realistic dose distributions
- 3D voxelized scoring for dose-depth and lateral profiles

### Performance
- Numba JIT compilation (10-100x speedup)
- Binary data format (60x faster loading than ASCII)
- Optimized parallelization (4x improvement from baseline)
- GPU-ready architecture (CUDA support planned)
- Multiprocessing support for large particle batches

### Current Performance
- 230 particles/sec on single core (10,000 particles in 44s)
- 4x speedup from baseline after recent optimizations
- Target: 8-10x with OpenMP-style parallelization (in progress)
- Future: 100-1000x with GPU implementation

## Installation

```bash
# Clone the repository
git clone https://github.com/wcomaskey/COULOMB_LET.git
cd COULOMB_LET

# Install with conda (recommended)
bash install.sh

# OR install with pip
bash install_pip.sh

# Activate environment
conda activate coulomb_mc
```

## Quick Start

```python
from coulomb_mc.transport.engine import TransportEngine

# Create transport engine
engine = TransportEngine(material='water')

# Create particle beam (Carbon-12 at 400 MeV/u)
beam = engine.create_beam('C-12', energy=400.0, n_particles=10000)

# Transport particles through material
stats = engine.transport(beam, max_depth=30.0, verbose=True)

# Get dose-depth profile
depth, dose = engine.get_dose_depth(normalize=True)

# Plot results
import matplotlib.pyplot as plt
plt.plot(depth, dose)
plt.xlabel('Depth (cm)')
plt.ylabel('Relative Dose')
plt.title('Bragg Peak - C-12 400 MeV/u in Water')
plt.show()
```

## Running Examples

```bash
# Validation tests
python examples/scripts/test_week2.py

# Performance profiling
python examples/scripts/profile_parallelization.py
```

## Project Structure

```
COULOMB_LET/
├── coulomb_mc/              # Main package
│   ├── core/                # Particle data structures
│   ├── physics/             # Stopping power, scattering
│   ├── transport/           # Monte Carlo engine
│   ├── scoring/             # Dose deposition
│   └── io/                  # Data loading/saving
├── data/nist/               # NIST stopping power tables
├── examples/scripts/        # Test and example scripts
├── tests/                   # Test suite
├── docs/                    # Technical documentation
└── install.sh               # Installation script
```

## Physics Validation

### Stopping Power
- Validated against NIST PSTAR (protons) and ASTAR (alpha/heavy ions)
- Range accuracy: <3% error for protons 1-400 MeV
- Energy loss: Linear interpolation on log-log grid

### Multiple Scattering
- Highland formula for RMS angle
- Molière distribution for angular sampling
- Validated against Geant4 reference data

### Bragg Peak Position
- Carbon-12 @ 400 MeV/u: 26.4 cm (expected: 26-27 cm)
- Proton @ 200 MeV: 26.0 cm (expected: 25-27 cm)
- Adaptive stepping maintains <1% energy loss per step near peak

## Development Status

### Completed
- Core particle data structures
- NIST stopping power integration
- Multiple scattering (Highland + Molière)
- Transport engine with adaptive stepping
- Dose scoring in 1D/3D
- Binary data optimization
- Performance profiling and analysis

### In Progress
- OpenMP-style particle parallelization
- Multiprocessing for large batches
- Cylindrical coordinate scoring

### Planned
- GPU implementation (CUDA)
- Nuclear fragmentation
- Geometry import (CT/voxel)
- Variance reduction techniques
- Clinical beam models

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for details.

## Performance

### Recent Improvements

| Optimization | Speedup | Status |
|--------------|---------|--------|
| Binary data loading | 60x | Complete |
| Removed bad parallelization | 3.6x | Complete |
| Total improvement | 4x | Complete |

### Next Steps
- OpenMP particle-level parallelization: +2-3x (target: 8-10x total)
- GPU CUDA implementation: +100-1000x

## 3D Simulation Requirements

3D simulations require significantly more particles than 1D due to radial beam spread.

| Geometry | Particles for 1% statistics | Simulation time |
|----------|----------------------------|-----------------|
| 1D (depth-dose) | 10,000 | 44 seconds |
| 3D (realistic beam) | 10,000,000 | 12 hours |
| 3D with GPU | 100,000,000 | 15 minutes (target) |

Solutions:
1. Increase particle count (requires GPU)
2. Use cylindrical coordinates (radially-adaptive voxels)
3. Variance reduction techniques

See [docs/3D_VOXEL_STRATEGY.md](docs/3D_VOXEL_STRATEGY.md) for analysis.

## Requirements

- Python 3.11+
- NumPy >= 1.24
- SciPy >= 1.10
- Numba >= 0.57
- Matplotlib >= 3.7

Optional (future):
- CUDA Toolkit 11+
- CuPy

## Contributing

Contributions welcome. Priority areas:

1. GPU CUDA kernels
2. Nuclear fragmentation models
3. Variance reduction techniques
4. Clinical beam models
5. Geometry import (DICOM, CT)

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{coulomb_let_2025,
  title = {COULOMB\_LET: Monte Carlo Heavy-Ion Transport Code},
  author = {Comaskey, William},
  year = {2025},
  url = {https://github.com/wcomaskey/COULOMB_LET}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## References

1. Highland, V.L. (1975). "Some Practical Remarks on Multiple Scattering"
2. ICRU Report 49 (1993). "Stopping Powers and Ranges for Protons and Alpha Particles"
3. Molière, G. (1948). "Theorie der Streuung schneller geladener Teilchen"
4. NIST PSTAR: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
5. NIST ASTAR: https://physics.nist.gov/PhysRefData/Star/Text/ASTAR.html

## Contact

William Comaskey - GitHub: [@wcomaskey](https://github.com/wcomaskey)
