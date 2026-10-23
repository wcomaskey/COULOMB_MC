# COULOMB_MC - Monte Carlo Radiation Transport Code

Monte Carlo simulation of heavy-ion transport in matter, with 3D particle tracking and depth-dose scoring.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

COULOMB_MC is a Python Monte Carlo code for heavy-ion transport in matter. It tracks each particle's position and direction in 3D through a homogeneous material, step by step, and accounts for:

- Continuous energy loss, from NIST PSTAR proton stopping powers scaled to heavier ions by a Barkas effective charge
- Multiple Coulomb scattering, as a Gaussian with the Highland width
- Adaptive step sizing near the end of range
- Depth-dose scoring (1D, 500 bins)

The Python code is a rewrite of the Fortran programs COULOMB_LET.f90 and GCR_Modern.f90. The Fortran originals are not in this repository.

## Features

### Physics
- Ion transport for protons through heavier ions (He-4, C-12, O-16, ...)
- Accepted particle names are proton, H-1, deuteron, H-2, triton, H-3, He-3, He-4, alpha, Li-7, Be-9, B-11, C-12, N-14, O-16, Ne-20, Si-28 and Fe-56. Any other name raises ValueError.
- NIST PSTAR proton stopping-power and range tables, interpolated as power laws between table points. Only the proton table is used. The helium table is shipped but never loaded.
- Barkas effective-charge scaling of the proton stopping power for heavier ions
- Multiple Coulomb scattering: each step draws two independent projected angles from a Gaussian of Highland width θ0, so the polar angle is Rayleigh distributed and the azimuth is uniform
- Adaptive stepping: `calculate_adaptive_step_size` compares an energy budget of 0.01 × E (E in MeV/u) with the whole-ion dE/dx, so each step loses at most 1%/A of the ion's energy. In the last 1 to 2 MeV/u a minimum step of 1 µm overrides this. A limit of 5% of the remaining range exists but does not set the step for the beams described here.

### Performance
- Numba JIT compilation of the step kernels
- NIST tables shipped in NumPy binary format (regenerated from the ASCII tables if missing)
- Multiprocessing over particle chunks: 3.19x on 4 cores, measured before the 2026-10-23 scattering fix with 1,000 C-12 particles (see [docs/PARALLELIZATION_PLAN.md](docs/PARALLELIZATION_PLAN.md))

### Measured run times (one process, Apple M5 Pro, 2026-10-23)
- C-12 at 400 MeV/u in water, 10,000 particles: about 95 s
- Protons at 200 MeV in water, 10,000 particles: about 11 s

## Installation

```bash
# Clone the repository
git clone https://github.com/wcomaskey/COULOMB_MC.git
cd COULOMB_MC

# Install with conda (recommended)
bash install.sh
conda activate coulomb_mc

# OR install with pip (creates the virtual environment venv_coulomb_mc)
bash install_pip.sh
source venv_coulomb_mc/bin/activate
```

## Quick Start

```python
from coulomb_mc.transport.engine import TransportEngine

# Create transport engine
engine = TransportEngine(material='water')

# Create particle beam (Carbon-12 at 400 MeV/u)
beam = engine.create_beam('C-12', 400.0, n_particles=10000)

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

## Running Tests and Examples

```bash
# Unit tests (scattering statistics and a pencil-beam regression)
python -m pytest tests/unit -q

# Clear the Numba caches before a validation run. Numba's on-disk cache does not
# invalidate a caller when a jitted function in another file changes.
find coulomb_mc -name __pycache__ -type d -prune -exec rm -rf {} +

# Bragg-peak validation with a numeric JSON record
python examples/scripts/bragg_validation.py --n 10000 --seed 12345 \
    --out examples/results/bragg_validation_seed12345.json

# Bragg-peak plots and range table
python examples/scripts/bragg_peak_simple.py

# Performance profiling
python examples/scripts/profile_parallelization.py
```

## Project Structure

```
COULOMB_MC/
├── coulomb_mc/              # Main package
│   ├── core/                # Particle data structures
│   ├── physics/             # Stopping power, scattering
│   ├── transport/           # Monte Carlo engine and depth-dose scoring
│   ├── scoring/             # (placeholder)
│   └── io/                  # (placeholder)
├── data/nist/               # NIST stopping power tables
├── examples/scripts/        # Test and example scripts
├── examples/results/        # Validation records (JSON)
├── tests/unit/              # Unit tests
├── docs/                    # Technical notes and plans
└── install.sh               # Installation script
```

## Physics Validation

### Multiple scattering
`tests/unit/test_scattering.py` holds the unit tests, and 13 tests pass. Over 200,000 samples, and for five incoming directions including ±z and x, they check that one scattering step gives projected angles with zero mean, width θ0 and no correlation. They check that the polar angle is Rayleigh distributed and the azimuth is uniform, and that `rotate_direction` tilts the direction by exactly θ and places the azimuth correctly. A transported pencil beam of 400 C-12 particles must have equal x and y spread (within 25%) and no net lateral drift.

### Bragg peak position
Water, 10,000 particles per case, beam along +z. Depths are bin centres of 0.1 cm bins, and R80 is the distal depth where the dose falls to 80% of its maximum. The references are CSDA ranges in liquid water in g/cm². The code's water density is 1.0 g/cm³, so its depths in cm are comparable to these ranges in g/cm².

| Beam | Peak depth | R80 | Reference CSDA range | Peak vs reference | R80 vs reference | Code's own CSDA range |
|---|---|---|---|---|---|---|
| C-12, 400 MeV/u | 27.35 cm | 27.39 cm | 27.5 g/cm² (ICRU Report 90, Table A.15, interpolated) | -0.55% | -0.38% | 27.43 g/cm² |
| Proton, 200 MeV | 25.95 cm | 26.02 cm | 25.96 g/cm² (NIST PSTAR) | -0.04% | +0.22% | 26.0 g/cm² |

The C-12 reference comes from ICRU Report 90, Table A.15, which lists 20.51 g/cm² at 4000 MeV and 29.40 g/cm² at 5000 MeV. Power-law interpolation to 4800 MeV (400 MeV/u × 12) gives 27.5 g/cm². The C-12 result is within 0.6% of this value and the proton result is within 0.25% of PSTAR. The earlier README compared C-12 with 26.4 cm, attributed to ICRU Report 73. That value could not be found in any ICRU source and is withdrawn.

The code's own C-12 range is its proton range at 400 MeV (82.3 g/cm²) times A/Z² = 12/36. For protons it is a three-significant-figure entry of the PSTAR table. The C-12 R80 agrees with the code's own C-12 range to 0.15%.

Energy loss is deterministic (no straggling), so the peak and R80 depths do not depend on the random seed at 0.1 cm resolution. The seed changes only the lateral positions, and seed 1 gives the same depths as seed 12345. The records, with depth-dose curves, are in [examples/results/](examples/results/):

- `bragg_validation_seed12345.json`
- `bragg_validation_seed1.json`
- `bragg_validation_before_scattering_fix_seed12345.json` (the code before the fix described below)

### Range table
`examples/scripts/bragg_peak_simple.py` prints the peak depth of six beams (2000 particles each) against reference ranges in g/cm²: protons at 200 MeV (25.96, NIST PSTAR) and 100 MeV (7.718, NIST PSTAR), He-4 at 250 MeV/u (38.30, ICRU 90 Table A.12 at 1000 MeV), C-12 at 400 MeV/u (27.5) and 290 MeV/u (16.3, both ICRU 90 Table A.15, interpolated), and O-16 at 400 MeV/u (20.6, ICRU 73 revised water data, integrated). All six agree within 0.9%, and for the short ranges this is limited by the 0.1 cm bins. An earlier version of the table used wrong references for He-4 at 400 MeV/u (27.8), C-12 at 290 MeV/u (13.1) and O-16 at 400 MeV/u (23.0). He-4 at 400 MeV/u actually ranges about 82 cm, beyond the 50 cm scoring grid.

### Lateral spread (not validated)
The stop positions are recorded, but the lateral spread has not been validated and is lower than published values. For seed 12345 the rms of the stop position in x/y is 0.122/0.123 cm for C-12 and 0.478/0.476 cm for protons. The published end-of-range rms for 200 MeV protons in water is 2.25% of range, or 0.584 cm (B. Gottschalk, arXiv:1204.4470, Table 1). For C-12 at 400 MeV/u, Eq. 7 of Kanematsu (Phys. Med. Biol. 54, N67 (2009)) gives about 0.170 cm. The code is therefore about 18% low for protons and about 28% low for C-12. The causes are listed under Known Limitations. The scattering demo (`python coulomb_mc/physics/scattering.py`) prints this per-step deficit (-18.4% for its test case) and shows that the sampling matches the per-step expectation to 0.2%.

### Fixes (2026-10-23)
Scattering. Before this date, the scattering step sampled an azimuth it never used and drew the polar angle from a half-Gaussian. Every deflection then turned the direction the same way about the same axis, so tracks curved in one plane and stopped short. Tracks ended on average 3.65 cm (C-12) and 3.94 cm (proton) off-axis in -x, with no spread in y. For the same cases that version (main at commit 4419136) gave peaks at 26.95 cm for C-12 (-2.0% against 27.5) and 25.35 cm for protons (-2.35% against 25.96). Its record is [examples/results/bragg_validation_before_scattering_fix_seed12345.json](examples/results/bragg_validation_before_scattering_fix_seed12345.json).

Dose scoring. `_score_dose_adaptive` is called before the particles move, but it treated the current position as the end of the step. It therefore scored each deposit one step upstream and dropped the deposit of the first step. It now uses the start position plus half a step. The peak bins did not change.

## Known Limitations

- No range straggling, so the Bragg peak is sharper than measured peaks
- No nuclear fragmentation, so there is no fragment tail beyond the peak for heavy ions
- Heavy-ion stopping power is the scaled proton value, with no ion-specific tables
- Highland width applied per step, with the per-step variances added in quadrature. PDG (RPP 2024, Sec. 34.3) calls this systematically too small. The log term is ln(x/X0), which lacks the z²/β² factor of the PDG form ln(x z²/(X0 β²)). Against published values the lateral spread is about 18% low for protons and about 28% low for C-12, and the deficit grows as the steps get smaller.
- Lateral spread is recorded but not validated
- The beam is a zero-width, zero-divergence pencil beam along +z
- Each step loses at most 1%/A of the ion's energy, and a minimum step of 1 µm applies in the last 1 to 2 MeV/u
- Scoring uses a fixed 0-50 cm grid of 500 bins
- Gaussian angular distribution only, with no single-scattering tail
- Homogeneous material, depth-dose scoring only, and particles below 0.1 MeV/u are dropped

## Development Status

### Completed
- Core particle data structures
- NIST stopping power integration
- Multiple scattering (Gaussian, Highland width)
- Transport engine with adaptive stepping
- Depth-dose scoring (1D)
- Binary data caching
- Multiprocessing over particle chunks
- Unit tests for scattering and a numeric Bragg-peak validation record

### Planned
- Range straggling
- Cylindrical and 3D dose scoring (see [docs/3D_VOXEL_STRATEGY.md](docs/3D_VOXEL_STRATEGY.md))
- Nuclear fragmentation
- Geometry import (CT/voxel)
- Variance reduction techniques
- GPU implementation (CUDA)

See [DEVELOPMENT_PLAN.md](DEVELOPMENT_PLAN.md) for details.

## Requirements

- Python 3.10+
- NumPy >= 1.24
- Numba >= 0.58
- Matplotlib >= 3.7
- pytest (for the unit tests)

`setup.py` also declares SciPy, h5py, PyYAML and tqdm, but nothing in the package, examples or tests imports them.

## Contributing

Contributions welcome. Priority areas:

1. Range straggling and ion-specific stopping powers
2. Nuclear fragmentation models
3. Cylindrical and 3D dose scoring
4. Variance reduction techniques
5. GPU CUDA kernels

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

```bibtex
@software{coulomb_mc_2025,
  title = {COULOMB\_MC: Monte Carlo Heavy-Ion Transport Code},
  author = {Comaskey, William},
  year = {2025},
  url = {https://github.com/wcomaskey/COULOMB_MC}
}
```

## License

MIT License - see [LICENSE](LICENSE) file.

## References

1. Highland, V.L. (1975). "Some Practical Remarks on Multiple Scattering", Nucl. Instrum. Methods 129, 497
2. ICRU Report 49 (1993). "Stopping Powers and Ranges for Protons and Alpha Particles"
3. ICRU Report 73 (2005). "Stopping of Ions Heavier Than Helium"
4. ICRU Report 90 (2016). "Key Data for Ionizing-Radiation Dosimetry: Measurement Standards and Applications", J. ICRU 14(1), doi:10.1093/jicru/ndw039
5. NIST PSTAR: https://physics.nist.gov/PhysRefData/Star/Text/PSTAR.html
6. Particle Data Group, Review of Particle Physics (RPP 2024), Sec. 34.3
7. Gottschalk, B., arXiv:1204.4470, Table 1
8. Kanematsu, N. (2009). Phys. Med. Biol. 54, N67

## Contact

William Comaskey - GitHub: [@wcomaskey](https://github.com/wcomaskey)
