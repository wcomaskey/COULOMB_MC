# Contributing to COULOMB_LET

Guidelines for contributing to the project.

---

## Getting Started

### 1. Fork and Clone

```bash
# Fork the repository on GitHub, then:
git clone https://github.com/YOUR_USERNAME/COULOMB_LET.git
cd COULOMB_LET
```

### 2. Set Up Development Environment

```bash
# Install dependencies
bash install.sh
conda activate coulomb_mc

# Install in development mode
pip install -e .
```

### 3. Run Tests

```bash
# Run validation tests
python examples/scripts/test_week1.py
python examples/scripts/test_week2.py

# Run unit tests (when available)
pytest tests/
```

---

## Development Workflow

### Branch Strategy

- `main` - Stable release branch
- `develop` - Integration branch for new features
- `feature/*` - Feature development branches
- `bugfix/*` - Bug fix branches

### Making Changes

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes:**
   - Write clean, documented code
   - Follow existing code style
   - Add tests for new features
   - Update documentation

3. **Test your changes:**
   ```bash
   python examples/scripts/test_week2.py
   # Ensure all tests pass
   ```

4. **Commit with clear messages:**
   ```bash
   git add .
   git commit -m "Add: Brief description of what you added"
   ```

5. **Push and create PR:**
   ```bash
   git push origin feature/your-feature-name
   # Then create Pull Request on GitHub
   ```

---

## Code Style

### Python Style Guide

- Follow **PEP 8** conventions
- Use **type hints** where possible
- Maximum line length: **100 characters**
- Use **docstrings** for all functions/classes

### Example:

```python
def calculate_stopping_power(energy: float, material: str) -> float:
    """
    Calculate stopping power for given energy and material.

    Parameters:
        energy: Particle kinetic energy [MeV/u]
        material: Material name (e.g., 'water', 'tissue')

    Returns:
        Stopping power [MeV cm^2/g]

    Raises:
        ValueError: If energy is negative or material unknown
    """
    if energy < 0:
        raise ValueError(f"Energy must be positive, got {energy}")
    # Implementation...
```

### Numba-Optimized Code

- Mark performance-critical functions with `@numba.njit`
- Avoid `parallel=True` unless profiling confirms speedup
- Use typed arrays and avoid Python objects in JIT functions

---

## Priority Contribution Areas

### 1. GPU Implementation (High Priority)
**Skills needed:** CUDA, Numba CUDA, GPU programming

- Port transport kernels to CUDA
- Implement atomic dose scoring
- Optimize memory transfers
- Target: 100-1000x speedup

### 2. Nuclear Fragmentation Models
**Skills needed:** Nuclear physics, Monte Carlo methods

- Implement QMD fragmentation cross-sections
- Add projectile/target fragmentation
- Validate against experimental data

### 3. Performance Optimization
**Skills needed:** Python optimization, profiling

- OpenMP-style particle parallelization
- Shared memory multiprocessing
- Lookup table optimization
- Cache-friendly data structures

### 4. Geometry and I/O
**Skills needed:** Medical imaging, file formats

- DICOM/CT import
- Voxelized geometry from images
- HU → density conversion
- Geometry visualization

### 5. Clinical Features
**Skills needed:** Radiation therapy physics

- Pencil beam scanning
- Treatment planning integration
- Biological dose models (RBE)
- Quality assurance tools

### 6. Variance Reduction
**Skills needed:** Monte Carlo theory

- Importance sampling
- Particle splitting/Russian roulette
- Weight windows
- Correlated sampling

---

## Testing Guidelines

### Unit Tests
- Test individual components in isolation
- Use `pytest` framework
- Aim for >80% code coverage

### Integration Tests
- Test complete workflows
- Validate physics accuracy
- Check performance benchmarks

### Validation Tests
- Compare against experimental data
- Reference codes (Geant4, FLUKA, etc.)
- Document validation results

---

## Documentation

### Code Documentation
- Docstrings for all public functions/classes
- Inline comments for complex logic
- Type hints for function signatures

### User Documentation
- Update README.md for new features
- Add examples in `examples/`
- Create tutorials for major features

### Developer Documentation
- Architecture decisions in `docs/`
- Performance analysis in profiling docs
- Physics validation in validation docs

---

## Pull Request Process

### Before Submitting PR

1. - All tests pass
2. - Code follows style guide
3. - Documentation updated
4. - No merge conflicts with `develop`
5. - Commits are clean and well-described

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Performance improvement
- [ ] Documentation update
- [ ] Refactoring

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] Validation tests pass

## Performance Impact
- Benchmark results (if applicable)
- Memory usage changes
- GPU compatibility

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] Tests added/passing
- [ ] No breaking changes (or documented)
```

---

## Code Review Guidelines

### For Reviewers
- Check correctness of physics implementation
- Verify performance claims with profiling
- Ensure code is maintainable
- Suggest improvements constructively

### For Authors
- Respond to feedback promptly
- Be open to suggestions
- Explain design decisions
- Update based on reviews

---

## Physics Validation Standards

### Required Validation
1. **Range tests:** <3% error vs. NIST
2. **Stopping power:** Match PSTAR/ASTAR within uncertainties
3. **Scattering:** Compare angular distributions to theory
4. **Bragg peak:** Position and width within 5%

### Documentation
- Include validation plots
- Reference data sources
- Document uncertainties
- Compare to other codes

---

## Performance Benchmarking

### Standard Benchmarks
```python
# Benchmark: 10,000 particles, C-12 @ 400 MeV/u in water
from coulomb_mc.transport.engine import TransportEngine
import time

engine = TransportEngine(material='water')
beam = engine.create_beam('C-12', 400.0, n_particles=10000)

start = time.time()
stats = engine.transport(beam, max_depth=30.0, verbose=False)
elapsed = time.time() - start

print(f"Time: {elapsed:.1f}s")
print(f"Rate: {10000/elapsed:.0f} particles/sec")
```

### Report Performance Changes
- Include before/after benchmarks
- Test on representative hardware
- Document CPU/GPU used

---

## Questions or Issues?

- **Open an issue** for bugs or feature requests
- **Start a discussion** for general questions
- **Email maintainer** for private concerns

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
