# 3D Voxel Scoring Strategy

## Executive Summary

**Problem:** In 3D Monte Carlo, only ~0.035% of particles hit the central voxel due to lateral scattering. With 10,000 particles, you get 3.5 particles per central voxel (53% noise) - completely unusable.

**Solutions:**
1. Use **10^7 particles** for high-resolution 3D (requires GPU)
2. Use **larger voxels** (0.5-1.0 cm instead of 0.1 cm)
3. Use **cylindrical scoring** with radially-adaptive voxel sizing

---

## Recommended Configurations

### Configuration 1: Development/Testing (Current)
```python
# 1D depth-dose only
engine = TransportEngine(material='water')
beam = engine.create_beam('C-12', 400.0, n_particles=10_000)
# Runtime: ~1 minute, Noise: ~1%
```

### Configuration 2: 3D with Coarse Voxels  
```python
# Acceptable for initial 3D work
engine = TransportEngine(material='water')
engine.n_bins = 200  # 0.25 cm voxels instead of 0.1 cm
beam = engine.create_beam('C-12', 400.0, n_particles=100_000)
# Runtime: ~10 minutes, Noise: ~10-15%
```

### Configuration 3: High-Resolution 3D (GPU Required)
```python
# Production quality
engine = TransportEngine(material='water', device='gpu')  # Week 7-8
engine.n_bins = 500  # 0.1 cm voxels
beam = engine.create_beam('C-12', 400.0, n_particles=10_000_000)
# Runtime: ~15 minutes on GPU, Noise: ~2%
```

###Configuration 4: Cylindrical Scoring (Smart Approach)
```python
# Best statistics with fine on-axis resolution
engine = TransportEngine(material='water', scoring_mode='cylindrical')
engine.set_radial_bins([0, 0.1, 0.3, 0.6, 1.0, 1.5, 2.5, 4.0])  # cm
beam = engine.create_beam('C-12', 400.0, n_particles=1_000_000)
# Runtime: ~100 minutes, Even statistics across all bins
```

---

## Particle Count vs. Voxel Size Trade-off

| Goal | Voxel Size | Min Particles | Recommended | Runtime (CPU) |
|------|------------|---------------|-------------|---------------|
| Quick test | 1.0 cm | 10,000 | 50,000 | 5 min |
| Development | 0.5 cm | 100,000 | 500,000 | 50 min |
| Validation | 0.2 cm | 1,000,000 | 5,000,000 | 8 hours |
| Production | 0.1 cm | 10,000,000 | 50,000,000 | 3 days |

**With GPU (Week 7-8), divide runtimes by ~100x**

---

## Implementation Roadmap

### Week 3: Basic 3D Support
- Add warning if particles < 100,000 for 3D
- Implement coarser binning option
- Add particle count recommendations to docs

### Week 4-6: Cylindrical Scoring
- Implement (r, z) binning instead of (x, y, z)
- Radially-adaptive voxel sizing
- Visualization tools for cylindrical data

### Week 7-8: GPU Acceleration **[CRITICAL for 3D]**
- Port transport loop to CUDA
- Enable 10^7-10^8 particles in reasonable time
- Batch processing for memory management

### Week 9+: Variance Reduction
- Particle splitting near ROI
- Russian roulette for low-importance particles  
- Importance sampling

---

## Immediate Actions

For your current work, I recommend:

1. **Stay with 1D for now** (depth-dose curves)
   - 10,000 particles is perfect
   - No noise issues
   - Fast iteration

2. **When moving to 3D:**
   - Start with 0.5 cm voxels + 500,000 particles
   - Or use cylindrical scoring with 1,000,000 particles
   - Accept ~10% noise for development

3. **Plan GPU implementation for Week 7-8**
   - Essential for production 3D work
   - Target: 10^7 particles in 10-15 minutes

---

*Created: 2025-12-29*
*COULOMB_MC Development Notes*
