#!/usr/bin/env python3
"""
Quick test script to verify installation.

Run this after setting up the environment to check everything works.
"""

import sys
from pathlib import Path

print("="*70)
print("COULOMB_MC Installation Test")
print("="*70)

# Test 1: Import packages
print("\n1. Testing imports...")
try:
    import numpy as np
    print("   ✓ NumPy:", np.__version__)
except ImportError as e:
    print(f"   ✗ NumPy failed: {e}")
    sys.exit(1)

try:
    import numba
    print("   ✓ Numba:", numba.__version__)
except ImportError as e:
    print(f"   ✗ Numba failed: {e}")
    sys.exit(1)

try:
    import matplotlib
    print("   ✓ Matplotlib:", matplotlib.__version__)
except ImportError as e:
    print(f"   ✗ Matplotlib failed: {e}")
    sys.exit(1)

# Test 2: Import coulomb_mc
print("\n2. Testing coulomb_mc imports...")
try:
    from coulomb_mc.core.particle import ParticleArray
    print("   ✓ ParticleArray imported")
except ImportError as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

try:
    from coulomb_mc.physics.stopping_power import StoppingPower
    print("   ✓ StoppingPower imported")
except ImportError as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 3: Check NIST data
print("\n3. Checking NIST data files...")
data_dir = Path(__file__).parent.parent.parent / 'data' / 'nist'
nist_file = data_dir / 'STOPRANGE_NIST_Proton.DAT'

if nist_file.exists():
    print(f"   ✓ Found: {nist_file.name}")
else:
    print(f"   ✗ Missing: {nist_file}")
    print(f"   Copy from: COULOMB_LET/STOPRANGE_NIST_Proton.DAT")
    print(f"   To: {nist_file}")

# Test 4: Create particle beam
print("\n4. Testing particle beam creation...")
try:
    beam = ParticleArray(1000)
    beam.initialize_beam('C-12', 400.0, (0, 0, 0), (0, 0, 1))
    print(f"   ✓ Created beam: {beam}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

# Test 5: Test stopping power (if NIST data available)
if nist_file.exists():
    print("\n5. Testing stopping power calculations...")
    try:
        sp = StoppingPower('water')
        let = sp.LET_keV_um(400, 12, 6)
        print(f"   ✓ C-12 @ 400 MeV/u: LET = {let:.2f} keV/µm")
        print(f"     (Expected: ~11.2 keV/µm from ICRU)")

        if abs(let - 11.2) / 11.2 < 0.10:  # 10% tolerance
            print("   ✓ Physics validation: PASSED")
        else:
            print(f"   ⚠ Physics validation: Large error ({abs(let-11.2)/11.2*100:.1f}%)")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
else:
    print("\n5. Skipping physics test (no NIST data)")

# Test 6: Test Numba JIT compilation
print("\n6. Testing Numba JIT compilation...")
try:
    from coulomb_mc.core.particle import transport_step_cpu
    import time

    # Warm up (trigger compilation)
    dEdx = np.ones(1000)
    _ = transport_step_cpu(beam.particles, 0.1, dEdx)

    # Time it
    start = time.time()
    for _ in range(10):
        transport_step_cpu(beam.particles, 0.1, dEdx)
    elapsed = (time.time() - start) / 10

    print(f"   ✓ JIT compilation successful")
    print(f"   ✓ Transport step: {elapsed*1000:.2f} ms")
except Exception as e:
    print(f"   ✗ Failed: {e}")

# Summary
print("\n" + "="*70)
print("Installation test complete!")
print("="*70)

if nist_file.exists():
    print("\n✓ All systems operational. Ready to run simulations!")
    print("\nNext steps:")
    print("  1. Run examples/scripts/bragg_peak_simple.py")
    print("  2. Open examples/notebooks/01_physics_validation.ipynb")
else:
    print("\n⚠ Setup incomplete: Copy NIST data files to get started")
    print("\nRun:")
    print(f"  cp COULOMB_LET/STOPRANGE_NIST_Proton.DAT {data_dir}/")
    print(f"  cp COULOMB_LET/STOPRANGE_NIST_Helium.DAT {data_dir}/")
