"""
Convert NIST stopping power data from ASCII to binary NumPy format.

This provides ~100x faster loading:
- ASCII loadtxt: ~10-50ms
- Binary np.load: ~0.5ms

Critical for multiprocessing where data is loaded multiple times.
"""

import numpy as np
from pathlib import Path
import time
import sys

# Add parent directory to path to import coulomb_mc
sys.path.insert(0, str(Path(__file__).parent.parent))


def convert_dat_to_npy(data_dir='data/nist'):
    """Convert all .DAT files to .npy format."""
    data_path = Path(__file__).parent.parent / data_dir

    if not data_path.exists():
        print(f"Error: {data_path} does not exist")
        return

    dat_files = list(data_path.glob('*.DAT'))

    if not dat_files:
        print(f"No .DAT files found in {data_path}")
        return

    print(f"Found {len(dat_files)} NIST data files")
    print(f"Converting ASCII → binary NumPy format...\n")

    total_time_ascii = 0
    total_time_binary = 0

    for dat_file in sorted(dat_files):
        print(f"Processing: {dat_file.name}")

        # Time ASCII loading (skip 3-line header)
        start = time.time()
        data = np.loadtxt(dat_file, skiprows=3)
        time_ascii = time.time() - start
        total_time_ascii += time_ascii
        print(f"  ASCII load: {time_ascii*1000:.1f}ms ({data.shape[0]} rows, {data.shape[1]} cols)")

        # Save as binary
        npy_file = dat_file.with_suffix('.npy')
        np.save(npy_file, data)

        # Time binary loading
        start = time.time()
        loaded = np.load(npy_file)
        time_binary = time.time() - start
        total_time_binary += time_binary
        print(f"  Binary load: {time_binary*1000:.1f}ms")

        # Verify correctness
        assert np.allclose(data, loaded), "Data mismatch!"

        speedup = time_ascii / time_binary if time_binary > 0 else float('inf')
        print(f"  Speedup: {speedup:.0f}x faster")
        print(f"  ✓ Saved: {npy_file.name}\n")

    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Files converted: {len(dat_files)}")
    print(f"Total ASCII load time: {total_time_ascii*1000:.1f}ms")
    print(f"Total binary load time: {total_time_binary*1000:.1f}ms")

    if total_time_binary > 0:
        overall_speedup = total_time_ascii / total_time_binary
        print(f"Overall speedup: {overall_speedup:.0f}x faster")

    print("\nBinary files (.npy) will be used automatically by StoppingPower class.")


if __name__ == "__main__":
    convert_dat_to_npy()
