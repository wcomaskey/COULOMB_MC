"""
Bragg-peak validation, with a numeric record of depth and lateral metrics.

Runs C-12 at 400 MeV/u and protons at 200 MeV in water, then writes the
results to JSON so the validation numbers live in a file, not only in plot
annotations.

For each case it reports:
    - peak depth: depth of the dose maximum (bin centre, 0.1 cm bins)
    - R80: distal depth where dose falls to 80% of the maximum (interpolated)
    - mean and rms of the stopping positions in x, y and z (recorded only;
      the lateral rms has no reference comparison here)
    - wall time, platform and package versions

Reference CSDA ranges in liquid water, in g/cm² (the code's water density is
1.0 g/cm³, so its depths in cm are g/cm²):
    - C-12 400 MeV/u: 27.5 (ICRU Report 90, Table A.15: 20.51 at 4000 MeV and
      29.40 at 5000 MeV, power-law interpolation to 4800 MeV)
    - proton 200 MeV: 25.96 (NIST PSTAR)

The code has no range straggling, so the dose maximum sits at the end of
range, both depth metrics are compared with the CSDA range, and the depth
metrics do not change with the random seed at 0.1 cm resolution.

Usage:
    python examples/scripts/bragg_validation.py --n 10000 --seed 12345 \
        --out examples/results/bragg_validation.json
"""

import argparse
import json
import platform
import time
from pathlib import Path
import sys

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coulomb_mc.transport.engine import TransportEngine


CASES = [
    ('C-12', 400.0, 27.5, 'ICRU Report 90, Table A.15, interpolated'),
    ('proton', 200.0, 25.96, 'NIST PSTAR'),
]


@numba.njit
def _seed_numba(seed):
    np.random.seed(seed)


def distal_depth(depth, dose, level):
    """Depth beyond the dose maximum where dose falls to `level` x max."""
    i_peak = int(np.argmax(dose))
    target = level * dose[i_peak]
    for i in range(i_peak, len(dose) - 1):
        if dose[i] >= target > dose[i + 1]:
            frac = (dose[i] - target) / (dose[i] - dose[i + 1])
            return float(depth[i] + frac * (depth[i + 1] - depth[i]))
    return float('nan')


def run_case(particle, energy, ref_range, ref_source, n, seed):
    np.random.seed(seed)
    _seed_numba(seed)

    engine = TransportEngine(material='water')
    beam = engine.create_beam(particle, energy, n_particles=n)

    t0 = time.time()
    stats = engine.transport(beam, max_depth=50.0, verbose=False)
    wall = time.time() - t0

    depth, dose = engine.get_dose_depth(normalize=True)
    peak = float(depth[int(np.argmax(dose))])
    r80 = distal_depth(depth, dose, 0.80)

    pos = beam.particles['position']
    result = {
        'particle': particle,
        'energy_MeV_u': energy,
        'n_particles': n,
        'seed': seed,
        'reference_csda_range_g_cm2': ref_range,
        'reference_source': ref_source,
        'peak_depth_cm': peak,
        'peak_vs_reference_pct': 100.0 * (peak - ref_range) / ref_range,
        'R80_cm': r80,
        'R80_vs_reference_pct': 100.0 * (r80 - ref_range) / ref_range,
        'stop_mean_x_cm': float(np.mean(pos[:, 0])),
        'stop_mean_y_cm': float(np.mean(pos[:, 1])),
        'stop_mean_z_cm': float(np.mean(pos[:, 2])),
        'stop_rms_x_cm': float(np.std(pos[:, 0])),
        'stop_rms_y_cm': float(np.std(pos[:, 1])),
        'n_steps': int(stats['n_steps']),
        'wall_time_s': wall,
    }
    return result, depth.tolist(), dose.tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--n', type=int, default=10000, help='particles per case')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--out', type=str, default='bragg_validation.json')
    parser.add_argument('--label', type=str, default='', help='free-text label stored in the output')
    args = parser.parse_args()

    out = {
        'label': args.label,
        'platform': platform.platform(),
        'processor': platform.processor(),
        'python': platform.python_version(),
        'numpy': np.__version__,
        'numba': numba.__version__,
        'cases': [],
        'curves': {},
    }
    for particle, energy, ref_range, ref_source in CASES:
        result, depth, dose = run_case(particle, energy, ref_range, ref_source,
                                       args.n, args.seed)
        out['cases'].append(result)
        out['curves'][f'{particle}_{energy:g}'] = {'depth_cm': depth, 'dose_pct': dose}
        print(f"{particle:7s} {energy:5.0f} MeV/u  peak {result['peak_depth_cm']:6.2f} cm "
              f"({result['peak_vs_reference_pct']:+.2f}%)  R80 {result['R80_cm']:6.2f} cm "
              f"({result['R80_vs_reference_pct']:+.2f}%)  "
              f"stop x {result['stop_mean_x_cm']:+.3f}+/-{result['stop_rms_x_cm']:.3f}  "
              f"y {result['stop_mean_y_cm']:+.3f}+/-{result['stop_rms_y_cm']:.3f} cm  "
              f"[{result['wall_time_s']:.0f} s]")

    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(f"wrote {path}")


if __name__ == '__main__':
    main()
