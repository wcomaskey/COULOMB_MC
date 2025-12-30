"""
Bragg Peak Simulation - Simple Example

Simulates depth-dose distribution for heavy-ion beams in water.
Demonstrates the characteristic Bragg peak where dose increases
with depth until particles stop.

This example validates:
    - Stopping power implementation
    - Range calculations
    - Dose scoring accuracy

Expected results for C-12 @ 400 MeV/u in water:
    - Range: ~26.4 cm (ICRU Report 73)
    - Peak-to-entrance ratio: ~3-4
    - Peak width: ~1-2 cm
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coulomb_mc.transport.engine import TransportEngine
from coulomb_mc.physics.stopping_power import StoppingPower


def simulate_bragg_peak(particle_type: str, energy_MeV_u: float,
                       n_particles: int = 10000, material: str = 'water',
                       max_depth: float = 50.0):
    """
    Simulate Bragg peak for given beam parameters.

    Parameters:
        particle_type: Particle name (e.g., 'C-12', 'proton')
        energy_MeV_u: Beam energy [MeV/u]
        n_particles: Number of particles to simulate
        material: Target material
        max_depth: Maximum simulation depth [cm]

    Returns:
        depth, dose: Arrays of depth [cm] and dose [% of peak]
    """
    print(f"\n{'='*70}")
    print(f"Bragg Peak Simulation")
    print(f"{'='*70}")
    print(f"  Particle: {particle_type}")
    print(f"  Energy: {energy_MeV_u} MeV/u")
    print(f"  Material: {material}")
    print(f"  Particles: {n_particles:,}")
    print(f"{'='*70}\n")

    # Create transport engine
    engine = TransportEngine(material=material)

    # Create beam
    beam = engine.create_beam(particle_type, energy_MeV_u, n_particles=n_particles)

    # Transport
    stats = engine.transport(beam, max_depth=max_depth, verbose=True)

    # Get results
    depth, dose = engine.get_dose_depth(normalize=True)

    # Analysis
    peak_idx = np.argmax(dose)
    peak_depth = depth[peak_idx]
    peak_dose = dose[peak_idx]

    # Entrance dose (average first 1 cm)
    entrance_mask = depth < 1.0
    entrance_dose = np.mean(dose[entrance_mask])

    # Peak-to-entrance ratio
    peak_to_entrance = peak_dose / entrance_dose if entrance_dose > 0 else 0

    print(f"\n{'='*70}")
    print(f"Results:")
    print(f"{'='*70}")
    print(f"  Bragg peak depth: {peak_depth:.2f} cm")
    print(f"  Peak dose: {peak_dose:.1f}% (normalized)")
    print(f"  Entrance dose: {entrance_dose:.1f}%")
    print(f"  Peak-to-entrance ratio: {peak_to_entrance:.2f}")
    print(f"  Total steps: {stats['n_steps']}")
    print(f"  Particles stopped: {stats['n_stopped']}")
    print(f"{'='*70}\n")

    return depth, dose, peak_depth, peak_to_entrance


def plot_bragg_peak(depth, dose, particle_type, energy_MeV_u,
                   peak_depth, expected_range=None, save_path=None):
    """
    Create publication-quality Bragg peak plot.

    Parameters:
        depth: Depth array [cm]
        dose: Dose array [% of peak]
        particle_type: Particle name for title
        energy_MeV_u: Energy for title
        peak_depth: Peak position [cm]
        expected_range: Expected range from literature [cm] (optional)
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))

    # Main dose curve
    plt.plot(depth, dose, 'b-', linewidth=2.5, label='Monte Carlo')

    # Mark peak
    plt.axvline(peak_depth, color='r', linestyle='--', linewidth=1.5,
                alpha=0.7, label=f'Peak: {peak_depth:.1f} cm')

    # Mark expected range if provided
    if expected_range is not None:
        plt.axvline(expected_range, color='g', linestyle=':', linewidth=1.5,
                   alpha=0.7, label=f'ICRU: {expected_range:.1f} cm')

        # Show difference
        diff = abs(peak_depth - expected_range)
        error_pct = 100 * diff / expected_range
        plt.text(0.98, 0.02, f'Difference: {diff:.2f} cm ({error_pct:.1f}%)',
                transform=plt.gca().transAxes, fontsize=10,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.xlabel('Depth in Water [cm]', fontsize=14, fontweight='bold')
    plt.ylabel('Dose [% of peak]', fontsize=14, fontweight='bold')
    plt.title(f'Bragg Peak: {particle_type} @ {energy_MeV_u} MeV/u',
             fontsize=16, fontweight='bold')

    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='upper left')

    plt.xlim(0, np.max(depth))
    plt.ylim(0, 110)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved: {save_path}")

    return plt.gcf()


def compare_multiple_energies(particle_type: str = 'C-12',
                              energies: list = [200, 300, 400],
                              n_particles: int = 5000):
    """
    Compare Bragg peaks at different energies.

    Parameters:
        particle_type: Particle name
        energies: List of energies [MeV/u]
        n_particles: Number of particles per simulation
    """
    plt.figure(figsize=(12, 7))

    colors = ['blue', 'green', 'red', 'purple', 'orange']

    for i, energy in enumerate(energies):
        print(f"\n{'='*70}")
        print(f"Energy: {energy} MeV/u")
        print(f"{'='*70}")

        engine = TransportEngine(material='water')
        beam = engine.create_beam(particle_type, energy, n_particles=n_particles)

        max_depth = 50.0 if energy <= 400 else 100.0
        engine.transport(beam, max_depth=max_depth, verbose=False)

        depth, dose = engine.get_dose_depth(normalize=True)

        peak_idx = np.argmax(dose)
        peak_depth = depth[peak_idx]

        plt.plot(depth, dose, color=colors[i % len(colors)], linewidth=2,
                label=f'{energy} MeV/u (peak: {peak_depth:.1f} cm)')

    plt.xlabel('Depth in Water [cm]', fontsize=14, fontweight='bold')
    plt.ylabel('Dose [% of peak]', fontsize=14, fontweight='bold')
    plt.title(f'Bragg Peaks: {particle_type} at Multiple Energies',
             fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=11, loc='upper left')
    plt.xlim(0, max(depth))
    plt.ylim(0, 110)
    plt.tight_layout()

    save_path = Path(__file__).parent / f'bragg_peak_comparison_{particle_type}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nComparison plot saved: {save_path}")

    return plt.gcf()


def validate_range():
    """
    Validate range calculations against ICRU data.

    Tests multiple ions and energies.
    """
    print(f"\n{'='*70}")
    print(f"Range Validation vs ICRU Report 73")
    print(f"{'='*70}\n")

    # Test cases: (particle, energy [MeV/u], expected_range [cm])
    test_cases = [
        ('proton', 200, 25.9),   # NIST PSTAR
        ('proton', 100, 7.72),
        ('He-4', 400, 27.8),     # NIST ASTAR
        ('C-12', 400, 26.4),     # ICRU Report 73
        ('C-12', 290, 13.1),
        ('O-16', 400, 23.0),
    ]

    results = []

    for particle, energy, expected_range in test_cases:
        # Quick simulation
        engine = TransportEngine(material='water')
        beam = engine.create_beam(particle, energy, n_particles=2000)
        engine.transport(beam, max_depth=expected_range * 1.5, verbose=False)

        depth, dose = engine.get_dose_depth(normalize=False)
        peak_idx = np.argmax(dose)
        simulated_range = depth[peak_idx]

        diff = simulated_range - expected_range
        error_pct = 100 * abs(diff) / expected_range

        results.append({
            'particle': particle,
            'energy': energy,
            'expected': expected_range,
            'simulated': simulated_range,
            'diff': diff,
            'error_pct': error_pct
        })

        status = "✓ PASS" if error_pct < 5.0 else "✗ FAIL"
        print(f"{status} {particle:8s} {energy:4.0f} MeV/u: "
              f"Expected {expected_range:5.1f} cm, "
              f"Got {simulated_range:5.1f} cm "
              f"({diff:+5.2f} cm, {error_pct:4.1f}%)")

    # Summary
    mean_error = np.mean([r['error_pct'] for r in results])
    max_error = np.max([r['error_pct'] for r in results])
    n_pass = sum(1 for r in results if r['error_pct'] < 5.0)

    print(f"\n{'='*70}")
    print(f"Validation Summary:")
    print(f"{'='*70}")
    print(f"  Tests passed: {n_pass}/{len(test_cases)}")
    print(f"  Mean error: {mean_error:.2f}%")
    print(f"  Max error: {max_error:.2f}%")
    print(f"  Target: <5% error")
    print(f"{'='*70}\n")

    return results


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Example 1: Single Bragg peak (C-12 @ 400 MeV/u)
    print("\n" + "="*70)
    print("Example 1: Carbon-12 Bragg Peak")
    print("="*70)

    depth, dose, peak_depth, peak_ratio = simulate_bragg_peak(
        'C-12', 400.0, n_particles=10000
    )

    fig = plot_bragg_peak(
        depth, dose, 'C-12', 400.0, peak_depth,
        expected_range=26.4,  # ICRU Report 73
        save_path='bragg_peak_C12_400MeV.png'
    )
    plt.show()

    # Example 2: Proton Bragg peak
    print("\n" + "="*70)
    print("Example 2: Proton Bragg Peak")
    print("="*70)

    depth_p, dose_p, peak_p, ratio_p = simulate_bragg_peak(
        'proton', 200.0, n_particles=10000
    )

    fig_p = plot_bragg_peak(
        depth_p, dose_p, 'Proton', 200.0, peak_p,
        expected_range=25.9,  # NIST PSTAR
        save_path='bragg_peak_proton_200MeV.png'
    )
    plt.show()

    # Example 3: Energy comparison
    print("\n" + "="*70)
    print("Example 3: Multiple Energies")
    print("="*70)

    compare_multiple_energies('C-12', [200, 290, 400], n_particles=5000)
    plt.show()

    # Example 4: Range validation
    print("\n" + "="*70)
    print("Example 4: Range Validation")
    print("="*70)

    validation_results = validate_range()

    print("\n" + "="*70)
    print("All examples complete!")
    print("="*70 + "\n")
