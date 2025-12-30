"""
Multiple Coulomb scattering calculations.

Implements Highland/Molière theory for small-angle scattering
in amorphous materials.

Ported from COULOMB_LET.f90 lines 415-481 and 636-695.

References:
    - Highland, NIM 129, 497 (1975)
    - Molière, Z. Naturforsch. A 2, 133 (1947)
    - PDG Review of Particle Physics (Passage of particles through matter)
"""

import numpy as np
import numba
from typing import Tuple

# Material properties database
MATERIAL_PROPERTIES = {
    'water': {
        'Z': 7.42,           # Effective Z
        'A': 13.37,          # Effective A
        'rho': 1.0,          # Density [g/cm³]
        'X0': 36.08,         # Radiation length [cm]
        'I': 75.0,           # Mean excitation energy [eV]
    },
    'aluminum': {
        'Z': 13.0,
        'A': 26.98,
        'rho': 2.70,
        'X0': 8.897,
        'I': 166.0,
    },
    'polyethylene': {
        'Z': 5.45,
        'A': 11.19,
        'rho': 0.94,
        'X0': 47.46,
        'I': 57.4,
    },
    'muscle': {
        'Z': 7.46,
        'A': 13.52,
        'rho': 1.05,
        'X0': 34.36,
        'I': 74.7,
    },
    'air': {
        'Z': 7.37,
        'A': 14.46,
        'rho': 0.001205,
        'X0': 30423,
        'I': 85.7,
    },
    'graphite': {
        'Z': 6.0,
        'A': 12.01,
        'rho': 2.21,
        'X0': 19.32,
        'I': 78.0,
    },
}


@numba.njit(fastmath=True, cache=True)
def highland_angle(energy_MeV_u: float, Z_particle: float, A_particle: float,
                   step_length_cm: float, X0_cm: float) -> float:
    """
    Calculate RMS scattering angle using Highland approximation.

    Ported from COULOMB_LET.f90:415-481 (calculate_rms_angle)

    Highland formula (accurate to ~11% for 10⁻³ < x/X0 < 100):
        θ_rms = (13.6 MeV / βcp) * Z * sqrt(x/X0) * [1 + 0.038*ln(x/X0)]

    where:
        β = v/c (velocity)
        c = speed of light
        p = momentum [MeV/c]
        Z = charge of incident particle
        x = path length
        X0 = radiation length

    Parameters:
        energy_MeV_u: Kinetic energy per nucleon [MeV/u]
        Z_particle: Atomic number of incident particle
        A_particle: Atomic mass of incident particle [u]
        step_length_cm: Path length in material [cm]
        X0_cm: Radiation length of material [cm]

    Returns:
        RMS scattering angle [radians]

    References:
        Highland, NIM 129, 497 (1975)
    """
    # Constants
    proton_mass = 938.272  # MeV/c²

    # Total kinetic energy
    E_kinetic = energy_MeV_u * A_particle  # MeV

    # Rest mass energy
    E_rest = proton_mass * A_particle  # MeV

    # Total energy
    E_total = E_rest + E_kinetic

    # Momentum
    momentum = np.sqrt(E_total**2 - E_rest**2)  # MeV/c

    # Velocity (β = v/c)
    beta = momentum / E_total

    # Momentum × velocity product
    beta_p = beta * momentum  # MeV

    # Path length in radiation lengths
    x_over_X0 = step_length_cm / X0_cm

    # Highland formula
    if x_over_X0 > 1e-10:  # Avoid log(0)
        theta_rms = (13.6 / beta_p) * Z_particle * np.sqrt(x_over_X0) * \
                    (1.0 + 0.038 * np.log(x_over_X0))
    else:
        theta_rms = 0.0

    return theta_rms


@numba.njit(fastmath=True, cache=True)
def moliere_angle_detailed(energy_MeV_u: float, Z_particle: float, A_particle: float,
                           step_length_cm: float, X0_cm: float, Z_target: float) -> float:
    """
    Calculate RMS scattering angle using full Molière theory.

    More accurate than Highland for extreme cases, but slower.

    Molière's formula includes:
        - Screening parameter (χ_a)
        - Coulomb logarithm corrections
        - Finite nuclear size effects

    Parameters:
        Same as highland_angle, plus:
        Z_target: Effective atomic number of target material

    Returns:
        RMS scattering angle [radians]

    References:
        Molière, Z. Naturforsch. A 2, 133 (1947)
        Bethe, Phys. Rev. 89, 1256 (1953)
    """
    # For now, use Highland (Molière implementation is complex)
    # TODO: Implement full Molière formula for high-precision work
    return highland_angle(energy_MeV_u, Z_particle, A_particle, step_length_cm, X0_cm)


@numba.njit(fastmath=True, cache=True)
def sample_scattering_angle(theta_rms: float) -> Tuple[float, float]:
    """
    Sample scattering angles from Gaussian distribution.

    Ported from COULOMB_LET.f90:636-695 (scatter_particle)

    Multiple Coulomb scattering is well-approximated by a Gaussian
    for small angles (θ << 1 radian), which is valid for most
    radiation transport scenarios.

    Parameters:
        theta_rms: RMS scattering angle [radians]

    Returns:
        (theta, phi): Polar and azimuthal scattering angles [radians]
            theta: Scattering angle (0 to π)
            phi: Azimuthal angle (0 to 2π)

    Notes:
        - For large-angle scattering, use nuclear elastic scattering models
        - Gaussian approximation breaks down for θ > 0.5 radians
    """
    # Sample from Gaussian distribution
    theta = np.abs(np.random.normal(0.0, theta_rms))

    # Uniform azimuthal angle
    phi = np.random.uniform(0.0, 2.0 * np.pi)

    return theta, phi


@numba.njit(fastmath=True, cache=True)
def rotate_direction(direction: np.ndarray, theta: float, phi: float) -> np.ndarray:
    """
    Rotate direction vector by scattering angles (theta, phi).

    Uses Rodrigues' rotation formula:
        v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))

    where k is the rotation axis perpendicular to v.

    Parameters:
        direction: Initial direction unit vector [x, y, z]
        theta: Polar scattering angle [radians]
        phi: Azimuthal angle [radians]

    Returns:
        Rotated direction unit vector [x, y, z]

    Algorithm:
        1. Find perpendicular axis in x-y plane
        2. Rotate by phi around original direction
        3. Rotate by theta around perpendicular axis
    """
    # Current direction (assumed normalized)
    ux, uy, uz = direction[0], direction[1], direction[2]

    # Small angle: no scattering
    if theta < 1e-10:
        return direction.copy()

    # Create perpendicular axis for rotation
    # If uz ≈ ±1, use different perpendicular
    if abs(uz) > 0.99:
        # Direction nearly along z-axis
        perp_x = 1.0
        perp_y = 0.0
        perp_z = 0.0
    else:
        # Perpendicular in x-y plane
        norm = np.sqrt(ux**2 + uy**2)
        perp_x = -uy / norm
        perp_y = ux / norm
        perp_z = 0.0

    # Rotate perpendicular axis by phi around original direction
    # This gives the scattering plane orientation
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)

    # Rotation matrix around direction vector
    # Using compact form for rotation by phi
    temp_x = perp_x * cos_phi + (uy * perp_z - uz * perp_y) * sin_phi
    temp_y = perp_y * cos_phi + (uz * perp_x - ux * perp_z) * sin_phi
    temp_z = perp_z * cos_phi + (ux * perp_y - uy * perp_x) * sin_phi

    # Now rotate by theta around this new perpendicular axis
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Rodrigues' formula
    new_x = ux * cos_theta + (temp_y * uz - temp_z * uy) * sin_theta + \
            temp_x * (temp_x * ux + temp_y * uy + temp_z * uz) * (1.0 - cos_theta)
    new_y = uy * cos_theta + (temp_z * ux - temp_x * uz) * sin_theta + \
            temp_y * (temp_x * ux + temp_y * uy + temp_z * uz) * (1.0 - cos_theta)
    new_z = uz * cos_theta + (temp_x * uy - temp_y * ux) * sin_theta + \
            temp_z * (temp_x * ux + temp_y * uy + temp_z * uz) * (1.0 - cos_theta)

    # Normalize (should be normalized already, but ensure numerical stability)
    norm = np.sqrt(new_x**2 + new_y**2 + new_z**2)

    result = np.empty(3, dtype=np.float64)
    result[0] = new_x / norm
    result[1] = new_y / norm
    result[2] = new_z / norm

    return result


@numba.njit(fastmath=True, cache=True)
def scatter_particle(direction: np.ndarray, theta_rms: float) -> np.ndarray:
    """
    Apply multiple Coulomb scattering to particle direction.

    Combines angle sampling and direction rotation.

    Parameters:
        direction: Initial direction unit vector [x, y, z]
        theta_rms: RMS scattering angle [radians]

    Returns:
        New direction unit vector [x, y, z]
    """
    theta, phi = sample_scattering_angle(theta_rms)
    return rotate_direction(direction, theta, phi)


class MultipleScattering:
    """
    High-level interface for multiple Coulomb scattering calculations.

    Usage:
        ms = MultipleScattering(material='water')
        theta_rms = ms.calculate_rms_angle(energy=400, Z=6, A=12, step=0.1)
        new_direction = ms.scatter(old_direction, theta_rms)
    """

    def __init__(self, material: str = 'water'):
        """
        Initialize scattering calculator.

        Parameters:
            material: Material name (see MATERIAL_PROPERTIES)
        """
        self.material = material.lower()

        if self.material not in MATERIAL_PROPERTIES:
            raise ValueError(f"Unknown material '{material}'. "
                           f"Available: {list(MATERIAL_PROPERTIES.keys())}")

        self.props = MATERIAL_PROPERTIES[self.material]

    def calculate_rms_angle(self, energy_MeV_u: float, Z: float, A: float,
                           step_length_cm: float) -> float:
        """
        Calculate RMS scattering angle for a given step.

        Parameters:
            energy_MeV_u: Kinetic energy per nucleon [MeV/u]
            Z: Atomic number of particle
            A: Atomic mass of particle [u]
            step_length_cm: Step length [cm]

        Returns:
            RMS scattering angle [radians]
        """
        return highland_angle(energy_MeV_u, Z, A, step_length_cm, self.props['X0'])

    def scatter(self, direction: np.ndarray, theta_rms: float) -> np.ndarray:
        """
        Apply scattering to direction vector.

        Parameters:
            direction: Initial direction [x, y, z]
            theta_rms: RMS scattering angle [radians]

        Returns:
            New direction [x, y, z]
        """
        return scatter_particle(direction, theta_rms)

    def get_radiation_length(self) -> float:
        """Get radiation length of material [cm]."""
        return self.props['X0']

    def get_density(self) -> float:
        """Get density of material [g/cm³]."""
        return self.props['rho']


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    # Test Highland formula
    print("\n" + "="*70)
    print("Multiple Coulomb Scattering Test")
    print("="*70)

    # Carbon-12 at 400 MeV/u in water
    energy = 400.0  # MeV/u
    Z = 6.0
    A = 12.0
    step = 0.1  # cm

    ms = MultipleScattering('water')

    theta_rms = ms.calculate_rms_angle(energy, Z, A, step)

    print(f"\nCarbon-12 @ {energy} MeV/u in water:")
    print(f"  Step length: {step} cm")
    print(f"  RMS angle: {theta_rms*1000:.3f} mrad")
    print(f"  RMS angle: {np.degrees(theta_rms):.4f} degrees")

    # Test scattering over many steps
    print(f"\nScattering simulation (1000 particles, 100 steps):")

    n_particles = 1000
    n_steps = 100

    # Initial direction (along z-axis)
    initial_dir = np.array([0.0, 0.0, 1.0])

    # Final directions after transport
    final_directions = np.zeros((n_particles, 3))

    for i in range(n_particles):
        direction = initial_dir.copy()

        # Transport through 100 steps
        for step_num in range(n_steps):
            theta_rms = ms.calculate_rms_angle(energy, Z, A, step)
            direction = ms.scatter(direction, theta_rms)

        final_directions[i] = direction

    # Calculate angular spread
    angles = np.arccos(np.clip(final_directions[:, 2], -1.0, 1.0))
    mean_angle = np.mean(angles)
    rms_angle = np.sqrt(np.mean(angles**2))

    print(f"  Mean deflection: {mean_angle*1000:.2f} mrad")
    print(f"  RMS deflection: {rms_angle*1000:.2f} mrad")

    # Theoretical prediction
    total_length = n_steps * step
    X0 = ms.get_radiation_length()

    # Highland formula for total path
    theta_theory = highland_angle(energy, Z, A, total_length, X0)

    print(f"  Theoretical RMS: {theta_theory*1000:.2f} mrad")
    print(f"  Agreement: {(rms_angle/theta_theory - 1.0)*100:.1f}% difference")

    # Test with different materials
    print(f"\nRMS angles for 0.1 cm step in different materials:")
    for mat in ['water', 'air', 'aluminum', 'polyethylene']:
        ms_mat = MultipleScattering(mat)
        theta = ms_mat.calculate_rms_angle(energy, Z, A, 0.1)
        print(f"  {mat:15s}: {theta*1000:6.3f} mrad (X0={ms_mat.get_radiation_length():.2f} cm)")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70 + "\n")
