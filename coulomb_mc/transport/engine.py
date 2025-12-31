"""
Monte Carlo transport engine for heavy-ion radiation.

Integrates:
    - Stopping power (continuous energy loss)
    - Multiple Coulomb scattering
    - Particle tracking through geometry
    - Step-size control

Ported from COULOMB_LET.f90 with significant architectural improvements.
"""

import numpy as np
import numba
from typing import Optional, Tuple
from pathlib import Path

from coulomb_mc.physics.stopping_power import StoppingPower
from coulomb_mc.physics.scattering import MultipleScattering, highland_angle
from coulomb_mc.core.particle import ParticleArray, PARTICLE_DTYPE


@numba.njit(fastmath=True, cache=True)  # Removed parallel=True - causes 3.6x slowdown!
def transport_step_with_scattering_adaptive(particles: np.ndarray, step_lengths: np.ndarray,
                                            energy_loss: np.ndarray, theta_rms: np.ndarray):
    """
    Advance particles one step with INDIVIDUAL step sizes, energy loss, and scattering.

    This version allows each particle to take its own step size based on remaining range.

    Parameters:
        particles: Structured array of particles (PARTICLE_DTYPE)
        step_lengths: Step length per particle [cm]
        energy_loss: Energy loss per particle [MeV/u]
        theta_rms: RMS scattering angle per particle [radians]

    Modifies particles in-place.
    """
    for i in range(len(particles)):  # Changed from numba.prange
        if particles['alive'][i]:
            step_length = step_lengths[i]

            # Update position
            particles['position'][i, 0] += step_length * particles['direction'][i, 0]
            particles['position'][i, 1] += step_length * particles['direction'][i, 1]
            particles['position'][i, 2] += step_length * particles['direction'][i, 2]

            # Update energy
            particles['energy'][i] -= energy_loss[i]

            # Kill particle if energy too low
            if particles['energy'][i] < 0.1:  # 0.1 MeV/u cutoff
                particles['alive'][i] = False
            else:
                # Apply scattering (inlined from scatter_particle to avoid nested JIT)
                if theta_rms[i] > 1e-10:  # Only scatter if angle is significant
                    # Sample scattering angle from Gaussian
                    theta = np.abs(np.random.normal(0.0, theta_rms[i]))
                    phi = np.random.uniform(0.0, 2.0 * np.pi)

                    # Rotate direction using Rodrigues formula
                    # Get current direction
                    dx = particles['direction'][i, 0]
                    dy = particles['direction'][i, 1]
                    dz = particles['direction'][i, 2]

                    # Create rotation axis perpendicular to direction
                    # Using (0,0,1) as reference unless direction is already along z
                    if np.abs(dz) < 0.9:
                        # Use z-axis as reference
                        kx = -dy
                        ky = dx
                        kz = 0.0
                    else:
                        # Use x-axis as reference
                        kx = 0.0
                        ky = -dz
                        kz = dy

                    # Normalize rotation axis
                    k_norm = np.sqrt(kx*kx + ky*ky + kz*kz)
                    if k_norm > 1e-10:
                        kx /= k_norm
                        ky /= k_norm
                        kz /= k_norm

                        # Rotate axis by azimuthal angle phi
                        cos_phi = np.cos(phi)
                        sin_phi = np.sin(phi)

                        # Apply polar rotation by theta using Rodrigues formula
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)

                        # v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
                        # Cross product: k × v
                        cross_x = ky * dz - kz * dy
                        cross_y = kz * dx - kx * dz
                        cross_z = kx * dy - ky * dx

                        # Dot product: k · v
                        dot = kx * dx + ky * dy + kz * dz

                        # Rodrigues formula
                        one_minus_cos = 1.0 - cos_theta
                        new_dx = dx * cos_theta + cross_x * sin_theta + kx * dot * one_minus_cos
                        new_dy = dy * cos_theta + cross_y * sin_theta + ky * dot * one_minus_cos
                        new_dz = dz * cos_theta + cross_z * sin_theta + kz * dot * one_minus_cos

                        # Normalize (important for numerical stability)
                        norm = np.sqrt(new_dx*new_dx + new_dy*new_dy + new_dz*new_dz)
                        if norm > 1e-10:
                            particles['direction'][i, 0] = new_dx / norm
                            particles['direction'][i, 1] = new_dy / norm
                            particles['direction'][i, 2] = new_dz / norm


@numba.njit(fastmath=True, cache=True)  # Removed parallel=True - causes 3.6x slowdown!
def transport_step_with_scattering(particles: np.ndarray, step_length: float,
                                   energy_loss: np.ndarray, theta_rms: np.ndarray):
    """
    Advance particles one step with energy loss and scattering.

    Note: parallel=True removed - it caused 3.6x slowdown due to synchronization overhead.
    Particle-level parallelization will be implemented at a higher level.

    Parameters:
        particles: Structured array of particles (PARTICLE_DTYPE)
        step_length: Step length [cm]
        energy_loss: Energy loss per particle [MeV]
        theta_rms: RMS scattering angle per particle [radians]

    Modifies particles in-place.

    Note: Scattering logic is inlined here to avoid Numba nested JIT issues.
    """
    for i in range(len(particles)):  # Changed from numba.prange
        if particles['alive'][i]:
            # Update position
            particles['position'][i, 0] += step_length * particles['direction'][i, 0]
            particles['position'][i, 1] += step_length * particles['direction'][i, 1]
            particles['position'][i, 2] += step_length * particles['direction'][i, 2]

            # Update energy
            particles['energy'][i] -= energy_loss[i]

            # Kill particle if energy too low
            if particles['energy'][i] < 0.1:  # 0.1 MeV/u cutoff
                particles['alive'][i] = False
            else:
                # Apply scattering (inlined from scatter_particle to avoid nested JIT)
                if theta_rms[i] > 1e-10:  # Only scatter if angle is significant
                    # Sample scattering angle from Gaussian
                    theta = np.abs(np.random.normal(0.0, theta_rms[i]))
                    phi = np.random.uniform(0.0, 2.0 * np.pi)

                    # Rotate direction using Rodrigues formula
                    # Get current direction
                    dx = particles['direction'][i, 0]
                    dy = particles['direction'][i, 1]
                    dz = particles['direction'][i, 2]

                    # Create rotation axis perpendicular to direction
                    # Using (0,0,1) as reference unless direction is already along z
                    if np.abs(dz) < 0.9:
                        # Use z-axis as reference
                        kx = -dy
                        ky = dx
                        kz = 0.0
                    else:
                        # Use x-axis as reference
                        kx = 0.0
                        ky = -dz
                        kz = dy

                    # Normalize rotation axis
                    k_norm = np.sqrt(kx*kx + ky*ky + kz*kz)
                    if k_norm > 1e-10:
                        kx /= k_norm
                        ky /= k_norm
                        kz /= k_norm

                        # Rotate axis by azimuthal angle phi
                        cos_phi = np.cos(phi)
                        sin_phi = np.sin(phi)

                        # Rotate k around direction vector by angle phi
                        # This is a simplified version - full rotation would use matrix
                        # For small angles this approximation is sufficient

                        # Apply polar rotation by theta using Rodrigues formula
                        cos_theta = np.cos(theta)
                        sin_theta = np.sin(theta)

                        # v_rot = v*cos(θ) + (k × v)*sin(θ) + k*(k·v)*(1-cos(θ))
                        # Cross product: k × v
                        cross_x = ky * dz - kz * dy
                        cross_y = kz * dx - kx * dz
                        cross_z = kx * dy - ky * dx

                        # Dot product: k · v
                        dot = kx * dx + ky * dy + kz * dz

                        # Rodrigues formula
                        one_minus_cos = 1.0 - cos_theta
                        new_dx = dx * cos_theta + cross_x * sin_theta + kx * dot * one_minus_cos
                        new_dy = dy * cos_theta + cross_y * sin_theta + ky * dot * one_minus_cos
                        new_dz = dz * cos_theta + cross_z * sin_theta + kz * dot * one_minus_cos

                        # Normalize (important for numerical stability)
                        norm = np.sqrt(new_dx*new_dx + new_dy*new_dy + new_dz*new_dz)
                        if norm > 1e-10:
                            particles['direction'][i, 0] = new_dx / norm
                            particles['direction'][i, 1] = new_dy / norm
                            particles['direction'][i, 2] = new_dz / norm


@numba.njit(fastmath=True, cache=True)
def calculate_step_size(energy_MeV_u: float, stopping_power_MeV_cm2_g: float,
                       rho_g_cm3: float, max_energy_loss_fraction: float = 0.01) -> float:
    """
    Calculate adaptive step size to limit energy loss per step.

    Ensures:
        ΔE / E < max_energy_loss_fraction (default 1%)

    This maintains accuracy of continuous slowing-down approximation.

    Parameters:
        energy_MeV_u: Kinetic energy per nucleon [MeV/u]
        stopping_power_MeV_cm2_g: Stopping power [MeV cm²/g]
        rho_g_cm3: Material density [g/cm³]
        max_energy_loss_fraction: Maximum fractional energy loss per step

    Returns:
        Step size [cm]
    """
    # Convert stopping power to MeV/cm
    dEdx = stopping_power_MeV_cm2_g * rho_g_cm3

    if dEdx < 1e-10:
        return 10.0  # Large step if no energy loss

    # Step size for target energy loss
    max_energy_loss = energy_MeV_u * max_energy_loss_fraction
    step = max_energy_loss / dEdx

    # Limit step size (min 0.001 cm, max 10 cm)
    step = max(0.001, min(step, 10.0))

    return step


@numba.njit(fastmath=True, cache=True)
def calculate_adaptive_step_size(energy_MeV_u: float, A: float, Z: float,
                                stopping_power_MeV_cm2_g: float,
                                remaining_range_cm: float, rho_g_cm3: float,
                                max_energy_loss_fraction: float = 0.01,
                                range_fraction_limit: float = 0.02) -> float:
    """
    Calculate adaptive step size based on both energy loss and remaining range.

    Uses range-based adaptation to take finer steps near Bragg peak:
    - Far from peak: limited by max_energy_loss_fraction (1%)
    - Near peak: limited by range_fraction_limit (2% of remaining range)

    This prevents taking huge steps that overshoot the Bragg peak and
    ensures better resolution near end of range.

    Parameters:
        energy_MeV_u: Kinetic energy per nucleon [MeV/u]
        A: Atomic mass number
        Z: Atomic number
        stopping_power_MeV_cm2_g: Stopping power [MeV cm²/g]
        remaining_range_cm: Estimated remaining range [cm]
        rho_g_cm3: Material density [g/cm³]
        max_energy_loss_fraction: Maximum fractional energy loss per step
        range_fraction_limit: Maximum fraction of remaining range per step

    Returns:
        Step size [cm]
    """
    # Step size based on energy loss (as before)
    dEdx = stopping_power_MeV_cm2_g * rho_g_cm3
    if dEdx > 1e-10:
        max_energy_loss = energy_MeV_u * max_energy_loss_fraction
        step_energy = max_energy_loss / dEdx
    else:
        step_energy = 10.0

    # Step size based on remaining range
    # Take smaller steps as we approach the Bragg peak
    if remaining_range_cm > 0.0:
        step_range = remaining_range_cm * range_fraction_limit
    else:
        step_range = 0.001  # Minimum if range unknown

    # Use the more restrictive limit
    step = min(step_energy, step_range)

    # Apply absolute limits
    # Use smaller minimum (0.0001 cm = 1 micron) for better resolution near Bragg peak
    # Max 10 cm for efficiency far from peak
    step = max(0.0001, min(step, 10.0))

    return step


# Global engine instance for each worker process
_worker_engine = None

def _init_worker(material, max_energy_loss_fraction):
    """Initialize worker process with shared engine instance."""
    global _worker_engine
    _worker_engine = TransportEngine(material=material,
                                     max_energy_loss_fraction=max_energy_loss_fraction)

def _transport_particle_worker(work_item):
    """
    Worker function for parallel particle transport.

    Uses a global engine instance created once per worker process
    to avoid repeatedly loading NIST data.

    Parameters:
        work_item: Dictionary with particle state

    Returns:
        Tuple of (final_particle, dose_contributions)
    """
    global _worker_engine

    # Extract parameters
    particle_id = work_item['particle_id']
    max_depth = work_item['max_depth']

    # Create initial state
    initial_state = {
        'position': work_item['position'],
        'direction': work_item['direction'],
        'energy': work_item['energy'],
        'A': work_item['A'],
        'Z': work_item['Z'],
        'weight': work_item['weight']
    }

    # Transport this particle using the shared engine
    return _worker_engine._transport_single_particle(particle_id, initial_state, max_depth)


class TransportEngine:
    """
    Main transport engine for Monte Carlo simulation.

    Handles:
        - Particle initialization
        - Step-by-step transport
        - Energy loss and scattering
        - Geometry boundaries (future)
        - Dose scoring (future)

    Example:
        engine = TransportEngine(material='water')
        beam = engine.create_beam('C-12', 400, n_particles=10000)
        engine.transport(beam, max_depth=30.0)
        depth, dose = engine.get_dose_depth()
    """

    def __init__(self, material: str = 'water', max_energy_loss_fraction: float = 0.01):
        """
        Initialize transport engine.

        Parameters:
            material: Material name
            max_energy_loss_fraction: Max ΔE/E per step (controls accuracy)
        """
        self.material = material
        self.max_energy_loss_fraction = max_energy_loss_fraction

        # Initialize physics modules
        self.stopping_power = StoppingPower(material)
        self.scattering = MultipleScattering(material)

        # Material properties
        self.density = self.scattering.get_density()
        self.X0 = self.scattering.get_radiation_length()

        # Scoring arrays (depth dose)
        self.dose_bins = None
        self.dose_deposit = None
        self.depth_min = 0.0
        self.depth_max = 50.0
        self.n_bins = 500

        self._initialize_scoring()

    def _initialize_scoring(self):
        """Initialize dose scoring arrays."""
        self.dose_bins = np.linspace(self.depth_min, self.depth_max, self.n_bins + 1)
        self.dose_deposit = np.zeros(self.n_bins)

    def create_beam(self, particle_type: str, energy_MeV_u: float,
                   n_particles: int = 10000,
                   position: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                   direction: Tuple[float, float, float] = (0.0, 0.0, 1.0),
                   energy_spread: float = 0.0) -> ParticleArray:
        """
        Create particle beam.

        Parameters:
            particle_type: Particle name (e.g., 'C-12', 'proton')
            energy_MeV_u: Beam energy [MeV/u]
            n_particles: Number of particles
            position: Beam starting position [cm]
            direction: Beam direction (normalized automatically)
            energy_spread: Energy spread (σ) [MeV/u]

        Returns:
            ParticleArray initialized with beam
        """
        beam = ParticleArray(n_particles)
        beam.initialize_beam(particle_type, energy_MeV_u, position, direction, energy_spread)
        return beam

    def transport(self, beam: ParticleArray, max_depth: float = 50.0,
                 verbose: bool = True) -> dict:
        """
        Transport particles through material.

        Parameters:
            beam: ParticleArray to transport
            max_depth: Maximum depth to simulate [cm]
            verbose: Print progress information

        Returns:
            Dictionary with simulation statistics
        """
        n_particles = beam.n_particles
        n_alive_initial = beam.n_alive

        step_count = 0
        total_distance = 0.0

        if verbose:
            print(f"\nTransporting {n_alive_initial} particles...")
            print(f"  Material: {self.material}")
            print(f"  Density: {self.density} g/cm³")
            print(f"  Max depth: {max_depth} cm")

        # Main transport loop
        while beam.n_alive > 0:
            # Get current particle states
            alive_mask = beam.particles['alive']
            energies = beam.particles['energy'][alive_mask]
            A_values = beam.particles['A'][alive_mask]
            Z_values = beam.particles['Z'][alive_mask]
            positions = beam.particles['position'][alive_mask]

            # Check depth cutoff
            depths = positions[:, 2]  # z-coordinate
            beyond_max = depths > max_depth

            if np.any(beyond_max):
                # Kill particles beyond max depth
                alive_indices = np.where(alive_mask)[0]
                beam.particles['alive'][alive_indices[beyond_max]] = False

                if beam.n_alive == 0:
                    break

                # Recalculate alive particles
                alive_mask = beam.particles['alive']
                energies = beam.particles['energy'][alive_mask]
                A_values = beam.particles['A'][alive_mask]
                Z_values = beam.particles['Z'][alive_mask]
                positions = beam.particles['position'][alive_mask]

            # Calculate physics for alive particles
            n_alive = np.sum(alive_mask)

            # Arrays for physics calculations
            stopping_powers = np.zeros(n_alive)
            energy_losses = np.zeros(n_alive)
            theta_rms_array = np.zeros(n_alive)
            step_sizes = np.zeros(n_alive)

            # Calculate step size and physics for each particle
            for i in range(n_alive):
                # Stopping power
                sp = self.stopping_power.stopping_power_MeV_cm2_g(
                    energies[i], A_values[i], Z_values[i]
                )
                stopping_powers[i] = sp

                # Calculate remaining range for range-based adaptive stepping
                remaining_range = self.stopping_power.range_g_cm2(
                    energies[i], A_values[i], Z_values[i]
                ) / self.density  # Convert g/cm² to cm

                # Range-based adaptive step size
                step = calculate_adaptive_step_size(
                    energies[i], A_values[i], Z_values[i], sp,
                    remaining_range, self.density,
                    self.max_energy_loss_fraction,
                    range_fraction_limit=0.05  # 5% of remaining range for better resolution
                )
                step_sizes[i] = step

                # Energy loss for this particle's actual step
                dEdx = sp * self.density  # MeV/cm (total ion)
                energy_loss_total = dEdx * step  # MeV (total)
                energy_losses[i] = energy_loss_total / A_values[i]  # MeV/u

                # Scattering angle for this particle's actual step
                theta_rms_array[i] = highland_angle(
                    energies[i], Z_values[i], A_values[i], step, self.X0
                )

            # Create full arrays (including dead particles) with individual step sizes
            energy_loss_full = np.zeros(n_particles)
            theta_rms_full = np.zeros(n_particles)
            step_sizes_full = np.zeros(n_particles)

            energy_loss_full[alive_mask] = energy_losses
            theta_rms_full[alive_mask] = theta_rms_array
            step_sizes_full[alive_mask] = step_sizes

            # Score dose using individual step sizes
            self._score_dose_adaptive(beam.particles, energy_loss_full, step_sizes_full)

            # Advance particles with individual step sizes
            transport_step_with_scattering_adaptive(
                beam.particles,
                step_sizes_full,
                energy_loss_full,
                theta_rms_full
            )

            # Track average step for progress reporting
            step_length = np.mean(step_sizes)

            step_count += 1
            total_distance += step_length

            # Progress update
            if verbose and step_count % 100 == 0:
                print(f"  Step {step_count}: {beam.n_alive}/{n_alive_initial} alive, "
                      f"depth={total_distance:.2f} cm")

        if verbose:
            print(f"\nTransport complete!")
            print(f"  Total steps: {step_count}")
            print(f"  Average distance: {total_distance:.2f} cm")
            print(f"  Particles stopped: {n_alive_initial - beam.n_alive}")

        return {
            'n_steps': step_count,
            'total_distance': total_distance,
            'n_stopped': n_alive_initial - beam.n_alive,
        }

    def _score_dose_adaptive(self, particles: np.ndarray, energy_loss: np.ndarray,
                            step_lengths: np.ndarray):
        """
        Score dose deposition with individual step sizes per particle.

        Uses midpoint of particle step to reduce statistical noise from scattering.

        Parameters:
            particles: Particle array
            energy_loss: Energy loss per particle [MeV/u]
            step_lengths: Step length per particle [cm]
        """
        alive_mask = particles['alive']

        # Get positions AFTER the step (current position)
        z_after = particles['position'][alive_mask, 2]

        # Get direction vectors and step lengths for alive particles
        directions = particles['direction'][alive_mask]
        steps = step_lengths[alive_mask]

        # Calculate position at MIDPOINT of step
        z_mid = z_after - 0.5 * steps * directions[:, 2]

        weights = particles['weight'][alive_mask]
        A_values = particles['A'][alive_mask]

        # Convert energy loss from MeV/u to total MeV for dose scoring
        energy_deposited = energy_loss[alive_mask] * A_values

        # Find bin indices using midpoint depth
        bin_indices = np.digitize(z_mid, self.dose_bins) - 1

        # Accumulate dose in bins
        for i in range(len(z_mid)):
            if 0 <= bin_indices[i] < self.n_bins:
                self.dose_deposit[bin_indices[i]] += energy_deposited[i] * weights[i]

    def _score_dose(self, particles: np.ndarray, energy_loss: np.ndarray,
                   step_length: float):
        """
        Score dose deposition in depth bins.

        Uses midpoint of particle step to reduce statistical noise from scattering.

        Parameters:
            particles: Particle array
            energy_loss: Energy loss per particle [MeV/u]
            step_length: Step length [cm]
        """
        alive_mask = particles['alive']

        # Get positions AFTER the step (current position)
        z_after = particles['position'][alive_mask, 2]

        # Get direction vectors
        directions = particles['direction'][alive_mask]

        # Calculate position at MIDPOINT of step (more accurate for binning)
        # Midpoint = current_position - (step_length / 2) * direction
        # We only care about z-component for 1D dose
        z_mid = z_after - 0.5 * step_length * directions[:, 2]

        weights = particles['weight'][alive_mask]
        A_values = particles['A'][alive_mask]

        # Convert energy loss from MeV/u to total MeV for dose scoring
        energy_deposited = energy_loss[alive_mask] * A_values

        # Find bin indices using midpoint depth
        bin_indices = np.digitize(z_mid, self.dose_bins) - 1

        # Accumulate dose in bins
        for i in range(len(z_mid)):
            if 0 <= bin_indices[i] < self.n_bins:
                self.dose_deposit[bin_indices[i]] += energy_deposited[i] * weights[i]

    def _transport_single_particle(self, particle_id: int, initial_state: dict,
                                   max_depth: float) -> Tuple[np.ndarray, list]:
        """
        Transport a single particle from initialization to death.

        This function is designed for parallel execution - each particle is
        transported independently without synchronization.

        Parameters:
            particle_id: Particle identifier
            initial_state: Initial particle state dict with keys:
                          position, direction, energy, A, Z, weight
            max_depth: Maximum depth cutoff [cm]

        Returns:
            final_particle: Final particle state (1-element array)
            dose_contributions: List of dose deposition events
        """
        # Create single particle array
        particle = np.zeros(1, dtype=PARTICLE_DTYPE)
        particle['alive'][0] = True
        particle['position'][0] = initial_state['position']
        particle['direction'][0] = initial_state['direction']
        particle['energy'][0] = initial_state['energy']
        particle['A'][0] = initial_state['A']
        particle['Z'][0] = initial_state['Z']
        particle['weight'][0] = initial_state['weight']

        dose_contributions = []
        step_count = 0

        while particle['alive'][0] and particle['position'][0, 2] < max_depth:
            # Get current state
            energy = particle['energy'][0]
            A = particle['A'][0]
            Z = particle['Z'][0]
            position = particle['position'][0].copy()

            # Calculate stopping power
            sp = self.stopping_power.stopping_power_MeV_cm2_g(energy, A, Z)

            # Calculate remaining range
            remaining_range = self.stopping_power.range_g_cm2(energy, A, Z) / self.density

            # Calculate adaptive step size
            step = calculate_adaptive_step_size(
                energy, A, Z, sp, remaining_range, self.density,
                self.max_energy_loss_fraction, range_fraction_limit=0.05
            )

            # Calculate energy loss
            dEdx = sp * self.density  # MeV/cm
            energy_loss_total = dEdx * step  # MeV
            energy_loss_per_u = energy_loss_total / A  # MeV/u

            # Calculate scattering angle
            theta_rms = highland_angle(energy, Z, A, step, self.X0)

            # Record dose contribution (before step)
            dose_contributions.append({
                'position': position.copy(),
                'depth': position[2],
                'energy_deposited': energy_loss_total,
                'step_length': step,
                'weight': particle['weight'][0]
            })

            # Advance particle
            transport_step_with_scattering_adaptive(
                particle,
                np.array([step]),
                np.array([energy_loss_per_u]),
                np.array([theta_rms])
            )

            step_count += 1

        return particle, dose_contributions

    def transport_parallel(self, beam: ParticleArray, max_depth: float = 50.0,
                          n_processes: int = None, verbose: bool = True) -> dict:
        """
        Transport particles in parallel using multiprocessing.

        This implements particle-level parallelization where each particle
        is transported independently from birth to death. No synchronization
        is required during transport - only at the end for dose accumulation.

        Parameters:
            beam: ParticleArray with initial particle states
            max_depth: Maximum depth to transport [cm]
            n_processes: Number of parallel processes (default: cpu_count)
            verbose: Print progress information

        Returns:
            Dictionary with transport statistics
        """
        import multiprocessing as mp
        import time

        if n_processes is None:
            n_processes = mp.cpu_count()

        n_particles = len(beam.particles)

        if verbose:
            print(f"\nParallel transport: {n_particles} particles on {n_processes} cores")
            print(f"  Material: {self.material}")
            print(f"  Density: {self.density} g/cm³")
            print(f"  Max depth: {max_depth} cm")

        # Prepare initial states with engine parameters
        work_items = []
        for i in range(n_particles):
            work_items.append({
                'particle_id': i,
                'position': beam.particles['position'][i].copy(),
                'direction': beam.particles['direction'][i].copy(),
                'energy': beam.particles['energy'][i],
                'A': beam.particles['A'][i],
                'Z': beam.particles['Z'][i],
                'weight': beam.particles['weight'][i],
                'max_depth': max_depth,
                'material': self.material,
                'density': self.density,
                'X0': self.X0,
                'max_energy_loss_fraction': self.max_energy_loss_fraction,
                'dose_bins': self.dose_bins
            })

        # Parallel execution
        start_time = time.time()

        with mp.Pool(n_processes, initializer=_init_worker,
                     initargs=(self.material, self.max_energy_loss_fraction)) as pool:
            results = pool.map(_transport_particle_worker, work_items)

        elapsed = time.time() - start_time

        # Accumulate dose from all particles
        total_steps = 0
        for particle_state, dose_list in results:
            total_steps += len(dose_list)
            for dose_event in dose_list:
                # Find bin index using midpoint depth
                z_mid = dose_event['depth'] + 0.5 * dose_event['step_length']
                bin_idx = np.digitize([z_mid], self.dose_bins)[0] - 1

                if 0 <= bin_idx < self.n_bins:
                    self.dose_deposit[bin_idx] += dose_event['energy_deposited'] * dose_event['weight']

        # Update beam with final states
        for i, (final_particle, _) in enumerate(results):
            beam.particles[i] = final_particle[0]

        # Calculate rate
        rate = n_particles / elapsed

        if verbose:
            print(f"\nTransport complete:")
            print(f"  Time: {elapsed:.1f}s")
            print(f"  Rate: {rate:.0f} particles/sec")
            print(f"  Total steps: {total_steps:,}")
            print(f"  Steps/particle: {total_steps/n_particles:.1f}")

        return {
            'n_particles': n_particles,
            'n_alive': np.sum(beam.particles['alive']),
            'total_steps': total_steps,
            'elapsed_time': elapsed,
            'particles_per_sec': rate
        }

    def get_dose_depth(self, normalize: bool = True, smooth: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get depth-dose distribution.

        Parameters:
            normalize: Normalize to peak dose = 100
            smooth: Apply moving average smoothing with this window size (0 = no smoothing)
                   Recommended: 3-5 for noisy data, 0 for high statistics

        Returns:
            (depth, dose): Depth [cm] and dose [arbitrary units or % of peak]
        """
        # Bin centers
        depth = (self.dose_bins[:-1] + self.dose_bins[1:]) / 2.0

        # Dose per bin
        bin_width = self.dose_bins[1] - self.dose_bins[0]
        dose = self.dose_deposit / bin_width  # Dose per cm

        # Apply smoothing if requested
        if smooth > 0:
            # Simple moving average
            kernel = np.ones(smooth) / smooth
            dose = np.convolve(dose, kernel, mode='same')

        if normalize and np.max(dose) > 0:
            dose = 100.0 * dose / np.max(dose)

        return depth, dose

    def reset_scoring(self):
        """Reset dose scoring arrays."""
        self._initialize_scoring()


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("\n" + "="*70)
    print("Transport Engine Test")
    print("="*70)

    # Create transport engine
    engine = TransportEngine(material='water')

    # Create carbon-12 beam at 400 MeV/u
    beam = engine.create_beam('C-12', 400.0, n_particles=1000)

    print(f"\nBeam created: {beam}")

    # Transport particles
    stats = engine.transport(beam, max_depth=30.0, verbose=True)

    # Get dose depth distribution
    depth, dose = engine.get_dose_depth(normalize=True)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(depth, dose, 'b-', linewidth=2)
    plt.xlabel('Depth [cm]', fontsize=12)
    plt.ylabel('Dose [% of peak]', fontsize=12)
    plt.title('Bragg Peak: C-12 @ 400 MeV/u in Water', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 30)
    plt.ylim(0, 110)

    # Find peak position
    peak_idx = np.argmax(dose)
    peak_depth = depth[peak_idx]
    plt.axvline(peak_depth, color='r', linestyle='--', alpha=0.5,
                label=f'Bragg peak at {peak_depth:.1f} cm')
    plt.legend()

    plt.tight_layout()
    plt.savefig('bragg_peak_test.png', dpi=150)
    print(f"\nPlot saved: bragg_peak_test.png")

    print(f"\nBragg peak position: {peak_depth:.2f} cm")
    print(f"Expected (ICRU): ~26.4 cm")
    print(f"Difference: {abs(peak_depth - 26.4):.1f} cm")

    print("\n" + "="*70)
    print("Test complete!")
    print("="*70 + "\n")
