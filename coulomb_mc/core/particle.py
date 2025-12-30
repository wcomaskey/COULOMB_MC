"""
Particle state management using NumPy structured arrays.

Optimized for vectorization and GPU transfer.
"""

import numpy as np
import numba
from typing import Tuple, Optional


# Define particle dtype (Structure of Arrays for SIMD)
PARTICLE_DTYPE = np.dtype([
    ('position', np.float64, 3),      # x, y, z [cm]
    ('direction', np.float64, 3),     # unit vector
    ('energy', np.float64),           # MeV/u
    ('A', np.float32),                # atomic mass
    ('Z', np.float32),                # atomic number
    ('weight', np.float64),           # statistical weight
    ('alive', np.bool_)               # is particle active?
])


class Particle:
    """Single particle (for convenience)."""

    def __init__(self, A: float, Z: float, energy: float,
                 position: Tuple[float, float, float],
                 direction: Tuple[float, float, float],
                 weight: float = 1.0):
        """
        Initialize a single particle.

        Parameters:
            A: Atomic mass [u]
            Z: Atomic number
            energy: Kinetic energy [MeV/u]
            position: (x, y, z) position [cm]
            direction: (dx, dy, dz) direction (normalized internally)
            weight: Statistical weight
        """
        self.A = A
        self.Z = Z
        self.energy = energy
        self.position = np.array(position, dtype=np.float64)

        # Normalize direction
        dir_array = np.array(direction, dtype=np.float64)
        self.direction = dir_array / np.linalg.norm(dir_array)

        self.weight = weight
        self.alive = True

    def to_structured_array(self) -> np.ndarray:
        """Convert to structured array format."""
        particle_array = np.zeros(1, dtype=PARTICLE_DTYPE)
        particle_array['position'][0] = self.position
        particle_array['direction'][0] = self.direction
        particle_array['energy'][0] = self.energy
        particle_array['A'][0] = self.A
        particle_array['Z'][0] = self.Z
        particle_array['weight'][0] = self.weight
        particle_array['alive'][0] = self.alive
        return particle_array


class ParticleArray:
    """Efficient particle state storage for Monte Carlo."""

    def __init__(self, n_particles: int):
        """
        Initialize particle array.

        Parameters:
            n_particles: Number of particles to allocate
        """
        self.particles = np.zeros(n_particles, dtype=PARTICLE_DTYPE)
        self.n_particles = n_particles

    def initialize_beam(self, particle_type: str, energy: float,
                        position: Tuple[float, float, float],
                        direction: Tuple[float, float, float],
                        energy_spread: float = 0.0):
        """
        Initialize monoenergetic parallel beam.

        Parameters:
            particle_type: 'C-12', 'He-4', 'proton', etc.
            energy: Kinetic energy [MeV/u]
            position: (x, y, z) starting position [cm]
            direction: (dx, dy, dz) direction (will be normalized)
            energy_spread: Energy spread (sigma) [MeV/u]
        """
        # Parse particle type
        A, Z = self._parse_particle_type(particle_type)

        # Normalize direction
        dir_array = np.array(direction, dtype=np.float64)
        dir_norm = dir_array / np.linalg.norm(dir_array)

        # Set all particles
        self.particles['position'] = position
        self.particles['direction'] = dir_norm
        self.particles['A'] = A
        self.particles['Z'] = Z
        self.particles['weight'] = 1.0
        self.particles['alive'] = True

        # Energy with optional spread
        if energy_spread > 0:
            energies = np.random.normal(energy, energy_spread, self.n_particles)
            energies = np.maximum(energies, 0.0)  # No negative energies
            self.particles['energy'] = energies
        else:
            self.particles['energy'] = energy

    @staticmethod
    def _parse_particle_type(particle_type: str) -> Tuple[float, float]:
        """Parse 'C-12' → (A=12, Z=6)"""
        particles = {
            'proton': (1, 1),
            'H-1': (1, 1),
            'deuteron': (2, 1),
            'H-2': (2, 1),
            'triton': (3, 1),
            'H-3': (3, 1),
            'He-3': (3, 2),
            'He-4': (4, 2),
            'alpha': (4, 2),
            'Li-7': (7, 3),
            'Be-9': (9, 4),
            'B-11': (11, 5),
            'C-12': (12, 6),
            'N-14': (14, 7),
            'O-16': (16, 8),
            'Ne-20': (20, 10),
            'Si-28': (28, 14),
            'Fe-56': (56, 26),
        }
        return particles.get(particle_type, (12, 6))  # Default C-12

    @property
    def n_alive(self) -> int:
        """Get number of alive particles."""
        return int(np.sum(self.particles['alive']))

    def get_alive_indices(self) -> np.ndarray:
        """Get indices of alive particles."""
        return np.where(self.particles['alive'])[0]

    def kill_particle(self, index: int):
        """Mark particle as dead."""
        self.particles['alive'][index] = False

    def get_statistics(self) -> dict:
        """Get statistics about particle array."""
        alive = self.particles['alive']
        energies = self.particles['energy'][alive]

        return {
            'n_total': self.n_particles,
            'n_alive': np.sum(alive),
            'n_dead': np.sum(~alive),
            'mean_energy': np.mean(energies) if len(energies) > 0 else 0.0,
            'max_energy': np.max(energies) if len(energies) > 0 else 0.0,
            'min_energy': np.min(energies) if len(energies) > 0 else 0.0,
        }

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (f"ParticleArray(n={stats['n_total']}, "
                f"alive={stats['n_alive']}, "
                f"<E>={stats['mean_energy']:.1f} MeV/u)")


# ============================================================================
# Numba-accelerated transport kernels
# ============================================================================

@numba.njit(parallel=True, fastmath=True, cache=True)
def transport_step_cpu(particles: np.ndarray, step_length: float,
                       energy_loss_rate: np.ndarray):
    """
    Advance all particles by one step (CPU parallel).

    Parameters:
        particles: Structured array of particles
        step_length: Step size [cm]
        energy_loss_rate: dE/dx for each particle [MeV/cm]
    """
    n = len(particles)

    for i in numba.prange(n):  # Parallel loop
        if particles['alive'][i]:
            # Move particle
            particles['position'][i] += step_length * particles['direction'][i]

            # Lose energy
            energy_loss = energy_loss_rate[i] * step_length
            particles['energy'][i] -= energy_loss

            # Kill if below threshold
            if particles['energy'][i] < 0.1:  # MeV
                particles['alive'][i] = False


# ============================================================================
# Example usage
# ============================================================================

if __name__ == "__main__":
    print("Creating 10k carbon ion beam...")

    # Create 10k carbon ions
    beam = ParticleArray(n_particles=10000)
    beam.initialize_beam(
        particle_type='C-12',
        energy=400.0,  # MeV/u
        position=(0, 0, -10),  # Start 10 cm before phantom
        direction=(0, 0, 1),   # Along z-axis
        energy_spread=5.0      # 5 MeV/u spread
    )

    print(f"\nInitialized: {beam}")
    print(f"Memory usage: {beam.particles.nbytes / 1e6:.2f} MB")

    stats = beam.get_statistics()
    print(f"\nStatistics:")
    print(f"  Total particles: {stats['n_total']}")
    print(f"  Alive: {stats['n_alive']}")
    print(f"  Mean energy: {stats['mean_energy']:.2f} MeV/u")
    print(f"  Energy range: [{stats['min_energy']:.2f}, {stats['max_energy']:.2f}] MeV/u")

    # Simulate one transport step
    print(f"\nSimulating one step...")
    dEdx = np.full(beam.n_particles, 1.12)  # 11.2 keV/µm = 1.12 MeV/cm
    transport_step_cpu(beam.particles, step_length=0.1, energy_loss_rate=dEdx)

    stats_after = beam.get_statistics()
    print(f"After step: {beam}")
    print(f"  Mean energy: {stats_after['mean_energy']:.2f} MeV/u")
    print(f"  Energy loss: {stats['mean_energy'] - stats_after['mean_energy']:.3f} MeV/u")
