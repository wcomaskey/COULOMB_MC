"""
COULOMB_MC: 3D Radiation Monte Carlo Simulation

A high-performance Monte Carlo code for heavy-ion radiation transport
with machine learning integration.

Modules:
    core: Particle state management, geometry
    physics: Stopping power, scattering, fragmentation
    transport: CPU and GPU transport engines
    scoring: Dose, LET, and fluence tallies
    ml: Machine learning surrogate models
    io: Input/output handling
"""

__version__ = "0.1.0"
__author__ = "William Comaskey"

from coulomb_mc.core.particle import ParticleArray, Particle
from coulomb_mc.physics.stopping_power import StoppingPower
from coulomb_mc.physics.scattering import MultipleScattering
from coulomb_mc.transport.engine import TransportEngine

__all__ = [
    "ParticleArray",
    "Particle",
    "StoppingPower",
    "MultipleScattering",
    "TransportEngine",
]
