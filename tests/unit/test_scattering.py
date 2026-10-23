"""
Statistical tests for multiple Coulomb scattering sampling.

Highland's theta_rms is the projected (plane) width θ0, so one scattering
step must give two independent projected angles of width θ0, a uniform
azimuth, and no preferred transverse direction for any incoming direction.

Run with:  python -m pytest tests/unit -q
"""

from pathlib import Path
import sys

import numba
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from coulomb_mc.physics.scattering import (scatter_direction, scatter_particle,
                                           sample_scattering_angle, rotate_direction)
from coulomb_mc.transport.engine import TransportEngine


THETA0 = 0.01
N = 200_000


@numba.njit
def _seed(seed):
    np.random.seed(seed)


@numba.njit
def _many_scatters(d, theta0, n):
    out = np.empty((n, 3))
    for k in range(n):
        out[k, 0], out[k, 1], out[k, 2] = scatter_direction(d[0], d[1], d[2], theta0)
    return out


@numba.njit
def _many_angles(theta0, n):
    out = np.empty((n, 2))
    for k in range(n):
        out[k, 0], out[k, 1] = sample_scattering_angle(theta0)
    return out


def _frame(d):
    """Any orthonormal pair perpendicular to d, built independently of the code."""
    a = np.array([1.0, 0.0, 0.0]) if abs(d[0]) < 0.5 else np.array([0.0, 1.0, 0.0])
    p = np.cross(d, a)
    p /= np.linalg.norm(p)
    q = np.cross(d, p)
    return p, q


DIRECTIONS = [
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
    (0.6, 0.0, 0.8),
    (1.0, 0.0, 0.0),
    (0.3, -0.5, 0.812403840463596),
]


@pytest.mark.parametrize('d', DIRECTIONS)
def test_projected_angles_are_isotropic_with_width_theta0(d):
    d = np.array(d) / np.linalg.norm(d)
    _seed(7)
    out = _many_scatters(d, THETA0, N)

    assert np.allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-12)

    p, q = _frame(d)
    ap = np.arctan2(out @ p, out @ d)
    aq = np.arctan2(out @ q, out @ d)

    se_mean = THETA0 / np.sqrt(N)
    assert abs(ap.mean()) < 5 * se_mean
    assert abs(aq.mean()) < 5 * se_mean
    # standard error of a sample std is about sigma / sqrt(2N)
    se_std = THETA0 / np.sqrt(2 * N)
    assert abs(ap.std() - THETA0) < 5 * se_std
    assert abs(aq.std() - THETA0) < 5 * se_std
    assert abs(np.corrcoef(ap, aq)[0, 1]) < 5 / np.sqrt(N)


def test_polar_angle_is_rayleigh_and_azimuth_uniform():
    _seed(11)
    ang = _many_angles(THETA0, N)
    theta, phi = ang[:, 0], ang[:, 1]

    assert np.all(theta >= 0.0)
    assert np.all((phi >= 0.0) & (phi < 2 * np.pi))
    # Rayleigh(θ0): E[θ²] = 2 θ0², E[θ] = θ0 sqrt(π/2)
    assert abs(np.mean(theta**2) / (2 * THETA0**2) - 1.0) < 0.01
    assert abs(np.mean(theta) / (THETA0 * np.sqrt(np.pi / 2)) - 1.0) < 0.01
    assert abs(np.mean(np.cos(phi))) < 5 / np.sqrt(N)
    assert abs(np.mean(np.sin(phi))) < 5 / np.sqrt(N)


@pytest.mark.parametrize('d', DIRECTIONS)
def test_rotate_direction_tilts_by_theta(d):
    d = np.array(d) / np.linalg.norm(d)
    for theta in (1e-3, 0.1, 0.7):
        for phi in (0.0, 1.0, 2.5, 4.0):
            r = rotate_direction(d, theta, phi)
            assert abs(np.linalg.norm(r) - 1.0) < 1e-12
            assert abs(np.arccos(np.clip(r @ d, -1, 1)) - theta) < 1e-9
            # Azimuth: the transverse part at phi + pi/2 is d x (transverse part at phi)
            r2 = rotate_direction(d, theta, phi + np.pi / 2)
            t1 = r - (r @ d) * d
            t2 = r2 - (r2 @ d) * d
            assert np.allclose(np.cross(d, t1), t2, atol=1e-9)


def test_scatter_particle_matches_helper_statistics():
    _seed(3)
    d = np.array([0.0, 0.0, 1.0])
    out = np.array([scatter_particle(d, THETA0) for _ in range(20_000)])
    assert abs(np.std(out[:, 0]) / THETA0 - 1.0) < 0.05
    assert abs(np.std(out[:, 1]) / THETA0 - 1.0) < 0.05


def test_pencil_beam_has_no_coherent_lateral_drift():
    """Regression for the unused-azimuth defect, which pushed every track to -x."""
    _seed(5)
    np.random.seed(5)
    engine = TransportEngine(material='water')
    beam = engine.create_beam('C-12', 400.0, n_particles=400)
    engine.transport(beam, max_depth=50.0, verbose=False)
    pos = beam.particles['position']
    sx, sy = pos[:, 0].std(), pos[:, 1].std()

    assert sx > 0 and sy > 0
    assert abs(sx / sy - 1.0) < 0.25
    assert abs(pos[:, 0].mean()) < 5 * sx / np.sqrt(len(pos))
    assert abs(pos[:, 1].mean()) < 5 * sy / np.sqrt(len(pos))
