"""Shared fixtures for the pyanal test suite (excluding ARay)."""
import numpy as np
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with -m 'not slow')")


@pytest.fixture
def tiny_two_body():
    """Two particles, one harmonic bond, parallel z-dipoles.

    Designed so the analytic energies are computable on paper:
      r_ij = 1.5,  k=10, r0=1.0   -> bond E = 0.5*10*0.25 = 1.25
      mu=(0,0,1),  H=(0,0,2)      -> Zeeman E = -4 (sum), -2 per particle
    """
    box_l = np.array([100.0, 100.0, 100.0])
    ids = np.array([0, 1], dtype=np.int32)
    poss = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float64)
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    bonds = [
        [(0, 10.0, 1.0, 0, 1)],
        [],
    ]
    return box_l, ids, poss, dips, bonds


@pytest.fixture
def dipole_pair_z():
    """Two parallel-z dipoles 1 unit apart along z. Analytic energy = -2."""
    box_l = np.array([100.0, 100.0, 100.0])
    poss = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return box_l, poss, dips

@pytest.fixture
def dipole_pair_x():
    """Two parallel-z dipoles 1 unit apart along x. Analytic energy = 1."""
    box_l = np.array([100.0, 100.0, 100.0])
    poss = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return box_l, poss, dips


@pytest.fixture
def two_particle_cluster_npy(tmp_path):
    """Saves a poss/dips fixture for cluster_anal_worker tests.

    4 particles: pair (0,1) close enough to bond at cutoff 2^(1/6)*1.3 ~= 1.459,
    particles 2 and 3 isolated. Dipoles parallel.
    """
    poss = np.array(
        [
            [0.0, 0.0, 5.0],
            [1.0, 0.0, 5.0],
            [10.0, 10.0, 5.0],
            [20.0, 20.0, 5.0],
        ],
        dtype=np.float64,
    )
    dips = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    poss_path = str(tmp_path / "poss.npy")
    dips_path = str(tmp_path / "dips.npy")
    np.save(poss_path, poss)
    np.save(dips_path, dips)
    return poss_path, dips_path, poss, dips
