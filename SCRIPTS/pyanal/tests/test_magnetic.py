"""Tests for pyanal.magnetic."""
import numpy as np
import pytest

from pyanal.magnetic import M_z, M_z_dipm


def test_M_z_arithmetic_mean_of_z_component():
    # Dipoles (0,0,1), (0,0,-1), (0,0,2) -> sum z = 2, /N=3 -> 2/3
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 2.0]])
    assert M_z(dips) == pytest.approx(2.0 / 3.0, rel=1e-6)


def test_M_z_dipm_uses_unit_direction():
    # Same dipoles: unit-z components are 1, -1, 1 -> mean = 1/3
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 2.0]])
    assert M_z_dipm(dips) == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_M_z_dipm_skips_zero_dipoles_but_divides_by_N():
    # Two unit-z dipoles + one zero dipole.
    # Numerator sums (1) + (-1) -> 0; denominator is still N=3 -> result 0
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
    assert M_z_dipm(dips) == pytest.approx(0.0, abs=1e-7)


def test_M_z_dipm_skipped_zero_still_in_denominator():
    # Two same-sign unit dipoles + one zero -> numerator = 2, /N=3 -> 2/3
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]])
    assert M_z_dipm(dips) == pytest.approx(2.0 / 3.0, rel=1e-6)
