"""Tests for pyanal.energy.

Hand-computed cases on a large box so PBC images don't contribute.
FASTMATH is on -- assert via allclose with reasonable tolerance.
"""
import numpy as np
import pytest

from pyanal.energy import (
    Zeeman_dipm_energy,
    Zeeman_energy,
    bond_energy_harmonic,
    calc_energies,
    convert_bonds_list_for_numba,
    dipolar_energy_direct_sum,
)


def test_Zeeman_energy_two_parallel_z_dipoles():
    # mu_i = (0,0,1), H=(0,0,2)  ->  -mu.H = -2 each, sum = -4
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    H = np.array([0.0, 0.0, 2.0])
    assert Zeeman_energy(dips, dips.shape[0], H) == pytest.approx(-4.0, rel=1e-6)


def test_Zeeman_dipm_energy_uses_unit_vec_skips_zero():
    # |mu_i|=3 then 4 then 0; unit z-components 1, 1, skipped
    # -mu_hat . H = -5 + -5 + 0  = -10
    dips = np.array([[0.0, 0.0, 3.0], [0.0, 0.0, 4.0], [0.0, 0.0, 0.0]])
    H = np.array([0.0, 0.0, 5.0])
    assert Zeeman_dipm_energy(dips, dips.shape[0], H) == pytest.approx(-10.0, rel=1e-6)


def test_dipolar_energy_direct_sum_parallel_along_z():
    # mu_i = mu_j = (0,0,1), r = (0,0,1)
    # E = (mu.mu)/r^3 - 3 (mu.r_hat)(mu.r_hat)/r^3
    #   = 1 - 3 * 1 * 1 = -2
    box = np.array([100.0, 100.0, 100.0])
    poss = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    E = dipolar_energy_direct_sum(box, poss, dips, 2)
    assert E == pytest.approx(-2.0, rel=1e-5)

def test_dipolar_energy_direct_sum_parallel_along_z_2():
    # mu_i = mu_j = (0,0,1), r = (0,0,1)
    # E = (mu.mu)/r^3 - 3 (mu.r_hat)(mu.r_hat)/r^3
    #   = 1 - 3 * 0 * 0 = 1
    box = np.array([100.0, 100.0, 100.0])
    poss = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    E = dipolar_energy_direct_sum(box, poss, dips, 2)
    assert E == pytest.approx(1.0, rel=1e-5)


def test_dipolar_energy_direct_sum_perpendicular():
    # mu_i = (0,0,1), mu_j = (1,0,0), r along z -> (mu_i.mu_j)=0, (mu_i.r)=1, (mu_j.r)=0 -> E = 0
    box = np.array([100.0, 100.0, 100.0])
    poss = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    dips = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])
    E = dipolar_energy_direct_sum(box, poss, dips, 2)
    assert E == pytest.approx(0.0, abs=1e-5)


def test_bond_energy_harmonic_via_calc_energies(tiny_two_body):
    # r=1.5, r0=1.0, k=10 -> 0.5 * 10 * 0.5**2 = 1.25 (sum), /N=2 -> 0.625 per-particle
    box_l, ids, poss, dips, bonds = tiny_two_body
    _, _, E_bond, _ = calc_energies(box_l, ids, poss, dips, bonds, H_ext=np.zeros(3))
    assert E_bond == pytest.approx(0.625, rel=1e-5)


def test_convert_bonds_list_for_numba_layout():
    ids = np.array([5, 7, 9], dtype=np.int32)
    bonds = [
        [(0, 1.0, 2.0, 0, 7)],
        [],
        [(0, 3.0, 4.0, 0, 5)],
    ]
    bond_ends, bond_idxs, bond_ks, bond_r0s = convert_bonds_list_for_numba(ids, bonds, 3)
    np.testing.assert_array_equal(bond_ends, [1, 1, 2])
    np.testing.assert_array_equal(bond_idxs, [1, 0])
    np.testing.assert_allclose(bond_ks, [1.0, 3.0])
    np.testing.assert_allclose(bond_r0s, [2.0, 4.0])


def test_calc_energies_tuple_order_and_normalization(tiny_two_body):
    box_l, ids, poss, dips, bonds = tiny_two_body
    H = np.array([0.0, 0.0, 2.0])
    E_dip, E_Zee, E_bond, E_Zee_dipm = calc_energies(box_l, ids, poss, dips, bonds, H)
    # bond -> 0.625 (see above)
    assert E_bond == pytest.approx(0.625, rel=1e-5)
    # Zeeman: sum -4, /N=2 -> -2
    assert E_Zee == pytest.approx(-2.0, rel=1e-5)
    # dipoles parallel z, r along x at dist 1.5: mu.mu = 1, mu.r=0,mu.r=0
    # E_pair = 1/1.5^3 = 0.2963, /N=2 -> 0.14815
    assert E_dip == pytest.approx(1.0 / 1.5**3 / 2, rel=1e-4)
    # |mu|=1 so E_Zee_dipm matches E_Zee per-particle
    assert E_Zee_dipm == pytest.approx(-2.0, rel=1e-5)


def test_calc_energies_p3m_raises(tiny_two_body):
    box_l, ids, poss, dips, bonds = tiny_two_body
    with pytest.raises(NotImplementedError):
        calc_energies(box_l, ids, poss, dips, bonds, H_ext=np.zeros(3), p3m=True)


def test_periodicity_flag_inflates_box_in_open_axis():
    # Place two dipoles at z=0.5 and z=9.5 in a 10-cube box.
    #   periodicity z=True  -> min-image dz = 1.0
    #   periodicity z=False -> internal box z = 1e6+10, so dz = 9.0 (no wrap)
    # The dipolar energy differs by ~9^3 between the two cases.
    box_l = np.array([10.0, 10.0, 10.0])
    ids = np.array([0, 1], dtype=np.int32)
    poss = np.array([[5.0, 5.0, 0.5], [5.0, 5.0, 9.5]])
    dips = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    bonds = [[], []]
    E_dip_periodic_z, _, _, _ = calc_energies(
        box_l, ids, poss, dips, bonds, H_ext=np.zeros(3),
        periodicity=[True, True, True])
    E_dip_open_z, _, _, _ = calc_energies(
        box_l, ids, poss, dips, bonds, H_ext=np.zeros(3),
        periodicity=[True, True, False])
    # min-image dz=1: parallel-z dipoles at r=1 along z -> E_pair = 1 - 3 = -2
    assert E_dip_periodic_z == pytest.approx(-2.0 / 2, rel=1e-4)
    # dz=9: parallel-z dipoles -> E_pair = -2/9^3 ~= -0.00274
    assert E_dip_open_z == pytest.approx(-2.0 / 9.0**3 / 2, rel=1e-3)
