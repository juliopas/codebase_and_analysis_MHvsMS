"""Tests for pyanal.bond_to_parts.

Direct kernel calls only. The full subprocess wrapper
(calculate_and_write_bond_particles_h5) needs an ESPResSo h5 fixture and is skipped.
"""
import numpy as np
import pytest

from pyanal.bond_to_parts import (
    calculate_bond_particles_flat,
    calculate_connectivity,
    calculate_total_number_of_bond_particles,
)


def test_calculate_total_number_of_bond_particles_simple():
    # 2 particles at (0,0,0) and (2.5,0,0). 1 bond i=0 -> j=1.
    # bond_part_distance = 1.0 -> int(2.5/1) = 2
    poss = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float64)
    idxs_2 = np.array([1], dtype=np.int32)
    flat_offsets = np.array([0, 1, 1], dtype=np.int32)
    box_l = np.array([100.0, 100.0, 100.0])
    out = calculate_total_number_of_bond_particles(
        poss, 1.0, 2, idxs_2, flat_offsets, box_l
    )
    np.testing.assert_array_equal(out, [2])


def test_calculate_bond_particles_flat_positions():
    # Particles at (0,0,0) and (2.5,0,0); bond_part_distance=1; Nbp=2
    poss = np.array([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]], dtype=np.float64)
    idxs_2 = np.array([1], dtype=np.int32)
    flat_offsets = np.array([0, 1, 1], dtype=np.int32)
    bonds_part_flat_offset = np.array([0, 2], dtype=np.int32)
    box_l = np.array([100.0, 100.0, 100.0])
    bond_parts = calculate_bond_particles_flat(
        poss, 1.0, 2, 2, idxs_2, flat_offsets, bonds_part_flat_offset, box_l
    )
    assert bond_parts.shape == (2, 3)
    # First bond particle at xi + 0.5*r_hat*d = (0.5, 0, 0); second at +(1,0,0) -> (1.5, 0, 0)
    np.testing.assert_allclose(bond_parts[0], [0.5, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(bond_parts[1], [1.5, 0.0, 0.0], atol=1e-5)


def test_calculate_connectivity_layout():
    # 2 particles, 1 bond (i=0 -> j=1), 2 bond particles.
    N = 2
    Nb = 1
    Nbp = 2
    ids = np.array([10, 20], dtype=np.int32)
    idxs_2 = np.array([1], dtype=np.int32)
    bond_part_ids = np.array([100, 101], dtype=np.int32)
    flat_offsets = np.array([0, 1, 1], dtype=np.int32)
    bonds_part_flat_offset = np.array([0, 2], dtype=np.int32)
    bp_to_bond, b_to_part = calculate_connectivity(
        N, Nb, Nbp, ids, idxs_2, bond_part_ids, flat_offsets, bonds_part_flat_offset,
    )
    # bond_to_particle: anchor row (j=0): [0, id1=10]; partner row: [0, id_of_idx_2=20]
    np.testing.assert_array_equal(b_to_part[0], [0, 10])
    np.testing.assert_array_equal(b_to_part[1], [0, 20])
    # bond_part_to_bond: [bond_part_id, j (bond index)]
    np.testing.assert_array_equal(bp_to_bond[0], [100, 0])
    np.testing.assert_array_equal(bp_to_bond[1], [101, 0])


# H5-dependent paths are skipped because they need a real ESPResSo simulation file.
# No subprocess smoke test for bond_to_parts since its only wrapper needs h5 input.

@pytest.mark.skip(reason="requires pressomancy h5 fixture")
def test_calculate_elastomer_bond_particles_h5():
    pass


@pytest.mark.skip(reason="requires pressomancy h5 fixture")
def test_calculate_and_write_bond_particles_h5_worker():
    pass


@pytest.mark.skip(reason="requires pressomancy h5 fixture")
def test_get_n_bond_to_parts_worker():
    pass
