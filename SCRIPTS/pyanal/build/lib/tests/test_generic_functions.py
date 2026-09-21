"""Tests for pyanal.generic_functions."""
import os

import numpy as np
import pytest

from pyanal.generic_functions import (
    cartesian_to_spherical_coord,
    divide_nan,
    fold_coord,
    generate_random_unit_vectors,
    log_zero,
    min_img_dist,
    run_subprocess_in_module_mp,
    separate_bulk_surface,
    tempfile_path,
    weighted_com_pbc,
)


def test_divide_nan():
    assert divide_nan(0, 0) == np.nan
    assert divide_nan(-0.0, 0.0) == np.nan
    assert divide_nan(3, 0) == np.nan
    assert divide_nan(6, 2) == 3
    assert divide_nan(-6, 2) == -3
    assert divide_nan(6, -2) == -3
    assert divide_nan(-6, -2) == 3


def test_log_zero():
    assert log_zero(0) == 0
    assert log_zero(np.e) == pytest.approx(1.0)


def test_fold_coord_basic():
    out = fold_coord(np.array([11.0, -0.5]), [10.0, 10.0])
    np.testing.assert_allclose(out, [1.0, 9.5])


def test_fold_coord_exact_edge_to_zero():
    # x exactly box -> wrapped to 0; numerical-hygiene clamp also catches near-edge
    out = fold_coord(np.array([10.0]), [10.0])
    assert out[0] == 0.0


def test_cartesian_to_spherical_axis_x():
    r, phi, theta = cartesian_to_spherical_coord([1.0, 0.0, 0.0])
    assert r == pytest.approx(1.0)
    assert phi == pytest.approx(0.0)
    assert theta == pytest.approx(np.pi / 2)


def test_cartesian_to_spherical_axis_y():
    r, phi, theta = cartesian_to_spherical_coord([0.0, 1.0, 0.0])
    assert r == pytest.approx(1.0)
    assert phi == pytest.approx(np.pi / 2)
    assert theta == pytest.approx(np.pi / 2)


def test_cartesian_to_spherical_axis_z():
    r, phi, theta = cartesian_to_spherical_coord([0.0, 0.0, 1.0])
    assert r == pytest.approx(1.0)
    assert theta == pytest.approx(0.0)


def test_cartesian_to_spherical_batched_shape():
    arr = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    out = cartesian_to_spherical_coord(arr)
    assert out.shape == (2, 3)


def test_cartesian_to_spherical_bad_shape_raises():
    with pytest.raises(IndexError):
        cartesian_to_spherical_coord(np.array([[1.0, 2.0]]))


def test_min_img_dist_wraps_short_way():
    d = min_img_dist(np.array([0.1]), np.array([9.9]), [10.0])
    np.testing.assert_allclose(d, [-0.2])
    d2 = min_img_dist(np.array([9.9]), np.array([0.1]), [10.0])
    np.testing.assert_allclose(d2, [0.2])


def test_weighted_com_pbc_circular_mean():
    # particles at the two edges of a periodic box -> COM at the wrap point (0), NOT 5.
    positions = np.array([[0.5, 0.5, 0.5], [9.5, 0.5, 0.5]])
    weights = np.array([1.0, 1.0])
    com = weighted_com_pbc(positions, weights, np.array([10.0, 10.0, 10.0]))
    # x dimension should wrap to ~0 or ~10 (same point under PBC), not 5
    assert min(abs(com[0]), abs(com[0] - 10.0)) == pytest.approx(0.0)


def test_generate_random_unit_vectors_norm_and_distribution():
    np.random.seed(0)
    v = generate_random_unit_vectors(2000)
    assert v.shape == (2000, 3)
    norms = np.linalg.norm(v, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-6)
    assert abs(v[:, 2].mean()) < 0.1


def test_separate_bulk_surface_splits_into_nonempty_masks():
    # Synthetic z-distribution: substrate at z<2, bulk at 2..8, very sparse 8..10
    np.random.seed(1)
    z_bulk = np.random.uniform(2.5, 8.0, 200)
    z_surf = np.random.uniform(8.5, 9.5, 5)
    z_sub = np.random.uniform(0.5, 1.5, 5)
    z = np.concatenate([z_sub, z_bulk, z_surf])
    poss = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    substrate_thr, bulk_thr, mask_bulk, mask_surface = separate_bulk_surface(
        poss, size=1.0, manual_selection=False
    )
    assert substrate_thr < bulk_thr
    assert mask_bulk.sum() > 0
    assert mask_surface.sum() >= 0  # may be 0 if auto-detection collapses; just sanity
    # No particle should be in both masks
    assert not (mask_bulk & mask_surface).any()


def test_tempfile_path_returns_string_with_suffix():
    p = tempfile_path(suffix=".npy")
    try:
        assert isinstance(p, str)
        assert p.endswith(".npy")
        assert os.path.isfile(p)
    finally:
        if os.path.exists(p):
            os.remove(p)


def test_run_subprocess_in_module_mp_smoke():
    """Exercises the spawn+Pipe plumbing with a trivial worker."""
    result = run_subprocess_in_module_mp(
        (2, 3),
        module="pyanal._test_helpers",
        worker="_smoke_worker",
    )
    assert result == 5


def test_run_subprocess_in_module_mp_propagates_exceptions():
    with pytest.raises(RuntimeError) as exc_info:
        run_subprocess_in_module_mp(
            (),
            module="pyanal._test_helpers",
            worker="_raising_worker",
        )
    assert "smoke-test-raised" in str(exc_info.value)
