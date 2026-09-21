"""Tests for pyanal.structure.

Direct worker calls + one subprocess smoke for cluster_anal.
pyscal3/pywigxjpf tests use importorskip.
"""
import os

import numpy as np
import pytest

from pyanal.structure import (
    BOP_analysis,
    cluster_anal,
    cluster_anal_worker,
    cluster_by_E_dip,
    fcc_lattice,
    get_energetic_neighbors,
    get_energetic_neighbors_csr,
    get_energetic_neighbors_of_i,
    neighbors_csr_to_list,
    steinhardt_ylm,
)


# ---------- cluster_anal_worker direct call ----------

def test_cluster_anal_worker_two_clustered_two_isolated(two_particle_cluster_npy, tmp_path):
    poss_path, dips_path, poss, dips = two_particle_cluster_npy
    cutoff = 2 ** (1 / 6) * 1.3  # ~1.459
    box_l = [100, 100, 100]
    return_file_path = cluster_anal_worker(
        poss_path, dips_path, mask=None, cutoff=cutoff, box_l=box_l, pbc=True,
    )
    data = np.load(return_file_path, allow_pickle=True)
    # 1 bonded pair + 2 singletons -> 3 clusters
    assert int(data['no_clusters']) == 3
    # cluster sizes sorted should be [1, 1, 2]
    assert sorted(data['cluster_sizes'].tolist()) == [1, 1, 2]
    # The bonded pair has parallel z-dipoles -> cosine should be ~ 1.0
    assert np.all(np.isclose(data['cosines'], 1.0, atol=1e-5))
    # num_neighbors: pair members see 1 neighbor; singletons see 0
    nn = data['num_neighbors']
    assert sorted(nn.tolist()) == [0, 0, 1, 1]
    os.remove(return_file_path)


def test_cluster_anal_subprocess_smoke(two_particle_cluster_npy):
    """Exercises run_subprocess_in_module_mp end-to-end for structure.py."""
    _, _, poss, dips = two_particle_cluster_npy
    cutoff = 2 ** (1 / 6) * 1.3
    box_l = [100, 100, 100]
    result = cluster_anal(poss, dips, mask=None, cutoff=cutoff, box_l=box_l, pbc=True)
    (num_neighbors, dist_hist, dist_bin_edges, cosines, cosine_avg_per_neigh,
     no_clusters, cluster_sizes) = result
    assert int(no_clusters) == 3
    assert sorted(cluster_sizes.tolist()) == [1, 1, 2]


# ---------- fcc_lattice ----------

def test_fcc_lattice_lattice_constant_and_bounds():
    r = 0.5
    vol = [5.0, 5.0, 5.0]
    pts = fcc_lattice(radius=r, volume_sides=vol)
    assert pts.ndim == 2 and pts.shape[1] == 3
    # all within volume
    assert (pts >= 0).all()
    assert (pts < vol).all()
    # Nearest-neighbor distance should equal 2r within float tol (or larger if
    # auto-rescaled). Sample a single pair:
    from scipy.spatial import cKDTree
    tree = cKDTree(pts)
    nn_dists, _ = tree.query(pts, k=2)
    min_d = nn_dists[:, 1].min()
    # lattice constant a = 2r/sqrt(2); NN distance along face diagonal = a*sqrt(2)/... = 2r
    assert min_d == pytest.approx(2 * r, rel=1e-3) or min_d > 2 * r


# ---------- BOP_analysis (pyscal-free numba Steinhardt) ----------

def _lattice(basis_frac, a, reps):
    """Build a periodic lattice; returns (positions, box_edge)."""
    cells = np.array([[i, j, k] for i in range(reps) for j in range(reps)
                      for k in range(reps)], dtype=float)
    pts = (cells[:, None, :] + np.asarray(basis_frac)[None, :, :]).reshape(-1, 3) * a
    return pts, reps * a


try:                                   # scipy >= 1.15
    from scipy.special import sph_harm_y as _sph_harm_y
except ImportError:                    # older scipy: sph_harm(m, n, azimuth, polar)
    from scipy.special import sph_harm as _sph_harm
    def _sph_harm_y(n, m, theta_polar, phi_azimuth):
        return _sph_harm(m, n, phi_azimuth, theta_polar)


def _ql_reference(neighbor_dirs, l):
    """Independent Steinhardt q_l from explicit neighbor unit vectors, using scipy's
    spherical harmonics (NOT the pyanal recurrence under test).

    q_l = sqrt(4*pi/(2l+1) * sum_m |<Y_lm>|^2), <.> averaged over the neighbor directions.
    For a single perfect coordination shell every atom is identical, so the
    neighbor-averaged q equals this per-atom q -- the correct target for a perfect lattice.
    """
    d = neighbor_dirs / np.linalg.norm(neighbor_dirs, axis=1, keepdims=True)
    theta = np.arccos(d[:, 2])              # polar (colatitude)
    phi = np.arctan2(d[:, 1], d[:, 0])      # azimuth
    qlm = np.array([np.mean(_sph_harm_y(l, m, theta, phi)) for m in range(-l, l + 1)])
    return float(np.sqrt(4 * np.pi / (2 * l + 1) * np.sum(np.abs(qlm) ** 2)))


# Analytic first-shell neighbor directions.
_FCC_DIRS = np.array([[s1, s2, 0] for s1 in (1, -1) for s2 in (1, -1)]
                     + [[s1, 0, s2] for s1 in (1, -1) for s2 in (1, -1)]
                     + [[0, s1, s2] for s1 in (1, -1) for s2 in (1, -1)], dtype=float)  # 12
_BCC_DIRS = np.array([[s1, s2, s3] for s1 in (1, -1) for s2 in (1, -1)
                      for s3 in (1, -1)], dtype=float)  # 8


def test_BOP_analysis_fcc_matches_independent_reference():
    """Perfect FCC: averaged q_l matches (a) an independent scipy spherical-harmonic
    computation over the 12 analytic NN directions and (b) the canonical published
    constants q4=0.190941, q6=0.574524 (Steinhardt-Nelson-Ronchetti 1983; Lechner-Dellago 2008).

    cutoff is set between the 1st (a/sqrt2) and 2nd (a) shells -> 12 neighbors."""
    a = 1.7
    pos, L = _lattice([[0, 0, 0], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]], a, reps=3)
    _, q, _ = BOP_analysis(pos, np.zeros_like(pos), cutoff_E=np.inf, cutoff_r=0.8 * a,
                           l_range=[4, 6], box_l=[L, L, L], pbc=True)
    q4, q6 = float(np.mean(q[0])), float(np.mean(q[1]))
    assert q4 == pytest.approx(_ql_reference(_FCC_DIRS, 4), abs=1e-4)
    assert q6 == pytest.approx(_ql_reference(_FCC_DIRS, 6), abs=1e-4)
    assert q4 == pytest.approx(0.190941, abs=1e-4)   # canonical literature
    assert q6 == pytest.approx(0.574524, abs=1e-4)


def test_BOP_analysis_bcc_matches_independent_reference():
    """Perfect BCC (8 NN): averaged q_l matches an independent scipy computation over
    the 8 analytic (+-1,+-1,+-1) directions (q4~0.5092, q6~0.6285).

    cutoff between the 1st (a*sqrt3/2) and 2nd (a) shells -> 8 neighbors."""
    a = 1.7
    pos, L = _lattice([[0, 0, 0], [.5, .5, .5]], a, reps=3)
    _, q, _ = BOP_analysis(pos, np.zeros_like(pos), cutoff_E=np.inf, cutoff_r=0.95 * a,
                           l_range=[4, 6], box_l=[L, L, L], pbc=True)
    q4, q6 = float(np.mean(q[0])), float(np.mean(q[1]))
    assert q4 == pytest.approx(_ql_reference(_BCC_DIRS, 4), abs=1e-4)
    assert q6 == pytest.approx(_ql_reference(_BCC_DIRS, 6), abs=1e-4)


def test_BOP_analysis_pbc_false_matches_true_for_interior():
    """For a cluster far from the box faces, pbc on/off give identical results."""
    rng = np.random.default_rng(2)
    box_l = [40.0, 40.0, 40.0]; l_range = list(range(2, 13))
    pos = rng.uniform(15, 25, size=(120, 3))  # interior: no neighbor crosses a face
    dips = rng.normal(size=(120, 3))
    _, q_t, _ = BOP_analysis(pos, dips, cutoff_E=np.inf, cutoff_r=2.5, l_range=l_range, box_l=box_l, pbc=True)
    _, q_f, _ = BOP_analysis(pos, dips, cutoff_E=np.inf, cutoff_r=2.5, l_range=l_range, box_l=box_l, pbc=False)
    assert np.array_equal(np.isnan(q_t), np.isnan(q_f))
    assert np.nanmax(np.abs(q_t.astype(np.float64) - q_f.astype(np.float64))) == 0.0


def test_BOP_analysis_isolated_atom_is_nan():
    """An atom with no neighbors yields NaN; a bonded pair stays finite."""
    pos = np.array([[5.0, 5.0, 5.0], [5.3, 5.0, 5.0], [20.0, 20.0, 20.0]])
    dips = np.zeros((3, 3))
    _, q, _ = BOP_analysis(pos, dips, cutoff_E=np.inf, cutoff_r=0.6,
                           l_range=[4, 6], box_l=[40, 40, 40], pbc=False)
    assert np.all(np.isnan(q[:, 2]))
    assert np.all(np.isfinite(q[:, :2]))


def test_BOP_analysis_energy_criterion_filters():
    """cutoff_E=0 with zero dipoles -> e_ij=0 is not < 0, so no energetic neighbors -> all NaN."""
    a = 1.7
    pos, L = _lattice([[0, 0, 0], [.5, .5, .5]], a, reps=3)
    dips = np.zeros_like(pos)
    _, q, _ = BOP_analysis(pos, dips, cutoff_E=0.0, cutoff_r=0.95 * a,
                           l_range=[4, 6], box_l=[L, L, L], pbc=True)
    assert np.all(np.isnan(q))


def test_BOP_analysis_mask_restricts_atoms():
    """mask selects the analysed subset -> q_vals has one column per masked atom."""
    rng = np.random.default_rng(3)
    L = 8.0; box_l = [L, L, L]
    pos = rng.uniform(0, L, size=(60, 3)); dips = rng.normal(size=(60, 3))
    mask = np.zeros(60, dtype=bool); mask[:25] = True
    _, q, _ = BOP_analysis(pos, dips, mask=mask, cutoff_E=np.inf, cutoff_r=2.0,
                           l_range=[4, 6], box_l=box_l, pbc=True)
    assert q.shape == (2, 25)


# ---------- spherical harmonics vs analytical closed forms ----------

_SQRT = np.sqrt
_Y_POS = {                                   # closed-form Y_l^m, m >= 0 (Condon-Shortley)
    (0, 0): lambda t, p: 0.5 * _SQRT(1 / np.pi) + 0j,
    (1, 0): lambda t, p: 0.5 * _SQRT(3 / np.pi) * np.cos(t),
    (1, 1): lambda t, p: -0.5 * _SQRT(3 / (2 * np.pi)) * np.sin(t) * np.exp(1j * p),
    (2, 0): lambda t, p: 0.25 * _SQRT(5 / np.pi) * (3 * np.cos(t) ** 2 - 1),
    (2, 1): lambda t, p: -0.5 * _SQRT(15 / (2 * np.pi)) * np.sin(t) * np.cos(t) * np.exp(1j * p),
    (2, 2): lambda t, p: 0.25 * _SQRT(15 / (2 * np.pi)) * np.sin(t) ** 2 * np.exp(2j * p),
    (3, 0): lambda t, p: 0.25 * _SQRT(7 / np.pi) * (5 * np.cos(t) ** 3 - 3 * np.cos(t)),
    (3, 1): lambda t, p: -0.125 * _SQRT(21 / np.pi) * np.sin(t) * (5 * np.cos(t) ** 2 - 1) * np.exp(1j * p),
    (3, 2): lambda t, p: 0.25 * _SQRT(105 / (2 * np.pi)) * np.sin(t) ** 2 * np.cos(t) * np.exp(2j * p),
    (3, 3): lambda t, p: -0.125 * _SQRT(35 / np.pi) * np.sin(t) ** 3 * np.exp(3j * p),
    (4, 0): lambda t, p: (3 / 16) * _SQRT(1 / np.pi) * (35 * np.cos(t) ** 4 - 30 * np.cos(t) ** 2 + 3),
    (4, 1): lambda t, p: -(3 / 8) * _SQRT(5 / np.pi) * np.sin(t) * (7 * np.cos(t) ** 3 - 3 * np.cos(t)) * np.exp(1j * p),
    (4, 2): lambda t, p: (3 / 8) * _SQRT(5 / (2 * np.pi)) * np.sin(t) ** 2 * (7 * np.cos(t) ** 2 - 1) * np.exp(2j * p),
    (4, 3): lambda t, p: -(3 / 8) * _SQRT(35 / np.pi) * np.sin(t) ** 3 * np.cos(t) * np.exp(3j * p),
    (4, 4): lambda t, p: (3 / 16) * _SQRT(35 / (2 * np.pi)) * np.sin(t) ** 4 * np.exp(4j * p),
}


def _Y_analytical(l, m, t, p):
    """Closed-form standard complex Y_l^m; negative m via Y_l^-m = (-1)^m conj(Y_l^m)."""
    if m >= 0:
        return _Y_POS[(l, m)](t, p)
    return (-1.0) ** m * np.conj(_Y_POS[(l, -m)](t, p))


def test_steinhardt_ylm_matches_analytical_table():
    """Every Y_l^m from the pyanal recurrence (steinhardt_ylm, the exact code BOP_analysis
    runs) matches the closed-form analytical spherical harmonics (standard complex,
    Condon-Shortley) for l=0..4 over several generic directions. Exercises the real and
    imaginary parts of all m, including m<0 (which requires the negative-m imaginary-sign
    convention in _accumulate_ylm)."""
    lmax = 4
    angles = [(0.7, 0.4), (1.9, 2.3), (2.5, -1.1), (np.pi / 2, np.pi / 3)]
    for t, p in angles:
        Y = steinhardt_ylm(t, p, lmax)
        for l in range(lmax + 1):
            for m in range(-l, l + 1):
                got = Y[l, lmax + m]
                exp = _Y_analytical(l, m, t, p)
                assert got == pytest.approx(exp, abs=1e-10), (l, m, t, p, got, exp)


# ---------- wigner_3j_wigxjpf (optional dep) ----------

def test_wigner_3j_wigxjpf_l2_returns_nonzero():
    pytest.importorskip("pywigxjpf")
    from pyanal.structure import wigner_3j_wigxjpf
    # 3j(2,2,2;0,0,0) is a known non-zero Wigner 3j symbol
    val = wigner_3j_wigxjpf(2, 0, 0, 0)
    assert abs(val) > 0


# ---------- cluster_by_E_dip / energetic neighbor list ----------
# Hand-checked cases with parallel z-dipoles. For two parallel-z dipoles
# separated by r along z: e_ij = (1 - 3)/r^3 = -2/r^3 (attractive).

def _partition_from_labels(labels):
    """Return the partition induced by a labels array as a set of frozensets."""
    parts = {}
    for idx, lab in enumerate(labels):
        parts.setdefault(int(lab), set()).add(idx)
    return frozenset(frozenset(s) for s in parts.values())


def test_cluster_by_E_dip_two_clusters_one_singleton():
    # Pair (0,1) close along z (e=-2 < 0); particle 2 far away.
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [10.0, 10.0, 10.0]])
    dipoles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    box_l = [20.0, 20.0, 20.0]
    labels = cluster_by_E_dip(positions, dipoles, box_l, cutoff_E=0.0, cutoff_r=2.0)
    assert labels.shape == (3,)
    parts = _partition_from_labels(labels)
    assert parts == frozenset([frozenset({0, 1}), frozenset({2})])


def test_cluster_by_E_dip_transitive_merging_chain():
    # Chain 0--1--2; cutoff_r=1.5 lets only adjacent pairs qualify, but the
    # connected-components logic must still merge 0,1,2 into one cluster.
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 2.0]])
    dipoles = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    box_l = [20.0, 20.0, 20.0]
    labels = cluster_by_E_dip(positions, dipoles, box_l, cutoff_E=0.0, cutoff_r=1.5)
    parts = _partition_from_labels(labels)
    assert parts == frozenset([frozenset({0, 1, 2})])


def test_get_energetic_neighbors_of_i_returns_right_sized_arrays():
    # Particle 0 has 2 close z-neighbors and 1 far neighbor.
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, -1.0],
        [10.0, 0.0, 0.0],
    ])
    dipoles = np.array([[0.0, 0.0, 1.0]] * 4)
    box_l = [20.0, 20.0, 20.0]
    idxs, energies = get_energetic_neighbors_of_i(0, positions, dipoles, box_l,
                                             cutoff_E=0.0, cutoff_r=1.5)
    assert idxs.shape == (2,)
    assert energies.shape == (2,)
    assert set(idxs.tolist()) == {1, 2}
    assert (energies < 0.0).all()
    # Each parallel-z pair at r=1: e = -2.
    np.testing.assert_allclose(np.sort(energies), [-2.0, -2.0], rtol=1e-5)


def test_get_energetic_neighbors_returns_list_of_tuples():
    # 4 particles: pairs (0,1) and (1,2) qualify; (0,2) at r=2 fails cutoff_r=1.5; (3) is isolated.
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [10.0, 10.0, 10.0],
    ])
    dipoles = np.array([[0.0, 0.0, 1.0]] * 4)
    box_l = [20.0, 20.0, 20.0]
    result = get_energetic_neighbors(
        positions, dipoles, box_l, cutoff_E=0.0, cutoff_r=1.5)
    assert len(result) == 4
    # Per-particle neighbors (sort each row for cell-list order independence).
    sorted_idxs = [sorted(result[i][0].tolist()) for i in range(4)]
    assert sorted_idxs == [[1], [0, 2], [1], []]
    # Parallel-z dipoles at r=1 -> e = -2 on every emitted pair.
    for i in range(4):
        es = result[i][1]
        np.testing.assert_allclose(es, [-2.0] * len(es), rtol=1e-5)

def test_get_energetic_neighbors_rejects_by_cutoff_E():
    # 4 particles: pairs (0,1) qualify; (1,2) fails cutoff_E=-2.0; (0,2) at r=2 fails cutoff_r=1.5; (3) is isolated.
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [10.0, 10.0, 10.0],
    ])
    dipoles = np.array([[0.0, 0.0, 2.0]] + [[0.0, 0.0, 1.0]] * 3)
    box_l = [20.0, 20.0, 20.0]
    result = get_energetic_neighbors(
        positions, dipoles, box_l, cutoff_E=-2.0, cutoff_r=1.5)
    assert len(result) == 4
    # Per-particle neighbors (sort each row for cell-list order independence).
    sorted_idxs = [sorted(result[i][0].tolist()) for i in range(4)]
    assert sorted_idxs == [[1], [0], [], []]
    # Parallel-z dipoles at r=1 i=0 dipole with dipm of 2 -> e = -4 for (0,1).
    for i in range(4):
        es = result[i][1]
        np.testing.assert_allclose(es, [-4.0] * len(es), rtol=1e-5)


def test_get_energetic_neighbors_csr_returns_flat_arrays():
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [10.0, 10.0, 10.0],
    ])
    dipoles = np.array([[0.0, 0.0, 1.0]] * 4)
    box_l = [20.0, 20.0, 20.0]
    offsets, flat_neighbors, flat_energies = get_energetic_neighbors_csr(
        positions, dipoles, box_l, cutoff_E=0.0, cutoff_r=1.5)
    # Symmetric: 0->[1], 1->[0,2], 2->[1], 3->[]
    np.testing.assert_array_equal(offsets, [0, 1, 3, 4, 4])
    nbrs_per_row = [sorted(flat_neighbors[offsets[i]:offsets[i+1]].tolist())
                    for i in range(4)]
    assert nbrs_per_row == [[1], [0, 2], [1], []]
    assert (flat_energies < 0.0).all()
    np.testing.assert_allclose(flat_energies, [-2.0] * 4, rtol=1e-5)


def test_cluster_and_neighbors_partition_consistent():
    # Same fixture as the CSR test; cluster_ids must induce the same partition
    # that the symmetric neighbor list does.
    positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 0.0, 2.0],
        [10.0, 10.0, 10.0],
    ])
    dipoles = np.array([[0.0, 0.0, 1.0]] * 4)
    box_l = [20.0, 20.0, 20.0]
    labels = cluster_by_E_dip(positions, dipoles, box_l, cutoff_E=0.0, cutoff_r=1.5)
    parts_from_cluster = _partition_from_labels(labels)
    assert parts_from_cluster == frozenset(
        [frozenset({0, 1, 2}), frozenset({3})]
    )


def test_cell_list_matches_brute_on_random():
    # Force the two code paths and compare. cutoff_r=1.5 in box [6,6,6]:
    # 6/1.5 = 4 >= 3 -> cell list is selected by the public API.
    from pyanal.structure import _emit_neighbors_csr_brute, _emit_neighbors_csr_cells, _build_cell_list_3d
    rng = np.random.default_rng(None)
    N = 30
    box_l = np.array([6.0, 6.0, 6.0])
    positions = rng.uniform(0.0, 6.0, size=(N, 3))
    dipoles = rng.normal(size=(N, 3))
    dipoles /= np.linalg.norm(dipoles, axis=1, keepdims=True)
    cutoff_r = 1.5

    off_b, fn_b, fe_b = _emit_neighbors_csr_brute(
        positions, dipoles, box_l, 0.0, cutoff_r, True, True)

    ncx = ncy = ncz = 4
    inv_cs = 4.0 / 6.0
    x = np.ascontiguousarray(positions[:, 0])
    y = np.ascontiguousarray(positions[:, 1])
    z = np.ascontiguousarray(positions[:, 2])
    cell_starts, cell_particles, p_cell_indice = _build_cell_list_3d(
        x, y, z, ncx, ncy, ncz, inv_cs, inv_cs, inv_cs, True)
    off_c, fn_c, fe_c = _emit_neighbors_csr_cells(
        positions, dipoles, box_l,
        cell_starts, cell_particles, p_cell_indice,
        ncx, ncy, ncz,
        0.0, cutoff_r, True, True)

    np.testing.assert_array_equal(off_b, off_c)
    for i in range(N):
        idx_b = fn_b[off_b[i]:off_b[i+1]]
        e_b = fe_b[off_b[i]:off_b[i+1]]
        ob = np.argsort(idx_b)
        idx_c = fn_c[off_c[i]:off_c[i+1]]
        e_c = fe_c[off_c[i]:off_c[i+1]]
        oc = np.argsort(idx_c)
        np.testing.assert_array_equal(idx_b[ob], idx_c[oc])
        np.testing.assert_allclose(e_b[ob], e_c[oc], rtol=1e-5)


def test_neighbors_csr_to_list_roundtrip():
    offsets = np.array([0, 2, 3, 3, 5], dtype=np.int32)
    flat_neighbors = np.array([1, 2, 0, 0, 1], dtype=np.int32)
    flat_energies = np.array([-1.0, -2.0, -3.0, -4.0, -5.0], dtype=np.float64)

    only_idx = neighbors_csr_to_list(offsets, flat_neighbors)
    assert len(only_idx) == 4
    np.testing.assert_array_equal(only_idx[0], [1, 2])
    np.testing.assert_array_equal(only_idx[1], [0])
    np.testing.assert_array_equal(only_idx[2], [])
    np.testing.assert_array_equal(only_idx[3], [0, 1])

    with_e = neighbors_csr_to_list(offsets, flat_neighbors, flat_energies)
    assert len(with_e) == 4
    np.testing.assert_array_equal(with_e[0][0], [1, 2])
    np.testing.assert_allclose(with_e[0][1], [-1.0, -2.0])
    np.testing.assert_array_equal(with_e[3][0], [0, 1])
    np.testing.assert_allclose(with_e[3][1], [-4.0, -5.0])
