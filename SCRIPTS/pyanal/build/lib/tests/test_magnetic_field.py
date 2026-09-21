"""Tests for dipolar_field_at_points, dipolar_field_xy_slice, H_lorentz, H_internal."""
import numpy as np
import pytest

from pyanal.magnetic import (
    dipolar_field_at_points, dipolar_field_xy_slice,
    H_lorentz_bare, H_internal_bare, H_internal,
    surface_point_cloud,
    _image_offsets,
)

# ─── helpers ───────────────────────────────────────────────────────────────

_LARGE_BOX = np.array([1e6, 1e6, 1e6])
_NO_PBC    = [False, False, False]
_SLAB_PBC  = [True, True, False]

def _analytic_dipole_field(r_vec, mu):
    """B_dip at r_vec from a dipole mu at origin (Gaussian-CGS, no μ₀/4π)."""
    r  = np.linalg.norm(r_vec)
    r3 = r**3
    r5 = r**5
    r_hat = r_vec / r
    return (3 * np.dot(mu, r_hat) * r_hat - mu) / r3


def _slab_lattice(nx=5, ny=5, nz=3, spacing=1.0):
    """Helper: build a uniform z-magnetised lattice."""
    grid = np.array([[x * spacing, y * spacing, z * spacing]
                     for x in range(nx) for y in range(ny) for z in range(nz)])
    pos  = grid.astype(np.float64)
    N    = pos.shape[0]
    dips = np.zeros((N, 3)); dips[:, 2] = 1.0
    box_l = np.array([nx * spacing, ny * spacing, 1e6])
    return pos, dips, box_l


# ─── 1. Single-dipole analytic checks ──────────────────────────────────────

@pytest.mark.parametrize("r_vec, mu", [
    (np.array([0.0, 0.0, 2.0]),  np.array([0.0, 0.0, 1.0])),  # axial
    (np.array([2.0, 0.0, 0.0]),  np.array([0.0, 0.0, 1.0])),  # perpendicular
    (np.array([1.0, 0.0, 1.0]),  np.array([0.0, 0.0, 1.0])),  # oblique (45°)
    (np.array([1.0, 1.0, 1.0]),  np.array([1.0, 0.0, 0.0])),  # off-axis dipole
])
def test_single_dipole_analytic(r_vec, mu):
    """Field at a single point from a single dipole matches closed-form."""
    pos    = np.array([[0.0, 0.0, 0.0]])
    dips   = mu.reshape(1, 3)
    pt     = r_vec.reshape(1, 3)

    B = dipolar_field_at_points(pos, dips, pt, _LARGE_BOX, _NO_PBC)
    B_expected = _analytic_dipole_field(r_vec, mu)

    np.testing.assert_allclose(B[0], B_expected, atol=1e-10)


# ─── 2. PBC sanity — translate source+point by one box length ──────────────

def test_pbc_translation_invariance():
    """Field is unchanged when source and evaluation point are both
    translated by an integer multiple of the box length (PBC in x)."""
    box_l   = np.array([10.0, 50.0, 50.0])
    pbc     = [True, False, False]

    pos_orig = np.array([[0.3, 0.0, 0.0]])
    dips     = np.array([[0.0, 0.0, 1.0]])
    pt_orig  = np.array([[2.5, 0.0, 0.0]])

    B_orig = dipolar_field_at_points(pos_orig, dips, pt_orig, box_l, pbc)

    pos_shifted = pos_orig + np.array([[box_l[0], 0, 0]])
    pt_shifted  = pt_orig  + np.array([[box_l[0], 0, 0]])
    B_shifted   = dipolar_field_at_points(pos_shifted, dips, pt_shifted, box_l, pbc)

    np.testing.assert_allclose(B_orig, B_shifted, atol=1e-12)


# ─── 3. Self-exclusion + H_ext wiring (H_lorentz) ──────────────────────────

def test_H_lorentz_single_particle_returns_H_ext():
    """A single particle's field at itself is skipped (r=0 → skip),
    so H_lorentz == H_ext."""
    pos    = np.array([[0.0, 0.0, 0.0]])
    dips   = np.array([[0.0, 0.0, 1.0]])
    H_ext  = np.array([0.3, 0.0, 2.5])

    H = H_lorentz_bare(pos, dips, _LARGE_BOX, _NO_PBC, H_ext)
    np.testing.assert_allclose(H, H_ext, atol=1e-12)


def test_H_lorentz_mask_averages_subset():
    """With mask=[True,False], H_lorentz averages only over particle 0.
    Result = (field at p0 from p1) + H_ext."""
    box_l = np.array([1e6, 1e6, 1e6])
    pos   = np.array([[0.0, 0.0, 0.0],
                      [0.0, 0.0, 2.0]])
    dips  = np.array([[0.0, 0.0, 1.0],
                      [0.0, 0.0, 1.0]])
    H_ext = np.array([0.0, 0.0, 0.5])
    mask  = np.array([True, False])

    H = H_lorentz_bare(pos, dips, box_l, _NO_PBC, H_ext, mask=mask)

    expected_B = _analytic_dipole_field(np.array([0.0, 0.0, -2.0]),
                                        np.array([0.0, 0.0, 1.0]))
    np.testing.assert_allclose(H, expected_B + H_ext, atol=1e-10)


def test_H_lorentz_empty_mask_returns_H_ext():
    """Empty mask (no particles selected) → returns H_ext unchanged."""
    pos   = np.array([[0.0, 0.0, 0.0]])
    dips  = np.array([[0.0, 0.0, 1.0]])
    H_ext = np.array([1.0, 2.0, 3.0])
    mask  = np.array([False])

    H = H_lorentz_bare(pos, dips, _LARGE_BOX, _NO_PBC, H_ext, mask=mask)
    np.testing.assert_allclose(H, H_ext, atol=1e-15)


# ─── 4. Lateral neighbour analytic check ───────────────────────────────────

def test_lateral_neighbor_demagnetizes_exact(dipole_pair_x):
    """A lateral neighbour (perpendicular to μ) demagnetises in z with exact value.

    For μ=(0,0,1) at (1,0,0) evaluated at the origin:
        r = (1,0,0),  B_z = (3(μ·r̂)r̂ - μ)_z / r³ = (0 - 1)/1 = -1.
    """
    box_l, poss, dips = dipole_pair_x
    H_ext = np.array([0.0, 0.0, 0.0])
    mask  = np.array([True, False])

    H = H_lorentz_bare(poss, dips, box_l, _NO_PBC, H_ext, mask=mask)

    assert H[2] < 0, "Lateral neighbour must demagnetise (B_z < 0)"
    np.testing.assert_allclose(H[2], -1.0, atol=1e-10)
    np.testing.assert_allclose(H[0], 0.0,  atol=1e-10)
    np.testing.assert_allclose(H[1], 0.0,  atol=1e-10)


def test_slab_demag_sign_symmetry():
    """5×5×3 slab: demag opposes M and lateral components are small."""
    pos, dips, box_l = _slab_lattice()
    pbc   = [True, True, False]
    H_ext = np.array([0.0, 0.0, 0.0])
    mask  = (pos[:, 2] == 1.0)

    H = H_lorentz_bare(pos, dips, box_l, pbc, H_ext, mask=mask)

    assert H[2] < 0
    assert abs(H[0]) < abs(H[2]) * 0.05
    assert abs(H[1]) < abs(H[2]) * 0.05


# ─── 5. Rename smoke-test ───────────────────────────────────────────────────

def test_exports():
    """Bare diagnostics and the surface-based H_internal are exported from pyanal."""
    import pyanal
    assert callable(pyanal.H_lorentz_bare)
    assert callable(pyanal.H_internal_bare)
    assert callable(pyanal.H_internal)


# ─── 6. _image_offsets shape / count checks ────────────────────────────────

def test_image_offsets_n0_always_one():
    """n_image_shells=0 → single home-cell offset."""
    for pbc in ([True, True, False], [True, True, True], [False, False, False]):
        offs = _image_offsets(np.array([5., 5., 5.]), pbc, 0, 'square')
        assert offs.shape == (1, 3), f"Expected 1 offset for n=0, got {offs.shape[0]}"
        np.testing.assert_array_equal(offs[0], [0., 0., 0.])


def test_image_offsets_square_slab():
    """Square truncation for slab PBC (T,T,F): (2N+1)² offsets, all z=0."""
    box_l = np.array([2., 3., 1e6])
    pbc   = [True, True, False]
    for N in (1, 3, 5):
        offs = _image_offsets(box_l, pbc, N, 'square')
        expected = (2 * N + 1) ** 2
        assert offs.shape[0] == expected, f"N={N}: expected {expected}, got {offs.shape[0]}"
        np.testing.assert_array_equal(offs[:, 2], 0.)  # non-periodic z always 0


def test_image_offsets_disk_slab():
    """Disk truncation for slab PBC (T,T,F): only n_x²+n_y²≤N² offsets."""
    box_l = np.array([1., 1., 1e6])
    pbc   = [True, True, False]
    for N in (1, 2, 3):
        offs  = _image_offsets(box_l, pbc, N, 'disk')
        expected = sum(1 for nx in range(-N, N+1)
                         for ny in range(-N, N+1)
                         if nx*nx + ny*ny <= N*N)
        assert offs.shape[0] == expected, f"N={N}: expected {expected}, got {offs.shape[0]}"
        np.testing.assert_array_equal(offs[:, 2], 0.)


def test_image_offsets_1d_pbc():
    """1D PBC (T,F,F): disk and square are equivalent, count = 2N+1."""
    box_l = np.array([1., 1e6, 1e6])
    pbc   = [True, False, False]
    for N in (2, 5):
        sq = _image_offsets(box_l, pbc, N, 'square')
        dk = _image_offsets(box_l, pbc, N, 'disk')
        assert sq.shape[0] == 2 * N + 1
        assert dk.shape[0] == 2 * N + 1


# ─── 7. Replica reduces to MIC for isolated source ──────────────────────────

def test_replica_matches_mic_large_box():
    """With a very large box, image replicas are far away; replica sum ≈ MIC."""
    box_l = np.array([1e4, 1e4, 1e4])
    pbc   = [True, True, True]
    pos   = np.array([[0.0, 0.0, 0.0]])
    dips  = np.array([[0.0, 0.0, 1.0]])
    pt    = np.array([[1.0, 0.0, 0.0]])

    B_mic     = dipolar_field_at_points(pos, dips, pt, box_l, pbc, n_image_shells=0)
    B_replica = dipolar_field_at_points(pos, dips, pt, box_l, pbc, n_image_shells=5)

    np.testing.assert_allclose(B_mic, B_replica, atol=1e-10)


# ─── 8. PBC translation invariance with replicas ───────────────────────────

@pytest.mark.parametrize("n_shells", [0, 5])
def test_pbc_translation_invariance_replicas(n_shells):
    """Shift source+point by box_l[0]; field must be unchanged (with and without replicas)."""
    box_l = np.array([10.0, 50.0, 50.0])
    pbc   = [True, False, False]

    pos_orig = np.array([[0.3, 0.0, 0.0]])
    dips     = np.array([[0.0, 0.0, 1.0]])
    pt_orig  = np.array([[2.5, 0.0, 0.0]])

    B_orig    = dipolar_field_at_points(pos_orig, dips, pt_orig, box_l, pbc,
                                        n_image_shells=n_shells)
    pos_shift = pos_orig + np.array([[box_l[0], 0, 0]])
    pt_shift  = pt_orig  + np.array([[box_l[0], 0, 0]])
    B_shifted = dipolar_field_at_points(pos_shift, dips, pt_shift, box_l, pbc,
                                        n_image_shells=n_shells)

    np.testing.assert_allclose(B_orig, B_shifted, atol=1e-12)


# ─── 9. Analytical convergence tests (Gaussian-CGS) ────────────────────────

@pytest.mark.slow
def test_1d_madelung():
    """1D dipolar lattice sum converges to −2·ζ(3).

    Single μ=ẑ at origin, PBC x only (box_x=1, box_yz=1e6).
    Self is excluded (r²=0 skip); periodic images along x contribute.
    At n_image_shells=200: |B_z + 2·ζ(3)| < 5e-4 (0.02% of reference).
    """
    from scipy.special import zeta
    ref = 2.0 * zeta(3)   # ≈ 2.40411

    box_l = np.array([1.0, 1e6, 1e6])
    pbc   = [True, False, False]
    pos   = np.array([[0.0, 0.0, 0.0]])
    dips  = np.array([[0.0, 0.0, 1.0]])
    pt    = np.array([[0.0, 0.0, 0.0]])

    B = dipolar_field_at_points(pos, dips, pt, box_l, pbc, n_image_shells=200)

    assert abs(B[0, 2] + ref) < 5e-4, \
        f"1D Madelung: B_z={B[0,2]:.6f}, expected ≈ −{ref:.5f}"
    assert abs(B[0, 0]) < 1e-10, f"|B_x| = {abs(B[0,0]):.2e}, expected 0 by symmetry"
    assert abs(B[0, 1]) < 1e-10, f"|B_y| = {abs(B[0,1]):.2e}, expected 0 by symmetry"


@pytest.mark.slow
def test_2d_madelung_convergence():
    """2D square-lattice dipolar Madelung sum converges to −S₂.

    S₂ = Σ_{(m,n)≠0} (m²+n²)^{-3/2} ≈ 9.03362 (Topping/Born-Bradburn).
    Reference computed independently in-test at N_ref=2000.
    Assertions:
      - Cauchy convergence: successive |ΔB_z| shrink as n goes 10→50→200.
      - At n=200: |B_z + S2_ref| / S2_ref < 5e-3 (0.5%).
      - Lateral components < 1e-10 by symmetry.
    """
    # Reference: independent direct sum at N_ref=2000
    N_ref = 2000
    S2_ref = sum(
        (m*m + n*n)**(-1.5)
        for m in range(-N_ref, N_ref+1)
        for n in range(-N_ref, N_ref+1)
        if not (m == 0 and n == 0)
    )

    box_l = np.array([1.0, 1.0, 1e6])
    pbc   = [True, True, False]
    pos   = np.array([[0.0, 0.0, 0.0]])
    dips  = np.array([[0.0, 0.0, 1.0]])
    pt    = np.array([[0.0, 0.0, 0.0]])

    Bz_vals = []
    for n in [10, 50, 200]:
        B = dipolar_field_at_points(pos, dips, pt, box_l, pbc, n_image_shells=n)
        Bz_vals.append(B[0, 2])

    # Cauchy convergence: successive differences shrink
    d01 = abs(Bz_vals[1] - Bz_vals[0])
    d12 = abs(Bz_vals[2] - Bz_vals[1])
    assert d12 < d01, f"No Cauchy convergence: |Δ(10→50)|={d01:.4f}, |Δ(50→200)|={d12:.4f}"

    # Accuracy at n=200
    rel_err = abs(Bz_vals[2] + S2_ref) / S2_ref
    assert rel_err < 5e-3, \
        f"2D Madelung: B_z(n=200)={Bz_vals[2]:.5f}, S2_ref={S2_ref:.5f}, rel_err={rel_err:.4%}"

    B200 = dipolar_field_at_points(pos, dips, pt, box_l, pbc, n_image_shells=200)
    assert abs(B200[0, 0]) < 1e-10
    assert abs(B200[0, 1]) < 1e-10


@pytest.mark.slow
def test_lorentz_lorenz_volume_identity():
    """Cell-average of the BARE dipole field is ~0 for a cubic 3D-periodic crystal.

    Single μ=ẑ at origin, box=(1,1,1), full PBC, cubic symmetry.

    NOTE: the macroscopic Lorentz-Lorenz identity ⟨H⟩_cell = −4π·M does NOT
    hold for the bare lattice sum that H_internal computes. The bare dipole
    field Σ[3(m·r̂)r̂−m]/r³ summed over a cubic-symmetric region vanishes
    identically (Σx²/r⁵ = Σy²/r⁵ = Σz²/r⁵ ⟹ Σ(3z²−r²)/r⁵ = 0); the −4π·M
    macroscopic value requires the omitted contact/continuum-shape term.
    This test pins the correct bare-field behaviour: H[z] ≈ 0.
    """
    box_l = np.array([1.0, 1.0, 1.0])
    pbc   = [True, True, True]
    pos   = np.array([[0.0, 0.0, 0.0]])
    dips  = np.array([[0.0, 0.0, 1.0]])
    H_ext = np.array([0.0, 0.0, 0.0])

    n = 8
    coords = (np.arange(n) + 0.5) / n  # cell-centred: 0.0625, 0.1875, ...
    grid = np.array([[x, y, z] for x in coords for y in coords for z in coords])

    for shape in ('square', 'disk'):
        H = H_internal_bare(pos, dips, grid, box_l, pbc, H_ext,
                       exclusion_radius=0.15, n_image_shells=50,
                       replica_shape=shape)
        assert abs(H[2]) < 1e-2, \
            f"bare cubic cell-average ({shape}): H[z]={H[2]:.4e}, expected ≈ 0"
        assert abs(H[0]) < 1e-2, f"|H_x|={abs(H[0]):.2e} ({shape})"
        assert abs(H[1]) < 1e-2, f"|H_y|={abs(H[1]):.2e} ({shape})"


@pytest.mark.slow
def test_cubic_lattice_lorentz_field():
    """Bare site-sum in a simple cubic 3D-periodic lattice vanishes by symmetry.

    Single μ=ẑ at origin, box=(1,1,1), full PBC.

    NOTE: the continuum cubic Lorentz field is −(8π/3)·M, but H_lorentz sums the
    *bare* dipole field, whose cubic-symmetric lattice sum is exactly 0 for BOTH
    'square' and 'disk' truncations (each shell has Σ(3cos²θ−1)=0). The −(8π/3)·M
    value requires the omitted contact term. This test pins H[z] ≈ 0.
    """
    box_l = np.array([1.0, 1.0, 1.0])
    pbc   = [True, True, True]
    pos   = np.array([[0.0, 0.0, 0.0]])
    dips  = np.array([[0.0, 0.0, 1.0]])
    H_ext = np.array([0.0, 0.0, 0.0])

    for shape in ('square', 'disk'):
        H = H_lorentz_bare(pos, dips, box_l, pbc, H_ext, n_image_shells=20,
                      replica_shape=shape)
        assert abs(H[2]) < 1e-10, \
            f"bare cubic site-sum ({shape}): H[z]={H[2]:.2e}, expected ≈ 0"
        assert abs(H[0]) < 1e-10, f"|H_x| = {abs(H[0]):.2e} ({shape})"
        assert abs(H[1]) < 1e-10, f"|H_y| = {abs(H[1]):.2e} ({shape})"


# ─── 10. exclusion_radius behaviour ────────────────────────────────────────


# ─── 12. exclusion_radius behaviour ────────────────────────────────────────

def test_H_internal_exclusion_radius_none_keeps_all():
    """exclusion_radius=None → all points averaged (default)."""
    pos  = np.array([[0.0, 0.0, 0.0]])
    dips = np.array([[0.0, 0.0, 1.0]])
    H_ext = np.array([0.0, 0.0, 0.0])
    # One near-singular point very close to the source
    pts  = np.array([[0.05, 0.0, 0.0],
                     [2.0,  0.0, 0.0],
                     [4.0,  0.0, 0.0]])

    H_all = H_internal_bare(pos, dips, pts, _LARGE_BOX, _NO_PBC, H_ext,
                       exclusion_radius=None)
    H_far = H_internal_bare(pos, dips, pts[1:], _LARGE_BOX, _NO_PBC, H_ext,
                       exclusion_radius=None)
    # With the near-singular point included, result differs from far-only average
    assert not np.allclose(H_all, H_far, atol=1e-4)


def test_H_internal_exclusion_radius_drops_near_point():
    """exclusion_radius=0.1 removes the near-singular point at (0.05,0,0)."""
    pos  = np.array([[0.0, 0.0, 0.0]])
    dips = np.array([[0.0, 0.0, 1.0]])
    H_ext = np.array([0.0, 0.0, 0.0])
    pts  = np.array([[0.05, 0.0, 0.0],
                     [2.0,  0.0, 0.0],
                     [4.0,  0.0, 0.0]])

    H_excl = H_internal_bare(pos, dips, pts, _LARGE_BOX, _NO_PBC, H_ext,
                        exclusion_radius=0.1)
    H_far  = H_internal_bare(pos, dips, pts[1:], _LARGE_BOX, _NO_PBC, H_ext,
                        exclusion_radius=None)
    np.testing.assert_allclose(H_excl, H_far, atol=1e-12)


def test_H_internal_exclusion_radius_drops_all_returns_H_ext():
    """exclusion_radius large enough to drop all points → H_ext returned with warning."""
    pos  = np.array([[0.0, 0.0, 0.0]])
    dips = np.array([[0.0, 0.0, 1.0]])
    H_ext = np.array([1.0, 2.0, 3.0])
    pts  = np.array([[0.5, 0.0, 0.0]])

    with pytest.warns(UserWarning, match="exclusion_radius"):
        H = H_internal_bare(pos, dips, pts, _LARGE_BOX, _NO_PBC, H_ext,
                       exclusion_radius=10.0)
    np.testing.assert_allclose(H, H_ext, atol=1e-15)


# ─── 13. Disk ≡ square for absolutely convergent 2D sum ────────────────────

@pytest.mark.slow
def test_disk_equals_square_2d_madelung():
    """Disk and square truncations agree to 0.2% for the 2D Madelung sum.

    The 2D ⊥-dipole sum converges absolutely (1/r³ in 2D), so the
    truncation shape affects only convergence rate, not the limit.
    Setup: single μ=ẑ, box=(1,1,1e6), PBC (T,T,F), eval at origin.
    n_image_shells=80: |B_z(sq) − B_z(disk)| / |B_z(sq)| < 2e-3.
    """
    box_l = np.array([1.0, 1.0, 1e6])
    pbc   = [True, True, False]
    pos   = np.array([[0.0, 0.0, 0.0]])
    dips  = np.array([[0.0, 0.0, 1.0]])
    pt    = np.array([[0.0, 0.0, 0.0]])

    B_sq = dipolar_field_at_points(pos, dips, pt, box_l, pbc,
                                   n_image_shells=80, replica_shape='square')
    B_dk = dipolar_field_at_points(pos, dips, pt, box_l, pbc,
                                   n_image_shells=80, replica_shape='disk')

    rel_diff = abs(B_sq[0, 2] - B_dk[0, 2]) / abs(B_sq[0, 2])
    assert rel_diff < 2e-3, \
        f"disk vs square: B_z(sq)={B_sq[0,2]:.5f}, B_z(disk)={B_dk[0,2]:.5f}, rel_diff={rel_diff:.4%}"


# ─── 14. Surface-based H_internal (volume-average demag) ────────────────────

def _cubic_slab(n_lat, n_layers, ng, axis='z', a=1.0, z0=3.0):
    """Simple-cubic slab + flat top/bottom surfaces on a grid of ng cells per a.

    Returns (positions, dipoles, top, bottom, mesh_spacing, box_l). One dipole of
    unit moment along `axis` per a³ cell ⇒ M_vol = 1/a³. The slab is periodic in
    xy and `n_layers` cells thick, so the perpendicular demag is N_G→4π and the
    in-plane demag is →0 in the infinite-slab / fine-grid limit.
    """
    pos = np.array([[i * a, j * a, z0 + k * a]
                    for i in range(n_lat) for j in range(n_lat)
                    for k in range(n_layers)], dtype=np.float64)
    dips = np.zeros((pos.shape[0], 3))
    dips[:, {'x': 0, 'y': 1, 'z': 2}[axis]] = 1.0
    box_l = np.array([n_lat * a, n_lat * a, 2 * z0 + n_layers * a])
    dx = a / ng
    Nx = int(round(box_l[0] / dx))
    mesh = np.array([box_l[0] / Nx, box_l[1] / Nx, dx])
    top = np.full((Nx, Nx), z0 + (n_layers - 1) * a + a / 2)
    bot = np.full((Nx, Nx), z0 - a / 2)
    return pos, dips, top, bot, mesh, box_l


def test_H_internal_returns_vector_scalar_and_preserves_lateral():
    """Shape/units smoke test: (3,) H_int + scalar N_G; the applied field's
    lateral components pass through unchanged (no scalar+vector or np.sum bug),
    and a z-magnetised slab demagnetises (N_G > 0)."""
    pos, dips, top, bot, mesh, box_l = _cubic_slab(4, 2, ng=4, axis='z')
    H_ext = np.array([0.7, 0.0, 5.0])
    H_int, N_G = H_internal(pos, dips, top, bot, mesh, box_l,
                            [True, True, False], H_ext, sphere_radius=0.5,
                            n_image_shells=6)
    assert H_int.shape == (3,)
    assert np.isscalar(N_G) or np.ndim(N_G) == 0
    np.testing.assert_allclose(H_int[0], 0.7, atol=1e-9)   # x untouched
    np.testing.assert_allclose(H_int[1], 0.0, atol=1e-6)   # y stays ~0
    assert N_G > 0                                          # demag opposes M
    assert H_int[2] < H_ext[2]                              # z reduced


def test_H_internal_all_excluded_returns_H_ext_nan():
    """sphere_radius large enough to drop every matrix point → (H_ext, NaN) + warning."""
    pos, dips, top, bot, mesh, box_l = _cubic_slab(3, 2, ng=2, axis='z')
    H_ext = np.array([0.0, 0.0, 1.0])
    with pytest.warns(UserWarning, match="sphere_radius"):
        H_int, N_G = H_internal(pos, dips, top, bot, mesh, box_l,
                                [True, True, False], H_ext, sphere_radius=100.0,
                                n_image_shells=0)
    np.testing.assert_allclose(H_int, H_ext, atol=1e-15)
    assert np.isnan(N_G)


@pytest.mark.slow
def test_H_internal_cubic_slab_demag_converges():
    """The volume-average H_internal reproduces the slab demag (Gaussian).

    Perpendicular slab → N_G → 4π (N_SI → 1); in-plane → N_G ≈ 0. The residual
    is O(mesh_spacing) matrix-grid discretisation, so refining the grid drives
    the perpendicular N_G toward 4π. (Cross-checks dev_demag/demag_analysis.py's
    frozen-field N_G.)
    """
    FOURPI = 4.0 * np.pi
    pbc = [True, True, False]
    H_ext = np.zeros(3)

    def slab_N(ng, axis):
        pos, dips, top, bot, mesh, box_l = _cubic_slab(5, 3, ng=ng, axis=axis)
        return H_internal(pos, dips, top, bot, mesh, box_l, pbc, H_ext,
                          sphere_radius=0.5, n_image_shells=16)[1]

    # Perpendicular: refining the grid moves N_G closer to 4π, and the fine
    # grid is within 3%.
    N_coarse = slab_N(2, 'z')
    N_fine   = slab_N(8, 'z')
    assert abs(N_fine - FOURPI) < abs(N_coarse - FOURPI), \
        f"refinement did not improve: N_coarse={N_coarse:.3f}, N_fine={N_fine:.3f}, 4π={FOURPI:.3f}"
    assert abs(N_fine / FOURPI - 1.0) < 0.03, \
        f"perpendicular N_SI={N_fine/FOURPI:.4f}, expected ≈1 (N_G={N_fine:.3f}, 4π={FOURPI:.3f})"

    # In-plane: fine grid → N_G ≈ 0. Within 1%
    N_inplane = slab_N(8, 'x')
    assert abs(N_inplane) < 0.01, f"in-plane N_G={N_inplane:+.4f}, expected ≈0"
