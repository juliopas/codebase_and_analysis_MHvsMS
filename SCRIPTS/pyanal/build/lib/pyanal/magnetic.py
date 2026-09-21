from numba import njit, prange
import numpy as np
import math
import warnings

FASTMATH = True


def moment_avg(dipoles):
    """Mean dipole-moment vector ⟨m⟩ = (1/N) Σ m_i, shape (3,)."""
    return np.asarray(dipoles, dtype=np.float64).mean(axis=0)


def Magnetization(dipoles, volume_fraction, particle_size):
    """Magnetization of sample of volume V, given by the volume fraction of magnetic particles and their diameter/size."""
    return volume_fraction / ( np.pi / 6 * particle_size**3 ) * moment_avg(dipoles)


@njit(parallel=True, fastmath=FASTMATH, cache=True)
def moment_dipm_avg(dipoles):
    N = dipoles.shape[0]
    mx = 0.0
    my = 0.0
    mz = 0.0
    for i in prange(N):
        mui0, mui1, mui2 = dipoles[i]
        mui_sqr = mui0*mui0 + mui1*mui1 + mui2*mui2
        if mui_sqr < 1e-6:
            continue
        inv_mui = 1 / math.sqrt(mui_sqr)
        mx += mui0 * inv_mui
        my += mui1 * inv_mui
        mz += mui2 * inv_mui
    inv_N = 1.0 / N
    mx *= inv_N
    my *= inv_N
    mz *= inv_N
    return np.array([mx, my, mz], dtype=np.float64)


def _effective_box(box_l, periodicity):
    """Expand non-periodic axes by 1e6 so MIC never wraps in those directions."""
    eff = np.asarray(box_l, dtype=np.float64).copy()
    for d in range(3):
        if not periodicity[d]:
            eff[d] += 1e6
    return eff


def _image_offsets(box_l, periodicity, n_image_shells, replica_shape='square'):
    """Build (No, 3) cartesian image offsets for periodic-image replica sum.

    n_image_shells=0  → single [(0,0,0)] (home cell only).
    'square' : |n_d| ≤ N for each periodic axis.
    'disk'   : in the periodic subspace Σ n_d² ≤ N² (disk for slab,
               3-ball for full-PBC); non-periodic axes always have n_d=0.
    """
    box_l = np.asarray(box_l, dtype=np.float64)
    N  = int(n_image_shells)
    N2 = N * N

    ranges = []
    for d in range(3):
        ranges.append(range(-N, N + 1) if periodicity[d] else [0])

    offsets = []
    for n0 in ranges[0]:
        for n1 in ranges[1]:
            for n2 in ranges[2]:
                if replica_shape == 'disk':
                    r2 = 0
                    if periodicity[0]: r2 += n0 * n0
                    if periodicity[1]: r2 += n1 * n1
                    if periodicity[2]: r2 += n2 * n2
                    if r2 > N2:
                        continue
                offsets.append([n0 * box_l[0], n1 * box_l[1], n2 * box_l[2]])

    return np.array(offsets, dtype=np.float64)


@njit(parallel=True, fastmath=FASTMATH, cache=True)
def _dipolar_field_kernel_mic(positions, dipoles, points, eff_box, ids):
    """MIC dipolar field (Gaussian-CGS) at each point from dipoles[ids]. Skips r²=0."""
    Np = points.shape[0]
    Ns = ids.shape[0]

    inv_box0 = 1.0 / eff_box[0]
    inv_box1 = 1.0 / eff_box[1]
    inv_box2 = 1.0 / eff_box[2]

    result = np.zeros((Np, 3))

    for i in prange(Np):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]

        Bx = 0.0
        By = 0.0
        Bz = 0.0

        for k in range(Ns):
            j   = ids[k]
            mj0 = dipoles[j, 0]
            mj1 = dipoles[j, 1]
            mj2 = dipoles[j, 2]

            dx = positions[j, 0] - px
            dy = positions[j, 1] - py
            dz = positions[j, 2] - pz

            # minimum image convention
            dx -= eff_box[0] * math.floor(dx * inv_box0 + 0.5)
            dy -= eff_box[1] * math.floor(dy * inv_box1 + 0.5)
            dz -= eff_box[2] * math.floor(dz * inv_box2 + 0.5)

            r2 = dx*dx + dy*dy + dz*dz
            if r2 == 0.0:
                continue

            inv_r  = 1.0 / math.sqrt(r2)
            inv_r2 = inv_r  * inv_r
            inv_r3 = inv_r2 * inv_r
            inv_r5 = inv_r3 * inv_r2

            mjdotr = mj0*dx + mj1*dy + mj2*dz

            Bx += 3.0 * mjdotr * dx * inv_r5 - mj0 * inv_r3
            By += 3.0 * mjdotr * dy * inv_r5 - mj1 * inv_r3
            Bz += 3.0 * mjdotr * dz * inv_r5 - mj2 * inv_r3

        result[i, 0] = Bx
        result[i, 1] = By
        result[i, 2] = Bz

    return result


@njit(parallel=True, fastmath=FASTMATH, cache=True)
def _dipolar_field_kernel_replicas(positions, dipoles, points, ids, offsets):
    """Replica-sum dipolar field (Gaussian-CGS) at each point from dipoles[ids].

    Sums explicitly over all image offsets; no MIC is applied.
    r²=0 contributions are skipped (self-exclusion).
    """
    Np = points.shape[0]
    Ns = ids.shape[0]
    No = offsets.shape[0]

    result = np.zeros((Np, 3))

    for i in prange(Np):
        px = points[i, 0]
        py = points[i, 1]
        pz = points[i, 2]

        Bx = 0.0
        By = 0.0
        Bz = 0.0

        for k in range(Ns):
            j   = ids[k]
            mj0 = dipoles[j, 0]
            mj1 = dipoles[j, 1]
            mj2 = dipoles[j, 2]
            sx  = positions[j, 0]
            sy  = positions[j, 1]
            sz  = positions[j, 2]

            for o in range(No):
                dx = sx + offsets[o, 0] - px
                dy = sy + offsets[o, 1] - py
                dz = sz + offsets[o, 2] - pz

                r2 = dx*dx + dy*dy + dz*dz
                if r2 == 0.0:
                    continue

                inv_r  = 1.0 / math.sqrt(r2)
                inv_r2 = inv_r  * inv_r
                inv_r3 = inv_r2 * inv_r
                inv_r5 = inv_r3 * inv_r2

                mjdotr = mj0*dx + mj1*dy + mj2*dz

                Bx += 3.0 * mjdotr * dx * inv_r5 - mj0 * inv_r3
                By += 3.0 * mjdotr * dy * inv_r5 - mj1 * inv_r3
                Bz += 3.0 * mjdotr * dz * inv_r5 - mj2 * inv_r3

        result[i, 0] = Bx
        result[i, 1] = By
        result[i, 2] = Bz

    return result


@njit(parallel=True, cache=True)
def _keep_points_mask(points, positions, offsets, r2_thresh):
    """Bool mask: True for points farther than sqrt(r2_thresh) from every source image."""
    Np = points.shape[0]
    Ns = positions.shape[0]
    No = offsets.shape[0]
    keep = np.ones(Np, dtype=np.bool_)
    for i in prange(Np):
        px = points[i, 0]; py = points[i, 1]; pz = points[i, 2]
        ok = True
        for j in range(Ns):
            if not ok:
                break
            sx = positions[j, 0]; sy = positions[j, 1]; sz = positions[j, 2]
            for o in range(No):
                dx = sx + offsets[o, 0] - px
                dy = sy + offsets[o, 1] - py
                dz = sz + offsets[o, 2] - pz
                if dx*dx + dy*dy + dz*dz < r2_thresh:
                    ok = False
                    break
        keep[i] = ok
    return keep


def dipolar_field_at_points(positions, dipoles, points, box_l, periodicity,
                             mask=None, n_image_shells=0, replica_shape='square'):
    """Dipolar field (Gaussian-CGS, no μ₀/4π) at arbitrary 3D points.

    Parameters
    ----------
    positions      : (Nd, 3) array — source dipole positions
    dipoles        : (Nd, 3) array — source dipole moments
    points         : (Np, 3) array — evaluation points
    box_l          : (3,) array   — periodic box lengths
    periodicity    : (3,) bool    — PBC per axis
    mask           : (Nd,) bool or None — source-dipole mask; None = all dipoles
    n_image_shells : int — number of periodic image shells (0 = MIC only)
    replica_shape  : 'square' | 'disk' — truncation shape in periodic plane

    Returns
    -------
    (Np, 3) float64 — B_dip at each point.

    Notes
    -----
    n_image_shells=0: minimum-image convention; bit-for-bit identical to
    prior behaviour. n_image_shells>0: explicit replica sum, no MIC applied.
    r=0 contributions are skipped in both paths (self-exclusion).
    """
    positions = np.asarray(positions, dtype=np.float64)
    dipoles   = np.asarray(dipoles,   dtype=np.float64)
    points    = np.asarray(points,    dtype=np.float64)

    if mask is None:
        ids = np.arange(positions.shape[0], dtype=np.int64)
    else:
        ids = np.flatnonzero(np.asarray(mask, dtype=bool)).astype(np.int64)

    if n_image_shells == 0:
        eff_box = _effective_box(box_l, periodicity)
        return _dipolar_field_kernel_mic(positions, dipoles, points, eff_box, ids)
    else:
        offsets = _image_offsets(box_l, periodicity, n_image_shells, replica_shape)
        return _dipolar_field_kernel_replicas(positions, dipoles, points, ids, offsets)


def dipolar_field_xy_slice(positions, dipoles, z, box_l, periodicity,
                           grid_spacing, mask=None, n_image_shells=0, replica_shape='square'):
    """Dipolar field on a 2D XY grid at a fixed z height.

    Parameters
    ----------
    positions    : (Nd, 3) array
    dipoles      : (Nd, 3) array
    z            : float   — z coordinate of the evaluation plane
    box_l        : (3,) array
    periodicity  : (3,) bool array
    grid_spacing : float   — cell size in real-space units
    mask         : (Nd,) bool or None — source-dipole mask
    n_image_shells : int — image shells for replica sum
    replica_shape  : 'square' | 'disk'

    Returns
    -------
    X     : (Nx, Ny) meshgrid x coordinates (cell centres)
    Y     : (Nx, Ny) meshgrid y coordinates (cell centres)
    B_dip : (Nx, Ny, 3) dipolar field at each grid point
    """
    box_l = np.asarray(box_l, dtype=np.float64)
    x_bins = np.arange(0, box_l[0] + grid_spacing / 2, grid_spacing)
    y_bins = np.arange(0, box_l[1] + grid_spacing / 2, grid_spacing)

    x_centres = (x_bins[:-1] + x_bins[1:]) / 2
    y_centres = (y_bins[:-1] + y_bins[1:]) / 2

    X, Y = np.meshgrid(x_centres, y_centres, indexing='ij')
    Nx, Ny = X.shape

    X_flat = X.ravel()
    Y_flat = Y.ravel()
    Z_flat = np.full_like(X_flat, z)
    pts = np.column_stack([X_flat, Y_flat, Z_flat])

    B_flat = dipolar_field_at_points(positions, dipoles, pts, box_l, periodicity,
                                     mask=mask, n_image_shells=n_image_shells,
                                     replica_shape=replica_shape)
    B_dip  = B_flat.reshape(Nx, Ny, 3)

    return X, Y, B_dip


def H_lorentz_bare(positions, dipoles, box_l, periodicity, H_ext,
              mask=None, n_image_shells=0, replica_shape='square'):
    """Mean of the BARE dipole field evaluated at particle sites + H_ext.

    Computes ⟨B_dip(rᵢ)⟩ for i in mask (or all particles), where B_dip
    at each site is the *bare* point-dipole field Σ[3(m·r̂)r̂ − m]/r³ from
    ALL dipoles (self-interaction excluded via r=0 skip).

    WARNING — despite the name, this is NOT the continuum Lorentz local field
    H_macro + (4π/3)·M. It is the raw discrete site-sum, which omits the
    contact/self term and is conditionally convergent (shape-dependent):
      • a cubic-symmetric 3D-periodic site-sum is **exactly 0**, not −(8π/3)·M;
      • a slab in-plane site-sum gives the 2D Madelung constant, not −4π·M.
    Useful as a structural diagnostic (e.g. anisotropy of the discrete field),
    but do not read a Lorentz cavity factor or demag factor off it directly.

    Parameters
    ----------
    positions      : (N, 3) array
    dipoles        : (N, 3) array
    box_l          : (3,) array
    periodicity    : (3,) bool
    H_ext          : (3,) array — external field (sim units)
    mask           : (N,) bool or None — selects which PARTICLE SITES to
                     average over; sources are always all dipoles
    n_image_shells : int — 0=MIC, >0=explicit replica sum
    replica_shape  : 'square' | 'disk'

    Returns
    -------
    (3,) float64 — ⟨B_dip at particle sites⟩ + H_ext.
    """
    positions = np.asarray(positions, dtype=np.float64)
    H_ext     = np.asarray(H_ext,     dtype=np.float64)

    if mask is None:
        pts = positions
    else:
        pts = positions[np.asarray(mask, dtype=bool)]

    if pts.shape[0] == 0:
        return H_ext.copy()

    B_at_pts = dipolar_field_at_points(positions, dipoles, pts, box_l, periodicity,
                                       mask=None,
                                       n_image_shells=n_image_shells,
                                       replica_shape=replica_shape)
    return B_at_pts.mean(axis=0) + H_ext


def H_internal_bare(positions, dipoles, points, box_l, periodicity, H_ext,
               mask=None, exclusion_radius=None, n_image_shells=0, replica_shape='square'):
    """Mean of the BARE dipole field over a user-supplied 3D point cloud + H_ext.

    Caller constructs the point cloud filling the sample volume (e.g. a
    3D grid of cell-centred points inside the slab).

    WARNING — this is NOT the macroscopic internal field H = H_ext − N·M.
    It returns ⟨B_dip⟩ + H_ext where B_dip is the *bare* point-dipole field
    Σ[3(m·r̂)r̂ − m]/r³ (no contact/self term, no continuum shape term). The
    macroscopic demagnetizing field differs from this average by the omitted
    contact/Lorentz term, and the bare lattice sum is conditionally convergent
    (shape-dependent). Concretely:
      • a cubic-symmetric 3D-periodic sum is **exactly 0** (not −4π/3·M);
      • the matrix average is dominated by near-field points and does NOT
        converge to −N·M.
    Do not use this to read off a demagnetizing factor. Measure N
    thermodynamically (e.g. 1/χ_app − 1/χ_int) or via P3M+DLC instead.

    Parameters
    ----------
    positions        : (N, 3) array — particle positions (field sources)
    dipoles          : (N, 3) array — particle dipoles (field sources)
    points           : (Np, 3) array — evaluation points (define the volume)
    box_l            : (3,) array
    periodicity      : (3,) bool
    H_ext            : (3,) array
    mask             : (N,) bool or None — SOURCE-dipole mask (which particles emit);
                       None = all dipoles contribute
    exclusion_radius : float or None — exclude evaluation points within this
                       distance of any source image (image-aware); None = keep all
    n_image_shells   : int — 0=MIC, >0=explicit replica sum
    replica_shape    : 'square' | 'disk'

    Returns
    -------
    (3,) float64 — ⟨B_dip⟩ + H_ext.

    Notes
    -----
    `mask` here selects SOURCE dipoles, unlike in H_lorentz where it selects
    averaging targets. This matches the mask semantics of dipolar_field_at_points.
    """
    positions = np.asarray(positions, dtype=np.float64)
    points    = np.asarray(points,    dtype=np.float64)
    H_ext     = np.asarray(H_ext,     dtype=np.float64)

    if exclusion_radius is not None:
        offsets = _image_offsets(box_l, periodicity, n_image_shells, replica_shape)
        keep    = _keep_points_mask(points, positions, offsets,
                                    float(exclusion_radius) ** 2)
        points  = points[keep]
        if points.shape[0] == 0:
            warnings.warn(
                "H_internal_bare: exclusion_radius dropped all evaluation points; "
                "returning H_ext."
            )
            return H_ext.copy()

    if points.shape[0] == 0:
        return H_ext.copy()

    B_at_pts = dipolar_field_at_points(positions, dipoles, points, box_l, periodicity,
                                       mask=mask,
                                       n_image_shells=n_image_shells,
                                       replica_shape=replica_shape)
    return B_at_pts.mean(axis=0) + H_ext


def H_internal(positions, dipoles,
               top_surface, bottom_surface, mesh_spacing,
               box_l, periodicity, H_ext,
               sphere_radius, n_image_shells=0, replica_shape='square'):
    """Macroscopic internal H-field of a slab, by volume-averaging the micro-field.

    Gaussian-CGS, prefactor μ₀/4π = 1 (B = H). The macroscopic field is the volume average of the microscopic H over the sample; for a slab periodic
    in xy and finite in z this equals ``H_ext − N·M``. The sample volume is
    split into particle spheres + matrix:

      • Matrix points (kept ≥ sphere_radius from every particle image) lie
        OUTSIDE all spheres, where the field of a uniformly magnetised sphere is
        exactly the point-dipole field — so the bare dipole sum is exact there.
      • Inside sphere i the average of the field from the OTHER particles equals
        its value at the centre (mean-value theorem; the external field is
        harmonic inside the non-overlapping sphere), and the sphere's own
        uniform demag field is −(4π/3)·M_i (Gaussian).

    With V_sph = (4/3)π·r³, dV = ∏mesh_spacing, m_tot = Σ m_i = Σ M_i·V_i,
    V_tot = N_p·V_sph + N_m·dV, B_centre_i = bare field at centre i (home-cell
    self r²=0 skipped),
    B_mat_p = bare field at matrix point p:

        H_demag = (1/V_tot)[ V_sph·Σ_i B_centre_i − (4π/3)·m_tot + dV·Σ_p B_mat_p ]
        H_int   = H_ext + H_demag

    The returned demag factor is the projection along the magnetisation,
    N_G = −(H_demag·M̂)/|M_vol|  (M_vol = m_tot/V_tot), which is the best single
    scalar for reconstructing H_int = H_ext − N_G·M_vol along M (it equals N_zz
    when M∥ẑ). N_G sums to 4π (sphere 4π/3, slab⊥ 4π); N_SI = N_G/4π. V_tot
    cancels in N_G, so N_G is insensitive to the exact total volume.

    Parameters
    ----------
    positions      : (N, 3) array — particle positions (field sources)
    dipoles        : (N, 3) array — particle dipoles (field sources)
    top_surface    : (Nx, Ny) array — upper z of the slab on the xy grid
    bottom_surface : (Nx, Ny) array — lower z of the slab on the xy grid
    mesh_spacing   : (3,) array — matrix-grid cell size (dx, dy, dz). The lateral
                     dx, dy must match the surface grid (origin 0,
                     dx = box_l[0]/Nx, dy = box_l[1]/Ny), as produced by
                     ``detect_surface_with_spheres``.
    box_l          : (3,) array
    periodicity    : (3,) bool
    H_ext          : (3,) array — applied field (sim units; B=H)
    sphere_radius  : float — physical particle radius. Used to exclude matrix
                     points (image-aware), to weight V_sph, and for the self
                     term, so the excluded matrix volume equals the sphere volume.
    n_image_shells : int — 0=MIC, >0=explicit replica sum (raise for in-plane
                     convergence)
    replica_shape  : 'square' | 'disk'

    Returns
    -------
    H_int : (3,) float64 — macroscopic internal field H_ext + H_demag.
    N_G   : float — Gaussian demag factor along M (H_int = H_ext − N_G·M_vol);
            NaN if all matrix points were excluded or |M_vol| = 0.

    Notes
    -----
    Accuracy is controlled by (i) the matrix-grid spacing and (ii) n_image_shells.
    The matrix integral converges as O(mesh_spacing): the near-field of the
    nearest particle just outside the exclusion sphere dominates the error, so a
    fine lateral/vertical spacing is needed — empirically dx ≈ σ/8 (≈ r/4) gives
    a perpendicular cubic slab N_SI within ~1 % of 1, while dx ≈ σ/2 is off by
    tens of %. Increase n_image_shells for in-plane convergence (the slab is the
    finite disk of radius ≈ n_image_shells·box_l; N_G → 4π/0 only as it grows).
    """
    positions      = np.asarray(positions,      dtype=np.float64)
    dipoles        = np.asarray(dipoles,         dtype=np.float64)
    top_surface    = np.asarray(top_surface,     dtype=np.float64)
    bottom_surface = np.asarray(bottom_surface,  dtype=np.float64)
    H_ext          = np.asarray(H_ext,           dtype=np.float64)

    r     = float(sphere_radius)
    V_sph = 4.0 / 3.0 * np.pi * r ** 3
    dV    = float(np.prod(mesh_spacing))
    N_p   = positions.shape[0]

    # Matrix grid filling the slab, excluding points inside any particle sphere.
    points  = surface_point_cloud(top_surface, bottom_surface, mesh_spacing)
    offsets = _image_offsets(box_l, periodicity, n_image_shells, replica_shape)
    keep    = _keep_points_mask(points, positions, offsets, r * r)
    points  = points[keep]
    N_m     = points.shape[0]

    if N_m == 0:
        warnings.warn(
            "H_internal: sphere_radius dropped all matrix evaluation points; "
            "returning H_ext with NaN demag factor."
        )
        return H_ext.copy(), np.nan

    # One bare-dipole-field evaluation for matrix points AND sphere centres.
    # At a centre the home-cell self term (r²=0) is skipped → field from the
    # other dipoles (and periodic images); the sphere's own demag is added
    # analytically as −(4π/3)·M_i below.
    eval_pts = np.vstack((points, positions))
    B = dipolar_field_at_points(positions, dipoles, eval_pts, box_l, periodicity,
                                n_image_shells=n_image_shells,
                                replica_shape=replica_shape)
    B_mat     = B[:N_m]
    B_centres = B[N_m:]

    m_tot = dipoles.sum(axis=0)
    V_tot = N_p * V_sph + N_m * dV

    H_demag = (V_sph * B_centres.sum(axis=0)
               - 4.0 / 3.0 * np.pi * m_tot
               + dV * B_mat.sum(axis=0)) / V_tot
    H_int = H_ext + H_demag

    M_vol = m_tot / V_tot
    M_mag = np.linalg.norm(M_vol)
    if M_mag == 0.0:
        N_G = np.nan
    else:
        N_G = -float(np.dot(H_demag, M_vol)) / (M_mag * M_mag)

    return H_int, N_G

@njit(parallel=True, cache=True)
def _count_points(top: np.ndarray, bottom: np.ndarray, dz: float) -> np.ndarray:
    """Number of full, cell-centred z-cells of height dz per (i,j) column."""
    N, M = top.shape
    counts = np.empty((N, M), dtype=np.int64)
    for i in prange(N):
        for j in range(M):
            h = top[i, j] - bottom[i, j]
            counts[i, j] = int(math.floor(h / dz)) if h > 0.0 else 0
    return counts


@njit(parallel=True, cache=True)
def _fill_columns(
    bottom: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    offsets: np.ndarray,    # (N*M+1,) prefix sum → thread-safe write positions
    counts: np.ndarray,
    dz: float,
    out: np.ndarray,
) -> None:
    """Cell-centred uniform fill: z = bottom + (k + 0.5)·dz → constant cell volume."""
    N, M = bottom.shape
    for i in prange(N):
        for j in range(M):
            n   = counts[i, j]
            idx = offsets[i * M + j]
            z0  = bottom[i, j]
            xi  = x[i]
            yj  = y[j]
            for k in range(n):
                out[idx + k, 0] = xi
                out[idx + k, 1] = yj
                out[idx + k, 2] = z0 + (k + 0.5) * dz


def surface_point_cloud(
    top_surface:    np.ndarray,
    bottom_surface: np.ndarray,
    mesh_spacing: np.ndarray,
) -> np.ndarray:
    """
    Fill the volume between two surfaces with a uniform 3-D point cloud.

    Each vertical column (i, j) is sampled at cell-centred heights
    ``bottom[i,j] + (k + 0.5)·dz`` for the ``floor((top−bottom)/dz)`` full cells
    of size dz = mesh_spacing[2]. Every point therefore represents the SAME cell
    volume ``∏mesh_spacing``, so a volume average over the cloud is a plain mean.

    The lateral coordinates are ``x_i = i·mesh_spacing[0]``,
    ``y_j = j·mesh_spacing[1]`` (origin 0), matching the grid produced by
    ``detect_surface_with_spheres`` when mesh_spacing[:2] = box_l[:2]/shape.

    Parameters
    ----------
    top_surface    : (N, M) array  – upper Z values on the XY grid
    bottom_surface : (N, M) array  – lower Z values on the XY grid
    mesh_spacing   : (3,) array - grid spacing (x,y,z)

    Returns
    -------
    points : (P, 3) float64 array  – columns [x, y, z]
    """
    N, M = top_surface.shape

    dz = float(mesh_spacing[2])
    x = np.arange(N, dtype=np.float64) * mesh_spacing[0]
    y = np.arange(M, dtype=np.float64) * mesh_spacing[1]

    # contiguous float64 — required by Numba
    top    = np.ascontiguousarray(top_surface,    dtype=np.float64)
    bottom = np.ascontiguousarray(bottom_surface, dtype=np.float64)
    x      = np.ascontiguousarray(x,              dtype=np.float64)
    y      = np.ascontiguousarray(y,              dtype=np.float64)

    # Pass 1 – parallel count
    counts = _count_points(top, bottom, dz)

    # Prefix sum (sequential, O(N²) in C – negligible vs the fill)
    offsets = np.empty(N * M + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts.ravel(), out=offsets[1:])

    # Pass 2 – parallel fill into exact-size buffer
    out = np.empty((int(offsets[-1]), 3), dtype=np.float64)
    _fill_columns(bottom, x, y, offsets, counts, dz, out)

    return out
