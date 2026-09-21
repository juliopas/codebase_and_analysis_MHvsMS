import numpy as np
import math
import scipy.ndimage as ndi
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator

from numba import njit, prange, get_num_threads

import os

from .generic_functions import fold_coord, weighted_com_pbc, run_subprocess_in_module, run_subprocess_in_module_mp, tempfile_path

FASTMATH=True

def get_module_name():
    return "pyanal.surface"

##### main #####
## HEIGHT + Peaks ##

##
#####

def _build_box_l_starting_grid(box_l, grid_res_x= None, grid_res_y= None, grid_bottom=None):
    if grid_bottom is None: # Defaults to z=0
        # Default to 1, if not inserted
        grid_res_x = grid_res_x if grid_res_x is not None else 1
        grid_res_y = grid_res_y if grid_res_y is not None else 1
        nx = math.ceil(box_l[0] / grid_res_x)
        ny = math.ceil(box_l[1] / grid_res_y)
        grid_res_x = box_l[0] / nx
        grid_res_y = box_l[1] / ny
        grid_bottom = np.zeros((nx, ny), dtype=np.float32)
    elif isinstance(grid_bottom, (int, float)):
        # Default to 1, if not inserted
        grid_res_x = grid_res_x if grid_res_x is not None else 1
        grid_res_y = grid_res_y if grid_res_y is not None else 1
        nx = math.ceil(box_l[0] / grid_res_x)
        ny = math.ceil(box_l[1] / grid_res_y)
        grid_res_x = box_l[0] / nx
        grid_res_y = box_l[1] / ny
        grid_bottom = np.full((nx, ny), grid_bottom, dtype=np.float32)
    else:
        if grid_res_x is not None or grid_res_y is not None:
            raise ValueError("If you input a grid_bottom, you cannot define the grid resolution, as it will be defined by the grid_bottom.")
        nx, ny = grid_bottom.shape
        grid_res_x = box_l[0] / nx
        grid_res_y = box_l[1] / ny

    return grid_bottom, (nx, ny), (grid_res_x, grid_res_y)

##### Height maps #####
def grid_max_with_radius(positions, radius, box_l, grid_res_x=None, grid_res_y=None, grid_bottom=None):
    grid_bottom, (nx, ny), (grid_res_x, grid_res_y) = _build_box_l_starting_grid(box_l, grid_res_x, grid_res_y, grid_bottom)
    x = np.ascontiguousarray(positions[:,0])
    y = np.ascontiguousarray(positions[:,1])
    z = np.ascontiguousarray(positions[:,2])
    radius = np.ascontiguousarray(radius)
    grid = grid_max_with_radius_numba(x, y, z, radius, nx, ny, grid_res_x, grid_res_y, grid_bottom.copy())
    return grid

@njit(fastmath=FASTMATH)
def grid_max_with_radius_numba(x, y, z, radius, nx, ny, grid_res_x, grid_res_y, grid_bottom):
    grid = grid_bottom

    inv_res_x = 1 / grid_res_x
    inv_res_y = 1 / grid_res_y
    half_res_x = grid_res_x * 0.5
    half_res_y = grid_res_y * 0.5

    for i in range(z.shape[0]):
        ix = int(x[i] * inv_res_x)
        iy = int(y[i] * inv_res_y)

        rgx = math.ceil(radius[i] * inv_res_x)

        radius_sqr = radius[i] * radius[i]
        xi = x[i]
        yi = y[i]
        zi = z[i]

        for gx in range(ix - rgx, ix + rgx + 1):
            cx = gx * grid_res_x + half_res_x
            dx = cx - xi
            dx2 = dx*dx
            if dx2 > radius_sqr:
                continue

            gx0 = gx
            if gx0 < 0:
                gx0 += nx
            elif gx0 >= nx:
                gx0 -= nx

            rgy = int(math.sqrt(radius_sqr - dx2) * inv_res_y)
            for gy in range(iy - rgy, iy + rgy + 1):
                cy = gy * grid_res_y + half_res_y
                dy = cy - yi
                dy2 = dy * dy

                d2 = dx2 + dy2
                if d2 > radius_sqr:
                    continue

                z_surface = zi + math.sqrt(radius_sqr - d2)

                gy0 = gy
                if gy0 < 0:
                    gy0 += ny
                elif gy0 >= ny:
                    gy0 -= ny

                if z_surface > grid[gx0, gy0]:
                    grid[gx0, gy0] = z_surface

    return grid

def grid_max(positions, box_l, grid_res_x=None, grid_res_y=None, grid_bottom=None):
    grid_bottom, _, (grid_res_x, grid_res_y) = _build_box_l_starting_grid(box_l, grid_res_x, grid_res_y, grid_bottom)
    grid = _grid_max_numba(positions, grid_res_x, grid_res_y, grid_bottom.copy())
    return grid

def _grid_max_numba(positions, grid_res_x, grid_res_y, grid):
    n = positions.shape[0]
    nx, ny = grid.shape
    if n > 10000 and n > 4 * nx * ny and nx * ny <= 500_000:
        return _grid_max_parallel(np.ascontiguousarray(positions), grid_res_x, grid_res_y, grid)
    return _grid_max_serial(np.ascontiguousarray(positions), grid_res_x, grid_res_y, grid)
@njit(fastmath=FASTMATH, boundscheck=False, cache=True)
def _grid_max_serial(positions, grid_res_x, grid_res_y, grid):
    inv_res_x = 1.0 / grid_res_x
    inv_res_y = 1.0 / grid_res_y
    for i in range(positions.shape[0]):
        ix = int(positions[i, 0] * inv_res_x)
        iy = int(positions[i, 1] * inv_res_y)
        zi = positions[i, 2]
        if zi > grid[ix, iy]:
            grid[ix, iy] = zi
    return grid
@njit(parallel=True, fastmath=FASTMATH, boundscheck=False, cache=True)
def _grid_max_parallel(positions, grid_res_x, grid_res_y, grid):
    inv_res_x = 1.0 / grid_res_x
    inv_res_y = 1.0 / grid_res_y
    nx, ny = grid.shape
    n = positions.shape[0]
    nthr = max(get_num_threads(), 2)
    local = np.full((nthr, nx, ny), -np.inf)
    for t in prange(nthr):
        for i in range(t, n, nthr):
            ix = int(positions[i, 0] * inv_res_x)
            iy = int(positions[i, 1] * inv_res_y)
            zi = positions[i, 2]
            if zi > local[t, ix, iy]:
                local[t, ix, iy] = zi
    for t in range(nthr):
        for i in range(nx):
            for j in range(ny):
                v = local[t, i, j]
                if v > grid[i, j]:
                    grid[i, j] = v
    return grid
#####


##### Peaks #####
## Find peaks in 2d grid surface map##
def find_peaks(Z, size_float, grid_res_x, grid_res_y, min_peak_z, box_l):
    sigma = 1.6 * size_float / max(grid_res_x, grid_res_y)
    Zs = ndi.gaussian_filter(Z, sigma=sigma, mode='wrap')

    footprint = np.ones((3, 3))
    local_max = (Zs == ndi.maximum_filter(Zs, footprint=footprint, mode='wrap'))
    # remove flat numerical plateaus
    local_max &= (Zs > ndi.minimum_filter(Zs, footprint=footprint, mode='wrap'))

    r = int(np.ceil(size_float / max(grid_res_x, grid_res_y)))
    grid_res_array = np.array([grid_res_x, grid_res_y])
    box_array = np.array(box_l[:2])

    peak_coords = np.argwhere(local_max & (Zs > min_peak_z))
    Z_region = []
    coords_zone = []
    coords_max_point = []

    shape = np.array(Zs.shape)
    for i, j in peak_coords:
        # PBC-aware slicing
        i_indices = (np.arange(i-r, i+r+1)) % shape[0]
        j_indices = (np.arange(j-r, j+r+1)) % shape[1]

        region = Zs[np.ix_(i_indices, j_indices)]
        mask = region >= (Zs[i, j] - 0.25*np.std(region))

        if mask.sum() < 3:
            continue

        ii, jj = np.meshgrid(i_indices, j_indices, indexing='ij')
        center = np.column_stack([ii[mask], jj[mask]])
        max_pixel = center[np.argmax(region[mask])]
        center = weighted_com_pbc(center, weights=region[mask], box_l=shape[:2])

        coords_zone.append((center*grid_res_array) % box_array)
        coords_max_point.append((max_pixel*grid_res_array) % box_array)
        Z_region.append(region.max())

    return np.array(Z_region, dtype=np.float32), np.array(coords_zone, dtype=np.float32).reshape(-1, 2), np.array(coords_max_point, dtype=np.float32).reshape(-1, 2)
##

## Calculate Peak statistics (distribution of distances and heights)
def peak_statistics(Z_peaks, pos_peaks, box_l):
    height_mean = np.mean(Z_peaks)
    height_std = np.std(Z_peaks)

    peak_dist_dist = peak_distance_dist(pos_peaks, box_l)
    dist_mean = np.mean(peak_dist_dist)
    dist_std = np.std(peak_dist_dist)

    return (Z_peaks, height_mean, height_std), (peak_dist_dist, dist_mean, dist_std)
@njit(fastmath=FASTMATH, cache=True)
def peak_distance_dist(pos_peaks, box_l):
    n = pos_peaks.shape[0]
    n_pairs = n*(n-1)//2
    dists = np.empty(n_pairs, dtype=np.float32)

    inv_box0 = 1 / box_l[0]
    inv_box1 = 1/ box_l[1]

    idx = 0
    for i in range(n-1):
        for j in range(i+1, n):
            dx = pos_peaks[i,0] - pos_peaks[j,0]
            dy = pos_peaks[i,1] - pos_peaks[j,1]
            # minimum image
            dx -= box_l[0] * math.floor(dx * inv_box0 + 0.5)
            dy -= box_l[1] * math.floor(dy * inv_box1 + 0.5)
            dists[idx] = math.sqrt(dx*dx + dy*dy)
            idx += 1
    return dists
##
#####

##### Surface mapping #####
@njit(cache=True)
def _build_cell_list_2d(x, y, ncx, ncy, inv_csx, inv_csy, pbc):
    n = x.shape[0]
    n_cells = ncx * ncy
    counts = np.zeros(n_cells, dtype=np.int32)
    for i in range(n):
        if pbc:
            cx = int(x[i] * inv_csx) % ncx
            cy = int(y[i] * inv_csy) % ncy
        else:
            cx = int(x[i] * inv_csx)
            if cx < 0:
                cx = 0
            elif cx >= ncx:
                cx = ncx - 1
            cy = int(y[i] * inv_csy)
            if cy < 0:
                cy = 0
            elif cy >= ncy:
                cy = ncy - 1
        counts[cx * ncy + cy] += 1

    cell_starts = np.zeros(n_cells + 1, dtype=np.int32)
    for c in range(n_cells):
        cell_starts[c + 1] = cell_starts[c] + counts[c]

    cursor = cell_starts[:n_cells].copy()
    cell_particles = np.empty(n, dtype=np.int32)
    for i in range(n):
        if pbc:
            cx = int(x[i] * inv_csx) % ncx
            cy = int(y[i] * inv_csy) % ncy
        else:
            cx = int(x[i] * inv_csx)
            if cx < 0:
                cx = 0
            elif cx >= ncx:
                cx = ncx - 1
            cy = int(y[i] * inv_csy)
            if cy < 0:
                cy = 0
            elif cy >= ncy:
                cy = ncy - 1
        cell = cx * ncy + cy
        cell_particles[cursor[cell]] = i
        cursor[cell] += 1

    return cell_starts, cell_particles


def _itim_numba_setup(poss_folded, radii, R, box_l, grid_spacing=None, grid_bottom=None, pbc=True):
    grid_bottom, (nx, ny), (grid_res_x, grid_res_y) = _build_box_l_starting_grid(box_l, grid_spacing, grid_spacing, grid_bottom)

    x_grid = np.arange(nx) * grid_res_x
    y_grid = np.arange(ny) * grid_res_y
    X, Y = np.meshgrid(x_grid, y_grid, indexing='ij')

    assert grid_bottom.shape == X.shape == Y.shape

    cutoff = float(R + np.max(radii))
    if pbc:
        assert 2.0 * cutoff < min(box_l[0], box_l[1]), \
            "cutoff (R + max(radii)) must be < half-box for PBC 3x3 stencil"

    ncx = max(1, int(box_l[0] / cutoff))
    ncy = max(1, int(box_l[1] / cutoff))
    cell_size_x = box_l[0] / ncx
    cell_size_y = box_l[1] / ncy
    inv_csx = 1.0 / cell_size_x
    inv_csy = 1.0 / cell_size_y

    cell_starts, cell_particles = _build_cell_list_2d(
        np.ascontiguousarray(poss_folded[:, 0], dtype=np.float32),
        np.ascontiguousarray(poss_folded[:, 1], dtype=np.float32),
        ncx, ncy, inv_csx, inv_csy, pbc,
    )

    lattice_points = np.column_stack([X.ravel(), Y.ravel()]).astype(np.float32)
    flat_bottom = grid_bottom.copy().ravel()

    return (X, Y), lattice_points, cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy, flat_bottom

def detect_surface_particles_itim(particle_positions, radii, R, box_l, grid_spacing=None, grid_bottom=None, pbc=True):
    poss_tmp_path = tempfile_path(suffix=".npy")
    radii_tmp_path = tempfile_path(suffix=".npy")
    grid_bottom_tmp_path = None
    return_file_path = None
    np.save(poss_tmp_path, particle_positions)
    np.save(radii_tmp_path, radii)
    if grid_bottom is not None:
        grid_bottom_tmp_path = tempfile_path(suffix=".npy")
        np.save(grid_bottom_tmp_path, grid_bottom)
    args = (poss_tmp_path, radii_tmp_path, R, box_l, grid_spacing, grid_bottom_tmp_path, pbc)
    try:
        return_file_path = run_subprocess_in_module_mp(args, module=get_module_name(), worker=detect_surface_particles_itim_worker.__name__)
        data = np.load(return_file_path)
        return data['poss_surface_part']
    finally:
        for p in (poss_tmp_path, radii_tmp_path, grid_bottom_tmp_path, return_file_path):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass

def detect_surface_particles_itim_worker(poss_path, radii_path, R, box_l, grid_spacing=None, grid_bottom_path=None, pbc=True):
    particle_positions = np.load(poss_path)
    radii = np.load(radii_path)
    grid_bottom = np.load(grid_bottom_path) if grid_bottom_path is not None else None

    box_l_special = [box_l[0], box_l[1], 1E6]
    poss_folded = fold_coord(particle_positions, box_dim=box_l_special)

    (_, _), lattice_points, cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy, flat_bottom = _itim_numba_setup(
        poss_folded, radii, R, box_l, grid_spacing, grid_bottom, pbc=pbc)

    box_l0 = float(box_l[0])
    box_l1 = float(box_l[1])
    inv_box0 = 1.0 / box_l0
    inv_box1 = 1.0 / box_l1

    part_idxs = compute_surface_particles_idx_itim_numba(
        poss_folded, radii, lattice_points,
        cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy,
        box_l0, box_l1, inv_box0, inv_box1, pbc,
        R, flat_bottom,
    )

    return_file_path = tempfile_path(suffix=".npz")
    np.savez(return_file_path, poss_surface_part=poss_folded[part_idxs])
    return return_file_path
@njit(parallel=True, fastmath=FASTMATH, cache=True)
def compute_surface_particles_idx_itim_numba(
    centers, radii, lattice_points,
    cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy,
    box_l0, box_l1, inv_box0, inv_box1, pbc,
    R, flat_bottom,
):
    n_lattice = lattice_points.shape[0]
    n_particles = centers.shape[0]

    # Race on writes is benign: we only ever set True (idempotent).
    seen = np.zeros(n_particles, dtype=np.bool_)
    for lattice_i in prange(n_lattice):
        lx = lattice_points[lattice_i, 0]
        ly = lattice_points[lattice_i, 1]
        z_max = flat_bottom[lattice_i]
        idx_max = -1

        if pbc:
            cx = int(lx * inv_csx) % ncx
            cy = int(ly * inv_csy) % ncy
        else:
            cx = int(lx * inv_csx)
            if cx < 0:
                cx = 0
            elif cx >= ncx:
                cx = ncx - 1
            cy = int(ly * inv_csy)
            if cy < 0:
                cy = 0
            elif cy >= ncy:
                cy = ncy - 1

        for ddx in range(-1, 2):
            ncx_i = cx + ddx
            if pbc:
                if ncx_i < 0:
                    ncx_i += ncx
                elif ncx_i >= ncx:
                    ncx_i -= ncx
            else:
                if ncx_i < 0 or ncx_i >= ncx:
                    continue
            for ddy in range(-1, 2):
                ncy_j = cy + ddy
                if pbc:
                    if ncy_j < 0:
                        ncy_j += ncy
                    elif ncy_j >= ncy:
                        ncy_j -= ncy
                else:
                    if ncy_j < 0 or ncy_j >= ncy:
                        continue
                cell = ncx_i * ncy + ncy_j
                for k in range(cell_starts[cell], cell_starts[cell + 1]):
                    idx = cell_particles[k]
                    dx = lx - centers[idx, 0]
                    dy = ly - centers[idx, 1]
                    if pbc:
                        dx -= box_l0 * math.floor(dx * inv_box0 + 0.5)
                        dy -= box_l1 * math.floor(dy * inv_box1 + 0.5)
                    d2 = dx * dx + dy * dy

                    r_sum = R + radii[idx]
                    r_sum_sqr = r_sum * r_sum
                    diff = r_sum_sqr - d2
                    if diff < 0.0:
                        continue

                    cz = centers[idx, 2]
                    rhs = z_max - cz + R
                    if rhs >= 0.0 and diff <= rhs * rhs:
                        continue

                    z = cz + math.sqrt(diff) - R
                    if z > z_max:
                        z_max = z
                        idx_max = idx

        if idx_max >= 0:
            seen[idx_max] = True   # write-only race; safe
    # Compact (serial)
    count = 0
    for i in range(n_particles):
        if seen[i]:
            count += 1
    out = np.empty(count, dtype=np.int64)
    j = 0
    for i in range(n_particles):
        if seen[i]:
            out[j] = i
            j += 1
    return out

def detect_surface_with_spheres(particle_positions, radii, R, box_l, grid_spacing=None, grid_bottom=None, pbc=True):
    """
    Detect top surface by dropping spheres of radius R, accounting for particle radii. Assume floor at z=0

    Parameters:
        particle_positions : (N, 3) array of particle positions
        radii : (N,) array of particle radii
        R : Sphere radius
        grid_spacing : Spacing between drop points in x-y plane
        grid_bottom : bottom of the grid (lowest limit of surface)
        pbc : if True, periodic in X and Y (never in Z)

    Returns:
        X, Y, Z: Grid coordinates and surface heights
    """
    poss_tmp_path = tempfile_path(suffix=".npy")
    radii_tmp_path = tempfile_path(suffix=".npy")
    grid_bottom_tmp_path = None
    return_file_path = None
    np.save(poss_tmp_path, particle_positions)
    np.save(radii_tmp_path, radii)
    if grid_bottom is not None:
        grid_bottom_tmp_path = tempfile_path(suffix=".npy")
        np.save(grid_bottom_tmp_path, grid_bottom)
    args = (poss_tmp_path, radii_tmp_path, R, box_l, grid_spacing, grid_bottom_tmp_path, pbc)
    try:
        return_file_path = run_subprocess_in_module_mp(args, module=get_module_name(), worker=detect_surface_with_spheres_worker.__name__)
        data = np.load(return_file_path)
        return data['X'], data['Y'], data['Z']
    finally:
        for p in (poss_tmp_path, radii_tmp_path, grid_bottom_tmp_path, return_file_path):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass

def detect_surface_with_spheres_worker(poss_path, radii_path, R, box_l, grid_spacing=None, grid_bottom_path=None, pbc=True):
    particle_positions = np.load(poss_path)
    radii = np.load(radii_path)
    grid_bottom = np.load(grid_bottom_path) if grid_bottom_path is not None else None

    box_l_special = [box_l[0], box_l[1], 1E6]
    poss_folded = fold_coord(particle_positions, box_dim=box_l_special)

    (X, Y), lattice_points, cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy, flat_bottom = _itim_numba_setup(
        poss_folded, radii, R, box_l, grid_spacing, grid_bottom, pbc=pbc)

    box_l0 = float(box_l[0])
    box_l1 = float(box_l[1])
    inv_box0 = 1.0 / box_l0
    inv_box1 = 1.0 / box_l1

    Z = compute_sphere_contact_heights_on_lattice_flat(
        poss_folded, radii, lattice_points,
        cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy,
        box_l0, box_l1, inv_box0, inv_box1, pbc,
        R, flat_bottom,
    ).reshape(X.shape)

    return_file_path = tempfile_path(suffix=".npz")
    np.savez(return_file_path, X=X, Y=Y, Z=Z)
    return return_file_path

@njit(parallel=True, fastmath=FASTMATH)
def compute_sphere_contact_heights_on_lattice_flat(
    centers, radii, lattice_points,
    cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy,
    box_l0, box_l1, inv_box0, inv_box1, pbc,
    R, flat_bottom,
):
    n_lattice_points = lattice_points.shape[0]
    Z = flat_bottom
    for lattice_i in prange(n_lattice_points):
        lx = lattice_points[lattice_i, 0]
        ly = lattice_points[lattice_i, 1]
        z_max = flat_bottom[lattice_i]

        if pbc:
            cx = int(lx * inv_csx) % ncx
            cy = int(ly * inv_csy) % ncy
        else:
            cx = int(lx * inv_csx)
            if cx < 0:
                cx = 0
            elif cx >= ncx:
                cx = ncx - 1
            cy = int(ly * inv_csy)
            if cy < 0:
                cy = 0
            elif cy >= ncy:
                cy = ncy - 1

        for ddx in range(-1, 2):
            ncx_i = cx + ddx
            if pbc:
                if ncx_i < 0:
                    ncx_i += ncx
                elif ncx_i >= ncx:
                    ncx_i -= ncx
            else:
                if ncx_i < 0 or ncx_i >= ncx:
                    continue
            for ddy in range(-1, 2):
                ncy_j = cy + ddy
                if pbc:
                    if ncy_j < 0:
                        ncy_j += ncy
                    elif ncy_j >= ncy:
                        ncy_j -= ncy
                else:
                    if ncy_j < 0 or ncy_j >= ncy:
                        continue
                cell = ncx_i * ncy + ncy_j
                for k in range(cell_starts[cell], cell_starts[cell + 1]):
                    idx = cell_particles[k]
                    dx = lx - centers[idx, 0]
                    dy = ly - centers[idx, 1]
                    if pbc:
                        dx -= box_l0 * math.floor(dx * inv_box0 + 0.5)
                        dy -= box_l1 * math.floor(dy * inv_box1 + 0.5)
                    d2 = dx * dx + dy * dy

                    r_sum = R + radii[idx]
                    r_sum_sqr = r_sum * r_sum
                    if d2 <= r_sum_sqr:
                        z = centers[idx, 2] + math.sqrt(r_sum_sqr - d2) - R
                        if z > z_max:
                            z_max = z

        Z[lattice_i] = z_max
    return Z

def detect_surface_itim(particle_positions, radii, R, box_l, grid_spacing=None, grid_bottom=None, pbc=True):
    """
    Run ITIM once and return BOTH the sphere-drop height map and the interfacial
    particle indices, reusing a single cell-list / lattice setup.

    Parameters:
        particle_positions : (N, 3) array of particle positions
        radii : (N,) array of particle radii
        R : Sphere (probe) radius
        box_l : box dimensions [Lx, Ly, Lz]
        grid_spacing : Spacing between drop points in x-y plane (mutually exclusive
                       with passing a 2D grid_bottom array)
        grid_bottom : bottom of the grid (lowest limit of surface)
        pbc : if True, periodic in X and Y (never in Z)

    Returns:
        X, Y, Z : grid coordinates and surface heights (each shaped (nx, ny))
        part_idxs : int64 indices of the interfacial particles into
                    particle_positions (ITIM preserves input order, so
                    particle_positions[part_idxs] are the surface particles)
    """
    poss_tmp_path = tempfile_path(suffix=".npy")
    radii_tmp_path = tempfile_path(suffix=".npy")
    grid_bottom_tmp_path = None
    return_file_path = None
    np.save(poss_tmp_path, particle_positions)
    np.save(radii_tmp_path, radii)
    if grid_bottom is not None:
        grid_bottom_tmp_path = tempfile_path(suffix=".npy")
        np.save(grid_bottom_tmp_path, grid_bottom)
    args = (poss_tmp_path, radii_tmp_path, R, box_l, grid_spacing, grid_bottom_tmp_path, pbc)
    try:
        return_file_path = run_subprocess_in_module_mp(args, module=get_module_name(), worker=detect_surface_itim_worker.__name__)
        data = np.load(return_file_path)
        return data['X'], data['Y'], data['Z'], data['part_idxs']
    finally:
        for p in (poss_tmp_path, radii_tmp_path, grid_bottom_tmp_path, return_file_path):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass

def detect_surface_itim_worker(poss_path, radii_path, R, box_l, grid_spacing=None, grid_bottom_path=None, pbc=True):
    particle_positions = np.load(poss_path)
    radii = np.load(radii_path)
    grid_bottom = np.load(grid_bottom_path) if grid_bottom_path is not None else None

    box_l_special = [box_l[0], box_l[1], 1E6]
    poss_folded = fold_coord(particle_positions, box_dim=box_l_special)

    (X, Y), lattice_points, cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy, flat_bottom = _itim_numba_setup(
        poss_folded, radii, R, box_l, grid_spacing, grid_bottom, pbc=pbc)

    box_l0 = float(box_l[0])
    box_l1 = float(box_l[1])
    inv_box0 = 1.0 / box_l0
    inv_box1 = 1.0 / box_l1

    # Order matters: the particle-index kernel only READS flat_bottom as its starting
    # threshold, while the height kernel aliases Z = flat_bottom and writes it in place.
    # Run the index kernel first so it sees the original floor.
    part_idxs = compute_surface_particles_idx_itim_numba(
        poss_folded, radii, lattice_points,
        cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy,
        box_l0, box_l1, inv_box0, inv_box1, pbc,
        R, flat_bottom,
    )

    Z = compute_sphere_contact_heights_on_lattice_flat(
        poss_folded, radii, lattice_points,
        cell_starts, cell_particles, ncx, ncy, inv_csx, inv_csy,
        box_l0, box_l1, inv_box0, inv_box1, pbc,
        R, flat_bottom,
    ).reshape(X.shape)

    return_file_path = tempfile_path(suffix=".npz")
    np.savez(return_file_path, X=X, Y=Y, Z=Z, part_idxs=part_idxs)
    return return_file_path

def detect_surface_by_interpolation_of_surface_particles(surface_points, box_l, grid_spacing=None, grid_bottom=None, pbc=True, method='cubic'):
    surface_points_tmp_path = tempfile_path(suffix=".npy")
    grid_bottom_tmp_path = None
    return_file_path = None
    np.save(surface_points_tmp_path, surface_points)
    if grid_bottom is not None:
        grid_bottom_tmp_path = tempfile_path(suffix=".npy")
        np.save(grid_bottom_tmp_path, grid_bottom)
    args = (surface_points_tmp_path, grid_bottom_tmp_path, box_l, grid_spacing, pbc, method)
    try:
        return_file_path = run_subprocess_in_module_mp(args, module=get_module_name(), worker=detect_surface_by_interpolation_of_surface_particles_worker.__name__)
        data = np.load(return_file_path)
        return data['X'], data['Y'], data['Z']
    finally:
        for p in (surface_points_tmp_path, grid_bottom_tmp_path, return_file_path):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass

def detect_surface_by_interpolation_of_surface_particles_worker(surface_points_path, grid_bottom_path, box_l, grid_spacing=None, pbc=True, method='cubic'):
    surface_points = np.load(surface_points_path)
    grid_bottom = np.load(grid_bottom_path) if grid_bottom_path is not None else None

    grid_bottom, (nx, ny), (dx, dy) = _build_box_l_starting_grid(
        box_l, grid_spacing, grid_spacing, grid_bottom)
    
    X, Y = np.meshgrid(np.arange(nx) * dx, np.arange(ny) * dy, indexing='ij')

    if surface_points.shape[0] < 1:
        return_file_path = tempfile_path(suffix=".npz")
        np.savez(return_file_path, X=X, Y=Y, Z=grid_bottom)
        return return_file_path

    xs = surface_points[:, 0]
    ys = surface_points[:, 1]
    zs = surface_points[:, 2]

    if pbc:
        Lx, Ly = float(box_l[0]), float(box_l[1])
        edge_dist = max(Lx, Ly) / 4.0
        L = xs < edge_dist
        Rm = xs >= Lx - edge_dist
        B = ys < edge_dist
        T = ys >= Ly - edge_dist
        LB, RB, LT, RT = L & B, Rm & B, L & T, Rm & T
        xy = np.column_stack([
            np.concatenate([xs,
                xs[L] + Lx, xs[Rm] - Lx, xs[B],       xs[T],
                xs[LB] + Lx, xs[RB] - Lx, xs[LT] + Lx, xs[RT] - Lx]),
            np.concatenate([ys,
                ys[L],       ys[Rm],      ys[B] + Ly, ys[T] - Ly,
                ys[LB] + Ly, ys[RB] + Ly, ys[LT] - Ly, ys[RT] - Ly]),
        ])
        z = np.concatenate([zs,
            zs[L], zs[Rm], zs[B], zs[T],
            zs[LB], zs[RB], zs[LT], zs[RT]])
    else:
        xy = np.column_stack([xs, ys])
        z = zs.copy()
    del xs, ys, zs

    if method == 'cubic':
        interp = CloughTocher2DInterpolator(xy, z, fill_value=np.nan)
    elif method == 'linear':
        interp = LinearNDInterpolator(xy, z, fill_value=np.nan)
    else:
        raise ValueError(f"unknown interpolation method: {method!r}")
    del xy, z

    Z = np.asarray(interp(X, Y))
    Z = np.where(np.isnan(Z), grid_bottom, Z)
    Z = np.maximum(Z, grid_bottom)

    return_file_path = tempfile_path(suffix=".npz")
    np.savez(return_file_path, X=X, Y=Y, Z=Z)
    return return_file_path

def detect_surface_by_interpolation_worker(poss_tmp_path, grid_bottom_tmp_path, box_l, grid_spacing=None, known_threshold=0., pbc=True):
    particle_positions = np.load(poss_tmp_path)
    grid_bottom = np.load(grid_bottom_tmp_path) if grid_bottom_tmp_path is not None else None

    grid_bottom, (nx, ny), (grid_res_x, grid_res_y) = _build_box_l_starting_grid(box_l, grid_spacing, grid_spacing, grid_bottom)

    grid = _grid_max_numba(particle_positions,
                          grid_res_x, grid_res_y, grid_bottom.copy())

    del grid_bottom, particle_positions

    # Known points: skip meshgrid entirely, build (M,2) directly.
    ki, kj = np.nonzero(grid > known_threshold)
    ki = ki.astype(np.int32, copy=False)
    kj = kj.astype(np.int32, copy=False)
    vals = grid[ki, kj]
    del grid

    if pbc:
        h = int(max(nx, ny) / 4)
        L = ki < h; R = ki >= nx - h
        B = kj < h; T = kj >= ny - h
        LB, RB, LT, RT = L & B, R & B, L & T, R & T
        # central + 4 edges + 4 corners
        ki = np.concatenate([ki,
            ki[L] + nx, ki[R] - nx, ki[B],       ki[T],
            ki[LB] + nx, ki[RB] - nx, ki[LT] + nx, ki[RT] - nx])
        kj = np.concatenate([kj,
            kj[L],       kj[R],      kj[B] + ny, kj[T] - ny,
            kj[LB] + ny, kj[RB] + ny, kj[LT] - ny, kj[RT] - ny])
        vals = np.concatenate([vals,
            vals[L], vals[R], vals[B], vals[T],
            vals[LB], vals[RB], vals[LT], vals[RT]])

    pts = np.empty((ki.size, 2), dtype=np.float64)
    pts[:, 0] = ki * grid_res_x
    pts[:, 1] = kj * grid_res_y
    del ki, kj

    interp = CloughTocher2DInterpolator(pts, vals)
    del pts, vals

    X, Y = np.meshgrid(np.arange(nx)*grid_res_x, np.arange(ny)*grid_res_y, indexing='ij')
    Z = np.asarray(interp(X, Y))

    return_file_path = tempfile_path(suffix=".npz")
    np.savez(return_file_path,
             X=X,
             Y=Y,
             Z=Z)
    return return_file_path
def detect_surface_by_grid_interpolation(particle_positions, box_l, grid_spacing=None, grid_bottom=None, known_threshold=0., pbc=True):
    poss_tmp_path = tempfile_path(suffix=".npy")
    grid_bottom_tmp_path = tempfile_path(suffix=".npy") if grid_bottom is not None else None
    return_file_path = None
    np.save(poss_tmp_path, particle_positions)
    if grid_bottom_tmp_path is not None:
        assert grid_bottom is not None
        np.save(grid_bottom_tmp_path, grid_bottom)
    args = (poss_tmp_path, grid_bottom_tmp_path, box_l, grid_spacing, known_threshold, pbc)
    try:
        return_file_path = run_subprocess_in_module_mp(args, module=get_module_name(), worker=detect_surface_by_interpolation_worker.__name__)
        data = np.load(return_file_path)
        return (data['X'], data['Y'], data['Z'])
    finally:
        for p in (poss_tmp_path, grid_bottom_tmp_path, return_file_path):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass
#####

##### Volume with surface mapping #####
@njit(parallel=True, fastmath=FASTMATH)
def volume_on_rectangular_grid(top_surface_flat, bottom_surface_flat, cell_area):
    volume = 0.0
    for i in prange(top_surface_flat.shape[0]):
        volume += top_surface_flat[i] - bottom_surface_flat[i]
    return volume * cell_area
