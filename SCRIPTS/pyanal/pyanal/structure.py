import os

import numpy as np
import math
from scipy.spatial import KDTree
import igraph as ig

from numba import njit, prange

from .generic_functions import fold_coord, run_subprocess_in_module_mp, tempfile_path

FASTMATH = True

def get_module_name():
    return "pyanal.structure"

##### Energetic clustering / neighbor lists #####
# Shared dipolar pair-energy criterion: e_ij < cutoff_E AND r_ij < cutoff_r.
# - cluster_by_E_dip: connected components via edges-from-CSR + igraph.
# - get_energetic_neighbors_csr: symmetric CSR neighbor list (offsets, idx, energies).
# - get_energetic_neighbors: ^^ formated as a python list of neighbors for each particle.
# - get_energetic_neighbors_of_i(i, ...): O(N) single-particle path.
# All three share _dipolar_pair_e_ij and dispatch to brute O(N^2) or a 3D cell
# list based on cutoff_r vs box size.

def cluster_E_anal_worker(poss_tmp_path, dips_tmp_path, mask=None, cutoff_E=0.0, cutoff_r=2**(1/6)*1.3, box_l=[int(1E6),int(1E6),int(1E6)], pbc=True):
    """
    Memory-efficient cluster analysis. Processes large arrays incrementally.
    """
    poss = np.ascontiguousarray(fold_coord(np.load(poss_tmp_path), box_l), dtype=np.float64)
    dips = np.ascontiguousarray(fold_coord(np.load(dips_tmp_path), box_l), dtype=np.float64)
    box_l = np.asarray(box_l, dtype=np.float64)

    if mask is not None:
        poss = poss[mask]
        dips = dips[mask]

    ids_internal = np.arange(len(poss))

    if not pbc:
        safe_layer_set = set(ids_internal[(poss[:,2] > cutoff_r) & (poss[:,2] < (box_l[2] - cutoff_r))])
    else:
        safe_layer_set = set(ids_internal)

    neighbors_E = get_energetic_neighbors(poss, dips, box_l, cutoff_E, cutoff_r, pbc)

    num_neighbors = np.array([len(nlist) for i, (nlist, _) in enumerate(neighbors_E) if i in safe_layer_set])

    clusters_E = labels_to_components(cluster_by_E_dip(poss, dips, box_l, cutoff_E, cutoff_r, pbc), idx=(np.fromiter(sorted(safe_layer_set), dtype=np.int64) if not pbc else None))

    no_clusters = len(clusters_E)

    cluster_sizes = np.array([cluster.size for cluster in clusters_E])

    return_file_path = tempfile_path(suffix=".npz")
    np.savez(return_file_path,
             num_neighbors=num_neighbors,
             no_clusters=no_clusters,
             cluster_sizes=np.asarray(cluster_sizes, dtype=np.int32)
             )

    return return_file_path

@njit(inline='always', fastmath=FASTMATH, cache=True)
def _dipolar_pair_e_ij(ri0, ri1, ri2, mui0, mui1, mui2,
                       rj0, rj1, rj2, muj0, muj1, muj2,
                       bx, by, bz, ibx, iby, ibz, pbc):
    r0 = rj0 - ri0
    r1 = rj1 - ri1
    r2 = rj2 - ri2
    if pbc:
        r0 -= bx * math.floor(r0 * ibx + 0.5)
        r1 -= by * math.floor(r1 * iby + 0.5)
        r2 -= bz * math.floor(r2 * ibz + 0.5)
    r2sq = r0*r0 + r1*r1 + r2*r2
    if r2sq < 1e-6:
        return 0.0, 0.0
    inv_r2 = 1.0 / r2sq
    inv_r3 = inv_r2 * math.sqrt(inv_r2)
    inv_r5 = inv_r3 * inv_r2
    mu_dot = mui0*muj0 + mui1*muj1 + mui2*muj2
    mur_i = mui0*r0 + mui1*r1 + mui2*r2
    mur_j = muj0*r0 + muj1*r1 + muj2*r2
    return mu_dot * inv_r3 - 3.0 * mur_i * mur_j * inv_r5, r2sq


@njit(inline='always', cache=True)
def _unflatten_cell(ci, ncy, ncz):
    cz = ci % ncz
    cy = (ci // ncz) % ncy
    cx = ci // (ncy * ncz)
    return cx, cy, cz


@njit(inline='always', cache=True)
def _neighbor_cell_coord(c0, dc, nc, pbc):
    """Returns wrapped neighbor coord, or -1 if out of range (pbc=False)."""
    if pbc:
        return (c0 + dc) % nc
    c = c0 + dc
    if c < 0 or c >= nc:
        return -1
    return c


@njit(cache=True)
def _build_cell_list_3d(x, y, z, ncx, ncy, ncz, inv_csx, inv_csy, inv_csz, pbc):
    n = x.shape[0]
    n_cells = ncx * ncy * ncz
    counts = np.zeros(n_cells, dtype=np.int32)
    p_cell_indice = np.empty(n, dtype=np.int32)
    for p in range(n):
        cx = int(x[p] * inv_csx); cy = int(y[p] * inv_csy); cz = int(z[p] * inv_csz)
        if pbc:
            cx %= ncx; cy %= ncy; cz %= ncz
        else:
            if cx < 0: cx = 0
            elif cx >= ncx: cx = ncx - 1
            if cy < 0: cy = 0
            elif cy >= ncy: cy = ncy - 1
            if cz < 0: cz = 0
            elif cz >= ncz: cz = ncz - 1
        ci = (cx * ncy + cy) * ncz + cz
        counts[ci] += 1
        p_cell_indice[p] = ci
    cell_starts = np.empty(n_cells + 1, dtype=np.int32)
    cell_starts[0] = 0
    for k in range(n_cells):
        cell_starts[k+1] = cell_starts[k] + counts[k]
    cell_particles = np.empty(n, dtype=np.int32)
    write_idx = cell_starts.copy()
    for p in range(n):
        ci = p_cell_indice[p]
        cell_particles[write_idx[ci]] = p
        write_idx[ci] += 1
    return cell_starts, cell_particles, p_cell_indice


@njit(parallel=True, fastmath=FASTMATH, cache=True)
def _emit_neighbors_csr_brute(positions, dipoles, box_l,
                              cutoff_E, cutoff_r, pbc, symmetric):
    N = positions.shape[0]
    bx = box_l[0]; by = box_l[1]; bz = box_l[2]
    ibx = 1.0 / bx; iby = 1.0 / by; ibz = 1.0 / bz
    cutoff_r2 = cutoff_r * cutoff_r

    counts = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        ri0 = positions[i, 0]; ri1 = positions[i, 1]; ri2 = positions[i, 2]
        mui0 = dipoles[i, 0]; mui1 = dipoles[i, 1]; mui2 = dipoles[i, 2]
        c = 0
        for j in range(N):
            if j == i:
                continue
            if (not symmetric) and j < i:
                continue
            e_ij, r2sq = _dipolar_pair_e_ij(
                ri0, ri1, ri2, mui0, mui1, mui2,
                positions[j, 0], positions[j, 1], positions[j, 2],
                dipoles[j, 0], dipoles[j, 1], dipoles[j, 2],
                bx, by, bz, ibx, iby, ibz, pbc)
            if r2sq < 1e-6 or r2sq > cutoff_r2:
                continue
            if e_ij < cutoff_E:
                c += 1
        counts[i] = c

    offsets = np.empty(N + 1, dtype=np.int32)
    offsets[0] = 0
    for i in range(N):
        offsets[i+1] = offsets[i] + counts[i]
    total = offsets[N]
    flat_neighbors = np.empty(total, dtype=np.int32)
    flat_energies = np.empty(total, dtype=np.float64)

    for i in prange(N):
        ri0 = positions[i, 0]; ri1 = positions[i, 1]; ri2 = positions[i, 2]
        mui0 = dipoles[i, 0]; mui1 = dipoles[i, 1]; mui2 = dipoles[i, 2]
        w = offsets[i]
        for j in range(N):
            if j == i:
                continue
            if (not symmetric) and j < i:
                continue
            e_ij, r2sq = _dipolar_pair_e_ij(
                ri0, ri1, ri2, mui0, mui1, mui2,
                positions[j, 0], positions[j, 1], positions[j, 2],
                dipoles[j, 0], dipoles[j, 1], dipoles[j, 2],
                bx, by, bz, ibx, iby, ibz, pbc)
            if r2sq < 1e-6 or r2sq > cutoff_r2:
                continue
            if e_ij < cutoff_E:
                flat_neighbors[w] = j
                flat_energies[w] = e_ij
                w += 1

    return offsets, flat_neighbors, flat_energies


@njit(parallel=True, fastmath=FASTMATH, cache=True)
def _emit_neighbors_csr_cells(positions, dipoles, box_l,
                              cell_starts, cell_particles, p_cell_indice,
                              ncx, ncy, ncz,
                              cutoff_E, cutoff_r, pbc, symmetric):
    N = positions.shape[0]
    bx = box_l[0]; by = box_l[1]; bz = box_l[2]
    ibx = 1.0 / bx; iby = 1.0 / by; ibz = 1.0 / bz
    cutoff_r2 = cutoff_r * cutoff_r

    counts = np.zeros(N, dtype=np.int32)
    for i in prange(N):
        ri0 = positions[i, 0]; ri1 = positions[i, 1]; ri2 = positions[i, 2]
        mui0 = dipoles[i, 0]; mui1 = dipoles[i, 1]; mui2 = dipoles[i, 2]
        cx0, cy0, cz0 = _unflatten_cell(p_cell_indice[i], ncy, ncz)
        c = 0
        for dx in range(-1, 2):
            cx = _neighbor_cell_coord(cx0, dx, ncx, pbc)
            if cx < 0:
                continue
            for dy in range(-1, 2):
                cy = _neighbor_cell_coord(cy0, dy, ncy, pbc)
                if cy < 0:
                    continue
                for dz in range(-1, 2):
                    cz = _neighbor_cell_coord(cz0, dz, ncz, pbc)
                    if cz < 0:
                        continue
                    ci = (cx * ncy + cy) * ncz + cz
                    for k in range(cell_starts[ci], cell_starts[ci+1]):
                        j = cell_particles[k]
                        if j == i:
                            continue
                        if (not symmetric) and j < i:
                            continue
                        e_ij, r2sq = _dipolar_pair_e_ij(
                            ri0, ri1, ri2, mui0, mui1, mui2,
                            positions[j, 0], positions[j, 1], positions[j, 2],
                            dipoles[j, 0], dipoles[j, 1], dipoles[j, 2],
                            bx, by, bz, ibx, iby, ibz, pbc)
                        if r2sq < 1e-6 or r2sq > cutoff_r2:
                            continue
                        if e_ij < cutoff_E:
                            c += 1
        counts[i] = c

    offsets = np.empty(N + 1, dtype=np.int32)
    offsets[0] = 0
    for i in range(N):
        offsets[i+1] = offsets[i] + counts[i]
    total = offsets[N]
    flat_neighbors = np.empty(total, dtype=np.int32)
    flat_energies = np.empty(total, dtype=np.float64)

    for i in prange(N):
        ri0 = positions[i, 0]; ri1 = positions[i, 1]; ri2 = positions[i, 2]
        mui0 = dipoles[i, 0]; mui1 = dipoles[i, 1]; mui2 = dipoles[i, 2]
        cx0, cy0, cz0 = _unflatten_cell(p_cell_indice[i], ncy, ncz)
        w = offsets[i]
        for dx in range(-1, 2):
            cx = _neighbor_cell_coord(cx0, dx, ncx, pbc)
            if cx < 0:
                continue
            for dy in range(-1, 2):
                cy = _neighbor_cell_coord(cy0, dy, ncy, pbc)
                if cy < 0:
                    continue
                for dz in range(-1, 2):
                    cz = _neighbor_cell_coord(cz0, dz, ncz, pbc)
                    if cz < 0:
                        continue
                    ci = (cx * ncy + cy) * ncz + cz
                    for k in range(cell_starts[ci], cell_starts[ci+1]):
                        j = cell_particles[k]
                        if j == i:
                            continue
                        if (not symmetric) and j < i:
                            continue
                        e_ij, r2sq = _dipolar_pair_e_ij(
                            ri0, ri1, ri2, mui0, mui1, mui2,
                            positions[j, 0], positions[j, 1], positions[j, 2],
                            dipoles[j, 0], dipoles[j, 1], dipoles[j, 2],
                            bx, by, bz, ibx, iby, ibz, pbc)
                        if r2sq < 1e-6 or r2sq > cutoff_r2:
                            continue
                        if e_ij < cutoff_E:
                            flat_neighbors[w] = j
                            flat_energies[w] = e_ij
                            w += 1

    return offsets, flat_neighbors, flat_energies


@njit(fastmath=FASTMATH, cache=True)
def _neighbors_for_i_brute(i, positions, dipoles, box_l, cutoff_E, cutoff_r, pbc):
    N = positions.shape[0]
    bx = box_l[0]; by = box_l[1]; bz = box_l[2]
    ibx = 1.0 / bx; iby = 1.0 / by; ibz = 1.0 / bz
    cutoff_r2 = cutoff_r * cutoff_r

    ri0 = positions[i, 0]; ri1 = positions[i, 1]; ri2 = positions[i, 2]
    mui0 = dipoles[i, 0]; mui1 = dipoles[i, 1]; mui2 = dipoles[i, 2]

    scratch_idx = np.empty(N, dtype=np.int32)
    scratch_e = np.empty(N, dtype=np.float64)
    c = 0
    for j in range(N):
        if j == i:
            continue
        e_ij, r2sq = _dipolar_pair_e_ij(
            ri0, ri1, ri2, mui0, mui1, mui2,
            positions[j, 0], positions[j, 1], positions[j, 2],
            dipoles[j, 0], dipoles[j, 1], dipoles[j, 2],
            bx, by, bz, ibx, iby, ibz, pbc)
        if r2sq < 1e-6 or r2sq > cutoff_r2:
            continue
        if e_ij < cutoff_E:
            scratch_idx[c] = j
            scratch_e[c] = e_ij
            c += 1
    return scratch_idx[:c].copy(), scratch_e[:c].copy()


@njit(fastmath=FASTMATH, cache=True)
def _neighbors_for_i_cells(i, positions, dipoles, box_l,
                           cell_starts, cell_particles, p_cell_indice,
                           ncx, ncy, ncz,
                           cutoff_E, cutoff_r, pbc):
    bx = box_l[0]; by = box_l[1]; bz = box_l[2]
    ibx = 1.0 / bx; iby = 1.0 / by; ibz = 1.0 / bz
    cutoff_r2 = cutoff_r * cutoff_r

    ri0 = positions[i, 0]; ri1 = positions[i, 1]; ri2 = positions[i, 2]
    mui0 = dipoles[i, 0]; mui1 = dipoles[i, 1]; mui2 = dipoles[i, 2]
    cx0, cy0, cz0 = _unflatten_cell(p_cell_indice[i], ncy, ncz)

    N = positions.shape[0]
    scratch_idx = np.empty(N, dtype=np.int32)
    scratch_e = np.empty(N, dtype=np.float64)
    c = 0
    for dx in range(-1, 2):
        cx = _neighbor_cell_coord(cx0, dx, ncx, pbc)
        if cx < 0:
            continue
        for dy in range(-1, 2):
            cy = _neighbor_cell_coord(cy0, dy, ncy, pbc)
            if cy < 0:
                continue
            for dz in range(-1, 2):
                cz = _neighbor_cell_coord(cz0, dz, ncz, pbc)
                if cz < 0:
                    continue
                ci = (cx * ncy + cy) * ncz + cz
                for k in range(cell_starts[ci], cell_starts[ci+1]):
                    j = cell_particles[k]
                    if j == i:
                        continue
                    e_ij, r2sq = _dipolar_pair_e_ij(
                        ri0, ri1, ri2, mui0, mui1, mui2,
                        positions[j, 0], positions[j, 1], positions[j, 2],
                        dipoles[j, 0], dipoles[j, 1], dipoles[j, 2],
                        bx, by, bz, ibx, iby, ibz, pbc)
                    if r2sq < 1e-6 or r2sq > cutoff_r2:
                        continue
                    if e_ij < cutoff_E:
                        scratch_idx[c] = j
                        scratch_e[c] = e_ij
                        c += 1
    return scratch_idx[:c].copy(), scratch_e[:c].copy()


def _should_use_cell_list(box_l, cutoff_r):
    # Need >= 3 cells per axis for the 27-cell scan to capture all in-range
    # pairs without redundant cell visits.
    return cutoff_r > 0 and all(box_l[k] / cutoff_r >= 3.0 for k in range(3))


def _build_cell_list_for(positions, box_l, cutoff_r, pbc):
    ncx = max(1, int(box_l[0] / cutoff_r))
    ncy = max(1, int(box_l[1] / cutoff_r))
    ncz = max(1, int(box_l[2] / cutoff_r))
    inv_csx = ncx / box_l[0]
    inv_csy = ncy / box_l[1]
    inv_csz = ncz / box_l[2]
    x = np.ascontiguousarray(positions[:, 0])
    y = np.ascontiguousarray(positions[:, 1])
    z = np.ascontiguousarray(positions[:, 2])
    cell_starts, cell_particles, p_cell_indice = _build_cell_list_3d(
        x, y, z, ncx, ncy, ncz, inv_csx, inv_csy, inv_csz, pbc)
    return ncx, ncy, ncz, cell_starts, cell_particles, p_cell_indice


def _emit_neighbors_csr(positions, dipoles, box_l, cutoff_E, cutoff_r, pbc, symmetric):
    if _should_use_cell_list(box_l, cutoff_r):
        ncx, ncy, ncz, cell_starts, cell_particles, p_cell_indice = \
            _build_cell_list_for(positions, box_l, cutoff_r, pbc)
        return _emit_neighbors_csr_cells(
            positions, dipoles, box_l,
            cell_starts, cell_particles, p_cell_indice,
            ncx, ncy, ncz,
            cutoff_E, cutoff_r, pbc, symmetric)
    return _emit_neighbors_csr_brute(
        positions, dipoles, box_l, cutoff_E, cutoff_r, pbc, symmetric)


def get_energetic_neighbors(positions, dipoles, box_l, cutoff_E=0., cutoff_r=1E6, pbc=True):
    """Symmetric per-particle neighbor list for the dipolar pair-energy criterion.

    Returns a list of length N where element ``i`` is the tuple ``(idxs, energies)``:
      ``idxs`` is an int32 array of particle ``i``'s neighbors and ``energies`` the corresponding pair energies (float64).

    Symmetric: each pair ``(i, j)`` appears once in ``i``'s list and once in ``j``'s.
    For the flat-array (CSR) form, use :func:`get_energetic_neighbors_csr`.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    dipoles = np.ascontiguousarray(dipoles, dtype=np.float64)
    box_l = np.asarray(box_l, dtype=np.float64)
    return neighbors_csr_to_list(*_emit_neighbors_csr(positions, dipoles, box_l,
                               float(cutoff_E), float(cutoff_r), pbc, True))


def get_energetic_neighbors_csr(positions, dipoles, box_l,
                                    cutoff_E=0., cutoff_r=1E6, pbc=True):
    """Flat-CSR variant of :func:`get_energetic_neighbors` (perf / numba interop).

    Returns
    -------
    offsets : int32[N+1]
    flat_neighbors : int32[total]
    flat_energies  : float64[total]
        Neighbors of particle ``i``: ``flat_neighbors[offsets[i]:offsets[i+1]]``;
        same slice in ``flat_energies`` is the corresponding pair energies.
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    dipoles = np.ascontiguousarray(dipoles, dtype=np.float64)
    box_l = np.asarray(box_l, dtype=np.float64)
    return _emit_neighbors_csr(positions, dipoles, box_l,
                               float(cutoff_E), float(cutoff_r), pbc, True)


def get_energetic_neighbors_of_i(i, positions, dipoles, box_l, cutoff_E=0., cutoff_r=1E6, pbc=True):
    """Neighbors of a single particle i; right-sized (neighbor_idxs, pair_energies)."""
    assert isinstance(i, int), f"i must be integer: {i}"
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    dipoles = np.ascontiguousarray(dipoles, dtype=np.float64)
    box_l = np.asarray(box_l, dtype=np.float64)
    if _should_use_cell_list(box_l, cutoff_r):
        ncx, ncy, ncz, cell_starts, cell_particles, p_cell_indice = \
            _build_cell_list_for(positions, box_l, cutoff_r, pbc)
        return _neighbors_for_i_cells(
            i, positions, dipoles, box_l,
            cell_starts, cell_particles, p_cell_indice,
            ncx, ncy, ncz,
            float(cutoff_E), float(cutoff_r), pbc)
    return _neighbors_for_i_brute(
        i, positions, dipoles, box_l,
        float(cutoff_E), float(cutoff_r), pbc)


def cluster_by_E_dip(positions, dipoles, box_l, cutoff_E=0., cutoff_r=1E6, pbc=True):
    """
    Cluster particles using a dipolar pair-energy criterion.

    Particles are connected if their pair interaction satisfies the energy
    threshold.

    Parameters
    ----------
    positions : (N, 3) array_like, float
        Particle positions.
    dipoles : (N, 3) array_like, float
        Dipole vectors associated with each particle.
    box_l : array_like, shape (3,)
        Simulation box lengths.
    cutoff_E : float, optional
        Energy cutoff used to determine whether an edge is created.
        Default is 0.0.
    cutoff_r : float, optional
        Maximum interaction distance cutoff.
        Default is 1e6.
    pbc : bool, optional
        If True, periodic boundary conditions are applied.

    Returns
    -------
    labels : (N,) ndarray of int32
        Cluster label for each particle (connected component membership).
    """
    positions = np.ascontiguousarray(positions, dtype=np.float64)
    dipoles = np.ascontiguousarray(dipoles, dtype=np.float64)
    box_l = np.asarray(box_l, dtype=np.float64)
    N = positions.shape[0]
    if N == 0:
        return np.empty(0, dtype=np.int32)
    # Upper-triangular CSR: one (i, j) per qualifying pair.
    offsets, flat_neighbors, _ = _emit_neighbors_csr(
        positions, dipoles, box_l,
        float(cutoff_E), float(cutoff_r), pbc, False)
    src = np.repeat(np.arange(N, dtype=np.int32), np.diff(offsets))
    edges = list(zip(src.tolist(), flat_neighbors.tolist()))
    g = ig.Graph(n=N, edges=edges)
    g.simplify()
    return np.array(g.components().membership, dtype=np.int32)


def neighbors_csr_to_list(offsets, flat_neighbors, flat_energies=None):
    """Convert CSR neighbor list to a per-particle list-of-arrays.

    Without energies: list of int32 arrays. With: list of (idxs, energies) tuples.
    """
    N = len(offsets) - 1
    if flat_energies is None:
        return [flat_neighbors[offsets[i]:offsets[i+1]] for i in range(N)]
    return [(flat_neighbors[offsets[i]:offsets[i+1]],
             flat_energies[offsets[i]:offsets[i+1]]) for i in range(N)]

def labels_to_components(labels, idx=None):
    """
    Convert integer cluster labels (igraph-style membership array)
    into list of index arrays.

    Parameters
    ----------
    labels : (N,) array_like of int
        Cluster label per node.

    Returns
    -------
    list of ndarray
        Each entry contains node indices belonging to one cluster.
    """
    labels = np.asarray(labels)
    if labels.size == 0: return []

    if idx is not None:
        idx = np.asarray(idx, dtype=np.int32)
        labels = labels[idx]
        if labels.size == 0: return []
        inv_map = idx
    else:
        inv_map = None

    uniq, inv = np.unique(labels, return_inverse=True)
    comps = [[] for _ in range(uniq.size)]

    if inv_map is None:
        for i, c in enumerate(inv): comps[c].append(i)
    else:
        for i, c in enumerate(inv): comps[c].append(inv_map[i])

    return [np.asarray(c, dtype=np.int32) for c in comps]
#####

##### Cluster analysis #####
@njit(fastmath=FASTMATH, cache=True)
def _pair_distance_histogram(positions, box_l, pbc, bin_width, nbins):
    """O(N^2)-time, O(nbins)-memory histogram of all i<j pairwise (min-image) distances.

    Bins are uniform of width ``bin_width`` starting at 0 (i.e. edges
    ``np.arange(0, nbins*bin_width + bin_width, bin_width)``). Distances at or beyond
    the top edge are dropped, matching ``np.histogram`` with a bounded ``bins`` range.
    """
    N = positions.shape[0]
    bx = box_l[0]; by = box_l[1]; bz = box_l[2]
    ibx = 1.0 / bx; iby = 1.0 / by; ibz = 1.0 / bz
    inv_w = 1.0 / bin_width
    counts = np.zeros(nbins, dtype=np.int64)
    for i in range(N):
        ri0 = positions[i, 0]; ri1 = positions[i, 1]; ri2 = positions[i, 2]
        for j in range(i + 1, N):
            r0 = positions[j, 0] - ri0
            r1 = positions[j, 1] - ri1
            r2 = positions[j, 2] - ri2
            if pbc:
                r0 -= bx * math.floor(r0 * ibx + 0.5)
                r1 -= by * math.floor(r1 * iby + 0.5)
                r2 -= bz * math.floor(r2 * ibz + 0.5)
            r2sq = r0*r0 + r1*r1 + r2*r2
            if r2sq < 1e-12:
                continue
            b = int(math.sqrt(r2sq) * inv_w)
            if 0 <= b < nbins:
                counts[b] += 1
    return counts


def cluster_anal_worker(poss_tmp_path, dips_tmp_path, mask=None, cutoff=2**(1/6)*1.3, box_l=[int(1E6),int(1E6),int(1E6)], pbc=True, dist_bins=None):
    """
    Memory-efficient cluster analysis. Processes large arrays incrementally.

    Pairwise distances are reduced to a histogram (``dist_bins`` edges, uniform width)
    inside this worker -- the raw O(N^2) distance list is never materialized.
    """
    poss = np.load(poss_tmp_path)
    dips = np.load(dips_tmp_path)

    if mask is not None:
        poss = poss[mask]
        dips = dips[mask]

    ids_internal = np.arange(len(poss))

    if not pbc:
        safe_layer_set = set(ids_internal[(poss[:,2] > cutoff) & (poss[:,2] < (box_l[2] - cutoff))])
    else:
        safe_layer_set = set(ids_internal)

    n_part = len(poss)

    pos_folded = fold_coord(poss, box_dim=box_l)

    tree = KDTree(pos_folded, boxsize=box_l)
    neighbors = tree.query_ball_point(pos_folded, cutoff, return_sorted=True, workers=-1)

    num_neighbors = np.array([len(nlist) - 1 for i, nlist in enumerate(neighbors) if i in safe_layer_set])

    # Histogram all pairwise distances in O(nbins) memory (no full distance matrix).
    if dist_bins is None:
        dist_bins = np.arange(0, np.linalg.norm(box_l) + 1e-4, 0.5)
    dist_bin_edges = np.ascontiguousarray(dist_bins, dtype=np.float64)
    bin_width = float(dist_bin_edges[1] - dist_bin_edges[0])
    nbins = int(len(dist_bin_edges) - 1)
    dist_hist = _pair_distance_histogram(
        np.ascontiguousarray(pos_folded, dtype=np.float64),
        np.ascontiguousarray(box_l, dtype=np.float64), pbc, bin_width, nbins)
    if not pbc:
        print("WARNING: pbc not implemented for distances in cluster_anal")

    # neighbours cosines
    cosines= []
    cosine_avg_per_neighbourhood= []
    for i, nlist in enumerate(neighbors):
        if i not in safe_layer_set:
            continue
        nlist = [j for j in nlist if j != i]
        if len(nlist) < 1:
            continue
        dip_i = dips[i]
        director_i = dip_i / np.linalg.norm(dip_i)
        dip_j_array = dips[nlist]
        director_j_array = dip_j_array / np.linalg.norm(dip_j_array, axis=-1, keepdims=True)
        dot_products = np.dot(director_i, director_j_array.T)
        cosines.extend(dot_products)
        cosine_avg_per_neighbourhood.append(np.mean(dot_products))

    cosines = np.asarray(cosines, dtype=np.float32)  # Use float32 to save memory
    cosine_avg_per_neighbourhood = np.asarray(cosine_avg_per_neighbourhood, dtype=np.float32)

    # Build graph: nodes = particles, edges = neighbour pairs. Connected-component
    # sizes only -- no per-subgraph object materialization.
    if not pbc:
        print("WARNING: pbc not implemented for distances in cluster_anal graphs")
    edges = [(i, j) for i, nlist in enumerate(neighbors) for j in nlist if j > i]

    graph = ig.Graph(n=n_part, edges=edges)
    graph.simplify()
    comps = graph.connected_components()
    no_clusters = len(comps)
    cluster_sizes = comps.sizes()

    return_file_path = tempfile_path(suffix=".npz")
    np.savez(return_file_path,
             num_neighbors=num_neighbors,
             dist_hist=dist_hist,
             dist_bin_edges=dist_bin_edges,
             cosines=cosines,
             cosine_avg_per_neighbourhood=cosine_avg_per_neighbourhood,
             no_clusters=no_clusters,
             cluster_sizes=np.asarray(cluster_sizes, dtype=np.int64))

    return return_file_path
#####

##### BOP #####
def _steinhardt_factors(lmax):
    """Associated-Legendre recurrence coefficients for the Steinhardt harmonics.

    Port of pyscal3 ``calculate_factors`` (sh.cpp). Returns ``(alm, blm, dl, el)``;
    pyscal's ``clm`` is unused by ``calculate_plm`` and omitted.
    """
    alm = np.zeros((lmax + 1, lmax + 1))
    blm = np.zeros((lmax + 1, lmax + 1))
    dl = np.zeros(lmax + 1)
    el = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        temp1 = 4.0 * l * l - 1.0
        temp2 = l * l - 2.0 * l + 1.0
        for m in range(l - 1):  # m in 0 .. l-2
            alm[l, m] = np.sqrt(temp1 / (l * l - m * m))
            blm[l, m] = -np.sqrt((temp2 - m * m) / (4.0 * temp2 - 1.0))
    for l in range(2, lmax + 1):
        dl[l] = np.sqrt(2.0 * (l - 1) + 3.0)
        el[l] = np.sqrt(1.0 + 0.5 / l)
    return alm, blm, dl, el


@njit(fastmath=FASTMATH, inline='always')
def _accumulate_ylm(qlm_real, qlm_imag, i, costheta, sintheta, cosphi, sinphi,
                    lmax, l_values, col_start, alm, blm, dl, el, plm):
    """Accumulate Ylm(theta, phi) into row ``i`` of ``(qlm_real, qlm_imag)`` for the
    requested ``l_values`` (CSR layout via ``col_start``), using ``plm`` as scratch.

    pyscal3 recurrence (calculate_plm / calculate_ylm); values are the standard complex
    spherical harmonics Y_l^m with Condon-Shortley phase. Single source of truth shared
    by ``_averaged_steinhardt_csr`` and ``steinhardt_ylm`` (forced-inlined, so the
    kernel hot path is unchanged). The -m column holds Y_l^{-m} = (-1)^m conj(Y_l^m).
    """
    nl = l_values.shape[0]
    zv = 1.0 / math.sqrt(2.0)
    a0 = math.sqrt(0.5 / math.pi)
    a2 = math.sqrt(3.0)
    a3 = math.sqrt(1.5)
    # --- calculate_plm ---
    plm[0, 0] = a0
    diag = a0
    if lmax >= 1:
        plm[1, 0] = a0 * costheta * a2
        diag = a0 * sintheta * (-a3)
        plm[1, 1] = diag
        for l in range(2, lmax + 1):
            for m in range(l - 1):
                plm[l, m] = alm[l, m] * (costheta * plm[l - 1, m]
                                         + blm[l, m] * plm[l - 2, m])
            plm[l, l - 1] = dl[l] * costheta * diag
            diag = diag * (-el[l] * sintheta)
            plm[l, l] = diag
    # --- accumulate Ylm into qlm row (calculate_ylm) ---
    for li in range(nl):                       # m = 0
        l = l_values[li]
        qlm_real[i, col_start[l] + l] += plm[l, 0] * zv
    two_cosphi = 2.0 * cosphi
    cosi = 1.0; cosf = cosphi
    sini = 0.0; sinf = -sinphi
    li_start = 0
    for m in range(1, lmax + 1):               # m >= 1
        ci = two_cosphi * cosi - cosf
        si = two_cosphi * sini - sinf
        sinf = sini; sini = si
        cosf = cosi; cosi = ci
        sign = 1.0 if (m % 2 == 0) else -1.0   # (-1)^m for the -m index
        while li_start < nl and l_values[li_start] < m:
            li_start += 1                      # l_values sorted: drop l < m once
        for li in range(li_start, nl):
            l = l_values[li]
            cs = col_start[l]
            fa = plm[l, m] * ci * zv
            fb = plm[l, m] * si * zv
            qlm_real[i, cs + l + m] += fa
            qlm_imag[i, cs + l + m] += fb
            qlm_real[i, cs + l - m] += fa * sign
            qlm_imag[i, cs + l - m] -= fb * sign   # Y_l^{-m} = (-1)^m conj(Y_l^m)


@njit(parallel=True, fastmath=FASTMATH, cache=True)
def _averaged_steinhardt_csr(positions, offsets, flat_neighbors, box_l, pbc,
                             l_values, col_start, S, lmax, alm, blm, dl, el):
    """Neighbor-averaged Steinhardt ``q_l`` over a CSR neighbor list (no pyscal).

    Mirrors pyscal3 exactly: pass 1 builds per-atom ``qlm = (1/Nb) Σ_nbrs Ylm``; pass 2
    forms ``avg_qlm = (qlm_i + Σ_nbrs qlm_j)/(Nb+1)`` then ``q_l = sqrt(4π/(2l+1) Σ_m |avg_qlm|²)``.
    Spherical harmonics use pyscal's recurrence (calculate_plm / calculate_ylm); angles are
    taken directly from the displacement (costheta=dz/r, etc.), matching theta=acos(z/r),
    phi=atan2(y,x). Minimum-image only when ``pbc``.

    Zero-neighbor atoms -> NaN. FASTMATH-safe: NaN is only ever written as an explicit
    constant. The neighbor list is symmetric, so every neighbor has >=1 neighbor and all
    arithmetic stays finite.
    """
    N = positions.shape[0]
    nl = l_values.shape[0]
    bx = box_l[0]; by = box_l[1]; bz = box_l[2]
    ibx = 1.0 / bx; iby = 1.0 / by; ibz = 1.0 / bz

    qlm_real = np.zeros((N, S))
    qlm_imag = np.zeros((N, S))

    # ---- Pass 1: per-atom qlm = (1/Nb) Σ_nbrs Ylm ----
    for i in prange(N):
        start = offsets[i]; end = offsets[i + 1]
        n = end - start
        if n == 0:
            continue
        xi = positions[i, 0]; yi = positions[i, 1]; zi = positions[i, 2]
        plm = np.empty((lmax + 1, lmax + 1))
        for k in range(start, end):
            j = flat_neighbors[k]
            dx = positions[j, 0] - xi
            dy = positions[j, 1] - yi
            dz = positions[j, 2] - zi
            if pbc:
                dx -= bx * math.floor(dx * ibx + 0.5)
                dy -= by * math.floor(dy * iby + 0.5)
                dz -= bz * math.floor(dz * ibz + 0.5)
            r = math.sqrt(dx * dx + dy * dy + dz * dz)
            costheta = dz / r
            rho = math.sqrt(dx * dx + dy * dy)
            sintheta = rho / r
            if rho > 1e-13:
                cosphi = dx / rho
                sinphi = dy / rho
            else:
                cosphi = 1.0
                sinphi = 0.0
            _accumulate_ylm(qlm_real, qlm_imag, i, costheta, sintheta, cosphi, sinphi,
                            lmax, l_values, col_start, alm, blm, dl, el, plm)
        inv = 1.0 / n
        for s in range(S):
            qlm_real[i, s] *= inv
            qlm_imag[i, s] *= inv

    # ---- Pass 2: average over self + neighbors, then q_l ----
    q_vals = np.empty((nl, N))
    norm = np.empty(nl)                                 # sqrt(4π/(2l+1)) per requested l
    for li in range(nl):
        norm[li] = math.sqrt(4.0 * math.pi / (2 * l_values[li] + 1))
    for i in prange(N):
        start = offsets[i]; end = offsets[i + 1]
        n = end - start
        if n == 0:
            for li in range(nl):
                q_vals[li, i] = np.nan
            continue
        avg_re = np.empty(S); avg_im = np.empty(S)
        for s in range(S):
            avg_re[s] = qlm_real[i, s]
            avg_im[s] = qlm_imag[i, s]
        for k in range(start, end):
            j = flat_neighbors[k]
            for s in range(S):
                avg_re[s] += qlm_real[j, s]
                avg_im[s] += qlm_imag[j, s]
        inv = 1.0 / (n + 1)                             # |avg_qlm| = (self+nbrs) / (n+1)
        for li in range(nl):
            cs = col_start[l_values[li]]
            summ = 0.0
            for idx in range(2 * l_values[li] + 1):
                re = avg_re[cs + idx]
                im = avg_im[cs + idx]
                summ += re * re + im * im
            q_vals[li, i] = norm[li] * inv * math.sqrt(summ)

    return q_vals


def steinhardt_ylm(theta, phi, lmax):
    """Standard complex spherical harmonics Y_l^m for a single direction, via the exact
    recurrence used by ``BOP_analysis`` (``_accumulate_ylm``).

    Returns a complex array ``Y`` of shape ``(lmax + 1, 2 * lmax + 1)`` with
    ``Y[l, lmax + m] = Y_l^m`` (entries with ``|m| > l`` are 0).
    """
    lmax = int(lmax)
    l_values = np.arange(lmax + 1, dtype=np.int64)
    col_start = np.empty(lmax + 1, dtype=np.int64)
    S = 0
    for l in range(lmax + 1):
        col_start[l] = S
        S += 2 * l + 1
    alm, blm, dl, el = _steinhardt_factors(lmax)
    qr = np.zeros((1, S)); qi = np.zeros((1, S))
    plm = np.empty((lmax + 1, lmax + 1))
    _accumulate_ylm(qr, qi, 0, math.cos(theta), math.sin(theta),
                    math.cos(phi), math.sin(phi),
                    lmax, l_values, col_start, alm, blm, dl, el, plm)
    Y = np.zeros((lmax + 1, 2 * lmax + 1), dtype=np.complex128)
    for l in range(lmax + 1):
        cs = col_start[l]
        for m in range(-l, l + 1):
            Y[l, lmax + m] = qr[0, cs + l + m] + 1j * qi[0, cs + l + m]
    return Y


def BOP_analysis(poss, dips, mask=None, substrate_z_threshhold=0.,
                 cutoff_E=0.0, cutoff_r=2**(1/6)*1.3,
                 l_range: list= [2,3,4,5,6,7,8],
                 box_l=[int(1E6), int(1E6), int(1E6)], pbc=True):
    """Neighbor-averaged Steinhardt bond-order parameters, computed in numba (no pyscal).

    Neighbors use the unified dipolar criterion ``r <= cutoff_r AND e_ij < cutoff_E``
    (``cutoff_E=+inf`` -> pure distance cutoff). Runs in-process. ``pbc`` selectable:
    True folds + uses minimum image; False uses raw displacements and includes every atom
    (boundary atoms are biased -- filter via ``mask`` if needed).

    Returns ``(l_range, q_vals, None)`` with ``q_vals`` of shape ``(len(l_range), n_atoms)``,
    float32; atoms with no neighbors are NaN (matches pyscal; use ``np.nanmean`` to aggregate).
    """
    poss = np.ascontiguousarray(poss, dtype=np.float64)
    dips = np.ascontiguousarray(dips, dtype=np.float64)
    if mask is not None:
        poss = poss[mask]
        dips = dips[mask]
    box_l = np.asarray(box_l, dtype=np.float64)

    pos = fold_coord(poss, box_dim=box_l) if pbc else poss
    pos = np.ascontiguousarray(pos - np.array([0.0, 0.0, substrate_z_threshhold]), dtype=np.float64)

    l_values = np.sort(np.asarray(list(l_range), dtype=np.int64))  # ascending: kernel assumes sorted
    lmax = int(l_values.max())
    col_start = np.full(lmax + 1, -1, dtype=np.int64)
    S = 0
    for l in l_values:
        col_start[l] = S
        S += 2 * int(l) + 1

    alm, blm, dl, el = _steinhardt_factors(lmax)

    offsets, flat_neighbors, _ = get_energetic_neighbors_csr(
        pos, dips, box_l, cutoff_E=cutoff_E, cutoff_r=cutoff_r, pbc=pbc)

    q_vals = _averaged_steinhardt_csr(
        pos, offsets, flat_neighbors, box_l, pbc,
        l_values, col_start, S, lmax, alm, blm, dl, el)

    return l_values.astype(np.int8), q_vals.astype(np.float32), None


_wig_initialized = False
def _ensure_wig_initialized():
    global _wig_initialized
    if not _wig_initialized:
        from pywigxjpf import wig_table_init, wig_temp_init
        wig_table_init(2*100, 3)  # type: ignore . 2*jmax, 3 for 3j symbols.
        wig_temp_init(2*100) # type: ignore
        _wig_initialized = True

def wigner_3j_wigxjpf(l, m1, m2, m3):
    from pywigxjpf import wig3jj
    _ensure_wig_initialized()
    try:
        return wig3jj(2*l, 2*l, 2*l, 2*m1, 2*m2, 2*m3) # type: ignore
    except Exception:
        return 0.0
#####


def fcc_lattice(radius, volume_sides, scaling_factor=1., max_points_per_side=100):
    """
    Generates a face-centered cubic (FCC) lattice of points within a cuboid volume. The function creates an FCC crystal structure where spheres of given radius are arranged such that they touch along the face diagonal of the unit lattice.

    Parameters
    ----------
    radius : float
        Radius of the spheres in the lattice.
    volume_side : iterable of float of size 3
        Length of the cuboid volume's sides.
    scaling_factor : float, optional
        Factor to scale the radius of the spheres. Default is 1.0.
    max_points_per_side : int, optional
        Maximum number of points allowed per dimension. Default is 100.
        If exceeded, lattice constant is increased.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) containing the coordinates of the lattice points,
        where N is the number of points in the FCC lattice.

    Notes
    -----
    - The lattice constant is calculated as 2*radius_scaled/sqrt(2), where
      radius_scaled = radius*scaling_factor.
    - If the number of points per side exceeds max_points_per_side, the lattice
      constant is gradually increased until the constraint is satisfied.
    - The function ensures the lattice fits within the given volume by removing
      the last row of points to avoid periodic boundary condition overlaps.
    - When the lattice constant is increased, a warning message is logged with
      the new value.
    """
    assert len(volume_sides)==3, "this metthod assumes volume_sides to be have len of 3"
    volume_sides = np.asarray(volume_sides)

    radius_scaled = radius*scaling_factor
    lattice_constant = 2 * radius_scaled / np.sqrt(2)
    while True:
        num_points = np.ceil(volume_sides / lattice_constant).astype(int)
        if (num_points <= max_points_per_side).all():
            break
        lattice_constant *= 1.1

    indices = [np.arange(num-1) for num in num_points ]
    x, y, z = np.meshgrid(indices[0], indices[1], indices[2], indexing='ij')
    sum_indices = x + y + z
    mask = sum_indices % 2 == 0
    lattice_points = np.column_stack(
        (x[mask], y[mask], z[mask])) * lattice_constant + np.ones(shape=3)*radius
    return lattice_points


##### Multiprocessing helper for multiple calls to cluster_anal #####
def cluster_anal(poss, dips, mask=None, cutoff=2**(1/6)*1.3, box_l=[int(1E6),int(1E6),int(1E6)], pbc=True, dist_bins=None):
    poss_tmp_path = tempfile_path(suffix=".npy")
    dips_tmp_path = tempfile_path(suffix=".npy")
    return_file_path = None
    np.save(poss_tmp_path, poss)
    np.save(dips_tmp_path, dips)

    args = (poss_tmp_path, dips_tmp_path, mask, cutoff, box_l, pbc, dist_bins)
    try:
        return_file_path = run_subprocess_in_module_mp(args, module=get_module_name(), worker=cluster_anal_worker.__name__)
        data = np.load(return_file_path, allow_pickle=True)
        return (data['num_neighbors'], data['dist_hist'], data['dist_bin_edges'],
                data['cosines'], data['cosine_avg_per_neighbourhood'],
                data['no_clusters'], data['cluster_sizes'])
    finally:
        for p in (poss_tmp_path, dips_tmp_path, return_file_path):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass

def cluster_E_anal(poss, dips, mask=None, cutoff_E=0.0, cutoff_r=2**(1/6)*1.3, box_l=[int(1E6),int(1E6),int(1E6)], pbc=True):
    poss_tmp_path = tempfile_path(suffix=".npy")
    dips_tmp_path = tempfile_path(suffix=".npy")
    return_file_path = None
    np.save(poss_tmp_path, poss)
    np.save(dips_tmp_path, dips)

    args = (poss_tmp_path, dips_tmp_path, mask, cutoff_E, cutoff_r, box_l, pbc)
    try:
        return_file_path = run_subprocess_in_module_mp(args, module=get_module_name(), worker=cluster_E_anal_worker.__name__)
        data = np.load(return_file_path)
        return (data['num_neighbors'],
                data['no_clusters'], data['cluster_sizes'])
    finally:
        for p in (poss_tmp_path, dips_tmp_path, return_file_path):
            if p is not None:
                try:
                    os.remove(p)
                except OSError:
                    pass
#####
