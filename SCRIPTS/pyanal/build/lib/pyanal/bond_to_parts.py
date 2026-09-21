# pyanal.bond_to_parts

# h5 files
from pressomancy.analysis import H5DataSelector
import h5py

# math and array operations
import numpy as np
import math

# numba
from numba import njit, prange

from .generic_functions import run_subprocess_in_module, run_subprocess_in_module_mp
from .global_definitions import A_to_obj_dict

FASTMATH=True

def get_module_name():
    return "pyanal.bond_to_parts"

def calculate_and_write_bond_particles_h5(read_simulation_h5_path, write_bond_particles_h5_path, A, bond_part_center_distance, time_step=None, new=False):
    args = ( read_simulation_h5_path, write_bond_particles_h5_path, A, bond_part_center_distance, time_step, new )
    run_subprocess_in_module_mp(args, module=get_module_name(), worker=calculate_and_write_bond_particles_h5_worker.__name__)

def calculate_and_write_bond_particles_h5_worker(read_simulation_h5_path, write_bond_particles_h5_path, A, bond_part_center_distance, time_step=None, new=False):
    if time_step is None:
        raise NotImplementedError

    bond_parts, bond_part_ids, bond_particle_to_bond, bond_to_particle = calculate_elastomer_bond_particles_h5(read_simulation_h5_path, A, bond_part_center_distance, time_step=time_step)

    if bond_parts is None:
        raise ValueError(f"Something wrong with the bond to parts calculation for {read_simulation_h5_path} at time {time_step}")

    write_bond_part_h5(write_bond_particles_h5_path, bond_parts, bond_part_ids, bond_particle_to_bond, bond_to_particle, bond_part_center_distance, time_step=time_step, new=new)

def write_bond_part_h5(write_bond_particles_h5_path, bond_parts, bond_part_ids, bond_particle_to_bond, bond_to_particle, bond_particle_size, time_step=None, new=False):
    if time_step is None:
        raise NotImplementedError

    ts = int(time_step)

    if new:
        file_open_mode = 'w'
    else:
        file_open_mode = 'a'

    total_number_of_bond_particles = bond_parts.shape[0]

    assert total_number_of_bond_particles > 0
    assert bond_part_ids.shape[0] == total_number_of_bond_particles
    assert bond_parts.shape[1] == 3

    with h5py.File(write_bond_particles_h5_path, file_open_mode) as h5_file:
        # Bond particle size
        sys_grp = h5_file.require_group(f"properties/BondParticle{ts}")
        sys_grp.attrs["size"] = bond_particle_size

        # Connectivity dataset
        connect_grp = h5_file.require_group(f"connectivity/BondParticle{ts}")
        connect_grp.create_dataset(
            f"BondParticle{ts}_to_Bond",
            data=bond_particle_to_bond,
            dtype=np.int32,
            shape=(bond_particle_to_bond.shape)
        )
        connect_grp.create_dataset(
            f"Bond_to_Particle",
            data=bond_to_particle,
            dtype=np.int32,
            shape=(bond_to_particle.shape)
        )

        # Particle positions dataset
        bond_part_grp = h5_file.require_group(f"particles/BondParticle{ts}")
        # id
        id_group = bond_part_grp.require_group("id")
        id_step = id_group.create_dataset("step", shape=(1,), dtype=np.int32)
        id_time = id_group.create_dataset("time", shape=(1,), dtype=np.float32)
        id_value = id_group.create_dataset(
            "value",
            shape=(1, total_number_of_bond_particles,),
            dtype=np.int32,
            chunks=(1, total_number_of_bond_particles),
            compression="gzip",
            compression_opts=4
        )
        id_step[0] = time_step
        id_time[0] = time_step
        id_value[0, :] = np.asarray(bond_part_ids, dtype=np.int32)
        # pos
        pos_group = bond_part_grp.require_group("pos")
        pos_step = pos_group.create_dataset("step", shape=(1,), dtype=np.int32)
        pos_time = pos_group.create_dataset("time", shape=(1,), dtype=np.float32)
        pos_value= pos_group.create_dataset(
            "value",
            shape=(1, total_number_of_bond_particles, 3),
            dtype=np.float32,
            chunks=(1, total_number_of_bond_particles, 3),
            compression="gzip",
            compression_opts=4
        )
        pos_step[0] = time_step
        pos_time[0] = time_step
        pos_value[0, :, :] = np.asarray(bond_parts, dtype=np.float32)

        h5_file.flush()

def calculate_elastomer_bond_particles_h5(h5_file_path, A, bond_part_size, time_step=None):
    with h5py.File(h5_file_path, "r") as h5_file:
        box_l = h5_file["sys"].attrs['box_l']

        elastomer_h5 = H5DataSelector(h5_file=h5_file, particle_group="Elastomer")
        max_id = np.asarray(elastomer_h5.id).max()

        M_part = elastomer_h5.select_particles_by_object(A_to_obj_dict[A])

        if A=="HM":
            bonded_parts = M_part.timestep[0]
            assert (np.isin(M_part.type, (3, 61))).all() and (np.isin(bonded_parts.type, (3, 61))).all()
        elif A in ("SM", "SMfl", "SMdumb"):
            M_part = M_part.select_particles_by_type((4, 62))
            bonded_parts = M_part.timestep[0]

        if time_step is not None:
            M_part = M_part.time(time_step)

        poss = np.asarray(M_part.pos, dtype=np.float32)
        n_part = poss.shape[0]
        if n_part == 0:
            return None, None, None, None
        ids = np.asarray(M_part.id, dtype=np.int32)
        bonds = bonded_parts.bonds

        bond_counts = np.array([len(b) for b in bonds], dtype=np.int32)
        bonds_flat_offset = np.zeros(n_part + 1, dtype=np.int32)
        bonds_flat_offset[1:] = np.cumsum(bond_counts)
        total_bonds = bonds_flat_offset[-1]

        id_to_idx_dict = {pid: idx for idx, pid in enumerate(ids)}
        bond_idxs2 = np.empty(total_bonds, dtype=np.int32)
        for i in range(n_part):
            start_idx = bonds_flat_offset[i]
            end_idx = bonds_flat_offset[i+1]
            bond_idxs2[start_idx:end_idx] = [id_to_idx_dict[b[4]] for b in bonds[i]]

        bond_part_counts = calculate_total_number_of_bond_particles(poss, bond_part_size, n_part, bond_idxs2, bonds_flat_offset, box_l)
        bonds_part_flat_offset = np.zeros(total_bonds + 1, dtype=np.int32)
        bonds_part_flat_offset[1:] = np.cumsum(bond_part_counts)
        total_bond_parts = bonds_part_flat_offset[-1]

        assert total_bond_parts > 0 and total_bond_parts < 2e9, f"{total_bond_parts}"

        bond_parts = calculate_bond_particles_flat(poss, bond_part_size, n_part, total_bond_parts, bond_idxs2, bonds_flat_offset, bonds_part_flat_offset, box_l)
        bond_part_ids = np.arange(max_id, max_id + total_bond_parts, 1, dtype=np.int32)

        assert bond_parts.shape[0] == bond_part_ids.shape[0]

        bond_particle_to_bond, bond_to_particle = calculate_connectivity(n_part, total_bonds, total_bond_parts, ids, bond_idxs2, bond_part_ids, bonds_flat_offset, bonds_part_flat_offset)

    return np.asarray(bond_parts, dtype=np.float32), np.asarray(bond_part_ids, dtype=np.int32), np.asarray(bond_particle_to_bond, dtype=np.int32), np.asarray(bond_to_particle, dtype=np.int32)

## helper numba functions
@njit(parallel=True, fastmath=FASTMATH)
def calculate_total_number_of_bond_particles(poss, bond_part_distance, N, idxs_2, flat_offsets, box_l):
    inv_box0 = 1 / box_l[0]
    inv_box1 = 1 / box_l[1]
    inv_box2 = 1 / box_l[2]

    inv_d = 1 / bond_part_distance

    bond_part_len = np.empty(flat_offsets[-1], dtype=np.int32)

    for i in prange(N):
        xi0, xi1, xi2 = poss[i]
        for j in range(flat_offsets[i], flat_offsets[i+1]):
            xj0, xj1, xj2 = poss[idxs_2[j]]

            r0 = xj0 - xi0
            r1 = xj1 - xi1
            r2 = xj2 - xi2

            # ---- minimum image convention ----
            r0 -= box_l[0] * math.floor(r0 * inv_box0 + 0.5)
            r1 -= box_l[1] * math.floor(r1 * inv_box1 + 0.5)
            r2 -= box_l[2] * math.floor(r2 * inv_box2 + 0.5)

            r_ij = math.sqrt(r0*r0 + r1*r1 + r2*r2)

            n_bond_part_ij = int(r_ij * inv_d)

            bond_part_len[j] = n_bond_part_ij


    return bond_part_len
@njit(parallel=True, fastmath=FASTMATH)
def calculate_bond_particles_flat(poss, bond_part_distance, N, Nbp, idxs_2, flat_offsets, bonds_part_flat_offset, box_l):
    inv_box0 = 1 / box_l[0]
    inv_box1 = 1 / box_l[1]
    inv_box2 = 1 / box_l[2]

    bond_parts_flat = np.empty((Nbp, 3), dtype=np.float32)

    for i in prange(N):
        xi0, xi1, xi2 = poss[i]
        # ---- fold coords ----
        xi0 -= box_l[0] * math.floor(xi0 * inv_box0)
        xi1 -= box_l[1] * math.floor(xi1 * inv_box1)
        xi2 -= box_l[2] * math.floor(xi2 * inv_box2)
        for j in range(flat_offsets[i], flat_offsets[i+1]):
            xj0, xj1, xj2 = poss[idxs_2[j]]

            r0 = xj0 - xi0
            r1 = xj1 - xi1
            r2 = xj2 - xi2

            # ---- minimum image convention ----
            r0 -= box_l[0] * math.floor(r0 * inv_box0 + 0.5)
            r1 -= box_l[1] * math.floor(r1 * inv_box1 + 0.5)
            r2 -= box_l[2] * math.floor(r2 * inv_box2 + 0.5)

            r_ij = math.sqrt(r0*r0 + r1*r1 + r2*r2)
            inv_r = 1 / r_ij

            r0 *= inv_r
            r1 *= inv_r
            r2 *= inv_r

            r0d = r0 * bond_part_distance
            r1d = r1 * bond_part_distance
            r2d = r2 * bond_part_distance

            x0 = xi0 + 0.5 * r0d
            x1 = xi1 + 0.5 * r1d
            x2 = xi2 + 0.5 * r2d

            for k in range(bonds_part_flat_offset[j], bonds_part_flat_offset[j+1]):
                bond_parts_flat[k][0] = x0
                bond_parts_flat[k][1] = x1
                bond_parts_flat[k][2] = x2

                x0 += r0d
                x1 += r1d
                x2 += r2d

    return bond_parts_flat

def calculate_bond_particles(particle_positions, bond_pairs, box_l, bond_part_distance=0.25):
    """
    Compute the in-memory positions of the "bond particles": beads laid every
    ``bond_part_distance`` along each bond, from the anchor toward its partner.

    Reuses the numba kernels ``calculate_total_number_of_bond_particles`` and
    ``calculate_bond_particles_flat`` (same geometry / minimum-image convention as
    ``calculate_elastomer_bond_particles_h5``), but returns the beads directly
    instead of writing an h5 cache.

    Parameters:
        particle_positions : (N, 3) anchor particle positions
        bond_pairs : (n_bonds, 2) int indices (anchor, partner) into particle_positions
        box_l : box dimensions [Lx, Ly, Lz] (for minimum image + folding)
        bond_part_distance : spacing between consecutive beads along a bond

    Returns:
        (M, 3) float32 array of bead positions (empty (0, 3) if there are no bonds).
    """
    poss = np.ascontiguousarray(particle_positions, dtype=np.float32)
    N = poss.shape[0]
    box_l = np.asarray(box_l, dtype=np.float64)
    bond_pairs = np.asarray(bond_pairs)

    if bond_pairs.size == 0:
        return np.empty((0, 3), dtype=np.float32)

    # Build the CSR structure the kernels expect (partners grouped by anchor),
    # mirroring calculate_elastomer_bond_particles_h5.
    anchors = bond_pairs[:, 0].astype(np.int64)
    partners = bond_pairs[:, 1].astype(np.int64)
    order = np.argsort(anchors, kind="stable")
    idxs_2 = partners[order].astype(np.int32)
    counts = np.bincount(anchors, minlength=N).astype(np.int32)
    flat_offsets = np.zeros(N + 1, dtype=np.int32)
    flat_offsets[1:] = np.cumsum(counts)

    bond_part_len = calculate_total_number_of_bond_particles(
        poss, bond_part_distance, N, idxs_2, flat_offsets, box_l)
    bonds_part_flat_offset = np.zeros(bond_part_len.shape[0] + 1, dtype=np.int32)
    bonds_part_flat_offset[1:] = np.cumsum(bond_part_len)
    total_bond_parts = int(bonds_part_flat_offset[-1])

    if total_bond_parts == 0:
        return np.empty((0, 3), dtype=np.float32)

    bond_parts = calculate_bond_particles_flat(
        poss, bond_part_distance, N, total_bond_parts,
        idxs_2, flat_offsets, bonds_part_flat_offset, box_l)
    return np.asarray(bond_parts, dtype=np.float32)

@njit(parallel=True, fastmath=FASTMATH)
def calculate_connectivity(N, Nb, Nbp, ids, idxs_2, bond_part_ids, flat_offsets, bonds_part_flat_offset):
    bond_part_to_bond = np.empty((Nbp, 2), dtype=np.int32)
    bond_to_particle = np.empty((Nb*2, 2), dtype=np.int32)

    for i in prange(N):
        id1 = ids[i]
        for j in range(flat_offsets[i], flat_offsets[i+1]):
            twoj = 2 * j
            # Anchor particle is first
            bond_to_particle[twoj, 0] = j
            bond_to_particle[twoj, 1] = id1
            # Other bonded particle in odd idxs
            bond_to_particle[twoj+1, 0] = j
            bond_to_particle[twoj+1, 1] = ids[idxs_2[j]]
            for k in range(bonds_part_flat_offset[j], bonds_part_flat_offset[j+1]):
                bond_part_to_bond[k, 0] = bond_part_ids[k]
                bond_part_to_bond[k, 1] = j

    return bond_part_to_bond, bond_to_particle

##### Test #####
def get_n_bond_to_parts_worker(h5_file_path, time_step):
    ts = int(time_step)
    with h5py.File(h5_file_path, "r") as h5_file:
        if f"properties/BondParticle{ts}" not in h5_file:
            return None, None
        size = h5_file[f"properties/BondParticle{ts}"].attrs['size']
        dataset = h5_file[f"particles/BondParticle{ts}/pos/value"]
        assert isinstance(dataset, h5py.Dataset)
        n_part = dataset[0, :, :].shape[0]
        assert n_part > 0
    return n_part, size
#####
