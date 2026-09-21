# h5 files
from pressomancy.analysis import H5DataSelector
import h5py

# math and array operations\
import numpy as np

from .generic_functions import fold_coord, run_subprocess_in_module_mp
from .global_definitions import A_to_obj_dict

from .bond_to_parts import get_module_name as get_bonds_to_parts_module_name
from .bond_to_parts import get_n_bond_to_parts_worker

def get_module_name():
    return "pyanal.io"

##### In #####
## Elastomer main from h5 ##
def get_elastomer_h5(h5_file_path, A, time_step=None, float_type: type[np.floating] =np.float64, int_type: type[np.integer] =np.int32):
    # Calculate the bond particles positions from the simulation data
    with h5py.File(h5_file_path, "r") as h5_file:
        box_l = h5_file["sys"].attrs['box_l']

        elastomer_h5 = H5DataSelector(h5_file=h5_file, particle_group="Elastomer")
        M_part = elastomer_h5.select_particles_by_object(A_to_obj_dict[A])
        M_part_at_0 = M_part.timestep[0]
        if time_step is not None:
            M_part = M_part.time(time_step)

        if A=="HM":
            # assert (np.isin(M_part.type, (3, 61))).all()
            ids = M_part.get_property("id")
            poss = M_part.get_property("pos")
            dips = M_part.get_property("dip")
            # bonds at time 0
            bonds = M_part_at_0.get_property("bonds")
        elif A in ("SM", "SMfl", "SMdumb"):
            real_ds = M_part.select_particles_by_type((4, 62))
            # assert (np.isin(real_ds.type, (4, 62))).all()
            ids = real_ds.get_property("id")
            poss = real_ds.get_property("pos")
            virt_ds = M_part.select_particles_by_type(666)
            # assert (np.asarray(virt_ds.type) == 666).all()
            dips = virt_ds.get_property("dip")
            # bonds at time 0
            bonds_ds = M_part_at_0.select_particles_by_type((4, 62))
            # assert (np.isin(bonds_ds.type, (4, 62))).all()
            bonds = bonds_ds.get_property("bonds")

    poss_folded = fold_coord(poss, box_dim=box_l)

    return np.asarray(box_l, float_type), np.asarray(ids, dtype=int_type), np.asarray(poss, dtype=float_type), np.asarray(poss_folded, dtype=float_type), np.asarray(dips, dtype=float_type), list(bonds)
##

## Bonds from h5 ##
def get_bonds_to_parts_h5(h5_file_path, time_step=None, float_type: type[np.floating] =np.float64):
    if time_step is None:
            raise NotImplementedError
    ts = int(time_step)
    with h5py.File(h5_file_path, "r") as h5_file:
        if f"properties/BondParticle{ts}" not in h5_file:
            return np.array([]), None
        size = h5_file[f"properties/BondParticle{ts}"].attrs['size']
        dataset = h5_file[f"particles/BondParticle{ts}/pos/value"]; assert isinstance(dataset, h5py.Dataset)
        poss = dataset[0, :, :]
        dataset = h5_file[f"particles/BondParticle{ts}/id/value"]; assert isinstance(dataset, h5py.Dataset)
        ids = dataset[0, :]
        assert ids.shape[0] == poss.shape[0], f"{ids.shape} {poss.shape}"
    return np.asarray(poss, dtype=float_type), size

def get_n_bond_to_parts(h5_file_path, time_step=None, in_process=True):
    if in_process:
        return get_n_bond_to_parts_worker(h5_file_path, time_step)
    else:
        return run_subprocess_in_module_mp((h5_file_path, time_step), module=get_bonds_to_parts_module_name(), worker=get_n_bond_to_parts_worker.__name__)
##

## Energy files ##
def read_E_file(path, ts, H, seed):
    with open(path, "r") as f:
        for line in f:
            line_split= line.split()
            assert len(line_split) == 7 # ts H seed E_bond E_dip E_Zee E_Zee_dipm
            if int(line_split[0]) == int(ts) and float(line_split[1]) == float(H) and int(line_split[2]) == int(seed):
                return int(line_split[0]), float(line_split[1]), int(line_split[2]), tuple(map(float, line_split[3:]))
    raise ValueError # no timestep and H found
##
#####

##### Out #####
#####

##### Tests #####
def elastomer_h5_readable(h5_file_path, verbose=False):
    try:
        with h5py.File(h5_file_path, "r") as h5_file:
            elastomer_h5 = H5DataSelector(h5_file=h5_file, particle_group="Elastomer")
        return True
    except Exception as e:
        if verbose:
            print(e)
        return False

def elastomer_h5_ts_exists(h5_file_path, time):
    if not elastomer_h5_readable(h5_file_path, verbose=True):
        raise RuntimeError(f"File at {h5_file_path} not readeable. See error message above.")
    with h5py.File(h5_file_path, "r") as h5_file:
        elastomer_h5 = H5DataSelector(h5_file=h5_file, particle_group="Elastomer")
        return time in elastomer_h5.times_array

def poke_elastomer_h5(h5_file_path, A, time_step=None):
    run_subprocess_in_module_mp((h5_file_path, A, time_step), module=get_module_name(), worker=poke_elastomer_h5_worker.__name__)

def poke_elastomer_h5_worker(h5_file_path, A, time_step=None):
    # Calculate the bond particles positions from the simulation data
    with h5py.File(h5_file_path, "r") as h5_file:
        box_l = h5_file["sys"].attrs['box_l']

        elastomer_h5 = H5DataSelector(h5_file=h5_file, particle_group="Elastomer")
        M_part = elastomer_h5.select_particles_by_object(A_to_obj_dict[A])
        M_part_at_0 = M_part.timestep[0]
        if time_step is not None:
            M_part = M_part.time(time_step)

        if A=="HM":
            # assert (np.isin(M_part.type, (3, 61))).all()
            ids = M_part.get_property("id")
            poss = M_part.get_property("pos")
            dips = M_part.get_property("dip")
            # bonds at time 0
            bonds = M_part_at_0.get_property("bonds")
        elif A in ("SM", "SMfl", "SMdumb"):
            real_ds = M_part.select_particles_by_type((4, 62))
            # assert (np.isin(real_ds.type, (4, 62))).all()
            ids = real_ds.get_property("id")
            poss = real_ds.get_property("pos")
            virt_ds = M_part.select_particles_by_type(666)
            # assert (np.asarray(virt_ds.type) == 666).all()
            dips = virt_ds.get_property("dip")
            # bonds at time 0
            bonds_ds = M_part_at_0.select_particles_by_type((4, 62))
            # assert (np.isin(bonds_ds.type, (4, 62))).all()
            bonds = bonds_ds.get_property("bonds")

#####
