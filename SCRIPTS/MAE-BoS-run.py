# Track time to kill process before walltime
import signal
WALLTIME = 72 * 3600      # total job walltime (seconds) - hours * seconds in an hour
SAFETY   = 20 * 60        # safety timer (seconds) - minutes * seconds in a minute
STOP_REQUESTED = False
def stop_handler(signum, frame):
   global STOP_REQUESTED
   print("Wall-e getting close...", flush=True)
   STOP_REQUESTED = True
signal.signal(signal.SIGALRM, stop_handler)
signal.setitimer(signal.ITIMER_REAL, WALLTIME - SAFETY)

import time
global_start_time = time.monotonic_ns() # in nanoseconds

import espressomd
from espressomd.magnetostatics import DipolarDirectSum
import espressomd.checkpointing
# global variables for checkpointing
global sim_params_cp
global system_sys; global system_part_types; global system_seed; global system_kT
global elastomer_n_parts
global sim_time
global OUTPUTPATH; global CFPNAME
global magnetize_func
global TRACK_TOTAL_TIME_NS

DIPM_PDS = 1.0

# MH only: sim-time units to equilibrate at H=0 (let in-plane chains form) before applying the
# field. 0 disables. Tune to a cheap pre-step (<~20% of total_sim_time); eq wall-time is printed.
EQ_TIME = 100
EQ_TIMESTEP = 0.0005

# using this
SIZE = 1.
SIGMA = 1.

# Not using this
# SIZE = None # leave None here if you want to choose sigma by hand
# SIGMA = 1. if SIZE is None else SIZE * 0.890898718 # else SIZE / 2^(1/6)
# SIZE = SIGMA * 1.12246205 if SIZE is None else SIZE # SIGMA * 2^(1/6) if size is not inputed

espressomd.assert_features(['WCA', 'ROTATION', 'EXTERNAL_FORCES',
                            'VIRTUAL_SITES', 'VIRTUAL_SITES_RELATIVE',
                            'DIPOLES', 'CUDA'])

from pressomancy.simulation import Simulation, Elastomer, PointDipolePermanent, PointDipoleSuperpara

import h5py

import numpy as np

import sys, os
import argparse

from tqdm import tqdm


# Default data location, resolved relative to the repository (see
# SCRIPTS/repo_paths.py). --SAMPLE_DIR and --OUT_DIR override it.
from repo_paths import REPO_ROOT, DATA_DIR

#############
# Functions #

def write_and_flush_h5(system, time):
   system.sys.integrator.run(0, recalc_forces=True)
   system.write_part_group_to_h5(time=time, unique_time=True)
   system.io_dict['h5_file'].flush()

def load_from_h5(system, h5_path, is_sample=False, time=None, sim_params=None):
   if sim_params is not None:
      A = sim_params['A']
      K = sim_params['K']
      HEIGHT = sim_params['HEIGHT']
      seed = sim_params['seed']
   dipm_HM = 1.
   dipm_SM = 1.

   # ensure system clean state
   system.reinitialize_instance()
   system.reset_non_bonded_inter()

   with h5py.File(h5_path, "r") as h5_file:
      # Pressomancy system parameters
      seed_file = h5_file["sim_inst"].attrs['seed']
      kT = h5_file["sim_inst"].attrs['kT']

      if seed_file != seed:
         print("WARNING: chose different seed that previously. should not be a problem. Will use inpute seed and ignore seed from file.")

      # System parameters
      box_l = h5_file["sys"].attrs['box_l']
      time_step = h5_file["sys"].attrs['time_step']
      periodicity = h5_file["sys"].attrs['periodicity']

      system.sys.box_l = box_l
      system.set_sys(time_step=time_step)
      system.sys.periodicity = periodicity
      system.seed = seed
      system.kT = kT

      # Elastomer
      ## associated objects
      A_to_name_dict = {"HM": "PointDipolePermanent", "SM": "PointDipoleSuperpara", "SMfl": "PointDipoleSuperpara"}

      if is_sample:
         types = np.asarray(h5_file[f"particles/Elastomer/type/value"][-1,:,0])
         n_parts = types[types==1].shape[0]
      else:
         n_parts = np.asarray(h5_file[f"connectivity/Elastomer/Elastomer_to_{A_to_name_dict[A]}"]).shape[0]

      if A == "HM":
         E_associated_objects = [PointDipolePermanent(config=PointDipolePermanent.config.specify(dipm=dipm_HM, espresso_handle=system.sys)) for _ in range(n_parts)]
      elif A == "SM":
         E_associated_objects = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(dipm=dipm_SM, Xi_0=0.1, mag_func=0, espresso_handle=system.sys)) for _ in range(n_parts)]
      elif A == "SMfl":
         E_associated_objects = [PointDipoleSuperpara(config=PointDipoleSuperpara.config.specify(dipm=dipm_SM, Xi_0=0.1, mag_func=1, espresso_handle=system.sys)) for _ in range(n_parts)]
      else:
         raise ValueError(f"Particel type not recognized: {A}")

      ## elastomer object
      bond_K_limits = {"hard": (0.01, 0.1), "soft": (0.001, 0.01)}
      config_E = Elastomer.config.specify(layer_height=HEIGHT, n_parts=n_parts, associated_objects=E_associated_objects, bond_K_lims=bond_K_limits[K], size=SIZE, sigma=SIGMA, espresso_handle=system.sys, seed=system.seed)
      elastomer=[Elastomer(config=config_E) for _ in range(1)]
      system.store_objects(elastomer)

      system.set_objects(elastomer)
      elastomer= elastomer[0]
      
      system.set_steric(tuple((type_name for type_name in system.part_types.keys() if "real" in type_name)), sigma=SIGMA)

      if is_sample:
         h5_step = -1 if time is None else time
         step_bonds = -1 if time is None else time
      elif time is None:
         h5_step = -1
         step_bonds = 0
      else:
         times = h5_file["particles/Elastomer/id/time"][:]
         h5_step = np.nonzero(times == time)[0][0]
         step_bonds = 0

      types = h5_file["particles/Elastomer/type/value"][h5_step,:,0]
      vip_type_mask = np.isin(types, (1, 61, 62))
      ids = h5_file["particles/Elastomer/id/value"][h5_step,:,0][vip_type_mask]
      poss = h5_file["particles/Elastomer/pos/value"][h5_step,:,:][vip_type_mask]
      dips = h5_file["particles/Elastomer/dip/value"][h5_step,:,:][vip_type_mask]
      fixs = h5_file["particles/Elastomer/fix/value"][h5_step,:,:][vip_type_mask]
      # bonds require a bit more work for older pressomancy
      ds = h5_file["particles/Elastomer/bonds/value"]
      bond_full=[]
      print("bonds from h5 and shape", ds, ds.shape)
      for i in range(ds.shape[1]):
         try:
            assert len(ds[step_bonds, i]) > 0
            bond_full.append(ds[step_bonds, i])
         except:
            bond_full.append([])
      bonds = np.asarray([bond_full], dtype=object)[0,:][vip_type_mask]

      print("particle types from h5 and how many mag parts", set(types), len(types[np.isin(types, (1,61,62))]))
      print("created particles and how many mag parts", set(system.sys.part.all().type), len(system.sys.part.select(lambda p: p.type in (61,62))))

      id_to_espressoid = {}
      if A == "HM":
         sys_part_pdp = system.sys.part.select(lambda p: p.type == 61) # 3 is legacy
         assert len(sys_part_pdp) == n_parts
         for i, part in enumerate(sys_part_pdp):
               id_to_espressoid[ids[i]] = part.id
               part.pos = poss[i]
               part.fix = list(map(bool, fixs[i]))
               if any(part.fix):
                  assert part.pos[2] < 1.9, "Particles are not correctly adhered to substrate."
               if not is_sample:
                  part.dip = dips[i] / np.linalg.norm(dips[i]) * dipm_HM
               assert np.isclose(part.dipm, dipm_HM)
      else:
         sys_part_pds_vip = system.sys.part.select(lambda p: p.type == 62) # 4 is legacy
         assert len(sys_part_pds_vip) == n_parts
         for i, part in enumerate(sys_part_pds_vip):
               id_to_espressoid[ids[i]] = part.id
               id_virt = part.id + 1 # get current id of virtual site
               part.pos = poss[i] # position of real particle
               part.fix = list(map(bool, fixs[i]))
               if any(part.fix):
                  assert part.pos[2] < 1.9, "Particles are not correctly adhered to substrate."
               part_virt = system.sys.part.by_id(id_virt) # get virtual site
               assert part_virt.vs_relative[0]==part.id # assert the correct virtual site
               part_virt.pos = part.pos # same position as real particle
               if not is_sample:
                  part_virt.dip = dips[i] # virtual site is the one with the dipoe moment

      # bond particles by ids
      assert len(bonds) == n_parts
      assert len(bonds[0]) > 0
      for i, bond_list in enumerate(bonds):
         id1 = id_to_espressoid[ids[i]]
         for bond in bond_list:
            id2 = id_to_espressoid[bond[4]] # get espresso system ids
            k_bond = bond[1]; r_0_bond = bond[2]
            elastic_bond = espressomd.interactions.HarmonicBond(k=k_bond, r_0=r_0_bond)
            system.sys.bonded_inter.add(elastic_bond)
            system.sys.part.by_id(id1).add_bond((elastic_bond, id2))
            assert id1 < id2
            assert k_bond > bond_K_limits[K][0] and k_bond < bond_K_limits[K][1]
            assert r_0_bond < 5.001

   return elastomer

def get_steps_per_time(time_step):
   """Map an espresso time_step to the number of integration steps per sim-time unit."""
   if np.isclose(time_step, 0.0005):
      return 1000
   elif np.isclose(time_step, 0.0008):
      return 625
   elif np.isclose(time_step, 0.001):
      return 500
   elif np.isclose(time_step, 0.0001):
      return 5000
   else:
      raise ValueError(f"sample_steps could not be defined, as time_step is not ususal: {time_step}")

# end Functions #
#################

# Simulation parameters
parser = argparse.ArgumentParser()
parser.add_argument("--A", action ="store", dest="A", type=str, help="Defines the magnetic particles magnetic behaviour ('HM', 'SM', 'SMfl', 'SMdumb').")
parser.add_argument("--N_FULL_BOX", action="store", dest="N_FULL_BOX", type=int, help="Number of particles that would fill the cubic system box, at a volume density of 0.3.")

parser.add_argument("--DENS", action ="store", dest="DENS", type=float, help="Volume density of particles 'A' (0 < DENS_A <= 0.3).")
parser.add_argument("--K", action ="store", dest="K", type=str, help="Spring constant type of bonds (soft or hard).")
parser.add_argument("--HEIGHT", action="store", dest="HEIGHT", type=int, help="Height of the MAE layer, in units of particle diameters (>= 1).")

parser.add_argument("--seed", action="store", dest="seed", type=int, help="Simulation seed.")

parser.add_argument("--H", action ="store", dest="H", type=float, help="z-component of the external magnetic field.")

parser.add_argument("--H_prev", action ="store", dest="H_prev", type=float, help="Sample H - sample will be a simulated MAE with the input parameters and H = H_prev. Must be a float. Omit when initializing from a sample created with mae-BoS-create.py.")
parser.add_argument("--time_prev", action ="store", dest="time_prev", type=int, help="Simulation time of the sample - sample will be a simulated MAE with the input parameters and H = H_prev, and we will take the time = time_prev. Must be integer.")

parser.add_argument("--start_from", action="store", dest="start_from", type=str, default="sample", choices=["sample", "eqH0"], help="Initial config source for a point run (i.e. when --H_prev is absent): 'sample' = bare cured sample (+ in-run MH equilibration); 'eqH0' = a pre-equilibrated H=0 run output (cp{A}-{seed}-H0.00.h5 under {pdp}-.../ in SAMPLE_DIR), which skips the in-run equilibration. Ignored when --H_prev is given. Output naming is canonical either way.")

parser.add_argument("--SAMPLE_DIR", action="store", dest="SAMPLE_DIR", type=str, default=os.path.join(DATA_DIR, "samples"), help="Path of the sample directory. Defaults to DATA/samples inside the repository. Either the path for the samples created with mae-BoS-create or the path to the output directory, with a previously simulated mae with H = H_prev and time = time_prev.")

parser.add_argument("--total_sim_time", action ="store", dest="total_sim_time", type=int, help="The total time simulated. Time unit used is 0.5 espresso time_step. Must be integer.")

parser.add_argument("--save_h5_each", action ="store", dest="save_h5_each", type=int, help="Save system configuration each X sim_time. Must be greater than 0 and integer.")
parser.add_argument("--checkpoint_each", action="store", dest="checkpoint_each", type=int, help="Checkpoint each X sim_time. Must be greater or equal to 0 (0 does not save any checkpoint).")

parser.add_argument("--sys_time_step", action="store", dest="sys_time_step", type=float, help="The espressomd system time_step. Not related to the sim_time. A lower value means a more stable simulaitno, but slower.")

parser.add_argument("--OUT_DIR", action="store", dest="OUT_DIR", type=str, default=os.path.join(DATA_DIR, "MAE-BoS", "cluster", "sim_data"), help="The path to the output directory of the simulation data. Defaults to DATA/MAE-BoS/cluster/sim_data inside the repository.")
parser.add_argument("--CHECKPOINT_DIR", action="store", dest="CHECKPOINT_DIR", type=str, help="The path to the directory containing the espresso checkpoints (the parent dict, not the individual folder containin the espresso checkpoint files for a single simulation).")

sim_params=vars(parser.parse_args())

print("")
print("<------------------------HEHE------------------------>")
print(sim_params)

magnetize_func_map = {"HM": None, "SM": "magnetize", "SMfl": "magnetize_froelich_kennelly", "SMdumb": "magnetize_dumb"}
magnetize_func = magnetize_func_map[sim_params['A']]
mag_func_to_int = {"SM": 0, "SMfl": 1}

current_time = time.monotonic_ns()
print(f"-- imported EVERYTHING -- {(current_time-global_start_time) // 1_000_000_000} (seconds)")
previous_time = current_time

# Init checkpoint
CHECKPOINT_DIR = sim_params['CHECKPOINT_DIR']
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
assert sim_params['A'] in ("HM", "SM", "SMfl", "SMdumb"), f"only valid values for --A are \"HM\" and \"SM\" (or \"SMfl\" or \"SMdumb\" for different magnetizations): {sim_params['A']}"
assert sim_params['K'] == "soft" or sim_params['K'] == "hard", f"--K must be \"soft\" or  \"hard\": {sim_params['K']}"
checkpoint_id= f"{sim_params['A']}-{sim_params['DENS']:.2f}_{sim_params['K']}-{sim_params['N_FULL_BOX']}_{sim_params['HEIGHT']}-{sim_params['seed']}--{sim_params['H']:.2f}"
if sim_params['H_prev'] is not None:
   checkpoint_id += f"--fromH{sim_params['H_prev']:.2f}"
print("CHECKPOINT_DIR", CHECKPOINT_DIR, "checkpoint_id", checkpoint_id)
checkpoint = espressomd.checkpointing.Checkpoint(checkpoint_id=checkpoint_id, checkpoint_path=CHECKPOINT_DIR)
if checkpoint.has_checkpoints():
   checkpoint_loaded= True
else: # if no available checkpoints, start from sample file
   checkpoint_loaded= False
   print("WARNING: did not found any valild checkpoints. Initializing from sample file.")

current_time = time.monotonic_ns()
print(f"-- Init checkpoint class -- {(current_time-previous_time) // 1_000_000_000} (seconds)")
previous_time = current_time

if not checkpoint_loaded:

   SAMPLE_DIR = sim_params['SAMPLE_DIR']
   if os.path.isdir(SAMPLE_DIR):
      pdA_dict = {"HM": "pdp", "SM": "pds", "SMfl": "pds", "SMdumb": "pds"}
      if sim_params['H_prev'] is None and sim_params['start_from'] == "eqH0":
         # Independent point started from a pre-equilibrated H=0 run output (already has the in-plane
         # chains). Loaded with is_sample=False so the equilibrated dipole orientations are read; the
         # in-run EQ_TIME equilibration is skipped. H_prev stays None -> canonical output naming.
         SAMPLEPATH = os.path.join(SAMPLE_DIR, f"{pdA_dict[sim_params['A']]}-{sim_params['DENS']:.2f}_{sim_params['K']}-{sim_params['N_FULL_BOX']}_{sim_params['HEIGHT']}", f"cp{sim_params['A']}-{sim_params['seed']}-H0.00.h5")
         print(f"sp-path pointed to a pre-equilibrated H=0 output. Simulating from {SAMPLEPATH}.")
         sample_is_sample = False
      elif sim_params['H_prev'] is None:
         SAMPLEPATH = os.path.join(SAMPLE_DIR, f"A-{sim_params['DENS']:.2f}_{sim_params['K']}-{sim_params['N_FULL_BOX']}_{sim_params['HEIGHT']}-{sim_params['seed']}.h5")
         print(f"sp-path pointed to asamples directory. Simulating with the sample correspondent to input params.")
         sample_is_sample = True
      else:
         SAMPLEPATH = os.path.join(SAMPLE_DIR, f"{pdA_dict[sim_params['A']]}-{sim_params['DENS']:.2f}_{sim_params['K']}-{sim_params['N_FULL_BOX']}_{sim_params['HEIGHT']}", f"cp{sim_params['A']}-{sim_params['seed']}-H{sim_params['H_prev']:.2f}.h5")
         print(f"sp-path pointed to a sim_data directory. Simulating with the sample correspondent to input params with H_prev={sim_params['H_prev']}.")
         sample_is_sample = False
   else:
      raise FileNotFoundError(f"SAMPLE_DIR should be a path to a directory: {SAMPLE_DIR}\nEither it does not exist, or there is a file with the same name.")

   current_time = time.monotonic_ns()
   print(f"SAMPLEPATH = {SAMPLEPATH} -- {(current_time-previous_time) // 1_000_000_000} (seconds)")
   previous_time = current_time

   # Create system
   system = Simulation(box_dim=[10]*3) # hold size

   # reinitialize system and add elastomer
   elastomer = load_from_h5(system, h5_path=SAMPLEPATH, is_sample=sample_is_sample, time=sim_params['time_prev'], sim_params=sim_params)
   system.set_sys(time_step=0.001)
   system.kT = 1E-6

   # Add extra io settings for saves (see default)
   system.io_dict['bonds'] = "all"
   system.io_dict['properties'].extend([('fix', 3)])
   print(system.io_dict)

   print("bonds of part 0", system.sys.part.by_id(0).bonds)

   if sim_params['K'] == "soft":
      assert elastomer.params['bond_K_lims'][0] == 0.001 and elastomer.params['bond_K_lims'][1] == 0.01, f"Not soft: elastomer bond lims {elastomer.params['bond_K_lims']}"
   elif sim_params['K'] == "hard":
      assert elastomer.params['bond_K_lims'][0] == 0.01 and elastomer.params['bond_K_lims'][1] == 0.1, f"Not hard: elastomer bond lims {elastomer.params['bond_K_lims']}"
   else:
      print(f"WARNING: K is not an expected values. Be sure to confirm the limits for K: {sim_params['K']} {elastomer.params['bond_K_lims']}")

   current_time = time.monotonic_ns()
   print(f"-- Loaded elastomer -- {(current_time-previous_time) // 1_000_000_000} (seconds)")
   previous_time = current_time

   #### output nice system stuff ####

   print(f"system.part_types {system.part_types}, (part_type: n_parts_type) {{{(lambda types: {typ: len(system.sys.part.select(type=typ)) for typ in types})(set(system.sys.part.all().type))}}}")
   print(system.sys.cell_system.get_state())
   print("dipm of 0 1", system.sys.part.by_id(0).dipm, system.sys.part.by_id(1).dipm)

   #### Run the sample with external H ####

   # Add thermostat
   system.sys.thermostat.set_langevin(kT=system.kT, gamma=100, seed=system.seed)

   # Add magnetic dipole interactions - direct sum, non-preiodic in z
   system.sys.periodicity = [True, True, False]
   system.init_magnetic_inter(DipolarDirectSum(prefactor=1, n_replicas=2, gpu=True))

   print("-|- non-bonded inter -|- at least two should be not 0")
   print(system.sys.non_bonded_inter[98, 61].wca.epsilon, system.sys.non_bonded_inter[98, 61].wca.sigma)
   print(system.sys.non_bonded_inter[61, 61].wca.epsilon, system.sys.non_bonded_inter[61, 61].wca.sigma)
   print(system.sys.non_bonded_inter[98, 62].wca.epsilon, system.sys.non_bonded_inter[98, 62].wca.sigma)
   print(system.sys.non_bonded_inter[62, 62].wca.epsilon, system.sys.non_bonded_inter[62, 62].wca.sigma)

   # Mark particles to magnetize. Careful to use python lists, and not espressomd particle slices
   parts_to_magnetize= list(system.sys.part.select(type=system.part_types['pds_virt'])) if 'pds_virt' in system.part_types else []

   print("n_parts_to_magnetize", len(parts_to_magnetize))

   current_time = time.monotonic_ns()
   print(f"-- Added magnetism -- {(current_time-previous_time) // 1_000_000_000} (seconds)")
   previous_time = current_time

   # WHEN THERE ARE MH AND MS IN SAME SIMULATION add relaxation of MS dipoles due to the dipolar fields of the MH dipoles (keeop here, as a reminder)

   # EQUILIBRATE MAGNETICALLY-HARD PARTICLES AT ZERO FIELD
   # MH carry permanent dipoles that form in-plane chains at H=0 in the slab geometry; let them
   # form before the field is applied. MS need none (zero induced moment at H=0), and a reverse
   # (H_prev) run already starts from an equilibrated configuration.
   if sim_params['A'] == "HM" and sim_params['H_prev'] is None and sim_params['start_from'] != "eqH0" and EQ_TIME > 0:
      system.sys.time_step = EQ_TIMESTEP
      eq_steps_per_time = get_steps_per_time(system.sys.time_step)
      system.set_H_ext(H=[0., 0., 0.])
      print(f"-- Equilibrating MH at H=0 for {EQ_TIME} sim-time units ({EQ_TIME * eq_steps_per_time} steps, time_step {system.sys.time_step}) --")
      system.sys.integrator.run(EQ_TIME * eq_steps_per_time, recalc_forces=True)
      current_time = time.monotonic_ns()
      eq_seconds = (current_time - previous_time) // 1_000_000_000
      print(f"-- Equilibrated MH at H=0 -- {eq_seconds // 60} (minutes) {eq_seconds % 60} (seconds)")
      previous_time = current_time

   # STABILIZE MAE WITH MAGNETIC FIELD
   system.sys.time = 0.
   system.sys.time_step = 0.001

   ext_B_z = sim_params['H']
   H_ext = [0.,0.,ext_B_z]
   system.set_H_ext(H=H_ext)

   pdA_dict = {"HM": "pdp", "SM": "pds", "SMfl": "pds", "SMdumb": "pds"}
   OUTPUTPATH = os.path.join(sim_params['OUT_DIR'], f"{pdA_dict[sim_params['A']]}-{sim_params['DENS']:.2f}_{sim_params['K']}-{sim_params['N_FULL_BOX']}_{sim_params['HEIGHT']}")
   if sim_params['H_prev'] is not None:
      OUTPUTPATH += f"-reverse"
      if sim_params['time_prev'] is not None:
         OUTPUTPATH += f"-t{sim_params['time_prev']}"
   os.makedirs(OUTPUTPATH, exist_ok=True)
   assert os.path.isdir(OUTPUTPATH), f"Failed to create directory: {OUTPUTPATH}"

   if sim_params['H_prev'] is None:
      CFPNAME = f"cp{sim_params['A']}-{system.seed}-H{ext_B_z:.2f}.h5"
   else:
      CFPNAME = f"cp{sim_params['A']}-{system.seed}-H{ext_B_z:.2f}-fromH{sim_params['H_prev']:.2f}.h5"
   h5_step = system.inscribe_part_group_to_h5(group_type=[Elastomer], h5_data_path=os.path.join(OUTPUTPATH, CFPNAME), mode='NEW')

   current_time = time.monotonic_ns()
   print(f"-- Inscribed h5 -- {(current_time-previous_time) // 1_000_000_000} (seconds)")
   previous_time = current_time

   # save initial configuration
   assert h5_step == 0
   sim_time = 0
   write_and_flush_h5(system, sim_time)

   current_time = time.monotonic_ns()
   print(f"-- Write step 0 h5 -- {(current_time-previous_time) // 1_000_000_000} (seconds)")
   previous_time = current_time

   # Magnetize particles before starting to simulate. This means that the external field is taken into account from the start of the first integration step
   system.sys.integrator.run(0, recalc_forces=True)

   # continues after else block - which is the case for when initializing from a checkpoint

   # stuff for checkpointing with espresso checkpoint
   system_sys= system.sys
   system_part_types= dict(system.part_types)
   system_seed= system.seed
   system_kT= system.kT
   elastomer_n_parts= elastomer.params['n_parts']
   TRACK_TOTAL_TIME_NS = 0

else: # Only meant to keep simulation going. System is not really initialized like one might expect
   print(f"WARNING: There is a valid checkpoint file. Initializing from checkpoint. Taking H as inputed {sim_params['H']} and ignoring all other inputs; and resuming simulation from the last checkpoint in {CHECKPOINT_DIR}")

   system = Simulation(box_dim=[10]*3) # hold size

   checkpoint.load()

   print("sim_params_cp", sim_params_cp)

   system.rebind_sys(system_sys)
   system.part_types.update(system_part_types)
   system.seed= system_seed
   system.kT = system_kT

   print(f"sim_time {sim_time}, system.sys {system.sys}, len(system.sys.part.all()) {len(system.sys.part.all())}, system.sys.part.by_id(0).pos {system.sys.part.by_id(0).pos}, (part_type: n_parts_type) {{{(lambda types: {typ: len(system.sys.part.select(type=typ)) for typ in types})(set(system.sys.part.all().type))}}}, system.part_types {system.part_types}, system.seed {system.seed}, elastomer_n_parts {elastomer_n_parts}, OUTPUTPATH {OUTPUTPATH}, CFPNAME {CFPNAME}")

   # Add extra io settings for saves (see default)
   system.io_dict['bonds'] = "all"
   system.io_dict['properties'].extend([('fix', 3)])
   print(system.io_dict)

   #### output nice system stuff ####

   print(system.sys.cell_system.get_state())

   # Checkpoint pressomancy stuff
   if sim_params['H_prev'] is None:
      CFPNAME = f"cp{sim_params['A']}-{system.seed}-H{sim_params['H']:.2f}.h5"
   else:
      CFPNAME = f"cp{sim_params['A']}-{system.seed}-H{sim_params['H']:.2f}-fromH{sim_params['H_prev']:.2f}.h5"
   step_h5 = system.inscribe_part_group_to_h5(group_type=[Elastomer], h5_data_path=os.path.join(OUTPUTPATH, CFPNAME), mode='LOAD_NEW')

   # STABILIZE MAE WITH MAGNETIC FIELD
   ext_B_z = sim_params['H']
   H_ext = [0.,0.,ext_B_z]
   system.set_H_ext(H=H_ext)

   # Mark particles to magnetize
   parts_to_magnetize= list(system.sys.part.select(type=system.part_types['pds_virt'])) if 'pds_virt' in system.part_types else []

   print("n_parts_to_magnetize", len(parts_to_magnetize))

   current_time = time.monotonic_ns()
   print(f"-- Loaded checkpoint -- {(current_time-previous_time) // 1_000_000_000} (seconds)")
   previous_time = current_time

print("OUTPUTPATH", OUTPUTPATH)
print("H_ext:", system.get_H_ext())

system.sys.time_step = sim_params['sys_time_step'] # 0.001 or 0.0005 or 0.0008

# Sample and integration steps
steps_per_time = get_steps_per_time(system.sys.time_step)
total_sim_time = sim_params['total_sim_time']
time_start = sim_time
print(f"time_start {time_start} of total_sim_time {total_sim_time} | steps_per_time {steps_per_time}")
assert isinstance(total_sim_time, int) and isinstance(time_start, int) and isinstance(steps_per_time, int)
integrate_start_time = time.monotonic_ns()
if time_start < total_sim_time: # if there are samples to run
   # actually run the things
   print("mag_func int for", sim_params['A'], mag_func_to_int.get(sim_params['A'], None))
   save_h5_each_times  = sim_params["save_h5_each"]
   cp_each_times = max(int(sim_params["checkpoint_each"]), 1) if sim_params["checkpoint_each"] != 0 else 0
   print(f"save_h5_each {save_h5_each_times} and checkpoint_each {cp_each_times}")

   if cp_each_times == 0: # no checkpointing (NONEEEEE!)
      # to resume simualtion from checkpoint, it is best to re-use forces, to keep kinetics. Also reuse forces, since they were pre-calculated if not checkpoint
      system.sys.integrator.run(steps_per_time, reuse_forces=True)
      if STOP_REQUESTED:
         system.io_dict['h5_file'].close()
         sys.exit(1)
      sim_time += 1
      if (sim_time %  save_h5_each_times == 0):
         write_and_flush_h5(system, sim_time)
      for _ in tqdm(range(sim_time, total_sim_time+1)):
         system.sys.integrator.run(steps_per_time)
         if STOP_REQUESTED:
            system.io_dict['h5_file'].close()
            sys.exit(1)
         sim_time += 1
         if (sim_time %  save_h5_each_times == 0):
            write_and_flush_h5(system, sim_time)
      current_time = time.monotonic_ns()
      TRACK_TOTAL_TIME_NS += (current_time - previous_time)
      previous_time = current_time

   else:
      if not checkpoint_loaded:
         sim_params_cp = sim_params
         checkpoint.register("system_sys")
         checkpoint.register("system_part_types")
         checkpoint.register("system_seed")
         checkpoint.register("system_kT")
         checkpoint.register("elastomer_n_parts")
         checkpoint.register("sim_time")
         checkpoint.register("OUTPUTPATH")
         checkpoint.register("CFPNAME")
         checkpoint.register("magnetize_func")
         checkpoint.register("TRACK_TOTAL_TIME_NS")
         checkpoint.register("sim_params_cp")
         print("registered checkpoint stuff")

      # to resume simualtion from checkpoint, it is best to re-use forces, to keep kinetics. Also re-use forces, because they were pre-computed if not checkpoint
      system.sys.integrator.run(steps_per_time, reuse_forces=True)
      if STOP_REQUESTED:
         system.io_dict['h5_file'].close()
         sys.exit(1)
      sim_time += 1
      if (sim_time %  save_h5_each_times == 0):
         write_and_flush_h5(system, sim_time)
      if (sim_time % cp_each_times == 0):
            current_time = time.monotonic_ns()
            TRACK_TOTAL_TIME_NS += (current_time - previous_time)
            previous_time = current_time
            checkpoint.save()
      for _ in tqdm(range(sim_time, total_sim_time+1)):
         system.sys.integrator.run(steps_per_time)
         if STOP_REQUESTED:
            system.io_dict['h5_file'].close()
            sys.exit(1)
         sim_time+= 1
         if (sim_time %  save_h5_each_times == 0):
            write_and_flush_h5(system, sim_time)
         if (sim_time % cp_each_times == 0):
            current_time = time.monotonic_ns()
            TRACK_TOTAL_TIME_NS += (current_time - previous_time)
            previous_time = current_time
            checkpoint.save()
      system.io_dict['h5_file'].close() # safely flush and close h5 file
      if (sim_time % cp_each_times != 0):
         current_time = time.monotonic_ns()
         TRACK_TOTAL_TIME_NS += (current_time - previous_time)
         previous_time = current_time
         checkpoint.save()
else: # if there are no samples to run
   print("Already did this shit. What the fuck - don't do that again, this simulation is over; or make n_samples larger")
print(f"->{sim_time} simulation time units simulated, from {total_sim_time} s.t.u<-")
print(f"time simulated in this run: {(sim_time - time_start)}")

current_time = time.monotonic_ns()
minutes = (current_time - integrate_start_time) // 1_000_000_000 // 60
print(f"-- Simulated {(sim_time - time_start)} steps -- {minutes // 60} (hours) {minutes} (minutes)")

seconds = ( TRACK_TOTAL_TIME_NS + (current_time - previous_time) ) // 1_000_000_000
minutes = seconds // 60
hours = minutes // 60
print(f"-- Total simulation time - total steps {sim_time} -- {hours // 24} (days) {hours % 24} (hours) {minutes % 60} (minutes) {seconds % 60} seconds")