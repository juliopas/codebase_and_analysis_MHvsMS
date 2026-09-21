import time
global_start_time = time.monotonic_ns()

import espressomd
import espressomd.checkpointing

espressomd.assert_features(['WCA', 'ROTATION',
                            'EXTERNAL_FORCES',
                            'CUDA'])

from pressomancy.simulation import Simulation, Elastomer

import numpy as np

import os
import argparse

from tqdm import tqdm

# Default data location, resolved relative to the repository (see
# SCRIPTS/repo_paths.py). --SAMPLE_DIR and --OUT_DIR override it.
from repo_paths import REPO_ROOT, DATA_DIR

# Simulation parameters
parser = argparse.ArgumentParser()
parser.add_argument("--N_FULL_BOX", action="store", dest="N_FULL_BOX", type=int, help="Numer of particles that would fill a square box at the chosen density.")

parser.add_argument("--DENS", action="store", dest="DENS", type=float, help="Volume density of magnetic particles.")
parser.add_argument("--K", action="store", dest="K", type=str, help="Spring constant type of bonds (soft or hard)")
parser.add_argument("--HEIGHT", action="store", dest="HEIGHT", type=int, help="Thickness of the MAE layer (height in the z direction) - in number of particle diameters.")

parser.add_argument("--seed", action="store", dest="seed", type=int, help="Random number generator MAE layer sample seed.")

parser.add_argument("--SAMPLE_DIR", action="store", dest="SAMPLE_DIR", type=str, default=os.path.join(DATA_DIR, "samples"), help="Path of the sample directory. Defaults to DATA/samples inside the repository.")
sim_params=vars(parser.parse_args())

print(sim_params)

SAMPLE_DIR = sim_params['SAMPLE_DIR']
os.makedirs(SAMPLE_DIR, exist_ok=True)

# System params
DENS = sim_params['DENS']
HEIGHT = sim_params['HEIGHT'] # in part size units
N_FULL_BOX = sim_params['N_FULL_BOX']

SIZE = None
SIGMA = 1. if SIZE is None else SIZE * 0.890898718 # else SIZE / 2^(1/6)
SIZE = SIGMA * 1.12246205 if SIZE is None else SIZE # SIGMA * 2^(1/6) if size is not inputed

RADIUS = SIGMA / 2

BONDS_MAX_LENGHT = 5. # Default 5.
K = sim_params['K']
valid_bond_limits = {'soft': (0.001, 0.01), 'hard': (0.01, 0.1)}
BOND_LIMITS = valid_bond_limits.get(K)
if BOND_LIMITS is None:
   raise ValueError(f"Invalid bond type: {K}. Valid --K values -> {valid_bond_limits.keys()}")

MAE_LAYER_HEIGHT = HEIGHT * SIZE # in system units
BOX_SIZE = np.cbrt( N_FULL_BOX * 4/3*np.pi / 0.3 ) * RADIUS
BOX_Z_MAX = 4 * MAE_LAYER_HEIGHT

N_PART = round(DENS * BOX_SIZE**2 * MAE_LAYER_HEIGHT / ( 4/3 * np.pi * RADIUS**3))

assert MAE_LAYER_HEIGHT<=BOX_Z_MAX
assert DENS >= 0.2 and DENS<=0.3

box_l = [BOX_SIZE, BOX_SIZE, BOX_Z_MAX]

dens = N_PART * 4/3 * np.pi * RADIUS**3 / (BOX_SIZE * BOX_SIZE * MAE_LAYER_HEIGHT)

assert dens - DENS < 0.005, f"{dens} - {DENS}"

print("--- Initial parameters ---")
print(f"-> dens {dens} K {K} HEIGHT(MAE_LAYER_HEIGHT) {HEIGHT}({MAE_LAYER_HEIGHT}) N_FULL_BOX(N_PART) {N_FULL_BOX}({N_PART}) seed {sim_params['seed']}")
print(f"-> SIGMA {SIGMA} SIZE {SIZE} BONDS_MAX_LENGHT {BONDS_MAX_LENGHT} BOND_LIMITS {BOND_LIMITS}")
print("--------------------------")

# INITIALIZE SYSTEM
system = Simulation(box_dim=box_l)
system.seed = sim_params['seed']
system.set_sys(time_step=0.001)

config_E = Elastomer.config.specify(layer_height=MAE_LAYER_HEIGHT, n_parts=N_PART, bond_K_lims=BOND_LIMITS, size=SIZE, sigma=SIGMA, espresso_handle=system.sys, seed=system.seed)
elastomer=[Elastomer(config=config_E) for _ in range(1)]
system.store_objects(elastomer)
system.set_objects(elastomer)
elastomer= elastomer[0]

system.io_dict['bonds'] = 'all' # save bonds
system.io_dict['properties'].append(("fix",3))

system.set_steric(("real",), sigma=SIGMA)

print(f"box_E {elastomer.params['box_E']}")

# save initial config
CFPNAME = f"A-{DENS:.2f}_{K}-{N_FULL_BOX}_{HEIGHT}-{sim_params['seed']}.h5"
h5_step = system.inscribe_part_group_to_h5(group_type=[Elastomer], h5_data_path=os.path.join(SAMPLE_DIR, CFPNAME), mode='NEW')
assert h5_step == 0
system.write_part_group_to_h5(step=h5_step)

current_time = time.monotonic_ns()
print(f"-- Created elastomer -- {(current_time-global_start_time) // 1_000_000_000} (seconds)")
previous_time = current_time

#elastomer.mix_elastomer_stuff()
# add iniziatilation process, to get a nice random distribution before bonding
from pressomancy.helper_functions import add_box_constraints_func, remove_box_constraints_func
old_time_step= float(elastomer.sys.time_step)
n_iter_1 = int(1000000 * 10)
elastomer.sys.time_step = 0.0001
if elastomer.substrate is None:
   raise ValueError("Substrate must be created before mix_elastomer_stuff().")
# Add temporary wall (top and bottom only).
types_M = tuple(typ for key, typ in elastomer.part_types.items() if "real" in key)
add_box_constraints_func(
   sides=['top', 'bottom'],
   top=elastomer.params['box_E'][2],
   bottom=elastomer._substrate_size,
   inter='wca',
   types_=types_M,
   sys=elastomer.sys,
)
elastomer.sys.thermostat.set_langevin(kT=1e-3, gamma=10, seed=elastomer.params['seed'])
elastomer.sys.integrator.run(n_iter_1)
# Remove temporary box particles
remove_box_constraints_func(sys=elastomer.sys)
elastomer.sys.thermostat.turn_off()
elastomer.sys.time_step = old_time_step

current_time = time.monotonic_ns()
print(f"-- Mixed elastomer -- {(current_time-global_start_time) // 1_000_000_000} (seconds)")
previous_time = current_time

elastomer.cure_elastomer()

current_time = time.monotonic_ns()
print(f"-- Cured elastomer -- {(current_time-global_start_time) // 1_000_000_000} (seconds)")
previous_time = current_time

# Save bare material sample
system.write_part_group_to_h5(step=h5_step, bonds_once=False)

current_time = time.monotonic_ns()
minutes = (current_time-previous_time) // 1_000_000_000 // 60
print(f"-- Saved elastomer sample -- {minutes // 60} (hours) {minutes} (minutes)")

seconds = (current_time-global_start_time) // 1_000_000_000
minutes = seconds // 60
hours = minutes // 60
print(f"-- Total sample creation time -- {hours // 24} (days) {hours % 24} (hours) {minutes % 60} (minutes) {seconds % 60} seconds")