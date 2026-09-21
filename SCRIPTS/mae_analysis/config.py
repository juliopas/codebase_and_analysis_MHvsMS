"""Shared run-time knobs for mae_analysis: paths, sweeps, units, style, toggles.

This is the ONE file you edit per run for anything cross-cutting: which datasets,
where they live, the unit system, figure style, and the calc/plot/reverse
toggles. Each analysis family keeps its own do_*/plot_* switches in its own
config (surface/config.py, cluster/config.py), which re-export everything here.
Nothing here triggers analysis at import time, so subprocess workers re-import it
cheaply to recover every setting.

IMPORTANT: do not change this file while mae_analysis is running.

Sections
--------
0. CLI toggles       : do_CALCULATE, plot_PLOT, REVERSE (set by __main__ flags / env)
1. Directories       : where data is read from / written to (reverse-aware)
2. Sample parameters : the physical parameter grids (A, ts, DENS, K, H, ...)
3. Reverse params    : A_CONC + H0_list pairing for reverse-field runs
4. Units             : simulation -> SI unit system (switchable)
5. Matplotlib style  : font sizes, marker size, output format
"""

import os
import math

from pyanal.units import UnitSystem, SIM

import repo_paths as _repo_paths

VERBOSE = False


def _env_bool(name, default):
    """Boolean override read from the environment (set by the CLI in `__main__`).

    Lets CLI flags flip a default without editing this file; subprocess workers
    inherit the environment, so their re-imported config stays in sync.
    """
    val = os.environ.get(name)
    if val is None:
        return default
    return val not in ("0", "", "false", "False", "no")


# ============================================================== 0. CLI toggles ==
do_BOND_TO_PARTS  = _env_bool("MAE_DO_BONDTOPARTS", False)# create bond-to-parts h5 file (--bond-to-parts)
do_CALCULATE      = _env_bool("MAE_DO_CALCULATE", True)   # False -> read processed file (--plot-only)
plot_PLOT         = _env_bool("MAE_PLOT", True)           # False -> no figures (--no-plot)
REVERSE           = _env_bool("MAE_REVERSE", False)       # reverse-field run (--reverse)
do_OVERWRITE_DATA = _env_bool("MAE_OVERWRITE", False)   # overwrite the data selected below
plot_ONLY_SAMPLE_H= _env_bool("MAE_PLOT_ONLY_SAMPLES", False) # plot only sample_H plots (1)
AUTO_XYLIM           = True # use matplotlib default axes limits (enures all data is visible)
PER_A_COLOR_FAMILIES = False # each A gets its own colospace

# ============================================================ 1. Directories ==
# Paths derive from the repository location (see SCRIPTS/repo_paths.py), so a
# clone works wherever it is unpacked; set $MAE_DATA_DIR to read the dataset from
# outside the repository. Reverse runs read/write a parallel set (sim_data-h5 /
# figs-reverse) so they never clobber forward data.
DATA_DIR = _repo_paths.DATA_DIR

OUTPUTPATH = os.path.join(DATA_DIR, "figs-reverse" if REVERSE else "figs")
os.makedirs(OUTPUTPATH, exist_ok=True)

BOND_PART_DIR = os.path.join(OUTPUTPATH, "bond_part_h5_files")
os.makedirs(BOND_PART_DIR, exist_ok=True)
DATA_ARAY_DIR = os.path.join(OUTPUTPATH, "data_aray")
os.makedirs(DATA_ARAY_DIR, exist_ok=True)

FORWARD_READ_BASE_DIR = os.path.join(DATA_DIR, "MAE-BoS", "cluster", "sim_data")
REVERSE_READ_BASE_DIR = os.path.join(DATA_DIR, "MAE-BoS", "cluster", "sim_data")
READ_BASE_DIR = REVERSE_READ_BASE_DIR if REVERSE else FORWARD_READ_BASE_DIR

# ===================================================== 2. Sample parameters ==
N_FULL_BOX = 6000 * 4   # ignore for eager sims
sampled_time_per_step = 0.5

A_list = ["HM","SM"]
ts_list = [500]#range(1000, 1000 + 1, 100)
DENS_list = [0.20]
K_list = ["soft"]
HEIGHT_list = [10]
H_list =  [0,0.2,0.4,0.6,0.8,1,1.5,2,3,4,6,10,15,20,25,30,40,50,60,70]; H_list = H_list[:-4]
seed_list = [1, 2, 3, 4]

SIZE = 1
SIZE_BOND_PART = 0.9
ITIM_DIAMETER = SIZE * 0.2
ITIM_DIAMETER_INTERP = ITIM_DIAMETER * 3   # laser-emulation node-selection probe; smaller = spikier

SUBSTRATE_Z_OFFSET = 1  # bottom of MAE layer should be above 0, but close to

# ======================================================== 3. Reverse params ==
# Reverse-field runs read files named cp{A}-{seed}-H{H}-fromH{H0}.h5 from a
# "{A_help}-{A_CONC}-{DENS}_{K}-{N_FULL_BOX}_{HEIGHT}-reverse" directory. H0 is
# paired 1:1 with H (start at H0, relax down to H). Used only when REVERSE.
# When running --reverse, set H_list / H0_list to the paired reverse sweep, e.g.:
#   H_list  = [0,   0.2, 0.4, 0.6, 0.8, 1,   1.5, 2, 3, 4, 6 ]
#   H0_list = [0.2, 0.4, 0.6, 0.8, 1,   1.5, 2,   3, 4, 6, 10]
H0_list = H_list[1:] + [H_list[-1]]

if REVERSE:
    assert len(H_list) == len(H0_list), \
        f"reverse needs H_list and H0_list paired 1:1 (got {len(H_list)} vs {len(H0_list)})"

# =================================================================== 4. Units ==
## Simulation units -> SI units
unit_L   = 3e-6                     # m
unit_mag = 2.7030920551246454e-11   # Am^2
unit_Mag = 1001145.142125713        # A/m
unit_B   = 0.009086930786900675     # T
unit_E   = 4.326041443131127e-18    # J
unit_T   = 313333.9062376554        # K (assumes k_B=1 in espresso)
unit_M   = 1.1131605249464713e-13   # kg
# time is then determined (in seconds)
unit_t   = unit_L * math.sqrt(unit_M / unit_E)  # s
unit_t   = sampled_time_per_step * unit_t

SI = UnitSystem(
    scales={
        "length":        unit_L,
        "energy":        unit_E,
        "field":         unit_B,
        "moment":        unit_mag,
        "magnetization": unit_Mag,
        "volume":        unit_L ** 3,
        "density":       1.0 / unit_L,
        "time":          unit_t,
    },
    labels={
        "length":        f" [{ r"$\mathrm{m}$" }]",
        "energy":        f" [{ r"$\mathrm{J}$" }]",
        "field":          f" [{ r"$\mathrm{T}$" }]",
        "moment":        f" [{ r"$\mathrm{A}m^{2}$" }]",
        "magnetization": f" [{ r"$\mathrm{A/m}$" }]",
        "volume":        f" [{ r"$\mathrm{m}^3$" }]",
        "density":       f" [{ r"$\mathrm{m}^{-1}$" }]",
        "time":          f" [{ r"$\mathrm{t}$" }]",
    },
    name="SI",
)

## Nondimensional units -- used only by the surface `plot_nondim_h` Langevin plot.
# Each quantity is expressed in its natural reduced unit.
TIME      = 0.5 # each sampled ts is 0.5 sim units
MASS      = 1.0
MU        = 1.0                # single-particle dipole moment (sim units)
K_BT      = 1.61e-8               # thermal energy (sim units)
M_SAT_REF = MU / (math.pi / 6 * SIZE ** 3)   # bulk saturation magnetization (sim units)

NONDIM = UnitSystem(
    scales={
        "field":      MU / K_BT,        # H -> xi = mu*H/k_BT
        "energy":        1.0 / K_BT,       # E -> E/k_BT
        "magnetization": 1.0 / M_SAT_REF,  # M -> M/M_s
        "moment":        1.0 / MU,         # m -> m/m_max
        "length":        1.0 / SIZE,              # L -> L/size
        "volume":        1.0 / SIZE**3,
        "density":       SIZE,
        "time":          1.0 / TIME,
    },
    labels={
        "field": "",          # xi axis label is self-contained (set on the AxisSpec)
        "energy":        r"$/k_{\text{B}}\text{T}$",
        "magnetization": r"$/M_{\text{s}}$",
        "moment":        r"$/\mu_{0}$",
        "length": r"$/d$",
        "volume": r"$/d^{3}$",
        "density": r"$d$"
    },
    name="nondim",
)

UNIT_SYSTEM = SIM   # switch here: SIM (sim units) or SI

# ======================================================== 5. Matplotlib style ==
fontsize_ticks  = 14 * 2
fontsize_legend = 16 * 2
fontsize_label  = 22 * 2

markersize_1 = 8 * 2

savefig_format = ".png"
dpi = 600

# Pool for color families for the different K values in sample plots. None for default
# if PER_A_COLOR_FAMILIES
A_CM_POOL = None#["Reds", "Oranges"]
if A_CM_POOL is not None and PER_A_COLOR_FAMILIES:
    assert len(A_CM_POOL) >= len(A_list) * len(K_list), f"{len(A_CM_POOL)} < {len(A_list) * len(K_list)}"
A_LS_POOL = None#[":", "--"] # None is default

# H is an ordered (~log-spaced) sweep parameter, so its colours should read as a
# progression. "sequential" samples H_SEQUENTIAL_CMAP by rank; "categorical" uses the
# colour-blind-safe (Okabe-Ito) qualitative palette. Used by both the surface
# and cluster plots via style.H_color_map.
H_COLORMAP_MODE = "categorical"   # "sequential" | "categorical"
# Any colormap name available to matplotlib (includes compatible colormap libraries eg. colorcet)
H_SEQUENTIAL_CMAP = "plasma" #"cet_CET_CBD2"
# inner sampling range for the sequential map (avoids the near-white / near-black ends)
H_SEQUENTIAL_RANGE = (0.12, 0.92)
# Okabe-Ito qualitative palette -- colour-blind safe, ordered for sequence.
H_CATEGORICAL_COLORS = [
    "#E69F00", "#56B4E9", "#009E73", "#D55E00",
    "#0072B2", "#CC79A7", "#F0E442", "#000000",
]
# alpha of the emphasised histogram fills + the patch_alpha legend swatch
HIST_FILL_ALPHA = 0.3
