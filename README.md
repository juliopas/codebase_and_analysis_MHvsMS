# Code and data: magnetically hard vs. soft particles in magneto-active elastomers

Simulation code, analysis code and simulation data accompanying the manuscript

> **Filler restructuring and surface roughness in magneto-active elastomers with
> magnetically hard versus soft particles**

This repository contains everything needed to re-run the simulations, re-run the
analysis, and regenerate the figures of the manuscript.

- **Authors:** Júlio P. A. Santos
- **DOI:**  to be filled in.
- **Contact:** julio.palma.de.assuncao.santos@univie.ac.at

---

## Data availability

The code lives here; the **dataset is distributed separately** because it is
2.0 GB, which exceeds what GitHub is meant to hold (and two files exceed
GitHub's 100 MB per-file limit).

> **Dataset:** 10.5281/zenodo.22872568

To use it, download and unpack the archive so that its `DATA/` directory sits at
the root of this repository:

```
codebase_and_analysis_MHvsMS/
├── DATA/          <- unpacked here
├── SCRIPTS/
├── espresso/
└── pressomancy/
```

If you prefer to keep it elsewhere (a scratch filesystem, say), point the
`MAE_DATA_DIR` environment variable at it instead:

```bash
export MAE_DATA_DIR=/scratch/mae-data
```

Every path in this repository is resolved relative to the repository root, via
[`SCRIPTS/repo_paths.py`](SCRIPTS/repo_paths.py). Nothing is tied to a particular
machine, user or working directory.

---

## Contents

| Path | What it is |
|---|---|
| `SCRIPTS/MAE-BoS-create.py` | Builds a cured elastomer sample (crosslinked matrix + magnetic filler) |
| `SCRIPTS/MAE-BoS-run.py` | Runs one simulation point: one sample under one external field |
| `SCRIPTS/mae_analysis/` | The analysis package — surface and cluster families, plotting, configuration |
| `SCRIPTS/pyanal/` | Supporting library: energies, ITIM surface extraction, bond-to-particle conversion, height maps |
| `SCRIPTS/repo_paths.py` | Repository-relative path resolution |
| `requirements.txt` | Analysis dependencies |
| `LICENSE` | GPL-3.0-or-later, the licence for all code here |
| `SCRIPTS/data_check.ipynb` | Integrity sweep over the simulation data; its saved output records the damaged files listed below |
| `SCRIPTS/rms.ipynb` | Surface RMS roughness against the Langevin parameter |
| `SCRIPTS/pressomancy_h5_tool.ipynb` | Converts simulation checkpoints to VTK for visualisation |
| `SCRIPTS/tmp.ipynb` | Working notebook comparing simulated and experimental magnetisation |
| `espresso/` | The ESPResSo 5.0.0 source tree the simulations were built against |
| `pressomancy/` | Pressomancy 1.0.0 — the object layer wrapping ESPResSo |
| `DATA/` | The dataset (not in this repository; see **Data availability**) |

The ESPResSo build tree (`espresso/build/`) and any local virtual environment are
not tracked; both are platform-specific and must be rebuilt locally.

Some notebooks additionally read data that is **not** part of the published
dataset — `rms.ipynb` and `tmp.ipynb` use experimental magnetisation curves, and
`pressomancy_h5_tool.ipynb` writes VTK output to directories that are created on
demand. They are included because they document how the corresponding figures
were produced, not because they run unmodified from the archive alone.

---

## Setup

**To reproduce the figures you do not need to build ESPResSo.** The analysis
reads the published HDF5 files directly and depends only on packages available
from PyPI:

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install ./pressomancy ./SCRIPTS/pyanal
```

That is the whole install. It was verified from a clean environment on
Python 3.13 with no ESPResSo present.

**To re-run the simulations** you additionally need ESPResSo built from the
`espresso/` source tree here. Simulation scripts are then launched through its
`pypresso` wrapper, which puts the compiled `espressomd` module on the path:

```bash
cd espresso && mkdir -p build && cd build
cmake .. && make -j          # see espresso/doc for the full instructions
```

The simulations were produced with Python 3.12 and ESPResSo 5.0.0 on Linux
(NumPy 2.4.4, SciPy 1.17.1, h5py 3.16.0, Cython 3.0.11), with Pressomancy 1.0.0
and pyanal 0.1. The build tree is platform-specific and is not distributed;
build it for your own machine.

---

## Reproducing the analysis

The analysis reads `DATA/` and writes figures and intermediate arrays to
`DATA/figs/`. Sweeps, unit system and plot style all live in one place,
[`SCRIPTS/mae_analysis/config.py`](SCRIPTS/mae_analysis/config.py); the
command-line flags only select what runs.

```bash
cd SCRIPTS

python -m mae_analysis --surface        # surface roughness / height maps
python -m mae_analysis --cluster        # filler cluster structure
python -m mae_analysis --hysteresis     # overlay forward and reverse field sweeps
python -m mae_analysis                  # all of the above
```

Useful flags:

| Flag | Effect |
|---|---|
| `--plot-only` | Skip the (slow) calculation, reload the saved arrays and re-plot |
| `--no-plot` | Calculate only, write no figures |
| `--overwrite` | Overwrite the stored result arrays with this run's |
| `--file-id N` | Pick saved array *N* non-interactively; pairs well with `--plot-only` |
| `--bond-to-parts` | Regenerate `DATA/figs/bond_part_h5_files/` (slow; not shipped, see below) |
| `--reverse` | Read the reverse-field sweep and write to a parallel output directory (that sweep is not part of the published dataset) |

Start with `--plot-only`: the stored arrays under `DATA/figs/data_aray/` and
`DATA/figs/data_aray_cluster/` already hold the processed results, so the figures
come back in seconds without re-reading the full dataset.

## Reproducing the simulations

Each simulation point is one sample under one field, run independently — the
sweep is trivially parallel and was run as a job array on a cluster.

```bash
cd SCRIPTS

# 1. Build a cured sample
pypresso MAE-BoS-create.py --N_FULL_BOX 24000 --DENS 0.20 --K soft --HEIGHT 10 --seed 1

# 2. Run that sample under an external field
pypresso MAE-BoS-run.py --A HM --N_FULL_BOX 24000 --DENS 0.20 --K soft --HEIGHT 10 \
                        --seed 1 --H 1.0 --total_sim_time 500 --save_h5_each 1
```

`--A` selects the particle model: `HM` for magnetically hard (permanent dipoles,
directory prefix `pdp`), `SM` for magnetically soft (induced dipoles, prefix
`pds`). `--SAMPLE_DIR` and `--OUT_DIR` default to the corresponding directories
inside `DATA/`, so neither needs to be given for a standard run.

---

## The dataset

### Layout

```
DATA/
├── MAE-BoS/cluster/sim_data/
│   └── {pdp,pds}-{DENS}_{K}-24000_10/cp{A}-{seed}-H{H}.h5    simulation configurations
├── samples/
│   └── A-{DENS}_{K}-24000_10-{seed}.h5                       cured samples, before any field
└── figs/
    ├── data_aray/            processed surface results ({A}--{sweep}-{value}.h5)
    ├── data_aray_cluster/    processed cluster results
    └── bond_part_h5_files/   empty — regenerate with `--bond-to-parts`
```

### Parameter grid

| Parameter | Values |
|---|---|
| Particle model `A` | `HM` (magnetically hard, `pdp`), `SM` (magnetically soft, `pds`) |
| Filler volume fraction `DENS` | 0.20, 0.25, 0.30 |
| Matrix stiffness `K` | `soft`, `hard` |
| Layer thickness `HEIGHT` | 10 particle diameters |
| Box filling `N_FULL_BOX` | 24000 |
| Seed | 1, 2, 3, 4 |
| External field `H` | 0, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 10, 15, 20, 25, 30 |

That is 760 configuration files and 24 samples. Forward field sweeps only.
Higher fields (H = 40, 50, 60, 70) were simulated but are not included, as the
manuscript does not use them.

### What the files contain

Each simulation ran for 500 sampled time steps, saving every step. **The
published files keep only the final configuration**, at `time == 500` — the
equilibrated state the analysis uses. The sample files likewise keep only their
last save, the crosslinked one.

Within a kept frame nothing is altered: particle positions, folded positions,
dipole moments, directors, forces, fixing constraints, ids, image-box indices and
types are all bit-identical to the corresponding frame of the full trajectory, as
are the `connectivity/` tables and the `sys` / `sim_inst` attributes (`box_l`,
`periodicity`, `time_step`, `kT`, `seed`). Files keep the original gzip-4,
one-frame-per-chunk layout. Dropping the other 501 frames takes the configuration
files from 183 GB to 1.6 GB; the dataset as a whole is 2.0 GB.

> **Note on reading bonds.** The `particles/Elastomer/bonds` dataset uses a
> variable-length compound dtype containing a variable-length string. Some h5py
> builds cannot convert it and raise
> `TypeError: Data type conversion failed` on any read — this affects the
> original files as much as the published ones. `MAE-BoS-run.py` catches that and
> substitutes empty bond lists, which silently yields an unbonded system, so
> check that bonds load before relying on them. `h5dump` reads the dataset
> correctly regardless. In the sample files the `bonds` group retains both saves
> while the other groups retain one; the loader indexes it with `-1`, so the
> crosslinked save is still the one used.

### Known gaps

`pdp-0.30_soft-24000_10` and `pdp-0.30_hard-24000_10` hold 60 files each rather
than 64. Eight magnetically-hard, seed-1 points at `DENS = 0.30` are unusable in
the raw simulation output and are therefore absent:

| File | Problem |
|---|---|
| `pdp-0.30_hard/cpHM-1-H0.40.h5` | corrupt — the file will not open |
| `pdp-0.30_hard/cpHM-1-H0.80.h5` | corrupt — the file will not open |
| `pdp-0.30_hard/cpHM-1-H1.50.h5` | run terminated early, at t = 368 of 500 |
| `pdp-0.30_hard/cpHM-1-H3.00.h5` | run terminated early, at t = 373 |
| `pdp-0.30_soft/cpHM-1-H0.20.h5` | run terminated early, at t = 366 |
| `pdp-0.30_soft/cpHM-1-H0.40.h5` | run terminated early, at t = 374 |
| `pdp-0.30_soft/cpHM-1-H0.60.h5` | run terminated early, at t = 376 |
| `pdp-0.30_soft/cpHM-1-H30.00.h5` | run terminated early, at t = 375 |

The magnetically-hard model at `DENS = 0.30` therefore has three independent
seeds at those six fields and four everywhere else. `SCRIPTS/data_check.ipynb`
reproduces this check over the whole dataset.

---

## Licence

**Code in this repository: GNU General Public License v3.0 or later
(GPL-3.0-or-later).** The full text is in [LICENSE](LICENSE).

In short: you are free to use, modify and redistribute this code, including for
commercial purposes, provided you keep it under the GPL and pass on the source.

**Data: Creative Commons Attribution 4.0 International (CC BY 4.0).** The dataset
is a separate deposit (see [Data availability](#data-availability)) and is not
covered by the GPL — it is measurement output, not a derived work of the code.
CC BY 4.0 lets anyone reuse it with attribution, which is what most journals and
funders now expect of a data-availability statement.

## Citing

If you use this code or data, please cite the manuscript, and cite ESPResSo and
Pressomancy separately — for ESPResSo as requested in `espresso/CITATION.cff`.
