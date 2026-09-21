"""Cluster-family run-time flags + cluster-specific dirs/constants.

Edit the cluster do_*/plot_* switches here. Everything cross-cutting (paths,
sweeps, units, style, calc/plot/reverse toggles) lives in the shared
``mae_analysis/config.py`` and is re-exported below, so cluster modules import a
single ``from .config import ...`` namespace (mirrors surface/config.py).
"""

import os
from types import MappingProxyType

from ..config import *          # noqa: F401,F403  -- re-export shared knobs
from ..config import OUTPUTPATH, plot_PLOT   # names referenced explicitly below

# --- Bulk/surface separation ---
do_MANUAL_BULK_SURFACE = False
do_USE_DUMB_THRESHOLD = False

# --- Cluster calculation flags ---
do_ENERGETIC_CLUSTER = False
do_CLUSTER = True
do_BOP = True
do_ENERGETIC_BOP = False
do_G2 = False   # not implemented

# --- Plotting flags ---
plot_H = True

if not plot_PLOT:               # --no-plot cascade
    plot_H = False

# --- H-histogram overlay style ---
# Every H gets a filled body and: emphasised H at HIST_FILL_ALPHA,
# the rest at the lower HIST_FILL_ALPHA_LOW. Emphasised H also get a bold outline; the others
# get thin top caps only. Integer-binned figures (num_neighbours, q_l) draw bars centred on
# the value; continuous ones draw stairs over bin edges.
# H_EMPHASIS = None -> emphasise [min(H_list), max(H_list)]; else a list of H.
H_EMPHASIS: None | list = None
# HIST_OUTLINE_STYLE applies to EMPHASISED curves only (and only to the continuous figures;
# integer bars always use top caps): "cap" -> top line only, "step" -> full step outline.
HIST_OUTLINE_STYLE = "cap"      # "cap" | "step"
HIST_FILL_ALPHA_LOW = 0.10      # fill alpha for non-emphasised H
HIST_CAP_LW = 1.6               # cap linewidth for emphasised H
HIST_CAP_LW_THIN = 0.6          # cap linewidth for non-emphasised H

# --- Cluster-specific directories & constants ---
# Cluster results live in their own ARay store, separate from the surface one.
DATA_ARAY_DIR = os.path.join(OUTPUTPATH, "data_aray_cluster")
os.makedirs(DATA_ARAY_DIR, exist_ok=True)

l_list = list(range(2, 12 + 1, 1))   # Steinhardt bond-order-parameter degrees


def _flags_with_prefix(prefix):
    """Collect the module-level boolean flags whose names start with `prefix`."""
    return {name: value for name, value in globals().items()
            if name.startswith(prefix) and isinstance(value, bool)}


do_FLAGS   = MappingProxyType(_flags_with_prefix("do_"))
plot_FLAGS = MappingProxyType(_flags_with_prefix("plot_"))
