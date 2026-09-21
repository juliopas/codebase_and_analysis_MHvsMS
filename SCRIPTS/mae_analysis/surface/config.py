"""Surface-family run-time flags + the bonds x surface VARIANT machinery.

Edit the surface do_*/plot_*/save_* switches here. Everything cross-cutting
(paths, sweeps, units, style, calc/plot/reverse toggles) lives in the shared
``mae_analysis/config.py`` and is re-exported below, so surface modules can keep
importing a single ``from .config import ...`` namespace.
"""

import warnings
from types import MappingProxyType
from dataclasses import dataclass

from ..config import *          # noqa: F401,F403  -- re-export shared knobs
from ..config import H_list, plot_PLOT, plot_ONLY_SAMPLE_H   # names referenced explicitly below

# --- Main calculation flags ---
do_H_INTERNAL = True
do_ENERGIES = True
do_MAGNETIZATION = True
do_HEIGHTS = True
do_PEAK_STATISTICS = True  # needs HEIGHTS
do_DENSITY_Z = True
do_VOLUME = True           # needs SURFACE_SHAPE

# --- "Use these in calculations" flags ---
do_USE_BOND_TO_PARTS = True   # in HIST, SURFACE, PHI_Z, SURFACE params
do_USE_SURFACE_SHAPE = True   # recommended with BOND_TO_PARTS  # in HIST, SURFACE params
do_USE_HEIGHT_AT_H0 = True

# --- Plotting flags ---
# uses saved data from ARay
plot_AFTER = True
plot_SAMPLE_H = True; plot_NONDIM_H = True
plot_H_TIME = True; plot_TIME_H = True
plot_DENSITY_Z = True
# during analysis (need do_CALCULATE + specific calculations)
plot_HIST = True
plot_SURFACE_SHAPE = True
plot_PEAK_DISTRIBUTIONS = True

# --- Saving flags ---
save_HEIGHT_MAPS = False

# ===================================================== Flag consistency ====
if plot_ONLY_SAMPLE_H:
    plot_SAMPLE_H = plot_NONDIM_H = True
    plot_H_TIME = plot_TIME_H = plot_DENSITY_Z = False
if do_PEAK_STATISTICS:
    assert do_HEIGHTS, "PEAK_STATISTICS needs HEIGHTS"
if do_VOLUME:
    assert do_USE_SURFACE_SHAPE, "VOLUME needs SURFACE_SHAPE"
    if not do_USE_BOND_TO_PARTS:
        warnings.warn("Using VOLUME without considering the bonds between particles for the shape of the "
                      "material. This leads to worst, and most likely unrealistic, results")

if not plot_PLOT:
    # disable every surface plot_* flag in one place (--no-plot cascade)
    for _plot_flag in [name for name in list(globals()) if name.startswith("plot_")]:
        globals()[_plot_flag] = False

if do_USE_HEIGHT_AT_H0:
    assert 0 in H_list, "To use HEIGHTS_AT_H0 you must include the elastomer data for H=0"


def _flags_with_prefix(prefix):
    """Collect the module-level boolean flags whose names start with `prefix`."""
    return {name: value for name, value in globals().items()
            if name.startswith(prefix) and isinstance(value, bool)}


# Grouped, read-only flag views passed to functions (cleaner than threading each flag through).
do_FLAGS   = MappingProxyType(_flags_with_prefix("do_"))
plot_FLAGS = MappingProxyType(_flags_with_prefix("plot_"))

# ============================================ Variant iteration (bonds x surface)
# VARIANTS drives all 4-way (bonds x surface) branches throughout the analysis.

@dataclass(frozen=True)
class VariantSpec:
    suffix:      str   # "" | "_with_bonds" | "_with_surface" | "_with_bonds_with_surface"
    use_bonds:   bool
    use_surface: bool


VARIANTS: list = [VariantSpec("", False, False)]
if do_USE_BOND_TO_PARTS:
    VARIANTS.append(VariantSpec("_with_bonds", True, False))
if do_USE_SURFACE_SHAPE:
    VARIANTS.append(VariantSpec("_with_surface", False, True))
    VARIANTS.append(VariantSpec("_interpolated", False, True))
if do_USE_BOND_TO_PARTS and do_USE_SURFACE_SHAPE:
    VARIANTS.append(VariantSpec("_with_bonds_with_surface", True, True))

# Sub-group frequently iterated separately: peaks / volume / dens_z (no surface variants)
VARIANTS_NO_SURFACE = [v for v in VARIANTS if not v.use_surface]
