"""Plot styling + the `Series` abstraction that drives multi-`A` overlays.

Encoding (consistent across every plot type):
  * K            -> colour family   (hard = warm ramp, soft = cool ramp)
  * varying par  -> shade in family + marker shape   (DENS, or HEIGHT in the other branch)
  * A (model)    -> linestyle

So overlaying models means: same colours/markers, one linestyle per model. Adding
a model costs one linestyle and zero colours.
"""

import os
from dataclasses import dataclass

import pyanal
from pyanal import ARay
from pyanal.plot import darken_color
from matplotlib import pyplot as plt
import colorcet

from .config import (
    A_list, K_list, DENS_list, HEIGHT_list,
    H_COLORMAP_MODE, H_SEQUENTIAL_CMAP, H_SEQUENTIAL_RANGE, H_CATEGORICAL_COLORS,
    A_CM_POOL, A_LS_POOL,
)


@dataclass(frozen=True)
class Series:
    """One model's dataset plus its per-model style. A list of these is what the
    plot functions consume: length 1 reproduces the single-model plots, length N
    overlays N models on the shared figures."""
    aray: ARay
    A: str
    linestyle: str
    filled: bool


def _sample_key_fn_and_varying():
    """Return (key_fn, varying_values) matching `data.resolve_label_and_basedir`'s
    `label_cm_or_mm` convention for the active constant-parameter branch."""
    if len(HEIGHT_list) == 1:        # constant HEIGHT -> density varies
        return (lambda K, dens: f"{dens:.2f}-{K}"), list(DENS_list)
    elif len(DENS_list) == 1:        # constant DENS -> height varies
        return (lambda K, height: f"{K}-{height}"), list(HEIGHT_list)
    else:
        raise ValueError("need to update plot design dicts. to accommodate more varying parameters")


def sample_color_map(k_cmaps=None):
    """{sample_key: colour} — K is the colour family, varying parameter the shade.
    `k_cmaps` overrides the per-K colormap families (e.g. for per-model colours)."""
    key_fn, varying = _sample_key_fn_and_varying()
    return pyanal.sample_color_map(K_list, varying, key_fn, k_cmaps=k_cmaps)


def sample_marker_map():
    """{sample_key: marker} — marker encodes the varying parameter (e.g. density)."""
    key_fn, varying = _sample_key_fn_and_varying()
    return pyanal.sample_marker_map(K_list, varying, key_fn)


def a_linestyle_map(a_list=None):
    """{A: linestyle} for the models being plotted (defaults to config.A_list)."""
    return pyanal.A_linestyle_map(list(a_list) if a_list is not None else A_list, linestyles=A_LS_POOL)


def a_markerfill_map(a_list=None):
    """{A: bool} marker-fill flags (True = solid, False = white-cored open ring)."""
    return pyanal.A_markerfill_map(list(a_list) if a_list is not None else A_list)


def a_kcmaps_map(a_list=None):
    """{A: {K: cmap_name}} giving each model its own per-K colour families."""
    a = list(a_list) if a_list is not None else A_list
    return pyanal.A_kcmaps_map(a, K_list, pool=A_CM_POOL)


def H_color_map(H_list, mode=None):
    """{H: rgba} for the magnetic field sweep. H is ordered, so `"sequential"` samples
    `H_SEQUENTIAL_CMAP` by rank within `H_SEQUENTIAL_RANGE`; `"categorical"` cycles the
    improved (colour-blind-safe) `H_CATEGORICAL_COLORS`. Keys are sorted ascending so the
    colour ramp reads monotonically. Shared by the surface and cluster plots."""
    mode = mode if mode is not None else H_COLORMAP_MODE
    Hs = sorted(H_list)
    if mode == "sequential":
        cmap = plt.get_cmap(H_SEQUENTIAL_CMAP)
        lo, hi = H_SEQUENTIAL_RANGE
        n = len(Hs)
        return {H: cmap(lo + (hi - lo) * (i / (n - 1) if n > 1 else 0.5))
                for i, H in enumerate(Hs)}
    if mode == "categorical":
        return {H: H_CATEGORICAL_COLORS[i % len(H_CATEGORICAL_COLORS)]
                for i, H in enumerate(Hs)}
    raise ValueError(f"unknown H_COLORMAP_MODE {mode!r} (expected 'sequential' or 'categorical')")


def combined_cm(ts_list=None, H_list=None, dark=False, k_cmaps=None):
    """Colour map for figures that mix sample / H / ts keys (sample_h, dens_z):
    ts colours from `create_cm`, H colours from the ordered `H_color_map`, sample
    colours from the systematic scheme.
    `dark=True` darkens every entry -- used to draw the reverse-field series in
    hysteresis overlays so forward and reverse are visually distinct.
    `k_cmaps` overrides the per-K colour families for the sample colours (used to
    give each model its own families when `PER_A_COLOR_FAMILIES` is on)."""
    if ts_list is not None:
        cm = pyanal.create_cm(ts_list=ts_list)
    if H_list is not None:
        cm.update(H_color_map(H_list))
    cm.update(sample_color_map(k_cmaps=k_cmaps))
    if dark:
        cm = {k: darken_color(v) for k, v in cm.items()}
    return cm


def join_out_label(series_list):
    """Output sub-directory under OUTPUTPATH for a set of series:
    a single model name, or `_joined/<A1>-<A2>-...` for an overlay."""
    if len(series_list) == 1:
        return series_list[0].A
    return os.path.join("-".join(s.A for s in series_list))


def join_name(series_list):
    """Filename-safe model label, e.g. "SM" or "HM-SM" (no path separators)."""
    return "-".join(s.A for s in series_list)


def is_join(series_list):
    return len(series_list) > 1
