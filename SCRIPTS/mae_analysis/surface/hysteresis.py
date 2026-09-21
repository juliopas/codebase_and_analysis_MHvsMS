"""Hysteresis loops: overlay forward and reverse surface ARays on shared figures.

Reuses the surface scalar-metric engine (`_accumulate_metrics` / `_save_metrics`):
the forward series is drawn in normal colours, the reverse series in darkened
colours with a ``-rev`` label, so the two field paths are visually distinct on the
same axes. Replaces the old ``plot-reverse-with-normal.py``.

Plot-only and combined: both the forward (``figs/data_aray``) and reverse
(``figs-reverse/data_aray``) ARays must already exist -- build them first with
``--surface`` and ``--surface --reverse`` respectively.
"""

import os
import time
import itertools

import pyanal
from pyanal import ARay

from ..config import DATA_DIR, A_list, ts_list, DENS_list, K_list, HEIGHT_list, H_list
from ..style import combined_cm, sample_marker_map, a_linestyle_map
from .data import resolve_label_and_basedir
from .metrics import AxisSpec
from .plotting import _accumulate_metrics, _save_metrics

_H_AXIS = AxisSpec(r"$H$", "field")


def _aray_filename(A):
    """Match the ARay name produced by surface.data.init_ARay for this sweep."""
    if len(HEIGHT_list) == 1:
        return f"{A}--HEIGHT-{HEIGHT_list[0]}.h5"
    if len(DENS_list) == 1:
        return f"{A}--DENS-{DENS_list[0]}.h5"
    raise ValueError("hysteresis needs a constant HEIGHT or DENS to name the ARay")


def _require(path, kind):
    if not os.path.isfile(path):
        hint = " --reverse" if kind == "reverse" else ""
        raise FileNotFoundError(
            f"hysteresis needs the {kind} ARay: {path}\nRun `--surface{hint}` first.")
    return path


def run():
    """Overlay forward+reverse surface metrics vs H for every model A in config.A_list."""
    fwd_dir = os.path.join(DATA_DIR, "figs", "data_aray")
    rev_dir = os.path.join(DATA_DIR, "figs-reverse", "data_aray")

    cm     = combined_cm(ts_list=ts_list, H_list=H_list)
    cm_rev = combined_cm(ts_list=ts_list, H_list=H_list, dark=True)
    mm     = sample_marker_map()
    linestyles = a_linestyle_map(A_list)

    for A in A_list:
        fwd = ARay.from_file(_require(os.path.join(fwd_dir, _aray_filename(A)), "forward"))
        rev = ARay.from_file(_require(os.path.join(rev_dir, _aray_filename(A)), "reverse"))
        ls = linestyles[A]

        start = time.time()
        for ts in ts_list:
            base_dir = None
            for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
                label, label_cm_or_mm, _ = resolve_label_and_basedir(A, DENS, K, HEIGHT)
                sel = dict(ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT)
                _accumulate_metrics(fwd.get(**sel), H_list, _H_AXIS,
                                    cm, mm, label, label_cm_or_mm, ls)
                _accumulate_metrics(rev.get(**sel), H_list, _H_AXIS,
                                    cm_rev, mm, label + "-rev", label_cm_or_mm, ls)
                sub = f"HEIGHT-{HEIGHT}" if len(HEIGHT_list) == 1 else f"DENS-{DENS}"
                base_dir = os.path.join(DATA_DIR, "figs", A, "hysteresis", sub)
            _save_metrics(os.path.join(f"{base_dir}", "1-sample_H", f"t{ts}"),
                          _H_AXIS, energy_ref_lines=True)
            pyanal.plt_close("all")
        print(f"Plotted hysteresis for {A}.", time.time() - start)
