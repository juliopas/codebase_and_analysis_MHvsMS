"""All plotting. Every plot function consumes a LIST of `Series`:

    * len 1  -> the single-model plots (output under OUTPUTPATH/<A>/...)
    * len N  -> the N models overlaid on shared figures
                (scalar metrics under OUTPUTPATH/_joined/<A1>-<A2>/...)

Scalar metrics (energies, magnetization, heights, peaks, thickness, volume) are
driven by the `metrics.METRIC_GROUPS` registry and overlay automatically: each
series is accumulated onto figures keyed by `plt.figure(num=spec.field)` before a
single save. Per-series style: colour = sample key, marker = density, linestyle =
model `A` (see `style.py`).
"""

import os
import time
import itertools

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D as plt_Line2D

import pyanal

from numpy import atleast_1d, logical_and, isnan, issubdtype, floating, asarray

from .config import (
    OUTPUTPATH, N_FULL_BOX, UNIT_SYSTEM, NONDIM, AUTO_XYLIM, savefig_format, dpi,
    fontsize_label, fontsize_ticks, markersize_1,
    ts_list, DENS_list, K_list, HEIGHT_list, H_list, VARIANTS_NO_SURFACE,
    PER_A_COLOR_FAMILIES,
)
from .metrics import METRIC_GROUPS, AxisSpec, PlotCtx
from .data import resolve_label_and_basedir
from ..style import (
    combined_cm, sample_marker_map, a_kcmaps_map, H_color_map,
    join_out_label, join_name, is_join,
)
from ..legends import save_frameless_legend, save_handle_legend_set

plt.rcParams['figure.max_open_warning'] = 0


# --------------------------------------------------------------- small helpers --

def _decorate_label(label: str, unit_str: str) -> str:
    """Append a [unit] suffix to a LaTeX label ending in $, or return unchanged."""
    if not unit_str:
        return label
    return label + unit_str


def _scale(value, factor):
    """Multiply value by factor; return unchanged when factor is identity (1.0)."""
    return value if factor == 1.0 else value * factor


# ----------------------------------------------- scalar-metric plotting engine --

def _accumulate_metrics(data_aray, axis_list, xaxis: AxisSpec,
                        cm, mm, label, label_cm_or_mm, linestyle, label_prefix="", filled=True,
                        unit_system=None):
    """Add lines for one data slice (one inner-loop iteration of one series) onto
    the running metric figures."""
    us = unit_system or UNIT_SYSTEM
    x_scale = us.scale(xaxis.unit)
    # axis_list is the requested subset (may include coords absent from this data
    # file). Subset the ARay to the present ones, then take x from its own coords so
    # x and the per-field set_mask are aligned by construction.
    (axis_name,) = data_aray.mapping_dict           # exactly one axis remains after .get(...)
    present = data_aray.coords(axis_name)
    wanted = [v for v in axis_list if v in present]
    data_aray = data_aray.get(**{axis_name: wanted})
    scaled_x = _scale(atleast_1d(data_aray.coords(axis_name)), x_scale)
    # Fields actually present in THIS series' ARay: lets older data files (missing
    # newer fields) coexist with current ones instead of crashing the engine.
    available = {name for name, _ in data_aray.dtype_str}
    for grp in METRIC_GROUPS:
        if grp.guard not in available:
            continue
        if not data_aray.set_mask[grp.guard].any():
            continue
        for spec in grp.specs:
            if spec.op is not None:                          # derived
                if not all(s in available for s in spec.sources):
                    continue
                mask = logical_and.reduce([data_aray.set_mask[s] for s in spec.sources])
                if not mask.any():
                    continue
                y = spec.op(*(data_aray.array[s][mask] for s in spec.sources))
            else:                                            # stored
                if spec.field not in available:
                    continue
                # Per-field masking: each metric figure plots every x where THIS field
                # is set, independent of sibling fields. The field's set-mask is one
                # bool array over the axis (positional, same order as `axis_list`), so
                # x and y stay aligned.
                mask = data_aray.set_mask[spec.field]
                y = data_aray.array[spec.field][mask]
            scaled_x_plot = scaled_x[mask]
            if issubdtype(asarray(y).dtype, floating):   # int metrics (e.g. n_peaks) can't hold NaN
                keep = ~isnan(y)
                y = y[keep]
                scaled_x_plot = scaled_x_plot[keep]
            if y.size == 0:                              # nothing left -> don't open an empty figure / ghost legend entry
                continue
            y_scale = us.scale(spec.yunit)
            plt.figure(num=spec.field)
            plt.plot(scaled_x_plot, _scale(y, y_scale),
                     color=cm[label_cm_or_mm], marker=mm[label_cm_or_mm],
                     markerfacecolor=cm[label_cm_or_mm] if filled else "white",
                     markersize=markersize_1, linestyle=linestyle, label=label_prefix + label)


def _save_metrics(outdir, xaxis: AxisSpec, unit_system=None):
    """Save every metric figure that was populated this outer-loop iteration.

    Overlay-safe: a figure exists iff at least one series accumulated onto it
    (`_accumulate_metrics` creates `plt.figure(num=spec.field)` only when the
    group has data), so we save on `plt.fignum_exists`.
    """
    os.makedirs(outdir, exist_ok=True)
    us = unit_system or UNIT_SYSTEM
    x_scale = us.scale(xaxis.unit)
    x_label = _decorate_label(xaxis.label, us.label(xaxis.unit))
    for grp in METRIC_GROUPS:
        for spec in grp.specs:
            if not plt.fignum_exists(spec.field):
                continue
            y_scale = us.scale(spec.yunit)
            y_label = _decorate_label(spec.ylabel, us.label(spec.yunit))
            ylim    = tuple(v * y_scale for v in spec.ylim) if spec.ylim is not None else None
            if spec.axes_fn is not None:
                plt.figure(num=spec.field)
                spec.axes_fn(plt.gca(), PlotCtx(xaxis, x_scale, y_scale))
            pyanal.save_plot_simple(
                num=spec.field,
                filename=os.path.join(outdir, spec.save_name),
                format=savefig_format,
                xlabel=x_label, ylabel=y_label, ylim=ylim,
                ignore_xylim=AUTO_XYLIM,
                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks,
                axes_kw=spec.axes_kw,
                dpi=dpi)


# ---------------------------------------------------------- scalar-metric plots --

def plot_sample_h(series_list):
    """Saved quantities over H, one line per elastomer sample (per ts)."""
    print("Plotting saved quantities over H, for multiple elastomer samples...")
    mm = sample_marker_map()
    out_label = join_out_label(series_list)
    joined = is_join(series_list)
    kcmaps = a_kcmaps_map([s.A for s in series_list]) if PER_A_COLOR_FAMILIES else None
    start_time = time.time()
    for ts in ts_list:
        BASE_DIR = None
        for s in series_list:
            cm = combined_cm(ts_list=ts_list, H_list=H_list,
                             k_cmaps=(kcmaps[s.A] if kcmaps else None))
            label_prefix = f"{s.A} " if joined else ""
            for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
                if not DENS in s.aray.coords('DENS') or not K in s.aray.coords('K') or not HEIGHT in s.aray.coords('HEIGHT'):
                    continue
                label, label_cm_or_mm, BASE_DIR = resolve_label_and_basedir(out_label, DENS, K, HEIGHT)
                data_ARay_ts_sample = s.aray.get(ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT)
                _accumulate_metrics(data_ARay_ts_sample, H_list, AxisSpec(r"$H$", "field"),
                                    cm, mm, label, label_cm_or_mm, s.linestyle, label_prefix, s.filled)
        assert BASE_DIR is not None
        outputpath_sample_H_plots = os.path.join(BASE_DIR, "1-sample_H", f"t{ts}")
        _save_metrics(outputpath_sample_H_plots, AxisSpec(r"$H$", "field"))
        pyanal.plt_close("all")

    print("Ploted 'sample_H' plots.", time.time() - start_time)


def plot_nondim_h(series_list):
    """Like `plot_sample_h`, but rendered through the NONDIM unit system: the field
    axis becomes the Langevin parameter xi = mu*H/k_BT and every metric is shown in its
    reduced unit (energy/k_BT, M/M_sat, ...). Tuning K_BT in config rescales xi to match a
    target initial susceptibility. One line per elastomer sample, per ts."""
    print("Plotting nondimensional (Langevin-parameter) quantities, for multiple elastomer samples...")
    mm = sample_marker_map()
    out_label = join_out_label(series_list)
    joined = is_join(series_list)
    kcmaps = a_kcmaps_map([s.A for s in series_list]) if PER_A_COLOR_FAMILIES else None
    xi = AxisSpec(r"$\xi = \mu_{0} H / k_{\text{B}}\text{T}$", "field")
    start_time = time.time()
    for ts in ts_list:
        BASE_DIR = None
        for s in series_list:
            cm = combined_cm(ts_list=ts_list, H_list=H_list,
                             k_cmaps=(kcmaps[s.A] if kcmaps else None))
            label_prefix = f"{s.A} " if joined else ""
            for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
                if not DENS in s.aray.coords('DENS') or not K in s.aray.coords('K') or not HEIGHT in s.aray.coords('HEIGHT'):
                    continue
                label, label_cm_or_mm, BASE_DIR = resolve_label_and_basedir(out_label, DENS, K, HEIGHT)
                data_ARay_ts_sample = s.aray.get(ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT)
                _accumulate_metrics(data_ARay_ts_sample, H_list, xi,
                                    cm, mm, label, label_cm_or_mm, s.linestyle, label_prefix, s.filled,
                                    unit_system=NONDIM)
        assert BASE_DIR is not None
        outputpath_nondim_H_plots = os.path.join(BASE_DIR, "4-nondim_H", f"t{ts}")
        _save_metrics(outputpath_nondim_H_plots, xi, unit_system=NONDIM)
        pyanal.plt_close("all")

    print("Ploted 'nondim_H' plots.", time.time() - start_time)


def plot_h_time(series_list):
    """Saved quantities over time; each line is an H (per sample)."""
    print("Plotting saved quantities over time, for multiple values of H...")
    cm = pyanal.create_cm(H_list=H_list)
    mm = pyanal.create_mm(default=lambda: "o")
    out_label = join_out_label(series_list)
    joined = is_join(series_list)
    start_time = time.time()
    for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
        save_name_sample = f"DENS{DENS:.2f}K{K}HEIGHT{HEIGHT}"
        *_, BASE_DIR = resolve_label_and_basedir(out_label, DENS, K, HEIGHT)
        for s in series_list:
            label_prefix = f"{s.A} " if joined else ""
            for H in H_list:
                if not H in s.aray.coords('H'):
                    continue
                label = f"H{H}"
                label_cm_or_mm = H
                data_ARay_sample_H = s.aray.get(DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
                _accumulate_metrics(data_ARay_sample_H, ts_list, AxisSpec(r"$t$", "time"),
                                    cm, mm, label, label_cm_or_mm, s.linestyle, label_prefix, s.filled)

        outputpath_H_time_plots = os.path.join(BASE_DIR, "3-H_time", f"{save_name_sample}")
        _save_metrics(outputpath_H_time_plots, AxisSpec(r"$t$", "time"))
        pyanal.plt_close("all")

    pyanal.plt_close("all")
    print("Ploted 'H_TIME' plots.", time.time() - start_time)


def plot_time_h(series_list):
    """Saved quantities over H; each line is a time (per sample)."""
    print("Plotting saved quantities over H, for multiple values of time...")
    cm = pyanal.create_cm(ts_list=ts_list)
    mm = pyanal.create_mm(default=lambda: "o")
    out_label = join_out_label(series_list)
    joined = is_join(series_list)
    start_time = time.time()
    for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
        save_name_sample = f"DENS{DENS:.2f}K{K}HEIGHT{HEIGHT}"
        *_, BASE_DIR = resolve_label_and_basedir(out_label, DENS, K, HEIGHT)
        for s in series_list:
            label_prefix = f"{s.A} " if joined else ""
            for ts in ts_list:
                if not ts in s.aray.coords('ts'):
                    continue
                label = f"t{ts}"
                label_cm_or_mm = ts
                data_ARay_sample_ts = s.aray.get(DENS=DENS, K=K, HEIGHT=HEIGHT, ts=ts)
                _accumulate_metrics(data_ARay_sample_ts, H_list, AxisSpec(r"$H$", "field"),
                                    cm, mm, label, label_cm_or_mm, s.linestyle, label_prefix, s.filled)

        outputpath_time_H_plots = os.path.join(BASE_DIR, "2-time_H", f"{save_name_sample}")
        _save_metrics(outputpath_time_H_plots, AxisSpec(r"$H$", "field"))
        pyanal.plt_close("all")

    pyanal.plt_close("all")
    print("Ploted 'TIME_H' plots.", time.time() - start_time)


# --------------------------------------------------------- density-z profiles --
# Density profiles phi(z) are plotted per model (each series -> its own OUTPUTPATH/<A>/...).
# The systematic colours/markers still apply; overlaying models on a single phi(z)
# figure is intentionally left out (the scalar-metric plots are the overlay target).

def plot_dens_z_profiles(series_list):
    """Plot phi(z) density-profile curves for each model in `series_list`."""
    kcmaps = a_kcmaps_map([s.A for s in series_list]) if PER_A_COLOR_FAMILIES else None
    for s in series_list:
        cm = combined_cm(ts_list=ts_list, H_list=H_list,
                         k_cmaps=(kcmaps[s.A] if kcmaps else None))
        _plot_dens_z_one(s.aray, s.A, cm)


def _plot_dens_z_one(data_ARay, A, cm):
    z_scale   = UNIT_SYSTEM.scale("length")
    phi_scale = UNIT_SYSTEM.scale("density")
    z_label   = _decorate_label(r"$z$",       UNIT_SYSTEM.label("length"))
    phi_label = _decorate_label(r"$\phi(z)$", UNIT_SYSTEM.label("density"))

    for ts in ts_list:
        for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
            label, label_cm_or_mm, BASE_DIR = resolve_label_and_basedir(A, DENS, K, HEIGHT)
            label_full = f"{A}-{DENS:.2f}_{K}-{N_FULL_BOX}_{HEIGHT}"
            save_name_sample = f"DENS{DENS:.2f}K{K}HEIGHT{HEIGHT}"

            if not ts in data_ARay.coords('ts') or not DENS in data_ARay.coords('DENS') or not K in data_ARay.coords('K') or not HEIGHT in data_ARay.coords('HEIGHT'):
                    continue
            data_ARay_ts_sample = data_ARay.get(ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT)

            for v in VARIANTS_NO_SURFACE:
                if data_ARay_ts_sample[f"thickness{v.suffix}"].get_set_entries().array.shape[0] == 0:
                    continue
                dens_z_aray = data_ARay_ts_sample[["dens_z_bin_edges", f"dens_z{v.suffix}"]]
                H_list_plot = [H for H in H_list if H in dens_z_aray.coords('H') and dens_z_aray.get(H=H).is_set()]
                for H in H_list_plot:
                    if not H in dens_z_aray.coords('H'):
                        continue
                    dza = dens_z_aray.get(H=H).array
                    bin_edges = _scale(dza["dens_z_bin_edges"][:-1], z_scale)
                    dens      = _scale(dza[f"dens_z{v.suffix}"], phi_scale)
                    # for each sample and time, for different H
                    plt.figure(num=f"phiz_H{v.suffix}")
                    plt.plot(bin_edges, dens, "-", color=cm[H], label=f"H={H}")
                    # for each H and time, for different samples
                    plt.figure(num=f"phiz_sample{v.suffix}{H}")
                    plt.plot(bin_edges, dens, "-", color=cm[label_cm_or_mm], label=label)
                    # for each H and sample, for different times
                    plt.figure(num=f"phiz_time{v.suffix}{label_full}{H}")
                    plt.plot(bin_edges, dens, "-", color=cm[ts], label=f"t={ts}")

                # Save phiz_H for this (ts, DENS, K, HEIGHT) and close immediately
                bonds_subdir = "with_bonds" if v.use_bonds else "without_bonds"
                outputpath_tmp = os.path.join(BASE_DIR, "phiz_z_with_H", bonds_subdir, save_name_sample)
                os.makedirs(outputpath_tmp, exist_ok=True)
                pyanal.save_plot_simple(num=f"phiz_H{v.suffix}",
                                        filename=os.path.join(outputpath_tmp, f"t{ts}"), format=savefig_format,
                                        xlabel=z_label, ylabel=phi_label,
                                        ignore_xylim=AUTO_XYLIM,
                                        fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks,
                                        dpi=dpi)
                pyanal.plt_close(f"phiz_H{v.suffix}")

        # Save phiz_sample (per ts+H) after inner loop has finished all samples
        for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
            *_, BASE_DIR = resolve_label_and_basedir(A, DENS, K, HEIGHT)
            for v in VARIANTS_NO_SURFACE:
                bonds_subdir = "with_bonds" if v.use_bonds else "without_bonds"
                for H in H_list:
                    if not plt.fignum_exists(f"phiz_sample{v.suffix}{H}"):
                        continue
                    outputpath_tmp = os.path.join(BASE_DIR, "phi_z_with_sample", bonds_subdir, f"H{H}")
                    os.makedirs(outputpath_tmp, exist_ok=True)
                    pyanal.save_plot_simple(num=f"phiz_sample{v.suffix}{H}",
                                            filename=os.path.join(outputpath_tmp, f"t{ts}"), format=savefig_format,
                                            xlabel=z_label, ylabel=phi_label,
                                            ignore_xylim=AUTO_XYLIM,
                                            fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks,
                                            dpi=dpi)

        pyanal.plt_close("all", but_func=lambda lbl: isinstance(lbl, str) and lbl.startswith("phiz_time"))

    # Save phiz_time (per DENS, K, HEIGHT, H) after all ts have been processed
    for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
        label_full = f"{A}-{DENS:.2f}_{K}-{N_FULL_BOX}_{HEIGHT}"
        save_name_sample = f"DENS{DENS:.2f}K{K}HEIGHT{HEIGHT}"
        *_, BASE_DIR = resolve_label_and_basedir(A, DENS, K, HEIGHT)
        for H in H_list:
            if not DENS in data_ARay.coords('DENS') or not K in data_ARay.coords('K') or not HEIGHT in data_ARay.coords('HEIGHT') or not H in data_ARay.coords('H'):
                continue
            data_ARay_sample_H = data_ARay.get(DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
            for v in VARIANTS_NO_SURFACE:
                if data_ARay_sample_H[f"thickness{v.suffix}"].get_set_entries().array.shape[0] == 0:
                    continue
                bonds_subdir = "with_bonds" if v.use_bonds else "without_bonds"
                outputpath_tmp = os.path.join(BASE_DIR, "phiz_z_with_time", bonds_subdir, save_name_sample)
                os.makedirs(outputpath_tmp, exist_ok=True)
                pyanal.save_plot_simple(num=f"phiz_time{v.suffix}{label_full}{H}",
                                        filename=os.path.join(outputpath_tmp, f"H{H}"), format=savefig_format,
                                        xlabel=z_label, ylabel=phi_label,
                                        ignore_xylim=AUTO_XYLIM,
                                        fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks,
                                        dpi=dpi)
    pyanal.plt_close("all")


# -------------------------------------------------------------------- legends --

def plot_legends(series_list):
    """Save all legend images once, to OUTPUTPATH/<out_label>/legends/."""
    out_label = join_out_label(series_list)
    joined = is_join(series_list)
    name = join_name(series_list)
    outdir_base = os.path.join(OUTPUTPATH, out_label, "legends")
    cm_H        = H_color_map(H_list)
    cm_ts       = pyanal.create_cm(ts_list=ts_list)
    cm_sample   = combined_cm(ts_list=ts_list, H_list=H_list)   # sample + H + ts keys (model-agnostic base)
    mm          = sample_marker_map()
    kcmaps      = a_kcmaps_map([s.A for s in series_list]) if PER_A_COLOR_FAMILIES else None

    # 1-sample_H: per ts — (a) sample-label legend, (b) H-colour legends
    outdir_1 = os.path.join(outdir_base, "1-sample_H")
    n_models = len(series_list)
    n_K = len(K_list)
    # Inner-most sample ordering: the varying sample parameter (DENS / HEIGHT) ramps
    # ascending, so the colour shade reads light->dark down each column.
    sample_inner = list(itertools.product(DENS_list, HEIGHT_list))
    for ts in ts_list:
        # (a) one Line2D per (model, sample): colour/marker = sample key, linestyle/fill
        # = model. Built into a dict keyed by (model, K, DENS, HEIGHT) so it can be
        # re-ordered into the different column groupings below.
        handle_of = {}
        for m_idx, s in enumerate(series_list):
            cm_s = (combined_cm(ts_list=ts_list, H_list=H_list, k_cmaps=kcmaps[s.A])
                    if kcmaps else cm_sample)
            label_prefix = f"M{s.A[0]}{s.A[2:]} " if joined else ""
            for K in K_list:
                for DENS, HEIGHT in sample_inner:
                    label, label_cm_or_mm, _ = resolve_label_and_basedir(out_label, DENS, K, HEIGHT)
                    handle_of[(m_idx, K, DENS, HEIGHT)] = plt_Line2D(
                        [0], [0], color=cm_s[label_cm_or_mm], marker=mm[label_cm_or_mm],
                        markerfacecolor=cm_s[label_cm_or_mm] if s.filled else "white",
                        linestyle=s.linestyle, label=label_prefix + label)
        # matplotlib fills legends column-major (handles split into `ncols` contiguous
        # top-to-bottom chunks), so handle order + ncols define the grid:
        #   bymodelK -> each column is one (model, K) colour ramp
        #   byK      -> each column is one K family, models stacked
        #   bymodel  -> each column is one full model
        order_mk = [(m, K, d, h) for m in range(n_models) for K in K_list for d, h in sample_inner]
        order_km = [(m, K, d, h) for K in K_list for m in range(n_models) for d, h in sample_inner]
        variants = [("bymodelK", order_mk, n_models * n_K),
                    ("byK",      order_km, n_K),
                    ("bymodel",  order_mk, n_models)]
        seen = set()
        for vname, order, ncols in variants:
            dedup_key = (tuple(order), ncols)
            if dedup_key in seen:                 # collapses to an identical grid (e.g. single model)
                continue
            seen.add(dedup_key)
            save_frameless_legend(outdir_1,
                                  os.path.join(outdir_1, f"{name}-ts{ts}-legend-{vname}.png"),
                                  [handle_of[k] for k in order], ncols, dpi)
        # (b) H-colour patch/line/point legends
        save_handle_legend_set(outdir_1, f"{name}-ts{ts}", cm_sample, H_list, lambda H: f"H{H}")

    # 2-H_time and 3-time_H: per sample
    for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
        save_name_sample = f"DENS{DENS:.2f}K{K}HEIGHT{HEIGHT}"
        save_handle_legend_set(os.path.join(outdir_base, "2-H_time"),
                               f"{save_name_sample}-H",
                               cm_H, H_list, lambda H: f"H{H}")
        save_handle_legend_set(os.path.join(outdir_base, "3-time_H"),
                               f"{save_name_sample}-time",
                               cm_ts, ts_list, lambda ts: f"t{ts}")
