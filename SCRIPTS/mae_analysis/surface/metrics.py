"""Declarative metadata for the scalar plots.

`METRIC_GROUPS` is the single source of truth for what scalar quantities get
plotted, how they are labelled, their y-limits, and their physical unit. The
plotting engine (`plotting.py`) walks this registry; adding a new scalar metric
means adding a `MetricSpec` here, not touching the plot code.
"""

from collections import namedtuple
from dataclasses import dataclass
from typing import Callable

import pyanal
from pyanal.generic_functions import divide_abs_nan

from ..config import H_list


PlotCtx = namedtuple("PlotCtx", ("xaxis", "x_scale", "y_scale"))


def _zeeman_ref(ax, ctx):
    if ctx.xaxis.unit != "field":
        return
    ax.plot([min(H_list), max(H_list) * ctx.x_scale], [min(H_list), -max(H_list) * ctx.y_scale],
            "-", color="black", label="Zeeman", zorder=-10)


def _zoom_axes(*, xraw):
    """Return an `axes_fn` that re-frames the (shared) figure to the raw x-window `xraw`.

    Used to re-save an accumulated figure as a tight, standalone zoom: the x-window is
    `xraw` scaled by `ctx.x_scale`, and y auto-fits the data already on the axes inside
    that window (with a small pad), so the canvas fills instead of leaving whitespace.
    """
    def _run(ax, ctx):
        x0, x1 = xraw[0] * ctx.x_scale, xraw[1] * ctx.x_scale
        ax.set_xlim(x0, x1)
        ys_in = [y for line in ax.get_lines()
                 for x, y in zip(line.get_xdata(), line.get_ydata())
                 if x0 <= x <= x1 and y == y]
        if ys_in:
            lo, hi = min(ys_in), max(ys_in)
            pad = 0.05 * (hi - lo) if hi > lo else 0.05 * (abs(hi) or 1.0)
            ax.set_ylim(lo - pad, hi + pad)
    return _run


@dataclass(frozen=True)
class AxisSpec:
    label: str
    unit:  str = ""


@dataclass(frozen=True)
class MetricSpec:
    field:     str            # ARay field name = plt.figure identifier
    save_name: str            # output filename (without extension)
    ylabel:    str            # y-axis label
    ylim:      tuple | None   # y-axis limits (used when AUTO_XYLIM=False)
    yunit:     str = ""       # dimension key into UNIT_SYSTEM ("" = dimensionless)
    # derived metric: when op is set, field is a virtual id (figure/filename only)
    sources:   tuple = ()              # source ARay field names
    op:        Callable | None = None  # combine sources (numerator first)
    # per-plot matplotlib customization
    axes_kw:   "dict | None" = None   # ax.set(**axes_kw): yscale, xscale, ylim, ...
    axes_fn:   Callable | None = None  # axes_fn(ax, ctx): imperative hook


@dataclass(frozen=True)
class MetricGroup:
    guard:  str    # ARay field to check for data presence before processing
    specs:  tuple  # MetricSpec instances in this group


METRIC_GROUPS = (
    MetricGroup(
        guard="E_dip",
        specs=(
            MetricSpec("E_bond",    "E_bond",
                       r"$\left<E_{\text{bond}}\right>$",
                       ylim=(-0.05,0.355),      yunit="energy"),
            MetricSpec("E_dip",     "E_dip",
                       r"$\left<E_{\text{dipolar}}\right>$",
                       ylim=(-2.05, 0.05),      yunit="energy"),
            MetricSpec("E_Zee",     "E_Zee",
                       r"$\left<E_{\text{Zeeman}}\right>$",
                       ylim=(-10.5,0.3),        yunit="energy", axes_fn=_zeeman_ref),
            MetricSpec("E_Zee_dipm","E_Zee_dipm",
                       r"$\left<E_{\text{Zeeman},\mu_{0}}\right>$",
                       ylim=(-10.5,0.3),        yunit="energy", axes_fn=_zeeman_ref),
            MetricSpec("E_Zee_dipm","E_Zee_dipm_zoom",
                       r"$\left<E_{\text{Zeeman},\mu_{0}}\right>$",
                       ylim=None,               yunit="energy",
                       axes_fn=_zoom_axes(xraw=(0, 6))),
            MetricSpec("E_bond_dip","E_bond_dip",
                       r"$\left<E_{\text{bond}}\right>/\left<E_{\text{dipolar}}\right>$",
                       ylim=(-1.2, 0.05),
                       sources=("E_bond", "E_dip"), op=pyanal.divide_nan),
            MetricSpec("E_Zee_dip", "E_Zee_dip",
                       r"$\left<E_{\text{Zeeman}}\right>/\left<E_{\text{dipolar}}\right>$",
                       ylim=(-0.1, 25),
                       sources=("E_Zee", "E_dip"),  op=pyanal.divide_nan),
            MetricSpec("E_Zee_bond", "E_Zee_bond",
                       r"$\left<E_{\text{Zeeman}}\right>/\left<E_{\text{bond}}\right>$",
                       ylim=(-415, 10),
                       sources=("E_Zee", "E_bond"), op=divide_abs_nan,
                       axes_kw = {"yscale": "log"}),
            MetricSpec("E_dip_bond","E_dip_bond",
                       r"$\left<E_{\text{dipolar}}\right>/\left<E_{\text{bond}}\right>$",
                       ylim=None,
                       sources=("E_dip", "E_bond"), op=pyanal.divide_nan,
                       axes_kw = {"yscale": "symlog"}),
            MetricSpec("E_dip_Zee","E_dip_Zee",
                       r"$\left<E_{\text{dipolar}}\right>/\left<E_{\text{Zee}}\right>$",
                       ylim=(-1, 1e3),
                       sources=("E_dip", "E_Zee"), op=pyanal.divide_nan,
                       axes_kw = {"yscale": "symlog"}),
            MetricSpec("E_chains",  "E_chains",
                       r"$-2\left<\mu^{2}\right>/R^{3}$",
                       ylim=(-2.05, 0.05),      yunit="energy"),
        )),
    MetricGroup(
        guard="m_z",
        specs=(
            MetricSpec("m_z",     "m_z",
                       r"$\left<\mu_{z}\right>$",
                       ylim=(-0.05, 1.05),      yunit="moment"),
            MetricSpec("m_z_dipm","m_z_dipm",
                       r"$\left<\mu_{z}/\mu\right>$",
                       ylim=(-0.05,1.05)),
            MetricSpec("M_z",     "Mag_z",
                       r"$M$",
                       ylim=(-0.05, 0.6),      yunit="magnetization"),
        )),
    MetricGroup(
        guard="rms",
        specs=(
            MetricSpec("rms",  "R_rms",
                       r"$R_{\text{rms}}$",
                       ylim=(1,8),              yunit="length"),
            MetricSpec("h_avg","h_avg",
                       r"$\left<h_{\text{max}}\right>$",
                       ylim=(3,6.7),            yunit="length"),
        )),
    MetricGroup(
        guard="rms_with_bonds",
        specs=(
            MetricSpec("rms_with_bonds",  "R_rms_with_bonds",
                       r"$R_{\text{rms}}$",
                       ylim=(0.3,1.62),         yunit="length"),
            MetricSpec("h_avg_with_bonds","h_avg_with_bonds",
                       r"$\left<h_{\text{max}}\right>$",
                       ylim=(3,10),             yunit="length"),
        )),
    MetricGroup(
        guard="rms_with_surface",
        specs=(
            MetricSpec("rms_with_surface",  "R_rms_with_surface",
                       r"$R_{\text{rms}}$",
                       ylim=(0.,1.62),          yunit="length"),
            MetricSpec("h_avg_with_surface","h_avg_with_surface",
                       r"$\left<h_{\text{max}}\right>$",
                       ylim=(3,10),             yunit="length"),
        )),
    MetricGroup(
        guard="rms_interpolated",
        specs=(
            MetricSpec("rms_interpolated",  "R_rms_interpolated",
                       r"$R_{\text{rms}}$",
                       ylim=(0.,1.62),          yunit="length"),
            MetricSpec("h_avg_interpolated","h_avg_interpolated",
                       r"$\left<h_{\text{max}}\right>$",
                       ylim=(3,10),             yunit="length"),
        )),
    MetricGroup(
        guard="rms_with_bonds_with_surface",
        specs=(
            MetricSpec("rms_with_bonds_with_surface",
                       "R_rms_with_bonds_with_surface",
                       r"$R_{\text{rms}}$",
                       ylim=(0.25,1.5),         yunit="length"),
            MetricSpec("h_avg_with_bonds_with_surface",
                       "h_avg_with_bonds_with_surface",
                       r"$\left<h_{\text{max}}\right>$",
                       ylim=(3,10),             yunit="length"),
        )),
    MetricGroup(
        guard="n_peaks",
        specs=(
            MetricSpec("n_peaks",          "n_peaks",
                       r"number of peaks",
                       ylim=None),
            MetricSpec("peak_h_avg",       "peak_h_avg",
                       r"$\left<h_{\text{peak}}\right>$",
                       ylim=None,              yunit="length"),
            MetricSpec("peak_h_std",       "peak_h_std",
                       r"$std(h_{\text{peak}})$",
                       ylim=None,              yunit="length"),
            MetricSpec("peak_distance_avg","peak_distance_avg",
                       r"$\left<\Delta r_{\text{peaks}}\right>$",
                       ylim=None,              yunit="length"),
            MetricSpec("peak_distance_std","peak_distance_std",
                       r"$std(\Delta r_{\text{peaks}})$",
                       ylim=None,              yunit="length"),
        )),
    MetricGroup(
        guard="n_peaks_with_bonds",
        specs=(
            MetricSpec("n_peaks_with_bonds",          "n_peaks_with_bonds",
                       r"number of peaks",
                       ylim=None),
            MetricSpec("peak_h_avg_with_bonds",       "peak_h_avg_with_bonds",
                       r"$\left<h_{\text{peak}}\right>$",
                       ylim=None,              yunit="length"),
            MetricSpec("peak_h_std_with_bonds",       "peak_h_std_with_bonds",
                       r"$std(h_{\text{peak}})$",
                       ylim=None,              yunit="length"),
            MetricSpec("peak_distance_avg_with_bonds","peak_distance_avg_with_bonds",
                       r"$\left<\Delta r_{\text{peaks}}\right>$",
                       ylim=None,              yunit="length"),
            MetricSpec("peak_distance_std_with_bonds","peak_distance_std_with_bonds",
                       r"$std(\Delta r_{\text{peaks}})$",
                       ylim=None,              yunit="length"),
        )),
    MetricGroup(
        guard="thickness",
        specs=(MetricSpec("thickness","thickness",r"thickness",ylim=None,yunit="length"),)),
    MetricGroup(
        guard="thickness_with_bonds",
        specs=(MetricSpec("thickness_with_bonds","thickness_with_bonds",
                          r"thickness",
                          ylim=None,            yunit="length"),)),
    MetricGroup(
        guard="volume_avg",
        specs=(MetricSpec("volume_avg","volume_H",
                          r"$\left<V\right>$",
                          ylim=None,            yunit="volume"),)),
    MetricGroup(
        guard="volume_avg_with_bonds",
        specs=(MetricSpec("volume_avg_with_bonds","volume_H_with_bonds",
                          r"$\left<V\right>$",
                          ylim=None,            yunit="volume"),)),
)
