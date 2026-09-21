from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle as plt_Circle
from matplotlib.colors import to_rgb as matplot_to_rgb
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D

import numpy as np

from collections import defaultdict

from .generic_functions import divide_x_by_zero_gets_a

##### Color/Marker maps #####
def darken_color(color, factor=0.6):
    r, g, b = matplot_to_rgb(color)
    return (r*factor, g*factor, b*factor)

def create_cm(samples=True, ts_list=None, H_list=None, default=None, dark=False):
    if default is None:
        cm = {}
    else:
        cm = defaultdict(default)

    if samples:
        cm.update(
            {'0.20-hard': "aquamarine", '0.20-soft': "blue",
            '0.25-hard': "yellowgreen", '0.25-soft': "royalblue",
            '0.30-hard': "gold", '0.30-soft': "cyan"}
        )

        #add heights scenario
        cm.update(
            {"hard-4": "red", "hard-6": "orange",
            "hard-8": "yellow", "hard-10": "yellowgreen",
            "hard-20": "green", "soft-4": "blue",
            "soft-6": "cyan", "soft-8": "purple",
            "soft-10": "pink", "soft-20": "magenta"}
        )

    if ts_list is not None:
        # add time step colors
        max_ts= max(ts_list)
        for ts in ts_list:
            cm[ts] = plt.cm.spring(divide_x_by_zero_gets_a(ts, max_ts, a=0.))

    if H_list is not None:
        cm_custom = ["red","blue","green","purple","orange","cyan","magenta","gold","brown","lime","navy","darkviolet","teal","deeppink","olive","darkorange","turquoise","maroon","dodgerblue","black",]
        assert len(cm_custom) >= len(H_list), f"Not enough custom colors for H values. Need at least {len(H_list)}, but only have {len(cm_custom)}. Consider adding more colors to the cm_custom list or using a colormap instead."
        # add magnetic field colors
        max_H= max(H_list); log_max_H = np.log(max_H); sqrt_max_H = np.sqrt(max_H); log_sp_max_H = np.log(max_H+1)
        for i, H in enumerate(H_list):
            # cm[H] = plt.cm.brg(i/(len(H_list)-1))
            # cm[H] = plt.cm.Paired(i)
            # cm[H] = plt.cm.Set1(i)
            cm[H] = cm_custom[i]

    if dark:
        cm = {k: darken_color(v) for k, v in cm.items()}

    return cm

def create_mm(samples=True, default=lambda:"o"):
    if default is None:
        mm = {}
    else:
        mm = defaultdict(default)

    if samples:
        mm.update({'0.20-hard': "v", '0.20-soft': "o",
            '0.25-hard': "D", '0.25-soft': "s",
            '0.30-hard': "<", '0.30-soft': "^"})

        mm.update(
            {"hard-4": "v", "hard-6": "o",
            "hard-8": "D", "hard-10": "s",
            "hard-20": "<", "soft-4": "^",
            "soft-6": ".", "soft-8": "X",
            "soft-10": ">", "soft-20": "P"}
        )

    return mm

# --- Systematic sample styling: K = colour family, varying param = shade + marker ---
# These build maps keyed by the *sample key* via `key_fn(K, value) -> str`, so the
# caller controls the key convention (e.g. f"{DENS:.2f}-{K}" or f"{K}-{HEIGHT}").
# Used to overlay several models (`A`) cleanly: colour encodes K (+ shade = density),
# marker encodes density, and the caller adds linestyle per `A` (see A_linestyle_map).

# Default ColorBrewer sequential families: warm for "hard", cool for "soft".
_K_CMAPS_DEFAULT = {"hard": "Oranges", "soft": "Blues"}
_K_CMAPS_FALLBACK = ["Greens", "Purples", "Greys", "Reds"]
_DENSITY_MARKERS_DEFAULT = ["o", "s", "^", "D", "v", "P", "X", "*"]
_A_LINESTYLES_DEFAULT = ["-", "--", ":", "-.", (0, (3, 1, 1, 1)), (0, (5, 1))]
_A_FILLS_DEFAULT = [True, False]   # model 0 filled, others open; cycles for >2 models
_K_CMAP_POOL_DEFAULT = ["Oranges", "Blues", "Greens", "Purples", "Greys", "Reds"]


def sample_color_map(K_list, varying_values, key_fn, k_cmaps=None, light=0.4, dark=0.9):
    """Map each (K, varying_value) sample to a colour.

    `K` chooses a sequential colormap (a colour *family*); the varying value
    (e.g. density) chooses a shade within it, light->dark as the value increases.
    Returns ``{key_fn(K, value): (r, g, b, a)}``.
    """
    if k_cmaps is None:
        k_cmaps = _K_CMAPS_DEFAULT
    cm = {}
    n = len(varying_values)
    for ki, K in enumerate(K_list):
        cmap_name = k_cmaps.get(K, _K_CMAPS_FALLBACK[ki % len(_K_CMAPS_FALLBACK)])
        cmap = plt.get_cmap(cmap_name)
        for vi, value in enumerate(varying_values):
            frac = light + (dark - light) * (vi / (n - 1) if n > 1 else 0.5)
            cm[key_fn(K, value)] = cmap(frac)
    return cm


def sample_marker_map(K_list, varying_values, key_fn, markers=None):
    """Map each (K, varying_value) sample to a marker determined by the varying
    value alone, so the same density shares a marker across every K (and model).
    Returns ``{key_fn(K, value): marker}``.
    """
    if markers is None:
        markers = _DENSITY_MARKERS_DEFAULT
    assert len(markers) >= len(varying_values), (
        f"Not enough markers for {len(varying_values)} varying values.")
    value_marker = {value: markers[i] for i, value in enumerate(varying_values)}
    return {key_fn(K, value): value_marker[value] for K in K_list for value in varying_values}


def A_linestyle_map(A_list, linestyles=None):
    """Map each model `A` to a distinct linestyle. Returns ``{A: linestyle}``."""
    if linestyles is None:
        linestyles = _A_LINESTYLES_DEFAULT
    assert len(linestyles) >= len(A_list), (
        f"Not enough linestyles for {len(A_list)} models; add more to linestyles.")
    return {A: linestyles[i] for i, A in enumerate(A_list)}


def A_markerfill_map(A_list, fills=None):
    """{A: bool} marker-fill flags (True = solid, False = white-cored open ring).

    A second, redundant model channel for overlays: combined with `A_linestyle_map`
    it distinguishes models by both linestyle and marker fill.
    """
    if fills is None:
        fills = _A_FILLS_DEFAULT
    return {A: fills[i % len(fills)] for i, A in enumerate(A_list)}


def A_kcmaps_map(A_list, K_list, pool=None):
    """Assign each model a contiguous block of len(K_list) sequential colormaps
    from `pool`, so models differ by colour *family*. Returns ``{A: {K: cmap_name}}``,
    consumable as `sample_color_map`'s `k_cmaps`."""
    if pool is None:
        pool = _K_CMAP_POOL_DEFAULT
    nK = len(K_list)
    assert len(pool) >= nK * len(A_list), "K-cmap pool too small for models x K."
    return {A: {K: pool[ai * nK + ki] for ki, K in enumerate(K_list)}
            for ai, A in enumerate(A_list)}
#####

def save_plot_simple(num, filename, format=".png",
    xlabel=None, ylabel=None, xlim=None, ylim=None, grid=False, legend=False,
    ignore_xylim=False,
    fontsize_ticks=14, fontsize_legend=16, fontsize_label=22,
    axes_kw=None,
    dpi=100,
    show=False):

    plt.figure(num=num)
    if xlabel:
        plt.xlabel(xlabel, fontsize=fontsize_label)
    if ylabel:
        plt.ylabel(ylabel, fontsize=fontsize_label)
    if xlim and not ignore_xylim:
        plt.xlim(xlim)
    if ylim and not ignore_xylim:
        plt.ylim(ylim)
    plt.xticks(fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)
    if grid:
        plt.grid()
    if legend:
        plt.legend(fontsize=fontsize_legend)
    if axes_kw:
        plt.gca().set(**axes_kw)
    plt.savefig(filename+format, bbox_inches='tight', dpi=dpi)
    if show:
        plt.show()

def plt_scatter(plt_or_ax, data_x, data_y, data_z=None, size=None, **kwargs):
    ax = plt_or_ax if isinstance(plt_or_ax, (Axes, Axes3D)) else plt_or_ax.gca()
    # Input validation
    data_x = np.asarray(data_x)
    data_y = np.asarray(data_y)
    if len(data_x) != len(data_y):
        raise ValueError("data_x and data_y must have the same length")
    if data_z is not None:
        assert isinstance(ax, Axes3D)
        data_z = np.asarray(data_z)
        if len(data_z) != len(data_x):
            raise ValueError("data_z must have the same length as data_x and data_y")
    else:
        assert isinstance(ax, Axes)

    if size is None:
        if data_z is not None:
            return ax.scatter(data_x, data_y, data_z, **kwargs) # type: ignore
        else:
            return ax.scatter(data_x, data_y, **kwargs)
    else:
        if data_z is not None:
            raise NotImplementedError
        else:
            radius = size / 2
            circles = [plt_Circle((xi,yi), radius=radius, linewidth=0, **kwargs) for xi,yi in zip(data_x,data_y)]
            c = PatchCollection(circles, match_original=True)
            return ax.add_collection(c)

def plt_close(fignum, but_list=[], but_func=None):
    """Close matplotlib figures, with an exclusion mechanism.

    Args:
        fignum: Figures to close. "all", a single fig number/label/object,
            or a list/tuple of them.
        but_list: Figure(s) to keep open (numbers/labels/objects).
        but_func: Optional callable(label) -> bool. Receives each figure's
            *label* (str); return True to KEEP that figure open.

    A figure is kept if it's in but_list OR but_func(label) is True.
    """
    # Get a list of figure numbers
    if fignum == "all":
        fignum = list(plt.get_fignums())
    elif isinstance(fignum, (list, tuple)):
        fignum = list(map(get_fig_number, fignum))
    else:
        fignum = [get_fig_number(fignum)]

    # Make sure buyt_list is referencing figure numbers
    if not isinstance(but_list, (list, tuple)):
        but_list = [get_fig_number(but_list)]
    else:
        but_list = list(map(get_fig_number, but_list))
    # Make but_func into a callable to test if number or label should be closed (True means not closed)
    if callable(but_func):
        but = lambda num: but_func(get_fig_label(num)) or num in but_list
    else:
        but = lambda num: num in but_list

    # Look trhough numbers that are to be close. Close those who are not exculded those who are not members of the but list, or return false from the but function
    for num in fignum:
        if but(num):
            continue
        plt.close(num)

def get_fig_number(label_or_number):
    if plt.fignum_exists(label_or_number):
        return plt.figure(label_or_number).number
    else:
        return False

def get_fig_label(label_or_number):
    if plt.fignum_exists(label_or_number):
        if plt.figure(label_or_number).get_label() == "":
            return plt.figure(label_or_number).number
        else:
            return plt.figure(label_or_number).get_label()
    else:
        return False

def plot_ridgeline(data_list, H_values, colors, bins=None, centered=False, overlap=0.5, alpha=1., figsize=(10, 8), density=False):
    """
    Create a ridgeline plot for multiple histograms

    Parameters:
    - data_list: list of arrays, each containing data for one distribution
    - H_values: list of H values for labeling
    - colors: list of colors for each distribution
    - overlap: how much the distributions overlap (0-1)
    - alpha: transparency level
    - figsize: figure size
    """

    fig, ax = plt.subplots(figsize=figsize)

    # Calculate bin range from all data
    if bins is None:
        min_list= []; max_list= []
        for data in data_list:
            min_list.append(np.min(data)); max_list.append(np.max(data))
        bin_edges = np.arange(min(min_list), max(max_list) + 2)
    else:
        bin_edges = bins

    if centered:
        bin_edges = bin_edges - (bin_edges[1]-bin_edges[0])/2

    # Calculate spacing between distributions
    max_count = max([np.histogram(data, bins=bin_edges, density=density)[0].max() for data in data_list])
    spacing = max_count * (1 - overlap)

    # Plot each distribution
    for i, (data, H, color) in enumerate(zip(data_list, H_values, colors)):
        # Calculate histogram
        counts, bins = np.histogram(data, bins=bin_edges, density=density)
        counts = counts * (bins[1]-bins[0])
        bin_centers = (bins[:-1] + bins[1:]) / 2

        # Offset for ridgeline effect
        y_offset = i * spacing

        # Plot each bin as a rectangle
        for j in range(len(counts)):
            if counts[j] > 0:  # Only draw non-zero bins
                rect = plt.Rectangle((bins[j], y_offset),
                                   bins[j+1] - bins[j],
                                   counts[j],
                                   alpha=alpha,
                                   facecolor=color,
                                   edgecolor=color,
                                   linewidth=1,
                                   zorder=-H)
                ax.add_patch(rect)

        # Add outline
        ax.step(bins[1:], counts + y_offset, color=color, linewidth=1.5)

        # Add H label
        ax.text(bin_centers[-1] + 0.5, y_offset + max_count/2, f'H = {H}',
               ha='left', va='center', fontweight='bold')

    # Remove y-axis ticks and spines for cleaner look
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)

    # Adjust x limits to show all data
    ax.set_xlim(bin_edges[0], bin_edges[-1])
    return fig, ax
