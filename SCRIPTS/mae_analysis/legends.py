"""Standalone legend-image savers, shared by the surface and cluster plots.

Legends are saved as separate, frameless, transparent PNGs (so individual pieces can be
cropped and composited consistently). `save_handle_legend_set` emits the four style
variants (patch / patch_alpha / line / point) into per-variant subfolders, each in both a
1-column and a 2-column version.
"""

import os

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D as plt_Line2D
from matplotlib.patches import Patch as plt_Patch

from .config import fontsize_legend, HIST_FILL_ALPHA


def save_frameless_legend(outdir, filename, handles, ncols, dpi_):
    """Render `handles` as a standalone, box-less, transparent legend image."""
    os.makedirs(outdir, exist_ok=True)
    fig = plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.legend(handles=handles, loc="center", ncols=ncols,
               fontsize=fontsize_legend, frameon=False)
    plt.savefig(filename, dpi=dpi_, bbox_inches="tight", transparent=True)
    plt.close(fig)


def save_handle_legend_set(outdir, stem, cm, keys, label_fn, dpi_=600, alpha=HIST_FILL_ALPHA):
    """Save patch / patch_alpha / line / point legend images for a key set.

    Each style variant lands in its own subfolder; both a 1-column and a 2-column
    version are written (2-column skipped for a single key). Keys are drawn in
    ascending order so the colour ramp reads monotonically. `alpha` sets the
    patch_alpha shade (match the histogram fill alpha)."""
    keys = sorted(keys)
    patch_h = [plt_Patch(color=cm[k], label=label_fn(k)) for k in keys]
    alpha_h = [plt_Patch(color=cm[k], alpha=alpha, label=label_fn(k)) for k in keys]
    line_h  = [plt_Line2D([0], [0], color=cm[k], label=label_fn(k)) for k in keys]
    point_h = [plt_Line2D([0], [0], marker='o', linestyle="", color=cm[k], label=label_fn(k)) for k in keys]
    ncols_list = (1,) if len(keys) <= 1 else (1, 2)
    for handles, variant in ((patch_h, "patch"), (alpha_h, "patch_alpha"),
                             (line_h, "line"), (point_h, "point")):
        variant_dir = os.path.join(outdir, variant)
        for ncols in ncols_list:
            save_frameless_legend(variant_dir,
                                  os.path.join(variant_dir, f"{stem}-legend-c{ncols}.png"),
                                  handles, ncols, dpi_)
