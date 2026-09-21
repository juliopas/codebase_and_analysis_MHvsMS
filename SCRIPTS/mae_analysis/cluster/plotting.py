"""Cluster plots: per-H distribution histograms, BOP q_l, q4-q6 scatter, the
scalar n(_E)_clusters_avg vs H, and the legend set. Ported from
anal-cluster-h5.py::plot_H. `plot_liquid` is a standalone random-liquid
benchmark (not wired into the pipeline)."""

import os
import itertools

import numpy as np
from matplotlib import pyplot as plt

import pyanal
from pyanal.global_definitions import K_legend_dict

from .config import (
    OUTPUTPATH, SIZE,
    ts_list, DENS_list, K_list, HEIGHT_list, H_list, l_list,
    savefig_format, dpi, fontsize_label, fontsize_ticks, markersize_1,
    AUTO_XYLIM,
    H_EMPHASIS, HIST_OUTLINE_STYLE, HIST_FILL_ALPHA,
    HIST_FILL_ALPHA_LOW, HIST_CAP_LW, HIST_CAP_LW_THIN,
)
from ..style import combined_cm, sample_marker_map
from ..legends import save_handle_legend_set

plt.rcParams['figure.max_open_warning'] = 0


def _overlay_hist(num, hist, bins, color, *, emphasized, discrete, outline_style,
                  fill_alpha, fill_alpha_low, cap_lw, cap_lw_thin, cap_frac=0.9):
    """Overlay one H's histogram onto figure `num`.

    Every H gets a filled body so it stays visible: `emphasized` H fill at `fill_alpha`,
    the rest at the lower `fill_alpha_low`. `discrete=True` draws bars centred on each
    integer value (counts, BOP degree `l`); `discrete=False` draws stairs over the bin
    edges (continuous quantities).

    Outline on top: a full step outline only for an emphasized *continuous* figure in
    `"step"` mode; otherwise per-bin top caps (bold `cap_lw` for emphasized, thin
    `cap_lw_thin` for the rest). `hist`/`bins` are np.histogram-style
    (``len(bins) == len(hist) + 1``)."""
    plt.figure(num=num)
    hist = np.asarray(hist, dtype=float)
    bins = np.asarray(bins, dtype=float)
    centers = 0.5 * (bins[:-1] + bins[1:])
    widths = np.diff(bins)
    alpha = fill_alpha if emphasized else fill_alpha_low
    body_z = 2 if emphasized else 1
    if discrete:
        plt.bar(centers, hist, width=cap_frac * widths, color=color, alpha=alpha,
                align="center", zorder=body_z)
    else:
        plt.stairs(hist, bins, fill=True, color=color, alpha=alpha, zorder=body_z)
    if emphasized and not discrete and outline_style == "step":
        plt.stairs(hist, bins, fill=False, color=color, zorder=3)
    else:
        lw = cap_lw if emphasized else cap_lw_thin
        cf = cap_frac if discrete else 1.0   # caps span the full bin for continuous figures
        for i in range(len(hist)):
            half = 0.5 * cf * widths[i]
            plt.plot([centers[i] - half, centers[i] + half], [hist[i], hist[i]],
                     color=color, lw=lw, zorder=3)


def plot_H(data_ARay, A):
    BASE_DIR = os.path.join(OUTPUTPATH, A, "CLUSTER")
    os.makedirs(BASE_DIR, exist_ok=True)

    cm = combined_cm(ts_list=ts_list, H_list=H_list)   # H = ordered ramp, samples = systematic
    mm = sample_marker_map()                           # marker by sample key (matches surface)
    # Outline every H; alpha-fill only these anchors (default: the field-sweep endpoints).
    emphasis_set = set(H_EMPHASIS) if H_EMPHASIS is not None else {min(H_list), max(H_list)}

    # A cutoff/observable group is only in the dtype if its do_* flag was on at
    # compute time; guard every block so e.g. an energetic-cluster-only ARay
    # (do_CLUSTER off) doesn't trip over the missing cutoff_cluster field.
    available = {name for name, _ in data_ARay.dtype_str}

    def _any_set(aray_slice, field):
        return field in available and aray_slice[field].set_mask.any()

    def _is_set(aray_slice, field):
        return field in available and aray_slice[field].is_set()

    for ts in ts_list:
        for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
            save_name_sample = f"DENS{DENS:.2f}K{K}HEIGHT{HEIGHT}"

            data_ARay_ts_sample = data_ARay.get(ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT)

            if len(HEIGHT_list) == 1:        # constant HEIGHT
                label = f"{DENS:.2f}-{K_legend_dict[K]}"
                label_cm_or_mm = f"{DENS:.2f}-{K}"
                BASE_DIR_EXTERNAL = os.path.join(OUTPUTPATH, A, f"HEIGHT-{HEIGHT}")
            elif len(DENS_list) == 1:        # constant density of mag parts
                label = f"{K_legend_dict[K]}-{HEIGHT}"
                label_cm_or_mm = f"{K}-{HEIGHT}"
                BASE_DIR_EXTERNAL = os.path.join(OUTPUTPATH, A, f"DENS-{DENS}")
            else:
                raise ValueError("need to update plot design dicts. to accomodate more varying parameters")

            for H in H_list:
                data_ARay_ts_sample_H = data_ARay.get(ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

                def _ov(num, hist, bins, discrete=False):
                    """Overlay this H onto figure `num`. `discrete=True` -> integer-centred
                    bars; emphasised H fill bold, the rest fill faint + thin caps."""
                    _overlay_hist(num, hist, bins, cm[H], emphasized=(H in emphasis_set),
                                  discrete=discrete, outline_style=HIST_OUTLINE_STYLE,
                                  fill_alpha=HIST_FILL_ALPHA, fill_alpha_low=HIST_FILL_ALPHA_LOW,
                                  cap_lw=HIST_CAP_LW, cap_lw_thin=HIST_CAP_LW_THIN)

                if _is_set(data_ARay_ts_sample_H, "cutoff_E_cluster"):
                    edges = np.arange(int(data_ARay_ts_sample_H["num_E_neighbors_list"].array.max())+1)-0.5
                    hist, bins = np.histogram(data_ARay_ts_sample_H["num_E_neighbors_list"], bins=edges, density=True)
                    _ov("num_E_neighbours", hist, bins, discrete=True)

                    hist, bins = np.histogram(data_ARay_ts_sample_H["cluster_E_sizes_list"], bins=np.arange(data_ARay_ts_sample_H["num_E_neighbors_list"].array.shape[0]+1))
                    _ov("cluster_E_sizes", hist, bins)

                if _is_set(data_ARay_ts_sample_H, "cutoff_cluster"):
                    edges = np.arange(int(data_ARay_ts_sample_H["num_neighbors_list"].array.max())+1)-0.5
                    hist, bins = np.histogram(data_ARay_ts_sample_H["num_neighbors_list"], bins=edges, density=True)
                    _ov("num_neighbours", hist, bins, discrete=True)

                    counts = np.asarray(data_ARay_ts_sample_H["distances_hist"], dtype=np.float64)
                    bins = np.asarray(data_ARay_ts_sample_H["distances_bins"])
                    total = counts.sum()
                    prob = counts / total if total > 0 else counts
                    _ov("distances", prob, bins)

                    hist, bins = np.histogram(data_ARay_ts_sample_H["cosines_list"], bins=np.arange(-1, 1+0.0001, 0.1), density=True)
                    _ov("cosines", hist*(bins[1]-bins[0]), bins)

                    hist, bins = np.histogram(data_ARay_ts_sample_H["cosine_avg_per_neighbourhood"], bins=np.arange(-1, 1+0.0001, 0.1), density=True)
                    _ov("cosine_avg_per_neighbourhood", hist*(bins[1]-bins[0]), bins)

                    hist, bins = np.histogram(data_ARay_ts_sample_H["cluster_sizes_list"], bins=np.arange(data_ARay_ts_sample_H["num_neighbors_list"].array.shape[0]+1))
                    _ov("cluster_sizes", hist, bins)

                if _is_set(data_ARay_ts_sample_H, "cutoff_BOP"):
                    l_vals = data_ARay_ts_sample_H["l_list"].array.astype(int)
                    q_avg_list = [data_ARay_ts_sample_H[f"q_{l}"].array for l in l_vals]
                    _ov("q_l", q_avg_list, np.append(l_vals - 0.5, l_vals[-1] + 0.5), discrete=True)
                    for l in l_vals:
                        plt.figure(num=f"q_{l}_vals")
                        hist, bins = np.histogram(data_ARay_ts_sample_H[f"q_{l}_vals"], bins=np.arange(0, 1., 0.01))
                        plt.plot((bins[:-1]+bins[1:])/2, hist, color=cm[H], alpha=0.2, label=f"H{H}")

                    # Point spread over q_4 and q_6 (and may add other combinations)
                    plt.figure(num="q_6_q_4")
                    plt.plot(data_ARay_ts_sample_H["q_4_vals"].array, data_ARay_ts_sample_H["q_6_vals"].array, 'o', alpha=0.5, color=cm[H], label=f"H{H}", markersize=1.5)

                if _is_set(data_ARay_ts_sample_H, "cutoff_E_BOP"):
                    l_vals = data_ARay_ts_sample_H["l_list_E"].array.astype(int)
                    q_avg_list = [data_ARay_ts_sample_H[f"q_{l}_E"].array for l in l_vals]
                    _ov("q_l_E", q_avg_list, np.append(l_vals - 0.5, l_vals[-1] + 0.5), discrete=True)
                    for l in l_vals:
                        plt.figure(num=f"q_{l}_E_vals")
                        hist, bins = np.histogram(data_ARay_ts_sample_H[f"q_{l}_E_vals"], bins=np.arange(0, 1., 0.01))
                        plt.plot((bins[:-1]+bins[1:])/2, hist, color=cm[H], alpha=0.2, label=f"H{H}")

                    # Point spread over q_4 and q_6 (and may add other combinations)
                    plt.figure(num="q_6_q_4_E")
                    plt.plot(data_ARay_ts_sample_H["q_4_E_vals"].array, data_ARay_ts_sample_H["q_6_E_vals"].array, 'o', alpha=0.5, color=cm[H], label=f"H{H}", markersize=1.5)

            if _any_set(data_ARay_ts_sample, "cutoff_E_cluster"):
                outputpath_ts_sample_H = os.path.join(BASE_DIR, "cluster_H", f"t{ts}", save_name_sample)
                os.makedirs(outputpath_ts_sample_H, exist_ok=True)
                pyanal.save_plot_simple(num="num_E_neighbours",
                                filename=os.path.join(outputpath_ts_sample_H, "num_E_neighbours"), format=savefig_format,
                                xlabel=r"Number of Energetic Neighbors", ylabel=r"density",
                                xlim=[-0.5,7.5], ylim=[0., 0.33], ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                pyanal.save_plot_simple(num="cluster_E_sizes",
                                filename=os.path.join(outputpath_ts_sample_H, "cluster_E_sizes"), format=savefig_format,
                                xlabel=r"Energetic cluster size (in number of particles)", ylabel=r"Frequency", ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)

                plt.figure(num="n_E_clusters_avg")
                n_E_clusters_avg = data_ARay_ts_sample["n_E_clusters_avg"].get(H=H_list)
                H_list_plot = [H for H in H_list if n_E_clusters_avg.get(H=H).is_set()]
                n_E_clusters_avg = n_E_clusters_avg.get_set_entries().array
                plt.plot(H_list_plot, n_E_clusters_avg, color=cm[label_cm_or_mm], marker=mm[label_cm_or_mm], markersize=markersize_1, linestyle="dashed", label=label_cm_or_mm)

            if _any_set(data_ARay_ts_sample, "cutoff_cluster"):
                outputpath_ts_sample_H = os.path.join(BASE_DIR, "cluster_H", f"t{ts}", save_name_sample)
                os.makedirs(outputpath_ts_sample_H, exist_ok=True)
                pyanal.save_plot_simple(num="num_neighbours",
                                filename=os.path.join(outputpath_ts_sample_H, "num_neighbours"), format=savefig_format,
                                xlabel=r"Number of Neighbors", ylabel=r"density",
                                xlim=[-0.5,10.5], ylim=[0., 0.252], ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                pyanal.save_plot_simple(num="distances",
                                filename=os.path.join(outputpath_ts_sample_H, "distances"), format=savefig_format,
                                xlabel=r"$\Delta_r$", ylabel=r"P", ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                pyanal.save_plot_simple(num="cosines",
                                filename=os.path.join(outputpath_ts_sample_H, "cosines"), format=savefig_format,
                                xlabel=r"$\cos(\theta)$", ylabel=r"P", 
                                xlim=[-1., 1.], ylim=[0., 1.05], ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                pyanal.save_plot_simple(num="cosine_avg_per_neighbourhood",
                                filename=os.path.join(outputpath_ts_sample_H, "cosines_neighb"), format=savefig_format,
                                xlabel=r"$\left<\cos(\theta)\right>_i$", ylabel=r"P",
                                xlim=[-1., 1.], ylim=[0.,1.05], ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                pyanal.save_plot_simple(num="cluster_sizes",
                                filename=os.path.join(outputpath_ts_sample_H, "cluster_sizes"), format=savefig_format,
                                xlabel=r"Cluster size (in number of particles)", ylabel=r"Frequency", ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)

                plt.figure(num="n_clusters_avg")
                n_clusters_avg = data_ARay_ts_sample["n_clusters_avg"].get(H=H_list)
                H_list_plot = [H for H in H_list if n_clusters_avg.get(H=H).is_set()]
                n_clusters_avg = n_clusters_avg.get_set_entries().array
                plt.plot(H_list_plot, n_clusters_avg, color=cm[label_cm_or_mm], marker=mm[label_cm_or_mm], markersize=markersize_1, linestyle="dashed", label=label_cm_or_mm)

            if _any_set(data_ARay_ts_sample, "cutoff_BOP"):
                outputpath_ts_sample_H = os.path.join(BASE_DIR, "BOP_H", f"t{ts}", save_name_sample)
                os.makedirs(outputpath_ts_sample_H, exist_ok=True)
                pyanal.save_plot_simple(num="q_l",
                                filename=os.path.join(outputpath_ts_sample_H, "BOP_q_l"), format=savefig_format,
                                xlabel=r"$l$", ylabel=r"$\left<q_l\right>$",
                                ylim=[0, 0.16], ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                pyanal.save_plot_simple(num="q_6_q_4",
                                filename=os.path.join(outputpath_ts_sample_H, "BOP_q_6_4"), format=savefig_format,
                                xlabel=r"$\left<q_4\right>$", ylabel=r"$\left<q_6\right>$",
                                xlim=[0, 0.35], ylim=[0, 0.35], grid=True, ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                for l in l_vals:
                    pyanal.save_plot_simple(num=f"q_{l}_vals",
                                    filename=os.path.join(outputpath_ts_sample_H, f"BOP_q_{l}.png"), format=savefig_format,
                                    xlabel=rf"$\left<q_{{{l}}}\right>$", ylabel=r"$P$", ignore_xylim=AUTO_XYLIM,
                                    fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)

            if _any_set(data_ARay_ts_sample, "cutoff_E_BOP"):
                outputpath_ts_sample_H = os.path.join(BASE_DIR, "BOP_E_H", f"t{ts}", save_name_sample)
                os.makedirs(outputpath_ts_sample_H, exist_ok=True)
                pyanal.save_plot_simple(num="q_l_E",
                                filename=os.path.join(outputpath_ts_sample_H, "BOP_q_l_E"), format=savefig_format,
                                xlabel=r"$l$", ylabel=r"$\left<q_l\right>$",
                                ylim=[0, 1.05], ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                pyanal.save_plot_simple(num="q_6_q_4_E",
                                filename=os.path.join(outputpath_ts_sample_H, "BOP_q_6_4_E"), format=savefig_format,
                                xlabel=r"$\left<q_4\right>$", ylabel=r"$\left<q_6\right>$",
                                xlim=[0, 1.05], ylim=[0, 1.05], grid=True, ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
                for l in l_vals:
                    pyanal.save_plot_simple(num=f"q_{l}_E_vals",
                                    filename=os.path.join(outputpath_ts_sample_H, f"BOP_q_{l}_E.png"), format=savefig_format,
                                    xlabel=rf"$\left<q_{{{l}}}\right>$", ylabel=r"$P$", ignore_xylim=AUTO_XYLIM,
                                    fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)

            pyanal.plt_close("all", but_func=lambda label: label in ("n_clusters_avg", "n_E_clusters_avg"))

        outputpath_external_sample_H_plots = os.path.join(BASE_DIR_EXTERNAL, "1-sample_H", f"t{ts}")
        os.makedirs(outputpath_external_sample_H_plots, exist_ok=True)
        if _any_set(data_ARay_ts_sample, "cutoff_E_cluster"):
            pyanal.save_plot_simple(num="n_E_clusters_avg",
                             filename=os.path.join(outputpath_external_sample_H_plots, "n_E_clusters_avg"), format=savefig_format,
                             xlabel=r"$H$", ylabel="Number of clusters", ignore_xylim=AUTO_XYLIM,
                             fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
        if _any_set(data_ARay_ts_sample, "cutoff_cluster"):
            pyanal.save_plot_simple(num="n_clusters_avg",
                             filename=os.path.join(outputpath_external_sample_H_plots, "n_clusters_avg"), format=savefig_format,
                             xlabel=r"$H$", ylabel="Number of clusters", ignore_xylim=AUTO_XYLIM,
                             fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)

        plt.close("all")

    # output legend labels: frameless H-colour legends, per-style subfolders, 1&2 col
    # (same scheme as the surface plots).
    outputpath_legend_H = os.path.join(BASE_DIR, "legend_H")
    save_handle_legend_set(outputpath_legend_H, f"{A}-{ts_list[0]}", cm, H_list,
                           lambda H: f"H{H}", dpi_=dpi)
    plt.close("all")


def plot_liquid(box_l=(10, 10, 10), dens=0.3, l_list=tuple(range(2, 12 + 1, 1))):
    """Random-liquid reference benchmark (standalone; not called by the pipeline).

    The original also tried to overlay a BOP q_l bar/scatter using analysis-loop
    variables (cm/H/data_ARay_ts_sample_H) that don't exist here -- that broken
    trailing block is omitted; the structural histograms below are the useful part.
    """
    box_l = list(box_l)
    n_part = round(dens * np.prod(box_l) / (4/3 * np.pi * SIZE**3))

    poss = np.random.random((n_part, 3))
    dips = pyanal.generic_functions.generate_random_unit_vectors(n_part)

    cutoff_cluster = SIZE * 2**(1/6)*2.6
    num_neighbors, dist_hist, dist_bin_edges, cosines, cosine_avg_per_neighbourhood, no_clusters, cluster_sizes = pyanal.cluster_anal(
        poss, dips, cutoff=cutoff_cluster, box_l=box_l, pbc=True)

    outputpath_liquid = os.path.join(OUTPUTPATH, "liquid", "CLUSTER", f"dens{dens}-n{n_part}")

    def _hist_bar(num, data, bins, density, fname, xlabel, ylabel):
        plt.figure(num=num, clear=True)
        hist, b = np.histogram(data, bins=bins, density=density)
        w = b[1] - b[0]
        plt.stairs(hist * (w if density else 1), b, fill=True, alpha=0.3, color="pink")
        pyanal.save_plot_simple(num=num, filename=os.path.join(outputpath_liquid, fname), format=savefig_format,
                                xlabel=xlabel, ylabel=ylabel, ignore_xylim=AUTO_XYLIM,
                                fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)

    _hist_bar("num_neighbours", num_neighbors, np.arange(12+1)-0.5, False, "num_neighbours", r"Number of Neighbors", r"Frequency")
    # distances come back pre-histogrammed (counts over dist_bin_edges); plot as probability per bin.
    plt.figure(num="distances", clear=True)
    _dcounts = np.asarray(dist_hist, dtype=np.float64); _dtotal = _dcounts.sum()
    plt.stairs((_dcounts / _dtotal if _dtotal > 0 else _dcounts), dist_bin_edges, fill=True, alpha=0.3, color="pink")
    pyanal.save_plot_simple(num="distances", filename=os.path.join(outputpath_liquid, "distances"), format=savefig_format,
                            xlabel=r"$\Delta_r$", ylabel=r"P", ignore_xylim=AUTO_XYLIM,
                            fontsize_label=fontsize_label, fontsize_ticks=fontsize_ticks, dpi=dpi)
    _hist_bar("cosines", cosines, np.arange(-1, 1+0.0001, 0.1), True, "cosines", r"$\cos(\theta)$", r"P")
    _hist_bar("cosines_neighb", cosine_avg_per_neighbourhood, np.arange(-1, 1+0.0001, 0.1), True, "cosines_neighb", r"$\left<\cos(\theta)\right>_i$", r"P")
    _hist_bar("cluster_sizes", cluster_sizes, np.arange(num_neighbors.shape[0]+1), False, "cluster_sizes", r"Cluster size (in number of particles)", r"Frequency")
