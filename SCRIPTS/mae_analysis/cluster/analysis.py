"""Cluster compute stage: subprocess fan-out + the per-(ts,sample) worker.

Ported from the monolithic anal-cluster-h5.py. The physics is unchanged -- it
delegates to pyanal (separate_bulk_surface, cluster_E_anal, cluster_anal,
BOP_analysis). Sim-data paths come from the shared reverse-aware resolver, so
cluster gains `--reverse` for free once reverse cluster sweeps are configured.
"""

import os
import sys
import time
import pickle
import tempfile
import itertools
import subprocess

import numpy as np

import pyanal
from pyanal import ARay

from ..paths import sample_h5_path
from .config import (
    N_FULL_BOX, SIZE, l_list,
    ts_list, DENS_list, K_list, HEIGHT_list, H_list, seed_list,
    do_OVERWRITE_DATA, VERBOSE,
    do_MANUAL_BULK_SURFACE, do_USE_DUMB_THRESHOLD,
    do_ENERGETIC_CLUSTER, do_CLUSTER, do_BOP, do_ENERGETIC_BOP,
    SUBSTRATE_Z_OFFSET,
)

# __file__ is mae_analysis/cluster/analysis.py -> three dirnames reach the package parent,
# so the worker subprocess can be launched with `-m mae_analysis` regardless of cwd.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _concat(chunks):
    """Concatenate a list of per-seed numpy arrays, or return an empty array if none."""
    return np.concatenate(chunks) if chunks else np.empty(0)


def analysis_main(data_ARay: ARay, A):
    """Outer loop: spawn one worker subprocess per (ts, DENS, K, HEIGHT, H)."""
    for ts, DENS, K, HEIGHT, H in itertools.product(ts_list, DENS_list, K_list, HEIGHT_list, H_list):
        args = (data_ARay.file_path, A, ts, DENS, K, HEIGHT, H)
        with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
            pickle.dump(args, f)
            tmp_path = f.name
        try:
            subprocess.run([sys.executable, "-m", "mae_analysis", "cluster_analysis_per_ts_sample_H", tmp_path],
                            check=True, cwd=_PKG_PARENT)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Cluster worker failed for ts={ts}, DENS={DENS}, K={K}, HEIGHT={HEIGHT}, H={H} (exit code {e.returncode})")
        finally:
            os.remove(tmp_path)


def analysis_per_ts_sample_H(args_pickle_path: str):
    """Worker: cluster/BOP observables for one (ts, DENS, K, HEIGHT) across all H/seeds."""
    with open(args_pickle_path, "rb") as f:
        DATA_ARAY_PATH, A, ts, DENS, K, HEIGHT, H = pickle.load(f)

    data_ARay = ARay.from_file(DATA_ARAY_PATH)

    start_time_H = time.time()

    # Accumulate per-seed numpy arrays as chunks and concatenate once at store time
    # (avoids boxing numpy arrays into Python lists element-by-element).
    num_E_neighbors_chunks = []
    no_E_clusters_avg = 0; cluster_E_sizes_chunks = []
    num_neighbors_chunks = []
    cosines_chunks = []; cosine_avg_per_neighbourhood_chunks = []
    no_clusters_avg = 0; cluster_sizes_chunks = []

    # Pairwise distances are reduced to a histogram inside cluster_anal; accumulate
    # bin counts across seeds instead of raw distances. Bins match the plot consumer.
    dist_bins = np.arange(0, np.sqrt(800 + 4 * HEIGHT**2) + 1e-4, 0.5)
    dist_hist_total = np.zeros(len(dist_bins) - 1, dtype=np.int64)

    data_ARay_ts_sample_H = data_ARay.get(ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

    if data_ARay_ts_sample_H.is_set() and not do_OVERWRITE_DATA:
        return 0  # skip if all of the possible things have been calculated
    else:
        if VERBOSE:
            print(data_ARay_ts_sample_H.find_unset_values())

    seed_count = 0  # keep track of number of seeds with data
    for seed in seed_list:
        start_time_seed = time.time()

        h5_path = sample_h5_path(A, DENS, K, HEIGHT, H, seed)
        print(h5_path)

        if not os.path.isfile(h5_path):
            if VERBOSE:
                print("no file in", h5_path, flush=True)
            continue

        # Get the box_l, positions and dipole moments for the time step ts
        try:
            box_l, _, poss, poss_folded, dips, _ = pyanal.get_elastomer_h5(
                h5_path, A, time_step=ts, float_type=np.float32, int_type=np.int32)
        except Exception:
            if VERBOSE:
                print(f"timestep {ts} does not exist in", h5_path, flush=True)
            continue

        # shift particles to have bottom layer at z=0
        poss -= SUBSTRATE_Z_OFFSET
        poss_folded -= SUBSTRATE_Z_OFFSET

        assert poss.shape[0] == poss_folded.shape[0] == dips.shape[0] and poss.shape[1] == poss_folded.shape[1] == dips.shape[1] == 3, "Algo de errado não está certo."

        substrate_z_threshhold, bulk_z_threshhold, mask_array_bulk, _ = pyanal.separate_bulk_surface(
            poss, size=SIZE, manual_selection=do_MANUAL_BULK_SURFACE,
            dumb_threshold=(HEIGHT-SIZE) if do_USE_DUMB_THRESHOLD else 0)
        box_l_bulk = [box_l[0], box_l[1], bulk_z_threshhold - substrate_z_threshhold + SIZE]

        if do_ENERGETIC_CLUSTER and (not data_ARay_ts_sample_H["cutoff_E_cluster"].is_set() or do_OVERWRITE_DATA):
            cutoff_E_cluster = 0.0  # energy cutoff
            cutoff_r_cluster = 2**(1/6)*1.3  # max distance cutoff
            num_E_neighbors, no_E_clusters, cluster_E_sizes = pyanal.cluster_E_anal(
                poss, dips, mask=mask_array_bulk, cutoff_E=cutoff_E_cluster, cutoff_r=cutoff_r_cluster, box_l=box_l_bulk, pbc=True)

            num_E_neighbors_chunks.append(np.asarray(num_E_neighbors))
            no_E_clusters_avg += no_E_clusters
            cluster_E_sizes_chunks.append(np.asarray(cluster_E_sizes))

        if do_CLUSTER and (not data_ARay_ts_sample_H["cutoff_cluster"].is_set() or do_OVERWRITE_DATA):
            cutoff_cluster = 2**(1/6)*1.3
            num_neighbors, dist_hist, _dist_bin_edges, cosines, cosine_avg_per_neighbourhood, no_clusters, cluster_sizes = pyanal.cluster_anal(
                poss, dips, mask=mask_array_bulk, cutoff=cutoff_cluster, box_l=box_l_bulk, pbc=True, dist_bins=dist_bins)

            if A in ("SM", "SMfl") and H == 0:
                cosines = np.empty(0, dtype=np.float32); cosine_avg_per_neighbourhood = np.empty(0, dtype=np.float32)

            num_neighbors_chunks.append(np.asarray(num_neighbors))
            dist_hist_total += dist_hist
            cosines_chunks.append(np.asarray(cosines)); cosine_avg_per_neighbourhood_chunks.append(np.asarray(cosine_avg_per_neighbourhood))

            no_clusters_avg += no_clusters
            cluster_sizes_chunks.append(np.asarray(cluster_sizes))

        if do_BOP and (not data_ARay_ts_sample_H["cutoff_BOP"].is_set() or do_OVERWRITE_DATA):
            cutoff_BOP = 2**(1/6)*2.6
            l_vals, q_vals, w_vals = pyanal.BOP_analysis(
                poss, dips, mask=mask_array_bulk, substrate_z_threshhold=substrate_z_threshhold,
                cutoff_E=np.inf, cutoff_r=cutoff_BOP, l_range=l_list, box_l=box_l_bulk, pbc=True)

            if seed_count == 0:
                data_ARay.set({"l_list": l_vals}, ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
                q_vals_list = [list(q_vals[i]) for i in range(len(l_vals))]  # per-l, concatenated across seeds
            else:
                for i in range(len(l_vals)):
                    q_vals_list[i].extend(q_vals[i])

        if do_ENERGETIC_BOP and (not data_ARay_ts_sample_H["cutoff_E_BOP"].is_set() or do_OVERWRITE_DATA):
            cutoff_E_BOP = 0.0  # energy cutoff
            cutoff_r_BOP = 2**(1/6)*1.3  # max distance cutoff
            l_vals_E, q_vals_E, _ = pyanal.BOP_analysis(
                poss, dips, mask=mask_array_bulk, substrate_z_threshhold=substrate_z_threshhold,
                cutoff_E=cutoff_E_BOP, cutoff_r=cutoff_r_BOP, l_range=l_list, box_l=box_l_bulk, pbc=True)

            if seed_count == 0:
                data_ARay.set({"l_list_E": l_vals_E}, ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
                q_vals_E_list = [list(q_vals_E[i]) for i in range(len(l_vals_E))]  # per-l, concatenated across seeds
            else:
                for i in range(len(l_vals_E)):
                    q_vals_E_list[i].extend(q_vals_E[i])

        seed_count += 1
        end_time = time.time()
        print(ts, DENS, K, HEIGHT, H, f"seed {seed}", "---", end_time - start_time_seed)

    if seed_count < 1:
        return 0

    data_ARay.set({"time_step": ts, "seed_count": seed_count},
                    ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

    if do_ENERGETIC_CLUSTER and (not data_ARay_ts_sample_H["cutoff_E_cluster"].is_set() or do_OVERWRITE_DATA):
        data_ARay.set({"cutoff_E_cluster": cutoff_E_cluster, "cutoff_r_cluster": cutoff_r_cluster},
                        ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
        no_E_clusters_avg /= seed_count
        data_ARay.set({"num_E_neighbors_list": _concat(num_E_neighbors_chunks)},
                        ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
        data_ARay.set({"n_E_clusters_avg": no_E_clusters_avg, "cluster_E_sizes_list": _concat(cluster_E_sizes_chunks)},
                        ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

    if do_CLUSTER and (not data_ARay_ts_sample_H["cutoff_cluster"].is_set() or do_OVERWRITE_DATA):
        data_ARay.set({"cutoff_cluster": cutoff_cluster}, ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
        no_clusters_avg /= seed_count
        data_ARay.set({"num_neighbors_list": _concat(num_neighbors_chunks),
                        "distances_hist": dist_hist_total,
                        "distances_bins": dist_bins,
                        "cosines_list": _concat(cosines_chunks),
                        "cosine_avg_per_neighbourhood": _concat(cosine_avg_per_neighbourhood_chunks)},
                        ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
        data_ARay.set({"n_clusters_avg": no_clusters_avg, "cluster_sizes_list": _concat(cluster_sizes_chunks)},
                        ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

    if do_BOP and (not data_ARay_ts_sample_H["cutoff_BOP"].is_set() or do_OVERWRITE_DATA):
        data_ARay.set({"cutoff_BOP": cutoff_BOP}, ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
        l_vals = data_ARay_ts_sample_H.array["l_list"].astype(int)
        for i, l in enumerate(l_vals):
            data_ARay.set({f"q_{l}_vals": np.asarray(q_vals_list[i], dtype=np.float32),
                            f"q_{l}": np.nanmean(q_vals_list[i])},
                            ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

    if do_ENERGETIC_BOP and (not data_ARay_ts_sample_H["cutoff_E_BOP"].is_set() or do_OVERWRITE_DATA):
        data_ARay.set({"cutoff_E_BOP": cutoff_E_BOP, "cutoff_r_BOP": cutoff_r_BOP},
                        ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
        l_vals_E = data_ARay_ts_sample_H.array["l_list_E"].astype(int)
        for i, l in enumerate(l_vals_E):
            data_ARay.set({f"q_{l}_E_vals": np.asarray(q_vals_E_list[i], dtype=np.float32),
                            f"q_{l}_E": np.nanmean(q_vals_E_list[i])},
                            ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

    data_ARay.write_file(DATA_ARAY_PATH)

    end_time = time.time()
    print("---", ts, DENS, K, HEIGHT, H, "---", end_time - start_time_H, "---", "\n")

    # save data_ARay back to the file
    data_ARay.write_file(data_ARay.file_path)
    return 0
