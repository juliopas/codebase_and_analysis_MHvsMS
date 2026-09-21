"""The heavy calculation stage: bond->particles preprocessing, the outer loop
that fans work out to subprocess workers, and the per-(ts,sample) worker itself.

`analysis_per_ts_sample` is now a thin orchestrator: it owns the cross-seed
accumulators and the writes into the ARay, and delegates every physics
calculation to `observables.py`.
"""

import os
import sys
import time
import pickle
import tempfile
import itertools
import subprocess

import numpy as np
from tqdm import tqdm

import pyanal
from pyanal import ARay

from .config import (
    READ_BASE_DIR, BOND_PART_DIR, N_FULL_BOX, SIZE,
    ts_list, DENS_list, K_list, HEIGHT_list, H_list, seed_list,
    do_OVERWRITE_DATA, VERBOSE,
    do_ENERGIES, do_MAGNETIZATION, do_HEIGHTS, do_PEAK_STATISTICS,
    do_DENSITY_Z, do_VOLUME, do_USE_BOND_TO_PARTS, do_USE_SURFACE_SHAPE,
    plot_HIST, plot_SURFACE_SHAPE,
    VARIANTS, VARIANTS_NO_SURFACE,
)
from ..paths import sample_h5_path, bond_part_path
from .data import resolve_label_and_basedir
from .observables import (
    SurfaceResult, HeightAccum,
    load_sample_frame, detect_surfaces,
    calc_energies_for_seed, calc_magnetization_for_seed,
    heights_and_peaks_for_seed, density_z_for_seed,
)

# Directory that contains the `mae_analysis` package, so the worker subprocess
# can be launched with `-m mae_analysis` regardless of the caller's cwd.
# __file__ is mae_analysis/surface/analysis.py -> three dirnames reach the package parent.
_PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def bond_to_parts(A):
    """Created particles to represent the elastomer bonds, for a better estimate of
    the volume and surface of the elastomer. Saves the bond particles' positions in
    an h5 file, with connectivity info (which bond particles belong to which bond,
    and which bond belongs to which espresso particle)."""
    bond_part_center_distance = 0.25
    for DENS, K, HEIGHT, H, seed in tqdm(
            itertools.product(DENS_list, K_list, HEIGHT_list, H_list, seed_list),
            total=len(DENS_list)*len(K_list)*len(HEIGHT_list)*len(H_list)*len(seed_list),
            desc="Bonding..."):
        # Iterate over all combinations of elastomer parameters (physical params,
        # initial positions by seed, and applied external field).

        # check if simulations h5 file exists for these parameters
        h5_path = sample_h5_path(A, DENS, K, HEIGHT, H, seed)
        if not os.path.isfile(h5_path):
            continue  # file does not exist
        if not pyanal.elastomer_h5_readable(h5_path):
            continue  # file is not readable (probably corrupted: bad object header version number)

        # path to bond file for the elastomer parameters
        bonds_to_paths = bond_part_path(A, DENS, K, HEIGHT, H, seed)

        # check if bond file exists
        bond_file_exists = False
        if os.path.isfile(bonds_to_paths):
            if do_OVERWRITE_DATA:
                os.remove(bonds_to_paths)
            else:
                bond_file_exists = True

        # Check if bonds to parts are already saved and save accordingly
        for ts in ts_list:
            if not pyanal.elastomer_h5_ts_exists(h5_path, time=ts):
                continue  # MAE h5 file exists, but does not have the time ts

            if not bond_file_exists:
                # Get box_l, positions and dipole moments of the relevant elastomer magnetic
                # particles for the time step ts, calculate the bond particle positions, and
                # write into a new file in bonds_to_paths
                pyanal.calculate_and_write_bond_particles_h5(h5_path, bonds_to_paths, A, bond_part_center_distance, time_step=ts, new=True)
            else:
                n_bond_parts, size = pyanal.get_n_bond_to_parts(bonds_to_paths, time_step=ts)
                if size not in (None, bond_part_center_distance):
                    raise ValueError("Must use overwrite to change distance between bond particles. To change the "
                                     "size of especific ones, run only for those that you which to change."
                                     + f"\n\t{size}!={bond_part_center_distance}")
                if n_bond_parts is None or do_OVERWRITE_DATA:
                    pyanal.calculate_and_write_bond_particles_h5(h5_path, bonds_to_paths, A, bond_part_center_distance, time_step=ts, new=False)


def analysis_main(data_ARay: ARay, A):
    """Outer loop: spawn one worker subprocess per (ts, DENS, K, HEIGHT)."""
    for ts in ts_list:  # Iterate over time steps
        for DENS, K, HEIGHT in itertools.product(DENS_list, K_list, HEIGHT_list):
            # For each combination of elastomer parameters (volume density, spring
            # elastic constant distribution, and layer height).
            _, label_cm_or_mm, BASE_DIR = resolve_label_and_basedir(A, DENS, K, HEIGHT)
            args = (data_ARay.file_path, BASE_DIR, label_cm_or_mm, A, ts, DENS, K, HEIGHT)
            with tempfile.NamedTemporaryFile(mode="wb", suffix=".pkl", delete=False) as f:
                pickle.dump(args, f)
                tmp_path = f.name
            try:
                subprocess.run([sys.executable, "-m", "mae_analysis", "analysis_per_ts_sample", tmp_path],
                               check=True, cwd=_PKG_PARENT)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Worker failed for ts={ts}, DENS={DENS}, K={K}, HEIGHT={HEIGHT} (exit code {e.returncode})")
            finally:
                os.remove(tmp_path)


def analysis_per_ts_sample(args_pickle_path: str):
    """Worker: compute every enabled observable for one (ts, DENS, K, HEIGHT)
    across all H and seeds, accumulate, and write into the ARay file."""
    with open(args_pickle_path, "rb") as f:
        DATA_ARAY_PATH, BASE_DIR, label_cm_or_mm, A, ts, DENS, K, HEIGHT = pickle.load(f)

    data_ARay = ARay.from_file(DATA_ARAY_PATH)

    cm = mm = None
    if plot_HIST or plot_SURFACE_SHAPE:
        cm = pyanal.create_cm(ts_list=ts_list, H_list=H_list)  # color map dict
        mm = pyanal.create_mm()                                # marker map dict

    H_list_plot = []
    height_at_h0 = None  # threaded across the H loop: set at H==0, reused for H>0
    for H in H_list:
        start_time_H = time.time()

        E_bond = 0; E_dip = 0; E_Zee = 0; E_Zee_dipm = 0; E_chains = 0
        net_m = np.zeros((3)); net_m_dipm = np.zeros((3))
        # height accumulators keyed by variant suffix
        rms   = {v.suffix: 0. for v in VARIANTS}
        h_avg = {v.suffix: 0. for v in VARIANTS}
        # peak / volume / dens_z accumulators (no surface variants)
        n_peaks       = {v.suffix: 0. for v in VARIANTS_NO_SURFACE}
        peak_h_avg    = {v.suffix: 0. for v in VARIANTS_NO_SURFACE}
        peak_h_std    = {v.suffix: 0. for v in VARIANTS_NO_SURFACE}
        peak_dist_avg = {v.suffix: 0. for v in VARIANTS_NO_SURFACE}
        peak_dist_std = {v.suffix: 0. for v in VARIANTS_NO_SURFACE}
        volume_avg    = {v.suffix: 0. for v in VARIANTS_NO_SURFACE}
        dens_z: dict[str, np.ndarray | None] = {v.suffix: None for v in VARIANTS_NO_SURFACE}

        accum_height = HeightAccum(rms=rms, h_avg=h_avg, n_peaks=n_peaks,
                            peak_h_avg=peak_h_avg, peak_h_std=peak_h_std,
                            peak_dist_avg=peak_dist_avg, peak_dist_std=peak_dist_std)

        data_ARay_ts_sample_H = data_ARay.get(ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

        if data_ARay_ts_sample_H.is_set() and not do_OVERWRITE_DATA:
            continue  # skip if all of the possible things have been calculated
        else:
            if VERBOSE:
                print(data_ARay_ts_sample_H.find_unset_values())

        dens_z_bin_edges = None
        seed_count = 0  # keep track of number of seeds with data
        for seed in seed_list:
            start_time_seed = time.time()

            h5_path = sample_h5_path(A, DENS, K, HEIGHT, H, seed)

            if not os.path.isfile(h5_path):
                if VERBOSE:
                    print("no file in", h5_path, flush=True)
                continue

            if seed_count == 0:
                H_list_plot.append(H)

            frame, height_at_h0 = load_sample_frame(
                A, ts, DENS, K, HEIGHT, H, seed, h5_path, BASE_DIR, label_cm_or_mm, height_at_h0)
            if frame is None:
                continue  # timestep missing in this file

            surfaces = SurfaceResult()
            if do_USE_SURFACE_SHAPE:
                compute_vol_no_bonds   = ( (not data_ARay_ts_sample_H["volume_avg"].is_set() or do_OVERWRITE_DATA) and do_VOLUME)
                compute_vol_with_bonds = ( (not data_ARay_ts_sample_H["volume_avg_with_bonds"].is_set() or do_OVERWRITE_DATA) and do_USE_BOND_TO_PARTS and do_VOLUME )
                surfaces = detect_surfaces(frame, compute_vol_no_bonds, compute_vol_with_bonds)
                volume_avg[""] += surfaces.vol_no_bonds
                if compute_vol_with_bonds:
                    assert surfaces.vol_with_bonds is not None
                    volume_avg["_with_bonds"] += surfaces.vol_with_bonds

            if do_ENERGIES and (not data_ARay_ts_sample_H["E_dip"].is_set() or do_OVERWRITE_DATA):
                E_bond_seed, E_dip_seed, E_Zee_seed, E_Zee_dipm_seed, E_chain_seed = calc_energies_for_seed(frame)
                E_bond += E_bond_seed
                E_dip += E_dip_seed
                E_Zee += E_Zee_seed
                E_Zee_dipm += E_Zee_dipm_seed
                E_chains += E_chain_seed

            if do_MAGNETIZATION and (not data_ARay_ts_sample_H["m_z"].is_set() or do_OVERWRITE_DATA):
                m_seed, m_dipm_seed = calc_magnetization_for_seed(frame)
                net_m += m_seed
                net_m_dipm += m_dipm_seed

            if do_HEIGHTS:
                heights_and_peaks_for_seed(frame, surfaces, accum_height, cm, data_ARay_ts_sample_H)

            if do_DENSITY_Z:
                hist, hist_with_bonds, dens_z_bin_edges = density_z_for_seed(frame)
                dens_z[""] = hist if dens_z[""] is None else dens_z[""] + hist
                if do_USE_BOND_TO_PARTS:
                    dens_z["_with_bonds"] = hist_with_bonds if dens_z["_with_bonds"] is None else dens_z["_with_bonds"] + hist_with_bonds

            seed_count += 1
            end_time = time.time()
            print(ts, DENS, K, HEIGHT, H, f"seed {seed}", "---", end_time - start_time_seed)

        if seed_count < 1:
            continue

        data_ARay.set({"time_step": ts, "seed_count": seed_count},
                      ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

        if do_ENERGIES and (not data_ARay_ts_sample_H["E_dip"].is_set() or do_OVERWRITE_DATA):
            E_bond /= seed_count; E_dip /= seed_count; E_Zee /= seed_count
            E_Zee_dipm /= seed_count; E_chains /= seed_count
            data_ARay.set(
                {"E_bond": E_bond, "E_dip": E_dip, "E_Zee": E_Zee, "E_Zee_dipm": E_Zee_dipm,
                 "E_chains": E_chains},
                ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

        if do_MAGNETIZATION and (not data_ARay_ts_sample_H["m_z"].is_set() or do_OVERWRITE_DATA):
            net_m = net_m / seed_count; net_m_dipm = net_m_dipm / seed_count
            M_z = net_m[2] / (np.pi / 6 * SIZE**3) * DENS
            data_ARay.set({"m_z": net_m[2], "m_z_dipm": net_m_dipm[2],
                           "M_z": M_z},
                          ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

        if do_HEIGHTS:
            # height & rms for all variants
            for v in VARIANTS:
                if not data_ARay_ts_sample_H[f"rms{v.suffix}"].is_set() or do_OVERWRITE_DATA:
                    rms[v.suffix]   /= seed_count
                    h_avg[v.suffix] /= seed_count
                    data_ARay.set({f"rms{v.suffix}": rms[v.suffix], f"h_avg{v.suffix}": h_avg[v.suffix]},
                                  ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
            # peak statistics (no surface variants)
            if do_PEAK_STATISTICS:
                for v in VARIANTS_NO_SURFACE:
                    if not data_ARay_ts_sample_H[f"n_peaks{v.suffix}"].is_set() or do_OVERWRITE_DATA:
                        n_peaks[v.suffix]       /= seed_count
                        peak_h_avg[v.suffix]    /= seed_count
                        peak_h_std[v.suffix]    /= seed_count
                        peak_dist_avg[v.suffix] /= seed_count
                        peak_dist_std[v.suffix] /= seed_count
                        data_ARay.set(
                            {f"n_peaks{v.suffix}":            n_peaks[v.suffix],
                             f"peak_h_avg{v.suffix}":         peak_h_avg[v.suffix],
                             f"peak_h_std{v.suffix}":         peak_h_std[v.suffix],
                             f"peak_distance_avg{v.suffix}":  peak_dist_avg[v.suffix],
                             f"peak_distance_std{v.suffix}":  peak_dist_std[v.suffix]},
                            ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

        if do_DENSITY_Z:
            assert dens_z_bin_edges is not None
            bin_size_half = (dens_z_bin_edges[1] - dens_z_bin_edges[0]) / 2
            if not data_ARay_ts_sample_H["dens_z_bin_edges"].is_set() or do_OVERWRITE_DATA:
                data_ARay.set({"dens_z_bin_edges": dens_z_bin_edges},
                              ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)
            for v in VARIANTS_NO_SURFACE:
                if not data_ARay_ts_sample_H[f"dens_z{v.suffix}"].is_set() or do_OVERWRITE_DATA:
                    assert (dz := dens_z[v.suffix]) is not None
                    dz /= seed_count
                    thickness = (sum(dz[i] * (edge + bin_size_half)
                                     for i, edge in enumerate(dens_z_bin_edges[:-1]))
                                 / np.sum(dz))
                    data_ARay.set({f"dens_z{v.suffix}": dz,
                                   f"thickness{v.suffix}": thickness},
                                  ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

        if do_VOLUME:
            for v in VARIANTS_NO_SURFACE:
                if not data_ARay_ts_sample_H[f"volume_avg{v.suffix}"].is_set() or do_OVERWRITE_DATA:
                    volume_avg[v.suffix] /= seed_count
                    data_ARay.set({f"volume_avg{v.suffix}": volume_avg[v.suffix]},
                                  ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H)

        data_ARay.write_file(DATA_ARAY_PATH)

        end_time = time.time()
        print("---", ts, DENS, K, HEIGHT, H, "---", end_time - start_time_H, "---", "\n")

    # save data_ARay back to the file
    data_ARay.write_file(data_ARay.file_path)
    return 0
