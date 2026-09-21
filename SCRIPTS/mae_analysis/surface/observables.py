"""Per-seed observable calculations, extracted from the old monolithic worker.

Each helper takes a loaded `SampleFrame` (one ts/sample/H/seed) plus whatever it
needs, computes ONE observable, and returns its contribution (or mutates the
accumulator passed in). The orchestrator in `analysis.py` owns the cross-seed
accumulation and the writes to the ARay.
"""

import os
import math
from dataclasses import dataclass
from numpy.typing import NDArray

import numpy as np
from matplotlib import pyplot as plt

import pyanal
from pyanal.global_definitions import A_help_dict

from ..paths import bond_part_path
from .config import (
    READ_BASE_DIR, N_FULL_BOX, OUTPUTPATH,
    SUBSTRATE_Z_OFFSET, SIZE, SIZE_BOND_PART, ITIM_DIAMETER, ITIM_DIAMETER_INTERP,
    savefig_format, dpi, fontsize_legend, VERBOSE,
    do_USE_HEIGHT_AT_H0, do_USE_BOND_TO_PARTS, do_USE_SURFACE_SHAPE, do_VOLUME,
    do_PEAK_STATISTICS, do_OVERWRITE_DATA,
    save_HEIGHT_MAPS, plot_HIST, plot_SURFACE_SHAPE, plot_PEAK_DISTRIBUTIONS,
)

plt.rcParams['figure.max_open_warning'] = 0

if save_HEIGHT_MAPS:
    from gwyfile.objects import GwyContainer, GwyDataField


# ---------------------------------------------------------------- containers --

@dataclass
class SampleFrame:
    """Everything loaded/derived for one (ts, sample, H, seed)."""
    # identity / paths
    A: str; ts: int; DENS: float; K: str; HEIGHT: int; H: float; seed: int
    BASE_DIR: str; save_name_sample: str; save_name_sample_H: str; label_cm_or_mm: str
    # magnetic particles
    box_l: NDArray[np.floating]; ids: NDArray[np.integer]; poss: NDArray[np.floating]; poss_folded: NDArray[np.floating]
    dips: NDArray[np.floating]; bonds: list; radius: NDArray[np.floating]; n_parts: int
    HEIGHT_AT_H0: float | None = None
    # bond particles (only when do_USE_BOND_TO_PARTS)
    poss_bond_parts_folded: np.ndarray | None = None
    poss_all_folded: np.ndarray | None = None
    radius_bonds: np.ndarray | None = None
    n_bond_parts: int | None = None


@dataclass
class SurfaceResult:
    """ITIM surface outputs reused by both the surface plots and the height block."""
    Z_itim: NDArray[np.floating] | None = None
    Z_itim_with_bonds: NDArray[np.floating] | None = None
    Z_interpolated: NDArray[np.floating] | None = None
    poss_surface_part: NDArray[np.floating] | None = None
    poss_surface_part_bonds: NDArray[np.floating] | None = None
    vol_no_bonds: float = 0.
    vol_with_bonds: float = 0.


@dataclass
class HeightAccum:
    """Per-variant-suffix accumulators mutated by `heights_and_peaks_for_seed`."""
    rms: dict; h_avg: dict
    n_peaks: dict; peak_h_avg: dict; peak_h_std: dict
    peak_dist_avg: dict; peak_dist_std: dict


# ----------------------------------------------------------- figure-IO helpers --

def save_height_map_gwy(outdir, filename_stem, data, box_l):
    """Save a 2-D height array as a Gwyddion .gwy file."""
    os.makedirs(outdir, exist_ok=True)
    gwy_obj = GwyContainer()
    gwy_obj['/0/data/title'] = str(outdir) + filename_stem
    gwy_obj['/0/data'] = GwyDataField(
        data=np.asarray(data, dtype=np.float64),
        xreal=box_l[0], yreal=box_l[1],
        xoff=0, yoff=0,
        si_unit_xy="sim", si_unit_z="sim")
    gwy_obj.tofile(os.path.join(outdir, filename_stem + ".gwy"))


def save_imshow_height(outdir, filename_stem, data, box_l, vmax):
    """Save an imshow of a 2-D height map."""
    os.makedirs(outdir, exist_ok=True)
    plt.figure(num="tmp", clear=True, figsize=(6, 5))
    plt.imshow(data.T, extent=(0, box_l[0], 0, box_l[1]), origin='lower',
               cmap='viridis', vmin=0, vmax=vmax)
    plt.colorbar(label="Height")
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, filename_stem + savefig_format), dpi=dpi)


# --------------------------------------------------------------------- loaders --

def load_sample_frame(A, ts, DENS, K, HEIGHT, H, seed, h5_path, BASE_DIR,
                      label_cm_or_mm, height_at_h0) -> tuple[SampleFrame | None, float]:
    """Load + preprocess one (ts, sample, H, seed). The caller has already checked
    that `h5_path` exists. Returns (SampleFrame | None, height_at_h0).

    Returns None for the frame if the requested timestep is missing in the file.
    `height_at_h0` is threaded through the H-loop: set at H==0, reused for H>0.
    """
    # file names that include information of the elastomer sample (+ H)
    save_name_sample   = f"DENS{DENS:.2f}K{K}HEIGHT{HEIGHT}-s{seed}"
    save_name_sample_H = f"DENS{DENS:.2f}K{K}HEIGHT{HEIGHT}-H{H}-s{seed}"

    # Get the box_l, positions and dipole moments for the time step ts
    try:
        box_l, ids, poss, poss_folded, dips, bonds = pyanal.get_elastomer_h5(
            h5_path, A, time_step=ts, float_type=np.float32, int_type=np.int32)
    except Exception:
        if VERBOSE:
            print(f"timestep {ts} does not exist in", h5_path, flush=True)
        return None, height_at_h0

    # shift particles to have bottom layer at z=0
    poss -= SUBSTRATE_Z_OFFSET
    poss_folded -= SUBSTRATE_Z_OFFSET

    assert (poss[:, 2] >= 0.).all() and (poss[:, 2] < 3 * box_l[0]).all(), f"{poss[:,2].min()} {poss[:,2].max()}"

    assert poss.shape[0] == poss_folded.shape[0] == dips.shape[0] and poss.shape[1] == poss_folded.shape[1] == dips.shape[1] == 3, "Algo de errado não está certo."
    n_parts = poss.shape[0]  # number of magnetic particles in elastomer

    radius = np.full((n_parts,), SIZE / 2, dtype=np.float32)

    if do_USE_HEIGHT_AT_H0:
        if SIZE_BOND_PART > SIZE:
            raise NotImplementedError
        if H == 0:
            height_at_h0 = float(poss[:, 2].max() + SIZE / 2)

    frame = SampleFrame(
        A=A, ts=ts, DENS=DENS, K=K, HEIGHT=HEIGHT, H=H, seed=seed,
        BASE_DIR=BASE_DIR, save_name_sample=save_name_sample,
        save_name_sample_H=save_name_sample_H, label_cm_or_mm=label_cm_or_mm,
        box_l=box_l, ids=ids, poss=poss, poss_folded=poss_folded,
        dips=dips, bonds=bonds, radius=radius, n_parts=n_parts,
        HEIGHT_AT_H0=height_at_h0)

    if do_USE_BOND_TO_PARTS:
        # Reads the positions of the bond particles from a h5 file (saved before the main loop)
        bonds_to_paths = bond_part_path(A, DENS, K, HEIGHT, H, seed)

        if not os.path.isfile(bonds_to_paths):
            raise FileNotFoundError(
                f"No bond part file in {h5_path}.\nUse do_BOND_TO_PARTS to create bond parts file.")

        poss_bond_parts, bond_part_center_distance = pyanal.get_bonds_to_parts_h5(
            bonds_to_paths, time_step=ts, float_type=np.float32)

        # shift to match previous shift to particles (to have bottom layer at z=0)
        poss_bond_parts -= SUBSTRATE_Z_OFFSET

        assert (poss_bond_parts[:, 2] >= 0.).all() and (poss_bond_parts[:, 2] < np.max(poss[:, 2])).all(), f"{poss_bond_parts[:,2].min()} {poss_bond_parts[:,2].max()}"

        n_bond_parts = len(poss_bond_parts)
        assert n_bond_parts > 0
        poss_bond_parts_folded = pyanal.fold_coord(poss_bond_parts, box_dim=box_l)

        poss_all_folded = np.vstack((poss_folded, poss_bond_parts_folded))
        n_parts_all = poss_all_folded.shape[0]

        radius_bonds = np.full((n_bond_parts,), SIZE_BOND_PART / 2, dtype=np.float32)

        assert n_parts_all == (n_parts + n_bond_parts), "Total number of particles is not equal to the sum of all particles... Interesting."

        frame.poss_bond_parts_folded = poss_bond_parts_folded
        frame.poss_all_folded = poss_all_folded
        frame.radius_bonds = radius_bonds
        frame.n_bond_parts = n_bond_parts

    return frame, height_at_h0


# ---------------------------------------------------------- surface detection --

def detect_surfaces(frame, compute_vol_no_bonds, compute_vol_with_bonds) -> SurfaceResult:
    """ITIM surface detection (with/without bonds), surface plots, and volume.

    Returns a SurfaceResult. Only called when do_USE_SURFACE_SHAPE is on.
    """
    box_l = frame.box_l
    res = SurfaceResult()

    # ITIM grid
    grid_spacing = ITIM_DIAMETER / 4
    grid_surface_bottom = np.full((math.ceil(box_l[0] / grid_spacing), math.ceil(box_l[1] / grid_spacing)),
                          0. if not do_USE_HEIGHT_AT_H0 else frame.HEIGHT_AT_H0)

    # without bonds
    ball_radius = ITIM_DIAMETER / 2
    assert ball_radius > grid_spacing
    X_itim, Y_itim, Z_itim = pyanal.detect_surface_with_spheres(
        frame.poss_folded, frame.radius, R=ball_radius, grid_bottom=grid_surface_bottom, box_l=box_l)

    poss_surface_part = pyanal.detect_surface_particles_itim(
        frame.poss_folded, frame.radius, R=ball_radius, box_l=box_l, grid_bottom=grid_surface_bottom)

    res.Z_itim = Z_itim
    res.poss_surface_part = poss_surface_part

    # laser-emulation interpolated surface: linear interp of particle apexes (no-bonds only)
    if ITIM_DIAMETER_INTERP == ITIM_DIAMETER:
        nodes = poss_surface_part
    else:
        nodes = pyanal.detect_surface_particles_itim(
            frame.poss_folded, frame.radius, R=ITIM_DIAMETER_INTERP / 2,
            box_l=box_l, grid_bottom=grid_surface_bottom)
    nodes_apex = nodes.copy()
    nodes_apex[:, 2] += SIZE / 2          # particle apex (laser-visible top)
    _, _, Z_interp = pyanal.detect_surface_by_interpolation_of_surface_particles(
        nodes_apex, box_l, grid_bottom=np.full_like(grid_surface_bottom, 6.0), method='linear')
    res.Z_interpolated = Z_interp

    assert Z_interp.max() < (frame.poss_folded[:,2].max() + SIZE)

    if plot_SURFACE_SHAPE:
        substrate = np.zeros_like(Z_itim)  # bottom of box
        outputpath_tmp = os.path.join(frame.BASE_DIR, "surface_3d", f"without_bonds-{ITIM_DIAMETER}", f"t{frame.ts}")
        os.makedirs(outputpath_tmp, exist_ok=True)
        fig = plt.figure(num="tmp", clear=True)
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(frame.poss_folded[:, 0], frame.poss_folded[:, 1], frame.poss_folded[:, 2], color='green', s=20, alpha=0.2)
        ax.scatter(poss_surface_part[:, 0], poss_surface_part[:, 1], poss_surface_part[:, 2], label="surface particles",
                          color='red', s=20, alpha=0.6)
        ax.plot_surface(X_itim, Y_itim, Z_itim,
                               color='royalblue', alpha=0.5, label='Top')
        ax.plot_surface(X_itim, Y_itim, substrate,
                               color='lightgreen', alpha=0.5, label='Bottom')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height (Z)')
        ax.set_zlim3d(bottom=0., top=25.)
        ax.zaxis.set_ticks(np.arange(0.,25.,2.5))
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

        outputpath_tmp = os.path.join(frame.BASE_DIR, "surface_2d", "without_bonds", f"t{frame.ts}")
        os.makedirs(outputpath_tmp, exist_ok=True)
        fig, ax = plt.subplots(num="tmp", clear=True)
        for i_tmp in range(X_itim.shape[1] - 1):
            ax.plot(Y_itim[:, i_tmp], Z_itim[:, i_tmp], "-", color='royalblue', alpha=0.7)
        ax.plot(Y_itim[:, -1], Z_itim[:, -1], "-", color='royalblue', alpha=0.7, label='Top')
        ax.plot(Y_itim[0], substrate[0], "-", color='lightgreen', alpha=0.7, label='Bottom')
        pyanal.plt_scatter(ax, frame.poss_folded[:, 1], frame.poss_folded[:, 2], size=SIZE, color='red', alpha=0.2)
        ax.set_xlabel('X')
        ax.set_ylabel('Z')
        ax.set_aspect('equal')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

    if do_VOLUME and compute_vol_no_bonds:
        # calculate the volume from the z=0 plane to the estimated surface
        Z_itim_flat = Z_itim.ravel()
        substrate = np.zeros_like(Z_itim_flat, np.float32)
        res.vol_no_bonds = pyanal.volume_on_rectangular_grid(
            top_surface_flat=Z_itim_flat, bottom_surface_flat=substrate, cell_area=grid_spacing**2)

    if do_USE_BOND_TO_PARTS:
        # Use bonds to better estimate the surface
        ball_radius = ITIM_DIAMETER / 2
        assert ball_radius > grid_spacing
        X_itim_with_bonds, Y_itim_with_bonds, Z_itim_with_bonds = pyanal.detect_surface_with_spheres(
            frame.poss_bond_parts_folded, frame.radius_bonds, R=ball_radius, grid_bottom=Z_itim, box_l=box_l)

        poss_surface_part_bonds = pyanal.detect_surface_particles_itim(
            frame.poss_all_folded, np.concatenate((frame.radius, frame.radius_bonds)), R=ball_radius, box_l=box_l, grid_bottom=grid_surface_bottom)

        res.Z_itim_with_bonds = Z_itim_with_bonds
        res.poss_surface_part_bonds = poss_surface_part_bonds

        if plot_SURFACE_SHAPE:
            substrate = np.zeros_like(Z_itim_with_bonds)  # bottom of box
            outputpath_tmp = os.path.join(frame.BASE_DIR, "surface_3d", f"with_bonds-{ITIM_DIAMETER}", f"t{frame.ts}")
            os.makedirs(outputpath_tmp, exist_ok=True)
            fig = plt.figure(num="tmp", clear=True)
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(frame.poss_folded[:, 0], frame.poss_folded[:, 1], frame.poss_folded[:, 2],
                              color='green', s=20, alpha=0.2)
            # ax.scatter(poss_surface_part_bonds[:, 0], poss_surface_part_bonds[:, 1], poss_surface_part_bonds[:, 2], label="surface particles",
            #                   color='red', s=20, alpha=0.6)
            ax.plot_surface(X_itim_with_bonds, Y_itim_with_bonds, Z_itim_with_bonds,
                                   color='royalblue', alpha=0.5, label='Top')
            ax.plot_surface(X_itim_with_bonds, Y_itim_with_bonds, substrate,
                                   color='lightgreen', alpha=0.5, label='Bottom')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Height (Z)')
            ax.set_zlim3d(bottom=0., top=25.)
            ax.zaxis.set_ticks(np.arange(0.,25.,2.5))
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

            outputpath_tmp = os.path.join(frame.BASE_DIR, "surface_2d", "with_bonds", f"t{frame.ts}")
            os.makedirs(outputpath_tmp, exist_ok=True)
            fig, ax = plt.subplots(num="tmp", clear=True)
            for i_tmp in range(X_itim_with_bonds.shape[1] - 1):
                ax.plot(Y_itim_with_bonds[:, i_tmp], Z_itim_with_bonds[:, i_tmp], "-", color='royalblue', alpha=0.7)
            ax.plot(Y_itim_with_bonds[:, -1], Z_itim_with_bonds[:, -1], "-", color='royalblue', alpha=0.7, label='Top')
            ax.plot(Y_itim_with_bonds[0], substrate[0], "-", color='lightgreen', alpha=0.7, label='Bottom')
            pyanal.plt_scatter(ax, frame.poss_folded[:, 1], frame.poss_folded[:, 2], size=2**(1/6), color='red', alpha=0.2)
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_aspect('equal')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

        if do_VOLUME and compute_vol_with_bonds:
            # calculate the volume from the z=0 plane to the estimated surface
            Z_itim_flat = Z_itim_with_bonds.ravel()
            substrate = np.zeros_like(Z_itim_flat, dtype=np.float32)
            res.vol_with_bonds = pyanal.volume_on_rectangular_grid(
                top_surface_flat=Z_itim_flat, bottom_surface_flat=substrate, cell_area=grid_spacing**2)

    return res


# ------------------------------------------------------------------- energies --

def calc_energies_for_seed(frame):
    """Return (E_bond, E_dip, E_Zee, E_Zee_dipm, E_chains) for one seed."""
    E_dip_seed, E_Zee_seed, E_bond_seed, E_Zee_dipm_seed = pyanal.calc_energies(
        frame.box_l, frame.ids, frame.poss, frame.dips, frame.bonds, H_ext=np.array([0, 0, frame.H]), periodicity=[True, True, False])
    E_chain_seed = pyanal.energy.dipolar_energy_chains(frame.dips, SIZE)

    return E_bond_seed, E_dip_seed, E_Zee_seed, E_Zee_dipm_seed, E_chain_seed


# -------------------------------------------------------------- magnetization --

def calc_magnetization_for_seed(frame):
    """Return (M_z vector, M_z_dipm vector) for one seed."""
    return pyanal.moment_avg(frame.dips), pyanal.moment_dipm_avg(frame.dips)


# ---------------------------------------------------------- heights and peaks --

def heights_and_peaks_for_seed(frame, surfaces, accum, cm, data_ARay_ts_sample_H):
    """The full do_HEIGHTS block for one seed; mutates `accum` (HeightAccum)."""
    box_l = frame.box_l
    HEIGHT = frame.HEIGHT
    ts = frame.ts

    RADIUS = np.float32(SIZE / 2)
    grid_res_x = 0.1
    grid_res_y = 0.1

    grid_bottom = np.zeros((math.ceil(box_l[0] / grid_res_x), math.ceil(box_l[1] / grid_res_y)))
    if save_HEIGHT_MAPS:
        grid_surface_bottom = grid_bottom
        if do_USE_HEIGHT_AT_H0:
            grid_surface_bottom = np.full_like(grid_surface_bottom, frame.HEIGHT_AT_H0)

    height_map = pyanal.grid_max_with_radius(frame.poss_folded, np.full(frame.n_parts, RADIUS), box_l, grid_bottom=grid_bottom)

    if save_HEIGHT_MAPS:
        save_height_map_gwy(
            os.path.join(frame.BASE_DIR, "height_map", "without_bonds", f"t{ts}"),
            frame.save_name_sample_H, height_map, box_l)

        grid_max_heights = pyanal.grid_max(frame.poss_folded, box_l, grid_bottom=grid_bottom)
        save_height_map_gwy(
            os.path.join(frame.BASE_DIR, "height_map_2", "without_bonds", f"t{ts}"),
            frame.save_name_sample_H, grid_max_heights, box_l)

        X_tmp, Y_tmp, Z_tmp = pyanal.detect_surface_by_interpolation_of_surface_particles(
            pyanal.detect_surface_particles_itim(
                frame.poss_folded, frame.radius, ITIM_DIAMETER/2, box_l, grid_bottom=grid_bottom
            ),
            box_l, grid_bottom=grid_surface_bottom
        )
        save_height_map_gwy(
            os.path.join(frame.BASE_DIR, "height_map_3", f"without_bonds-{ITIM_DIAMETER}", f"t{ts}"),
            frame.save_name_sample_H, Z_tmp, box_l)

        substrate = np.zeros_like(Z_tmp)  # bottom of box
        outputpath_tmp = os.path.join(frame.BASE_DIR, "surface_3d-3", f"without_bonds-{ITIM_DIAMETER}", f"t{ts}")
        os.makedirs(outputpath_tmp, exist_ok=True)
        fig = plt.figure(num="tmp", clear=True)
        ax = fig.add_subplot(projection='3d')
        ax.scatter(frame.poss_folded[:, 0], frame.poss_folded[:, 1], frame.poss_folded[:, 2],
                          color='green', s=20, alpha=0.2)
        ax.scatter(surfaces.poss_surface_part[:, 0], surfaces.poss_surface_part[:, 1], surfaces.poss_surface_part[:, 2], label="surface particles",
                          color='red', s=20, alpha=0.6)
        ax.plot_surface(X_tmp, Y_tmp, Z_tmp,
                        color='royalblue', alpha=0.5, label='Top')
        ax.plot_surface(X_tmp, X_tmp, substrate,
                               color='lightgreen', alpha=0.5, label='Bottom')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height (Z)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

    if plot_HIST:
        save_imshow_height(
            os.path.join(frame.BASE_DIR, "hist", "without_bonds", f"t{ts}"),
            frame.save_name_sample_H, height_map, box_l, HEIGHT*1.75)

    heights = height_map.ravel()
    mean_h = np.mean(heights) if len(heights) > 0 else 0.

    accum.h_avg[""] += mean_h
    accum.rms[""] += np.sqrt(np.mean((heights - mean_h)**2))

    if do_PEAK_STATISTICS and (not data_ARay_ts_sample_H["peak_h_avg"].is_set() or do_OVERWRITE_DATA):
        min_peak_z = HEIGHT + 1
        peak_heights, peaks_pos, _ = pyanal.find_peaks(height_map, size_float=RADIUS, grid_res_x=grid_res_x, grid_res_y=grid_res_y, min_peak_z=min_peak_z, box_l=box_l)

        peak_distances = pyanal.peak_distance_dist(peaks_pos, box_l=box_l)

        n_peaks_tmp = peak_heights.shape[0]

        if n_peaks_tmp > 0:
            peak_heights -= HEIGHT

        if n_peaks_tmp > 1:
            accum.n_peaks[""] += n_peaks_tmp

            accum.peak_h_avg[""] += np.mean(peak_heights)
            accum.peak_h_std[""] += np.std(peak_heights)

            accum.peak_dist_avg[""] += np.mean(peak_distances)
            accum.peak_dist_std[""] += np.std(peak_distances)

        if plot_PEAK_DISTRIBUTIONS:
            outputpath_tmp = os.path.join(frame.BASE_DIR, "peak_height", "without_bonds", f"t{ts}")
            os.makedirs(outputpath_tmp, exist_ok=True)
            fig, ax = plt.subplots(num="tmp", clear=True, figsize=(6, 5))
            ax.hist(peak_heights, bins=np.arange(0, 2*HEIGHT, 0.5), color=cm[frame.label_cm_or_mm], label=f"{len(peak_heights)}") # type: ignore
            ax.set_xlabel(r"$\Delta height_{peak}$")
            ax.set_ylabel(r"$p$")
            ax.legend(fontsize=fontsize_legend)
            fig.tight_layout()
            fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

            outputpath_tmp = os.path.join(frame.BASE_DIR, "peak_distance", "without_bonds", f"t{ts}")
            os.makedirs(outputpath_tmp, exist_ok=True)
            fig, ax = plt.subplots(num="tmp", clear=True, figsize=(6, 5))
            ax.hist(peak_distances, bins=np.arange(2, np.ceil(box_l[0]), 1), color=cm[frame.label_cm_or_mm], label=f"{len(peak_distances)}")
            ax.set_xlabel(r"$\Delta r_{peak}$")
            ax.set_ylabel(r"$p$")
            ax.legend(fontsize=fontsize_legend)
            fig.tight_layout()
            fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

    if do_USE_BOND_TO_PARTS:
        height_map_with_bonds = pyanal.grid_max_with_radius(frame.poss_bond_parts_folded, frame.radius_bonds, box_l=box_l, grid_bottom=height_map)

        if save_HEIGHT_MAPS:
            save_height_map_gwy(
                os.path.join(frame.BASE_DIR, "height_map", "with_bonds", f"t{ts}"),
                frame.save_name_sample_H, height_map_with_bonds, box_l)

            data_tmp = pyanal.grid_max(frame.poss_bond_parts_folded, box_l, grid_bottom=grid_max_heights)
            save_height_map_gwy(
                os.path.join(frame.BASE_DIR, "height_map_2", "with_bonds", f"t{ts}"),
                frame.save_name_sample_H, data_tmp, box_l)

            X_tmp, Y_tmp, Z_tmp = pyanal.detect_surface_by_interpolation_of_surface_particles(
                pyanal.detect_surface_particles_itim(
                    frame.poss_all_folded, np.concatenate((frame.radius, frame.radius_bonds)), ITIM_DIAMETER/2, box_l, grid_bottom=grid_bottom
                ),
                box_l, grid_bottom=grid_surface_bottom
            )
            save_height_map_gwy(
                os.path.join(frame.BASE_DIR, "height_map_3", f"with_bonds-{ITIM_DIAMETER}", f"t{ts}"),
                frame.save_name_sample_H, Z_tmp, box_l)

            substrate = np.zeros_like(Z_tmp)  # bottom of box
            outputpath_tmp = os.path.join(frame.BASE_DIR, "surface_3d-3", f"with_bonds-{ITIM_DIAMETER}", f"t{ts}")
            os.makedirs(outputpath_tmp, exist_ok=True)
            fig = plt.figure(num="tmp", clear=True)
            ax = fig.add_subplot(projection='3d')
            ax.scatter(frame.poss_folded[:, 0], frame.poss_folded[:, 1], frame.poss_folded[:, 2],
                              color='green', s=20, alpha=0.2)
            ax.scatter(surfaces.poss_surface_part_bonds[:, 0], surfaces.poss_surface_part_bonds[:, 1], surfaces.poss_surface_part_bonds[:, 2], label="surface particles",
                              color='red', s=20, alpha=0.6)
            ax.plot_surface(X_tmp, Y_tmp, Z_tmp,
                            color='royalblue', alpha=0.5, label='Top')
            ax.plot_surface(X_tmp, X_tmp, substrate,
                                   color='lightgreen', alpha=0.5, label='Bottom')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Height (Z)')
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

        if plot_HIST:
            save_imshow_height(
                os.path.join(frame.BASE_DIR, "hist", "with_bonds", f"t{ts}"),
                frame.save_name_sample_H, height_map_with_bonds, box_l, HEIGHT*1.75)

        heights = height_map_with_bonds.ravel()
        mean_h = np.mean(heights) if len(heights) > 0 else 0.

        accum.h_avg["_with_bonds"] += mean_h
        accum.rms["_with_bonds"] += np.sqrt(np.mean((heights - mean_h)**2))

        if do_PEAK_STATISTICS and (not data_ARay_ts_sample_H["peak_h_avg_with_bonds"].is_set() or do_OVERWRITE_DATA):
            peak_heights, peaks_pos, _ = pyanal.find_peaks(height_map_with_bonds, size_float=max(RADIUS, frame.radius_bonds.mean()), grid_res_x=grid_res_x, grid_res_y=grid_res_y, min_peak_z=min_peak_z, box_l=box_l)

            peak_distances = pyanal.peak_distance_dist(peaks_pos, box_l=box_l)

            n_peaks_tmp = peak_heights.shape[0]

            if n_peaks_tmp > 0:
                peak_heights -= HEIGHT

            if n_peaks_tmp > 1:
                accum.n_peaks["_with_bonds"] += n_peaks_tmp

                accum.peak_h_avg["_with_bonds"] += np.mean(peak_heights)
                accum.peak_h_std["_with_bonds"] += np.std(peak_heights)

                accum.peak_dist_avg["_with_bonds"] += np.mean(peak_distances)
                accum.peak_dist_std["_with_bonds"] += np.std(peak_distances)

            if plot_PEAK_DISTRIBUTIONS:
                outputpath_tmp = os.path.join(frame.BASE_DIR, "peak_height", "with_bonds", f"t{ts}")
                os.makedirs(outputpath_tmp, exist_ok=True)
                fig, ax = plt.subplots(num="tmp", clear=True, figsize=(6, 5))
                ax.hist(peak_heights, bins=np.arange(0, 2*HEIGHT, 0.5), color=cm[frame.label_cm_or_mm], label=f"{len(peak_heights)}") # type: ignore
                ax.set_xlabel(r"$\Delta height_{peak}$")
                ax.set_ylabel(r"$p$")
                ax.legend(fontsize=fontsize_legend)
                fig.tight_layout()
                fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

                outputpath_tmp = os.path.join(frame.BASE_DIR, "peak_distance", "with_bonds", f"t{ts}")
                os.makedirs(outputpath_tmp, exist_ok=True)
                fig, ax = plt.subplots(num="tmp", clear=True, figsize=(6, 5))
                ax.hist(peak_distances, bins=np.arange(2, np.ceil(box_l[0]), 1), color=cm[frame.label_cm_or_mm], label=f"{len(peak_distances)}")
                ax.set_xlabel(r"$\Delta r_{peak}$")
                ax.set_ylabel(r"$p$")
                ax.legend(fontsize=fontsize_legend)
                fig.tight_layout()
                fig.savefig(os.path.join(outputpath_tmp, frame.save_name_sample_H + savefig_format), dpi=dpi)

    if do_USE_SURFACE_SHAPE:
        # Use ITIM surface map for heights
        # without bond particles
        surface_map = surfaces.Z_itim

        if plot_HIST:
            save_imshow_height(
                os.path.join(frame.BASE_DIR, "hist_surface", "without_bonds", f"t{ts}"),
                frame.save_name_sample_H, surface_map, box_l, HEIGHT*1.75)

        heights = surface_map.ravel()
        mean_h = np.mean(heights) if len(heights) > 0 else 0.

        accum.h_avg["_with_surface"] += mean_h
        accum.rms["_with_surface"] += np.sqrt(np.mean((heights - mean_h)**2))

        # interpolated surface (no-bonds only)
        surface_map_interp = surfaces.Z_interpolated

        if plot_HIST:
            save_imshow_height(
                os.path.join(frame.BASE_DIR, "hist_surface", "interpolated", f"t{ts}"),
                frame.save_name_sample_H, surface_map_interp, box_l, HEIGHT*1.75)

        heights = surface_map_interp.ravel()
        mean_h = np.mean(heights) if len(heights) > 0 else 0.

        accum.h_avg["_interpolated"] += mean_h
        accum.rms["_interpolated"] += np.sqrt(np.mean((heights - mean_h)**2))

        if do_USE_BOND_TO_PARTS:
            # Use ITIM surface map for heights
            # with bond particles
            surface_map_with_bonds = surfaces.Z_itim_with_bonds

            if save_HEIGHT_MAPS:
                save_height_map_gwy(
                    os.path.join(frame.BASE_DIR, "height_map", "with_bonds_surface", f"t{ts}"),
                    frame.save_name_sample_H, surface_map_with_bonds, box_l)

            if plot_HIST:
                save_imshow_height(
                    os.path.join(frame.BASE_DIR, "hist_surface", "with_bonds", f"t{ts}"),
                    frame.save_name_sample_H, surface_map_with_bonds, box_l, HEIGHT*1.75)

            heights = surface_map_with_bonds.ravel()
            mean_h = np.mean(heights) if len(heights) > 0 else 0.

            accum.h_avg["_with_bonds_with_surface"] += mean_h
            accum.rms["_with_bonds_with_surface"] += np.sqrt(np.mean((heights - mean_h)**2))


# -------------------------------------------------------------- density along z --

def density_z_for_seed(frame) -> tuple[NDArray, NDArray | None, NDArray]:
    """Return (hist_no_bonds, hist_with_bonds | None, bin_edges) for one seed."""
    bins_ = int(frame.HEIGHT * 2 + 3); range_ = (0, bins_); density_ = True
    hist, dens_z_bin_edges = np.histogram(frame.poss_folded[:, 2], bins=bins_, range=range_, density=density_)
    hist_with_bonds = None
    if do_USE_BOND_TO_PARTS:
        hist_with_bonds, _ = np.histogram(frame.poss_all_folded[:, 2], bins=bins_, range=range_, density=density_)
    return hist, hist_with_bonds, dens_z_bin_edges
