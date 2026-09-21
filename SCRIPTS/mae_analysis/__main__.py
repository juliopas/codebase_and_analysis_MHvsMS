"""Driver + CLI dispatch for mae_analysis.

    python -m mae_analysis --surface [--reverse] [--plot-only] [--no-plot] [--file-id N]
    python -m mae_analysis --cluster [--plot-only] [--no-plot] [--file-id N]
    python -m mae_analysis --surface --cluster      # both families, separate outputs
    python -m mae_analysis --hysteresis             # overlay forward + reverse surface ARays
    python -m mae_analysis analysis_per_ts_sample <pkl>          # (internal) surface worker
    python -m mae_analysis cluster_analysis_per_ts_sample_H <pkl>  # (internal) cluster worker

CLI flags override config defaults via environment variables. Config-dependent
imports are deferred into the run_* functions so overrides apply *before* config
is imported -- and because subprocess workers inherit the environment, their
re-imported config stays in sync with the parent.
"""

import os
import sys
import time


def run_surface(file_id=None):
    """Surface family: compute one ARay per model A, then overlay-plot."""
    from pyanal import ARay

    from .config import DATA_ARAY_DIR, A_list, do_CALCULATE, plot_PLOT
    from .surface.config import (
        do_BOND_TO_PARTS,
        plot_AFTER, plot_SAMPLE_H, plot_NONDIM_H, plot_TIME_H, plot_H_TIME, plot_DENSITY_Z,
    )
    from .surface.data import init_ARay
    from .surface.analysis import bond_to_parts, analysis_main
    from .style import Series, a_linestyle_map, a_markerfill_map
    from .surface.plotting import (
        plot_sample_h, plot_nondim_h, plot_time_h, plot_h_time, plot_dens_z_profiles, plot_legends,
    )

    start_time_main = time.time()

    # ---- Calculation phase: one ARay per model A (independent, as before) ----
    aray_paths = {}
    for A in A_list:
        data_ARay, DATA_ARAY_FILENAME = init_ARay(A, use_file_id=file_id, user_input=True)
        data_Aray_file_path = data_ARay.write_file(os.path.join(DATA_ARAY_DIR, DATA_ARAY_FILENAME))
        print("\n")

        # Convert bonds into particles (slow)
        if do_BOND_TO_PARTS and do_CALCULATE:
            print("Converting bonds into particles...")
            start_time = time.time()
            bond_to_parts(A)
            print("Created bond particles.", time.time() - start_time, "\n")

        # Main analysis loop (slow)
        if do_CALCULATE:
            print("Analysing data...")
            start_time = time.time()
            analysis_main(data_ARay, A)
            print("Finished analysing data.", time.time() - start_time, "\n")

        aray_paths[A] = data_Aray_file_path
        del data_ARay

    if not plot_AFTER:
        return

    # ---- Plotting phase: overlay all models on shared figures (fast) ----
    linestyles = a_linestyle_map(A_list, )
    fills      = a_markerfill_map(A_list)
    series_list = [Series(ARay.from_file(aray_paths[A]), A, linestyles[A], fills[A]) for A in A_list]

    if plot_SAMPLE_H:
        plot_sample_h(series_list)
        print("\n")
    if plot_NONDIM_H:
        plot_nondim_h(series_list)
        print("\n")
    if plot_TIME_H:
        plot_time_h(series_list)
        print("\n")
    if plot_H_TIME:
        plot_h_time(series_list)
    if plot_DENSITY_Z:
        plot_dens_z_profiles(series_list)
    if plot_PLOT:
        plot_legends(series_list)

    print(f"Surface done in {(time.time() - start_time_main) // 60:.0f} minutes.")


def run_cluster(file_id=None):
    """Cluster family (filled in Phase 2C)."""
    from .cluster.driver import run as cluster_run
    cluster_run(file_id=file_id)


def run_hysteresis():
    """Overlay forward + reverse surface ARays into hysteresis loops (Phase 2B)."""
    from .surface.hysteresis import run as hysteresis_run
    hysteresis_run()


def _run_worker(key, argv):
    """Internal subprocess entry for one compute shard."""
    if key == "analysis_per_ts_sample":
        from .surface.analysis import analysis_per_ts_sample
        return analysis_per_ts_sample(*argv)
    if key == "cluster_analysis_per_ts_sample_H":
        from .cluster.analysis import analysis_per_ts_sample_H
        return analysis_per_ts_sample_H(*argv)
    raise RuntimeError(f"unknown worker key: {key}")


_WORKER_KEYS = ("analysis_per_ts_sample", "cluster_analysis_per_ts_sample_H")


def _apply_cli_overrides():
    """Parse CLI flags and export them as env vars consumed by config.py.

    Returns the parsed args. Must be called BEFORE the run_* deferred imports.
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="mae_analysis",
        description="Analyse MAE simulation data. Sweeps/paths live in config.py; "
                    "these flags pick what runs and override the calc/plot defaults.",
    )
    # Analysis families (combine freely; default is --surface when none given).
    parser.add_argument("--surface", action="store_true",
                        help="run the surface analysis family")
    parser.add_argument("--cluster", action="store_true",
                        help="run the cluster analysis family")
    parser.add_argument("--hysteresis", action="store_true",
                        help="overlay forward+reverse surface ARays (hysteresis loops)")
    # Run-mode toggles (override config.py defaults via env vars).
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite data file with data calculated this run")
    parser.add_argument("--reverse", action="store_true",
                        help="reverse-field run: read sim_data-h5 / -fromH files, write to figs-reverse")
    parser.add_argument("--plot-only", action="store_true",
                        help="skip calculation; reload the saved ARay and (re)plot -- fast")
    parser.add_argument("--no-plot", action="store_true",
                        help="calculate only; produce no figures")
    parser.add_argument("--bond-to-parts", action="store_true",
                        help="calculate bond-to-parts file")
    parser.add_argument("--file-id", type=int, default=None,
                        help="pick saved ARay #N non-interactively (handy with --plot-only)")
    # Surface plot specific flags
    parser.add_argument("--plot-samples", action="store_true",
                        help="if plotting, plot only sample plots (1)")
    
    args = parser.parse_args()

    if args.overwrite:
        os.environ["MAE_OVERWRITE"] = "1"
    if args.plot_only:
        os.environ["MAE_DO_CALCULATE"] = "0"
    if args.no_plot:
        os.environ["MAE_PLOT"] = "0"
    if args.bond_to_parts:
        os.environ["MAE_DO_BONDTOPARTS"] = "1"
    if args.reverse:
        os.environ["MAE_REVERSE"] = "1"
    if args.plot_samples:
        os.environ["MAE_PLOT_ONLY_SAMPLES"] = "1"

    return args


if __name__ == "__main__":
    # Internal subprocess worker: dispatch directly, bypassing argparse. The env
    # set by the parent is inherited, so the re-imported config matches the run.
    if len(sys.argv) > 1 and sys.argv[1] in _WORKER_KEYS:
        sys.exit(_run_worker(sys.argv[1], sys.argv[2:]))

    _args = _apply_cli_overrides()

    _ran_any = False

    if _args.surface:
        run_surface(file_id=_args.file_id); _ran_any = True
    if _args.cluster:
        run_cluster(file_id=_args.file_id); _ran_any = True
    if _args.hysteresis:
        run_hysteresis(); _ran_any = True

    if not _ran_any:
        # run forwards (not really forward) / static eq
        run_surface(file_id=_args.file_id)
        run_cluster(file_id=_args.file_id)
        # run backwards
        os.environ["MAE_REVERSE"] = "1"
        run_surface(file_id=_args.file_id)
        run_cluster(file_id=_args.file_id)
        # plot hysteresis
        run_hysteresis()
