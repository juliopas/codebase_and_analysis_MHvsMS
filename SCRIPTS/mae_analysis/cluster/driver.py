"""Cluster family orchestration: init ARay -> compute -> plot (mirrors run_surface)."""

import os
import time

from pyanal import ARay

from ..config import A_list, do_CALCULATE
from .config import DATA_ARAY_DIR, plot_H as do_plot_H
from .data import init_ARay
from .analysis import analysis_main
from .plotting import plot_H


def run(file_id=None):
    """Compute the cluster ARay per model A, then plot."""
    start_time_main = time.time()
    for A in A_list:
        data_ARay, DATA_ARAY_FILENAME = init_ARay(A, use_file_id=file_id, user_input=True)
        data_Aray_file_path = data_ARay.write_file(os.path.join(DATA_ARAY_DIR, DATA_ARAY_FILENAME))
        print("\n")

        if do_CALCULATE:
            print("Analysing cluster data...")
            start_time = time.time()
            analysis_main(data_ARay, A)
            print("Finished analysing cluster data.", time.time() - start_time, "\n")

        if do_plot_H:
            print("Plotting cluster data...")
            start_time = time.time()
            plot_H(ARay.from_file(data_Aray_file_path), A)
            print("Finished plotting cluster data.", time.time() - start_time, "\n")

        del data_ARay

    print(f"Cluster done in {(time.time() - start_time_main) // 60:.0f} minutes.")
