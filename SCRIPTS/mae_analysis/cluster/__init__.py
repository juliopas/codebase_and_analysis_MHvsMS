"""Cluster analysis family: energetic/distance clustering, Steinhardt BOP, correlations.

Mirrors the surface family on the same ARay (ts,DENS,K,HEIGHT,H) and reuses the
pyanal physics (cluster_E_anal, cluster_anal, BOP_analysis, separate_bulk_surface).

Modules
-------
config    cluster do_*/plot_* flags, l_list, cluster ARay dir   -- edit per run
data      cluster ARay creation / selection
analysis  subprocess fan-out + the per-(ts,sample) compute worker
plotting  plot_H (per-H histograms, BOP q_l, q4-q6 scatter, scalars, legends)
driver    init -> compute -> plot orchestration (called by `run_cluster`)

Run via:  python -m mae_analysis --cluster
"""
