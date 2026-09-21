"""mae_analysis -- analysis of magneto-active elastomer (MAE) simulations.

Layout
------
config       shared run-time knobs (paths, sweeps, units, style)   -- edit per run
style        Series abstraction + K=hue / density=shade+marker / A=linestyle scheme
surface/     surface analysis family (energies, magnetization, heights, density, volume)
cluster/     cluster analysis family (placeholder; filled in step 2)
__main__     `python -m mae_analysis` driver + CLI flags + subprocess worker dispatch

Entry points
------------
    python -m mae_analysis --surface                     # surface: calc + plot
    python -m mae_analysis --surface --plot-only         # reload saved ARay, replot
    python -m mae_analysis analysis_per_ts_sample <pkl>  # (internal) subprocess worker
"""
