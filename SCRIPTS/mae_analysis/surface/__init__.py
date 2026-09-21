"""Surface analysis family: energies, magnetization, heights/peaks, density, volume.

Modules
-------
data         ARay creation / selection and sample identity helper
metrics      declarative registry of scalar plots
observables  per-seed observable calculations (energies, heights, ...)
analysis     bond->parts preprocessing, worker fan-out, the worker orchestrator
plotting     the scalar-metric engine + plot functions (consume a list of Series)

Shared infrastructure (config, style) lives one level up in the `mae_analysis`
package and is imported as `..config` / `..style`.
"""
