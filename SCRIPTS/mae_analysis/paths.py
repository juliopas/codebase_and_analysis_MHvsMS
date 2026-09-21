"""Reverse-aware resolution of simulation-data and bond-particle h5 paths.

Forward and reverse runs name their files differently, so centralise the
conventions here and let the analysis just ask for a path. ``REVERSE`` (set by
``--reverse`` via the environment) selects the convention:

  forward:  <READ_BASE>/{A_help}-{DENS}_{K}-{N_FULL_BOX}_{HEIGHT}/cp{A}-{seed}-H{H}.h5
  reverse:  <READ_BASE>/{A_help}-{A_CONC}-{DENS}_{K}-{N_FULL_BOX}_{HEIGHT}-reverse/
                cp{A}-{seed}-H{H}-fromH{H0}.h5        (H0 paired 1:1 with H)

``READ_BASE_DIR`` and ``BOND_PART_DIR`` already point at the reverse set
(sim_data-h5, figs-reverse) when ``REVERSE`` -- see config.py.
"""

import os

from pyanal.global_definitions import A_help_dict

from .config import (
    READ_BASE_DIR, BOND_PART_DIR, N_FULL_BOX,
    REVERSE, H_list, H0_list,
)


def _h0_for(H):
    """The starting field H0 paired with final field H in a reverse run."""
    return dict(zip(list(H_list), list(H0_list)))[H]


def _sample_dir(A, DENS, K, HEIGHT):
    """Per-sample sim-data directory (forward vs reverse naming)."""
    if REVERSE:
        return os.path.join(
            READ_BASE_DIR,
            f"{A_help_dict[A]}-{DENS:.2f}_{K}-{N_FULL_BOX}_{HEIGHT}-reverse")
    return os.path.join(
        READ_BASE_DIR,
        f"{A_help_dict[A]}-{DENS:.2f}_{K}-{N_FULL_BOX}_{HEIGHT}")


def sample_h5_path(A, DENS, K, HEIGHT, H, seed):
    """Path to the simulation checkpoint h5 for one (A, DENS, K, HEIGHT, H, seed)."""
    rev = f"-fromH{_h0_for(H):.2f}" if REVERSE else ""
    filename = f"cp{A}-{seed}-H{H:.2f}{rev}.h5"
    return os.path.join(_sample_dir(A, DENS, K, HEIGHT), filename)


def bond_part_path(A, DENS, K, HEIGHT, H, seed):
    """Path to the bond-particle h5 for one (A, DENS, K, HEIGHT, H, seed)."""
    rev = f"-fromH{_h0_for(H):.2f}" if REVERSE else ""
    filename = f"bp{A}-{DENS:.2f}_{K}-{N_FULL_BOX}_{HEIGHT}-{seed}-H{H:.2f}{rev}.h5"
    return os.path.join(BOND_PART_DIR, filename)
