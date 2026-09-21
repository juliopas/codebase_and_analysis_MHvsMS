"""Repository-relative path resolution.

Every path in this repository is resolved from the location of this file, so a
clone works wherever it is unpacked and whatever the working directory is. No
path is tied to a particular machine or to ``$HOME``.

``DATA_DIR`` defaults to ``<repo>/DATA``. The dataset is distributed separately
(see the README), so point ``MAE_DATA_DIR`` at wherever you unpacked it if you
keep it outside the repository:

    export MAE_DATA_DIR=/scratch/mae-data
"""

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS_DIR = os.path.join(REPO_ROOT, "SCRIPTS")
DATA_DIR = os.environ.get("MAE_DATA_DIR") or os.path.join(REPO_ROOT, "DATA")

__all__ = ["REPO_ROOT", "SCRIPTS_DIR", "DATA_DIR"]
