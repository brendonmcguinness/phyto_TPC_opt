"""
Resolve the repository's `figures/` directory independently of the
current working directory.

All driver scripts live in `src/` and write their PNGs to the sibling
`figures/` folder. Importing `fig_path` and wrapping save paths keeps
output in one place whether a script is run from the repo root
(`python src/run_1sp_1res.py`) or from inside `src/`.
"""

from __future__ import annotations

import os

FIG_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), os.pardir, "figures"
)
os.makedirs(FIG_DIR, exist_ok=True)


def fig_path(name):
    """Return an absolute path inside the figures/ directory for `name`."""
    return os.path.join(FIG_DIR, name)
