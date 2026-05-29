"""Ensure the repo root is importable when collecting fusion tests.

`tests/fusion/` is a package but `tests/` is not, so pytest's prepend import
mode does not place the repo root on sys.path. Add it here so `slam_core`,
`visual_slam`, etc. resolve under `.venv/bin/pytest`.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
