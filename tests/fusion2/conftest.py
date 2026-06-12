"""Ensure the repo root is importable when collecting fusion2 tests.

Same bootstrap as tests/fusion/conftest.py: `tests/fusion2/` is a package but
`tests/` is not, so add the repo root to sys.path for `slam_core` etc.
"""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
