from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _discover_project_root() -> Path:
    workspace_root = Path(__file__).resolve().parents[1]
    candidates = [workspace_root, workspace_root / "crisis-dispatch-env"]

    for candidate in candidates:
        if (candidate / "app").is_dir() and (candidate / "inference.py").is_file():
            return candidate

    raise RuntimeError(
        "Could not discover project root with app/ and inference.py. "
        f"Checked: {[str(path) for path in candidates]}"
    )


PROJECT_ROOT = _discover_project_root()

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT
