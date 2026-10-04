"""Shared JSON output helper for the ablation scripts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_json(path: str | Path, payload: Any) -> Path:
    """Write ``payload`` as indented JSON, creating the parent directory.

    ``reports/`` and ``logs/`` are gitignored, so they are routinely absent on a
    fresh clone; creating them here keeps the scripts runnable from scratch.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    return path
