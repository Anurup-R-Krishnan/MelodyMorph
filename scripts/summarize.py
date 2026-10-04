"""Print the ablation reports as one table.

    python -m scripts.summarize reports/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COLS = ["pitch_ppl", "pitch_nll", "dur_nll", "note_nll", "pos_nll",
        "all_bits_per_note", "token_ppl", "n_notes"]


def _cell(value: object) -> str:
    if isinstance(value, bool) or value is None:
        return f"{str(value):>18s}"
    if isinstance(value, float):
        return f"{value:>18.4f}"
    if isinstance(value, int):
        return f"{value:>18d}"
    return f"{str(value):>18s}"


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    root = Path(argv[0] if argv else "reports")
    if not root.is_dir():
        raise SystemExit(f"error: {root} is not a directory")

    rows = []
    for path in sorted(root.glob("*.json")):
        rep = json.loads(path.read_text())
        if "transformer" in rep and not path.stem.startswith("kn_"):
            rows.append((path.stem, rep["transformer"]))
        for order, r in rep.get("kneser_ney", {}).items():
            rows.append((f"KN-{order} ({path.stem.removeprefix('kn_')})", r))

    if not rows:
        raise SystemExit(f"error: no report JSON found in {root}")

    print(f"{'run':28s}" + "".join(f"{c:>18s}" for c in COLS))
    for name, r in rows:
        print(f"{name:28s}" + "".join(_cell(r.get(c)) for c in COLS))


if __name__ == "__main__":
    main()
