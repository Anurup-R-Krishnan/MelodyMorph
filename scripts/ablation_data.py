"""Build the corpora for the metre-alignment ablation.

    data/melodies.jsonl               metre-aligned, duple metres only (the default)
    data/melodies_legacy.jsonl        old extraction: every metre, pickup shifted to step 0
    data/melodies_legacy_duple.jsonl  old extraction restricted to the aligned corpus' tunes

``legacy_duple`` vs the default isolates the alignment fix (same tunes, same
split); ``legacy`` vs the default is the net effect including the dropped metres.

    python -m scripts.ablation_data

This rebuilds ``data/melodies.jsonl`` from scratch (``force=True``), so it runs
only under the ``__main__`` guard -- importing this module must stay free of
side effects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from melodymorph.corpus import build_corpus

ALIGNED = "data/melodies.jsonl"
LEGACY = "data/melodies_legacy.jsonl"
LEGACY_DUPLE = "data/melodies_legacy_duple.jsonl"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    aligned = build_corpus(ALIGNED, force=True)
    legacy = build_corpus(LEGACY, force=True, align_meter=False)
    ids = {r.id for r in aligned}

    out = Path(LEGACY_DUPLE)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in legacy:
            if r.id in ids:
                notes = [[n.onset, n.pitch, n.dur] for n in r.notes]
                fh.write(json.dumps({"id": r.id, "meter": r.meter, "notes": notes}) + "\n")

    print("aligned", len(aligned),
          "legacy", len(legacy),
          "legacy_duple", sum(r.id in ids for r in legacy))


if __name__ == "__main__":
    main()
