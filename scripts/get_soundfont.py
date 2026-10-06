"""Download the General MIDI soundfont the app uses to render real instruments.

    python -m scripts.get_soundfont [--dest assets/soundfonts]

GeneralUser GS (S. Christian Collins) is fetched from its public GitHub
repository. It is not bundled with this project: check its licence terms before
redistributing it. Any other General MIDI ``.sf2`` placed in the same folder (or
named by ``$MELODYMORPH_SOUNDFONT``) works too.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

URL = "https://github.com/mrbumpy409/GeneralUser-GS/raw/main/GeneralUser-GS.sf2"
NAME = "GeneralUser-GS.sf2"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", default="assets/soundfonts", help="folder to save into")
    ap.add_argument("--force", action="store_true", help="download again even if present")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / NAME
    if target.exists() and not args.force:
        print(f"already have {target} ({target.stat().st_size / 1e6:.1f} MB)")
        return 0
    part = target.with_suffix(".part")
    print(f"downloading {URL}")
    try:
        with urllib.request.urlopen(URL, timeout=60) as resp, part.open("wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while chunk := resp.read(1 << 20):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:5.1f} / {total / 1e6:.1f} MB", end="", flush=True)
        part.replace(target)
    except OSError as exc:
        part.unlink(missing_ok=True)
        print(f"\ndownload failed: {exc}", file=sys.stderr)
        return 1
    print(f"\nsaved {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
