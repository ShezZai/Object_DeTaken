"""
bootstrap_labels.py
Pre-label pairs using the detector diff (missing_items.py), so you hand-correct
instead of annotating 400 pairs from scratch.

Writes label.json into every pair folder that does not already have one:
    {"missing": true, "items": ["cup"], "source": "detector", "verified": false}

    python bootstrap_labels.py data
    python bootstrap_labels.py data --overwrite-unverified

Then open the ones you doubt, fix "items"/"missing", and set "verified": true.
Anything still marked verified:false is a guess -- keep those OUT of your
held-out evaluation set, or you measure agreement with the detector rather
than correctness.
"""

import argparse
import json
from pathlib import Path

from missing_items import find_missing

IMG_EXT = (".jpg", ".jpeg", ".png")


def find_image(pair_dir, stem):
    for ext in IMG_EXT:
        p = pair_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?", default="data")
    ap.add_argument("--overwrite-unverified", action="store_true",
                    help="re-run on labels with verified:false")
    ap.add_argument("--conf-before", type=float, default=0.45)
    ap.add_argument("--conf-after", type=float, default=0.25)
    args = ap.parse_args()

    written = skipped = 0
    for pair_dir in sorted(p for p in Path(args.root).glob("*/*") if p.is_dir()):
        before, after = find_image(pair_dir, "before"), find_image(pair_dir, "after")
        if before is None or after is None:
            print(f"  skip {pair_dir}: missing image")
            continue

        lf = pair_dir / "label.json"
        if lf.exists():
            try:
                existing = json.loads(lf.read_text())
            except json.JSONDecodeError:
                existing = {}
            if existing.get("verified") or not args.overwrite_unverified:
                skipped += 1
                continue

        r = find_missing(str(before), str(after), args.conf_before, args.conf_after)
        lf.write_text(json.dumps({
            "missing": bool(r["any_missing"]),
            "items": r["missing"],
            "source": "detector",
            "verified": False,
        }, indent=2) + "\n")
        written += 1
        print(f"  {pair_dir.parent.name}/{pair_dir.name}: "
              f"{'MISSING ' + ', '.join(r['missing']) if r['missing'] else 'nothing missing'}")

    print(f"\nwrote {written} label(s), left {skipped} existing untouched")
    print('now hand-check them and set "verified": true on the ones you confirm')


if __name__ == "__main__":
    main()
