#!/usr/bin/env python3
"""Reorganize aligned/ pairs into aligned_ordered/<scene>/<pair_num>/.

Each pair folder contains before.jpg, after.jpg, and label.json of the form
{"missing": true, "items": ["<removed object name>"]}. Pair numbers restart
at 0 per scene and follow the manifest order, merging all of a scene's
objects into one sequence.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aligned",
        type=Path,
        default=Path(__file__).resolve().parent / "aligned",
        help="input folder produced by make_aligned_pairs.py",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output folder (default: <aligned>/../aligned_ordered)",
    )
    args = parser.parse_args()
    args.output = args.output or args.aligned.parent / "aligned_ordered"
    return args


def main() -> None:
    args = parse_args()
    manifest_path = args.aligned / "pairs.csv"
    if not manifest_path.is_file():
        raise SystemExit(f"error: manifest not found: {manifest_path}")

    if args.output.exists():
        shutil.rmtree(args.output)

    scene_counters: Counter[str] = Counter()
    with open(manifest_path, newline="") as manifest_file:
        for row in csv.DictReader(manifest_file):
            scene, item = row["object"].split("/", 1)
            source_directory = args.aligned / "scene" / row["object"]
            before = source_directory / f"before_{row['index']}.jpg"
            after = source_directory / f"after_{row['index']}.jpg"
            if not before.is_file() or not after.is_file():
                raise SystemExit(f"error: missing pair images for {row}")

            pair_number = scene_counters[scene]
            scene_counters[scene] += 1
            pair_directory = args.output / scene / str(pair_number)
            pair_directory.mkdir(parents=True)
            shutil.copy2(before, pair_directory / "before.jpg")
            shutil.copy2(after, pair_directory / "after.jpg")
            (pair_directory / "label.json").write_text(
                json.dumps({"missing": True, "items": [item]}, indent=2) + "\n"
            )

    total = sum(scene_counters.values())
    for scene in sorted(scene_counters):
        print(f"{scene}: {scene_counters[scene]} pairs")
    print(f"Total: {total} pairs in {args.output}")


if __name__ == "__main__":
    main()
