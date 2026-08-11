#!/usr/bin/env python3
"""Build no-change pairs from positive-only scenes by re-cropping one photo.

Reads a text file of paths (one per line, relative to --root) naming either
a pair folder or a whole scene folder -- a scene expands to every pair it
currently holds -- and, for each pair, writes TWO new negative pairs:

    before.jpg -> before trimmed right+top   vs  before trimmed bottom+left
    after.jpg  -> after  trimmed right+top   vs  after  trimmed bottom+left

Both images of a new pair come from the SAME photo, so nothing went missing
between them -- only the framing shifted, which is exactly the "the camera
moved, the scene did not" case a removal detector must not fall for. Trim
amounts are random per edge (--trim-min..--trim-max pixels).

New pairs land in the source pair's scene folder, continuing its numbering,
labelled {"missing": false, "items": []}.

    python make_crop_negatives.py pairs.txt --root somethings_missing_here
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2

IMG_EXT = (".jpg", ".jpeg", ".png")


def find_image(pair_dir: Path, stem: str) -> Path | None:
    for ext in IMG_EXT:
        candidate = pair_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None


def is_generated(pair_dir: Path) -> bool:
    """True for pairs this script produced, so re-runs never crop a crop."""
    label = pair_dir / "label.json"
    if not label.is_file():
        return False
    try:
        source = json.loads(label.read_text()).get("source", "")
    except json.JSONDecodeError:
        return False
    return str(source).startswith("crop-shift")


def expand(entry: str, root: Path, problems: list) -> list[str]:
    """A pair path stays as is; a scene path becomes all its pairs.

    The scene is listed once, up front, so pairs written during this run are
    never themselves used as sources.
    """
    path = root / entry
    if not path.is_dir():
        problems.append(f"{entry}: no such folder")
        return []
    if any(find_image(path, stem) for stem in ("before", "after")):
        return [entry]                       # a pair folder
    pairs = sorted((p for p in path.glob("pair_*") if p.is_dir()),
                   key=lambda p: p.name)
    fresh = [p for p in pairs if not is_generated(p)]
    if not pairs:
        problems.append(f"{entry}: neither a pair nor a scene with pairs")
    skipped = len(pairs) - len(fresh)
    print(f"{entry}: scene with {len(fresh)} pairs"
          + (f" ({skipped} generated pairs skipped)" if skipped else ""))
    return [str(p.relative_to(root)) for p in fresh]


def next_pair_number(scene: Path) -> int:
    numbers = [int(p.name.split("_")[1]) for p in scene.glob("pair_*")
               if p.is_dir() and p.name.split("_")[1].isdigit()]
    return max(numbers, default=0) + 1


def crop_pair(image, rng, trim_min, trim_max):
    """Two crops of one image, shifted in opposite directions.

    The same trim amounts are applied to OPPOSITE edges -- right+top on one
    crop, bottom+left on the other -- so the two windows sit at different
    offsets (the shift) but come out exactly the same size. Differing sizes
    break the shared augmentation, which requires all targets to match.
    """
    height, width = image.shape[:2]
    dx = rng.randint(trim_min, trim_max)
    dy = rng.randint(trim_min, trim_max)
    if min(height - dy, width - dx) < 32:
        raise ValueError(f"image too small to trim ({width}x{height})")
    return (image[dy:, :width - dx],          # trimmed right + top
            image[:height - dy, dx:])         # trimmed bottom + left


def repair(root: Path, dry_run: bool) -> None:
    """Equalize the image sizes of crop-shift pairs already on disk.

    Early runs trimmed each edge by an independent amount, so the two crops
    could differ by a few pixels -- which the shared augmentation rejects
    ("Height and Width of image, mask or masks should be equal"). Trim each
    image a little further on the edges it was already trimmed on, so both
    reach a common size and the shift direction is preserved.
    """
    fixed = already = 0
    for label in sorted(root.rglob("label.json")):
        pair = label.parent
        if not is_generated(pair):
            continue
        before_path, after_path = pair / "before.jpg", pair / "after.jpg"
        before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
        after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
        if before is None or after is None:
            continue
        if before.shape[:2] == after.shape[:2]:
            already += 1
            continue

        height = min(before.shape[0], after.shape[0])
        width = min(before.shape[1], after.shape[1])
        # before was trimmed right+top: take more off the top and the right.
        before_fixed = before[before.shape[0] - height:, :width]
        # after was trimmed bottom+left: take more off the bottom and left.
        after_fixed = after[:height, after.shape[1] - width:]
        print(f"{'would fix' if dry_run else 'fixing'} "
              f"{pair.relative_to(root)}: "
              f"{before.shape[1]}x{before.shape[0]} / "
              f"{after.shape[1]}x{after.shape[0]} -> {width}x{height}")
        if not dry_run:
            cv2.imwrite(str(before_path), before_fixed,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            cv2.imwrite(str(after_path), after_fixed,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        fixed += 1

    print(f"\n{fixed} crop-shift pairs equalized, {already} already matched"
          + (" (dry run)" if dry_run else ""))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("list_file", type=Path, nargs="?",
                        help="text file of pair or scene folders, one per "
                             "line (blank lines and # comments ignored)")
    parser.add_argument("--repair", action="store_true",
                        help="skip generation; instead equalize the image "
                             "sizes of crop-shift pairs already under --root")
    parser.add_argument("--root", type=Path, default=Path("."),
                        help="folder the listed paths are relative to")
    parser.add_argument("--trim-min", type=int, default=10)
    parser.add_argument("--trim-max", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be written, write nothing")
    args = parser.parse_args()
    if args.trim_min < 1 or args.trim_max < args.trim_min:
        parser.error("need 1 <= --trim-min <= --trim-max")
    if not args.repair and args.list_file is None:
        parser.error("list_file is required (or use --repair)")
    return args


def main() -> None:
    args = parse_args()
    if args.repair:
        repair(args.root, args.dry_run)
        return
    rng = random.Random(args.seed)

    lines = [line.strip() for line in args.list_file.read_text().splitlines()]
    listed = [line for line in lines if line and not line.startswith("#")]

    problems = []
    entries = [pair for entry in listed
               for pair in expand(entry.rstrip("/"), args.root, problems)]

    written = 0
    counters = {}          # scene -> next number, so several listed pairs
    for entry in entries:  # from one scene keep counting up
        pair_dir = args.root / entry
        scene = pair_dir.parent
        number = counters.get(scene) or next_pair_number(scene)
        for stem in ("before", "after"):
            source = find_image(pair_dir, stem)
            if source is None:
                problems.append(f"{entry}: no {stem} image")
                continue
            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                problems.append(f"{entry}: could not decode {source.name}")
                continue
            try:
                first, second = crop_pair(image, rng, args.trim_min,
                                          args.trim_max)
            except ValueError as error:
                problems.append(f"{entry}: {error}")
                continue

            target = scene / f"pair_{number:02d}"
            print(f"{'would write' if args.dry_run else 'writing'} "
                  f"{target.relative_to(args.root)} "
                  f"(from {entry}/{source.name})")
            if not args.dry_run:
                target.mkdir()
                cv2.imwrite(str(target / "before.jpg"), first,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                cv2.imwrite(str(target / "after.jpg"), second,
                            [cv2.IMWRITE_JPEG_QUALITY, 95])
                (target / "label.json").write_text(json.dumps({
                    "missing": False,
                    "items": [],
                    "source": f"crop-shift of {entry}/{source.name}",
                }, indent=2) + "\n")
            written += 1
            number += 1
            counters[scene] = number

    print(f"\n{written} negative pairs from {len(entries)} source pairs "
          f"({len(listed)} lines)" + (" (dry run)" if args.dry_run else ""))
    for problem in problems:
        print(f"  PROBLEM {problem}")


if __name__ == "__main__":
    main()
