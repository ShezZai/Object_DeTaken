#!/usr/bin/env python3
"""Align two pictures, then mark objects unique to the first picture."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


SCRIPT_DIRECTORY = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run align_images.py followed by compare_yolo_objects.py in one command."
        )
    )
    parser.add_argument("pic1", type=Path)
    parser.add_argument("pic2", type=Path)
    parser.add_argument(
        "-o", "--output", type=Path, default=Path("unique_objects.jpg")
    )
    parser.add_argument("--overlap", type=float, default=0.15)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--pattern-threshold", type=float, default=0.50)
    parser.add_argument("--difference-min-area", type=float, default=0.001)
    parser.add_argument("--alignment-max-size", type=int, default=800)
    parser.add_argument(
        "--keep-aligned",
        action="store_true",
        help="keep intermediate <stem>_aligned images in the current directory",
    )
    args = parser.parse_args()

    for path in (args.pic1, args.pic2):
        if not path.is_file():
            parser.error(f"image does not exist or is not a file: {path}")
    for name in ("overlap", "confidence", "pattern_threshold"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.alignment_max_size < 32:
        parser.error("--alignment-max-size must be at least 32")
    if not 0.0 < args.difference_min_area <= 1.0:
        parser.error("--difference-min-area must be greater than 0 and at most 1")
    return args


def run_pipeline(args: argparse.Namespace, aligned1: Path, aligned2: Path) -> None:
    alignment_command = [
        sys.executable,
        str(SCRIPT_DIRECTORY / "align_images.py"),
        str(args.pic1),
        str(args.pic2),
        "--output1",
        str(aligned1),
        "--output2",
        str(aligned2),
        "--alignment-max-size",
        str(args.alignment_max_size),
    ]
    comparison_command = [
        sys.executable,
        str(SCRIPT_DIRECTORY / "compare_yolo_objects.py"),
        str(aligned1),
        str(aligned2),
        "--output",
        str(args.output),
        "--overlap",
        str(args.overlap),
        "--confidence",
        str(args.confidence),
        "--pattern-threshold",
        str(args.pattern_threshold),
        "--difference-min-area",
        str(args.difference_min_area),
    ]

    print("Step 1/2: aligning pictures", flush=True)
    subprocess.run(alignment_command, check=True)
    print("Step 2/2: detecting and comparing objects", flush=True)
    subprocess.run(comparison_command, check=True)


def main() -> None:
    args = parse_args()
    try:
        if args.keep_aligned:
            suffix1 = args.pic1.suffix or ".png"
            suffix2 = args.pic2.suffix or ".png"
            aligned1 = Path(f"{args.pic1.stem}_aligned{suffix1}")
            aligned2 = Path(f"{args.pic2.stem}_aligned{suffix2}")
            run_pipeline(args, aligned1, aligned2)
            print(f"Kept aligned images at {aligned1} and {aligned2}")
        else:
            with tempfile.TemporaryDirectory(prefix="align-and-compare-") as directory:
                temporary_directory = Path(directory)
                run_pipeline(
                    args,
                    temporary_directory / "pic1_aligned.jpg",
                    temporary_directory / "pic2_aligned.jpg",
                )
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"error: pipeline step failed with exit code {error.returncode}"
        ) from error

    print(f"Pipeline complete. Final marked image: {args.output}")


if __name__ == "__main__":
    main()
