#!/usr/bin/env python3
"""Download the somethings_missing_here dataset from Hugging Face.

The repo is private, so a token is needed: read from the HF_TOKEN
environment variable, or from the .env file next to this script
(HF_TOKEN="hf_...").

    python download_dataset.py                # into ./somethings_missing_here
    python download_dataset.py --output /elsewhere
    python download_dataset.py --fresh        # wipe the local copy first

A plain run is an incremental update: new and changed files are fetched,
but files deleted or renamed on the Hub linger locally. --fresh removes the
local copy first so the result mirrors the Hub exactly.
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
DEFAULT_REPO = "Shay85Gil/somethings_missing_here"


def load_env_token() -> str | None:
    """Read HF_TOKEN from the environment or a .env file next to this script."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip('"').strip("'")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=DEFAULT_REPO,
                        help=f"dataset repo id (default: {DEFAULT_REPO})")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "somethings_missing_here",
                        help="target folder (default: ./somethings_missing_here)")
    parser.add_argument("--fresh", action="store_true",
                        help="delete the local copy first and re-download "
                             "from scratch (removes stale renamed/deleted files)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = load_env_token()
    if token is None:
        raise SystemExit(
            "error: no HF token found -- set HF_TOKEN or put it in .env")

    if args.fresh and args.output.exists():
        # Only wipe something that actually looks like a dataset copy, so a
        # mistyped --output cannot delete an unrelated folder.
        markers = ("README.md", "training", "test", ".cache")
        if not any((args.output / m).exists() for m in markers):
            raise SystemExit(
                f"error: {args.output} does not look like a dataset copy; "
                "refusing to delete it")
        print(f"Removing local copy at {args.output}")
        shutil.rmtree(args.output)

    path = snapshot_download(args.repo, repo_type="dataset",
                             local_dir=args.output, token=token)

    pairs = sum(1 for _ in Path(path).rglob("label.json"))
    print(f"Downloaded {args.repo} to {path} ({pairs} labeled pairs)")


if __name__ == "__main__":
    main()
