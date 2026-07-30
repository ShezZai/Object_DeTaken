#!/usr/bin/env python3
"""Load the Remove360 dataset from Hugging Face.

By default this streams a few samples to verify access without downloading
the full ~30 GB dataset. Pass --download to fetch and cache everything.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "simkoc/Remove360"


def load_env_token() -> str | None:
    """Read HF_TOKEN from the environment or a .env file next to this script."""
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.is_file():
        for line in env_path.read_text().splitlines():
            key, _, value = line.partition("=")
            if key.strip() == "HF_TOKEN":
                return value.strip().strip('"').strip("'")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--download",
        action="store_true",
        help="download and cache the full dataset (~30 GB) instead of streaming",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=3,
        help="number of streamed samples to preview (default: 3)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    token = load_env_token()

    if args.download:
        ds = load_dataset(DATASET_NAME, token=token)
        print(ds)
        return

    ds = load_dataset(DATASET_NAME, streaming=True, token=token)
    print(f"Streaming preview of {DATASET_NAME} (use --download for the full set)")
    for split_name, split in ds.items():
        print(f"\nSplit '{split_name}': features {list(split.features)}")
        for index, sample in enumerate(split):
            if index >= args.samples:
                break
            described = {
                key: getattr(value, "size", value) for key, value in sample.items()
            }
            print(f"  sample {index}: {described}")


if __name__ == "__main__":
    main()
