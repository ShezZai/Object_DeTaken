"""
pairs_io.py
Read the scene/pair folder tree straight into memory: no manifest file, no
generated folders. Positive pairs are reversed on the fly to synthesise hard
negatives (removing an item, read backwards, is adding one).

    from pairs_io import load_pairs, ImageCache

    rows = load_pairs("data")                 # includes reversals
    cache = ImageCache(max_side=640)          # decode each JPEG once

Because a reversed pair points at the SAME two files as its source, the cache
makes reversal free: no extra decode, no extra pixels held in RAM.

Layout expected:
    data/scene_03/pair_001/{before.jpg, after.jpg, label.json}
    label.json: {"missing": true, "items": ["cup"]}
"""

from pathlib import Path
import json

import cv2
import numpy as np

IMG_EXT = (".jpg", ".jpeg", ".png")


# --------------------------------------------------------------------------- #
# scanning
# --------------------------------------------------------------------------- #
def _find_image(pair_dir, stem):
    for ext in IMG_EXT:
        p = pair_dir / f"{stem}{ext}"
        if p.exists():
            return str(p)
    return None


def _is_positive(meta):
    m = meta.get("missing")
    return m is True or (isinstance(m, list) and len(m) > 0)


def load_pairs(root, reverse_positives=True, verbose=True):
    """
    Returns a list of dicts:
        {before, after, y, scene, items, reversed, pair}

    `y` is 1.0 if something went missing. Reversed rows carry y=0.0 and keep
    the same `scene`, so scene-grouped CV keeps a reversal in the same fold as
    its source -- otherwise the model sees the same photographs in train and
    val and validation becomes meaningless.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"no such folder: {root}")

    rows, problems = [], []

    for pair_dir in sorted(p for p in root.glob("*/*") if p.is_dir()):
        before = _find_image(pair_dir, "before")
        after = _find_image(pair_dir, "after")
        label_file = pair_dir / "label.json"

        if before is None or after is None:
            problems.append(f"{pair_dir}: missing before/after image")
            continue
        if not label_file.exists():
            problems.append(f"{pair_dir}: no label.json (unlabelled?)")
            continue
        try:
            meta = json.loads(label_file.read_text())
        except json.JSONDecodeError as e:
            problems.append(f"{label_file}: bad JSON ({e})")
            continue
        if "missing" not in meta:
            problems.append(f"{label_file}: no 'missing' key")
            continue

        positive = _is_positive(meta)
        scene = pair_dir.parent.name

        rows.append({
            "before": before, "after": after,
            "y": 1.0 if positive else 0.0,
            "scene": scene, "items": meta.get("items", []),
            "reversed": False, "pair": pair_dir.name,
        })

        if positive and reverse_positives:
            rows.append({
                "before": after, "after": before,      # swapped
                "y": 0.0,
                "scene": scene, "items": [],
                "reversed": True, "pair": pair_dir.name,
            })

    if verbose:
        summarise(rows, problems)
    return rows


def summarise(rows, problems=()):
    scenes = sorted({r["scene"] for r in rows})
    pos = int(sum(r["y"] for r in rows))
    rev = sum(r["reversed"] for r in rows)
    files = len({r["before"] for r in rows} | {r["after"] for r in rows})

    print(f"{len(rows)} pairs ({len(rows)-rev} real + {rev} reversed) "
          f"from {files} image files")
    print(f"  scenes:   {len(scenes)}")
    print(f"  positive: {pos}   negative: {len(rows)-pos}")

    for s in scenes:
        sr = [r for r in rows if r["scene"] == s and not r["reversed"]]
        p = sum(r["y"] for r in sr)
        if len(sr) > 2 and p in (0, len(sr)):
            print(f"  WARNING: scene {s} is entirely "
                  f"{'positive' if p else 'negative'} ({len(sr)} real pairs)")

    if problems:
        print(f"  {len(problems)} problem(s):")
        for p in list(problems)[:20]:
            print(f"    {p}")


# --------------------------------------------------------------------------- #
# decode cache
# --------------------------------------------------------------------------- #
class ImageCache:
    """
    Decode each file once, hold it as RGB uint8, hand out references.

    Sizing: a 640-max-side image is ~1.2 MB. 500 pairs == ~1000 files == ~1.2 GB.
    Drop max_side to 384 (~0.4 MB each, ~400 MB) if that is too much, but keep it
    at least as large as your training input or you upsample blurry pixels.

    Set max_side=None to cache originals untouched (watch your RAM with phone
    photos -- a 12 MP image is ~36 MB decoded).

    Not safe to share across DataLoader workers: each worker gets its own copy,
    so use num_workers=0 when caching, or accept N copies. With a cache warm,
    num_workers=0 is usually faster anyway -- there is nothing left to parallelise.
    """

    def __init__(self, max_side=640):
        self.max_side = max_side
        self._store = {}

    def get(self, path):
        img = self._store.get(path)
        if img is None:
            raw = cv2.imread(path, cv2.IMREAD_COLOR)
            if raw is None:
                raise FileNotFoundError(f"could not decode {path}")
            img = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
            if self.max_side:
                h, w = img.shape[:2]
                s = self.max_side / max(h, w)
                if s < 1.0:
                    img = cv2.resize(img, (round(w * s), round(h * s)),
                                     interpolation=cv2.INTER_AREA)
            img.flags.writeable = False          # guard against in-place augs
            self._store[path] = img
        return img

    def warm(self, rows, verbose=True):
        paths = {r["before"] for r in rows} | {r["after"] for r in rows}
        for p in paths:
            self.get(p)
        if verbose:
            print(f"cached {len(self._store)} images, {self.nbytes/1e6:.0f} MB")
        return self

    @property
    def nbytes(self):
        return sum(v.nbytes for v in self._store.values())

    def __len__(self):
        return len(self._store)


# --------------------------------------------------------------------------- #
def scene_folds(rows, n_folds=5, seed=42):
    """Yield (train_rows, val_rows) with whole scenes held out."""
    scenes = sorted({r["scene"] for r in rows})
    if len(scenes) < n_folds:
        raise ValueError(f"need >= {n_folds} scenes, found {len(scenes)}")
    order = np.random.default_rng(seed).permutation(scenes)
    for chunk in np.array_split(order, n_folds):
        held = set(chunk)
        yield ([r for r in rows if r["scene"] not in held],
               [r for r in rows if r["scene"] in held])


if __name__ == "__main__":
    import sys
    rows = load_pairs(sys.argv[1] if len(sys.argv) > 1 else "data")
    ImageCache(max_side=640).warm(rows)
