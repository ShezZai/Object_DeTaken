---
license: cc-by-nc-4.0
pretty_name: Something's Missing Here
size_categories:
  - n<1K
task_categories:
  - image-classification
tags:
  - change-detection
  - missing-object-detection
  - image-pairs
  - siamese
---

# Something's Missing Here

A dataset of viewpoint-aligned before/after image pairs for training and
evaluating missing-object detection: each pair shows the same scene from
(nearly) the same viewpoint, once with one or more objects present and once
after they were removed, with a label naming what went missing.

The dataset originated from pairs derived from the
[simkoc/Remove360](https://huggingface.co/datasets/simkoc/Remove360) dataset
(`training/Remove360_based/`), extended with original photographs collected
by the dataset author (`training/DeTaken/`), plus a held-out `test/` split of
scenes that appear nowhere in training. A separate `challenging/` split
holds harder cases — object swaps, replacements, and rearrangements — where
telling "something is missing" apart from "things merely changed" takes
more than spotting a difference (see below).

## Structure

Both splits are organized by collection, then scene, then pair:

```
training/
├── DeTaken/                 original photographs by the dataset author
│   └── <scene>/             bullseye, cabinet_window, corner_table, michal_01,
│       └── pair_<nn>/       shay_01, vases, ...
│           ├── before.jpg   earlier image
│           ├── after.jpg    later image
│           └── label.json   {"missing": true, "items": ["<removed object>"]}
│                            or {"missing": false, "items": []} for no-change pairs
└── Remove360_based/         pairs derived from Remove360 (see below)
    └── <scene>/             backyard_big_tree, backyard_bricks, backyard_stones,
        └── pair_<nn>/       backyard_toys, bedroom, living-room, office, park,
                             stairwell
test/
└── DeTaken/                 held-out scenes, never in training
    └── test_<scene>/        test_broom, test_chair, test_class_01..05,
        └── pair_<nn>/       test_phone, test_stroller, ...
challenging/                 harder cases: swaps, replacements, rearrangements
├── training/
│   └── <scene>/             same scene names as training/DeTaken
│       └── pair_<nn>/
└── test/
    └── pair_<nn>/           held-out challenging pairs (flat)
```

Overview (crop-shift = synthetic no-change pairs, see below):

| split | collection | scenes | pairs | positive | negative | of which crop-shift |
|---|---|---|---|---|---|---|
| training | DeTaken | 18 | 186 | 74 | 112 | 100 |
| training | Remove360_based | 9 | 372 | 124 | 248 | 248 |
| test | DeTaken | 13 | 108 | 51 | 57 | 16 |
| challenging | training | 17 | 81 | 27 | 54 | 0 |
| challenging | test | 1 | 12 | 6 | 6 | 0 |

Negative pairs that were derived rather than photographed carry a `source`
field in `label.json` documenting what they were built from.

## Crop-shift negatives

Many scenes were photographed only with something removed, leaving no
"nothing changed" pairs to learn from. Those scenes are supplemented with
**crop-shift negatives**: both images of the pair are crops of the *same*
photograph, taken at different offsets — one window trimmed at the right and
top edges, the other trimmed by the same amounts at the bottom and left. The
content is therefore identical and only the framing moves, which is exactly
the "the camera shifted, the scene did not" case a removal detector must not
mistake for a disappearance.

Two negatives are generated per source pair (one from its `before.jpg`, one
from its `after.jpg`), the trim is 10–20 px per axis, both crops come out the
same size, and each carries
`{"missing": false, "items": [], "source": "crop-shift of <pair>/<file>"}`.

They are cheap and plentiful, but they only vary translation — unlike real
re-shot no-change pairs, the lighting, focus, and perspective are identical.
Treat them as a supplement: the `crop-shift` column above shows how much of
each split's negative set is synthetic, and the held-out `test/` split keeps
its negatives mostly real so a model that has merely learned "small shift ⇒
nothing missing" shows up as false alarms.

In `Remove360_based/` pairs (both splits), `before.jpg` is warped into
`after.jpg`'s camera frame and both images are cropped to their shared valid
region, so the two images are pixel-aligned with identical dimensions.
`DeTaken/` and `challenging/` pairs are handheld re-shots from approximately
the same viewpoint and are not pixel-aligned. The `class_*` / `test_class_*`
scenes are tabletop object arrangements shot in rapid succession (seconds
apart), with roughly half of each scene being no-change negative pairs.

## The challenging split

`challenging/` holds the deliberately hard cases, separated from the main
splits so models can be trained and evaluated with or without them. In the
main splits, the change between `before` and `after` is a clean object
removal (or nothing at all) seen under a small viewpoint change. In
`challenging/`, other things happen too — objects are **swapped** with one
another, **replaced** by different objects, or **rearranged** within the
scene:

- Its **negatives** (`missing: false`, the majority here) are hard: items
  moved, swapped, or substituted — the scene visibly changed, yet nothing
  went missing. A model leaning on "the images differ, so something is
  gone" fails these.
- Its **positives** (`missing: true`) hide a genuine removal among such
  distractions, so spotting the difference is not enough — the model must
  identify that the change is specifically a disappearance.

Labels use the same `label.json` format. `challenging/test/` pairs are held
out from all training and sit directly under the folder (no scene
subfolders).

## The Remove360_based subset

[Remove360](https://huggingface.co/datasets/simkoc/Remove360) provides
separate pre-removal and post-removal camera walks of real indoor and
outdoor scenes. Its before and after images are independent captures — they
are **not** pixel-aligned pairs — so this subset was built by finding and
aligning the closest matching viewpoints between the two walks.

372 pairs — 124 positives (something was removed) and 248 crop-shift
negatives — across 9 scenes and 9 removed objects. Remove360's single large
backyard scene is split into four sub-scenes by area (big tree lawn, brick
patio, stones, toy corner):

| scene | positive | negative | | removed object | pairs |
|---|---|---|---|---|---|
| stairwell | 24 | 48 | | chairs | 36 |
| backyard_toys | 23 | 46 | | backpack | 24 |
| living-room | 21 | 42 | | stroller | 16 |
| backyard_bricks | 16 | 32 | | sofa | 15 |
| office | 13 | 26 | | deckchair | 11 |
| backyard_big_tree | 11 | 22 | | bicycle | 10 |
| park | 10 | 20 | | pillows | 6 |
| bedroom | 3 | 6 | | table | 3 |
| backyard_stones | 3 | 6 | | toy-truck | 3 |

### How it was generated

1. **Download** — the full `simkoc/Remove360` repository (file tree of
   `<scene>/<object>/train|test|masks`), where `train/` holds pre-removal
   ("before") images and `test/` holds post-removal ("after") images.
2. **Valid-region cropping** — a subset of Remove360's images is truncated
   at fixed byte boundaries on the Hub itself (all of `backyard/stroller` at
   2.75 MiB, all of `backyard/playhouse` at 256 KiB); truncated JPEGs decode
   with a uniform gray tail. Each image was cropped to its real content
   before matching, and images with less than 15% real content were
   discarded.
3. **Viewpoint matching** — every after image was ranked against all before
   images of the same object by SIFT feature matches (Lowe ratio 0.75); the
   top 3 candidates were verified with a RANSAC homography (reprojection
   threshold 4 px, minimum 40 inliers).
4. **Acceptance criteria** — a pair was kept only if each frame covers at
   least **85%** of the other under the homography (mutual frame coverage)
   and the warped before image correlates with the after image at
   **≥ 0.475** zero-mean normalized correlation. Each before image was used
   in at most one pair.
5. **Alignment and cropping** — the accepted before image was warped into
   the after frame at full resolution and both images were cropped to the
   largest rectangle of shared valid pixels.
6. **Labeling** — each pair's `label.json` records the removed object (the
   Remove360 object folder the pair came from) as
   `{"missing": true, "items": ["<object>"]}`.
7. **Manual curation** — the automatically accepted pairs were reviewed and
   some were deleted by hand; the remaining pairs were renumbered
   contiguously.
8. **No-change negatives** — this subset's negatives are the crop-shift
   pairs described above, two per positive. An earlier approach paired
   same-walk images (two pre-removal frames, or two post-removal frames,
   matched with the same gates as the positives) and a few such pairs may
   remain; their `label.json` `source` field names the two images they came
   from.

### Known limitations

- **No `playhouse` pairs** — all of Remove360's `backyard/playhouse` images
  are truncated to ~6% of their content on the Hub, which is below the
  usability floor.
- **Stroller pairs are half-height** — `backyard/stroller` images are
  truncated to roughly their top half, so its pairs are wide bands
  (~3900×1000) rather than full frames (~4000×2200).
- **Residual parallax** — alignment uses a single homography per pair; small
  parallax between the two camera positions can remain, especially on
  close foreground geometry.
- Some objects other than the labeled one may have shifted slightly between
  Remove360's two capture sessions.

## License and attribution

The `Remove360_based/` subset is a derivative of
[simkoc/Remove360](https://huggingface.co/datasets/simkoc/Remove360) and is
distributed under the same **CC-BY-NC-4.0** license, which this dataset
adopts as a whole. If you use it, please also cite the original Remove360
paper ([arXiv:2508.11431](https://arxiv.org/abs/2508.11431)).
