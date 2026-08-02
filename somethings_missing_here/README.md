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
scenes that appear nowhere in training.

## Structure

Both splits are organized by collection, then scene, then pair:

```
training/
├── DeTaken/                 original photographs by the dataset author
│   └── <scene>/             boxes, cabinet_window, magnets, red_stool, ...
│       └── pair_<nn>/
│           ├── before.jpg   earlier image
│           ├── after.jpg    later image
│           └── label.json   {"missing": true, "items": ["<removed object>"]}
│                            or {"missing": false, "items": []} for no-change pairs
└── Remove360_based/         pairs derived from Remove360 (see below)
    └── <scene>/             backyard_big_tree, backyard_bricks, backyard_toys,
        └── pair_<nn>/       bedroom, living-room, office, park, stairwell
test/
├── DeTaken/                 held-out scenes, never in training
│   └── test_<scene>/        test_broom, test_chair, ... (one pair each;
│       └── pair_<nn>/       test_michal_01 has 17 pairs)
└── Remove360_based/
    └── test_backyard/       3 held-out Remove360 pairs
```

Overview:

| split | collection | scenes | pairs | positive | negative |
|---|---|---|---|---|---|
| training | DeTaken | 13 | 87 | 50 | 37 |
| training | Remove360_based | 8 | 143 | 121 | 22 |
| test | DeTaken | 8 | 24 | 24 | 0 |
| test | Remove360_based | 1 | 3 | 3 | 0 |

Remove360-derived negative (no-change) pairs additionally carry a `source`
field in `label.json` documenting which images they were built from.

In `Remove360_based/` pairs (both splits), `before.jpg` is warped into
`after.jpg`'s camera frame and both images are cropped to their shared valid
region, so the two images are pixel-aligned with identical dimensions.
`DeTaken/` pairs (both splits) are handheld re-shots from approximately the
same viewpoint and are not pixel-aligned.

## The Remove360_based subset

[Remove360](https://huggingface.co/datasets/simkoc/Remove360) provides
separate pre-removal and post-removal camera walks of real indoor and
outdoor scenes. Its before and after images are independent captures — they
are **not** pixel-aligned pairs — so this subset was built by finding and
aligning the closest matching viewpoints between the two walks.

146 pairs — 124 positives (something was removed) and 22 negatives (nothing
changed) — across 9 scenes and 9 removed objects. Remove360's single large
backyard scene is split into three sub-scenes by area (big tree lawn, brick
patio, toy corner), and 3 backyard pairs are held out as `test/`
`Remove360_based/test_backyard`:

| scene | positive | negative | | removed object | pairs |
|---|---|---|---|---|---|
| backyard_toys | 23 | 5 | | chairs | 36 |
| backyard_bricks | 16 | 0 | | backpack | 24 |
| backyard_big_tree | 11 | 2 | | stroller | 16 |
| stairwell | 24 | 7 | | sofa | 15 |
| living-room | 21 | 5 | | deckchair | 11 |
| office | 13 | 1 | | bicycle | 10 |
| park | 10 | 2 | | pillows | 6 |
| bedroom | 3 | 0 | | table | 3 |
| test_backyard (test) | 3 | 0 | | toy-truck | 3 |

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
8. **No-change negatives** — within one scene, the before images of two
   different pairs that share the same removed-item label come from the same
   pre-removal camera walk (the item is present in both), and likewise the
   after images from the same post-removal walk (absent in both). Such
   same-side image pairs were matched and aligned with the same gates as the
   positives — with mutual frame coverage additionally capped at 96% so
   overlap statistics cannot separate the classes — and saved as
   `{"missing": false}` pairs.

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
