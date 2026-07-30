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
(the `Remove360_based/` folder), and is extended with original photographs
collected by the dataset author in additional folders alongside it.

## Structure

```
Remove360_based/             pairs derived from Remove360 (see below)
└── <scene>/                 backyard, bedroom, living-room, office, park, stairwell
    └── pair_<nn>/           pair_01, pair_02, ... (contiguous per scene)
        ├── before.jpg       aligned image with the object present
        ├── after.jpg        image with the object missing
        └── label.json       {"missing": true, "items": ["<removed object>"]}
```

Additional collections of original photographs follow the same
pair-folder layout with `before.jpg`, `after.jpg`, and `label.json`.

Within a pair, `before.jpg` is aligned to `after.jpg`'s camera frame and
both images are cropped to their shared valid region, so the two images are
pixel-aligned and have identical dimensions.

## The Remove360_based subset

[Remove360](https://huggingface.co/datasets/simkoc/Remove360) provides
separate pre-removal and post-removal camera walks of real indoor and
outdoor scenes. Its before and after images are independent captures — they
are **not** pixel-aligned pairs — so this subset was built by finding and
aligning the closest matching viewpoints between the two walks.

124 pairs across 6 scenes and 9 removed objects:

| scene | pairs | | removed object | pairs |
|---|---|---|---|---|
| backyard | 53 | | chairs | 36 |
| stairwell | 24 | | backpack | 24 |
| living-room | 21 | | stroller | 16 |
| office | 13 | | sofa | 15 |
| park | 10 | | deckchair | 11 |
| bedroom | 3 | | bicycle | 10 |
| | | | pillows | 6 |
| | | | table | 3 |
| | | | toy-truck | 3 |

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
