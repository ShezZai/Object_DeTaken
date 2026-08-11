# Object DeTaken

A project by **Shay Gil** and **Michal Peri Markovich**.

Given two photos of the same scene — a "before" and an "after" — decide
whether something went missing, and point at what was taken.

## Motivation

Spotting that an object disappeared sounds trivial (subtract the images?)
but is not: the two photos are taken from slightly different viewpoints,
under different lighting, sometimes hours apart. A useful detector must
ignore camera shift, exposure, and shadows, yet still fire on a missing
cup — and must *not* fire when things merely moved. The removed-object case
matters wherever a space is handed over and checked: rentals, offices,
labs, exhibitions, care settings ("who took the phone?").

The project attacks the problem twice, with a purpose-built dataset:

1. a **classical pipeline** — no training, works out of the box;
2. a **learned siamese model** — trained to answer "did anything go
   missing?" directly.

## The dataset

[`Shay85Gil/somethings_missing_here`](https://huggingface.co/datasets/Shay85Gil/somethings_missing_here)
— 559 viewpoint-aligned before/after pairs with labels naming the removed
items: 350 training and 116 held-out test pairs across 40 scenes, plus a
93-pair `challenging` split (swaps, replacements, rearrangements) where
"things changed" must be told apart from "something is gone". Original
photographs by the authors plus pairs derived from
[Remove360](https://huggingface.co/datasets/simkoc/Remove360). Test pairs
contain no synthetic imagery. `download_dataset.py` fetches it into
`somethings_missing_here/`.

## The detection pipeline (`pipeline/`)

Classical computer vision, one command:

```bash
python pipeline/align_and_compare.py before.jpg after.jpg -o unique_objects.jpg
```

It ECC-aligns the pair, detects objects with YOLO11 and adds
label-independent candidates from image differencing, keeps picture-1
boxes with no overlapping counterpart and no template match in picture 2,
and names the survivors with an ImageNet classifier. On the dataset's test
split it flags 95 of 97 removals with 12 false alarms on 19 no-change
pairs (pair-level accuracy 0.88, recall 0.98) — high recall, but noisy on
scenes that merely shifted, and its object *names* are limited by the
ImageNet vocabulary. Details, options, and per-step usage:
[`pipeline/README.md`](pipeline/README.md).

## The learned model (`siamese_missing/`)

A siamese binary classifier (`train_missing.py`): one shared pretrained
encoder embeds both photos, and a small head reads
`[f_before, f_after, f_before − f_after]` — the *signed* difference, so
removal and addition are distinguishable. Training holds out whole scenes
for validation (a scene never appears in both train and validation), uses
heavy shared-geometry/independent-photometry augmentation, and picks the
decision threshold on validation (Youden's J); at inference every
checkpoint matching a glob is averaged, so multi-seed ensembles come free.
Two interchangeable implementations behind one CLI: Keras 3 +
EfficientNetB0 (`--keras`, the default) and PyTorch + ResNet18 (`--torch`).

**The best results came from the Keras/EfficientNetB0 implementation** —
which is why it is the default.
Other lessons the experiments taught us, mostly the hard way:

- **Seed variance dominates.** The identical configuration scored test
  AUC 0.93 with one seed and 0.73 with another — bigger than any single
  hyperparameter effect we measured. Conclusions need multi-seed runs;
  single-run comparisons at this dataset size (hundreds of pairs) mostly
  compare luck.
- **Validation can stop tracking the real task.** Adding reversed-pair
  negatives (`--reverse`) raises validation AUC while changing what
  validation measures; the held-out natural test pairs are the only
  arbiter, which is why `evaluate` refuses to score reversals.
- **Fine-tuning the encoder is fragile on this data size** — freezing it
  avoids sudden post-unfreeze collapses, and the per-epoch `--graph`
  curves make that failure mode visible.

Training, prediction, evaluation, and Grad-CAM visualization are
documented in [`siamese_missing/README.md`](siamese_missing/README.md).

## Setup

Requires Python 3.10+.

Linux / macOS:

```bash
./setup_venv.sh
source .venv/bin/activate
```

Windows:

```powershell
py setup_venv.py
.venv\Scripts\Activate.ps1
```

The script creates a `.venv` virtual environment and installs
`requirements.txt`. YOLO model weights (`yolo11n.pt`, `yolo11n-cls.pt`)
are downloaded automatically by Ultralytics on first run.

## Files

- `pipeline/` — the classical alignment + detection pipeline
  (see `pipeline/README.md`)
- `siamese_missing/` — the trainable siamese model
  (see `siamese_missing/README.md`)
- `download_dataset.py` — fetch the dataset from Hugging Face
- `make_crop_negatives.py` — generate crop-shift no-change pairs
  (training splits only)
- `setup_venv.sh` / `setup_venv.py` — environment setup
