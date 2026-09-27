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
Three interchangeable implementations behind one CLI: Keras 3 +
EfficientNetB0 (`--keras`, the default), PyTorch + ResNet18 (`--torch`),
and a PyTorch ViT that compares patch tokens (`--vit`, see below).

**Before the ViT addition, Keras/EfficientNetB0 was the winner** of the
two CNN encoders (EfficientNetB0 vs. ResNet18), which is why it is the
default. The ViT concept now beats it — see
[ViT addition and its checks against Keras](#vit-addition-and-its-checks-against-keras-efficientnetb0).
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

## ViT addition and its checks against Keras (EfficientNetB0)

Both CNN models pool each photo into one vector before comparing. A
missing cup covers ~1% of the image, so after pooling it is ~1% of the
feature vector. The ViT model (`siamese_missing/Vit_siamese.py`,
`--vit`) compares *patches* instead:

1. a frozen DINOv2 ViT-S/14 keeps its patch tokens (no pooling);
2. every *before* patch looks for itself among the *after* patches with
   cross-attention — a moved object is found elsewhere, a taken one is
   not (the learned version of the pipeline's template-matching step);
3. a small transformer scores each before-patch as "gone", and the pair
   logit is a smooth max over those scores — one unexplained patch is
   enough. The per-patch scores are trained by the classification loss
   and double as a localization map (`evaluate --viz`).

Only ~315k parameters train; the encoder stays frozen.

**Result: the ViT concept is better, and far more robust across seeds.**
Before this addition Keras/EfficientNetB0 was the winner (EfficientNetB0
vs. ResNet18 encoders); against the ViT it loses on every seed.

### Setup of the check

The same command for both, only the backend flag swapped
(`train --keras` / `train --vit`, each with its own defaults: 256 px for
Keras, 224 px for the ViT), on the dataset's `training` split, 5 seeds
(42, 7, 1, 2, 3). Each seed holds out different scenes for validation.
Every model is then scored on the 116 held-out `test` pairs (97 with a
removal, 19 without) and the 12 `challenging/test` pairs; the ensemble
averages all 5 seeds.

Decision rule: the ViT uses **Platt scaling** (a logistic fit of the
model's logit on validation, with Platt's smoothed targets, cut at 0.5);
Keras uses the **stored validation threshold** (Youden's J, middle of the
best plateau). These are the per-backend defaults of `--calibration` —
each backend got the rule that measured better for it (see
[Choosing the decision rule](#choosing-the-decision-rule)).

### `test` — per seed (116 pairs)

TP = removal caught, FP = false alarm, FN = removal missed, TN = no-change
pair correctly passed.

| Seed | Model | Val AUC | Test AUC | Acc | TP | FP | FN | TN | Precision | Recall |
|---|---|---|---|---|---|---|---|---|---|---|
| 42 | Keras | 0.685 | 0.759 | 0.698 | 69 | 7 | 28 | 12 | 0.908 | 0.711 |
| 42 | **ViT** | 0.983 | **0.950** | **0.888** | 88 | 4 | 9 | 15 | 0.957 | 0.907 |
| 7 | Keras | 0.644 | 0.789 | 0.750 | 72 | 4 | 25 | 15 | 0.947 | 0.742 |
| 7 | **ViT** | 0.964 | **0.938** | **0.888** | 86 | 2 | 11 | 17 | 0.977 | 0.887 |
| 1 | Keras | 0.528 | 0.849 | 0.784 | 79 | 7 | 18 | 12 | 0.919 | 0.814 |
| 1 | **ViT** | 0.924 | **0.923** | **0.879** | 85 | 2 | 12 | 17 | 0.977 | 0.876 |
| 2 | Keras | 0.796 | 0.823 | **0.784** | 77 | 5 | 20 | 14 | 0.939 | 0.794 |
| 2 | **ViT** | 1.000 | **0.900** | 0.759 | 69 | 0 | 28 | 19 | 1.000 | 0.711 |
| 3 | Keras | 0.704 | 0.679 | 0.690 | 73 | 12 | 24 | 7 | 0.859 | 0.753 |
| 3 | **ViT** | 0.927 | **0.918** | **0.853** | 84 | 4 | 13 | 15 | 0.955 | 0.866 |

### `test` — across seeds, and the 5-seed ensembles

| | Keras | ViT |
|---|---|---|
| Test AUC, mean (range) | 0.780 (0.679–0.849) | **0.926 (0.900–0.950)** |
| Accuracy, mean (range) | 0.741 (0.690–0.784) | **0.853 (0.759–0.888)** |
| Validation AUC, range | 0.528–0.796 | 0.924–1.000 |
| Seeds won on test AUC | 0 / 5 | **5 / 5** |
| Train time per seed | 81–278 s | 75–92 s |
| Model latency, GPU | 60.9 ms | **6.9 ms** |

Confusion matrices summed over the 5 single-seed models (5 × 116 = 580
decisions each):

| Keras, 5 seeds | Predicted missing | Predicted no change |
|---|---|---|
| **Actually missing** (485) | TP 370 | FN 115 |
| **Actually no change** (95) | FP 35 | TN 60 |

| ViT, 5 seeds | Predicted missing | Predicted no change |
|---|---|---|
| **Actually missing** (485) | TP **412** | FN **73** |
| **Actually no change** (95) | FP **12** | TN **83** |

Recall 76.3% → 84.9%, false-alarm rate 36.8% → 12.6%: the ViT catches
more removals *and* raises a third of the false alarms.

5-seed ensembles:

| Keras ensemble (AUC 0.817, acc 0.767) | Predicted missing | Predicted no change |
|---|---|---|
| **Actually missing** (97) | TP 76 | FN 21 |
| **Actually no change** (19) | FP 6 | TN 13 |

| ViT ensemble (AUC 0.958, acc 0.905) | Predicted missing | Predicted no change |
|---|---|---|
| **Actually missing** (97) | TP **87** | FN **10** |
| **Actually no change** (19) | FP **1** | TN **18** |

The ViT ensemble reaches the classical pipeline's accuracy (0.88 there,
0.905 here) with 1 false alarm instead of 12, at a lower recall (0.897
vs. 0.98).

### Cost: training time, parameters, latency

Measured on one machine: NVIDIA RTX 5060 Ti 16 GB, Intel i7-14700F, each
backend at its default input size (Keras 256 px, ViT 224 px).

**Parameters**

| | Keras (EfficientNetB0) | ViT (DINOv2 ViT-S/14) |
|---|---|---|
| Encoder | 4,049,571 | 22,056,192 |
| Head | 492,033 | 314,625 |
| **Total** | **4,541,604** | **22,370,817** |
| Trained | 492,033 for 10 epochs, then all 4,541,604 | 314,625 — the encoder is never unfrozen |

The ViT is ~5× larger in total but trains ~36% fewer parameters, and none
of them in the encoder — which is why it cannot suffer the post-unfreeze
collapses the CNNs are prone to.

**Training time** (`--cache`, wall clock incl. startup and image caching;
epochs = best epoch + 15 patience, capped at 60)

| Seed | Keras | ViT |
|---|---|---|
| 42 | 107 s (22 epochs) | 77 s (29 epochs) |
| 7 | 121 s (25 epochs) | 86 s (34 epochs) |
| 1 | 81 s (16 epochs) | 92 s (38 epochs) |
| 2 | 239 s (52 epochs) | 75 s (27 epochs) |
| 3 | 278 s (60 epochs) | 92 s (37 epochs) |
| **5 seeds** | **826 s (13.8 min), ~4.7 s/epoch** | **422 s (7.0 min), ~2.6 s/epoch** |

The ViT's epochs are ~45% faster (its encoder never needs a backward
pass, and it runs at 224 px against Keras' 256 px), and its run length is steadier: 27–38 epochs against
Keras' 16–60.

**Inference latency per pair** (batch 1, median of 50 after warm-up, one
model — an ensemble costs this × its size)

| | Keras GPU | ViT GPU | Keras CPU | ViT CPU |
|---|---|---|---|---|
| Model only (both photos through the network) | 60.9 ms | **6.9 ms** | 163.5 ms | **43.5 ms** |
| End to end (`predict`: 2 JPEG decodes + resize + model) | 129.8 ms | **74.4 ms** | 223.2 ms | **112.6 ms** |

The model itself is ~9× faster on GPU and ~4× on CPU. Keras' time is
real compute, not API overhead (a direct model call measures the same
59 ms): the Keras 3 torch backend runs EfficientNetB0's depthwise
convolutions op by op at batch 1, while the ViT is a few large matrix
multiplies. End to end, decoding the two full-size photos dominates the
ViT's time.

### `challenging/test` (12 pairs: 6 removals, 6 swaps/rearrangements)

| Seed | Model | AUC | Acc | TP | FP | FN | TN |
|---|---|---|---|---|---|---|---|
| 42 | Keras | 0.389 | 0.500 | 2 | 2 | 4 | 4 |
| 42 | ViT | 0.444 | 0.500 | 6 | 6 | 0 | 0 |
| 7 | Keras | 0.250 | 0.500 | 1 | 1 | 5 | 5 |
| 7 | ViT | 0.389 | 0.500 | 6 | 6 | 0 | 0 |
| 1 | Keras | 0.250 | 0.250 | 2 | 5 | 4 | 1 |
| 1 | ViT | 0.444 | 0.333 | 4 | 6 | 2 | 0 |
| 2 | Keras | 0.528 | 0.583 | 2 | 1 | 4 | 5 |
| 2 | ViT | 0.611 | 0.500 | 4 | 4 | 2 | 2 |
| 3 | Keras | 0.472 | 0.500 | 2 | 2 | 4 | 4 |
| 3 | ViT | 0.167 | 0.500 | 6 | 6 | 0 | 0 |
| ensemble | Keras | 0.361 | 0.500 | 2 | 2 | 4 | 4 |
| ensemble | ViT | 0.500 | 0.417 | 5 | 6 | 1 | 0 |

**Neither model solves this split.** Both sit at or below chance; they
just fail differently — Keras mostly answers "no change", the ViT
flags nearly every swap as a removal (28 false alarms on 30 no-change
decisions). The ViT's "is every before-patch found in after?" question
has no answer for "it is found, but it is a different object"; training
with `--challenge-root` is the next thing to try. With 12 pairs, one pair
moves accuracy by 8 points, so read this table as a direction only.

### Choosing the decision rule

With the per-model threshold picked at the edge of the best validation
cutoff range, the ViT's threshold jumped between seeds (0.886 on one run,
0.010 on another) because it separates validation almost perfectly and
every cutoff in the gap looks equally good. Two replacements were tried
on the same 10 checkpoints (`test` accuracy):

| Seed | Keras, threshold | Keras, Platt | ViT, threshold | ViT, Platt |
|---|---|---|---|---|
| 42 | 0.698 | 0.733 | 0.871 | **0.888** |
| 7 | **0.750** | 0.595 | 0.879 | **0.888** |
| 1 | **0.784** | 0.164 | 0.871 | **0.879** |
| 2 | **0.784** | 0.776 | 0.741 | **0.759** |
| 3 | **0.690** | 0.664 | **0.879** | 0.853 |
| ensemble | **0.767** | 0.716 | **0.905** | **0.905** |
| false alarms, 5 seeds | 35 | 26 | 18 | **12** |

Platt helps the ViT (4 of 5 seeds, a third fewer false alarms) because
its validation scores track its test scores. It hurts Keras: Keras'
validation AUC is near chance on some seeds (0.53 on seed 1), Platt then
honestly fits a near-zero slope, and that model answers "no change" to
everything — even though it ranks the test pairs well (AUC 0.85).

### Caveats

- **The decision rule per backend was chosen on these test results**, so
  the accuracy figures above are slightly optimistic for both. The AUCs
  do not depend on the rule, and they carry the conclusion.
- **Runs are not bit-reproducible**: GPU training is nondeterministic, so
  the same `--seed` gives a different model on a rerun (Keras seed 42
  scored test AUC 0.864 in an earlier run, 0.759 here). This is one more
  reason to compare seed distributions, not single runs.
- **`test` is removal-heavy** (84% of pairs) while training is 43%; any
  rule learned on training data under-calls removals there. Pass
  `--prior` (Platt only) when the deployment rate is known.
- One ViT seed (2) had a validation split with only 15 removals out of 75
  and still under-calls removals on test (28 missed) under either rule.

Reproduce:

```bash
cd siamese_missing
for seed in 42 7 1 2 3; do
  python train_missing.py train --keras --root ../somethings_missing_here/training --cache --seed $seed --prefix keras_s$seed
  python train_missing.py train --vit   --root ../somethings_missing_here/training --cache --seed $seed --prefix vit_s$seed
done
python train_missing.py evaluate --keras --root ../somethings_missing_here/test --cache --ckpt 'keras_s*.keras'
python train_missing.py evaluate --vit   --root ../somethings_missing_here/test --cache --ckpt 'vit_s*.pt'
```

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
