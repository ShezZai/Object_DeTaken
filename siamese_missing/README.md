# Missing-item detection: siamese pipeline

Two photos of a scene in, one boolean out: did anything go missing?

## Files

| File | Role |
|---|---|
| `pairs_io.py` | Reads the folder tree into memory, synthesises reversed negatives, image cache, scene-grouped train/val split |
| `train_missing.py` | **Main entry point.** `train` / `predict` / `evaluate`; `--keras` (EfficientNetB0, default, saves `kmodel.keras` + `.json` sidecar) , `--torch` (ResNet18, saves `model.pt`), or `--vit` (DINOv2 ViT, saves `vmodel.pt`) |
| `Vit_siamese.py` | The `--vit` model: frozen DINOv2 patch tokens, before→after cross-attention, per-patch "gone" scores |
| `keras_flow.ipynb` | Self-contained notebook of the Keras flow — installs, downloads the dataset, trains, evaluates, visualizes; runs top to bottom on its own |
| `missing_items.py` | Detector-diff baseline (YOLO). Standalone, no training |
| `bootstrap_labels.py` | Optional: pre-label pairs with the detector so you hand-correct instead of annotating from scratch |
| `requirements.txt` | Dependencies |

Superseded, kept only for reference: `siamese_missing.py` (multi-label version),
`siamese_missing_bool.py`, `build_manifest.py`, `augment_reversed.py`.
The current pipeline needs none of them — no manifest file, no generated folders.

## Data hierarchy

```
data/
├── scene_01/
│   ├── pair_001/
│   │   ├── before.jpg
│   │   ├── after.jpg
│   │   └── label.json
│   ├── pair_002/
│   │   ├── before.jpg
│   │   ├── after.jpg
│   │   └── label.json
│   └── pair_003/ ...
├── scene_02/
│   └── pair_001/ ...
└── scene_47/ ...

holdout/            # same layout, scenes NEVER used in training
└── scene_90/
    └── pair_001/ ...
```

Pair folders are found at ANY depth under the root; a pair's parent path
relative to the root is its scene id. So both `<scene>/<pair>/` and
collection layouts like `training/DeTaken/<scene>/<pair>/` work — scenes
from different collections stay distinct (`DeTaken/boxes` vs
`Remove360_based/backyard`). `.png` and `.jpeg` work too.

### `label.json`

```json
{"missing": true, "items": ["cup", "scissors"], "verified": true}
```

- `missing` — the only required field. `true` / `false`, or a list of labels (non-empty means true).
- `items` — optional, unused by the bool model. Record it anyway; it costs nothing now and saves re-annotating if you later want per-class output.
- `verified` — optional, set by hand. Only meaningful with `bootstrap_labels.py`.

### Rules that matter

1. **One scene = one physical location/camera setup.** The train/validation split is by scene, so a scene is entirely in train or entirely in val. Getting this wrong is the difference between an honest 0.85 and a fake 0.97.
2. **Both labels in every scene.** Shoot a "nothing removed" pair at each location. A scene that is all-positive lets the model learn "that room ⇒ true"; `pairs_io` warns about this.
3. **20–50 scenes, 200–500 pairs.** Scene diversity matters more than pair count.
4. **Never reverse pairs on disk.** Reversed negatives are generated in memory and stay in their source scene automatically.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only PyTorch is fine for this dataset size (~30–60 min for a full run;
a few minutes on a free Colab T4).

## Commands

Check the data before training — this parses every `label.json` and warns about
single-label scenes and unlabelled pairs:

```bash
python pairs_io.py data
```

Train (whole scenes held out for validation — see `--val-fraction` — saves
`kmodel.keras` + `kmodel.json`):

```bash
python train_missing.py train --root data --cache
# e.g. on the somethings_missing_here dataset (both collections):
python train_missing.py train --root ../somethings_missing_here/training --cache --identity-negatives
```

Predict on one new pair (hflip TTA with `--flip`; every checkpoint matching
`--ckpt` is averaged — logits and thresholds both, in logit space — so
several runs, e.g. different seeds, form an ensemble):

```bash
python train_missing.py predict --before a.jpg --after b.jpg
```

```json
{ "any_missing": true, "prob": 0.812, "threshold": 0.463,
  "models": 1, "per_model": [0.812] }
```

Evaluate on a held-out folder (natural pairs only, no reversals):

```bash
python train_missing.py evaluate --root holdout --cache
# e.g. the somethings_missing_here held-out split:
python train_missing.py evaluate --root ../somethings_missing_here/test --cache
```

The default implementation is Keras 3 + EfficientNetB0 — it gave the best
results (keras.applications has no ResNet18; sets `KERAS_BACKEND=torch`
behind the scenes so the GPU works wherever torch does — export
`KERAS_BACKEND` yourself to override). The PyTorch + ResNet18
implementation is one flag away and saves `model.pt`:

```bash
python train_missing.py train --torch --root data --cache
python train_missing.py predict --torch --before a.jpg --after b.jpg
```

`--vit` (torch-only, saves `vmodel.pt`) swaps the pooled-feature CNN for
`Vit_siamese.py`: a frozen DINOv2 ViT-S/14 keeps its patch tokens, every
*before* patch looks for itself among the *after* patches via
cross-attention, and a small transformer scores each before-patch as
"gone". The pair logit is a smooth max over those scores, so a single
unexplained patch is enough — the point for small objects that global
pooling dilutes. Defaults change with it: backbone
`vit_small_patch14_dinov2.lvd142m`, 224×224 (use multiples of 14), and the
encoder is never unfrozen unless you pass `--freeze-epochs`. With
`evaluate --viz`, the panels show the model's own trained patch scores
instead of Grad-CAM (absolute scale: a dark map means no patch was
confident).

```bash
python train_missing.py train --vit --root data --cache
python train_missing.py evaluate --vit --root holdout --cache --viz viz
```

Any invocation can live in a JSON config instead of flags — keys mirror the
flag names (dashes or underscores), an optional `"cmd"` key picks the
subcommand, and explicit CLI flags override config values:

```bash
python train_missing.py --config run.json
python train_missing.py train --config run.json --bs 8   # config + override
```

```json
{
  "cmd": "train",
  "keras": true,
  "root": "../somethings_missing_here/training",
  "cache": true,
  "identity-negatives": true,
  "height": 384, "width": 384, "bs": 8,
  "prefix": "kmodel384"
}
```

Optional detector baseline and label bootstrap:

```bash
python missing_items.py before.jpg after.jpg --save comparison.jpg
python bootstrap_labels.py data
```

## Useful flags

| Flag | Default | When to change it |
|---|---|---|
| `--cache` | off | Turn it on. Decodes each image once into RAM; often 3–5× faster since JPEG decode is the real bottleneck |
| `--cache-side` | 640 | Lower to 384 if RAM is tight (~1.2 GB at 640 for 500 pairs) |
| `--height/--width` | 256 (224 with `--vit`) | Raise to 384 if items are small in frame. Halve `--bs` if you hit OOM |
| `--backbone` | `resnet18` | `resnet34` above ~1000 pairs. `convnext_tiny` needs 12 GB+. With `--vit`: `vit_small_patch14_dinov2.lvd142m`; `vit_base_patch14_dinov2.lvd142m` is the bigger option |
| `--bs` | 16 | Lower on small GPUs; the head uses LayerNorm so small batches are safe |
| `--val-fraction` | 0.2 | Fraction of rows (whole scenes) held out for validation |
| `--dropout` | 0.4 | Raise to 0.5–0.6 if train/val AUC diverge |
| `--freeze-epochs` | 10 (never with `--vit`) | Raise under ~250 pairs; the frozen encoder is doing most of the work |
| `--no-reverse` | off | Only if you already materialised reversals on disk |

## Reading the output

```
 68 val pairs | AUC 0.883 | acc 0.831 | TP 24 FP 5 FN 6 TN 33 | best epoch 12
```

Validation is one scene-held-out subset, so the score moves with which
scenes landed in it — rerun with a few `--seed` values before trusting a
comparison.

- **0.85–0.92 AUC** — working as expected.
- **Big swings across seeds** — validation scenes dominate the score. Add scenes, not epochs.
- **Below 0.75** — usually items too small in frame (try `--height 384 --width 384`) or too little scene diversity.
- **Val AUC ≫ its own held-out score** — a leak. Check that no location appears under two scene names.

Compare against `missing_items.py` on the *same* held-out scenes. If the
detector diff wins, that is a real result: it needs no training data and no GPU.
The siamese model earns its place when "missing" depends on context the
detector cannot name — an object absent from its designated slot, or items
outside COCO's 80 classes.

## Design notes

- **Shared encoder, called twice** — that is the siamese part. One set of weights.
- **Signed difference** `[fa, fb, fa - fb]`, not `abs()`. Abs is symmetric and cannot tell removed from added, which is the whole distinction.
- **Reversed positives as negatives.** A removal read backwards is an addition. Doubles the data, balances the classes, and forces the model to use that sign.
- **Geometric augmentation shared, photometric independent.** Independent brightness/colour jitter is what teaches "the light changed" ≠ "the item is gone".
- **Frozen encoder stays in `eval()`** so ImageNet BatchNorm running statistics survive fine-tuning on a few hundred images.
- **Threshold from Youden's J** on the validation split, stored in the checkpoint — not a hardcoded 0.5. It is the middle (in logit space) of the range of cutoffs that all reach the best J, not the edge of it: when validation separates cleanly, the edge is an arbitrary validation score and does not transfer.
- **Platt scaling** is stored alongside it (a logistic fit of the logit on validation with Platt's smoothed targets, moved to the training data's base rate). `--calibration platt|threshold` picks the rule at `predict`/`evaluate`; the default is `platt` with `--vit` and `threshold` otherwise — over 5 seeds Platt helped the ViT and hurt Keras, whose validation AUC is near chance on some seeds (see the top-level README). `--prior 0.8` moves the Platt decision to a known deployment rate of removals. Checkpoints trained before this can be calibrated without retraining: `python train_missing.py calibrate --vit --root data --seed 42 --ckpt vmodel.pt` (newer checkpoints store their seed and split settings, so `--seed` is not needed).

## Troubleshooting

| Symptom | Cause |
|---|---|
| `need >= 2 scenes` | Validation holds out whole scenes, so at least two scene folders are required. Split your data by location |
| `no label.json (unlabelled?)` | Pair folder without a label. `pairs_io.py` lists them |
| Every prediction ~0.5 | Model learned nothing. Check that positives and negatives are not visually identical (wrong `before`/`after` naming) |
| `AttributeError` in albumentations | Version 2.x renamed transform arguments. `pip install "albumentations<2"` |
| CUDA OOM | Lower `--bs` before lowering resolution — resolution matters more for small items |
| Slow epochs, low GPU use | JPEG decode bound. Add `--cache`, or pre-resize your source images |
