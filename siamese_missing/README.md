# Missing-item detection: siamese pipeline

Two photos of a scene in, one boolean out: did anything go missing?

## Files

| File | Role |
|---|---|
| `pairs_io.py` | Reads the folder tree into memory, synthesises reversed negatives, image cache, scene-grouped folds |
| `train_missing.py` | **Main entry point.** `train` / `predict` / `evaluate` |
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

Exactly two levels: `<scene>/<pair>/`. `.png` and `.jpeg` work too.

### `label.json`

```json
{"missing": true, "items": ["cup", "scissors"], "verified": true}
```

- `missing` — the only required field. `true` / `false`, or a list of labels (non-empty means true).
- `items` — optional, unused by the bool model. Record it anyway; it costs nothing now and saves re-annotating if you later want per-class output.
- `verified` — optional, set by hand. Only meaningful with `bootstrap_labels.py`.

### Rules that matter

1. **One scene = one physical location/camera setup.** Folds are split by scene, so a scene is entirely in train or entirely in val. Getting this wrong is the difference between an honest 0.85 and a fake 0.97.
2. **Both labels in every scene.** Shoot a "nothing removed" pair at each location. A scene that is all-positive lets the model learn "that room ⇒ true"; `pairs_io` warns about this.
3. **20–50 scenes, 200–500 pairs.** Scene diversity matters more than pair count.
4. **Never reverse pairs on disk.** Reversed negatives are generated in memory and stay in their source scene automatically.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

CPU-only PyTorch is fine for this dataset size (~3–5 h for a full 5-fold run;
~20–40 min on a free Colab T4).

## Commands

Check the data before training — this parses every `label.json` and warns about
single-label scenes and unlabelled pairs:

```bash
python pairs_io.py data
```

Train (scene-grouped 5-fold, saves `fold0.pt` … `fold4.pt`):

```bash
python train_missing.py train --root data --cache
```

Predict on one new pair (ensembles all fold checkpoints, hflip TTA):

```bash
python train_missing.py predict --before a.jpg --after b.jpg
```

```json
{ "any_missing": true, "prob": 0.812, "threshold": 0.463,
  "models": 5, "per_model": [0.79, 0.85, 0.74, 0.88, 0.8] }
```

Evaluate on a held-out folder (natural pairs only, no reversals):

```bash
python train_missing.py evaluate --root holdout --cache
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
| `--height/--width` | 256 | Raise to 384 if items are small in frame. Halve `--bs` if you hit OOM |
| `--backbone` | `resnet18` | `resnet34` above ~1000 pairs. `convnext_tiny` needs 12 GB+ |
| `--bs` | 16 | Lower on small GPUs; the head uses LayerNorm so small batches are safe |
| `--folds` | 5 | Needs at least this many scenes |
| `--dropout` | 0.4 | Raise to 0.5–0.6 if train/val AUC diverge |
| `--freeze-epochs` | 10 | Raise under ~250 pairs; the frozen encoder is doing most of the work |
| `--no-reverse` | off | Only if you already materialised reversals on disk |

## Reading the output

```
AUC 0.883 +/- 0.041   acc 0.831 +/- 0.052
```

The spread matters as much as the mean at this data size.

- **0.85–0.92 AUC, std < 0.05** — working as expected.
- **High mean, std > 0.10** — one lucky fold. Add scenes, not epochs.
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
- **Threshold from Youden's J** on each fold's validation split, stored in the checkpoint — not a hardcoded 0.5.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `need >= 5 scenes` | Fewer scene folders than `--folds`. Lower `--folds` or split your data by location |
| `no label.json (unlabelled?)` | Pair folder without a label. `pairs_io.py` lists them |
| Every prediction ~0.5 | Model learned nothing. Check that positives and negatives are not visually identical (wrong `before`/`after` naming) |
| `AttributeError` in albumentations | Version 2.x renamed transform arguments. `pip install "albumentations<2"` |
| CUDA OOM | Lower `--bs` before lowering resolution — resolution matters more for small items |
| Slow epochs, low GPU use | JPEG decode bound. Add `--cache`, or pre-resize your source images |
