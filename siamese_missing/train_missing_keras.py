"""
train_missing_keras.py
Keras 3 port of train_missing.py: siamese binary "did anything go missing
between these two photos?" model. Same CLI, same data layer (pairs_io), same
fold/threshold logic; EfficientNetB0 backbone instead of ResNet18 (no
ResNet18 exists in keras.applications).

    python train_missing_keras.py train    --root data
    python train_missing_keras.py predict  --before a.jpg --after b.jpg
    python train_missing_keras.py evaluate --root holdout

Checkpoints are saved as <prefix><fold>.keras with a sidecar
<prefix><fold>.json holding the threshold and metadata (a .keras file cannot
stash arbitrary extras the way torch.save could).
"""

import argparse
import glob
import json
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import keras
from keras import layers
from sklearn.metrics import roc_auc_score, roc_curve

from pairs_io import ImageCache, load_pairs, scene_folds


def seed_all(seed):
    keras.utils.set_random_seed(seed)


# --------------------------------------------------------------------------- #
# dataset
# --------------------------------------------------------------------------- #
class PairSequence(keras.utils.PyDataset):
    """
    Same augmentation contract as the torch PairSet:
      geometric   -> identical on both images (one shared transform)
      photometric -> independent per image ('lighting change != item gone')
      jitter      -> small extra warp on `after` only, for registration slack

    Yields ((a, b), y) float32 batches. EfficientNet preprocessing (scaling
    to its expected input distribution) is part of the backbone in Keras, so
    images stay 0-255 float here.
    """

    def __init__(self, rows, size=(256, 256), train=True, cache=None,
                 batch_size=16, shuffle=None, seed=0, **kwargs):
        super().__init__(**kwargs)
        self.rows = list(rows)
        self.train = train
        self.cache = cache
        self.batch_size = batch_size
        self.shuffle = train if shuffle is None else shuffle
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.rows))
        h, w = size

        shared = [
            A.LongestMaxSize(max_size=max(h, w)),
            A.PadIfNeeded(h, w, border_mode=cv2.BORDER_CONSTANT),
            A.CenterCrop(h, w),
        ]
        if train:
            shared += [
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.9, 1.1), translate_percent=0.05,
                         rotate=(-7, 7), p=0.7),
            ]
        self.shared = A.Compose(shared, additional_targets={"image2": "image"})
        self.jitter = (A.Affine(translate_percent=0.02, rotate=(-2, 2), p=0.5)
                       if train else None)
        self.photo = A.Compose([
            A.RandomBrightnessContrast(0.3, 0.3, p=0.8),
            A.HueSaturationValue(12, 20, 12, p=0.5),
            A.GaussNoise(p=0.2),
        ]) if train else None

        if self.shuffle:
            self.rng.shuffle(self.order)

    def _read(self, path):
        if self.cache is not None:
            return self.cache.get(path)
        raw = cv2.imread(path, cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"could not decode {path}")
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    def _sample(self, r):
        out = self.shared(image=self._read(r["before"]),
                          image2=self._read(r["after"]))
        a, b = out["image"], out["image2"]
        if self.jitter is not None:
            b = self.jitter(image=b)["image"]
        if self.photo is not None:
            a = self.photo(image=a)["image"]
            b = self.photo(image=b)["image"]
        return a.astype(np.float32), b.astype(np.float32), np.float32(r["y"])

    def __len__(self):
        return (len(self.rows) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, index):
        idx = self.order[index * self.batch_size:(index + 1) * self.batch_size]
        samples = [self._sample(self.rows[i]) for i in idx]
        a = np.stack([s[0] for s in samples])
        b = np.stack([s[1] for s in samples])
        y = np.array([s[2] for s in samples], dtype=np.float32)
        return (a, b), y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def build_model(size=(256, 256), pretrained=True, p_drop=0.4):
    h, w = size
    # ONE backbone applied to both inputs == shared weights (the siamese part).
    # EfficientNet includes its input rescaling/normalization layers, so the
    # model takes raw 0-255 images.
    backbone = keras.applications.EfficientNetB0(
        include_top=False, pooling="avg",
        weights="imagenet" if pretrained else None,
        input_shape=(h, w, 3),
    )
    inp_a = keras.Input((h, w, 3), name="before")
    inp_b = keras.Input((h, w, 3), name="after")
    fa, fb = backbone(inp_a), backbone(inp_b)
    # SIGNED difference: the sign separates 'removed' from 'added'
    x = layers.Concatenate()([fa, fb, layers.Subtract()([fa, fb])])
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128)(x)
    # LayerNorm not BatchNorm: batches are small, so batch statistics are
    # noisy and batch-size dependent.
    x = layers.LayerNormalization()(x)
    x = layers.Activation("gelu")(x)
    x = layers.Dropout(p_drop)(x)
    out = layers.Dense(1, name="logit")(x)
    model = keras.Model([inp_a, inp_b], out)
    model.backbone = backbone
    return model


def compile_model(model, lr):
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=lr, weight_decay=1e-4),
        loss=keras.losses.BinaryCrossentropy(from_logits=True),
    )


# --------------------------------------------------------------------------- #
# evaluation helpers
# --------------------------------------------------------------------------- #
def probs_and_labels(model, seq):
    ps, ys = [], []
    for i in range(len(seq)):
        (a, b), y = seq[i]
        logits = model.predict_on_batch((a, b)).squeeze(-1)
        ps.append(1.0 / (1.0 + np.exp(-logits)))
        ys.append(y)
    return np.concatenate(ps), np.concatenate(ys)


def best_threshold(y, p):
    """Youden's J: maximise TPR - FPR. Beats a fixed 0.5 on imbalanced data."""
    if len(set(y)) < 2:
        return 0.5
    fpr, tpr, thr = roc_curve(y, p)
    return float(np.clip(thr[np.argmax(tpr - fpr)], 0.01, 0.99))


def confusion(y, p, thr):
    pred = (p > thr).astype(float)
    return (int(((pred == 1) & (y == 1)).sum()), int(((pred == 1) & (y == 0)).sum()),
            int(((pred == 0) & (y == 1)).sum()), int(((pred == 0) & (y == 0)).sum()))


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def train_fold(tr_rows, va_rows, args, fold, cache):
    seed_all(args.seed + fold)
    size = (args.height, args.width)

    tr = PairSequence(tr_rows, size, True, cache, args.bs, seed=args.seed + fold)
    va = PairSequence(va_rows, size, False, cache, args.bs, shuffle=False)

    model = build_model(size, p_drop=args.dropout)
    model.backbone.trainable = False   # frozen BN stays in inference mode
    compile_model(model, lr=1e-3)

    best = {"auc": -1.0, "thr": 0.5, "weights": None, "epoch": -1}
    patience = 0

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:      # unfreeze, low LR on everything
            model.backbone.trainable = True
            compile_model(model, lr=1e-4)

        model.fit(tr, epochs=1, verbose=0)

        p, y = probs_and_labels(model, va)
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")

        if auc > best["auc"]:
            best = {"auc": float(auc), "thr": best_threshold(y, p),
                    "epoch": epoch, "weights": model.get_weights()}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    model.set_weights(best["weights"])
    p, y = probs_and_labels(model, va)
    tp, fp, fn, tn = confusion(y, p, best["thr"])
    acc = (tp + tn) / max(len(y), 1)

    path = f"{args.prefix}{fold}.keras"
    model.save(path)
    Path(f"{args.prefix}{fold}.json").write_text(json.dumps({
        "size": list(size), "threshold": best["thr"], "val_auc": best["auc"],
        "backbone": "efficientnetb0", "dropout": args.dropout,
    }))
    return {"auc": best["auc"], "acc": acc, "epoch": best["epoch"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "path": path,
            "n_val": len(y)}


def cmd_train(args):
    rows = load_pairs(args.root, reverse_positives=not args.no_reverse,
                      identity_negatives=args.identity_negatives)
    cache = ImageCache(args.cache_side).warm(rows) if args.cache else None
    print()

    results = []
    for k, (tr_rows, va_rows) in enumerate(scene_folds(rows, args.folds, args.seed)):
        # identity negatives are train-only: without augmentation they are
        # pixel-identical and would flatter validation
        va_rows = [r for r in va_rows if not r.get("identity", False)]
        r = train_fold(tr_rows, va_rows, args, k, cache)
        results.append(r)
        print(f"fold {k}: {r['n_val']:3d} val pairs | AUC {r['auc']:.3f} | "
              f"acc {r['acc']:.3f} | TP {r['tp']} FP {r['fp']} FN {r['fn']} "
              f"TN {r['tn']} | best epoch {r['epoch']}")

    aucs = np.array([r["auc"] for r in results])
    accs = np.array([r["acc"] for r in results])
    print(f"\nAUC {aucs.mean():.3f} +/- {aucs.std():.3f}"
          f"   acc {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"checkpoints: {args.prefix}0.keras ... {args.prefix}{args.folds - 1}.keras")
    print("read the spread as well as the mean -- at this data size one lucky "
          "fold can carry the average")


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #
def load_ensemble(ckpt_glob):
    paths = sorted(glob.glob(ckpt_glob))
    if not paths:
        raise SystemExit(f"no checkpoints matched {ckpt_glob!r}")
    models = []
    for cp in paths:
        meta = json.loads(Path(cp).with_suffix(".json").read_text())
        models.append((keras.models.load_model(cp),
                       tuple(meta["size"]), meta["threshold"]))
    return models


def pair_prob(model, before, after, size, cache=None):
    seq = PairSequence([{"before": before, "after": after, "y": 0.0}],
                       size, train=False, cache=cache, batch_size=1,
                       shuffle=False)
    (a, b), _ = seq[0]
    # hflip TTA: cheap, reduces variance on a small model
    logits = [model.predict_on_batch((a, b)).squeeze(),
              model.predict_on_batch((np.flip(a, 2).copy(),
                                      np.flip(b, 2).copy())).squeeze()]
    return float(1.0 / (1.0 + np.exp(-np.mean(logits))))


def cmd_predict(args):
    models = load_ensemble(args.ckpt)
    probs = [pair_prob(m, args.before, args.after, size)
             for m, size, _ in models]
    thr = float(np.mean([t for _, _, t in models]))
    p = float(np.mean(probs))
    print(json.dumps({
        "any_missing": bool(p > thr),
        "prob": round(p, 3),
        "threshold": round(thr, 3),
        "models": len(models),
        "per_model": [round(x, 3) for x in probs],
    }, indent=2))


def cmd_evaluate(args):
    models = load_ensemble(args.ckpt)
    # natural pairs only: reversals would flatter the score
    rows = load_pairs(args.root, reverse_positives=False)
    cache = ImageCache(args.cache_side).warm(rows) if args.cache else None
    print()

    y = np.array([r["y"] for r in rows])
    p = np.array([
        float(np.mean([pair_prob(m, r["before"], r["after"], size, cache)
                       for m, size, _ in models]))
        for r in rows
    ])
    thr = float(np.mean([t for _, _, t in models]))
    tp, fp, fn, tn = confusion(y, p, thr)
    auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")

    print(f"{len(rows)} pairs | threshold {thr:.3f}")
    print(f"AUC {auc:.3f}   acc {(tp + tn) / len(rows):.3f}")
    print(f"TP {tp}  FP {fp}  FN {fn}  TN {tn}")
    if tp + fp:
        print(f"precision {tp / (tp + fp):.3f}", end="   ")
    if tp + fn:
        print(f"recall {tp / (tp + fn):.3f}")

    print("\nworst mistakes:")
    for i in np.argsort(-np.abs(p - y))[:10]:
        r = rows[i]
        print(f"  p={p[i]:.3f} y={y[i]:.0f}  {r['scene']}/{r['pair']}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--root", default="data")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--freeze-epochs", type=int, default=10)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--dropout", type=float, default=0.4)
    t.add_argument("--height", type=int, default=256)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--prefix", default="kfold")
    t.add_argument("--cache", action="store_true", help="decode images once into RAM")
    t.add_argument("--cache-side", type=int, default=640)
    t.add_argument("--no-reverse", action="store_true",
                   help="skip synthesised reversed negatives")
    t.add_argument("--identity-negatives", action="store_true",
                   help="add same-photo no-change negatives (for datasets "
                        "with no natural negative pairs)")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--ckpt", default="kfold*.keras")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("evaluate")
    e.add_argument("--root", required=True, help="held-out folder, same layout")
    e.add_argument("--ckpt", default="kfold*.keras")
    e.add_argument("--cache", action="store_true")
    e.add_argument("--cache-side", type=int, default=640)
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
