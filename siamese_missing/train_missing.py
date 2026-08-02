"""
train_missing.py
Siamese binary "did anything go missing between these two photos?" model,
tuned for a small dataset (200-500 pairs across 20-50 scenes).

    python train_missing.py train    --root data
    python train_missing.py predict  --before a.jpg --after b.jpg
    python train_missing.py evaluate --root holdout

Two implementations behind one CLI, chosen per subcommand:
    --torch (default)  ResNet18 encoder, saves fold<k>.pt
    --keras            Keras 3 + EfficientNetB0, saves kfold<k>.keras plus a
                       kfold<k>.json sidecar (threshold/metadata). Sets
                       KERAS_BACKEND=torch behind the scenes so the GPU works
                       wherever torch does; export KERAS_BACKEND yourself to
                       override (e.g. tensorflow on a supported GPU).

Reads the folder tree directly via pairs_io -- no manifest file, no generated
folders. Reversed negatives are synthesised in memory.
"""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import timm
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from pairs_io import ImageCache, load_pairs, scene_folds

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def to_tensor(img):
    x = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
    return (x - MEAN) / STD


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# augmentation (shared by both implementations)
# --------------------------------------------------------------------------- #
def build_transforms(size, train):
    """
    geometric   -> identical on both images (one shared transform)
    photometric -> independent per image (teaches 'lighting change != item gone')
    jitter      -> small extra warp on `after` only, for registration slack
    """
    h, w = size
    shared = [
        A.LongestMaxSize(max_size=max(h, w)),
        A.PadIfNeeded(h, w, border_mode=cv2.BORDER_CONSTANT),
        A.CenterCrop(h, w),          # guarantees an exact, collatable shape
    ]
    if train:
        shared += [
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.9, 1.1), translate_percent=0.05,
                     rotate=(-7, 7), p=0.7),
        ]
    shared = A.Compose(shared, additional_targets={"image2": "image"})
    jitter = A.Affine(translate_percent=0.02, rotate=(-2, 2), p=0.5) if train else None
    photo = A.Compose([
        A.RandomBrightnessContrast(0.3, 0.3, p=0.8),
        A.HueSaturationValue(12, 20, 12, p=0.5),
        A.GaussNoise(p=0.2),
    ]) if train else None
    return shared, jitter, photo


def read_image(path, cache=None):
    if cache is not None:
        return cache.get(path)
    raw = cv2.imread(path, cv2.IMREAD_COLOR)
    if raw is None:
        raise FileNotFoundError(f"could not decode {path}")
    return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)


# --------------------------------------------------------------------------- #
# dataset (torch)
# --------------------------------------------------------------------------- #
class PairSet(Dataset):
    def __init__(self, rows, size=(256, 256), train=True, cache=None):
        self.rows, self.train, self.cache = rows, train, cache
        self.shared, self.jitter, self.photo = build_transforms(size, train)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        out = self.shared(image=read_image(r["before"], self.cache),
                          image2=read_image(r["after"], self.cache))
        a, b = out["image"], out["image2"]
        if self.jitter is not None:
            b = self.jitter(image=b)["image"]
        if self.photo is not None:
            a = self.photo(image=a)["image"]
            b = self.photo(image=b)["image"]
        return to_tensor(a), to_tensor(b), torch.tensor(float(r["y"]))


# --------------------------------------------------------------------------- #
# model (torch)
# --------------------------------------------------------------------------- #
class SiameseBool(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, p_drop=0.4):
        super().__init__()
        # ONE encoder called twice == shared weights. That is the siamese part.
        # Not two encoders: (1) the head subtracts features, which only means
        # anything if both images are embedded in the same feature space --
        # independent encoders drift apart and make fa - fb arbitrary;
        # (2) a second encoder would double the parameters on a tiny dataset;
        # (3) before/after are the same kind of image, so there is nothing for
        # a second encoder to specialise on.
        self.enc = timm.create_model(backbone, pretrained=pretrained, num_classes=0)
        d = self.enc.num_features
        # LayerNorm not BatchNorm: batches are small, so batch statistics are
        # noisy and batch-size dependent.
        self.head = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(3 * d, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Dropout(p_drop),
            nn.Linear(128, 1),
        )
        self.enc_frozen = False

    def forward(self, a, b):
        fa, fb = self.enc(a), self.enc(b)
        # SIGNED difference: the sign separates 'removed' from 'added'.
        # abs() would be symmetric and could not tell the two apart, which is
        # exactly the distinction the reversed-pair negatives train.
        return self.head(torch.cat([fa, fb, fa - fb], dim=1)).squeeze(1)

    def freeze_encoder(self, flag=True):
        for p in self.enc.parameters():
            p.requires_grad = not flag
        self.enc_frozen = flag

    def train_mode(self):
        """Frozen encoder stays in eval() so its ImageNet BatchNorm running
        stats are not overwritten by a few hundred new-domain images."""
        self.train()
        if self.enc_frozen:
            self.enc.eval()


# --------------------------------------------------------------------------- #
# evaluation helpers (shared)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def probs_and_labels(model, loader, device):
    model.eval()
    ps, ys = [], []
    for a, b, y in loader:
        ps.append(torch.sigmoid(model(a.to(device), b.to(device))).cpu().numpy())
        ys.append(y.numpy())
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
# training (torch)
# --------------------------------------------------------------------------- #
def train_fold(tr_rows, va_rows, args, device, fold, cache):
    seed_all(args.seed + fold)
    size = (args.height, args.width)
    workers = 0 if cache is not None else args.workers

    tr = DataLoader(PairSet(tr_rows, size, True, cache), batch_size=args.bs,
                    shuffle=True, num_workers=workers)
    va = DataLoader(PairSet(va_rows, size, False, cache), batch_size=args.bs,
                    num_workers=workers)

    model = SiameseBool(args.backbone, p_drop=args.dropout).to(device)
    model.freeze_encoder(True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()

    best = {"auc": -1.0, "thr": 0.5, "state": None, "epoch": -1}
    patience = 0

    for epoch in range(args.epochs):
        if epoch == args.freeze_epochs:          # unfreeze, low LR on the backbone
            model.freeze_encoder(False)
            opt = torch.optim.AdamW([
                {"params": model.enc.parameters(), "lr": 1e-4},
                {"params": model.head.parameters(), "lr": 5e-4},
            ], weight_decay=1e-4)

        model.train_mode()
        for a, b, y in tr:
            loss = crit(model(a.to(device), b.to(device)), y.to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()

        p, y = probs_and_labels(model, va, device)
        auc = roc_auc_score(y, p) if len(set(y)) > 1 else float("nan")

        if auc > best["auc"]:
            best = {"auc": float(auc), "thr": best_threshold(y, p), "epoch": epoch,
                    "state": {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}}
            patience = 0
        else:
            patience += 1
            if patience >= args.patience:
                break

    model.load_state_dict(best["state"])
    p, y = probs_and_labels(model, va, device)
    tp, fp, fn, tn = confusion(y, p, best["thr"])
    acc = (tp + tn) / max(len(y), 1)

    path = f"{args.prefix}{fold}.pt"
    torch.save({"model": best["state"], "backbone": args.backbone, "size": size,
                "threshold": best["thr"], "val_auc": best["auc"],
                "dropout": args.dropout}, path)
    return {"auc": best["auc"], "acc": acc, "epoch": best["epoch"],
            "tp": tp, "fp": fp, "fn": fn, "tn": tn, "path": path,
            "n_val": len(y)}


# --------------------------------------------------------------------------- #
# keras implementation (lazy: only imported when --keras is used)
# --------------------------------------------------------------------------- #
_KERAS_IMPL = None


def keras_impl():
    """Import keras on first use, defaulting to the torch backend so the GPU
    works wherever torch does. Export KERAS_BACKEND yourself to override."""
    global _KERAS_IMPL
    if _KERAS_IMPL is not None:
        return _KERAS_IMPL

    os.environ.setdefault("KERAS_BACKEND", "torch")
    import keras
    from keras import layers

    class PairSequence(keras.utils.PyDataset):
        """Same augmentation contract as PairSet; yields ((a, b), y) float32
        batches. EfficientNet embeds its own input rescaling, so images stay
        0-255 float here."""

        def __init__(self, rows, size=(256, 256), train=True, cache=None,
                     batch_size=16, shuffle=None, seed=0, **kwargs):
            super().__init__(**kwargs)
            self.rows = list(rows)
            self.cache = cache
            self.batch_size = batch_size
            self.shuffle = train if shuffle is None else shuffle
            self.rng = np.random.default_rng(seed)
            self.order = np.arange(len(self.rows))
            self.shared, self.jitter, self.photo = build_transforms(size, train)
            if self.shuffle:
                self.rng.shuffle(self.order)

        def _sample(self, r):
            out = self.shared(image=read_image(r["before"], self.cache),
                              image2=read_image(r["after"], self.cache))
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

    def build_model(size=(256, 256), pretrained=True, p_drop=0.4):
        h, w = size
        # ONE backbone applied to both inputs == shared weights (the siamese
        # part). In the functional API, calling the same layer instance twice
        # creates two graph nodes over one set of parameters; summary() shows
        # two branches but count_params() counts the backbone once. Not two
        # encoders: (1) the head subtracts features, which only means anything
        # if both images are embedded in the same feature space -- independent
        # encoders drift apart and make fa - fb arbitrary; (2) a second
        # encoder would double the parameters on a tiny dataset; (3)
        # before/after are the same kind of image, so there is nothing for a
        # second encoder to specialise on.
        backbone = keras.applications.EfficientNetB0(
            include_top=False, pooling="avg",
            weights="imagenet" if pretrained else None,
            input_shape=(h, w, 3),
        )
        inp_a = keras.Input((h, w, 3), name="before")
        inp_b = keras.Input((h, w, 3), name="after")
        fa, fb = backbone(inp_a), backbone(inp_b)
        # SIGNED difference: the sign separates 'removed' from 'added'.
        # abs() would be symmetric and could not tell the two apart, which is
        # exactly the distinction the reversed-pair negatives train.
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

    def k_probs_and_labels(model, seq):
        ps, ys = [], []
        for i in range(len(seq)):
            (a, b), y = seq[i]
            logits = model.predict_on_batch((a, b)).squeeze(-1)
            ps.append(1.0 / (1.0 + np.exp(-logits)))
            ys.append(y)
        return np.concatenate(ps), np.concatenate(ys)

    def k_train_fold(tr_rows, va_rows, args, fold, cache):
        keras.utils.set_random_seed(args.seed + fold)
        size = (args.height, args.width)

        tr = PairSequence(tr_rows, size, True, cache, args.bs,
                          seed=args.seed + fold)
        va = PairSequence(va_rows, size, False, cache, args.bs, shuffle=False)

        model = build_model(size, p_drop=args.dropout)
        model.backbone.trainable = False   # frozen BN stays in inference mode
        compile_model(model, lr=1e-3)

        best = {"auc": -1.0, "thr": 0.5, "weights": None, "epoch": -1}
        patience = 0

        for epoch in range(args.epochs):
            if epoch == args.freeze_epochs:  # unfreeze, low LR on everything
                model.backbone.trainable = True
                compile_model(model, lr=1e-4)

            model.fit(tr, epochs=1, verbose=0)

            p, y = k_probs_and_labels(model, va)
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
        p, y = k_probs_and_labels(model, va)
        tp, fp, fn, tn = confusion(y, p, best["thr"])
        acc = (tp + tn) / max(len(y), 1)

        path = f"{args.prefix}{fold}.keras"
        model.save(path)
        Path(f"{args.prefix}{fold}.json").write_text(json.dumps({
            "size": [args.height, args.width], "threshold": best["thr"],
            "val_auc": best["auc"], "backbone": "efficientnetb0",
            "dropout": args.dropout,
        }))
        return {"auc": best["auc"], "acc": acc, "epoch": best["epoch"],
                "tp": tp, "fp": fp, "fn": fn, "tn": tn, "path": path,
                "n_val": len(y)}

    def k_load_ensemble(ckpt_glob):
        paths = sorted(glob.glob(ckpt_glob))
        if not paths:
            raise SystemExit(f"no checkpoints matched {ckpt_glob!r}")
        models = []
        for cp in paths:
            meta = json.loads(Path(cp).with_suffix(".json").read_text())
            models.append((keras.models.load_model(cp),
                           tuple(meta["size"]), meta["threshold"]))
        return models

    def k_pair_prob(model, before, after, size, cache=None):
        seq = PairSequence([{"before": before, "after": after, "y": 0.0}],
                           size, train=False, cache=cache, batch_size=1,
                           shuffle=False)
        (a, b), _ = seq[0]
        # hflip TTA: cheap, reduces variance on a small model
        logits = [model.predict_on_batch((a, b)).squeeze(),
                  model.predict_on_batch((np.flip(a, 2).copy(),
                                          np.flip(b, 2).copy())).squeeze()]
        return float(1.0 / (1.0 + np.exp(-np.mean(logits))))

    from types import SimpleNamespace
    _KERAS_IMPL = SimpleNamespace(
        train_fold=k_train_fold, load_ensemble=k_load_ensemble,
        pair_prob=k_pair_prob,
    )
    return _KERAS_IMPL


# --------------------------------------------------------------------------- #
# commands (dispatch on --keras / --torch)
# --------------------------------------------------------------------------- #
def resolve_defaults(args):
    if getattr(args, "prefix", None) is None:
        args.prefix = "kfold" if args.keras else "fold"
    if getattr(args, "ckpt", None) is None:
        args.ckpt = "kfold*.keras" if args.keras else "fold*.pt"


def cmd_train(args):
    resolve_defaults(args)
    impl = keras_impl() if args.keras else None
    device = None
    if not args.keras:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"device: {device}\n")

    rows = load_pairs(args.root, reverse_positives=not args.no_reverse,
                      identity_negatives=args.identity_negatives)
    cache = ImageCache(args.cache_side).warm(rows) if args.cache else None
    print()

    results = []
    for k, (tr_rows, va_rows) in enumerate(scene_folds(rows, args.folds, args.seed)):
        # identity negatives are train-only: without augmentation they are
        # pixel-identical and would flatter validation
        va_rows = [r for r in va_rows if not r.get("identity", False)]
        if args.keras:
            r = impl.train_fold(tr_rows, va_rows, args, k, cache)
        else:
            r = train_fold(tr_rows, va_rows, args, device, k, cache)
        results.append(r)
        print(f"fold {k}: {r['n_val']:3d} val pairs | AUC {r['auc']:.3f} | "
              f"acc {r['acc']:.3f} | TP {r['tp']} FP {r['fp']} FN {r['fn']} "
              f"TN {r['tn']} | best epoch {r['epoch']}")

    aucs = np.array([r["auc"] for r in results])
    accs = np.array([r["acc"] for r in results])
    suffix = "keras" if args.keras else "pt"
    print(f"\nAUC {aucs.mean():.3f} +/- {aucs.std():.3f}"
          f"   acc {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"checkpoints: {args.prefix}0.{suffix} ... "
          f"{args.prefix}{args.folds - 1}.{suffix}")
    print("read the spread as well as the mean -- at this data size one lucky "
          "fold can carry the average")


# --------------------------------------------------------------------------- #
# inference (torch)
# --------------------------------------------------------------------------- #
def load_ensemble(ckpt_glob, device):
    paths = sorted(glob.glob(ckpt_glob))
    if not paths:
        raise SystemExit(f"no checkpoints matched {ckpt_glob!r}")
    models = []
    for cp in paths:
        ck = torch.load(cp, map_location=device, weights_only=False)
        m = SiameseBool(ck["backbone"], pretrained=False,
                        p_drop=ck.get("dropout", 0.4)).to(device)
        m.load_state_dict(ck["model"])
        m.eval()
        models.append((m, ck["size"], ck["threshold"]))
    return models


@torch.no_grad()
def pair_prob(model, before, after, size, device, cache=None):
    ds = PairSet([{"before": before, "after": after, "y": 0.0}],
                 size, train=False, cache=cache)
    a, b, _ = ds[0]
    a, b = a[None].to(device), b[None].to(device)
    # hflip TTA: cheap, reduces variance on a small model
    logits = torch.stack([model(a, b),
                          model(torch.flip(a, [-1]), torch.flip(b, [-1]))])
    return float(torch.sigmoid(logits.mean()).item())


def make_pair_scorer(args, cache=None):
    """Return (models, score(model, size, before, after)) for either backend."""
    if args.keras:
        impl = keras_impl()
        models = impl.load_ensemble(args.ckpt)
        return models, lambda m, size, bf, af: impl.pair_prob(m, bf, af, size, cache)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble(args.ckpt, device)
    return models, lambda m, size, bf, af: pair_prob(m, bf, af, size, device, cache)


def cmd_predict(args):
    if not args.before or not args.after:
        raise SystemExit("error: --before and --after are required "
                         "(as flags or config keys)")
    resolve_defaults(args)
    models, score = make_pair_scorer(args)
    probs = [score(m, size, args.before, args.after) for m, size, _ in models]
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
    if not args.root:
        raise SystemExit("error: --root is required (as a flag or config key)")
    resolve_defaults(args)
    # natural pairs only: reversals would flatter the score
    rows = load_pairs(args.root, reverse_positives=False)
    cache = ImageCache(args.cache_side).warm(rows) if args.cache else None
    models, score = make_pair_scorer(args, cache)
    print()

    y = np.array([r["y"] for r in rows])
    p = np.array([
        float(np.mean([score(m, size, r["before"], r["after"])
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

    failures = np.where((p > thr) != (y == 1))[0]
    print(f"\nfailed pairs ({len(failures)}/{len(rows)}):")
    for i in sorted(failures, key=lambda i: -abs(p[i] - y[i])):
        r = rows[i]
        kind = "missed" if y[i] == 1 else "false alarm"
        items = ", ".join(r["items"]) or "-"
        print(f"  {kind:11s} p={p[i]:.3f} (thr {thr:.3f})  "
              f"{r['scene']}/{r['pair']}  items: {items}")


# --------------------------------------------------------------------------- #
def load_config(path):
    """Read a JSON config whose keys mirror the CLI flags.

    Keys may use dashes or underscores ("freeze-epochs" == "freeze_epochs").
    An optional "cmd" key selects the subcommand so the whole invocation can
    live in the file:  python train_missing.py --config run.json
    Explicit CLI flags always override config values.
    """
    cfg = json.loads(Path(path).read_text())
    if not isinstance(cfg, dict):
        raise SystemExit(f"error: {path} must contain a JSON object")
    return {k.replace("-", "_"): v for k, v in cfg.items()}


def main():
    backend = argparse.ArgumentParser(add_help=False)
    group = backend.add_mutually_exclusive_group()
    group.add_argument("--keras", action="store_true",
                       help="Keras implementation (EfficientNetB0); sets "
                            "KERAS_BACKEND=torch unless already set")
    group.add_argument("--torch", dest="keras", action="store_false",
                       help="PyTorch implementation (ResNet18, default)")
    backend.set_defaults(keras=False)
    backend.add_argument("--device", default=None,
                         help="cuda | cpu | mps (torch implementation only)")
    backend.add_argument("--config", default=None,
                         help="JSON file with flag values (CLI flags override)")

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None,
                    help="JSON file with flag values; may include \"cmd\"")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", parents=[backend])
    t.add_argument("--root", default="data")
    t.add_argument("--backbone", default="resnet18",
                   help="timm model name (torch only; keras always uses "
                        "EfficientNetB0)")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--freeze-epochs", type=int, default=10)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--dropout", type=float, default=0.4)
    t.add_argument("--height", type=int, default=256)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--prefix", default=None,
                   help="checkpoint prefix (default: fold / kfold)")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--cache", action="store_true", help="decode images once into RAM")
    t.add_argument("--cache-side", type=int, default=640)
    t.add_argument("--no-reverse", action="store_true",
                   help="skip synthesised reversed negatives")
    t.add_argument("--identity-negatives", action="store_true",
                   help="add same-photo no-change negatives (for datasets "
                        "with no natural negative pairs)")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", parents=[backend])
    p.add_argument("--ckpt", default=None,
                   help="checkpoint glob (default: fold*.pt / kfold*.keras)")
    p.add_argument("--before", default=None, help="required (flag or config)")
    p.add_argument("--after", default=None, help="required (flag or config)")
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("evaluate", parents=[backend])
    e.add_argument("--root", default=None,
                   help="held-out folder, same layout; required (flag or config)")
    e.add_argument("--ckpt", default=None,
                   help="checkpoint glob (default: fold*.pt / kfold*.keras)")
    e.add_argument("--cache", action="store_true")
    e.add_argument("--cache-side", type=int, default=640)
    e.set_defaults(func=cmd_evaluate)

    # --config: pre-scan argv, then install config values as parser defaults
    # for the target subcommand -- explicit CLI flags override naturally.
    argv = sys.argv[1:]
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", default=None)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        cfg = load_config(known.config)
        subcommands = {"train": t, "predict": p, "evaluate": e}
        cmd = next((a for a in argv if a in subcommands), None)
        if cmd is None:
            cmd = cfg.get("cmd")
            if cmd not in subcommands:
                raise SystemExit(
                    "error: no subcommand on the command line and no valid "
                    f"\"cmd\" in {known.config}")
            argv = [cmd] + argv
        parser = subcommands[cmd]
        valid = {action.dest for action in parser._actions}
        unknown = set(cfg) - valid - {"cmd"}
        if unknown:
            raise SystemExit(
                f"error: unknown config keys for '{cmd}': "
                f"{', '.join(sorted(unknown))}")
        parser.set_defaults(**{k: v for k, v in cfg.items() if k != "cmd"})

    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
