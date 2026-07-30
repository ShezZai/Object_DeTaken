"""
train_missing.py
Siamese binary "did anything go missing between these two photos?" model,
tuned for a small dataset (200-500 pairs across 20-50 scenes).

    python train_missing.py train    --root data
    python train_missing.py predict  --before a.jpg --after b.jpg
    python train_missing.py evaluate --root holdout

Reads the folder tree directly via pairs_io -- no manifest file, no generated
folders. Reversed negatives are synthesised in memory.
"""

import argparse
import glob
import json

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
# dataset
# --------------------------------------------------------------------------- #
class PairSet(Dataset):
    """
    geometric   -> identical on both images (one shared transform)
    photometric -> independent per image (teaches 'lighting change != item gone')
    jitter      -> small extra warp on `after` only, for registration slack
    """

    def __init__(self, rows, size=(256, 256), train=True, cache=None):
        self.rows, self.train, self.cache = rows, train, cache
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
        self.shared = A.Compose(shared, additional_targets={"image2": "image"})

        self.jitter = A.Affine(translate_percent=0.02, rotate=(-2, 2), p=0.5) if train else None
        self.photo = A.Compose([
            A.RandomBrightnessContrast(0.3, 0.3, p=0.8),
            A.HueSaturationValue(12, 20, 12, p=0.5),
            A.GaussNoise(p=0.2),
        ]) if train else None

    def _read(self, path):
        if self.cache is not None:
            return self.cache.get(path)
        raw = cv2.imread(path, cv2.IMREAD_COLOR)
        if raw is None:
            raise FileNotFoundError(f"could not decode {path}")
        return cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        out = self.shared(image=self._read(r["before"]), image2=self._read(r["after"]))
        a, b = out["image"], out["image2"]
        if self.jitter is not None:
            b = self.jitter(image=b)["image"]
        if self.photo is not None:
            a = self.photo(image=a)["image"]
            b = self.photo(image=b)["image"]
        return to_tensor(a), to_tensor(b), torch.tensor(float(r["y"]))


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
class SiameseBool(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, p_drop=0.4):
        super().__init__()
        # ONE encoder called twice == shared weights. That is the siamese part.
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
        # SIGNED difference: the sign separates 'removed' from 'added'
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
# evaluation helpers
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
# training
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


def cmd_train(args):
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
        r = train_fold(tr_rows, va_rows, args, device, k, cache)
        results.append(r)
        print(f"fold {k}: {r['n_val']:3d} val pairs | AUC {r['auc']:.3f} | "
              f"acc {r['acc']:.3f} | TP {r['tp']} FP {r['fp']} FN {r['fn']} "
              f"TN {r['tn']} | best epoch {r['epoch']}")

    aucs = np.array([r["auc"] for r in results])
    accs = np.array([r["acc"] for r in results])
    print(f"\nAUC {aucs.mean():.3f} +/- {aucs.std():.3f}"
          f"   acc {accs.mean():.3f} +/- {accs.std():.3f}")
    print(f"checkpoints: {args.prefix}0.pt ... {args.prefix}{args.folds - 1}.pt")
    print("read the spread as well as the mean -- at this data size one lucky "
          "fold can carry the average")


# --------------------------------------------------------------------------- #
# inference
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


def cmd_predict(args):
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble(args.ckpt, device)
    probs = [pair_prob(m, args.before, args.after, size, device)
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
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble(args.ckpt, device)
    # natural pairs only: reversals would flatter the score
    rows = load_pairs(args.root, reverse_positives=False)
    cache = ImageCache(args.cache_side).warm(rows) if args.cache else None
    print()

    y = np.array([r["y"] for r in rows])
    p = np.array([
        float(np.mean([pair_prob(m, r["before"], r["after"], size, device, cache)
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
    ap.add_argument("--device", default=None, help="cuda | cpu | mps")
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--root", default="data")
    t.add_argument("--backbone", default="resnet18")
    t.add_argument("--folds", type=int, default=5)
    t.add_argument("--epochs", type=int, default=60)
    t.add_argument("--freeze-epochs", type=int, default=10)
    t.add_argument("--patience", type=int, default=15)
    t.add_argument("--bs", type=int, default=16)
    t.add_argument("--dropout", type=float, default=0.4)
    t.add_argument("--height", type=int, default=256)
    t.add_argument("--width", type=int, default=256)
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--prefix", default="fold")
    t.add_argument("--workers", type=int, default=2)
    t.add_argument("--cache", action="store_true", help="decode images once into RAM")
    t.add_argument("--cache-side", type=int, default=640)
    t.add_argument("--no-reverse", action="store_true",
                   help="skip synthesised reversed negatives")
    t.add_argument("--identity-negatives", action="store_true",
                   help="add same-photo no-change negatives (for datasets "
                        "with no natural negative pairs)")
    t.set_defaults(func=cmd_train)

    p = sub.add_parser("predict")
    p.add_argument("--ckpt", default="fold*.pt")
    p.add_argument("--before", required=True)
    p.add_argument("--after", required=True)
    p.set_defaults(func=cmd_predict)

    e = sub.add_parser("evaluate")
    e.add_argument("--root", required=True, help="held-out folder, same layout")
    e.add_argument("--ckpt", default="fold*.pt")
    e.add_argument("--cache", action="store_true")
    e.add_argument("--cache-side", type=int, default=640)
    e.set_defaults(func=cmd_evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
