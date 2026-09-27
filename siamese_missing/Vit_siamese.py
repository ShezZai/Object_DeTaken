"""Drop-in ViT alternative to SiameseBool (siamese_missing/train_missing.py).

Same interface: forward(a, b) -> logit, freeze_encoder(flag), train_mode().
Differences from the CNN version:

  1. Frozen DINOv2 ViT encoder, and we keep the PATCH TOKENS instead of
     global-pooling them away. A missing cup is ~1% of the image; after
     average pooling it is ~1% of the feature vector. Per-patch it is 100%
     of a few tokens.
  2. Cross-attention: every "before" patch queries all "after" patches.
     If the object merely moved (or the camera shifted), attention finds it
     elsewhere and the residual is small. If it was taken, nothing matches
     and the residual is large. This is the learned version of the
     template-matching step in pipeline/compare_yolo_objects.py.
  3. Only BEFORE queries AFTER, so the model is asymmetric by construction:
     "something in before is unexplained by after" = removal. Reversed-pair
     negatives still work unchanged.
  4. A tiny transformer reads the residual tokens and scores every patch.
     The pair logit is a smooth max over those scores (log-mean-exp: "at
     least one patch is unexplained"), so the per-patch scores are trained
     by the classification loss itself and double as a localization map
     (replaces Grad-CAM).

Use --height/--width multiples of 14 (224, 336, 448) for DINOv2.
"""
import math

import torch
import torch.nn as nn
import timm


class SiameseViT(nn.Module):
    def __init__(self, backbone="vit_small_patch14_dinov2.lvd142m",
                 pretrained=True, p_drop=0.4, hidden=128, heads=4, depth=1,
                 **_):
        super().__init__()
        self.enc = timm.create_model(backbone, pretrained=pretrained,
                                     num_classes=0, dynamic_img_size=True,
                                     dynamic_img_pad=True)
        self.n_prefix = getattr(self.enc, "num_prefix_tokens", 1)
        d = self.enc.num_features

        self.proj = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden))
        # before-patches (queries) look for themselves among after-patches
        self.cross = nn.MultiheadAttention(hidden, heads, dropout=0.1,
                                           batch_first=True)
        # per-patch evidence: [before, matched-after, before - matched,
        #                      before - same-position-after]
        self.fuse = nn.Sequential(nn.Linear(4 * hidden, hidden),
                                  nn.GELU(), nn.Dropout(p_drop))
        layer = nn.TransformerEncoderLayer(hidden, heads, 2 * hidden,
                                           dropout=0.1, batch_first=True,
                                           activation="gelu", norm_first=True)
        self.reason = nn.TransformerEncoder(layer, depth,
                                            enable_nested_tensor=False)
        # per-patch "this before-patch is gone" logit == localization map
        self.patch_score = nn.Sequential(nn.LayerNorm(hidden),
                                         nn.Dropout(p_drop),
                                         nn.Linear(hidden, 1))
        self.enc_frozen = False

    def tokens(self, x):
        t = self.enc.forward_features(x)          # [B, prefix + N, D]
        return self.proj(t[:, self.n_prefix:])    # drop CLS/register tokens

    def forward(self, a, b, return_map=False):
        ta, tb = self.tokens(a), self.tokens(b)   # [B, N, H] each
        matched, _ = self.cross(ta, tb, tb)       # best explanation of each
                                                  # before-patch from after
        z = self.fuse(torch.cat([ta, matched, ta - matched, ta - tb], -1))
        s = self.patch_score(self.reason(z)).squeeze(-1)       # [B, N]
        # log-mean-exp: a smooth max that does not grow with the patch count,
        # so one confident patch is enough and resolution does not shift it
        logit = torch.logsumexp(s, 1) - math.log(s.shape[1])
        if return_map:
            return logit, s
        return logit

    def patch_grid(self, h, w):
        """(rows, cols, patch) of the token grid for an h x w input; the
        grid covers the right/bottom-padded image (dynamic_img_pad)."""
        ph, pw = self.enc.patch_embed.patch_size
        return math.ceil(h / ph), math.ceil(w / pw), (ph, pw)

    def freeze_encoder(self, flag=True):
        for p in self.enc.parameters():
            p.requires_grad = not flag
        self.enc_frozen = flag

    def train_mode(self):
        self.train()
        if self.enc_frozen:
            self.enc.eval()


if __name__ == "__main__":
    m = SiameseViT(pretrained=False)
    m.freeze_encoder(True)
    a, b = torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224)
    logit, pmap = m(a, b, return_map=True)
    trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(logit.shape, pmap.shape, f"{trainable/1e3:.0f}k trainable params")