"""Fast integrity tests that do not require an external EEG dataset."""
from __future__ import annotations

import numpy as np
import torch

from missingness import make_mask
from ogct_model import OGCTImpute
from losses import LossWeights, ogct_loss
from evaluate import reconstruction_metrics


def synthetic_eeg(n=4, c=8, t=256, fs=128, seed=1):
    rng = np.random.default_rng(seed)
    time = np.arange(t) / fs
    x = []
    for _ in range(n):
        seg = []
        for ch in range(c):
            s = np.sin(2 * np.pi * (8 + ch * 0.2) * time) + 0.35 * np.sin(2 * np.pi * 18 * time + ch / 3)
            s += 0.05 * rng.normal(size=t)
            seg.append(s)
        x.append(seg)
    return np.asarray(x, dtype=np.float32)


def main():
    fs = 128
    X = synthetic_eeg(fs=fs)
    rng = np.random.default_rng(42)
    masks = np.stack([make_mask(x, "mixed", 0.25, rng, fs) for x in X])
    incomplete = X * masks

    model = OGCTImpute(d_model=32, cnn_kernel_size=5, cnn_layers=2, temporal_heads=4, temporal_layers=1, channel_heads=4, dropout=0.0)
    xt = torch.from_numpy(incomplete); mt = torch.from_numpy(masks); yt = torch.from_numpy(X)
    out = model(xt, mt)
    assert out.shape == yt.shape
    observed = mt > 0.5
    assert torch.allclose(out[observed], xt[observed], atol=1e-7), "Observed samples changed."
    losses = ogct_loss(out, yt, mt, LossWeights())
    assert torch.isfinite(losses["total"])
    metrics = reconstruction_metrics(X[0], out[0].detach().numpy(), masks[0], fs)
    assert np.isfinite(metrics["rmse"])

    # Deterministic mask check.
    r1 = np.random.default_rng(7); r2 = np.random.default_rng(7)
    assert np.array_equal(make_mask(X[0], "segment", 0.2, r1, fs), make_mask(X[0], "segment", 0.2, r2, fs))
    print("All OGCT-Impute pipeline integrity tests passed.")


if __name__ == "__main__":
    main()
