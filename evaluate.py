"""Reconstruction metrics and systematic missingness evaluation."""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Optional
import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import pearsonr
import torch

from missingness import make_mask
from train import resolve_device


def reconstruction_metrics(target: np.ndarray, pred: np.ndarray, mask: np.ndarray, fs: float = 256.0) -> Dict[str, float]:
    missing = mask < 0.5
    if not np.any(missing):
        raise ValueError("Evaluation mask contains no missing samples.")
    y = target[missing].astype(float)
    p = pred[missing].astype(float)
    err = p - y
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    corr = float(pearsonr(y, p).statistic) if len(y) > 2 and np.std(y) > 0 and np.std(p) > 0 else np.nan
    signal_power = float(np.sum(y ** 2))
    noise_power = float(np.sum(err ** 2)) + 1e-12
    snr = float(10.0 * np.log10((signal_power + 1e-12) / noise_power))

    # PSD correlation is computed across full reconstructed signals so that spectral preservation is assessed globally.
    psd_corrs = []
    for ch in range(target.shape[0]):
        _, a = welch(target[ch], fs=fs, nperseg=min(256, target.shape[-1]))
        _, b = welch(pred[ch], fs=fs, nperseg=min(256, pred.shape[-1]))
        if np.std(a) > 0 and np.std(b) > 0:
            psd_corrs.append(pearsonr(a, b).statistic)
    return {"mae": mae, "rmse": rmse, "correlation": corr, "snr_db": snr, "psd_correlation": float(np.nanmean(psd_corrs))}


@torch.no_grad()
def predict_model(model: torch.nn.Module, incomplete: np.ndarray, mask: np.ndarray, device: str = "auto") -> np.ndarray:
    dev = resolve_device(device)
    model.eval().to(dev)
    x = torch.from_numpy(incomplete[None].astype(np.float32)).to(dev)
    m = torch.from_numpy(mask[None].astype(np.float32)).to(dev)
    return model(x, m).squeeze(0).cpu().numpy()


def evaluate_condition(
    X: np.ndarray,
    predictor: Callable[[np.ndarray, np.ndarray], np.ndarray],
    kind: str,
    ratio: float,
    fs: float,
    seed: int,
    repetition: int = 0,
    segment_seconds: Optional[float] = None,
    n_channels: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict] = []
    for i, target in enumerate(X):
        rng = np.random.default_rng(seed + repetition * 10_000_019 + i * 7919)
        mask = make_mask(target, kind, ratio, rng, fs, segment_seconds, n_channels)
        incomplete = target * mask
        pred = predictor(incomplete, mask)
        metrics = reconstruction_metrics(target, pred, mask, fs)
        rows.append({"segment": i, "repetition": repetition, "missingness": kind, "ratio": ratio,
                     "segment_seconds": segment_seconds, "n_missing_channels": n_channels, **metrics})
    return pd.DataFrame(rows)


def aggregate_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = ["mae", "rmse", "correlation", "snr_db", "psd_correlation"]
    group_cols = [c for c in ["method", "missingness", "ratio", "segment_seconds", "n_missing_channels"] if c in df.columns]
    agg = df.groupby(group_cols, dropna=False)[metric_cols].agg(["mean", "std", "count"]).reset_index()
    agg.columns = ["_".join([str(x) for x in col if x != ""]) if isinstance(col, tuple) else col for col in agg.columns]
    return agg
