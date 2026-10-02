"""Downstream EEG utility evaluation using leakage-safe band-power features."""
from __future__ import annotations

from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 45.0),
}


def bandpower_features(X: np.ndarray, fs: float) -> np.ndarray:
    feats = []
    for x in X:
        f, pxx = welch(x, fs=fs, axis=-1, nperseg=min(256, x.shape[-1]))
        row = []
        for lo, hi in BANDS.values():
            idx = (f >= lo) & (f < hi)
            row.extend(np.log1p(np.trapezoid(pxx[:, idx], f[idx], axis=-1)).tolist())
        feats.append(row)
    return np.asarray(feats, dtype=np.float32)


def evaluate_classifier(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray, fs: float) -> Dict[str, float]:
    if train_y is None or test_y is None:
        raise ValueError("Downstream evaluation requires class labels y in the dataset NPZ.")
    xtr = bandpower_features(train_x, fs)
    xte = bandpower_features(test_x, fs)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(xtr, train_y)
    pred = clf.predict(xte)
    average = "binary" if len(np.unique(train_y)) == 2 else "macro"
    return {
        "accuracy": accuracy_score(test_y, pred) * 100.0,
        "precision": precision_score(test_y, pred, average=average, zero_division=0) * 100.0,
        "recall": recall_score(test_y, pred, average=average, zero_division=0) * 100.0,
        "f1": f1_score(test_y, pred, average=average, zero_division=0) * 100.0,
    }
