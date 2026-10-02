"""Leakage-safe EEG preprocessing for OGCT-Impute."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly


@dataclass
class ChannelNormalizer:
    mean: Optional[np.ndarray] = None
    std: Optional[np.ndarray] = None
    clip_std: float = 8.0

    def fit(self, x: np.ndarray) -> "ChannelNormalizer":
        if x.ndim != 3:
            raise ValueError("Expected X with shape [segments, channels, time].")
        self.mean = np.nanmean(x, axis=(0, 2), keepdims=True)
        self.std = np.nanstd(x, axis=(0, 2), keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted on the training partition first.")
        z = (x - self.mean) / self.std
        return np.clip(z, -self.clip_std, self.clip_std).astype(np.float32)

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("Normalizer must be fitted first.")
        return x * self.std + self.mean


def bandpass_filter(x: np.ndarray, fs: float, low: float, high: float, order: int = 4) -> np.ndarray:
    if not 0 < low < high < fs / 2:
        raise ValueError(f"Invalid band-pass [{low}, {high}] for fs={fs}.")
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def notch_filter(x: np.ndarray, fs: float, notch_hz: float, q: float = 30.0) -> np.ndarray:
    if notch_hz <= 0 or notch_hz >= fs / 2:
        return x.astype(np.float32)
    b, a = iirnotch(notch_hz, q, fs=fs)
    return filtfilt(b, a, x, axis=-1).astype(np.float32)


def resample_eeg(x: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    if original_fs == target_fs:
        return x.astype(np.float32)
    from math import gcd
    g = gcd(int(original_fs), int(target_fs))
    return resample_poly(x, target_fs // g, original_fs // g, axis=-1).astype(np.float32)


def preprocess_partition(
    x: np.ndarray,
    fs: float,
    low: float,
    high: float,
    notch_hz: float,
) -> np.ndarray:
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    x = bandpass_filter(x, fs, low, high)
    x = notch_filter(x, fs, notch_hz)
    return x


def fit_and_apply_normalization(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray,
    clip_std: float = 8.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ChannelNormalizer]:
    normalizer = ChannelNormalizer(clip_std=clip_std).fit(train)
    return (
        normalizer.transform(train),
        normalizer.transform(val),
        normalizer.transform(test),
        normalizer,
    )
