"""Realistic missingness simulation for multichannel EEG.

Mask convention: 1 = observed, 0 = unavailable/reconstructed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np


@dataclass(frozen=True)
class MaskedEEG:
    incomplete: np.ndarray
    mask: np.ndarray
    target: np.ndarray


def _validate(x: np.ndarray) -> Tuple[int, int]:
    if x.ndim != 2:
        raise ValueError("A single EEG segment must have shape [channels, time].")
    return x.shape


def random_mask(shape: Tuple[int, int], ratio: float, rng: np.random.Generator) -> np.ndarray:
    if not 0 <= ratio < 1:
        raise ValueError("ratio must be in [0, 1).")
    mask = np.ones(shape, dtype=np.float32)
    n = int(round(mask.size * ratio))
    if n > 0:
        idx = rng.choice(mask.size, size=n, replace=False)
        mask.flat[idx] = 0.0
    return mask


def segment_mask(shape: Tuple[int, int], ratio: float, rng: np.random.Generator, segment_length: Optional[int] = None) -> np.ndarray:
    c, t = shape
    mask = np.ones(shape, dtype=np.float32)
    target_missing = max(1, int(round(c * t * ratio)))
    missing = 0
    while missing < target_missing:
        ch = int(rng.integers(0, c))
        if segment_length is None:
            max_len = max(1, int(t * min(max(ratio, 0.02), 0.5)))
            length = int(rng.integers(1, max_len + 1))
        else:
            length = min(int(segment_length), t)
        start = int(rng.integers(0, max(1, t - length + 1)))
        before = np.count_nonzero(mask[ch, start:start + length] == 0)
        mask[ch, start:start + length] = 0.0
        missing += length - before
        if np.count_nonzero(mask == 0) >= target_missing:
            break
    return mask


def channel_mask(shape: Tuple[int, int], ratio: float, rng: np.random.Generator, n_channels: Optional[int] = None) -> np.ndarray:
    c, _ = shape
    mask = np.ones(shape, dtype=np.float32)
    n = n_channels if n_channels is not None else max(1, int(round(c * ratio)))
    n = min(max(1, int(n)), max(1, c - 1))
    idx = rng.choice(c, size=n, replace=False)
    mask[idx, :] = 0.0
    return mask


def mixed_mask(shape: Tuple[int, int], ratio: float, rng: np.random.Generator) -> np.ndarray:
    # Split the target corruption approximately equally between channel and segment loss.
    c, _ = shape
    channel_ratio = min(0.5 * ratio, (c - 1) / max(c, 1))
    m1 = channel_mask(shape, max(channel_ratio, 1.0 / max(c, 1)), rng)
    m2 = segment_mask(shape, max(0.01, ratio * 0.5), rng)
    mask = m1 * m2
    # Do not permit a completely unobserved segment.
    if np.all(mask == 0):
        mask[int(rng.integers(0, c)), :] = 1.0
    return mask.astype(np.float32)


def make_mask(
    x: np.ndarray,
    kind: str,
    ratio: float,
    rng: np.random.Generator,
    sampling_rate: Optional[float] = None,
    segment_seconds: Optional[float] = None,
    n_channels: Optional[int] = None,
) -> np.ndarray:
    shape = _validate(x)
    kind = kind.lower()
    if kind == "random":
        return random_mask(shape, ratio, rng)
    if kind == "segment":
        length = None
        if segment_seconds is not None:
            if sampling_rate is None:
                raise ValueError("sampling_rate is required when segment_seconds is set.")
            length = max(1, int(round(segment_seconds * sampling_rate)))
        return segment_mask(shape, ratio, rng, length)
    if kind == "channel":
        return channel_mask(shape, ratio, rng, n_channels=n_channels)
    if kind == "mixed":
        return mixed_mask(shape, ratio, rng)
    raise ValueError(f"Unknown missingness kind: {kind}")


def apply_mask(x: np.ndarray, mask: np.ndarray, fill_value: float = 0.0) -> MaskedEEG:
    if x.shape != mask.shape:
        raise ValueError("Signal and mask shapes must match.")
    incomplete = np.where(mask > 0.5, x, fill_value).astype(np.float32)
    return MaskedEEG(incomplete=incomplete, mask=mask.astype(np.float32), target=x.astype(np.float32))
