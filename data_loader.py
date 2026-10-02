"""Dataset loading, fixed-length segmentation, and subject-disjoint splitting."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import numpy as np


@dataclass
class EEGDatasetBundle:
    X: np.ndarray                 # [segments, channels, time]
    subject_id: np.ndarray        # [segments]
    y: Optional[np.ndarray] = None
    sampling_rate: Optional[float] = None


def segment_recording(x: np.ndarray, window: int, overlap: float = 0.5) -> np.ndarray:
    if x.ndim != 2:
        raise ValueError("Recording must have shape [channels, time].")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1).")
    step = max(1, int(round(window * (1 - overlap))))
    if x.shape[-1] < window:
        pad = window - x.shape[-1]
        x = np.pad(x, ((0, 0), (0, pad)), mode="reflect")
    starts = range(0, x.shape[-1] - window + 1, step)
    return np.stack([x[:, s:s + window] for s in starts], axis=0).astype(np.float32)


def load_npz(path: str, signal_key: str = "X", subject_key: str = "subject_id") -> EEGDatasetBundle:
    arr = np.load(path, allow_pickle=True)
    if signal_key not in arr:
        raise KeyError(f"{signal_key!r} not found in {path}.")
    x = np.asarray(arr[signal_key], dtype=np.float32)
    if x.ndim != 3:
        raise ValueError("NPZ X must have shape [segments, channels, time].")
    subjects = np.asarray(arr[subject_key]) if subject_key in arr else np.arange(len(x))
    y = np.asarray(arr["y"]) if "y" in arr else None
    fs = float(arr["sampling_rate"]) if "sampling_rate" in arr else None
    if len(subjects) != len(x):
        raise ValueError("subject_id must contain one identifier per segment.")
    return EEGDatasetBundle(x, subjects, y, fs)


def load_edf_directory(
    path: str,
    sampling_rate: int,
    window_seconds: float,
    overlap: float = 0.5,
    channels: Optional[Sequence[str]] = None,
) -> EEGDatasetBundle:
    try:
        import mne
    except ImportError as e:
        raise ImportError("MNE is required for EDF loading. Install requirements.txt.") from e

    files = sorted(Path(path).rglob("*.edf"))
    if not files:
        raise FileNotFoundError(f"No EDF files found under {path}.")

    xs: List[np.ndarray] = []
    ids: List[str] = []
    expected_channels: Optional[List[str]] = list(channels) if channels is not None else None
    for fp in files:
        raw = mne.io.read_raw_edf(fp, preload=True, verbose="ERROR")
        if expected_channels is None:
            expected_channels = list(raw.ch_names)
        keep = [ch for ch in expected_channels if ch in raw.ch_names]
        if len(keep) != len(expected_channels):
            continue  # montage mismatch: explicit alignment should be performed before benchmarking
        raw.pick(keep)
        raw.resample(sampling_rate, npad="auto", verbose="ERROR")
        data = raw.get_data().astype(np.float32)
        segs = segment_recording(data, int(round(window_seconds * sampling_rate)), overlap)
        xs.append(segs)
        # By default, parent directory is treated as subject identity.
        ids.extend([fp.parent.name or fp.stem] * len(segs))

    if not xs:
        raise RuntimeError("No EDF recording matched the selected channel montage.")
    return EEGDatasetBundle(np.concatenate(xs, axis=0), np.asarray(ids), None, float(sampling_rate))


def load_dataset(cfg: Dict) -> EEGDatasetBundle:
    path = str(cfg["path"])
    p = Path(path)
    if p.is_file() and p.suffix.lower() == ".npz":
        return load_npz(path, cfg.get("signal_key", "X"), cfg.get("subject_key", "subject_id"))
    if p.is_dir():
        return load_edf_directory(
            path,
            sampling_rate=int(cfg.get("target_sampling_rate", cfg.get("sampling_rate", 256))),
            window_seconds=float(cfg.get("window_seconds", 4.0)),
            overlap=float(cfg.get("window_overlap", 0.5)),
        )
    raise FileNotFoundError(
        f"Dataset path {path!r} was not found. Provide a .npz file or a directory containing EDF files."
    )


def subject_disjoint_split(
    subjects: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.")
    unique = np.unique(subjects)
    if len(unique) < 3:
        raise ValueError("At least three unique subjects are required for subject-disjoint train/val/test splitting.")
    rng = np.random.default_rng(seed)
    unique = rng.permutation(unique)
    n = len(unique)
    n_train = max(1, int(round(n * train_ratio)))
    n_val = max(1, int(round(n * val_ratio)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    train_s = set(unique[:n_train])
    val_s = set(unique[n_train:n_train + n_val])
    test_s = set(unique[n_train + n_val:])
    train_idx = np.flatnonzero(np.isin(subjects, list(train_s)))
    val_idx = np.flatnonzero(np.isin(subjects, list(val_s)))
    test_idx = np.flatnonzero(np.isin(subjects, list(test_s)))
    return train_idx, val_idx, test_idx
