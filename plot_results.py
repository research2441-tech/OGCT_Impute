"""Publication-quality result plotting. Figures are generated from experiment CSVs."""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import welch


def plot_training_history(history_csv: str, output: str) -> None:
    df = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(df["epoch"], df["train_loss"], label="Training")
    ax.plot(df["epoch"], df["val_loss"], label="Validation")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.legend(); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def plot_missingness(df: pd.DataFrame, output: str, metric: str = "rmse") -> None:
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    for method, g in df.groupby("method"):
        s = g.groupby("ratio")[metric].mean().sort_index()
        ax.plot(s.index * 100, s.values, marker="o", label=method)
    ax.set_xlabel("Missingness ratio (%)"); ax.set_ylabel(metric.upper()); ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def plot_reconstruction(target: np.ndarray, incomplete: np.ndarray, pred: np.ndarray, fs: float, output: str, channel: int = 0) -> None:
    t = np.arange(target.shape[-1]) / fs
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    ax.plot(t, target[channel], label="Original", linewidth=1.0)
    ax.plot(t, incomplete[channel], label="Masked", linewidth=0.9)
    ax.plot(t, pred[channel], label="Reconstructed", linewidth=0.9)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Normalized amplitude"); ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def plot_psd(target: np.ndarray, pred: np.ndarray, fs: float, output: str, channel: int = 0) -> None:
    f1, p1 = welch(target[channel], fs=fs, nperseg=min(512, target.shape[-1]))
    f2, p2 = welch(pred[channel], fs=fs, nperseg=min(512, pred.shape[-1]))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.semilogy(f1, p1, label="Original")
    ax.semilogy(f2, p2, label="Reconstructed")
    ax.set_xlim(0, min(45, fs / 2)); ax.set_xlabel("Frequency (Hz)"); ax.set_ylabel("PSD"); ax.legend(); ax.grid(alpha=0.2)
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)
