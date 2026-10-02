"""Classical and neural baseline implementations for fair EEG imputation comparison."""
from __future__ import annotations

from typing import Optional
import numpy as np
import torch
from torch import nn
from scipy.interpolate import CubicSpline
from sklearn.impute import KNNImputer


def linear_interpolate(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = x.copy().astype(np.float32)
    c, t = x.shape
    time = np.arange(t)
    for ch in range(c):
        obs = mask[ch] > 0.5
        if obs.sum() == 0:
            out[ch] = 0.0
        elif obs.sum() == 1:
            out[ch] = x[ch, obs][0]
        else:
            out[ch, ~obs] = np.interp(time[~obs], time[obs], x[ch, obs])
    return out


def spline_interpolate(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    out = x.copy().astype(np.float32)
    c, t = x.shape
    time = np.arange(t)
    for ch in range(c):
        obs = mask[ch] > 0.5
        if obs.sum() < 4:
            out[ch] = linear_interpolate(x[[ch]], mask[[ch]])[0]
        else:
            cs = CubicSpline(time[obs], x[ch, obs], bc_type="natural", extrapolate=True)
            out[ch, ~obs] = cs(time[~obs])
    return out


def knn_impute(x: np.ndarray, mask: np.ndarray, n_neighbors: int = 5) -> np.ndarray:
    # Treat time points as samples and channels as features.
    arr = x.T.astype(float)
    arr[mask.T < 0.5] = np.nan
    n_neighbors = max(1, min(n_neighbors, max(1, arr.shape[0] - 1)))
    imp = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    y = imp.fit_transform(arr)
    if y.shape[1] != x.shape[0]:
        # KNNImputer can discard all-missing features; fallback to linear for those cases.
        return linear_interpolate(x, mask)
    return y.T.astype(np.float32)


class ConvImputer(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels * 2, hidden, 7, padding=3), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, channels, 3, padding=1),
        )
    def forward(self, x, mask):
        pred = self.net(torch.cat([x, mask], dim=1))
        return mask * x + (1 - mask) * pred


class RNNImputer(nn.Module):
    def __init__(self, channels: int, hidden: int = 96, kind: str = "lstm"):
        super().__init__()
        rnn_cls = nn.LSTM if kind.lower() == "lstm" else nn.GRU
        self.rnn = rnn_cls(channels * 2, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden * 2, channels)
    def forward(self, x, mask):
        z = torch.cat([x, mask], dim=1).transpose(1, 2)
        z, _ = self.rnn(z)
        pred = self.out(z).transpose(1, 2)
        return mask * x + (1 - mask) * pred


class AutoencoderImputer(nn.Module):
    def __init__(self, channels: int, hidden: int = 64):
        super().__init__()
        self.enc = nn.Sequential(nn.Conv1d(channels * 2, hidden, 5, padding=2), nn.GELU(), nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.GELU())
        self.dec = nn.Sequential(nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1), nn.GELU(), nn.Conv1d(hidden, channels, 5, padding=2))
    def forward(self, x, mask):
        pred = self.dec(self.enc(torch.cat([x, mask], dim=1)))[..., : x.shape[-1]]
        return mask * x + (1 - mask) * pred


class TransformerImputer(nn.Module):
    def __init__(self, channels: int, d_model: int = 96, heads: int = 4, layers: int = 2):
        super().__init__()
        self.inp = nn.Linear(channels * 2, d_model)
        layer = nn.TransformerEncoderLayer(d_model, heads, d_model * 4, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers)
        self.out = nn.Linear(d_model, channels)
    def forward(self, x, mask):
        z = torch.cat([x, mask], dim=1).transpose(1, 2)
        pred = self.out(self.enc(self.inp(z))).transpose(1, 2)
        return mask * x + (1 - mask) * pred

class CNNLSTMImputer(nn.Module):
    def __init__(self, channels: int, hidden: int = 96):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(channels * 2, hidden, 7, padding=3), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2), nn.GELU(),
        )
        self.rnn = nn.LSTM(hidden, hidden, batch_first=True, bidirectional=True)
        self.out = nn.Linear(hidden * 2, channels)
    def forward(self, x, mask):
        z = self.conv(torch.cat([x, mask], dim=1)).transpose(1, 2)
        z, _ = self.rnn(z)
        pred = self.out(z).transpose(1, 2)
        return mask * x + (1 - mask) * pred


class CNNTransformerImputer(nn.Module):
    def __init__(self, channels: int, d_model: int = 96, heads: int = 4, layers: int = 2, dropout: float = 0.15):
        super().__init__()
        if d_model % heads != 0:
            d_model = heads * max(1, d_model // heads)
        self.conv = nn.Sequential(
            nn.Conv1d(channels * 2, d_model, 7, padding=3), nn.BatchNorm1d(d_model), nn.GELU(),
            nn.Conv1d(d_model, d_model, 5, padding=2), nn.BatchNorm1d(d_model), nn.GELU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=heads, dim_feedforward=d_model * 4, dropout=dropout,
            batch_first=True, norm_first=True, activation="gelu"
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=layers)
        self.out = nn.Linear(d_model, channels)
    def forward(self, x, mask):
        z = self.conv(torch.cat([x, mask], dim=1)).transpose(1, 2)
        pred = self.out(self.enc(z)).transpose(1, 2)
        return mask * x + (1 - mask) * pred
