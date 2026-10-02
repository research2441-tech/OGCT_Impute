"""OGCT-Impute: Optimization-Guided CNN-Transformer EEG reconstruction network."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 16384):
        super().__init__()
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class ResidualTemporalConv(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size, padding=pad),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, kernel_size, padding=pad),
            nn.BatchNorm1d(d_model),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TemporalCNNEncoder(nn.Module):
    """Shared per-channel CNN. Signal and availability mask are two input features."""
    def __init__(self, d_model: int, kernel_size: int, layers: int, dropout: float):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.stem = nn.Sequential(
            nn.Conv1d(2, d_model, kernel_size, padding=kernel_size // 2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualTemporalConv(d_model, kernel_size, dropout) for _ in range(max(1, layers - 1))
        ])

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x/mask: [B,C,T] -> [B,C,T,D]
        b, c, t = x.shape
        z = torch.stack([x, mask], dim=2).reshape(b * c, 2, t)
        z = self.blocks(self.stem(z)).transpose(1, 2)
        return z.reshape(b, c, t, -1)


class ChannelAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float, ff_multiplier: int = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_multiplier), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * ff_multiplier, d_model), nn.Dropout(dropout),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # [B,C,T,D] -> attention over channels at every time point
        b, c, t, d = z.shape
        q = z.permute(0, 2, 1, 3).reshape(b * t, c, d)
        h = self.norm1(q)
        a, _ = self.attn(h, h, h, need_weights=False)
        q = q + a
        q = q + self.ff(self.norm2(q))
        return q.reshape(b, t, c, d).permute(0, 2, 1, 3)


class ReconstructionDecoder(nn.Module):
    def __init__(self, d_model: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)


class OGCTImpute(nn.Module):
    def __init__(
        self,
        d_model: int = 96,
        cnn_kernel_size: int = 7,
        cnn_layers: int = 3,
        temporal_heads: int = 4,
        temporal_layers: int = 3,
        channel_heads: int = 4,
        dropout: float = 0.15,
        ff_multiplier: int = 4,
    ):
        super().__init__()
        if d_model % temporal_heads != 0 or d_model % channel_heads != 0:
            raise ValueError("d_model must be divisible by both temporal_heads and channel_heads.")
        self.cnn = TemporalCNNEncoder(d_model, cnn_kernel_size, cnn_layers, dropout)
        self.pos = PositionalEncoding(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=temporal_heads,
            dim_feedforward=d_model * ff_multiplier,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(layer, num_layers=temporal_layers)
        self.channel_attention = ChannelAttention(d_model, channel_heads, dropout, ff_multiplier)
        self.decoder = ReconstructionDecoder(d_model, dropout)

    def forward(self, incomplete: torch.Tensor, mask: torch.Tensor, preserve_observed: bool = True) -> torch.Tensor:
        if incomplete.shape != mask.shape or incomplete.ndim != 3:
            raise ValueError("incomplete and mask must have shape [B,C,T].")
        local = self.cnn(incomplete, mask)
        b, c, t, d = local.shape
        temp = local.reshape(b * c, t, d)
        temp = self.temporal(self.pos(temp)).reshape(b, c, t, d)
        spatiotemporal = self.channel_attention(temp)
        predicted = self.decoder(spatiotemporal)
        if preserve_observed:
            return mask * incomplete + (1.0 - mask) * predicted
        return predicted

    @torch.no_grad()
    def impute(self, incomplete: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        self.eval()
        return self.forward(incomplete, mask, preserve_observed=True)
