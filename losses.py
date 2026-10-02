"""Multi-component reconstruction objective for OGCT-Impute."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * weight).sum() / (weight.sum() + eps)


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    missing = 1.0 - mask
    return _masked_mean(torch.abs(pred - target), missing)


def temporal_consistency_loss(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    dp = pred[..., 1:] - pred[..., :-1]
    dt = target[..., 1:] - target[..., :-1]
    missing_edge = torch.maximum(1.0 - mask[..., 1:], 1.0 - mask[..., :-1])
    return _masked_mean(torch.abs(dp - dt), missing_edge)


def spectral_preservation_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.fft.rfft(pred, dim=-1)
    t = torch.fft.rfft(target, dim=-1)
    p_power = torch.log1p(p.real.square() + p.imag.square())
    t_power = torch.log1p(t.real.square() + t.imag.square())
    return F.smooth_l1_loss(p_power, t_power)


def interchannel_consistency_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Match channel covariance structure per batch item.
    p = pred - pred.mean(dim=-1, keepdim=True)
    t = target - target.mean(dim=-1, keepdim=True)
    denom = max(1, pred.shape[-1] - 1)
    pc = torch.matmul(p, p.transpose(1, 2)) / denom
    tc = torch.matmul(t, t.transpose(1, 2)) / denom
    return F.smooth_l1_loss(pc, tc)


@dataclass
class LossWeights:
    reconstruction: float = 1.0
    temporal: float = 0.20
    spectral: float = 0.15
    channel: float = 0.10


def ogct_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weights: LossWeights,
) -> Dict[str, torch.Tensor]:
    l_rec = reconstruction_loss(pred, target, mask)
    l_tmp = temporal_consistency_loss(pred, target, mask)
    l_spc = spectral_preservation_loss(pred, target)
    l_ch = interchannel_consistency_loss(pred, target)
    total = (
        weights.reconstruction * l_rec
        + weights.temporal * l_tmp
        + weights.spectral * l_spc
        + weights.channel * l_ch
    )
    return {
        "total": total,
        "reconstruction": l_rec,
        "temporal": l_tmp,
        "spectral": l_spc,
        "channel": l_ch,
    }
