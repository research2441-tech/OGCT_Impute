"""Training utilities with deterministic masking and early stopping."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from losses import LossWeights, ogct_loss
from missingness import make_mask


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


class MaskedEEGDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        sampling_rate: float,
        kinds: Sequence[str],
        ratio_min: float,
        ratio_max: float,
        seed: int,
        fixed: bool = False,
    ):
        self.x = x.astype(np.float32)
        self.fs = float(sampling_rate)
        self.kinds = list(kinds)
        self.ratio_min = float(ratio_min)
        self.ratio_max = float(ratio_max)
        self.seed = int(seed)
        self.fixed = fixed
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.x)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int):
        # Fixed validation masks; epoch-dependent deterministic training masks.
        e = 0 if self.fixed else self.epoch
        rng = np.random.default_rng(self.seed + idx * 1009 + e * 1000003)
        kind = self.kinds[int(rng.integers(0, len(self.kinds)))]
        ratio = float(rng.uniform(self.ratio_min, self.ratio_max))
        m = make_mask(self.x[idx], kind, ratio, rng, sampling_rate=self.fs)
        incomplete = self.x[idx] * m
        return (
            torch.from_numpy(incomplete),
            torch.from_numpy(m),
            torch.from_numpy(self.x[idx]),
        )


@dataclass
class TrainResult:
    best_val: float
    best_epoch: int
    history: list


def fit_model(
    model: torch.nn.Module,
    train_x: np.ndarray,
    val_x: np.ndarray,
    cfg: Dict,
    loss_weights: LossWeights,
    checkpoint: Optional[str] = None,
    epochs_override: Optional[int] = None,
    verbose: bool = True,
) -> TrainResult:
    seed = int(cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(cfg.get("device", "auto"))
    model.to(device)

    tr_cfg = cfg["training"]
    miss = cfg["missingness"]
    fs = float(cfg["data"].get("target_sampling_rate", cfg["data"].get("sampling_rate", 256)))
    train_ds = MaskedEEGDataset(
        train_x, fs, miss["types"], miss["train_ratio_min"], miss["train_ratio_max"], seed, fixed=False
    )
    val_ds = MaskedEEGDataset(
        val_x, fs, miss["types"], miss["train_ratio_min"], miss["train_ratio_max"], seed + 77, fixed=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=int(tr_cfg["batch_size"]), shuffle=True,
        num_workers=int(tr_cfg.get("num_workers", 0)), drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(tr_cfg["batch_size"]), shuffle=False,
        num_workers=int(tr_cfg.get("num_workers", 0)), drop_last=False
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(tr_cfg["learning_rate"]), weight_decay=float(tr_cfg.get("weight_decay", 0.0))
    )
    epochs = int(epochs_override or tr_cfg["epochs"])
    patience = int(tr_cfg.get("patience", 15))
    use_amp = bool(tr_cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_val = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    best_state = None

    for epoch in range(1, epochs + 1):
        train_ds.set_epoch(epoch)
        model.train()
        tr_sum, tr_n = 0.0, 0
        iterator = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False, disable=not verbose)
        for incomplete, mask, target in iterator:
            incomplete, mask, target = incomplete.to(device), mask.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred = model(incomplete, mask)
                losses = ogct_loss(pred, target, mask, loss_weights)
                loss = losses["total"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(tr_cfg.get("gradient_clip", 1.0)))
            scaler.step(optimizer)
            scaler.update()
            tr_sum += float(loss.detach()) * len(target)
            tr_n += len(target)

        model.eval()
        va_sum, va_n = 0.0, 0
        with torch.no_grad():
            for incomplete, mask, target in val_loader:
                incomplete, mask, target = incomplete.to(device), mask.to(device), target.to(device)
                pred = model(incomplete, mask)
                loss = ogct_loss(pred, target, mask, loss_weights)["total"]
                va_sum += float(loss) * len(target)
                va_n += len(target)
        tr_loss = tr_sum / max(1, tr_n)
        va_loss = va_sum / max(1, va_n)
        history.append({"epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss})

        if va_loss < best_val - 1e-8:
            best_val, best_epoch, stale = va_loss, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if checkpoint:
        Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict(), "best_val": best_val, "best_epoch": best_epoch}, checkpoint)
    return TrainResult(best_val=best_val, best_epoch=best_epoch, history=history)
