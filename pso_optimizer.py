"""Particle Swarm Optimization for OGCT-Impute hyperparameters.

The objective callable MUST evaluate only the validation partition; the test set
must never be exposed during search.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple
import json
import math
import numpy as np


PARAMS = [
    "cnn_kernel_size", "d_model", "temporal_heads", "temporal_layers",
    "learning_rate", "dropout", "loss_temporal", "loss_spectral", "loss_channel",
]


@dataclass
class Particle:
    position: np.ndarray
    velocity: np.ndarray
    best_position: np.ndarray
    best_score: float = float("inf")


class ParticleSwarmOptimizer:
    def __init__(self, bounds: Dict[str, List[float]], population_size: int = 8, iterations: int = 10,
                 inertia: float = 0.729, cognitive: float = 1.49445, social: float = 1.49445,
                 seed: int = 42, tolerance: float = 1e-4, patience: int = 3):
        self.bounds = np.array([bounds[p] for p in PARAMS], dtype=float)
        self.population_size = int(population_size)
        self.iterations = int(iterations)
        self.w, self.c1, self.c2 = float(inertia), float(cognitive), float(social)
        self.rng = np.random.default_rng(seed)
        self.tolerance = float(tolerance)
        self.patience = int(patience)
        self.history: List[Dict] = []

    def _decode(self, x: np.ndarray) -> Dict:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        x = np.clip(x, lo, hi)
        kernel = int(round(x[0])); kernel += 1 - kernel % 2
        heads = max(1, int(round(x[2])))
        d_model = max(heads, int(round(x[1])))
        d_model = max(heads, int(round(d_model / heads)) * heads)
        # channel heads are chosen to divide d_model as well; 4 is preferred when possible.
        channel_heads = next((h for h in (4, 3, 2, 1) if d_model % h == 0), 1)
        return {
            "cnn_kernel_size": kernel,
            "d_model": d_model,
            "temporal_heads": heads,
            "channel_heads": channel_heads,
            "temporal_layers": max(1, int(round(x[3]))),
            "learning_rate": float(x[4]),
            "dropout": float(x[5]),
            "loss_temporal": float(x[6]),
            "loss_spectral": float(x[7]),
            "loss_channel": float(x[8]),
        }

    def optimize(self, objective: Callable[[Dict], float]) -> Tuple[Dict, float, List[Dict]]:
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        span = hi - lo
        swarm = []
        for _ in range(self.population_size):
            pos = self.rng.uniform(lo, hi)
            vel = self.rng.uniform(-0.1 * span, 0.1 * span)
            swarm.append(Particle(pos, vel, pos.copy()))
        global_pos = swarm[0].position.copy()
        global_score = float("inf")
        stale = 0

        for iteration in range(1, self.iterations + 1):
            improved = False
            for i, p in enumerate(swarm):
                cfg = self._decode(p.position)
                score = float(objective(cfg))
                if not np.isfinite(score):
                    score = float("inf")
                if score < p.best_score:
                    p.best_score = score
                    p.best_position = p.position.copy()
                if score < global_score - self.tolerance:
                    global_score = score
                    global_pos = p.position.copy()
                    improved = True
                self.history.append({"iteration": iteration, "particle": i, "score": score, **cfg})
            stale = 0 if improved else stale + 1
            if stale >= self.patience:
                break
            for p in swarm:
                r1 = self.rng.random(len(PARAMS)); r2 = self.rng.random(len(PARAMS))
                p.velocity = (
                    self.w * p.velocity
                    + self.c1 * r1 * (p.best_position - p.position)
                    + self.c2 * r2 * (global_pos - p.position)
                )
                p.velocity = np.clip(p.velocity, -0.25 * span, 0.25 * span)
                p.position = np.clip(p.position + p.velocity, lo, hi)

        return self._decode(global_pos), global_score, self.history
