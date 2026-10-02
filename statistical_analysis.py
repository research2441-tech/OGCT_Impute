"""Paired statistical analysis for reconstruction experiments."""
from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, t
from statsmodels.stats.multitest import multipletests


def mean_ci(x: np.ndarray, confidence: float = 0.95):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return float(np.nanmean(x)), np.nan, np.nan, np.nan
    mean = float(np.mean(x)); sd = float(np.std(x, ddof=1))
    half = float(t.ppf((1 + confidence) / 2, len(x) - 1) * sd / np.sqrt(len(x)))
    return mean, sd, mean - half, mean + half


def rank_biserial_from_differences(d: np.ndarray) -> float:
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    if len(d) == 0:
        return 0.0
    ranks = np.argsort(np.argsort(np.abs(d))) + 1
    pos = ranks[d > 0].sum(); neg = ranks[d < 0].sum()
    return float((pos - neg) / (pos + neg))


def paired_method_comparison(df: pd.DataFrame, metric: str = "rmse", id_col: str = "segment", method_col: str = "method") -> Dict[str, pd.DataFrame]:
    pivot = df.pivot_table(index=id_col, columns=method_col, values=metric, aggfunc="mean").dropna()
    methods = list(pivot.columns)
    if len(methods) < 2:
        raise ValueError("At least two methods are required.")
    friedman = None
    if len(methods) >= 3 and len(pivot) >= 2:
        stat, p = friedmanchisquare(*[pivot[m].values for m in methods])
        friedman = pd.DataFrame([{"metric": metric, "statistic": stat, "p_value": p, "n_pairs": len(pivot)}])

    rows = []
    raw_p = []
    for a, b in combinations(methods, 2):
        d = pivot[a].values - pivot[b].values
        try:
            stat, p = wilcoxon(pivot[a].values, pivot[b].values, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            stat, p = 0.0, 1.0
        ma, sda, loa, hia = mean_ci(pivot[a].values)
        mb, sdb, lob, hib = mean_ci(pivot[b].values)
        rows.append({"method_a": a, "method_b": b, "metric": metric, "wilcoxon_statistic": stat,
                     "p_raw": p, "effect_rank_biserial_a_minus_b": rank_biserial_from_differences(d),
                     "mean_a": ma, "sd_a": sda, "ci95_low_a": loa, "ci95_high_a": hia,
                     "mean_b": mb, "sd_b": sdb, "ci95_low_b": lob, "ci95_high_b": hib,
                     "n_pairs": len(d)})
        raw_p.append(p)
    corrected = multipletests(raw_p, alpha=0.05, method="holm")[1] if raw_p else []
    for row, pc in zip(rows, corrected):
        row["p_holm"] = float(pc)
    return {"friedman": friedman, "pairwise": pd.DataFrame(rows)}
