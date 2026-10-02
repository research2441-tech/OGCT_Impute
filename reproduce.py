"""End-to-end reproducibility driver for OGCT-Impute.

The script intentionally computes manuscript tables from model predictions; no
paper result is hard-coded. The test partition is isolated until final
benchmarking, and PSO fitness is evaluated only on validation data.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Dict, Callable

import numpy as np
import pandas as pd
import torch
import yaml
from scipy.signal import welch

from data_loader import load_dataset, subject_disjoint_split
from preprocessing import preprocess_partition, resample_eeg, fit_and_apply_normalization
from ogct_model import OGCTImpute
from losses import LossWeights
from train import fit_model, set_seed, resolve_device
from pso_optimizer import ParticleSwarmOptimizer
from evaluate import evaluate_condition, predict_model, aggregate_results
from baselines import (
    linear_interpolate, spline_interpolate, knn_impute,
    AutoencoderImputer, RNNImputer, ConvImputer, TransformerImputer,
    CNNLSTMImputer, CNNTransformerImputer,
)
from statistical_analysis import paired_method_comparison
from missingness import make_mask
from downstream_evaluation import evaluate_classifier


def read_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration file must contain a YAML mapping.")
    return cfg


def model_from_cfg(cfg: Dict) -> OGCTImpute:
    m = cfg["model"]
    return OGCTImpute(
        d_model=int(m["d_model"]),
        cnn_kernel_size=int(m["cnn_kernel_size"]),
        cnn_layers=int(m.get("cnn_layers", 3)),
        temporal_heads=int(m["temporal_heads"]),
        temporal_layers=int(m["temporal_layers"]),
        channel_heads=int(m.get("channel_heads", 4)),
        dropout=float(m["dropout"]),
        ff_multiplier=int(m.get("ff_multiplier", 4)),
    )


def weights_from_cfg(cfg: Dict) -> LossWeights:
    w = cfg["loss"]
    return LossWeights(float(w["reconstruction"]), float(w["temporal"]), float(w["spectral"]), float(w["channel"]))


def apply_candidate(base: Dict, cand: Dict) -> Dict:
    cfg = copy.deepcopy(base)
    cfg["model"]["cnn_kernel_size"] = int(cand["cnn_kernel_size"])
    cfg["model"]["d_model"] = int(cand["d_model"])
    cfg["model"]["temporal_heads"] = int(cand["temporal_heads"])
    cfg["model"]["channel_heads"] = int(cand["channel_heads"])
    cfg["model"]["temporal_layers"] = int(cand["temporal_layers"])
    cfg["model"]["dropout"] = float(cand["dropout"])
    cfg["training"]["learning_rate"] = float(cand["learning_rate"])
    cfg["loss"]["temporal"] = float(cand["loss_temporal"])
    cfg["loss"]["spectral"] = float(cand["loss_spectral"])
    cfg["loss"]["channel"] = float(cand["loss_channel"])
    return cfg


def prepare_data(cfg: Dict):
    bundle = load_dataset(cfg["data"])
    fs_orig = int(bundle.sampling_rate or cfg["data"].get("sampling_rate", 256))
    fs = int(cfg["data"].get("target_sampling_rate", fs_orig))
    X = resample_eeg(bundle.X, fs_orig, fs)

    tr_idx, va_idx, te_idx = subject_disjoint_split(
        bundle.subject_id,
        float(cfg["data"].get("train_ratio", 0.70)),
        float(cfg["data"].get("val_ratio", 0.15)),
        float(cfg["data"].get("test_ratio", 0.15)),
        int(cfg.get("seed", 42)),
    )
    p = cfg["preprocessing"]
    train = preprocess_partition(X[tr_idx], fs, float(p["bandpass_low_hz"]), float(p["bandpass_high_hz"]), float(p["notch_hz"]))
    val = preprocess_partition(X[va_idx], fs, float(p["bandpass_low_hz"]), float(p["bandpass_high_hz"]), float(p["notch_hz"]))
    test = preprocess_partition(X[te_idx], fs, float(p["bandpass_low_hz"]), float(p["bandpass_high_hz"]), float(p["notch_hz"]))
    train, val, test, normalizer = fit_and_apply_normalization(train, val, test, float(p.get("robust_clip_std", 8.0)))
    y = bundle.y
    labels = None if y is None else (y[tr_idx], y[va_idx], y[te_idx])
    split_info = {
        "sampling_rate": fs,
        "n_channels": int(train.shape[1]),
        "window_samples": int(train.shape[2]),
        "n_train_segments": int(len(train)),
        "n_val_segments": int(len(val)),
        "n_test_segments": int(len(test)),
        "n_train_subjects": int(len(np.unique(bundle.subject_id[tr_idx]))),
        "n_val_subjects": int(len(np.unique(bundle.subject_id[va_idx]))),
        "n_test_subjects": int(len(np.unique(bundle.subject_id[te_idx]))),
    }
    return train, val, test, labels, split_info


def run_pso(cfg: Dict, train: np.ndarray, val: np.ndarray, outdir: Path) -> Dict:
    pso_cfg = cfg["pso"]
    if not pso_cfg.get("enabled", True):
        return cfg

    def objective(candidate: Dict) -> float:
        local = apply_candidate(cfg, candidate)
        model = model_from_cfg(local)
        result = fit_model(
            model, train, val, local, weights_from_cfg(local),
            checkpoint=None,
            epochs_override=int(pso_cfg.get("fitness_epochs", 12)),
            verbose=False,
        )
        # Free accelerator memory between particles.
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result.best_val

    opt = ParticleSwarmOptimizer(
        pso_cfg["bounds"],
        population_size=int(pso_cfg["population_size"]),
        iterations=int(pso_cfg["iterations"]),
        inertia=float(pso_cfg["inertia"]),
        cognitive=float(pso_cfg["cognitive"]),
        social=float(pso_cfg["social"]),
        seed=int(cfg.get("seed", 42)),
        tolerance=float(pso_cfg.get("tolerance", 1e-4)),
        patience=int(pso_cfg.get("patience", 3)),
    )
    best, score, history = opt.optimize(objective)
    pd.DataFrame(history).to_csv(outdir / "pso_history.csv", index=False)
    with open(outdir / "pso_best.json", "w", encoding="utf-8") as f:
        json.dump({"validation_fitness": score, "parameters": best}, f, indent=2)
    return apply_candidate(cfg, best)


def train_named_model(name: str, model: torch.nn.Module, train: np.ndarray, val: np.ndarray, cfg: Dict, outdir: Path, full_loss: bool = False):
    weights = weights_from_cfg(cfg) if full_loss else LossWeights(1.0, 0.0, 0.0, 0.0)
    start = time.perf_counter()
    result = fit_model(model, train, val, cfg, weights, checkpoint=str(outdir / f"{name}.pt"), verbose=True)
    elapsed = time.perf_counter() - start
    pd.DataFrame(result.history).to_csv(outdir / f"{name}_history.csv", index=False)
    return model, elapsed, result


def classical_predictor(fn: Callable):
    return lambda incomplete, mask: fn(incomplete, mask)


def model_predictor(model, cfg):
    return lambda incomplete, mask: predict_model(model, incomplete, mask, cfg.get("device", "auto"))


def evaluate_overall(methods: Dict[str, Callable], test: np.ndarray, cfg: Dict, fs: float) -> pd.DataFrame:
    rows = []
    repetitions = int(cfg.get("evaluation", {}).get("repetitions", 5))
    kind = str(cfg.get("evaluation", {}).get("overall_missingness", "mixed"))
    ratio = float(cfg.get("evaluation", {}).get("overall_ratio", 0.30))
    for name, predictor in methods.items():
        for rep in range(repetitions):
            d = evaluate_condition(test, predictor, kind, ratio, fs, int(cfg.get("seed", 42)), rep)
            d["method"] = name
            rows.append(d)
    return pd.concat(rows, ignore_index=True)


def evaluate_ogct_robustness(predictor, test: np.ndarray, cfg: Dict, fs: float) -> pd.DataFrame:
    pieces = []
    seed = int(cfg.get("seed", 42))
    reps = int(cfg.get("evaluation", {}).get("repetitions", 5))
    for kind in cfg["missingness"]["types"]:
        for ratio in cfg["missingness"]["ratios"]:
            for rep in range(reps):
                d = evaluate_condition(test, predictor, kind, float(ratio), fs, seed, rep)
                d["method"] = "OGCT-Impute"
                pieces.append(d)
    for sec in cfg["missingness"].get("segment_lengths_seconds", []):
        for rep in range(reps):
            d = evaluate_condition(test, predictor, "segment", 0.30, fs, seed, rep, segment_seconds=float(sec))
            d["method"] = "OGCT-Impute"
            pieces.append(d)
    for n_ch in cfg["missingness"].get("channel_counts", []):
        if int(n_ch) >= test.shape[1]:
            continue
        for rep in range(reps):
            d = evaluate_condition(test, predictor, "channel", float(n_ch) / test.shape[1], fs, seed, rep, n_channels=int(n_ch))
            d["method"] = "OGCT-Impute"
            pieces.append(d)
    return pd.concat(pieces, ignore_index=True)


def spectral_band_table(model, test: np.ndarray, cfg: Dict, fs: float) -> pd.DataFrame:
    bands = {"Delta": (0.5,4), "Theta": (4,8), "Alpha": (8,13), "Beta": (13,30), "Gamma": (30,45)}
    originals = {k: [] for k in bands}; imputeds = {k: [] for k in bands}
    seed = int(cfg.get("seed", 42))
    for i, target in enumerate(test):
        rng = np.random.default_rng(seed + i * 1223)
        mask = make_mask(target, "mixed", 0.30, rng, fs)
        pred = predict_model(model, target * mask, mask, cfg.get("device", "auto"))
        f, po = welch(target, fs=fs, axis=-1, nperseg=min(256, target.shape[-1]))
        _, pi = welch(pred, fs=fs, axis=-1, nperseg=min(256, pred.shape[-1]))
        for name, (lo, hi) in bands.items():
            idx = (f >= lo) & (f < hi)
            originals[name].append(float(np.mean(np.trapezoid(po[:, idx], f[idx], axis=-1))))
            imputeds[name].append(float(np.mean(np.trapezoid(pi[:, idx], f[idx], axis=-1))))
    rows = []
    for name in bands:
        o = float(np.mean(originals[name])); p = float(np.mean(imputeds[name]))
        rows.append({"eeg_band": name, "original_power": o, "imputed_power": p, "relative_error_percent": abs(p-o)/(abs(o)+1e-12)*100})
    return pd.DataFrame(rows)




def impute_dataset_with_fixed_masks(predictor, X: np.ndarray, kind: str, ratio: float, fs: float, seed: int) -> np.ndarray:
    out = []
    for i, target in enumerate(X):
        rng = np.random.default_rng(seed + i * 7919)
        mask = make_mask(target, kind, ratio, rng, fs)
        out.append(predictor(target * mask, mask))
    return np.asarray(out, dtype=np.float32)


def run_ablation(train: np.ndarray, val: np.ndarray, test: np.ndarray, base_cfg: Dict, tuned_cfg: Dict, outdir: Path, fs: float) -> pd.DataFrame:
    c = train.shape[1]
    variants = []

    # 1) CNN only
    cnn, _, _ = train_named_model("ablation_cnn", ConvImputer(c), train, val, base_cfg, outdir, full_loss=False)
    variants.append(("CNN only", model_predictor(cnn, base_cfg)))

    # 2) Transformer only
    tr, _, _ = train_named_model("ablation_transformer", TransformerImputer(c), train, val, base_cfg, outdir, full_loss=False)
    variants.append(("Transformer only", model_predictor(tr, base_cfg)))

    # 3) CNN + Transformer
    ct, _, _ = train_named_model("ablation_cnn_transformer", CNNTransformerImputer(c), train, val, base_cfg, outdir, full_loss=False)
    variants.append(("CNN + Transformer", model_predictor(ct, base_cfg)))

    # 4) CNN + Transformer + Channel Attention (fixed, non-PSO configuration)
    ca = model_from_cfg(base_cfg)
    ca, _, _ = train_named_model("ablation_channel_attention", ca, train, val, base_cfg, outdir, full_loss=False)
    variants.append(("CNN + Transformer + Channel Attention", model_predictor(ca, base_cfg)))

    # 5) CNN + Transformer + Optimization (PSO configuration, channel attention removed)
    opt_model = model_from_cfg(tuned_cfg)
    opt_model.channel_attention = torch.nn.Identity()
    opt_model, _, _ = train_named_model("ablation_optimized_no_channel", opt_model, train, val, tuned_cfg, outdir, full_loss=False)
    variants.append(("CNN + Transformer + Optimization", model_predictor(opt_model, tuned_cfg)))

    # 6) Full OGCT-Impute (optimized configuration + channel attention + all losses)
    full = model_from_cfg(tuned_cfg)
    full, _, _ = train_named_model("ablation_full_ogct", full, train, val, tuned_cfg, outdir, full_loss=True)
    variants.append(("Full OGCT-Impute", model_predictor(full, tuned_cfg)))

    rows = []
    for name, predictor in variants:
        d = evaluate_condition(test, predictor, "mixed", 0.30, fs, int(base_cfg.get("seed", 42)), 0)
        d["model_configuration"] = name
        rows.append(d)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(outdir / "table6_ablation_per_segment.csv", index=False)
    summary = raw.groupby("model_configuration")[["mae", "rmse", "correlation"]].agg(["mean", "std"]).reset_index()
    summary.columns = ["_".join([str(x) for x in col if x != ""]) if isinstance(col, tuple) else col for col in summary.columns]
    summary.to_csv(outdir / "table6_ablation_summary.csv", index=False)
    return raw


def main():
    ap = argparse.ArgumentParser(description="Reproduce OGCT-Impute experiments from an EEG dataset.")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--data", default=None, help="Override data.path in config.yaml")
    ap.add_argument("--skip-pso", action="store_true")
    ap.add_argument("--full-baselines", action="store_true", help="Train neural baseline models in addition to classical baselines.")
    ap.add_argument("--ablation", action="store_true", help="Train manuscript-oriented ablation variants and write Table 6 evidence.")
    args = ap.parse_args()

    cfg = read_config(args.config)
    if args.data:
        cfg["data"]["path"] = args.data
    if args.skip_pso:
        cfg["pso"]["enabled"] = False
    set_seed(int(cfg.get("seed", 42)))
    outdir = Path(cfg.get("output_dir", "results")); outdir.mkdir(parents=True, exist_ok=True)

    train, val, test, labels, info = prepare_data(cfg)
    with open(outdir / "dataset_split.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)
    fs = float(info["sampling_rate"])

    tuned_cfg = run_pso(cfg, train, val, outdir)
    with open(outdir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(tuned_cfg, f, sort_keys=False)

    ogct = model_from_cfg(tuned_cfg)
    ogct, training_seconds, result = train_named_model("ogct_impute", ogct, train, val, tuned_cfg, outdir, full_loss=True)

    methods: Dict[str, Callable] = {
        "Linear Interpolation": classical_predictor(linear_interpolate),
        "Spline Interpolation": classical_predictor(spline_interpolate),
        "KNN": classical_predictor(knn_impute),
        "OGCT-Impute": model_predictor(ogct, tuned_cfg),
    }

    if args.full_baselines:
        c = train.shape[1]
        baseline_specs = {
            "Autoencoder": AutoencoderImputer(c),
            "LSTM": RNNImputer(c, kind="lstm"),
            "GRU": RNNImputer(c, kind="gru"),
            "CNN": ConvImputer(c),
            "Transformer": TransformerImputer(c),
            "CNN-LSTM": CNNLSTMImputer(c),
            "CNN-Transformer": CNNTransformerImputer(c),
        }
        for name, model in baseline_specs.items():
            key = name.lower().replace("-", "_").replace(" ", "_")
            trained, _, _ = train_named_model(key, model, train, val, tuned_cfg, outdir, full_loss=False)
            methods[name] = model_predictor(trained, tuned_cfg)

    overall = evaluate_overall(methods, test, tuned_cfg, fs)
    overall.to_csv(outdir / "table1_overall_per_segment.csv", index=False)
    aggregate_results(overall).to_csv(outdir / "table1_overall_summary.csv", index=False)

    robust = evaluate_ogct_robustness(methods["OGCT-Impute"], test, tuned_cfg, fs)
    robust.to_csv(outdir / "robustness_per_segment.csv", index=False)
    aggregate_results(robust).to_csv(outdir / "robustness_summary.csv", index=False)

    spectral_band_table(ogct, test, tuned_cfg, fs).to_csv(outdir / "table5_frequency_preservation.csv", index=False)

    # Table 7-style downstream evaluation is generated when class labels are supplied.
    if labels is not None:
        train_y, _, test_y = labels
        downstream_rows = [{"input_data": "Complete EEG", **evaluate_classifier(train, train_y, test, test_y, fs)}]
        kind = str(tuned_cfg.get("evaluation", {}).get("overall_missingness", "mixed"))
        ratio = float(tuned_cfg.get("evaluation", {}).get("overall_ratio", 0.30))
        for name, predictor in methods.items():
            imputed = impute_dataset_with_fixed_masks(predictor, test, kind, ratio, fs, int(tuned_cfg.get("seed", 42)))
            downstream_rows.append({"input_data": name, **evaluate_classifier(train, train_y, imputed, test_y, fs)})
        pd.DataFrame(downstream_rows).to_csv(outdir / "table7_downstream_classification.csv", index=False)

    if args.ablation:
        run_ablation(train, val, test, cfg, tuned_cfg, outdir, fs)

    stats = paired_method_comparison(overall, metric="rmse", id_col="segment", method_col="method")
    if stats["friedman"] is not None:
        stats["friedman"].to_csv(outdir / "statistics_friedman_rmse.csv", index=False)
    stats["pairwise"].to_csv(outdir / "statistics_pairwise_rmse.csv", index=False)

    params = sum(p.numel() for p in ogct.parameters())
    with open(outdir / "computational_summary.json", "w", encoding="utf-8") as f:
        json.dump({"parameters": params, "training_seconds": training_seconds, "best_validation_loss": result.best_val, "best_epoch": result.best_epoch}, f, indent=2)

    print(f"Reproduction completed. Outputs are in: {outdir.resolve()}")
    print("No manuscript result values were hard-coded; all CSV values originate from the current run.")


if __name__ == "__main__":
    main()
