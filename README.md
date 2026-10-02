# OGCT-Impute

**Optimization-Guided CNN-Transformer Network for Robust Multichannel EEG Data Imputation under Realistic Missingness Conditions**

This repository provides a flat, reproducible implementation of **OGCT-Impute**, the multichannel EEG reconstruction framework described in the accompanying manuscript. The implementation combines a shared one-dimensional temporal CNN, long-range Transformer encoding, inter-channel self-attention, a mask-preserving reconstruction decoder, and Particle Swarm Optimization (PSO). The code evaluates random-sample loss, continuous temporal gaps, complete-channel loss, and mixed temporal-spatial missingness.

The repository does **not** hard-code the numerical values shown in manuscript tables. All evaluation CSV files are generated from predictions produced by the current run. This is important because the manuscript states that the displayed experimental values are an example template that must be replaced by actual measurements after executing the model.

## Repository files

The project is deliberately flat: no source folders, no nested experiment folders, and no license file.

| File | Purpose |
|---|---|
| `README.md` | Reproduction protocol and reviewer-facing documentation |
| `requirements.txt` | Python environment |
| `config.yaml` | Central experiment and PSO configuration |
| `reproduce.py` | End-to-end experiment driver |
| `data_loader.py` | NPZ/EDF loading and subject-disjoint splitting |
| `preprocessing.py` | Filtering, resampling, and train-only normalization |
| `missingness.py` | Random, segment, channel, and mixed missingness |
| `ogct_model.py` | CNN + Transformer + channel attention + decoder |
| `losses.py` | Reconstruction, temporal, spectral, and inter-channel losses |
| `pso_optimizer.py` | Validation-only Particle Swarm Optimization |
| `baselines.py` | Interpolation, KNN, AE, RNN, CNN, Transformer, and hybrid baselines |
| `train.py` | Deterministic training and early stopping |
| `evaluate.py` | MAE, RMSE, correlation, SNR, and PSD correlation |
| `downstream_evaluation.py` | Leakage-safe downstream EEG classification utilities |
| `statistical_analysis.py` | Friedman/Wilcoxon/Holm and effect-size analysis |
| `plot_results.py` | Publication-quality figures generated from results |
| `test_pipeline.py` | Fast integrity tests using synthetic EEG |

## Method-to-code mapping

The implementation follows the manuscript pipeline:

1. EEG preprocessing -> `preprocessing.py`
2. Availability/missingness mask -> `missingness.py`
3. CNN local temporal representation -> `ogct_model.py`
4. Transformer long-range temporal modelling -> `ogct_model.py`
5. Inter-channel representation -> `ogct_model.py`
6. PSO model configuration -> `pso_optimizer.py`
7. Mask-preserving reconstruction -> `ogct_model.py`
8. Multi-component training objective -> `losses.py`
9. Reconstruction and spectral evaluation -> `evaluate.py`
10. Paired statistical testing -> `statistical_analysis.py`

The final reconstruction obeys the manuscript's preservation rule:

```text
final = observed_mask * measured_EEG + missing_mask * network_prediction
```

Consequently, samples marked as observed are copied unchanged into the output. `test_pipeline.py` explicitly tests this property.

## Data format

### Recommended NPZ input

For a controlled benchmark, provide a NumPy archive containing complete EEG windows:

```python
np.savez(
    "data.npz",
    X=X,                    # float32 [segments, channels, time]
    subject_id=subject_id,  # one subject identifier per segment
    y=y,                    # optional downstream class label per segment
    sampling_rate=256,
)
```

The subject identifier is mandatory for a rigorous subject-disjoint split. The same person's windows are never allowed to occur in both training and test partitions.

### EDF input

`data.path` may alternatively point to a directory containing EDF files. MNE-Python is used for loading. The default EDF loader treats each file's parent directory as the subject identity and requires a consistent montage. For a publication benchmark, dataset-specific channel alignment should be defined before combining recordings acquired with different montages.

No patient recordings or proprietary EEG data are included in this repository.

## Installation

Python 3.11 is recommended.

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Integrity test

Run this before using a real dataset:

```bash
python test_pipeline.py
```

The test verifies:

- four-dimensional internal feature flow is shape-consistent;
- missingness masks are deterministic for a fixed seed;
- the network produces finite reconstruction loss;
- observed EEG samples are not changed by reconstruction;
- evaluation metrics execute correctly.

## Full reproduction

Set `data.path` in `config.yaml`, then run:

```bash
python reproduce.py --config config.yaml
```

or override the dataset path:

```bash
python reproduce.py --config config.yaml --data /path/to/data.npz
```

This command performs:

```text
load EEG
 -> subject-disjoint 70/15/15 partition
 -> filtering/resampling
 -> training-only normalization
 -> PSO on training + validation only
 -> final OGCT-Impute training
 -> untouched test evaluation
 -> realistic missingness experiments
 -> frequency preservation analysis
 -> paired statistical testing
 -> reproducibility outputs
```

To run without the expensive PSO stage:

```bash
python reproduce.py --config config.yaml --skip-pso
```

To additionally train the neural comparison models:

```bash
python reproduce.py --config config.yaml --full-baselines
```

To execute the manuscript-oriented ablation sequence (CNN only, Transformer only, CNN–Transformer, channel-attention variant, PSO-optimized hybrid, and full OGCT-Impute):

```bash
python reproduce.py --config config.yaml --ablation
```

## Leakage controls

Several safeguards are intentionally built into the code.

**Subject-disjoint splitting.** Splitting occurs on unique subject identifiers rather than randomly splitting EEG windows.

**Training-only normalization.** Mean and standard deviation are estimated from the training partition and then applied unchanged to validation and test data.

**Validation-only PSO.** Candidate architecture configurations are selected using validation reconstruction loss. The test partition is not available to the optimizer.

**Fixed final test masks.** Repeated test conditions are seeded reproducibly. Methods compared within the same condition therefore receive the same corruption pattern.

**No test-guided early stopping.** Early stopping monitors validation loss only.

## Missingness experiments

The code supports the four scenarios described in the manuscript.

### Random missingness

Individual time-channel observations are removed throughout the segment.

### Continuous-segment missingness

Contiguous temporal intervals are removed. The default manuscript-oriented durations are:

```text
0.5 s, 1.0 s, 2.0 s, 3.0 s, 5.0 s
```

### Missing-channel reconstruction

One or more complete electrode channels are removed. By default the requested counts are:

```text
1, 2, 4, 6, 8 channels
```

Counts that are not valid for a particular montage are automatically skipped.

### Mixed missingness

Channel loss and continuous temporal loss are combined to emulate simultaneous spatial and temporal acquisition failure.

The default missingness-ratio sweep is:

```text
10%, 20%, 30%, 40%, 50%, 60%
```

## OGCT-Impute architecture

For each EEG channel, the incomplete normalized waveform and binary observation mask are stacked as two input features. A shared temporal CNN learns local morphology without assigning a separate convolutional parameter set to every electrode. The encoded sequence is passed through a positional encoding and a Transformer encoder for long-range temporal dependencies. The resulting representation is reorganized so that multi-head attention operates across channels at each temporal position. A reconstruction decoder then estimates channel-specific values.

This design is particularly relevant to complete-channel loss: the model can infer a missing electrode from channel context while retaining long-range information within the recording.

## Multi-component objective

Training combines four terms:

```text
L = lambda_rec * L_rec
  + lambda_temporal * L_temporal
  + lambda_spectral * L_spectral
  + lambda_channel * L_channel
```

`L_rec` is evaluated over unavailable observations. `L_temporal` penalizes disagreement in first-order temporal differences around missing regions. `L_spectral` matches log power spectra. `L_channel` matches channel covariance structure. Their weights can be optimized by PSO.

## Particle Swarm Optimization

Each PSO particle represents a candidate containing:

- CNN kernel size;
- Transformer embedding dimension;
- number of temporal attention heads;
- Transformer depth;
- learning rate;
- dropout;
- temporal-loss weight;
- spectral-loss weight;
- inter-channel-loss weight.

The implementation uses inertia, cognitive, and social terms, explicit velocity clipping, bounded positions, deterministic initialization, early termination, and validation-only fitness evaluation. Search ranges and swarm settings are recorded in `config.yaml`, while each evaluated particle is written to `results/pso_history.csv`.

## Generated evidence

A successful run produces machine-readable evidence such as:

```text
results/dataset_split.json
results/resolved_config.yaml
results/pso_history.csv
results/pso_best.json
results/ogct_impute.pt
results/ogct_impute_history.csv
results/table1_overall_per_segment.csv
results/table1_overall_summary.csv
results/robustness_per_segment.csv
results/robustness_summary.csv
results/table5_frequency_preservation.csv
results/table7_downstream_classification.csv   # when class labels y are available
results/table6_ablation_summary.csv            # when --ablation is requested
results/statistics_friedman_rmse.csv
results/statistics_pairwise_rmse.csv
results/computational_summary.json
```

These outputs should be used to populate the final manuscript tables. They should not be replaced by manually entered values.

## Metrics

The implementation reports:

- Mean Absolute Error (MAE);
- Root Mean Square Error (RMSE);
- Pearson correlation;
- signal-to-noise ratio (SNR);
- power spectral density (PSD) correlation.

The statistical module performs paired comparisons across the same reconstructed segments. When at least three methods are present, a Friedman omnibus test is computed. Pairwise Wilcoxon signed-rank tests are subsequently corrected using Holm's procedure. The output also reports rank-biserial effect sizes and confidence intervals.

## Downstream EEG analysis

`downstream_evaluation.py` contains a reproducible downstream classifier based on EEG band-power features and logistic regression. It is intentionally separate from the reconstruction network so that the usefulness of imputed EEG can be assessed with an independent model. If class labels are present in the NPZ archive, the utility of complete and reconstructed EEG can be compared using accuracy, precision, recall, and F1-score.

## Reproducibility notes for the manuscript

The final paper should report the exact dataset release, subject count, montage, sampling frequency, channel alignment policy, window length, filtering settings, random seed, number of repeated masks, hardware, software versions, PSO population size, iterations, stopping criterion, and the final PSO-selected configuration. Statistical results should be calculated from independent subject- or segment-level paired observations rather than treating individual EEG samples as independent measurements.

The present repository contains no DOI declaration. After the implementation and its outputs have been verified, the exact version submitted with the revised manuscript can be archived in a DOI-assigning repository as requested by the editor.
