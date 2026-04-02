# additional_R50 Scratch-UniSub Follow-Up

This folder contains the later 9-model ResNet-50 analysis package. It used to live under `ResNet50/` and is now renamed `additional_R50/` so it is clearly separated from the scratch ResNet-50 experiments in `CNN/`.

This is additional analysis. It includes:

- alpha-filtered layer diagnostics
- spatial-wise factorization for common `3x3` convolutions
- scree plots
- reconstruction summaries
- low-rank coefficient rescue

For the consolidated overview, start with [`../Main_weight_analysis/additional_R50/README.md`](../Main_weight_analysis/additional_R50/README.md).

## What To Read

- Main summary: [results/scratch_alpha_summary.json](results/scratch_alpha_summary.json)
- Source model list: [results/scratch_source_models.csv](results/scratch_source_models.csv)
- Raw summary rows: [results/scratch_raw_mean_summary.csv](results/scratch_raw_mean_summary.csv)
- Raw per-target rows: [results/scratch_raw_operating_points.csv](results/scratch_raw_operating_points.csv)
- Coefficient calibration: [results/scratch_iid_coeff_calibration.csv](results/scratch_iid_coeff_calibration.csv)
- Coefficient-plus-head calibration: [results/scratch_ood_coeff_head_calibration.csv](results/scratch_ood_coeff_head_calibration.csv)
- Retained layers: [results/scratch_retained_layers.csv](results/scratch_retained_layers.csv)
- Factorized layers: [results/scratch_factorized_layers.csv](results/scratch_factorized_layers.csv)
- Scree figures: [figures/mean_scree.pdf](figures/mean_scree.pdf), [figures/channel_mean_scree.pdf](figures/channel_mean_scree.pdf)
- Alpha large-vs-small plot: [figures/alpha_large_vs_small_delta.pdf](figures/alpha_large_vs_small_delta.pdf)

## Setup

- Source family size: `9`
- Datasets in the source pool: `CIFAR10`, `CIFAR100`, `SVHN`, `ImageNet-1k`, `ImageNet-21k`, `iNaturalist`, `Places365`, `RadImageNet`, `VGGFace2`
- Common layers: `52`
- Retained layers: `52`
- Spatially factorized `3x3` layers: `10`

## Alpha Metric

The alpha metric is used here as a layer-selection heuristic based on the spectral shape of each common layer across the source family. We fit a power-law exponent per layer and prefer layers whose alpha statistics are stable and stay in a usable range. That gives a better shared-subspace candidate set than treating every layer equally.

## Raw Reconstruction Results

Mean top-1 over the five evaluated scratch targets:

| VT | UniSub mean top-1 | Delta | Savings % |
| --- | --- | --- | --- |
| `0.8` | `0.7826` | `-0.0274` | `8.42` |
| `0.9` | `0.8045` | `-0.0055` | `5.23` |
| `0.95` | `0.8085` | `-0.0015` | `2.78` |

## Minimal Reproduction Code

The wrapper scripts in [scripts/](scripts) show the exact command pattern used for this study.

- [scripts/run_scratch_pipeline.sh](scripts/run_scratch_pipeline.sh)
- [scripts/download_or_train_models.sh](scripts/download_or_train_models.sh)
- [scripts/compute_alpha_metrics.sh](scripts/compute_alpha_metrics.sh)
- [scripts/reconstruct_and_evaluate.sh](scripts/reconstruct_and_evaluate.sh)
- [scripts/low_rank_calibration.sh](scripts/low_rank_calibration.sh)
