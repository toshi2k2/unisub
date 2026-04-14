# GeomDist UniSub Public Reproduction

This folder packages the commands and validated snapshots needed to reproduce the GeomDist alpha-filtered UniSub analysis, including the lower-rank `80%` and `70%` ablations used in the paper appendix.

## What This Runs

The full workflow has up to four stages:

1. `geomdist_alpha_unisub.py`
   - discovers the compatible GeomDist checkpoint family
   - computes layer-wise alpha metrics
   - filters spectral layers with the alpha suitability rule
   - extracts the shared UniSub basis
   - reconstructs IID and OOD checkpoints
   - exports original / IID / OOD point clouds
   - computes Chamfer metrics
2. `geomdist_finalize_analysis.py`
   - writes `summary.json`
   - renders the qualitative paper figure and per-scene 3D views from the saved PLYs
3. `geomdist_coeff_finetune.py` (optional)
   - reuses the retained GeomDist UniSub basis
   - optimizes only the low-rank coefficient vector for selected IID or OOD targets
   - keeps the non-spectral/control state fixed
   - saves per-scene rescue checkpoints, convergence plots, and rescue figures
4. `geomdist_compose_paper_figures.py` (optional)
   - reuses saved point clouds
   - produces denser paper-facing figure panels
   - writes colored PLY exports for external 3D inspection

## Environment

This reproduction folder assumes the main GeomDist repo is available elsewhere and contains:

- `geomdist_alpha_unisub.py`
- `geomdist_finalize_analysis.py`
- `geomdist_coeff_finetune.py`
- `geomdist_compose_paper_figures.py`
- `mesh_utils.py`
- `geom_dist_ckpt/`
- the local `shapes/` assets needed for GT Chamfer on the reference-available subset

Validated Python packages in this environment:

- `python`
- `torch`
- `numpy`
- `matplotlib`
- optional: `scienceplots`

The wrapper sets:

- `OMP_NUM_THREADS=1`
- `MKL_NUM_THREADS=1`
- `OPENBLAS_NUM_THREADS=1`
- `MPLCONFIGDIR=<this folder>/.mplconfig`

Those settings are required on this host because the default threaded BLAS / NumPy interop path was unstable during plotting and metric evaluation.

## Quick Start

Set the main repo path and run:

```bash
cd unisub/GeomDist
export GEOMDIST_REPO_DIR=/path/to/GeomDist
bash run_public_pipeline.sh \
  --repo-dir "$GEOMDIST_REPO_DIR" \
  --device cuda:0 \
  --sample-points 16384 \
  --reference-points 16384 \
  --num-steps 64
```

Outputs go to:

- `results/analysis_alpha_unisub_v1`

If you only want the lightweight finalizer on an already-computed analysis root:

```bash
cd unisub/GeomDist
export GEOMDIST_REPO_DIR=/path/to/GeomDist
python "$GEOMDIST_REPO_DIR/geomdist_finalize_analysis.py" \
  --analysis-root results/analysis_alpha_unisub_v1 \
  --figure-models valley,wukong,tower,lamp \
  --figure-points 8000
```

Optional OOD coefficient rescue on the strongest OOD scenes:

```bash
cd unisub/GeomDist
export GEOMDIST_REPO_DIR=/path/to/GeomDist
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
MPLCONFIGDIR="$PWD/.mplconfig" \
python "$GEOMDIST_REPO_DIR/geomdist_coeff_finetune.py" \
  --analysis-root results/analysis_alpha_unisub_v1 \
  --output-root results/analysis_alpha_unisub_v1/coeff_finetune_ood_v1 \
  --scene-names valley,wukong \
  --protocol ood \
  --device cuda:0 \
  --num-updates 400 \
  --eval-every 25 \
  --batch-size 16384 \
  --train-pool-points 131072 \
  --sample-points 16384 \
  --reference-points 16384 \
  --num-steps 64 \
  --figure-points 28000
```

Optional dense paper figures and colored PLY exports:

```bash
cd unisub/GeomDist
export GEOMDIST_REPO_DIR=/path/to/GeomDist
MPLCONFIGDIR="$PWD/.mplconfig" \
python "$GEOMDIST_REPO_DIR/geomdist_compose_paper_figures.py" \
  --analysis-root results/analysis_alpha_unisub_v1 \
  --rescue-root results/analysis_alpha_unisub_v1/coeff_finetune_ood_v1 \
  --scene-names loong,archimedes \
  --protocols original,iid \
  --output-prefix results/analysis_alpha_unisub_v1/evaluation/paper_figures/dense_iid_best \
  --figure-points 28000

MPLCONFIGDIR="$PWD/.mplconfig" \
python "$GEOMDIST_REPO_DIR/geomdist_compose_paper_figures.py" \
  --analysis-root results/analysis_alpha_unisub_v1 \
  --rescue-root results/analysis_alpha_unisub_v1/coeff_finetune_ood_v1 \
  --scene-names valley,wukong \
  --protocols original,ood,ood_ft \
  --output-prefix results/analysis_alpha_unisub_v1/evaluation/paper_figures/dense_ood_rescue \
  --figure-points 28000
```

## Validated Snapshot

The validated run keeps `18/22` spectral layers with:

- `q_all = 0.1501039669601017`
- `q_arch = 0.197672229867697`

Rejected layers:

- `model.layers.1.1.weight`
- `model.layers.2.1.weight`
- `model.layers.3.1.weight`
- `model.layers.5.2.weight`

Global scree:

- `90%`: `16` components
- `95%`: `18` components
- `99%`: `19` components

Validated metric summary:

- original GT Chamfer mean: `0.05056`
- IID GT Chamfer mean: `0.08674`
- IID pair Chamfer mean: `0.02497`
- OOD GT Chamfer mean: `0.42622`
- OOD pair Chamfer mean: `0.45132`

Best OOD reference-available examples in the validated run:

- `valley`
- `wukong`
- `tower`
- `lamp`

Validated OOD rescue summary:

- `valley`: GT Chamfer `0.24099 -> 0.23448`, pairwise Chamfer `0.24010 -> 0.23330`
- `wukong`: no improvement beyond the projected initialization

Validated dense figures:

- `validated_summary_model_comparisons.*`
- `validated_dense_iid_best.*`
- `validated_dense_ood_rescue.*`

## Lower-rank Snapshot

This package also includes the appendix-style lower-rank snapshots:

- `validated_rank80_summary.json`
- `validated_rank80_paper_metric_summary.csv`
- `validated_rank80_ood_rescue_summary.json`
- `validated_rank70_summary.json`
- `validated_rank70_paper_metric_summary.csv`
- `validated_rank70_ood_rescue_summary.json`

Compact paper-facing tables:

- `validated_compact_paper_metrics.csv`
- `validated_compact_ood_rescue.csv`
- `validated_paper_tables.tex`

The direct-projection comparison is:

- baseline `95%`, rank `18`, full savings `8.10%`
  - IID GT CD `0.08674`, OOD GT CD `0.42622`, OOD pair CD `0.45132`
- `80%`, rank `14`, full savings `24.30%`
  - IID GT CD `0.12140`, OOD GT CD `0.42607`, OOD pair CD `0.44651`
- `70%`, rank `12`, full savings `32.40%`
  - IID GT CD `0.13679`, OOD GT CD `0.43190`, OOD pair CD `0.44885`

So `80%` is the best compressed setting overall: it preserves OOD quality much better than `70%` while still giving a meaningful storage reduction relative to the validated `95%` run.

## Files Included Here

- `run_public_pipeline.sh`
- `validated_summary.json`
- `validated_retained_layer_summary.csv`
- `validated_q_threshold_sweep.csv`
- `validated_paper_metric_summary.csv`
- `validated_memory_savings_summary.csv`
- `validated_summary_model_comparisons.png`
- `validated_summary_model_comparisons.svg`
- `validated_dense_iid_best.png`
- `validated_dense_iid_best.svg`
- `validated_dense_ood_rescue.png`
- `validated_dense_ood_rescue.svg`
- `validated_compact_paper_metrics.csv`
- `validated_compact_ood_rescue.csv`
- `validated_paper_tables.tex`
- `validated_rank80_summary.json`
- `validated_rank80_paper_metric_summary.csv`
- `validated_rank80_ood_rescue_summary.json`
- `validated_rank70_summary.json`
- `validated_rank70_paper_metric_summary.csv`
- `validated_rank70_ood_rescue_summary.json`

## Caveats

- GT Chamfer is only computed for the `10` models whose reference mesh or primitive exists locally.
- The texture-conditioned `spot_color` checkpoint is excluded from the common family analysis.
- This package does not download checkpoints; it expects the GeomDist repo to already contain the local checkpoint family under `geom_dist_ckpt/`.
