# SDFStudio UniSub Reproduction

This folder is a public-facing companion for reproducing the alpha-filtered UniSub analysis on the shared-init NeUS-facto plain-MLP benchmark in `sdfstudio/`.

It assumes:

- the main `sdfstudio/` repo is available locally
- the shared-init NeUS-facto plain-MLP training runs already exist, or you will create them with `scripts/neus_plainmlp_shared_init.py`
- you are using the same Python environment that already runs `sdfstudio/`

This companion folder now also includes a compact snapshot of the validated run:

- `validated_summary.json`
- `validated_retained_layer_summary.csv`
- `validated_layer_basis_summary.csv`
- `validated_paper_metric_summary.csv`
- `validated_q_threshold_sweep.csv`
- `validated_selected_scene_alpha_summary.csv`
- `figures/*.png`

## Main Repo Requirement

Set the main repo location:

```bash
export SDFSTUDIO_REPO_DIR=/path/to/sdfstudio
cd "$SDFSTUDIO_REPO_DIR"
```

This package is intentionally lightweight. The actual code still lives in the main repo, especially:

- `scripts/neus_plainmlp_shared_init.py`
- `scripts/neus_plainmlp_alpha_unisub.py`
- `scripts/merge_neus_plainmlp_eval_shards.py`
- `run_eval_shards.sh`

## Data And Training

If you already have the completed shared-init benchmark at:

```bash
/mnt/data0/prakhar/sdfstudio_uwsh
```

you can skip directly to the alpha/UniSub analysis section.

If not, first:

1. download the SDFStudio datasets
2. run `scripts/neus_plainmlp_shared_init.py` to train the 32 scene-specific plain-MLP models from the same shared initialization

The alpha/UniSub analysis expects the completed run root to contain:

- `run_manifest.json`
- per-scene `checkpoints/model_final.pt`
- per-scene `vectors/layouts.json`

## Environment Setup

This companion folder does not define a separate environment from the main repo. Use the same environment that already runs `sdfstudio/`.

Minimal packages used directly by the public commands are listed in `requirements_public.txt`.

## One-Command Wrapper

The simplest supported path is:

```bash
cd /path/to/public_sdfstudio_unisub
bash run_public_pipeline.sh \
  --repo-dir "$SDFSTUDIO_REPO_DIR" \
  --device cuda:0
```

The wrapper defaults to the validated settings:

- `--alpha-min-fraction-in-range 0.3`
- `--eval-max-images 4`
- `--figure-images-per-scene 4`
- output root `/mnt/data0/prakhar/sdfstudio_uwsh/analysis_alpha_unisub_v1`

Useful variants:

```bash
# Alpha-only, without checkpoint reconstruction or image evaluation
bash run_public_pipeline.sh \
  --repo-dir "$SDFSTUDIO_REPO_DIR" \
  --device cpu \
  --skip-reconstruction \
  --skip-eval
```

```bash
# Quick single-scene smoke
bash run_public_pipeline.sh \
  --repo-dir "$SDFSTUDIO_REPO_DIR" \
  --run-root /mnt/data0/prakhar/sdfstudio_uwsh \
  --analysis-root /tmp/sdfstudio_alpha_unisub_smoke \
  --device cpu \
  --eval-max-images 1 \
  --figure-images-per-scene 1 \
  --scene-keys dtu/scan105
```

## Direct Commands

Main validated run:

```bash
SDFSTUDIO_USE_VENDOR_PY=1 python scripts/neus_plainmlp_alpha_unisub.py \
  --run-root /mnt/data0/prakhar/sdfstudio_uwsh \
  --analysis-root /mnt/data0/prakhar/sdfstudio_uwsh/analysis_alpha_unisub_v1 \
  --alpha-min-fraction-in-range 0.3 \
  --eval-max-images 4 \
  --figure-images-per-scene 4 \
  --device cuda:0
```

Alpha-only dry run:

```bash
SDFSTUDIO_USE_VENDOR_PY=1 python scripts/neus_plainmlp_alpha_unisub.py \
  --run-root /mnt/data0/prakhar/sdfstudio_uwsh \
  --analysis-root /mnt/data0/prakhar/sdfstudio_uwsh/analysis_alpha_unisub_v1 \
  --alpha-min-fraction-in-range 0.3 \
  --skip-reconstruction \
  --skip-eval
```

Eval-only sharded rerun after reconstructions already exist:

```bash
/mnt/data1/prakhar/sdfstudio/run_eval_shards.sh \
  /mnt/data1/prakhar/sdfstudio \
  /mnt/data0/prakhar/sdfstudio_uwsh/analysis_alpha_unisub_v1 \
  /mnt/data0/prakhar/sdfstudio_uwsh
```

## What The Script Does

`scripts/neus_plainmlp_alpha_unisub.py`:

1. loads the completed plain-MLP shared-init runs
2. computes per-layer spectral alpha statistics on the common matrix-valued `weight_v` layers
3. computes `q_l`, `q_all`, and `q_arch`
4. keeps the alpha-qualified spectral layers and carries the low-dimensional control layers (`bias`, `weight_g`, `variance`)
5. fits per-layer UniSub bases
6. reconstructs IID and OOD checkpoints
7. evaluates original vs reconstructed models on held-out eval views
8. writes aggregate scree plots, alpha plots, CSV summaries, reconstructed checkpoints, and qualitative figures

## Validated Run Summary

The validated run analyzed all `32` completed scenes from:

- `dtu` (`15`)
- `replica` (`8`)
- `scannet` (`4`)
- `tanks-and-temple` (`4`)
- `tanks-and-temple-highres` (`1`)

Validated alpha summary:

- `q_all = 0.13890791663134228`
- `q_arch = 0.19163092636979379`
- `spectral_layer_count = 5`
- `retained_spectral_layer_count = 4`
- `control_layer_count = 14`
- `retained_total_layer_count = 18`

Retained spectral layers:

- `glin0.weight_v`
- `glin1.weight_v`
- `clin0.weight_v`
- `clin1.weight_v`

Excluded spectral layer:

- `glin2.weight_v`

Retained control layers:

- `glin0.bias`
- `glin0.weight_g`
- `glin1.bias`
- `glin1.weight_g`
- `glin2.bias`
- `glin2.weight_g`
- `deviation_network.variance`
- `clin0.bias`
- `clin0.weight_g`
- `clin1.bias`
- `clin1.weight_g`
- `clin2.bias`
- `clin2.weight_g`
- `clin2.weight_v`

Aggregate scree summary over retained layers:

- `90%` variance at `172` components
- `95%` variance at `207` components
- `99%` variance at `250` components
- top-1 explained variance `0.2820`
- top-3 cumulative explained variance `0.3721`
- top-5 cumulative explained variance `0.3891`

The spectral-only scree over the retained alpha-qualified `weight_v` layers is weaker:

- `90%` variance at `172` components
- top-1 explained variance `0.1029`

## Validated Evaluation Summary

The validated evaluation used the first `4` eval images per scene.

Aggregate metrics from `validated_paper_metric_summary.csv`:

- `original`
  - GT PSNR mean `23.6837`
  - GT PSNR std `4.5942`
- `iid`
  - GT PSNR mean `7.2965`
  - GT PSNR std `3.4911`
  - pair MAE mean `0.39115`
  - pair MSE mean `0.23783`
  - pair PSNR mean `7.6640`
- `ood`
  - GT PSNR mean `7.2952`
  - GT PSNR std `3.4922`
  - pair MAE mean `0.39124`
  - pair MSE mean `0.23792`
  - pair PSNR mean `7.6626`

Interpretation:

- the retained-layer UniSub reconstruction preserves the shared parameter structure well enough to generate stable IID and OOD reconstructions
- reconstruction quality remains much worse than the original scene-specific checkpoints under this retained-layer basis
- IID and OOD performance are effectively the same under the final retained-layer set

## Output Layout

Main output root:

```bash
/mnt/data0/prakhar/sdfstudio_uwsh/analysis_alpha_unisub_v1
```

Important artifacts:

- `summary.json`
- `alpha/layer_summary.csv`
- `alpha/retained_layer_summary.csv`
- `alpha/q_threshold_sweep.csv`
- `basis_iid/layer_basis_summary.csv`
- `basis_iid/aggregate_scree/group_mean_explained_variance.png`
- `basis_iid/spectral_only_scree/group_mean_explained_variance.png`
- `reconstructed_models/manifest.csv`
- `evaluation/scene_metrics.csv`
- `evaluation/paper_metric_summary.csv`
- `evaluation/scene_grids/*.png`
- `evaluation/paper_figures/summary_scene_comparisons.png`

## Notes

- The main run uses `--alpha-min-fraction-in-range 0.3` because the stricter `0.5` filter removed `glin0.weight_v` even though its alpha mean/std and `q_l` remained usable.
- On this host we replaced the external WeightWatcher dependency with an in-script spectral-tail alpha estimator because the local SciPy stack crashed during WeightWatcher import.
- By default the script saves PNG plots only on this host because repeated vector export was unstable.
- Set `SDFSTUDIO_SAVE_SVG=1` and/or `SDFSTUDIO_SAVE_PDF=1` if your environment can handle vector export.
- Set `SDFSTUDIO_SAVE_LAYER_PLOTS=1` to also save per-layer scree plots.
