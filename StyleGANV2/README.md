# StyleGAN2 Universal-Subspace Experiment

This folder packages the commands and verified outputs needed to rerun the StyleGAN2 universal-subspace experiment.

The validated run uses:
- a heterogeneous pool of `20` public StyleGAN2 checkpoints from `models.json`
- an alpha-filtered, all-common convolution-layer selection rule
- spatial-wise universal subspaces following the Guth et al. style factorization for `3x3` synthesis convolutions
- paired generator reconstructions in both `IID` and leave-one-model-out `OOD` modes
- no-reference image-quality reporting via Inception Score (`IS`)

This folder can live outside the main repo. The commands below therefore make the repository path explicit instead of assuming this package lives inside `awesome-pretrained-stylegan2/`.

## What This Release Contains

- `run_public_pipeline.sh`
  - one-command wrapper for download + WeightWatcher prepass + alpha-filtered UniSub run
- `requirements_public.txt`
  - minimal Python packages needed on top of the vendored code already present in the main repo
- `retained_layers_conservative.csv`
  - the 9 retained layers from the validated conservative all-20 run
- `selected_models_all20.csv`
  - the 20 source checkpoints used in the conservative all-20 run
- `iid_ood_is_summary.csv`
  - aggregate `IS` numbers for the validated conservative run
- `experiment_snapshot.json`
  - `q_all`, `q_arch`, retained-layer list, selected-model count, and skipped image-eval targets

## Main Repository Requirement

This package is a lightweight companion folder, not a full copy of the codebase. You still need a checkout of the main repository that contains:

- `download_stylegan_models.py`
- `stylegan_all20_weightwatcher_eval.py`
- `stylegan_alpha_common20_eval.py`
- `compose_stylegan_paper_figures.py`
- the vendored `stylegan2-ada-pytorch` and `weightwatcher` code

Set:

```bash
export STYLEGAN_REPO_DIR=/path/to/awesome-pretrained-stylegan2
cd /path/to/StyleGANV2
```

If you do not want to use `STYLEGAN_REPO_DIR`, pass `--repo-dir "$STYLEGAN_REPO_DIR"` to the wrapper script.

## Environment Setup

Tested with Python `3.10`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements_public.txt
```

Direct plotting and detector caches into this folder:

```bash
export MPLCONFIGDIR="$PWD/.mplconfig"
export XDG_CACHE_HOME="$PWD/.cache"
```

## One-Command Wrapper

Simplest supported path:

```bash
bash run_public_pipeline.sh \
  --repo-dir "$STYLEGAN_REPO_DIR" \
  --device cuda:0 \
  --num-samples 512 \
  --batch-size 4 \
  --verbose
```

Useful variants:

```bash
# Reuse downloaded checkpoints and WeightWatcher outputs
bash run_public_pipeline.sh \
  --repo-dir "$STYLEGAN_REPO_DIR" \
  --skip-download \
  --skip-weightwatcher \
  --device cuda:0
```

```bash
# Only run up through the alpha stage, without IID/OOD image metrics
bash run_public_pipeline.sh \
  --repo-dir "$STYLEGAN_REPO_DIR" \
  --skip-download \
  --skip-weightwatcher \
  --skip-reconstruction \
  --device cpu
```

Outputs go to:

```text
models/
results/
```

## Step 1: Download the StyleGAN2 Checkpoints

This writes checkpoints into `models/` and a manifest into `results/`.

```bash
python "$STYLEGAN_REPO_DIR/download_stylegan_models.py" \
  --destination-dir models \
  --manifest-path results/download_manifest.json \
  --verbose
```

Expected behavior:
- the script attempts all checkpoint URLs from `models.json`
- some URLs are dead in the original public list, so successful downloads may be fewer than 36
- the validated run recovered 20 checkpoints locally

## Step 2: Run the Broad All-20 WeightWatcher + Spatial Prepass

This step computes:
- all-model layer coverage for shared `3x3` synthesis convs
- per-model WeightWatcher layer diagnostics
- the broader all-20 spatial basis summaries used by the alpha-filtered stage

```bash
python "$STYLEGAN_REPO_DIR/stylegan_all20_weightwatcher_eval.py" \
  --manifest-path results/download_manifest.json \
  --output-dir results/all20_weightwatcher_v1 \
  --skip-reconstruction \
  --device cpu \
  --verbose
```

Key output:

```text
results/all20_weightwatcher_v1/weightwatcher/all_models.csv
```

This CSV is the input to the alpha-filtered common-layer experiment below.

## Step 3: Run the Conservative Alpha Analysis

This stage:
- restricts to the synthesis `3x3` conv layers common to all downloaded models
- computes the layer-wise alpha statistics
- applies the conservative keep rule
- writes the retained layer set and scree plots

```bash
python "$STYLEGAN_REPO_DIR/stylegan_alpha_common20_eval.py" \
  --manifest-path results/download_manifest.json \
  --weightwatcher-csv results/all20_weightwatcher_v1/weightwatcher/all_models.csv \
  --output-dir results/all20_alpha_common20_v1 \
  --skip-reconstruction \
  --device cpu \
  --alpha-std-max 2.0 \
  --alpha-q-threshold 0.05 \
  --verbose
```

Main outputs:

```text
results/all20_alpha_common20_v1/summary.json
results/all20_alpha_common20_v1/alpha/layer_summary.csv
results/all20_alpha_common20_v1/alpha/retained_layers.csv
results/all20_alpha_common20_v1/common20_spatial_scree_alpha_retained/group_mean_explained_variance.svg
```

## Step 4: Build the Universal Subspace Reconstructions and Report IS

This reruns the same alpha-filtered stage, but now also reconstructs target generators in `IID` and `OOD` mode, generates sample grids, and computes Inception Score.

GPU command used for the validated run:

```bash
python "$STYLEGAN_REPO_DIR/stylegan_alpha_common20_eval.py" \
  --manifest-path results/download_manifest.json \
  --weightwatcher-csv results/all20_weightwatcher_v1/weightwatcher/all_models.csv \
  --output-dir results/all20_alpha_common20_v1 \
  --device cuda:0 \
  --num-samples 512 \
  --batch-size 4 \
  --alpha-std-max 2.0 \
  --alpha-q-threshold 0.05 \
  --verbose
```

If you only have CPU, reduce the sample count:

```bash
python "$STYLEGAN_REPO_DIR/stylegan_alpha_common20_eval.py" \
  --manifest-path results/download_manifest.json \
  --weightwatcher-csv results/all20_weightwatcher_v1/weightwatcher/all_models.csv \
  --output-dir results/all20_alpha_common20_v1 \
  --device cpu \
  --num-samples 64 \
  --batch-size 2 \
  --alpha-std-max 2.0 \
  --alpha-q-threshold 0.05 \
  --verbose
```

Main outputs:

```text
results/all20_alpha_common20_v1/reconstruction_metrics.csv
results/all20_alpha_common20_v1/paper_metric_summary.csv
results/all20_alpha_common20_v1/grids/*.png
results/all20_alpha_common20_v1/skipped_targets.csv
```

## Conservative Retained Layer Set

The validated conservative all-20 run retained these 9 layers:

- `synthesis.b8.conv1.weight`
- `synthesis.b16.conv0.weight`
- `synthesis.b32.conv1.weight`
- `synthesis.b64.conv0.weight`
- `synthesis.b64.conv1.weight`
- `synthesis.b128.conv0.weight`
- `synthesis.b128.conv1.weight`
- `synthesis.b256.conv0.weight`
- `synthesis.b256.conv1.weight`

These are also copied into `retained_layers_conservative.csv`.

## Validated Quality Summary

From the validated conservative all-20 run:

- `q_all = 0.05722518915489639`
- `q_arch = 0.09070011931249121`
- `num_selected_models = 20`
- `num_retained_layers = 9`

Interpretation:
- `q_all` is the full-family alpha suitability before layer filtering
- `q_arch` is the retained-layer suitability for the conservative chosen subset

## Validated IS Summary

Aggregate Inception Score from the validated conservative run:

- IID:
  - original IS mean = `3.20826232425645`
  - reconstructed IS mean = `3.2041762198607744`
  - delta = `-0.004086104395675151`
- OOD:
  - original IS mean = `3.20826232425645`
  - reconstructed IS mean = `3.2042376821599863`
  - delta = `-0.004024642096463126`

These values are copied into `iid_ood_is_summary.csv`.

## Optional: Compose Paper Figures from Saved Grids

Once the `grids/` directory exists, you can compose the draft and appendix comparison figures:

```bash
python "$STYLEGAN_REPO_DIR/compose_stylegan_paper_figures.py" \
  --grid-root results/all20_alpha_common20_v1/grids \
  --summary-path results/all20_alpha_common20_v1/summary.json \
  --metric-summary-path results/all20_alpha_common20_v1/paper_metric_summary.csv \
  --output-dir results/all20_alpha_common20_v1/paper_figures \
  --verbose
```

## Optional Appendix Ablations

Looser all-20 layer rule:

```bash
python "$STYLEGAN_REPO_DIR/stylegan_alpha_common20_eval.py" \
  --manifest-path results/download_manifest.json \
  --weightwatcher-csv results/all20_weightwatcher_v1/weightwatcher/all_models.csv \
  --output-dir results/all20_alpha_common20_loose_v1 \
  --skip-reconstruction \
  --device cpu \
  --alpha-std-max 2.0 \
  --alpha-q-threshold 0.04 \
  --verbose
```

Good-model subset:

```bash
python "$STYLEGAN_REPO_DIR/stylegan_alpha_common20_eval.py" \
  --manifest-path results/download_manifest.json \
  --weightwatcher-csv results/all20_weightwatcher_v1/weightwatcher/all_models.csv \
  --output-dir results/all20_alpha_common20_modelsubset09_v1 \
  --skip-reconstruction \
  --device cpu \
  --alpha-std-max 2.0 \
  --alpha-q-threshold 0.05 \
  --model-min-fraction-in-range 0.9 \
  --verbose
```

## Important Caveats

- The current `stylegan_alpha_common20_eval.py` pipeline reconstructs target generators in memory, generates images, and writes metrics/grids. It does **not** export standalone reconstructed `.pkl` checkpoints by default.
- The validated image-evaluation run covers `18` targets, not all `20`. Two checkpoints are still skipped during live generator reconstruction:
  - `beetles`: TensorFlow pickle version too low
  - `wikiart`: generator shape mismatch during live load
- The source pool is heterogeneous. The local metadata explicitly marks `faces-ffhq-config-e-256x256` and `cakes` as trained from scratch, and `ukiyoe-faces` and `flowers` as fine-tuned. Many other checkpoints are simply public pretrained releases without uniform provenance labels.
- This release reports `IS` because it is implemented and validated in the current code. CLIP score is **not** part of the current validated StyleGAN alpha-filtered run.

## Still Missing

If you want this public package to be fully self-sufficient for external users, the one thing still missing is explicit checkpoint export for the reconstructed `UniSub` generators. The current pipeline already reconstructs them internally, so adding serialized `.pkl` emission would be straightforward, but it is not implemented yet.
