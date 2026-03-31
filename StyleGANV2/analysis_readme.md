# StyleGAN2 UniSub Analysis

Primary validated analysis root:

- `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_v1`

Lower-rank ablation roots:

- `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_80_v1`
- `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_70_v1`

This experiment uses the alpha-filtered all-20 common-layer StyleGAN2 setup:

- `20` downloaded public checkpoints
- `9` retained common `3x3` synthesis conv layers
- spatial-wise basis extraction on the retained conv kernels
- IID and leave-one-model-out OOD generator reconstruction
- paired pixel metrics and no-reference Inception Score

## Alpha / Layer Selection

The retained layer set is unchanged across the `70%`, `80%`, and validated `90%` runs.

- `q_all = 0.05722518915489639`
- `q_arch = 0.09070011931249121`
- retained layers:
  - `synthesis.b8.conv1.weight`
  - `synthesis.b16.conv0.weight`
  - `synthesis.b32.conv1.weight`
  - `synthesis.b64.conv0.weight`
  - `synthesis.b64.conv1.weight`
  - `synthesis.b128.conv0.weight`
  - `synthesis.b128.conv1.weight`
  - `synthesis.b256.conv0.weight`
  - `synthesis.b256.conv1.weight`

Relevant files:

- validated summary: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_v1/summary.json`
- 80% summary: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_80_v1/summary.json`
- 70% summary: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_70_v1/summary.json`

## Validated 90% Run

From `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_v1/paper_metric_summary.csv`:

- IID PSNR `19.9836 dB`
- OOD PSNR `19.9154 dB`
- reconstructed IS `3.2042` IID, `3.2042` OOD
- implicit mean selected spatial rank about `6.33`

Interpretation:

- this remains the best paper-facing reconstruction setting
- it preserves matched-latent image structure far better than the lower-rank ablations

## Lower-rank Ablations

These runs use the same retained layers and alpha diagnostics, but truncate each retained spatial
basis earlier during reconstruction.

### 80% Explained Variance

From `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_80_v1/paper_metric_summary.csv`:

- IID PSNR `14.0712 dB`
- OOD PSNR `13.9725 dB`
- reconstructed IS `3.3039` IID, `3.3032` OOD
- mean selected spatial rank `4.12`

### 70% Explained Variance

From `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_70_v1/paper_metric_summary.csv`:

- IID PSNR `11.4558 dB`
- OOD PSNR `11.4564 dB`
- reconstructed IS `3.3488` IID, `3.3445` OOD
- mean selected spatial rank `3.33`

## Comparison

Paired reconstruction fidelity drops sharply below the validated `90%` setting:

- `90% -> 80%`: about `-5.91 dB` IID PSNR, `-5.94 dB` OOD PSNR
- `90% -> 70%`: about `-8.53 dB` IID PSNR, `-8.46 dB` OOD PSNR

No-reference Inception Score stays roughly flat and even rises slightly in the lower-rank runs.
That means the lower-rank generators can still look broadly plausible to the Inception detector,
but they no longer match the original generators well on shared latent seeds.

## Recommendation

For StyleGAN2:

- keep the validated `90%` run as the main reconstruction result
- use `80%` only as a compression ablation
- do not use `70%` as the main setting

Main artifacts:

- validated metrics: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_v1/paper_metric_summary.csv`
- 80% metrics: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_80_v1/paper_metric_summary.csv`
- 70% metrics: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_70_v1/paper_metric_summary.csv`
- validated grids: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_v1/grids/`
- 80% grids: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_80_v1/grids/`
- 70% grids: `/mnt/data0/prakhar/stylegan_paper/all20_alpha_common20_70_v1/grids/`
