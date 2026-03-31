# ViT Main Weight Analysis

This folder is the additional ViT analysis release. It augments the original paper-era [`../../VIT`](../../VIT/README.md) folder with the newer alpha, scree, retained-layer, and memory-threshold outputs.

## Validated Snapshot

- Models analyzed: `464`
- Matrix-valued layers analyzed: `40`
- Retained alpha-qualified layers: `18`
- `q_all = 0.0518`
- `q_arch = 0.3545`
- Mean alpha over all values: `4.7066`

Family memory savings:

| Variance threshold | Family savings | Compression multiple |
| --- | --- | --- |
| `0.80` | `76.99%` | `4.35x` |
| `0.90` | `62.62%` | `2.68x` |
| `0.95` | `61.24%` | `2.58x` |

## Scree

[<img src="./artifacts/aggregate_scree.png" alt="ViT aggregate scree" width="100%">](./artifacts/aggregate_scree.png)

This aggregate scree plot is the main public-facing summary because the x-axis extends through `100` components and directly shows the shared low-rank structure.

## Included Artifacts

- [alpha_summary.json](./artifacts/alpha_summary.json)
- [summary.json](./artifacts/summary.json)
- [aggregate_scree.png](./artifacts/aggregate_scree.png)
- [aggregate_cumulative.png](./artifacts/aggregate_cumulative.png)
- [alpha_vs_savings.png](./artifacts/alpha_vs_savings.png)

## Code In This Folder

The original paper-era scripts are copied into [`scripts/`](./scripts/):

- [scripts/download_models.py](./scripts/download_models.py)
- [scripts/run_pca.py](./scripts/run_pca.py)
- [scripts/generate_plots.py](./scripts/generate_plots.py)
- [scripts/vit.txt](./scripts/vit.txt)

Run path:

```bash
cd /mnt/data0/prakhar/unisub/Main_weight_analysis/vit

python scripts/download_models.py --model_name_file scripts/vit.txt --target_folder ./vit_models
python scripts/run_pca.py --model_directory ./vit_models --target_folder ./vit_plots
python scripts/generate_plots.py --pca_dir ./vit_plots/pca --output_dir ./results --plot_type both
```

The committed alpha and memory-threshold artifacts come from the broader follow-up transformer run and are included here as the validated additional-analysis snapshot.
