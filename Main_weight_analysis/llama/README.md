# LLaMA Main Weight Analysis

This folder is the additional LLaMA analysis release. It augments the original paper-era [`../../LLaMA`](../../LLaMA/README.md) folder with the newer alpha, scree, retained-layer, and memory-threshold outputs.

## Validated Snapshot

- Models analyzed: `50`
- Matrix-valued layers analyzed: `224`
- Retained alpha-qualified layers: `116`
- `q_all = 0.0940`
- `q_arch = 0.3290`

Family memory savings:

| Variance threshold | Family savings | Compression multiple |
| --- | --- | --- |
| `0.80` | `67.46%` | `3.07x` |
| `0.90` | `67.13%` | `3.04x` |
| `0.95` | `67.04%` | `3.03x` |

## Scree

[<img src="./artifacts/aggregate_scree.png" alt="LLaMA aggregate scree" width="100%">](./artifacts/aggregate_scree.png)

This aggregate scree plot is the main public-facing summary because it keeps the broad component axis and shows the family-wide tail directly.

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
- [scripts/llama_models.txt](./scripts/llama_models.txt)

Run path:

```bash
cd /mnt/data0/prakhar/unisub/Main_weight_analysis/llama

python scripts/download_models.py --model_name_file scripts/llama_models.txt --target_folder ./llama_models
python scripts/run_pca.py --model_directory ./llama_models --target_folder ./llama_plots
python scripts/generate_plots.py --pca_dir ./llama_plots/pca --output_dir ./results --plot_type both
```

The committed alpha and memory-threshold artifacts come from the broader follow-up transformer run and are included here as the validated additional-analysis snapshot.
