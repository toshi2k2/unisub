# Transformer UniSub Analysis

This note summarizes the recent transformer-family UniSub release in a user-facing form. For the main visuals and linked artifacts, start with the family pages in this directory.

## Included Families

| Family | Models | Layers | `q_all` | `q_arch` | Retained layers | Savings @ `90%` | README |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ViT | 464 | 40 | 0.0518 | 0.3545 | 18 | 62.62% | [vit](./vit/README.md) |
| GPT-2 | 177 | 49 | 0.0725 | 0.3059 | 23 | 64.57% | [gpt2](./gpt2/README.md) |
| LLaMA | 50 | 224 | 0.0940 | 0.3290 | 116 | 67.13% | [llama](./llama/README.md) |
| Flan-T5 GLUE | 196 | 216 | 0.0224 | 0.4850 | 179 | 85.13% | [flan_t5_glue](./flan_t5_glue/README.md) |

## What The Release Contains

- Family-level scree plots and cumulative explained-variance plots
- Alpha summaries and retained-layer summaries
- Savings-versus-alpha visual summaries
- Memory-threshold CSVs for the committed recent-analysis families

## Quick Reading Guide

- Use each family `README.md` for the public summary and the main scree figure.
- Use `alpha_summary.json` and `summary.json` inside each `artifacts/` folder for the machine-readable snapshot.
- Use `memory_thresholds.csv` and `alpha_retained_layers.csv` when you want the threshold-by-threshold tables rather than the short README summary.

## Notes

- The committed files here are validated recent-analysis snapshots rather than a complete mirror of the original analysis workspace.
- Flan-T5 GLUE is included as an artifact-backed recent-analysis release page even though there is no separate paper-era experiment folder for it in this repo.
