# Transformer UniSub Analysis

This package summarizes the analyzed matrix layers from the existing saved PCA runs for ViT, GPT2, LLAMA, and Flan-T5 GLUE.
Storage assumes fp32 tensors (4 bytes per parameter).
Activations, residual adds, softmax, dropout, and other non-parameterized ops always contribute zero parameters and are ignored.
The memory reporting below is split into analyzed matrix parameters versus excluded or unmodeled tensors from the reference checkpoint.
Excluded or unmodeled tensors are every reference-model parameters outside the saved PCA layer set, including norms, biases, embeddings, heads, adapters, and any other tensor never captured by the PCA artifacts.

## Main Outputs

- `alpha_family_violin.pdf` / `alpha_family_violin.png`: top-level violin of the legacy saved-PCA tail slope across families.
- `<family>/aggregate_scree.pdf` / `.png`: notebook-style scree plot matched to `vit_total_new2.ipynb` execution-count 13, with the in-plot title removed.
- `<family>/aggregate_cumulative.pdf` / `.png`: mean cumulative explained-variance curve.
- `<family>/alpha_values_long.csv`: per-model per-layer alpha values where raw source weights are locally available.
- `<family>/alpha_model_summary.csv`: per-model alpha mean/std across all analyzed layers.
- `<family>/alpha_layer_metrics.csv`: per-layer alpha mean/std/count, `q_l`, and retained-layer flags.
- `<family>/alpha_layer_labels.csv`: short layer labels used on the per-layer alpha violin plot.
- `<family>/alpha_retained_layers.csv`: final retained layers from the alpha metric pipeline, when available.
- `<family>/alpha_violin_all_layers.pdf` / `.png`: one violin per layer over all locally available source models.
- `<family>/memory_thresholds.csv`: memory and parameter counts for 60/70/80/90/95 explained variance.
- `<family>/retained_layers_ev60.csv` ... `<family>/retained_layers_ev95.csv`: retained-rank layer tables for each explained-variance threshold.

## PCA Summary

| Family | Models | Layers | Mean saved-PCA tail slope | Fraction in [2, 6] | 90% family savings | 90% compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ViT | 470 | 40 | 0.9163 | 0.000 | 62.62% | 2.68x |
| GPT2 | 177 | 49 | 0.5611 | 0.020 | 64.57% | 2.82x |
| LLAMA | 50 | 224 | 0.4654 | 0.000 | 67.13% | 3.04x |
| Flan-T5 GLUE | 196 | 216 | 1.2376 | 0.042 | 85.13% | 6.72x |

These PCA-summary tail slopes come only from the saved PCA scree curves. They are not WeightWatcher alpha and should not be compared directly to the original WeightWatcher ViT/LLAMA paper results.

## Alpha Metrics

| Family | Alpha source models | Alpha layers | Mean alpha (all values) | Mean alpha (per model) | q_all | q_arch | Retained layers | Files |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| ViT | 464 | 40 | 4.7066 | 4.7020 | 0.05183 | 0.3545 | 18 | `vit/alpha_values_long.csv`, `vit/alpha_model_summary.csv`, `vit/alpha_layer_metrics.csv`, `vit/alpha_violin_all_layers.pdf` |
| GPT2 | 177 | 49 | 4.0951 | 4.0919 | 0.07249 | 0.3059 | 23 | `gpt2/alpha_values_long.csv`, `gpt2/alpha_model_summary.csv`, `gpt2/alpha_layer_metrics.csv`, `gpt2/alpha_violin_all_layers.pdf` |
| LLAMA | 50 | 224 | 5.2391 | 5.2074 | 0.09399 | 0.329 | 116 | `llama/alpha_values_long.csv`, `llama/alpha_model_summary.csv`, `llama/alpha_layer_metrics.csv`, `llama/alpha_violin_all_layers.pdf` |
| Flan-T5 GLUE | 196 | 216 | 3.9136 | 3.9136 | 0.02244 | 0.485 | 179 | `flan_t5_glue/alpha_values_long.csv`, `flan_t5_glue/alpha_model_summary.csv`, `flan_t5_glue/alpha_layer_metrics.csv`, `flan_t5_glue/alpha_violin_all_layers.pdf` |

These alpha metrics come from the true cross-model layerwise alpha pipeline (`q_l`, `q_all`, `q_arch`) rather than the older single-spectrum alpha fit from the saved PCA curves.
For raw-weight alpha estimation, this script now uses a WeightWatcher-style power-law tail fit on the eigenvalue spectrum instead of the earlier ad hoc log-rank tail slope.
Only alpha summaries written with `alpha_method = ww_powerlaw_tail` are treated as current below; older on-disk alpha CSV/JSON files are considered stale and need rerunning with the patched script.
For each family with local raw weights, `alpha_layer_metrics.csv` contains `alpha_mean`, `alpha_std`, `num_valid_alpha`, `q_l`, `stat_keep`, `q_keep`, and `retained` for every analyzed layer, and `alpha_retained_layers.csv` lists the final retained alpha layers.

## WeightWatcher References

- General HTSR rule from the WeightWatcher catalog: well-trained layers typically have `alpha` near `2`, or at least in the `2 <= alpha <= 6` band. Reference: `https://weightwatcher.ai/models.html`.
- GPT reference: `https://weightwatcher.ai/models/GPT-summary.html` reports that `openai-gpt` has many outlier layers above the well-trained threshold, while `gpt2` has all but one layer inside `2 <= alpha <= 6`.
- Flan-T5 reference: `https://weightwatcher.ai/models/FlanT5-summary.html` reports that most Flan-T5 layers lie in `2 <= alpha <= 6`, and stronger checkpoints have fewer layers with `alpha > 6`.
- LLaMA / instruct-tuning reference: `https://www.weightwatcher.ai/fine_tuned.html` reports that base models often have many underfit layers with `alpha > 6`, while instruct fine-tuned models have most layers in the `2 <= alpha <= 6` safe zone. The same page explicitly shows this for Llama-3.1-Instruct histograms and correlation-flow plots.
- ViT note: the current WeightWatcher catalog does not provide a one-to-one plain ViT image-classification family page matching our local HF ViT pool. For ViT we therefore use the general HTSR alpha band as the external sanity check, not a direct model-for-model public WW table.

## Retained Layers

The `retained_layers_evXX.csv` files below are the compression-retained layers: every analyzed layer stays in the family summary, and each file records the minimum rank that reaches the requested explained-variance threshold for that layer.
Where a true cross-model alpha matrix is available, the alpha-retained layers are separately saved in `alpha_retained_layers.csv`.

### ViT

- Per-layer retained-rank tables: `vit/retained_layers_ev60.csv` ... `vit/retained_layers_ev95.csv`
- Per-layer alpha values for all locally available models: `vit/alpha_values_long.csv`
- Per-model alpha summary across layers: `vit/alpha_model_summary.csv`
- Layer labels used in the alpha violin: `vit/alpha_layer_labels.csv`
- Per-layer alpha violin over all models: `vit/alpha_violin_all_layers.pdf`
- Alpha-retained layers: `vit/alpha_retained_layers.csv`
- Alpha metrics: mean alpha over all values = 4.7066, mean alpha over model means = 4.7020, q_all = 0.05183, q_arch = 0.3545, retained alpha layers = 18
- ViT raw-checkpoint recovery manifest: `vit/vit_model_resolution_manifest.json` records the recovered 470-model repo-id list and the ambiguous repo-name slugs resolved by last occurrence from the historical HF crawl.
- Lowest 90% retained ranks: Q:120 (vit.encoder.layer.0.attention.attention.query.weight), K:124 (vit.encoder.layer.0.attention.attention.key.weight), V:209 (vit.encoder.layer.0.attention.attention.value.weight), Attn Out:249 (vit.encoder.layer.0.attention.output.dense.weight), K:276 (vit.encoder.layer.1.attention.attention.key.weight)

### GPT2

- Per-layer retained-rank tables: `gpt2/retained_layers_ev60.csv` ... `gpt2/retained_layers_ev95.csv`
- Per-layer alpha values for all locally available models: `gpt2/alpha_values_long.csv`
- Per-model alpha summary across layers: `gpt2/alpha_model_summary.csv`
- Layer labels used in the alpha violin: `gpt2/alpha_layer_labels.csv`
- Per-layer alpha violin over all models: `gpt2/alpha_violin_all_layers.pdf`
- Alpha-retained layers: `gpt2/alpha_retained_layers.csv`
- Alpha metrics: mean alpha over all values = 4.0951, mean alpha over model means = 4.0919, q_all = 0.07249, q_arch = 0.3059, retained alpha layers = 23
- Lowest 90% retained ranks: Classifier:169 (score.weight), Attn Out:311 (transformer.h.0.attn.c_proj.weight), Attn Out:333 (transformer.h.1.attn.c_proj.weight), Attn Out:351 (transformer.h.11.attn.c_proj.weight), Attn Out:361 (transformer.h.6.attn.c_proj.weight)

### LLAMA

- Per-layer retained-rank tables: `llama/retained_layers_ev60.csv` ... `llama/retained_layers_ev95.csv`
- Per-layer alpha values for all locally available models: `llama/alpha_values_long.csv`
- Per-model alpha summary across layers: `llama/alpha_model_summary.csv`
- Layer labels used in the alpha violin: `llama/alpha_layer_labels.csv`
- Per-layer alpha violin over all models: `llama/alpha_violin_all_layers.pdf`
- Alpha-retained layers: `llama/alpha_retained_layers.csv`
- Alpha metrics: mean alpha over all values = 5.2391, mean alpha over model means = 5.2074, q_all = 0.09399, q_arch = 0.329, retained alpha layers = 116
- Lowest 90% retained ranks: Up:300 (model.layers.0.mlp.up_proj.weight), K:414 (model.layers.8.self_attn.k_proj.weight), K:421 (model.layers.9.self_attn.k_proj.weight), V:448 (model.layers.0.self_attn.v_proj.weight), K:448 (model.layers.13.self_attn.k_proj.weight)

### Flan-T5 GLUE

- Per-layer retained-rank tables: `flan_t5_glue/retained_layers_ev60.csv` ... `flan_t5_glue/retained_layers_ev95.csv`
- Per-layer alpha values for all locally available models: `flan_t5_glue/alpha_values_long.csv`
- Per-model alpha summary across layers: `flan_t5_glue/alpha_model_summary.csv`
- Layer labels used in the alpha violin: `flan_t5_glue/alpha_layer_labels.csv`
- Per-layer alpha violin over all models: `flan_t5_glue/alpha_violin_all_layers.pdf`
- Alpha-retained layers: `flan_t5_glue/alpha_retained_layers.csv`
- Alpha metrics: mean alpha over all values = 3.9136, mean alpha over model means = 3.9136, q_all = 0.02244, q_arch = 0.485, retained alpha layers = 179
- Lowest 90% retained ranks: Self V:3 (decoder.block.0.layer.0.SelfAttention.v.weight), Self K:10 (decoder.block.0.layer.0.SelfAttention.k.weight), Self K:10 (decoder.block.6.layer.0.SelfAttention.k.weight), Self K:11 (decoder.block.4.layer.0.SelfAttention.k.weight), Self Q:11 (decoder.block.6.layer.0.SelfAttention.q.weight)

## Memory Tables

The tables below use one shared basis per layer for the whole family plus one coefficient matrix per model.
Original totals are repeated across thresholds because the analyzed source family stays fixed; only the retained ranks change.
For each family, the analyzed totals are exact for the PCA-covered matrix layers. The reference-full totals keep the reference checkpoint's excluded or unmodeled tensors unchanged and assume every model in the family has the same excluded-parameter count as that reference checkpoint.

### ViT

- Analyzed matrix params per model: 23,592,960 params (90.00 MiB / 0.088 GiB)
- Reference excluded or unmodeled params per model: 62,974,696 params (240.23 MiB / 0.235 GiB)
- Reference full params per model: 86,567,656 params (330.23 MiB / 0.322 GiB)
- Analyzed matrix params across all 470 models: 11,088,691,200 params (42300.00 MiB / 41.309 GiB)
- Reference full params across all 470 models under the fixed-reference assumption: 40,686,798,320 params (155207.82 MiB / 151.570 GiB)
- Reference accounting note: Reference-full counts come from the reference checkpoint only. Excluded or unmodeled params are every reference tensor outside the analyzed PCA layer set, including norms, biases, embeddings, heads, adapters, and any other tensor not present in the saved PCA artifacts. Activations and other non-parameterized ops always contribute zero parameters.

| EV | Basis Once | Coeff / model | Analyzed family compressed total | Analyzed savings | Analyzed compression | Reference full family compressed total | Reference full savings | Reference full compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60% | 1,423,872 params (5.43 MiB / 0.005 GiB) | 1,423,872 params (5.43 MiB / 0.005 GiB) | 670,643,712 params (2558.30 MiB / 2.498 GiB) | 93.95% | 16.53x | 30,268,750,832 params (115466.12 MiB / 112.760 GiB) | 25.61% | 1.34x |
| 70% | 2,823,936 params (10.77 MiB / 0.011 GiB) | 2,823,936 params (10.77 MiB / 0.011 GiB) | 1,330,073,856 params (5073.83 MiB / 4.955 GiB) | 88.01% | 8.34x | 30,928,180,976 params (117981.65 MiB / 115.216 GiB) | 23.98% | 1.32x |
| 80% | 5,417,472 params (20.67 MiB / 0.020 GiB) | 5,417,472 params (20.67 MiB / 0.020 GiB) | 2,551,629,312 params (9733.69 MiB / 9.506 GiB) | 76.99% | 4.35x | 32,149,736,432 params (122641.51 MiB / 119.767 GiB) | 20.98% | 1.27x |
| 90% | 8,800,512 params (33.57 MiB / 0.033 GiB) | 8,800,512 params (33.57 MiB / 0.033 GiB) | 4,145,041,152 params (15812.08 MiB / 15.441 GiB) | 62.62% | 2.68x | 33,743,148,272 params (128719.90 MiB / 125.703 GiB) | 17.07% | 1.21x |
| 95% | 9,124,608 params (34.81 MiB / 0.034 GiB) | 9,124,608 params (34.81 MiB / 0.034 GiB) | 4,297,690,368 params (16394.39 MiB / 16.010 GiB) | 61.24% | 2.58x | 33,895,797,488 params (129302.21 MiB / 126.272 GiB) | 16.69% | 1.20x |

- CSV export: `vit/memory_thresholds.csv`

### GPT2

- Analyzed matrix params per model: 84,936,192 params (324.01 MiB / 0.316 GiB)
- Reference excluded or unmodeled params per model: 39,505,152 params (150.70 MiB / 0.147 GiB)
- Reference full params per model: 124,441,344 params (474.71 MiB / 0.464 GiB)
- Analyzed matrix params across all 177 models: 15,033,705,984 params (57349.04 MiB / 56.005 GiB)
- Reference full params across all 177 models under the fixed-reference assumption: 22,026,117,888 params (84022.97 MiB / 82.054 GiB)
- Reference accounting note: Reference-full counts come from the reference checkpoint only. Excluded or unmodeled params are every reference tensor outside the analyzed PCA layer set, including norms, biases, embeddings, heads, adapters, and any other tensor not present in the saved PCA artifacts. Activations and other non-parameterized ops always contribute zero parameters.

| EV | Basis Once | Coeff / model | Analyzed family compressed total | Analyzed savings | Analyzed compression | Reference full family compressed total | Reference full savings | Reference full compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60% | 30,241,536 params (115.36 MiB / 0.113 GiB) | 20,321,448 params (77.52 MiB / 0.076 GiB) | 3,627,137,832 params (13836.43 MiB / 13.512 GiB) | 75.87% | 4.14x | 10,619,549,736 params (40510.37 MiB / 39.561 GiB) | 51.79% | 2.07x |
| 70% | 39,748,608 params (151.63 MiB / 0.148 GiB) | 26,333,396 params (100.45 MiB / 0.098 GiB) | 4,700,759,700 params (17931.98 MiB / 17.512 GiB) | 68.73% | 3.20x | 11,693,171,604 params (44605.91 MiB / 43.560 GiB) | 46.91% | 1.88x |
| 80% | 41,647,872 params (158.87 MiB / 0.155 GiB) | 29,719,564 params (113.37 MiB / 0.111 GiB) | 5,302,010,700 params (20225.57 MiB / 19.752 GiB) | 64.73% | 2.84x | 12,294,422,604 params (46899.50 MiB / 45.800 GiB) | 44.18% | 1.79x |
| 90% | 41,810,688 params (159.50 MiB / 0.156 GiB) | 29,855,570 params (113.89 MiB / 0.111 GiB) | 5,326,246,578 params (20318.02 MiB / 19.842 GiB) | 64.57% | 2.82x | 12,318,658,482 params (46991.95 MiB / 45.891 GiB) | 44.07% | 1.79x |
| 95% | 41,829,888 params (159.57 MiB / 0.156 GiB) | 29,855,620 params (113.89 MiB / 0.111 GiB) | 5,326,274,628 params (20318.13 MiB / 19.842 GiB) | 64.57% | 2.82x | 12,318,686,532 params (46992.06 MiB / 45.891 GiB) | 44.07% | 1.79x |

- CSV export: `gpt2/memory_thresholds.csv`

### LLAMA

- Analyzed matrix params per model: 6,979,321,856 params (26624.00 MiB / 26.000 GiB)
- Reference excluded or unmodeled params per model: 1,050,939,392 params (4009.02 MiB / 3.915 GiB)
- Reference full params per model: 8,030,261,248 params (30633.02 MiB / 29.915 GiB)
- Analyzed matrix params across all 50 models: 348,966,092,800 params (1331200.00 MiB / 1300.000 GiB)
- Reference full params across all 50 models under the fixed-reference assumption: 401,513,062,400 params (1531650.78 MiB / 1495.753 GiB)
- Reference accounting note: Reference-full counts come from the reference checkpoint only. Excluded or unmodeled params are every reference tensor outside the analyzed PCA layer set, including norms, biases, embeddings, heads, adapters, and any other tensor not present in the saved PCA artifacts. Activations and other non-parameterized ops always contribute zero parameters.

| EV | Basis Once | Coeff / model | Analyzed family compressed total | Analyzed savings | Analyzed compression | Reference full family compressed total | Reference full savings | Reference full compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60% | 1,543,641,088 params (5888.52 MiB / 5.751 GiB) | 1,740,398,592 params (6639.09 MiB / 6.483 GiB) | 88,563,570,688 params (337843.21 MiB / 329.925 GiB) | 74.62% | 3.94x | 141,110,540,288 params (538293.99 MiB / 525.678 GiB) | 64.86% | 2.85x |
| 70% | 1,814,759,424 params (6922.76 MiB / 6.761 GiB) | 2,203,451,392 params (8405.50 MiB / 8.208 GiB) | 111,987,329,024 params (427197.76 MiB / 417.185 GiB) | 67.91% | 3.12x | 164,534,298,624 params (627648.54 MiB / 612.938 GiB) | 59.02% | 2.44x |
| 80% | 1,826,738,176 params (6968.45 MiB / 6.805 GiB) | 2,234,516,480 params (8524.00 MiB / 8.324 GiB) | 113,552,562,176 params (433168.65 MiB / 423.016 GiB) | 67.46% | 3.07x | 166,099,531,776 params (633619.43 MiB / 618.769 GiB) | 58.63% | 2.42x |
| 90% | 1,852,237,824 params (7065.73 MiB / 6.900 GiB) | 2,256,791,552 params (8608.98 MiB / 8.407 GiB) | 114,691,815,424 params (437514.55 MiB / 427.260 GiB) | 67.13% | 3.04x | 167,238,785,024 params (637965.34 MiB / 623.013 GiB) | 58.35% | 2.40x |
| 95% | 1,859,053,568 params (7091.73 MiB / 6.926 GiB) | 2,263,078,912 params (8632.96 MiB / 8.431 GiB) | 115,012,999,168 params (438739.77 MiB / 428.457 GiB) | 67.04% | 3.03x | 167,559,968,768 params (639190.55 MiB / 624.210 GiB) | 58.27% | 2.40x |

- CSV export: `llama/memory_thresholds.csv`

### Flan-T5 GLUE

- Analyzed matrix params per model: 358,612,992 params (1368.00 MiB / 1.336 GiB)
- Reference excluded or unmodeled params per model: 424,537,088 params (1619.48 MiB / 1.582 GiB)
- Reference full params per model: 783,150,080 params (2987.48 MiB / 2.917 GiB)
- Analyzed matrix params across all 196 models: 70,288,146,432 params (268128.00 MiB / 261.844 GiB)
- Reference full params across all 196 models under the fixed-reference assumption: 153,497,415,680 params (585546.17 MiB / 571.822 GiB)
- Reference accounting note: Reference-full counts come from the reference checkpoint only. Excluded or unmodeled params are every reference tensor outside the analyzed PCA layer set, including norms, biases, embeddings, heads, adapters, and any other tensor not present in the saved PCA artifacts. Activations and other non-parameterized ops always contribute zero parameters.

| EV | Basis Once | Coeff / model | Analyzed family compressed total | Analyzed savings | Analyzed compression | Reference full family compressed total | Reference full savings | Reference full compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 60% | 12,124,160 params (46.25 MiB / 0.045 GiB) | 12,828,416 params (48.94 MiB / 0.048 GiB) | 2,526,493,696 params (9637.81 MiB / 9.412 GiB) | 96.41% | 27.82x | 85,735,762,944 params (327055.98 MiB / 319.391 GiB) | 44.15% | 1.79x |
| 70% | 19,830,272 params (75.65 MiB / 0.074 GiB) | 20,964,608 params (79.97 MiB / 0.078 GiB) | 4,128,893,440 params (15750.48 MiB / 15.381 GiB) | 94.13% | 17.02x | 87,338,162,688 params (333168.65 MiB / 325.360 GiB) | 43.10% | 1.76x |
| 80% | 29,377,280 params (112.07 MiB / 0.109 GiB) | 33,599,232 params (128.17 MiB / 0.125 GiB) | 6,614,826,752 params (25233.56 MiB / 24.642 GiB) | 90.59% | 10.63x | 89,824,096,000 params (342651.73 MiB / 334.621 GiB) | 41.48% | 1.71x |
| 90% | 44,550,656 params (169.95 MiB / 0.166 GiB) | 53,111,040 params (202.60 MiB / 0.198 GiB) | 10,454,314,496 params (39880.04 MiB / 38.945 GiB) | 85.13% | 6.72x | 93,663,583,744 params (357298.22 MiB / 348.924 GiB) | 38.98% | 1.64x |
| 95% | 60,896,512 params (232.30 MiB / 0.227 GiB) | 72,209,408 params (275.46 MiB / 0.269 GiB) | 14,213,940,480 params (54221.88 MiB / 52.951 GiB) | 79.78% | 4.95x | 97,423,209,728 params (371640.05 MiB / 362.930 GiB) | 36.53% | 1.58x |

- CSV export: `flan_t5_glue/memory_thresholds.csv`

## Reconstruction / Accuracy Notes

- The main alpha analysis above is the layerwise all-model analysis. The older GPT2 2+2 bucket comparison is still saved in `gpt2_demo/group_alpha_violin.pdf`; scratch-like mean alpha = 0.4242, pretrained-like mean alpha = 0.4250. This small local 2+2 split does not show a meaningful alpha gap.
- GPT2 reconstructed-vs-original prompt comparison: mean next-token KL = 3.4336, mean top-1 agreement = 0.0833. Saved in `gpt2_demo/prompt_comparison.json` and `gpt2_demo/prompt_comparison.csv`.
- ViT image-classification accuracy is still not reported here. I did not find a matched local validation/test dataset for the cached ViT checkpoints; the readily available local datasets in this workspace are the ResNet50 CIFAR/SVHN assets, which are not a clean target for the ViT family mix in `downloaded_vit_models.txt`.
- ViT full-family alpha metrics are also unavailable from current local artifacts. The original ViT PCA run did not keep an `all_layerdata.npy` archive, and only one raw local model directory remains under `vit_mm/hug`, so there is no faithful way to reconstruct the original 470-model ViT alpha matrix from what is currently on disk.
- LLAMA and Flan-T5 do not yet have a comparable local task-evaluation pass in this package. The current package therefore reports compression, alpha diagnostics, and the GPT2 generative comparison only.

