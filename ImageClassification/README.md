# ViT Adapter Experiments

This directory covers the image-classification adapter experiments built around Vision Transformers.

The main training script is [train_vit.py](./train_vit.py). It supports:

- `lora`
- `vera`
- `elora` (`SubspaceAdapter`)
- `none` for full fine-tuning

Datasets supported in the script include `CIFAR10`, `CIFAR100`, `StanfordCars`, `Flowers102`, `Food101`, and ImageNet-style folder layouts.

## Main Files

- [train_vit.py](./train_vit.py): training and evaluation entry point
- [run_training.sh](./run_training.sh): wrapper for the standard experiment configurations
- [utils.py](./utils.py): subset loading, LoRA consolidation, eigenvector extraction, and initialization helpers

## Quick Start

Train a LoRA baseline:

```bash
bash run_training.sh lora CIFAR100
```

Train a SubspaceAdapter model:

```bash
bash run_training.sh elora CIFAR100
```

Important detail: the `elora` path initializes SubspaceAdapter from previously trained LoRA checkpoints, so the LoRA checkpoints need to exist first.

For more complete PEFT details and the maintained adapter-focused code path, refer to [EigenLoRA](https://github.com/toshi2k2/EigenLoRA/).
