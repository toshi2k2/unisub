# CNN Workflow (Paper ResNet-50 Scratch Family)

This folder contains the scratch ResNet-50 training and shared-subspace workflow used in the paper.

Use this folder for the main CNN experiments. The later 9-model spatial-wise and alpha-based follow-up package now lives in [`../additional_R50`](../additional_R50/README.md), and the consolidated follow-up summaries live in [`../Main_weight_analysis`](../Main_weight_analysis/README.md).

## What This Folder Contains

- `cnn.py`: train ResNet-50 from scratch with `weights=None`
- `subspace.py`: compute shared eigenspaces and reconstruct checkpoints
- `cifar10.sh`, `cifar100.sh`, `caltech.sh`, `eurosat.sh`, `imagenet.sh`: convenience training sweeps
- `subspace.sh`: full shared-subspace flow

## Position In The Repo

- `CNN/`: original paper scratch-family code path
- `additional_R50/`: later 9-model follow-up using spatial-wise calculation and alpha analysis
- `Main_weight_analysis/`: follow-up analysis summaries that sit alongside the main experiment directories

## Supported Datasets

`cnn.py` supports:

- `cifar10`
- `cifar100`
- `pets`
- `caltech101`
- `eurosat`
- `imagenet`

For non-ImageNet datasets, data is auto-downloaded under `--data_root`.

### ImageNet Layout

For `--dataset imagenet`, the script expects either:

- `<data_root>/train` and `<data_root>/val`
- `<data_root>/imagenet/train` and `<data_root>/imagenet/val`

Each split must be class-subfolder based.

## Train From Scratch

```bash
python cnn.py \
  --dataset cifar100 \
  --data_root ./data \
  --epochs 100 \
  --learning_rate 1e-3 \
  --batch_size 256 \
  --device cuda:0
```

Important notes:

- the model is initialized from scratch via `resnet50(weights=None)`
- the best checkpoint is written to `models/`
- the local sweep scripts are the fastest starting point if you want the original paper-style workflow

## Build The Shared Subspace

Compute eigenspaces only:

```bash
python subspace.py \
  --mode compute-eig \
  --model_paths "cifar10:/path/cifar10.pt,cifar100:/path/cifar100.pt,eurosat:/path/eurosat.pt,pets:/path/pets.pt,imagenet:/path/imagenet.pt" \
  --eig_path 64_resnet50.pt \
  --rank 64 \
  --niter 50
```

Reconstruct only:

```bash
python subspace.py \
  --mode reconstruct \
  --model_paths "cifar10:/path/cifar10.pt,cifar100:/path/cifar100.pt,eurosat:/path/eurosat.pt,pets:/path/pets.pt,imagenet:/path/imagenet.pt" \
  --eig_path 64_resnet50.pt \
  --out_dir ./outputs \
  --recon_suffix _64_rc
```

Full flow:

```bash
python subspace.py \
  --mode all \
  --eig_path 64_resnet50.pt \
  --out_dir ./outputs \
  --recon_suffix _64_rc
```

Or simply:

```bash
bash subspace.sh
```
