#!/usr/bin/env bash
set -euo pipefail

: "${RESNET50_REPO_DIR:?Set RESNET50_REPO_DIR to a checkout containing the ResNet50 UniSub scripts.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_MANIFEST="${DOWNLOAD_MANIFEST:-${PACKAGE_DIR}/results/download_manifest_distinct.json}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${PACKAGE_DIR}/results/alpha_unisub_factorized_spatial_v1}"
IID_OUT="${IID_OUT:-${PACKAGE_DIR}/results/calib_factorized_spatial_vt60_80_iid_coeff_v1}"
OOD_OUT="${OOD_OUT:-${PACKAGE_DIR}/results/calib_factorized_spatial_vt60_80_ood_coeff_fc_probe2048_v1}"
TARGETS="cifar10__edadaltocg_resnet50_cifar10,cifar100__edadaltocg_resnet50_cifar100,svhn__edadaltocg_resnet50_svhn"

"${PYTHON_BIN}" "${RESNET50_REPO_DIR}/calibrate_resnet50_unisub.py" \
  --model-manifest "${DOWNLOAD_MANIFEST}" \
  --reconstruction-manifest "${ANALYSIS_ROOT}/reconstruction_manifest.json" \
  --output-dir "${IID_OUT}" \
  --targets "${TARGETS}" \
  --protocols iid \
  --variance-thresholds 0.60,0.70,0.80 \
  --coefficient-keep-fractions 1.0 \
  --calibration-split train \
  --calibration-samples 2048 \
  --steps 50 \
  --lr 1e-4 \
  --num-workers 0 \
  --device "${DEVICE}" \
  --verbose

"${PYTHON_BIN}" "${RESNET50_REPO_DIR}/calibrate_resnet50_unisub.py" \
  --model-manifest "${DOWNLOAD_MANIFEST}" \
  --reconstruction-manifest "${ANALYSIS_ROOT}/reconstruction_manifest.json" \
  --output-dir "${OOD_OUT}" \
  --targets "${TARGETS}" \
  --protocols ood \
  --variance-thresholds 0.60,0.70,0.80 \
  --coefficient-keep-fractions 1.0 \
  --calibration-split train \
  --calibration-samples 2048 \
  --eval-samples 2048 \
  --steps 50 \
  --lr 1e-4 \
  --num-workers 0 \
  --tune-last-layer \
  --device "${DEVICE}" \
  --verbose
