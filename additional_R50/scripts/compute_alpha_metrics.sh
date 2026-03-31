#!/usr/bin/env bash
set -euo pipefail

: "${RESNET50_REPO_DIR:?Set RESNET50_REPO_DIR to a checkout containing the ResNet50 UniSub scripts.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_MANIFEST="${DOWNLOAD_MANIFEST:-${PACKAGE_DIR}/results/download_manifest_distinct.json}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${PACKAGE_DIR}/results/alpha_unisub_factorized_spatial_v1}"

"${PYTHON_BIN}" "${RESNET50_REPO_DIR}/resnet50_alpha_unisub.py" \
  --manifest-path "${DOWNLOAD_MANIFEST}" \
  --output-dir "${ANALYSIS_ROOT}" \
  --reconstruction-mode spatial \
  --factorized-kernel-size 3 \
  --variance-thresholds 0.60,0.70,0.80,0.85,0.90,0.95 \
  --coefficient-keep-fractions 1.0 \
  --reconstruction-stage-min 2 \
  --reconstruction-stage-max 3 \
  --verbose
