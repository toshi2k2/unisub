#!/usr/bin/env bash
set -euo pipefail

: "${RESNET50_REPO_DIR:?Set RESNET50_REPO_DIR to a checkout containing the ResNet50 UniSub scripts.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cpu}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DOWNLOAD_MANIFEST="${DOWNLOAD_MANIFEST:-${PACKAGE_DIR}/results/download_manifest_distinct.json}"
ANALYSIS_ROOT="${ANALYSIS_ROOT:-${PACKAGE_DIR}/results/alpha_unisub_factorized_spatial_v1}"
EVAL_ROOT="${EVAL_ROOT:-${PACKAGE_DIR}/results/eval_factorized_spatial_v1_merged}"

"${PYTHON_BIN}" "${RESNET50_REPO_DIR}/evaluate_resnet50_unisub.py" \
  --model-manifest "${DOWNLOAD_MANIFEST}" \
  --reconstruction-manifest "${ANALYSIS_ROOT}/reconstruction_manifest.json" \
  --compression-csv "${ANALYSIS_ROOT}/compression_summary.csv" \
  --output-dir "${EVAL_ROOT}" \
  --device "${DEVICE}" \
  --verbose
