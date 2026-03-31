#!/usr/bin/env bash
set -euo pipefail

: "${RESNET50_REPO_DIR:?Set RESNET50_REPO_DIR to a checkout containing the ResNet50 UniSub scripts.}"
PYTHON_BIN="${PYTHON_BIN:-python}"
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${RESULTS_DIR:-${PACKAGE_DIR}/results}"
MODELS_DIR="${MODELS_DIR:-${PACKAGE_DIR}/models}"
TRAIN_OUT="${TRAIN_OUT:-${PACKAGE_DIR}/train_runs}"
CANDIDATES_JSON="${CANDIDATES_JSON:-${RESULTS_DIR}/source_candidates_distinct.json}"
DOWNLOAD_MANIFEST="${DOWNLOAD_MANIFEST:-${RESULTS_DIR}/download_manifest_distinct.json}"

mkdir -p "${RESULTS_DIR}" "${MODELS_DIR}" "${TRAIN_OUT}"

"${PYTHON_BIN}" "${RESNET50_REPO_DIR}/curate_scratch_resnet50_sources.py" \
  --output-path "${CANDIDATES_JSON}" \
  --verbose

"${PYTHON_BIN}" "${RESNET50_REPO_DIR}/download_scratch_resnet50_models.py" \
  --candidates-path "${CANDIDATES_JSON}" \
  --output-dir "${MODELS_DIR}" \
  --manifest-path "${DOWNLOAD_MANIFEST}" \
  --skip-existing \
  --verbose

# Optional local training examples for owned datasets:
# "${PYTHON_BIN}" "${RESNET50_REPO_DIR}/train_resnet50_scratch.py" --dataset cifar10 --output-root "${TRAIN_OUT}" --emit-script
# "${PYTHON_BIN}" "${RESNET50_REPO_DIR}/train_resnet50_scratch.py" --dataset cifar100 --output-root "${TRAIN_OUT}" --emit-script
# "${PYTHON_BIN}" "${RESNET50_REPO_DIR}/train_resnet50_scratch.py" --dataset svhn --output-root "${TRAIN_OUT}" --emit-script
