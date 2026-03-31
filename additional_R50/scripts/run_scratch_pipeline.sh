#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/download_or_train_models.sh"
"${SCRIPT_DIR}/compute_alpha_metrics.sh"
"${SCRIPT_DIR}/reconstruct_and_evaluate.sh"
"${SCRIPT_DIR}/low_rank_calibration.sh"
