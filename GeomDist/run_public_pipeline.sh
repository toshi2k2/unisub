#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${GEOMDIST_REPO_DIR:-}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(pwd)/results/analysis_alpha_unisub_v1}"
DEVICE="cuda:0"
SAMPLE_POINTS="16384"
REFERENCE_POINTS="16384"
NUM_STEPS="64"
FIGURE_MODELS="valley,wukong,tower,lamp"
FIGURE_POINTS="8000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --sample-points)
      SAMPLE_POINTS="$2"
      shift 2
      ;;
    --reference-points)
      REFERENCE_POINTS="$2"
      shift 2
      ;;
    --num-steps)
      NUM_STEPS="$2"
      shift 2
      ;;
    --figure-models)
      FIGURE_MODELS="$2"
      shift 2
      ;;
    --figure-points)
      FIGURE_POINTS="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${REPO_DIR}" ]]; then
  echo "Set --repo-dir or GEOMDIST_REPO_DIR to the main GeomDist repository." >&2
  exit 1
fi

if [[ ! -f "${REPO_DIR}/geomdist_alpha_unisub.py" ]]; then
  echo "Could not find geomdist_alpha_unisub.py under ${REPO_DIR}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUTPUT_ROOT}")"
mkdir -p "$(pwd)/.mplconfig"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$(pwd)/.mplconfig"

python "${REPO_DIR}/geomdist_alpha_unisub.py" \
  --analysis-root "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  --sample-points "${SAMPLE_POINTS}" \
  --reference-points "${REFERENCE_POINTS}" \
  --num-steps "${NUM_STEPS}"

python "${REPO_DIR}/geomdist_finalize_analysis.py" \
  --analysis-root "${OUTPUT_ROOT}" \
  --figure-models "${FIGURE_MODELS}" \
  --figure-points "${FIGURE_POINTS}"

echo "GeomDist UniSub analysis complete: ${OUTPUT_ROOT}"
