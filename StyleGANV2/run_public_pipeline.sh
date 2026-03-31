#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Run the public StyleGAN2 UniSub pipeline end-to-end.

Usage:
  bash run_public_pipeline.sh --repo-dir /path/to/awesome-pretrained-stylegan2 [options]

Options:
  --python PATH                 Python executable to use. Default: python
  --repo-dir PATH               Path to the awesome-pretrained-stylegan2 repo. Can also be set via STYLEGAN_REPO_DIR
  --device DEVICE               Device for the alpha/reconstruction stage. Default: cpu
  --num-samples N               Number of samples per target for IID/OOD metrics. Default: 512
  --batch-size N                Batch size for generation and detector passes. Default: 4
  --alpha-q-threshold Q         Final q_l threshold. Default: 0.05
  --alpha-std-max S             Alpha std cap. Default: 2.0
  --skip-download               Reuse existing downloaded checkpoints and manifest
  --skip-weightwatcher          Reuse existing all20 WeightWatcher outputs
  --skip-reconstruction         Only run the alpha stage without IID/OOD image metrics
  --verbose                     Enable verbose logging on Python entrypoints
  -h, --help                    Show this message

Outputs are written under:
  ./models
  ./results
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${STYLEGAN_REPO_DIR:-}"

PYTHON_BIN="python"
DEVICE="cpu"
NUM_SAMPLES="512"
BATCH_SIZE="4"
ALPHA_Q_THRESHOLD="0.05"
ALPHA_STD_MAX="2.0"
SKIP_DOWNLOAD="0"
SKIP_WEIGHTWATCHER="0"
SKIP_RECONSTRUCTION="0"
VERBOSE="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --alpha-q-threshold)
      ALPHA_Q_THRESHOLD="$2"
      shift 2
      ;;
    --alpha-std-max)
      ALPHA_STD_MAX="$2"
      shift 2
      ;;
    --skip-download)
      SKIP_DOWNLOAD="1"
      shift
      ;;
    --skip-weightwatcher)
      SKIP_WEIGHTWATCHER="1"
      shift
      ;;
    --skip-reconstruction)
      SKIP_RECONSTRUCTION="1"
      shift
      ;;
    --verbose)
      VERBOSE="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${REPO_DIR}" ]]; then
  for candidate in \
    "${SCRIPT_DIR}/.." \
    "${PWD}" \
    "${PWD}/awesome-pretrained-stylegan2"
  do
    if [[ -f "${candidate}/download_stylegan_models.py" ]] && \
       [[ -f "${candidate}/stylegan_all20_weightwatcher_eval.py" ]] && \
       [[ -f "${candidate}/stylegan_alpha_common20_eval.py" ]]; then
      REPO_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${REPO_DIR}" ]]; then
  echo "Could not locate the awesome-pretrained-stylegan2 repo." >&2
  echo "Pass --repo-dir /path/to/awesome-pretrained-stylegan2 or set STYLEGAN_REPO_DIR." >&2
  exit 1
fi

REPO_DIR="$(cd "${REPO_DIR}" && pwd)"

for required in \
  "${REPO_DIR}/download_stylegan_models.py" \
  "${REPO_DIR}/stylegan_all20_weightwatcher_eval.py" \
  "${REPO_DIR}/stylegan_alpha_common20_eval.py" \
  "${REPO_DIR}/models.json"
do
  if [[ ! -f "${required}" ]]; then
    echo "Missing required repo file: ${required}" >&2
    exit 1
  fi
done

MODELS_DIR="${SCRIPT_DIR}/models"
RESULTS_DIR="${SCRIPT_DIR}/results"
MANIFEST_PATH="${RESULTS_DIR}/download_manifest.json"
WW_OUTPUT_DIR="${RESULTS_DIR}/all20_weightwatcher_v1"
ALPHA_OUTPUT_DIR="${RESULTS_DIR}/all20_alpha_common20_v1"
WEIGHTWATCHER_CSV="${WW_OUTPUT_DIR}/weightwatcher/all_models.csv"

mkdir -p "${MODELS_DIR}" "${RESULTS_DIR}" "${SCRIPT_DIR}/.mplconfig" "${SCRIPT_DIR}/.cache"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${SCRIPT_DIR}/.mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${SCRIPT_DIR}/.cache}"

COMMON_ARGS=()
if [[ "${VERBOSE}" == "1" ]]; then
  COMMON_ARGS+=(--verbose)
fi

run_cmd() {
  echo
  echo "+ $*"
  "$@"
}

if [[ "${SKIP_DOWNLOAD}" != "1" ]]; then
  run_cmd \
    "${PYTHON_BIN}" "${REPO_DIR}/download_stylegan_models.py" \
    --destination-dir "${MODELS_DIR}" \
    --manifest-path "${MANIFEST_PATH}" \
    "${COMMON_ARGS[@]}"
fi

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  echo "Missing manifest: ${MANIFEST_PATH}" >&2
  exit 1
fi

if [[ "${SKIP_WEIGHTWATCHER}" != "1" ]]; then
  run_cmd \
    "${PYTHON_BIN}" "${REPO_DIR}/stylegan_all20_weightwatcher_eval.py" \
    --manifest-path "${MANIFEST_PATH}" \
    --output-dir "${WW_OUTPUT_DIR}" \
    --skip-reconstruction \
    --device cpu \
    "${COMMON_ARGS[@]}"
fi

if [[ ! -f "${WEIGHTWATCHER_CSV}" ]]; then
  echo "Missing WeightWatcher CSV: ${WEIGHTWATCHER_CSV}" >&2
  exit 1
fi

ALPHA_ARGS=(
  --manifest-path "${MANIFEST_PATH}"
  --weightwatcher-csv "${WEIGHTWATCHER_CSV}"
  --output-dir "${ALPHA_OUTPUT_DIR}"
  --device "${DEVICE}"
  --alpha-std-max "${ALPHA_STD_MAX}"
  --alpha-q-threshold "${ALPHA_Q_THRESHOLD}"
)

if [[ "${SKIP_RECONSTRUCTION}" == "1" ]]; then
  ALPHA_ARGS+=(--skip-reconstruction)
else
  ALPHA_ARGS+=(
    --num-samples "${NUM_SAMPLES}"
    --batch-size "${BATCH_SIZE}"
  )
fi

if [[ "${VERBOSE}" == "1" ]]; then
  ALPHA_ARGS+=(--verbose)
fi

run_cmd \
  "${PYTHON_BIN}" "${REPO_DIR}/stylegan_alpha_common20_eval.py" \
  "${ALPHA_ARGS[@]}"

echo
echo "Done."
echo "Repo root: ${REPO_DIR}"
echo "Manifest: ${MANIFEST_PATH}"
echo "WeightWatcher root: ${WW_OUTPUT_DIR}"
echo "Alpha/UniSub root: ${ALPHA_OUTPUT_DIR}"
echo "Retained layers: ${ALPHA_OUTPUT_DIR}/alpha/retained_layers.csv"
if [[ "${SKIP_RECONSTRUCTION}" != "1" ]]; then
  echo "IID/OOD metrics: ${ALPHA_OUTPUT_DIR}/paper_metric_summary.csv"
  echo "Grids: ${ALPHA_OUTPUT_DIR}/grids"
fi
