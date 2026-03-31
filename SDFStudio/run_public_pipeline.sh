#!/usr/bin/env bash
set -euo pipefail

REPO_DIR=""
RUN_ROOT="/mnt/data0/prakhar/sdfstudio_uwsh"
ANALYSIS_ROOT="/mnt/data0/prakhar/sdfstudio_uwsh/analysis_alpha_unisub_v1"
DEVICE="cuda:0"
ALPHA_MIN_FRACTION="0.3"
EVAL_MAX_IMAGES="4"
FIGURE_IMAGES_PER_SCENE="4"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --analysis-root)
      ANALYSIS_ROOT="$2"
      shift 2
      ;;
    --device)
      DEVICE="$2"
      shift 2
      ;;
    --alpha-min-fraction-in-range)
      ALPHA_MIN_FRACTION="$2"
      shift 2
      ;;
    --eval-max-images)
      EVAL_MAX_IMAGES="$2"
      shift 2
      ;;
    --figure-images-per-scene)
      FIGURE_IMAGES_PER_SCENE="$2"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$REPO_DIR" ]]; then
  if [[ -n "${SDFSTUDIO_REPO_DIR:-}" ]]; then
    REPO_DIR="$SDFSTUDIO_REPO_DIR"
  else
    echo "Missing --repo-dir or SDFSTUDIO_REPO_DIR" >&2
    exit 1
  fi
fi

cd "$REPO_DIR"

SDFSTUDIO_USE_VENDOR_PY=1 python scripts/neus_plainmlp_alpha_unisub.py \
  --run-root "$RUN_ROOT" \
  --analysis-root "$ANALYSIS_ROOT" \
  --alpha-min-fraction-in-range "$ALPHA_MIN_FRACTION" \
  --eval-max-images "$EVAL_MAX_IMAGES" \
  --figure-images-per-scene "$FIGURE_IMAGES_PER_SCENE" \
  --device "$DEVICE" \
  "${EXTRA_ARGS[@]}"
