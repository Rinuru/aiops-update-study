#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_DIR="${DATA_DIR:-data}"
GOOGLE_FILE="${GOOGLE_FILE:-google_job_failure_g2.csv}"
RESULT_DIR="${RESULT_DIR:-results/final/google}"
RUN_NAME="${RUN_NAME:-google_fixed_final_10seeds}"

python -m early_prediction_v2.runner \
  --dataset google \
  --data-dir "${DATA_DIR}" \
  --google-file "${GOOGLE_FILE}" \
  --upstream-config-file early_prediction_v2/configs/drift_detector_config_v1.json \
  --google-model-config-file early_prediction_v2/configs/google_model_config_v2.json \
  --models if pca ae deep_svdd neutral_ad \
  --strategies stationary periodic drift_adaptive \
  --drift-detectors kswin lsdd kdqtree d3 dawidd \
  --feature-mode numeric_only \
  --reference-start 1 \
  --reference-end 14 \
  --eval-start 15 \
  --eval-end 28 \
  --periodic-interval 1 \
  --parameter-policy auto \
  --drift-feature-view augmented \
  --drift-ref-periods 14 \
  --drift-confirm-consecutive 1 \
  --drift-cooldown-periods 0 \
  --drift-seed-mode experiment \
  --n-experiments 10 \
  --seed 50000 \
  --seed-step 1000 \
  --result-dir "${RESULT_DIR}" \
  --run-name "${RUN_NAME}" \
  --resume
