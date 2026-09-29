#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_DIR="${DATA_DIR:-data}"
BACKBLAZE_FILE="${BACKBLAZE_FILE:-disk_failure_v2.csv}"
RESULT_DIR="${RESULT_DIR:-results/final/backblaze}"
RUN_NAME="${RUN_NAME:-backblaze_initial_to_maintenance_final_10seeds}"

python -m early_prediction_v2.runner \
  --dataset backblaze \
  --data-dir "${DATA_DIR}" \
  --backblaze-file "${BACKBLAZE_FILE}" \
  --upstream-config-file early_prediction_v2/configs/drift_detector_config_v1.json \
  --backblaze-model-config-file early_prediction_v2/configs/backblaze_maintenance_v1.json \
  --models if pca ae deep_svdd neutral_ad \
  --strategies stationary periodic drift_adaptive \
  --drift-detectors kswin lsdd kdqtree d3 dawidd \
  --feature-mode numeric_only \
  --backblaze-log-transform \
  --backblaze-feature-group full \
  --reference-start 1 \
  --reference-end 18 \
  --eval-start 19 \
  --eval-end 36 \
  --periodic-interval 1 \
  --parameter-policy auto \
  --drift-feature-view augmented \
  --drift-ref-periods 18 \
  --drift-confirm-consecutive 1 \
  --drift-cooldown-periods 0 \
  --drift-seed-mode experiment \
  --n-experiments 10 \
  --seed 50000 \
  --seed-step 1000 \
  --result-dir "${RESULT_DIR}" \
  --run-name "${RUN_NAME}" \
  --resume
