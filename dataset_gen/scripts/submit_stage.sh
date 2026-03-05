#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_ROOT/.." && pwd)

source "$PROJECT_DIR/scripts/common/config_common.sh"

STAGE=${1:-}
DATASET_CONFIG=${2:-}

if [ -z "$STAGE" ] || [ -z "$DATASET_CONFIG" ]; then
  echo "Usage: $0 <segmentation|pi3|vggt> <dataset_config>" >&2
  exit 1
fi

if ! command -v qsub >/dev/null 2>&1; then
  echo "[ERROR] qsub not found in PATH" >&2
  exit 1
fi

dataset_gen_load_config "$PROJECT_DIR" "$DATASET_CONFIG"

PREFIX=$(dataset_gen_stage_prefix "$STAGE")

JOB_NAME=${JOB_NAME:-$(dataset_gen_get_prefixed_var "$PREFIX" JOB_NAME)}
QUEUE=${QUEUE:-$(dataset_gen_get_prefixed_var "$PREFIX" PBS_QUEUE)}
PROJECT_CODE=${PROJECT_CODE:-$(dataset_gen_get_prefixed_var "$PREFIX" PBS_PROJECT)}
SELECT_SPEC=${SELECT_SPEC:-$(dataset_gen_get_prefixed_var "$PREFIX" PBS_SELECT)}
WALLTIME=${WALLTIME:-$(dataset_gen_get_prefixed_var "$PREFIX" PBS_WALLTIME)}

if [ -z "$JOB_NAME" ] || [ -z "$QUEUE" ] || [ -z "$PROJECT_CODE" ] || [ -z "$SELECT_SPEC" ] || [ -z "$WALLTIME" ]; then
  echo "[ERROR] Incomplete PBS config for stage=$STAGE dataset=$DATASET_ID" >&2
  exit 1
fi

echo "[INFO] stage: $STAGE"
echo "[INFO] dataset: $DATASET_ID"
echo "[INFO] config: $DATASET_CONFIG_PATH"
echo "[INFO] qsub -N $JOB_NAME -q $QUEUE -P $PROJECT_CODE -l select=$SELECT_SPEC -l walltime=$WALLTIME"

qsub -V \
  -N "$JOB_NAME" \
  -q "$QUEUE" \
  -P "$PROJECT_CODE" \
  -l "select=$SELECT_SPEC" \
  -l "walltime=$WALLTIME" \
  -v "STAGE=$STAGE,DATASET_CONFIG=$DATASET_CONFIG_PATH" \
  "$PROJECT_DIR/scripts/run_stage.sh"
