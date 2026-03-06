#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_ROOT/.." && pwd)

DATASET_CONFIG=${1:-}

if [ -z "$DATASET_CONFIG" ]; then
  echo "Usage: $0 <dataset_config>" >&2
  exit 1
fi

echo "[INFO] submitting segmentation stage for $DATASET_CONFIG"
SEG_OUTPUT=$("$PROJECT_DIR/scripts/submit_stage.sh" segmentation "$DATASET_CONFIG")
printf '%s\n' "$SEG_OUTPUT"
SEG_JOB_ID=$(printf '%s\n' "$SEG_OUTPUT" | awk 'NF { line=$0 } END { print line }')

if [ -z "$SEG_JOB_ID" ]; then
  echo "[ERROR] failed to capture segmentation job id" >&2
  exit 1
fi

echo "[INFO] submitting pi3 stage after segmentation job $SEG_JOB_ID"
QSUB_AFTEROK="$SEG_JOB_ID" "$PROJECT_DIR/scripts/submit_stage.sh" pi3 "$DATASET_CONFIG"
