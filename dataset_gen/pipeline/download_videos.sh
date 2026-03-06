#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_ROOT/.." && pwd)

source "$PROJECT_DIR/pipeline/common/config_common.sh"
source "$PROJECT_DIR/pipeline/common/job_common.sh"

DATASET_CONFIG=${1:-default}
dataset_gen_load_config "$PROJECT_DIR" "$DATASET_CONFIG"

: "${VIDEO_LIST_CSV:?VIDEO_LIST_CSV is required}"
: "${DOWNLOAD_ROOT:?DOWNLOAD_ROOT is required}"

DATASET_GEN_STRICT_DOWNLOAD=${DATASET_GEN_STRICT_DOWNLOAD:-0}


dataset_gen_prepare_project_dir
dataset_gen_load_modules

VENV_ACTIVATE=${VENV_ACTIVATE:-$PROJECT_DIR/.venv_pi3/bin/activate}
dataset_gen_activate_venv "$VENV_ACTIVATE" "dataset_gen venv"

echo "[INFO] dataset: $DATASET_ID"
echo "[INFO] config: $DATASET_CONFIG_PATH"
echo "[INFO] video list: $VIDEO_LIST_CSV"
echo "[INFO] download root: $DOWNLOAD_ROOT"
if [ -n "${DOWNLOAD_ARCHIVE:-}" ]; then
  echo "[INFO] download archive: $DOWNLOAD_ARCHIVE"
fi
if [ -n "${DOWNLOAD_FAILURE_LOG:-}" ]; then
  echo "[INFO] download failure log: $DOWNLOAD_FAILURE_LOG"
fi

set +e
python "$PROJECT_DIR/pipeline/download_video_list.py" \
  --csv "$VIDEO_LIST_CSV" \
  --output-root "$DOWNLOAD_ROOT" \
  --archive "${DOWNLOAD_ARCHIVE:-}" \
  --failure-log "${DOWNLOAD_FAILURE_LOG:-}"
status=$?
set -e

if [ "$status" -eq 2 ] && [ "$DATASET_GEN_STRICT_DOWNLOAD" != "1" ]; then
  echo "[WARN] Some downloads failed; continuing because DATASET_GEN_STRICT_DOWNLOAD=0"
  exit 0
fi

exit "$status"
