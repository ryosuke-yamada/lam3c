#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_ROOT/.." && pwd)

source "$PROJECT_DIR/scripts/common/config_common.sh"

STAGE=${1:-${STAGE:-}}
DATASET_CONFIG=${2:-${DATASET_CONFIG:-}}

if [ -z "$STAGE" ] || [ -z "$DATASET_CONFIG" ]; then
  echo "Usage: $0 <segmentation|pi3|vggt> <dataset_config>" >&2
  exit 1
fi

dataset_gen_load_config "$PROJECT_DIR" "$DATASET_CONFIG"

case "$STAGE" in
  segmentation)
    SCRIPT_DIR="$PROJECT_DIR/scripts/segmentation"
    source "$PROJECT_DIR/scripts/common/segmentation_submit_common.sh"

    VIDEO_ROOT=${VIDEO_ROOT:-$(dataset_gen_get_prefixed_var SEG VIDEO_ROOT)}
    OUTPUT_ROOT=${OUTPUT_ROOT:-$(dataset_gen_get_prefixed_var SEG OUTPUT_ROOT)}
    GPU_IDS=${GPU_IDS:-$(dataset_gen_get_prefixed_var SEG GPU_IDS)}
    CONCURRENCY=${CONCURRENCY:-$(dataset_gen_get_prefixed_var SEG CONCURRENCY)}
    NUM_SHARDS=${NUM_SHARDS:-$(dataset_gen_get_prefixed_var SEG NUM_SHARDS)}
    LOG_BASE="$(dataset_gen_get_prefixed_var SEG LOG_PREFIX).${PBS_JOBID:-noid}"
    RUNNER_MODE=${RUNNER_MODE:-$(dataset_gen_get_prefixed_var SEG RUNNER_MODE)}

    dataset_gen_run_segmentation_submit
    ;;

  pi3)
    SCRIPT_DIR="$PROJECT_DIR/scripts/pi3"
    source "$PROJECT_DIR/scripts/common/pi3_submit_common.sh"

    LAYOUT=${LAYOUT:-$(dataset_gen_get_prefixed_var PI3 LAYOUT)}
    INPUT_BASE=${INPUT_BASE:-$(dataset_gen_get_prefixed_var PI3 INPUT_BASE)}
    OUTPUT_BASE=${OUTPUT_BASE:-$(dataset_gen_get_prefixed_var PI3 OUTPUT_BASE)}
    INTERVAL=${INTERVAL:-$(dataset_gen_get_prefixed_var PI3 INTERVAL)}
    NUM_GPUS=${NUM_GPUS:-$(dataset_gen_get_prefixed_var PI3 NUM_GPUS)}
    PIXEL_LIMIT=${PIXEL_LIMIT:-$(dataset_gen_get_prefixed_var PI3 PIXEL_LIMIT)}
    MAX_ENTRIES=${MAX_ENTRIES:-$(dataset_gen_get_prefixed_var PI3 MAX_ENTRIES)}
    TARGET_FRAMES=${TARGET_FRAMES:-$(dataset_gen_get_prefixed_var PI3 TARGET_FRAMES)}
    IMAGE_TARGET_FRAMES=${IMAGE_TARGET_FRAMES:-$(dataset_gen_get_prefixed_var PI3 IMAGE_TARGET_FRAMES)}
    PRESERVE_ORDER=${PRESERVE_ORDER:-$(dataset_gen_get_prefixed_var PI3 PRESERVE_ORDER)}
    VIDEO_SKIP_SECONDS=${VIDEO_SKIP_SECONDS:-$(dataset_gen_get_prefixed_var PI3 VIDEO_SKIP_SECONDS)}
    ADJUST_VIDEO_INTERVAL=${ADJUST_VIDEO_INTERVAL:-$(dataset_gen_get_prefixed_var PI3 ADJUST_VIDEO_INTERVAL)}
    NUM_SHARDS=${NUM_SHARDS:-$(dataset_gen_get_prefixed_var PI3 NUM_SHARDS)}
    SCENE_JSON=${SCENE_JSON:-$(dataset_gen_get_prefixed_var PI3 SCENE_JSON)}
    OVERWRITE_EXISTING=${OVERWRITE_EXISTING:-$(dataset_gen_get_prefixed_var PI3 OVERWRITE_EXISTING)}
    INCLUDE_PROCESSED=${INCLUDE_PROCESSED:-$(dataset_gen_get_prefixed_var PI3 INCLUDE_PROCESSED)}
    LOG_BASE="$(dataset_gen_get_prefixed_var PI3 LOG_PREFIX).${PBS_JOBID:-noid}.${PBS_ARRAY_INDEX:-noidx}"
    START_MESSAGE=${START_MESSAGE:-$DATASET_LABEL}

    dataset_gen_run_pi3_submit
    ;;

  vggt)
    SCRIPT_DIR="$PROJECT_DIR/scripts/vggt"
    source "$PROJECT_DIR/scripts/common/vggt_submit_common.sh"

    VIDEO_BASE=${VIDEO_BASE:-$(dataset_gen_get_prefixed_var VGGT VIDEO_BASE)}
    PI3_BASE=${PI3_BASE:-$(dataset_gen_get_prefixed_var VGGT PI3_BASE)}
    OUTPUT_BASE=${OUTPUT_BASE:-$(dataset_gen_get_prefixed_var VGGT OUTPUT_BASE)}
    WORK_BASE_REL=${WORK_BASE_REL:-$(dataset_gen_get_prefixed_var VGGT WORK_BASE_REL)}
    WORK_BASE=${WORK_BASE:-$PROJECT_DIR/${WORK_BASE_REL:-tmp/vggt}}
    TARGET_IMAGES=${TARGET_IMAGES:-$(dataset_gen_get_prefixed_var VGGT TARGET_IMAGES)}
    PIXEL_LIMIT=${PIXEL_LIMIT:-$(dataset_gen_get_prefixed_var VGGT PIXEL_LIMIT)}
    INTERVAL=${INTERVAL:-$(dataset_gen_get_prefixed_var VGGT INTERVAL)}
    CONF_THRES_VALUE=${CONF_THRES_VALUE:-$(dataset_gen_get_prefixed_var VGGT CONF_THRES_VALUE)}
    NUM_SHARDS=${NUM_SHARDS:-$(dataset_gen_get_prefixed_var VGGT NUM_SHARDS)}
    LOG_BASE="$(dataset_gen_get_prefixed_var VGGT LOG_PREFIX).${PBS_JOBID:-noid}.${PBS_ARRAY_INDEX:-noidx}"

    dataset_gen_run_vggt_submit
    ;;

  *)
    echo "[ERROR] Unsupported stage: $STAGE" >&2
    exit 1
    ;;
esac
