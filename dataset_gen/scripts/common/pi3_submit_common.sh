#!/usr/bin/env bash

source "$PROJECT_DIR/scripts/common/job_common.sh"

dataset_gen_run_pi3_submit() {
  dataset_gen_resolve_array_shard "$NUM_SHARDS"
  dataset_gen_prepare_project_dir
  dataset_gen_setup_logging "$LOG_BASE"
  dataset_gen_load_modules
  dataset_gen_print_runtime_info

  VENV_PATH=${VENV_PATH:-$PROJECT_DIR/.venv_pi3/bin/activate}
  dataset_gen_activate_venv "$VENV_PATH" "Pi3 venv"

  echo "[INFO] INPUT_BASE: $INPUT_BASE"
  echo "[INFO] OUTPUT_BASE: $OUTPUT_BASE"
  echo "[INFO] LAYOUT: $LAYOUT"
  echo "[INFO] INTERVAL: $INTERVAL"
  echo "[INFO] NUM_GPUS: $NUM_GPUS"
  echo "[INFO] VIDEO_TARGET_FRAMES: $TARGET_FRAMES"
  echo "[INFO] Starting Pi3 batch processing for $START_MESSAGE"

  EXTRA_ARGS=()
  if [ -n "${SCENE_JSON:-}" ]; then
    echo "[INFO] Scene JSON: $SCENE_JSON"
    EXTRA_ARGS+=(--scene_json "$SCENE_JSON")
  fi
  if [ "${PRESERVE_ORDER:-0}" = "1" ]; then
    EXTRA_ARGS+=(--preserve_order)
  fi
  if [ "${ADJUST_VIDEO_INTERVAL:-0}" = "1" ]; then
    EXTRA_ARGS+=(--video_adjust_for_high_fps)
  fi
  if [ "${INCLUDE_PROCESSED:-0}" = "1" ] || [ "${OVERWRITE_EXISTING:-0}" = "1" ]; then
    echo "[INFO] include_processed enabled"
    EXTRA_ARGS+=(--include_processed)
  fi
  if [ "${OVERWRITE_EXISTING:-0}" = "1" ]; then
    echo "[INFO] overwrite_existing enabled"
    EXTRA_ARGS+=(--overwrite_existing)
  fi

  python "$SCRIPT_DIR/pi3_batch_datasets.py" \
    --layout "$LAYOUT" \
    --input_base "$INPUT_BASE" \
    --output_base "$OUTPUT_BASE" \
    --interval "$INTERVAL" \
    --num_gpus "$NUM_GPUS" \
    --num_shards "$NUM_SHARDS" \
    --shard_id "$SHARD_ID" \
    --pixel_limit "$PIXEL_LIMIT" \
    --max_entries "$MAX_ENTRIES" \
    --video_skip_seconds "$VIDEO_SKIP_SECONDS" \
    --video_target_frames "$TARGET_FRAMES" \
    --image_target_frames "$IMAGE_TARGET_FRAMES" \
    "${EXTRA_ARGS[@]}"

  echo "[INFO] Done."
}
