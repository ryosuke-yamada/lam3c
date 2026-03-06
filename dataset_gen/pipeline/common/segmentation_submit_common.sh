#!/usr/bin/env bash

source "$PROJECT_DIR/pipeline/common/job_common.sh"

dataset_gen_run_segmentation_submit() {
  dataset_gen_prepare_project_dir
  dataset_gen_setup_logging "$LOG_BASE"

  echo "[INFO] JobID: ${PBS_JOBID:-none}"
  echo "[INFO] VIDEO_ROOT:  $VIDEO_ROOT"
  echo "[INFO] OUTPUT_ROOT: $OUTPUT_ROOT"
  echo "[INFO] GPU_IDS:    $GPU_IDS"
  echo "[INFO] CONCURRENCY: $CONCURRENCY"
  if [ -n "${NUM_SHARDS:-}" ]; then
    echo "[INFO] NUM_SHARDS: $NUM_SHARDS"
  fi

  dataset_gen_load_modules

  VENV_ACTIVATE=${VENV_ACTIVATE:-$PROJECT_DIR/.venv_pi3/bin/activate}
  dataset_gen_activate_venv "$VENV_ACTIVATE" "segmentation venv"
  dataset_gen_print_runtime_info

  if [ "$RUNNER_MODE" = "batch" ]; then
    dataset_gen_resolve_optional_shard
    GPU_IDS="$GPU_IDS" NUM_SHARDS="${NUM_SHARDS:-0}" SHARD_ID="$SHARD_ID" \
      "$SCRIPT_DIR/batch_run_label_segmentation.sh" "$VIDEO_ROOT" "$OUTPUT_ROOT" "$CONCURRENCY"
  else
    GPU_IDS="$GPU_IDS" "$SCRIPT_DIR/batch_run_label_segmentation_queue.sh" \
      "$VIDEO_ROOT" "$OUTPUT_ROOT" "$CONCURRENCY"
  fi

  echo "[INFO] Job completed"
}
