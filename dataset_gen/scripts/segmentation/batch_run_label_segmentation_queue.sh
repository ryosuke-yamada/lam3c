#!/usr/bin/env bash
# Dynamic task queue runner for room tour segmentation.
# Each worker repeatedly pops a video path from a shared queue and processes it.

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ $# -lt 3 ]; then
  echo "Usage: $0 <video_root> <output_root> <concurrency>" >&2
  exit 1
fi

VIDEO_ROOT="$1"
OUTPUT_ROOT="$2"
CONCURRENCY="$3"
GPU_IDS=${GPU_IDS:-}
QUEUE_FILE=${QUEUE_FILE:-"$OUTPUT_ROOT/_queue.txt"}

mkdir -p "$OUTPUT_ROOT"
QUEUE_DIR=$(dirname "$QUEUE_FILE")
mkdir -p "$QUEUE_DIR"

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_IDS"
GPU_COUNT=${#GPU_ARRAY[@]}

if [ $GPU_COUNT -eq 0 ] && command -v nvidia-smi >/dev/null 2>&1; then
  mapfile -t GPU_ARRAY < <(nvidia-smi --query-gpu=index --format=csv,noheader)
  GPU_COUNT=${#GPU_ARRAY[@]}
fi

ensure_queue() {
  local lock_file="$QUEUE_FILE.init.lock"
  exec 200>"$lock_file"
  flock -x 200
  if [ ! -s "$QUEUE_FILE" ]; then
    echo "[INFO] Initializing task queue at $QUEUE_FILE"
    python "$SCRIPT_DIR/init_segmentation_queue.py" "$VIDEO_ROOT" "$OUTPUT_ROOT" "$QUEUE_FILE" >&2
  fi
  flock -u 200
}

pop_task() {
  local video
  if video=$(python "$SCRIPT_DIR/pop_segmentation_queue.py" "$QUEUE_FILE" 2>/dev/null); then
    if [ -z "$video" ]; then
      return 1
    fi
    printf '%s' "$video"
    return 0
  fi
  return 1
}

process_video() {
  local video="$1"
  local gpu_id="$2"
  local rel="${video#$VIDEO_ROOT/}"
  if [ "$rel" = "$video" ]; then
    rel=$(basename "$video")
  fi
  local rel_dir=$(dirname "$rel")
  local filename=$(basename "$video")
  local stem="${filename%.*}"
  local out_dir="$OUTPUT_ROOT"
  if [ "$rel_dir" != "." ] && [ -n "$rel_dir" ]; then
    out_dir+="/$rel_dir/$stem"
  else
    out_dir+="/$stem"
  fi

  if [ -d "$out_dir/scenes" ] && find "$out_dir/scenes" -maxdepth 1 -type f -name '*.mp4' -print -quit >/dev/null 2>&1; then
    echo "[SKIP] Already processed: $video"
    return
  fi
  if [ -f "$out_dir/SKIP_NO_FRAMES" ]; then
    echo "[SKIP] Previously marked no frames: $video"
    return
  fi

  mkdir -p "$out_dir"
  echo "[RUN] $(date '+%Y-%m-%d %H:%M:%S') video=$video gpu=${gpu_id:-cpu}"
  if [ -n "$gpu_id" ]; then
    CUDA_VISIBLE_DEVICES="$gpu_id" DEVICE=cuda "$SCRIPT_DIR/run_label_segmentation.sh" "$video" "$out_dir"
  else
    DEVICE=cpu "$SCRIPT_DIR/run_label_segmentation.sh" "$video" "$out_dir"
  fi
}

worker() {
  local gpu_id="$1"
  while true; do
    local task
    if ! task=$(pop_task); then
      break
    fi
    process_video "$task" "$gpu_id"
  done
}

ensure_queue

if [ "$CONCURRENCY" -le 0 ]; then
  echo "[ERROR] CONCURRENCY must be >=1" >&2
  exit 1
fi

pids=()
for ((i=0; i<CONCURRENCY; i++)); do
  gpu_id=""
  if [ $GPU_COUNT -gt 0 ]; then
    gpu_id="${GPU_ARRAY[$((i % GPU_COUNT))]}"
  fi
  worker "$gpu_id" &
  pids+=($!)
  sleep 0.2
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

remaining=$(wc -l < "$QUEUE_FILE" 2>/dev/null || echo 0)
echo "[INFO] Queue processing finished. Remaining tasks: $remaining"
exit $status
