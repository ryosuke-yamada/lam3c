#!/usr/bin/env bash
#
# Room tour batch processor with optional parallelism & GPU assignment.
# Walks through each immediate subdirectory under VIDEO_ROOT (plus root itself)
# and runs run_label_segmentation.sh in parallel, mirroring outputs under OUTPUT_ROOT.

set -euo pipefail

VIDEO_ROOT=${1:-}
OUTPUT_ROOT=${2:-}
CONCURRENCY_INPUT=${3:-}

if [ -z "$VIDEO_ROOT" ] || [ -z "$OUTPUT_ROOT" ]; then
  echo "Usage: $0 <video_root> <output_root> [concurrency]" >&2
  exit 1
fi

# Detect available GPU IDs (via GPU_IDS env or nvidia-smi).
GPU_IDS_STR=${GPU_IDS:-}
if [ -z "$GPU_IDS_STR" ] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_IDS_STR=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr $'\n' ',' | sed 's/,$//') || true
fi

# Build clean array of numeric GPU IDs.
GPU_IDS=()
if [ -n "$GPU_IDS_STR" ]; then
  IFS=',' read -r -a RAW_GPU_IDS <<< "${GPU_IDS_STR// /,}"
  for id in "${RAW_GPU_IDS[@]}"; do
    if [[ $id =~ ^[0-9]+$ ]]; then
      GPU_IDS+=("$id")
    fi
  done
fi
GPU_COUNT=${#GPU_IDS[@]}

# Decide concurrency: prefer explicit arg > env > GPU count > CPU heuristic > 1.
if [ -n "$CONCURRENCY_INPUT" ]; then
  CONCURRENCY=$CONCURRENCY_INPUT
elif [ -n "${CONCURRENCY:-}" ]; then
  CONCURRENCY=$CONCURRENCY
elif [ $GPU_COUNT -gt 0 ]; then
  CONCURRENCY=$GPU_COUNT
else
  if command -v nproc >/dev/null 2>&1; then
    CPU_CORES=$(nproc)
    CONCURRENCY=$(( CPU_CORES / 8 ))
    [ $CONCURRENCY -lt 1 ] && CONCURRENCY=1
  else
    CONCURRENCY=1
  fi
fi

if ! [[ "$CONCURRENCY" =~ ^[0-9]+$ ]] || [ "$CONCURRENCY" -lt 1 ]; then
  echo "[ERROR] Invalid CONCURRENCY: $CONCURRENCY" >&2
  exit 1
fi

if [ ! -d "$VIDEO_ROOT" ]; then
  echo "[ERROR] VIDEO_ROOT not found: $VIDEO_ROOT" >&2
  exit 1
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
RUN_SCRIPT="$SCRIPT_DIR/run_label_segmentation.sh"
if [ ! -x "$RUN_SCRIPT" ]; then
  echo "[ERROR] Missing executable run_label_segmentation.sh at $RUN_SCRIPT" >&2
  exit 1
fi

echo "VIDEO_ROOT:  $VIDEO_ROOT"
echo "OUTPUT_ROOT: $OUTPUT_ROOT"
echo "CONCURRENCY: $CONCURRENCY"
if [ $GPU_COUNT -gt 0 ]; then
  echo "GPU_IDS:    ${GPU_IDS[*]}"
else
  echo "GPU_IDS:    (none detected)"
fi

mkdir -p "$OUTPUT_ROOT"
SKIP_LOG_FILE="$OUTPUT_ROOT/_skipped_videos.tsv"

PATTERN=(-iname '*.mp4' -o -iname '*.mov' -o -iname '*.m4v')

process_dir() {
  local dir="$1"
  local gpu_id="$2"
  local rel=""
  if [ "$dir" = "$VIDEO_ROOT" ]; then
    rel=""
  else
    rel=${dir#"$VIDEO_ROOT"/}
  fi
  echo "
>>> Directory: ${rel:-.} (GPU: ${gpu_id:-none})"

  find "$dir" -type f \( "${PATTERN[@]}" \) -print0 |
    while IFS= read -r -d '' video; do
      local rel_path=${video#"$VIDEO_ROOT"/}
      local rel_dir=$(dirname -- "$rel_path")
      local stem=$(basename -- "$video")
      stem=${stem%.*}
      [ "$rel_dir" = "." ] && rel_dir=""
      local out_dir="$OUTPUT_ROOT"
      if [ -n "$rel_dir" ]; then
        out_dir+="/$rel_dir"
      fi
      out_dir+="/$stem"

      if [ -d "$out_dir/scenes" ] && find "$out_dir/scenes" -maxdepth 1 -type f -name '*.mp4' -print -quit >/dev/null; then
        echo "[SKIP] Already processed: $video"
        continue
      fi
      if [ -f "$out_dir/SKIP_NO_FRAMES" ]; then
        echo "[SKIP] Previously marked as no usable frames: $video"
        continue
      fi

      mkdir -p "$out_dir"
      echo "=== Processing: $video"
      echo "Output dir: $out_dir"
      if [ -n "$gpu_id" ]; then
        CUDA_VISIBLE_DEVICES="$gpu_id" DEVICE=cuda SKIP_LOG="$SKIP_LOG_FILE" "$RUN_SCRIPT" "$video" "$out_dir"
      else
        DEVICE=cpu SKIP_LOG="$SKIP_LOG_FILE" "$RUN_SCRIPT" "$video" "$out_dir"
      fi
    done
}

mapfile -t SUBDIRS < <(find "$VIDEO_ROOT" -mindepth 1 -maxdepth 1 -type d | sort)
TARGET_DIRS=()
if find "$VIDEO_ROOT" -maxdepth 1 -type f \( "${PATTERN[@]}" \) -print -quit >/dev/null; then
  TARGET_DIRS+=("$VIDEO_ROOT")
fi
TARGET_DIRS+=("${SUBDIRS[@]}")

NUM_SHARDS=${NUM_SHARDS:-0}
if [ ${NUM_SHARDS:-0} -gt 0 ]; then
  if [ -z "${SHARD_ID:-}" ]; then
    echo "[ERROR] SHARD_ID is required when NUM_SHARDS > 0" >&2
    exit 1
  fi
  if [ "$SHARD_ID" -lt 0 ] || [ "$SHARD_ID" -ge "$NUM_SHARDS" ]; then
    echo "[ERROR] SHARD_ID=$SHARD_ID out of range for NUM_SHARDS=$NUM_SHARDS" >&2
    exit 1
  fi
  echo "[INFO] Sharding enabled: shard $SHARD_ID / $NUM_SHARDS"
  FILTERED_TARGETS=()
  for idx in "${!TARGET_DIRS[@]}"; do
    if [ $((idx % NUM_SHARDS)) -eq "$SHARD_ID" ]; then
      FILTERED_TARGETS+=("${TARGET_DIRS[$idx]}")
    fi
  done
  TARGET_DIRS=("${FILTERED_TARGETS[@]}")
  if [ ${#TARGET_DIRS[@]} -eq 0 ]; then
    echo "[INFO] No directories assigned to this shard"
    exit 0
  fi
fi

if [ ${#TARGET_DIRS[@]} -eq 0 ]; then
  echo "[INFO] No videos found under $VIDEO_ROOT"
  exit 0
fi

assign_gpu() {
  local idx=$1
  if [ $GPU_COUNT -eq 0 ]; then
    echo ""
  else
    local gpu=${GPU_IDS[$(( idx % GPU_COUNT ))]}
    echo "$gpu"
  fi
}

pids=()
idx=0

start_job() {
  local dir="$1"
  local gpu_id="$2"
  ( set -euo pipefail; process_dir "$dir" "$gpu_id" ) &
  local pid=$!
  pids+=($pid)
}

wait_for_pid() {
  local pid="$1"
  if ! wait "$pid"; then
    echo "[ERROR] A parallel task failed (PID $pid)." >&2
    exit 1
  fi
}

for dir in "${TARGET_DIRS[@]}"; do
  gpu_assigned=$(assign_gpu $idx)
  start_job "$dir" "$gpu_assigned"
  idx=$((idx + 1))
  while [ ${#pids[@]} -ge "$CONCURRENCY" ]; do
    first=${pids[0]}
    wait_for_pid "$first"
    if [ ${#pids[@]} -gt 1 ]; then
      pids=("${pids[@]:1}")
    else
      pids=()
    fi
  done
  # brief pause to stagger startups when heavily parallel (optional)
  sleep 0.5
done

for pid in "${pids[@]}"; do
  wait_for_pid "$pid"

done

echo "
Batch processing completed."
