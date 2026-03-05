#!/bin/bash
#PBS -N roomtour-label
#PBS -q rt_HF
#PBS -P gag51402
#PBS -l select=1
#PBS -l walltime=24:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
VIDEO_ROOT=${VIDEO_ROOT:-/groups/gag51402/datasets/RoomTours/raw_videos/1st_download}
OUTPUT_ROOT=${OUTPUT_ROOT:-/groups/gag51402/datasets/RoomTours/processed_label_segments_v2}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
CONCURRENCY=${CONCURRENCY:-8}
NUM_SHARDS=${NUM_SHARDS:-0}

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_BASE="roomtour_label.${PBS_JOBID:-noid}"
exec > >(tee -a "$LOG_DIR/$LOG_BASE.out") 2> >(tee -a "$LOG_DIR/$LOG_BASE.err" >&2)

echo "[INFO] JobID: ${PBS_JOBID:-none}"
echo "[INFO] VIDEO_ROOT:  $VIDEO_ROOT"
echo "[INFO] OUTPUT_ROOT: $OUTPUT_ROOT"
echo "[INFO] GPU_IDS:    $GPU_IDS"
echo "[INFO] CONCURRENCY: $CONCURRENCY"
echo "[INFO] NUM_SHARDS: $NUM_SHARDS"

if command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh || true
  module load python/3.12/3.12.9 || true
  module load cuda/12.6 || true
else
  echo "[WARN] module command not found; skipping module loads"
fi

cd "$PROJECT_DIR"

VENV_ACTIVATE="${VENV_ACTIVATE:-$PROJECT_DIR/.venv_pi3/bin/activate}"
if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "[ERROR] Virtualenv activate script not found at $VENV_ACTIVATE"
  exit 1
fi

source "$VENV_ACTIVATE"

if [ "${SHARD_ID:-}" != "" ]; then
  SHARD_ID=$SHARD_ID
elif [ "${PBS_ARRAY_INDEX:-}" != "" ]; then
  SHARD_ID=$PBS_ARRAY_INDEX
else
  SHARD_ID=0
fi
if [ "$SHARD_ID" -gt 0 ]; then
  SHARD_ID=$((SHARD_ID - 1))
fi

echo "[INFO] Python: $(python -V 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU info:" && nvidia-smi
fi

GPU_IDS="$GPU_IDS" NUM_SHARDS="$NUM_SHARDS" SHARD_ID="$SHARD_ID" "$SCRIPT_DIR/batch_run_label_segmentation.sh" "$VIDEO_ROOT" "$OUTPUT_ROOT" "$CONCURRENCY"

echo "[INFO] Job completed"

# Example submission:
# qsub -P gag51402 scripts/segmentation/submit_roomtour_segmentation.sh
# qsub -P gag51402 -v VIDEO_ROOT=/path/to/videos,OUTPUT_ROOT=/path/to/out scripts/segmentation/submit_roomtour_segmentation.sh


# qsub -P gag51402 -J 1-8 \
#   -v NUM_SHARDS=8,GPU_IDS=0,1,2,3,4,5,6,7,CONCURRENCY=8 \
#   scripts/segmentation/submit_roomtour_segmentation.sh
