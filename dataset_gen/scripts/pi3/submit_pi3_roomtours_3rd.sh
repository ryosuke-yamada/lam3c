#!/bin/bash
#PBS -N pi3-roomtours-3rd
#PBS -q rt_HF
#PBS -P gag51492
#PBS -l select=1
#PBS -l walltime=24:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG=roomtours
INPUT_BASE=${INPUT_BASE:-/groups/gag51402/datasets/RoomTours/processed_label_segments_3rd_download}
OUTPUT_BASE=${OUTPUT_BASE:-/groups/gag51402/datasets/roomtours_pi3_3rd}
INTERVAL=${INTERVAL:-1}
NUM_GPUS=${NUM_GPUS:-8}
PIXEL_LIMIT=${PIXEL_LIMIT:-255000}
MAX_ENTRIES=${MAX_ENTRIES:-0}
ROOMTOURS_TARGET_FRAMES=${ROOMTOURS_TARGET_FRAMES:-400}

NUM_SHARDS=${NUM_SHARDS:-8}
if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
  SHARD_ID=$((PBS_ARRAY_INDEX - 1))
else
  SHARD_ID=${SHARD_ID:-0}
fi

if [ "$SHARD_ID" -lt 0 ] || [ "$SHARD_ID" -ge "$NUM_SHARDS" ]; then
  echo "[ERROR] SHARD_ID=$SHARD_ID out of range for NUM_SHARDS=$NUM_SHARDS"
  exit 1
fi

echo "[INFO] JobID: ${PBS_JOBID:-none}, ArrayIndex(raw): ${PBS_ARRAY_INDEX:-none}"
echo "[INFO] Using shard (0-based) ${SHARD_ID}/${NUM_SHARDS}"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
LOG_BASE="pi3_roomtours_3rd.${PBS_JOBID:-noid}.${PBS_ARRAY_INDEX:-noidx}"
LOG_OUT="$LOG_DIR/$LOG_BASE.out"
LOG_ERR="$LOG_DIR/$LOG_BASE.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

if command -v module >/dev/null 2>&1; then
  echo "[INFO] Loading modules..."
  source /etc/profile.d/modules.sh || true
  module load python/3.12/3.12.9 || true
  module load cuda/12.6 || true
else
  echo "[WARN] module command not found; skipping module loads"
fi

echo "[INFO] Python version: $(python -V 2>&1 || echo 'python not found')"
echo "[INFO] Which python: $(which python || echo 'not found')"
echo "[INFO] CUDA: $(nvidia-smi | head -n 1 || echo 'nvidia-smi not found')"

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv_pi3/bin/activate}"
if [ -f "$VENV_PATH" ]; then
  echo "[INFO] Activating venv at $VENV_PATH"
  source "$VENV_PATH"
else
  echo "[ERROR] venv not found: $VENV_PATH"
  exit 1
fi

echo "[INFO] Starting Pi3 batch processing for RoomTours 3rd download"
ROOMTOURS_TARGET_FRAMES="$ROOMTOURS_TARGET_FRAMES" python "$SCRIPT_DIR/pi3_batch_datasets.py" \
  --config "$CONFIG" \
  --input_base "$INPUT_BASE" \
  --output_base "$OUTPUT_BASE" \
  --interval "$INTERVAL" \
  --num_gpus "$NUM_GPUS" \
  --num_shards "$NUM_SHARDS" \
  --shard_id "$SHARD_ID" \
  --pixel_limit "$PIXEL_LIMIT" \
  --max_entries "$MAX_ENTRIES"

echo "[INFO] Done."

# Example:
# qsub -J 1-8 -P gag51402 scripts/pi3/submit_pi3_roomtours_3rd.sh
