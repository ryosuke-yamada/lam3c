#!/bin/bash
#PBS -N roomtour-label-v4-gh
#PBS -q rt_HG
#PBS -P gag51402
#PBS -l select=1:ngpus=1
#PBS -l walltime=72:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
VIDEO_ROOT=${VIDEO_ROOT:-/groups/gag51402/datasets/RoomTours/raw_videos/v4_relaxed_download}
OUTPUT_ROOT=${OUTPUT_ROOT:-/groups/gag51402/datasets/RoomTours/processed_label_segments_v4_relaxed}
GPU_IDS=${GPU_IDS:-0}
CONCURRENCY=${CONCURRENCY:-1}

LOG_DIR="$PROJECT_DIR/logs"
mkdir -p "$LOG_DIR"
LOG_BASE="roomtour_label_v4_gh.${PBS_JOBID:-noid}"
exec > >(tee -a "$LOG_DIR/$LOG_BASE.out") 2> >(tee -a "$LOG_DIR/$LOG_BASE.err" >&2)

echo "[INFO] JobID: ${PBS_JOBID:-none}"
echo "[INFO] VIDEO_ROOT:  $VIDEO_ROOT"
echo "[INFO] OUTPUT_ROOT: $OUTPUT_ROOT"
echo "[INFO] GPU_IDS:    $GPU_IDS"
echo "[INFO] CONCURRENCY: $CONCURRENCY"

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

echo "[INFO] Python: $(python -V 2>&1)"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "[INFO] GPU info:" && nvidia-smi
fi

GPU_IDS="$GPU_IDS" scripts/batch_run_label_segmentation_queue.sh "$VIDEO_ROOT" "$OUTPUT_ROOT" "$CONCURRENCY"

echo "[INFO] Job completed"

# Example submission:
# qsub -P gag51402 -v VIDEO_ROOT=/path/to/videos,OUTPUT_ROOT=/path/to/out scripts/segmentation/submit_roomtour_segmentation_v4_relaxed_rt_HG.sh
