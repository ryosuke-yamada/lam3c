#!/bin/bash
#PBS -N roomtour-label-house
#PBS -q rt_HF
#PBS -P gag51402
#PBS -l select=1
#PBS -l walltime=24:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
VIDEO_ROOT=${VIDEO_ROOT:-/groups/gag51402/datasets/HouseTours/data/files/official-housetour-dataset/videos}
OUTPUT_ROOT=${OUTPUT_ROOT:-/groups/gag51402/datasets/RoomTours/processed_label_segments_HouseTours}
GPU_IDS=${GPU_IDS:-0,1,2,3,4,5,6,7}
CONCURRENCY=${CONCURRENCY:-8}

source "$PROJECT_DIR/scripts/common/segmentation_submit_common.sh"

LOG_BASE="roomtour_label_housetours.${PBS_JOBID:-noid}"
RUNNER_MODE="queue"
dataset_gen_run_segmentation_submit


# Example submission:
# export GPU_IDS="0,1,2,3,4,5,6,7"
# qsub -P gag51402 -J 1-8 -v CONCURRENCY=8 scripts/segmentation/submit_roomtour_segmentation_housetours.sh
