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


source "$PROJECT_DIR/scripts/common/segmentation_submit_common.sh"

LOG_BASE="roomtour_label.${PBS_JOBID:-noid}"
RUNNER_MODE="batch"
dataset_gen_run_segmentation_submit


# Example submission:
# qsub -P gag51402 scripts/segmentation/submit_roomtour_segmentation.sh
# qsub -P gag51402 -v VIDEO_ROOT=/path/to/videos,OUTPUT_ROOT=/path/to/out scripts/segmentation/submit_roomtour_segmentation.sh


# qsub -P gag51402 -J 1-8 \
#   -v NUM_SHARDS=8,GPU_IDS=0,1,2,3,4,5,6,7,CONCURRENCY=8 \
#   scripts/segmentation/submit_roomtour_segmentation.sh
