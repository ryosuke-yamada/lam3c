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


source "$PROJECT_DIR/scripts/common/segmentation_submit_common.sh"

LOG_BASE="roomtour_label_v4_gh.${PBS_JOBID:-noid}"
RUNNER_MODE="queue"
dataset_gen_run_segmentation_submit


# Example submission:
# qsub -P gag51402 -v VIDEO_ROOT=/path/to/videos,OUTPUT_ROOT=/path/to/out scripts/segmentation/submit_roomtour_segmentation_v4_relaxed_rt_HG.sh
