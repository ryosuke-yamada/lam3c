#!/bin/bash
#PBS -N pi3-roomtours-v2
#PBS -q rt_HF
#PBS -P gag51492
#PBS -l select=1
#PBS -l walltime=24:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG=roomtours
INPUT_BASE=${INPUT_BASE:-/groups/gag51402/datasets/RoomTours/processed_label_segments_v2}
OUTPUT_BASE=${OUTPUT_BASE:-/groups/gag51402/datasets/roomtours_pi3_v2}
INTERVAL=${INTERVAL:-1}
NUM_GPUS=${NUM_GPUS:-8}
ROOMTOURS_TARGET_FRAMES=${ROOMTOURS_TARGET_FRAMES:-400}

PIXEL_LIMIT=${PIXEL_LIMIT:-255000}
MAX_ENTRIES=${MAX_ENTRIES:-0}

NUM_SHARDS=${NUM_SHARDS:-8}

source "$PROJECT_DIR/scripts/common/pi3_submit_common.sh"

LOG_BASE="pi3_roomtours_v2.${PBS_JOBID:-noid}.${PBS_ARRAY_INDEX:-noidx}"
START_MESSAGE="RoomTours v2"
dataset_gen_run_pi3_submit
