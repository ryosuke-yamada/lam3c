#!/bin/bash
#PBS -N pi3-roomtours-batch-v8
#PBS -q rt_HG
#PBS -P gag51402
#PBS -l select=1:ngpus=1
#PBS -l walltime=96:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG=roomtours
INPUT_BASE=${INPUT_BASE:-/groups/gag51402/datasets/RoomTours/processed_label_segments_batch_v8}
OUTPUT_BASE=${OUTPUT_BASE:-/groups/gag51402/datasets/roomtours_pi3_batch_v8}
INTERVAL=${INTERVAL:-1}
NUM_GPUS=${NUM_GPUS:-1}
PIXEL_LIMIT=${PIXEL_LIMIT:-255000}
MAX_ENTRIES=${MAX_ENTRIES:-0}
ROOMTOURS_TARGET_FRAMES=${ROOMTOURS_TARGET_FRAMES:-400}
ROOMTOURS_SCENE_JSON=${ROOMTOURS_SCENE_JSON:-}
OVERWRITE_EXISTING=${OVERWRITE_EXISTING:-0}
INCLUDE_PROCESSED=${INCLUDE_PROCESSED:-0}

NUM_SHARDS=${NUM_SHARDS:-1}

source "$PROJECT_DIR/scripts/common/pi3_submit_common.sh"

LOG_BASE="pi3_roomtours_batch_v8.${PBS_JOBID:-noid}.${PBS_ARRAY_INDEX:-noidx}"
START_MESSAGE="RoomTours batch v8"
dataset_gen_run_pi3_submit
