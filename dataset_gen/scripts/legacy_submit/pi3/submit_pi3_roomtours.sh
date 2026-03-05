#!/bin/bash
#PBS -N pi3-roomtours
#PBS -q rt_HF
#PBS -P gag51492
#PBS -l select=1
#PBS -l walltime=24:00:00
# NOTE: Some PBS systems do not expand env vars in -o/-e. Send logs to a directory and handle naming inside the script.

set -euo pipefail

# ------ User-configurable parameters ------
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG=roomtours
INPUT_BASE=${INPUT_BASE:-/groups/gag51402/datasets/RoomTours/processed_label_segments_v2}
OUTPUT_BASE=${OUTPUT_BASE:-/groups/gag51402/datasets/roomtours_pi3_v2_200}
INTERVAL=${INTERVAL:-1}
NUM_GPUS=${NUM_GPUS:-8}
ROOMTOURS_TARGET_FRAMES=${ROOMTOURS_TARGET_FRAMES:-200}

# Optional overrides via environment variables
PIXEL_LIMIT=${PIXEL_LIMIT:-255000}
MAX_ENTRIES=${MAX_ENTRIES:-0}

# Sharding settings (can also be overridden at submission time via environment variables)
NUM_SHARDS=${NUM_SHARDS:-8}

source "$PROJECT_DIR/scripts/common/pi3_submit_common.sh"

LOG_BASE="pi3_roomtours.${PBS_JOBID:-noid}.${PBS_ARRAY_INDEX:-noidx}"
START_MESSAGE="RoomTours v2_200"
dataset_gen_run_pi3_submit


# Submission examples (1-based array indices on ABCI):
# qsub -J 1-8 -P gag51402 scripts/pi3/submit_pi3_roomtours.sh
# qsub -J 1-5 -P gag51402 -v NUM_SHARDS=5 scripts/pi3/submit_pi3_roomtours.sh
# Customize interval/frames:
# qsub -P gag51402 -v INTERVAL=2,NUM_SHARDS=5,SHARD_ID=0 scripts/pi3/submit_pi3_roomtours.sh
# qsub -P gag51402 -v MAX_ENTRIES=2000,NUM_SHARDS=5 scripts/pi3/submit_pi3_roomtours.sh
# Legacy dataset example:
# qsub -P gag51402 -v INPUT_BASE=/path/to/processed_label_segments,OUTPUT_BASE=/path/to/roomtours_pi3 scripts/pi3/submit_pi3_roomtours.sh
