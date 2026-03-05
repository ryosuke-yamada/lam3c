#!/usr/bin/env bash
DATASET_ID=roomtours_housetours
DATASET_LABEL='RoomTours HouseTours'

SEG_JOB_NAME=roomtour-label-house
SEG_LOG_PREFIX=roomtour_label_housetours
SEG_PBS_QUEUE=rt_HF
SEG_PBS_PROJECT=gag51402
SEG_PBS_SELECT=1
SEG_PBS_WALLTIME=24:00:00
SEG_VIDEO_ROOT=/groups/gag51402/datasets/HouseTours/data/files/official-housetour-dataset/videos
SEG_OUTPUT_ROOT=/groups/gag51402/datasets/RoomTours/processed_label_segments_HouseTours
SEG_GPU_IDS=0,1,2,3,4,5,6,7
SEG_CONCURRENCY=8
SEG_NUM_SHARDS=0
SEG_RUNNER_MODE=queue

PI3_JOB_NAME=pi3-roomtours-housetours
PI3_LOG_PREFIX=pi3_roomtours_housetours
PI3_PBS_QUEUE=rt_HF
PI3_PBS_PROJECT=gag51492
PI3_PBS_SELECT=1
PI3_PBS_WALLTIME=24:00:00
PI3_CONFIG=roomtours
PI3_INPUT_BASE=/groups/gag51402/datasets/RoomTours/processed_label_segments_HouseTours
PI3_OUTPUT_BASE=/groups/gag51402/datasets/roomtours_pi3_HouseTours
PI3_INTERVAL=1
PI3_NUM_GPUS=8
PI3_PIXEL_LIMIT=255000
PI3_MAX_ENTRIES=0
PI3_TARGET_FRAMES=400
PI3_NUM_SHARDS=8
PI3_ROOMTOURS_SCENE_JSON=''
PI3_OVERWRITE_EXISTING=0
PI3_INCLUDE_PROCESSED=0
