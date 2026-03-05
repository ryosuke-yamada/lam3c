#!/usr/bin/env bash
DATASET_ID=roomtours_batch_v7
DATASET_LABEL='RoomTours batch v7'

SEG_JOB_NAME=roomtour-label-batch-v7
SEG_LOG_PREFIX=roomtour_label_batch_v7
SEG_PBS_QUEUE=rt_HG
SEG_PBS_PROJECT=gag51402
SEG_PBS_SELECT=1:ngpus=1
SEG_PBS_WALLTIME=72:00:00
SEG_VIDEO_ROOT=/groups/gag51402/datasets/RoomTours/raw_videos/batch_v7_download
SEG_OUTPUT_ROOT=/groups/gag51402/datasets/RoomTours/processed_label_segments_batch_v7
SEG_GPU_IDS=0
SEG_CONCURRENCY=1
SEG_NUM_SHARDS=0
SEG_RUNNER_MODE=queue

PI3_JOB_NAME=pi3-roomtours-batch-v7
PI3_LOG_PREFIX=pi3_roomtours_batch_v7
PI3_PBS_QUEUE=rt_HG
PI3_PBS_PROJECT=gag51402
PI3_PBS_SELECT=1:ngpus=1
PI3_PBS_WALLTIME=96:00:00
PI3_CONFIG=roomtours
PI3_INPUT_BASE=/groups/gag51402/datasets/RoomTours/processed_label_segments_batch_v7
PI3_OUTPUT_BASE=/groups/gag51402/datasets/roomtours_pi3_batch_v7
PI3_INTERVAL=1
PI3_NUM_GPUS=1
PI3_PIXEL_LIMIT=255000
PI3_MAX_ENTRIES=0
PI3_TARGET_FRAMES=400
PI3_NUM_SHARDS=1
PI3_ROOMTOURS_SCENE_JSON=''
PI3_OVERWRITE_EXISTING=0
PI3_INCLUDE_PROCESSED=0
