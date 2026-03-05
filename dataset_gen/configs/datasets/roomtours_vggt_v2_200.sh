#!/usr/bin/env bash
DATASET_ID=roomtours_vggt_v2_200
DATASET_LABEL='RoomTours VGGT v2 200'

SEG_JOB_NAME=roomtour-label
SEG_LOG_PREFIX=roomtour_label
SEG_PBS_QUEUE=rt_HF
SEG_PBS_PROJECT=gag51402
SEG_PBS_SELECT=1
SEG_PBS_WALLTIME=24:00:00
SEG_VIDEO_ROOT=/groups/gag51402/datasets/RoomTours/raw_videos/1st_download
SEG_OUTPUT_ROOT=/groups/gag51402/datasets/RoomTours/processed_label_segments_v2
SEG_GPU_IDS=0,1,2,3,4,5,6,7
SEG_CONCURRENCY=8
SEG_NUM_SHARDS=0
SEG_RUNNER_MODE=batch

VGGT_JOB_NAME=vggt-roomtours-v2
VGGT_LOG_PREFIX=vggt_roomtours_v2
VGGT_PBS_QUEUE=rt_HF
VGGT_PBS_PROJECT=gag51492
VGGT_PBS_SELECT=1:ngpus=1
VGGT_PBS_WALLTIME=24:00:00
VGGT_VIDEO_BASE=/groups/gag51402/datasets/RoomTours/processed_label_segments_v2
VGGT_PI3_BASE=/groups/gag51402/datasets/roomtours_pi3_v2
VGGT_OUTPUT_BASE=/groups/gag51402/datasets/roomtours_vggt_v2_200
VGGT_WORK_BASE_REL=tmp/vggt_roomtours_v2
VGGT_TARGET_IMAGES=200
VGGT_PIXEL_LIMIT=255000
VGGT_INTERVAL=1
VGGT_CONF_THRES_VALUE=0.1
VGGT_NUM_SHARDS=8
