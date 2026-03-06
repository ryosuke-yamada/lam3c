#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../base/seg_rt_hg_queue.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/pi3_rt_hg_1gpu.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/roomtours_release_common.sh"

DATASET_ID=roomtours_batch_v7
DATASET_LABEL='RoomTours batch v7'
ROOMTOURS_RAW_VIDEO_ROOT=/groups/gag51402/datasets/RoomTours/raw_videos/batch_v7_download
ROOMTOURS_SEGMENT_ROOT=/groups/gag51402/datasets/RoomTours/processed_label_segments_batch_v7
ROOMTOURS_PI3_OUTPUT_ROOT=/groups/gag51402/datasets/roomtours_pi3_batch_v7
ROOMTOURS_SEG_SUFFIX_DASH=batch-v7
ROOMTOURS_PI3_SUFFIX_DASH=batch-v7
ROOMTOURS_TARGET_FRAMES=400

dataset_gen_apply_roomtours_seg_defaults
dataset_gen_apply_roomtours_pi3_defaults
