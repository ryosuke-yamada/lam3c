#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../base/seg_rt_hf_queue.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/pi3_rt_hf_8gpu.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/roomtours_release_common.sh"

DATASET_ID=roomtours_2nd
DATASET_LABEL='RoomTours 2nd'
ROOMTOURS_RAW_VIDEO_ROOT=/groups/gag51402/datasets/RoomTours/raw_videos/2nd_download
ROOMTOURS_SEGMENT_ROOT=/groups/gag51402/datasets/RoomTours/processed_label_segments_2nd_download
ROOMTOURS_PI3_OUTPUT_ROOT=/groups/gag51402/datasets/roomtours_pi3_2nd
ROOMTOURS_SEG_SUFFIX_DASH=2nd
ROOMTOURS_PI3_SUFFIX_DASH=2nd
ROOMTOURS_TARGET_FRAMES=400
PI3_PBS_PROJECT=gag51402

dataset_gen_apply_roomtours_seg_defaults
dataset_gen_apply_roomtours_pi3_defaults
