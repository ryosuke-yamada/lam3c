#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../base/seg_rt_hf_queue.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/pi3_rt_hf_8gpu.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/roomtours_release_common.sh"

DATASET_ID=roomtours_v4_relaxed
DATASET_LABEL='RoomTours v4 relaxed'
ROOMTOURS_RAW_VIDEO_ROOT=/groups/gag51402/datasets/RoomTours/raw_videos/v4_relaxed_download
ROOMTOURS_SEGMENT_ROOT=/groups/gag51402/datasets/RoomTours/processed_label_segments_v4_relaxed
ROOMTOURS_PI3_OUTPUT_ROOT=/groups/gag51402/datasets/roomtours_pi3_v4_relaxed
ROOMTOURS_SEG_SUFFIX_DASH=v4
ROOMTOURS_PI3_SUFFIX_DASH=v4relaxed
ROOMTOURS_PI3_SUFFIX_UNDERSCORE=v4_relaxed
ROOMTOURS_TARGET_FRAMES=400

dataset_gen_apply_roomtours_seg_defaults
dataset_gen_apply_roomtours_pi3_defaults
