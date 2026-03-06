#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../base/seg_rt_hf_queue.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/pi3_rt_hf_8gpu.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/roomtours_release_common.sh"

DATASET_ID=roomtours_housetours
DATASET_LABEL='RoomTours HouseTours'
ROOMTOURS_RAW_VIDEO_ROOT=/groups/gag51402/datasets/HouseTours/data/files/official-housetour-dataset/videos
ROOMTOURS_SEGMENT_ROOT=/groups/gag51402/datasets/RoomTours/processed_label_segments_HouseTours
ROOMTOURS_PI3_OUTPUT_ROOT=/groups/gag51402/datasets/roomtours_pi3_HouseTours
ROOMTOURS_SEG_SUFFIX_DASH=house
ROOMTOURS_SEG_SUFFIX_UNDERSCORE=housetours
ROOMTOURS_PI3_SUFFIX_DASH=housetours
ROOMTOURS_TARGET_FRAMES=400

dataset_gen_apply_roomtours_seg_defaults
dataset_gen_apply_roomtours_pi3_defaults
