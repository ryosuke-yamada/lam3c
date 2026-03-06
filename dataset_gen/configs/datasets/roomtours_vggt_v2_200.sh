#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/../base/seg_rt_hf_batch.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/vggt_rt_hf_1gpu.sh"
source "$(dirname "${BASH_SOURCE[0]}")/../base/roomtours_release_common.sh"

DATASET_ID=roomtours_vggt_v2_200
DATASET_LABEL='RoomTours VGGT v2 200'
ROOMTOURS_RAW_VIDEO_ROOT=/groups/gag51402/datasets/RoomTours/raw_videos/1st_download
ROOMTOURS_SEGMENT_ROOT=/groups/gag51402/datasets/RoomTours/processed_label_segments_v2
ROOMTOURS_PI3_OUTPUT_ROOT=/groups/gag51402/datasets/roomtours_pi3_v2
ROOMTOURS_VGGT_OUTPUT_ROOT=/groups/gag51402/datasets/roomtours_vggt_v2_200
ROOMTOURS_VGGT_SUFFIX_DASH=v2

dataset_gen_apply_roomtours_seg_defaults
dataset_gen_apply_roomtours_vggt_defaults
