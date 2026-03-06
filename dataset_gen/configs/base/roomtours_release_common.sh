#!/usr/bin/env bash


dataset_gen_roomtours_append_suffix() {
  local base="$1"
  local suffix="${2:-}"
  local separator="$3"
  if [ -n "$suffix" ]; then
    printf '%s%s%s\n' "$base" "$separator" "$suffix"
  else
    printf '%s\n' "$base"
  fi
}


dataset_gen_roomtours_normalize_log_suffix() {
  local suffix="${1:-}"
  printf '%s\n' "${suffix//-/_}"
}


dataset_gen_apply_roomtours_seg_defaults() {
  : "${ROOMTOURS_RAW_VIDEO_ROOT:?ROOMTOURS_RAW_VIDEO_ROOT is required}"
  : "${ROOMTOURS_SEGMENT_ROOT:?ROOMTOURS_SEGMENT_ROOT is required}"

  local seg_suffix_dash="${ROOMTOURS_SEG_SUFFIX_DASH:-}"
  local seg_suffix_underscore="${ROOMTOURS_SEG_SUFFIX_UNDERSCORE:-$(dataset_gen_roomtours_normalize_log_suffix "$seg_suffix_dash")}"

  SEG_JOB_NAME=${SEG_JOB_NAME:-$(dataset_gen_roomtours_append_suffix "roomtour-label" "$seg_suffix_dash" "-")}
  SEG_LOG_PREFIX=${SEG_LOG_PREFIX:-$(dataset_gen_roomtours_append_suffix "roomtour_label" "$seg_suffix_underscore" "_")}
  SEG_VIDEO_ROOT=${SEG_VIDEO_ROOT:-$ROOMTOURS_RAW_VIDEO_ROOT}
  SEG_OUTPUT_ROOT=${SEG_OUTPUT_ROOT:-$ROOMTOURS_SEGMENT_ROOT}
}


dataset_gen_apply_roomtours_pi3_defaults() {
  : "${ROOMTOURS_SEGMENT_ROOT:?ROOMTOURS_SEGMENT_ROOT is required}"
  : "${ROOMTOURS_PI3_OUTPUT_ROOT:?ROOMTOURS_PI3_OUTPUT_ROOT is required}"

  local pi3_suffix_dash="${ROOMTOURS_PI3_SUFFIX_DASH:-}"
  local pi3_suffix_underscore="${ROOMTOURS_PI3_SUFFIX_UNDERSCORE:-$(dataset_gen_roomtours_normalize_log_suffix "$pi3_suffix_dash")}"

  PI3_JOB_NAME=${PI3_JOB_NAME:-$(dataset_gen_roomtours_append_suffix "pi3-roomtours" "$pi3_suffix_dash" "-")}
  PI3_LOG_PREFIX=${PI3_LOG_PREFIX:-$(dataset_gen_roomtours_append_suffix "pi3_roomtours" "$pi3_suffix_underscore" "_")}
  PI3_INPUT_BASE=${PI3_INPUT_BASE:-$ROOMTOURS_SEGMENT_ROOT}
  PI3_OUTPUT_BASE=${PI3_OUTPUT_BASE:-$ROOMTOURS_PI3_OUTPUT_ROOT}
  PI3_LAYOUT=${PI3_LAYOUT:-roomtours_scenes}
  PI3_PRESERVE_ORDER=${PI3_PRESERVE_ORDER:-1}
  PI3_VIDEO_SKIP_SECONDS=${PI3_VIDEO_SKIP_SECONDS:-10}
  PI3_ADJUST_VIDEO_INTERVAL=${PI3_ADJUST_VIDEO_INTERVAL:-1}
  PI3_TARGET_FRAMES=${PI3_TARGET_FRAMES:-${ROOMTOURS_TARGET_FRAMES:-400}}
  PI3_IMAGE_TARGET_FRAMES=${PI3_IMAGE_TARGET_FRAMES:-1500}
}


dataset_gen_apply_roomtours_vggt_defaults() {
  : "${ROOMTOURS_SEGMENT_ROOT:?ROOMTOURS_SEGMENT_ROOT is required}"
  : "${ROOMTOURS_PI3_OUTPUT_ROOT:?ROOMTOURS_PI3_OUTPUT_ROOT is required}"
  : "${ROOMTOURS_VGGT_OUTPUT_ROOT:?ROOMTOURS_VGGT_OUTPUT_ROOT is required}"

  local vggt_suffix_dash="${ROOMTOURS_VGGT_SUFFIX_DASH:-}"
  local vggt_suffix_underscore="${ROOMTOURS_VGGT_SUFFIX_UNDERSCORE:-$(dataset_gen_roomtours_normalize_log_suffix "$vggt_suffix_dash")}"

  VGGT_JOB_NAME=${VGGT_JOB_NAME:-$(dataset_gen_roomtours_append_suffix "vggt-roomtours" "$vggt_suffix_dash" "-")}
  VGGT_LOG_PREFIX=${VGGT_LOG_PREFIX:-$(dataset_gen_roomtours_append_suffix "vggt_roomtours" "$vggt_suffix_underscore" "_")}
  VGGT_VIDEO_BASE=${VGGT_VIDEO_BASE:-$ROOMTOURS_SEGMENT_ROOT}
  VGGT_PI3_BASE=${VGGT_PI3_BASE:-$ROOMTOURS_PI3_OUTPUT_ROOT}
  VGGT_OUTPUT_BASE=${VGGT_OUTPUT_BASE:-$ROOMTOURS_VGGT_OUTPUT_ROOT}
}
