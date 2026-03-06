#!/usr/bin/env bash

dataset_gen_resolve_config_path() {
  local project_dir="$1"
  local config_ref="$2"

  if [ -z "$config_ref" ]; then
    config_ref=default
  fi

  if [ -f "$config_ref" ]; then
    printf '%s\n' "$config_ref"
    return
  fi

  if [ -f "$project_dir/configs/datasets/$config_ref.sh" ]; then
    printf '%s\n' "$project_dir/configs/datasets/$config_ref.sh"
    return
  fi

  if [ -f "$project_dir/configs/datasets/$config_ref" ]; then
    printf '%s\n' "$project_dir/configs/datasets/$config_ref"
    return
  fi

  echo "[ERROR] Dataset config not found: $config_ref" >&2
  exit 1
}

dataset_gen_load_config() {
  local project_dir="$1"
  local config_ref="$2"
  DATASET_CONFIG_PATH=$(dataset_gen_resolve_config_path "$project_dir" "$config_ref")
  # shellcheck disable=SC1090
  source "$DATASET_CONFIG_PATH"
  DATASET_ID=${DATASET_ID:-$(basename "$DATASET_CONFIG_PATH" .sh)}
  DATASET_LABEL=${DATASET_LABEL:-$DATASET_ID}
}

dataset_gen_stage_prefix() {
  case "$1" in
    segmentation) printf 'SEG\n' ;;
    pi3) printf 'PI3\n' ;;
    *)
      echo "[ERROR] Unsupported stage: $1" >&2
      exit 1
      ;;
  esac
}

dataset_gen_get_prefixed_var() {
  local prefix="$1"
  local name="$2"
  eval "printf '%s\n' \"\${${prefix}_${name}:-}\""
}
