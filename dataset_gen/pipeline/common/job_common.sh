#!/usr/bin/env bash

dataset_gen_prepare_project_dir() {
  mkdir -p "$PROJECT_DIR/logs"
  cd "$PROJECT_DIR"
}

dataset_gen_setup_logging() {
  local log_base="$1"
  local log_dir="$PROJECT_DIR/logs"
  mkdir -p "$log_dir"
  exec > >(tee -a "$log_dir/$log_base.out") 2> >(tee -a "$log_dir/$log_base.err" >&2)
}

dataset_gen_load_modules() {
  if command -v module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh || true
    module load python/3.12/3.12.9 || true
    module load cuda/12.6 || true
  else
    echo "[WARN] module command not found; skipping module loads"
  fi
}

dataset_gen_activate_venv() {
  local venv_path="$1"
  local label="${2:-venv}"
  if [ ! -f "$venv_path" ]; then
    echo "[ERROR] ${label} not found: $venv_path"
    exit 1
  fi
  echo "[INFO] Activating ${label}: $venv_path"
  # shellcheck disable=SC1090
  source "$venv_path"
}

dataset_gen_print_runtime_info() {
  echo "[INFO] Python version: $(python -V 2>&1 || echo 'python not found')"
  echo "[INFO] Which python: $(which python || echo 'not found')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    echo "[INFO] GPU info:"
    nvidia-smi
  fi
}

dataset_gen_resolve_array_shard() {
  local num_shards="$1"
  if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
    SHARD_ID=$((PBS_ARRAY_INDEX - 1))
  else
    SHARD_ID=${SHARD_ID:-0}
  fi

  if [ "$SHARD_ID" -lt 0 ] || [ "$SHARD_ID" -ge "$num_shards" ]; then
    echo "[ERROR] SHARD_ID=$SHARD_ID out of range for NUM_SHARDS=$num_shards"
    exit 1
  fi

  echo "[INFO] JobID: ${PBS_JOBID:-none}, ArrayIndex(raw): ${PBS_ARRAY_INDEX:-none}"
  echo "[INFO] Using shard (0-based) ${SHARD_ID}/${num_shards}"
}

dataset_gen_resolve_optional_shard() {
  if [ -n "${SHARD_ID:-}" ]; then
    SHARD_ID=$SHARD_ID
  elif [ -n "${PBS_ARRAY_INDEX:-}" ]; then
    SHARD_ID=$PBS_ARRAY_INDEX
  else
    SHARD_ID=0
  fi

  if [ "$SHARD_ID" -gt 0 ]; then
    SHARD_ID=$((SHARD_ID - 1))
  fi
}
