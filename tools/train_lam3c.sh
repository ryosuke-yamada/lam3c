#!/usr/bin/env bash
set -eu

# Usage:
#   bash tools/train_lam3c.sh [CONFIG_PATH] [SAVE_PATH] [NUM_GPUS]
#
# Example (recommended: PTv3-Base):
#   bash tools/train_lam3c.sh \
#     configs/lam3c/pretrain/lam3c_v1m1_ptv3_base.py \
#     logs/lam3c_pretrain_base
#
# Optional (PTv3-Large):
#   # bash tools/train_lam3c.sh \
#   #   configs/lam3c/pretrain/lam3c_v1m1_ptv3_large.py \
#   #   logs/lam3c_pretrain_large
#
CONFIG="${1:-configs/lam3c/pretrain/lam3c_v1m1_ptv3_base.py}"
SAVE_PATH="${2:-./logs/lam3c_pretrain_test}"
NUM_GPUS="${3:-8}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POINTCEPT_ROOT="${REPO_ROOT}/third_party/pointcept"

export PYTHONPATH="${POINTCEPT_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${SAVE_PATH}"

python "${POINTCEPT_ROOT}/tools/train.py" \
  --config-file "${REPO_ROOT}/${CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --options save_path="${REPO_ROOT}/${SAVE_PATH}"