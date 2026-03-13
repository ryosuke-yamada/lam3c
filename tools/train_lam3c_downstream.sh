#!/usr/bin/env bash
set -eu

CONFIG="${1:-configs/lam3c/downstream/semseg/semseg-lam3c-scannet-lin.py}"
SAVE_PATH="${2:-./logs/lam3c_scannet_lin}"
NUM_GPUS="${3:-8}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POINTCEPT_ROOT="${REPO_ROOT}/third_party/pointcept"

export PYTHONPATH="${POINTCEPT_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

mkdir -p "${SAVE_PATH}"

OPTIONS=( "save_path=${SAVE_PATH}" )
if [[ -n "${LAM3C_PRETRAIN_WEIGHT:-}" ]]; then
  OPTIONS+=( "weight=${LAM3C_PRETRAIN_WEIGHT}" )
fi
if [[ -n "${LAM3C_SCANNET_ROOT:-}" ]]; then
  OPTIONS+=( "data.train.data_root=${LAM3C_SCANNET_ROOT}" )
  OPTIONS+=( "data.val.data_root=${LAM3C_SCANNET_ROOT}" )
  OPTIONS+=( "data.test.data_root=${LAM3C_SCANNET_ROOT}" )
fi

python "${POINTCEPT_ROOT}/tools/train.py" \
  --config-file "${REPO_ROOT}/${CONFIG}" \
  --num-gpus "${NUM_GPUS}" \
  --options "${OPTIONS[@]}"
