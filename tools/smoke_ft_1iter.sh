#!/usr/bin/env bash
set -eu

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
POINTCEPT_ROOT="${REPO_ROOT}/third_party/pointcept"
cd "${REPO_ROOT}"

export PYTHONPATH="${POINTCEPT_ROOT}:${REPO_ROOT}:${PYTHONPATH:-}"

# Override by environment variables when needed.
BASE_WEIGHT="${LAM3C_PRETRAIN_BASE_WEIGHT:-${REPO_ROOT}/logs/lam3c_pretrain_test/model/model_last.pth}"
LARGE_WEIGHT="${LAM3C_PRETRAIN_LARGE_WEIGHT:-${REPO_ROOT}/logs/lam3c_pretrain_test/model/model_last.pth}"
SCANNET_ROOT="${LAM3C_SCANNET_ROOT:-/groups/gag51404/yamada/2025/CVPR/Pointcept/data/scannet}"
SCANNETPP_ROOT="${LAM3C_SCANNETPP_ROOT:-/groups/gag51404/yamada/2025/CVPR/Pointcept/data/scannetpp}"
S3DIS_ROOT="${LAM3C_S3DIS_ROOT:-/groups/gag51404/yamada/2025/CVPR/Pointcept/data/s3dis}"

FT_CONFIGS=(
  "configs/lam3c/downstream/semseg/semseg-lam3c-large-scannet-ft.py"
  "configs/lam3c/downstream/semseg/semseg-lam3c-large-scannet200-ft.py"
  "configs/lam3c/downstream/semseg/semseg-lam3c-scannet200-ft.py"
  "configs/lam3c/downstream/semseg/semseg-lam3c-scannetpp-ft.py"
  "configs/lam3c/downstream/semseg/semseg-lam3c-large-scannetpp-ft.py"
  "configs/lam3c/downstream/semseg/semseg-lam3c-large-s3dis-ft.py"
  "configs/lam3c/downstream/semseg/semseg-lam3c-s3dis-ft.py"
)

echo "[INFO] REPO_ROOT=${REPO_ROOT}"
echo "[INFO] BASE_WEIGHT=${BASE_WEIGHT}"
echo "[INFO] LARGE_WEIGHT=${LARGE_WEIGHT}"
echo "[INFO] SCANNET_ROOT=${SCANNET_ROOT}"
echo "[INFO] SCANNETPP_ROOT=${SCANNETPP_ROOT}"
echo "[INFO] S3DIS_ROOT=${S3DIS_ROOT}"

for path in "${BASE_WEIGHT}" "${LARGE_WEIGHT}"; do
  if [[ ! -f "${path}" ]]; then
    echo "[ERROR] Weight not found: ${path}" >&2
    exit 1
  fi
done

for cfg in "${FT_CONFIGS[@]}"; do
  name="$(basename "${cfg}" .py)"
  save_path="${REPO_ROOT}/logs/smoke1iter_${name}"

  if [[ "${name}" == *large* ]]; then
    weight="${LARGE_WEIGHT}"
  else
    weight="${BASE_WEIGHT}"
  fi

  if [[ "${name}" == *scannetpp* ]]; then
    data_root="${SCANNETPP_ROOT}"
  elif [[ "${name}" == *s3dis* ]]; then
    data_root="${S3DIS_ROOT}"
  else
    data_root="${SCANNET_ROOT}"
  fi

  echo "=== 1-ITER FT SMOKE: ${cfg} ==="
  echo "    data_root=${data_root}"
  echo "    weight=${weight}"

  python "${POINTCEPT_ROOT}/tools/train.py" \
    --config-file "${REPO_ROOT}/${cfg}" \
    --num-gpus 1 \
    --options \
      save_path="${save_path}" \
      weight="${weight}" \
      enable_wandb=False \
      epoch=1 \
      eval_epoch=1 \
      num_worker=2 \
      batch_size=2 \
      data.train.loop=1 \
      data.val.loop=1 \
      data.test.loop=1 \
      data.train.data_root="${data_root}" \
      data.val.data_root="${data_root}" \
      data.test.data_root="${data_root}"
done

echo "[DONE] Finished 1-iteration smoke checks for 7 FT configs."
