#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PI3_VENV=${PI3_VENV:-$ROOT_DIR/.venv_pi3}
VGGT_VENV=${VGGT_VENV:-$ROOT_DIR/.venv_vggt}

create_venv() {
  local venv_path="$1"
  if [ ! -d "$venv_path" ]; then
    "$PYTHON_BIN" -m venv "$venv_path"
  fi
  "$venv_path/bin/pip" install --upgrade pip setuptools wheel
}

install_pi3_stack() {
  create_venv "$PI3_VENV"
  "$PI3_VENV/bin/pip" install -r "$ROOT_DIR/third_party/Pi3/requirements.txt"
  "$PI3_VENV/bin/pip" install clip-anytorch 'scenedetect[opencv]'
}

install_vggt_stack() {
  create_venv "$VGGT_VENV"
  "$VGGT_VENV/bin/pip" install -r "$ROOT_DIR/third_party/vggt/requirements.txt"
  "$VGGT_VENV/bin/pip" install -r "$ROOT_DIR/third_party/vggt/requirements_demo.txt"
}

install_pi3_stack
install_vggt_stack

echo "[INFO] Created: $PI3_VENV"
echo "[INFO] Created: $VGGT_VENV"
echo "[INFO] Pi3 + segmentation stack installed into .venv_pi3"
echo "[INFO] VGGT stack installed into .venv_vggt"
