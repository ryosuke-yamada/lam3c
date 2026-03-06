#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
PI3_VENV=${PI3_VENV:-$ROOT_DIR/.venv_pi3}

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

install_pi3_stack

echo "[INFO] Created: $PI3_VENV"
echo "[INFO] Pi3 + segmentation stack installed into .venv_pi3"
