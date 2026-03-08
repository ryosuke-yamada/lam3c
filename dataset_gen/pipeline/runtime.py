#!/usr/bin/env python3
"""Internal runtime helpers for the public dataset-generation entrypoints."""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, Sequence

ROOT_DIR = Path(__file__).resolve().parents[1]
VIDEO_LIST_CSV = ROOT_DIR.parent / "video_lists.csv"
WORK_ROOT = ROOT_DIR / "data" / "roomtours"
DOWNLOAD_ROOT = WORK_ROOT / "videos"
DOWNLOAD_ARCHIVE = WORK_ROOT / "download_archive.txt"
DOWNLOAD_FAILURE_LOG = WORK_ROOT / "download_failures.tsv"
SEGMENTATION_ROOT = WORK_ROOT / "segmentation"
PI3_ROOT = WORK_ROOT / "pi3"
BASH_BIN = shutil.which("bash") or "/bin/bash"


def get_python_executable() -> Path:
    override = os.environ.get("DATASET_GEN_PYTHON", "").strip()
    if override:
        return Path(override)
    if sys.executable:
        return Path(sys.executable)
    resolved = shutil.which("python3") or shutil.which("python")
    if resolved:
        return Path(resolved)
    raise SystemExit("Python executable not found. Activate an environment first.")


def build_runtime_env(extra_env: dict[str, str] | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHON_BIN", str(get_python_executable()))
    if extra_env:
        env.update(extra_env)
    return env


def detect_gpu_count() -> int:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        return len([item for item in visible.split(",") if item.strip()])

    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return 0

    completed = subprocess.run(
        [nvidia_smi, "--query-gpu=index", "--format=csv,noheader"],
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return 0
    return len([line for line in completed.stdout.splitlines() if line.strip()])


def default_segmentation_concurrency() -> int:
    gpu_count = detect_gpu_count()
    if gpu_count > 0:
        return gpu_count
    return 1


def default_pi3_num_gpus() -> int:
    gpu_count = detect_gpu_count()
    if gpu_count > 0:
        return gpu_count
    return 1


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex_quote(part) for part in parts)


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def print_command(parts: Iterable[str]) -> None:
    print("[INFO] command:", shell_join(list(parts)), flush=True)
