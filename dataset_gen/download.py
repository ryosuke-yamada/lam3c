#!/usr/bin/env python3
"""Public entrypoint for downloading the canonical RoomTours video list."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
from typing import Dict, List

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = "default"
INTERNAL_COMMAND = ROOT_DIR / "pipeline" / "download_videos.sh"


def parse_env_overrides(items: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected NAME=VALUE.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid override '{item}'. NAME cannot be empty.")
        overrides[key] = value
    return overrides


def main() -> int:
    parser = argparse.ArgumentParser(description="Download videos from video_lists.csv")
    parser.add_argument(
        "dataset_config",
        nargs="?",
        default=DEFAULT_CONFIG,
        help="Dataset config name under configs/datasets/ or an explicit config path",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Temporary environment-variable override passed to the pipeline",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved command without executing it",
    )
    args = parser.parse_args()

    try:
        overrides = parse_env_overrides(args.overrides)
    except ValueError as exc:
        parser.error(str(exc))

    if not INTERNAL_COMMAND.exists():
        parser.error(f"Missing internal command: {INTERNAL_COMMAND}")

    cmd = [str(INTERNAL_COMMAND), args.dataset_config]
    print("[INFO] command:", " ".join(cmd), flush=True)
    if overrides:
        for key, value in sorted(overrides.items()):
            print(f"[INFO] env override: {key}={value}", flush=True)

    if args.dry_run:
        return 0

    env = os.environ.copy()
    env.update(overrides)
    completed = subprocess.run(cmd, cwd=ROOT_DIR, env=env)
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
