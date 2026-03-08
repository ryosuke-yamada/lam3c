#!/usr/bin/env python3
"""Download the canonical RoomTours video list."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from pipeline.runtime import (
    DOWNLOAD_ARCHIVE,
    DOWNLOAD_FAILURE_LOG,
    DOWNLOAD_ROOT,
    ROOT_DIR,
    VIDEO_LIST_CSV,
    build_runtime_env,
    get_python_executable,
    print_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download videos listed in video_lists.csv")
    parser.add_argument("--csv", type=Path, default=VIDEO_LIST_CSV, help="Path to the input CSV manifest")
    parser.add_argument(
        "--output-root", type=Path, default=DOWNLOAD_ROOT, help="Directory where downloaded videos are stored"
    )
    parser.add_argument(
        "--archive", type=Path, default=DOWNLOAD_ARCHIVE, help="yt-dlp download archive path"
    )
    parser.add_argument(
        "--failure-log",
        type=Path,
        default=DOWNLOAD_FAILURE_LOG,
        help="Path to write failed downloads as TSV",
    )
    parser.add_argument("--yt-dlp-bin", default="yt-dlp", help="yt-dlp executable name or path")
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Maximum number of manifest entries to process (0 = all entries)",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Restrict downloads to specific video_id values. Can be passed multiple times; use --video-id=<id> when the id starts with '-'.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it")
    args = parser.parse_args()

    python_bin = get_python_executable()
    command = [
        str(python_bin),
        str(ROOT_DIR / "pipeline" / "download_video_list.py"),
        "--csv",
        str(args.csv),
        "--output-root",
        str(args.output_root),
        "--archive",
        str(args.archive),
        "--failure-log",
        str(args.failure_log),
        "--yt-dlp-bin",
        args.yt_dlp_bin,
    ]
    if args.max_videos:
        command.extend(["--max-videos", str(args.max_videos)])
    for video_id in args.video_id:
        command.append(f"--video-id={video_id}")
    print_command(command)

    if args.dry_run:
        return 0

    completed = subprocess.run(command, cwd=ROOT_DIR, env=build_runtime_env())
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
