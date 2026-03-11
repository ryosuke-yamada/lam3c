#!/usr/bin/env python3
"""Run CLIP-based indoor filtering and segmentation locally."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}

from pipeline.runtime import (
    BASH_BIN,
    DOWNLOAD_ROOT,
    ROOT_DIR,
    SEGMENTATION_ROOT,
    build_runtime_env,
    default_segmentation_concurrency,
    print_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the segmentation stage locally")
    parser.add_argument("--input-root", type=Path, default=DOWNLOAD_ROOT, help="Directory containing input videos")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=SEGMENTATION_ROOT,
        help="Directory where segmentation outputs are written",
    )
    parser.add_argument(
        "--runner",
        choices=["queue", "batch"],
        default="queue",
        help="queue: dynamic task queue, batch: directory-wise batch runner",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=default_segmentation_concurrency(),
        help="Number of concurrent workers",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        action="append",
        default=[],
        help="Process only the specified input video file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Process only the downloaded file matching this video_id under --input-root. Can be passed multiple times; use --video-id=<id> when the id starts with '-'.",
    )
    parser.add_argument("--gpu-ids", default="", help="Comma-separated GPU IDs, e.g. 0,1,2,3")
    parser.add_argument("--num-shards", type=int, default=0, help="Optional shard count for batch mode")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards-1]")
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it")
    args = parser.parse_args()

    if args.concurrency < 1:
        parser.error("--concurrency must be >= 1")
    if args.num_shards < 0:
        parser.error("--num-shards must be >= 0")
    if args.num_shards and (args.shard_id < 0 or args.shard_id >= args.num_shards):
        parser.error("--shard-id must be in [0, num_shards-1]")

    if args.num_shards > 0 and args.runner != "batch":
        parser.error("--num-shards is only supported with --runner batch")
    if (args.video_path or args.video_id) and args.num_shards > 0:
        parser.error("--num-shards is not supported together with --video-path/--video-id")

    explicit_videos = resolve_explicit_videos(args.input_root, args.video_path, args.video_id, parser)
    if explicit_videos:
        script_path = ROOT_DIR / "pipeline" / "segmentation" / "batch_run_label_segmentation_queue.sh"
    elif args.runner == "queue":
        script_path = ROOT_DIR / "pipeline" / "segmentation" / "batch_run_label_segmentation_queue.sh"
    else:
        script_path = ROOT_DIR / "pipeline" / "segmentation" / "batch_run_label_segmentation.sh"

    extra_env = {}
    temp_queue_path: str | None = None
    input_root = args.input_root
    if explicit_videos:
        input_root = resolve_selected_video_root(args.input_root, explicit_videos)
        temp_queue_path = write_explicit_queue(explicit_videos)
        extra_env["QUEUE_FILE"] = temp_queue_path

    command = [
        BASH_BIN,
        str(script_path),
        str(input_root),
        str(args.output_root),
        str(args.concurrency),
    ]
    print_command(command)
    if explicit_videos:
        print(f"[INFO] selected videos: {len(explicit_videos)}", flush=True)

    if args.dry_run:
        if temp_queue_path is not None:
            try:
                os.unlink(temp_queue_path)
            except OSError:
                pass
        return 0

    if args.gpu_ids:
        extra_env["GPU_IDS"] = args.gpu_ids
    if args.num_shards > 0:
        extra_env["NUM_SHARDS"] = str(args.num_shards)
        extra_env["SHARD_ID"] = str(args.shard_id)

    env = build_runtime_env(extra_env)
    try:
        completed = subprocess.run(command, cwd=ROOT_DIR, env=env)
        return int(completed.returncode)
    finally:
        if temp_queue_path is not None:
            try:
                os.unlink(temp_queue_path)
            except OSError:
                pass


def resolve_explicit_videos(
    input_root: Path,
    video_paths: list[Path],
    video_ids: list[str],
    parser: argparse.ArgumentParser,
) -> list[Path]:
    selected: list[Path] = []

    for video_path in video_paths:
        resolved = video_path.expanduser().resolve()
        if not resolved.is_file():
            parser.error(f"--video-path not found: {video_path}")
        if resolved.suffix.lower() not in VIDEO_EXTENSIONS:
            parser.error(f"--video-path is not a supported video file: {video_path}")
        selected.append(resolved)

    if video_ids:
        root = input_root.expanduser().resolve()
        if not root.exists():
            parser.error(f"--input-root not found: {input_root}")
        discovered = {path.resolve() for path in root.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS}
        for video_id in video_ids:
            matches = sorted(path for path in discovered if path.stem == video_id)
            if not matches:
                parser.error(f"--video-id not found under {root}: {video_id}")
            selected.extend(matches)

    deduped: list[Path] = []
    seen = set()
    for path in selected:
        key = str(path)
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped


def resolve_selected_video_root(default_root: Path, explicit_videos: list[Path]) -> Path:
    default_root = default_root.expanduser().resolve()
    if explicit_videos and all(is_relative_to(path, default_root) for path in explicit_videos):
        return default_root

    parent_strings = [str(path.parent) for path in explicit_videos]
    return Path(os.path.commonpath(parent_strings))


def write_explicit_queue(explicit_videos: list[Path]) -> str:
    with tempfile.NamedTemporaryFile("w", prefix="roomtours_gen_seg_", suffix=".txt", delete=False) as handle:
        for path in explicit_videos:
            handle.write(str(path) + "\n")
        return handle.name


def is_relative_to(path: Path, base: Path) -> bool:
    try:
        path.relative_to(base)
        return True
    except ValueError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
