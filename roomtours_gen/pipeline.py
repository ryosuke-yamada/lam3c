#!/usr/bin/env python3
"""Run the full RoomTours dataset-generation pipeline locally."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import subprocess

from pipeline.runtime import (
    DOWNLOAD_ROOT,
    PI3_ROOT,
    PREPROCESS_ROOT,
    ROOT_DIR,
    SEGMENTATION_ROOT,
    VIDEO_LIST_CSV,
    build_runtime_env,
    default_pi3_num_gpus,
    default_segmentation_concurrency,
    get_python_executable,
    print_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the full RoomTours dataset-generation pipeline locally")
    parser.add_argument("--csv", type=Path, default=VIDEO_LIST_CSV, help="Path to the input CSV manifest")
    parser.add_argument("--download-root", type=Path, default=DOWNLOAD_ROOT, help="Directory for downloaded videos")
    parser.add_argument(
        "--segmentation-root",
        type=Path,
        default=SEGMENTATION_ROOT,
        help="Directory for segmentation outputs",
    )
    parser.add_argument("--pi3-root", type=Path, default=PI3_ROOT, help="Directory for Pi3 outputs")
    parser.add_argument(
        "--preprocess-root",
        type=Path,
        default=PREPROCESS_ROOT,
        help="Directory for preprocessing outputs",
    )
    parser.add_argument(
        "--max-videos",
        type=int,
        default=0,
        help="Limit the run to the first N manifest entries (0 = all entries)",
    )
    parser.add_argument(
        "--video-id",
        action="append",
        default=[],
        help="Restrict the run to specific video_id values. Can be passed multiple times.",
    )
    parser.add_argument(
        "--segmentation-concurrency",
        type=int,
        default=default_segmentation_concurrency(),
        help="Number of concurrent segmentation workers",
    )
    parser.add_argument("--gpu-ids", default="", help="Comma-separated GPU IDs for segmentation")
    parser.add_argument(
        "--pi3-num-gpus",
        type=int,
        default=default_pi3_num_gpus(),
        help="Number of GPUs to use for Pi3",
    )
    parser.add_argument("--pi3-num-shards", type=int, default=1, help="Shard count for Pi3")
    parser.add_argument("--pi3-shard-id", type=int, default=0, help="Shard index for Pi3")
    parser.add_argument(
        "--save-intermediates",
        action="store_true",
        help="Save intermediate point clouds during preprocessing",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose preprocessing logs")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download stage")
    parser.add_argument("--skip-segmentation", action="store_true", help="Skip the segmentation stage")
    parser.add_argument("--skip-pi3", action="store_true", help="Skip the Pi3 stage")
    parser.add_argument("--skip-preprocess", action="store_true", help="Skip the preprocessing stage")
    parser.add_argument("--overwrite-existing", action="store_true", help="Overwrite existing outputs when supported")
    parser.add_argument("--dry-run", action="store_true", help="Print stage commands without executing them")
    args = parser.parse_args()

    if args.max_videos < 0:
        parser.error("--max-videos must be >= 0")
    if args.segmentation_concurrency < 1:
        parser.error("--segmentation-concurrency must be >= 1")
    if args.pi3_num_gpus < 1:
        parser.error("--pi3-num-gpus must be >= 1")
    if args.pi3_num_shards < 1:
        parser.error("--pi3-num-shards must be >= 1")
    if args.pi3_shard_id < 0 or args.pi3_shard_id >= args.pi3_num_shards:
        parser.error("--pi3-shard-id must be in [0, pi3-num-shards-1]")

    target_video_ids = resolve_target_video_ids(args.csv, args.video_id, args.max_videos, parser)

    if not args.skip_download:
        download_command = build_download_command(args, target_video_ids)
        if run_stage("download", download_command, dry_run=args.dry_run) != 0:
            return 1

    if not args.skip_segmentation:
        segmentation_command = build_segmentation_command(args, target_video_ids)
        if run_stage("segmentation", segmentation_command, dry_run=args.dry_run) != 0:
            return 1

    scene_paths: list[Path] = []
    if not args.skip_pi3:
        if target_video_ids:
            scene_paths = discover_scene_paths(args.segmentation_root, target_video_ids)
            if not scene_paths and not args.dry_run:
                raise SystemExit("No segmented scenes were found for the requested video IDs.")
            if not scene_paths and args.dry_run:
                print(
                    "[INFO] dry-run: no segmented scenes exist yet, so the Pi3 command is shown without "
                    "per-scene filtering",
                    flush=True,
                )
        pi3_command = build_pi3_command(args, scene_paths)
        if run_stage("pi3", pi3_command, dry_run=args.dry_run) != 0:
            return 1

    if not args.skip_preprocess:
        ply_paths: list[Path] = []
        if target_video_ids:
            ply_paths = discover_pi3_outputs(args.pi3_root, target_video_ids)
            if not ply_paths and not args.dry_run:
                raise SystemExit("No Pi3 outputs were found for the requested video IDs.")
            if not ply_paths and args.dry_run:
                print(
                    "[INFO] dry-run: no Pi3 outputs exist yet, so the preprocessing command is shown without "
                    "per-file filtering",
                    flush=True,
                )
        preprocess_command = build_preprocess_command(args, ply_paths)
        if run_stage("preprocess", preprocess_command, dry_run=args.dry_run) != 0:
            return 1

    return 0


def resolve_target_video_ids(
    csv_path: Path,
    explicit_video_ids: list[str],
    max_videos: int,
    parser: argparse.ArgumentParser,
) -> list[str]:
    if explicit_video_ids:
        deduped: list[str] = []
        seen = set()
        for video_id in explicit_video_ids:
            if video_id not in seen:
                seen.add(video_id)
                deduped.append(video_id)
        return deduped

    if max_videos == 0:
        return []

    if not csv_path.exists():
        parser.error(f"--csv not found: {csv_path}")

    resolved: list[str] = []
    with csv_path.open("r", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            video_id = row[0].strip()
            if not video_id or video_id == "video_id":
                continue
            resolved.append(video_id)
            if len(resolved) >= max_videos:
                break
    return resolved


def build_download_command(args: argparse.Namespace, target_video_ids: list[str]) -> list[str]:
    command = [
        str(get_python_executable()),
        str(ROOT_DIR / "download.py"),
        "--csv",
        str(args.csv),
        "--output-root",
        str(args.download_root),
    ]
    if target_video_ids:
        for video_id in target_video_ids:
            command.append(f"--video-id={video_id}")
    elif args.max_videos > 0:
        command.extend(["--max-videos", str(args.max_videos)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_segmentation_command(args: argparse.Namespace, target_video_ids: list[str]) -> list[str]:
    command = [
        str(get_python_executable()),
        str(ROOT_DIR / "segmentation.py"),
        "--input-root",
        str(args.download_root),
        "--output-root",
        str(args.segmentation_root),
        "--concurrency",
        str(args.segmentation_concurrency),
    ]
    if args.gpu_ids:
        command.extend(["--gpu-ids", args.gpu_ids])
    for video_id in target_video_ids:
        command.append(f"--video-id={video_id}")
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_pi3_command(args: argparse.Namespace, scene_paths: list[Path]) -> list[str]:
    command = [
        str(get_python_executable()),
        str(ROOT_DIR / "pi3.py"),
        "--input-root",
        str(args.segmentation_root),
        "--output-root",
        str(args.pi3_root),
        "--num-gpus",
        str(args.pi3_num_gpus),
        "--num-shards",
        str(args.pi3_num_shards),
        "--shard-id",
        str(args.pi3_shard_id),
    ]
    if args.overwrite_existing:
        command.append("--overwrite-existing")
    if scene_paths:
        for scene_path in scene_paths:
            command.extend(["--scene-path", str(scene_path)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def build_preprocess_command(args: argparse.Namespace, ply_paths: list[Path]) -> list[str]:
    command = [
        str(get_python_executable()),
        str(ROOT_DIR / "preprocess.py"),
        "--input-root",
        str(args.pi3_root),
        "--output-root",
        str(args.preprocess_root),
    ]
    if args.overwrite_existing:
        command.append("--overwrite-existing")
    if args.save_intermediates:
        command.append("--save-intermediates")
    if args.verbose:
        command.append("--verbose")
    if ply_paths:
        for ply_path in ply_paths:
            command.extend(["--ply-path", str(ply_path)])
    if args.dry_run:
        command.append("--dry-run")
    return command


def discover_scene_paths(segmentation_root: Path, target_video_ids: list[str]) -> list[Path]:
    paths: list[Path] = []
    for video_id in target_video_ids:
        scene_dir = segmentation_root.expanduser().resolve() / video_id / "scenes"
        if not scene_dir.exists():
            continue
        paths.extend(sorted(path.resolve() for path in scene_dir.glob("*.mp4") if path.is_file()))
    return paths


def discover_pi3_outputs(pi3_root: Path, target_video_ids: list[str]) -> list[Path]:
    paths: list[Path] = []
    resolved_root = pi3_root.expanduser().resolve()
    for video_id in target_video_ids:
        video_root = resolved_root / video_id
        if not video_root.exists():
            continue
        paths.extend(sorted(path.resolve() for path in video_root.rglob("pi3.ply") if path.is_file()))
    return paths


def run_stage(stage_name: str, command: list[str], dry_run: bool) -> int:
    print(f"[INFO] stage: {stage_name}", flush=True)
    print_command(command)
    if dry_run:
        return 0
    completed = subprocess.run(command, cwd=ROOT_DIR, env=build_runtime_env())
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
