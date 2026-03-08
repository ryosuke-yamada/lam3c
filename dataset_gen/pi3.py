#!/usr/bin/env python3
"""Run Pi3 point-cloud generation locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

from pipeline.runtime import (
    PI3_ROOT,
    ROOT_DIR,
    SEGMENTATION_ROOT,
    build_runtime_env,
    default_pi3_num_gpus,
    get_python_executable,
    print_command,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Pi3 stage locally")
    parser.add_argument("--input-root", type=Path, default=SEGMENTATION_ROOT, help="Segmentation output root")
    parser.add_argument("--output-root", type=Path, default=PI3_ROOT, help="Directory where Pi3 outputs are written")
    parser.add_argument(
        "--layout",
        default="roomtours_scenes",
        choices=["roomtours_scenes", "recursive_images"],
        help="Input layout consumed by the Pi3 batch script",
    )
    parser.add_argument("--interval", type=int, default=1, help="Sampling interval for input frames/images")
    parser.add_argument("--num-gpus", type=int, default=default_pi3_num_gpus(), help="Number of GPUs to use")
    parser.add_argument("--num-shards", type=int, default=1, help="Optional shard count for manual parallel runs")
    parser.add_argument("--shard-id", type=int, default=0, help="Shard index in [0, num_shards-1]")
    parser.add_argument("--pixel-limit", type=int, default=255000, help="Approximate per-frame resize budget")
    parser.add_argument("--max-entries", type=int, default=0, help="Limit entries processed (0 = no limit)")
    parser.add_argument("--scene-json", type=Path, default=None, help="Optional explicit scene target list")
    parser.add_argument(
        "--scene-path",
        type=Path,
        action="append",
        default=[],
        help="Process only the specified segmented scene mp4 file. Can be passed multiple times.",
    )
    parser.add_argument("--video-skip-seconds", type=int, default=10, help="Initial seconds to skip per video")
    parser.add_argument("--video-target-frames", type=int, default=400, help="Target frame cap for videos")
    parser.add_argument("--image-target-frames", type=int, default=1500, help="Target image cap for image directories")
    parser.add_argument("--preserve-order", action="store_true", help="Preserve discovered input order")
    parser.add_argument(
        "--video-adjust-for-high-fps",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Increase sampling interval automatically for high-FPS videos",
    )
    parser.add_argument(
        "--include-processed",
        action="store_true",
        help="Include entries whose outputs already exist",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Overwrite existing outputs instead of skipping them",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved command without executing it")
    args = parser.parse_args()

    python_bin = get_python_executable()

    if args.num_gpus < 1:
        parser.error("--num-gpus must be >= 1")
    if args.num_shards < 1:
        parser.error("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        parser.error("--shard-id must be in [0, num_shards-1]")
    if args.scene_json is not None and args.scene_path:
        parser.error("--scene-json and --scene-path cannot be used together")
    if args.scene_path and args.layout != "roomtours_scenes":
        parser.error("--scene-path is only supported with --layout roomtours_scenes")

    temp_scene_json: str | None = None
    command = [
        str(python_bin),
        str(ROOT_DIR / "pipeline" / "pi3" / "pi3_batch_datasets.py"),
        "--layout",
        args.layout,
        "--input_base",
        str(args.input_root),
        "--output_base",
        str(args.output_root),
        "--interval",
        str(args.interval),
        "--num_gpus",
        str(args.num_gpus),
        "--num_shards",
        str(args.num_shards),
        "--shard_id",
        str(args.shard_id),
        "--pixel_limit",
        str(args.pixel_limit),
        "--max_entries",
        str(args.max_entries),
        "--video_skip_seconds",
        str(args.video_skip_seconds),
        "--video_target_frames",
        str(args.video_target_frames),
        "--image_target_frames",
        str(args.image_target_frames),
    ]
    scene_json_path = args.scene_json
    if args.scene_path:
        temp_scene_json = write_explicit_scene_json(args.input_root, args.output_root, args.scene_path, parser)
        scene_json_path = Path(temp_scene_json)
    if scene_json_path is not None:
        command.extend(["--scene_json", str(scene_json_path)])
    if args.preserve_order:
        command.append("--preserve_order")
    if args.video_adjust_for_high_fps:
        command.append("--video_adjust_for_high_fps")
    if args.include_processed:
        command.append("--include_processed")
    if args.overwrite_existing:
        command.append("--overwrite_existing")

    print_command(command)
    if args.scene_path:
        print(f"[INFO] selected scenes: {len(args.scene_path)}", flush=True)

    if args.dry_run:
        if temp_scene_json is not None:
            try:
                Path(temp_scene_json).unlink()
            except OSError:
                pass
        return 0

    try:
        completed = subprocess.run(command, cwd=ROOT_DIR, env=build_runtime_env())
        return int(completed.returncode)
    finally:
        if temp_scene_json is not None:
            try:
                Path(temp_scene_json).unlink()
            except OSError:
                pass


def write_explicit_scene_json(
    input_root: Path,
    output_root: Path,
    scene_paths: list[Path],
    parser: argparse.ArgumentParser,
) -> str:
    entries = []
    for scene_path in scene_paths:
        resolved = scene_path.expanduser().resolve()
        if not resolved.is_file():
            parser.error(f"--scene-path not found: {scene_path}")
        if resolved.suffix.lower() != ".mp4":
            parser.error(f"--scene-path must point to an .mp4 file: {scene_path}")
        output_path = default_scene_output_path(resolved, input_root.expanduser().resolve(), output_root.expanduser().resolve())
        entries.append(
            {
                "input_path": str(resolved),
                "output_path": str(output_path),
            }
        )

    with tempfile.NamedTemporaryFile("w", prefix="dataset_gen_pi3_", suffix=".json", delete=False) as handle:
        json.dump(entries, handle, indent=2)
        handle.write("\n")
        return handle.name


def default_scene_output_path(scene_path: Path, input_root: Path, output_root: Path) -> Path:
    try:
        rel_path = scene_path.relative_to(input_root)
    except ValueError:
        return output_root / scene_path.stem / "pi3.ply"

    parent = rel_path.parent
    if parent.name == "scenes":
        parent = parent.parent
    if not parent.parts:
        return output_root / rel_path.stem / "pi3.ply"
    return output_root / parent / rel_path.stem / "pi3.ply"


if __name__ == "__main__":
    raise SystemExit(main())
