#!/usr/bin/env python3
"""Download the canonical RoomTours video list with stable local filenames."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Iterable, List


@dataclass(frozen=True)
class VideoEntry:
    video_id: str
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download videos listed in video_lists.csv")
    parser.add_argument("--csv", required=True, help="Path to the input CSV manifest")
    parser.add_argument("--output-root", required=True, help="Directory where downloaded videos are stored")
    parser.add_argument("--archive", default="", help="yt-dlp download archive path")
    parser.add_argument("--failure-log", default="", help="Path to write failed downloads as TSV")
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
    return parser.parse_args()


def load_entries(csv_path: Path) -> List[VideoEntry]:
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV manifest not found: {csv_path}")

    entries: List[VideoEntry] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV manifest has no header: {csv_path}")
        required_columns = {"video_id", "url"}
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV manifest is missing required columns {sorted(missing)}: {csv_path}"
            )

        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            url = (row.get("url") or "").strip()
            if not video_id or not url:
                continue
            entries.append(VideoEntry(video_id=video_id, url=url))
    return entries


def filter_entries(
    entries: List[VideoEntry], selected_video_ids: List[str], max_videos: int
) -> List[VideoEntry]:
    filtered = entries
    if selected_video_ids:
        requested = {video_id.strip() for video_id in selected_video_ids if video_id.strip()}
        filtered = [entry for entry in entries if entry.video_id in requested]
        missing = sorted(requested.difference(entry.video_id for entry in filtered))
        for video_id in missing:
            print(f"[WARN] video_id not found in manifest: {video_id}", file=sys.stderr)

    if max_videos > 0:
        filtered = filtered[:max_videos]
    return filtered


def existing_downloads(output_root: Path, video_id: str) -> List[Path]:
    matches = []
    for path in sorted(output_root.glob(f"{video_id}.*")):
        if not path.is_file():
            continue
        try:
            if path.stat().st_size <= 0:
                continue
        except OSError:
            continue
        matches.append(path)
    return matches


def build_command(yt_dlp_bin: str, entry: VideoEntry, output_root: Path, archive_path: Path | None) -> List[str]:
    command = [
        yt_dlp_bin,
        "--ignore-config",
        "--no-overwrites",
        "--no-part",
        "--newline",
        "--extractor-args",
        "youtube:player_client=android",
        "--output",
        str(output_root / f"{entry.video_id}.%(ext)s"),
    ]
    if archive_path is not None:
        command.extend(["--download-archive", str(archive_path)])
    command.append(entry.url)
    return command


def write_failure_log(path: Path, failures: Iterable[tuple[str, str, str]]) -> None:
    rows = list(failures)
    if not rows:
        if path.exists():
            path.unlink()
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("video_id\turl\terror\n")
        for video_id, url, error in rows:
            handle.write(f"{video_id}\t{url}\t{error}\n")


def main() -> int:
    args = parse_args()
    if args.max_videos < 0:
        print("[ERROR] --max-videos must be >= 0", file=sys.stderr)
        return 1

    csv_path = Path(args.csv).resolve()
    output_root = Path(args.output_root).resolve()
    archive_path = Path(args.archive).resolve() if args.archive else None
    failure_log = Path(args.failure_log).resolve() if args.failure_log else None

    yt_dlp_bin = shutil.which(args.yt_dlp_bin)
    if yt_dlp_bin is None:
        print(f"[ERROR] yt-dlp not found: {args.yt_dlp_bin}", file=sys.stderr)
        return 1

    try:
        entries = load_entries(csv_path)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    entries = filter_entries(entries, args.video_id, args.max_videos)
    if not entries:
        print("[ERROR] No video entries selected for download.", file=sys.stderr)
        return 1

    output_root.mkdir(parents=True, exist_ok=True)
    if archive_path is not None:
        archive_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] video list: {csv_path}")
    print(f"[INFO] output root: {output_root}")
    if archive_path is not None:
        print(f"[INFO] download archive: {archive_path}")
    print(f"[INFO] entries: {len(entries)}")

    downloaded = 0
    skipped = 0
    failed: List[tuple[str, str, str]] = []

    for index, entry in enumerate(entries, start=1):
        existing = existing_downloads(output_root, entry.video_id)
        if existing:
            skipped += 1
            print(
                f"[{index}/{len(entries)}] skip existing {entry.video_id}: "
                f"{', '.join(path.name for path in existing)}"
            )
            continue

        command = build_command(yt_dlp_bin, entry, output_root, archive_path)
        print(f"[{index}/{len(entries)}] download {entry.video_id}: {entry.url}")
        completed = subprocess.run(command, check=False)
        if completed.returncode == 0:
            downloaded += 1
            continue

        failed.append((entry.video_id, entry.url, f"yt-dlp exit code {completed.returncode}"))
        print(
            f"[WARN] failed {entry.video_id} with exit code {completed.returncode}",
            file=sys.stderr,
        )

    if failure_log is not None:
        write_failure_log(failure_log, failed)
        if failed:
            print(f"[WARN] failure log written to: {failure_log}")

    print(
        f"[INFO] download summary: downloaded={downloaded}, skipped_existing={skipped}, failures={len(failed)}"
    )
    if failed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
