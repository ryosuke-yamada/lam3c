#!/usr/bin/env python3
import os
import sys
from pathlib import Path

if len(sys.argv) < 4:
    print("Usage: init_segmentation_queue.py <video_root> <output_root> <queue_file>", file=sys.stderr)
    sys.exit(1)

video_root = Path(sys.argv[1]).resolve()
output_root = Path(sys.argv[2]).resolve()
queue_file = Path(sys.argv[3]).resolve()

if not video_root.exists():
    print(f"[ERROR] Video root not found: {video_root}", file=sys.stderr)
    sys.exit(1)

extensions = {'.mp4', '.mov', '.m4v', '.avi', '.mkv', '.webm'}
tasks = []

for path in video_root.rglob('*'):
    if not path.is_file():
        continue
    # Skip rsync partials or transient folders (e.g., rsync in-progress)
    if any(part == '.rsync-partial' for part in path.parts):
        continue
    if path.suffix.lower() not in extensions:
        continue
    rel = path.relative_to(video_root)
    out_dir = output_root / rel.parent / path.stem
    scenes_dir = out_dir / 'scenes'
    skip_file = out_dir / 'SKIP_NO_FRAMES'
    processed = False
    if skip_file.exists():
        processed = True
    elif scenes_dir.exists():
        try:
            for item in scenes_dir.iterdir():
                if item.is_file() and item.suffix.lower() in {'.mp4', '.mkv', '.mov', '.m4v', '.webm'}:
                    processed = True
                    break
        except PermissionError:
            pass
    if not processed:
        tasks.append(str(path))

queue_file.parent.mkdir(parents=True, exist_ok=True)
with queue_file.open('w') as f:
    for item in sorted(tasks):
        f.write(item + '\n')

print(f"[INFO] queued {len(tasks)} videos", file=sys.stderr)
