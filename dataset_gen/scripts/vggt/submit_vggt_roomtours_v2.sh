#!/bin/bash
#PBS -N vggt-roomtours-v2
#PBS -q rt_HF
#PBS -P gag51492
#PBS -l select=1:ngpus=1
#PBS -l walltime=24:00:00

set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
VIDEO_BASE=${VIDEO_BASE:-/groups/gag51402/datasets/RoomTours/processed_label_segments_v2}
PI3_BASE=${PI3_BASE:-/groups/gag51402/datasets/roomtours_pi3_v2}
OUTPUT_BASE=${OUTPUT_BASE:-/groups/gag51402/datasets/roomtours_vggt_v2_200}
WORK_BASE=${WORK_BASE:-$PROJECT_DIR/tmp/vggt_roomtours_v2}
TARGET_IMAGES=${TARGET_IMAGES:-200}
PIXEL_LIMIT=${PIXEL_LIMIT:-255000}
INTERVAL=${INTERVAL:-1}
PI3_PYTHON=${PI3_PYTHON:-$PROJECT_DIR/.venv_pi3/bin/python}
CONF_THRES_VALUE=${CONF_THRES_VALUE:-0.1}

NUM_SHARDS=${NUM_SHARDS:-8}
if [ -n "${PBS_ARRAY_INDEX:-}" ]; then
  SHARD_ID=$((PBS_ARRAY_INDEX - 1))
else
  SHARD_ID=${SHARD_ID:-0}
fi

if [ "$SHARD_ID" -lt 0 ] || [ "$SHARD_ID" -ge "$NUM_SHARDS" ]; then
  echo "[ERROR] SHARD_ID=$SHARD_ID out of range for NUM_SHARDS=$NUM_SHARDS"
  exit 1
fi

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

LOG_DIR="$PROJECT_DIR/logs"
LOG_BASE="vggt_roomtours_v2.${PBS_JOBID:-noid}.${PBS_ARRAY_INDEX:-noidx}"
LOG_OUT="$LOG_DIR/$LOG_BASE.out"
LOG_ERR="$LOG_DIR/$LOG_BASE.err"
exec > >(tee -a "$LOG_OUT") 2> >(tee -a "$LOG_ERR" >&2)

if command -v module >/dev/null 2>&1; then
  source /etc/profile.d/modules.sh || true
  module load python/3.12/3.12.9 || true
  module load cuda/12.6 || true
fi

VENV_PATH="${VENV_PATH:-$PROJECT_DIR/.venv_vggt/bin/activate}"
if [ -f "$VENV_PATH" ]; then
  source "$VENV_PATH"
else
  echo "[ERROR] venv not found: $VENV_PATH"; exit 1
fi

export PYTHONPATH="$PROJECT_DIR/third_party/vggt:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

collect_videos() {
  VIDEO_BASE="$VIDEO_BASE" PI3_BASE="$PI3_BASE" "$PI3_PYTHON" - <<'PY'
import os, sys
from pathlib import Path

video_base = Path(os.environ['VIDEO_BASE'])
pi3_base = Path(os.environ['PI3_BASE'])

candidates = []
for ply_path in pi3_base.glob('*/*/*/pi3.ply'):
    try:
        channel_dir, video_dir, scene_dir = ply_path.parent.relative_to(pi3_base).parts
    except ValueError:
        continue
    scene_mp4 = video_base / channel_dir / video_dir / 'scenes' / f"{scene_dir}.mp4"
    if scene_mp4.exists():
        candidates.append(scene_mp4)
    else:
        print(f"[WARN] Missing source video for {ply_path}", file=sys.stderr)

for path in sorted({p.as_posix() for p in candidates}):
    print(path)
PY
}

mapfile -t ALL_VIDEOS < <(collect_videos)
TOTAL=${#ALL_VIDEOS[@]}
if [ "$TOTAL" -eq 0 ]; then
  echo "[ERROR] No RoomTours scene videos matched Pi3 outputs"
  exit 1
fi

echo "[INFO] Total videos: $TOTAL"

SELECTED_VIDEOS=()
for i in "${!ALL_VIDEOS[@]}"; do
  if (( i % NUM_SHARDS == SHARD_ID )); then
    SELECTED_VIDEOS+=("${ALL_VIDEOS[$i]}")
  fi
done

echo "[INFO] Shard $SHARD_ID will process ${#SELECTED_VIDEOS[@]} videos"

extract_frames_pi3() {
  local video="$1"
  local dst="$2"
  mkdir -p "$dst"
  rm -f "$dst"/* 2>/dev/null || true

  VIDEO_PATH="$video" OUTPUT_DIR="$dst" INTERVAL="$INTERVAL" PIXEL_LIMIT="$PIXEL_LIMIT" TARGET_IMAGES="$TARGET_IMAGES" PI3_SCRIPT_DIR="$PROJECT_DIR/scripts/pi3" "$PI3_PYTHON" - <<'PY'
import os, sys
from pathlib import Path
import torch
from torchvision import transforms

sys.path.insert(0, os.environ["PI3_SCRIPT_DIR"])
from pi3_batch_datasets import load_video_as_tensor_with_skip  # noqa: E402

video_path = os.environ['VIDEO_PATH']
dst_dir = Path(os.environ['OUTPUT_DIR'])
interval = int(os.environ.get('INTERVAL', '1'))
pixel_limit = int(os.environ.get('PIXEL_LIMIT', '255000'))
target_frames = int(os.environ.get('TARGET_IMAGES', '1500'))

dst_dir.mkdir(parents=True, exist_ok=True)

imgs = load_video_as_tensor_with_skip(
    video_path,
    interval=interval,
    skip_seconds=10,
    PIXEL_LIMIT=pixel_limit,
    adjust_for_high_fps=True,
    target_frames=target_frames,
    dataset_type='roomtours',
)

if imgs.numel() == 0:
    print(f"[WARN] No frames extracted from {video_path}", file=sys.stderr)
    sys.exit(0)

to_pil = transforms.ToPILImage()
for idx in range(imgs.shape[0]):
    pil = to_pil(imgs[idx].cpu())
    pil.save(dst_dir / f"{idx:06d}.jpg", quality=95)

print(f"[INFO] Pi3-aligned sampling produced {imgs.shape[0]} frames for {Path(video_path).name}")
PY
}

process_video() {
  local video="$1"
  local rel="${video#$VIDEO_BASE/}"
  local rel_no_ext="${rel%.mp4}"
  local scene_name="$(basename "$rel_no_ext")"
  local scenes_rel="$(dirname "$rel_no_ext")"
  local video_rel="$(dirname "$scenes_rel")"
  local output_scene_dir="$OUTPUT_BASE/$video_rel/$scene_name"
  local sparse_dir="$output_scene_dir/sparse"

  if [ -f "$sparse_dir/points3D.bin" ] || [ -f "$sparse_dir/images.bin" ]; then
    echo "[INFO] Skip $scene_name: output already exists"
    return
  fi

  local work_scene_dir="$WORK_BASE/$video_rel/$scene_name"
  local work_images_dir="$work_scene_dir/images"
  extract_frames_pi3 "$video" "$work_images_dir"

  local frame_count
  frame_count=$(find "$work_images_dir" -maxdepth 1 -type f -name '*.jpg' | wc -l | tr -d ' ')
  if [ "$frame_count" -eq 0 ]; then
    echo "[WARN] No frames extracted for $scene_name"
    rm -rf "$work_scene_dir"
    return
  fi

  echo "[INFO] Running VGGT on $scene_name with $frame_count frames"
  python "$SCRIPT_DIR/demo_colmap.py" \
    --scene_dir "$work_scene_dir" \
    --conf_thres_value "$CONF_THRES_VALUE"

  mkdir -p "$output_scene_dir/images" "$sparse_dir"
  rsync -a --delete "$work_images_dir/" "$output_scene_dir/images/"
  if [ -d "$work_scene_dir/sparse" ]; then
    rsync -a --delete "$work_scene_dir/sparse/" "$sparse_dir/"
  else
    echo "[WARN] VGGT output missing for $scene_name"
  fi
  rm -rf "$work_scene_dir" || true
}

for video in "${SELECTED_VIDEOS[@]}"; do
  process_video "$video"
done

echo "[INFO] Shard $SHARD_ID completed"
