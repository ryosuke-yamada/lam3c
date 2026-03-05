#!/usr/bin/env bash

source "$PROJECT_DIR/scripts/common/job_common.sh"

dataset_gen_run_vggt_submit() {
  dataset_gen_resolve_array_shard "$NUM_SHARDS"
  dataset_gen_prepare_project_dir
  dataset_gen_setup_logging "$LOG_BASE"
  dataset_gen_load_modules

  VENV_PATH=${VENV_PATH:-$PROJECT_DIR/.venv_vggt/bin/activate}
  dataset_gen_activate_venv "$VENV_PATH" "VGGT venv"

  export PYTHONPATH="$PROJECT_DIR/third_party/vggt:${PYTHONPATH:-}"
  export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
  PI3_PYTHON=${PI3_PYTHON:-$PROJECT_DIR/.venv_pi3/bin/python}

  dataset_gen_print_runtime_info
  echo "[INFO] VIDEO_BASE: $VIDEO_BASE"
  echo "[INFO] PI3_BASE: $PI3_BASE"
  echo "[INFO] OUTPUT_BASE: $OUTPUT_BASE"
  echo "[INFO] WORK_BASE: $WORK_BASE"

  collect_videos() {
    VIDEO_BASE="$VIDEO_BASE" PI3_BASE="$PI3_BASE" "$PI3_PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

video_base = Path(os.environ["VIDEO_BASE"])
pi3_base = Path(os.environ["PI3_BASE"])

candidates = []
for ply_path in pi3_base.glob("*/*/*/pi3.ply"):
    try:
        channel_dir, video_dir, scene_dir = ply_path.parent.relative_to(pi3_base).parts
    except ValueError:
        continue
    scene_mp4 = video_base / channel_dir / video_dir / "scenes" / f"{scene_dir}.mp4"
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
    echo "[ERROR] No scene videos matched existing Pi3 outputs"
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

    VIDEO_PATH="$video" OUTPUT_DIR="$dst" INTERVAL="$INTERVAL" PIXEL_LIMIT="$PIXEL_LIMIT" TARGET_IMAGES="$TARGET_IMAGES" \
    PI3_SCRIPT_DIR="$PROJECT_DIR/scripts/pi3" "$PI3_PYTHON" - <<'PY'
import os
import sys
from pathlib import Path
import torch
from torchvision import transforms

sys.path.insert(0, os.environ["PI3_SCRIPT_DIR"])
from pi3_batch_datasets import load_video_as_tensor_with_skip  # noqa: E402

video_path = os.environ["VIDEO_PATH"]
dst_dir = Path(os.environ["OUTPUT_DIR"])
interval = int(os.environ.get("INTERVAL", "1"))
pixel_limit = int(os.environ.get("PIXEL_LIMIT", "255000"))
target_frames = int(os.environ.get("TARGET_IMAGES", "1500"))

dst_dir.mkdir(parents=True, exist_ok=True)

imgs = load_video_as_tensor_with_skip(
    video_path,
    interval=interval,
    skip_seconds=10,
    PIXEL_LIMIT=pixel_limit,
    adjust_for_high_fps=True,
    target_frames=target_frames,
    dataset_type="roomtours",
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

  echo "[INFO] Done."
}
