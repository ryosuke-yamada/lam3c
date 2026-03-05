#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

if [ ${#@} -lt 1 ]; then
  echo "Usage: $0 <input_video> [output_dir]" >&2
  exit 1
fi

VIDEO="$1"
OUTDIR="${2:-outputs_label_avi}"

# チューニング用パラメータ（環境変数で上書き可能）
# SAMPLE_STRIDE: フレームのサンプリング間隔。1で全フレーム処理、2なら1/2に間引き。
SAMPLE_STRIDE=${SAMPLE_STRIDE:-1}
# BATCH_SIZE: CLIP推論のバッチサイズ。GPU/CPUメモリに合わせて調整。
BATCH_SIZE=${BATCH_SIZE:-64}
# DEVICE: 利用デバイスの指定。"cuda"/"cpu" など。空なら自動判定。
DEVICE=${DEVICE:-}
# SMOOTH_SEC: 室内確率の移動平均（秒）。大きいほど変化に鈍感で安定、区切りは粗くなりやすい。
SMOOTH_SEC=${SMOOTH_SEC:-0.5}
# MIN_INSIDE_SEC: 室内の連続区間がこの秒数未満なら短すぎるとみなし削除（ノイズ除去）。
MIN_INSIDE_SEC=${MIN_INSIDE_SEC:-10.0}
# OUTSIDE_MARGIN_SEC: 室外と判定された箇所の前後に追加で削除するマージン（秒）。
OUTSIDE_MARGIN_SEC=${OUTSIDE_MARGIN_SEC:-0.2}
# TARGET_SEG_FPS: ラベル分割時のCLIP判定のサンプリングFPS。高いほど細かく切れるが計算量が増える。
TARGET_SEG_FPS=${TARGET_SEG_FPS:-5.0}
# MIN_ROOM_SEC: ラベル分割で保持する最小部屋区間（秒）。この長さ未満は隣接区間にマージ。
MIN_ROOM_SEC=${MIN_ROOM_SEC:-10.0}
SKIP_LOG_PATH=${SKIP_LOG:-}
if [ -z "$SKIP_LOG_PATH" ]; then
  SKIP_LOG_PATH="$(dirname "$OUTDIR")/_skipped_videos.tsv"
fi

mkdir -p "$OUTDIR"

# Stage 1: Generate inside_only.avi (MJPG) using CLIP filtering to avoid MP4 decode issues.
VIDEO="$VIDEO" OUTDIR="$OUTDIR" SKIP_LOG="$SKIP_LOG_PATH" SAMPLE_STRIDE="$SAMPLE_STRIDE" BATCH_SIZE="$BATCH_SIZE" DEVICE="$DEVICE" \
SMOOTH_SEC="$SMOOTH_SEC" MIN_INSIDE_SEC="$MIN_INSIDE_SEC" OUTSIDE_MARGIN_SEC="$OUTSIDE_MARGIN_SEC" \
SEGMENTATION_SCRIPT_DIR="$SCRIPT_DIR" \
python - <<'PY'
import csv
import os
import sys
import cv2
script_dir = os.environ["SEGMENTATION_SCRIPT_DIR"]
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from room_tour_pipeline import (
    classify_probs_per_frame,
    smooth_probs,
    suppress_short_runs,
    apply_outside_margin,
    get_fps_safe,
)

VIDEO = os.environ['VIDEO']
OUTDIR = os.environ['OUTDIR']
SAMPLE_STRIDE = int(float(os.environ.get('SAMPLE_STRIDE','1')))
BATCH_SIZE = int(float(os.environ.get('BATCH_SIZE','64')))
DEVICE = os.environ.get('DEVICE') or None
SMOOTH_SEC = float(os.environ.get('SMOOTH_SEC','0.3'))
MIN_INSIDE_SEC = float(os.environ.get('MIN_INSIDE_SEC','0.5'))
OUTSIDE_MARGIN_SEC = float(os.environ.get('OUTSIDE_MARGIN_SEC','0.2'))

print(f"[Stage1] CLIP classify: {VIDEO}")
probs_full, probs_sampled, fps, total_frames = classify_probs_per_frame(
    VIDEO, batch_size=BATCH_SIZE, device_preference=DEVICE, sample_stride=SAMPLE_STRIDE
)
print(f"[Stage1] Frames={total_frames}, FPS={fps:.2f}")
# Save debug
with open(os.path.join(OUTDIR, 'clip_probs.csv'), 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['sample_index','frame_index','time_sec','prob_inside'])
    for k,p in enumerate(probs_sampled):
        fidx = k * SAMPLE_STRIDE; t = fidx / fps if fps>0 else 0
        w.writerow([k, fidx, f"{t:.3f}", f"{p:.6f}"])

# Stage1 filtering (smoothing + min duration + margin)
if SMOOTH_SEC > 0:
    wf = max(1, int(round(SMOOTH_SEC * fps)))
    base_vals = smooth_probs([1.0 if p >= 0.5 else 0.0 for p in probs_full], wf)
else:
    base_vals = [1.0 if p >= 0.5 else 0.0 for p in probs_full]
mask = [v >= 0.5 for v in base_vals]
if MIN_INSIDE_SEC > 0:
    min_inside_frames = max(1, int(round(MIN_INSIDE_SEC * fps)))
    mask = suppress_short_runs(mask, min_inside_frames)
if OUTSIDE_MARGIN_SEC > 0:
    margin_frames = max(0, int(round(OUTSIDE_MARGIN_SEC * fps)))
    mask = apply_outside_margin(mask, margin_frames)
kept = sum(1 for x in mask if x)
print(f"[Stage1] Kept frames: {kept}/{len(mask)} ({(kept/max(1,len(mask)))*100:.1f}%)")
skip_marker = os.path.join(OUTDIR, 'SKIP_NO_FRAMES')
if os.path.exists(skip_marker):
    try:
        os.remove(skip_marker)
    except Exception:
        pass
if kept <= 0:
    reason = "[Stage1] No frames kept; skipping video."
    try:
        with open(skip_marker, 'w') as f:
            f.write(reason + "\n")
    except Exception as write_err:
        print(f"[Stage1] Warning: failed to write skip marker: {write_err}")
    skip_log_path = os.environ.get('SKIP_LOG')
    if skip_log_path:
        try:
            dir_name = os.path.dirname(skip_log_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            with open(skip_log_path, 'a') as log_f:
                log_f.write(f"{VIDEO}\tno_frames_kept\n")
        except Exception as log_err:
            print(f"[Stage1] Warning: failed to append skip log: {log_err}")
    print(reason)
    raise SystemExit(0)

# Write AVI (MJPG) to avoid MP4 decode issues across nodes
avi_path = os.path.join(OUTDIR, 'inside_only.avi')
print(f"[Stage1] Writing AVI -> {avi_path}")
cap = cv2.VideoCapture(VIDEO)
if not cap.isOpened():
    raise SystemExit(f"[Stage1] Failed to reopen input: {VIDEO}")
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)); h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
out = cv2.VideoWriter(avi_path, fourcc, fps, (w, h))
if not out.isOpened():
    cap.release(); raise SystemExit("[Stage1] AVI writer open failed")
idx = 0; written = 0
while True:
    ret, frame = cap.read()
    if not ret: break
    if idx < len(mask) and mask[idx]:
        out.write(frame); written += 1
    idx += 1
out.release(); cap.release()
print(f"[Stage1] AVI frames written: {written}")
PY

if [ -f "$OUTDIR/SKIP_NO_FRAMES" ]; then
  echo "[INFO] Stage1 found no usable frames; skipping segmentation."
  exit 0
fi

VIDEO_AVI="$OUTDIR/inside_only.avi"
if [ ! -f "$VIDEO_AVI" ]; then
  echo "[WARN] inside_only.avi not found; skipping segmentation."
  exit 0
fi

# Stage 2: Label-based segmentation on AVI (ffmpeg not required)
VIDEO_AVI="$OUTDIR/inside_only.avi"
SCENES_DIR="$OUTDIR/scenes"
mkdir -p "$SCENES_DIR"
VIDEO_AVI="$VIDEO_AVI" SCENES_DIR="$SCENES_DIR" TARGET_SEG_FPS="$TARGET_SEG_FPS" MIN_ROOM_SEC="$MIN_ROOM_SEC" \
SEGMENTATION_SCRIPT_DIR="$SCRIPT_DIR" \
python - <<'PY'
import os
import sys

script_dir = os.environ["SEGMENTATION_SCRIPT_DIR"]
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

from room_tour_pipeline import segment_by_labels

VIDEO_AVI = os.environ['VIDEO_AVI']
SCENES_DIR = os.environ['SCENES_DIR']
TARGET_SEG_FPS = float(os.environ.get('TARGET_SEG_FPS','5.0'))
MIN_ROOM_SEC = float(os.environ.get('MIN_ROOM_SEC','2.0'))
print(f"[Stage2] Label segmentation on {VIDEO_AVI} -> {SCENES_DIR}")
info = segment_by_labels(VIDEO_AVI, SCENES_DIR, target_fps=TARGET_SEG_FPS, min_room_sec=MIN_ROOM_SEC)
print(f"[Stage2] Wrote {info['segments']} labeled segments.")
PY

echo "Done. Outputs in $OUTDIR (inside_only.avi, scenes/)"

# ./scripts/run_label_segmentation.sh \
#   "/groups/gag51402/datasets/RoomTours/raw_videos/1st_download/Crisna_at_Remax/2OFrEwK6Wvc_For Sale ｜ WALKTHROUGH ｜ CRISNA ｜ RE⧸MAX AMANZIMTOTI.mp4" \
#   "scripts/test/2OFrEwK6Wvc"
