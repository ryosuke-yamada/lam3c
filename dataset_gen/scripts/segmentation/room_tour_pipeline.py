#!/usr/bin/env python3
import argparse
import os
import csv
import math
from typing import List, Tuple, Optional

import cv2
import numpy as np


def try_import_clip():
    try:
        import clip  # provided by clip-anytorch
        import torch
        from PIL import Image
        return clip, torch, Image
    except Exception as e:
        raise RuntimeError(
            "CLIP (clip-anytorch) and its dependencies are required.\n"
            "Install: pip install clip-anytorch pillow torch torchvision torchaudio (cpu or cuda wheels)\n"
            f"Original import error: {e}"
        )


def try_import_scenedetect():
    try:
        from scenedetect import open_video, SceneManager
        from scenedetect.detectors import ContentDetector
        from scenedetect.video_splitter import split_video_ffmpeg
        return open_video, SceneManager, ContentDetector, split_video_ffmpeg
    except Exception:
        return None, None, None, None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_fps_safe(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    return float(fps)


def get_default_room_labels() -> list:
    """Return default (label, prompts) tuples for room classification."""
    return [
        ("living_room", [
            "a wide angle living room interior",
            "a modern living room interior",
            "a cozy living room",
            "living room interior photo",
        ]),
        ("bedroom", [
            "a bedroom interior",
            "a cozy bedroom interior",
            "bedroom interior with a bed",
        ]),
        ("bathroom", [
            "a bathroom interior",
            "modern bathroom interior",
            "bathroom with shower and sink",
        ]),
    ]


def resolve_device(device_preference, torch_module, context=''):
    """Resolve torch.device with graceful CUDA fallback.

    Args:
        device_preference: str or None
        torch_module: torch module
        context: string prefix for warnings
    Returns:
        torch.device
    """
    if device_preference:
        pref = str(device_preference).strip().lower()
        if pref in ('cuda', 'gpu') or pref.startswith('cuda:'):
            if torch_module.cuda.is_available():
                return torch_module.device('cuda')
            warning = '[WARN]' + (context + ' ' if context else ' ') + 'CUDA requested but not available; falling back to CPU.'
            print(warning.strip())
            return torch_module.device('cpu')
        try:
            return torch_module.device(pref)
        except Exception:
            warning = '[WARN]' + (context + ' ' if context else ' ') + f"Unknown device '{device_preference}', falling back to CPU."
            print(warning.strip())
            return torch_module.device('cpu')
    if torch_module.cuda.is_available():
        return torch_module.device('cuda')
    return torch_module.device('cpu')



def classify_probs_per_frame(
    video_path: str,
    batch_size: int = 64,
    device_preference: Optional[str] = None,
    sample_stride: int = 1,
    text_prompts_inside: Optional[List[str]] = None,
    text_prompts_outside: Optional[List[str]] = None,
) -> Tuple[List[float], List[float], float, int]:
    """Return per-frame CLIP probabilities for being inside."""

    clip, torch, Image = try_import_clip()

    if text_prompts_inside is None:
        text_prompts_inside = [
            "indoor scene",
        ]
    if text_prompts_outside is None:
        text_prompts_outside = [
            "outdoor scene",
        ]

    model, preprocess = clip.load("ViT-B/32", device="cpu")
    device = resolve_device(device_preference, torch, context="[CLIP classify]")
    model = model.to(device)
    model.eval()

    with torch.no_grad():
        txt_in = clip.tokenize(text_prompts_inside).to(device)
        txt_out = clip.tokenize(text_prompts_outside).to(device)
        emb_in = model.encode_text(txt_in)
        emb_out = model.encode_text(txt_out)
        emb_in = emb_in / emb_in.norm(dim=-1, keepdim=True)
        emb_out = emb_out / emb_out.norm(dim=-1, keepdim=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        alt = os.path.splitext(video_path)[0] + ".avi"
        if os.path.exists(alt):
            cap = cv2.VideoCapture(alt)
            if cap.isOpened():
                print(f"[INFO] Using AVI fallback for segmentation: {alt}")
                video_path = alt
            else:
                raise RuntimeError(f"Failed to open video: {video_path} and {alt}")
        else:
            raise RuntimeError(f"Failed to open video: {video_path}")

    fps = get_fps_safe(cap)

    probs_sampled: List[float] = []
    frame_idx = 0
    batch_imgs = []

    def flush_batch():
        nonlocal probs_sampled
        if not batch_imgs:
            return
        with torch.no_grad():
            imgs_tensor = torch.stack(batch_imgs).to(device)
            img_emb = model.encode_image(imgs_tensor)
            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            sim_in = (img_emb @ emb_in.t()).max(dim=1).values
            sim_out = (img_emb @ emb_out.t()).max(dim=1).values
            sims = torch.stack([sim_in, sim_out], dim=1)
            probs = sims.softmax(dim=1)[:, 0]
            probs_sampled.extend(probs.tolist())
        batch_imgs.clear()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sample_stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = preprocess(Image.fromarray(rgb))
            batch_imgs.append(img)
            if len(batch_imgs) >= batch_size:
                flush_batch()
        frame_idx += 1
    flush_batch()
    cap.release()

    total_frames = frame_idx

    probs_full = [0.0] * total_frames
    if sample_stride <= 1:
        for i in range(min(total_frames, len(probs_sampled))):
            probs_full[i] = probs_sampled[i]
        if probs_sampled:
            for i in range(len(probs_sampled), total_frames):
                probs_full[i] = probs_sampled[-1]
    else:
        for k, p in enumerate(probs_sampled):
            start = k * sample_stride
            end = min(total_frames, (k + 1) * sample_stride)
            for i in range(start, end):
                probs_full[i] = p
        if probs_sampled:
            last_end = len(probs_sampled) * sample_stride
            for i in range(last_end, total_frames):
                probs_full[i] = probs_sampled[-1]

    return probs_full, probs_sampled, fps, total_frames


def smooth_probs(probs: List[float], window_frames: int) -> List[float]:
    if window_frames <= 1:
        return probs
    arr = np.asarray(probs, dtype=np.float32)
    kernel = np.ones(window_frames, dtype=np.float32) / float(window_frames)
    smoothed = np.convolve(arr, kernel, mode="same")
    return smoothed.tolist()



def suppress_short_runs(mask: List[bool], min_len: int) -> List[bool]:
    if min_len <= 1:
        return mask
    n = len(mask)
    out = mask[:]
    i = 0
    while i < n:
        if out[i]:
            j = i
            while j < n and out[j]:
                j += 1
            if (j - i) < min_len:
                for k in range(i, j):
                    out[k] = False
            i = j
        else:
            i += 1
    return out


def apply_outside_margin(keep_mask: List[bool], margin_frames: int) -> List[bool]:
    if margin_frames <= 0:
        return keep_mask
    n = len(keep_mask)
    out = keep_mask[:]
    for idx, keep in enumerate(keep_mask):
        if not keep:
            start = max(0, idx - margin_frames)
            end = min(n, idx + margin_frames + 1)
            for j in range(start, end):
                out[j] = False
    return out


def write_inside_video(src_path: str, dst_path: str, keep_mask: List[bool], fps: float) -> int:
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {src_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (w, h))
    if not out.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to create output video writer: {dst_path}")

def split_video_with_opencv_labels(video_path: str, output_dir: str, labeled_boundaries):
    """Split by (start_frame, end_frame, label) boundaries using OpenCV."""
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for splitting: {video_path}")
    fps = get_fps_safe(cap)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    def sanitize(label: str) -> str:
        return "".join(ch if (ch.isalnum() or ch in ("_","-")) else "_" for ch in label)
    writer = None; seg_idx = 0; fidx = 0
    def close_writer():
        nonlocal writer
        if writer is not None:
            writer.release(); writer=None
    while True:
        ret, frame = cap.read()
        if not ret: break
        while seg_idx < len(labeled_boundaries) and fidx >= labeled_boundaries[seg_idx][1]:
            close_writer(); seg_idx += 1
        if seg_idx < len(labeled_boundaries):
            s_f, e_f, lab = labeled_boundaries[seg_idx]
            if s_f <= fidx < e_f:
                if writer is None:
                    out_name = f"scene-{seg_idx+1:03d}_{sanitize(lab)}.mp4"
                    out_path = os.path.join(output_dir, out_name)
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
                    if not writer.isOpened():
                        cap.release(); raise RuntimeError(f"Failed to create scene writer: {out_path}")
                writer.write(frame)
        fidx += 1
    close_writer(); cap.release()
    return len(labeled_boundaries)


    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx < len(keep_mask) and keep_mask[frame_idx]:
            out.write(frame)
            written += 1
        frame_idx += 1

    out.release()
    cap.release()
    return written


def split_video_with_opencv(video_path: str, output_dir: str, scene_list) -> int:
    """Fallback splitter using OpenCV when ffmpeg is unavailable."""
    ensure_dir(output_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video for splitting: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0
    fps = float(fps)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    boundaries = []
    for start_tc, end_tc in scene_list:
        start_f = int(start_tc.get_frames())
        end_f = int(end_tc.get_frames())
        if end_f > start_f:
            boundaries.append((start_f, end_f))

    writer = None
    current_scene = 0
    frame_idx = 0
    total = len(boundaries)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        while current_scene < total and frame_idx >= boundaries[current_scene][1]:
            if writer is not None:
                writer.release()
                writer = None
            current_scene += 1

        if current_scene < total:
            start_f, end_f = boundaries[current_scene]
            if start_f <= frame_idx < end_f:
                if writer is None:
                    out_path = os.path.join(output_dir, f"scene-{current_scene + 1:03d}.mp4")
                    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
                    if not writer.isOpened():
                        cap.release()
                        raise RuntimeError(f"Failed to create scene writer: {out_path}")
                writer.write(frame)

        frame_idx += 1

    if writer is not None:
        writer.release()
    cap.release()
    return total


def run_pyscenedetect_split(video_path: str, output_dir: str, threshold: float = 30.0):
    open_video, SceneManager, ContentDetector, _ = try_import_scenedetect()
    if open_video is None:
        print("[WARN] PySceneDetect not available. Skipping scene split. Install with: pip install scenedetect")
        return []

    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))
    scene_manager.detect_scenes(video, show_progress=False)
    scene_list = scene_manager.get_scene_list()

    if not scene_list:
        print("[INFO] PySceneDetect found no scene cuts; skipping splitting step.")
        return []

    ensure_dir(output_dir)
    count = split_video_with_opencv(video_path, output_dir, scene_list)
    print(f"[INFO] OpenCV wrote {count} scene clips to {output_dir}")
    return [(int(start.get_frames()), int(end.get_frames())) for start, end in scene_list]


def write_debug_csv(csv_path: str, probs_sampled: List[float], sample_stride: int, fps: float):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_index", "frame_index", "time_sec", "prob_inside"])
        for idx, prob in enumerate(probs_sampled):
            frame_index = idx * sample_stride
            time_sec = frame_index / fps if fps > 0 else 0
            writer.writerow([idx, frame_index, f"{time_sec:.3f}", f"{prob:.6f}"])


def segment_by_labels(video_path: str, output_dir: str, target_fps: float = 5.0, min_room_sec: float = 2.0, batch_size: int = 64, device_preference: str = None):
    """Segment inside_only video by CLIP room labels and write labeled clips."""
    clip, torch, Image = try_import_clip()
    labels_prompts = get_default_room_labels()
    label_names = [lp[0] for lp in labels_prompts]
    # Flatten prompts and keep slices per label
    prompt_list = []
    slices = []
    start = 0
    for _, prompts in labels_prompts:
        prompt_list.extend(prompts)
        end = start + len(prompts)
        slices.append((start, end))
        start = end
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    device = resolve_device(device_preference, torch, context="[CLIP labels]")
    model = model.to(device).eval()
    with torch.no_grad():
        toks = clip.tokenize(prompt_list).to(device)
        txt_emb = model.encode_text(toks)
        txt_emb = txt_emb / txt_emb.norm(dim=-1, keepdim=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        alt = os.path.splitext(video_path)[0] + ".avi"
        if os.path.exists(alt):
            cap = cv2.VideoCapture(alt)
            if cap.isOpened():
                print(f"[INFO] Using AVI fallback for segmentation: {alt}")
                video_path = alt
            else:
                raise RuntimeError(f"Failed to open video: {video_path} and {alt}")
        else:
            raise RuntimeError(f"Failed to open video: {video_path}")
    fps = get_fps_safe(cap)
    stride = max(1, int(round(fps / target_fps)))
    frames_idx = []
    pred_indices = []
    batch_imgs = []
    def flush():
        nonlocal pred_indices
        if not batch_imgs: return
        with torch.no_grad():
            x = torch.stack(batch_imgs).to(device)
            im_emb = model.encode_image(x); im_emb = im_emb / im_emb.norm(dim=-1, keepdim=True)
            sim = im_emb @ txt_emb.t()
            import numpy as _np
            for row in sim.cpu().numpy():
                scores = []
                for s0, s1 in slices:
                    scores.append(float(row[s0:s1].max() if s1 > s0 else -1e9))
                arr = _np.array(scores)
                arr = arr - arr.max(); p = _np.exp(arr); p = p / (p.sum() + 1e-9)
                pred_indices.append(int(p.argmax()))
        batch_imgs.clear()
    fi = 0; B = batch_size
    while True:
        ret, frame = cap.read()
        if not ret: break
        if fi % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = preprocess(Image.fromarray(rgb))
            batch_imgs.append(img); frames_idx.append(fi)
            if len(batch_imgs) >= B: flush()
        fi += 1
    flush(); total_frames = fi; cap.release()
    # Build runs
    runs = []
    if pred_indices:
        s0 = 0
        for i in range(1, len(pred_indices) + 1):
            if i == len(pred_indices) or pred_indices[i] != pred_indices[s0]:
                runs.append((s0, i, pred_indices[s0]))
                s0 = i
    # Merge only very short runs to suppress jitter
    min_merge_sec = min(2.0, max(0.5, min_room_sec / 5.0))
    min_len_samples = max(1, int(round(min_merge_sec * (fps / stride))))
    merged = []
    for rs, re, lab in runs:
        if (re - rs) < min_len_samples and merged:
            ps, pe, pl = merged[-1]
            merged[-1] = (ps, re, pl)
        else:
            merged.append((rs, re, lab))
    # Collapse adjacent same labels
    collapsed = []
    for seg in merged:
        if not collapsed:
            collapsed.append(seg)
        else:
            ps, pe, pl = collapsed[-1]
            cs, ce, cl = seg
            if pl == cl and cs <= pe:
                collapsed[-1] = (ps, ce, pl)
            else:
                collapsed.append(seg)
    # Frame boundaries with labels, enforcing minimum segment duration
    segments = []
    for rs, re, lab in collapsed:
        if rs >= len(frames_idx):
            continue
        start_f = frames_idx[rs]
        end_index = min(re, len(frames_idx)) - 1
        if end_index < rs:
            continue
        end_f = frames_idx[end_index] + stride
        end_f = min(end_f, total_frames)
        segments.append([start_f, end_f, lab])

    if not segments and label_names:
        segments.append([0, total_frames, 0])

    if segments:
        segments[0][0] = 0
        segments[-1][1] = total_frames

    min_room_frames = max(1, int(round(min_room_sec * fps)))

    def duration(seg):
        return seg[1] - seg[0]

    changed = True
    while changed and len(segments) > 1:
        changed = False
        for idx, seg in enumerate(segments):
            if duration(seg) >= min_room_frames:
                continue
            left_exists = idx > 0
            right_exists = idx < len(segments) - 1
            if not left_exists and not right_exists:
                break
            # Prefer merging with same-label neighbor if available
            if left_exists and segments[idx - 1][2] == seg[2]:
                segments[idx - 1][1] = seg[1]
                segments.pop(idx)
                changed = True
                break
            if right_exists and segments[idx + 1][2] == seg[2]:
                segments[idx + 1][0] = seg[0]
                segments.pop(idx)
                changed = True
                break
            # Choose neighbor with larger duration after merge
            if left_exists and right_exists:
                left_dur = duration(segments[idx - 1])
                right_dur = duration(segments[idx + 1])
                if right_dur > left_dur:
                    segments[idx + 1][0] = seg[0]
                    segments.pop(idx)
                else:
                    segments[idx - 1][1] = seg[1]
                    segments.pop(idx)
            elif left_exists:
                segments[idx - 1][1] = seg[1]
                segments.pop(idx)
            else:
                segments[idx + 1][0] = seg[0]
                segments.pop(idx)
            changed = True
            break

    labeled = []
    for start_f, end_f, lab in segments:
        start = max(0, int(start_f))
        end = min(total_frames, int(max(start + 1, end_f)))
        label_idx = lab if 0 <= lab < len(label_names) else 0
        labeled.append((start, end, label_names[label_idx]))

    ensure_dir(output_dir)
    count = split_video_with_opencv_labels(video_path, output_dir, labeled)
    return {"segments": count, "labels": label_names}


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Room tour cleaner: CLIP filter + segmentation (scenedetect/label)")
    parser.add_argument("input", help="Input video path")
    parser.add_argument("output_dir", help="Output directory for results")
    parser.add_argument("--sample-stride", type=int, default=1, help="Process every Nth frame (default: 1)")
    parser.add_argument("--batch-size", type=int, default=64, help="CLIP batch size")
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g., cuda or cpu")
    parser.add_argument("--smooth-sec", type=float, default=0.5, help="Moving-average window in seconds")
    parser.add_argument("--min-inside-sec", type=float, default=10.0, help="Drop inside runs shorter than this duration")
    parser.add_argument("--outside-margin-sec", type=float, default=1.0, help="Also drop this many seconds around outside frames")
    parser.add_argument("--scenedetect-threshold", type=float, default=30.0, help="PySceneDetect ContentDetector threshold")
    parser.add_argument("--segmentation", choices=["scenedetect","label","none"], default="scenedetect", help="Segmentation method for inside_only video")
    parser.add_argument("--min-room-sec", type=float, default=10.0, help="Minimum duration for a room segment when using label segmentation")
    parser.add_argument("--target-seg-fps", type=float, default=5.0, help="Sampling fps for label segmentation")
    parser.add_argument("--skip-splitting", action="store_true", help="Skip scene splitting step")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    input_path = args.input
    output_dir = args.output_dir
    ensure_dir(output_dir)

    print(f"Input: {input_path}")
    print(f"Output dir: {output_dir}")

    print("[1/5] CLIP classification (per-frame probabilities)...")
    probs_full, probs_sampled, fps, total_frames = classify_probs_per_frame(
        input_path,
        batch_size=args.batch_size,
        device_preference=args.device,
        sample_stride=args.sample_stride,
    )
    print(f"Frames: {total_frames}, FPS: {fps:.2f}")

    write_debug_csv(os.path.join(output_dir, "clip_probs.csv"), probs_sampled, args.sample_stride, fps)

    print("[2/5] Smoothing indoor/outdoor probabilities...")
    base_mask = [prob >= 0.5 for prob in probs_full]
    base_values = [1.0 if m else 0.0 for m in base_mask]

    if args.smooth_sec > 0:
        window_frames = max(1, int(round(args.smooth_sec * fps)))
        probs_smooth = smooth_probs(base_values, window_frames)
    else:
        probs_smooth = base_values

    keep_mask = [p >= 0.5 for p in probs_smooth]

    print("[3/5] Removing short inside runs...")
    if args.min_inside_sec > 0:
        min_inside_frames = max(1, int(round(args.min_inside_sec * fps)))
        keep_mask = suppress_short_runs(keep_mask, min_inside_frames)

    print("[4/5] Applying outside margin...")
    if args.outside_margin_sec > 0:
        margin_frames = max(0, int(round(args.outside_margin_sec * fps)))
        keep_mask = apply_outside_margin(keep_mask, margin_frames)
    kept = sum(1 for keep in keep_mask if keep)
    print(f"Kept frames after filtering: {kept}/{len(keep_mask)} ({(kept / max(1, len(keep_mask))) * 100:.1f}%)")
    if kept <= 0:
        print("[WARN] No frames remain after filtering. Exiting.")
        return 1

    inside_video = os.path.join(output_dir, "inside_only.mp4")
    print(f"[5/5] Writing inside-only video -> {inside_video}")
    _ = write_inside_video(input_path, inside_video, keep_mask, fps)
    # Validate file exists; if not, fallback to AVI container (MJPG)
    try:
        size_ok = os.path.exists(inside_video) and os.path.getsize(inside_video) > 0
    except Exception:
        size_ok = False
    if not size_ok:
        print("[WARN] inside_only.mp4 missing or empty; rewriting as AVI fallback.")
        alt_path = os.path.join(output_dir, "inside_only.avi")
        cap2 = cv2.VideoCapture(input_path)
        if not cap2.isOpened():
            print("[ERROR] Cannot reopen input for AVI fallback.")
            return 1
        w2 = int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)); h2 = int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc2 = cv2.VideoWriter_fourcc(*"MJPG")
        out2 = cv2.VideoWriter(alt_path, fourcc2, fps, (w2, h2))
        if not out2.isOpened():
            print("[ERROR] AVI fallback writer open failed.")
            cap2.release()
            return 1
        idx2 = 0
        while True:
            ret2, fr2 = cap2.read()
            if not ret2: break
            if idx2 < len(keep_mask) and keep_mask[idx2]:
                out2.write(fr2)
            idx2 += 1
        out2.release(); cap2.release()
        if os.path.exists(alt_path) and os.path.getsize(alt_path) > 0:
            inside_video = alt_path
            print(f"[INFO] Using AVI fallback at {alt_path}")
        else:
            print("[ERROR] AVI fallback also failed.")
            return 1
    print(f"Inside-only video frames written: {kept}")

    if not args.skip_splitting:
        scenes_dir = os.path.join(output_dir, "scenes")
        if args.segmentation == "label":
            print(f"Scene segmentation by room labels -> {scenes_dir}")
            try:
                info = segment_by_labels(
                    inside_video,
                    scenes_dir,
                    target_fps=args.target_seg_fps,
                    min_room_sec=args.min_room_sec,
                    batch_size=args.batch_size,
                    device_preference=args.device,
                )
                print(f"[INFO] Wrote {info['segments']} labeled segments.")
            except Exception as exc:
                print(f"[WARN] Label-based segmentation failed: {exc}")
        elif args.segmentation == "scenedetect":
            print(f"Scene splitting with PySceneDetect -> {scenes_dir}")
            try:
                run_pyscenedetect_split(inside_video, scenes_dir, threshold=args.scenedetect_threshold)
            except Exception as exc:
                print(f"[WARN] Scene splitting failed: {exc}")
        else:
            print("Segmentation skipped by option.")
    else:
        print("Scene splitting skipped.")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

