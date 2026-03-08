import torch
import argparse
import os
import gc
import time
import json
from pathlib import Path
import sys
from typing import List, Tuple, Dict, Optional
import multiprocessing as mp

VENDORED_PI3_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Pi3"
if str(VENDORED_PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_PI3_ROOT))

from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3
import cv2
import math
from PIL import Image
from torchvision import transforms

def load_video_as_tensor_with_skip(
    video_path,
    interval=1,
    skip_seconds=0,
    PIXEL_LIMIT=255000,
    adjust_for_high_fps=False,
    target_frames=1500,
):
    """Load and resize video frames with optional front skipping and interval adjustment."""
    sources = []
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")
    
    # 動画の基本情報を取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if skip_seconds > 0:
        skip_frames = int(fps * skip_seconds)
        available_frames = total_frames - skip_frames
        print(f"Loading frames from video: {video_path} (skipping first {skip_seconds} seconds)")
    else:
        skip_frames = 0
        available_frames = total_frames
        print(f"Loading frames from video: {video_path} (no skipping)")

    original_interval = interval

    if adjust_for_high_fps and target_frames > 0:
        if available_frames > target_frames:
            optimal_interval = max(1, int(round(available_frames / target_frames)))

            print(f"Video info: FPS={fps:.2f}, Total frames={total_frames}, Available frames={available_frames}")
            print(f"Target frames: {target_frames}, Optimal interval: {optimal_interval}")
            print(f"Adjusting interval from {original_interval} to {optimal_interval} for target frame count")
            interval = optimal_interval
        else:
            print(f"Video info: FPS={fps:.2f}, Total frames={total_frames}, Available frames={available_frames}")
            print(f"Using all {available_frames} frames (≤ {target_frames}), interval: {interval}")
    else:
        print(f"Video FPS: {fps:.2f}, using original interval: {interval}")
    
    if skip_frames > 0:
        print(f"Skipping first {skip_frames} frames, then sampling every {interval} frames")
    else:
        print(f"Sampling every {interval} frames from start")
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 最初のskip_frames分をスキップ
        if frame_idx < skip_frames:
            frame_idx += 1
            continue
        
        # intervalに従ってフレームを選択
        if (frame_idx - skip_frames) % interval == 0:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            sources.append(Image.fromarray(rgb_frame))
        
        frame_idx += 1
    
    cap.release()
    
    if not sources:
        print("No frames found or loaded after skipping")
        return torch.empty(0)

    print(f"Found {len(sources)} frames after skipping. Processing...")

    # 最初の画像から統一サイズを決定
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")

    # 画像をテンソルに変換
    tensor_list = []
    to_tensor_transform = transforms.ToTensor()
    
    for img_pil in sources:
        try:
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return torch.empty(0)

    return torch.stack(tensor_list, dim=0)


def get_video_duration_seconds(video_path: str) -> float:
    """Return video duration in seconds (0 when unavailable)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[WARN] Could not open video for duration: {video_path}")
        return 0.0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    finally:
        cap.release()

    fps = float(fps) if fps and fps > 0 else 0.0
    frame_count = float(frame_count) if frame_count and frame_count > 0 else 0.0

    if fps > 0:
        return frame_count / fps if frame_count > 0 else 0.0
    return frame_count


def roomtours_video_key(name: str) -> str:
    text = str(name or "")
    if "_" in text:
        return text.split("_", 1)[0]
    return text


def normalize_roomtours_name(name: str) -> str:
    return str(name or "").replace(" ", "_")


def build_roomtours_video_index(base_path: Path) -> Dict[str, List[Path]]:
    index: Dict[str, List[Path]] = {}
    if not base_path.exists():
        return index
    for entry in base_path.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        key = roomtours_video_key(entry.name)
        index.setdefault(key, []).append(entry)
    return index


def get_roomtours_video_dirs(video_id: Optional[str], base_path: Path, index: Dict[str, List[Path]]) -> List[Path]:
    if not video_id:
        return []

    vid = str(video_id)
    candidates: List[Path] = []

    direct = base_path / vid
    if direct.is_dir():
        candidates.append(direct)

    key = roomtours_video_key(vid)
    for cand in index.get(key, []):
        if cand not in candidates:
            candidates.append(cand)

    target_norm = normalize_roomtours_name(vid)
    candidates.sort(key=lambda p: (normalize_roomtours_name(p.name) != target_norm, p.name))
    return candidates


def load_roomtours_entries_from_json(
    scene_json_path: str,
    input_base: str,
    output_base: str,
    include_processed: bool = False,
) -> List[Tuple[str, str]]:
    """Load explicit RoomTours scene targets from a JSON list.

    Supported per-entry keys:
      - input_path / scene_mp4
      - output_path / source_ply
      - video_id + scene_dir (or scene_name)
    """
    json_path = Path(scene_json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"RoomTours scene JSON not found: {scene_json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)

    if not isinstance(payload, list):
        raise ValueError(f"RoomTours scene JSON must be a list, got: {type(payload).__name__}")

    input_base_path = Path(input_base)
    output_base_path = Path(output_base)
    input_video_index = build_roomtours_video_index(input_base_path)
    output_video_index = build_roomtours_video_index(output_base_path)
    entries: List[Tuple[str, str]] = []
    seen_pairs = set()
    skipped_unresolved = 0
    skipped_duplicates = 0
    skipped_exists = 0

    for idx, item in enumerate(payload):
        if not isinstance(item, dict):
            skipped_unresolved += 1
            continue

        video_id = item.get("video_id")
        scene_dir = item.get("scene_dir") or item.get("scene_name")
        source_ply = item.get("source_ply")
        input_path = item.get("input_path") or item.get("scene_mp4")
        output_path = item.get("output_path")

        # Fallback: recover video/scene from source_ply path
        if source_ply and (not video_id or not scene_dir):
            source_path = Path(str(source_ply))
            if source_path.name == "pi3.ply" and len(source_path.parts) >= 3:
                if not scene_dir:
                    scene_dir = source_path.parent.name
                if not video_id:
                    video_id = source_path.parent.parent.name

        scene_file = None
        scene_stem = None
        if scene_dir:
            scene_file = str(scene_dir)
            if not scene_file.endswith(".mp4"):
                scene_file = f"{scene_file}.mp4"
            scene_stem = scene_file[:-4] if scene_file.endswith(".mp4") else scene_file

        if input_path is None and video_id and scene_file:
            input_path = str(input_base_path / str(video_id) / "scenes" / scene_file)

        if scene_file and video_id:
            needs_input_resolve = (not input_path) or (not Path(str(input_path)).exists())
            if needs_input_resolve:
                input_candidates = get_roomtours_video_dirs(video_id, input_base_path, input_video_index)
                resolved_input = None
                for cand in input_candidates:
                    cand_input = cand / "scenes" / scene_file
                    if cand_input.exists():
                        resolved_input = cand_input
                        break
                if resolved_input is None and input_candidates:
                    resolved_input = input_candidates[0] / "scenes" / scene_file
                if resolved_input is not None:
                    input_path = str(resolved_input)

        if output_path is None and source_ply:
            output_path = str(source_ply)
        if output_path is None and video_id and scene_stem:
            output_path = str(output_base_path / str(video_id) / scene_stem / "pi3.ply")

        if scene_stem and video_id:
            needs_output_resolve = (not output_path) or (not Path(str(output_path)).exists())
            if needs_output_resolve:
                output_candidates = get_roomtours_video_dirs(video_id, output_base_path, output_video_index)
                resolved_output = None
                for cand in output_candidates:
                    cand_output = cand / scene_stem / "pi3.ply"
                    if cand_output.exists():
                        resolved_output = cand_output
                        break
                if resolved_output is None and output_candidates:
                    resolved_output = output_candidates[0] / scene_stem / "pi3.ply"
                if resolved_output is not None:
                    output_path = str(resolved_output)

        if not input_path or not output_path:
            skipped_unresolved += 1
            continue

        pair = (str(Path(input_path)), str(Path(output_path)))
        if pair in seen_pairs:
            skipped_duplicates += 1
            continue
        seen_pairs.add(pair)

        if not include_processed and Path(pair[1]).exists():
            skipped_exists += 1
            continue

        entries.append(pair)

    print(
        f"[INFO] RoomTours JSON entries loaded: {len(entries)} "
        f"(unresolved={skipped_unresolved}, duplicate={skipped_duplicates}, exists={skipped_exists})"
    )
    if entries:
        missing_inputs = sum(1 for input_path, _ in entries if not Path(input_path).exists())
        if missing_inputs > 0:
            print(f"[WARN] RoomTours JSON contains {missing_inputs} entries with missing input mp4 files")

    return entries



def load_images_as_tensor_with_range(directory: str, interval: int = 1, start_idx: int = 0, end_idx: int = None, PIXEL_LIMIT: int = 255000, return_meta: bool = False) -> torch.Tensor:
    """指定範囲の画像を読み込んでテンソルに変換（チャンク処理用）"""
    sources = []
    paths_meta = []
    original_sizes = []
    resized_size = None
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']

    # 画像ファイルを取得してソート
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(directory).glob(ext))
    image_files.sort()

    if end_idx is None:
        end_idx = len(image_files)

    # 指定範囲内の画像のみを選択
    selected_files = image_files[start_idx:end_idx]

    print(f"Loading images from directory: {directory}")
    print(f"Found {len(image_files)} total images, processing range {start_idx}-{end_idx-1} ({len(selected_files)} images)")

    if not selected_files:
        return (torch.empty(0), {}) if return_meta else torch.empty(0)

    # intervalに従って画像を選択
    for i, img_path in enumerate(selected_files):
        if i % interval == 0:
            try:
                img = Image.open(img_path).convert('RGB')
                sources.append(img)
                if return_meta:
                    paths_meta.append(str(img_path.resolve()))
                    original_sizes.append((img.height, img.width))
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    if not sources:
        print("No images were successfully loaded")
        return (torch.empty(0), {}) if return_meta else torch.empty(0)

    print(f"Found {len(sources)} images/frames after interval selection. Processing...")

    # 最初の画像から統一サイズを決定
    first_img = sources[0]
    W_orig, H_orig = first_img.size
    scale = math.sqrt(PIXEL_LIMIT / (W_orig * H_orig)) if W_orig * H_orig > 0 else 1
    W_target, H_target = W_orig * scale, H_orig * scale
    k, m = round(W_target / 14), round(H_target / 14)
    while (k * 14) * (m * 14) > PIXEL_LIMIT:
        if k / m > W_target / H_target:
            k -= 1
        else:
            m -= 1
    TARGET_W, TARGET_H = max(1, k) * 14, max(1, m) * 14
    print(f"All images will be resized to a uniform size: ({TARGET_W}, {TARGET_H})")
    if return_meta:
        resized_size = (TARGET_H, TARGET_W)

    # 画像をテンソルに変換
    tensor_list = []
    to_tensor_transform = transforms.ToTensor()

    for img_pil in sources:
        try:
            resized_img = img_pil.resize((TARGET_W, TARGET_H), Image.Resampling.LANCZOS)
            img_tensor = to_tensor_transform(resized_img)
            tensor_list.append(img_tensor)
        except Exception as e:
            print(f"Error processing an image: {e}")

    if not tensor_list:
        print("No images were successfully processed.")
        return (torch.empty(0), {}) if return_meta else torch.empty(0)

    stacked = torch.stack(tensor_list, dim=0)

    if not return_meta:
        return stacked

    metadata = {
        'paths': paths_meta,
        'original_sizes': original_sizes,
        'resized_size': resized_size,
        'interval': interval,
        'pixel_limit': PIXEL_LIMIT,
        'source': str(Path(directory).resolve()),
        'start_idx': start_idx,
        'end_idx': end_idx,
    }

    return stacked, metadata




class Pi3BatchProcessor:
    def __init__(
        self,
        device: str = 'cuda',
        ckpt: str = None,
        save_correspondence: bool = False,
        correspondence_suffix: str = '_correspondence.pt',
        pixel_limit: int = 255000,
        overwrite_existing: bool = False,
    ):
        self.device = torch.device(device)
        self.model = None
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
        self.save_correspondence = save_correspondence
        self.correspondence_suffix = correspondence_suffix
        self.confidence_threshold = 0.1
        # Resize target pixel budget per frame
        self.pixel_limit = int(pixel_limit)
        self.overwrite_existing = bool(overwrite_existing)
        self._load_model(ckpt)

    def _load_model(self, ckpt: str = None):
        """モデルを一度だけロード"""
        print(f"[GPU {self.device}] Loading model...")

        if ckpt is not None:
            self.model = Pi3().to(self.device).eval()
            if ckpt.endswith('.safetensors'):
                from safetensors.torch import load_file
                weight = load_file(ckpt)
            else:
                weight = torch.load(ckpt, map_location=self.device, weights_only=False)
            self.model.load_state_dict(weight)
        else:
            self.model = Pi3.from_pretrained("yyfz233/Pi3").to(self.device).eval()

        print(f"[GPU {self.device}] Model loaded successfully!")

    def _clear_gpu_memory(self):
        """GPUメモリを明示的にクリア"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        if torch.cuda.is_available():
            try:
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                print(f"Memory after clear - Allocated: {allocated/1024**3:.2f}GB, Reserved: {reserved/1024**3:.2f}GB")
            except RuntimeError as e:
                if "ECC error" in str(e):
                    print(f"Warning: ECC error during memory check, continuing...")
                else:
                    print(f"Warning: Error during memory check: {e}")

    def process_single_directory(
        self,
        input_path: str,
        output_path: str,
        interval: int = 1,
        processing_config: Optional[Dict[str, object]] = None,
    ) -> str:
        """Process a single scene video or image directory."""
        try:
            # 入力チェック
            if not os.path.exists(input_path):
                print(f"[GPU {self.device}] SKIP: Input not found: {input_path}")
                return "skip_no_input"

            # 出力チェック
            if os.path.exists(output_path):
                if not self.overwrite_existing:
                    print(f"[GPU {self.device}] SKIP: Output file already exists: {output_path}")
                    return "skip_exists"
                print(f"[GPU {self.device}] OVERWRITE: Existing output will be replaced: {output_path}")
                if os.path.isfile(output_path):
                    try:
                        os.remove(output_path)
                    except Exception as e:
                        print(f"[GPU {self.device}] WARN: Failed to remove existing output before overwrite: {e}")

            print(f"[GPU {self.device}] Processing: {input_path} -> {output_path}")

            effective_interval = interval
            image_meta = None

            processing_config = processing_config or {}
            video_skip_seconds = int(processing_config.get("video_skip_seconds", 0) or 0)
            video_target_frames = int(processing_config.get("video_target_frames", 1500) or 0)
            video_adjust_for_high_fps = bool(processing_config.get("video_adjust_for_high_fps", False))
            image_target_frames = int(processing_config.get("image_target_frames", 1500) or 0)

            if input_path.endswith('.mp4'):
                print(f"[GPU {self.device}] Processing video file")
                if video_target_frames > 0:
                    print(f"[GPU {self.device}] Target frames capped at {video_target_frames} for video sampling")
                imgs = load_video_as_tensor_with_skip(
                    input_path,
                    interval=interval,
                    skip_seconds=video_skip_seconds,
                    adjust_for_high_fps=video_adjust_for_high_fps,
                    target_frames=video_target_frames,
                ).to(self.device)
                if self.save_correspondence:
                    image_meta = {
                        'paths': None,
                        'original_sizes': None,
                        'resized_size': None,
                        'interval': interval,
                        'pixel_limit': None,
                        'source': input_path,
                    }
            else:
                image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
                image_files = []
                for ext in image_extensions:
                    image_files.extend(Path(input_path).glob(ext))

                if not image_files:
                    print(f"[GPU {self.device}] SKIP: No image files found in: {input_path}")
                    return "skip_no_images"

                total_images = len(image_files)
                loader_return = None

                if image_target_frames > 0 and total_images > image_target_frames:
                    optimal_interval = max(1, int(round(total_images / image_target_frames)))
                    print(
                        f"[GPU {self.device}] Found {total_images} images, adjusting interval "
                        f"from {effective_interval} to {optimal_interval} for target count {image_target_frames}"
                    )
                    effective_interval = optimal_interval
                    loader_return = load_images_as_tensor(
                        input_path,
                        interval=effective_interval,
                        PIXEL_LIMIT=self.pixel_limit,
                        return_meta=self.save_correspondence,
                    )
                else:
                    print(
                        f"[GPU {self.device}] Found {total_images} images, using all images "
                        f"(≤ {image_target_frames or total_images}), interval: {effective_interval}"
                    )
                    loader_return = load_images_as_tensor(
                        input_path,
                        interval=effective_interval,
                        PIXEL_LIMIT=self.pixel_limit,
                        return_meta=self.save_correspondence,
                    )

                if self.save_correspondence:
                    imgs, image_meta = loader_return
                    if image_meta is not None:
                        image_meta = dict(image_meta)
                        image_meta['interval'] = effective_interval
                        image_meta.setdefault('source', input_path)
                else:
                    imgs = loader_return

                imgs = imgs.to(self.device)

            if imgs.numel() == 0:
                print(f"[GPU {self.device}] SKIP: No valid frames/images loaded")
                return "skip_no_images"

            print(f"[GPU {self.device}] Loaded {imgs.shape[0]} frames/images")

            # 推論実行（チャンク対応）
            print(f"[GPU {self.device}] Running model inference...")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            def run_model_on_tensor(t: torch.Tensor):
                with torch.no_grad():
                    with torch.amp.autocast('cuda', dtype=self.dtype):
                        return self.model(t[None])

            total_frames = int(imgs.shape[0])
            chunk_size = total_frames

            all_points_cpu = []
            all_colors_cpu = []

            if chunk_size >= total_frames:
                # Single pass
                try:
                    res = run_model_on_tensor(imgs)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"[GPU {self.device}] ERROR: CUDA OOM in single-pass. Consider reducing --pixel_limit. {e}")
                    self._clear_gpu_memory()
                    return "error_oom"

                conf_scores = torch.sigmoid(res['conf'][..., 0])
                conf_mask = conf_scores > self.confidence_threshold
                non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
                combined_mask = torch.logical_and(conf_mask, non_edge)
                masks_gpu = combined_mask[0]
                masks_cpu = masks_gpu.cpu()

                num_points = masks_gpu.sum().item()
                if num_points == 0:
                    print(f"[GPU {self.device}] SKIP: No valid points after masking: {input_path}")
                    del imgs, res, masks_gpu, masks_cpu, conf_scores, conf_mask, non_edge, combined_mask
                    self._clear_gpu_memory()
                    return "skip_no_points"

                color_tensor = imgs
                if self.save_correspondence:
                    color_tensor_cpu = imgs.detach().cpu()
                    color_tensor = color_tensor_cpu
                points_selected = res['points'][0][masks_gpu].detach().cpu()
                if self.save_correspondence:
                    colors_selected = color_tensor.permute(0, 2, 3, 1)[masks_cpu]
                else:
                    colors_selected = color_tensor.permute(0, 2, 3, 1)[masks_gpu].detach().cpu()

                all_points_cpu.append(points_selected)
                all_colors_cpu.append(colors_selected)

                # Optional correspondence saving (single pass only to avoid huge memory)
                if self.save_correspondence:
                    try:
                        points_world = res['points'][0].detach().cpu()
                        local_points = res['local_points'][0].detach().cpu()
                        camera_poses = res['camera_poses'][0].detach().cpu()
                        conf_logits = res['conf'][0, ..., 0].detach().cpu()
                        conf_prob = torch.sigmoid(conf_logits)
                        conf_mask_cpu = conf_mask[0].detach().cpu()
                        non_edge_cpu = non_edge[0].detach().cpu()
                        final_mask_cpu = masks_cpu.detach().clone()

                        N, H, W, _ = points_world.shape
                        u_coords = torch.arange(W, dtype=torch.int32).view(1, 1, W).expand(N, H, W)
                        v_coords = torch.arange(H, dtype=torch.int32).view(1, H, 1).expand(N, H, W)
                        pixel_coords = torch.stack((u_coords, v_coords), dim=-1)
                        frame_indices = torch.arange(N, dtype=torch.int32)

                        rgb_resized = color_tensor_cpu.permute(0, 2, 3, 1).contiguous()

                        meta_payload = {
                            'paths': image_meta.get('paths') if image_meta else None,
                            'original_sizes_hw': image_meta.get('original_sizes') if image_meta else None,
                            'resized_size_hw': image_meta.get('resized_size') if image_meta and image_meta.get('resized_size') else (H, W),
                            'interval': image_meta.get('interval') if image_meta else effective_interval,
                            'pixel_limit': image_meta.get('pixel_limit') if image_meta else None,
                            'source': image_meta.get('source') if image_meta else input_path,
                            'start_idx': image_meta.get('start_idx') if image_meta else None,
                            'end_idx': image_meta.get('end_idx') if image_meta else None,
                        }

                        correspondence_data = {
                            'points_world': points_world,
                            'local_points': local_points,
                            'camera_poses': camera_poses,
                            'rgb_resized': rgb_resized,
                            'confidence_logits': conf_logits,
                            'confidence_prob': conf_prob,
                            'confidence_mask': conf_mask_cpu,
                            'non_edge_mask': non_edge_cpu,
                            'final_mask': final_mask_cpu,
                            'pixel_coords_uv': pixel_coords,
                            'frame_indices': frame_indices,
                            'resized_size_hw': (H, W),
                            'confidence_threshold': self.confidence_threshold,
                            'metadata': meta_payload,
                        }

                        output_path_obj = Path(output_path)
                        correspondence_path = output_path_obj.with_name(output_path_obj.stem + self.correspondence_suffix)
                        torch.save(correspondence_data, correspondence_path)
                        print(f"[GPU {self.device}] Correspondence data saved to: {correspondence_path}")
                    except Exception as e:
                        print(f"[GPU {self.device}] WARNING: Failed to save correspondence data: {e}")

                del res, masks_gpu, masks_cpu, conf_scores, conf_mask, non_edge, combined_mask
                self._clear_gpu_memory()
            else:
                # Chunked processing
                if self.save_correspondence:
                    print(f"[GPU {self.device}] WARNING: save_correspondence is disabled in chunked mode to avoid OOM.")
                num_chunks = (total_frames + chunk_size - 1) // chunk_size
                print(f"[GPU {self.device}] Chunked inference: {num_chunks} chunks of up to {chunk_size} frames")
                start_idx = 0
                for ci in range(num_chunks):
                    end_idx = min(start_idx + chunk_size, total_frames)
                    chunk = imgs[start_idx:end_idx]
                    print(f"[GPU {self.device}] Chunk {ci+1}/{num_chunks}: frames {start_idx}..{end_idx-1}")

                    # Retry by halving chunk size on OOM
                    local_chunk = chunk
                    local_chunk_size = local_chunk.shape[0]
                    while True:
                        try:
                            res = run_model_on_tensor(local_chunk)
                            break
                        except torch.cuda.OutOfMemoryError as e:
                            if local_chunk_size <= 1:
                                print(f"[GPU {self.device}] ERROR: CUDA OOM even on single frame: {e}")
                                self._clear_gpu_memory()
                                return "error_oom"
                            self._clear_gpu_memory()
                            local_chunk_size = local_chunk_size // 2
                            print(f"[GPU {self.device}] OOM; retrying chunk with smaller size {local_chunk_size}")
                            local_chunk = chunk[:local_chunk_size]

                    conf_scores = torch.sigmoid(res['conf'][..., 0])
                    conf_mask = conf_scores > self.confidence_threshold
                    non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
                    combined_mask = torch.logical_and(conf_mask, non_edge)
                    masks_gpu = combined_mask[0]

                    color_tensor = local_chunk
                    points_selected = res['points'][0][masks_gpu].detach().cpu()
                    colors_selected = color_tensor.permute(0, 2, 3, 1)[masks_gpu].detach().cpu()
                    all_points_cpu.append(points_selected)
                    all_colors_cpu.append(colors_selected)

                    del res, masks_gpu, conf_scores, conf_mask, non_edge, combined_mask
                    self._clear_gpu_memory()

                    # Move window
                    start_idx = end_idx

            # Write concatenated outputs
            if len(all_points_cpu) == 0:
                print(f"[GPU {self.device}] SKIP: No valid points accumulated: {input_path}")
                del imgs
                self._clear_gpu_memory()
                return "skip_no_points"

            points_cat = torch.cat(all_points_cpu, dim=0)
            colors_cat = torch.cat(all_colors_cpu, dim=0)
            write_ply(points_cat, colors_cat, output_path)

            del imgs, points_cat, colors_cat
            self._clear_gpu_memory()

            print(f"[GPU {self.device}] SUCCESS: Output saved to: {output_path}")
            return "success"

        except torch.cuda.OutOfMemoryError as e:
            print(f"[GPU {self.device}] ERROR: CUDA OOM in {input_path}: {e}")
            self._clear_gpu_memory()
            return "error_oom"
        except Exception as e:
            print(f"[GPU {self.device}] ERROR: Failed to process {input_path}: {e}")
            self._clear_gpu_memory()
            return "error"


def has_images(directory: Path, image_extensions: List[str]) -> bool:
    """Return True if the directory contains at least one supported image file."""
    return any(directory.glob(ext) for ext in image_extensions)


def discover_recursive_image_directories(
    input_base: str,
    output_base: str,
    include_processed: bool = False,
) -> List[Tuple[str, str]]:
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
    directories = []

    for root, _, _ in os.walk(input_base):
        root_path = Path(root)
        if not has_images(root_path, image_extensions):
            continue

        rel_path = os.path.relpath(root, input_base)
        if root_path.name == "images":
            scene_name = root_path.parent.name
            output_file = Path(output_base) / scene_name / "images" / "pi3.ply"
        else:
            output_file = Path(output_base) / rel_path / "pi3.ply"

        if include_processed or (not output_file.exists()):
            directories.append((root, str(output_file)))

    return directories


def discover_roomtours_scene_directories(
    dataset_config: Dict[str, object],
    include_processed: bool = False,
) -> List[Tuple[str, str]]:
    input_base = str(dataset_config['input_base'])
    output_base = str(dataset_config['output_base'])
    max_entries = int(dataset_config.get('max_entries', 0) or 0)
    scene_json = dataset_config.get("scene_json")

    base_path = Path(input_base)
    directories: List[Tuple[str, str]] = []

    if scene_json:
        print(f"[INFO] RoomTours: loading explicit scene targets from {scene_json}")
        directories = load_roomtours_entries_from_json(
            scene_json_path=str(scene_json),
            input_base=input_base,
            output_base=output_base,
            include_processed=include_processed,
        )
        if max_entries > 0 and len(directories) > max_entries:
            print(f"[INFO] RoomTours(JSON): limiting to first {max_entries} entries out of {len(directories)}")
            directories = directories[:max_entries]
        print(f"[INFO] RoomTours(JSON): selected {len(directories)} scenes for processing")
        return directories

    prioritized_entries = []
    unsorted_entries = []

    for channel_dir in base_path.iterdir():
        if channel_dir.name.startswith("."):
            continue

        if channel_dir.is_dir():
            direct_scenes_dir = channel_dir / "scenes"
            if direct_scenes_dir.exists() and direct_scenes_dir.is_dir():
                for scene_file in direct_scenes_dir.glob("*.mp4"):
                    if not scene_file.is_file():
                        continue
                    output_file = Path(output_base) / channel_dir.name / scene_file.stem / "pi3.ply"
                    if not include_processed and output_file.exists():
                        continue
                    if max_entries > 0:
                        duration = get_video_duration_seconds(scene_file)
                        prioritized_entries.append((duration, str(scene_file), str(output_file)))
                    else:
                        unsorted_entries.append((str(scene_file), str(output_file)))
                continue

            for video_dir in channel_dir.iterdir():
                if video_dir.name.startswith(".") or (not video_dir.is_dir()):
                    continue
                scenes_dir = video_dir / "scenes"
                if not scenes_dir.exists() or (not scenes_dir.is_dir()):
                    continue
                for scene_file in scenes_dir.glob("*.mp4"):
                    if not scene_file.is_file():
                        continue
                    output_file = Path(output_base) / channel_dir.name / video_dir.name / scene_file.stem / "pi3.ply"
                    if not include_processed and output_file.exists():
                        continue
                    if max_entries > 0:
                        duration = get_video_duration_seconds(scene_file)
                        prioritized_entries.append((duration, str(scene_file), str(output_file)))
                    else:
                        unsorted_entries.append((str(scene_file), str(output_file)))
        elif channel_dir.is_file() and channel_dir.suffix.lower() == ".mp4":
            output_file = Path(output_base) / channel_dir.stem / "pi3.ply"
            if not include_processed and output_file.exists():
                continue
            if max_entries > 0:
                duration = get_video_duration_seconds(channel_dir)
                prioritized_entries.append((duration, str(channel_dir), str(output_file)))
            else:
                unsorted_entries.append((str(channel_dir), str(output_file)))

    if max_entries > 0:
        prioritized_entries.sort(key=lambda x: x[0], reverse=True)
        total_candidates = len(prioritized_entries)
        if total_candidates > max_entries:
            print(f"[INFO] RoomTours: limiting to top {max_entries} scenes by duration out of {total_candidates} candidates")
            prioritized_entries = prioritized_entries[:max_entries]
        print(f"[INFO] RoomTours: selected {len(prioritized_entries)} scenes for processing (limit {max_entries}).")
        directories.extend((input_path, output_path) for _, input_path, output_path in prioritized_entries)
    else:
        print(f"[INFO] RoomTours: processing all remaining scenes without duration sorting. Total candidates: {len(unsorted_entries)}")
        directories.extend(unsorted_entries)

    return directories


DISCOVERY_HANDLERS = {
    "recursive_images": lambda config, include_processed=False: discover_recursive_image_directories(
        str(config["input_base"]),
        str(config["output_base"]),
        include_processed=include_processed,
    ),
    "roomtours_scenes": discover_roomtours_scene_directories,
}


def get_dataset_directories(dataset_config: Dict, include_processed: bool = False) -> List[Tuple[str, str]]:
    """Resolve input/output pairs from the configured input layout."""
    layout = dataset_config["layout"]
    handler = DISCOVERY_HANDLERS.get(layout)
    if handler is None:
        raise ValueError(f"Unsupported layout: {layout}")
    return handler(dataset_config, include_processed=include_processed)

def worker_process(
    gpu_id: int,
    directories: List[Tuple[str, str]],
    interval: int,
    ckpt: str,
    results_queue,
    processing_config: Optional[Dict[str, object]] = None,
    save_correspondence: bool = False,
    pixel_limit: int = 255000,
    overwrite_existing: bool = False,
):
    """ワーカープロセス：指定されたGPUで割り当てられたディレクトリを処理"""
    device = f"cuda:{gpu_id}"
    processor = Pi3BatchProcessor(
        device=device,
        ckpt=ckpt,
        save_correspondence=save_correspondence,
        pixel_limit=pixel_limit,
        overwrite_existing=overwrite_existing,
    )
    
    total = len(directories)
    results = {
        'success': 0,
        'skip_exists': 0,
        'skip_no_input': 0,
        'skip_no_images': 0,
        'error_oom': 0,
        'error': 0
    }
    
    print(f"[GPU {gpu_id}] Starting processing {total} directories")
    
    for i, (input_path, output_path) in enumerate(directories, 1):
        print(f"[GPU {gpu_id}] Progress: {i}/{total}")
        
        result = processor.process_single_directory(
            input_path,
            output_path,
            interval,
            processing_config=processing_config,
        )
        
        if result in results:
            results[result] += 1
        else:
            results['error'] += 1
    
    results['gpu_id'] = gpu_id
    results_queue.put(results)
    print(f"[GPU {gpu_id}] Completed processing {total} directories")

def split_directories(directories: List, num_gpus: int) -> List[List]:
    """ディレクトリリストをGPU数に分割"""
    dirs_per_gpu = len(directories) // num_gpus
    remainder = len(directories) % num_gpus
    
    split_dirs = []
    start_idx = 0
    
    for i in range(num_gpus):
        current_count = dirs_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + current_count
        
        if start_idx < len(directories):
            split_dirs.append(directories[start_idx:end_idx])
        else:
            split_dirs.append([])
        
        start_idx = end_idx
    
    return split_dirs


LAYOUT_DESCRIPTIONS = {
    "recursive_images": "Generic recursive image-directory scan",
    "roomtours_scenes": "RoomTours segmented scenes layout",
}


def build_runtime_config(args: argparse.Namespace) -> Dict[str, object]:
    layout = args.layout
    if layout not in LAYOUT_DESCRIPTIONS:
        raise ValueError(f"Unknown layout: {layout}")

    return {
        "layout": layout,
        "input_base": args.input_base,
        "output_base": args.output_base,
        "description": LAYOUT_DESCRIPTIONS[layout],
        "preserve_order": args.preserve_order,
        "scene_json": args.scene_json,
        "max_entries": args.max_entries,
        "processing": {
            "video_skip_seconds": args.video_skip_seconds,
            "video_target_frames": args.video_target_frames,
            "video_adjust_for_high_fps": args.video_adjust_for_high_fps,
            "image_target_frames": args.image_target_frames,
        },
    }

def main():
    parser = argparse.ArgumentParser(description="Batch process scene videos or image directories with Pi3")
    parser.add_argument("--layout", "--config", dest="layout", required=True,
                        help="Input layout (roomtours_scenes or recursive_images)")
    parser.add_argument("--input_base", type=str, required=True,
                        help="Input base directory")
    parser.add_argument("--output_base", type=str, required=True,
                        help="Output base directory")
    parser.add_argument("--interval", type=int, default=4,
                        help="Sampling interval for images")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use")
    parser.add_argument("--include_processed", action='store_true',
                        help="Include entries whose output already exists (they will be skipped at processing time)")
    parser.add_argument("--save_correspondence", action='store_true',
                        help="Save per-pixel correspondence tensors alongside the PLY output")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards to split the work into (for multi-node parallelism)")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="Shard index for this job in [0, num_shards-1]")
    parser.add_argument("--pixel_limit", type=int, default=255000,
                        help="Approximate pixel budget per resized frame (e.g., 255000 for ~510x500)")
    parser.add_argument("--max_entries", type=int, default=0,
                        help="Limit total entries processed (0 = no limit)")
    parser.add_argument("--scene_json", "--roomtours_scene_json", dest="scene_json", type=str, default=None,
                        help="Optional JSON list describing explicit scene targets")
    parser.add_argument("--preserve_order", action='store_true',
                        help="Preserve discovered input order instead of sorting by input path")
    parser.add_argument("--video_skip_seconds", type=int, default=0,
                        help="Number of initial seconds to skip when sampling videos")
    parser.add_argument("--video_target_frames", type=int, default=1500,
                        help="Target frame count for video sampling when interval auto-adjust is enabled")
    parser.add_argument("--video_adjust_for_high_fps", action='store_true',
                        help="Increase the video sampling interval when the discovered frame count exceeds the target")
    parser.add_argument("--image_target_frames", type=int, default=1500,
                        help="Target image count for image-directory inputs before interval auto-adjust")
    parser.add_argument("--overwrite_existing", action='store_true',
                        help="Overwrite existing output files instead of skipping")
    
    args = parser.parse_args()
    try:
        dataset_config = build_runtime_config(args)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        print(f"Available layouts: {sorted(LAYOUT_DESCRIPTIONS.keys())}")
        return

    print(f"Processing dataset: {dataset_config['description']}")
    
    # 入力・出力ディレクトリのチェック
    if not os.path.exists(args.input_base):
        print(f"ERROR: Input base directory not found: {args.input_base}")
        return
    
    os.makedirs(args.output_base, exist_ok=True)
    
    # ディレクトリリストを取得
    print("Scanning for directories to process...")
    directories = get_dataset_directories(dataset_config, include_processed=args.include_processed)
    processing_config = dict(dataset_config.get("processing", {}))

    preserve_order = dataset_config.get("preserve_order", False)
    if not preserve_order:
        # 安定化のため入力パスでソート
        directories.sort(key=lambda x: x[0])
    else:
        print("Preserving directory order as provided by dataset configuration")

    # シャーディング（複数ノード並列用）
    if args.num_shards > 1:
        if args.shard_id < 0 or args.shard_id >= args.num_shards:
            print(f"ERROR: shard_id must be in [0, {args.num_shards-1}], got {args.shard_id}")
            return
        original_len = len(directories)
        directories = directories[args.shard_id::args.num_shards]
        print(f"Sharding enabled: shard {args.shard_id}/{args.num_shards}. This shard has {len(directories)} of {original_len} total entries")
    
    print(f"Found {len(directories)} directories to process")
    
    if len(directories) == 0:
        print("No directories found to process!")
        return
    
    # 最初の数個のディレクトリを表示
    print("Sample directories:")
    for i, (input_path, output_path) in enumerate(directories[:5]):
        print(f"  {i+1}. {input_path} -> {output_path}")
    if len(directories) > 5:
        print(f"  ... and {len(directories) - 5} more")
    
    # GPU数を確認
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
    if num_gpus < 1:
        print("ERROR: No CUDA devices available")
        return
    print(f"Using {num_gpus} GPUs (out of {available_gpus} available)")
    
    # ディレクトリをGPU数に分割
    split_dirs = split_directories(directories, num_gpus)
    
    # 各GPUの処理数を表示
    for i, dirs in enumerate(split_dirs):
        print(f"GPU {i}: {len(dirs)} directories")
    
    # マルチプロセシング開始
    print(f"\nStarting parallel processing with {num_gpus} GPUs...")
    start_time = time.time()
    
    # 結果を受け取るためのキュー
    results_queue = mp.Queue()
    processes = []
    
    # 各GPUでプロセスを開始
    for gpu_id in range(num_gpus):
        if len(split_dirs[gpu_id]) > 0:
            p = mp.Process(
                target=worker_process,
                args=(
                    gpu_id,
                    split_dirs[gpu_id],
                    args.interval,
                    args.ckpt,
                    results_queue,
                    processing_config,
                    args.save_correspondence,
                    args.pixel_limit,
                    args.overwrite_existing,
                ),
            )
            p.start()
            processes.append(p)
    
    # 全プロセスの完了を待機
    for p in processes:
        p.join()
    
    # 結果を集計
    total_results = {
        'success': 0,
        'skip_exists': 0,
        'skip_no_input': 0,
        'skip_no_images': 0,
        'error_oom': 0,
        'error': 0
    }
    
    while not results_queue.empty():
        gpu_results = results_queue.get()
        gpu_id = gpu_results.pop('gpu_id')
        
        print(f"GPU {gpu_id} results:")
        for key, value in gpu_results.items():
            print(f"  {key}: {value}")
            total_results[key] += value
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"\n=== Final Results ===")
    print(f"Dataset: {dataset_config['description']}")
    print(f"Successfully processed: {total_results['success']}")
    print(f"Skipped (already exists): {total_results['skip_exists']}")
    print(f"Skipped (no input): {total_results['skip_no_input']}")
    print(f"Skipped (no images): {total_results['skip_no_images']}")
    print(f"Failed (CUDA OOM): {total_results['error_oom']}")
    print(f"Failed (other errors): {total_results['error']}")
    print(f"Total processed: {sum(total_results.values())}")
    print(f"Total time: {elapsed_time:.2f} seconds")
    
    if total_results['error'] + total_results['error_oom'] > 0:
        print("WARNING: Some directories failed to process.")
    else:
        print("All directories processed successfully!")

if __name__ == '__main__':
    # マルチプロセシング用の設定
    mp.set_start_method('spawn', force=True)
    main()
