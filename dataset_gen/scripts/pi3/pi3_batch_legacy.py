import torch
import argparse
import os
import gc
import time
from pathlib import Path
import sys
from typing import List, Tuple
import multiprocessing as mp

VENDORED_PI3_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "Pi3"
if str(VENDORED_PI3_ROOT) not in sys.path:
    sys.path.insert(0, str(VENDORED_PI3_ROOT))

from pi3.utils.basic import load_images_as_tensor, write_ply
from pi3.utils.geometry import depth_edge
from pi3.models.pi3 import Pi3

class Pi3BatchProcessor:
    def __init__(self, device: str = 'cuda', ckpt: str = None):
        self.device = torch.device(device)
        self.model = None
        self.dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
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
            # ECCエラーを避けるため、synchronizeをスキップ
            # torch.cuda.synchronize()
        gc.collect()
        
        # より強力なメモリクリア
        if torch.cuda.is_available():
            try:
                # 現在のGPUメモリ使用量を確認
                allocated = torch.cuda.memory_allocated()
                reserved = torch.cuda.memory_reserved()
                print(f"Memory after clear - Allocated: {allocated/1024**3:.2f}GB, Reserved: {reserved/1024**3:.2f}GB")
            except RuntimeError as e:
                if "ECC error" in str(e):
                    print(f"Warning: ECC error during memory check, continuing...")
                else:
                    print(f"Warning: Error during memory check: {e}")
    
    def process_single_directory(self, input_path: str, output_path: str, interval: int = 1) -> str:
        """単一ディレクトリを処理"""
        try:
            # 入力チェック
            if not os.path.exists(input_path):
                print(f"[GPU {self.device}] SKIP: Input directory not found: {input_path}")
                return "skip_no_input"
                
            # 出力チェック
            if os.path.exists(output_path):
                print(f"[GPU {self.device}] SKIP: Output file already exists: {output_path}")
                return "skip_exists"
            
            # 画像の存在チェック
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(input_path).glob(ext))
            
            if not image_files:
                print(f"[GPU {self.device}] SKIP: No image files found in: {input_path}")
                return "skip_no_images"
            
            # ARKitScenesの場合、画像ファイルの破損チェック
            if "arkitscenes" in input_path.lower() or "lowres_wide" in input_path:
                print(f"[GPU {self.device}] ARKitScenes: Found {len(image_files)} images in {input_path}")
                if len(image_files) > 0:
                    print(f"[GPU {self.device}] First image: {image_files[0]}")
                    print(f"[GPU {self.device}] Last image: {image_files[-1]}")
                    
                    # 最初の数枚の画像をチェック
                    try:
                        import cv2
                        corrupted_count = 0
                        for i, img_file in enumerate(image_files[:3]):  # 最初の3枚をチェック
                            try:
                                img = cv2.imread(str(img_file))
                                if img is None:
                                    print(f"[GPU {self.device}] WARNING: Corrupted image detected: {img_file}")
                                    corrupted_count += 1
                                else:
                                    print(f"[GPU {self.device}] Image {i+1} OK: {img_file} (shape: {img.shape})")
                            except Exception as e:
                                print(f"[GPU {self.device}] ERROR reading image {img_file}: {e}")
                                corrupted_count += 1
                        
                        if corrupted_count > 0:
                            print(f"[GPU {self.device}] WARNING: {corrupted_count} corrupted images detected in first 3 images")
                            return "error_corrupted_images"
                    except ImportError:
                        print(f"[GPU {self.device}] Warning: cv2 not available, skipping image corruption check")
            
            print(f"[GPU {self.device}] Processing: {input_path} -> {output_path}")
            
            # 画像読み込み
            imgs = load_images_as_tensor(input_path, interval=interval).to(self.device)
            
            # 推論実行
            print(f"[GPU {self.device}] Running model inference...")
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=self.dtype):
                    res = self.model(imgs[None])  # バッチ次元を追加
            
            # マスク処理
            masks = torch.sigmoid(res['conf'][..., 0]) > 0.1
            non_edge = ~depth_edge(res['local_points'][..., 2], rtol=0.03)
            masks = torch.logical_and(masks, non_edge)[0]
            
            # 結果保存
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            write_ply(res['points'][0][masks].cpu(), imgs.permute(0, 2, 3, 1)[masks], output_path)
            
            # メモリクリア
            del imgs, res, masks, non_edge
            
            # ARKitScenesの場合、より慎重なメモリクリア
            if "arkitscenes" in input_path.lower() or "lowres_wide" in input_path:
                print(f"[GPU {self.device}] ARKitScenes: Performing careful memory cleanup")
                # 段階的にメモリをクリア
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                    time.sleep(0.1)  # 少し待機
                except Exception as e:
                    print(f"[GPU {self.device}] Warning: Error during ARKitScenes memory cleanup: {e}")
            else:
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

def worker_process(gpu_id: int, directories: List[Tuple[str, str]], interval: int, ckpt: str, results_queue):
    """ワーカープロセス：指定されたGPUで割り当てられたディレクトリを処理"""
    device = f"cuda:{gpu_id}"
    processor = Pi3BatchProcessor(device=device, ckpt=ckpt)
    
    total = len(directories)
    results = {
        'success': 0,
        'skip_exists': 0,
        'skip_no_input': 0,
        'skip_no_images': 0,
        'error_oom': 0,
        'error_corrupted_images': 0,
        'error': 0
    }
    
    print(f"[GPU {gpu_id}] Starting processing {total} directories")
    
    for i, (input_path, output_path) in enumerate(directories, 1):
        print(f"[GPU {gpu_id}] Progress: {i}/{total}")
        
        result = processor.process_single_directory(input_path, output_path, interval)
        
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
        # 余りがある場合は最初のGPUに1つずつ多く割り当て
        current_count = dirs_per_gpu + (1 if i < remainder else 0)
        end_idx = start_idx + current_count
        
        if start_idx < len(directories):
            split_dirs.append(directories[start_idx:end_idx])
        else:
            split_dirs.append([])
        
        start_idx = end_idx
    
    return split_dirs

def main():
    parser = argparse.ArgumentParser(description="Batch process directories with Pi3 model using 8 GPUs")
    parser.add_argument("--input_base", type=str, 
                        default="/groups/gag51402/user/yamada/scannet_rgb",
                        help="Input base directory (default: /groups/gag51402/user/yamada/scannet_rgb)")
    parser.add_argument("--output_base", type=str, 
                        default="./outputs/scannet_rgb_pi3",
                        help="Output base directory (default: ./outputs/scannet_rgb_pi3)")
    parser.add_argument("--dataset_type", type=str, default="scannet",
                        choices=["scannet", "arkitscenes", "structure3d"],
                        help="Dataset type: scannet, arkitscenes, or structure3d")
    parser.add_argument("--interval", type=int, default=4,
                        help="Sampling interval for images")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to model checkpoint")
    parser.add_argument("--num_gpus", type=int, default=8,
                        help="Number of GPUs to use")
    
    args = parser.parse_args()
    
    # 入力・出力ディレクトリのチェック
    if not os.path.exists(args.input_base):
        print(f"ERROR: Input base directory not found: {args.input_base}")
        return
    
    os.makedirs(args.output_base, exist_ok=True)
    
    # データセットタイプに応じてディレクトリを収集
    base_path = Path(args.input_base)
    
    if args.dataset_type == "scannet":
        # ScanNet用: colorディレクトリを検索
        all_input_dirs = list(base_path.glob("**/color"))
        if not all_input_dirs:
            print(f"ERROR: No color directories found in {args.input_base}")
            return
        
        # 未処理のディレクトリのみを選択
        directories = []
        for input_dir in all_input_dirs:
            scene_dir = input_dir.parent
            scene_name = scene_dir.name
            output_file = Path(args.output_base) / scene_name / "pi3.ply"
            
            if not output_file.exists():
                directories.append((str(input_dir), str(output_file)))
    
    elif args.dataset_type == "arkitscenes":
        # ARKitScenes用: xxx_frames/lowres_wideディレクトリを検索
        all_input_dirs = list(base_path.glob("**/*_frames/lowres_wide"))
        if not all_input_dirs:
            print(f"ERROR: No *_frames/lowres_wide directories found in {args.input_base}")
            return
        
        # TrainingとValidationを分離して処理
        directories = []
        training_count = 0
        validation_count = 0
        
        for input_dir in all_input_dirs:
            # パス構造: .../Training/40753686/40753686_frames/lowres_wide
            # または: .../Validation/40753686/40753686_frames/lowres_wide
            scene_dir = input_dir.parent.parent  # 40753686_framesの親ディレクトリ
            scene_name = scene_dir.name  # 40753686
            
            # TrainingかValidationかを判定
            parent_dir = scene_dir.parent  # TrainingまたはValidationディレクトリ
            split_type = parent_dir.name  # "Training" または "Validation"
            
            # 出力先を適切に設定
            output_file = Path(args.output_base) / split_type / scene_name / "pi3.ply"
            
            if not output_file.exists():
                directories.append((str(input_dir), str(output_file)))
                if split_type == "Training":
                    training_count += 1
                elif split_type == "Validation":
                    validation_count += 1
        
        print(f"Found {training_count} Training directories and {validation_count} Validation directories to process")
    
    elif args.dataset_type == "structure3d":
        # Structure3D用: シーンディレクトリを検索
        all_input_dirs = list(base_path.glob("scene_*"))
        if not all_input_dirs:
            print(f"ERROR: No scene_* directories found in {args.input_base}")
            return
        
        # 未処理のディレクトリのみを選択
        directories = []
        for input_dir in all_input_dirs:
            scene_name = input_dir.name  # scene_XXXXX
            output_file = Path(args.output_base) / scene_name / "pi3.ply"
            
            if not output_file.exists():
                directories.append((str(input_dir), str(output_file)))
    
    else:
        print(f"ERROR: Unknown dataset type: {args.dataset_type}")
        return
    
    print(f"Found {len(directories)} directories to process (out of {len(all_input_dirs)} total)")
    
    if len(directories) == 0:
        print("All directories have already been processed!")
        return
    
    # GPU数を確認
    available_gpus = torch.cuda.device_count()
    num_gpus = min(args.num_gpus, available_gpus)
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
        if len(split_dirs[gpu_id]) > 0:  # 処理するディレクトリがある場合のみ
            p = mp.Process(
                target=worker_process,
                args=(gpu_id, split_dirs[gpu_id], args.interval, args.ckpt, results_queue)
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
        'error_corrupted_images': 0,
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
    print(f"Successfully processed: {total_results['success']}")
    print(f"Skipped (already exists): {total_results['skip_exists']}")
    print(f"Skipped (no input): {total_results['skip_no_input']}")
    print(f"Skipped (no images): {total_results['skip_no_images']}")
    print(f"Failed (CUDA OOM): {total_results['error_oom']}")
    print(f"Failed (corrupted images): {total_results['error_corrupted_images']}")
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
