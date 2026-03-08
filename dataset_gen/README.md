# RoomTours Dataset Generation

Tools for reproducing the RoomTours processing pipeline from a CSV list of video URLs.

## Overview

This directory covers four steps:

1. download source videos listed in `../video_lists.csv`
2. split each video into indoor room-tour scenes with CLIP-based filtering
3. reconstruct per-scene point clouds with Pi3
4. run point-cloud preprocessing on Pi3 outputs

The public release is scheduler-agnostic. It does not assume ABCI, `qsub`, or any other cluster-specific environment.

## Requirements

- Python 3.9+
- CUDA GPU(s) for practical Pi3 execution
- enough disk space for downloaded videos and intermediate outputs

`ffmpeg` is not required by the current segmentation path.

## Environment Setup

Choose one of the following.

### Option 1: `venv`

```bash
cd dataset_gen
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Option 2: `uv`

```bash
cd dataset_gen
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Option 3: `conda`

```bash
cd dataset_gen
conda env create -f environment.yml
conda activate lam3c-dataset-gen
```

## Quick Start

After activating one of the environments above:

```bash
cd dataset_gen
python download.py
python segmentation.py
python pi3.py
python preprocess.py
```

## Commands

### 1. Download videos

```bash
python download.py
```

By default, this processes all entries in `../video_lists.csv`.

Common options:

```bash
python download.py --csv ../video_lists.csv --output-root ./data/roomtours/videos
python download.py --max-videos 1
python download.py --video-id=-09htWFYXaA
python download.py --video-id=-09htWFYXaA --video-id=-0DkX-2ZeYA
python download.py --max-videos 10 --dry-run
```

### 2. Run segmentation

```bash
python segmentation.py
```

Optional arguments:

```bash
python segmentation.py --input-root ./data/roomtours/videos --output-root ./data/roomtours/segmentation
python segmentation.py --gpu-ids 0,1,2,3 --concurrency 4
python segmentation.py --video-path ./data/roomtours/videos/-09htWFYXaA.mp4 --concurrency 1
python segmentation.py --video-id=-09htWFYXaA --concurrency 1
```

### 3. Run Pi3

```bash
python pi3.py
```

Optional arguments:

```bash
python pi3.py --input-root ./data/roomtours/segmentation --output-root ./data/roomtours/pi3
python pi3.py --num-gpus 4 --num-shards 4 --shard-id 0
python pi3.py --scene-path ./data/roomtours/segmentation/-09htWFYXaA/scenes/scene_000.mp4 --num-gpus 1
```

### 4. Run point-cloud preprocessing

```bash
python preprocess.py
```

By default, this writes only the final output and report for each input point cloud:

- `final_result.ply`
- `report.json`

Optional arguments:

```bash
python preprocess.py --input-root ./data/roomtours/pi3 --output-root ./data/roomtours/preprocess
python preprocess.py --ply-path ./data/roomtours/pi3/-09htWFYXaA/scene-004_bedroom/pi3.ply
python preprocess.py --max-entries 1 --verbose
python preprocess.py --ply-path ./data/roomtours/pi3/-09htWFYXaA/scene-004_bedroom/pi3.ply --save-intermediates
```

## Default Paths

- video manifest: `../video_lists.csv`
- downloads: `data/roomtours/videos`
- segmentation outputs: `data/roomtours/segmentation`
- Pi3 outputs: `data/roomtours/pi3`
- preprocessing outputs: `data/roomtours/preprocess`

## Repository Layout

- `download.py`: public download entrypoint
- `segmentation.py`: public segmentation entrypoint
- `pi3.py`: public Pi3 entrypoint
- `preprocess.py`: public point-cloud preprocessing entrypoint
- `requirements.txt`: pip/venv/uv dependency definition
- `environment.yml`: conda environment definition
- `pipeline/download_video_list.py`: manifest-driven downloader
- `pipeline/segmentation/`: segmentation implementation
- `pipeline/pi3/`: Pi3 batch implementation
- `pipeline/preprocess/`: point-cloud preprocessing implementation
- `third_party/Pi3/`: vendored Pi3 code

## Notes

- The downloader skips files that already exist for the same `video_id`.
- Failed downloads are recorded in `data/roomtours/download_failures.tsv`.
- The segmentation step can run on CPU, but it is much slower.
- Pi3 is expected to run on CUDA GPUs.
- Public entrypoints use the currently active Python environment. They do not require a fixed `.venv` name.

## License Notes

This directory does not yet ship with a single top-level `LICENSE` file. See `LICENSING.md` and `THIRD_PARTY.md` for the boundary between repository-maintained code and vendored third-party code.
