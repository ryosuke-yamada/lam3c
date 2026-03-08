# RoomTours Dataset Generation

Reproduction code for the RoomTours dataset-generation pipeline used in `lam3c`. Starting from `../video_lists.csv`, this directory downloads source videos, extracts indoor room-tour scenes, reconstructs per-scene point clouds with Pi3, and optionally runs point-cloud preprocessing.

The public release is scheduler-agnostic. It does not depend on ABCI, `qsub`, or any other cluster-specific environment.

## Overview

The pipeline consists of four stages:

1. `download.py`: download source videos from `../video_lists.csv`
2. `segmentation.py`: filter indoor frames and split videos into room-tour scenes
3. `pi3.py`: reconstruct point clouds from segmented scenes
4. `preprocess.py`: run point-cloud cleanup and normalization on Pi3 outputs

For a standard run, use the end-to-end entrypoint:

```bash
python pipeline.py
```

## Requirements

- Python 3.9 or newer
- CUDA GPU(s) for practical Pi3 execution
- Sufficient disk space for videos and intermediate outputs

For the full release-scale run, budget at least:

- about `117 GB` for `data/roomtours/videos`
- about `96 GB` for `data/roomtours/segmentation`

Pi3 and preprocessing outputs require additional headroom.

## Installation

Create and activate a Python environment with your preferred tool, then install the dependencies:

```bash
cd dataset_gen
python -m pip install -r requirements.txt
```

`environment.yml` is included for reproducibility, but `requirements.txt` is the primary installation path documented here.

## End-to-End Usage

Run the full pipeline with the default manifest and default output locations:

```bash
cd dataset_gen
python pipeline.py
```

Useful variants:

```bash
python pipeline.py --max-videos 1
python pipeline.py --video-id=-09htWFYXaA
python pipeline.py --skip-download
python pipeline.py --skip-preprocess
python pipeline.py --pi3-num-gpus 4 --segmentation-concurrency 4
```

By default, `pipeline.py` executes:

1. `python download.py`
2. `python segmentation.py`
3. `python pi3.py`
4. `python preprocess.py`

## Stage Entry Points

### Download

```bash
python download.py
python download.py --max-videos 1
python download.py --video-id=-09htWFYXaA
```

### Segmentation

```bash
python segmentation.py
python segmentation.py --video-id=-09htWFYXaA --concurrency 1
python segmentation.py --video-path ./data/roomtours/videos/-09htWFYXaA.mp4 --concurrency 1
```

### Pi3

```bash
python pi3.py
python pi3.py --num-gpus 4 --num-shards 4 --shard-id 0
python pi3.py --scene-path ./data/roomtours/segmentation/-09htWFYXaA/scenes/scene-004_bedroom.mp4 --num-gpus 1
```

### Preprocessing

```bash
python preprocess.py
python preprocess.py --ply-path ./data/roomtours/pi3/-09htWFYXaA/scene-004_bedroom/pi3.ply
python preprocess.py --ply-path ./data/roomtours/pi3/-09htWFYXaA/scene-004_bedroom/pi3.ply --save-intermediates
```

By default, preprocessing writes:

- `final_result.ply`
- `report.json`

## Input and Output Layout

Default inputs and outputs are:

- input manifest: `../video_lists.csv`
- downloaded videos: `data/roomtours/videos`
- segmentation outputs: `data/roomtours/segmentation`
- Pi3 outputs: `data/roomtours/pi3`
- preprocessing outputs: `data/roomtours/preprocess`

Typical segmentation output layout:

```text
data/roomtours/segmentation/<video_id>/
  inside_only.avi
  scenes/
    scene-000_*.mp4
    scene-001_*.mp4
    ...
```

Typical Pi3 output layout:

```text
data/roomtours/pi3/<video_id>/<scene_name>/pi3.ply
```

Typical preprocessing output layout:

```text
data/roomtours/preprocess/<video_id>/<scene_name>/
  final_result.ply
  report.json
```

## Repository Layout

- `pipeline.py`: end-to-end public entrypoint
- `download.py`: download entrypoint
- `segmentation.py`: segmentation entrypoint
- `pi3.py`: Pi3 entrypoint
- `preprocess.py`: preprocessing entrypoint
- `pipeline/`: internal implementation details
- `third_party/Pi3/`: vendored Pi3 code

## Notes

- Existing downloads are skipped automatically when the same `video_id` is already present.
- Failed downloads are recorded in `data/roomtours/download_failures.tsv`.
- Segmentation can run on CPU, but it is substantially slower than GPU execution.
- Public entrypoints use the currently active Python environment.

## License and Third-Party Code

This directory does not ship with a single top-level `LICENSE` file yet. See `LICENSING.md` and `THIRD_PARTY.md` for the boundary between repository-maintained code and vendored third-party code.
