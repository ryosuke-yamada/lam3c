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

- about `117 GB` for the source videos in `data/roomtours/videos`
- about `96 GB` for the final RoomTours point-cloud data after preprocessing

Segmentation and Pi3 intermediates require additional temporary headroom.

## Installation

Create and activate a Python environment with your preferred tool, then install the dependencies:

```bash
cd roomtours_gen
python -m pip install -r requirements.txt
```

`environment.yml` is included for reproducibility, but `requirements.txt` is the primary installation path documented here.

## End-to-End Usage

Run the full pipeline with the default manifest and default output locations:

```bash
cd roomtours_gen
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

### `pipeline.py` arguments

| Argument | Description |
| --- | --- |
| `--csv` | Input CSV manifest. Defaults to `../video_lists.csv`. |
| `--download-root` | Output directory for downloaded videos. |
| `--segmentation-root` | Output directory for segmentation results. |
| `--pi3-root` | Output directory for Pi3 point clouds. |
| `--preprocess-root` | Output directory for preprocessed point clouds. |
| `--max-videos` | Limit the run to the first `N` manifest entries. `0` means all entries. |
| `--video-id` | Restrict the run to specific video IDs. Can be passed multiple times. |
| `--segmentation-concurrency` | Number of segmentation workers. |
| `--gpu-ids` | Comma-separated GPU list for segmentation, for example `0,1,2,3`. |
| `--pi3-num-gpus` | Number of GPUs used by Pi3. |
| `--pi3-num-shards` | Number of Pi3 shards for manual parallelization. |
| `--pi3-shard-id` | Pi3 shard index in `[0, pi3-num-shards-1]`. |
| `--save-intermediates` | Save intermediate point clouds during preprocessing. |
| `--verbose` | Enable verbose preprocessing logs. |
| `--skip-download` | Skip the download stage. |
| `--skip-segmentation` | Skip the segmentation stage. |
| `--skip-pi3` | Skip the Pi3 stage. |
| `--skip-preprocess` | Skip the preprocessing stage. |
| `--overwrite-existing` | Overwrite existing outputs for stages that support it. |
| `--dry-run` | Print resolved stage commands without executing them. |

## Stage Entry Points

### Download

```bash
python download.py
python download.py --max-videos 1
python download.py --video-id=-09htWFYXaA
```

| Argument | Description |
| --- | --- |
| `--csv` | Input CSV manifest. Defaults to `../video_lists.csv`. |
| `--output-root` | Directory where downloaded videos are written. |
| `--archive` | `yt-dlp` archive file used to avoid re-downloading the same videos. |
| `--failure-log` | TSV file for failed downloads. |
| `--yt-dlp-bin` | `yt-dlp` executable name or path. |
| `--max-videos` | Limit to the first `N` manifest entries. `0` means all entries. |
| `--video-id` | Download only specific video IDs. Can be passed multiple times. |
| `--dry-run` | Print the resolved command without executing it. |

### Segmentation

```bash
python segmentation.py
python segmentation.py --video-id=-09htWFYXaA --concurrency 1
python segmentation.py --video-path ./data/roomtours/videos/-09htWFYXaA.mp4 --concurrency 1
```

| Argument | Description |
| --- | --- |
| `--input-root` | Directory containing downloaded source videos. |
| `--output-root` | Directory where segmentation outputs are written. |
| `--runner` | Internal execution mode. `queue` is the default and the recommended mode. |
| `--concurrency` | Number of concurrent segmentation workers. |
| `--video-path` | Process only explicit video paths. Can be passed multiple times. |
| `--video-id` | Process only videos matching specific `video_id` values under `--input-root`. |
| `--gpu-ids` | Comma-separated GPU IDs used by the segmentation stage. |
| `--num-shards` | Optional shard count for batch mode. |
| `--shard-id` | Shard index in batch mode. |
| `--dry-run` | Print the resolved command without executing it. |

### Pi3

```bash
python pi3.py
python pi3.py --num-gpus 4 --num-shards 4 --shard-id 0
python pi3.py --scene-path ./data/roomtours/segmentation/-09htWFYXaA/scenes/scene-004_bedroom.mp4 --num-gpus 1
```

| Argument | Description |
| --- | --- |
| `--input-root` | Root directory containing segmentation outputs. |
| `--output-root` | Directory where Pi3 outputs are written. |
| `--layout` | Input layout expected by the Pi3 batch script. Use `roomtours_scenes` for the default pipeline. |
| `--interval` | Frame or image sampling interval. |
| `--num-gpus` | Number of GPUs used by Pi3. |
| `--num-shards` | Number of shards for manual parallel execution. |
| `--shard-id` | Shard index in `[0, num-shards-1]`. |
| `--pixel-limit` | Approximate resize budget per frame before Pi3 inference. |
| `--max-entries` | Limit the number of discovered inputs. `0` means no limit. |
| `--scene-json` | Explicit JSON file listing scene inputs and outputs. |
| `--scene-path` | Process only explicit segmented scene MP4 files. Can be passed multiple times. |
| `--video-skip-seconds` | Number of initial seconds skipped for each input video scene. |
| `--video-target-frames` | Maximum target frame count for videos after sampling. |
| `--image-target-frames` | Maximum target frame count for image-directory inputs. |
| `--preserve-order` | Preserve discovered input order instead of reordering internally. |
| `--video-adjust-for-high-fps` | Increase sampling interval automatically for high-FPS videos. Enabled by default. |
| `--include-processed` | Include inputs whose outputs already exist. |
| `--overwrite-existing` | Overwrite existing Pi3 outputs instead of skipping them. |
| `--dry-run` | Print the resolved command without executing it. |

### Preprocessing

```bash
python preprocess.py
python preprocess.py --ply-path ./data/roomtours/pi3/-09htWFYXaA/scene-004_bedroom/pi3.ply
python preprocess.py --ply-path ./data/roomtours/pi3/-09htWFYXaA/scene-004_bedroom/pi3.ply --save-intermediates
```

By default, preprocessing writes:

- `final_result.ply`
- `report.json`

| Argument | Description |
| --- | --- |
| `--input-root` | Root directory containing Pi3 outputs. |
| `--output-root` | Directory where preprocessing outputs are written. |
| `--config` | JSON config for preprocessing. Defaults to `pipeline/preprocess/default_config.json`. |
| `--ply-path` | Process only explicit Pi3 `.ply` files. Can be passed multiple times. |
| `--max-entries` | Limit the number of discovered `.ply` inputs. `0` means all entries. |
| `--overwrite-existing` | Overwrite existing preprocessing outputs. |
| `--save-intermediates` | Save intermediate point clouds in addition to the final output. |
| `--verbose` | Enable verbose preprocessing logs. |
| `--dry-run` | Print resolved preprocessing targets without executing them. |

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
