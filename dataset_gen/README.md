# Dataset Generation

This directory contains the public RoomTours reproduction pipeline.

## Scope

The canonical flow is:

1. read the input manifest at `../video_lists.csv`
2. download videos with `yt-dlp`
3. run CLIP-based indoor-frame filtering and room-level scene segmentation
4. run Pi3 point-cloud generation on the segmented scene videos

## Public entrypoints

The public interface is intentionally minimal:

- `download.py`: download the videos listed in `../video_lists.csv`
- `segmentation.py`: submit the segmentation stage
- `pi3.py`: submit the Pi3 stage

Everything else under `pipeline/` is internal implementation detail.

## Layout

- `download.py`: public download entrypoint
- `segmentation.py`: public segmentation entrypoint
- `pi3.py`: public Pi3 entrypoint
- `configs/datasets/default.sh`: default runtime and cluster settings
- `docs/roomtours_dataset_provenance.md`: canonical provenance note for the public pipeline
- `pipeline/`: internal implementation code and runtime helpers
- `third_party/Pi3/`: vendored Pi3 runtime code
- `setup.sh`: creates the local virtual environment used by download / segmentation / Pi3
- `LICENSING.md`: boundary between repository-maintained code and vendored third-party code
- `THIRD_PARTY.md`: pinned upstream commit and license location

## Environment setup

Run from `dataset_gen/`:

```bash
./setup.sh
```

This creates `.venv_pi3` for the full pipeline.

## Public commands

Download the canonical video list:

```bash
python download.py
```

Submit segmentation:

```bash
python segmentation.py
```

Submit Pi3:

```bash
python pi3.py
```

Run them in order for the full reproduction flow.

All commands default to `configs/datasets/default.sh`, so an explicit dataset name is not required. If needed, you can still pass another config name or config path as the optional first argument.

Temporary overrides can be passed without editing the config file:

```bash
python segmentation.py --set SEG_PBS_PROJECT=gag51402
python pi3.py --set PI3_NUM_SHARDS=8 --set PI3_PBS_PROJECT=gag51402
```

The default working directories are:

- downloads: `data/roomtours/videos`
- segmentation outputs: `data/roomtours/segmentation`
- Pi3 outputs: `data/roomtours/pi3`

The PBS defaults are still cluster-specific. Adjust queue / project settings in `configs/datasets/default.sh` or via `--set` before release to other environments.

## Licensing note

`dataset_gen/` does not yet ship with a single top-level `LICENSE` file. Review `LICENSING.md` before assigning one, because `third_party/` remains under upstream licenses and model weights are not redistributed here.
