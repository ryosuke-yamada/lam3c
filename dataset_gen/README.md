# Dataset Generation

This directory contains the complete pipeline used to reproduce the released RoomTours-derived datasets.

## Scope

The pipeline covered here is:

1. raw video collection layout
2. CLIP-based indoor-frame filtering and room-level scene segmentation
3. Pi3 point-cloud generation from segmented scene videos
4. VGGT / COLMAP export for `roomtours_vggt_v2_200`

## Layout

- `docs/roomtours_dataset_provenance.md`: dataset-by-dataset provenance matrix
- `scripts/segmentation/`: raw video -> `inside_only.avi` -> `scenes/*.mp4`
- `scripts/pi3/`: `scenes/*.mp4` -> `pi3.ply`
- `scripts/vggt/`: Pi3-aligned frame sampling -> VGGT / COLMAP export
- `third_party/Pi3/`: vendored Pi3 runtime code
- `third_party/vggt/`: vendored VGGT runtime code
- `setup.sh`: creates local virtual environments for the pipeline
- `LICENSING.md`: boundary between repository-maintained code and vendored third-party code
- `THIRD_PARTY.md`: pinned upstream commits and license locations

## Environment setup

Run from `dataset_gen/`:

```bash
./setup.sh
```

This creates:

- `.venv_pi3`: segmentation + Pi3 runtime
- `.venv_vggt`: VGGT runtime

## Execution model

All submit wrappers under `scripts/` resolve paths relative to `dataset_gen/` and expect the dataset roots to be provided via environment variables or the defaults embedded in each wrapper.

The wrappers still contain cluster-specific PBS resource directives. Adjust queue / project settings for your environment before release if needed.

## Licensing note

`dataset_gen/` does not yet ship with a single top-level `LICENSE` file. Review `LICENSING.md` before assigning one, because `third_party/` remains under upstream licenses and model weights are not redistributed here.
