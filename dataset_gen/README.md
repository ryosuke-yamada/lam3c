# Dataset Generation

This directory contains the complete pipeline used to reproduce the released RoomTours-derived datasets.

## Scope

The pipeline covered here is:

1. raw video collection layout
2. CLIP-based indoor-frame filtering and room-level scene segmentation
3. Pi3 point-cloud generation from segmented scene videos

## Layout

- `configs/datasets/`: dataset-specific defaults for segmentation / Pi3
- `configs/base/`: shared stage defaults and RoomTours helper fragments
- `docs/roomtours_dataset_provenance.md`: dataset-by-dataset provenance matrix
- `scripts/submit_pipeline.sh`: one-shot qsub wrapper for segmentation -> Pi3
- `scripts/submit_stage.sh`: generic qsub entrypoint for all stages
- `scripts/run_stage.sh`: generic runtime entrypoint executed inside the job
- `scripts/common/`: shared config / job / stage helpers
- `scripts/segmentation/`: raw video -> `inside_only.avi` -> `scenes/*.mp4`
- `scripts/pi3/`: `scenes/*.mp4` -> `pi3.ply`
- `third_party/Pi3/`: vendored Pi3 runtime code
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

## Execution model

The primary workflow is now config-driven:

```bash
./scripts/submit_pipeline.sh roomtours_batch_v8
```

Or stage-by-stage if needed:

```bash
./scripts/submit_stage.sh segmentation roomtours_batch_v8
./scripts/submit_stage.sh pi3 roomtours_batch_v8
```

`submit_stage.sh` reads the dataset defaults from `configs/datasets/*.sh`, applies any environment-variable overrides currently set in the shell, and submits the generic runtime script `scripts/run_stage.sh`.

The dataset configs are intentionally thin. Shared PBS defaults live under `configs/base/`, and the RoomTours-family datasets derive their stage names, log prefixes, and layout defaults from `configs/base/roomtours_release_common.sh`.

The wrappers still contain cluster-specific PBS resource directives. Adjust queue / project settings for your environment before release if needed.

## Licensing note

`dataset_gen/` does not yet ship with a single top-level `LICENSE` file. Review `LICENSING.md` before assigning one, because `third_party/` remains under upstream licenses and model weights are not redistributed here.
