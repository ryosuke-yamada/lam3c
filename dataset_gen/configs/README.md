# Dataset Configs

Each file in `datasets/` defines the thin dataset-specific layer used by:

- `scripts/submit_stage.sh`
- `scripts/run_stage.sh`

Shared stage defaults and helper fragments live under `base/`. The RoomTours-family configs derive most stage names, log prefixes, and Pi3 layout defaults from `base/roomtours_release_common.sh`.

Typical usage:

```bash
./scripts/submit_stage.sh segmentation roomtours_batch_v8
./scripts/submit_stage.sh pi3 roomtours_batch_v8
```

Environment variables can override the defaults at submission time.
