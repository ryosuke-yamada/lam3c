# Dataset Configs

Each file in `datasets/` defines the default dataset-specific values used by:

- `scripts/submit_stage.sh`
- `scripts/run_stage.sh`

Typical usage:

```bash
./scripts/submit_stage.sh segmentation roomtours_batch_v8
./scripts/submit_stage.sh pi3 roomtours_batch_v8
./scripts/submit_stage.sh vggt roomtours_vggt_v2_200
```

Environment variables can override the defaults at submission time.
