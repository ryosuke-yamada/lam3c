# Dataset Configs

The public pipeline now uses a single default config:

- `datasets/default.sh`

This file defines:

- the canonical input manifest (`../video_lists.csv`)
- the default download / segmentation / Pi3 output roots
- the default PBS settings for segmentation and Pi3

Typical usage does not need an explicit config name anymore:

```bash
python download.py
python segmentation.py
python pi3.py
```

If needed, another config file path can still be passed as the optional first argument to each public entrypoint, and any variable can be overridden temporarily with `--set NAME=VALUE`.
