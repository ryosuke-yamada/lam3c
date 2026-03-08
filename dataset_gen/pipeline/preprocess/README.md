# Preprocess Pipeline

This directory contains the point-cloud preprocessing modules used by `../../preprocess.py`.

The public entrypoint is:

```bash
python preprocess.py
```

Do not import these modules through the old internal `scripts.code.*` path. In `dataset_gen`, the package path is:

```python
from pipeline.preprocess.main_pipeline import PreprocessPipeline
```

The default configuration file is:

- `default_config.json`
