# Licensing Boundary

This file defines the intended boundary between `dataset_gen`-maintained code and vendored third-party code before adding a top-level `LICENSE` file to `dataset_gen/`.

## 1. Vendored third-party code

The following directories are vendored copies of external projects and must remain under their own upstream licenses:

- `third_party/Pi3/`

These directories are excluded from any future top-level `dataset_gen/LICENSE` unless that file explicitly says otherwise.

Refer to `THIRD_PARTY.md` and the upstream license files stored inside each vendored directory:

- `third_party/Pi3/LICENSE`

## 2. `dataset_gen`-maintained code

The following paths are maintained as part of this repository's dataset-generation pipeline:

- `setup.sh`
- `.gitignore`
- `README.md`
- `THIRD_PARTY.md`
- `download.py`
- `segmentation.py`
- `pi3.py`
- `configs/`
- `docs/`
- `pipeline/`

This includes scripts that were reconstructed from job logs or adapted from the authors' internal working tree in order to document the released dataset pipeline.

## 3. Adapted integration scripts

Some implementation files under `pipeline/` were derived from pre-existing implementation code and then modified to run from `dataset_gen/` as a self-contained pipeline. They are repository-maintained integration code, not vendored upstream snapshots:

- `pipeline/pi3/pi3_batch_datasets.py`

They should be reviewed carefully before assigning a final top-level license to `dataset_gen/`.

## 4. Model weights

No model checkpoints or weights are redistributed inside `dataset_gen/`.

- `Pi3` weights are not vendored here.

The current code paths rely on upstream download mechanisms or user-provided environments at runtime. Any future release must keep the treatment of weights separate from the treatment of source code.

## 5. Practical rule for the future top-level LICENSE

When adding `dataset_gen/LICENSE`, it should apply only to the repository-maintained files listed in section 2 unless the file explicitly states a narrower scope.

It should not silently relicense anything under `third_party/`.
