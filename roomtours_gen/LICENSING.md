# Licensing Boundary

This file defines the intended boundary between `roomtours_gen`-maintained code and vendored third-party code before adding a top-level `LICENSE` file to `roomtours_gen/`.

## Vendored third-party code

The following directory is a vendored copy of an external project and must remain under its upstream license:

- `third_party/Pi3/`

Refer to `THIRD_PARTY.md` and the upstream license file stored inside the vendored directory:

- `third_party/Pi3/LICENSE`

## Repository-maintained code

The following paths are maintained as part of this repository's public dataset-generation pipeline:

- `requirements.txt`
- `environment.yml`
- `.gitignore`
- `README.md`
- `THIRD_PARTY.md`
- `download.py`
- `segmentation.py`
- `pi3.py`
- `preprocess.py`
- `docs/`
- `pipeline/`

This includes integration code adapted from the original internal working tree to make the public release self-contained and scheduler-agnostic.

## Adapted integration scripts

Some implementation files under `pipeline/` were derived from pre-existing implementation code and then modified to run from `roomtours_gen/` as a self-contained pipeline. They are repository-maintained integration code, not vendored upstream snapshots.

The main example is:

- `pipeline/pi3/pi3_batch_datasets.py`

## Model weights

No model checkpoints or weights are redistributed inside `roomtours_gen/`.

- `Pi3` weights are not vendored here.

Weights remain separate from the source-code release.

## Practical rule for a future top-level LICENSE

When adding `roomtours_gen/LICENSE`, it should apply only to the repository-maintained files listed above unless the file explicitly states a narrower scope.

It should not silently relicense anything under `third_party/`.
