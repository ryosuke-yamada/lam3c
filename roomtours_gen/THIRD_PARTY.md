# Third-Party Components

This directory vendors the external code required to reproduce the released dataset-generation pipeline.

## Pi3

- Source repository: local working copy from `3DV/Pi3`
- Upstream commit: `7f1054ecff3b745d534a35e47b2c1cac9d5fef0f`
- Vendored location: `third_party/Pi3/`
- Included files: `pi3/`, `LICENSE`, `README.md`, `requirements.txt`, `requirements_demo.txt`
- Vendored content: source code and metadata only
- Excluded content: model checkpoints / weights
- License: see `third_party/Pi3/LICENSE`

## Notes

- The vendored copies are kept under their original licenses.
- `roomtours_gen` vendors code, not pretrained weights.
- No model weights are redistributed inside `roomtours_gen/`.
- Pi3 checkpoints are expected to be obtained separately at runtime by the upstream loading code or by user-managed environments.
- `roomtours_gen` scripts import these vendored copies directly, so this directory is self-contained once the Python dependencies from `requirements.txt` or `environment.yml` are installed.
