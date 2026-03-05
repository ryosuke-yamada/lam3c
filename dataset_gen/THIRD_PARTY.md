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

## VGGT

- Source repository: local working copy from `3DV/vggt`
- Upstream commit: `a8f9d63398b438e6bca308c3eeb6b99c67a8328a`
- Vendored location: `third_party/vggt/`
- Included files: runtime package subset under `third_party/vggt/vggt/`, plus `LICENSE.txt`, `README.md`, `pyproject.toml`, `requirements.txt`, `requirements_demo.txt`
- Vendored content: source code and metadata only
- Excluded content: model checkpoints / weights
- License: see `third_party/vggt/LICENSE.txt`

## Notes

- The vendored copies are kept under their original licenses.
- `dataset_gen` vendors code, not pretrained weights.
- No model weights are redistributed inside `dataset_gen/`.
- Pi3 and VGGT checkpoints are expected to be obtained separately at runtime by the upstream loading code or by user-managed environments.
- `dataset_gen` scripts import these vendored copies directly, so this directory is self-contained once the virtual environments are created with `./setup.sh`.
