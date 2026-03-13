# LAM3C Pre-training

This document explains how to start LAM3C pre-training using the Pointcept code included in this repository.

## Start Here

1. Run commands from the repository root (`lam3c/`).
2. Install Pointcept dependencies first.
3. Open `configs/lam3c/pretrain/lam3c_v1m1_ptv3_base.py` and check that `data_root` points to your RoomTours-style dataset.

Then run:

```bash
bash tools/train_lam3c.sh \
  configs/lam3c/pretrain/lam3c_v1m1_ptv3_base.py \
  logs/lam3c_pretrain_base
```

## Installation

LAM3C pre-training uses the Pointcept code included under `third_party/pointcept`.

First, follow the official Pointcept installation guide and make sure your environment can import both `pointcept` and `torch_scatter`.

- Official Pointcept installation guide: [here](https://github.com/Pointcept/Pointcept?tab=readme-ov-file#installation)

Recommended installation completion checks:

```bash
python -c "import pointcept; print('pointcept import OK')"
python -c "import torch_scatter; print('torch_scatter import OK')"
```

If installation fails, first re-check Pointcept dependencies, especially sparse operator / extension packages (`torch_scatter`, `spconv`, pointops-related builds).

## Before You Start

Before launching training, verify:

- you are in repo root (`lam3c/`)
- the target config path exists
- the configured `data_root` exists and is correct for your machine (check `configs/lam3c/pretrain/lam3c_v1m1_ptv3_base.py`)
- requested GPU count matches available devices

## More Launch Examples

Optional (PTv3-Large):

```bash
bash tools/train_lam3c.sh \
  configs/lam3c/pretrain/lam3c_v1m1_ptv3_large.py \
  logs/lam3c_pretrain_large
```

To explicitly set GPU count:

```bash
bash tools/train_lam3c.sh \
  configs/lam3c/pretrain/lam3c_v1m1_ptv3_base.py \
  logs/lam3c_pretrain_base \
  1
```

## Training Entrypoint

`tools/train_lam3c.sh` is the recommended entrypoint.

It automatically:

- sets `PYTHONPATH` for `third_party/pointcept` and repo root
- forwards your config to Pointcept trainer
- writes logs/checkpoints to the specified save directory

Argument order:

- first argument: config file
- second argument: save directory
- third argument: number of GPUs (default: `8`)

## Main Configs

- `configs/lam3c/pretrain/lam3c_v1m1_ptv3_base.py` (recommended default)
- `configs/lam3c/pretrain/lam3c_v1m1_ptv3_large.py` (larger-scale option)

## What Success Looks Like

Training is considered started correctly when:

- training logs are printed continuously
- a checkpoint appears under `logs/.../model/`
- the latest checkpoint is usually saved as `logs/.../model/model_last.pth`

## Output Directory

Training outputs are written to the save path passed to `tools/train_lam3c.sh`.

Example:

```text
logs/lam3c_pretrain_base/
├── events.out.tfevents...
└── model/
    ├── model_last.pth
    └── ...
```

## Common Mistakes

- running from a directory other than repo root
- wrong `data_root` in config
- missing Pointcept dependencies (for example `torch_scatter`)
- GPU count argument not matching the actual environment

## Recommended Validation Before Long Runs

Before full multi-GPU jobs, run a short validation:

1. `Config.fromfile(...)` succeeds
2. `build_model(cfg.model)` succeeds
3. `build_dataset(cfg.data.train)` succeeds
4. 1-step trainer run succeeds
5. 5-10 step dry-run completes without NaN/Inf

If all of the above succeed, the setup is usually ready for a longer training run.
