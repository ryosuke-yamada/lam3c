<p align="center">
    <img src="assets/logo.png" width="150" style="margin-bottom: 0.2;"/>
</p>

<p align="center"><strong><font size="6">LAM3C & RoomTours</font></strong></p>
<p align="center"><strong><font size="4">3d <em>sans</em> 3d scans: scalable pre-training from video-generated point clouds</font></strong></p>

---

<p align="center">
  <a href="https://arxiv.org/abs/2512.23042"><img src="https://img.shields.io/badge/arXiv-2512.23042-b31b1b.svg" alt="arXiv"></a>
  <a href="https://arxiv.org/abs/2512.23042"><img src="https://img.shields.io/badge/CVPR-2026-blue.svg" alt="CVPR 2026"></a>
  <a href="#overview"><img src="https://img.shields.io/badge/Website-Project-8A2BE2" alt="Website"></a>
  <!-- <a href="./dataset_gen"><img src="https://img.shields.io/badge/Dataset-RoomTours-009688" alt="Dataset"></a> -->
  <a href="https://huggingface.co/aist-cvrt/lam3c"><img src="https://img.shields.io/badge/HuggingFace-LAM3C-yellow" alt="LAM3C"></a>
</p>


This repository contains the official implementation of LAM3C and the RoomTours pipeline.

![LAM3C scaling results](assets/lam3c_scaling.png)
> ### LAM3C Message
> **The bottleneck of 3D self-supervised learning is not algorithms alone, but the scarcity of 3D data. Turning the vast sea of unlabeled internet videos into 3D point clouds unlocks a scalable source of 3D supervision.**


---

## TL;DR
This paper shows that 3D self-supervised learning can be trained using only video-generated point clouds reconstructed from unlabeled videos, without relying on real 3D scans.

We introduce:

- **RoomTours** – a scalable pipeline that converts unlabeled indoor videos into video-generated point clouds
- **LAM3C** – a 3D self-supervised learning designed to learn robust representations from noisy video-generated point clouds

LAM3C transfers well to indoor semantic and instance segmentation.

---

## News

- Mar 2026: Released RoomTours generation code and demo visualization.
- Feb 2026: LAM3C was accepted to CVPR 2026.


---

## Overview

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Zoo](#pretrained-models)
- [RoomTours Pipeline](./dataset_gen/README.md)
- [Citation](#citation)

---

## Installation

We recommend using `conda` with `environment.yml`:

```bash
# 1) create env from pinned dependencies (PyTorch 2.5.0 + CUDA 12.4)
cd /path/to/lam3c
conda env create -f environment.yml
conda activate lam3c

# 2) install LAM3C package
pip install -e .
```

Optional dependencies:

- `flash-attn` can improve speed, but is optional.
- If `flash-attn` is unavailable, demos and inference still work with `enable_flash=False` fallback.

---

## Quick Start

Let's first begin with simple visualization demos with LAM3C, our pre-trained PointTransformerV3 (PTv3) model.

![LAM3C demo](assets/lam3c_demo.png)

### Requirements

- Conda
- Python 3.10
- PyTorch 2.5.0
- CUDA 12.4
- NVIDIA GPU for CUDA execution


### Visualization

We provide similarity heatmap and PCA visualization demos in the `demo` folder.

```bash
# optional: local checkpoints (useful on offline/HF-restricted environments)
export LAM3C_LOCAL_CKPT="$(pwd)/weights/lam3c_roomtours49k_ptv3-large.infer.pth"
export LAM3C_LOCAL_LINEAR_HEAD_CKPT="$(pwd)/weights/lam3c_linear_prob_head_sc.pth"

# headless mode (recommended on servers such as ABCI)
export LAM3C_HEADLESS=1
export PYTHONPATH=./

python demo/0_pca.py
python demo/1_similarity.py
python demo/2_sem_seg.py  # linear probed head on ScanNet
```

### Inference on custom data

Then, here are instructions to run inference on custom data with LAM3C.

**Data.** Prepare your input data as a dictionary:

```python
# single point cloud
point = {
  "coord": numpy.array,   # (N, 3)
  "color": numpy.array,   # (N, 3)
  "normal": numpy.array,  # (N, 3)
  "segment": numpy.array, # (N,) optional
}

# batched point clouds
point = {
  "coord": numpy.array,   # (N, 3)
  "color": numpy.array,   # (N, 3)
  "normal": numpy.array,  # (N, 3)
  "batch": numpy.array,   # (N,) optional
  "segment": numpy.array, # (N,) optional
}
```

One sample can be loaded by:

```python
point = lam3c.data.load("sample1")
```

**Transform.** The transform pipeline is shared with the Pointcept style. You can build it by:

```python
config = [
    dict(type="CenterShift", apply_z=True),
    dict(
        type="GridSample",
        grid_size=0.02,
        hash_type="fnv",
        mode="train",
        return_grid_coord=True,
        return_inverse=True,
    ),
    dict(type="NormalizeColor"),
    dict(type="ToTensor"),
    dict(
        type="Collect",
        keys=("coord", "grid_coord", "color", "inverse"),
        feat_keys=("coord", "color", "normal"),
    ),
]
transform = lam3c.transform.Compose(config)
```

The above default inference transform can also be obtained by:

```python
transform = lam3c.transform.default()
```

**Model.** Load the pre-trained model by:

```python
# from HuggingFace
# supported models: "lam3c"
# ckpt is cached in ~/.cache/lam3c/ckpt, path can be customized by `download_root`
model = lam3c.model.load("lam3c", repo_id="aist-cvrt/lam3c").cuda()

# or
from lam3c.model import PointTransformerV3
model = PointTransformerV3.from_pretrained("aist-cvrt/lam3c").cuda()

# from local path
model = lam3c.model.load("ckpt/lam3c.pth").cuda()
```

If FlashAttention is not available, load the model with:

```python
custom_config = dict(
    enc_patch_size=[1024 for _ in range(5)],
    enable_flash=False,  # reduce patch size if necessary
)
model = lam3c.model.load("lam3c", repo_id="aist-cvrt/lam3c", custom_config=custom_config).cuda()

# or
from lam3c.model import PointTransformerV3
model = PointTransformerV3.from_pretrained("aist-cvrt/lam3c", **custom_config).cuda()
```

**Inference.** Run inference by:

```python
point = transform(point)
for key in point.keys():
    if isinstance(point[key], torch.Tensor):
        point[key] = point[key].cuda(non_blocking=True)
point = model(point)
```

As LAM3C is a pre-trained encoder-only PTv3, the default output is the hierarchical encoded point cloud feature. Map it back to original scale by:

```python
for _ in range(2):
    assert "pooling_parent" in point.keys()
    assert "pooling_inverse" in point.keys()
    parent = point.pop("pooling_parent")
    inverse = point.pop("pooling_inverse")
    parent.feat = torch.cat([parent.feat, point.feat[inverse]], dim=-1)
    point = parent
while "pooling_parent" in point.keys():
    assert "pooling_inverse" in point.keys()
    parent = point.pop("pooling_parent")
    inverse = point.pop("pooling_inverse")
    parent.feat = point.feat[inverse]
    point = parent

feat = point.feat[point.inverse]
```

---

## Pre-trained Models

Download pre-trained checkpoints from Google Drive (recommended for now) and place them under `weights/`.
HuggingFace links are also kept below and will be updated as files are published.

- `lam3c_roomtours49k_ptv3-large.infer.pth` (inference backbone)  
  [Google Drive](https://drive.google.com/file/d/1hUK7JMZ_eTzFUDUasvJLeQkomD3SIHR3/view?usp=drive_link) | [HuggingFace](https://huggingface.co/aist-cvrt/lam3c)
- `lam3c_linear_prob_head_sc.pth` (ScanNet linear head for `demo/2_sem_seg.py`)  
  [Google Drive](https://drive.google.com/file/d/1hUK7JMZ_eTzFUDUasvJLeQkomD3SIHR3/view?usp=drive_link) | [HuggingFace](https://huggingface.co/aist-cvrt/lam3c)

---

## Citation

If you find our LAM3C work useful, please cite:

```bibtex
@inproceedings{yamada2026lam3c,
  title={3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds},
  author={Ryousuke Yamada and Kohsuke Ide and Yoshihiro Fukuhara and Hirokatsu Kataoka and Gilles Puy and Andrei Bursuc and Yuki M. Asano},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2026}
}
```

---

## Acknowledgment

We thank the ABCI team at AIST for providing computational resources.

---

## License

- **Code:** Apache-2.0.
- **Upstream attribution:** This repository includes/adapts code from Sonata (Apache-2.0).
- **Model weights:** Separate terms may apply (TBD).
- **Dataset / generated data:** Separate terms may apply (TBD).

See `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` for details.
