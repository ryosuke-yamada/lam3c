# LAM3C & RoomTours

[![arXiv](https://img.shields.io/badge/arXiv-2512.23042-b31b1b.svg)](https://arxiv.org/abs/2512.23042)
[![CVPR 2026](https://img.shields.io/badge/CVPR-2026-blue.svg)](https://arxiv.org/abs/2512.23042)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](#license)

Official repository for **LAM3C (Laplacian-Aware Multi-level 3D Clustering with Sinkhorn-Knopp)**  
and **RoomTours**, a pipeline for generating large-scale point clouds from unlabeled videos.

**Key idea:** 3D self-supervised learning can be trained entirely from video-generated point clouds reconstructed from unlabeled videos.

<!-- **Paper:** [3D sans 3D Scans: Scalable Pre-training from Video-Generated Point Clouds　(CVPR 2026 main track)](https://arxiv.org/pdf/2512.23042) -->

[[ **Project Page** ]](#overview) [[**Pre-trained Models**]](#pretrained-models) [[ **Paper** ]](https://arxiv.org/pdf/2512.23042) [[ **Bib** ]](#citation)

![LAM3C scaling results](assets/lam3c_scaling.png)
> ### LAM3C Message
> **The bottleneck of 3D self-supervised learning is not algorithms alone, but the scarcity of 3D data.**  
> **Turning the vast sea of unlabeled internet videos into 3D point clouds unlocks a scalable source of 3D supervision.**


---

## TL;DR
This paper shows that 3D self-supervised learning can be trained using only video-generated point clouds reconstructed from unlabeled videos, without relying on real 3D scans.

We introduce:

- **RoomTours** – a scalable pipeline that converts unlabeled indoor videos into video-generated point clouds
- **LAM3C** – a 3D self-supervised learning framework designed to learn robust representations from noisy video-generated point clouds

LAM3C transfers well to indoor semantic and instance segmentation.

---

## News

- Mar 2026: Released pre-training code with Pointcept support and demo visualization.
- Feb 2026: LAM3C was accepted to CVPR 2026.


---

## Overview

This repository contains the official implementation of LAM3C and the RoomTours pipeline.

LAM3C is a masked 3D self-supervised learning framework designed to learn robust representations from video-generated point clouds reconstructed from unlabeled videos.

Main components:

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Inference](#inference-on-custom-data)
- [Pre-trained Models](#pretrained-models)
- [Citation](#citation)

---

## Installation

Coming soon...

---

## Quick Start

Let's first begin with simple visualization demos with LAM3C, our pre-trained PointTransformerV3 (PTv3) model.

![LAM3C demo](assets/lam3c_demo.png)

### Requirements

- Python >= 3.9
- PyTorch >= 2.0
- CUDA >= 11.8

Additional dependencies are installed automatically via pip.


### Visualization

We provide similarity heatmap and PCA visualization demos in the `demo` folder.

```bash
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

Pre-trained model weights and download links will be listed here.

- `LAM3C-Base` - PTv3 backbone ([HuggingFace](https://huggingface.co/aist-cvrt/lam3c))
- `LAM3C-Large` - larger PTv3 backbone ([HuggingFace](https://huggingface.co/aist-cvrt/lam3c))

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