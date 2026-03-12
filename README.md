<p align="center">
    <img src="assets/logo.png" width="400" style="margin-bottom: 0.2;"/>
</p>

<h1 align="center">3D <em>sans</em> 3D Scans: Scalable Pre-training from Video-Generated Point Clouds</h1>


<p align="center">
  <a href="https://arxiv.org/abs/2512.23042"><img src="https://img.shields.io/badge/arXiv-2512.23042-b31b1b.svg" alt="arXiv"></a>
  <a href=""><img src="https://img.shields.io/badge/CVPR-2026-blue.svg" alt="CVPR 2026"></a>
  <a href=""><img src="https://img.shields.io/badge/Website-Project-8A2BE2" alt="Website"></a>
  <a href="https://huggingface.co/aist-cvrt/lam3c"><img src="https://img.shields.io/badge/HuggingFace-LAM3C-yellow" alt="LAM3C"></a>
   <a href=""><img src="https://img.shields.io/badge/Dataset-RoomTours-009688" alt="Dataset"></a>
</p>

---

This repository contains the official implementation of LAM3C and the RoomTours pipeline.

> ### LAM3C Message
> **The bottleneck of 3D self-supervised learning is not algorithms alone, but the scarcity of 3D data. Turning the vast sea of unlabeled internet videos into 3D point clouds unlocks a scalable source of 3D supervision.**

![LAM3C scaling results](assets/lam3c_scaling.png)


## TL;DR
This paper shows that 3D self-supervised learning can be trained using only video-generated point clouds reconstructed from unlabeled videos, without relying on real 3D scans. We introduce: **RoomTours** is a scalable pipeline that converts unlabeled indoor videos into video-generated point clouds. **LAM3C** is a 3D self-supervised learning designed to learn robust representations from noisy video-generated point clouds.LAM3C transfers well to indoor semantic and instance segmentation.


## News

- Mar 2026: Released RoomTours generation code and inference demo visualization.
- Feb 2026: LAM3C was accepted to CVPR 2026 (main track).


## Overview

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Zoo](#pretrained-models)
- [RoomTours Pipeline](./roomtours_gen/README.md)
- [Citation](#citation)


## Requirements

- Conda
- Python 3.9
- PyTorch 2.5.0
- CUDA 12.4
- NVIDIA GPU for CUDA execution

## Installation

This repo provide two ways of installation: **standalone mode** and **package mode**.

- The **standalone mode** is recommended for users who want to use the code for quick inference and visualization. We provide a most easy way to install the environment by using `conda` environment file. The whole environment including `cuda` and `pytorch` can be easily installed by running the following command:
  ```bash
  # Create and activate conda environment named as 'lam3c'

  # run `unset CUDA_PATH` if you have installed cuda in your local environment
  conda env create -f environment.yml --verbose
  conda activate lam3c

  # if torch-scatter installation fails, install explicitly from the PyG wheel index
  pip install --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-2.5.0+cu124.html

  # optional: install FlashAttention after torch is available in this env
  # (required on some systems to avoid pip build-isolation issues)
  pip install --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git
  ```

  *FlashAttention is optional. If installation fails in your environment, you can skip it and use the fallback path (see Model section in [Quick Start](#quick-start)).*

- The **package mode** is recommended for users who want to inject our model into their own codebase. We provide a `setup.py` file for installation. You can install the package by running the following command:
  ```bash
  # Ensure Cuda and Pytorch are already installed in your local environment

  # CUDA_VERSION: cuda version of local environment (e.g., 124), check by running 'nvcc --version'
  # TORCH_VERSION: torch version of local environment (e.g., 2.5.0), check by running 'python -c "import torch; print(torch.__version__)"'
  pip install spconv-cu${CUDA_VERSION}
  pip install --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-{TORCH_VERSION}+cu${CUDA_VERSION}.html
  # optional:
  pip install --no-build-isolation git+https://github.com/Dao-AILab/flash-attention.git
  pip install huggingface_hub timm

  # (optional, or directly copy the lam3c folder to your project)
  python setup.py install
  ```
  Additionally, for running our **demo code**, the following packages are also required:
  ```bash
  pip install open3d fast_pytorch_kmeans psutil numpy==1.26.4  # currently, open3d does not support numpy 2.x
  ```


## Quick Start

Let's first begin with simple visualization demos with LAM3C, our pre-trained PointTransformerV3 (PTv3) model.

### Visualization
![LAM3C demo](assets/lam3c_demo.png)

We provide demos for PCA feature visualization, similarity heatmaps, semantic segmentation, and batched forward inference in the `demo` folder.

```bash
# 1) activate environment
conda activate lam3c
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH=./

# 2) (optional) headless output on servers
# (writes .ply to outputs/<pretrained-model-name>/)
# export LAM3C_HEADLESS=1

# 3-A) run with default HuggingFace checkpoints (large)
python demo/0_pca.py
python demo/1_similarity.py
python demo/2_sem_seg.py
python demo/3_batch_forward.py

# 3-B) run with local preset checkpoints (switchable model size via --model-size)
python demo/0_pca.py --model-size base
python demo/1_similarity.py --model-size base
python demo/2_sem_seg.py --model-size base
python demo/3_batch_forward.py --model-size base

# (optional) run the same demos with local large checkpoints
python demo/0_pca.py --model-size large
python demo/1_similarity.py --model-size large
python demo/2_sem_seg.py --model-size large
python demo/3_batch_forward.py --model-size large

# 3-C) run with custom checkpoints
python demo/0_pca.py --ckpt /path/to/custom_backbone.infer.pth
python demo/2_sem_seg.py --ckpt /path/to/custom_backbone.infer.pth --head-ckpt /path/to/custom_linear_head.pth
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

`sample1` is a Pointcept-provided demo sample downloaded from the Hugging Face dataset repository `pointcept/demo` (i.e., not user-collected data in this repository).

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



## Model Zoo

Download pre-trained checkpoints from Google Drive (recommended for now) and place them under `weights/`.
HuggingFace links are also kept below and will be updated as files are published.


- `lam3c_roomtours49k_ptv3-large.infer.pth` (inference backbone)  
  [Google Drive](https://drive.google.com/file/d/1hUK7JMZ_eTzFUDUasvJLeQkomD3SIHR3/view?usp=drive_link) | [HuggingFace](https://huggingface.co/aist-cvrt/lam3c)
- `lam3c_linear_prob_head_sc.pth` (ScanNet linear head for `demo/2_sem_seg.py`)  
  [Google Drive](https://drive.google.com/file/d/1hUK7JMZ_eTzFUDUasvJLeQkomD3SIHR3/view?usp=drive_link) | [HuggingFace](https://huggingface.co/aist-cvrt/lam3c)


## RoomTours Pipeline

RoomTours converts unlabeled indoor videos into training-ready point clouds for LAM3C pre-training.
The pipeline includes video download, scene segmentation, Pi3 reconstruction, and point-cloud preprocessing.
For setup and commands, see [`roomtours_gen/README.md`](./roomtours_gen/README.md).

![RoomTours](assets/roomtours.png)


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
```bib
@inproceedings{wu2025sonata,
    title={Sonata: Self-Supervised Learning of Reliable Point Representations},
    author={Wu, Xiaoyang and DeTone, Daniel and Frost, Duncan and Shen, Tianwei and Xie, Chris and Yang, Nan and Engel, Jakob and Newcombe, Richard and Zhao, Hengshuang and Straub, Julian},
    booktitle={CVPR},
    year={2025}
}
```
```bib
@misc{pointcept2023,
    title={Pointcept: A Codebase for Point Cloud Perception Research},
    author={Pointcept Contributors},
    howpublished={\url{https://github.com/Pointcept/Pointcept}},
    year={2023}
}
```

## License

- **Code:** MIT License.
- **LAM3C weights:** Creative Commons BY-NC 4.0 (free for research/education, **no commercial use**).
- **RoomTours / generated data:** Creative Commons BY-NC 4.0 (free for research/education, **no commercial use**).
- **Demo sample data:** loaded from Pointcept's `pointcept/demo` dataset repository on Hugging Face.
- **Upstream attribution:** This repository includes/adapts code from Sonata.

See `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES.md` for details.
