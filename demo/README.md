# Demo and Custom Inference Notes

This page collects practical notes for running the demos and using LAM3C on custom data.

## Inference on custom data

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

## Credits

The demo implementations in `demo/` are built by adapting and improving code from [Sonata](https://github.com/facebookresearch/sonata) and [Pointcept](https://github.com/Pointcept/Pointcept).