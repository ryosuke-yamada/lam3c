# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modifications Copyright (c) 2026 Ryousuke Yamada.


import os
import numpy as np
import open3d as o3d
import lam3c
import torch
import torch.nn as nn

try:
    import flash_attn
except ImportError:
    flash_attn = None


def use_headless_mode():
    headless_env = os.getenv("LAM3C_HEADLESS")
    if headless_env is not None:
        return headless_env == "1"
    return not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


# ScanNet Meta data
VALID_CLASS_IDS_20 = (
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    14,
    16,
    24,
    28,
    33,
    34,
    36,
    39,
)


CLASS_LABELS_20 = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
)

SCANNET_COLOR_MAP_20 = {
    0: (0.0, 0.0, 0.0),
    1: (174.0, 199.0, 232.0),
    2: (152.0, 223.0, 138.0),
    3: (31.0, 119.0, 180.0),
    4: (255.0, 187.0, 120.0),
    5: (188.0, 189.0, 34.0),
    6: (140.0, 86.0, 75.0),
    7: (255.0, 152.0, 150.0),
    8: (214.0, 39.0, 40.0),
    9: (197.0, 176.0, 213.0),
    10: (148.0, 103.0, 189.0),
    11: (196.0, 156.0, 148.0),
    12: (23.0, 190.0, 207.0),
    14: (247.0, 182.0, 210.0),
    15: (66.0, 188.0, 102.0),
    16: (219.0, 219.0, 141.0),
    17: (140.0, 57.0, 197.0),
    18: (202.0, 185.0, 52.0),
    19: (51.0, 176.0, 203.0),
    20: (200.0, 54.0, 131.0),
    21: (92.0, 193.0, 61.0),
    22: (78.0, 71.0, 183.0),
    23: (172.0, 114.0, 82.0),
    24: (255.0, 127.0, 14.0),
    25: (91.0, 163.0, 138.0),
    26: (153.0, 98.0, 156.0),
    27: (140.0, 153.0, 101.0),
    28: (158.0, 218.0, 229.0),
    29: (100.0, 125.0, 154.0),
    30: (178.0, 127.0, 135.0),
    32: (146.0, 111.0, 194.0),
    33: (44.0, 160.0, 44.0),
    34: (112.0, 128.0, 144.0),
    35: (96.0, 207.0, 209.0),
    36: (227.0, 119.0, 194.0),
    37: (213.0, 92.0, 176.0),
    38: (94.0, 106.0, 211.0),
    39: (82.0, 84.0, 163.0),
    40: (100.0, 85.0, 144.0),
}

CLASS_COLOR_20 = [SCANNET_COLOR_MAP_20[id] for id in VALID_CLASS_IDS_20]


class SegHead(nn.Module):
    def __init__(self, backbone_out_channels, num_classes):
        super(SegHead, self).__init__()
        self.seg_head = nn.Linear(backbone_out_channels, num_classes)

    def forward(self, x):
        return self.seg_head(x)


if __name__ == "__main__":
    # set random seed
    lam3c.utils.set_seed(24525867)
    # Load model (prefer local checkpoint before HuggingFace)
    repo_id = os.getenv("LAM3C_HF_REPO_ID", "aist-cvrt/lam3c")
    # Used for local fallback checkpoints and headless output paths.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_ckpt = os.path.join(
        project_root, "weights", "lam3c_roomtours49k_ptv3-large.infer.pth"
    )
    local_ckpt = os.getenv("LAM3C_LOCAL_CKPT", default_ckpt)
    custom_config = None
    if flash_attn is None:
        custom_config = dict(
            enc_patch_size=[1024 for _ in range(5)],  # reduce patch size if necessary
            enable_flash=False,
        )

    try:
        model = lam3c.load("lam3c", repo_id=repo_id, custom_config=custom_config).cuda()
        model_source = f"huggingface:{repo_id}"
        print(f"[LAM3C] Loaded checkpoint from HuggingFace repo: {repo_id}")
    except Exception as e:
        if not os.path.isfile(local_ckpt):
            raise RuntimeError(
                f"Failed to load from HuggingFace ({repo_id}) and local checkpoint "
                f"not found at {local_ckpt}"
            ) from e
        print(f"[LAM3C] Falling back to local checkpoint: {local_ckpt}")
        model = lam3c.load(local_ckpt, custom_config=custom_config).cuda()
        model_source = f"local:{local_ckpt}"
    print(f"[LAM3C] Using backbone checkpoint source: {model_source}")
    # Load linear probing seg head
    default_head_ckpt = os.path.join(project_root, "weights", "lam3c_linear_prob_head_sc.pth")
    local_head_ckpt = os.getenv("LAM3C_LOCAL_LINEAR_HEAD_CKPT", default_head_ckpt)
    try:
        ckpt = lam3c.load(
            "lam3c_linear_prob_head_sc", repo_id=repo_id, ckpt_only=True
        )
        head_source = f"huggingface:{repo_id}"
        print(f"[LAM3C] Loaded linear head from HuggingFace repo: {repo_id}")
    except Exception as e:
        if not os.path.isfile(local_head_ckpt):
            raise RuntimeError(
                f"Failed to load linear head from HuggingFace ({repo_id}) and local "
                f"head checkpoint not found at {local_head_ckpt}"
            ) from e
        print(f"[LAM3C] Falling back to local linear head: {local_head_ckpt}")
        ckpt = lam3c.load(local_head_ckpt, ckpt_only=True)
        head_source = f"local:{local_head_ckpt}"
    print(f"[LAM3C] Using linear head source: {head_source}")
    seg_head = SegHead(**ckpt["config"]).cuda()
    seg_head.load_state_dict(ckpt["state_dict"])
    # Load default data transform pipeline
    transform = lam3c.transform.default()
    # Load data
    point = lam3c.data.load("sample1")
    point.pop("segment200")
    segment = point.pop("segment20")
    point["segment"] = segment  # two kinds of segment exist in ScanNet, only use one
    original_coord = point["coord"].copy()
    point = transform(point)

    # Inference
    model.eval()
    seg_head.eval()
    with torch.inference_mode():
        for key in point.keys():
            if isinstance(point[key], torch.Tensor):
                point[key] = point[key].cuda(non_blocking=True)
        # model forward:
        point = model(point)
        raw_feat = point.feat
        raw_dim = int(raw_feat.shape[-1])
        top_native_feat = None
        while "pooling_parent" in point.keys():
            assert "pooling_inverse" in point.keys()
            parent = point.pop("pooling_parent")
            inverse = point.pop("pooling_inverse")
            parent_native_feat = parent.feat
            if "pooling_parent" not in parent.keys():
                top_native_feat = parent_native_feat
            parent.feat = torch.cat([parent_native_feat, point.feat[inverse]], dim=-1)
            point = parent
        upcast_feat = point.feat
        upcast_dim = int(upcast_feat.shape[-1])
        head_in_dim = int(seg_head.seg_head.in_features)
        candidates = {
            upcast_dim: upcast_feat,
            raw_dim: raw_feat,
        }
        if top_native_feat is not None:
            candidates[int(top_native_feat.shape[-1])] = top_native_feat
        if head_in_dim in candidates:
            feat = candidates[head_in_dim]
        else:
            available_dims = sorted(candidates.keys())
            raise RuntimeError(
                "Linear head input mismatch: "
                f"head expects {head_in_dim}, but available feature dims are "
                f"{available_dims}."
            )
        seg_logits = seg_head(feat)
        pred = seg_logits.argmax(dim=-1).data.cpu().numpy()
        color = np.array(CLASS_COLOR_20)[pred]

    # Visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point.coord.cpu().detach().numpy())
    pcd.colors = o3d.utility.Vector3dVector(color / 255)
    if use_headless_mode():
        os.makedirs(os.path.join(project_root, "outputs"), exist_ok=True)
        out_path = os.path.join(project_root, "outputs", "demo2_sem_seg.ply")
        o3d.io.write_point_cloud(out_path, pcd)
        print(f"[LAM3C] Headless mode: wrote {out_path}")
    else:
        o3d.visualization.draw_geometries([pcd])
