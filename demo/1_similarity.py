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


import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import lam3c
import torch
import torch.nn.functional as F

try:
    import flash_attn
except ImportError:
    flash_attn = None


def use_headless_mode():
    headless_env = os.getenv("LAM3C_HEADLESS")
    if headless_env is not None:
        return headless_env == "1"
    return not (os.getenv("DISPLAY") or os.getenv("WAYLAND_DISPLAY"))


def parse_args():
    parser = argparse.ArgumentParser(description="LAM3C demo: similarity heatmap")
    parser.add_argument(
        "--model-size",
        choices=("base", "large"),
        default=os.getenv("LAM3C_MODEL_SIZE", "large"),
        help="Preset model size used for default local checkpoint path.",
    )
    parser.add_argument(
        "--ckpt",
        default=None,
        help="Optional local backbone checkpoint path. Overrides env/default.",
    )
    parser.add_argument(
        "--hf-repo-id",
        default=os.getenv("LAM3C_HF_REPO_ID", "aist-cvrt/lam3c"),
        help="HuggingFace model repo id.",
    )
    return parser.parse_args()


def get_model_tag(model_source):
    if model_source.startswith("local:"):
        ckpt_path = model_source.split(":", 1)[1]
        return os.path.splitext(os.path.basename(ckpt_path))[0]
    if model_source.startswith("huggingface:"):
        repo_id = model_source.split(":", 1)[1]
        return repo_id.replace("/", "__")
    return model_source.replace(":", "__").replace("/", "__")


def to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.clone().detach().cpu().numpy()
    assert isinstance(x, np.ndarray)
    return x


def get_point_cloud(coord, color=None, verbose=True):
    if not isinstance(coord, list):
        coord = [coord]
        if color is not None:
            color = [color]

    pcd_list = []
    for i in range(len(coord)):
        coord_ = to_numpy(coord[i])
        if color is not None:
            color_ = to_numpy(color[i])
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(coord_)
        pcd.colors = o3d.utility.Vector3dVector(
            np.ones_like(coord_) if color is None else color_
        )
        pcd_list.append(pcd)
    if verbose:
        o3d.visualization.draw_geometries(pcd_list)
    return pcd_list


def get_line_set(coord, line, color=(1.0, 0.0, 0.0), verbose=True):
    coord = to_numpy(coord)
    line = to_numpy(line)
    colors = np.array([color for _ in range(len(line))])
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(coord)
    line_set.lines = o3d.utility.Vector2iVector(line)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    if verbose:
        o3d.visualization.draw_geometries([line_set])
    return line_set


if __name__ == "__main__":
    args = parse_args()
    # set random seed
    lam3c.utils.set_seed(6463323)
    # Load model (prefer local checkpoint before HuggingFace)
    repo_id = args.hf_repo_id
    # Used for local fallback checkpoints and headless output paths.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_ckpt_name = {
        "base": "lam3c_ptv3-base_roomtours49k.pth",
        "large": "lam3c_ptv3-large_roomtours49k.pth",
    }[args.model_size]
    default_ckpt = os.path.join(
        project_root, "weights", default_ckpt_name
    )
    local_ckpt = args.ckpt or os.getenv("LAM3C_LOCAL_CKPT", default_ckpt)
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
    print(f"[LAM3C] Using checkpoint source: {model_source}")
    # Load default data transform pipeline
    transform = lam3c.transform.default()
    # Load data
    data = lam3c.data.load("sample1")
    data.pop("segment200")
    segment = data.pop("segment20")
    data["segment"] = segment  # two kinds of segment exist in ScanNet, only use one
    # Get local view
    coord = data["coord"]
    color = data["color"]
    center = np.array([0.48, 2.12, 0.67])  # sofa closed to the window
    size = 16384
    index = np.argsort(np.sum(np.square(coord - center), axis=-1))[:size]
    # get_point_cloud(coord[index], color[index] / 255)
    global_data = data
    local_data = dict()
    for key in data.keys():
        local_data[key] = data[key][index]
    global_data = transform(global_data)
    local_data = transform(local_data)

    # model forward:
    with torch.inference_mode():
        for key in global_data.keys():
            if isinstance(global_data[key], torch.Tensor):
                global_data[key] = global_data[key].cuda(non_blocking=True)
        for key in local_data.keys():
            if isinstance(local_data[key], torch.Tensor):
                local_data[key] = local_data[key].cuda(non_blocking=True)
        global_point = model(global_data)
        local_point = model(local_data)
        # upcast point feature
        # Point is a structure contains all the information during forward
        for _ in range(2):
            assert "pooling_parent" in global_point.keys()
            assert "pooling_inverse" in global_point.keys()
            parent = global_point.pop("pooling_parent")
            inverse = global_point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, global_point.feat[inverse]], dim=-1)
            global_point = parent
        while "pooling_parent" in global_point.keys():
            assert "pooling_inverse" in global_point.keys()
            parent = global_point.pop("pooling_parent")
            inverse = global_point.pop("pooling_inverse")
            parent.feat = global_point.feat[inverse]
            global_point = parent
        for _ in range(2):
            assert "pooling_parent" in local_point.keys()
            assert "pooling_inverse" in local_point.keys()
            parent = local_point.pop("pooling_parent")
            inverse = local_point.pop("pooling_inverse")
            parent.feat = torch.cat([parent.feat, local_point.feat[inverse]], dim=-1)
            local_point = parent
        while "pooling_parent" in local_point.keys():
            assert "pooling_inverse" in local_point.keys()
            parent = local_point.pop("pooling_parent")
            inverse = local_point.pop("pooling_inverse")
            parent.feat = local_point.feat[inverse]
            local_point = parent

        select_index = [[8706]]  # sofa arm
        target = F.normalize(local_point.feat, p=2, dim=-1)
        refer = F.normalize(global_point.feat, p=2, dim=-1)
        inner_self = target[select_index] @ target.t()
        inner_cross = target[select_index] @ refer.t()

        oral = 0.02
        highlight = 0.1
        reject = 0.5
        cmap = plt.get_cmap("Spectral_r")
        sorted_inner = torch.sort(inner_cross, descending=True)[0]
        oral = sorted_inner[0, int(global_point.offset[0] * oral)]
        highlight = sorted_inner[0, int(global_point.offset[0] * highlight)]
        reject = sorted_inner[0, -int(global_point.offset[0] * reject)]

        inner_self = inner_self - highlight
        inner_self[inner_self > 0] = F.sigmoid(
            inner_self[inner_self > 0] / (oral - highlight)
        )
        inner_self[inner_self < 0] = (
            F.sigmoid(inner_self[inner_self < 0] / (highlight - reject)) * 0.9
        )

        inner_cross = inner_cross - highlight
        inner_cross[inner_cross > 0] = F.sigmoid(
            inner_cross[inner_cross > 0] / (oral - highlight)
        )
        inner_cross[inner_cross < 0] = (
            F.sigmoid(inner_cross[inner_cross < 0] / (highlight - reject)) * 0.9
        )

        matched_index = torch.argmax(inner_cross)

        local_heat_color = cmap(inner_self.squeeze(0).cpu().numpy())[:, :3]
        global_heat_color = cmap(inner_cross.squeeze(0).cpu().numpy())[:, :3]
        # shift local view from global view
        bias = torch.tensor([[-3.5, 1, 0]]).cuda()  # original bias in our paper
        pcds = get_point_cloud(
            coord=[global_point.coord, local_point.coord + bias],
            color=[global_heat_color, local_heat_color],
            verbose=False,
        )
        pcds.append(
            get_line_set(
                coord=torch.cat(
                    [
                        local_point.coord[select_index] + bias,
                        global_point.coord[matched_index.unsqueeze(0)],
                    ]
                ),
                line=np.array([[0, 1]]),
                color=np.array([0, 0, 0]) / 255,
                verbose=False,
            )
        )
        if use_headless_mode():
            output_root = os.path.join(project_root, "outputs", get_model_tag(model_source))
            os.makedirs(output_root, exist_ok=True)
            out_global = os.path.join(output_root, "demo1_similarity_global.ply")
            out_local = os.path.join(output_root, "demo1_similarity_local.ply")
            out_line = os.path.join(output_root, "demo1_similarity_line.ply")
            o3d.io.write_point_cloud(out_global, pcds[0])
            o3d.io.write_point_cloud(out_local, pcds[1])
            o3d.io.write_line_set(out_line, pcds[2])
            print(f"[LAM3C] Headless mode: wrote {out_global}")
            print(f"[LAM3C] Headless mode: wrote {out_local}")
            print(f"[LAM3C] Headless mode: wrote {out_line}")
        else:
            o3d.visualization.draw_geometries(pcds)
