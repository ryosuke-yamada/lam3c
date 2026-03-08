#!/usr/bin/env python3
"""
Extract ScanNet linear head checkpoint from a full fine-tuning checkpoint.

Input checkpoint example keys:
- module.seg_head.weight
- module.seg_head.bias

Output checkpoint format (compatible with demo/2_sem_seg.py):
{
  "config": {
    "backbone_out_channels": <int>,
    "num_classes": <int>
  },
  "state_dict": {
    "seg_head.weight": <tensor>,
    "seg_head.bias": <tensor>
  }
}
"""

import argparse
import os

import torch


def _extract_head_state_dict(raw_state_dict):
    candidates = [
        ("module.seg_head.weight", "module.seg_head.bias"),
        ("seg_head.weight", "seg_head.bias"),
    ]
    for w_key, b_key in candidates:
        if w_key in raw_state_dict and b_key in raw_state_dict:
            return {
                "seg_head.weight": raw_state_dict[w_key],
                "seg_head.bias": raw_state_dict[b_key],
            }
    raise KeyError(
        "Could not find seg_head keys in state_dict. "
        "Expected one of {module.seg_head.*, seg_head.*}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract linear-probe head checkpoint for LAM3C demo."
    )
    parser.add_argument("--input", required=True, help="Path to full checkpoint .pth")
    parser.add_argument("--output", required=True, help="Path to output head .pth")
    args = parser.parse_args()

    ckpt = torch.load(args.input, map_location="cpu", weights_only=False)
    if "state_dict" not in ckpt:
        raise KeyError("Input checkpoint has no 'state_dict' key.")

    head_state = _extract_head_state_dict(ckpt["state_dict"])
    num_classes, backbone_out_channels = head_state["seg_head.weight"].shape
    out = {
        "config": {
            "backbone_out_channels": int(backbone_out_channels),
            "num_classes": int(num_classes),
        },
        "state_dict": head_state,
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(out, args.output)
    print(f"[LAM3C] Saved linear head checkpoint: {args.output}")
    print(
        f"[LAM3C] config: backbone_out_channels={backbone_out_channels}, "
        f"num_classes={num_classes}"
    )


if __name__ == "__main__":
    main()
