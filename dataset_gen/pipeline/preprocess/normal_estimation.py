#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal-estimation helpers."""

import numpy as np
import open3d as o3d


def estimate_normals_auto(coords, k=32, chunk_threshold=1000000, verbose=False):
    """Estimate outward-facing normals with Open3D KNN search."""
    if verbose:
        print("=== Normal estimation ===")
        print(f"Points: {len(coords):,}")

    num_points = len(coords)
    method = "chunked" if num_points > chunk_threshold else "standard"

    if verbose:
        print(f"Method: {method}")
        if method == "chunked":
            print(f"Chunk threshold: {chunk_threshold:,}")
        print(f"Running KNN normal estimation (k={k})")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))

    centroid = np.mean(coords, axis=0)
    if verbose:
        print(
            "Orientation reference: "
            f"[{centroid[0]:.3f}, {centroid[1]:.3f}, {centroid[2]:.3f}]"
        )

    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=k))
    pcd.orient_normals_towards_camera_location(camera_location=centroid)
    normals = np.asarray(pcd.normals).astype(np.float32)

    num_zero = np.sum(np.linalg.norm(normals, axis=1) < 1e-6)
    num_nan = np.sum(np.isnan(normals).any(axis=1))
    avg_norm = np.mean(np.linalg.norm(normals, axis=1))
    quality_score = 100.0 * (1.0 - (num_zero + num_nan) / num_points)

    if verbose:
        print(f"Estimated normals: {len(normals):,}")
        print("Quality summary:")
        print(f"  Zero vectors: {num_zero} ({num_zero / num_points * 100:.2f}%)")
        print(f"  NaNs: {num_nan}")
        print(f"  Mean norm length: {avg_norm:.6f}")
        print(f"  Quality score: {quality_score:.1f}%")
        print("=== Normal estimation complete ===\n")

    return normals
