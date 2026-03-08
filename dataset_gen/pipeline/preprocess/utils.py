#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Utility helpers for point-cloud preprocessing."""

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import ConvexHull


def calculate_scale(coords):
    """Return the point-cloud scale as the bounding-box diagonal length."""
    mins = np.min(coords, axis=0)
    maxs = np.max(coords, axis=0)
    diagonal = np.linalg.norm(maxs - mins)
    return float(diagonal)


def calculate_basic_density(coords):
    """Estimate density as point count divided by occupied volume."""
    try:
        hull = ConvexHull(coords)
        volume = hull.volume
        density = len(coords) / volume if volume > 0 else 0
        return density, volume
    except Exception:
        mins = np.min(coords, axis=0)
        maxs = np.max(coords, axis=0)
        volume = np.prod(maxs - mins)
        density = len(coords) / volume if volume > 0 else 0
        return density, volume


def compute_nearest_neighbor_spacing(coords, sample_size=10000):
    """Compute the median 1-NN spacing on a sampled subset."""
    from sklearn.neighbors import NearestNeighbors

    num_points = len(coords)
    if num_points > sample_size:
        indices = np.random.choice(num_points, sample_size, replace=False)
        sample_coords = coords[indices]
    else:
        sample_coords = coords

    nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(sample_coords)
    distances, _ = nbrs.kneighbors(sample_coords)
    nearest_distances = distances[:, 1]
    return float(np.median(nearest_distances))


def analyze_spacing_distribution(coords, sample_size=None):
    """Compute full 1-NN spacing statistics for the given point cloud."""
    from sklearn.neighbors import NearestNeighbors

    num_points = len(coords)
    if num_points == 0:
        return None

    sample_coords = coords
    nbrs = NearestNeighbors(n_neighbors=2, algorithm="kd_tree").fit(sample_coords)
    distances, _ = nbrs.kneighbors(sample_coords)
    nearest_distances = distances[:, 1]

    stats = {
        "mean": float(np.mean(nearest_distances)),
        "median": float(np.median(nearest_distances)),
        "std": float(np.std(nearest_distances)),
        "min": float(np.min(nearest_distances)),
        "max": float(np.max(nearest_distances)),
        "percentiles": {
            "25": float(np.percentile(nearest_distances, 25)),
            "75": float(np.percentile(nearest_distances, 75)),
            "90": float(np.percentile(nearest_distances, 90)),
            "95": float(np.percentile(nearest_distances, 95)),
        },
    }
    return stats


def save_ply(filepath, coords, colors=None, normals=None):
    """Write a point cloud to a binary PLY file."""
    num_points = len(coords)

    dtype_list = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
    ]
    if colors is not None:
        dtype_list += [
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    if normals is not None:
        dtype_list += [
            ("nx", "f4"),
            ("ny", "f4"),
            ("nz", "f4"),
        ]

    vertex_data = np.zeros(num_points, dtype=dtype_list)
    vertex_data["x"] = coords[:, 0]
    vertex_data["y"] = coords[:, 1]
    vertex_data["z"] = coords[:, 2]

    if colors is not None:
        vertex_data["red"] = colors[:, 0]
        vertex_data["green"] = colors[:, 1]
        vertex_data["blue"] = colors[:, 2]

    if normals is not None:
        vertex_data["nx"] = normals[:, 0]
        vertex_data["ny"] = normals[:, 1]
        vertex_data["nz"] = normals[:, 2]

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    element = PlyElement.describe(vertex_data, "vertex")
    PlyData([element], text=False).write(filepath)


def read_ply(filepath):
    """Read coordinates, colors, and normals from a PLY file."""
    plydata = PlyData.read(filepath)
    vertex = plydata["vertex"]

    coords = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)

    colors = None
    if "red" in vertex and "green" in vertex and "blue" in vertex:
        colors = np.column_stack([vertex["red"], vertex["green"], vertex["blue"]]).astype(np.uint8)

    normals = None
    if "nx" in vertex and "ny" in vertex and "nz" in vertex:
        normals = np.column_stack([vertex["nx"], vertex["ny"], vertex["nz"]]).astype(np.float32)

    return coords, colors, normals
