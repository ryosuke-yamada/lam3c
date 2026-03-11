# -*- coding: utf-8 -*-
"""Utilities for matching spacing and density to a reference distribution."""

from typing import Dict, Tuple, Optional, Any
import numpy as np
from sklearn.neighbors import NearestNeighbors


# =========================================================
# =========================================================

def calculate_basic_density(coords: np.ndarray) -> Tuple[float, float]:
    """

    density = N / volume

    Returns:
        density: float
        volume: float
    """
    if coords.size == 0:
        return 0.0, 0.0

    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    bbox = max_xyz - min_xyz
    volume = float(np.prod(bbox))
    if volume <= 0:
        return float(len(coords)), 0.0

    density = len(coords) / volume
    return float(density), float(volume)


def analyze_spacing_distribution(
    coords: np.ndarray,
    name: str = "",
    k: int = 2,
    sample_size: int = None
) -> Optional[Dict[str, Any]]:
    """


    Args:
        coords: (N, 3)




    Returns:
        {
          'mean': ...,
          'median': ...,
          'std': ...,
          'percentiles': np.ndarray(...),

        }

    """
    N = len(coords)
    if N < 2:
        return None

    sample_coords = coords

    nbrs = NearestNeighbors(n_neighbors=2, algorithm='kd_tree').fit(sample_coords)
    distances, _ = nbrs.kneighbors(sample_coords)
    nn_distances = distances[:, 1]

    stats = {
        "mean": float(np.mean(nn_distances)),
        "median": float(np.median(nn_distances)),
        "std": float(np.std(nn_distances)),
        "percentiles": np.percentile(nn_distances, [10, 25, 50, 75, 90, 95, 99]),
        "distances": nn_distances,
    }
    return stats


# =========================================================
# =========================================================

def perfect_spatial_sampling(
    coords: np.ndarray,
    colors: Optional[np.ndarray],
    normals: Optional[np.ndarray],
    target_num_points: int,
    grid_size: float,
    target_spacing_stats: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """









    """

    N = len(coords)
    if N <= target_num_points:
        return coords, colors, normals

    min_xyz = coords.min(axis=0)
    # voxel index (N, 3)
    voxel_idx = np.floor((coords - min_xyz) / grid_size).astype(np.int64)

    max_dim = voxel_idx.max() + 1
    voxel_keys = voxel_idx[:, 0] + voxel_idx[:, 1] * max_dim + voxel_idx[:, 2] * (max_dim ** 2)
    
    _, unique_indices = np.unique(voxel_keys, return_index=True)
    rep_indices = unique_indices

    if len(rep_indices) > target_num_points:
        sel = np.random.choice(rep_indices, size=target_num_points, replace=False)
        rep_indices = sel

    if len(rep_indices) < target_num_points:
        remain = target_num_points - len(rep_indices)
        all_indices = np.arange(N)
        mask = np.ones(N, dtype=bool)
        mask[rep_indices] = False
        candidates = all_indices[mask]
        if len(candidates) > 0:
            add = np.random.choice(candidates, size=min(remain, len(candidates)), replace=False)
            rep_indices = np.concatenate([rep_indices, add], axis=0)

    rep_coords = coords[rep_indices]
    rep_colors = colors[rep_indices] if colors is not None else None
    rep_normals = normals[rep_indices] if normals is not None else None

    return rep_coords, rep_colors, rep_normals


# =========================================================
# =========================================================

def perfect_spacing_sampling(
    coords: np.ndarray,
    colors: Optional[np.ndarray],
    normals: Optional[np.ndarray],
    target_spacing_stats: Dict[str, float],
    target_num_points: int,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """








    """
    N = len(coords)
    if N <= target_num_points:
        return coords, colors, normals

    target_median = target_spacing_stats["median"]
    perfect_grid_size = target_median * 0.7

    sampled_coords, sampled_colors, sampled_normals = perfect_spatial_sampling(
        coords,
        colors,
        normals,
        target_num_points,
        perfect_grid_size,
        target_spacing_stats,
    )

    return sampled_coords, sampled_colors, sampled_normals


# =========================================================
# =========================================================

def iterative_spacing_refinement(
    coords: np.ndarray,
    colors: Optional[np.ndarray],
    normals: Optional[np.ndarray],
    target_spacing_stats: Dict[str, float],
    max_iterations: int = 5,
    min_points_ratio: float = 0.70,
    verbose: bool = True
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """





    """
    if verbose:
        print(f"Starting iterative spacing refinement (up to {max_iterations})")

    current_coords = coords.copy()
    current_colors = colors.copy() if colors is not None else None
    current_normals = normals.copy() if normals is not None else None

    initial_num_points = len(coords)
    min_allowed_points = int(initial_num_points * min_points_ratio)

    if verbose:
        print(f"Initial points: {initial_num_points:,}")
        print(f"Minimum point threshold: {min_allowed_points:,} ({min_points_ratio:.0%})")

    target_mean = target_spacing_stats["mean"]

    initial_spacing = analyze_spacing_distribution(coords)
    if initial_spacing:
        current_mean = initial_spacing['mean']
        spacing_ratio = current_mean / target_mean
        
        if spacing_ratio < 1.0:
            grid_size_factor = (1.0 / spacing_ratio) ** 0.25
            grid_size_factor = min(1.25, max(0.95, grid_size_factor))
            if verbose:
                print(f"Initial spacing ratio: {spacing_ratio:.3f} -> Initial factor: {grid_size_factor:.3f} (mild optimization)")
        else:
            grid_size_factor = spacing_ratio ** 0.25
            grid_size_factor = min(1.05, max(0.8, grid_size_factor))
            if verbose:
                print(f"Initial spacing ratio: {spacing_ratio:.3f} -> Initial factor: {grid_size_factor:.3f} (mild optimization)")
    else:
        grid_size_factor = 1.0

    best_error = float('inf')
    best_coords = current_coords.copy()
    best_colors = current_colors.copy() if current_colors is not None else None
    best_normals = current_normals.copy() if current_normals is not None else None
    best_iteration = 0
    
    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")
        
        current_spacing = analyze_spacing_distribution(current_coords)
        if current_spacing is None:
            break
        
        current_mean = current_spacing['mean']
        
        if verbose:
            print(f"Current mean spacing: {current_mean:.6f}")
            print(f"Target mean spacing: {target_mean:.6f}")
        
        relative_error = abs(current_mean - target_mean) / target_mean
        current_num_points = len(current_coords)
        
        if verbose:
            print(f"Relative error: {relative_error:.4f} (best: {best_error:.4f})")
            print(f"Current points: {current_num_points:,} (minimum threshold: {min_allowed_points:,})")
        
        if current_num_points < min_allowed_points:
            if verbose:
                print(f"Warning: Point count fell below the minimum threshold ({current_num_points:,} < {min_allowed_points:,})")
                print(f"  Using best result from iteration {best_iteration}")
            break
        
        if relative_error < 0.02:
            if verbose:
                print(f"Reached the tolerance threshold (error={relative_error:.4f} < 0.02) -> stopping immediately")
                print(f"   Using the current result: {current_num_points:,} points")
            return current_coords, current_colors, current_normals
        
        if relative_error < best_error:
            best_error = relative_error
            best_coords = current_coords.copy()
            best_colors = current_colors.copy() if current_colors is not None else None
            best_normals = current_normals.copy() if current_normals is not None else None
            best_iteration = iteration + 1
            if verbose:
                print(f"  Updated best result: Error {best_error:.4f} (iteration {best_iteration})")
        else:
            if verbose:
                print(f"  Warning: stopping because the error increased ({relative_error:.4f} > {best_error:.4f})")
                print(f"  Using best result from iteration {best_iteration}")
            break
        
        if current_mean > target_mean:
            grid_size_factor *= 0.95
        else:
            grid_size_factor *= 1.05

        grid_size_factor = min(2.0, max(0.5, grid_size_factor))

        grid_size = target_spacing_stats["median"] * grid_size_factor
        
        current_coords, current_colors, current_normals = perfect_spatial_sampling(
            current_coords,
            current_colors,
            current_normals,
            len(current_coords),
            grid_size,
            target_spacing_stats,
        )
        if verbose:
            print(f"Updated grid size: {grid_size:.6f} (factor={grid_size_factor:.3f})")
            print(f"Iteration {iteration + 1} complete: {len(current_coords):,} points")

    if verbose:
        print(f"Final result: using result from iteration {best_iteration} (error={best_error:.4f})")
    return best_coords, best_colors, best_normals

