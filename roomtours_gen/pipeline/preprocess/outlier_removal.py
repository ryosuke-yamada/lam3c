#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Conservative outlier removal utilities for reconstructed point clouds."""

import numpy as np
import open3d as o3d
from typing import Tuple, Optional, Dict
from sklearn.neighbors import NearestNeighbors
import warnings


def estimate_spacing_scale(coords_sample: np.ndarray, k: int = 8) -> Tuple[float, float]:
    """

    
    Args:


    
    Returns:


    """
    if len(coords_sample) < k + 1:
        return 0.01, 0.01
    
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm='auto').fit(coords_sample)
    dists, _ = nbrs.kneighbors(coords_sample)
    
    nn1 = dists[:, 1]
    s_median = float(np.median(nn1))
    s_mean = float(np.mean(nn1))
    
    return s_median, s_mean


def enforce_removal_guard(N_before: int, N_after: int, hard_cap: float = 0.10, 
                         stage_name: str = "") -> None:
    """

    
    Args:




    
    Raises:

    """
    if N_before == 0:
        return
    
    removed = N_before - N_after
    ratio = removed / N_before
    
    if ratio > hard_cap:
        msg = f"Too many points removed in {stage_name}: {removed}/{N_before} ({ratio:.1%}). "
        msg += f"Hard cap is {hard_cap:.1%}. Loosen parameters and retry."
        raise RuntimeError(msg)


def light_preclean_simple(coords: np.ndarray,
                          colors: Optional[np.ndarray] = None,
                          s_med: Optional[float] = None,
                          radius_factor: float = 2.8,
                          r_min_pts: int = 2,
                          k_sor: int = 24,
                          std_ratio: float = 1.2,
                          hard_cap: float = 0.10,
                          verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """

    
    Args:









    
    Returns:



    """
    N_original = len(coords)
    
    if verbose:
        print(f"=== Light pre-clean start ===")
        print(f"Input points: {N_original:,}")
    
    # --- Spacing estimate ---
    if s_med is None:
        M = min(len(coords), 300_000)
        if len(coords) > M:
            idx = np.random.choice(len(coords), M, replace=False)
            sample = coords[idx]
        else:
            sample = coords
        s_med, s_mean = estimate_spacing_scale(sample)
    if verbose:
            print(f"Spacing estimate: median={s_med:.6f}, mean={s_mean:.6f}")
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(coords.astype(np.float64))
    
    if colors is not None:
        if colors.dtype == np.uint8:
            colors_normalized = colors.astype(np.float64) / 255.0
        else:
            colors_normalized = colors.astype(np.float64)
            if colors_normalized.max() > 1.0:
                colors_normalized = colors_normalized / 255.0
        pcd.colors = o3d.utility.Vector3dVector(colors_normalized)
    
    # --- Pass 1: radius-based removal(R-OR) ---
    r = radius_factor * s_med
    N0 = len(pcd.points)
    
    if verbose:
        print(f"\nPass 1: radius-based removal")
        print(f"  Radius: {r:.6f} (= {radius_factor} x spacing)")
        print(f"  Minimum neighbors: {r_min_pts}")
    
    pcd, mask1 = pcd.remove_radius_outlier(nb_points=r_min_pts, radius=r)
    N1 = len(pcd.points)
    removed1 = N0 - N1
    
    if verbose:
        print(f"  Removed: {removed1:,} / {N0:,} ({removed1/N0*100:.2f}%)")
    
    try:
        enforce_removal_guard(N0, N1, hard_cap * 0.6, "R-OR")
    except RuntimeError as e:
        if verbose:
            print(f"  Warning: {e}")
        warnings.warn(str(e))
    
    if verbose:
        print(f"\nPass 2: statistical outlier removal (SOR)")
        print(f"  Neighbors: {k_sor}")
        print(f"  Std-ratio: {std_ratio}")
    
    pcd, mask2 = pcd.remove_statistical_outlier(nb_neighbors=k_sor, std_ratio=std_ratio)
    N2 = len(pcd.points)
    removed2 = N1 - N2
    
    if verbose:
        print(f"  Removed: {removed2:,} / {N1:,} ({removed2/N1*100:.2f}%)")
    
    total_removed = N_original - N2
    try:
        enforce_removal_guard(N_original, N2, hard_cap, "Total (R-OR + SOR)")
    except RuntimeError as e:
        if verbose:
            print(f"  Warning: {e}")
        warnings.warn(str(e))
    
    coords_out = np.asarray(pcd.points, dtype=np.float32)
    colors_out = None
    if pcd.has_colors():
        colors_out = np.asarray(pcd.colors, dtype=np.float32)
        if colors is not None and colors.dtype == np.uint8:
            colors_out = (colors_out * 255).astype(np.uint8)
    
    info = {
        's_med': s_med,
        'removed_ratio': total_removed / N_original,
        'removed_r_or': removed1,
        'removed_sor': removed2,
        'N_original': N_original,
        'N_final': N2,
    }
    
    if verbose:
        print(f"\n=== Processing complete ===")
        print(f"Final points: {N2:,} / {N_original:,}")
        print(f"Reduction ratio: {info['removed_ratio']*100:.2f}%")
    
    return coords_out, colors_out, info


def compute_spatial_grid(coords: np.ndarray, cell_size: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """

    
    Args:

        cell_size: Cell size
    
    Returns:




    """
    min_xyz = coords.min(axis=0)
    max_xyz = coords.max(axis=0)
    
    cell_indices = np.floor((coords - min_xyz) / cell_size).astype(np.int32)
    
    n_cells = np.ceil((max_xyz - min_xyz) / cell_size).astype(np.int32)
    
    cell_indices = np.clip(cell_indices, 0, n_cells - 1)
    
    return cell_indices, min_xyz, max_xyz, n_cells


def light_preclean_grid(coords: np.ndarray,
                        colors: Optional[np.ndarray] = None,
                        s_med: Optional[float] = None,
                        cell_factor: float = 20.0,
                        margin_cells: int = 1,
                        radius_factor: float = 2.8,
                        r_min_pts: int = 2,
                        k_sor: int = 24,
                        std_ratio: float = 1.2,
                        hard_cap: float = 0.10,
                        verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """

    
    Args:



        cell_factor: Cell size = cell_factor x s_med







    
    Returns:



    """
    N_original = len(coords)
    
    if verbose:
        print(f"=== Grid pre-clean start ===")
        print(f"Input points: {N_original:,}")
    
    # --- Spacing estimate ---
    if s_med is None:
        M = min(len(coords), 300_000)
        if len(coords) > M:
            idx = np.random.choice(len(coords), M, replace=False)
            sample = coords[idx]
        else:
            sample = coords
        s_med, s_mean = estimate_spacing_scale(sample)
        if verbose:
            print(f"Spacing estimate: median={s_med:.6f}, mean={s_mean:.6f}")
    
    cell_size = cell_factor * s_med
    if verbose:
        print(f"Cell size: {cell_size:.6f} (= {cell_factor} x spacing)")
        print(f"Boundary margin: {margin_cells} cells")
    
    cell_indices, min_xyz, max_xyz, n_cells = compute_spatial_grid(coords, cell_size)
    total_cells = np.prod(n_cells)
    
    if verbose:
        print(f"Grid: {n_cells[0]} x {n_cells[1]} x {n_cells[2]} = {total_cells:,} cells")
    
    cell_1d = (cell_indices[:, 0] * n_cells[1] * n_cells[2] + 
               cell_indices[:, 1] * n_cells[2] + 
               cell_indices[:, 2])
    
    sorted_idx = np.argsort(cell_1d)
    sorted_cell_1d = cell_1d[sorted_idx]
    
    unique_cells, cell_starts = np.unique(sorted_cell_1d, return_index=True)
    cell_ends = np.append(cell_starts[1:], len(sorted_cell_1d))
    
    if verbose:
        print(f"Non-empty cells: {len(unique_cells):,}")
    
    keep_flags = np.ones(N_original, dtype=bool)
    
    r = radius_factor * s_med
    
    total_removed_r_or = 0
    total_removed_sor = 0
    
    for cell_idx, (start, end) in enumerate(zip(cell_starts, cell_ends)):
        if verbose and cell_idx % max(1, len(unique_cells) // 20) == 0:
            print(f"  Processing: {cell_idx+1}/{len(unique_cells)} cells", end='\r')
        
        cell_point_indices = sorted_idx[start:end]
        
        cell_i, cell_j, cell_k = np.unravel_index(unique_cells[cell_idx], n_cells)
        
        i_min = max(0, cell_i - margin_cells)
        i_max = min(n_cells[0] - 1, cell_i + margin_cells)
        j_min = max(0, cell_j - margin_cells)
        j_max = min(n_cells[1] - 1, cell_j + margin_cells)
        k_min = max(0, cell_k - margin_cells)
        k_max = min(n_cells[2] - 1, cell_k + margin_cells)
        
        mask = ((cell_indices[:, 0] >= i_min) & (cell_indices[:, 0] <= i_max) &
                (cell_indices[:, 1] >= j_min) & (cell_indices[:, 1] <= j_max) &
                (cell_indices[:, 2] >= k_min) & (cell_indices[:, 2] <= k_max))
        extended_indices = np.where(mask)[0]
        
        if len(extended_indices) < r_min_pts + 1:
            continue
        
        sub_coords = coords[extended_indices]
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(sub_coords.astype(np.float64))
        
        N_before_r_or = len(sub_pcd.points)
        sub_pcd, _ = sub_pcd.remove_radius_outlier(nb_points=r_min_pts, radius=r)
        N_after_r_or = len(sub_pcd.points)
        
        if N_after_r_or >= k_sor + 1:
            sub_pcd, _ = sub_pcd.remove_statistical_outlier(nb_neighbors=k_sor, std_ratio=std_ratio)
        
        N_after_sor = len(sub_pcd.points)
        
        kept_coords = np.asarray(sub_pcd.points, dtype=np.float32)
        
        for pt_idx in cell_point_indices:
            pt = coords[pt_idx]
            dists = np.linalg.norm(kept_coords - pt, axis=1)
            if dists.min() > 1e-6:
                keep_flags[pt_idx] = False
        
        removed_in_cell_r_or = N_before_r_or - N_after_r_or
        removed_in_cell_sor = N_after_r_or - N_after_sor
        total_removed_r_or += removed_in_cell_r_or
        total_removed_sor += removed_in_cell_sor
    
    if verbose:
        print()
    
    coords_out = coords[keep_flags]
    colors_out = colors[keep_flags] if colors is not None else None
    
    N_final = len(coords_out)
    total_removed = N_original - N_final
    
    try:
        enforce_removal_guard(N_original, N_final, hard_cap, "Total (Grid R-OR + SOR)")
    except RuntimeError as e:
        if verbose:
            print(f"  Warning: {e}")
        warnings.warn(str(e))
    
    info = {
        's_med': s_med,
        'removed_ratio': total_removed / N_original,
        'removed_r_or': total_removed_r_or,
        'removed_sor': total_removed_sor,
        'N_original': N_original,
        'N_final': N_final,
        'n_cells': total_cells,
        'non_empty_cells': len(unique_cells),
    }
    
    if verbose:
        print(f"\n=== Processing complete ===")
        print(f"Final points: {N_final:,} / {N_original:,}")
        print(f"Reduction ratio: {info['removed_ratio']*100:.2f}%")
        print(f"R-ORRemoved: {total_removed_r_or:,}")
        print(f"SORRemoved: {total_removed_sor:,}")
    
    return coords_out, colors_out, info


def light_preclean_auto(coords: np.ndarray,
                        colors: Optional[np.ndarray] = None,
                        memory_threshold: int = 50_000_000,
                        **kwargs) -> Tuple[np.ndarray, Optional[np.ndarray], Dict]:
    """

    
    Args:




    
    Returns:



    """
    N = len(coords)
    
    if N <= memory_threshold:
        return light_preclean_simple(coords, colors, **kwargs)
    else:
        return light_preclean_grid(coords, colors, **kwargs)


