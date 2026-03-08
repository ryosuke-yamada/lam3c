#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Spacing-normalization routines with conservative point-count preservation."""

import numpy as np
from typing import Tuple, Optional, Dict
from .utils import analyze_spacing_distribution


def perfect_spatial_sampling(coords: np.ndarray,
                            colors: Optional[np.ndarray],
                            normals: Optional[np.ndarray],
                            target_spacing: float,
                            grid_size_factor: float = 1.0,
                            verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """

    

    
    Args:


        normals: (N, 3) Normal
        target_spacing: Target spacing


    
    Returns:



    """
    grid_size = target_spacing * grid_size_factor
    
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    
    grid_dims = np.ceil((maxs - mins) / grid_size).astype(np.int64)
    
    if verbose:
        print(f"  Current points: {len(coords):,}")
        print(f"  Target spacing: {target_spacing:.6f}")
        print(f"  Grid size: {grid_size:.6f} (factor={grid_size_factor:.2f})")
        print(f"  Grid dimensions: {grid_dims}")
    
    grid_indices = np.floor((coords - mins) / grid_size).astype(np.int64)
    
    grid_keys = (grid_indices[:, 0] + 
                 grid_indices[:, 1] * grid_dims[0] + 
                 grid_indices[:, 2] * grid_dims[0] * grid_dims[1])
    
    unique_keys, inverse_indices = np.unique(grid_keys, return_inverse=True)
    
    if verbose:
        total_cells = int(grid_dims[0]) * int(grid_dims[1]) * int(grid_dims[2])
        occupied_cells = len(unique_keys)
        print(f"  Total grid cells: {total_cells:,}")
        print(f"  Occupied grid cells: {occupied_cells:,}")
        if total_cells > 0:
            print(f"  Occupancy: {occupied_cells/total_cells*100:.4f}%")
    
    
    random_priorities = np.random.random(len(coords))
    
    sort_indices = np.argsort(inverse_indices)
    sorted_inverse = inverse_indices[sort_indices]
    sorted_priorities = random_priorities[sort_indices]
    
    unique_inverse, group_starts = np.unique(sorted_inverse, return_index=True)
    
    sampled_indices = np.empty(len(unique_inverse), dtype=np.int64)
    for i in range(len(unique_inverse)):
        start = group_starts[i]
        end = group_starts[i + 1] if i + 1 < len(group_starts) else len(sorted_inverse)
        group_priorities = sorted_priorities[start:end]
        local_best = np.argmax(group_priorities)
        sampled_indices[i] = sort_indices[start + local_best]
    
    sampled_coords = coords[sampled_indices]
    sampled_colors = colors[sampled_indices] if colors is not None else None
    sampled_normals = normals[sampled_indices] if normals is not None else None
    
    if verbose:
        print(f"  Points after sampling: {len(sampled_coords):,} ({len(sampled_coords)/len(coords)*100:.1f}%)")
    
    return sampled_coords, sampled_colors, sampled_normals


DEFAULT_SCANNET_SPACING_STATS = {
    'mean': {
        'mean': 0.013329362514309568,
        'median': 0.013047804585617917,
        'std': 0.002155367280848707,
        'min': 0.012341868064472081,
        'max': 0.031871938705140275,
        'percentiles': [0.012813941035942552, 0.012924797598551458, 0.013047804585617917,
                       0.013192256293711216, 0.013346902432631876, 0.013485435641338509,
                       0.030136269920015713]
    },
    'median': {
        'mean': 0.012764181891882465,
        'median': 0.01249951015777981,
        'std': 0.002087843613725497
    },
    'std': {
        'mean': 0.004424077436133315,
        'median': 0.004331381558047868,
        'std': 0.0007265971390420328
    }
}


def iterative_spacing_refinement(coords: np.ndarray,
                                 colors: Optional[np.ndarray],
                                 normals: Optional[np.ndarray],
                                 target_spacing_stats: Dict,
                                 max_iterations: int = 5,
                                 min_points_ratio: float = 0.70,
                                 verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """


    


    
    Args:


        normals: (N, 3) Normal




    
    Returns:



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
    
    target_mean = target_spacing_stats['mean']
    
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
        
        spacing_ratio = current_mean / target_mean
        
        if spacing_ratio < 1.0:
            adjustment_factor = (1.0 / spacing_ratio) ** 0.5
        else:
            adjustment_factor = spacing_ratio ** 0.5
        
        grid_size_factor = grid_size_factor * adjustment_factor
        
        if verbose:
            print(f"Grid-size factor update: {grid_size_factor:.3f} (adjustment factor: {adjustment_factor:.3f})")
        
        current_coords, current_colors, current_normals = perfect_spatial_sampling(
            current_coords, current_colors, current_normals,
            target_mean, grid_size_factor, verbose=False
        )
        
        if verbose:
            print(f"Iteration {iteration + 1} complete: {len(current_coords):,} points")
    
    if verbose:
        print(f"\nFinal result: using result from iteration {best_iteration} (error={best_error:.4f})")
    
    return best_coords, best_colors, best_normals


def gentle_spatial_upsampling(coords: np.ndarray,
                              colors: Optional[np.ndarray],
                              normals: Optional[np.ndarray],
                              target_num_points: int,
                              max_neighbors: int = 8,
                              safe_distance_factor: float = 0.3,
                              verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """

    
    Args:


        normals: (N, 3) Normal

        max_neighbors: Up to Neighbors


    
    Returns:



    """
    current_num = len(coords)
    num_to_add = target_num_points - current_num
    
    if num_to_add <= 0:
        return coords, colors, normals
    
    if verbose:
        print(f"Upsampling: {current_num:,} -> {target_num_points:,} (+{num_to_add:,})")
    
    spacing_stats = analyze_spacing_distribution(coords)
    if spacing_stats is None:
        idx = np.random.choice(current_num, num_to_add, replace=True)
        upsampled_coords = np.vstack([coords, coords[idx]])
        upsampled_colors = np.vstack([colors, colors[idx]]) if colors is not None else None
        upsampled_normals = np.vstack([normals, normals[idx]]) if normals is not None else None
        return upsampled_coords, upsampled_colors, upsampled_normals
    
    safe_distance = spacing_stats['mean'] * safe_distance_factor
    
    from sklearn.neighbors import NearestNeighbors
    nbrs = NearestNeighbors(n_neighbors=min(max_neighbors + 1, current_num), algorithm='auto').fit(coords)
    
    new_coords = []
    new_colors = [] if colors is not None else None
    new_normals = [] if normals is not None else None
    
    attempts = 0
    max_attempts = num_to_add * 3
    
    while len(new_coords) < num_to_add and attempts < max_attempts:
        idx = np.random.randint(0, current_num)
        point = coords[idx]
        
        distances, indices = nbrs.kneighbors([point])
        neighbor_idx = indices[0][1:max_neighbors + 1]
        
        if len(neighbor_idx) == 0:
            attempts += 1
            continue
        
        neighbor = coords[np.random.choice(neighbor_idx)]
        
        direction = neighbor - point
        dist = np.linalg.norm(direction)
        if dist < safe_distance:
            attempts += 1
            continue
        
        t = np.random.uniform(0.3, 0.7)
        new_point = point + t * direction
        
        new_coords.append(new_point)
        
        if colors is not None:
            new_color = (colors[idx] * (1 - t) + colors[neighbor_idx[0]] * t).astype(colors.dtype)
            new_colors.append(new_color)
        
        if normals is not None:
            new_normal = normals[idx] * (1 - t) + normals[neighbor_idx[0]] * t
            new_normal = new_normal / (np.linalg.norm(new_normal) + 1e-8)
            new_normals.append(new_normal)
        
        attempts += 1
    
    if len(new_coords) < num_to_add:
        shortage = num_to_add - len(new_coords)
        if verbose:
            print(f"  Warning: {shortage:,} points filled by duplication")
        idx = np.random.choice(current_num, shortage, replace=True)
        new_coords.extend(coords[idx])
        if colors is not None:
            new_colors.extend(colors[idx])
        if normals is not None:
            new_normals.extend(normals[idx])
    
    new_coords = np.array(new_coords[:num_to_add])
    upsampled_coords = np.vstack([coords, new_coords])
    
    if colors is not None:
        new_colors = np.array(new_colors[:num_to_add])
        upsampled_colors = np.vstack([colors, new_colors])
    else:
        upsampled_colors = None
    
    if normals is not None:
        new_normals = np.array(new_normals[:num_to_add])
        upsampled_normals = np.vstack([normals, new_normals])
    else:
        upsampled_normals = None
    
    if verbose:
        print(f"  Complete: {len(upsampled_coords):,} points")
    
    return upsampled_coords, upsampled_colors, upsampled_normals


def adjust_spacing(coords: np.ndarray,
                  colors: Optional[np.ndarray] = None,
                  normals: Optional[np.ndarray] = None,
                  target_spacing: float = 0.01333,
                  spacing_error_threshold: float = 0.03,
                  max_refine_iters: int = 5,
                  min_points_ratio: float = 0.0,
                  verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """

    





    
    Args:


        normals: (N, 3) Normal
        target_spacing: Target spacing




    
    Returns:




    """
    from .space_matching import (
        analyze_spacing_distribution,
        perfect_spacing_sampling,
        iterative_spacing_refinement
    )
    
    N = len(coords)
    
    target_spacing_stats = {
        'mean': target_spacing,
        'median': target_spacing * 0.957,
        'std': target_spacing * 0.332,
        'percentiles': [
            target_spacing * 0.6,
            target_spacing * 0.75,
            target_spacing * 0.95,
            target_spacing * 1.15,
            target_spacing * 1.4,
            target_spacing * 1.6,
            target_spacing * 1.8
        ]
    }
    
    current_spacing = analyze_spacing_distribution(coords)
    if not current_spacing:
        if verbose:
            print("Spacing analysis failed; keeping the current point count")
        info = {
            'current_spacing': 0.0,
            'target_spacing': target_spacing,
            'final_spacing': 0.0,
            'error': 1.0,
            'N_original': N,
            'N_final': N,
            'method': 'failed'
        }
        return coords, colors, normals, info
    
    s_cur = current_spacing['mean']
    s_tgt = target_spacing_stats['mean']
    r = s_cur / s_tgt
    
    if verbose:
        print(f"\n=== Spacing normalization start ===")
        print(f"Input points: {N:,}")
        print(f"Current spacing: {s_cur:.6f}")
        print(f"Target spacing: {s_tgt:.6f}")
        print(f"Spacing ratio r: {r:.6f}")
    
    if verbose:
        print(f"\n[STEP 1] Keep-N attempt")
    
    trial_coords, trial_colors, trial_normals = perfect_spacing_sampling(
        coords, colors, normals, target_spacing_stats, N
    )
    
    trial_spacing = analyze_spacing_distribution(trial_coords)
    if trial_spacing:
        err = abs(trial_spacing['mean'] - s_tgt) / s_tgt
        if verbose:
            print(f"Keep-N result: spacing={trial_spacing['mean']:.6f}, error={err:.4f}")
        
        if err <= spacing_error_threshold:
            if verbose:
                print(f"Point count can stay unchanged; spacing already matches well(Error{err:.4f} <= {spacing_error_threshold})")
            info = {
                'current_spacing': s_cur,
                'target_spacing': s_tgt,
                'final_spacing': trial_spacing['mean'],
                'error': err,
                'N_original': N,
                'N_final': len(trial_coords),
                'method': 'keep-N'
            }
            return trial_coords, trial_colors, trial_normals, info
    
    if verbose:
        print(f"\n[STEP 2] Progressive sampling")
    
    raw_target = int(N * (r ** 3))
    
    if verbose:
        print(f"Theoretical r^3 target: {raw_target:,} points ({raw_target/N*100:.1f}%)")
    
    current_coords = coords.copy()
    current_colors = colors.copy() if colors is not None else None
    current_normals = normals.copy() if normals is not None else None
    
    max_steps = 10
    for step in range(max_steps):
        current_spacing = analyze_spacing_distribution(current_coords)
        if not current_spacing:
            break
        
        s_cur = current_spacing['mean']
        error_current = abs(s_cur - s_tgt) / s_tgt
        
        current_N = len(current_coords)
        
        if verbose:
            print(f"\n  Step {step+1}/{max_steps}:")
            print(f"    Current points: {current_N:,}")
            print(f"    Current spacing: {s_cur:.6f} (target: {s_tgt:.6f})")
            print(f"    Error: {error_current:.4f} ({error_current*100:.2f}%)")
        
        if error_current <= spacing_error_threshold:
            if verbose:
                print(f"    Target reached(Error {error_current:.4f} <= {spacing_error_threshold})")
            break
        
        if s_cur > s_tgt:
            if verbose:
                print(f"    Target exceeded (spacing {s_cur:.6f} > {s_tgt:.6f}); stopping")
            break
        
        ratio = s_cur / s_tgt
        theoretical_point_ratio = ratio ** 3
        
        step_ratio = 1.0 + (theoretical_point_ratio - 1.0) * 0.2
        target_N = max(int(current_N * step_ratio), int(current_N * 0.5))
        
        if verbose:
            print(f"    Theoretical required point ratio: {theoretical_point_ratio:.4f}")
            print(f"    Applied reduction ratio: {step_ratio:.4f}")
            print(f"    Target points: {target_N:,} ({target_N/current_N*100:.1f}%, Reduction ratio: {(1-target_N/current_N)*100:.1f}%)")
        
        # 
        # 
        # grid_size = target_spacing * grid_size_factor
        
        point_ratio = current_N / target_N
        grid_size_factor = point_ratio ** (1/3)
        
        if grid_size_factor < 1.02:
            if verbose:
                print(f"    Warning:  grid_size_factor={grid_size_factor:.4f} is too small; skipping this step")
            break
        
        if verbose:
            print(f"    grid_size_factor: {grid_size_factor:.4f} (point_ratio^(1/3) = {point_ratio:.4f}^(1/3))")
        
        sampled_coords, sampled_colors, sampled_normals = perfect_spatial_sampling(
            current_coords, current_colors, current_normals,
            s_tgt, grid_size_factor, verbose=False
        )
        
        if len(sampled_coords) == len(current_coords):
            if verbose:
                print(f"    Warning:  Point count did not change; stopping")
            break
        
        current_coords = sampled_coords
        current_colors = sampled_colors
        current_normals = sampled_normals
    
    adjusted_coords = current_coords
    adjusted_colors = current_colors
    adjusted_normals = current_normals
    
    final_spacing = analyze_spacing_distribution(adjusted_coords)
    err_after = (abs(final_spacing['mean'] - s_tgt) / s_tgt) if final_spacing else 1.0
    
    if verbose and final_spacing:
        print(f"\n  Progressive sampling complete:")
        print(f"    Final points: {len(adjusted_coords):,} ({len(adjusted_coords)/N*100:.1f}%)")
        print(f"    Final spacing: {final_spacing['mean']:.6f}")
        print(f"    Final error: {err_after:.4f} ({err_after*100:.2f}%)")
    
    if err_after > spacing_error_threshold:
        if verbose:
            print(f"\n[STEP 3] Layout refinement with perfect_spatial_sampling(Error{err_after:.4f} > {spacing_error_threshold})")
        
        target_N_maintain = len(adjusted_coords)
        
        current_s = final_spacing['mean']
        
        if current_s < s_tgt:
            grid_size_factor = 1.0 + (s_tgt / current_s - 1.0) * 0.3
        else:
            grid_size_factor = 1.0
        
        if verbose:
            print(f"  Target points to preserve: {target_N_maintain:,}")
            print(f"  Current spacing: {current_s:.6f}")
            print(f"  Grid-size factor: {grid_size_factor:.3f}")
        
        optimized_coords, optimized_colors, optimized_normals = perfect_spatial_sampling(
            adjusted_coords, adjusted_colors, adjusted_normals,
            s_tgt, grid_size_factor, verbose=False
        )
        
        optimized_spacing = analyze_spacing_distribution(optimized_coords)
        if optimized_spacing:
            err_optimized = abs(optimized_spacing['mean'] - s_tgt) / s_tgt
            
            n_change_pct = abs(len(optimized_coords) - target_N_maintain) / target_N_maintain * 100
        
            if verbose:
                print(f"  Points after layout refinement: {len(optimized_coords):,} (change: {n_change_pct:.1f}%)")
                print(f"  Spacing after layout refinement: {optimized_spacing['mean']:.6f}")
                print(f"  Error after layout refinement: {err_optimized:.4f}")
            
            if err_optimized < err_after and n_change_pct < 20.0:
                adjusted_coords = optimized_coords
                adjusted_colors = optimized_colors
                adjusted_normals = optimized_normals
                err_after = err_optimized
                if verbose:
                    print(f"  Accepting layout refinement (error improved and point count preserved)")
            else:
                if verbose:
                    if err_optimized >= err_after:
                        print(f"  Warning:  Rejecting layout refinement (error worsened: {err_optimized:.4f} >= {err_after:.4f})")
                    else:
                        print(f"  Warning:  Rejecting layout refinement (point-count change is too large: {n_change_pct:.1f}%)")
    
    final_spacing = analyze_spacing_distribution(adjusted_coords)
    err_after = (abs(final_spacing['mean'] - s_tgt) / s_tgt) if final_spacing else 1.0
    
    info = {
        'current_spacing': s_cur,
        'target_spacing': s_tgt,
        'final_spacing': final_spacing['mean'] if final_spacing else s_cur,
        'error': err_after,
        'N_original': N,
        'N_final': len(adjusted_coords),
        'method': 'progressive_sampling'
    }
    
    if verbose:
        print(f"\n=== Spacing normalization complete ===")
        print(f"Final spacing: {info['final_spacing']:.6f}")
        print(f"Error: {info['error']:.4f}")
        print(f"Points: {info['N_final']:,} / {info['N_original']:,} ({info['N_final']/info['N_original']*100:.1f}%)\n")
    
    return adjusted_coords, adjusted_colors, adjusted_normals, info


