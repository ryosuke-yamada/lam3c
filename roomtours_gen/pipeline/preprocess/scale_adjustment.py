#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Scale-adjustment helpers for aligning reconstructed point clouds to a reference distribution."""

import numpy as np
from typing import Tuple, Optional, Dict
from .utils import calculate_scale, calculate_basic_density


DEFAULT_SCANNET_SCALE_STATS = {
    'mean': 7.983953952789307,
    'median': 7.791001319885254,
    'std': 2.385934829711914,
    'min': 2.8349838256835938,
    'max': 20.039709091186523,
    'percentiles': [5.236302852630615, 6.3233723640441895, 7.791001319885254, 
                    9.36453628540039, 10.929191589355469, 12.139323234558105, 
                    15.189062118530273]
}


def sample_from_real_scale_distribution(real_stats: Dict, seed: Optional[int] = None) -> float:
    """


    
    Args:


    
    Returns:

    """
    if seed is not None:
        np.random.seed(seed)
    
    mean = real_stats['mean']           # 7.984
    std = real_stats['std']             # 2.386
    percentiles = real_stats['percentiles']  # [5.236, 6.323, 7.791, 9.365, 10.929, 12.139, 15.189]
    
    percentile_values = [
        real_stats['min'],      # 2.835 (0%)
        percentiles[0],         # 5.236 (10%)
        percentiles[1],         # 6.323 (25%)
        percentiles[2],         # 7.791 (50%)
        percentiles[3],         # 9.365 (75%)
        percentiles[4],         # 10.929 (90%)
        percentiles[5],         # 12.139 (95%)
        percentiles[6],
    ]
    
    weights = [
        0.01,   # 2.8-5.2  (1%)
        0.09,   # 5.2-6.3  (9%)
        0.25,
        0.30,
        0.20,   # 9.4-10.9 (20%)
        0.10,   # 10.9-12.1(10%)
        0.05
    ]
    
    selected_interval = np.random.choice(len(weights), p=weights)
    
    if selected_interval < len(percentile_values) - 1:
        min_val = percentile_values[selected_interval]
        max_val = percentile_values[selected_interval + 1]
        sampled_scale = np.random.uniform(min_val, max_val)
    else:
        sampled_scale = percentile_values[-1]
    
    max_allowed_scale = percentiles[6]  # 15.189
    sampled_scale = np.clip(sampled_scale, real_stats['min'], max_allowed_scale)
    
    return float(sampled_scale)


def adjust_scale_unified(coords: np.ndarray,
                        colors: Optional[np.ndarray] = None,
                        normals: Optional[np.ndarray] = None,
                        target_scale: Optional[float] = None,
                        real_stats: Optional[Dict] = None,
                        use_probabilistic: bool = False,
                        seed: Optional[int] = None,
                        verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """


    
    Args:








    
    Returns:




    """
    if verbose:
        print("=== Scale adjustment start ===")
    
    current_scale = calculate_scale(coords)
    
    if verbose:
        print(f"Current scale: {current_scale:.3f}")
    
    if real_stats is None:
        real_stats = DEFAULT_SCANNET_SCALE_STATS
        if verbose:
            print("Using default ScanNet statistics")
    
    real_mean = real_stats['mean']      # 7.984
    real_median = real_stats['median']  # 7.791
    real_std = real_stats['std']        # 2.386
    
    if verbose:
        print(f"Reference ScanNet scale stats: mean={real_mean:.3f}, std={real_std:.3f}")
    
    if target_scale is None:
        if use_probabilistic:
            target_scale = sample_from_real_scale_distribution(real_stats, seed)
            if verbose:
                print(f"[SCALE] Probabilistic sampling: {target_scale:.3f}")
        else:
            target_scale = real_mean
            if verbose:
                print(f"[SCALE] Using unified target scale: {target_scale:.3f}")
    else:
        if verbose:
            print(f"[SCALE] Using explicit target scale: {target_scale:.3f}")
    
    scale_factor = target_scale / current_scale
    
    FORCE_SCALE_FACTOR = 1.753
    if verbose:
        print(f"Warning:  Test mode: forcing scale_factor to {FORCE_SCALE_FACTOR}")
        print(f"   Original computed value: {scale_factor:.3f}")
    scale_factor = FORCE_SCALE_FACTOR
    # =================================================
    
    if verbose:
        print(f"Scale adjustment: {current_scale:.3f} -> {target_scale:.3f} (x{scale_factor:.3f})")
    
    scaled_coords = coords * scale_factor
    
    final_scale = calculate_scale(scaled_coords)
    
    if verbose:
        print(f"Adjusted scale: {final_scale:.3f}")
        print(f"Error: {abs(final_scale - target_scale):.6f}")
    
    info = {
        'current_scale': current_scale,
        'target_scale': target_scale,
        'final_scale': final_scale,
        'scale_factor': scale_factor,
        'error': abs(final_scale - target_scale),
        'use_probabilistic': use_probabilistic,
    }
    
    if verbose:
        print("=== Scale adjustment complete ===\n")
    
    return scaled_coords, colors, normals, info


def adjust_scale_to_volume(coords: np.ndarray,
                           colors: Optional[np.ndarray] = None,
                           normals: Optional[np.ndarray] = None,
                           target_volume: float = None,
                           verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """

    
    Args:


        normals: (N, 3) Normal
        target_volume: Target volume

    
    Returns:




    """
    if verbose:
        print("=== Volume-based scale adjustment start ===")
    
    # Current volume
    current_density, current_volume = calculate_basic_density(coords)
    
    if verbose:
        print(f"Current volume: {current_volume:.3f}")
        print(f"Target volume: {target_volume:.3f}")
    
    volume_ratio = target_volume / current_volume
    scale_factor = volume_ratio ** (1/3)
    
    if verbose:
        print(f"Volume ratio: {volume_ratio:.3f}")
        print(f"Scale factor: {scale_factor:.4f}")
    
    scaled_coords = coords * scale_factor
    
    final_density, final_volume = calculate_basic_density(scaled_coords)
    
    if verbose:
        print(f"Adjusted volume: {final_volume:.3f}")
        print(f"Volume match ratio: {final_volume/target_volume:.3f}")
        print("=== Volume-based scale adjustment complete ===\n")
    
    info = {
        'current_volume': current_volume,
        'target_volume': target_volume,
        'final_volume': final_volume,
        'volume_ratio': volume_ratio,
        'scale_factor': scale_factor,
    }
    
    return scaled_coords, colors, normals, info


