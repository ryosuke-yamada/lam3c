#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Density-adjustment helpers."""

import numpy as np

from .utils import calculate_basic_density, calculate_scale


def adjust_density(
    coords,
    colors=None,
    normals=None,
    min_final_points=40000,
    max_change_ratio=0.3,
    seed=None,
    verbose=False,
):
    """Apply conservative density correction only for extreme cases."""
    if verbose:
        print("=== Density adjustment ===")

    num_points = len(coords)
    density, volume = calculate_basic_density(coords)
    scale = calculate_scale(coords)

    if verbose:
        print("Current state:")
        print(f"  Points: {num_points:,}")
        print(f"  Density: {density:.1f}")
        print(f"  Volume: {volume:.3f}")
        print(f"  Scale: {scale:.3f}")

    scale_ref = 8.0
    density_ref = 1734.4
    expected_density = density_ref * (scale / scale_ref) ** 0
    density_ratio = density / expected_density

    if verbose:
        print("Density check:")
        print(f"  Current density: {density:.1f}")
        print(f"  Expected density: {expected_density:.1f}")
        print(f"  Density ratio: {density_ratio:.2f}")

    should_adjust = False
    if density_ratio > 1.5:
        should_adjust = True
        if verbose:
            print(f"  Density ratio {density_ratio:.2f} > 1.5, adjustment required")
    if num_points > 200000:
        should_adjust = True
        if verbose:
            print(f"  Point count {num_points:,} > 200,000, adjustment required")

    if not should_adjust:
        if verbose:
            print("  No density adjustment required")
            print("=== Density adjustment skipped ===\n")
        return coords, colors, normals, {
            "adjusted": False,
            "N_original": num_points,
            "N_final": num_points,
            "density_original": density,
            "density_final": density,
        }

    target_density = expected_density
    target_num_points = int(num_points * (target_density / density))
    target_num_points = max(target_num_points, min_final_points)
    max_reduction = int(num_points * (1.0 - max_change_ratio))
    max_increase = int(num_points * (1.0 + max_change_ratio))
    target_num_points = min(max(target_num_points, max_reduction), max_increase)

    if verbose:
        print(f"  Target point count: {target_num_points:,} ({target_num_points / num_points * 100:.1f}%)")

    if seed is not None:
        np.random.seed(seed)

    if target_num_points < num_points:
        indices = np.random.choice(num_points, target_num_points, replace=False)
        indices = np.sort(indices)
        adjusted_coords = coords[indices]
        adjusted_colors = colors[indices] if colors is not None else None
        adjusted_normals = normals[indices] if normals is not None else None
    elif target_num_points > num_points:
        additional = target_num_points - num_points
        extra_indices = np.random.choice(num_points, additional, replace=True)
        adjusted_coords = np.vstack([coords, coords[extra_indices]])
        adjusted_colors = np.vstack([colors, colors[extra_indices]]) if colors is not None else None
        adjusted_normals = np.vstack([normals, normals[extra_indices]]) if normals is not None else None
    else:
        adjusted_coords = coords
        adjusted_colors = colors
        adjusted_normals = normals

    adjusted_density, _ = calculate_basic_density(adjusted_coords)

    if verbose:
        print(f"  Adjusted point count: {len(adjusted_coords):,}")
        print(f"  Adjusted density: {adjusted_density:.1f}")
        print(f"  Adjusted ratio: {adjusted_density / expected_density:.2f}")
        print("=== Density adjustment complete ===\n")

    return adjusted_coords, adjusted_colors, adjusted_normals, {
        "adjusted": True,
        "N_original": num_points,
        "N_final": len(adjusted_coords),
        "density_original": density,
        "density_final": adjusted_density,
    }
