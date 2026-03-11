#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Z-up alignment utilities based on plane detection and fallback heuristics."""

import numpy as np
from typing import Tuple, Optional, Dict
from scipy.spatial.distance import cdist


def estimate_auto_threshold(points: np.ndarray, sample_size: int = 1000, seed: Optional[int] = None) -> float:
    """

    
    Args:
        points: (N, 3) array of 3D points


    
    Returns:

    """
    n_points = len(points)
    if n_points < 10:
        return 0.01
    
    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    
    if n_points > sample_size:
        indices = rng.choice(n_points, sample_size, replace=False)
        sample_points = points[indices]
    else:
        sample_points = points
    
    if len(sample_points) > 500:
        query_indices = rng.choice(len(sample_points), 500, replace=False)
        query_points = sample_points[query_indices]
    else:
        query_points = sample_points
    
    distances = cdist(query_points, sample_points)
    
    nearest_distances = []
    for i in range(len(query_points)):
        row_distances = distances[i]
        non_zero_distances = row_distances[row_distances > 0]
        if len(non_zero_distances) > 0:
            nearest_distances.append(np.min(non_zero_distances))
    
    if len(nearest_distances) == 0:
        return 0.01
    
    nearest_distances = np.array(nearest_distances)
    
    q25 = np.percentile(nearest_distances, 25)
    q75 = np.percentile(nearest_distances, 75)
    iqr = q75 - q25
    
    threshold = 0.5 * iqr
    
    threshold = np.clip(threshold, 0.001, 0.1)
    
    return threshold


def fit_plane_to_points(points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """


    
    Args:
        points: (3, 3) array of 3D points
    
    Returns:


    """
    p1, p2, p3 = points
    
    v1 = p2 - p1
    v2 = p3 - p1
    
    normal = np.cross(v1, v2)
    
    norm = np.linalg.norm(normal)
    if norm < 1e-8:
        return None, None
    
    normal = normal / norm
    
    # d = -(ax + by + cz) for point p1
    d = -np.dot(normal, p1)
    
    return normal, d


def point_to_plane_distance(points: np.ndarray, normal: np.ndarray, d: float) -> np.ndarray:
    """

    
    Args:
        points: (N, 3) array of 3D points


    
    Returns:

    """
    norm = np.linalg.norm(normal)
    if norm > 0:
        normal = normal / norm
        d = d / norm
    
    return np.abs(np.dot(points, normal) + d)


def svd_plane_fit(points: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """

    
    Args:
        points: (N, 3) array of 3D points
    
    Returns:


    """
    if len(points) < 3:
        return None, None
    
    centroid = np.mean(points, axis=0)
    
    centered_points = points - centroid
    
    U, S, Vt = np.linalg.svd(centered_points, full_matrices=False)
    
    normal = Vt[-1]
    
    normal = normal / np.linalg.norm(normal)
    
    d = -np.dot(normal, centroid)
    
    return normal, d


def align_to_z_up_voxel(coords, colors=None, normals=None,
                       spacing_factor=4.0, num_planes=2, max_iterations=500,
                       min_inliers=50, exact_axis=True, seed=42, verbose=False):
    """

    



    
    Args:


        normals: (N, 3) Normal







    
    Returns:
        coords_aligned, colors_aligned, normals_aligned, info
    """
    if verbose:
        print("=== Z-up alignment start ===")
        print(f"Input points: {len(coords):,}")
    
    target_normal = np.array([0, 0, 1])
    
    from scipy.spatial import cKDTree
    
    rng = np.random.default_rng(seed)
    
    sample_size = min(10000, len(coords))
    sample_indices = rng.choice(len(coords), sample_size, replace=False)
    sample_coords = coords[sample_indices]
    
    tree = cKDTree(sample_coords)
    distances, _ = tree.query(sample_coords, k=2)
    nearest_distances = distances[:, 1]
    
    spacing = np.median(nearest_distances)
    
    voxel_size = spacing * spacing_factor
    
    voxel_size = np.clip(voxel_size, 0.01, 0.2)
    
    if verbose:
        print(f"Voxel size: {voxel_size:.6f} (spacing x {spacing_factor})")
        print(f"Extracting voxel representatives...")
    
    bbox_min = coords.min(axis=0)
    bbox_max = coords.max(axis=0)
    voxel_coords = ((coords - bbox_min) / voxel_size).astype(int)
    
    unique_voxels, inverse_indices = np.unique(voxel_coords, axis=0, return_inverse=True)
    N_voxel = len(unique_voxels)
    
    voxel_points = np.zeros((N_voxel, 3), dtype=np.float64)
    for i in range(N_voxel):
        mask = (inverse_indices == i)
        voxel_points[i] = np.mean(coords[mask], axis=0)
    
    if verbose:
        print(f"Voxel representative points: {N_voxel:,} ({N_voxel/len(coords)*100:.1f}%)")
    
    if verbose:
        print(f"\nRunning RANSAC floor detection on voxel representatives")
    
    planes = detect_multiple_planes_ransac(
        voxel_points,
        num_planes=num_planes,
        max_iterations=max_iterations,
        distance_threshold=None,
        min_inliers=min_inliers,
        seed=seed,
        verbose=verbose
    )
    
    if len(planes) == 0:
        if verbose:
            print("Warning:  No floor plane was detected")
            print("Falling back to PCA...")
        
        centered_points = coords - np.mean(coords, axis=0)
        U, _, _ = np.linalg.svd(centered_points.T, full_matrices=False)
        z_direction = U[:, -1]
        
        if verbose:
            print(f"PCA fallback: minimum-variance direction [{z_direction[0]:.3f}, {z_direction[1]:.3f}, {z_direction[2]:.3f}]")
        
        if z_direction[2] < 0:
            z_direction = -z_direction
            if verbose:
                print(f"Flipped PCA normal toward +Z: [{z_direction[0]:.3f}, {z_direction[1]:.3f}, {z_direction[2]:.3f}]")
        
        if np.abs(np.dot(z_direction, target_normal)) > 0.999:
            R = np.eye(3) if np.dot(z_direction, target_normal) > 0 else np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        else:
            rotation_axis = np.cross(z_direction, target_normal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.dot(z_direction, target_normal)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                          [rotation_axis[2], 0, -rotation_axis[0]],
                          [-rotation_axis[1], rotation_axis[0], 0]])
            R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
        
        t = np.mean(coords, axis=0)
        
        if verbose:
            print(f"PCA fallback: translating the centroid to the origin t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
            print("Note: the PCA fallback cannot place the floor exactly at z=0 because no plane was detected.")
        
        coords_aligned = apply_zup(coords, R, t)
        colors_aligned = colors
        normals_aligned = None
        if normals is not None:
            normals_aligned = (R @ normals.T).T
        
        info = {
            'success': True,
            'N_voxel': N_voxel,
            'voxel_size': voxel_size,
            'method': 'PCA_fallback',
            'ground_normal': z_direction.tolist(),
            'rotation_matrix': R.tolist(),
            'translation': t.tolist()
        }
        
        return coords_aligned, colors_aligned, normals_aligned, info
    
    if verbose:
        print(f"\n=== Plane score comparison (coordinate-system agnostic) ===")
        for i, plane in enumerate(planes):
            bd = plane['score_breakdown']
            print(f"Plane {i+1}:")
            print(f"  Inliers: {plane['num_inliers']:,}")
            print(f"  Normal: [{plane['normal'][0]:.3f}, {plane['normal'][1]:.3f}, {plane['normal'][2]:.3f}] (reference Z={bd['normal_z']:.3f})")
            print(f"  Floor score: {plane['ground_score']:.1f} (count+areax100+flatnessx1000+horizontalx500)")
            print(f"    - count:      {bd['count']:.0f} x 1.0    = {bd['count_w']:.1f}")
            print(f"    - area:       {bd['area']:.3f} x 100.0  = {bd['area_w']:.1f}")
            print(f"    - flatness:   {bd['flatness']:.3f} x 1000.0 = {bd['flatness_w']:.1f}")
            print(f"    - horizontal: {bd['horizontal']:.3f} x 500.0  = {bd['horizontal_w']:.1f}")
            print("")
    
    best_idx = max(range(len(planes)), key=lambda i: planes[i]['ground_score'])
    best_plane = planes[best_idx]
    
    if verbose:
        print(f"=== Floor plane selection ===")
        print(f"Plane {best_idx + 1}  selected as the floor plane(Floor score: {best_plane['ground_score']:.1f})")
        print(f"Inliers: {best_plane['num_inliers']}/{N_voxel} ({best_plane['num_inliers']/N_voxel*100:.1f}%)")
        print(f"Normal vector: [{best_plane['normal'][0]:.3f}, {best_plane['normal'][1]:.3f}, {best_plane['normal'][2]:.3f}]")
    
    ground_normal = best_plane['normal'].copy()
    ground_d = best_plane['d']
    
    distances_full = point_to_plane_distance(coords, ground_normal, ground_d)
    threshold_full = estimate_auto_threshold(coords, seed=seed)
    ground_inliers = distances_full < threshold_full
    
    ground_points = coords[ground_inliers]
    
    ground_centroid = np.mean(ground_points, axis=0)
    overall_centroid = np.mean(coords, axis=0)
    up_direction = overall_centroid - ground_centroid
    dot_product_centroid = np.dot(ground_normal, up_direction)
    
    plane_distances = np.dot(coords, ground_normal) + ground_d
    points_above = np.sum(plane_distances > 0)
    points_below = np.sum(plane_distances < 0)
    
    z_bias = ground_normal[2]
    
    flip_votes = 0
    
    if dot_product_centroid < 0:
        flip_votes += 1
    
    if points_below > points_above:
        flip_votes += 1
    
    if z_bias < 0:
        flip_votes += 0.5
    
    if flip_votes >= 1.5:
        ground_normal = -ground_normal
        ground_d = -ground_d
        if verbose:
            print(f"Flipped floor normal upward: [{ground_normal[0]:.3f}, {ground_normal[1]:.3f}, {ground_normal[2]:.3f}] (votes: {flip_votes})")
    else:
        if verbose:
            print(f"Floor normal already points upward (votes: {flip_votes})")
    
    R = compute_z_up_rotation(ground_normal)
    
    if verbose:
        if np.abs(np.dot(ground_normal, target_normal)) > 0.999:
            if np.dot(ground_normal, target_normal) > 0:
                print("Floor normal already matches the z-axis; no rotation needed")
            else:
                print("Floor normal is opposite to the z-axis; rotating 180 degrees about X")
        else:
            rotation_axis = np.cross(ground_normal, target_normal)
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
            cos_angle = np.dot(ground_normal, target_normal)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            print(f"Rotation angle: {np.degrees(angle):.2f} deg")
            print(f"Rotation axis: [{rotation_axis[0]:.3f}, {rotation_axis[1]:.3f}, {rotation_axis[2]:.3f}]")
    
    p0 = -ground_d * ground_normal
    
    if verbose:
        print(f"Reference point on the plane p0: [{p0[0]:.3f}, {p0[1]:.3f}, {p0[2]:.3f}]")
    
    ground_points = coords[ground_inliers]
    ground_points_temp = apply_zup(ground_points, R, p0)
    
    min_z = ground_points_temp[:, 2].min()
    
    if verbose:
        ground_min_z = min_z
        coords_temp_all = apply_zup(coords, R, p0)
        all_min_z = coords_temp_all[:, 2].min()
        all_z_max = coords_temp_all[:, 2].max()
        print(f"Z values after rotation:")
        print(f"  Minimum Z among floor inliers: {ground_min_z:.6f} <- this value is shifted to Z=0")
        print(f"  Minimum Z over all points (reference): {all_min_z:.6f}")
        print(f"  Floor inliers: {np.sum(ground_inliers):,} / {len(coords):,} ({np.sum(ground_inliers)/len(coords)*100:.1f}%)")
        print(f"  Z range over all points before correction: [{all_min_z:.6f}, {all_z_max:.6f}]")
    
    z_offset_vector = np.array([0, 0, min_z])
    offset_in_original = R.T @ z_offset_vector
    
    t = p0 + offset_in_original
    
    if verbose:
        print(f"Placing the lowest floor point at Z=0: additional offset={offset_in_original}")
        print(f"Final translation vector t=[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}]")
    
    coords_aligned = apply_zup(coords, R, t)
    
    if verbose:
        final_z_min = coords_aligned[:, 2].min()
        final_z_max = coords_aligned[:, 2].max()
        z_negative_count = np.sum(coords_aligned[:, 2] < -1e-6)
        print(f"\nZ values after the final transform:")
        print(f"  Range: [{final_z_min:.6f}, {final_z_max:.6f}]")
        if z_negative_count > 0:
            z_negative_pct = z_negative_count / len(coords_aligned) * 100
            print(f"  Warning: Points with Z < 0: {z_negative_count:,} ({z_negative_pct:.2f}%)")
            print(f"  This may be numerical error (tolerance: -1e-6).")
        else:
            print(f"  All points were placed with Z >= 0.")
    
    colors_aligned = colors
    
    normals_aligned = None
    if normals is not None:
        normals_aligned = (R @ normals.T).T
    
    if verbose:
        print("\nApplying the transform to all points...")
        print("\n=== Processing complete ===")
        print(f"Rotation matrix R:")
        print(R)
        print(f"Translation t: {t}")
    
    info = {
        'success': True,
        'N_voxel': N_voxel,
        'voxel_size': voxel_size,
        'ground_normal': ground_normal.tolist(),
        'rotation_matrix': R.tolist(),
        'translation': t.tolist()
    }
    
    return coords_aligned, colors_aligned, normals_aligned, info


def compute_z_up_rotation(plane_normal: np.ndarray) -> np.ndarray:
    """

    
    Args:

    
    Returns:

    """
    target_z = np.array([0, 0, 1])
    
    if np.abs(np.dot(plane_normal, target_z)) > 0.999:
        if np.dot(plane_normal, target_z) > 0:
            return np.eye(3)
        else:
            return np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
    
    rotation_axis = np.cross(plane_normal, target_z)
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
    
    cos_angle = np.dot(plane_normal, target_z)
    angle = np.arccos(np.clip(cos_angle, -1, 1))
    
    K = np.array([[0, -rotation_axis[2], rotation_axis[1]],
                  [rotation_axis[2], 0, -rotation_axis[0]],
                  [-rotation_axis[1], rotation_axis[0], 0]])
    
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)
    
    return R


def apply_zup(points: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """

    
    Args:
        points: (N, 3) array of 3D points


    
    Returns:

    """
    return (R @ (points - t).T).T


def calculate_ground_score(points: np.ndarray, normal: np.ndarray, inlier_count: int) -> Tuple[float, Dict]:
    """

    
    Args:


        inlier_count: Inliers
    
    Returns:


    """
    if len(points) < 3:
        empty_breakdown = {
            'count': 0, 'count_w': 0,
            'area': 0, 'area_w': 0,
            'flatness': 0, 'flatness_w': 0,
            'horizontal': 0, 'horizontal_w': 0,
            'normal_z': 0
        }
        return 0.0, empty_breakdown
    
    count_score = inlier_count
    
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    if abs(normal[2]) < 0.9:
        u1 = np.cross(normal, [0, 0, 1])
    else:
        u1 = np.cross(normal, [1, 0, 0])
    u1 = u1 / np.linalg.norm(u1)
    u2 = np.cross(normal, u1)
    u2 = u2 / np.linalg.norm(u2)
    
    plane_coords = np.column_stack([
        np.dot(centered_points, u1),
        np.dot(centered_points, u2)
    ])
    
    area_variance = np.prod(np.var(plane_coords, axis=0))
    area_score = np.sqrt(area_variance)
    
    height_variance = np.var(np.dot(centered_points, normal))
    flatness_score = 1.0 / (1.0 + height_variance * 1000)
    
    horizontal_score = abs(normal[2])
    
    total_score = (
        count_score * 1.0 +           # Inliers
        area_score * 100.0 +
        flatness_score * 1000.0 +
        horizontal_score * 500.0
    )
    
    breakdown = {
        'count': count_score,
        'count_w': count_score * 1.0,
        'area': area_score,
        'area_w': area_score * 100.0,
        'flatness': flatness_score,
        'flatness_w': flatness_score * 1000.0,
        'horizontal': horizontal_score,
        'horizontal_w': horizontal_score * 500.0,
        'normal_z': normal[2]
    }
    
    return total_score, breakdown

def detect_multiple_planes_ransac(coords, num_planes=2, max_iterations=500,
                                  distance_threshold=0.02, min_inliers=50,
                                  seed=42, verbose=False):
    """

    
    Args:


        ...
    
    Returns:

            - 'normal': Normal vector


            - 'num_inliers': Inliers


    """
    if verbose:
        print(f"=== Floor plane detection start ===")
        print(f"Up to {num_planes} planes will be detected")
    
    remaining_points = coords.copy()
    remaining_indices = np.arange(len(coords))
    
    planes = []
    
    for plane_idx in range(num_planes):
        if len(remaining_points) < 100:
            break
        
        if verbose:
            print(f"\n--- Plane {plane_idx + 1}  detection ---")
        
        plane = ransac_plane_fit(
            remaining_points,
            max_iterations=max_iterations,
            distance_threshold=distance_threshold,
            min_inliers=min_inliers,
            seed=seed,
            rng=None,
            verbose=verbose
        )
        
        if plane is None:
            if verbose:
                print(f"Plane {plane_idx + 1}  detection failed")
            break
        
        global_inlier_mask = np.zeros(len(coords), dtype=bool)
        global_inlier_mask[remaining_indices] = plane['inlier_mask']
        
        inlier_count = np.sum(global_inlier_mask)
        
        inlier_points = coords[global_inlier_mask]
        ground_score, score_breakdown = calculate_ground_score(
            inlier_points,
            plane['normal'],     # Normal
            inlier_count         # Inliers
        )
        
        planes.append({
            'normal': plane['normal'],
            'd': plane['d'],
            'inlier_mask': global_inlier_mask,
            'num_inliers': inlier_count,
            'ground_score': ground_score,
            'score_breakdown': score_breakdown
        })
        
        if verbose:
            print(f"Plane {plane_idx + 1}: {inlier_count} inliers, Floor score: {ground_score:.3f}")
        
        local_outlier_mask = ~plane['inlier_mask']
        remaining_points = remaining_points[local_outlier_mask]
        remaining_indices = remaining_indices[local_outlier_mask]
    
    if verbose and planes:
        print(f"\nDetected planes: {len(planes)}")
        for i, p in enumerate(planes):
            print(f"  Plane {i + 1}: inliers {p['num_inliers']}, score {p['ground_score']:.1f}")
    
    return planes


def ransac_plane_fit(coords, max_iterations=500, distance_threshold=None,
                    min_inliers=50, seed=None, rng=None, verbose=False):
    """

    
    Returns:

    """
    n_points = len(coords)
    if n_points < 3:
        return None
    
    if distance_threshold is None:
        distance_threshold = estimate_auto_threshold(coords, seed=seed)
    
    if verbose:
        print(f"Starting RANSAC plane detection: {n_points} points, up to {max_iterations} iterations, threshold={distance_threshold:.4f}")
    
    if rng is None:
        rng = np.random.default_rng(seed)
    
    best_inlier_count = 0
    best_normal = None
    best_d = None
    best_inlier_mask = np.array([], dtype=bool)
    
    for iteration in range(max_iterations):
        sample_indices = rng.choice(n_points, 3, replace=False)
        sample_points = coords[sample_indices]
        
        normal, d = fit_plane_to_points(sample_points)
        if normal is None:
            continue
        
        distances = point_to_plane_distance(coords, normal, d)
        
        inlier_mask = distances < distance_threshold
        inlier_count = np.sum(inlier_mask)
        
        if inlier_count > best_inlier_count and inlier_count >= min_inliers:
            best_inlier_count = inlier_count
            best_normal = normal.copy()
            best_d = d
            best_inlier_mask = inlier_mask.copy()
    
    if best_normal is not None:
        inlier_points = coords[best_inlier_mask]
        refined_normal, refined_d = svd_plane_fit(inlier_points)
        
        if refined_normal is not None:
            best_normal = refined_normal
            best_d = refined_d
            
            if verbose:
                print(f"Plane detection complete: inliers {best_inlier_count}/{n_points} ({best_inlier_count/n_points*100:.1f}%)")
                print(f"SVD-refit normal: [{best_normal[0]:.3f}, {best_normal[1]:.3f}, {best_normal[2]:.3f}]")
        else:
            if verbose:
                print(f"Plane detection complete: inliers {best_inlier_count}/{n_points} ({best_inlier_count/n_points*100:.1f}%)")
                print(f"Plane normal: [{best_normal[0]:.3f}, {best_normal[1]:.3f}, {best_normal[2]:.3f}]")
    else:
        if verbose:
            print("Plane detection failed: not enough inliers were found.")
        return None
    
    return {
        'normal': best_normal,
        'd': best_d,
        'num_inliers': best_inlier_count,
        'inlier_mask': best_inlier_mask
    }
