#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Integrated point-cloud preprocessing pipeline."""

import numpy as np
import os
import json
import time
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
import warnings

from .outlier_removal import light_preclean_auto
from .z_up_alignment import align_to_z_up_voxel
from .scale_adjustment import adjust_scale_unified
from .spacing_normalization import adjust_spacing
from .density_adjustment import adjust_density
from .normal_estimation import estimate_normals_auto


class PreprocessPipeline:
    """Point-cloud preprocessing pipeline."""
    
    def __init__(self, config: Optional[Dict] = None, config_path: Optional[str] = None, verbose: bool = True):
        if config is not None:
            self.config = config
        elif config_path is not None and os.path.exists(config_path):
            self.config = self._load_config_from_file(config_path)
        else:
            self.config = self._load_default_config()
        
        self.verbose = verbose
        self.results = {}
        self.module_verbose = False

    def _log(self, message: str = "") -> None:
        if self.verbose:
            print(message)

    def _log_header(self, title: str) -> None:
        if not self.verbose:
            return
        print("=" * 80)
        print(title)
        print("=" * 80)
    
    @staticmethod
    def _load_config_from_file(config_path: str) -> Dict:
        with open(config_path, 'r') as f:
            config = json.load(f)
        for key in config:
            if isinstance(config[key], dict) and 'comment' in config[key]:
                del config[key]['comment']
        return config
    
    @staticmethod
    def _load_default_config() -> Dict:
        default_config_path = os.path.join(os.path.dirname(__file__), 'default_config.json')
        
        if not os.path.exists(default_config_path):
            raise FileNotFoundError(
                f"Configuration file not found: {default_config_path}\n"
                "default_config.json is required."
            )
        
        with open(default_config_path, 'r') as f:
            config = json.load(f)
        
        for key in config:
            if isinstance(config[key], dict) and 'comment' in config[key]:
                del config[key]['comment']
        
        return config
    
    def process(self,
               coords: np.ndarray,
               colors: Optional[np.ndarray] = None,
               normals: Optional[np.ndarray] = None,
               save_intermediates: bool = False,
               output_dir: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
        if self.verbose:
            self._log_header("Preprocessing Pipeline")
            self._log(f"Input points: {len(coords):,}")
            self._log(f"Input arrays: coords={coords.shape}, colors={colors.shape if colors is not None else None}")
            self._log()
        
        start_time = time.time()
        report = {
            'input': {
                'N': len(coords),
                'has_colors': colors is not None,
                'has_normals': normals is not None,
            },
            'steps': {}
        }
        
        current_coords = coords
        current_colors = colors
        current_normals = normals
        
        if 'pre_downsample' in self.config and self.config['pre_downsample']['enabled']:
            step_start = time.time()
            threshold = self.config['pre_downsample'].get('threshold', 1_000_000)

            if len(current_coords) > threshold:
                self._log_header(f"STEP 0: Pre-downsampling ({len(current_coords):,} > {threshold:,})")

                try:
                    N_before = len(current_coords)
                    target_points = threshold
                    self._log("Method: random sampling")
                    self._log(f"Target points: {target_points:,}")
                    self._log(f"Target ratio: {target_points / N_before:.1%}")

                    np.random.seed(self.config['pre_downsample'].get('seed', 42))
                    sample_indices = np.random.choice(N_before, target_points, replace=False)
                    sample_indices = np.sort(sample_indices)
                    
                    current_coords = current_coords[sample_indices]
                    if current_colors is not None:
                        current_colors = current_colors[sample_indices]
                    if current_normals is not None:
                        current_normals = current_normals[sample_indices]
                    
                    ratio = len(current_coords) / N_before
                    self._log(f"Downsampling complete: {N_before:,} -> {len(current_coords):,} ({ratio:.1%})")

                    report['steps']['pre_downsample'] = {
                        'success': True,
                        'method': 'random_sampling',
                        'N_before': N_before,
                        'N_after': len(current_coords),
                        'ratio': ratio,
                        'target_points': target_points,
                        'time': time.time() - step_start,
                    }

                    if save_intermediates and output_dir:
                        self._save_intermediate(output_dir, 'step0_downsampled',
                                              current_coords, current_colors, current_normals)
                except Exception as e:
                    report['steps']['pre_downsample'] = {'success': False, 'error': str(e)}
                    warnings.warn(f"Pre-downsampling failed: {e}")
            else:
                self._log(f"STEP 0 skipped: {len(current_coords):,} <= threshold {threshold:,}")
        
        if self.config['outlier_removal']['enabled']:
            step_start = time.time()
            self._log_header("STEP 1: Outlier Removal")
            
            try:
                current_coords, current_colors, info = light_preclean_auto(
                    current_coords, current_colors,
                    **{k: v for k, v in self.config['outlier_removal'].items() 
                       if k not in ('enabled', 'comment', 'method', 'n_jobs')},
                    verbose=self.module_verbose
                )
                self._log(
                    f"Output points: {info['N_final']:,} / {info['N_original']:,} "
                    f"(removed {info['removed_ratio'] * 100:.2f}%)"
                )
                
                report['steps']['outlier_removal'] = {
                    'success': True,
                    'N_before': info['N_original'],
                    'N_after': info['N_final'],
                    'removed_ratio': info['removed_ratio'],
                    'time': time.time() - step_start,
                }
                
                if save_intermediates and output_dir:
                    self._save_intermediate(output_dir, 'step1_outlier_removed', 
                                          current_coords, current_colors, current_normals)
            except Exception as e:
                report['steps']['outlier_removal'] = {'success': False, 'error': str(e)}
                warnings.warn(f"Outlier removal failed: {e}")
        
        if self.config['z_up_alignment']['enabled']:
            step_start = time.time()
            self._log_header("STEP 2: Z-up Alignment")
            
            try:
                current_coords, current_colors, current_normals, info = align_to_z_up_voxel(
                    current_coords, current_colors, current_normals,
                    **{k: v for k, v in self.config['z_up_alignment'].items() 
                       if k not in ('enabled', 'comment')},
                    verbose=self.module_verbose
                )
                self._log(f"Method: {info.get('method', 'RANSAC')}")
                if info.get('N_voxel') is not None:
                    self._log(f"Voxel representatives: {info['N_voxel']:,}")
                if info.get('ground_normal') is not None:
                    self._log(f"Ground normal: {info['ground_normal']}")
                
                report['steps']['z_up_alignment'] = {
                    'success': True,
                    'N_voxel': info.get('N_voxel'),
                    'voxel_size': info.get('voxel_size'),
                    'ground_normal': info.get('ground_normal'),
                    'method': info.get('method', 'RANSAC'),
                    'time': time.time() - step_start,
                }
                
                if save_intermediates and output_dir:
                    self._save_intermediate(output_dir, 'step2_z_up_aligned',
                                          current_coords, current_colors, current_normals)
            except Exception as e:
                report['steps']['z_up_alignment'] = {'success': False, 'error': str(e)}
                warnings.warn(f"Z-up alignment failed: {e}")
        
        if self.config['scale_adjustment']['enabled']:
            step_start = time.time()
            self._log_header("STEP 3: Scale Adjustment")
            
            try:
                current_coords, current_colors, current_normals, info = adjust_scale_unified(
                    current_coords, current_colors, current_normals,
                    **{k: v for k, v in self.config['scale_adjustment'].items() 
                       if k not in ('enabled', 'comment')},
                    verbose=self.module_verbose
                )
                self._log(
                    f"Scale: {info['current_scale']:.3f} -> {info['final_scale']:.3f} "
                    f"(x{info['scale_factor']:.3f})"
                )
                
                report['steps']['scale_adjustment'] = {
                    'success': True,
                    'scale_factor': info['scale_factor'],
                    'current_scale': info['current_scale'],
                    'final_scale': info['final_scale'],
                    'time': time.time() - step_start,
                }
                
                if save_intermediates and output_dir:
                    self._save_intermediate(output_dir, 'step3_scale_adjusted',
                                          current_coords, current_colors, current_normals)
            except Exception as e:
                report['steps']['scale_adjustment'] = {'success': False, 'error': str(e)}
                warnings.warn(f"Scale adjustment failed: {e}")
        
        if self.config['spacing_normalization']['enabled']:
            step_start = time.time()
            self._log_header("STEP 4: Spacing Normalization")
            
            try:
                spacing_params = {k: v for k, v in self.config['spacing_normalization'].items() if k not in ['enabled', 'comment']}
                self._log(f"Spacing parameters: {spacing_params}")
                
                current_coords, current_colors, current_normals, info = adjust_spacing(
                    current_coords, current_colors, current_normals,
                    **spacing_params,
                    verbose=self.module_verbose
                )
                self._log(
                    f"Spacing: {info['current_spacing']:.6f} -> {info['final_spacing']:.6f} "
                    f"(error={info['error']:.6f})"
                )
                
                report['steps']['spacing_normalization'] = {
                    'success': True,
                    'N_before': info['N_original'],
                    'N_after': info['N_final'],
                    'current_spacing': info['current_spacing'],
                    'final_spacing': info['final_spacing'],
                    'error': info['error'],
                    'time': time.time() - step_start,
                }
                
                if save_intermediates and output_dir:
                    self._save_intermediate(output_dir, 'step4_spacing_normalized',
                                          current_coords, current_colors, current_normals)
                
                max_refine_iterations = int(
                    self.config['spacing_normalization'].get('max_refine_iters', 5)
                )
                error_threshold = float(
                    self.config['spacing_normalization'].get('spacing_error_threshold', 0.10)
                )
                
                for refine_iter in range(1, max_refine_iterations + 1):
                    current_error = info['error']
                    
                    if current_error < error_threshold:
                        self._log(
                            f"Refine skipped: error {current_error:.4f} is below threshold {error_threshold:.4f}"
                        )
                        break
                    
                    self._log(f"Refine iteration {refine_iter}: retrying (error={current_error:.6f})")
                    
                    current_coords, current_colors, current_normals, refine_info = adjust_spacing(
                        current_coords, current_colors, current_normals,
                        **spacing_params,
                        verbose=self.module_verbose
                    )
                    
                    report['steps']['spacing_normalization'][f'refine_{refine_iter}'] = {
                        'N_after': refine_info['N_final'],
                        'final_spacing': refine_info['final_spacing'],
                        'error': refine_info['error']
                    }
                    
                    info = refine_info
                    
                    self._log(
                        f"Refine {refine_iter} result: N={refine_info['N_final']:,}, "
                        f"spacing={refine_info['final_spacing']:.6f}, error={refine_info['error']:.6f}"
                    )
                
                report['steps']['spacing_normalization']['N_after'] = info['N_final']
                report['steps']['spacing_normalization']['final_spacing'] = info['final_spacing']
                report['steps']['spacing_normalization']['error'] = info['error']
                
            except Exception as e:
                import traceback
                error_detail = traceback.format_exc()
                report['steps']['spacing_normalization'] = {'success': False, 'error': str(e), 'traceback': error_detail}
                warnings.warn(f"Spacing normalization failed: {e}")
                self._log(f"Detailed traceback:\n{error_detail}")
        
        if self.config['density_adjustment']['enabled']:
            step_start = time.time()
            self._log_header("STEP 5: Density Adjustment")
            
            try:
                current_coords, current_colors, current_normals, info = adjust_density(
                    current_coords, current_colors, current_normals,
                    **{k: v for k, v in self.config['density_adjustment'].items() 
                       if k not in ('enabled', 'comment')},
                    verbose=self.module_verbose
                )
                self._log(
                    f"Density adjusted: {info['adjusted']}, "
                    f"N={info['N_original']:,}->{info['N_final']:,}, "
                    f"density={info['density_original']:.1f}->{info['density_final']:.1f}"
                )
                
                report['steps']['density_adjustment'] = {
                    'success': True,
                    'adjusted': info['adjusted'],
                    'N_before': info['N_original'],
                    'N_after': info['N_final'],
                    'current_density': info['density_original'],
                    'final_density': info['density_final'],
                    'time': time.time() - step_start,
                }
                
                if save_intermediates and output_dir:
                    self._save_intermediate(output_dir, 'step5_density_adjusted',
                                          current_coords, current_colors, current_normals)
            except Exception as e:
                report['steps']['density_adjustment'] = {'success': False, 'error': str(e)}
                warnings.warn(f"Density adjustment failed: {e}")
        
        if self.config['normal_estimation']['enabled']:
            step_start = time.time()
            self._log_header("STEP 6: Normal Estimation")
            
            try:
                current_normals = estimate_normals_auto(
                    current_coords,
                    **{k: v for k, v in self.config['normal_estimation'].items() 
                       if k not in ('enabled', 'comment', 'method')},
                    verbose=self.module_verbose
                )
                self._log(f"Estimated normals: {len(current_normals):,}")
                
                report['steps']['normal_estimation'] = {
                    'success': True,
                    'N_normals': len(current_normals),
                    'time': time.time() - step_start,
                }
                
                if save_intermediates and output_dir:
                    self._save_intermediate(output_dir, 'step6_normals_estimated',
                                          current_coords, current_colors, current_normals)
            except Exception as e:
                report['steps']['normal_estimation'] = {'success': False, 'error': str(e)}
                warnings.warn(f"Normal estimation failed: {e}")
        
        total_time = time.time() - start_time
        report['output'] = {
            'N': len(current_coords),
            'has_colors': current_colors is not None,
            'has_normals': current_normals is not None,
        }
        report['total_time'] = total_time
        
        if self.verbose:
            self._log()
            self._log_header("Preprocessing Complete")
            self._log(
                f"Final points: {len(current_coords):,} / {len(coords):,} "
                f"({len(current_coords)/len(coords)*100:.1f}%)"
            )
            self._log(f"Total runtime: {total_time:.2f}s")
            self._log()
        
        return current_coords, current_colors, current_normals, report
    
    def _save_intermediate(self, output_dir: str, name: str,
                          coords: np.ndarray,
                          colors: Optional[np.ndarray],
                          normals: Optional[np.ndarray]):
        output_path = Path(output_dir) / name
        output_path.mkdir(parents=True, exist_ok=True)
        
        np.save(output_path / 'coord.npy', coords)
        if colors is not None:
            np.save(output_path / 'color.npy', colors)
        if normals is not None:
            np.save(output_path / 'normal.npy', normals)
        
        try:
            self._save_ply(output_path / f'{name}.ply', coords, colors, normals)
            if self.verbose:
                print(f"Saved intermediate PLY: {output_path / f'{name}.ply'}")
        except Exception as e:
            warnings.warn(f"Failed to save intermediate PLY: {e}")
    
    def _save_ply(self, filepath: str, 
                  coords: np.ndarray,
                  colors: Optional[np.ndarray] = None,
                  normals: Optional[np.ndarray] = None):
        from plyfile import PlyData, PlyElement
        
        vertex_data = []
        vertex_data.append((coords[:, 0].astype(np.float32), 'x'))
        vertex_data.append((coords[:, 1].astype(np.float32), 'y'))
        vertex_data.append((coords[:, 2].astype(np.float32), 'z'))
        if normals is not None:
            vertex_data.append((normals[:, 0].astype(np.float32), 'nx'))
            vertex_data.append((normals[:, 1].astype(np.float32), 'ny'))
            vertex_data.append((normals[:, 2].astype(np.float32), 'nz'))
        if colors is not None:
            if colors.max() <= 1.0:
                colors_uint8 = (colors * 255).astype(np.uint8)
            else:
                colors_uint8 = colors.astype(np.uint8)
            
            vertex_data.append((colors_uint8[:, 0], 'red'))
            vertex_data.append((colors_uint8[:, 1], 'green'))
            vertex_data.append((colors_uint8[:, 2], 'blue'))
        
        vertex_array = np.empty(len(coords), dtype=[(name, arr.dtype) for arr, name in vertex_data])
        for arr, name in vertex_data:
            vertex_array[name] = arr
        vertex_element = PlyElement.describe(vertex_array, 'vertex')
        ply_data = PlyData([vertex_element], text=False)
        ply_data.write(filepath)
    
    def save_report(self, report: Dict, output_path: str):
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        if self.verbose:
            print(f"Saved report: {output_path}")


def process_scene_simple(coords: np.ndarray,
                        colors: Optional[np.ndarray] = None,
                        normals: Optional[np.ndarray] = None,
                        config: Optional[Dict] = None,
                        verbose: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict]:
    """

    
    Args:


        normals: (N, 3) Normal


    
    Returns:




    """
    pipeline = PreprocessPipeline(config=config, verbose=verbose)
    return pipeline.process(coords, colors, normals)


