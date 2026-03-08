"""Core modules for the point-cloud preprocessing pipeline."""

__version__ = "2.0.0"
__author__ = "Reconstruction Project"

from .main_pipeline import PreprocessPipeline
from .utils import save_ply, analyze_spacing_distribution
from .space_matching import (
    analyze_spacing_distribution as analyze_spacing_full,
    perfect_spacing_sampling,
    iterative_spacing_refinement,
)

__all__ = [
    "PreprocessPipeline",
    "save_ply",
    "analyze_spacing_distribution",
    "analyze_spacing_full",
    "perfect_spacing_sampling",
    "iterative_spacing_refinement",
]
