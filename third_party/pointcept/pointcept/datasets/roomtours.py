"""
RoomTours dataset.

This dataset reuses ScanNet-style loading logic for RoomTours-formatted data.
The dedicated dataset name avoids confusion in public configs.
"""

from .builder import DATASETS
from .scannet import ScanNetDataset


@DATASETS.register_module()
class RoomToursDataset(ScanNetDataset):
    """RoomTours dataset backed by ScanNetDataset implementation."""

