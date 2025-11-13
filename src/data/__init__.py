"""
Data loading and preprocessing modules for Nonlinear Rectified Flows.

This package provides:
- Dataset loaders (COCO, CC3M, LAION)
- Data augmentation utilities
- Text embedding preprocessing
"""

# Import augmentation utilities
from .augmentation import (
    get_transforms,
    RandomHorizontalFlip,
    RandomResizedCrop,
    ColorJitter,
)

# Import dataset functions and classes
from .datasets import (
    create_dataloader,
    COCODataset,
    precompute_text_embeddings,
    download_coco_dataset,
)

__all__ = [
    "create_dataloader",
    "COCODataset",
    "precompute_text_embeddings",
    "download_coco_dataset",
    "get_transforms",
    "RandomHorizontalFlip",
    "RandomResizedCrop",
    "ColorJitter",
]
