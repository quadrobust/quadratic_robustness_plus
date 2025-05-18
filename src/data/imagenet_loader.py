#!/usr/bin/env python3
"""
imagenet_loader.py

ImageNet Validation Loader & QuadAug Integration
------------------------------------------------
Provides:
  - `build_imagenet_val(...)`: constructs an ImageFolder dataset for ImageNet-val,
     optionally applying QuadraticAug (QuadAugTransform) for on-the-fly QC corruptions.
  - `loader(...)`: wraps the dataset in a DataLoader with sensible defaults.

Usage example:
    from src.data.imagenet_loader import loader
    # Clean data
    clean_loader = loader(batch=128, qc=False)
    # Quadratic-C corruption at severity 0.3
    qc_loader    = loader(batch=128, qc=True, eps_aff=0.3, eps_trans=0.3)
"""

import pathlib
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from torch.utils.data import DataLoader
from .quad_aug import QuadAugTransform

# ── Paths ───────────────────────────────────────────────
# Determine the project root by going up two directories from the current file
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
# Define the validation dataset directory for ImageNet
VAL_ROOT     = PROJECT_ROOT / "data" / "imagenet" / "val"

# ── ImageNet Normalization Parameters ───────────────────
# Standard mean and standard deviation used for normalizing ImageNet images
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]
_NORMALIZE = T.Normalize(_MEAN, _STD)

# ── Builds the ImageNet Validation Dataset ──────────────
def build_imagenet_val(qc=False, eps_aff=0.30, eps_trans=0.30, apply_normalize: bool = True):
    """
    Constructs the ImageNet validation dataset with optional QuadAug transformations.

    Args:
        qc (bool): If True, apply QuadAugTransform with given parameters.
        eps_aff (float): Strength of affine transformations for QuadAug.
        eps_trans (float): Strength of translation transformations for QuadAug.

    Returns:
        torchvision.datasets.ImageFolder: Transformed ImageNet validation dataset.
    """
    base = [T.Resize(256), T.CenterCrop(224)]  # Standard preprocessing for ImageNet
    if qc:
        # Apply custom QuadAug transformation followed by normalization
        tfm = base + [
            QuadAugTransform(p=1.0, eps_aff=eps_aff, eps_trans=eps_trans)
        ]
        if apply_normalize:
            tfm.append(_NORMALIZE)
    else:
        # Apply standard ToTensor and normalization only
        tfm = base + [T.ToTensor()]
        if apply_normalize:
            tfm.append(_NORMALIZE)
    return ImageFolder(str(VAL_ROOT), transform=T.Compose(tfm))

# ── DataLoader Wrapper ──────────────────────────────────
def loader(batch=64,
           qc=False,
           eps_aff=0.30,
           eps_trans=0.30,
           apply_normalize: bool = True,
           num_workers=4):
    """
    Prepares a DataLoader for the ImageNet validation set.

    Args:
        batch (int): Batch size for loading the dataset.
        qc (bool): Whether to use QuadAugTransform or not.
        eps_aff (float): Strength of affine perturbation (if qc=True).
        eps_trans (float): Strength of translation perturbation (if qc=True).
        num_workers (int): Number of subprocesses used for data loading.

    Returns:
        torch.utils.data.DataLoader: DataLoader for the ImageNet validation dataset.
    """
    ds = build_imagenet_val(qc=qc,
                            eps_aff=eps_aff,
                            eps_trans=eps_trans,
                            apply_normalize=apply_normalize)
    return DataLoader(ds,
                      batch_size=batch,
                      shuffle=False,
                      num_workers=num_workers,
                      pin_memory=True)
