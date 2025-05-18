#!/usr/bin/env python3
"""
quad_aug.py

QuadraticAug Transform for PyTorch / torchvision
------------------------------------------------
On-the-fly data augmentation that applies a random degree-2 (quadratic)
spatial warp to each image tensor.

Key features:
  - Samples one of the 15 canonical quadratic forms per invocation.
  - Wraps two small random affine perturbations around each form.
  - Probability `p` of applying the warp; otherwise returns tensor unchanged.

Usage:
    from src.data.quad_aug import QuadAugTransform
    transform = QuadAugTransform(p=0.7, eps_aff=0.3, eps_trans=0.3)
    # as part of a torchvision pipeline:
    pipeline = torchvision.transforms.Compose([
        Resize(256), CenterCrop(224),
        transform,
        Normalize(mean, std),
    ])
"""

import sys, os

# Add the project root to the system path to allow relative imports
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import random
import torchvision.transforms as T
from geometry.warp import quadratic_warp  

class QuadAugTransform:
    """
    Applies a random quadratic warp to an image tensor. The transformation operates on 2D inputs (ℝ² → ℝ²)
    and is affine-equivalent within each orbit representative defined by the `form_id`.

    Parameters:
        p (float): Probability of applying the transformation.
        form_ids (list of int): List of orbit representatives to sample from. Each `form_id` corresponds
                                to a different quadratic transformation. Defaults to [1, 2, ..., 21].
        eps_aff (float): Perturbation magnitude for the linear part of the transformation matrix (A_L, A_R).
        eps_trans (float): Magnitude of translation noise applied to the transformation.

    Usage:
        transform = QuadAugTransform()
        transformed_image = transform(image)
    """
    def __init__(self,
                 p        = 0.7,
                 form_ids = None,
                 eps_aff  = 0.30,
                 eps_trans= 0.30):
        self.p         = p
        self.form_ids  = form_ids or list(range(1, 16))  # Default to form_ids 1 through 21
        self.eps_aff   = eps_aff
        self.eps_trans = eps_trans
        self.to_tensor = T.ToTensor()

    def __call__(self, img):
        """
        Apply the transformation to the input image with probability `p`.
        The image is first converted to a tensor and then, if the random condition is met,
        a quadratic warp is applied.

        Args:
            img (PIL.Image or similar): Input image.

        Returns:
            torch.Tensor: Transformed image tensor.
        """
        x = self.to_tensor(img)
        if random.random() < self.p:
            fid = random.choice(self.form_ids)
            x = quadratic_warp(x.unsqueeze(0),
                               form_id   = fid,
                               eps_aff   = self.eps_aff,
                               eps_trans = self.eps_trans
                               ).squeeze(0)
        return x
