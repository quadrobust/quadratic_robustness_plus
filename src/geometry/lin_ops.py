# src/geometry/lin_ops.py
# ---------------------------------------------------------
#  Random affine transformations:
#     GA(k) = GL(k) ⋉ ℝ^k   (x ↦ A·x + t)
# ---------------------------------------------------------
import torch, random
from typing import Tuple

# ---------------------------------------------------------
#  Helper function – random linear part from GL(k)
# ---------------------------------------------------------
def _random_gl(k: int,
               eps_aff: float = 0.3,
               allow_scale: bool = False,      # *** new parameter
               det_sigma: float = 0.4,         # *** log‑σ when allow_scale=True
               device: str = "cpu") -> torch.Tensor:
    """
    Returns a random matrix from GL(k), the general linear group.

    Parameters:
    - k: Dimension of the space.
    - eps_aff: Perturbation strength; controls deviation from the identity matrix.
    - allow_scale (bool): 
        If False – the matrix is normalized so that |det(A)| ≈ 1.
        If True – the determinant is allowed to vary, i.e. det(A) ≈ det(A) * exp(N(0, σ)).
    - det_sigma: Standard deviation in log-determinant space when allow_scale=True.
    - device: Device for tensor allocation ('cpu' or 'cuda').

    Returns:
    - A: A random matrix from GL(k) with controlled determinant properties.
    """
    A = torch.eye(k, device=device)
    A += eps_aff * torch.randn(k, k, device=device)

    # --- control of scale det(A) ---------------------------------
    if allow_scale:
        # apply a random scaling multiplier (log-normal distributed)
        scale = torch.exp(torch.randn(1, device=device) * det_sigma)
        A = A * scale          # results in det(A) ≈ scale^k
    else:
        # enforce |det(A)| ≈ 1
        det = torch.det(A).abs()
        A = A / det**(1/k)
    return A

# ---------------------------------------------------------
#  GA(2) – 2D affine transformation
# ---------------------------------------------------------
def sample_ga2(eps_aff: float = 0.3,
               eps_trans: float = 0.5,
               allow_scale: bool = False,
               det_sigma: float = 0.4,
               device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Samples a random 2D affine transformation (A, t), where:
    - A ∈ GL(2) is a random linear transformation.
    - t ∈ ℝ² is a translation vector.

    Parameters:
    - eps_aff: Perturbation of the linear component (A).
    - eps_trans: Standard deviation of the translation vector.
    - allow_scale: Whether to allow variable determinant in A.
    - det_sigma: Spread of determinant in log space if scaling is allowed.
    - device: Device on which tensors are created.

    Returns:
    - (A, t): A tuple representing an affine transformation x ↦ A·x + t.
    """
    A = _random_gl(2, eps_aff, allow_scale, det_sigma, device)
    t = eps_trans * torch.randn(2, device=device)
    return A, t
