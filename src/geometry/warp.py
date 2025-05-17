import torch, torch.nn.functional as F
from .canonical_forms import get_form
from .lin_ops import sample_ga2

# ---------------------------------------------------------
#  Quadratic warp = (A_L, t_L) ∘ F_type ∘ (A_R, t_R)
#  Applies a learned quadratic deformation to the image by:
#  1) Applying a random affine transformation (A_R, t_R)
#  2) Passing through a canonical quadratic form
#  3) Applying another random affine transformation (A_L, t_L)
# ---------------------------------------------------------

def quadratic_warp(img,                            # Input image tensor [B, 3, H, W]
                   form_id     = 1,                # ID of the quadratic form to use
                   eps_aff     = 0.3,              # Magnitude of affine transformation
                   eps_trans   = 0.3,              # Magnitude of translation component
                   device      = None,             # Device for computation
                   seed        = None,             # Optional seed for reproducibility
                   A_R         = None,
                   t_R         = None,
                   A_L         = None,
                   t_L         = None):            
    """
    Applies a quadratic warp to a batch of images using a fixed quadratic form
    and randomized affine transformations before and after the warping.

    Parameters:
        img (torch.Tensor): Input image tensor of shape [B, 3, H, W]
        form_id (int): ID of the canonical quadratic form (used in get_form)
        eps_aff (float): Standard deviation for sampling affine matrices
        eps_trans (float): Standard deviation for sampling translations
        device (torch.device): Device on which to run computation
        seed (int, optional): Seed for random number generation

    Returns:
        torch.Tensor: Warped image tensor of shape [B, 3, H, W]
    """
    
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = img.device
    B, _, H, W = img.shape

    # ----------  Generate input grid  ------------------------
    yy, xx = torch.linspace(-1, 1, H, device=device), torch.linspace(-1, 1, W, device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")   # meshgrid of shape [H, W]
    X0 = torch.stack((grid_x, grid_y), dim=0)                # stack into [2, H, W]

    # ----------  First affine transform (A_R, t_R)  ----------
    # Applies a sampled affine transform to the grid (pre-warp)
    if A_R is None or t_R is None:
        A_R, t_R = sample_ga2(eps_aff, eps_trans, device=device)  # 2×2 affine matrix and 2D translation vector
    X1 = A_R @ X0.reshape(2, -1) + t_R[:, None]               # Apply affine: shape [2, H*W]
    x1, y1 = X1.reshape(2, H, W)                              # Reshape back to [2, H, W]

    # ----------  Canonical quadratic form --------------------
    # Applies the quadratic form to the coordinates
    u, v = get_form(form_id)(x1, y1)                          # Maps (x1, y1) → (u, v)

    # ----------  Second affine transform (A_L, t_L) ----------
    # Applies another sampled affine transform to the warped coordinates
    if A_L is None or t_L is None:
        A_L, t_L = sample_ga2(eps_aff, eps_trans, device=device)
    U = A_L[0, 0] * u + A_L[0, 1] * v + t_L[0]
    V = A_L[1, 0] * u + A_L[1, 1] * v + t_L[1]

    # ----------  Final sampling grid --------------------------
    # Stack and expand the warped coordinates into grid shape
    grid = torch.stack((U, V), dim=-1)                        # [H, W, 2]
    grid = grid.expand(B, H, W, 2)                            # [B, H, W, 2]

    # ----------  Sample the image with the new grid  ----------
    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)
