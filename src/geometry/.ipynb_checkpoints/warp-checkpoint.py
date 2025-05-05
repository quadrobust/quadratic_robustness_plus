import torch, torch.nn.functional as F
from canonical_forms import get_form
from lin_ops import sample_ga2

# ---------------------------------------------------------
#  Quadratic warp  =  (A_L,t_L) ∘ F_type ∘ (A_R,t_R)
# ---------------------------------------------------------
def quadratic_warp(img,                            # tensor [B,3,H,W]
                   form_id     = 1,
                   eps_aff     = 0.3,   # odchyłka liniowej części
                   eps_trans   = 0.3,   # skala translacji
                   device      = None,
                   seed        = None):
    
    if seed is not None:
        torch.manual_seed(seed)
    if device is None:
        device = img.device
    B,_,H,W = img.shape

    # ----------  siatka wejściowa  ------------------------
    yy, xx = torch.linspace(-1,1,H,device=device), torch.linspace(-1,1,W,device=device)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")   # [H,W]
    X0 = torch.stack((grid_x, grid_y), dim=0)                # [2,H,W]

    # ----------  (A_R, t_R)  =  affine‑pre  ---------------
    A_R, t_R = sample_ga2(eps_aff, eps_trans, device=device)   # 2×2 , 2
    X1 = A_R @ X0.reshape(2,-1) + t_R[:,None]                  # [2,H*W]
    x1, y1 = X1.reshape(2,H,W)

    # ----------  Quadratic forma (reprezentant orbit) -----
    u, v = get_form(form_id)(x1, y1)        # dokładnie para (u,v)

    # ----------  (A_L, t_L)  =  affine‑post  ---------------
    A_L, t_L = sample_ga2(eps_aff, eps_trans, device=device)
    U = A_L[0,0]*u + A_L[0,1]*v + t_L[0]
    V = A_L[1,0]*u + A_L[1,1]*v + t_L[1]

    # ----------  final grid  ------------------------------
    grid = torch.stack((U, V), dim=-1)               # [H,W,2]
    grid = grid.expand(B, H, W, 2)

    return F.grid_sample(img, grid, mode="bilinear",
                         padding_mode="border", align_corners=True)
