# src/geometry/lin_ops.py
# ---------------------------------------------------------
#  Losowe przekształcenia afiniczne:
#     GA(k) = GL(k) ⋉ ℝ^k   (x ↦ A·x + t)
# ---------------------------------------------------------
import torch, random
from typing import Tuple

# ---------------------------------------------------------
#  Pomocnicza funkcja ‑ losowa część liniowa GL(k)
# ---------------------------------------------------------
def _random_gl(k: int,
               eps_aff: float = 0.3,
               allow_scale: bool = False,      # *** nowy parametr
               det_sigma: float = 0.4,         # *** log‑σ gdy allow_scale=True
               device: str = "cpu") -> torch.Tensor:
    """
    Zwraca losową macierz GL(k).

    eps_aff   – odchylenie od identyczności (dodawany biały szum)
    allow_scale=False  →  normalizujemy, by |det|≈1
    allow_scale=True   →  det może różnić się:  det = det*A * exp(N(0,σ))
    det_sigma – rozproszenie skali w przestrzeni log‑det gdy allow_scale=True
    """
    A = torch.eye(k, device=device)
    A += eps_aff * torch.randn(k, k, device=device)

    # --- kontrola skali det(A) ---------------------------------
    if allow_scale:
        #  losowy mnożnik skali (log‑normal)
        scale = torch.exp(torch.randn(1, device=device) * det_sigma)
        A = A * scale          # zmieniamy det(A) ≈ scale^k
    else:
        # wymuś |det|≈1
        det = torch.det(A).abs()
        A = A / det**(1/k)
    return A

# ---------------------------------------------------------
#  GA(2)
# ---------------------------------------------------------
def sample_ga2(eps_aff: float = 0.3,
               eps_trans: float = 0.5,
               allow_scale: bool = False,
               det_sigma: float = 0.4,
               device: str = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Zwraca (A, t)  – affine 2D.
    t generujemy z N(0, eps_trans^2).
    """
    A = _random_gl(2, eps_aff, allow_scale, det_sigma, device)
    t = eps_trans * torch.randn(2, device=device)
    return A, t
