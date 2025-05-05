# tests/test_warp.py
#
# Uruchamiaj:       pytest -q tests/test_warp.py
# Wymaga dowolnego JPEG‑a 224×224:   tests/dog.jpg
# ─────────────────────────────────────────────────────────
import torch, torchvision
from PIL import Image
from pathlib import Path
import sys, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

from src.geometry.warp import quadratic_warp

IMG_PATH = Path("root/tests/dog.jpg")      # wgraj cokolwiek z ImageNet‑val
assert IMG_PATH.exists(), "Wrzuć jeden obraz .jpg do tests/ !"

def test_quadratic_warp_sanity(tmp_path):
    """Obraz po warpie nie może być identyczny z wejściem, a tensor się zgadza."""
    img = Image.open(IMG_PATH)
    x   = torchvision.transforms.ToTensor()(img).unsqueeze(0).cuda()   # [1,3,H,W]
    warped = quadratic_warp(x, form_id=3).cpu()

    # 1) Ten sam rozmiar
    assert warped.shape == x.shape

    # 2) Nie‑zerowy błąd L2 (czyli faktycznie coś zniekształciliśmy)
    diff = torch.norm(warped - x).item()
    assert diff > 1e-2, "Warp nic nie zmienia!"

    # 3) Zapisywanie podglądu – opcjonalnie
    torchvision.utils.save_image(warped, tmp_path / "dog_warp.png")
