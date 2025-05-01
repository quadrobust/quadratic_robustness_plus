# tests/test_pipeline.py
import torch
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from src.geometry.warp import quadratic_warp

def test_pipeline_load_and_warp():
    # 1) Znajdź katalog tego pliku i wejdź o jeden poziom wyżej
    test_dir = Path(__file__).resolve().parent
    project_root = test_dir.parent

    # 2) Zbuduj ścieżkę do obrazka w data/imagenet
    img_path = project_root / "data" / "imagenet" / "n01514668" / "ILSVRC2012_val_00000001.JPEG"
    assert img_path.exists(), f"Brak pliku {img_path}"

    # 3) Załaduj obrazek i skonwertuj na tensor
    img = Image.open(img_path)
    tensor = T.ToTensor()(img).unsqueeze(0)  # [1,3,H,W]

    # 4) Wywołaj stub quadratic_warp i sprawdź kształt
    warped = quadratic_warp(tensor, form_id=1, strength=0.3)
    assert warped.shape == tensor.shape
