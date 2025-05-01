# tests/test_pipeline.py

import sys, os

import numpy as np
# 1) dodajemy katalog główny projektu na początek ścieżki importów
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import torch
from torchvision.io import read_image
from PIL import Image
import torchvision.transforms as T
from pathlib import Path
from src.geometry.warp import quadratic_warp

def test_pipeline_load_and_warp():
    test_dir = Path(__file__).resolve().parent
    project_root = test_dir.parent

    img_path = project_root / "data" / "imagenet" / "ILSVRC2012_val_00000001.JPEG"
    assert img_path.exists(), f"Brak pliku {img_path}"

    tensor = read_image(str(img_path)).unsqueeze(0).float().div(255.0)

    warped = quadratic_warp(tensor, form_id=1, strength=0.3)
    assert warped.shape == tensor.shape
