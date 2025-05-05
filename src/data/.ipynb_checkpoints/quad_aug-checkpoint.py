# src/data/quad_aug.py  (zamień całość)
import sys, os

root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import random
import torchvision.transforms as T
from src.geometry.warp import quadratic_warp    # ⬅ już nowa wersja

class QuadAugTransform:
    """
    Random Quadratic Warp  (R^2→R^2, affine‑equivalent).
    p           – prawdopodobieństwo zastosowania
    form_ids    – lista reprezentantów orbit
    eps_aff     – odchyłka części liniowej macierzy A_L, A_R
    eps_trans   – skala translacji
    """
    def __init__(self,
                 p        = 0.7,
                 form_ids = None,
                 eps_aff  = 0.30,
                 eps_trans= 0.30):
        self.p         = p
        self.form_ids  = form_ids or list(range(1,22))  # 1..21
        self.eps_aff   = eps_aff
        self.eps_trans = eps_trans
        self.to_tensor = T.ToTensor()

    def __call__(self, img):
        x = self.to_tensor(img)
        if random.random() < self.p:
            fid = random.choice(self.form_ids)
            x = quadratic_warp(x.unsqueeze(0),
                               form_id   = fid,
                               eps_aff   = self.eps_aff,
                               eps_trans = self.eps_trans
                               ).squeeze(0)
        return x
