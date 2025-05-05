# src/data/imagenet_loader.py



from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms as T
from .quad_aug import QuadAugTransform

# standardowe statystyki ImageNet
_MEAN = [0.485, 0.456, 0.406]
_STD  = [0.229, 0.224, 0.225]

def build_imagenet_val(root      = "data/imagenet/val",
                       qc        = False,
                       eps_aff   = 0.30,
                       eps_trans = 0.30):
    """
    root  – katalog z *posortowanymi* podfolderami synsetów (valprep!)
    qc    – czy włączyć Quadratic‑C deformację
    eps_* – parametry gęstości losowych afinicznych macierzy A_L,A_R
    """

    if qc:
        # QuadAug → tensor  → Normalize
        transform = T.Compose([
            QuadAugTransform(p=1.0,
                             eps_aff=eps_aff,
                             eps_trans=eps_trans),
            T.Normalize(_MEAN, _STD)
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(_MEAN, _STD)
        ])

    return ImageFolder(root, transform=transform)

def loader(batch       = 64,
           qc          = False,
           eps_aff     = 0.30,
           eps_trans   = 0.30,
           num_workers = 4,
           root        = "data/imagenet/val"):
    ds = build_imagenet_val(root=root,
                            qc=qc,
                            eps_aff=eps_aff,
                            eps_trans=eps_trans)
    return DataLoader(ds,
                      batch_size   = batch,
                      shuffle      = False,
                      num_workers  = num_workers,
                      pin_memory   = True)
