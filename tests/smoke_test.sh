#!/usr/bin/env bash
set -e

echo "=== Quick Smoke Test: 1 batch per model ==="
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"

python3 - "$ROOT" <<'PYCODE'
import time, torch, timm, sys
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from src.data.quad_aug import QuadAugTransform
from torchvision import transforms as T
ROOT = sys.argv[1]
device = "cuda" if torch.cuda.is_available() else "cpu"
models = ["resnet50", "efficientnet_b3", "vit_small_patch16_224"]

for model_name in models:
    print(f"\n> Model: {model_name}")
    # 1) Build tiny loader via build_imagenet_val (resize+crop embedded)
    from src.data.imagenet_loader import build_imagenet_val
    ds = build_imagenet_val(qc=True,
                            eps_aff=0.1,
                            eps_trans=0.1,
                            apply_normalize=True)
    loader = DataLoader(ds, batch_size=16,
                        shuffle=True, num_workers=4, pin_memory=True)
    # 2) One batch forward+backward
    model = timm.create_model(model_name, pretrained=True).to(device).train()
    opt   = AdamW(model.parameters(), lr=1e-4)
    loss_fn = CrossEntropyLoss()

    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)

    start = time.time()
    logits = model(xb)
    loss   = loss_fn(logits, yb)
    loss.backward()
    opt.step()
    dur = time.time() - start

    print(f"  ✔ one train step: loss={loss.item():.4f}, time={dur:.2f}s")
print("\nSmoke test passed.")
PYCODE
