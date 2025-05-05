#!/usr/bin/env bash
set -e

MODELS=(resnet50 efficientnet_b3 vit_small_patch16_224)

for MODEL in "${MODELS[@]}"; do
  echo "===>  Testing $MODEL"

  python - "$MODEL" << 'PY'
import sys, os
# dodajemy root do ścieżki, żeby importy działały:
root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, root)

import torch, timm, csv, pathlib
from src.data.imagenet_loader import loader

# nazwa modelu przekazana w sys.argv[1]
model_name = sys.argv[1]
device = "cuda"
model = timm.create_model(model_name, pretrained=True).to(device).eval()

def run(qc):
    top1, n = 0, 0
    for x, y in loader(batch=128, qc=qc):
        with torch.no_grad():
            logits = model(x.to(device, non_blocking=True))
        top1 += (logits.argmax(1).cpu() == y).sum().item()
        n    += y.size(0)
    return top1 / n * 100.0

acc_clean = run(False)
acc_qc    = run(True)

outdir = pathlib.Path("metrics")
outdir.mkdir(exist_ok=True)
with open(outdir / "baseline.csv", "a", newline="") as f:
    csv.writer(f).writerow([model_name, acc_clean, acc_qc])

print(f"{model_name}: clean {acc_clean:.2f}%, quadratic-C {acc_qc:.2f}%")
PY

done
