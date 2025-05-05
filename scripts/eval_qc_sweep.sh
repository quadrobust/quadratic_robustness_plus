#!/usr/bin/env bash
set -e

# ───────────────────────────────────────────────────────
# 1) Set PYTHONPATH to the root directory of the project
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
# ───────────────────────────────────────────────────────

# 2) Define models and deformation severity levels
MODELS=(resnet50 efficientnet_b3 vit_small_patch16_224)
SEVERITIES=(0.5 0.1 0.15 0.2 0.25 0.3)
BATCH=128

# 3) Prepare output directories and files
mkdir -p metrics
OUT_CSV=metrics/qc_sweep.csv
OUT_TXT=metrics/qc_sweep.txt

# Write CSV header
echo "model,eps,clean_acc,qc_acc" > "$OUT_CSV"

# Write TXT file header for human-readable results
cat > "$OUT_TXT" <<EOF
===========================================
  Quadratic-C Robustness Sweep Results
===========================================

EOF

# 4) Main loop over all models
for MODEL in "${MODELS[@]}"; do
  echo ">>> Sweeping model: $MODEL"

  # 4.1) Compute clean accuracy (no deformation)
  CLEAN=$(python3 - <<PYCODE
import torch, timm
from src.data.imagenet_loader import loader

model = timm.create_model("$MODEL", pretrained=True).cuda().eval()
correct, total = 0, 0
for x, y in loader(batch=$BATCH, qc=False):
    with torch.no_grad():
        logits = model(x.cuda(non_blocking=True))
    correct += (logits.argmax(1).cpu() == y).sum().item()
    total   += y.size(0)
print(correct/total*100)
PYCODE
)

  # 4.2) Write model section header to TXT file
  {
    echo "Model: $MODEL"
    echo -e "eps\tclean\tqc"
  } >> "$OUT_TXT"

  # 4.3) Loop over all deformation severity levels
  for eps in "${SEVERITIES[@]}"; do
    echo "  * eps=$eps"

    QC=$(python3 - <<PYCODE
import torch, timm
from src.data.imagenet_loader import loader

model = timm.create_model("$MODEL", pretrained=True).cuda().eval()
correct, total = 0, 0
for x, y in loader(batch=$BATCH, qc=True, eps_aff=$eps, eps_trans=$eps):
    with torch.no_grad():
        logits = model(x.cuda(non_blocking=True))
        correct += (logits.argmax(1).cpu() == y).sum().item()
        total   += y.size(0)
print(correct/total*100)
PYCODE
)

    # 4.4) Append results to both CSV and TXT files
    echo "$MODEL,$eps,$CLEAN,$QC" >> "$OUT_CSV"
    printf "%5.2f\t%5.2f\t%5.2f\n" "$eps" "$CLEAN" "$QC" >> "$OUT_TXT"
  done

  # 4.5) Add a blank line in TXT file to separate models
  echo "" >> "$OUT_TXT"
done

# 5) Final message
echo "Sweep done. CSV saved at $OUT_CSV, TXT saved at $OUT_TXT"
