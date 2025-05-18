#!/usr/bin/env bash

#
# eval_baseline.sh
#
# Evaluate both standard (pretrained) and robust-bench ImageNet models
# on:
#   1) Clean ImageNet-val
#   2) Quadratic-C (15 canonical quadratic warps at multiple ε levels)
#
# Outputs a CSV (metrics/baseline.csv) and a human-readable TXT (metrics/baseline.txt).
#

set -e

# ───────────────────────────────────────────────────────
# Set root directory and extend PYTHONPATH accordingly
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
# ───────────────────────────────────────────────────────

# 2) Configuration
IMAGENET_BATCH=128
SEVERITIES=(0.10 0.15 0.20 0.25 0.30 0.50)

# List of standard models (pretrained from TIMM)
STD_MODELS=(resnet50 efficientnet_b3 vit_small_patch16_224)

# List of robust models (custom robust implementations)
ROB_MODELS=(Hendrycks2020AugMix Erichson2022NoisyMix_new)

# Output files
OUT_CSV=metrics/baseline.csv
OUT_TXT=metrics/baseline.txt
mkdir -p metrics
echo "model,type,eps,clean,qc" > $OUT_CSV

# Header for text output
cat > $OUT_TXT <<EOF
================ Baseline + Robust ================
(clean – ImageNet-val,  QC – Quadratic-C)
==================================================
EOF

# ───────────────────────────────────────────────────────
# Python inline evaluation: evaluate accuracy on clean and QC data
py_eval() {
python3 - "$@" <<'PY'
import sys, torch, timm, importlib
from src.data.imagenet_loader import loader
from src.utils.compute_log import ComputeLogger

MODEL = sys.argv[1]          # Model name
EPS   = float(sys.argv[2])   # Distortion severity (ε)
BATCH = int(sys.argv[3])     # Batch size
KIND  = sys.argv[4]          # "std" or "robust"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Load the model -----
if KIND == "std":
    model = timm.create_model(MODEL, pretrained=True).to(device).eval()
    apply_norm = True
else:
    rb = importlib.import_module("src.models.robust_models")
    model = rb.get_robust_model(MODEL, device=device)
    apply_norm = False                # Avoid double normalization for robust models

# ----- Accuracy computation function -----
def accuracy(qc: bool):
    tag = f"{MODEL}_{'qc' if qc else 'clean'}"
    with ComputeLogger(tag=tag, extra={"eps": EPS}):    
        dl = loader(batch           = BATCH,
                    qc              = qc,
                    eps_aff         = EPS,
                    eps_trans       = EPS,
                    apply_normalize = apply_norm)
        correct = total = 0
        for x,y in dl:
            with torch.no_grad():
                preds = model(x.to(device, non_blocking=True)).argmax(1).cpu()
            correct += (preds==y).sum().item()
            total   += y.size(0)
        return correct/total*100
    


# Evaluate on clean and QC-augmented data
clean = accuracy(False)
qc    = accuracy(True)
print(f"{clean:.2f} {qc:.2f}")
PY
}

# ───────────────────────────────────────────────────────
# Run evaluation for standard models
for MODEL in "${STD_MODELS[@]}"; do
  echo ">>> STD model: $MODEL"
  for EPS in "${SEVERITIES[@]}"; do
    read CLEAN QC <<< $(py_eval $MODEL $EPS $IMAGENET_BATCH std)
    echo "$MODEL,std,$EPS,$CLEAN,$QC" >> $OUT_CSV
    printf "%s std  eps=%s  clean=%.2f  qc=%.2f\n" \
           "$MODEL" "$EPS" "$CLEAN" "$QC" >> $OUT_TXT
  done
  echo "" >> $OUT_TXT
done

# Run evaluation for robust models
for MODEL in "${ROB_MODELS[@]}"; do
  echo ">>> ROB model: $MODEL"
  for EPS in "${SEVERITIES[@]}"; do
    read CLEAN QC <<< $(py_eval $MODEL $EPS $IMAGENET_BATCH robust)
    echo "$MODEL,robust,$EPS,$CLEAN,$QC" >> $OUT_CSV
    printf "%s robust  eps=%s  clean=%.2f  qc=%.2f\n" \
           "$MODEL" "$EPS" "$CLEAN" "$QC" >> $OUT_TXT
  done
  echo "" >> $OUT_TXT
done

# Final output message
echo -e "\n✓  Results saved to:\n   • CSV → $OUT_CSV\n   • TXT → $OUT_TXT"

