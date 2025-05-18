#!/usr/bin/env bash

#
# eval_finetuned.sh
#
# Evaluate QuadraticAug fine-tuned ImageNet backbones on:
#   1) Clean ImageNet-val
#   2) Quadratic-C (15 canonical quadratic warps at multiple ε levels)
#
# Assumes that checkpoints (*.pth) live under models/ and writes:
#   • metrics/finetuned.csv – comma-separated results
#   • metrics/finetuned.txt – human-readable log
#

set -e

# ───────────────────────────────────────────────────────
# 1) Set the root project path
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT_DIR:$PYTHONPATH"
# This ensures Python modules can be imported correctly from the project root.
# ───────────────────────────────────────────────────────

# 2) Configuration
IMAGENET_BATCH=128  # Batch size for ImageNet evaluation
SEVERITIES=(0.10 0.15 0.20 0.25 0.30 0.50)  # List of ε values for QC perturbation severity

# Define fine-tuned models (corresponding .pth files must exist in the models/ directory)
FT_MODELS=(resnet50 efficientnet_b3 vit_small_patch16_224)

# Checkpoint paths for each fine-tuned model
declare -A CKPT
CKPT[resnet50]="models/resnet50_qaug.pth"
CKPT[efficientnet_b3]="models/efficientnet_b3_qaug.pth"
CKPT[vit_small_patch16_224]="models/vit_small_patch16_224_qaug.pth"

# Output files
OUT_CSV=metrics/finetuned.csv  # Path to output CSV with results
OUT_TXT=metrics/finetuned.txt  # Path to output TXT with logs
mkdir -p metrics  # Ensure output directory exists

# CSV header
echo "model,eps,clean,qc" > "$OUT_CSV"

# TXT header
cat > "$OUT_TXT" <<EOF
================  Evaluation of Fine-Tuned Models  ================
(clean – ImageNet-val,  QC – Quadratic-C; fine-tuned with ε=0.30)
====================================================================
EOF

# ───────────────────────────────────────────────────────
# 3) Inline Python function for accuracy evaluation
#    Args: MODEL EPS BATCH
#    Evaluates clean accuracy and QC (Quadratic Corruption) accuracy.
py_eval_ft() {
python3 - "$@" <<'PY'
import sys, torch, timm
from src.data.imagenet_loader import loader
from src.utils.compute_log import ComputeLogger

MODEL = sys.argv[1]
EPS   = float(sys.argv[2])
BATCH = int(sys.argv[3])
CKPT  = sys.argv[4]

device = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Load the model architecture and weights -----
model = timm.create_model(MODEL, pretrained=False).to(device).eval()
state = torch.load(CKPT, map_location=device)
model.load_state_dict(state)
 
# ----- Accuracy computation -----
# Evaluates the accuracy of the model on either clean or QC-perturbed data.
def accuracy(qc: bool):
    tag = f"{MODEL}_{'qc' if qc else 'clean'}"
    with ComputeLogger(tag=tag, extra={"eps": EPS}): 
        dl = loader(batch     = BATCH,
                    qc        = qc,
                    eps_aff   = EPS,
                    eps_trans = EPS)
        correct = total = 0
        for x,y in dl:
            with torch.no_grad():
                preds = model(x.to(device, non_blocking=True)).argmax(1).cpu()
            correct += (preds==y).sum().item()
            total   += y.size(0)
        return correct/total*100

clean = accuracy(False)  # Accuracy on clean (unperturbed) data
qc    = accuracy(True)   # Accuracy on QC-perturbed data
print(f"{clean:.2f} {qc:.2f}")
PY
}

# ───────────────────────────────────────────────────────
# 4) Loop over all fine-tuned models
for M in "${FT_MODELS[@]}"; do
  CKPT_PATH=${CKPT[$M]}
  echo -e "\n>>> Fine-tuned model: $M (ckpt: $CKPT_PATH)"

  for EPS in "${SEVERITIES[@]}"; do
    # Read clean and QC accuracies from the inline Python script
    read CLEAN QC <<< $(py_eval_ft $M $EPS $IMAGENET_BATCH $CKPT_PATH)
    
    # Append results to CSV
    echo "$M,$EPS,$CLEAN,$QC" >> "$OUT_CSV"
    
    # Append formatted results to TXT log
    printf "%-25s  eps=%4.2f  clean=%6.2f  qc=%6.2f\n" \
           "$M" "$EPS" "$CLEAN" "$QC" >> "$OUT_TXT"
  done

  echo "" >> "$OUT_TXT"  # Add spacing between model sections
done

# ───────────────────────────────────────────────────────
# 5) Final status message
echo -e "\n✓  Fine-tuned eval done. Results saved to:"
echo "   • CSV → $OUT_CSV"
echo "   • TXT → $OUT_TXT"
