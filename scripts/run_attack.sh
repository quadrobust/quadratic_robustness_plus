#!/usr/bin/env bash
set -e

# Define the root directory of the project
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:$PYTHONPATH"

# List of models to evaluate under adversarial attacks
#MODELS=(resnet50 resnet50_qaug Hendrycks2020AugMix)
MODELS=(resnet50 efficientnet_b3 vit_small_patch16_224 resnet50_qaug efficientnet_b3_qaug vit_small_patch16_224_qaug Hendrycks2020AugMix Erichson2022NoisyMix_new)

# Attack parameters
BATCH=48        # Batch size for data loading
SUBSET=5000     # Number of validation samples to use
ITERS=20        # Number of attack iterations
#EPS_LIST=(0.1)
EPS_LIST=(0.10 0.15 0.20 0.25 0.30 0.50)  # List of perturbation strengths to test

# Create directory for saving attack results
mkdir -p metrics
OUT_CSV=metrics/attack_sweep.csv

# Initialize CSV file for storing results
echo "model,eps,CA,ASR,RA,conf_drop" > $OUT_CSV
# Initialize TXT file for human-readable summary
TXT=metrics/attack_sweep.txt
echo "=== Quadratic-PGD Attack Sweep ===" > $TXT
echo "(model, eps, CA, ASR, RA, conf_drop)" >> $TXT

# Prepare a fixed list of image indices (for reproducibility)
IDX=metrics/indices_5k.txt
[ -f $IDX ] || python3 - <<PY
import random
random.seed(42)
idx = random.sample(range(50000), $SUBSET)
with open("$IDX","w") as f:
    for i in idx: f.write(f"{i}\n")
PY

# Perform adversarial attack sweep for each model and perturbation strength
for M in "${MODELS[@]}"; do
  echo ">>> Attack on model: $M"
  for eps in "${EPS_LIST[@]}"; do
    echo "   * eps=$eps"
    CSV_LINE=$(python3 - <<PY
import torch, timm, importlib
import torch.nn.functional as F
from torch.utils.data import Subset, DataLoader
from src.data.imagenet_loader import build_imagenet_val
from src.attacks.q_attack import q_pgd_batch
from src.geometry.canonical_forms import FORMS
from src.models.robust_models import get_robust_model
from src.utils.compute_log import ComputeLogger


# Setup model
name="$M"; device="cuda"
if name.endswith("_qaug"):
    # Load quantization-augmented model from checkpoint
    base = name.replace("_qaug","")
    model = timm.create_model(base, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(f"models/{base}_qaug.pth"))
    ds  = build_imagenet_val(qc=False,
                         eps_aff=$eps,
                         eps_trans=$eps)
elif name in ["resnet50","efficientnet_b3","vit_small_patch16_224"]:
    # Load standard pretrained model from timm
    model = timm.create_model(name, pretrained=True).to(device).eval()
    apply_norm = True
    ds  = build_imagenet_val(qc=False,
                         eps_aff=$eps,
                         eps_trans=$eps,
                         apply_normalize=apply_norm)
elif name in ["Hendrycks2020AugMix","Erichson2022NoisyMix_new"]:
    rb = importlib.import_module("src.models.robust_models")
    model = rb.get_robust_model(name, device=device)
    apply_norm = False                # Avoid double normalization for robust models
    ds  = build_imagenet_val(qc=False,
                         eps_aff=$eps,
                         eps_trans=$eps,
                         apply_normalize=apply_norm)
else:
    # fallback: RobustBench
    rb = importlib.import_module("src.models.robust_models")
    model = rb.get_robust_model(name).to(device)
    ds  = build_imagenet_val(qc=False,
                         eps_aff=$eps,
                         eps_trans=$eps,
                         apply_normalize=apply_norm)

# Load validation dataset subset
idx = [int(l) for l in open("$IDX")]

loader = DataLoader(Subset(ds, idx),
                    batch_size=$BATCH, shuffle=False,
                    num_workers=8, pin_memory=True)
 
# Run attack and collect statistics
total = 0
total_all = 0
success = 0
conf_drop = 0.0

for x, y in loader:
    x, y = x.to(device), y.to(device)

    # --- Only correctly classified clean examples -------------------------
     
    with torch.no_grad():
        logits_clean = model(x)   
    
    clean_pred = logits_clean.argmax(1)
    mask = clean_pred == y
    total_all += y.size(0)
    if mask.sum() == 0:
        continue

    total += mask.sum().item()
    conf_clean = F.softmax(logits_clean[mask], dim=1)[torch.arange(mask.sum(), device=device), y[mask]]

    # --- Adversarial attack  (log compute) ----------------------------------
    with ComputeLogger(tag=f"attack_{name}",
                       extra={"eps":$eps,"iters":$ITERS,"subset":$SUBSET}):
        adv, succ = q_pgd_batch(model,
                                x[mask], y[mask],
                                form_ids=list(FORMS.keys()),
                                eps_aff=$eps, eps_trans=$eps,
                                iters=$ITERS, step=$eps/10)


    success += succ.sum().item()

    with torch.no_grad():
        logits_adv = model(adv)
    conf_adv = F.softmax(logits_adv, dim=1)[torch.arange(mask.sum(), device=device), y[mask]]
    conf_drop += (conf_clean - conf_adv).sum().item()

# CA: Clean Accuracy — fraction of the subset that was correctly classified before the attack
# ASR: Attack Success Rate — proportion of correct examples that were misclassified after the attack
# RA: Robust Accuracy — proportion of correct examples that remained correct after the attack
# avg_conf_drop: average drop in confidence due to the attack
CA  = total / total_all if total_all else 0.0
ASR = success / total if total else 0.0
RA  = 1.0 - ASR
avg_conf_drop = conf_drop / total if total else 0.0
print(f"{CA:.3f},{ASR:.3f},{RA:.3f},{avg_conf_drop:.3f}")

PY
)
    
    IFS=, read -r CA_val ASR_val RA_val CD_val <<< "$CSV_LINE"

    
    echo "$M,$eps,$CSV_LINE" >> $OUT_CSV
    printf "%-30s  eps=%4s  CA=%5.3f  ASR=%5.3f  RA=%5.3f  conf_drop=%5.3f\n" \
           "$M" "$eps" "$CA_val" "$ASR_val" "$RA_val" "$CD_val" >> $TXT
  done
done

# Final message
echo "✓ Attack sweep done → $OUT_CSV"
