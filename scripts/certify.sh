#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────────────────────────────────────
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
# ───────────────────────────────────────────────────────

# Modele do certyfikacji
MODELS=(resnet50 efficientnet_b3 vit_small_patch16_224 \
        resnet50_qaug efficientnet_b3_qaug vit_small_patch16_224_qaug \
        Hendrycks2020AugMix Erichson2022NoisyMix_new)

BATCH=48
SUBSET=5000
SEV_LIST=(0.02 0.04 0.06 0.08 0.10)
GRID=5   # liczba punktów w gridzie (np. 5×5)

mkdir -p metrics
CSV=metrics/certify_sweep.csv
TXT=metrics/certify_sweep.txt

# Nagłówki
echo "=== Quadratic Certification Sweep ===" > $TXT
echo "model,form_id,severity,grid_size,CRA,mean_acc,worst_acc,seconds" > $CSV

# Przygotuj indeksy
IDX=metrics/indices_val.txt
if [ ! -f "$IDX" ]; then
  python3 - <<PY
import random
random.seed(42)
idx = random.sample(range(50000), $SUBSET)
with open("$IDX","w") as f:
    for i in idx:
        f.write(f"{i}\n")
PY
fi

# Pobierz listę form
FORM_IDS=( $(python3 - <<PY
from src.geometry.canonical_forms import FORMS
print(' '.join(str(k) for k in sorted(FORMS.keys())))
PY
) )

# Główna pętla
for M in "${MODELS[@]}"; do
  echo ">>> Certification for model: $M" | tee -a $TXT

  for FORM_ID in "${FORM_IDS[@]}"; do
    for SEP in "${SEV_LIST[@]}"; do
      echo "  * form=$FORM_ID severity=$SEP" >> $TXT

      # uruchamiamy certify_instance bezpośrednio
            read CRA mean worst < <(python3 - <<PY
import json, torch, timm, importlib
from torch.utils.data import Subset, DataLoader
from src.data.imagenet_loader import build_imagenet_val
from src.certify.cert_enum import certify_instance
from src.utils.compute_log import ComputeLogger

name = "$M"
device = "cuda" if torch.cuda.is_available() else "cpu"

# --- 1) Load model --------------------------------
if name.endswith("_qaug"):
    base = name.replace("_qaug", "")
    model = timm.create_model(base, pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(f"models/{base}_qaug.pth", map_location=device))
    apply_norm = False
elif name in ["resnet50", "efficientnet_b3", "vit_small_patch16_224"]:
    model = timm.create_model(name, pretrained=True).to(device).eval()
    apply_norm = True
else:
    rb = importlib.import_module("src.models.robust_models")
    model = rb.get_robust_model(name, device=device)
    apply_norm = False

# --- 2) DataLoader -------------------------------
ds = build_imagenet_val(qc=False,
                        eps_aff=0.0,
                        eps_trans=0.0,
                        apply_normalize=apply_norm)
idx = [int(x.strip()) for x in open("${IDX}")]
loader = DataLoader(Subset(ds, idx),
                    batch_size=$BATCH,
                    shuffle=False,
                    num_workers=8,
                    pin_memory=True)

# --- 3) Certification -----------------------------
with ComputeLogger(tag=f"certify_{name}",
                   extra={"form_id":$FORM_ID, "severity":$SEP, "grid":$GRID}):
    cra, mean_acc, worst_acc = certify_instance(
        model, loader,
        form_id=$FORM_ID,
        severity=$SEP,
        grid_size=$GRID,
        device=device
    )

# --- 4) Output -----------------------------
print(f"{cra:.4f} {mean_acc:.4f} {worst_acc:.4f}")
PY
      )

      # Zapisz do CSV i TXT
      echo "$M,$FORM_ID,$SEP,$GRID,$CRA,$mean,$worst" >> $CSV
      echo "      -> CRA = $CRA" >> $TXT

    done
  done

  echo "✓ Completed certification for $M" | tee -a $TXT
done

echo "✓ All certifications done → $CSV" | tee -a $TXT
