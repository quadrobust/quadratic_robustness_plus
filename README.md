
---
# Quadratic‑C & QuadraticAug
---

**Benchmark & augmentation for non‑affine spatial robustness**

This repository accompanies our paper *“Robustness to Smooth Quadratic Warps: The Quadratic‑C Benchmark and a Simple Data Augmentation”*.
It contains everything needed to **evaluate** existing ImageNet models under quadratic warps and to **train** new models that handle them better.

* **Quadratic‑C** – a corruption benchmark built from 15 canonical degree‑2 warps, rendered on the fly.
* **QuadraticAug** – a single‑line `torchvision` transform that samples random quadratic warps during training.
* End‑to‑end scripts for baseline evaluation, fine‑tuning, adaptive attack, and a deterministic robustness certificate.

Our aim is to make it painless to

1. quantify how severely quadratic distortions break a vision backbone, and
2. close part of that gap with a lightweight data‑augmentation pass.

---

## Table of Contents
---
1. [Repository Layout](#repository-layout)
2. [Quick‑start](#quick-start)  
3. [Running Experiments](#running-experiments)
4. [Reproducing paper numbers](#reproducing-paper-numbers)
5. [License & dataset usage](#license--dataset-usage)


---

## Repository Layout
---
```text
├── env.yml                      # Conda environment (name: quadrob)
├── src/                         # Core Python modules
│   ├── attacks/                 # Adaptive adversarial attacks
│   │   └── q_attack.py          # Quadratic‑PGD (parameter‑space) attack
│   ├── certify/                 # Deterministic certificate
│   │   └── cert_enum.py         # Exhaustive enumeration over warp grid
│   ├── data/                    # Data loading & augmentation
│   │   ├── imagenet_loader.py   # Thin wrapper around torchvision ImageFolder
│   │   └── quad_aug.py          # QuadraticAug – torchvision‑style transform
│   ├── geometry/                # Math backend (degree‑2 maps)
│   │   ├── canonical_forms.py   # 15 canonical quadratic mappings
│   │   ├── lin_ops.py           # Linear / affine helpers
│   │   └── warp.py              # Differentiable quadratic warp operator
│   ├── models/                  # Model zoo & checkpoint helpers
│   │   └── robust_models.py     # ResNet‑50, EfficientNet‑B3, ViT‑Small + RB baselines
│   ├── train/                   # Training / fine‑tuning
│   │   └── finetune_quadaug.py  # Fine‑tune backbone with QuadraticAug
│   └── utils/
│       └── compute_log.py       # Lightweight JSONL runtime/metric logger
├── scripts/                     # Convenience bash wrappers
│   ├── certify.sh               # Run CRA certificate sweep
│   ├── eval_baseline.sh         # Evaluate pretrained models
│   ├── eval_finetuned.sh        # Evaluate QuadraticAug‑fine‑tuned checkpoints
│   ├── run_attack.sh            # Adaptive quadratic‑PGD attack sweep
│   └── prepare_imagenet_val.py  # Re‑index ImageNet‑val into class folders
├── metrics/                     # CSV / TXT outputs reproduced for the paper
└── data/                        # Will contain ImageNet after manual download
```

---

## Quick‑start
---
### Clone & install

```bash
git clone https://github.com/quadrobust/quadratic_robustness_plus.git
cd quadratic_robustness_plus
conda env create -f env.yml
conda activate quadrob
```

### Download **ImageNet‑1k validation** split

Only the 50 000 validation JPEGs are needed (≈6.3 GB).

```bash
mkdir -p data/imagenet && cd data/imagenet
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar
wget https://image-net.org/data/ILSVRC/2012/ILSVRC2012_devkit_t12.tar.gz

tar -xf ILSVRC2012_img_val.tar  && rm ILSVRC2012_img_val.tar
tar -xf ILSVRC2012_devkit_t12.tar.gz && rm ILSVRC2012_devkit_t12.tar.gz
cd ../../
python scripts/prepare_imagenet_val.py --root data/imagenet
```

---

## Running experiments
---
Below commands assume you are in the repo root.

1. **Baseline accuracy (clean + Quadratic‑C)**

   ```bash
   bash scripts/eval_baseline.sh        # writes metrics/eval_baseline.csv (.txt)
   ```
2. **Fine‑tune a backbone with QuadraticAug**

   ```bash
   python3 src/train/finetune_quadaug.py   --eps    0.3   --epochs 10   --batch  128   --lr     5e-5   2>&1
   ```
3. **Evaluate the fine‑tuned checkpoint**

   ```bash
   bash scripts/eval_finetuned.sh       # outputs metrics/eval_finetuned.csv (.txt)
   ```
4. **Adaptive quadratic‑PGD attack**

   ```bash
   bash scripts/run_attack.sh           # outputs metrics/attack_sweep.csv (.txt)
   ```
5. **Deterministic certificate (CRA)**

   ```bash
   bash scripts/certify.sh              # outputs metrics/certify.csv (.txt)
   ```

Each shell wrapper simply passes sane defaults to the underlying Python module—you can open them to tweak severity lists, grid size, logging options, etc.

---

## Reproducing paper numbers
---
Running all five steps above for each backbone recreates the CSV files that back every figure/table in the submission.

Hardware used in the paper: **1× NVIDIA A100 80 GB**, runtime ≈24 GPU‑hours total.

---

## License & dataset usage
---
### Code

All source code in this repository is released under the permissive **MIT License**. You are free to use, modify and redistribute it, provided that the license notice remains in every derived file.

### ImageNet‑1k validation images

Quadratic‑C is a *streaming* corruption: it warps images **on‑the‑fly** in GPU memory. **No JPEGs are checked into this repo and no derivative images are saved to disk by default.**

Our own experiments were carried out under an approved **academic ImageNet licence** and strictly follow the obligations quoted below: the data were used *only* for non‑commercial research, kept on internal storage, and never redistributed. We instruct every user to do the same.

**By running any script in this repo you acknowledge that you have a valid licence and that you comply with its conditions.**  Commercial use requires a separate agreement with the ImageNet administrators.

---




