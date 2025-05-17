# Quadratic‑C & QuadraticAug

### Robustness to **Smooth Quadratic Warps** in ImageNet models

This repository accompanies our <u>research submission</u> on spatial robustness.  It contains 

* **Quadratic‑C** — an ImageNet‑val corruption benchmark built from 15 canonical *degree‑2* warps,
* **QuadraticAug** — a drop‑in `torchvision` transform that samples random quadratic warps during training,
* reference **training, evaluation, attack and certification** scripts reproducing the results in the paper draft.

The goal is to make it straightforward to **(i)** measure how fragile a vision backbone is to smooth non‑affine distortions and **(ii)** close part of that gap with a single data‑augmentation line.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Quick‑start](#quick-start)  
3. [Running Experiments](#running-experiments)
4. [Reproducing paper numbers](#reproducing-paper-results)
5. [License](#license)


---

## Repository Layout

```text
├── env.yml                      # Conda environment (Python ≥3.10, CUDA 11.8)
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

### 1 Clone & install

```bash
git clone https://github.com/quadrobust/quadratic_robustness_plus.git
cd quadratic_robustness_plus
conda env create -f env.yml
conda activate quadrob
```

### 2 Download **ImageNet‑1k validation** split

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

Below commands assume you are in the repo root and `$IMAGENET_ROOT` is set.

1. **Baseline accuracy (clean + Quadratic‑C)**

   ```bash
   bash scripts/eval_baseline.sh        # writes metrics/baseline.csv
   ```
2. **Fine‑tune a backbone with QuadraticAug**

   ```bash
   python src/train/finetune_quadaug.py \
          --model resnet50 --epochs 10 --batch-size 128 \
          --p 0.7 --eps-aff 0.3 --eps-trans 0.3 \
          --save-path models/resnet50_qaug.pth
   ```
3. **Evaluate the fine‑tuned checkpoint**

   ```bash
   bash scripts/eval_finetuned.sh       # outputs metrics/finetuned.csv
   ```
4. **Adaptive quadratic‑PGD attack**

   ```bash
   bash scripts/run_attack.sh           # outputs metrics/attack.csv
   ```
5. **Deterministic certificate (CRA)**

   ```bash
   bash scripts/certify.sh              # outputs metrics/cra.csv
   ```

Each shell wrapper simply passes sane defaults to the underlying Python module—you can open them to tweak severity lists, grid size, logging options, etc.

---

## Reproducing paper numbers

Running all five steps above for each backbone recreates the CSV files that back every figure/table in the submission.

Hardware used in the paper: **1× NVIDIA A100 80 GB**, runtime ≈24 GPU‑hours total.

---

## License

*Code* is released under the MIT License.
Quadratic‑C images are generated on‑the‑fly from the original **ImageNet‑1k validation** set, which is licensed for non‑commercial research – you must separately agree to the ImageNet terms.

