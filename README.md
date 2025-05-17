# Quadratic‑C & QuadraticAug

Robustness to **Smooth Quadratic Warps**

> Reference implementation for the NeurIPS 2025 paper:
> **“Robustness to Smooth Quadratic Warps: The Quadratic‑C Benchmark and a Simple Data Augmentation.”**
>
> This repo provides the benchmark, augmentation, training scripts, attacks and a deterministic certificate used in the paper.

---

## Table of Contents

1. [Repository Layout](#repository-layout)
2. [Quick‑start](#quick-start)
      2.1 Install the environment  |  2.2 Download ImageNet‑val  
3. [Running Experiments](#running-experiments)
4. [Reproducing Paper Results](#reproducing-paper-results)
5. [Python API](#python-api)
8. [License](#license)


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
│
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

### 2.1 Install the environment

```bash
# clone
git clone https://github.com/quadrobust/quadratic_robustness_plus.git
cd quadratic_robustness_plus

# create conda env
conda env create -f env.yml
conda activate quadrob
```

### 2.2 Download **ImageNet‑1k validation** split

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

*All commands assume the repository root and a valid `IMAGENET_ROOT`.*

| Task                                        | Command                                                                 | Output                           |
| ------------------------------------------- | ----------------------------------------------------------------------- | -------------------------------- |
| **Baseline accuracy** (clean & Quadratic‑C) | `bash scripts/eval_baseline.sh`                                         | `metrics/baseline.csv`           |
| **Fine‑tune with QuadraticAug**             | `python src/train/finetune_quadaug.py --model resnet50 --epochs 10 ...` | `models/resnet50_qaug.pth` + log |
| **Evaluate fine‑tuned**                     | `bash scripts/eval_finetuned.sh`                                        | `metrics/finetuned.csv`          |
| **Adaptive quadratic‑PGD attack**           | `bash scripts/run_attack.sh`                                            | `metrics/attack.csv`             |
| **Certified Robust Accuracy (CRA)**         | `bash scripts/certify.sh`                                               | `metrics/cra.csv`                |

See each script for optional flags (severity list, grid size, checkpoint paths, wandb logging, …).



---

## License

*Code* is released under the MIT License.
Quadratic‑C images are generated on‑the‑fly from the original **ImageNet‑1k validation** set, which is licensed for non‑commercial research – you must separately agree to the ImageNet terms.
