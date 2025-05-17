#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fine-tuning with QuadraticAug for three models:
  • resnet50
  • efficientnet_b3
  • vit_small_patch16_224

Checkpoints are saved in the models/ folder.
Training report (hyperparameters, durations, final statistics) is saved to models/training_report.txt.
"""

import os, sys
# -----------------------------------------------------
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
# -----------------------------------------------------

import time
import argparse
import torch
import timm
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.datasets import ImageFolder

from src.data.quad_aug import QuadAugTransform
from src.data.imagenet_loader import loader
from src.data.imagenet_loader import build_imagenet_val


# --------------------------------------------------------------------
# 1) Training function for a single model
# --------------------------------------------------------------------
def train_model(model_name: str,
                eps: float,
                epochs: int,
                batch_size: int,
                lr: float,
                root: str,
                device: str = "cuda"):
    # Create output folder and prepare report path
    os.makedirs("models", exist_ok=True)
    report_path = os.path.join("metrics", "training_report.txt")
    ckpt_path   = os.path.join("models", f"{model_name}_qaug.pth")
    start_time  = time.time()

    # Build model and data loader
    model = timm.create_model(model_name, pretrained=True).to(device)
    model.train()
    
    train_ds = build_imagenet_val(qc=True,
                                  eps_aff=eps,
                                  eps_trans=eps,
                                  apply_normalize=True)
    train_loader = DataLoader(train_ds,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)



    optimizer = AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Write training configuration to the report
    with open(report_path, "a") as f:
        f.write(f"\n=== Training {model_name} ===\n")
        f.write(f"Hyperparams: eps_aff={eps}, eps_trans={eps}, batch_size={batch_size}, lr={lr}, epochs={epochs}\n")
        f.write(f"Start: {time.ctime(start_time)}\n")

    # Epoch loop with tqdm progress bar
    for epoch in range(1, epochs+1):
        epoch_loss = 0.0
        bar = tqdm(train_loader, desc=f"{model_name}  ep{epoch}/{epochs}", leave=False)
        for xb, yb in bar:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss   = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            bar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader.dataset)

        # Log average epoch loss to the report
        with open(report_path, "a") as f:
            f.write(f"Epoch {epoch}: avg_loss={avg_loss:.4f}\n")

        # Save intermediate checkpoint every 5 epochs
        if epoch % 5 == 0:
            torch.save(model.state_dict(), ckpt_path.replace(".pth", f"_ep{epoch}.pth"))

    # Save final checkpoint
    torch.save(model.state_dict(), ckpt_path)

    # --- Accuracy measurement after training ---
    model.eval()
    def compute_acc(qc: bool):
        dl = loader(batch=batch_size,
                    qc=qc,
                    eps_aff=eps,
                    eps_trans=eps)
        correct = total = 0
        with torch.no_grad():
            for xb,yb in dl:
                preds = model(xb.to(device,non_blocking=True)).argmax(1).cpu()
                correct += (preds==yb).sum().item()
                total   += yb.size(0)
        return correct/total*100

    clean_acc = compute_acc(qc=False)
    qc_acc    = compute_acc(qc=True)

    # Append accuracy and timing to the training report
    end_time = time.time()
    duration = end_time - start_time
    with open(report_path, "a") as f:
        f.write(f"Post-train clean_acc={clean_acc:.2f}%  qc_acc={qc_acc:.2f}%\n")
        f.write(f"Finished: {time.ctime(end_time)}  Duration: {duration/3600:.2f} h\n\n")

    print(f"✓ {model_name} trained in {duration/60:.1f} min, "
          f"clean={clean_acc:.2f}% qc={qc_acc:.2f}% → {ckpt_path}")


# --------------------------------------------------------------------
# 2) Main function: triggers training for all three models
# --------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root",   default="data/imagenet/val", help="Path to the ImageNet validation folder")
    parser.add_argument("--eps",    type=float, default=0.3,      help="ε for eps_aff and eps_trans")
    parser.add_argument("--epochs", type=int,   default=10,       help="Number of training epochs")
    parser.add_argument("--batch",  type=int,   default=128,      help="Batch size")
    parser.add_argument("--lr",     type=float, default=5e-5,     help="Learning rate")
    args = parser.parse_args()

    # Model list and custom batch sizes (ViT needs smaller batch)
    tasks = [
        ("resnet50",               args.batch),
        ("efficientnet_b3",        args.batch),
        ("vit_small_patch16_224",  max(1, args.batch//2))
    ]

    # Train each model with the specified configuration
    for model_name, bs in tasks:
        train_model(model_name,
                    eps         = args.eps,
                    epochs      = args.epochs,
                    batch_size  = bs,
                    lr          = args.lr,
                    root        = args.root)

if __name__ == "__main__":
    main()
