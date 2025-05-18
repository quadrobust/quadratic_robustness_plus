#!/usr/bin/env python3
"""
q_attack.py

Adaptive Quadratic‐PGD Attack
-----------------------------
Performs a projected gradient descent in the space of quadratic warp parameters
to craft adversarial examples against a given vision model.

We parameterize each attack as two affine perturbations around identity:
  A_R = I + δA_R, t_R     (pre‐warp)
  A_L = I + δA_L, t_L     (post‐warp)
with constraints ‖δA‖_F ≤ eps_aff and ‖t‖_2 ≤ eps_trans.

Functions:
  - q_pgd_batch(...):
      Runs the PGD attack across a batch, cycling through multiple canonical
      quadratic forms (beam search). Returns the best adversarial images and
      a mask of which samples were successfully attacked.

  - _clamp(delta, eps):
      Projects parameter tensors onto the appropriate norm‐ball
      (Frobenius or Euclidean) to enforce the perturbation budget.

  - _apply_quad_affine(x, form_id, dAr, dAl, tr, tl):
      Helper applying the learned warp parameters to a batch of images
      (loop per‐sample, as batch sizes are small).
"""

import torch, torch.nn.functional as F
from tqdm import tqdm

def _clamp(delta, eps):
    """
    Projects:
     - 2D tensor (B,2)  → each row onto L2-ball of radius eps
     - 3D tensor (B,2,2)→ each 2×2 matrix onto Frobenius-ball of radius eps
    """
    with torch.no_grad():
        flat  = delta.view(delta.size(0), -1)
        norm  = flat.norm(dim=1, keepdim=True) + 1e-12
        factor = torch.where(norm > eps, eps / norm, torch.ones_like(norm))
        delta.mul_(factor.view(-1, *([1]*(delta.ndim-1))))




def q_pgd_batch(model,                # target neural network model
                x, y,                 # input batch [B,3,H,W] and corresponding labels
                form_ids,            # list of canonical form IDs to try (beam search)
                eps_aff   = 0.3,     # maximum allowed Frobenius norm for δA
                eps_trans = 0.3,     # maximum allowed L2 norm for translation vectors
                iters     = 40,      # number of PGD steps
                step      = 0.05):   # PGD step size
    device = x.device
    B, _, H, W = x.shape
    best_adv = x.clone()  # will store the best adversarial examples found
    best_success = torch.zeros(B, dtype=torch.bool, device=device)  # success flags per sample

    criterion = torch.nn.CrossEntropyLoss(reduction='none')  # per-sample loss

    for fid in form_ids:
        # --- Select only samples that have not yet been successfully attacked ---
        remaining = (~best_success).nonzero(as_tuple=True)[0]
        if remaining.numel() == 0:  # The entire batch has already been successfully attacked
            break

        x_sub = x[remaining]
        y_sub = y[remaining]
        Bsub = x_sub.size(0)  # Batch size for the remaining samples

        # --- Initialize attack parameters only for the remaining samples ---
        dAr = torch.zeros(Bsub, 2, 2, device=device, requires_grad=True)  # Right-side affine matrix (perturbation)
        dAl = torch.zeros(Bsub, 2, 2, device=device, requires_grad=True)  # Left-side affine matrix (perturbation)
        tr  = torch.zeros(Bsub, 2,     device=device, requires_grad=True)  # Right-side translation vector
        tl  = torch.zeros(Bsub, 2,     device=device, requires_grad=True)  # Left-side translation vector

        # --- Perform iterative optimization to craft adversarial examples ---
        for _ in range(iters):
            adv = _apply_quad_affine(x_sub, fid, dAr, dAl, tr, tl)  # Apply quadratic affine transformation
            loss = criterion(model(adv), y_sub).sum()               # Compute loss with respect to true labels
            loss.backward()                                        # Backpropagate to compute gradients

            # Perform gradient ascent step for each parameter
            for p in (dAr, dAl, tr, tl):
                p.data.add_(step * p.grad.sign())  # Update in the direction of the gradient sign
                p.grad.zero_()                     # Reset gradients

            # Clamp the parameter values to enforce perturbation limits
            _clamp(dAr, eps_aff)
            _clamp(dAl, eps_aff)
            _clamp(tr,  eps_trans)
            _clamp(tl,  eps_trans)

        # --- Update the global best adversarial examples and success mask ---
        with torch.no_grad():
            adv = _apply_quad_affine(x_sub, fid, dAr, dAl, tr, tl)        # Re-apply transformation with final parameters
            new_success = model(adv).argmax(1) != y_sub                   # Determine which samples are now misclassified
            idx_global = remaining[new_success]                           # Map to global indices
            best_adv[idx_global] = adv[new_success]                       # Save adversarial examples
            best_success[idx_global] = True                               # Mark as successfully attacked

    return best_adv, best_success

# ---------- Helper: apply parameterized quadratic warp ----------------------
from src.geometry.warp import quadratic_warp

def _apply_quad_affine(x, fid, dAr, dAl, tr, tl):
    """
    Applies a batch of quadratic affine transformations to input images using
    parameterized left and right affine components (A = I + δA, t).
    """
    B = x.size(0)
    out = torch.empty_like(x)
    for i in range(B):  # loop is acceptable for batch size ~32
        out[i] = quadratic_warp(
            x[i:i+1],
            form_id   = fid,
            eps_aff   = 0.0,  # disable randomness, use only optimized deltas
            eps_trans = 0.0,
            device    = x.device,
            A_R       = torch.eye(2, device=x.device) + dAr[i],
            t_R       = tr[i],
            A_L       = torch.eye(2, device=x.device) + dAl[i],
            t_L       = tl[i],
        ).squeeze(0)
    return out
