# src/certify/cert_enum.py
import argparse
import json
import time
import torch
from itertools import product
from torch.cuda.amp import autocast
from tqdm import tqdm
from src.geometry.warp import quadratic_warp

def certify_instance(model, loader, form_id: int, severity: float,
                     grid_size: int = 5, device: str = "cuda"):
    """
    Certified evaluation by enumerating all (eps_aff, eps_trans) on a grid.
    A sample is 'certified' if for every combination on that grid
    the warped input is classified correctly.

    Returns:
        CRA: fraction of samples certified robust
        mean_acc: same as CRA (for compatibility)
        worst_acc: same as CRA (we do not track per-subset minima here)
    """
    model = model.to(device).eval()
    # create grid of (eps_aff, eps_trans) values
    # from 0 to severity inclusive
    aff_values = torch.linspace(0, severity, grid_size).tolist()
    trans_values = torch.linspace(0, severity, grid_size).tolist()
    total_samples = 0
    certified_count = 0

    with torch.no_grad():
        for x, y in tqdm(loader, desc=f"Form {form_id} ϵ={severity}", leave=False):
            x, y = x.to(device), y.to(device)
            B = x.size(0)
            total_samples += B
            # start with all True
            robust_mask = torch.ones(B, dtype=torch.bool, device=device)

            # for each grid point, warp and test
            for eps_aff, eps_trans in product(aff_values, trans_values):
                if not robust_mask.any():
                    break   # nic więcej nie trzeba sprawdzać
                # warp only the subset still robust
                idx = robust_mask.nonzero(as_tuple=True)[0]
                x_sub = x[idx]
                y_sub = y[idx]
                # apply quadratic warp
                xw = quadratic_warp(
                    x_sub,
                    form_id=form_id,
                    eps_aff=eps_aff,
                    eps_trans=eps_trans,
                    device=device
                )
                with autocast():  # opcjonalnie bfloat16
                    preds = model(xw).argmax(dim=1)
                # update mask: tylko te, które dalej poprawne
                robust_mask[idx] = (preds == y_sub)

            # po wszystkich gridach, policz, ile zostało certyfikowanych
            certified_count += robust_mask.sum().item()

    CRA = certified_count / total_samples
    # zwracamy trójkę dla kompatybilności ze skryptem bash
    return CRA, CRA, CRA


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--model_arch', required=True)
    p.add_argument('--form_id',    type=int,   required=True)
    p.add_argument('--severity',   type=float, required=True)
    p.add_argument('--grid_size',  type=int,   default=5,
                   help="grid density per axis")
    p.add_argument('--subset',     type=int,   default=5000)
    p.add_argument('--batch',      type=int,   default=64)
    p.add_argument('--device',     default='cuda')
    args = p.parse_args()

    import timm, importlib
    from torch.utils.data import Subset, DataLoader
    from src.data.imagenet_loader import build_imagenet_val

    # --- load model ---
    if args.model_arch.endswith('_qaug'):
        arch = args.model_arch.replace('_qaug', '')
        model = timm.create_model(arch, pretrained=False)
        model.load_state_dict(torch.load(args.model_path,
                                         map_location='cpu'))
        apply_norm = False
    elif args.model_arch in ['resnet50','efficientnet_b3','vit_small_patch16_224']:
        model = timm.create_model(args.model_arch, pretrained=True)
        apply_norm = True
    else:
        rb = importlib.import_module('src.models.robust_models')
        model = rb.get_robust_model(args.model_arch, device=args.device)
        apply_norm = False

    # --- prepare dataset subset ---
    ds = build_imagenet_val(qc=False,
                            eps_aff=0., eps_trans=0.,
                            apply_normalize=apply_norm)
    # wczytaj deterministyczny indeks z pliku
    idx = [int(x) for x in open('metrics/indices_val.txt')]
    loader = DataLoader(Subset(ds, idx),
                        batch_size=args.batch, shuffle=False,
                        num_workers=8, pin_memory=True)

    tic = time.time()
    CRA, mean_acc, worst_acc = certify_instance(
        model, loader,
        form_id   = args.form_id,
        severity  = args.severity,
        grid_size = args.grid_size,
        device    = args.device
    )
    dur = time.time() - tic

    print(json.dumps({
        'CRA':        round(CRA,4),
        'mean_acc':   round(mean_acc,4),
        'worst_acc':  round(worst_acc,4),
        'seconds':    round(dur,2),
        'form_id':    args.form_id,
        'severity':   args.severity,
        'grid_size':  args.grid_size
    }))


if __name__ == '__main__':
    main()
