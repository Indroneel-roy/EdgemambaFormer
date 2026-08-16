#!/usr/bin/env python
"""
Evaluate a trained EdgeMambaFormer checkpoint on all four test sets
(Section 4.5, Table 1): Kvasir-SEG and CVC-ClinicDB (in-distribution),
CVC-ColonDB and ETIS-LaribPolypDB (zero-shot).

Usage:
    python scripts/evaluate.py \
        --checkpoint ./checkpoints/best_shared_edgemambaformer.pth \
        --kvasir /path/to/Kvasir-SEG \
        --clinicdb /path/to/CVC-ClinicDB \
        --colondb /path/to/CVC-ColonDB \
        --etis /path/to/ETIS-LaribPolypDB \
        --out results.json
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import (
    build_pairs_kvasir, build_pairs_clinicdb, build_pairs_flat,
    stratified_split, make_loader, get_val_transforms,
)
from model import build_model

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate EdgeMambaFormer on all four test sets")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--kvasir", required=True)
    p.add_argument("--clinicdb", required=True)
    p.add_argument("--colondb", required=True)
    p.add_argument("--etis", required=True)
    p.add_argument("--img-size", type=int, default=352)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--out", default=None, help="Optional path to save results as JSON")
    return p.parse_args()


@torch.no_grad()
def evaluate_test_set(model, pairs, cfg, device, name, threshold=0.5):
    loader = make_loader(pairs, get_val_transforms(cfg["img_size"]), cfg, shuffle=False)
    all_dice, all_iou, all_mae = [], [], []

    for imgs, masks in tqdm(loader, desc=f"Eval {name}", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        pred_bin = (torch.sigmoid(outputs["pred"]) > threshold).float()
        for b in range(imgs.shape[0]):
            p = pred_bin[b].view(-1)
            t = masks[b].view(-1)
            tp = (p * t).sum().item()
            fp = (p * (1 - t)).sum().item()
            fn = ((1 - p) * t).sum().item()
            all_dice.append(2 * tp / (2 * tp + fp + fn + 1e-8))
            all_iou.append(tp / (tp + fp + fn + 1e-8))
            all_mae.append((p - t).abs().mean().item())

    return {
        "name": name, "n": len(pairs),
        "dice": float(np.mean(all_dice)), "iou": float(np.mean(all_iou)), "mae": float(np.mean(all_mae)),
    }


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = {
        "img_size": args.img_size, "batch_size": args.batch_size, "num_workers": args.num_workers,
        "pvt_backbone": "pvt_v2_b2", "d_model": 64, "d_state": 8, "mamba_chunk": 32,
        "dec_dim": 64, "window": 8, "num_heads": 4,
    }

    model = build_model(cfg, device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')} (val Dice={ckpt.get('best_dice', float('nan')):.4f})")

    kvasir_pairs = build_pairs_kvasir(args.kvasir)
    clinicdb_pairs = build_pairs_clinicdb(args.clinicdb)
    colondb_pairs = build_pairs_flat(args.colondb)
    etis_pairs = build_pairs_flat(args.etis)

    _, kvasir_test = stratified_split(kvasir_pairs, args.train_ratio, SEED)
    _, clinicdb_test = stratified_split(clinicdb_pairs, args.train_ratio, SEED)

    test_sets = [
        ("Kvasir-SEG (in-dist.)", kvasir_test),
        ("CVC-ClinicDB (in-dist.)", clinicdb_test),
        ("CVC-ColonDB (zero-shot)", colondb_pairs),
        ("ETIS (zero-shot)", etis_pairs),
    ]

    results = [evaluate_test_set(model, pairs, cfg, device, name, args.threshold) for name, pairs in test_sets]

    print("\n" + "=" * 72)
    print(f'  {"Test set":26s} {"n":>5s}   {"mDice":>7s}  {"mIoU":>7s}  {"MAE":>7s}')
    print("  " + "-" * 68)
    for r in results:
        print(f'  {r["name"]:26s} {r["n"]:5d}   {r["dice"]:7.4f}  {r["iou"]:7.4f}  {r["mae"]:7.4f}')
    print("=" * 72)

    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.out}")


if __name__ == "__main__":
    main()
