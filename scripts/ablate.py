#!/usr/bin/env python
"""
Ablation study (Section 4.7, Table 3): trains four configurations on
Kvasir-SEG only, each removing one of the three proposed components (or
none, for the full model), under an identical protocol so the comparison
is fair.

Usage:
    python scripts/ablate.py --kvasir /path/to/Kvasir-SEG --epochs 70
"""

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import build_pairs_kvasir, stratified_split, make_loader, get_train_transforms, get_val_transforms
from model.ablation_variants import build_ablation_model
from train import run_training

SEED = 42

# (run_name, use_wavelet_eag, use_csmm, use_dual_branch_decoder)
CONFIGS = [
    ("wo_WaveletEAG", False, True, True),
    ("wo_CSMM", True, False, True),
    ("wo_DualBranchDecoder", True, True, False),
    ("Full_Model", True, True, True),
]


def parse_args():
    p = argparse.ArgumentParser(description="Run the EdgeMambaFormer ablation study on Kvasir-SEG")
    p.add_argument("--kvasir", required=True, help="Path to Kvasir-SEG root")
    p.add_argument("--img-size", type=int, default=352)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=70)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint", default="./checkpoints_ablation")
    p.add_argument("--out", default="ablation_results.json")
    return p.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = {
        "img_size": args.img_size, "batch_size": args.batch_size, "epochs": args.epochs,
        "lr": args.lr, "weight_decay": args.weight_decay, "num_workers": args.num_workers,
        "pvt_backbone": "pvt_v2_b2", "d_model": 64, "d_state": 8, "mamba_chunk": 32,
        "dec_dim": 64, "window": 8, "num_heads": 4,
        "lambda_pred": 1.0, "lambda_edge": 0.3,
    }
    os.makedirs(args.checkpoint, exist_ok=True)

    kvasir_pairs = build_pairs_kvasir(args.kvasir)
    print(f"Kvasir-SEG: {len(kvasir_pairs)} pairs")
    kvasir_train, kvasir_test = stratified_split(kvasir_pairs, args.train_ratio, SEED)
    train_loader = make_loader(kvasir_train, get_train_transforms(cfg["img_size"]), cfg, shuffle=True)
    val_loader = make_loader(kvasir_test, get_val_transforms(cfg["img_size"]), cfg, shuffle=False)

    summary = []
    for name, use_eag, use_csmm, use_decoder in CONFIGS:
        torch.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)

        model = build_ablation_model(
            cfg, device, use_wavelet_eag=use_eag, use_csmm=use_csmm, use_dual_branch_decoder=use_decoder,
        )
        result = run_training(
            name, model, cfg, train_loader, val_loader, device,
            epochs=cfg["epochs"], ckpt_dir=args.checkpoint,
        )
        summary.append({
            "config": name, "mDice": result["best_dice"],
            "params_M": result["params_M"], "ckpt_path": result["ckpt_path"],
        })

        del model
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("  ABLATION SUMMARY (Table 3)")
    print("=" * 60)
    for r in summary:
        print(f'  {r["config"]:24s}  mDice={r["mDice"]:.4f}  params={r["params_M"]:.2f}M')
    print("=" * 60)

    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to {args.out}")


if __name__ == "__main__":
    main()
