#!/usr/bin/env python
"""
Train the shared EdgeMambaFormer model on pooled Kvasir-SEG + CVC-ClinicDB
(Section 3.6 / 4.1). Checkpoints on best pooled in-distribution val Dice.

Usage:
    python scripts/train.py \
        --kvasir /path/to/Kvasir-SEG \
        --clinicdb /path/to/CVC-ClinicDB \
        --colondb /path/to/CVC-ColonDB \
        --etis /path/to/ETIS-LaribPolypDB \
        --epochs 100 --checkpoint ./checkpoints
"""

import argparse
import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from data import (
    build_pairs_kvasir, build_pairs_clinicdb, build_pairs_flat,
    stratified_split, make_loader, get_train_transforms, get_val_transforms,
)
from model import build_model
from train import run_training

SEED = 42


def parse_args():
    p = argparse.ArgumentParser(description="Train shared EdgeMambaFormer model")
    p.add_argument("--kvasir", required=True, help="Path to Kvasir-SEG root")
    p.add_argument("--clinicdb", required=True, help="Path to CVC-ClinicDB root")
    p.add_argument("--colondb", required=True, help="Path to CVC-ColonDB root (zero-shot, unused in training)")
    p.add_argument("--etis", required=True, help="Path to ETIS-LaribPolypDB root (zero-shot, unused in training)")
    p.add_argument("--img-size", type=int, default=352)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--train-ratio", type=float, default=0.9)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint", default="./checkpoints")
    return p.parse_args()


def main():
    args = parse_args()

    random.seed(SEED); np.random.seed(SEED)
    torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    cfg = {
        "img_size": args.img_size, "batch_size": args.batch_size, "epochs": args.epochs,
        "lr": args.lr, "weight_decay": args.weight_decay, "num_workers": args.num_workers,
        "checkpoint": args.checkpoint, "pvt_backbone": "pvt_v2_b2",
        "d_model": 64, "d_state": 8, "mamba_chunk": 32,
        "dec_dim": 64, "window": 8, "num_heads": 4,
        "lambda_pred": 1.0, "lambda_edge": 0.3, "train_ratio": args.train_ratio,
    }
    os.makedirs(cfg["checkpoint"], exist_ok=True)

    # -- Build pairs & the shared-training split (Section 4.1) --
    kvasir_pairs = build_pairs_kvasir(args.kvasir)
    clinicdb_pairs = build_pairs_clinicdb(args.clinicdb)
    colondb_pairs = build_pairs_flat(args.colondb)   # held out, zero-shot only
    etis_pairs = build_pairs_flat(args.etis)          # held out, zero-shot only

    print(f"Kvasir-SEG   : {len(kvasir_pairs)} pairs")
    print(f"CVC-ClinicDB : {len(clinicdb_pairs)} pairs")
    print(f"CVC-ColonDB  : {len(colondb_pairs)} pairs (zero-shot, not used for training)")
    print(f"ETIS         : {len(etis_pairs)} pairs (zero-shot, not used for training)")

    kvasir_train, kvasir_test = stratified_split(kvasir_pairs, cfg["train_ratio"], SEED)
    clinicdb_train, clinicdb_test = stratified_split(clinicdb_pairs, cfg["train_ratio"], SEED)

    train_pairs = kvasir_train + clinicdb_train
    random.Random(SEED).shuffle(train_pairs)
    val_pairs = kvasir_test + clinicdb_test

    print(f"Pooled TRAIN: {len(train_pairs)}  |  Pooled VAL: {len(val_pairs)}")

    train_loader = make_loader(train_pairs, get_train_transforms(cfg["img_size"]), cfg, shuffle=True)
    val_loader = make_loader(val_pairs, get_val_transforms(cfg["img_size"]), cfg, shuffle=False)

    # -- Train --
    model = build_model(cfg, device)
    result = run_training(
        "shared_edgemambaformer", model, cfg, train_loader, val_loader, device,
        epochs=cfg["epochs"], ckpt_dir=cfg["checkpoint"],
    )
    print(f"\nDone. Best checkpoint: {result['ckpt_path']}  (val Dice={result['best_dice']:.4f})")


if __name__ == "__main__":
    main()
