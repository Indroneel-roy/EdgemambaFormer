"""
Train/eval loop (Section 3.6). `run_training` is a general-purpose runner
used both for the shared pooled-training run and for each of the four
ablation configurations -- one call per run, each with its own model
instance, optimizer, scheduler, history, and checkpoint.
"""

import os

import torch
from tqdm import tqdm

from .losses import edgemamba_loss
from .metrics import SegMetrics


def train_one_epoch(model, loader, optimizer, device, lambdas):
    model.train()
    metrics = SegMetrics()
    running_loss = 0.0
    pbar = tqdm(loader, desc="Train", leave=False)

    for imgs, masks in pbar:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss, _ = edgemamba_loss(outputs, masks, lambdas)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        metrics.update(outputs["pred"], masks)
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return running_loss / len(loader), metrics.result()


@torch.no_grad()
def evaluate(model, loader, device, lambdas):
    model.eval()
    metrics = SegMetrics()
    running_loss = 0.0

    for imgs, masks in tqdm(loader, desc="Eval ", leave=False):
        imgs, masks = imgs.to(device), masks.to(device)
        outputs = model(imgs)
        loss, _ = edgemamba_loss(outputs, masks, lambdas)
        running_loss += loss.item()
        metrics.update(outputs["pred"], masks)

    return running_loss / len(loader), metrics.result()


def run_training(name, model, cfg, train_loader, val_loader, device, epochs, ckpt_dir):
    """
    Trains `model` for `epochs` epochs, checkpointing on best validation
    Dice. Used both for the shared model (name='shared_edgemambaformer')
    and for each ablation configuration (name='wo_WaveletEAG', etc.).

    Returns a dict with the run's history, best Dice, checkpoint path, and
    parameter count.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    lambdas = (cfg["lambda_pred"], cfg["lambda_edge"])

    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "train_mae": [], "val_mae": [],
    }
    best_dice = 0.0
    ckpt_path = os.path.join(ckpt_dir, f"best_{name}.pth")

    print(f"\n{'=' * 70}\nRun: {name}\n{'=' * 70}")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_m = train_one_epoch(model, train_loader, optimizer, device, lambdas)
        vl_loss, vl_m = evaluate(model, val_loader, device, lambdas)
        scheduler.step()

        history["train_loss"].append(tr_loss); history["val_loss"].append(vl_loss)
        history["train_dice"].append(tr_m["mDice"]); history["val_dice"].append(vl_m["mDice"])
        history["train_iou"].append(tr_m["mIoU"]); history["val_iou"].append(vl_m["mIoU"])
        history["train_mae"].append(tr_m["MAE"]); history["val_mae"].append(vl_m["MAE"])

        if vl_m["mDice"] > best_dice:
            best_dice = vl_m["mDice"]
            torch.save({
                "epoch": epoch, "model_state": model.state_dict(),
                "opt_state": optimizer.state_dict(), "best_dice": best_dice,
            }, ckpt_path)
            tag = " <- best"
        else:
            tag = ""

        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{name}] Epoch {epoch:3d}/{epochs} | "
                f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
                f"Dice {tr_m['mDice']:.4f}/{vl_m['mDice']:.4f} | "
                f"IoU {tr_m['mIoU']:.4f}/{vl_m['mIoU']:.4f} | LR {lr_now:.2e}{tag}"
            )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"{'-' * 70}\n[{name}] Best val Dice: {best_dice:.4f}  |  Params: {n_params / 1e6:.2f} M")

    return {
        "name": name, "history": history, "best_dice": best_dice,
        "ckpt_path": ckpt_path, "params_M": n_params / 1e6,
    }
