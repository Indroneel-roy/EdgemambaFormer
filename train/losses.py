"""
Objective function (Section 3.5): weighted BCE + weighted IoU on both the
main prediction head and the WaveletEAG auxiliary edge-gate head.

    L(y, y_hat) = L_wBCE(y, y_hat) + L_wIoU(y, y_hat)
    L_total     = lambda_pred * L(y_pred, y) + lambda_edge * L(y_edge, y)

with lambda_pred = 1.0, lambda_edge = 0.3. Only the main head is used at
inference; the edge term is an auxiliary regulariser.
"""

import torch


def weighted_bce(pred, target, eps=1e-6):
    """Weighted binary cross-entropy; minority-class pixels get higher weight."""
    pred = torch.clamp(torch.sigmoid(pred), eps, 1 - eps)
    pos_w = (target == 0).float().sum() / (target == 1).float().sum().clamp(min=1)
    loss = -(pos_w * target * torch.log(pred) + (1 - target) * torch.log(1 - pred))
    return loss.mean()


def weighted_iou(pred, target, eps=1e-6):
    """Standard soft, differentiable IoU loss."""
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum(dim=(2, 3))
    union = (pred + target - pred * target).sum(dim=(2, 3))
    iou = (inter + eps) / (union + eps)
    return (1 - iou).mean()


def edgemamba_loss(outputs, target, lambdas=(1.0, 0.3)):
    """
    Two-term supervision over {'pred', 'edge'} outputs, each scored as
    weighted BCE + weighted IoU against the same ground-truth mask.
    """
    lam_pred, lam_edge = lambdas
    keys = ["pred", "edge"]
    lam = [lam_pred, lam_edge]
    total = 0.0
    details = {}
    for k, lam_k in zip(keys, lam):
        pred = outputs[k]
        bce = weighted_bce(pred, target)
        iou = weighted_iou(pred, target)
        details[k] = (bce + iou).item()
        total += lam_k * (bce + iou)
    return total, details
