"""
Evaluation metrics (Section 4.2): mean Dice Similarity Coefficient (mDice),
mean Intersection-over-Union (mIoU), and Mean Absolute Error (MAE) between
the predicted probability map and the ground-truth mask.
"""

import torch


class SegMetrics:
    """Accumulates Dice, IoU, and MAE over a dataset, per-image then averaged."""

    def __init__(self, threshold: float = 0.5):
        self.thr = threshold
        self.reset()

    def reset(self):
        self.dice_sum = 0.0
        self.iou_sum = 0.0
        self.mae_sum = 0.0
        self.n = 0

    @torch.no_grad()
    def update(self, pred_logit, target):
        """
        pred_logit: (B,1,H,W) raw logits
        target:     (B,1,H,W) binary float
        """
        pred = (torch.sigmoid(pred_logit) > self.thr).float()
        B = pred.shape[0]
        for b in range(B):
            p = pred[b].view(-1)
            t = target[b].view(-1)
            tp = (p * t).sum()
            fp = (p * (1 - t)).sum()
            fn = ((1 - p) * t).sum()
            dice = (2 * tp / (2 * tp + fp + fn + 1e-8)).item()
            iou = (tp / (tp + fp + fn + 1e-8)).item()
            mae = (p - t).abs().mean().item()
            self.dice_sum += dice
            self.iou_sum += iou
            self.mae_sum += mae
            self.n += 1

    def result(self):
        n = max(self.n, 1)
        return {
            "mDice": self.dice_sum / n,
            "mIoU": self.iou_sum / n,
            "MAE": self.mae_sum / n,
        }
