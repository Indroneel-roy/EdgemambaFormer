"""
Wavelet Edge Attention Gate (WaveletEAG) -- Section 3.2.

Replaces a learned edge detector with a fixed, non-learned single-level 2-D
Haar Discrete Wavelet Transform (DWT) of the encoder's finest-resolution
stage f1. The three high-frequency subbands (LH, HL, HH) are fused and
gated against upsampled deep semantic context (f4) via a learned sigmoid
gate.

    LL = lo x lo   (smooth / approximation -- discarded)
    LH = lo x hi   (horizontal edges)
    HL = hi x lo   (vertical edges)
    HH = hi x hi   (diagonal edges)

with 1-D Haar low-pass lo = [1, 1]/sqrt(2) and high-pass hi = [1, -1]/sqrt(2).
The filters are fixed constants (not learned weights), so the decomposition
itself contributes zero trainable parameters -- only the subsequent fusion
and gating convolutions are learned (54,337 params total, < 0.3% of the
full 25.43M-parameter model).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _haar_filters(device, dtype):
    """Return the four 2-D Haar analysis filters as (1,1,2,2) tensors."""
    s = 1.0 / (2.0 ** 0.5)
    lo = torch.tensor([s, s], device=device, dtype=dtype)
    hi = torch.tensor([s, -s], device=device, dtype=dtype)
    LL = torch.outer(lo, lo).view(1, 1, 2, 2)
    LH = torch.outer(lo, hi).view(1, 1, 2, 2)
    HL = torch.outer(hi, lo).view(1, 1, 2, 2)
    HH = torch.outer(hi, hi).view(1, 1, 2, 2)
    return LL, LH, HL, HH


def dwt2d(x):
    """
    Single-level 2-D Haar DWT, pure PyTorch, channel-wise (depthwise, stride 2).

    Args:
        x: (B, C, H, W) -- H, W are reflect-padded to even if needed.

    Returns:
        LL, LH, HL, HH, each (B, C, H//2, W//2).
    """
    B, C, H, W = x.shape
    if H % 2 != 0 or W % 2 != 0:
        x = F.pad(x, (0, W % 2, 0, H % 2), mode="reflect")

    LL_f, LH_f, HL_f, HH_f = _haar_filters(x.device, x.dtype)
    LL_f = LL_f.expand(C, -1, -1, -1)
    LH_f = LH_f.expand(C, -1, -1, -1)
    HL_f = HL_f.expand(C, -1, -1, -1)
    HH_f = HH_f.expand(C, -1, -1, -1)

    LL = F.conv2d(x, LL_f, stride=2, groups=C)
    LH = F.conv2d(x, LH_f, stride=2, groups=C)
    HL = F.conv2d(x, HL_f, stride=2, groups=C)
    HH = F.conv2d(x, HH_f, stride=2, groups=C)
    return LL, LH, HL, HH


class EdgeAttentionGate(nn.Module):
    """
    WaveletEAG: gates fine-detail wavelet edge evidence against deep semantic
    context, per pixel.

    Inputs
    ------
    f1: (B, c_low,  H, W)   PVTv2-B2 stage-1 (1/4 scale)
    f4: (B, c_high, h, w)   PVTv2-B2 stage-4 (1/32 scale)

    Returns
    -------
    fe:    (B, c_low, H, W)  gated edge-guided feature (decoder local branch input)
    sigma: (B, 1,     H, W)  boundary-probability map (auxiliary supervision)
    """

    def __init__(self, c_low: int, c_high: int):
        super().__init__()
        self.proj_high = nn.Conv2d(c_high, c_low, 1)

        self.edge_proj = nn.Sequential(
            nn.Conv2d(c_low * 3, c_low, 1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
        )
        self.edge_refine = nn.Sequential(
            nn.Conv2d(c_low, c_low, 3, padding=1, groups=c_low, bias=False),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(c_low * 2, c_low, 1, bias=False),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_low, 1, 1),
        )

    def forward(self, f1, f4):
        B, C, H, W = f1.shape

        # 1. Haar DWT on f1
        _LL, LH, HL, HH = dwt2d(f1)  # each (B, C, H/2, W/2)

        # 2. Edge subbands -> edge_feat at f1 resolution
        edge_subbands = torch.cat([LH, HL, HH], dim=1)
        edge_feat = self.edge_proj(edge_subbands)
        edge_feat = F.interpolate(edge_feat, size=(H, W), mode="bilinear", align_corners=False)
        edge_feat = self.edge_refine(edge_feat)

        # 3. Semantic guidance from f4
        f4_up = F.interpolate(self.proj_high(f4), size=(H, W), mode="bilinear", align_corners=False)

        # 4. Sigmoid gate
        sigma = torch.sigmoid(self.gate(torch.cat([edge_feat, f4_up], dim=1)))

        # 5. Gated fusion: sigma -> 1 trusts wavelet edges, sigma -> 0 trusts semantics
        fe = edge_feat * sigma + f4_up * (1.0 - sigma)
        return fe, sigma
