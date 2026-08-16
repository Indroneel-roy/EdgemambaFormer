"""
Ablated component variants (Section 4.7) -- each class is a drop-in
replacement for the corresponding full-model module, with the mechanism
under test stripped out but the interface (inputs/outputs) kept identical,
so exactly one component differs between a run and the full model.

Also provides `EdgeMambaFormerAblation`, a configurable wrapper that swaps
any of the three components in/out via boolean flags, and `build_ablation_model`,
its factory.
"""

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from .wavelet_eag import EdgeAttentionGate
from .csmm import CrossScaleMambaModule
from .decoder import DualBranchDecoder


class NoWaveletEAG(nn.Module):
    """
    w/o WaveletEAG: drops the Haar DWT edge decomposition AND the sigmoid
    gate that blends fine edge detail with semantic context from f4. `fe`
    is just a plain depthwise-conv refinement of f1; the edge map comes
    from a single 1x1 conv (no wavelet prior, no cross-scale gating).
    """

    def __init__(self, c_low, c_high):
        super().__init__()
        self.refine = nn.Sequential(
            nn.Conv2d(c_low, c_low, 3, padding=1, groups=c_low, bias=False),
            nn.BatchNorm2d(c_low),
            nn.ReLU(inplace=True),
        )
        self.edge_head = nn.Conv2d(c_low, 1, 1)

    def forward(self, f1, f4):
        fe = self.refine(f1)  # no wavelet subbands, no f4 gating
        edge_map = self.edge_head(fe)
        return fe, edge_map


class NoCSMM(nn.Module):
    """
    w/o CSMM: drops the bidirectional selective-scan (Mamba) that mixes
    tokens across f2/f3/f4. Each scale is independently 1x1-conv projected
    to d_model channels -- no cross-scale sequence modelling, no long-range
    mixing.
    """

    def __init__(self, c2, c3, c4, d_model=128, d_state=16, chunk_size=32):
        super().__init__()
        self.proj2 = nn.Conv2d(c2, d_model, 1)
        self.proj3 = nn.Conv2d(c3, d_model, 1)
        self.proj4 = nn.Conv2d(c4, d_model, 1)

    def forward(self, f2, f3, f4):
        return self.proj2(f2), self.proj3(f3), self.proj4(f4)


class SimpleConvDecoder(nn.Module):
    """
    w/o Dual-Branch Decoder: drops both the local-window attention branch
    and the global self-attention + cross-branch attention fusion. Fine
    detail (f1+fe) and multi-scale context (f2'+f3'+f4') are simply
    concatenated and passed through stacked plain conv blocks -- no
    attention anywhere in the decoder.
    """

    def __init__(self, c_low, d_model, dec_dim=64, window=8, num_heads=4):
        super().__init__()
        self.in_proj = nn.Conv2d(c_low * 2, dec_dim, 1)
        self.ctx_proj = nn.Conv2d(d_model, dec_dim, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(dec_dim * 2, dec_dim, 3, padding=1), nn.BatchNorm2d(dec_dim), nn.ReLU(inplace=True),
            nn.Conv2d(dec_dim, dec_dim, 3, padding=1), nn.BatchNorm2d(dec_dim), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(dec_dim, 1, 1)

    def forward(self, f1, fe, f2p, f3p, f4p):
        fine = self.in_proj(torch.cat([f1, fe], dim=1))
        Ht, Wt = fine.shape[-2:]
        ctx = (
            F.adaptive_avg_pool2d(f2p, (Ht, Wt))
            + F.interpolate(f3p, size=(Ht, Wt), mode="bilinear", align_corners=False)
            + F.interpolate(f4p, size=(Ht, Wt), mode="bilinear", align_corners=False)
        )
        ctx = self.ctx_proj(ctx)
        fused = self.fuse(torch.cat([fine, ctx], dim=1))
        return self.head(fused)


class EdgeMambaFormerAblation(nn.Module):
    """
    Same PVTv2-B2 encoder + same forward graph as EdgeMambaFormer, but each
    of the three novel components can be independently swapped out for its
    ablated counterpart via boolean flags -- this keeps everything else
    (encoder weights init, channel dims, loss, training loop) identical
    across runs so the ablation isolates just that one component.
    """

    def __init__(
        self,
        backbone="pvt_v2_b2",
        pretrained=True,
        d_model=64,
        d_state=8,
        mamba_chunk=32,
        dec_dim=64,
        window=8,
        num_heads=4,
        use_wavelet_eag=True,
        use_csmm=True,
        use_dual_branch_decoder=True,
    ):
        super().__init__()

        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
        )
        c1, c2, c3, c4 = self.encoder.feature_info.channels()
        self._c = (c1, c2, c3, c4)

        self.eag = EdgeAttentionGate(c1, c4) if use_wavelet_eag else NoWaveletEAG(c1, c4)

        self.csmm = (
            CrossScaleMambaModule(c2, c3, c4, d_model=d_model, d_state=d_state, chunk_size=mamba_chunk)
            if use_csmm
            else NoCSMM(c2, c3, c4, d_model=d_model)
        )

        self.decoder = (
            DualBranchDecoder(c1, d_model, dec_dim=dec_dim, window=window, num_heads=num_heads)
            if use_dual_branch_decoder
            else SimpleConvDecoder(c1, d_model, dec_dim=dec_dim)
        )

    def forward(self, x):
        img_h, img_w = x.shape[-2:]
        f1, f2, f3, f4 = self.encoder(x)
        fe, edge_map = self.eag(f1, f4)
        f2p, f3p, f4p = self.csmm(f2, f3, f4)
        logit = self.decoder(f1, fe, f2p, f3p, f4p)

        size = (img_h, img_w)
        pred = F.interpolate(logit, size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(edge_map, size=size, mode="bilinear", align_corners=False)
        return {"pred": pred, "edge": edge}


def build_ablation_model(cfg, device, use_wavelet_eag=True, use_csmm=True, use_dual_branch_decoder=True):
    model = EdgeMambaFormerAblation(
        backbone=cfg["pvt_backbone"],
        pretrained=True,
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        mamba_chunk=cfg["mamba_chunk"],
        dec_dim=cfg["dec_dim"],
        window=cfg["window"],
        num_heads=cfg["num_heads"],
        use_wavelet_eag=use_wavelet_eag,
        use_csmm=use_csmm,
        use_dual_branch_decoder=use_dual_branch_decoder,
    ).to(device)
    return model
