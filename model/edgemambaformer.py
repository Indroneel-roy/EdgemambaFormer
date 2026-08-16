"""
EdgeMambaFormer -- full model assembly (Section 3).

    PVTv2-B2 encoder
      -> Wavelet Edge Attention Gate(f1, f4)        -> fe, edge_map
      -> Cross-Scale Mamba Module(f2, f3, f4)       -> f2', f3', f4'
      -> Dual-Branch Transformer Decoder            -> logit
    Main segmentation head + auxiliary edge-map supervision.
"""

import timm
import torch.nn as nn
import torch.nn.functional as F

from .wavelet_eag import EdgeAttentionGate
from .csmm import CrossScaleMambaModule
from .decoder import DualBranchDecoder


class EdgeMambaFormer(nn.Module):
    def __init__(
        self,
        backbone: str = "pvt_v2_b2",
        pretrained: bool = True,
        d_model: int = 64,
        d_state: int = 8,
        mamba_chunk: int = 32,
        dec_dim: int = 64,
        window: int = 8,
        num_heads: int = 4,
    ):
        super().__init__()

        # -- Encoder --
        self.encoder = timm.create_model(
            backbone, pretrained=pretrained, features_only=True, out_indices=(0, 1, 2, 3),
        )
        c1, c2, c3, c4 = self.encoder.feature_info.channels()  # e.g. [64,128,320,512]
        self._c = (c1, c2, c3, c4)

        # -- Wavelet Edge Attention Gate --
        self.eag = EdgeAttentionGate(c1, c4)

        # -- Cross-Scale Mamba Module --
        self.csmm = CrossScaleMambaModule(
            c2, c3, c4, d_model=d_model, d_state=d_state, chunk_size=mamba_chunk
        )

        # -- Dual-Branch Transformer Decoder --
        self.decoder = DualBranchDecoder(
            c1, d_model, dec_dim=dec_dim, window=window, num_heads=num_heads
        )

    def forward(self, x):
        img_h, img_w = x.shape[-2:]

        f1, f2, f3, f4 = self.encoder(x)

        fe, edge_map = self.eag(f1, f4)
        f2p, f3p, f4p = self.csmm(f2, f3, f4)
        logit = self.decoder(f1, fe, f2p, f3p, f4p)  # (B,1,H/4,W/4)

        size = (img_h, img_w)
        pred = F.interpolate(logit, size=size, mode="bilinear", align_corners=False)
        edge = F.interpolate(edge_map, size=size, mode="bilinear", align_corners=False)

        # 'pred' is the final segmentation logit; 'edge' is the auxiliary
        # boundary-gate logit, supervised against the same mask as a regulariser.
        # Only 'pred' is used at inference.
        return {"pred": pred, "edge": edge}


def build_model(cfg: dict, device) -> EdgeMambaFormer:
    """Factory matching the CFG dict used throughout training/eval scripts."""
    model = EdgeMambaFormer(
        backbone=cfg["pvt_backbone"],
        pretrained=True,
        d_model=cfg["d_model"],
        d_state=cfg["d_state"],
        mamba_chunk=cfg["mamba_chunk"],
        dec_dim=cfg["dec_dim"],
        window=cfg["window"],
        num_heads=cfg["num_heads"],
    ).to(device)
    return model
