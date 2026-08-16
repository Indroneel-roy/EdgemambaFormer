"""
PVTv2-B2 hierarchical encoder (Section 3.1).

Loaded via `timm` in `features_only` mode. Produces four feature maps
f1..f4 at strides {4, 8, 16, 32} with channel widths {64, 128, 320, 512}
(at a 352x352 input: 88x88, 44x44, 22x22, 11x11).
"""

import timm
import torch.nn as nn


class PVTv2Encoder(nn.Module):
    """Thin wrapper so the rest of the codebase doesn't depend on timm directly."""

    def __init__(self, backbone: str = "pvt_v2_b2", pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        self.channels = tuple(self.backbone.feature_info.channels())  # e.g. (64, 128, 320, 512)

    def forward(self, x):
        """Returns [f1, f2, f3, f4], finest to coarsest resolution."""
        return self.backbone(x)
