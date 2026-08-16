"""
Dual-Branch Transformer Decoder -- Section 3.4.

Local branch: fine detail (f1 + WaveletEAG output fe) via local windowed
self-attention. Global branch: pooled multi-scale context (f2', f3', f4'
from CSMM) via full self-attention. The two branches are merged with a
single cross-attention step before the segmentation head.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalWindowAttention(nn.Module):
    """Self-attention restricted to non-overlapping windows (fine boundaries)."""

    def __init__(self, dim, window=8, num_heads=4):
        super().__init__()
        self.window = window
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, x):  # x: (B,C,H,W)
        B, C, H, W = x.shape
        w = self.window
        pad_h, pad_w = (-H) % w, (-W) % w
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        Hp, Wp = x.shape[-2:]

        xt = x.permute(0, 2, 3, 1)
        xt = xt.view(B, Hp // w, w, Wp // w, w, C).permute(0, 1, 3, 2, 4, 5)
        xt = xt.reshape(B * (Hp // w) * (Wp // w), w * w, C)

        xn = self.norm(xt)
        attn_out, _ = self.attn(xn, xn, xn)
        xt = xt + attn_out
        xt = xt + self.mlp(self.norm2(xt))

        xt = xt.view(B, Hp // w, Wp // w, w, w, C).permute(0, 1, 3, 2, 4, 5)
        xt = xt.reshape(B, Hp, Wp, C).permute(0, 3, 1, 2)
        return xt[:, :, :H, :W]


class GlobalSelfAttention(nn.Module):
    """Standard full self-attention over a (small) token sequence -- semantics."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))

    def forward(self, tokens):  # (B,N,C)
        xn = self.norm(tokens)
        attn_out, _ = self.attn(xn, xn, xn)
        tokens = tokens + attn_out
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class CrossBranchFusion(nn.Module):
    """Local tokens (Q) attend to global tokens (K,V); concat + conv fuse."""

    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.fuse = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 3, padding=1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
        )

    def forward(self, local_feat, global_tokens):  # local_feat:(B,C,H,W) global_tokens:(B,N,C)
        B, C, H, W = local_feat.shape
        q = local_feat.flatten(2).transpose(1, 2)
        attn_out, _ = self.cross_attn(self.norm_q(q), self.norm_kv(global_tokens), global_tokens)
        fused_tokens = q + attn_out
        fused_map = fused_tokens.transpose(1, 2).reshape(B, C, H, W)
        out = self.fuse(torch.cat([local_feat, fused_map], dim=1))
        return out


class DualBranchDecoder(nn.Module):
    """
    Fine-detail input (f1 + fe)      -> local window attention
    Multi-scale context (f2'+f3'+f4') -> global self-attention
      -> cross-branch fusion -> 1x1 head -> logit (at f1 resolution)
    """

    def __init__(self, c_low, d_model, dec_dim=64, window=8, num_heads=4):
        super().__init__()
        self.in_proj = nn.Conv2d(c_low * 2, dec_dim, 1)  # f1 concat fe -> dec_dim
        self.local_attn = LocalWindowAttention(dec_dim, window=window, num_heads=num_heads)

        self.ctx_proj = nn.Conv2d(d_model, dec_dim, 1)
        self.pos_emb = nn.Parameter(torch.zeros(1, 4096, dec_dim))  # generous max length
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.global_attn = GlobalSelfAttention(dec_dim, num_heads=num_heads)

        self.cross_fuse = CrossBranchFusion(dec_dim, num_heads=num_heads)
        self.head = nn.Conv2d(dec_dim, 1, 1)

    def forward(self, f1, fe, f2p, f3p, f4p):
        # local / fine-detail branch
        fine = self.in_proj(torch.cat([f1, fe], dim=1))
        local_feat = self.local_attn(fine)

        # global / multi-scale context branch -- aggregate at f3' resolution
        Ht, Wt = f3p.shape[-2:]
        ctx = (
            F.adaptive_avg_pool2d(f2p, (Ht, Wt))
            + f3p
            + F.interpolate(f4p, size=(Ht, Wt), mode="bilinear", align_corners=False)
        )
        ctx = self.ctx_proj(ctx)
        B = ctx.shape[0]
        tokens = ctx.flatten(2).transpose(1, 2)
        N = tokens.shape[1]
        tokens = tokens + self.pos_emb[:, :N]
        global_tokens = self.global_attn(tokens)

        # cross-branch fusion + head
        fused = self.cross_fuse(local_feat, global_tokens)
        logit = self.head(fused)  # (B,1,H/4,W/4)
        return logit
