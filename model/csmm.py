"""
Cross-Scale Mamba Module (CSMM) -- Section 3.3.

f2, f3, f4 (the three coarser PVTv2-B2 stages) are projected to a shared
channel dimension, flattened to tokens, concatenated into one sequence, and
mixed with a bidirectional selective-scan (S6) block -- giving every scale
linear-cost access to context accumulated at every other scale, before the
tokens are split back into per-scale feature maps.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(x, dt, A, B, C, D, chunk_size=64):
    """
    Chunked, numerically-stable parallel selective scan (S6), pure PyTorch.

    Mathematically equivalent to the sequential recurrence
        h_t = exp(dt_t * A) * h_{t-1} + dt_t * B_t * x_t
        y_t = C_t . h_t + D * x_t
    but computed in O(L/chunk * chunk^2) vectorised work instead of a Python
    loop over every timestep, which is what makes it usable on full-resolution
    feature-map token sequences (thousands of tokens).

    Args:
        x:  (Bz, L, d_inner)
        dt: (Bz, L, d_inner)   already softplus'd, > 0
        A:  (d_inner, d_state) already negative
        B:  (Bz, L, d_state)
        C:  (Bz, L, d_state)
        D:  (d_inner,)

    Returns:
        y: (Bz, L, d_inner)
    """
    Bz, L, d_inner = x.shape
    d_state = A.shape[1]
    device, dtype = x.device, x.dtype

    pad = (-L) % chunk_size
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, pad))
    Lp = x.shape[1]
    n_chunks = Lp // chunk_size
    T = chunk_size

    xc = x.view(Bz, n_chunks, T, d_inner)
    dtc = dt.view(Bz, n_chunks, T, d_inner)
    Bc = B.view(Bz, n_chunks, T, d_state)
    Cc = C.view(Bz, n_chunks, T, d_state)

    la = dtc.unsqueeze(-1) * A.view(1, 1, 1, d_inner, d_state)  # (Bz,nc,T,d_inner,d_state)
    Lcum = la.cumsum(dim=2)  # inclusive cumsum within chunk

    dBx = dtc.unsqueeze(-1) * Bc.unsqueeze(3) * xc.unsqueeze(-1)  # (Bz,nc,T,d_inner,d_state)

    causal = torch.tril(torch.ones(T, T, device=device, dtype=torch.bool))

    def _chunk_step(Lcum_c, dBx_c, C_c, h_carry):
        diff = Lcum_c.unsqueeze(2) - Lcum_c.unsqueeze(1)  # (Bz,T(t),T(s),d_inner,d_state)
        diff = diff.masked_fill(~causal.view(1, T, T, 1, 1), float("-inf"))
        w = torch.exp(diff)

        h_partial = torch.einsum("btsdn,bsdn->btdn", w, dBx_c)
        decay_from_start = torch.exp(Lcum_c)
        h_t = h_partial + decay_from_start * h_carry.unsqueeze(1)

        y_c = torch.einsum("btdn,btn->btd", h_t, C_c)
        return y_c, h_t[:, -1]

    y_chunks = []
    h_carry = torch.zeros(Bz, d_inner, d_state, device=device, dtype=dtype)
    use_ckpt = x.requires_grad or dt.requires_grad or B.requires_grad or C.requires_grad
    for c in range(n_chunks):
        if use_ckpt:
            # Recompute each chunk's activations during backward instead of storing
            # them all -- memory would otherwise grow with sequence length.
            y_c, h_carry = torch.utils.checkpoint.checkpoint(
                _chunk_step, Lcum[:, c], dBx[:, c], Cc[:, c], h_carry, use_reentrant=False
            )
        else:
            y_c, h_carry = _chunk_step(Lcum[:, c], dBx[:, c], Cc[:, c], h_carry)
        y_chunks.append(y_c)

    y = torch.cat(y_chunks, dim=1)[:, :L]
    y = y + x[:, :L] * D.view(1, 1, d_inner)
    return y


class S6Block(nn.Module):
    """One-directional selective-SSM block (simplified Mamba, pure PyTorch)."""

    def __init__(self, d_model, d_state=16, expand=1, dt_rank=None, conv_kernel=3, chunk_size=32):
        super().__init__()
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or max(1, d_model // 16)
        self.chunk_size = chunk_size

        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=conv_kernel,
            padding=conv_kernel - 1, groups=self.d_inner,
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model)

    def forward(self, x):  # x: (B, L, d_model)
        Bz, L, _ = x.shape
        x_z = self.in_proj(x)
        x_in, z = x_z.chunk(2, dim=-1)

        x_in = x_in.transpose(1, 2)
        x_in = self.conv1d(x_in)[..., :L]
        x_in = F.silu(x_in).transpose(1, 2)

        d_state = self.A_log.shape[1]
        x_dbl = self.x_proj(x_in)
        dt, Bp, Cp = torch.split(x_dbl, [self.dt_rank, d_state, d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)

        y = selective_scan(x_in, dt, A, Bp, Cp, self.D, chunk_size=self.chunk_size)
        y = y * F.silu(z)
        return self.out_proj(y)


class BiMambaBlock(nn.Module):
    """Bidirectional selective-scan SSM: forward + backward scans, summed as a residual."""

    def __init__(self, d_model, d_state=16, expand=1, chunk_size=32):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.fwd = S6Block(d_model, d_state, expand, chunk_size=chunk_size)
        self.bwd = S6Block(d_model, d_state, expand, chunk_size=chunk_size)

    def forward(self, x):  # (B, L, d_model)
        xn = self.norm(x)
        y_f = self.fwd(xn)
        y_b = self.bwd(xn.flip(1)).flip(1)
        return x + y_f + y_b


class CrossScaleMambaModule(nn.Module):
    """
    f2 (1/8, c2), f3 (1/16, c3), f4 (1/32, c4)
      -> project to shared d_model, flatten, concat tokens
      -> bidirectional Mamba
      -> reshape & split -> f2', f3', f4' (each now carrying cross-scale context)
    """

    def __init__(self, c2, c3, c4, d_model=128, d_state=16, chunk_size=32):
        super().__init__()
        self.proj2 = nn.Conv2d(c2, d_model, 1)
        self.proj3 = nn.Conv2d(c3, d_model, 1)
        self.proj4 = nn.Conv2d(c4, d_model, 1)
        self.mamba = BiMambaBlock(d_model, d_state=d_state, expand=1, chunk_size=chunk_size)

    def forward(self, f2, f3, f4):
        B = f2.shape[0]
        p2, p3, p4 = self.proj2(f2), self.proj3(f3), self.proj4(f4)
        h2, w2 = p2.shape[-2:]
        h3, w3 = p3.shape[-2:]
        h4, w4 = p4.shape[-2:]

        t2 = p2.flatten(2).transpose(1, 2)
        t3 = p3.flatten(2).transpose(1, 2)
        t4 = p4.flatten(2).transpose(1, 2)
        tokens = torch.cat([t2, t3, t4], dim=1)

        out = self.mamba(tokens)

        n2, n3 = h2 * w2, h3 * w3
        o2, o3, o4 = out.split([n2, n3, h4 * w4], dim=1)
        f2p = o2.transpose(1, 2).reshape(B, -1, h2, w2)
        f3p = o3.transpose(1, 2).reshape(B, -1, h3, w3)
        f4p = o4.transpose(1, 2).reshape(B, -1, h4, w4)
        return f2p, f3p, f4p
