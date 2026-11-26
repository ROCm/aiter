# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Kimi Delta Attention (KDA) fused layer: training forward + backward interface.

One KDA layer, token-major (packed varlen is not supported yet):

    q, k, v = short_conv4(x @ W{q,k,v}) -> silu       # depthwise causal conv, width W
    q, k    = l2norm(q), l2norm(k)                    # per head
    g       = -exp(A_log) * softplus(f_b(f_a(x)) + dt_bias)   # F-decay low-rank gate [T,H,K]
    beta    = sigmoid(x @ W_beta)                     # [T,H]
    S_t     = diag(exp(g_t)) S_{t-1}
    S_t    += k_t (beta_t (v_t - S_t^T k_t))^T        # gated delta rule
    o_t     = scale * S_t^T q_t
    o       = rmsnorm(o) * norm_w * sigmoid(g_b(g_a(x)))     # gated RMSNorm, per head
    y       = o @ W_o

``kda_layer_fwd`` / ``kda_layer_bwd`` are the training entry points. Backends:

* ``"torch"``: chunked (WY / triangular-solve) PyTorch implementation, gradients
  from autograd. Correct but slow; used as the fallback and as the reference
  the fused kernels are graded against.
* ``"fused"``: the fused AITER kernel path. Not implemented yet; it will compose
  the Triton KDA chunk backward, the causal-conv1d weight-gradient kernel and
  fused gated-RMSNorm / gate backward.
"""

from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F

KDA_BACKENDS = ("torch", "fused")


@dataclass
class KDALayerWeights:
    """Weights of one KDA layer. Shapes use D=hidden, H=heads, K=head_dim,
    V=value head_dim, W=conv width, R=gate low rank."""

    w_q: torch.Tensor  # [H*K, D]
    w_k: torch.Tensor  # [H*K, D]
    w_v: torch.Tensor  # [H*V, D]
    conv_q: torch.Tensor  # [H*K, W]
    conv_k: torch.Tensor  # [H*K, W]
    conv_v: torch.Tensor  # [H*V, W]
    f_a: torch.Tensor  # [R, D]      decay gate down-proj
    f_b: torch.Tensor  # [H*K, R]    decay gate up-proj
    A_log: torch.Tensor  # [H]
    dt_bias: torch.Tensor  # [H*K]
    w_beta: torch.Tensor  # [H, D]
    g_a: torch.Tensor  # [R, D]      output gate down-proj
    g_b: torch.Tensor  # [H*V, R]    output gate up-proj
    norm_w: torch.Tensor  # [V]
    w_o: torch.Tensor  # [D, H*V]

    def tensors(self):
        return [getattr(self, f.name) for f in fields(self)]

    @classmethod
    def from_tensors(cls, ts):
        return cls(*ts)


def _short_conv(x, w):
    # x: [B, T, C], w: [C, W]; causal depthwise conv, no bias, then silu.
    T = x.shape[1]
    y = F.conv1d(
        x.transpose(1, 2), w.unsqueeze(1), padding=w.shape[1] - 1, groups=w.shape[0]
    )
    return F.silu(y[..., :T]).transpose(1, 2)


def kda_layer_preprocess(x, w: KDALayerWeights, num_heads, eps=1e-6):
    """Everything before the recurrence. Returns q, k [B,T,H,K], v [B,T,H,V],
    g (log decay) [B,T,H,K], beta [B,T,H], fp32."""
    B, T, _ = x.shape
    H = num_heads
    xf = x.float()
    q = _short_conv(xf @ w.w_q.float().t(), w.conv_q.float()).view(B, T, H, -1)
    k = _short_conv(xf @ w.w_k.float().t(), w.conv_k.float()).view(B, T, H, -1)
    v = _short_conv(xf @ w.w_v.float().t(), w.conv_v.float()).view(B, T, H, -1)
    q = F.normalize(q, dim=-1, eps=eps)
    k = F.normalize(k, dim=-1, eps=eps)
    gl = (xf @ w.f_a.float().t()) @ w.f_b.float().t() + w.dt_bias.float()
    g = -w.A_log.float().exp().view(H, 1) * F.softplus(gl.view(B, T, H, -1))
    beta = torch.sigmoid(xf @ w.w_beta.float().t())
    return q, k, v, g, beta


def kda_layer_postprocess(o, x, w: KDALayerWeights, eps=1e-6):
    """Gated RMSNorm + output projection. o: [B,T,H,V] fp32."""
    B, T, H, V = o.shape
    xf = x.float()
    gate = torch.sigmoid((xf @ w.g_a.float().t()) @ w.g_b.float().t()).view(B, T, H, V)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * w.norm_w.float() * gate
    return (o.reshape(B, T, H * V) @ w.w_o.float().t()).to(x.dtype)


def chunk_kda_torch(q, k, v, g, beta, scale, chunk_size=64):
    """Chunked gated delta rule with per-channel decay (differentiable).
    q,k,g: [B,T,H,K]; v: [B,T,H,V]; beta: [B,T,H]. Returns o [B,T,H,V] fp32."""
    B, T, H, K = q.shape
    V = v.shape[-1]
    C = chunk_size
    pad = (-T) % C
    if pad:
        # zero k/v/beta and zero decay keep the state unchanged on padded steps
        q, k, v, g = (F.pad(t, (0, 0, 0, 0, 0, pad)) for t in (q, k, v, g))
        beta = F.pad(beta, (0, 0, 0, pad))
    N = (T + pad) // C

    def chunks(t):  # [B, T, H, ...] -> [B, H, N, C, ...]
        return t.reshape(B, N, C, H, *t.shape[3:]).transpose(1, 3).transpose(2, 3)

    q, k, v, g, beta = map(chunks, (q, k, v, g, beta))
    G = g.cumsum(-2)  # [B,H,N,C,K]
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    # pairwise decay exp(G_i - G_j), masked to j <= i before exp for stability
    dG = G.unsqueeze(-2) - G.unsqueeze(-3)  # [B,H,N,C(i),C(j),K]
    dG = dG.masked_fill(~incl[..., None], float("-inf"))
    decay = dG.exp()
    A_kk = torch.einsum("bhnik,bhnjk,bhnijk->bhnij", k, k, decay) * strict
    A_qk = torch.einsum("bhnik,bhnjk,bhnijk->bhnij", q, k, decay)
    eye = torch.eye(C, device=q.device, dtype=q.dtype)
    L = eye + beta.unsqueeze(-1) * A_kk  # unit lower triangular
    k_end = k * (G[..., -1:, :] - G).exp()  # decay to chunk end
    q_g = q * G.exp()
    k_g = k * G.exp()

    S = q.new_zeros(B, H, K, V)
    out = []
    for n in range(N):
        rhs = beta[:, :, n, :, None] * (v[:, :, n] - k_g[:, :, n] @ S)
        u = torch.linalg.solve_triangular(
            L[:, :, n], rhs, upper=False, unitriangular=True
        )
        out.append(q_g[:, :, n] @ S + A_qk[:, :, n] @ u)
        S = G[:, :, n, -1, :, None].exp() * S + k_end[:, :, n].transpose(-1, -2) @ u
    o = torch.stack(out, 2) * scale  # [B,H,N,C,V]
    o = o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, V)
    return o[:, :T]


def _kda_layer_torch(x, w, num_heads, chunk_size, eps):
    q, k, v, g, beta = kda_layer_preprocess(x, w, num_heads, eps)
    o = chunk_kda_torch(q, k, v, g, beta, q.shape[-1] ** -0.5, chunk_size)
    return kda_layer_postprocess(o, x, w, eps)


def kda_layer_fwd(
    x, w: KDALayerWeights, num_heads, chunk_size=64, eps=1e-6, backend="torch"
):
    """x: [B,T,D] -> y: [B,T,D]. Differentiable for backend="torch"."""
    if backend == "torch":
        return _kda_layer_torch(x, w, num_heads, chunk_size, eps)
    if backend == "fused":
        raise NotImplementedError(
            "fused KDA layer forward for training is not implemented yet; use backend='torch'"
        )
    raise ValueError(f"unknown backend {backend!r}, expected one of {KDA_BACKENDS}")


def kda_layer_bwd(
    dy, x, w: KDALayerWeights, num_heads, chunk_size=64, eps=1e-6, backend="torch"
):
    """Backward of one KDA layer.

    Returns (dx, dw) where dw is a KDALayerWeights holding the gradient of every
    weight (q/k/v proj, short conv, F-decay low-rank gate + A_log/dt_bias, beta
    proj, output gate, norm weight, output proj). The forward is recomputed.
    """
    if backend == "fused":
        raise NotImplementedError(
            "fused KDA layer backward is not implemented yet; it depends on the Triton "
            "KDA chunk backward (ROCm/aiter#6498) and the causal-conv1d weight-gradient "
            "kernel (ROCm/aiter#8726). Use backend='torch'."
        )
    if backend != "torch":
        raise ValueError(f"unknown backend {backend!r}, expected one of {KDA_BACKENDS}")
    with torch.enable_grad():
        xs = x.detach().requires_grad_(True)
        ws = [t.detach().requires_grad_(True) for t in w.tensors()]
        y = _kda_layer_torch(
            xs, KDALayerWeights.from_tensors(ws), num_heads, chunk_size, eps
        )
        grads = torch.autograd.grad(y, [xs, *ws], dy)
    return grads[0], KDALayerWeights.from_tensors(list(grads[1:]))
