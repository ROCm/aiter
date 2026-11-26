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
  from autograd. Slow; it is the reference the fused path is graded against.
* ``"fused"``: hand-written backward. q/k L2 norm and decay/beta gates run as
  FlyDSL kernels (``aiter.ops.flydsl.kda_layer_bwd``); gated RMSNorm goes through
  ``register_gated_rmsnorm_bwd``; conv1d + SiLU reuses the Triton split-q/k/v
  conv backward; projections are GEMMs; the chunk delta-rule backward goes
  through ``register_chunk_bwd`` (which can also absorb the L2 norm and gate
  stages). Unregistered stages fall back to PyTorch.
"""

from collections.abc import Callable
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


ChunkBwdFn = Callable[..., tuple]
CHUNK_BWD_FUSIONS = frozenset({"qk_l2norm", "gate", "beta_sigmoid"})
_chunk_bwd_impl: ChunkBwdFn | None = None
_chunk_bwd_fuses: frozenset = frozenset()


def register_chunk_bwd(fn: ChunkBwdFn | None, fuses=()):
    """Install the chunk delta-rule backward used by ``backend="fused"``.

    Contract (fla ``chunk_kda`` layout and flag names, which the KDA chunk
    kernels share)::

        fn(do, q, k, v, g, beta, scale, chunk_size, *,
           use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
           A_log=None, dt_bias=None, use_beta_sigmoid_in_kernel=False)
          -> (dq, dk, dv, dg, dbeta, dA_log, ddt_bias)

    q, k, g ``[B, T, H, K]``, v and do ``[B, T, H, V]``, beta ``[B, T, H]``, no
    initial or final state; ``dA_log``/``ddt_bias`` are None unless the gate is
    fused. By default q, k are L2-normalized, g is the per-token log decay and
    beta is post-sigmoid. Each name in ``fuses`` (a subset of
    ``CHUNK_BWD_FUSIONS``) moves that prologue into ``fn``, which then gets the
    raw tensor and returns its gradient, and the matching flag is passed:

    * ``"qk_l2norm"``: q, k before the L2 norm.
    * ``"gate"``: g is ``f_b(f_a(x))`` without ``dt_bias``;
      ``g = -exp(A_log) * softplus(g + dt_bias)`` is applied inside.
    * ``"beta_sigmoid"``: beta is the logit.

    ``None`` restores the PyTorch fallback.
    """
    global _chunk_bwd_impl, _chunk_bwd_fuses
    fuses = frozenset(fuses)
    if not fuses <= CHUNK_BWD_FUSIONS:
        raise ValueError(f"unknown fusions {sorted(fuses - CHUNK_BWD_FUSIONS)}")
    _chunk_bwd_impl, _chunk_bwd_fuses = fn, fuses if fn is not None else frozenset()


GatedRMSNormBwdFn = Callable[..., tuple]
_gated_rmsnorm_bwd_impl: GatedRMSNormBwdFn | None = None


def register_gated_rmsnorm_bwd(fn: GatedRMSNormBwdFn | None):
    """Install the gated RMSNorm backward used by ``backend="fused"``.

    Contract: ``fn(o, gp, w, dy, eps) -> (do, dgp, dw)`` for
    ``out = o * rsqrt(mean(o^2, -1) + eps) * w * sigmoid(gp)`` with o, gp, dy
    ``[B, T, H, V]`` and w ``[V]``; outputs fp32. ``None`` restores the PyTorch
    fallback.
    """
    global _gated_rmsnorm_bwd_impl
    _gated_rmsnorm_bwd_impl = fn


def chunk_kda_bwd_torch(
    do,
    q,
    k,
    v,
    g,
    beta,
    scale,
    chunk_size=64,
    *,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    A_log=None,
    dt_bias=None,
    use_beta_sigmoid_in_kernel=False,
    eps=1e-6,
):
    """Reference chunk backward with the ``register_chunk_bwd`` contract."""
    with torch.enable_grad():
        leaves = [t.detach().float().requires_grad_(True) for t in (q, k, v, g, beta)]
        params = []
        if use_gate_in_kernel:
            params = [t.detach().float().requires_grad_(True) for t in (A_log, dt_bias)]
        q_, k_, v_, g_, b_ = leaves
        if use_qk_l2norm_in_kernel:
            q_, k_ = F.normalize(q_, dim=-1, eps=eps), F.normalize(k_, dim=-1, eps=eps)
        if use_gate_in_kernel:
            H = g_.shape[-2]
            z = g_ + params[1].view(H, -1)
            g_ = -params[0].exp().view(H, 1) * F.softplus(z)
        if use_beta_sigmoid_in_kernel:
            b_ = torch.sigmoid(b_)
        o = chunk_kda_torch(q_, k_, v_, g_, b_, scale, chunk_size)
        grads = torch.autograd.grad(o, leaves + params, do.float())
    return (*grads[:5], *(grads[5:] or (None, None)))


def conv_silu_bwd(a, w, dq, dk, dv):
    """Backward of ``silu(causal_depthwise_conv(a, w))`` split into q/k/v.

    a: [B, T, C] conv input; w: [C, W]; dq, dk: [B, T, HK]; dv: [B, T, HV].
    Returns (da [B, T, C], dw [C, W]), fp32. On GPU this is the Triton
    ``causal_conv1d_split_qkv_bwd_triton_fn`` (ROCm/aiter#8726) when installed.
    """
    B, T, C = a.shape
    hk, hv = dq.shape[-1], dv.shape[-1]
    try:
        from aiter.ops.triton.gated_delta_net import (
            causal_conv1d_split_qkv_bwd_triton_fn as fn,
        )
    except ImportError:
        fn = None
    if fn is None or not a.is_cuda:
        dc = torch.cat([dq, dk, dv], -1).float()
        with torch.enable_grad():
            a = a.detach().float().requires_grad_(True)
            w = w.detach().float().requires_grad_(True)
            return torch.autograd.grad(_short_conv(a, w), [a, w], dc)
    qsl = torch.arange(B + 1, device=a.device, dtype=torch.int32) * T
    da, dw, _, _ = fn(
        *(t.float().reshape(B * T, -1) for t in (dq, dk, dv)),
        a.float().reshape(B * T, C).t(),
        w.float().contiguous(),
        None,
        qsl,
        hk,
        hv,
        activation="silu",
    )
    return da.t().reshape(B, T, C), dw


def _flydsl_ops():
    from aiter.ops.flydsl import kda_layer_bwd

    return kda_layer_bwd


class _TorchOps:
    """Torch versions of the kernel stages: fallbacks and CPU stand-ins."""

    @staticmethod
    def l2norm_bwd(x, dy, eps=1e-6):
        with torch.enable_grad():
            x = x.detach().float().requires_grad_(True)
            return torch.autograd.grad(F.normalize(x, dim=-1, eps=eps), x, dy.float())[
                0
            ]

    @staticmethod
    def gate_beta_bwd(z, A_log, dg, braw, dbeta):
        H = z.shape[-2]
        neg_a = -A_log.float().exp().view(H, 1)
        dz = dg * neg_a * torch.sigmoid(z)
        dA_log = (dg * neg_a * F.softplus(z)).reshape(-1, *z.shape[-2:]).sum((0, 2))
        bt = torch.sigmoid(braw)
        return dz, dA_log, dbeta * bt * (1 - bt)

    @staticmethod
    def gated_rmsnorm_bwd(o, gp, w, dy, eps=1e-6):
        with torch.enable_grad():
            o, gp, w = (t.detach().float().requires_grad_(True) for t in (o, gp, w))
            out = (
                o
                * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
                * w
                * torch.sigmoid(gp)
            )
            return torch.autograd.grad(out, [o, gp, w], dy.float())


def _kda_layer_fused_fwd(x, w, H, chunk_size, eps):
    """Forward that keeps what the hand-written backward needs (fp32)."""
    B, T, _ = x.shape
    xf = x.float()
    w_qkv = torch.cat([w.w_q, w.w_k, w.w_v]).float()
    conv_qkv = torch.cat([w.conv_q, w.conv_k, w.conv_v]).float()
    hk, hv = w.w_q.shape[0], w.w_v.shape[0]
    a = xf @ w_qkv.t()  # [B, T, 2HK + HV]
    c = _short_conv(a, conv_qkv)
    q_pre, k_pre, v = c.split([hk, hk, hv], -1)
    q_pre, k_pre, v = (t.reshape(B, T, H, -1) for t in (q_pre, k_pre, v))
    q, k = F.normalize(q_pre, dim=-1, eps=eps), F.normalize(k_pre, dim=-1, eps=eps)
    lr = xf @ w.f_a.float().t()
    z = (lr @ w.f_b.float().t() + w.dt_bias.float()).view(B, T, H, -1)
    g = -w.A_log.float().exp().view(H, 1) * F.softplus(z)
    braw = xf @ w.w_beta.float().t()
    beta = torch.sigmoid(braw)
    scale = q.shape[-1] ** -0.5
    o = chunk_kda_torch(q, k, v, g, beta, scale, chunk_size)
    ga = xf @ w.g_a.float().t()
    gp = (ga @ w.g_b.float().t()).view_as(o)
    on = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * w.norm_w.float()
    on = (on * torch.sigmoid(gp)).reshape(B, T, -1)
    y = (on @ w.w_o.float().t()).to(x.dtype)
    saved = {
        "xf": xf,
        "w_qkv": w_qkv,
        "conv_qkv": conv_qkv,
        "a": a,
        "q_pre": q_pre,
        "k_pre": k_pre,
        "q": q,
        "k": k,
        "v": v,
        "lr": lr,
        "z": z,
        "g": g,
        "braw": braw,
        "beta": beta,
        "o": o,
        "ga": ga,
        "gp": gp,
        "on": on,
        "scale": scale,
    }
    return y, saved


def _kda_layer_fused_bwd(dy, x, w, H, chunk_size, eps, ops):
    _, s = _kda_layer_fused_fwd(x, w, H, chunk_size, eps)
    B, T, D = x.shape
    xf = s["xf"].reshape(B * T, D)
    dy = dy.float().reshape(B * T, D)

    d_wo = dy.t() @ s["on"].reshape(B * T, -1)
    d_on = (dy @ w.w_o.float()).view_as(s["o"])
    rms_bwd = _gated_rmsnorm_bwd_impl or _TorchOps.gated_rmsnorm_bwd
    do, dgp, d_norm = rms_bwd(s["o"], s["gp"], w.norm_w, d_on, eps)
    dgp = dgp.reshape(B * T, -1)
    ga = s["ga"].reshape(B * T, -1)
    d_gb = dgp.t() @ ga
    dga = dgp @ w.g_b.float()
    d_ga = dga.t() @ xf
    dx = dga @ w.g_a.float()

    chunk_bwd = _chunk_bwd_impl or chunk_kda_bwd_torch
    fuses = _chunk_bwd_fuses
    H_, K_ = s["z"].shape[-2:]
    kw = {}
    q_in, k_in, g_in, b_in = s["q"], s["k"], s["g"], s["beta"]
    if "qk_l2norm" in fuses:
        q_in, k_in = s["q_pre"], s["k_pre"]
        kw["use_qk_l2norm_in_kernel"] = True
    if "gate" in fuses:
        g_in = s["z"] - w.dt_bias.float().view(H_, K_)
        kw.update(use_gate_in_kernel=True, A_log=w.A_log, dt_bias=w.dt_bias)
    if "beta_sigmoid" in fuses:
        b_in = s["braw"]
        kw["use_beta_sigmoid_in_kernel"] = True
    dq, dk, dv, dg, dbeta, dA_log, ddt = chunk_bwd(
        do, q_in, k_in, s["v"], g_in, b_in, s["scale"], chunk_size, **kw
    )
    dz, db = dg.float(), dbeta.float()
    if not {"gate", "beta_sigmoid"} <= fuses:
        dz_k, dA_k, db_k = ops.gate_beta_bwd(s["z"], w.A_log, dz, s["braw"], db)
        if "gate" not in fuses:
            dz, dA_log = dz_k, dA_k
        if "beta_sigmoid" not in fuses:
            db = db_k
    dz = dz.reshape(B * T, -1)
    db = db.reshape(B * T, -1)
    if ddt is None:
        ddt = dz.sum(0)
    d_fb = dz.t() @ s["lr"].reshape(B * T, -1)
    dlr = dz @ w.f_b.float()
    d_fa = dlr.t() @ xf
    dx = dx + dlr @ w.f_a.float() + db @ w.w_beta.float()
    d_wbeta = db.t() @ xf

    if "qk_l2norm" not in fuses:
        dq = ops.l2norm_bwd(s["q_pre"], dq.float(), eps)
        dk = ops.l2norm_bwd(s["k_pre"], dk.float(), eps)
    da, d_conv = conv_silu_bwd(
        s["a"], s["conv_qkv"], *(t.reshape(B, T, -1) for t in (dq, dk, dv))
    )
    da = da.reshape(B * T, -1)
    d_wqkv = da.t() @ xf
    dx = dx + da @ s["w_qkv"]

    hk, hv = w.w_q.shape[0], w.w_v.shape[0]
    dwq, dwk, dwv = d_wqkv.split([hk, hk, hv])
    dcq, dck, dcv = d_conv.split([hk, hk, hv])
    grads = KDALayerWeights(
        w_q=dwq,
        w_k=dwk,
        w_v=dwv,
        conv_q=dcq,
        conv_k=dck,
        conv_v=dcv,
        f_a=d_fa,
        f_b=d_fb,
        A_log=dA_log,
        dt_bias=ddt,
        w_beta=d_wbeta,
        g_a=d_ga,
        g_b=d_gb,
        norm_w=d_norm,
        w_o=d_wo,
    )
    grads = KDALayerWeights.from_tensors(
        [gr.to(p.dtype) for gr, p in zip(grads.tensors(), w.tensors())]
    )
    return dx.view(B, T, D).to(x.dtype), grads


class _KDALayerFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_heads, chunk_size, eps, *ws):
        ctx.save_for_backward(x, *ws)
        ctx.cfg = (num_heads, chunk_size, eps)
        y, _ = _kda_layer_fused_fwd(
            x, KDALayerWeights.from_tensors(ws), num_heads, chunk_size, eps
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, *ws = ctx.saved_tensors
        dx, dw = _kda_layer_fused_bwd(
            dy, x, KDALayerWeights.from_tensors(ws), *ctx.cfg, _flydsl_ops()
        )
        return (dx, None, None, None, *dw.tensors())


def kda_layer_fwd(
    x, w: KDALayerWeights, num_heads, chunk_size=64, eps=1e-6, backend="torch"
):
    """x: [B,T,D] -> y: [B,T,D]. Differentiable for both backends."""
    if backend == "torch":
        return _kda_layer_torch(x, w, num_heads, chunk_size, eps)
    if backend == "fused":
        return _KDALayerFused.apply(x, num_heads, chunk_size, eps, *w.tensors())
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
        return _kda_layer_fused_bwd(dy, x, w, num_heads, chunk_size, eps, _flydsl_ops())
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
