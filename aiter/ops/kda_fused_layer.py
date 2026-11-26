# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Kimi Delta Attention (KDA) fused layer: training forward + backward interface.

One KDA layer, token-major (packed varlen through ``cu_seqlens``):

    q, k, v = short_conv4(x @ W{q,k,v}) -> silu       # depthwise causal conv, width W
    q, k    = l2norm(q), l2norm(k)                    # per head
    z       = f_b(f_a(x)) + dt_bias                   # F-decay low-rank gate [T,H,K]
    g       = -exp(A_log[h]) * softplus(z)            # or, with lower_bound (safe gate):
            = lower_bound * sigmoid(exp(A_log[h]) * z)
    (A_log holds one decay rate per head; checkpoints pad it, e.g. 96 heads -> [128])
    beta    = sigmoid(x @ W_beta)                     # [T,H]
    S_t     = diag(exp(g_t)) S_{t-1}
    S_t    += k_t (beta_t (v_t - S_t^T k_t))^T        # gated delta rule
    o_t     = scale * S_t^T q_t
    o       = rmsnorm(o) * norm_w * sigmoid(x @ g_proj^T)    # gated RMSNorm, per head
    y       = o @ W_o

``kda_layer_fwd`` / ``kda_layer_bwd`` are the training entry points. Backends:

* ``"torch"``: chunked (WY / triangular-solve) PyTorch implementation, gradients
  from autograd. Slow; it is the reference the fused path is graded against.
* ``"fused"``: hand-written backward. The f_b and g_proj projections run as
  gfx950 FlyDSL GEMMs with fused epilogues (decay gate fwd/bwd, sigmoid output
  gate); the q/k L2 norm is a FlyDSL kernel (``aiter.ops.flydsl.kda_layer_bwd``);
  gated RMSNorm goes through ``register_gated_rmsnorm_bwd``; conv1d + SiLU
  reuses the Triton split-q/k/v conv backward; the chunk delta-rule backward
  goes through ``register_chunk_bwd`` (which can also absorb the L2 norm and
  gate stages). Unregistered or unsupported stages fall back to PyTorch.
"""

import inspect
import itertools
import os
from collections.abc import Callable
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

KDA_BACKENDS = ("torch", "fused")
# q/k L2-norm epsilon: fixed, because the chunk kernels that fuse the L2 norm
# (FLA / #6498 / #8740) hard-code it; ``eps`` arguments are the RMSNorm's.
L2NORM_EPS = 1e-6


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
    A_log: torch.Tensor  # [>= H] fp32, per head; entries past H are padding
    dt_bias: torch.Tensor  # [H*K]
    w_beta: torch.Tensor  # [H, D]
    g_proj: torch.Tensor  # [H*V, D]    output gate proj
    norm_w: torch.Tensor  # [V]
    w_o: torch.Tensor  # [D, H*V]

    def tensors(self):
        return [getattr(self, f.name) for f in fields(self)]

    @classmethod
    def from_tensors(cls, ts):
        return cls(*ts)


def kda_gate(z, A_log, lower_bound=None):
    """Decay gate (log space) from z = f_b(f_a(x)) + dt_bias, [..., H, K].

    ``A_log`` is per head and may be padded past H (the Kimi-K3 checkpoints
    store [128] for 96 heads, zeros after); only the first H entries are read,
    as in the FLA kernels.
    """
    H = z.shape[-2]
    a = A_log[:H].float().exp().view(H, 1)
    if lower_bound is None:
        return -a * F.softplus(z)
    return lower_bound * torch.sigmoid(a * z)


def _short_conv(x, w):
    # x: [B, T, C], w: [C, W]; causal depthwise conv, no bias, then silu.
    T = x.shape[1]
    y = F.conv1d(
        x.transpose(1, 2), w.unsqueeze(1), padding=w.shape[1] - 1, groups=w.shape[0]
    )
    return F.silu(y[..., :T]).transpose(1, 2)


def kda_layer_preprocess(x, w: KDALayerWeights, num_heads, eps=1e-6, lower_bound=None):
    """Everything before the recurrence. Returns q, k [B,T,H,K], v [B,T,H,V],
    g (log decay) [B,T,H,K], beta [B,T,H], fp32."""
    B, T, _ = x.shape
    H = num_heads
    xf = x.float()
    q = _short_conv(xf @ w.w_q.float().t(), w.conv_q.float()).view(B, T, H, -1)
    k = _short_conv(xf @ w.w_k.float().t(), w.conv_k.float()).view(B, T, H, -1)
    v = _short_conv(xf @ w.w_v.float().t(), w.conv_v.float()).view(B, T, H, -1)
    q = F.normalize(q, dim=-1, eps=L2NORM_EPS)
    k = F.normalize(k, dim=-1, eps=L2NORM_EPS)
    gl = (xf @ w.f_a.float().t()) @ w.f_b.float().t() + w.dt_bias.float()
    g = kda_gate(gl.view(B, T, H, -1), w.A_log, lower_bound)
    beta = torch.sigmoid(xf @ w.w_beta.float().t())
    return q, k, v, g, beta


def kda_layer_postprocess(o, x, w: KDALayerWeights, eps=1e-6):
    """Gated RMSNorm + output projection. o: [B,T,H,V] fp32."""
    B, T, H, V = o.shape
    xf = x.float()
    gate = torch.sigmoid(xf @ w.g_proj.float().t()).view(B, T, H, V)
    o = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * w.norm_w.float() * gate
    return (o.reshape(B, T, H * V) @ w.w_o.float().t()).to(x.dtype)


def _chunk_step(S, q, k, v, g, beta):
    """One chunk of the gated delta rule. S: [B,H,K,V]; q,k,g: [B,H,C,K];
    v: [B,H,C,V]; beta: [B,H,C]. Returns (o_unscaled [B,H,C,V], S_next)."""
    C = q.shape[-2]
    G = g.cumsum(-2)
    strict = torch.ones(C, C, dtype=torch.bool, device=q.device).tril(-1)
    incl = torch.ones(C, C, dtype=torch.bool, device=q.device).tril()
    # pairwise decay exp(G_i - G_j), masked to j <= i before exp for stability
    dG = (G.unsqueeze(-2) - G.unsqueeze(-3)).masked_fill(
        ~incl[..., None], float("-inf")
    )
    decay = dG.exp()  # [B,H,C(i),C(j),K]
    A_kk = torch.einsum("bhik,bhjk,bhijk->bhij", k, k, decay) * strict
    A_qk = torch.einsum("bhik,bhjk,bhijk->bhij", q, k, decay)
    L = torch.eye(C, device=q.device, dtype=q.dtype) + beta.unsqueeze(-1) * A_kk
    eg = G.exp()
    rhs = beta.unsqueeze(-1) * (v - (k * eg) @ S)
    u = torch.linalg.solve_triangular(L, rhs, upper=False, unitriangular=True)
    o = (q * eg) @ S + A_qk @ u
    k_end = k * (G[..., -1:, :] - G).exp()  # decay to chunk end
    S = G[..., -1, :, None].exp() * S + k_end.transpose(-1, -2) @ u
    return o, S


def chunk_kda_torch(q, k, v, g, beta, scale, chunk_size=64):
    """Chunked gated delta rule with per-channel decay (differentiable).
    q,k,g: [B,T,H,K]; v: [B,T,H,V]; beta: [B,T,H]. Returns o [B,T,H,V] fp32.

    Each chunk is a separate step, checkpointed under autograd, so memory grows
    with one state per chunk rather than with the [C, C, K] decay of every chunk.
    """
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
    ckpt = torch.is_grad_enabled() and any(t.requires_grad for t in (q, k, v, g, beta))
    S = q.new_zeros(B, H, K, V)
    out = []
    for n in range(N):
        args = (S, q[:, :, n], k[:, :, n], v[:, :, n], g[:, :, n], beta[:, :, n])
        if ckpt:
            o, S = torch.utils.checkpoint.checkpoint(
                _chunk_step, *args, use_reentrant=False
            )
        else:
            o, S = _chunk_step(*args)
        out.append(o)
    o = torch.stack(out, 2) * scale  # [B,H,N,C,V]
    o = o.permute(0, 2, 3, 1, 4).reshape(B, N * C, H, V)
    return o[:, :T]


def _kda_layer_torch(x, w, num_heads, chunk_size, eps, lower_bound=None):
    q, k, v, g, beta = kda_layer_preprocess(x, w, num_heads, eps, lower_bound)
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
           A_log=None, dt_bias=None, use_beta_sigmoid_in_kernel=False,
           [lower_bound])
          -> (dq, dk, dv, dg, dbeta, dA_log, ddt_bias)

    q, k, g ``[B, T, H, K]``, v and do ``[B, T, H, V]``, beta ``[B, T, H]``, no
    initial or final state; ``dA_log``/``ddt_bias`` are None unless the gate is
    fused. By default q, k are L2-normalized, g is the per-token log decay and
    beta is post-sigmoid. Each name in ``fuses`` (a subset of
    ``CHUNK_BWD_FUSIONS``) moves that prologue into ``fn``, which then gets the
    raw tensor and returns its gradient, and the matching flag is passed:

    * ``"qk_l2norm"``: q, k before the L2 norm.
    * ``"gate"``: g is ``f_b(f_a(x))`` without ``dt_bias``;
      ``g = -exp(A_log) * softplus(g + dt_bias)`` is applied inside, or the
      safe gate ``lower_bound * sigmoid(exp(A_log) * (g + dt_bias))`` when the
      layer has a ``lower_bound``; that keyword is passed only then, and a
      kernel without a ``lower_bound`` parameter does not get the gate fused.
      ``A_log`` is the first H entries (``dA_log`` is ``[H]``).
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
    ``[B, T, H, V]`` and w ``[V]``; outputs fp32. ``None`` restores the default:
    the FlyDSL ``aiter.ops.flydsl.gated_rmsnorm_bwd`` (ROCm/aiter#8741) on GPU
    when it is installed, else PyTorch.
    """
    global _gated_rmsnorm_bwd_impl
    _gated_rmsnorm_bwd_impl = fn


def _default_gated_rmsnorm_bwd(on_gpu):
    if on_gpu:
        try:
            from aiter.ops.flydsl.gated_rmsnorm_bwd import gated_rmsnorm_bwd

            return gated_rmsnorm_bwd
        except ImportError:
            pass
    return _TorchOps.gated_rmsnorm_bwd


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
    lower_bound=None,
    eps=L2NORM_EPS,
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
            g_ = kda_gate(g_ + params[1].view(H, -1), params[0], lower_bound)
        if use_beta_sigmoid_in_kernel:
            b_ = torch.sigmoid(b_)
        o = chunk_kda_torch(q_, k_, v_, g_, b_, scale, chunk_size)
        grads = torch.autograd.grad(o, leaves + params, do.float())
    return (*grads[:5], *(grads[5:] or (None, None)))


# Which implementation served each stage in the latest fused call.
LAST_PIECES: dict = {}

# Stages with both Triton and FlyDSL kernels default to Triton; FlyDSL is used
# only when selected (``set_kda_pieces`` or ``AITER_KDA_PIECES``, e.g.
# ``chunk_fwd=flydsl,conv_bwd=torch``). A missing or rejecting kernel falls
# back to the next choice: flydsl -> triton -> torch, triton -> torch.
KDA_PIECE_DEFAULTS = {
    "chunk_fwd": "triton",
    "chunk_bwd": "triton",
    "conv_bwd": "triton",
}
KDA_PIECE_CHOICES = ("triton", "flydsl", "torch")
_piece_overrides: dict = {}


# Fusion combinations, A/B-able in one run (``set_kda_fusions`` or
# ``AITER_KDA_FUSIONS``, e.g. ``save_fwd=1,concat_proj=1,chunk_fuse=qk_l2norm+gate``):
#   save_fwd       1: the training forward keeps its activations and the backward
#                  skips the recompute (memory for time); 0: recompute
#   concat_proj    1: one GEMM for x @ [w_q; w_k; w_v; f_a; w_beta (; g_proj)]^T
#   proj_epilogue  0: plain torch GEMMs instead of the FlyDSL GEMM epilogues
#   chunk_fuse     auto (the kernel's own set), none, or a "+"-joined subset of
#                  qk_l2norm / gate / beta_sigmoid handed to the chunk backward
KDA_FUSION_DEFAULTS = {
    "save_fwd": "0",
    "concat_proj": "0",
    "proj_epilogue": "1",
    "chunk_fuse": "auto",
}
_fusion_overrides: dict = {}


def set_kda_fusions(**choices):
    """Override fusion switches, e.g. ``set_kda_fusions(save_fwd=1)``; ``None`` drops one."""
    for name, value in choices.items():
        if name not in KDA_FUSION_DEFAULTS:
            raise ValueError(
                f"unknown KDA fusion {name!r}, expected {sorted(KDA_FUSION_DEFAULTS)}"
            )
        if value is None:
            _fusion_overrides.pop(name, None)
        else:
            _fusion_overrides[name] = str(
                int(value) if isinstance(value, bool) else value
            )


def kda_fusion_choices():
    """Current fusion switches: overrides, then ``AITER_KDA_FUSIONS``, then defaults."""
    out = dict(KDA_FUSION_DEFAULTS)
    for item in filter(None, os.environ.get("AITER_KDA_FUSIONS", "").split(",")):
        name, _, value = item.strip().partition("=")
        if name in out and value:
            out[name] = value
    out.update(_fusion_overrides)
    return out


def _fusion_on(name):
    return kda_fusion_choices()[name] not in ("0", "false", "off", "")


def set_kda_pieces(**choices):
    """Select the implementation per stage, e.g. ``set_kda_pieces(chunk_bwd="flydsl")``.

    ``None`` drops an override (back to the environment / default).
    """
    for name, choice in choices.items():
        if name not in KDA_PIECE_DEFAULTS:
            raise ValueError(
                f"unknown KDA piece {name!r}, expected {sorted(KDA_PIECE_DEFAULTS)}"
            )
        if choice is None:
            _piece_overrides.pop(name, None)
        elif choice not in KDA_PIECE_CHOICES:
            raise ValueError(f"{name}={choice!r}, expected one of {KDA_PIECE_CHOICES}")
        else:
            _piece_overrides[name] = choice


def kda_piece_choices():
    """Current selection: overrides, then ``AITER_KDA_PIECES``, then defaults."""
    out = dict(KDA_PIECE_DEFAULTS)
    for item in filter(None, os.environ.get("AITER_KDA_PIECES", "").split(",")):
        name, _, choice = item.strip().partition("=")
        if name in out and choice in KDA_PIECE_CHOICES:
            out[name] = choice
    out.update(_piece_overrides)
    return out


def _fallback_order(choice):
    return {"flydsl": ("flydsl", "triton", "torch"), "triton": ("triton", "torch")}.get(
        choice, ("torch",)
    )


def _token_major(*ts, dtype=None):
    """Contiguous ``[B, T, H, *]`` copies (cast to ``dtype`` in the same pass).

    q/k/v come out of the short conv as a transposed, channel-major view; the
    chunk kernels index them token-major, and a dtype cast alone keeps the
    strides of a dense view.
    """
    # .to() returns a same-dtype tensor unchanged even with memory_format set,
    # so the layout is forced separately (a no-op when already contiguous)
    return [t.to(dtype or t.dtype).contiguous() for t in ts]


def _chunk_fwd_triton():
    from aiter.ops.triton.kimi_delta_attn import chunk_kimi_delta_attn

    def run(
        q,
        k,
        v,
        g,
        beta,
        scale,
        chunk_size,
        cu_seqlens=None,
        compute_dtype=None,
        lower_bound=None,
    ):
        q, k, v = _token_major(q, k, v, dtype=compute_dtype or torch.bfloat16)
        g, beta = _token_major(g, beta)
        # same gate mode as the backward: a bounded (safe) gate when the layer
        # has a lower bound; g is already the finished log decay
        kw = (
            {}
            if lower_bound is None
            else {"safe_gate": True, "lower_bound": lower_bound}
        )
        o, _ = chunk_kimi_delta_attn(
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            chunk_size=chunk_size,
            cu_seqlens=cu_seqlens,
            **kw,
        )
        return o.float()

    return run


def _chunk_fwd_flydsl():
    from aiter.ops.flydsl.kda_chunk_fwd import chunk_kda_fwd_flydsl

    native_varlen = _accepts(chunk_kda_fwd_flydsl, "cu_seqlens")

    def run(q, k, v, g, beta, scale, chunk_size, cu_seqlens=None):
        kw = {}
        if cu_seqlens is not None:
            if not native_varlen:
                raise NotImplementedError("this FlyDSL KDA forward has no varlen")
            kw["cu_seqlens"] = cu_seqlens
        q, k, v = _token_major(q, k, v, dtype=torch.bfloat16)
        g, beta = _token_major(g, beta)
        o, _ = chunk_kda_fwd_flydsl(q, k, v, g, beta, scale, chunk_size, **kw)
        return o.float()

    return run


def _chunk_bwd_triton():
    from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_bwd import (
        chunk_delta_attn_bwd,
    )

    def run(
        do,
        q,
        k,
        v,
        g,
        beta,
        scale,
        chunk_size,
        lower_bound=None,
        compute_dtype=None,
        **kw,
    ):
        if lower_bound is not None:
            kw.update(safe_gate=True, lower_bound=lower_bound)
        dtypes_in = [t.dtype for t in (q, k, v, g, beta)]
        q, k, v, do = _token_major(q, k, v, do, dtype=compute_dtype or torch.bfloat16)
        g, beta = _token_major(g, beta)
        out = chunk_delta_attn_bwd(
            do,
            None,
            q,
            k,
            v,
            g,
            beta,
            scale,
            chunk_size=chunk_size,
            **kw,
        )
        grads = [t.to(dt) for t, dt in zip(out[:5], dtypes_in)]
        return (*grads, out[5], out[6])

    return run, CHUNK_BWD_FUSIONS


def _chunk_bwd_flydsl():
    from aiter.ops.flydsl.kda_chunk_bwd import CHUNK_BWD_FUSES, chunk_kda_bwd_flydsl

    return chunk_kda_bwd_flydsl, CHUNK_BWD_FUSES


def _conv_bwd_triton():
    from aiter.ops.triton.gated_delta_net import causal_conv1d_split_qkv_bwd_triton_fn

    return causal_conv1d_split_qkv_bwd_triton_fn


def _conv_bwd_flydsl():
    from aiter.ops.flydsl.causal_conv1d_bwd_flydsl import (
        causal_conv1d_split_qkv_bwd_flydsl_fn,
    )

    return causal_conv1d_split_qkv_bwd_flydsl_fn


_PIECE_LOADERS = {
    ("chunk_fwd", "triton"): _chunk_fwd_triton,
    ("chunk_fwd", "flydsl"): _chunk_fwd_flydsl,
    ("chunk_bwd", "triton"): _chunk_bwd_triton,
    ("chunk_bwd", "flydsl"): _chunk_bwd_flydsl,
    ("conv_bwd", "triton"): _conv_bwd_triton,
    ("conv_bwd", "flydsl"): _conv_bwd_flydsl,
}


# Layer dtypes each FlyDSL piece is built for; other dtypes fall back to the
# next choice instead of feeding the kernel inputs it does not take.
FLYDSL_PIECE_DTYPES = {
    "chunk_fwd": (torch.bfloat16,),
    "chunk_bwd": (torch.bfloat16,),
    "conv_bwd": (torch.bfloat16, torch.float16, torch.float32),
}


def _resolve(name, on_gpu, dtype=None):
    """[(backend, impl), ...] to try in order for ``name``; torch is ``None``.

    ``dtype`` is the layer's input dtype; a FlyDSL piece is skipped (recorded as
    ``<name>_flydsl_skipped``) when it is not one the kernel is built for.
    """
    order = _fallback_order(kda_piece_choices()[name]) if on_gpu else ("torch",)
    found = []
    for backend in order:
        if backend == "torch":
            found.append(("torch", None))
            continue
        if backend == "flydsl" and dtype not in FLYDSL_PIECE_DTYPES[name]:
            LAST_PIECES[f"{name}_flydsl_skipped"] = f"dtype {dtype}"
            continue
        try:
            found.append((backend, _PIECE_LOADERS[(name, backend)]()))
        except ImportError as err:
            # recorded, not hidden: a selected kernel that is not in the tree
            LAST_PIECES[f"{name}_{backend}_missing"] = f"ImportError: {err}"
    return found


def _conv_bwd_kernel(on_gpu, dtype=None):
    backend, fn = _resolve("conv_bwd", on_gpu, dtype)[0]
    LAST_PIECES["conv_bwd"] = backend
    return fn


def _piece(impl):
    """Short label of where an optional stage implementation comes from."""
    module = getattr(impl, "__module__", None) or getattr(impl, "__name__", "")
    if impl is None or module == __name__:
        return "torch"
    return module.rsplit(".", 1)[-1]


def _chunk_fwd(
    q, k, v, g, beta, scale, chunk_size, dtype=None, cu_seqlens=None, lower_bound=None
):
    """Chunk forward by the selected backend; a kernel that rejects the call
    (``ValueError``) or fails to build falls through to the next choice."""
    for backend, fn in _resolve("chunk_fwd", q.is_cuda, dtype):
        if fn is None:
            break
        try:
            kw = {}
            if _has_param(fn, "compute_dtype"):
                kw["compute_dtype"] = _compute_dtype_of(dtype)
            if lower_bound is not None and _has_param(fn, "lower_bound"):
                kw["lower_bound"] = lower_bound
            o = fn(q, k, v, g, beta, scale, chunk_size, cu_seqlens, **kw)
        except (ValueError, NotImplementedError, RuntimeError) as err:
            LAST_PIECES[f"chunk_fwd_{backend}_skipped"] = type(err).__name__
            continue
        LAST_PIECES["chunk_fwd"] = backend
        return o
    LAST_PIECES["chunk_fwd"] = "torch"
    return _chunk_torch_varlen(q, k, v, g, beta, scale, chunk_size, cu_seqlens)


def _bounds(cu_seqlens, T):
    return [0, T] if cu_seqlens is None else [int(b) for b in cu_seqlens.tolist()]


def _chunk_torch_varlen(q, k, v, g, beta, scale, chunk_size, cu_seqlens=None):
    """``chunk_kda_torch`` restarting the state at every ``cu_seqlens`` boundary."""
    if cu_seqlens is None:
        return chunk_kda_torch(q, k, v, g, beta, scale, chunk_size)
    parts = [
        chunk_kda_torch(*(t[:, s:e] for t in (q, k, v, g, beta)), scale, chunk_size)
        for s, e in itertools.pairwise(_bounds(cu_seqlens, q.shape[1]))
    ]
    return torch.cat(parts, 1)


def _short_conv_varlen(a, w, cu_seqlens=None):
    """``_short_conv`` restarting the causal window at every boundary."""
    if cu_seqlens is None:
        return _short_conv(a, w)
    parts = [
        _short_conv(a[:, s:e], w)
        for s, e in itertools.pairwise(_bounds(cu_seqlens, a.shape[1]))
    ]
    return torch.cat(parts, 1)


def _compute_dtype(x):
    """GEMM input dtype: the layer's own 16-bit dtype, else fp32."""
    return x.dtype if x.dtype in (torch.bfloat16, torch.float16) else torch.float32


_MM_OUT_DTYPE_OK = True


def _mm(a, b, cd):
    """2-D a @ b with inputs rounded to ``cd`` and an fp32 result.

    16-bit inputs accumulate and store in fp32, so the output is never rounded
    to 16 bits before the L2 norm and the gates that follow:
    ``torch.mm(out_dtype=)`` where the build has it, else an fp32 GEMM on the
    rounded inputs (the same numbers, slower).
    """
    global _MM_OUT_DTYPE_OK
    if cd == torch.float32:
        return a.float() @ b.float()
    a, b = a.to(cd), b.to(cd)
    if a.is_cuda and _MM_OUT_DTYPE_OK:
        try:
            return torch.mm(a, b, out_dtype=torch.float32)
        except (RuntimeError, TypeError):
            _MM_OUT_DTYPE_OK = False
    return a.float() @ b.float()


def conv_silu_bwd(a, w, dq, dk, dv, fn=None, cu_seqlens=None):
    """Backward of ``silu(causal_depthwise_conv(a, w))`` split into q/k/v.

    a: [B, T, C] conv input; w: [C, W]; dq, dk: [B, T, HK]; dv: [B, T, HV].
    Returns (da [B, T, C], dw [C, W]), fp32. ``fn`` is a ROCm/aiter#8726
    split-q/k/v conv backward (Triton ``causal_conv1d_split_qkv_bwd_triton_fn``
    or FlyDSL ``causal_conv1d_split_qkv_bwd_flydsl_fn``, same signature), fed
    fp32 with ``query_start_loc = cu_seqlens`` for varlen.
    """
    B, T, C = a.shape
    hk, hv = dq.shape[-1], dv.shape[-1]
    if fn is None:
        fn = _conv_bwd_kernel(a.is_cuda)
    if fn is None:
        dc = torch.cat([dq, dk, dv], -1).float()
        with torch.enable_grad():
            a = a.detach().float().requires_grad_(True)
            w = w.detach().float().requires_grad_(True)
            return torch.autograd.grad(_short_conv_varlen(a, w, cu_seqlens), [a, w], dc)
    if cu_seqlens is None:
        qsl = torch.arange(B + 1, device=a.device, dtype=torch.int32) * T
    else:
        assert B == 1, "varlen packs the sequences into one row"
        qsl = cu_seqlens.to(device=a.device, dtype=torch.int32)
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


def _row_ops(x):
    return _flydsl_ops() if x.is_cuda else _TorchOps


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


class _TorchProj:
    """Torch versions of the fused projection GEMMs (``aiter.ops.flydsl.kda_layer_bwd``)."""

    @staticmethod
    def kda_gate_proj(lr, f_b, dt_bias, A_log, num_heads, lower_bound=None):
        z = (lr.float() @ f_b.float().t() + dt_bias.float()).unflatten(
            -1, (num_heads, -1)
        )
        return kda_gate(z, A_log, lower_bound).flatten(-2)

    @staticmethod
    def kda_gate_proj_bwd(lr, f_b, dt_bias, A_log, num_heads, dg, lower_bound=None):
        """(dz, per-element dA_log terms), fp32 [M, H*K]."""
        z = (lr.float() @ f_b.float().t() + dt_bias.float()).unflatten(
            -1, (num_heads, -1)
        )
        a = A_log[:num_heads].float().exp().view(num_heads, 1)
        dg = dg.float().view_as(z)
        if lower_bound is None:
            dz = -dg * a * torch.sigmoid(z)
            dterm = -dg * a * F.softplus(z)
        else:
            sg = torch.sigmoid(a * z)
            dz = dg * lower_bound * sg * (1 - sg) * a
            dterm = dz * z
        return dz.flatten(-2), dterm.flatten(-2)

    @staticmethod
    def gate_proj_bwd_gemms(dgp, x, g_proj):
        dgp = dgp.float()
        return dgp @ g_proj.float(), dgp.t() @ x.float()

    @staticmethod
    def sigmoid_gate_proj(x, g_proj, aux=None):
        out = torch.sigmoid(x.float() @ g_proj.float().t())
        return out if aux is None else out * aux.float()


def _proj_ops(x, w):
    """FlyDSL GEMM-with-epilogue projections on gfx950 for 16-bit weights."""
    if not _fusion_on("proj_epilogue"):
        return _TorchProj
    if x.is_cuda and w.f_b.dtype in (torch.bfloat16, torch.float16):
        from aiter.jit.utils.chip_info import get_gfx_runtime

        if get_gfx_runtime() == "gfx950":
            return _flydsl_ops()
    return _TorchProj


def _kda_layer_fused_fwd(
    x,
    w,
    H,
    chunk_size,
    eps,
    conv_graph=False,
    chunk_graph=False,
    for_backward=True,
    lower_bound=None,
    cu_seqlens=None,
):
    """Forward that keeps what the hand-written backward needs.

    Large GEMMs take their inputs in the layer's 16-bit dtype (fp32 for fp32
    layers) and return fp32; the element-wise stages run in fp32.
    ``conv_graph`` / ``chunk_graph`` record autograd graphs for the conv and
    chunk stages, so their PyTorch fallbacks differentiate the activations of
    this pass instead of recomputing the stage. With ``for_backward=False``
    the output gate is fused into the g_proj GEMM and only ``y`` is meaningful;
    with ``for_backward=True`` ``y`` is not computed.
    """
    B, T, D = x.shape
    cd = _compute_dtype(x)
    xc = x.to(cd).reshape(B * T, D)
    w_qkv = torch.cat([w.w_q, w.w_k, w.w_v]).to(cd)
    conv_qkv = torch.cat([w.conv_q, w.conv_k, w.conv_v]).float()
    hk, hv = w.w_q.shape[0], w.w_v.shape[0]
    side = None
    if _fusion_on("concat_proj"):
        # one pass over x for every projection that has no epilogue of its own
        extra = [w.f_a, w.w_beta] + ([w.g_proj] if for_backward else [])
        full = _mm(xc, torch.cat([w_qkv, *(t.to(cd) for t in extra)]).t(), cd)
        a, *side = full.split([w_qkv.shape[0], *(t.shape[0] for t in extra)], -1)
        a = a.contiguous().view(B, T, -1)
    else:
        a = _mm(xc, w_qkv.t(), cd).view(B, T, -1)  # [B, T, 2HK + HV]
    conv_in = None
    with torch.set_grad_enabled(conv_graph):
        if conv_graph:
            conv_in = [a.requires_grad_(True), conv_qkv.requires_grad_(True)]
        c = _short_conv_varlen(a, conv_qkv, cu_seqlens)
    if conv_graph:
        a, conv_qkv = a.detach(), conv_qkv.detach()
    q_pre, k_pre, v = c.split([hk, hk, hv], -1)
    q_pre, k_pre, v = (t.reshape(B, T, H, -1) for t in (q_pre, k_pre, v))
    q = F.normalize(q_pre, dim=-1, eps=L2NORM_EPS)
    k = F.normalize(k_pre, dim=-1, eps=L2NORM_EPS)
    proj = _proj_ops(x, w)
    lr = side[0].contiguous() if side else _mm(xc, w.f_a.t(), cd)
    g = proj.kda_gate_proj(lr, w.f_b, w.dt_bias, w.A_log, H, lower_bound)
    g = g.view(B, T, H, -1)
    braw = (side[1] if side else _mm(xc, w.w_beta.t(), cd)).reshape(B, T, H)
    beta = torch.sigmoid(braw)
    scale = q.shape[-1] ** -0.5
    chunk_in = None
    with torch.set_grad_enabled(chunk_graph):
        if chunk_graph:
            chunk_in = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
            o = _chunk_torch_varlen(*chunk_in, scale, chunk_size, cu_seqlens)
            LAST_PIECES["chunk_fwd"] = "torch(autograd)"
        else:
            o = _chunk_fwd(
                q, k, v, g, beta, scale, chunk_size, x.dtype, cu_seqlens, lower_bound
            )
    o_graph = o if chunk_graph else None
    o = o.detach()
    normed = o * torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps) * w.norm_w.float()
    gp, y = None, None
    if for_backward:
        gp = (side[2] if side else _mm(xc, w.g_proj.t(), cd)).reshape(o.shape)
        on = (normed * torch.sigmoid(gp)).reshape(B * T, -1)
    else:
        on = proj.sigmoid_gate_proj(xc, w.g_proj, normed.reshape(B * T, -1))
        y = _mm(on, w.w_o.t(), cd).view(B, T, D).to(x.dtype)
    saved = {
        "xc": xc,
        "cd": cd,
        "w_qkv": w_qkv,
        "conv_qkv": conv_qkv,
        "a": a,
        "q_pre": q_pre,
        "k_pre": k_pre,
        "q": q,
        "k": k,
        "v": v,
        "lr": lr,
        "g": g,
        "braw": braw,
        "beta": beta,
        "o": o,
        "gp": gp,
        "on": on,
        "scale": scale,
        "c": c if conv_graph else None,
        "conv_in": conv_in,
        "o_graph": o_graph,
        "chunk_in": chunk_in,
    }
    return y, saved


def _has_param(fn, name):
    """``fn`` names ``name`` explicitly (``**kwargs`` does not count)."""
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def _compute_dtype_of(dtype):
    """Kernel input dtype for a layer of ``dtype``: its own 16-bit type, else fp32."""
    return dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32


def _accepts(fn, name):
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind is p.VAR_KEYWORD for p in params)


def _chunk_bwd_varlen(chunk_bwd, cu_seqlens, do, q, k, v, g, beta, scale, cs, **kw):
    """Call ``chunk_bwd`` natively with ``cu_seqlens`` when it takes them, else
    once per sequence (gradients concatenated, dA_log / ddt_bias summed)."""
    if cu_seqlens is None:
        return chunk_bwd(do, q, k, v, g, beta, scale, cs, **kw)
    if _accepts(chunk_bwd, "cu_seqlens"):
        return chunk_bwd(do, q, k, v, g, beta, scale, cs, cu_seqlens=cu_seqlens, **kw)
    outs = [
        chunk_bwd(*(t[:, s:e] for t in (do, q, k, v, g, beta)), scale, cs, **kw)
        for s, e in itertools.pairwise(_bounds(cu_seqlens, q.shape[1]))
    ]
    grads = [torch.cat([o[i] for o in outs], 1) for i in range(5)]
    tail = [None if outs[0][i] is None else sum(o[i] for o in outs) for i in (5, 6)]
    return (*grads, *tail)


def _fused_plan(x, w, ops, lower_bound):
    """Pick the backward's kernels: (chunk_impl, fuses, conv_kernel, rms_bwd)."""
    LAST_PIECES.clear()
    if _chunk_bwd_impl is not None:
        chunk_impl, fuses = _chunk_bwd_impl, _chunk_bwd_fuses
        LAST_PIECES["chunk_bwd"] = f"registered:{_piece(chunk_impl)}"
    else:
        backend, impl = _resolve("chunk_bwd", x.is_cuda, x.dtype)[0]
        chunk_impl, fuses = impl if impl is not None else (None, frozenset())
        LAST_PIECES["chunk_bwd"] = backend
    if (
        lower_bound is not None
        and "gate" in fuses
        and not _accepts(chunk_impl, "lower_bound")
    ):
        # the kernel cannot apply the safe gate; feed it the finished decay
        fuses = fuses - {"gate"}
    wanted = kda_fusion_choices()["chunk_fuse"]
    if wanted != "auto":
        fuses = (
            fuses & frozenset() if wanted == "none" else fuses & set(wanted.split("+"))
        )
    conv_kernel = _conv_bwd_kernel(x.is_cuda, x.dtype)
    rms_bwd = _gated_rmsnorm_bwd_impl or _default_gated_rmsnorm_bwd(x.is_cuda)
    LAST_PIECES.update(
        proj_gemms=_piece(_proj_ops(x, w)),
        rmsnorm_bwd=_piece(rms_bwd),
        chunk_bwd_fuses=",".join(sorted(fuses)) or "-",
        l2norm_bwd="fused" if "qk_l2norm" in fuses else _piece(ops),
        fusions=",".join(f"{k}={v}" for k, v in sorted(kda_fusion_choices().items())),
    )
    return chunk_impl, frozenset(fuses), conv_kernel, rms_bwd


def _fused_saved_fwd(x, w, H, chunk_size, eps, plan, lower_bound, cu_seqlens):
    chunk_impl, _, conv_kernel, _ = plan
    with torch.enable_grad():
        return _kda_layer_fused_fwd(
            x,
            w,
            H,
            chunk_size,
            eps,
            conv_graph=conv_kernel is None,
            chunk_graph=chunk_impl is None,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
        )[1]


def _kda_layer_fused_bwd(
    dy,
    x,
    w,
    H,
    chunk_size,
    eps,
    ops,
    lower_bound=None,
    cu_seqlens=None,
    saved=None,
):
    """``saved`` = (plan, state) from a ``save_fwd`` forward skips the recompute."""
    if saved is None:
        plan = _fused_plan(x, w, ops, lower_bound)
        s = _fused_saved_fwd(x, w, H, chunk_size, eps, plan, lower_bound, cu_seqlens)
    else:
        plan, s = saved
    LAST_PIECES["recompute"] = "no (save_fwd)" if saved is not None else "yes"
    chunk_impl, fuses, conv_kernel, rms_bwd = plan
    B, T, D = x.shape
    xc, cd = s["xc"], s["cd"]
    dyc = dy.reshape(B * T, D).to(cd)

    d_wo = _mm(dyc.t(), s["on"], cd)
    d_on = _mm(dyc, w.w_o, cd).view_as(s["o"])
    do, dgp, d_norm = rms_bwd(s["o"], s["gp"], w.norm_w, d_on, eps)
    dgp = dgp.reshape(B * T, -1)
    dx, d_gproj = _proj_ops(x, w).gate_proj_bwd_gemms(dgp, xc, w.g_proj)

    if chunk_impl is None:

        def chunk_bwd(do, *_, **__):
            grads = torch.autograd.grad(s["o_graph"], s["chunk_in"], do.float())
            return (*grads, None, None)

        chunk_bwd.__signature__ = inspect.Signature(
            [inspect.Parameter("cu_seqlens", inspect.Parameter.KEYWORD_ONLY)]
        )

    else:
        chunk_bwd = chunk_impl
    lr = s["lr"].reshape(B * T, -1)
    kw = {}
    q_in, k_in, g_in, b_in = s["q"], s["k"], s["g"], s["beta"]
    if "qk_l2norm" in fuses:
        q_in, k_in = s["q_pre"], s["k_pre"]
        kw["use_qk_l2norm_in_kernel"] = True
    if "gate" in fuses:
        g_in = (lr @ w.f_b.float().t()).view_as(s["g"])
        kw.update(use_gate_in_kernel=True, A_log=w.A_log[:H], dt_bias=w.dt_bias)
        if lower_bound is not None:
            kw["lower_bound"] = lower_bound
    if "beta_sigmoid" in fuses:
        b_in = s["braw"]
        kw["use_beta_sigmoid_in_kernel"] = True
    if chunk_impl is not None and _has_param(chunk_impl, "compute_dtype"):
        kw["compute_dtype"] = _compute_dtype_of(x.dtype)
    v_in = s["v"]
    if chunk_impl is not None:
        # every chunk kernel indexes token-major; q/k/v are channel-major views
        do, q_in, k_in, v_in, g_in, b_in = _token_major(
            do, q_in, k_in, v_in, g_in, b_in
        )
    dq, dk, dv, dg, dbeta, dA_log, ddt = _chunk_bwd_varlen(
        chunk_bwd,
        cu_seqlens,
        do,
        q_in,
        k_in,
        v_in,
        g_in,
        b_in,
        s["scale"],
        chunk_size,
        **kw,
    )
    dz = dg.float().reshape(B * T, -1)
    if "gate" not in fuses:
        dz, dterm = _proj_ops(x, w).kda_gate_proj_bwd(
            lr, w.f_b, w.dt_bias, w.A_log, H, dz, lower_bound
        )
        dA_log = dterm.view(-1, *s["g"].shape[-2:]).sum((0, 2))
    # padded A_log entries get a zero gradient
    dA_log = F.pad(dA_log.float(), (0, w.A_log.numel() - dA_log.numel()))
    db = dbeta.float()
    if "beta_sigmoid" not in fuses:
        db = db * s["beta"] * (1 - s["beta"])
    db = db.reshape(B * T, -1)
    if ddt is None:
        ddt = dz.sum(0)
    d_fb = dz.t() @ lr
    dlr = dz @ w.f_b.float()
    d_fa = _mm(dlr.t(), xc, cd)
    dx = dx + _mm(dlr, w.f_a, cd) + _mm(db, w.w_beta, cd)
    d_wbeta = _mm(db.t(), xc, cd)

    if "qk_l2norm" not in fuses:
        dq = ops.l2norm_bwd(s["q_pre"], dq.float(), L2NORM_EPS)
        dk = ops.l2norm_bwd(s["k_pre"], dk.float(), L2NORM_EPS)
    dqkv = [t.reshape(B, T, -1) for t in (dq, dk, dv)]
    if conv_kernel is None:
        dc = torch.cat(dqkv, -1).float()
        da, d_conv = torch.autograd.grad(s["c"], s["conv_in"], dc)
    else:
        da, d_conv = conv_silu_bwd(
            s["a"], s["conv_qkv"], *dqkv, fn=conv_kernel, cu_seqlens=cu_seqlens
        )
    da = da.reshape(B * T, -1)
    d_wqkv = _mm(da.t(), xc, cd)
    dx = dx + _mm(da, s["w_qkv"], cd)

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
        g_proj=d_gproj,
        norm_w=d_norm,
        w_o=d_wo,
    )
    grads = KDALayerWeights.from_tensors(
        [gr.to(p.dtype) for gr, p in zip(grads.tensors(), w.tensors())]
    )
    LAST_PIECES["gemm"] = (
        "fp32"
        if cd == torch.float32
        else f"{cd}->fp32" if _MM_OUT_DTYPE_OK else f"{cd} inputs, fp32 GEMM"
    )
    return dx.view(B, T, D).to(x.dtype), grads


class _KDALayerFused(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, num_heads, chunk_size, eps, lower_bound, cu_seqlens, *ws):
        ctx.save_for_backward(x, *ws)
        ctx.cfg = (num_heads, chunk_size, eps)
        ctx.lower_bound, ctx.cu_seqlens = lower_bound, cu_seqlens
        w = KDALayerWeights.from_tensors(ws)
        ctx.saved_state = None
        if _fusion_on("save_fwd"):
            plan = _fused_plan(x, w, _row_ops(x), lower_bound)
            s = _fused_saved_fwd(
                x, w, num_heads, chunk_size, eps, plan, lower_bound, cu_seqlens
            )
            ctx.saved_state = (plan, s)
            B, T, D = x.shape
            return _mm(s["on"], w.w_o.t(), s["cd"]).view(B, T, D).to(x.dtype)
        y, _ = _kda_layer_fused_fwd(
            x,
            w,
            num_heads,
            chunk_size,
            eps,
            for_backward=False,
            lower_bound=lower_bound,
            cu_seqlens=cu_seqlens,
        )
        return y

    @staticmethod
    def backward(ctx, dy):
        x, *ws = ctx.saved_tensors
        dx, dw = _kda_layer_fused_bwd(
            dy,
            x,
            KDALayerWeights.from_tensors(ws),
            *ctx.cfg,
            _row_ops(x),
            lower_bound=ctx.lower_bound,
            cu_seqlens=ctx.cu_seqlens,
            saved=ctx.saved_state,
        )
        ctx.saved_state = None
        return (dx, None, None, None, None, None, *dw.tensors())


def _segments(x, cu_seqlens):
    """[(start, end), ...] token ranges; ``cu_seqlens`` needs B == 1."""
    if cu_seqlens is None:
        return None
    if x.shape[0] != 1:
        raise ValueError(f"cu_seqlens needs a packed batch of 1, got B={x.shape[0]}")
    bounds = [int(b) for b in cu_seqlens.tolist()]
    if bounds[0] != 0 or bounds[-1] != x.shape[1] or bounds != sorted(bounds):
        raise ValueError(f"cu_seqlens {bounds} does not cover T={x.shape[1]}")
    return [(s, e) for s, e in itertools.pairwise(bounds) if e > s]


def kda_layer_fwd(
    x,
    w: KDALayerWeights,
    num_heads,
    chunk_size=64,
    eps=1e-6,
    backend="torch",
    cu_seqlens=None,
    lower_bound=None,
):
    """x: [B,T,D] -> y: [B,T,D]. Differentiable for both backends.

    ``lower_bound`` (e.g. Kimi-K3 ``gate_lower_bound=-5``) selects the safe gate
    ``lower_bound * sigmoid(exp(A_log) * z)``.

    ``cu_seqlens`` ([N + 1] cumulative lengths, x packed as [1, total, D]) runs
    each sequence independently: the conv and the recurrence restart at every
    boundary.
    """
    if backend not in KDA_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {KDA_BACKENDS}")
    segments = _segments(x, cu_seqlens)
    if backend == "fused":
        return _KDALayerFused.apply(
            x, num_heads, chunk_size, eps, lower_bound, cu_seqlens, *w.tensors()
        )
    if segments is not None:
        return torch.cat(
            [
                kda_layer_fwd(
                    x[:, s:e], w, num_heads, chunk_size, eps, backend, None, lower_bound
                )
                for s, e in segments
            ],
            1,
        )
    return _kda_layer_torch(x, w, num_heads, chunk_size, eps, lower_bound)


def kda_layer_bwd(
    dy,
    x,
    w: KDALayerWeights,
    num_heads,
    chunk_size=64,
    eps=1e-6,
    backend="torch",
    cu_seqlens=None,
    lower_bound=None,
):
    """Backward of one KDA layer.

    Returns (dx, dw) where dw is a KDALayerWeights holding the gradient of every
    weight (q/k/v proj, short conv, F-decay low-rank gate + A_log/dt_bias, beta
    proj, output gate proj, norm weight, output proj). The forward is recomputed.
    ``cu_seqlens`` as in ``kda_layer_fwd``; weight gradients sum over sequences.
    """
    if backend not in KDA_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {KDA_BACKENDS}")
    segments = _segments(x, cu_seqlens)
    if backend == "fused":
        return _kda_layer_fused_bwd(
            dy,
            x,
            w,
            num_heads,
            chunk_size,
            eps,
            _row_ops(x),
            lower_bound,
            cu_seqlens if segments is not None else None,
        )
    if segments is not None:
        dxs, dw = [], None
        for s, e in segments:
            dx_s, dw_s = kda_layer_bwd(
                dy[:, s:e],
                x[:, s:e],
                w,
                num_heads,
                chunk_size,
                eps,
                backend,
                None,
                lower_bound,
            )
            dxs.append(dx_s)
            parts = [t.float() for t in dw_s.tensors()]
            dw = parts if dw is None else [a + b for a, b in zip(dw, parts)]
        dw = [d.to(p.dtype) for d, p in zip(dw, w.tensors())]
        return torch.cat(dxs, 1), KDALayerWeights.from_tensors(dw)
    with torch.enable_grad():
        xs = x.detach().requires_grad_(True)
        ws = [t.detach().requires_grad_(True) for t in w.tensors()]
        y = _kda_layer_torch(
            xs,
            KDALayerWeights.from_tensors(ws),
            num_heads,
            chunk_size,
            eps,
            lower_bound,
        )
        grads = torch.autograd.grad(y, [xs, *ws], dy)
    return grads[0], KDALayerWeights.from_tensors(list(grads[1:]))
