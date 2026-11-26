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

import contextlib
import inspect
import itertools
import os
import weakref
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


# Precision mode: "default" keeps every element-wise activation in fp32;
# "megatron" reproduces Megatron-LM #6872's KDA (docs/megatron-kda-precision-map
# in the project store): projection outputs, the conv pre-activation, SiLU
# output, L2-normed q/k, the chunk output, the normed value and the gated-norm
# output are rounded to bf16; beta, the gate, rstd and accumulations stay fp32;
# conv and norm weights are bf16 at runtime, A_log / dt_bias fp32.
KDA_PRECISIONS = ("default", "megatron")
_precision_override = None


def set_kda_precision(mode):
    """``"default"`` or ``"megatron"``; ``None`` falls back to ``AITER_KDA_PRECISION``."""
    global _precision_override
    if mode is not None and mode not in KDA_PRECISIONS:
        raise ValueError(f"precision {mode!r}, expected one of {KDA_PRECISIONS}")
    _precision_override = mode


def kda_precision():
    mode = _precision_override or os.environ.get("AITER_KDA_PRECISION", "default")
    return mode if mode in KDA_PRECISIONS else "default"


def _rb(t):
    """bf16 rounding in "megatron" mode (gradient passes straight through, as
    for a dtype cast); identity otherwise."""
    if kda_precision() != "megatron":
        return t
    r = t.to(torch.bfloat16).float()
    return t + (r - t).detach() if t.requires_grad else r


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
    # megatron: bf16 weights, fp32 accumulation rounded to bf16 before the SiLU,
    # bf16 output
    T = x.shape[1]
    w = _rb(w)
    y = F.conv1d(
        x.transpose(1, 2), w.unsqueeze(1), padding=w.shape[1] - 1, groups=w.shape[0]
    )
    return _rb(F.silu(_rb(y[..., :T]))).transpose(1, 2)


def kda_layer_preprocess(x, w: KDALayerWeights, num_heads, eps=1e-6, lower_bound=None):
    """Everything before the recurrence. Returns q, k [B,T,H,K], v [B,T,H,V],
    g (log decay) [B,T,H,K], beta [B,T,H], fp32."""
    B, T, _ = x.shape
    H = num_heads
    xf = x.float()
    q = _short_conv(_rb(xf @ w.w_q.float().t()), w.conv_q.float()).view(B, T, H, -1)
    k = _short_conv(_rb(xf @ w.w_k.float().t()), w.conv_k.float()).view(B, T, H, -1)
    v = _short_conv(_rb(xf @ w.w_v.float().t()), w.conv_v.float()).view(B, T, H, -1)
    q = _rb(F.normalize(q, dim=-1, eps=L2NORM_EPS))
    k = _rb(F.normalize(k, dim=-1, eps=L2NORM_EPS))
    raw = _rb(_rb(xf @ w.f_a.float().t()) @ w.f_b.float().t())
    g = kda_gate((raw + w.dt_bias.float()).view(B, T, H, -1), w.A_log, lower_bound)
    beta = torch.sigmoid(_rb(xf @ w.w_beta.float().t()))
    return q, k, v, g, beta


def kda_layer_postprocess(o, x, w: KDALayerWeights, eps=1e-6):
    """Gated RMSNorm + output projection. o: [B,T,H,V] fp32."""
    B, T, H, V = o.shape
    xf = x.float()
    o = _rb(o)
    gate = torch.sigmoid(_rb(xf @ w.g_proj.float().t())).view(B, T, H, V)
    rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    normed = _rb(o * rstd * _rb(w.norm_w.float()))
    o = _rb(normed * gate)
    return _rb(o.reshape(B, T, H * V) @ w.w_o.float().t()).to(x.dtype)


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


class _Guarded:
    """A kernel module whose calls fall back to torch instead of raising.

    ``checks[name](*args)`` returns why the kernel cannot take the call (a
    shape, dtype or arch constraint) or None; a kernel that still rejects the
    call at runtime falls back the same way. Either way the reason is recorded
    in ``LAST_PIECES['<name>_skipped']``.
    """

    _ERRORS = (ValueError, AssertionError, NotImplementedError, RuntimeError, TypeError)

    def __init__(self, primary, fallback, checks=None):
        self._primary, self._fallback, self._checks = primary, fallback, checks or {}
        self._label = _piece(primary)

    def __getattr__(self, name):
        kernel = getattr(self._primary, name)  # AttributeError if the kernel lacks it
        backup = getattr(self._fallback, name, None)
        if backup is None or not callable(kernel):
            return kernel
        check = self._checks.get(name)

        def call(*args, **kwargs):
            reason = check(*args, **kwargs) if check else None
            if reason is None:
                try:
                    return kernel(*args, **kwargs)
                except self._ERRORS as err:
                    reason = f"{type(err).__name__}: {err}"
            LAST_PIECES[f"{name}_skipped"] = reason
            return backup(*args, **kwargs)

        call._label = self._label
        return call


def _last_dim_multiple(n):
    def check(x, *_, **__):
        d = x.shape[-1]
        return None if d % n == 0 else f"last dim {d} is not a multiple of {n}"

    return check


def _gemm_epilogue_check(a, w, *_, **__):
    # gfx950 GEMM: 16-byte DMA rows and the store vector along N
    if w.dtype not in (torch.bfloat16, torch.float16):
        return f"weight dtype {w.dtype}"
    for name, n in (("K", a.shape[-1]), ("N", w.shape[0]), ("D", w.shape[-1])):
        if n % 8:
            return f"{name}={n} is not a multiple of 8"
    return None


def _default_gated_rmsnorm_bwd(on_gpu):
    if on_gpu:
        try:
            from aiter.ops.flydsl import gated_rmsnorm_bwd as rms_mod
        except ImportError:
            return _TorchOps.gated_rmsnorm_bwd
        # #8741 needs V % 64 == 0
        return _Guarded(
            rms_mod,
            _TorchOps,
            {"gated_rmsnorm_bwd": _last_dim_multiple(64)},
        ).gated_rmsnorm_bwd
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

# Each stage defaults to its faster kernel on MI350X: FlyDSL for the chunk
# forward (#8744, 22-31% faster than Triton), Triton elsewhere. Override with
# ``set_kda_pieces`` or ``AITER_KDA_PIECES`` (e.g. ``chunk_fwd=triton``). A
# missing, unsupported or rejecting kernel falls back to the next choice, with
# the reason in LAST_PIECES and the log: flydsl -> triton -> torch,
# triton -> torch.
KDA_PIECE_DEFAULTS = {
    "chunk_fwd": "flydsl",
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

    try:
        from aiter.ops.flydsl.kda_chunk_fwd import kda_fwd_flydsl_supported as supported
    except ImportError:
        supported = None
    native_varlen = _accepts(chunk_kda_fwd_flydsl, "cu_seqlens")

    def run(q, k, v, g, beta, scale, chunk_size, cu_seqlens=None):
        H, K, V = q.shape[2], q.shape[3], v.shape[3]
        if supported is not None and not supported(
            H=H,
            HV=v.shape[2],
            K=K,
            V=V,
            chunk_size=chunk_size,
            dtype=torch.bfloat16,
            is_varlen=cu_seqlens is not None,
        ):
            raise ValueError(
                f"unsupported shape for the FlyDSL chunk forward: H={H} "
                f"HV={v.shape[2]} K={K} V={V} chunk={chunk_size} (needs gfx950, "
                "K=V=128, chunk 64, H=HV)"
            )
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


_logged_fallbacks = set()


def _log_fallback(stage, backend, reason):
    """Log each distinct fallback once per process."""
    if (stage, backend, reason) in _logged_fallbacks:
        return
    _logged_fallbacks.add((stage, backend, reason))
    import logging

    logging.getLogger("aiter").info(
        "KDA %s: %s not used, falling back (%s)", stage, backend, reason
    )


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
            _log_fallback(name, backend, f"dtype {dtype}")
            continue
        try:
            found.append((backend, _PIECE_LOADERS[(name, backend)]()))
        except ImportError as err:
            # recorded, not hidden: a selected kernel that is not in the tree
            LAST_PIECES[f"{name}_{backend}_missing"] = f"ImportError: {err}"
            _log_fallback(name, backend, f"ImportError: {err}")
    return found


def _conv_bwd_kernel(on_gpu, dtype=None):
    backend, fn = _resolve("conv_bwd", on_gpu, dtype)[0]
    LAST_PIECES["conv_bwd"] = backend
    return fn


def _piece(impl):
    """Short label of where an optional stage implementation comes from."""
    if isinstance(getattr(impl, "__dict__", {}).get("_label"), str):
        return impl._label
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
            reason = f"{type(err).__name__}: {err}"
            LAST_PIECES[f"chunk_fwd_{backend}_skipped"] = reason
            _log_fallback("chunk_fwd", backend, reason)
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


ProjGemmHook = Callable[..., object]
_proj_gemm_hook: ProjGemmHook | None = None


def register_proj_gemm(fn: ProjGemmHook | None):
    """Install a projection-GEMM implementation (e.g. an MXFP8 recipe).

    ``fn(a, w, *, kind, cd, epilogue, name, **epilogue_args)`` gets every
    projection GEMM of the fused path, forward, dgrad and wgrad, including the
    concatenated projection and the epilogue-fused ones (see ``_proj_gemm`` for
    the operand convention). It returns the fp32 result, or ``NotImplemented``
    to leave that call to the default path. ``None`` uninstalls it.
    """
    global _proj_gemm_hook
    _proj_gemm_hook = fn


def _proj_gemm(a, w, kind, cd, epilogue=None, ops=None, name=None, **epi):
    """The single entry point for every projection GEMM of the fused path.

    kind  "fwd":   a [M, K] @ w[N, K]^T   (w a weight)
          "dgrad": a [M, N] @ w[N, K]     (a an output grad, w a weight)
          "wgrad": a[M, N]^T @ w[M, K]    (a an output grad, w the input)
    ``cd`` is the input dtype (fp32 accumulation and output). ``epilogue``
    ("kda_gate", "kda_gate_bwd", "sigmoid_mul"; ``kind="fwd"``) runs on
    ``ops`` (FlyDSL GEMM epilogues or ``_TorchProj``) with ``epi`` as its
    arguments. ``name`` labels the projection (qkv, f_a, f_b, w_beta, g_proj,
    w_o, ...) for an installed hook.
    """
    if _proj_gemm_hook is not None:
        out = _proj_gemm_hook(
            a, w, kind=kind, cd=cd, epilogue=epilogue, name=name, **epi
        )
        if out is not NotImplemented:
            return out
    if epilogue is not None:
        ops = ops or _TorchProj
        lb = epi.get("lower_bound")
        if epilogue == "kda_gate":
            return ops.kda_gate_proj(
                a, w, epi["dt_bias"], epi["A_log"], epi["num_heads"], lb
            )
        if epilogue == "kda_gate_bwd":
            return ops.kda_gate_proj_bwd(
                a, w, epi["dt_bias"], epi["A_log"], epi["num_heads"], epi["dg"], lb
            )
        if epilogue == "sigmoid_mul":
            return ops.sigmoid_gate_proj(a, w, epi.get("aux"))
        raise ValueError(f"unknown projection epilogue {epilogue!r}")
    if ops is not None and hasattr(ops, "proj_gemm"):
        return ops.proj_gemm(a, w, kind)
    if kind == "fwd":
        return _mm(a, w.t(), cd)
    if kind == "dgrad":
        return _mm(a, w, cd)
    if kind == "wgrad":
        return _mm(a.t(), w, cd)
    raise ValueError(f"unknown projection GEMM kind {kind!r}")


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
    if not x.is_cuda:
        return _TorchOps
    # the FlyDSL L2-norm kernel needs rows of a multiple of 64
    return _Guarded(_flydsl_ops(), _TorchOps, {"l2norm_bwd": _last_dim_multiple(64)})


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
    def gated_rmsnorm_bwd_megatron(o, gp, w, dy, eps=1e-6):
        """Megatron ``gated_norm_bwd``: grad·sigmoid(g) rounded to bf16 before the
        norm gradient; dgate = (grad·normed)·σ(1−σ) in fp32; normed rounded to
        bf16 as in the forward; bf16 weight."""
        o, gp, dy = o.float(), gp.float(), dy.float()
        w = _rb(w.float())
        sg = torch.sigmoid(gp)
        rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
        normed = _rb(o * rstd * w)
        dgp = dy * normed * sg * (1 - sg)
        dn = _rb(dy * sg)
        dxhat = dn * w
        do = rstd * (dxhat - o * rstd.pow(2) * (dxhat * o).mean(-1, keepdim=True))
        dw = (dn * o * rstd).reshape(-1, o.shape[-1]).sum(0)
        return do, dgp, dw

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
        return _TorchProj.gate_bwd_from_z(z, A_log, num_heads, dg, lower_bound)

    @staticmethod
    def gate_bwd_from_z(z, A_log, num_heads, dg, lower_bound=None):
        """Gate backward from the pre-activation z [M, H, K] (dt_bias included)."""
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
            return _Guarded(
                _flydsl_ops(),
                _TorchProj,
                {
                    name: _gemm_epilogue_check
                    for name in (
                        "kda_gate_proj",
                        "kda_gate_proj_bwd",
                        "sigmoid_gate_proj",
                        "proj_gemm",
                    )
                },
            )
    return _TorchProj


_WEIGHT_CAT_CACHE = {}
_WEIGHT_CAT_CACHE_MAX = 16


class _CachedCat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cat, *srcs):
        ctx.rows = [t.shape[0] for t in srcs]
        ctx.dtypes = [t.dtype for t in srcs]
        return cat

    @staticmethod
    def backward(ctx, g):
        return (None,) + tuple(p.to(dt) for p, dt in zip(g.split(ctx.rows), ctx.dtypes))


_WEIGHT_CACHE_MODE = None
_INVALIDATE_HOOKS = []


@contextlib.contextmanager
def _weight_cache_mode(is_first_microbatch):
    global _WEIGHT_CACHE_MODE
    prev, _WEIGHT_CACHE_MODE = _WEIGHT_CACHE_MODE, is_first_microbatch
    try:
        yield
    finally:
        _WEIGHT_CACHE_MODE = prev


def kda_weight_reuse():
    """True while a KDA call with ``is_first_microbatch`` True/False runs (its
    forward and its backward): derived weights may be cached across calls.
    False for ``is_first_microbatch=None``: build them per call."""
    return _WEIGHT_CACHE_MODE is not None


def register_weight_cache_invalidator(fn):
    """``fn(weights, anchors)`` runs on every ``invalidate_weight_caches``:
    ``weights`` as passed (None means everything), ``anchors`` the dropped
    ``cached_weight_cat`` results (w_qkv / w_all / conv_qkv) of those weights,
    e.g. for a cache of quantized weights keyed on them."""
    _INVALIDATE_HOOKS.append(fn)
    return fn


def invalidate_weight_caches(weights=None):
    """Drop cached derived weights (concatenations, and those of registered
    hooks) built from ``weights`` (a KDALayerWeights or tensors), or all of
    them. Needed after an update that keeps ``data_ptr`` and ``_version``
    (``p.data.copy_``, a distributed optimizer's shared buffer) when calls
    pass ``is_first_microbatch=False``."""
    if weights is None:
        drop = list(_WEIGHT_CAT_CACHE)
    else:
        ts = weights.tensors() if hasattr(weights, "tensors") else weights
        ids = {id(t) for t in ts if t is not None}
        drop = [
            k
            for k, (refs, _, _) in _WEIGHT_CAT_CACHE.items()
            if any(id(r()) in ids for r in refs)
        ]
    anchors = [_WEIGHT_CAT_CACHE.pop(k)[1] for k in drop]
    for fn in _INVALIDATE_HOOKS:
        fn(weights, anchors)


def cached_weight_cat(srcs, dtype):
    """``torch.cat(srcs).to(dtype)`` with Transformer Engine's
    ``is_first_microbatch`` semantics (set by ``kda_layer_fwd``/``_bwd``):
    None builds it fresh, True rebuilds and caches it, False reuses the
    cached copy. An optimizer update through ``p.data.copy_`` or a shared
    buffer is invisible to the cache, so pass True on the first microbatch
    after a step (or call ``invalidate_weight_caches``). Under grad mode the
    gradient is split back to the sources. Do not modify the result in place.
    """
    mode = _WEIGHT_CACHE_MODE
    key = (dtype,) + tuple(
        (t.data_ptr(), tuple(t.shape), t.stride(), t.dtype, t.device) for t in srcs
    )
    versions = tuple(t._version for t in srcs)
    hit = _WEIGHT_CAT_CACHE.get(key) if mode is False else None
    if (
        hit is not None
        and hit[2] == versions
        and all(r() is t for r, t in zip(hit[0], srcs))
    ):
        cat = hit[1]
    else:
        with torch.no_grad():
            cat = torch.cat([t.detach() for t in srcs]).to(dtype)
        if mode is not None:
            if len(_WEIGHT_CAT_CACHE) >= _WEIGHT_CAT_CACHE_MAX:
                _WEIGHT_CAT_CACHE.clear()
            _WEIGHT_CAT_CACHE[key] = ([weakref.ref(t) for t in srcs], cat, versions)
    if torch.is_grad_enabled() and any(t.requires_grad for t in srcs):
        return _CachedCat.apply(cat, *srcs)
    return cat


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
    chunk_override=None,
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
    w_qkv = cached_weight_cat([w.w_q, w.w_k, w.w_v], cd)
    conv_qkv = _rb(cached_weight_cat([w.conv_q, w.conv_k, w.conv_v], torch.float32))
    hk, hv = w.w_q.shape[0], w.w_v.shape[0]
    megatron = kda_precision() == "megatron"
    side = None
    if _fusion_on("concat_proj"):
        # one pass over x for every projection that has no epilogue of its own
        extra = [w.f_a, w.w_beta] + ([w.g_proj] if for_backward else [])
        w_all = cached_weight_cat([w.w_q, w.w_k, w.w_v, *extra], cd)
        full = _proj_gemm(xc, w_all, "fwd", cd, name="concat")
        a, *side = full.split([w_qkv.shape[0], *(t.shape[0] for t in extra)], -1)
        side = [_rb(t) for t in side]
        a = _rb(a.contiguous()).view(B, T, -1)
    else:
        a = _rb(_proj_gemm(xc, w_qkv, "fwd", cd, name="qkv")).view(B, T, -1)
    conv_in = None
    with torch.set_grad_enabled(conv_graph):
        if conv_graph:
            conv_in = [a.requires_grad_(True), conv_qkv.requires_grad_(True)]
        c = _short_conv_varlen(a, conv_qkv, cu_seqlens)
    if conv_graph:
        a, conv_qkv = a.detach(), conv_qkv.detach()
    q_pre, k_pre, v = c.split([hk, hk, hv], -1)
    q_pre, k_pre, v = (t.reshape(B, T, H, -1) for t in (q_pre, k_pre, v))
    q = _rb(F.normalize(q_pre, dim=-1, eps=L2NORM_EPS))
    k = _rb(F.normalize(k_pre, dim=-1, eps=L2NORM_EPS))
    proj = _proj_ops(x, w)
    if side:
        lr = side[0].contiguous()
    else:
        lr = _rb(_proj_gemm(xc, w.f_a, "fwd", cd, name="f_a"))
    raw_g = None
    if megatron:
        # bf16 raw_g, then the gate in fp32 (no GEMM epilogue)
        raw_g = _rb(_proj_gemm(lr, w.f_b, "fwd", cd, name="f_b")).view(B, T, H, -1)
        g = kda_gate(raw_g + w.dt_bias.float().view(H, -1), w.A_log, lower_bound)
    else:
        g = _proj_gemm(
            lr,
            w.f_b,
            "fwd",
            cd,
            epilogue="kda_gate",
            ops=proj,
            name="f_b",
            dt_bias=w.dt_bias,
            A_log=w.A_log,
            num_heads=H,
            lower_bound=lower_bound,
        )
    g = g.view(B, T, H, -1)
    if side:
        braw = side[1]
    else:
        braw = _rb(_proj_gemm(xc, w.w_beta, "fwd", cd, name="w_beta"))
    braw = braw.reshape(B, T, H)
    beta = torch.sigmoid(braw)
    scale = q.shape[-1] ** -0.5
    chunk_in, chunk_inter = None, None
    with torch.set_grad_enabled(chunk_graph):
        if chunk_graph:
            chunk_in = [t.detach().requires_grad_(True) for t in (q, k, v, g, beta)]
            o = _chunk_torch_varlen(*chunk_in, scale, chunk_size, cu_seqlens)
            LAST_PIECES["chunk_fwd"] = "torch(autograd)"
        elif chunk_override is not None:
            pre = {"q_pre": q_pre, "k_pre": k_pre, "q": q, "k": k, "v": v}
            pre.update(g=g, raw_g=raw_g, braw=braw, beta=beta, lr=lr, scale=scale)
            o, chunk_inter = chunk_override(pre)
        else:
            o = _chunk_fwd(
                q, k, v, g, beta, scale, chunk_size, x.dtype, cu_seqlens, lower_bound
            )
    o_graph = o if chunk_graph else None
    o = _rb(o.detach())
    rstd = torch.rsqrt(o.pow(2).mean(-1, keepdim=True) + eps)
    normed = _rb(o * rstd * _rb(w.norm_w.float()))
    gp, y = None, None
    if for_backward or megatron:
        gp = side[2] if side and for_backward else None
        if gp is None:
            gp = _rb(_proj_gemm(xc, w.g_proj, "fwd", cd, name="g_proj"))
        gp = gp.reshape(o.shape)
        on = _rb(normed * torch.sigmoid(gp)).reshape(B * T, -1)
        if not for_backward:
            y = _rb(_proj_gemm(on, w.w_o, "fwd", cd, name="w_o"))
            y = y.view(B, T, D).to(x.dtype)
    else:
        on = _proj_gemm(
            xc,
            w.g_proj,
            "fwd",
            cd,
            epilogue="sigmoid_mul",
            ops=proj,
            name="g_proj",
            aux=normed.reshape(B * T, -1),
        )
        y = _proj_gemm(on, w.w_o, "fwd", cd, name="w_o").view(B, T, D).to(x.dtype)
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
        "raw_g": raw_g,
        "chunk_inter": chunk_inter,
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
        precision=kda_precision(),
    )
    return chunk_impl, frozenset(fuses), conv_kernel, rms_bwd


def _fused_saved_fwd(
    x, w, H, chunk_size, eps, plan, lower_bound, cu_seqlens, keep_chunk=False
):
    """``keep_chunk`` (the ``save_fwd`` forward) also keeps the chunk forward's
    intermediates for a chunk backward that takes them."""
    chunk_impl, _, conv_kernel, _ = plan
    override = None
    if keep_chunk and chunk_impl is not None:
        override = _save_chunk_intermediates(
            x, w, H, chunk_size, plan, lower_bound, cu_seqlens
        )
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
            chunk_override=override,
        )[1]


def _chunk_bwd_args(s, w, H, fuses, lower_bound, chunk_impl, dtype):
    """(q, k, v, g, beta, kwargs) the chunk backward gets for the fused
    prologue ``fuses``, from a forward's saved state ``s``; token-major for a
    kernel."""
    B, T = s["q"].shape[:2]
    lr = s["lr"].reshape(B * T, -1)
    kw = {}
    q_in, k_in, g_in, b_in = s["q"], s["k"], s["g"], s["beta"]
    if "qk_l2norm" in fuses:
        q_in, k_in = s["q_pre"], s["k_pre"]
        kw["use_qk_l2norm_in_kernel"] = True
    if "gate" in fuses:
        g_in = s["raw_g"]
        if g_in is None:
            g_in = _proj_gemm(lr, w.f_b, "fwd", torch.float32, name="f_b")
            g_in = g_in.view_as(s["g"])
        kw.update(use_gate_in_kernel=True, A_log=w.A_log[:H], dt_bias=w.dt_bias)
        if lower_bound is not None:
            kw["lower_bound"] = lower_bound
    if "beta_sigmoid" in fuses:
        b_in = s["braw"]
        kw["use_beta_sigmoid_in_kernel"] = True
    if chunk_impl is not None and _has_param(chunk_impl, "compute_dtype"):
        kw["compute_dtype"] = _compute_dtype_of(dtype)
    v_in = s["v"]
    if chunk_impl is not None:
        # every chunk kernel indexes token-major; q/k/v are channel-major views
        q_in, k_in, v_in, g_in, b_in = _token_major(q_in, k_in, v_in, g_in, b_in)
    return q_in, k_in, v_in, g_in, b_in, kw


def _save_chunk_intermediates(x, w, H, chunk_size, plan, lower_bound, cu_seqlens):
    """``chunk_override`` for a ``save_fwd`` forward whose chunk backward takes
    ``fwd_intermediates`` (#8740 FlyDSL): a chunk forward on the backward's own
    inputs that also returns the prepare 10-tuple. The FlyDSL forward
    (``return_intermediates``, #8744; fp32 Aqk/Akk, o bit-identical to the
    plain call) when it is selected and serves the shape, else the Triton
    forward with ``disable_recompute=True`` (bf16 Aqk/Akk). None when neither
    applies."""
    chunk_impl, fuses = plan[0], plan[1]
    on_device = x.is_cuda or _chunk_bwd_impl is not None
    if not (on_device and _accepts(chunk_impl, "fwd_intermediates")):
        return None
    if cu_seqlens is not None and not _accepts(chunk_impl, "cu_seqlens"):
        return None
    cd = _compute_dtype_of(x.dtype)
    flydsl_fwd = triton_fwd = None
    if "flydsl" in _fallback_order(kda_piece_choices()["chunk_fwd"]):
        try:
            from aiter.ops.flydsl.kda_chunk_fwd import chunk_kda_fwd_flydsl

            if _accepts(chunk_kda_fwd_flydsl, "return_intermediates"):
                flydsl_fwd = chunk_kda_fwd_flydsl
            else:
                LAST_PIECES["chunk_fwd_save_flydsl_skipped"] = "no return_intermediates"
        except ImportError as err:
            LAST_PIECES["chunk_fwd_save_flydsl_missing"] = f"ImportError: {err}"
    try:
        from aiter.ops.triton._triton_kernels.kimi_delta_attn.chunk_fwd import (
            chunk_delta_attn_fwd as triton_fwd,
        )
    except ImportError as err:
        LAST_PIECES["chunk_fwd_save_triton_missing"] = f"ImportError: {err}"
    if flydsl_fwd is None and triton_fwd is None:
        return None

    def run(pre):
        q, k, v, g, beta, kw = _chunk_bwd_args(
            pre, w, H, fuses, lower_bound, chunk_impl, x.dtype
        )
        kw.pop("compute_dtype", None)
        q, k, v = _token_major(q, k, v, dtype=cd)
        if cu_seqlens is not None:
            kw["cu_seqlens"] = cu_seqlens
        if flydsl_fwd is not None:
            try:
                o, _, inter = flydsl_fwd(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    pre["scale"],
                    chunk_size,
                    return_intermediates=True,
                    **kw,
                )
                LAST_PIECES["chunk_fwd"] = "flydsl(save intermediates)"
                return o.float(), tuple(inter)
            except (ValueError, NotImplementedError, RuntimeError) as err:
                reason = f"{type(err).__name__}: {err}"
                LAST_PIECES["chunk_fwd_save_flydsl_skipped"] = reason
                _log_fallback("chunk_fwd(save intermediates)", "flydsl", reason)
                if triton_fwd is None:
                    raise
        o, _, gc, Aqk, Akk, wt, u, qg, kg = triton_fwd(
            q,
            k,
            v,
            g,
            beta,
            pre["scale"],
            None,
            False,
            chunk_size=chunk_size,
            safe_gate=lower_bound is not None,
            disable_recompute=True,
            **kw,
        )
        LAST_PIECES["chunk_fwd"] = "triton(save intermediates)"
        LAST_PIECES["chunk_fwd_note"] = (
            "save_fwd=1 with a chunk backward taking fwd_intermediates: the "
            "forward runs the Triton disable_recompute path for its "
            "intermediates (bf16 Aqk/Akk), since the FlyDSL forward with "
            "return_intermediates is not selected or does not serve this call"
        )
        return o.float(), (None, None, None, gc, wt, u, qg, kg, Aqk, Akk)

    return run


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

    d_wo = _proj_gemm(dyc, s["on"], "wgrad", cd, name="w_o")
    d_on = _proj_gemm(dyc, w.w_o, "dgrad", cd, name="w_o").view_as(s["o"])
    if kda_precision() == "megatron":
        rms_bwd = _TorchOps.gated_rmsnorm_bwd_megatron
        LAST_PIECES["rmsnorm_bwd"] = "torch(megatron rounding)"
    do, dgp, d_norm = rms_bwd(s["o"], s["gp"], w.norm_w, d_on, eps)
    dgp = dgp.reshape(B * T, -1)
    proj = _proj_ops(x, w)
    dx = _proj_gemm(dgp, w.g_proj, "dgrad", cd, ops=proj, name="g_proj")
    d_gproj = _proj_gemm(dgp, xc, "wgrad", cd, ops=proj, name="g_proj")

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
    q_in, k_in, v_in, g_in, b_in, kw = _chunk_bwd_args(
        s, w, H, fuses, lower_bound, chunk_impl, x.dtype
    )
    if chunk_impl is not None:
        do = _token_major(do)[0]
    if s.get("chunk_inter") is not None and _accepts(chunk_impl, "fwd_intermediates"):
        kw["fwd_intermediates"] = s["chunk_inter"]
        LAST_PIECES["chunk_bwd_fwd_intermediates"] = "reused"
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
    if "gate" not in fuses and s["raw_g"] is not None:
        z = s["raw_g"] + w.dt_bias.float().view(H, -1)
        dz, dterm = _TorchProj.gate_bwd_from_z(z, w.A_log, H, dz, lower_bound)
        dz, dterm = dz.reshape(B * T, -1), dterm.reshape(B * T, -1)
        dA_log = dterm.view(-1, *s["g"].shape[-2:]).sum((0, 2))
    elif "gate" not in fuses:
        dz, dterm = _proj_gemm(
            lr,
            w.f_b,
            "fwd",
            cd,
            epilogue="kda_gate_bwd",
            ops=proj,
            name="f_b",
            dt_bias=w.dt_bias,
            A_log=w.A_log,
            num_heads=H,
            dg=dz,
            lower_bound=lower_bound,
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
    # the low-rank f_b side stays fp32 (K = 128, negligible cost)
    d_fb = _proj_gemm(dz, lr, "wgrad", torch.float32, name="f_b")
    dlr = _proj_gemm(dz, w.f_b, "dgrad", torch.float32, name="f_b")
    d_fa = _proj_gemm(dlr, xc, "wgrad", cd, name="f_a")
    dx = dx + _proj_gemm(dlr, w.f_a, "dgrad", cd, name="f_a")
    dx = dx + _proj_gemm(db, w.w_beta, "dgrad", cd, name="w_beta")
    d_wbeta = _proj_gemm(db, xc, "wgrad", cd, name="w_beta")

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
    d_wqkv = _proj_gemm(da, xc, "wgrad", cd, name="qkv")
    dx = dx + _proj_gemm(da, s["w_qkv"], "dgrad", cd, name="qkv")

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
    def forward(
        ctx, x, num_heads, chunk_size, eps, lower_bound, cu_seqlens, first_mb, *ws
    ):
        with _weight_cache_mode(first_mb):
            return _KDALayerFused._forward(
                ctx, x, num_heads, chunk_size, eps, lower_bound, cu_seqlens, *ws
            )

    @staticmethod
    def _forward(ctx, x, num_heads, chunk_size, eps, lower_bound, cu_seqlens, *ws):
        # the backward reuses what this forward built, if it cached anything
        ctx.cache_mode = None if _WEIGHT_CACHE_MODE is None else False
        ctx.save_for_backward(x, *ws)
        ctx.cfg = (num_heads, chunk_size, eps)
        ctx.lower_bound, ctx.cu_seqlens = lower_bound, cu_seqlens
        w = KDALayerWeights.from_tensors(ws)
        ctx.saved_state = None
        if _fusion_on("save_fwd"):
            plan = _fused_plan(x, w, _row_ops(x), lower_bound)
            s = _fused_saved_fwd(
                x,
                w,
                num_heads,
                chunk_size,
                eps,
                plan,
                lower_bound,
                cu_seqlens,
                keep_chunk=True,
            )
            ctx.saved_state = (plan, s)
            B, T, D = x.shape
            y = _proj_gemm(s["on"], w.w_o, "fwd", s["cd"], name="w_o")
            return y.view(B, T, D).to(x.dtype)
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
        with _weight_cache_mode(ctx.cache_mode):
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
        return (dx, None, None, None, None, None, None, *dw.tensors())


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
    is_first_microbatch=None,
):
    """x: [B,T,D] -> y: [B,T,D]. Differentiable for both backends.

    ``is_first_microbatch`` follows Transformer Engine for the derived weights
    the fused backend caches (``cached_weight_cat``): None builds them per
    call, True rebuilds and caches them (first microbatch after an optimizer
    step; it first calls ``invalidate_weight_caches(w)``), False reuses them
    (later microbatches of the same step). ``kda_weight_reuse()`` reports
    which applies during the call's forward and backward.

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
        if is_first_microbatch:
            invalidate_weight_caches(w)
        return _KDALayerFused.apply(
            x,
            num_heads,
            chunk_size,
            eps,
            lower_bound,
            cu_seqlens,
            is_first_microbatch,
            *w.tensors(),
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
    is_first_microbatch=None,
):
    """Backward of one KDA layer.

    Returns (dx, dw) where dw is a KDALayerWeights holding the gradient of every
    weight (q/k/v proj, short conv, F-decay low-rank gate + A_log/dt_bias, beta
    proj, output gate proj, norm weight, output proj). The forward is recomputed.
    ``cu_seqlens`` and ``is_first_microbatch`` as in ``kda_layer_fwd``; weight
    gradients sum over sequences.
    """
    if backend not in KDA_BACKENDS:
        raise ValueError(f"unknown backend {backend!r}, expected one of {KDA_BACKENDS}")
    segments = _segments(x, cu_seqlens)
    if backend == "fused":
        if is_first_microbatch:
            invalidate_weight_caches(w)
        with _weight_cache_mode(is_first_microbatch):
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
