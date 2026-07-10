# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level FlyDSL Flash Attention APIs (gfx1201 / RDNA4).

Public entry point ``flydsl_flash_attn_func`` wraps the FlyDSL
``flash_attn_func_gfx1201`` (bf16/f16) and ``flash_attn_func_fp8_gfx1201``
(fp8) kernels with:
  - BSHD ([B, S, H, D]) input/output convention to match the upstream
    flash-attention layout.
  - Shape-driven ``(BLOCK_M, BLOCK_N)`` tile selection (``_pick_tiles``) and a
    per-shape build cache.
  - Automatic seq_len padding to ``BLOCK_M`` for in-bounds loads; the real
    (pre-pad) length is passed to the kernel, which bounds the non-causal KV
    loop at it. Unaligned non-causal lengths are handled by the kernel's
    per-column tail mask.
  - Self- and cross-attention: when ``seqlen_k != seqlen_q`` the ``cross_attn``
    build is used so Q and K/V address on independent lengths.
  - FP8 (per-tensor): pass ``q_descale/k_descale/v_descale`` (1-element fp32
    device tensors) together with fp8 ``q/k/v`` to route to the fp8 kernel;
    the output is bf16. Naming mirrors ``flash_attn_fp8_pertensor_func``
    in ``aiter/ops/mha.py``. Both self- and cross-attention are supported for
    fp8 (though for cross-attn the K/V are typically small, so quantizing them
    may not pay off in practice). Use ``flydsl_fp8_quant`` to produce the fp8
    q/k/v + descales from bf16/f16 inputs, with optional Hadamard rotation of
    Q/K -- rotation lives in that producer step, never inside the kernel.
"""

from __future__ import annotations

from functools import lru_cache

import torch
import torch.nn.functional as F

from .kernels.flash_attn_func_gfx1201 import build_flash_attn_func_module
from .kernels.flash_attn_func_fp8_gfx1201 import (
    build_flash_attn_func_module as build_flash_attn_fp8_func_module,
)
from .kernels.fmha_gfx1250.fmha_kernel import flash_attn_varlen_d192_gfx1250

__all__ = [
    "flydsl_flash_attn_func",
    "flydsl_flash_attn_varlen_func",
    "flydsl_fp8_quant",
]


# FP8 input dtype accepted by the fp8 kernel (e4m3, per-tensor descale).
_FP8_DTYPES = (torch.float8_e4m3fn,)


# ---------------------------------------------------------------------------
# FP8 quantization (+ optional Hadamard rotation) for the fp8 attention path.
#
# This is the producer-side preprocessing that turns bf16/f16 q/k/v into the
# fp8 q/k/v + per-tensor descales that ``flydsl_flash_attn_func`` consumes. The
# rotation lives HERE, not in the attention kernel: per-tensor fp8 scaling needs
# the global amax of the *rotated* tensor before the kernel launches, and the
# kernel only ever sees already-fp8 q/k/v.
# ---------------------------------------------------------------------------
try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover - triton is normally present
    _HAS_TRITON = False

_FP8_QUANT_DTYPE = torch.float8_e4m3fn
_FP8_QUANT_MAX = 448.0
_ROT_ROWS = 64  # rows per program for the fused rotate+quant kernels
_HADAMARD_CACHE: dict = {}


def _hadamard_matrix(head_dim: int, device, dtype):
    """Normalized ``head_dim x head_dim`` Hadamard (orthonormal), cached per
    ``(head_dim, device, dtype)``. Prefers aiter's ``create_hadamard_matrix``;
    falls back to a local Sylvester construction. Returns ``None`` when head_dim
    is not a power of two (rotation is then skipped)."""
    if head_dim & (head_dim - 1):
        return None
    key = (head_dim, device, dtype)
    R = _HADAMARD_CACHE.get(key)
    if R is not None:
        return R
    try:
        try:
            from aiter.ops.triton._triton_kernels.attention.fav3_sage_attention_mxfp4 import (
                create_hadamard_matrix,
            )
        except ImportError:
            from aiter.ops.triton.quant.sage_attention_quant_wrappers import (
                create_hadamard_matrix,
            )
        R = (create_hadamard_matrix(head_dim, dtype=dtype) / (head_dim**0.5)).to(device)
    except ImportError:
        H = torch.ones((1, 1), dtype=torch.float32, device=device)
        while H.shape[0] < head_dim:
            H = torch.cat([torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)], dim=0)
        R = (H / (head_dim**0.5)).to(dtype)
    _HADAMARD_CACHE[key] = R
    return R


def _rotate_bf16(x: torch.Tensor, R: torch.Tensor | None) -> torch.Tensor:
    """Full-head Hadamard rotation along head_dim (no-op if ``R`` is None)."""
    return x if R is None else torch.matmul(x, R.to(x.dtype))


def _quant_one_pertensor(x: torch.Tensor, R: torch.Tensor | None):
    """Torch rotate (optional) + per-tensor fp8 quant. Fallback for cross-attn,
    non-power-of-2 head_dim, rotation-off, or when triton is unavailable."""
    x = _rotate_bf16(x, R)
    amax = x.abs().max().to(torch.float32)
    s = (amax / _FP8_QUANT_MAX).clamp(min=1e-12).reshape(1)
    xq = (
        (x.to(torch.float32) / s).clamp(-_FP8_QUANT_MAX, _FP8_QUANT_MAX).to(_FP8_QUANT_DTYPE)
    )
    return xq, s


if _HAS_TRITON:

    @triton.jit
    def _rot_amax3(q, k, v, R, aq, ak, av, M, D: tl.constexpr, BLOCK_ROWS: tl.constexpr):
        """Fused Hadamard-rotate (Q/K only) + per-tensor amax over q/k/v. Row-
        tiled: each program rotates a ``(BLOCK_ROWS, D)`` tile in-register via
        ``tl.dot(x, R)`` on the matrix cores and atomic-max's its abs-max into
        that tensor's scalar. V is not rotated (compute-and-select keeps one
        branch-free launch). Rotated Q/K never round-trip to HBM."""
        pid = tl.program_id(0)
        rb = tl.cdiv(M, BLOCK_ROWS)
        t = pid // rb
        b = pid % rb
        rows = b * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        rmask = rows < M
        p = tl.where(t == 0, q, tl.where(t == 1, k, v))
        offs = rows[:, None] * D + tl.arange(0, D)[None, :]
        x = tl.load(p + offs, mask=rmask[:, None], other=0.0)
        Rm = tl.load(R + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :])
        xr = tl.dot(x, Rm)
        x = tl.where(t < 2, xr, x.to(tl.float32))
        val = tl.max(tl.abs(x))
        o = tl.where(t == 0, aq, tl.where(t == 1, ak, av))
        tl.atomic_max(o, val)

    @triton.jit
    def _rot_sc3(
        q, k, v, R, yq, yk, yv, sq, sk, sv, M, D: tl.constexpr, BLOCK_ROWS: tl.constexpr
    ):
        """Fused Hadamard-rotate (Q/K only) + scale/clamp/cast to fp8, row-tiled
        to match ``_rot_amax3`` (rotation recomputed, not reloaded from HBM)."""
        pid = tl.program_id(0)
        rb = tl.cdiv(M, BLOCK_ROWS)
        t = pid // rb
        b = pid % rb
        rows = b * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
        rmask = rows < M
        p = tl.where(t == 0, q, tl.where(t == 1, k, v))
        yp = tl.where(t == 0, yq, tl.where(t == 1, yk, yv))
        sp = tl.where(t == 0, sq, tl.where(t == 1, sk, sv))
        offs = rows[:, None] * D + tl.arange(0, D)[None, :]
        x = tl.load(p + offs, mask=rmask[:, None], other=0.0)
        Rm = tl.load(R + tl.arange(0, D)[:, None] * D + tl.arange(0, D)[None, :])
        xr = tl.dot(x, Rm)
        x = tl.where(t < 2, xr, x.to(tl.float32))
        x = x * (1.0 / tl.load(sp))
        # fp8 e4m3 max (448.0) inlined; triton @jit can't read module globals.
        x = tl.minimum(tl.maximum(x, -448.0), 448.0)
        tl.store(yp + offs, x.to(yp.dtype.element_ty), mask=rmask[:, None])


def _quant_pertensor_flydsl(x: torch.Tensor, rotate: bool):
    """Per-tensor rotate(optional)+fp8 quant of a single tensor via the fully
    FlyDSL 2-pass kernel. head_dim==128 only."""
    from .kernels.fp8_quant_gfx1201 import flydsl_fp8_pertensor_quant

    return flydsl_fp8_pertensor_quant(x, rotate=rotate)


def flydsl_fp8_quant(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    rotation: bool = True,
    backend: str = "flydsl",
):
    """Per-tensor fp8 (e4m3) quantization of bf16/f16 ``q/k/v`` for the flydsl
    fp8 attention path, with optional Hadamard rotation of Q/K.

    Returns ``(q_fp8, k_fp8, v_fp8, q_descale, k_descale, v_descale)`` ready to
    hand straight to ``flydsl_flash_attn_func(..., q_descale=, k_descale=,
    v_descale=)``. Descales are 1-element fp32 device tensors following the
    wrapper contract: ``real = fp8 * descale``.

    Rotation (recommended, default on) applies an orthonormal Hadamard ``R`` to Q
    and K only. It is QK-preserving -- ``(Q@R)(K@R)^T == Q@K^T`` -- so attention
    scores are unchanged, while spreading channel outliers so per-tensor e4m3
    clamps far less (the benefit shows on outlier-heavy real activations).

    ``backend`` selects the producer: ``"flydsl"`` (default) runs the fully-FlyDSL
    2-pass rotate+quant kernels (in-register FWHT, no Triton); ``"triton"`` runs
    the fused Triton passes; ``"torch"`` runs the reference. All three keep rotated
    Q/K off HBM (2 reads + 0.5 write). The FlyDSL path requires head_dim==128
    (per-tensor, so q/k/v may differ in size, e.g. cross-attention); it silently
    falls back to Triton/torch for other head_dims. The Triton fused path
    additionally requires same-size q/k/v.
    """
    head_dim = q.shape[-1]
    same_size = q.numel() == k.numel() == v.numel()
    be = backend.lower()

    # Fully-FlyDSL per-tensor path (no Triton). Each tensor is quantized by its
    # own independent launch, so q/k/v need not be the same size (cross-attention
    # is fine); only head_dim==128 (VEC=4) is required by the MVP kernel. Uses the
    # in-register FWHT, so it needs no host-side Hadamard matrix.
    if be == "flydsl" and head_dim == 128:
        q8, sq = _quant_pertensor_flydsl(q, rotate=rotation)
        k8, sk = _quant_pertensor_flydsl(k, rotate=rotation)
        v8, sv = _quant_pertensor_flydsl(v, rotate=False)
        return q8, k8, v8, sq, sk, sv

    # Triton/torch fallbacks rotate with an explicit host-side Hadamard matrix.
    R = _hadamard_matrix(head_dim, q.device, torch.bfloat16) if rotation else None
    if be == "torch" or not (_HAS_TRITON and R is not None and same_size):
        q8, sq = _quant_one_pertensor(q, R)
        k8, sk = _quant_one_pertensor(k, R)
        v8, sv = _quant_one_pertensor(v, None)
        return q8, k8, v8, sq, sk, sv

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    M = q.numel() // head_dim
    q2, k2, v2 = q.view(M, head_dim), k.view(M, head_dim), v.view(M, head_dim)
    grid = (3 * triton.cdiv(M, _ROT_ROWS),)
    aq = torch.zeros(1, dtype=torch.float32, device=q.device)
    ak = torch.zeros(1, dtype=torch.float32, device=q.device)
    av = torch.zeros(1, dtype=torch.float32, device=q.device)
    _rot_amax3[grid](q2, k2, v2, R, aq, ak, av, M, D=head_dim, BLOCK_ROWS=_ROT_ROWS)
    sq = (aq / _FP8_QUANT_MAX).clamp(min=1e-12)
    sk = (ak / _FP8_QUANT_MAX).clamp(min=1e-12)
    sv = (av / _FP8_QUANT_MAX).clamp(min=1e-12)
    q8 = torch.empty_like(q, dtype=_FP8_QUANT_DTYPE)
    k8 = torch.empty_like(k, dtype=_FP8_QUANT_DTYPE)
    v8 = torch.empty_like(v, dtype=_FP8_QUANT_DTYPE)
    _rot_sc3[grid](
        q2, k2, v2, R,
        q8.view(M, head_dim), k8.view(M, head_dim), v8.view(M, head_dim),
        sq, sk, sv, M, D=head_dim, BLOCK_ROWS=_ROT_ROWS,
    )
    return q8, k8, v8, sq, sk, sv


def _torch_dtype_to_str(dtype: torch.dtype) -> str:
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "f16"
    raise ValueError(f"flydsl_flash_attn_func only supports bf16/f16, got {dtype!r}")


def _pick_tiles(seq_len: int, head_dim: int) -> tuple[int, int]:
    """Shape-driven (BLOCK_M, BLOCK_N) selection (swept on gfx1201).

    head_dim <= 64: BLOCK_M=128 fills the CUs where 256 starves them; BLOCK_N=64.
    head_dim=128, long S (>=1280): BLOCK_M=256 quarters the q-tile count so each
      K/V stream is re-read from HBM far less often (dominant cost at long seq).
    head_dim=128, short S (<1280): BLOCK_M=256 underfills the CUs -> BLOCK_M=128,
      BLOCK_N=32.
    """
    if head_dim <= 64:
        return 128, 64
    if seq_len < 1280:
        return 128, 32
    return 256, 64


@lru_cache(maxsize=64)
def _get_bf16_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    dtype_str: str,
    waves_per_eu: int,
    daz: bool,
    block_m: int,
    block_n: int,
    tail_mask: bool,
    cross_attn: bool,
    sm_scale: float | None,
):
    return build_flash_attn_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str=dtype_str,
        sm_scale=sm_scale,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        block_n=block_n,
        tail_mask=tail_mask,
        cross_attn=cross_attn,
    )


@lru_cache(maxsize=64)
def _get_fp8_kernel(
    num_heads: int,
    head_dim: int,
    causal: bool,
    waves_per_eu: int,
    daz: bool,
    block_m: int,
    block_n: int,
    tail_mask: bool,
    cross_attn: bool,
    sm_scale: float | None,
):
    # Output/compute dtype is bf16; fp8 is the Q/K/V HBM width only.
    return build_flash_attn_fp8_func_module(
        num_heads=num_heads,
        head_dim=head_dim,
        causal=causal,
        dtype_str="bf16",
        sm_scale=sm_scale,
        waves_per_eu=waves_per_eu,
        daz=daz,
        block_m=block_m,
        block_n=block_n,
        tail_mask=tail_mask,
        cross_attn=cross_attn,
    )


def _pad_seq(t: torch.Tensor, pad: int) -> torch.Tensor:
    # BSHD: seq is dim 1 (last dim head_dim). F.pad counts from the last dim, so
    # (D_left, D_right, H_left, H_right, S_left, S_right).
    t = t.contiguous()
    if pad == 0:
        return t
    return F.pad(t, (0, 0, 0, 0, 0, pad))


def flydsl_flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    waves_per_eu: int = 2,
    daz: bool = True,
    stream: torch.cuda.Stream | None = None,
    # New optional params are appended after the original signature
    # (q, k, v, causal, waves_per_eu, daz, stream) so existing positional and
    # keyword callers keep working unchanged.
    softmax_scale: float | None = None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run FlyDSL Flash Attention on RDNA4 (gfx1201).

    Supports bf16/f16 self- and cross-attention, and a per-tensor fp8 self-attn
    fast path. Tiles are chosen from the shape; seq_len is padded internally.

    Args:
        q: ``[batch, seqlen_q, num_heads, head_dim]`` (BSHD).
        k, v: ``[batch, seqlen_k, num_heads, head_dim]``. ``seqlen_k`` may differ
            from ``seqlen_q`` (cross-attention). ``k`` and ``v`` must share shape.
        causal: apply causal masking when ``True``.
        softmax_scale: QK^T scale; defaults to ``1/sqrt(head_dim)``. Baked into
            the compiled kernel, so non-default values compile a new variant.
        q_descale, k_descale, v_descale: per-tensor fp8 descales as 1-element
            fp32 device tensors. Required (and only used) when ``q/k/v`` are fp8;
            selects the fp8 kernel. Naming matches
            ``flash_attn_fp8_pertensor_func``.
        out: optional preallocated output buffer ``[batch, seqlen_q, num_heads,
            head_dim]``. bf16 for the fp8 path, else ``q.dtype``.
        waves_per_eu: kernel occupancy hint.
        daz: enable denormals-are-zero.
        stream: optional CUDA/HIP stream; defaults to the current stream.

    Returns:
        Output tensor ``[batch, seqlen_q, num_heads, head_dim]``. Same dtype as
        ``q`` for bf16/f16; bf16 for the fp8 path.

    Raises:
        ValueError: on incompatible shapes/dtypes/devices, unmet ``head_dim``
            constraints, or missing fp8 descales.
    """
    if not (q.is_cuda and k.is_cuda and v.is_cuda):
        raise ValueError("flydsl_flash_attn_func requires CUDA/HIP tensors")
    if not (q.device == k.device == v.device):
        raise ValueError(
            "q/k/v must reside on the same device, got "
            f"q={q.device} k={k.device} v={v.device}"
        )
    try:
        arch = torch.cuda.get_device_properties(q.device.index).gcnArchName
    except Exception:
        arch = ""
    arch_base = arch.lower().split(":")[0] if arch else ""
    if not arch_base.startswith("gfx1201"):
        raise ValueError(f"flydsl_flash_attn_func requires gfx1201, got {arch!r}")
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            "expected 4D BSHD tensors, got ranks "
            f"q={q.dim()} k={k.dim()} v={v.dim()}"
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k/v must share shape, got k={tuple(k.shape)} v={tuple(v.shape)}"
        )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(f"q/k/v dtype must match: {q.dtype}/{k.dtype}/{v.dtype}")
    if q.shape[0] != k.shape[0] or q.shape[2] != k.shape[2] or q.shape[3] != k.shape[3]:
        raise ValueError(
            "q/k must share batch, num_heads and head_dim, got "
            f"q={tuple(q.shape)} k={tuple(k.shape)}"
        )

    batch, seq_q_real, num_heads, head_dim = q.shape
    seq_kv_real = k.shape[1]
    is_cross = seq_kv_real != seq_q_real
    is_fp8 = q.dtype in _FP8_DTYPES

    if head_dim < 64 or head_dim % 32 != 0:
        raise ValueError(
            f"kernel requires head_dim >= 64 and head_dim % 32 == 0, got {head_dim}"
        )

    block_m, block_n = _pick_tiles(seq_q_real, head_dim)

    if is_fp8:
        if q_descale is None or k_descale is None or v_descale is None:
            raise ValueError(
                "flydsl_flash_attn_func: fp8 inputs require q_descale, k_descale "
                "and v_descale (1-element fp32 device tensors)."
            )
        out_dtype = torch.bfloat16
    else:
        _torch_dtype_to_str(q.dtype)  # validate bf16/f16
        out_dtype = q.dtype

    # Non-causal, non-BLOCK_N-aligned lengths need the kernel's per-column tail
    # mask so padded K/V columns do not leak exp(0)=1 into the softmax. Causal
    # already excludes padded (future) columns.
    kv_len_for_mask = seq_kv_real if is_cross else seq_q_real
    tail_mask = (not causal) and (kv_len_for_mask % block_n != 0)

    seq_q_pad = ((seq_q_real + block_m - 1) // block_m) * block_m
    seq_kv_pad = ((seq_kv_real + block_m - 1) // block_m) * block_m

    q_p = _pad_seq(q, seq_q_pad - seq_q_real)
    k_p = _pad_seq(k, seq_kv_pad - seq_kv_real)
    v_p = _pad_seq(v, seq_kv_pad - seq_kv_real)
    o_p = torch.empty(
        (batch, seq_q_pad, num_heads, head_dim), dtype=out_dtype, device=q.device
    )

    # Wrap kernel build + launch in q.device context so multi-GPU callers
    # whose current device differs from q.device get the kernel compiled
    # and launched on the right device/stream.
    with torch.cuda.device(q.device.index):
        launch_stream = (
            torch.cuda.current_stream(q.device) if stream is None else stream
        )
        if launch_stream.device != q.device:
            raise ValueError(
                f"`stream` must be on {q.device}, got {launch_stream.device}"
            )
        # Kernel length args: Q tiles on its padded length (seq_q_pad); the KV loop
        # is bounded at the KV real length and addresses on the KV padded length.
        # For self-attn seq_kv_* collapse to seq_q_*, so one launch covers both
        # self- and cross-attention for each dtype.
        if is_fp8:
            exe = _get_fp8_kernel(
                num_heads=num_heads,
                head_dim=head_dim,
                causal=causal,
                waves_per_eu=waves_per_eu,
                daz=daz,
                block_m=block_m,
                block_n=block_n,
                tail_mask=tail_mask,
                cross_attn=is_cross,
                sm_scale=softmax_scale,
            )
            exe(
                q_p.reshape(-1),
                k_p.reshape(-1),
                v_p.reshape(-1),
                o_p.reshape(-1),
                batch,
                seq_q_pad,
                seq_kv_real,
                seq_kv_pad,
                q_descale,
                k_descale,
                v_descale,
                stream=launch_stream,
            )
        else:
            exe = _get_bf16_kernel(
                num_heads=num_heads,
                head_dim=head_dim,
                causal=causal,
                dtype_str=_torch_dtype_to_str(q.dtype),
                waves_per_eu=waves_per_eu,
                daz=daz,
                block_m=block_m,
                block_n=block_n,
                tail_mask=tail_mask,
                cross_attn=is_cross,
                sm_scale=softmax_scale,
            )
            exe(
                q_p.reshape(-1),
                k_p.reshape(-1),
                v_p.reshape(-1),
                o_p.reshape(-1),
                batch,
                seq_q_pad,
                seq_kv_real,
                seq_kv_pad,
                stream=launch_stream,
            )

    result = o_p[:, :seq_q_real, :, :] if seq_q_pad != seq_q_real else o_p
    if out is not None:
        expected = (batch, seq_q_real, num_heads, head_dim)
        if tuple(out.shape) != expected:
            raise ValueError(
                f"`out` must have shape {expected}, got {tuple(out.shape)}"
            )
        if out.dtype != out_dtype:
            raise ValueError(f"`out` must have dtype {out_dtype}, got {out.dtype}")
        out.copy_(result)
        return out
    return result.contiguous()


def flydsl_flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float | None = None,
    causal: bool = False,
    return_lse: bool = False,
    dropout_p: float = 0.0,
    window_size=(-1, -1),
    bias=None,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    out=None,
    sink=None,
):
    """FlyDSL MHA forward, varlen THD layout.

    Returns the result if FlyDSL can handle this configuration,
    otherwise returns None so the caller falls through to Triton/CK.
    """
    from ...jit.utils.chip_info import get_gfx

    # FlyDSL handles only plain MHA. Any unsupported feature (bias, alibi, sink,
    # dropout, sliding window, paging, probs/deterministic) falls through to
    # CK/Triton instead of being silently dropped.
    supported = (
        get_gfx() == "gfx1250"
        and q.shape[-1] == 192
        and v.shape[-1] == 128
        and q.dtype == torch.bfloat16
        and dropout_p == 0.0
        and tuple(window_size[:2]) == (-1, -1)
        and block_table is None
        and bias is None
        and alibi_slopes is None
        and sink is None
        and not deterministic
        and not return_attn_probs
    )
    if not supported:
        return None

    # gfx1250 — varlen THD, D_qk=192 D_v=128, bf16
    if out is None:
        out = torch.empty_like(q[:, :, : v.shape[-1]])
    return flash_attn_varlen_d192_gfx1250(
        q,
        k,
        v,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=softmax_scale,
        causal=causal,
        out=out,
        return_lse=return_lse,
    )
