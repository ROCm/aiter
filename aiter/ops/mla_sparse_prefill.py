# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MLA sparse prefill (DeepSeek DSA): one softmax per query token over its rows of a
paged prefix source (``unified_kv``) and of a flat extend source (``kv``), with a
per-head sink in the denominator only.

One public entry per KV layout: the name carries the layout (the model version that
introduced it) and the tensors' dtypes carry the precision, so a new precision of a
layout is a new dtype case, not a new entry.

* :func:`mla_sparse_prefill`     -- unscaled rows (bf16/fp16): opus.
* :func:`mla_sparse_prefill_v4`  -- v4: NoPE with its E8M0 scales in the row, plus bf16
  RoPE (NoPE fp8): asm on gfx1250 with H == 128 (opus if it declines), else opus.
* :func:`mla_sparse_prefill_v41` -- v4.1: rows with separate ``[*, 16]`` E8M0 scales
  (q, kv fp8): opus on gfx950.

``backend`` forces one backend. A configuration no backend serves raises RuntimeError;
a bad argument raises ValueError.
"""

import functools

import torch

from .. import logger
from ..jit.core import compile_ops
from ..jit.utils.asm_guard import require_gfx1250_asm
from ..jit.utils.chip_info import get_cu_num, get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard
from ..utility import dtypes
from .opus.mla import (
    opus_mla_v4_prefill_a8w8_gfx950_fwd,
    opus_mla_v4_prefill_a8w8_gfx1250_fwd,
    opus_mla_v4_prefill_a16w16_gfx950_fwd,
    opus_mla_v4_prefill_a16w16_gfx1250_fwd,
    opus_mla_v41_prefill_a8w8_gfx950_fwd,
)

MD_NAME = "module_mla_sparse_prefill_asm"

# Baked into the asm code object.
_ASM_HEADS = 128


def _check_backend(op: str, backend: str | None, backends: tuple) -> None:
    if backend is not None and backend not in backends:
        raise ValueError(
            f"{op}: backend must be one of {backends} or None, got {backend!r}"
        )


def _out_bf16(t: int, h: int, device, out: torch.Tensor | None) -> torch.Tensor:
    if out is None:
        return torch.empty((t, h, 512), dtype=torch.bfloat16, device=device)
    if out.shape != (t, h, 512) or out.dtype != torch.bfloat16:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={(t, h, 512)} dtype={torch.bfloat16}"
        )
    return out


# ---------------------------------------------------------------------------
# asm (gfx1250)
# ---------------------------------------------------------------------------


# NOTE: ctypes binds positionally off this signature -- the argument order here
# must match `mla_sparse_prefill_fp8_asm_fwd` in
# csrc/py_itfs_cu/asm_mla_sparse_prefill.cu one for one. A silent reorder here
# is a silent wrong-pointer launch, not a compile error.
@compile_ops(MD_NAME, fc_name="mla_sparse_prefill_fp8_asm_fwd", ffi_type="ctypes")
def mla_sparse_prefill_fp8_asm_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _asm_sparse_prefill_v4(
    q_nope,
    q_rope,
    unified_kv_nope,
    unified_kv_rope,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv_nope,
    kv_rope,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
):
    gfx = get_gfx_runtime()
    if gfx != "gfx1250":
        raise RuntimeError(f"mla_sparse_prefill_fp8_asm requires gfx1250, got {gfx}")
    require_gfx1250_asm("mla_sparse_prefill_fp8_asm")
    if attn_sink is None:
        raise RuntimeError("mla_sparse_prefill_fp8_asm requires attn_sink")

    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")

    t, h = q_nope.shape[0], q_nope.shape[1]
    if h != _ASM_HEADS:
        # Hard constraint, not a dispatch miss.
        raise RuntimeError(f"mla_sparse_prefill_fp8_asm requires H == 128, got {h}")
    out = _out_bf16(t, h, q_nope.device, out)

    mla_sparse_prefill_fp8_asm_fwd(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
    )
    return out


def _mla_sparse_prefill_fp8_asm_fake(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_fp8_asm_fake)
def mla_sparse_prefill_fp8_asm(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sparse prefill attention with split fp8 NoPE and bf16 RoPE inputs (asm).

    Signature-compatible with :func:`mla_sparse_prefill_v4` and
    :func:`aiter.ops.pa_sparse_prefill_opus.pa_sparse_prefill_fp8_opus`.

    Args:
      q_nope:            ``[T, H, 512]`` fp8 query without positional encoding.
      q_rope:            ``[T, H, 64]`` bf16 query RoPE encoding part.
      unified_kv_nope:   ``[total_pages, 512]`` fp8 prefix KV NoPE source.
      unified_kv_rope:   ``[total_pages, 64]`` bf16 prefix KV RoPE source.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        sources, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv_nope:           ``[total_tokens, 512]`` fp8 extend KV NoPE source.
      kv_rope:           ``[total_tokens, 64]`` bf16 extend KV RoPE source.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        sources, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias.
      softmax_scale:     float scalar applied to the combined QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output buffer; allocated
        if ``None``.

    Returns:
      ``out`` (``[T, H, 512]`` bf16).
    """
    return _asm_sparse_prefill_v4(
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
    )


# ---------------------------------------------------------------------------
# opus
# ---------------------------------------------------------------------------


def _opus_fwd(gfx: str, op_gfx950, op_gfx1250):
    """gfx950 compiles the kernel from source; gfx1250 loads a prebuilt code object,
    as its kernel needs a scheduler that release toolchains do not ship."""
    if gfx == "gfx1250":
        return op_gfx1250
    if gfx == "gfx950":
        return op_gfx950
    raise RuntimeError(
        f"opus MLA sparse prefill supports gfx950 and gfx1250, got {gfx}"
    )


# Split-K cuts each token's keys over num_kv_splits workgroups and folds them in a second
# launch. Auto takes, among the counts whose workgroups still fit in one wave of CUs (past
# it a split loses), the one that minimizes the keys a workgroup walks plus _KV_SPLIT_COST
# keys for the partials and their reduce. Fit on gfx1250 over topk 256-4096, N 8-768 and
# H 16-128.
_KV_SPLIT_COST = 1280


def _opus_heads_per_wg(gfx: str, h: int) -> int:
    """Heads one workgroup of the v4 kernel that serves ``h`` covers."""
    if gfx == "gfx1250":
        return 16 if h <= 16 else 32 if h <= 32 else 64
    return 16 if h <= 32 else 128


def _kv_splits(
    num_kv_splits: int | None, n: int, h: int, heads_per_wg: int, nnz: int
) -> int:
    """The split count. Auto reads host scalars only (``nnz``: all tokens' rows, from the
    index tensors' sizes), so a captured graph's choice is its shape's."""
    if num_kv_splits is None:
        if n == 0:
            return 1
        wgs = n * -(-h // heads_per_wg)
        keys = nnz / n
        fits = [s for s in (2, 4, 8) if wgs * s <= get_cu_num()]

        def cost(s: int) -> float:
            return keys / s + (_KV_SPLIT_COST if s > 1 else 0)

        return min([1, *fits], key=cost)
    if num_kv_splits not in (1, 2, 4, 8):
        # ValueError, a caller's error; RuntimeError would read as a missing kernel.
        raise ValueError(f"num_kv_splits must be 1, 2, 4 or 8, got {num_kv_splits}")
    return num_kv_splits


def _split_args(n: int, num_kv_splits: int, h: int, device) -> tuple:
    """The opus kernels' trailing split-K arguments, none without split-K: num_splits,
    partial_o [N, S, H, 512], partial_max and partial_sum [N, S, H], fp32."""
    if num_kv_splits == 1:
        return ()
    partial_o = torch.empty(
        n, num_kv_splits, h, 512, dtype=torch.float32, device=device
    )
    partial_max = torch.empty(n, num_kv_splits, h, dtype=torch.float32, device=device)
    return num_kv_splits, partial_o, partial_max, torch.empty_like(partial_max)


def _opus_sparse_prefill(
    q,
    unified_kv,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
    num_kv_splits,
):
    gfx = get_gfx_runtime()
    fwd = _opus_fwd(
        gfx,
        opus_mla_v4_prefill_a16w16_gfx950_fwd,
        opus_mla_v4_prefill_a16w16_gfx1250_fwd,
    )

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"expects fp16/bf16 q, got {q.dtype}")
    if gfx == "gfx1250" and q.dtype != torch.bfloat16:
        raise RuntimeError(
            f"the gfx1250 code object only provides the bf16 variant, got {q.dtype}"
        )
    if unified_kv.dtype != q.dtype:
        raise RuntimeError(
            f"unified_kv dtype mismatch: unified_kv={unified_kv.dtype}, q={q.dtype}"
        )
    if kv.dtype != q.dtype:
        raise RuntimeError(f"kv dtype mismatch: kv={kv.dtype}, q={q.dtype}")
    if unified_kv.size(-1) != kv.size(-1):
        raise RuntimeError(
            f"head_dim mismatch: unified_kv={unified_kv.size(-1)}, kv={kv.size(-1)}"
        )

    if out is None:
        out = torch.empty_like(q)
    elif out.shape != q.shape or out.dtype != q.dtype:
        raise RuntimeError(
            f"out shape/dtype mismatch: got shape={tuple(out.shape)} dtype={out.dtype}, "
            f"expected shape={tuple(q.shape)} dtype={q.dtype}"
        )

    n, h, _ = q.shape
    nnz = kv_indices_prefix.numel() + kv_indices_extend.numel()
    num_kv_splits = _kv_splits(num_kv_splits, n, h, _opus_heads_per_wg(gfx, h), nnz)
    fwd(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
        *_split_args(n, num_kv_splits, h, q.device),
    )
    return out


# NoPE dtype -> (gfx950, gfx1250) kernel.
_OPUS_V4_KERNELS = {
    dtypes.fp8: (
        opus_mla_v4_prefill_a8w8_gfx950_fwd,
        opus_mla_v4_prefill_a8w8_gfx1250_fwd,
    ),
}


def _opus_sparse_prefill_v4(
    q_nope,
    q_rope,
    unified_kv_nope,
    unified_kv_rope,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv_nope,
    kv_rope,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
    num_kv_splits=None,
):
    if q_nope.dtype != unified_kv_nope.dtype or q_nope.dtype != kv_nope.dtype:
        raise RuntimeError(
            f"NoPE dtype mismatch: q_nope={q_nope.dtype}, "
            f"unified_kv_nope={unified_kv_nope.dtype}, kv_nope={kv_nope.dtype}"
        )
    kernels = _OPUS_V4_KERNELS.get(q_nope.dtype)
    if kernels is None:
        raise RuntimeError(
            f"opus v4 sparse prefill supports NoPE dtypes {list(_OPUS_V4_KERNELS)}, "
            f"got {q_nope.dtype}"
        )
    gfx = get_gfx_runtime()
    fwd = _opus_fwd(gfx, *kernels)
    if q_rope.dtype != torch.bfloat16:
        raise RuntimeError(f"q_rope must be bf16, got {q_rope.dtype}")
    n, h, _ = q_nope.shape
    out = _out_bf16(n, h, q_nope.device, out)
    args = (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    )
    nnz = kv_indices_prefix.numel() + kv_indices_extend.numel()
    num_kv_splits = _kv_splits(num_kv_splits, n, h, _opus_heads_per_wg(gfx, h), nnz)
    fwd(
        *args,
        out,
        float(softmax_scale),
        *_split_args(n, num_kv_splits, h, q_nope.device),
    )
    return out


# (q dtype, KV dtype) -> kernel; unified_kv and kv share the KV dtype.
_OPUS_V41_KERNELS = {
    (dtypes.fp8, dtypes.fp8): opus_mla_v41_prefill_a8w8_gfx950_fwd,
}


def _opus_sparse_prefill_v41(
    q,
    q_scale,
    unified_kv,
    unified_kv_scale,
    kv_indices_prefix,
    kv_indptr_prefix,
    kv,
    kv_scale,
    kv_indices_extend,
    kv_indptr_extend,
    attn_sink,
    softmax_scale,
    out,
    num_kv_splits=None,
):
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"mla_sparse_prefill_v41 requires gfx950, got {gfx}")

    if unified_kv.dtype != kv.dtype:
        raise RuntimeError(
            f"KV dtype mismatch: unified_kv={unified_kv.dtype}, kv={kv.dtype}"
        )
    fwd = _OPUS_V41_KERNELS.get((q.dtype, kv.dtype))
    if fwd is None:
        raise RuntimeError(
            f"mla_sparse_prefill_v41 supports (q, kv) dtypes {list(_OPUS_V41_KERNELS)}, "
            f"got ({q.dtype}, {kv.dtype})"
        )
    for name, scale in (
        ("q_scale", q_scale),
        ("unified_kv_scale", unified_kv_scale),
        ("kv_scale", kv_scale),
    ):
        if scale.dtype not in (torch.uint8, dtypes.fp8_e8m0):
            raise RuntimeError(f"{name} must be E8M0 (uint8), got {scale.dtype}")
    n, h, _ = q.shape
    out = _out_bf16(n, h, q.device, out)
    args = (
        q,
        q_scale,
        unified_kv,
        unified_kv_scale,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_scale,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
    )
    nnz = kv_indices_prefix.numel() + kv_indices_extend.numel()
    num_kv_splits = _kv_splits(num_kv_splits, n, h, 64, nnz)
    fwd(*args, out, float(softmax_scale), *_split_args(n, num_kv_splits, h, q.device))
    return out


# ---------------------------------------------------------------------------
# Public entries
# ---------------------------------------------------------------------------


@functools.cache
def _warn_asm_fallback(reason: str) -> None:
    logger.warning(
        f"mla_sparse_prefill_v4: the asm backend declined ({reason}); using opus"
    )


def _mla_sparse_prefill_fake(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_fake)
def mla_sparse_prefill(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """MLA sparse prefill over bf16/fp16 Q/KV (D = 512).

    Args:
      q:                 ``[T, H, D]`` bf16/fp16 query.
      unified_kv:        ``[total_pages, D]`` prefix source, same dtype as ``q``.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into
        ``unified_kv``, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv:                ``[total_tokens, D]`` extend source, same dtype and row
        stride as ``unified_kv``.
      kv_indices_extend: ``[total_extend]`` int32 row indices into ``kv``,
        concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias, or ``None``
        for no sink.
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, D]`` output (``q``'s dtype);
        allocated if ``None``.
      num_kv_splits:     split each token's keys over this many workgroups and
        fold them in a second launch: 1, 2, 4 or 8. ``None`` picks it from the
        token, head and key counts.
      backend:           ``"opus"`` or ``None``.

    Returns:
      ``out``.
    """
    _check_backend("mla_sparse_prefill", backend, ("opus",))
    return _opus_sparse_prefill(
        q,
        unified_kv,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
        num_kv_splits,
    )


def _mla_sparse_prefill_v4_fake(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q_nope.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q_nope.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_v4_fake)
def mla_sparse_prefill_v4(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """MLA sparse prefill over the v4 layout: NoPE rows holding their own E8M0
    scales, plus bf16 RoPE. Supported NoPE dtypes: fp8.

    Args:
      q_nope:            ``[T, H, 512]`` NoPE (fp8: 448 values, 14 E8M0 scales, pad).
      q_rope:            ``[T, H, 64]`` bf16.
      unified_kv_nope:   ``[total_pages, 512]`` prefix NoPE source, as ``q_nope``.
      unified_kv_rope:   ``[total_pages, 64]`` bf16 prefix RoPE source.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        sources, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv_nope:           ``[total_tokens, 512]`` extend NoPE source, as ``q_nope``.
      kv_rope:           ``[total_tokens, 64]`` bf16 extend RoPE source.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        sources, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias, or ``None``
        for no sink (opus; ``None`` makes the default backend opus).
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output; allocated if ``None``.
      num_kv_splits:     as in :func:`mla_sparse_prefill` (opus only).
      backend:           ``"asm"``, ``"opus"`` or ``None``.

    Returns:
      ``out``.
    """
    _check_backend("mla_sparse_prefill_v4", backend, ("asm", "opus"))
    args = (
        q_nope,
        q_rope,
        unified_kv_nope,
        unified_kv_rope,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_nope,
        kv_rope,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
    )
    unsplit = num_kv_splits in (None, 1)
    if backend == "asm":
        if not unsplit:
            raise RuntimeError("the asm backend has no split-K")
        return _asm_sparse_prefill_v4(*args)
    if (
        backend is None
        and unsplit
        and attn_sink is not None
        and get_gfx_runtime() == "gfx1250"
        and q_nope.size(1) == _ASM_HEADS
    ):
        try:
            return _asm_sparse_prefill_v4(*args)
        except RuntimeError as e:  # asm is stricter about dtypes and shapes than opus
            _warn_asm_fallback(str(e))
    return _opus_sparse_prefill_v4(*args, num_kv_splits)


def _mla_sparse_prefill_v41_fake(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    if out is not None:
        return out
    t, h, _ = q.shape
    return torch.empty((t, h, 512), dtype=torch.bfloat16, device=q.device)


@torch_compile_guard(mutates_args=["out"], gen_fake=_mla_sparse_prefill_v41_fake)
def mla_sparse_prefill_v41(
    q: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor | None,
    softmax_scale: float,
    out: torch.Tensor | None = None,
    num_kv_splits: int | None = None,
    backend: str | None = None,
) -> torch.Tensor:
    """MLA v4.1 sparse prefill: rows of 512 values with separate E8M0 scales (uint8
    or fp8_e8m0, one per 32 values). Same argument order as
    :func:`mla_sparse_prefill_v4`, with the RoPE tensors replaced by the scales.
    Supported (q, kv) dtypes: (fp8, fp8), on gfx950.

    Args:
      q:                 ``[T, H, 512]`` query.
      q_scale:           ``[T, H, 16]`` scales of ``q``.
      unified_kv:        ``[total_pages, 512]`` prefix KV source.
      unified_kv_scale:  ``[total_pages, 16]`` scales of ``unified_kv``.
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into the prefix
        source, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv:                ``[total_tokens, 512]`` extend KV source, with the
        row stride of ``unified_kv``.
      kv_scale:          ``[total_tokens, 16]`` scales of ``kv``, with the row
        stride of ``unified_kv_scale``.
      kv_indices_extend: ``[total_extend]`` int32 row indices into the extend
        source, concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` fp32 per-head softmax-denom bias, or ``None``
        for no sink.
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, 512]`` bf16 output buffer; allocated
        if ``None``.
      num_kv_splits:     as in :func:`mla_sparse_prefill` (opus only).
      backend:           ``"opus"`` or ``None``.

    Returns:
      ``out`` (``[T, H, 512]`` bf16).
    """
    _check_backend("mla_sparse_prefill_v41", backend, ("opus",))
    return _opus_sparse_prefill_v41(
        q,
        q_scale,
        unified_kv,
        unified_kv_scale,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv,
        kv_scale,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        softmax_scale,
        out,
        num_kv_splits,
    )


__all__ = [
    "mla_sparse_prefill",
    "mla_sparse_prefill_fp8_asm",
    "mla_sparse_prefill_fp8_asm_fwd",
    "mla_sparse_prefill_v4",
    "mla_sparse_prefill_v41",
    "opus_mla_v41_prefill_a8w8_gfx950_fwd",
]
