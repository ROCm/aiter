# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""OPUS-based sparse paged prefill attention for DeepSeek-V4 on gfx950.

Two-region sparse scaled-dot-product attention over a paged prefix source
(``unified_kv``) and a flat per-fwd extend source (``kv``), with a per-head
softmax-denominator sink. The two regions share a single online-softmax
accumulator, making the order region-invariant.

The user-facing entry is :func:`pa_sparse_prefill_opus`; it forwards
to the JIT-compiled HIP kernel via
:func:`pa_sparse_prefill_opus_fwd`.

The kernel currently only compiles a single configuration:

* Head dim ``D == 512``.
* dtype ``bf16`` or ``fp16`` for Q/K/V/O; ``attn_sink`` is ``fp32``.
* Every entry in ``kv_indices_prefix`` / ``kv_indices_extend`` must be a
  valid row index into ``unified_kv`` / ``kv`` respectively. Empty CSR rows
  (``kv_indptr[i] == kv_indptr[i+1]``) are allowed.

See ``aiter/csrc/include/pa_sparse_prefill_opus.h`` for the C++ API.
"""

import torch
from typing import Optional

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx_runtime
from ..jit.utils.torch_guard import torch_compile_guard

MD_NAME = "module_pa_sparse_prefill_opus"


@compile_ops("module_pa_sparse_prefill_opus", develop=True)
def pa_sparse_prefill_opus_fwd(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def _pa_sparse_prefill_opus_fake(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return out if out is not None else torch.empty_like(q)


@torch_compile_guard(mutates_args=["out"], gen_fake=_pa_sparse_prefill_opus_fake)
def pa_sparse_prefill_opus(
    q: torch.Tensor,
    unified_kv: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Sparse prefill attention over two KV sources (paged ``unified_kv`` +
    flat per-fwd ``kv``), backed by the OPUS gfx950 HIP kernel.

    The trailing ``out`` keyword is an aiter-only convenience for callers
    that want to reuse a pre-allocated output buffer; pass ``None`` (the
    default) to have one allocated for you.

    Args:
      q:                 ``[T, H, D]`` bf16/fp16 query (T == N tokens).
      unified_kv:        ``[total_pages, D]`` prefix source (paged history).
      kv_indices_prefix: ``[total_prefix]`` int32 row indices into
        ``unified_kv``, concatenated per token.
      kv_indptr_prefix:  ``[T+1]`` int32 CSR row pointers.
      kv:                ``[total_tokens, D]`` extend source (current fwd's
        just-computed K).
      kv_indices_extend: ``[total_extend]`` int32 row indices into ``kv``,
        concatenated per token.
      kv_indptr_extend:  ``[T+1]`` int32 CSR row pointers.
      attn_sink:         ``[H]`` per-head softmax-denom bias.
      softmax_scale:     float scalar applied to the QK^T scores.
      out:               Optional ``[T, H, D]`` output buffer; allocated if
        ``None``.

    Returns:
      ``out`` (``[T, H, D]`` same dtype as ``q``).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_sparse_prefill_opus requires gfx950, got {gfx}")

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise RuntimeError(f"pa_sparse_prefill_opus expects fp16/bf16 q, got {q.dtype}")
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

    pa_sparse_prefill_opus_fwd(
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
    )
    return out


# ---------------------------------------------------------------------------
# FP8 (DeepSeek-V4 / asm-v4 layout) variant.
#   head dim 512 = NOPE(448, fp8 e4m3, per-64-tile fp32 scale) + ROPE(64, bf16),
#   for both Q and KV. First implementation dequantizes to bf16 scratch (via a
#   standalone device kernel) then runs the bf16 attention kernel.
# ---------------------------------------------------------------------------

_FP8_NOPE = 448
_FP8_ROPE = 64
_FP8_FULL = 512
_FP8_NUM_TILES = 7


@compile_ops("module_pa_sparse_prefill_opus", develop=True)
def pa_sparse_prefill_opus_fp8_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    q_bf16: torch.Tensor,
    unified_kv_bf16: torch.Tensor,
    kv_bf16: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def pa_sparse_prefill_opus_fp8(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """FP8 (v4-layout) sparse prefill attention.

    Q and KV are stored as NOPE(448, fp8 e4m3, per-64-tile fp32 scale) +
    ROPE(64, bf16). Output is bf16 ``[N, H, 512]``. The bf16 dequant scratch
    is allocated here and freed when it goes out of scope.

    Args mirror :func:`pa_sparse_prefill_opus` but with each q/unified_kv/kv
    split into ``*_nope`` (fp8 ``[..., 448]``), ``*_rope`` (bf16 ``[..., 64]``)
    and ``*_scale`` (fp32 ``[..., 7]``).
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(f"pa_sparse_prefill_opus_fp8 requires gfx950, got {gfx}")

    if q_nope.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"q_nope must be fp8 e4m3fn, got {q_nope.dtype}")
    if q_nope.size(-1) != _FP8_NOPE or q_rope.size(-1) != _FP8_ROPE:
        raise RuntimeError(
            f"expected nope last dim {_FP8_NOPE} and rope last dim {_FP8_ROPE}, "
            f"got {q_nope.size(-1)} / {q_rope.size(-1)}"
        )

    n, h = q_nope.size(0), q_nope.size(1)
    total_pages = unified_kv_nope.size(0)
    total_tokens = kv_nope.size(0)
    dev = q_nope.device

    q_bf16 = torch.empty((n, h, _FP8_FULL), dtype=torch.bfloat16, device=dev)
    unified_kv_bf16 = torch.empty(
        (total_pages, _FP8_FULL), dtype=torch.bfloat16, device=dev
    )
    kv_bf16 = torch.empty((total_tokens, _FP8_FULL), dtype=torch.bfloat16, device=dev)
    if out is None:
        out = torch.empty((n, h, _FP8_FULL), dtype=torch.bfloat16, device=dev)

    pa_sparse_prefill_opus_fp8_fwd(
        q_nope,
        q_rope,
        q_scale,
        unified_kv_nope,
        unified_kv_rope,
        unified_kv_scale,
        kv_nope,
        kv_rope,
        kv_scale,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        q_bf16,
        unified_kv_bf16,
        kv_bf16,
        out,
        float(softmax_scale),
    )
    return out


@compile_ops("module_pa_sparse_prefill_opus", develop=True)
def pa_sparse_prefill_opus_fp8_fused_fwd(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    out: torch.Tensor,
    softmax_scale: float,
) -> None: ...


def pa_sparse_prefill_opus_fp8_fused(
    q_nope: torch.Tensor,
    q_rope: torch.Tensor,
    q_scale: torch.Tensor,
    unified_kv_nope: torch.Tensor,
    unified_kv_rope: torch.Tensor,
    unified_kv_scale: torch.Tensor,
    kv_nope: torch.Tensor,
    kv_rope: torch.Tensor,
    kv_scale: torch.Tensor,
    kv_indices_prefix: torch.Tensor,
    kv_indptr_prefix: torch.Tensor,
    kv_indices_extend: torch.Tensor,
    kv_indptr_extend: torch.Tensor,
    attn_sink: torch.Tensor,
    softmax_scale: float,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused fp8 (v4-layout) sparse prefill attention (no bf16 KV scratch).

    QK-nope runs in fp8 MFMA with software per-64-tile scale; QK-rope in bf16;
    PV in bf16 with on-chip V dequant. H must be a multiple of 16.
    """
    gfx = get_gfx_runtime()
    if gfx != "gfx950":
        raise RuntimeError(
            f"pa_sparse_prefill_opus_fp8_fused requires gfx950, got {gfx}"
        )
    if q_nope.dtype != torch.float8_e4m3fn:
        raise RuntimeError(f"q_nope must be fp8 e4m3fn, got {q_nope.dtype}")
    n, h = q_nope.size(0), q_nope.size(1)
    if h % 16 != 0:
        raise RuntimeError(f"H must be a multiple of 16, got {h}")
    if out is None:
        out = torch.empty((n, h, _FP8_FULL), dtype=torch.bfloat16, device=q_nope.device)
    pa_sparse_prefill_opus_fp8_fused_fwd(
        q_nope,
        q_rope,
        q_scale,
        unified_kv_nope,
        unified_kv_rope,
        unified_kv_scale,
        kv_nope,
        kv_rope,
        kv_scale,
        kv_indices_prefix,
        kv_indptr_prefix,
        kv_indices_extend,
        kv_indptr_extend,
        attn_sink,
        out,
        float(softmax_scale),
    )
    return out


__all__ = [
    "pa_sparse_prefill_opus_fwd",
    "pa_sparse_prefill_opus",
    "pa_sparse_prefill_opus_fp8_fwd",
    "pa_sparse_prefill_opus_fp8",
    "pa_sparse_prefill_opus_fp8_fused_fwd",
    "pa_sparse_prefill_opus_fp8_fused",
]
