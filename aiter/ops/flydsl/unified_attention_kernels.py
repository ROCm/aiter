# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fp8 unified-attention backend for gfx950.

Adapts the vendored ``flash_attn_dualwave_swp`` fp8 kernel
(``kernels/flash_attn_fp8_gfx950.py``) to the ``unified_attention`` calling
convention that vLLM and SGLang import directly, so a supported gfx950 fp8
paged call routes here instead of Triton.

Dispatch follows the in-tree convention for a new backend competing for an
existing op: a pure code predicate that returns ``None`` when it cannot serve
the configuration, letting the caller fall through to Triton unchanged (see
``flydsl_flash_attn_varlen_func`` in ``fmha_kernels.py`` and the flydsl branch
in ``ops/gemm_op_a8w8.py``). There is deliberately no environment variable:
aiter already carries 19 undocumented backend-gate vars, and this path is arch-
and shape-scoped, so a code gate states the choice more honestly. For an A/B,
patch ``_FLYDSL_UNIFIED_ATTN_ARCH`` in the Triton module to ``False``.

Split-K is never built. ``varlen`` + ``num_kv_splits > 1`` is rejected by the
builder (``flash_attn_fp8_gfx950.py:110``), and every call arriving through this
API is varlen. Triton's 3d path is its split-KV tier; we simply do not take it,
which is why pure-decode shapes still favour Triton.
"""

from __future__ import annotations

import math
from functools import lru_cache

import torch

from .kernels.flash_attn_fp8_gfx950 import build_flash_attn_dualwave_swp_fp8_module
from .utils import is_flydsl_available

__all__ = ["flydsl_unified_attention"]


# Page size is structural, not a builder parameter: the paged path addresses KV
# in BLOCK_N-sized pages and BLOCK_N is pinned at 64 by the MFMA tile.
_PAGE_SIZE = 64

# Head dim is fixed by the kernel (it raises on anything else).
_HEAD_DIM = 128

# block_m, and with it the largest GQA group that can pack into the M dimension.
# _make_dualwave_swp_fp8_traits asserts block_m % gqa_group_size == 0.
_BLOCK_M = 256

# The block table is staged through a fixed LDS window of PAGED_BT_LDS_SIZE=2048
# entries. Past that the stager writes only `local_tile < segment_tiles` slots
# and silently drops the remaining page ids -- wrong output, not a fault -- so
# the KV length must be capped here. 2048 pages * 64 tokens = 131072 tokens,
# which is exactly the model's maximum context rather than comfortably above it.
_MAX_KV_TILES = 2048

_FP8_DTYPE = torch.float8_e4m3fn


@lru_cache(maxsize=32)
def _get_kernel(num_heads: int, num_kv_heads: int, causal: bool, out_dtype_str: str,
                use_sinks: bool):
    """Build (and cache) the paged+varlen fp8 launcher.

    Keyed on the head counts, the mask mode and the output dtype. Every other
    builder argument is either pinned by the support gate (head_dim, dtype,
    varlen, paged) or left at its default -- notably ``gqa_pack_m=None``, which
    auto-enables M-dimension packing for GQA > 1:1 without split-K. Adding fixed
    values to the key would only waste entries.

    ``causal``, ``out_dtype_str`` and ``use_sinks`` are runtime-selectable, so
    each must key the cache: they change compiled code (the mask path, the store
    packer, and the sink init/epilogue term).

    Do not key on batch/seqlen/device: ``_run_compiled`` memoizes the compiled
    function on the returned closure, so a finer key would defeat that cache and
    recompile per shape.
    """
    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        causal=causal,
        dtype_str="fp8",
        out_dtype_str=out_dtype_str,
        use_sinks=use_sinks,
        num_kv_heads=num_kv_heads,
        varlen=True,
        paged=True,
    )


def _strides_ok(q, out, k, v, block_table, num_kv_heads, head_size) -> bool:
    """Check the exact layout the kernel's two scalar strides imply.

    The launcher takes one ``stride_q_n`` and one ``stride_kv_n`` and derives
    every offset from them, so anything else must be declined rather than
    silently mis-addressed.
    """
    if q.stride(2) != 1 or q.stride(1) != head_size:
        return False
    if out.stride(2) != 1 or out.stride(1) != head_size:
        return False
    # One stride_q_n argument serves the Q read, the O write, and BOTH buffer
    # num_records bounds (init_descriptors), so Q and O must agree on it.
    if q.stride(0) != out.stride(0):
        return False
    page_row = num_kv_heads * head_size
    for t in (k, v):
        if t.stride(3) != 1 or t.stride(2) != head_size:
            return False
        if t.stride(1) != page_row or t.stride(0) != _PAGE_SIZE * page_row:
            return False
    return block_table.stride(1) == 1


def _supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_k,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    num_kv_heads,
    block_size,
    num_queries_per_kv,
    num_seqs,
    q_scales,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    shuffled_kv_cache,
    skip_reduce,
) -> bool:
    """Whether this exact configuration can be served. Kept separate from the
    marshalling so it can be unit-tested against meta tensors, with no GPU."""
    if not is_flydsl_available():
        return False

    head_size = q.shape[-1]
    num_query_heads = q.shape[1]

    return (
        # Dispatch mode. Both causal and non-causal are built (causal keys the
        # kernel cache); the paged path is the only one wired here, and the two
        # "reduce"/shuffle flags belong to Triton layouts we do not read.
        window_size[0] < 0
        and block_table is not None
        and not shuffled_kv_cache
        and not skip_reduce
        # Page geometry. Both are structural, not tunable.
        and block_size == _PAGE_SIZE
        and (max_seqlen_k + _PAGE_SIZE - 1) // _PAGE_SIZE <= _MAX_KV_TILES
        # Dtypes. Output packs via cvt_pk_bf16_f32 (bf16) or cvt_pkrtz (f16);
        # both store 2 bytes, so the store layout is identical.
        and q.dtype == _FP8_DTYPE
        and k.dtype == _FP8_DTYPE
        and v.dtype == _FP8_DTYPE
        and out.dtype in (torch.bfloat16, torch.float16)
        and cu_seqlens_q.dtype == torch.int32
        and seqused_k.dtype == torch.int32
        and block_table.dtype == torch.int32
        # Per-tensor fp8 descales are mandatory on this path -- the kernel reads
        # all three unconditionally to form c_logit_scale and the V dequant.
        and q_descale is not None
        and k_descale is not None
        and v_descale is not None
        and q_descale.dtype == torch.float32
        and k_descale.dtype == torch.float32
        and v_descale.dtype == torch.float32
        and q_descale.numel() == 1
        and k_descale.numel() == 1
        and v_descale.numel() == 1
        # Geometry. The GQA group must divide block_m for M-dimension packing.
        and head_size == _HEAD_DIM
        and num_query_heads % num_kv_heads == 0
        and _BLOCK_M % num_queries_per_kv == 0
        and cu_seqlens_q.numel() == num_seqs + 1
        # Features with no kernel support. Declining beats silently dropping.
        and softcap == 0
        and alibi_slopes is None
        and qq_bias is None
        and q_scales is None
        and output_scale is None
        # Sinks are served (per-head [num_query_heads] fp32) but only on the
        # single-split path. The adapter never builds split-K (see module
        # docstring), so any accepted sinks call is single-split; the guard is
        # defensive in case that ever changes.
        and (sinks is None
             or (sinks.dtype == torch.float32 and sinks.numel() == num_query_heads))
        and _strides_ok(q, out, k, v, block_table, num_kv_heads, head_size)
    )


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """fp8 buffers are passed to flydsl as int8 views (the kernel builds i8-typed
    descriptors so DMA and register loads share one byte view)."""
    return t.view(torch.int8) if t.dtype == _FP8_DTYPE else t


def _cu_seqlens_kv(seqused_k: torch.Tensor, num_seqs: int) -> torch.Tensor:
    """Lengths -> cumulative offsets.

    ``unified_attention`` passes per-sequence KV *lengths*; the kernel wants
    cumulative offsets and recovers the length as ``cu[i+1] - cu[i]``. Under
    paging the absolute bases are unobservable -- init_gmem_offsets drops
    kv_tok_base from the KV offset and the paged path builds per-page
    descriptors instead of a whole-tensor view -- so a synthesized cumsum is
    exactly equivalent to the real one.
    """
    cu = torch.zeros(num_seqs + 1, dtype=torch.int32, device=seqused_k.device)
    torch.cumsum(seqused_k, 0, out=cu[1:])
    return cu


def _scaled_q_descale(q_descale: torch.Tensor, softmax_scale: float, head_size: int):
    """Fold an arbitrary softmax scale into the Q descale.

    The kernel takes no runtime softmax scale: init_descale bakes in
    ``rsqrt(head_dim) * log2e`` and multiplies it by ``q_descale * k_descale``
    to form ``c_logit_scale``. Because that is a product, scaling q_descale by
    ``softmax_scale / rsqrt(head_dim)`` yields exactly the requested scale with
    no kernel change.

    This is sound only while q_descale reaches c_logit_scale and nothing else.
    That holds today (the loaded scalar has exactly two uses: the load and the
    multiply), and c_logit_scale is only ever applied to a DIFFERENCE of logits,
    which is what a softmax scale is. The V descale is loaded separately and
    cannot be affected. A test guards the invariant.

    For the near-universal ``softmax_scale == 1/sqrt(head_size)`` the ratio is
    exactly 1 and the tensor passes through untouched, so the common path costs
    nothing.
    """
    ratio = float(softmax_scale) * math.sqrt(head_size)
    if abs(ratio - 1.0) <= 1e-6:
        return q_descale
    return q_descale * ratio


def flydsl_unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    *,
    num_kv_heads,
    block_size,
    num_queries_per_kv,
    num_seqs,
    q_scales=None,
    alibi_slopes=None,
    output_scale=None,
    qq_bias=None,
    sinks=None,
    shuffled_kv_cache=False,
    skip_reduce=False,
):
    """Run unified attention on the FlyDSL fp8 gfx950 kernel.

    The positional parameters mirror ``unified_attention`` exactly so the hook is
    a mechanical forward. The keyword-only block is quantities the caller has
    already derived; recomputing them here would duplicate its layout unpacking.

    Returns ``out`` (written in place) if this configuration is supported, or
    ``None`` so the caller falls through to Triton.
    """
    if not _supported(
        q, k, v, out, cu_seqlens_q, seqused_k, max_seqlen_k, causal, window_size,
        block_table, softcap, q_descale, k_descale, v_descale, num_kv_heads,
        block_size, num_queries_per_kv, num_seqs, q_scales, alibi_slopes,
        output_scale, qq_bias, sinks, shuffled_kv_cache, skip_reduce,
    ):
        return None

    num_query_heads = q.shape[1]
    out_dtype_str = "f16" if out.dtype == torch.float16 else "bf16"
    kernel = _get_kernel(num_query_heads, num_kv_heads, bool(causal), out_dtype_str,
                         sinks is not None)

    with torch.cuda.device(q.device.index):
        kernel(
            _as_i8(q).view(-1),
            _as_i8(k).view(-1),
            _as_i8(v).view(-1),
            out.view(-1),
            num_seqs,
            # grid.y is ceil(seq_len / BLOCK_Q), so this must be the MAX q len;
            # per-sequence trimming comes from cu_seqlens_q via the active guard.
            int(max_seqlen_q),
            # The KV stride is the within-page row stride, not the page stride.
            k.stride(1),
            q.stride(0),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=_cu_seqlens_kv(seqused_k, num_seqs),
            q_descale=_scaled_q_descale(q_descale, softmax_scale, q.shape[-1]),
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table.view(-1),
            block_table_stride=int(block_table.stride(0)),
            sink=None if sinks is None else sinks.view(-1),
            stream=torch.cuda.current_stream(q.device),
        )
    return out
