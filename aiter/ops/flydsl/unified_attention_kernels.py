# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fp8 unified-attention backend for gfx942.

Adapts ``kernels/flash_attn_fp8_gfx942.py`` to the ``unified_attention``
calling convention, so a supported gfx942 fp8 paged call routes here instead
of Triton. Dispatch is a pure predicate returning ``None`` when it can't serve
the config, falling through to Triton unchanged.

Served: causal paged attention with head dim 256 or 512, a GQA group dividing
16, full attention or a left-only sliding window, plain (unshuffled) FP8
E4M3FNUZ K/V with page 32, 64 or 128, per-tensor fp32 descales, and bf16
output. That covers both Gemma-4 layer types. One launch serves a whole varlen
batch: multi-token sequences run as prefill tiles, one-token sequences as KV
splits that a second launch combines. A batch of only one-token sequences runs
a decode-only build of the same kernel.
"""

from __future__ import annotations

import importlib.util
from functools import cache, lru_cache

import torch

from .kernels.flash_attn_fp8_gfx942 import (
    build_flash_attn_fp8_gfx942_combine_module,
    build_flash_attn_fp8_gfx942_module,
    plan_num_kv_splits,
    prefill_block_q,
)

__all__ = ["flydsl_unified_attention"]

_FP8_DTYPE = torch.float8_e4m3fnuz
_HEAD_DIMS = (256, 512)
# vLLM pages Gemma-4's head-512 layers at twice the block size of its head-256
# layers (equal page bytes), so block size 64 gives them page 128.
_PAGE_SIZES = (32, 64, 128)
# fp32 split partials, one buffer per (device, stream), grown on demand.
_workspaces = {}


@lru_cache(maxsize=1)
def _is_flydsl_installed() -> bool:
    return importlib.util.find_spec("flydsl") is not None


@cache
def is_flydsl_available(device_index: int) -> bool:
    if not _is_flydsl_installed():
        return False
    props = torch.cuda.get_device_properties(device_index)
    return props.gcnArchName.split(":", 1)[0] == "gfx942"


def _window_keys(window_size):
    """Inclusive key window, None for full attention, or False if unsupported."""
    left, right = window_size
    if left < 0 and right < 0:
        return None
    if left >= 0 and right in (0, -1):
        return left + 1
    return False


def _dispatch_mode_ok(causal, window_size, block_table, shuffled_kv_cache, skip_reduce):
    """Causal and paged, plain KV layout, no right window. The reduce flag
    belongs to a Triton layout this kernel does not write."""
    return (
        bool(causal)
        and _window_keys(window_size) is not False
        and block_table is not None
        and not shuffled_kv_cache
        and not skip_reduce
    )


def _page_geometry_ok(block_size) -> bool:
    return block_size in _PAGE_SIZES


def _dtypes_ok(q, k, v, out, cu_seqlens_q, seqused_k, block_table) -> bool:
    """FNUZ fp8 QKV, bf16 output, int32 index tensors."""
    return (
        q.dtype == _FP8_DTYPE
        and k.dtype == _FP8_DTYPE
        and v.dtype == _FP8_DTYPE
        and out.dtype == torch.bfloat16
        and cu_seqlens_q.dtype == torch.int32
        and seqused_k.dtype == torch.int32
        and block_table.dtype == torch.int32
    )


def _descales_ok(q_descale, k_descale, v_descale) -> bool:
    """Per-tensor fp32 descales; numel()==1 admits the 0-dim scalars vLLM
    passes, which _as_1d_descale widens."""
    return all(
        d is not None and d.dtype == torch.float32 and d.numel() == 1
        for d in (q_descale, k_descale, v_descale)
    )


def _as_1d_descale(d):
    """FlyDSL's from_dlpack rejects a 0-dim tensor; reshape is a view."""
    return d.reshape(1) if d.ndim == 0 else d


def _geometry_ok(
    q, k, v, out, num_kv_heads, block_size, num_queries_per_kv, cu_seqlens_q, num_seqs
) -> bool:
    """Head dim the kernel is built for; the GQA group divides the 16 MFMA
    rows a wave owns; cu_seqlens covers every sequence."""
    if q.dim() != 3 or k.dim() != 4 or v.shape != k.shape or out.shape != q.shape:
        return False
    _, num_query_heads, head_size = q.shape
    return (
        head_size in _HEAD_DIMS
        and tuple(k.shape[1:]) == (block_size, num_kv_heads, head_size)
        and num_query_heads == num_kv_heads * num_queries_per_kv
        and 16 % num_queries_per_kv == 0
        and cu_seqlens_q.numel() == num_seqs + 1
    )


def _strides_ok(q, k, v, out, block_table) -> bool:
    """Contiguous Q and O. K and V may be strided views (vLLM splits one cache
    tensor into them); the kernel is built for their page, token, and head
    strides and needs a unit head-dim stride."""
    return (
        q.is_contiguous()
        and out.is_contiguous()
        and k.stride(3) == 1
        and v.stride(3) == 1
        and block_table.dim() == 2
        and block_table.stride(1) == 1
    )


def _no_unsupported_features(
    softcap, alibi_slopes, qq_bias, q_scales, output_scale, sinks
) -> bool:
    """Features the kernel has no path for; declining beats silently dropping."""
    return (
        not softcap
        and alibi_slopes is None
        and qq_bias is None
        and q_scales is None
        and output_scale is None
        and sinks is None
    )


def _supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
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
    """Whether this exact configuration can be served."""
    if not is_flydsl_available(q.device.index):
        return False
    return (
        _dispatch_mode_ok(
            causal, window_size, block_table, shuffled_kv_cache, skip_reduce
        )
        and _page_geometry_ok(block_size)
        and _dtypes_ok(q, k, v, out, cu_seqlens_q, seqused_k, block_table)
        and _descales_ok(q_descale, k_descale, v_descale)
        and _geometry_ok(
            q,
            k,
            v,
            out,
            num_kv_heads,
            block_size,
            num_queries_per_kv,
            cu_seqlens_q,
            num_seqs,
        )
        and _no_unsupported_features(
            softcap, alibi_slopes, qq_bias, q_scales, output_scale, sinks
        )
        and _strides_ok(q, k, v, out, block_table)
    )


def _cede_to_triton(head_size, max_seqlen_q, num_seqs, max_seqlen_k) -> bool:
    """Tiny head-512 decode-only batches stay on Triton: up to 4K keys in all
    and 2K per sequence (batch 1-2 to 2K, batch 4 at 1K), where launch latency
    dominates and Triton is 1.0-1.3x faster on MI325X. Every other batch is
    faster here and is served."""
    return (
        head_size == 512
        and max_seqlen_q == 1
        and max_seqlen_k <= 2048
        and num_seqs * max_seqlen_k <= 4096
    )


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """fp8 buffers are passed to FlyDSL as int8 views."""
    return t.view(torch.int8) if t.dtype == _FP8_DTYPE else t


def _workspace(device, stream, numel):
    key = (device, stream.cuda_stream)
    part = _workspaces.get(key)
    if part is None or part.numel() < numel:
        part = torch.empty(numel, device=device, dtype=torch.float32)
        _workspaces[key] = part
    return part


def _launch(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    window,
    block_table,
    q_descale,
    k_descale,
    v_descale,
    num_kv_heads,
    block_size,
    num_seqs,
    num_kv_splits=None,
):
    """Launch for a call that passed _supported. window is the inclusive key
    count or None; num_kv_splits forces the decode split count."""
    _, num_query_heads, head_size = q.shape
    decode_only = max_seqlen_q == 1
    # Triton's q-block count: an upper bound on any batch's prefill tiles.
    tile_slots = (
        0
        if decode_only
        else q.shape[0] // prefill_block_q(num_query_heads, num_kv_heads) + num_seqs
    )
    splits = num_kv_splits or plan_num_kv_splits(
        num_seqs, max_seqlen_k, num_kv_heads, window, head_size
    )
    # Each decode wave runs one split; the unified kernel has four per workgroup.
    waves = 1 if decode_only else 4
    groups = (splits + waves - 1) // waves
    stream = torch.cuda.current_stream(q.device)
    workspace = _workspace(
        q.device,
        stream,
        max(4, num_seqs * num_query_heads * groups * waves * (head_size + 4)),
    )
    with torch.cuda.device(q.device):
        kernel = build_flash_attn_fp8_gfx942_module(
            head_size,
            num_query_heads,
            num_kv_heads,
            window,
            block_size,
            k.stride()[:3] + v.stride()[:3],
            decode_only=decode_only,
        )
        combine = build_flash_attn_fp8_gfx942_combine_module(head_size, num_query_heads)
        kernel(
            _as_i8(q),
            _as_i8(k),
            _as_i8(v),
            out,
            workspace,
            cu_seqlens_q,
            seqused_k,
            block_table,
            _as_1d_descale(q_descale),
            _as_1d_descale(k_descale),
            _as_1d_descale(v_descale),
            num_seqs,
            tile_slots,
            groups,
            # Binary-search steps over cu_seqlens_q's num_seqs + 1 entries.
            max(1, num_seqs.bit_length()),
            block_table.stride(0),
            float(softmax_scale),
            tile_slots + num_seqs * groups,
            stream=stream,
        )
        combine(workspace, out, cu_seqlens_q, num_seqs, groups * waves, stream=stream)
    return out


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
    """Run unified attention on the FlyDSL fp8 gfx942 kernel.

    The positional parameters mirror ``unified_attention`` exactly;
    max_seqlen_q and max_seqlen_k are host integers that must bound every
    sequence. The keyword-only block is quantities the caller has already
    derived.

    Returns ``out`` (written in place) if this configuration is supported, or
    ``None`` so the caller falls through to Triton.
    """
    if not _supported(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
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
    ):
        return None
    if _cede_to_triton(q.shape[-1], max_seqlen_q, num_seqs, max_seqlen_k):
        return None
    return _launch(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        _window_keys(window_size),
        block_table,
        q_descale,
        k_descale,
        v_descale,
        num_kv_heads,
        block_size,
        num_seqs,
    )
