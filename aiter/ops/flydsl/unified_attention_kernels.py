# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL fp8 paged unified attention on gfx950.

The fmha_gfx950 builder serves single-pass prefill and pa_decode serves shuffled
decode. Causal calls outside that routing region return None so the caller falls
back to the Triton wrapper (Gluon on gfx950).
"""

from __future__ import annotations

import os
from functools import lru_cache

import torch
from flydsl.utils.env import runtime as _flydsl_runtime

from .kernels.flash_attn_func_fp8_gfx950 import (
    _FP8_MAX_FLAT_ELEMS,
    _fp8_auto_block_m,
    _fp8_rescale_threshold,
    _gpu_arch_cached,
    _is_valid_softmax_scale,
    _num_cu,
)
from .kernels.fmha_gfx950.flash_attn_fp8_gfx950 import (
    build_flash_attn_dualwave_swp_fp8_module,
)

__all__ = ["flydsl_unified_attention"]


def is_flydsl_available(device_index: int) -> bool:
    # The adapter's import boundary already handles an absent FlyDSL package.
    return _gpu_arch_cached(device_index) == "gfx950"


# Page size is structural, not a builder parameter: the paged path addresses KV
# in BLOCK_N-sized pages and BLOCK_N is pinned at 64 by the MFMA tile.
_PAGE_SIZE = 64

# Head dim is fixed by the kernel (it raises on anything else).
_HEAD_DIM = 128

# Vectorization width of the shuffled 5D KV cache: 16 fp8 elements = one
# 128-bit dwordx4.
_KV_VEC_SIZE = 16


# The block table is staged through a fixed LDS window of PAGED_BT_LDS_SIZE=2048
# entries. Past that the stager writes only `local_tile < segment_tiles` slots
# and silently drops the remaining page ids -- wrong output, not a fault -- so
# the KV length must be capped here: 2048 pages * 64 tokens = 131072 tokens.
_MAX_KV_TILES = 2048

_FP8_DTYPE = torch.float8_e4m3fn

# GQA group sizes routed to the packed BN64 body (group 1 packing is the identity).
_PACKED_BN64_GROUPS = (4, 8, 16)


@lru_cache(maxsize=64)
def _get_kernel(
    num_heads: int,
    num_kv_heads: int,
    causal: bool,
    out_dtype_str: str,
    shuffled_kv_cache: bool,
    rescale_threshold: float,
    block_m: int,
    packed_bn64: bool = False,
):
    """Cache only build-time specializations, not runtime batch/sequence sizes.

    ``packed_bn64`` selects the conventional BN64 body with the GQA group packed
    into M; that body fixes block_m=128 itself, so ``block_m`` is ignored for it.
    """
    variant = (
        {"body_variant": "conventional_bn64", "gqa_pack_m": True} if packed_bn64 else {}
    )
    return build_flash_attn_dualwave_swp_fp8_module(
        num_heads=num_heads,
        head_dim=_HEAD_DIM,
        num_kv_heads=num_kv_heads,
        causal=causal,
        varlen=True,
        cross_seqlen=True,
        paged=True,
        kv_cache_layout="shuffled" if shuffled_kv_cache else "linear",
        out_dtype=out_dtype_str,
        rescale_threshold=rescale_threshold,
        block_m=block_m,
        batch_interleave_group=1,
        num_kv_splits=1,
        **variant,
    )


# FlyDSL pa_decode route for all-decode calls on the shuffled fp8 KV cache. The
# partition count is a host-only static rule (no planner, no device sync).
_PA_DECODE_TILE = 256  # pa_decode's fixed context_partition_size
_PA_DECODE_MAX_NP = 32
# Target workgroups per CU for one launch. Measured best on gfx950 for both
# pa_decode tile variants: 144 VGPRs with wide KV addressing (3 WGs/CU resident)
# and 114 VGPRs without (4 resident). More WGs per CU adds partitions and reduce
# work without raising in-flight loads.
_PA_DECODE_WGS_PER_CU = 3


def _env_use_pa_decode(default: bool = True) -> bool:
    raw = os.environ.get("AITER_UNIFIED_ATTN_PA_DECODE")
    return default if raw is None else raw == "1"


_USE_PA_DECODE = _env_use_pa_decode()


def _pa_decode_num_partitions(
    num_seqs: int, num_kv_heads: int, max_seqlen_k: int, num_cus: int
) -> int:
    """Static partition count NP: split each sequence into about
    three workgroups per CU across the GPU, but never less than one 256-token
    tile per partition. Python ints only."""
    fit = _PA_DECODE_WGS_PER_CU * num_cus // (num_seqs * num_kv_heads)
    hi = min(-(-max_seqlen_k // _PA_DECODE_TILE), _PA_DECODE_MAX_NP)
    return max(1, min(fit, hi))


def _pa_decode_ok(max_seqlen_q, shuffled_kv_cache, out, num_seqs) -> bool:
    """Layout/dtype gate on top of _supported (which already pins fp8 QKV,
    5D shuffled strides, page 64, D128, int32 index tensors, no softcap/alibi)."""
    return (
        _USE_PA_DECODE
        and max_seqlen_q == 1
        and shuffled_kv_cache
        and out.dtype in (torch.bfloat16, torch.float16)
        and num_seqs > 0
    )


def _route_pa_decode(
    q,
    k,
    v,
    out,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    block_table,
    q_descale,
    k_descale,
    v_descale,
    *,
    num_kv_heads,
    num_seqs,
    sinks=None,
):
    """All-decode call -> FlyDSL pa_decode with fp8 Q. seqused_k is already the
    per-sequence context length, so no cumulative-offset conversion is needed."""
    from .pa_decode import pa_decode

    np_ = _pa_decode_num_partitions(
        num_seqs, num_kv_heads, int(max_seqlen_k), _num_cu(q.device)
    )
    with torch.cuda.device(q.device.index):
        pa_decode(
            out,
            q,
            k,
            v,
            seqused_k,
            block_table,
            float(softmax_scale),
            1,
            np_,
            _PA_DECODE_TILE,
            _FP8_DTYPE,
            q_descale.reshape(-1),
            k_descale.reshape(-1),
            v_descale.reshape(-1),
            sinks=sinks,
        )
    return out


def _kv_strides_ok_5d(k, v, num_kv_heads, head_size) -> bool:
    """Validate the shuffled 5D KV-cache shape/strides.

    Layout: K = ``[num_blocks, kv_heads, head_size//x, block_size, x]``, V =
    ``[num_blocks, kv_heads, block_size//x, head_size, x]``, x = _KV_VEC_SIZE.
    Requires the trailing (vectorized) dim to be exactly ``x`` elements,
    contiguous, and the whole tensor row-major in that 5D shape -- the
    vectorized loaders assume the fixed byte-offset formula that only holds
    under that layout.
    """
    if k.dim() != 5 or v.dim() != 5:
        return False
    x = _KV_VEC_SIZE
    if head_size % x != 0 or _PAGE_SIZE % x != 0:
        return False
    k_shape = (k.shape[0], num_kv_heads, head_size // x, _PAGE_SIZE, x)
    v_shape = (v.shape[0], num_kv_heads, _PAGE_SIZE // x, head_size, x)
    if tuple(k.shape) != k_shape or tuple(v.shape) != v_shape:
        return False
    for t, shape in ((k, k_shape), (v, v_shape)):
        stride = 1
        for dim in reversed(range(5)):
            if t.stride(dim) != stride:
                return False
            stride *= shape[dim]
    return True


def _shapes_ok(
    q,
    out,
    k,
    v,
    block_table,
    cu_seqlens_q,
    seqused_k,
    num_kv_heads,
    num_seqs,
    shuffled_kv_cache,
    q_descale,
    k_descale,
    v_descale,
) -> bool:
    """Exact shapes and devices the launcher addresses from Q alone.

    The kernel derives Q/O bounds and the Dv=D128 V addressing from Q and the
    head counts, so a smaller O or V (e.g. a column slice) must decline before
    any reshape can silently copy it.
    """
    if out.shape != q.shape:
        return False
    # Q/O are flattened and the C-ABI packs the dynamic dim as signed i32.
    if q.numel() >= _FP8_MAX_FLAT_ELEMS:
        return False
    if block_table.dim() != 2 or block_table.shape[0] < num_seqs:
        return False
    # Metadata is read as flat int32 vectors; the tensor adaptor needs a stride-1 axis.
    for meta in (seqused_k, cu_seqlens_q):
        if meta.dim() != 1 or meta.stride(0) != 1:
            return False
    if seqused_k.numel() != num_seqs or cu_seqlens_q.numel() != num_seqs + 1:
        return False
    if k.shape != v.shape and not shuffled_kv_cache:
        return False
    if k.dim() not in (4, 5) or k.shape[0] != v.shape[0]:
        return False
    if not shuffled_kv_cache and k.shape[1:] != (_PAGE_SIZE, num_kv_heads, _HEAD_DIM):
        return False
    # The shuffled K/V shapes are checked exactly in _kv_strides_ok_5d.
    return all(
        t.device == q.device
        for t in (
            out,
            k,
            v,
            block_table,
            cu_seqlens_q,
            seqused_k,
            q_descale,
            k_descale,
            v_descale,
        )
    )


def _strides_ok(
    q,
    out,
    k,
    v,
    block_table,
    num_query_heads,
    num_kv_heads,
    head_size,
    shuffled_kv_cache,
) -> bool:
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
    # Flattening must remain a view with the row stride passed to the kernel.
    if block_table.stride(1) != 1 or block_table.stride(0) != block_table.shape[1]:
        return False
    # `_run_compiled` launches on `q.reshape(-1)` / `out.reshape(-1)`, baking
    # the tensor's full memref into the kernel cache signature. A padded
    # (non-flattenable) layout would make reshape return a silent COPY --
    # the kernel writes the copy, the caller's real `out` stays untouched.
    # Requiring stride(0) == flattened row size restricts acceptance to
    # layouts where reshape(-1) is a view; padded layouts decline and fall
    # through to the Triton wrapper.
    if q.stride(0) != num_query_heads * head_size:
        return False
    if out.stride(0) != num_query_heads * head_size:
        return False
    # The flag and the tensor rank must agree: the vectorized loader's
    # byte-offset formula would read the wrong memory on a mismatch.
    if shuffled_kv_cache:
        if k.dim() != 5 or v.dim() != 5:
            return False
        return _kv_strides_ok_5d(k, v, num_kv_heads, head_size)
    if k.dim() != 4 or v.dim() != 4:
        return False
    page_row = num_kv_heads * head_size
    for t in (k, v):
        if t.stride(3) != 1 or t.stride(2) != head_size:
            return False
        if t.stride(1) != page_row or t.stride(0) != _PAGE_SIZE * page_row:
            return False
    return True


def _dispatch_mode_ok(window_size, block_table, skip_reduce) -> bool:
    """Paged, full-window, non-reduce. The reduce flag belongs to a Triton
    layout this kernel does not read. Causal and non-causal are both built.

    ``shuffled_kv_cache`` is not gated here: ``_get_kernel`` selects the layout
    and ``_strides_ok`` validates the 5D K/V shape.
    """
    return window_size == (-1, -1) and block_table is not None and not skip_reduce


def _page_geometry_ok(block_size, max_seqlen_k) -> bool:
    """Structural page shape, not tunable: fixed page size, KV within the staged
    block-table window."""
    return (
        block_size == _PAGE_SIZE
        and (max_seqlen_k + _PAGE_SIZE - 1) // _PAGE_SIZE <= _MAX_KV_TILES
    )


def _dtypes_ok(q, k, v, out, cu_seqlens_q, seqused_k, block_table) -> bool:
    """fp8 QKV, bf16/f16 output (both pack to 2 bytes), int32 index tensors."""
    return (
        q.dtype == _FP8_DTYPE
        and k.dtype == _FP8_DTYPE
        and v.dtype == _FP8_DTYPE
        and out.dtype in (torch.bfloat16, torch.float16)
        and cu_seqlens_q.dtype == torch.int32
        and seqused_k.dtype == torch.int32
        and block_table.dtype == torch.int32
    )


def _descales_ok(q_descale, k_descale, v_descale) -> bool:
    """Per-tensor fp8 descales are mandatory: the kernel reads all three to form
    c_logit_scale and the V dequant. numel()==1 admits both a 1-D [1] tensor and
    a 0-dim scalar; _as_1d_descale normalizes the latter before it reaches the
    kernel (see flydsl_unified_attention)."""
    return all(
        d is not None and d.dtype == torch.float32 and d.numel() == 1
        for d in (q_descale, k_descale, v_descale)
    )


def _as_1d_descale(d):
    """Normalize a 0-dim scalar descale to a 1-element 1-D tensor.

    vLLM passes per-tensor fp8 descales as 0-dim scalars (shape ()). FlyDSL's
    from_dlpack rejects those -- a scalar has no stride-1 axis to auto-mark
    layout-dynamic -- so widen to [1] here. reshape on a 0-dim tensor is a view,
    not a copy, and numel is unchanged so _descales_ok still passes. Passes None
    and already-1-D descales through untouched."""
    if d is not None and d.ndim == 0:
        return d.reshape(1)
    return d


def _geometry_ok(head_size, num_query_heads, num_kv_heads) -> bool:
    """Fixed head dim and integral GQA (cu_seqlens length is owned by _shapes_ok)."""
    return head_size == _HEAD_DIM and num_query_heads % num_kv_heads == 0


def _no_unsupported_features(
    softcap, alibi_slopes, qq_bias, q_scales, output_scale
) -> bool:
    """Features the kernel has no path for; declining beats silently dropping."""
    return (
        softcap == 0
        and alibi_slopes is None
        and qq_bias is None
        and q_scales is None
        and output_scale is None
    )


def _sinks_ok(sinks) -> bool:
    """Sinks are not served by FlyDSL; auto-backend calls fall back to the
    Triton wrapper. Explicit-FlyDSL sinks calls raise instead."""
    return sinks is None


def _supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    max_seqlen_k,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    num_kv_heads,
    block_size,
    num_seqs,
    q_scales,
    alibi_slopes,
    output_scale,
    qq_bias,
    sinks,
    shuffled_kv_cache,
    skip_reduce,
    softmax_scale,
) -> bool:
    """Return whether the configuration satisfies the FlyDSL device, dtype, feature,
    and layout requirements."""
    if not is_flydsl_available(q.device.index):
        return False
    # FMHA softmax masks in the unscaled domain; scale <= 0 or non-finite breaks it.
    if softmax_scale is None or not _is_valid_softmax_scale(softmax_scale):
        return False

    # Rank and positivity first: later checks index q.shape and take modulo.
    if q.dim() != 3 or num_kv_heads <= 0:
        return False
    head_size = q.shape[-1]
    num_query_heads = q.shape[1]

    return (
        _dispatch_mode_ok(window_size, block_table, skip_reduce)
        and _page_geometry_ok(block_size, max_seqlen_k)
        and _dtypes_ok(q, k, v, out, cu_seqlens_q, seqused_k, block_table)
        and _descales_ok(q_descale, k_descale, v_descale)
        and _geometry_ok(head_size, num_query_heads, num_kv_heads)
        and _no_unsupported_features(
            softcap, alibi_slopes, qq_bias, q_scales, output_scale
        )
        and _sinks_ok(sinks)
        and _shapes_ok(
            q,
            out,
            k,
            v,
            block_table,
            cu_seqlens_q,
            seqused_k,
            num_kv_heads,
            num_seqs,
            shuffled_kv_cache,
            q_descale,
            k_descale,
            v_descale,
        )
        and _strides_ok(
            q,
            out,
            k,
            v,
            block_table,
            num_query_heads,
            num_kv_heads,
            head_size,
            shuffled_kv_cache,
        )
    )


def _as_i8(t: torch.Tensor) -> torch.Tensor:
    """fp8 buffers are passed to flydsl as int8 views (the kernel builds i8-typed
    descriptors so DMA and register loads share one byte view)."""
    return t.view(torch.int8) if t.dtype == _FP8_DTYPE else t


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
    ``None`` so the caller falls back to the Triton wrapper.
    """
    # vLLM passes per-tensor descales as 0-dim scalars, which FlyDSL's
    # from_dlpack rejects; widen to [1] (a view) before the gate sees them.
    q_descale = _as_1d_descale(q_descale)
    k_descale = _as_1d_descale(k_descale)
    v_descale = _as_1d_descale(v_descale)

    # vLLM may pad the table with extra rows; launchers take exactly num_seqs rows.
    # A short table is declined by _supported, so slice only after it passes.
    if not _supported(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
        max_seqlen_k,
        window_size,
        block_table,
        softcap,
        q_descale,
        k_descale,
        v_descale,
        num_kv_heads,
        block_size,
        num_seqs,
        q_scales,
        alibi_slopes,
        output_scale,
        qq_bias,
        sinks,
        shuffled_kv_cache,
        skip_reduce,
        softmax_scale,
    ):
        return None
    block_table = block_table[:num_seqs]

    if max_seqlen_q == 1:
        # Decode kernels are not AOT-compiled; decline so run-only deployments
        # fall back.
        if _flydsl_runtime.run_only:
            return None
        if _pa_decode_ok(max_seqlen_q, shuffled_kv_cache, out, num_seqs):
            return _route_pa_decode(
                q,
                k,
                v,
                out,
                seqused_k,
                max_seqlen_k,
                softmax_scale,
                block_table,
                q_descale,
                k_descale,
                v_descale,
                num_kv_heads=num_kv_heads,
                num_seqs=num_seqs,
                sinks=sinks,
            )
        # Linear-cache decode has no FlyDSL kernel.
        return None

    # One query-length-1 row caps M at (B-1)*S+1, so a larger M has no decode rows.
    # Possibly mixed: BN64 pads decode rows to full tiles, slower than the fallback.
    # Non-causal has no fallback, so it stays on FlyDSL.
    if causal and not (num_seqs == 1 or q.shape[0] > (num_seqs - 1) * max_seqlen_q + 1):
        return None

    num_query_heads = q.shape[1]
    out_dtype_str = "f16" if out.dtype == torch.float16 else "bf16"
    target_num_prgms = _num_cu(q.device)
    block_m = _fp8_auto_block_m(
        num_seqs,
        num_query_heads,
        int(max_seqlen_q),
        int(max_seqlen_k),
        target_num_prgms,
    )
    # Packed BN64 (paged, single-pass, D128; pipeline.py allows groups 1/4/8/16,
    # only the packing groups 4/8/16 are routed) folds the GQA group into the
    # 128-row M tile: one workgroup per KV head and 128 // group query tokens.
    # Everything else keeps one query head per workgroup.
    group = num_query_heads // num_kv_heads
    packed_bn64 = group in _PACKED_BN64_GROUPS
    if packed_bn64:
        block_m = 128  # fixed by the body; keeps the cache key equal to the AOT job

    with torch.cuda.device(q.device.index):
        kernel = _get_kernel(
            num_query_heads,
            num_kv_heads,
            bool(causal),
            out_dtype_str,
            shuffled_kv_cache,
            _fp8_rescale_threshold(max_seqlen_k),
            block_m,
            packed_bn64=packed_bn64,
        )
        q_flat, out_flat = _as_i8(q).reshape(-1), out.reshape(-1)
        # A reshape copy would leave the caller's output buffer unchanged.
        assert q_flat.data_ptr() == q.data_ptr()
        assert out_flat.data_ptr() == out.data_ptr()
        kernel(
            q_flat,
            # Rank two avoids the C-ABI signed-i32 flat-pool shape overflow.
            _as_i8(k).reshape(k.shape[0], -1),
            _as_i8(v).reshape(v.shape[0], -1),
            out_flat,
            num_seqs,
            int(max_seqlen_q),
            num_kv_heads * _HEAD_DIM,
            q.stride(0),
            softmax_scale=float(softmax_scale),
            seq_len_kv=int(max_seqlen_k),
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_kv=seqused_k,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            block_table=block_table.reshape(-1),
            block_table_stride=int(block_table.stride(0)),
            stream=torch.cuda.current_stream(q.device),
        )
    return out
