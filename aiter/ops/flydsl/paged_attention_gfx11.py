# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM-style paged attention (the ``unified_attention`` form) on gfx11 with the
FlyDSL kernels: ``flydsl_unified_attention_gfx11`` takes one step's batch of
prefill chunks and decode tokens over a paged KV cache and writes ``out``.

Two kernels share the work. Sequences with more than ``16 / (heads per KV
head)`` queries run on the paged prefill kernel (128-row query blocks, many
waves per workgroup); the shorter ones (decode, speculative decode) on the
decode kernel, one wave per (sequence, KV head, KV segment) with the KV head's
query heads packed into the WMMA rows. An all-short batch splits the keys into
segments merged by aiter's Triton ``reduce_segments``; in a mixed batch the
short sequences take one segment and write their rows directly, so the two
kernels never touch the same output row and no reduce is needed.
"""

import functools
import math

import torch

from aiter.ops.flydsl.fmha_kernels import _gfx11_varlen_config
from aiter.ops.flydsl.kernels.flash_attn_paged_decode_gfx11 import (
    build_flash_attn_paged_decode_module_gfx11,
)
from aiter.ops.flydsl.kernels.flash_attn_paged_gfx11 import (
    build_flash_attn_paged_module_gfx11,
)

__all__ = ["flydsl_unified_attention_gfx11", "gfx11_paged_supported"]

_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16"}


@functools.lru_cache(maxsize=64)
def _prefill_kernel(
    device_index,
    H,
    HKV,
    D,
    page,
    dtype_str,
    scale,
    num_waves,
    block_n,
    vt_rows,
    head_group,
):
    # device_index intentionally participates in the cache key.
    return build_flash_attn_paged_module_gfx11(
        num_heads=H,
        head_dim=D,
        page_size=page,
        causal=True,
        dtype_str=dtype_str,
        sm_scale=scale,
        num_waves=num_waves,
        block_n=block_n,
        vt_rows=vt_rows,
        head_group=head_group,
        num_kv_heads=HKV,
    )


@functools.lru_cache(maxsize=64)
def _decode_kernel(
    device_index, H, HKV, D, page, dtype_str, scale, block_n, vt_rows, num_waves, direct
):
    return build_flash_attn_paged_decode_module_gfx11(
        num_heads=H,
        head_dim=D,
        page_size=page,
        dtype_str=dtype_str,
        sm_scale=scale,
        block_n=block_n,
        vt_rows=vt_rows,
        num_waves=num_waves,
        direct=direct,
        num_kv_heads=HKV,
    )


_DECODE_BLOCK_N = 16
_LDS_BYTES = 65536


def _decode_waves(D, block_n=_DECODE_BLOCK_N):
    """Waves per decode workgroup: as many as fit their V^T tiles in LDS, up
    to 8 (4 at head_dim 256, 8 at head_dim 128)."""
    dp = -(-D // 16) * 16
    tile_bytes = dp * (block_n + 8) * 2
    nw = 8
    while nw > 1 and nw * tile_bytes > _LDS_BYTES:
        nw //= 2
    return nw


def _short_limit(H, HKV):
    """Queries per sequence up to which the decode kernel takes it: the WMMA
    tile holds 16 / GROUP tokens of a KV head's GROUP query heads; 0 when the
    head ratio does not fit a tile (then everything runs on the prefill kernel)."""
    group = H // HKV
    return 16 // group if group <= 16 and 16 % group == 0 else 0


def _num_segments(batch, HKV, max_seqlen_k, block_n, target_wgs=160):
    """KV segments for an all-short batch: enough workgroups to fill the GPU
    (gfx1151: 20 WGPs x 8), at most one tile per segment, a power of two (the
    Triton reduce's segment vector)."""
    want = -(-target_wgs // max(1, batch * HKV))
    tiles = -(-max_seqlen_k // block_n)
    n = int(max(1, min(want, tiles, 32)))
    return 1 << (n.bit_length() - 1)


def gfx11_paged_supported(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    causal,
    window_size,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes,
    sinks,
    qq_bias,
    output_scale,
):
    """True when the FlyDSL kernels take the call: causal, bf16 / f16 all
    round, dense ``[blocks, page, kv_heads, head_dim]`` caches with a page a
    multiple of 8, ``head_dim % 8 == 0``, packed q / out, int32 tables, and
    none of the options the kernels have no code for."""
    if not causal or softcap or window_size[0] >= 0 or window_size[1] >= 0:
        return False
    if any(
        x is not None
        for x in (
            q_descale,
            k_descale,
            v_descale,
            alibi_slopes,
            sinks,
            qq_bias,
            output_scale,
        )
    ):
        return False
    if (
        q.dtype not in _DTYPE_STR
        or k.dtype != q.dtype
        or v.dtype != q.dtype
        or out.dtype != q.dtype
    ):
        return False
    if q.dim() != 3 or k.dim() != 4 or v.shape != k.shape or out.shape != q.shape:
        return False
    total_q, H, D = q.shape
    num_blocks, page, HKV, dk = k.shape
    if dk != D or D % 8 != 0 or page % 8 != 0 or H % HKV != 0:
        return False
    if not (
        q.is_contiguous()
        and out.is_contiguous()
        and k.is_contiguous()
        and v.is_contiguous()
    ):
        return False
    if (
        q.data_ptr() % 16
        or out.data_ptr() % 16
        or k.data_ptr() % 16
        or v.data_ptr() % 16
    ):
        return False
    for t in (cu_seqlens_q, seqused_k, block_table):
        if t.dtype != torch.int32 or not t.is_contiguous() or t.device != q.device:
            return False
    if block_table.dim() != 2 or block_table.shape[0] != seqused_k.numel():
        return False
    if cu_seqlens_q.numel() != seqused_k.numel() + 1:
        return False
    # Per-sequence Q / O descriptors address with 32-bit byte offsets.
    return total_q * H * D * q.element_size() < 2**31


def flydsl_unified_attention_gfx11(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    block_table,
    arch="gfx1151",
):
    """Paged causal attention over a vLLM step; ``gfx11_paged_supported`` must
    have accepted the call. Writes and returns ``out``."""
    total_q, H, D = q.shape
    page, HKV = k.shape[1], k.shape[2]
    batch = seqused_k.numel()
    dtype_str = _DTYPE_STR[q.dtype]
    scale = float(softmax_scale) if softmax_scale is not None else 1.0 / math.sqrt(D)
    dev = q.device.index
    stream = torch.cuda.current_stream(q.device)
    short = _short_limit(H, HKV)
    max_q, max_k = int(max_seqlen_q), int(max_seqlen_k)
    dnw = _decode_waves(D)
    bn = _DECODE_BLOCK_N

    if max_q > short:
        # Prefill chunks (and anything longer than a decode tile).
        num_waves, block_n, head_group, vt_rows = _gfx11_varlen_config(
            arch, max_q, max_k, H, D, batch, True
        )
        exe = _prefill_kernel(
            dev,
            H,
            HKV,
            D,
            page,
            dtype_str,
            scale,
            num_waves,
            block_n,
            vt_rows,
            head_group,
        )
        exe(
            q,
            k,
            v,
            out,
            cu_seqlens_q,
            seqused_k,
            block_table,
            batch,
            max_q,
            stream=stream,
            q_small=short,
        )
        if short:
            dec = _decode_kernel(
                dev, H, HKV, D, page, dtype_str, scale, bn, 2, dnw, True
            )
            dec(
                q,
                k,
                v,
                out,
                out,
                out,
                out,
                cu_seqlens_q,
                seqused_k,
                block_table,
                batch,
                short,
                1,
                stream=stream,
            )
        return out

    # All sequences are short: split the keys so the grid fills the GPU
    # (counted in waves: the workgroups already hold dnw of them).
    # One wave per workgroup here: with the segments supplying the parallelism,
    # the small LDS footprint of a single V^T tile keeps more workgroups
    # resident than the multi-wave merge does (measured 5-10% on 8-32 decodes).
    nseg = _num_segments(batch, HKV, max_k, bn)
    if nseg == 1:
        dec = _decode_kernel(dev, H, HKV, D, page, dtype_str, scale, bn, 2, 1, True)
        dec(
            q,
            k,
            v,
            out,
            out,
            out,
            out,
            cu_seqlens_q,
            seqused_k,
            block_table,
            batch,
            max_q,
            1,
            stream=stream,
        )
        return out
    dec = _decode_kernel(dev, H, HKV, D, page, dtype_str, scale, bn, 2, 1, False)
    dpad = dec.dpad2
    segm_output = torch.empty(
        total_q, H, nseg, dpad, dtype=torch.float32, device=q.device
    )
    segm_max = torch.empty(total_q, H, nseg, dtype=torch.float32, device=q.device)
    segm_expsum = torch.empty(total_q, H, nseg, dtype=torch.float32, device=q.device)
    dec(
        q,
        k,
        v,
        out,
        segm_output,
        segm_max,
        segm_expsum,
        cu_seqlens_q,
        seqused_k,
        block_table,
        batch,
        max_q,
        nseg,
        stream=stream,
    )
    _reduce(
        q,
        k,
        v,
        out,
        cu_seqlens_q,
        seqused_k,
        block_table,
        scale,
        max_q,
        max_k,
        segm_output,
        segm_max,
        segm_expsum,
        nseg,
        bn,
    )
    return out


def _reduce(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    scale,
    max_q,
    max_k,
    segm_output,
    segm_max,
    segm_expsum,
    nseg,
    tile,
):
    """Merge the decode kernel's segments with aiter's Triton reduce_segments."""
    from aiter.ops.triton.attention.unified_attention import (
        _reduce_segments_triton,
        _UAParams,
    )
    from aiter.ops.triton.utils.device_info import get_num_sms

    total_q, H, D = q.shape
    num_blocks, page, HKV, _ = k.shape
    batch = seqused_k.numel()
    cu = get_num_sms()
    params = _UAParams(
        q=q,
        k=k,
        v=v,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        softmax_scale=scale,
        softcap=0.0,
        causal=True,
        sliding_window=0,
        max_seqlen_q=max_q,
        max_seqlen_k=max_k,
        num_tokens=total_q,
        num_query_heads=H,
        num_kv_heads=HKV,
        num_queries_per_kv=H // HKV,
        head_size=D,
        head_size_v=D,
        num_seqs=batch,
        total_num_q_blocks=batch,
        num_2d_prgms=batch * HKV,
        num_blocks=num_blocks,
        block_size=page,
        k_width=8,
        scale_k_width=4,
        block_scales_size=16,
        q_dtype=q.dtype,
        kv_cache_dtype=k.dtype,
        v_cache_dtype=v.dtype,
        all_decode=max_q == 1,
        spec_decode=False,
        shuffled_kv_cache=False,
        use_alibi_slopes=False,
        use_qq_bias=False,
        num_sms=cu,
        target_num_prgms=cu * 4,
    )
    _reduce_segments_triton(params, segm_output, segm_max, segm_expsum, nseg, tile)
