# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM-style paged attention (the ``unified_attention`` form) on gfx11 with the
FlyDSL kernels: ``flydsl_unified_attention_gfx11`` takes one step's batch of
prefill chunks and decode tokens over a paged KV cache and writes ``out``.

Three kernels, all FlyDSL. Sequences with more than ``16 / (heads per KV
head)`` queries run on the paged prefill kernel (128-row query blocks, many
waves per workgroup); the shorter ones (decode, speculative decode) on the
decode kernel, one wave per (sequence, KV head, KV segment) with the KV head's
query heads packed into the WMMA rows; when the keys are split into segments,
the merge kernel combines them and writes those rows, leaving the prefill
kernel's rows alone. The two attention kernels never touch the same row.
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
from aiter.ops.flydsl.kernels.flash_attn_paged_reduce_gfx11 import (
    build_flash_attn_paged_reduce_module_gfx11,
)

__all__ = ["flydsl_unified_attention_gfx11", "gfx11_paged_supported"]

_DTYPE_STR = {torch.bfloat16: "bf16", torch.float16: "f16"}
_LDS_BYTES = 65536
# The decode side's tile and grid choices (the prefill tile comes from the
# varlen rule). One place, so a sweep can set them.
_TUNE = {
    "decode_block_n": 16,  # keys per decode tile (16 or 32)
    "decode_vt_rows": 2,  # keys per LDS write in the V^T staging (2, 4, 8)
    "target_wgs": 160,  # one-wave decode workgroups the segments aim for
    "max_segments": 32,  # KV segments per sequence at most
    # decode rows of a mixed batch: KV segments + merge (True) or one segment
    # with several waves per workgroup writing the rows themselves (False)
    "mixed_segments": True,
}


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


@functools.lru_cache(maxsize=64)
def _reduce_kernel(device_index, H, D, dtype_str):
    return build_flash_attn_paged_reduce_module_gfx11(
        num_heads=H, head_dim=D, dtype_str=dtype_str
    )


def _short_limit(H, HKV):
    """Queries per sequence up to which the decode kernel takes it: the WMMA
    tile holds 16 / GROUP tokens of a KV head's GROUP query heads; 0 when the
    head ratio does not fit a tile (then everything runs on the prefill kernel)."""
    group = H // HKV
    return 16 // group if group <= 16 and 16 % group == 0 else 0


def _decode_waves(D, block_n):
    """Waves per decode workgroup when it runs without segments: as many as
    fit their V^T tiles in LDS, up to 8 (4 at head_dim 256, 8 at head_dim 128)."""
    dp = -(-D // 16) * 16
    tile_bytes = dp * (block_n + 8) * 2
    nw = 8
    while nw > 1 and nw * tile_bytes > _LDS_BYTES:
        nw //= 2
    return nw


def _num_segments(batch, HKV, max_seqlen_k, block_n):
    """KV segments for the decode kernel: enough one-wave workgroups to reach
    ``target_wgs`` (gfx1151: 20 WGPs x 8), at least one tile per segment, a
    power of two, at most ``max_segments``."""
    want = -(-int(_TUNE["target_wgs"]) // max(1, batch * HKV))
    tiles = -(-max_seqlen_k // block_n)
    n = int(max(1, min(want, tiles, int(_TUNE["max_segments"]))))
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
    bn = int(_TUNE["decode_block_n"])
    vt = int(_TUNE["decode_vt_rows"])

    if max_q > short:
        # The long sequences (prefill chunks): the kernel skips the short ones.
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
        if short == 0:
            return out
    dec_q = min(max_q, short)  # the decode kernel's longest sequence

    # The short sequences: their keys split into segments so that the grid
    # fills the GPU (the batch may hold long sequences too: an upper bound).
    nseg = _num_segments(batch, HKV, max_k, bn)
    if max_q > short and not _TUNE["mixed_segments"]:
        nseg = 1
    if nseg == 1:
        # Enough workgroups without splitting; several waves per workgroup
        # split the tiles instead.
        nw = _decode_waves(D, bn) if max_q > short else 1
        dec = _decode_kernel(dev, H, HKV, D, page, dtype_str, scale, bn, vt, nw, True)
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
            dec_q,
            1,
            stream=stream,
        )
        return out
    dec = _decode_kernel(dev, H, HKV, D, page, dtype_str, scale, bn, vt, 1, False)
    dpad = dec.dpad2
    # Partials per (sequence, token slot): the long sequences' slots stay
    # unused, so the buffers do not grow with the prefill tokens.
    rows = batch * dec_q
    segm_output = torch.empty(rows, H, nseg, dpad, dtype=torch.float32, device=q.device)
    segm_max = torch.empty(rows, H, nseg, dtype=torch.float32, device=q.device)
    segm_expsum = torch.empty(rows, H, nseg, dtype=torch.float32, device=q.device)
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
        dec_q,
        nseg,
        stream=stream,
    )
    red = _reduce_kernel(dev, H, D, dtype_str)
    red(
        segm_output,
        segm_max,
        segm_expsum,
        out,
        cu_seqlens_q,
        batch,
        nseg,
        dec_q,
        stream=stream,
    )
    return out
