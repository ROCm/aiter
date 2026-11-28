# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM-style paged attention (the ``unified_attention`` form) on gfx11 with the
FlyDSL kernels: ``flydsl_unified_attention_gfx11`` takes one step's batch of
prefill chunks and decode tokens over a paged KV cache and writes ``out``.

Three kernels, all FlyDSL. Sequences with more than ``16 / (heads per KV
head)`` queries run on the paged prefill kernel (128-row query blocks, many
waves per workgroup); the shorter ones (decode, speculative decode) on the
decode kernel, one workgroup per (sequence, KV head, KV segment) with the KV
head's query heads packed into the WMMA rows and the head dim split over its
waves; when the keys are split into segments, the merge kernel combines them
and writes those rows, leaving the prefill kernel's rows alone. The two
attention kernels never touch the same row.
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
# Tile and grid choices. 0 / None = the shape rules below (from the gfx1151
# sweeps); a sweep sets explicit values here.
_TUNE = {
    "decode_block_n": 0,  # keys per decode tile (16 or 32)
    "decode_vt_rows": 0,  # keys per LDS write in the V^T staging (2, 4, 8)
    "decode_d_split": 0,  # waves per decode workgroup splitting the head dim
    "decode_q_reload": None,  # Q operand re-read per tile instead of in VGPRs
    "decode_vt_pad": None,  # V^T row padding in LDS, 0 or 8 (None = rule)
    "decode_skew_rows": None,  # V^T skew period in rows, 0 / 4 / 8 / 16 (None = rule)
    "decode_s_slots": 0,  # partial-S exchange slots, 1 or 2 (0 = rule)
    # prefill tile override: None = the rule, else (num_waves, block_n,
    # vt_rows, d_split, double_buffer[, q_reload, waves_per_eu[, vt_pad,
    # skew_rows]])
    "prefill_tile": None,
    "prefill_lds_prefetch": None,  # LDS operands fetched n WMMAs ahead (None = rule)
    "prefill_lds_sched": None,  # scheduling groups for that order (None = off)
    "target_wgs": 0,  # decode workgroups the segments aim for (0 = rule)
    "max_segments": 32,  # KV segments per sequence at most
    # decode rows of a mixed batch: KV segments + merge (True) or one segment
    # per sequence writing the rows itself (False)
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
    d_split,
    double_buffer,
    q_reload,
    waves_per_eu,
    lds_prefetch,
    lds_sched,
    vt_layout,
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
        d_split=d_split,
        q_reload=q_reload,
        lds_prefetch=lds_prefetch,
        lds_sched=lds_sched,
        vt_pad=vt_layout[0],
        skew_rows=vt_layout[1],
        double_buffer=double_buffer,
        head_group=head_group,
        waves_per_eu=waves_per_eu,
        num_kv_heads=HKV,
    )


@functools.lru_cache(maxsize=64)
def _decode_kernel(
    device_index,
    H,
    HKV,
    D,
    page,
    dtype_str,
    scale,
    block_n,
    vt_rows,
    d_split,
    q_reload,
    vt_pad,
    skew_rows,
    s_slots,
    direct,
):
    return build_flash_attn_paged_decode_module_gfx11(
        num_heads=H,
        head_dim=D,
        page_size=page,
        dtype_str=dtype_str,
        sm_scale=scale,
        block_n=block_n,
        vt_rows=vt_rows,
        d_split=d_split,
        q_reload=q_reload,
        direct=direct,
        vt_pad=vt_pad,
        skew_rows=skew_rows,
        s_slots=s_slots,
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


def _fit_d_split(D, n):
    """``n`` waves per row group, halved until the head dim splits into
    16-column slices."""
    while n > 1 and D % (16 * n):
        n //= 2
    return n


def _decode_tile(D, batch, HKV):
    """(block_n, vt_rows, d_split, q_reload) of the decode kernel. gfx1151
    sweep, head_dim 256 (8 / 2 heads): up to 32 (sequence, KV head) pairs
    16-key tiles over two waves; beyond that 32-key tiles over four waves with
    the Q operand re-read per tile (1.3-1.8x the Triton kernel at batch
    32-256, 1.2-1.5x with two waves). Head_dim 128 (32 / 8 heads): 16-key
    tiles over two waves (speculative 8 x 4 1.85x instead of 1.43x, the rest
    within 2% of any other variant; large batches are DRAM bound)."""
    if D > 128:
        bn, vt, ds, qr = (16, 8, 2, False) if batch * HKV <= 32 else (32, 8, 4, True)
    else:
        bn, vt, ds, qr = 16, 8, 2, False
    bn = int(_TUNE["decode_block_n"]) or bn
    vt = int(_TUNE["decode_vt_rows"]) or vt
    ds = _fit_d_split(D, int(_TUNE["decode_d_split"]) or ds)
    if _TUNE["decode_q_reload"] is not None:
        qr = bool(_TUNE["decode_q_reload"])
    # V^T layout: no row padding, skew every 8 rows (16-key tiles) or 4 (32).
    pad = 0 if _TUNE["decode_vt_pad"] is None else int(_TUNE["decode_vt_pad"])
    skr = 8 if bn == 16 else 4
    if _TUNE["decode_skew_rows"] is not None:
        skr = int(_TUNE["decode_skew_rows"])
    slots = int(_TUNE["decode_s_slots"]) or 2
    return bn, vt, ds, qr, (pad, skr, slots)


def _prefill_tile(arch, max_q, max_k, H, D, batch, total_q):
    """(num_waves, block_n, vt_rows, d_split, double_buffer, q_reload,
    head_group) of the
    prefill kernel. Head dim above 128 (gfx1151 sweep, 8 / 2 heads, head_dim
    256): a batch of few query rows or short contexts runs 64-row workgroups
    of 16-key double-buffered tiles with the head dim over two waves (twice
    the workgroups: prefill 512 1.6x the Triton kernel instead of 1.0x, six
    short sequences 2.2x instead of 1.4x); long prefills keep the varlen
    rule's 128-row, 32-key tile, which is faster per key despite spilling.
    Head dim up to 128 (32 / 8 heads, head_dim 128): no head-dim split (the
    exchange costs more than it saves); contexts of 4096 keys and up run
    256-row, 64-key tiles (16 waves; each K / V tile staged once per 256
    rows: a 256-query chunk over 8192 keys 1.02x instead of 0.79x), batches
    of short sequences 64-row, 64-key tiles (4 waves: six short sequences
    1.13x instead of 0.94x), contexts of 2048 keys and up the 256-row tile
    again, the rest the varlen rule."""
    pad = (8, 16)  # V^T layout (vt_pad, skew_rows)
    if _TUNE["prefill_tile"] is not None:
        nw, bn, vt, ds, db, *rest = _TUNE["prefill_tile"]
        qr = bool(rest[0]) if rest else False
        wpe = int(rest[1]) if len(rest) > 1 and rest[1] else None
        if len(rest) > 3:
            pad = (int(rest[2]), int(rest[3]))
        return int(nw), int(bn), int(vt), int(ds), bool(db), qr, wpe, pad, None
    if D > 128:
        if (total_q <= 768 or max_k <= 512) and D % 32 == 0:
            return 8, 16, 8, 2, True, False, None, pad, None
    elif max_k >= 4096:
        return 16, 64, 8, 1, True, False, None, pad, None
    elif max_q <= 256:
        # 4 waves x 32 keys, single buffer, padless V^T: 6 short sequences
        # 1.47x instead of 1.18x
        return 4, 32, 8, 1, False, False, None, (0, 4), None
    elif max_k >= 2048:
        return 16, 64, 8, 1, True, False, None, pad, None
    num_waves, block_n, head_group, vt_rows = _gfx11_varlen_config(
        arch, max_q, max_k, H, D, batch, True
    )
    return num_waves, block_n, vt_rows, 1, True, False, None, pad, head_group


def _num_segments(batch, HKV, max_seqlen_k, block_n, D):
    """KV segments for the decode kernel: enough workgroups to reach the
    target (gfx1151 sweeps: 160 at head_dim 256, 128 at head_dim 128, where
    16 x 2 speculative tokens run 0.97x the Triton kernel unsplit and 0.91x
    split in two), at least one tile per segment, at most ``max_segments``."""
    target = int(_TUNE["target_wgs"]) or (160 if D > 128 else 128)
    want = -(-target // max(1, batch * HKV))
    tiles = -(-max_seqlen_k // block_n)
    return int(max(1, min(want, tiles, int(_TUNE["max_segments"]))))


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
    bn, vt, ds, qr, (vpad, skr, slots) = _decode_tile(D, batch, HKV)

    if max_q > short:
        # The long sequences (prefill chunks): the kernel skips the short ones.
        num_waves, block_n, vt_rows, pds, dbuf, pqr, wpe, lay, head_group = (
            _prefill_tile(arch, max_q, max_k, H, D, batch, total_q)
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
            pds,
            dbuf,
            pqr,
            wpe,
            # LDS operands fetched this many WMMAs ahead: deeper at head_dim
            # 256, where the tile spills anyway and depth 2 only adds pressure
            (
                (4 if D > 128 else 2)
                if _TUNE["prefill_lds_prefetch"] is None
                else int(_TUNE["prefill_lds_prefetch"])
            ),
            (
                False
                if _TUNE["prefill_lds_sched"] is None
                else bool(_TUNE["prefill_lds_sched"])
            ),
            lay,
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
    nseg = _num_segments(batch, HKV, max_k, bn, D)
    if max_q > short and not _TUNE["mixed_segments"]:
        nseg = 1
    if nseg == 1:
        # Enough workgroups without splitting: the rows are written directly.
        dec = _decode_kernel(
            dev,
            H,
            HKV,
            D,
            page,
            dtype_str,
            scale,
            bn,
            vt,
            ds,
            qr,
            vpad,
            skr,
            slots,
            True,
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
            dec_q,
            1,
            stream=stream,
        )
        return out
    dec = _decode_kernel(
        dev, H, HKV, D, page, dtype_str, scale, bn, vt, ds, qr, vpad, skr, slots, False
    )
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
