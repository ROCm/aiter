# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import os

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.device_info import get_num_sms
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Flash-decoding (split-K) sliding-window paged decode over the shuffled fp8 KV
# layout used by the asm pa_fwd path. Same cache layout and window semantics as
# pa_decode_shuffle, but the windowed key range is split into partitions across
# the grid so the long-context case fills the GPU instead of serialising one
# program per (seq, kv head). A second pass reduces the per-partition softmax
# state. For a single partition (small window) the host calls the one-pass
# kernel directly so we don't pay the workspace round-trip.
#
# Shuffled layout (x = 16 // elem_size, = 16 for fp8):
#   K: [num_blocks, num_kv_heads, head_size // x, block_size, x]
#   V: [num_blocks, num_kv_heads, block_size // x, head_size, x]
#   K[d, slot] -> (d // x) * block_size * x + slot * x + (d % x)
#   V[slot, d] -> (slot // x) * head_size * x + d * x + (slot % x)

_REPR_KEYS = ["HEAD", "BLOCK", "GRP", "SCALE_SCALAR"]
_onepass_repr = make_kernel_repr("_pa_shuffle_swa_onepass_kernel", _REPR_KEYS)
_partition_repr = make_kernel_repr("_pa_shuffle_swa_partition_kernel", _REPR_KEYS)
_reduce_repr = make_kernel_repr(
    "_pa_shuffle_swa_reduce_kernel", ["HEAD", "GRP", "NPART_POW2"]
)


@triton.jit
def _pa_shuffle_swa_block(
    q,
    m_i,
    l_i,
    acc,
    k_cache_ptr,
    v_cache_ptr,
    k_scale_ptr,
    v_scale_ptr,
    ks_scalar,
    vs_scalar,
    block_id,
    kvh,
    blk_start,
    lo,
    hi,
    scale,
    ks_s0,
    ks_s1,
    ks_s2,
    vs_s0,
    vs_s1,
    vs_s2,
    k_blk_stride,
    k_head_stride,
    v_blk_stride,
    v_head_stride,
    HEAD: tl.constexpr,
    HEAD_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    X: tl.constexpr,
    SCALE_SCALAR: tl.constexpr,
):
    # one cache block of the online softmax, keys in [lo, hi) are valid
    s = tl.arange(0, BLOCK)
    d = tl.arange(0, HEAD_POW2)
    pos = blk_start + s
    valid = (pos < hi) & (pos >= lo)
    k_base = block_id * k_blk_stride + kvh * k_head_stride
    v_base = block_id * v_blk_stride + kvh * v_head_stride
    ld_mask = (s[:, None] < BLOCK) & (d[None, :] < HEAD)
    k_off = k_base + (d[None, :] // X) * (BLOCK * X) + s[:, None] * X + (d[None, :] % X)
    v_off = v_base + (s[:, None] // X) * (HEAD * X) + d[None, :] * X + (s[:, None] % X)
    k = tl.load(k_cache_ptr + k_off, mask=ld_mask, other=0.0).to(tl.float32)
    v = tl.load(v_cache_ptr + v_off, mask=ld_mask, other=0.0).to(tl.float32)
    if SCALE_SCALAR:
        k = k * ks_scalar
        v = v * vs_scalar
    else:
        ks = tl.load(
            k_scale_ptr + block_id * ks_s0 + kvh * ks_s1 + s * ks_s2,
            mask=s < BLOCK,
            other=0.0,
        )
        vs = tl.load(
            v_scale_ptr + block_id * vs_s0 + kvh * vs_s1 + s * vs_s2,
            mask=s < BLOCK,
            other=0.0,
        )
        k = k * ks[:, None]
        v = v * vs[:, None]
    # unused slots get p = 0, but 0 * NaN is NaN in the p @ v dot, so a NaN
    # left in an unused V slot or its scale has to be zeroed. Only the edge
    # blocks of [lo, hi) have unused slots, interior blocks skip the select.
    if (blk_start < lo) | (blk_start + BLOCK > hi):
        v = tl.where(valid[:, None], v, 0.0)
    scores = tl.dot(q, tl.trans(k)) * scale
    scores = tl.where(valid[None, :], scores, -float("inf"))
    m_new = tl.maximum(m_i, tl.max(scores, axis=1))
    p = tl.exp(scores - m_new[:, None])
    alpha = tl.exp(m_i - m_new)
    l_i = l_i * alpha + tl.sum(p, axis=1)
    acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
    return m_new, l_i, acc


@triton.jit(repr=_onepass_repr)
def _pa_shuffle_swa_onepass_kernel(
    out_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    blk_tbl_ptr,
    seq_lens_ptr,
    k_scale_ptr,
    v_scale_ptr,
    scale,
    sliding_window,
    q_s0,
    q_s1,
    q_s2,
    o_s0,
    o_s1,
    o_s2,
    bt_s0,
    ks_s0,
    ks_s1,
    ks_s2,
    vs_s0,
    vs_s1,
    vs_s2,
    k_blk_stride,
    k_head_stride,
    v_blk_stride,
    v_head_stride,
    HEAD: tl.constexpr,
    HEAD_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    X: tl.constexpr,
    GRP: tl.constexpr,
    GRP_POW2: tl.constexpr,
    SCALE_SCALAR: tl.constexpr,
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    seq_len = tl.load(seq_lens_ptr + seq)
    win_start = 0
    if sliding_window > 0:
        ws = seq_len - sliding_window
        win_start = tl.where(ws > 0, ws, 0)

    grp = tl.arange(0, GRP_POW2)
    d = tl.arange(0, HEAD_POW2)
    q_head = kvh * GRP + grp
    q_mask = (grp[:, None] < GRP) & (d[None, :] < HEAD)
    q = tl.load(
        q_ptr + seq * q_s0 + q_head[:, None] * q_s1 + d[None, :] * q_s2,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([GRP_POW2], -float("inf"), tl.float32)
    l_i = tl.zeros([GRP_POW2], tl.float32)
    acc = tl.zeros([GRP_POW2, HEAD_POW2], tl.float32)
    ks_scalar = 1.0
    vs_scalar = 1.0
    if SCALE_SCALAR:
        ks_scalar = tl.load(k_scale_ptr)
        vs_scalar = tl.load(v_scale_ptr)

    for b in range(win_start // BLOCK, tl.cdiv(seq_len, BLOCK)):
        block_id = tl.load(blk_tbl_ptr + seq * bt_s0 + b).to(tl.int64)
        m_i, l_i, acc = _pa_shuffle_swa_block(
            q,
            m_i,
            l_i,
            acc,
            k_cache_ptr,
            v_cache_ptr,
            k_scale_ptr,
            v_scale_ptr,
            ks_scalar,
            vs_scalar,
            block_id,
            kvh,
            b * BLOCK,
            win_start,
            seq_len,
            scale,
            ks_s0,
            ks_s1,
            ks_s2,
            vs_s0,
            vs_s1,
            vs_s2,
            k_blk_stride,
            k_head_stride,
            v_blk_stride,
            v_head_stride,
            HEAD,
            HEAD_POW2,
            BLOCK,
            X,
            SCALE_SCALAR,
        )

    # seq_len == 0 (cudagraph padding rows) never enters the loop: write zeros, not 0 / 0
    out = tl.where(l_i[:, None] > 0.0, acc / l_i[:, None], 0.0)
    tl.store(
        out_ptr + seq * o_s0 + q_head[:, None] * o_s1 + d[None, :] * o_s2,
        out.to(out_ptr.dtype.element_ty),
        mask=q_mask,
    )


@triton.jit(repr=_partition_repr)
def _pa_shuffle_swa_partition_kernel(
    m_ws_ptr,
    l_ws_ptr,
    acc_ws_ptr,
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    blk_tbl_ptr,
    seq_lens_ptr,
    k_scale_ptr,
    v_scale_ptr,
    scale,
    sliding_window,
    part_size,
    q_s0,
    q_s1,
    q_s2,
    bt_s0,
    ks_s0,
    ks_s1,
    ks_s2,
    vs_s0,
    vs_s1,
    vs_s2,
    k_blk_stride,
    k_head_stride,
    v_blk_stride,
    v_head_stride,
    m_s0,
    m_s1,
    m_s2,
    m_s3,
    a_s0,
    a_s1,
    a_s2,
    a_s3,
    a_s4,
    HEAD: tl.constexpr,
    HEAD_POW2: tl.constexpr,
    BLOCK: tl.constexpr,
    X: tl.constexpr,
    GRP: tl.constexpr,
    GRP_POW2: tl.constexpr,
    SCALE_SCALAR: tl.constexpr,
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    part = tl.program_id(2)
    seq_len = tl.load(seq_lens_ptr + seq)
    win_start = 0
    if sliding_window > 0:
        ws = seq_len - sliding_window
        win_start = tl.where(ws > 0, ws, 0)

    ctx_start = win_start + part * part_size
    ctx_end = ctx_start + part_size
    ctx_end = tl.where(ctx_end < seq_len, ctx_end, seq_len)

    grp = tl.arange(0, GRP_POW2)
    d = tl.arange(0, HEAD_POW2)
    m_off = seq * m_s0 + kvh * m_s1 + grp * m_s2 + part * m_s3

    if ctx_start >= seq_len:
        # empty partition: l=0, the reduce skips it and never reads its acc
        tl.store(
            m_ws_ptr + m_off,
            tl.full([GRP_POW2], -float("inf"), tl.float32),
            mask=grp < GRP,
        )
        tl.store(l_ws_ptr + m_off, tl.zeros([GRP_POW2], tl.float32), mask=grp < GRP)
        return

    q_head = kvh * GRP + grp
    q_mask = (grp[:, None] < GRP) & (d[None, :] < HEAD)
    q = tl.load(
        q_ptr + seq * q_s0 + q_head[:, None] * q_s1 + d[None, :] * q_s2,
        mask=q_mask,
        other=0.0,
    ).to(tl.float32)

    m_i = tl.full([GRP_POW2], -float("inf"), tl.float32)
    l_i = tl.zeros([GRP_POW2], tl.float32)
    acc = tl.zeros([GRP_POW2, HEAD_POW2], tl.float32)
    ks_scalar = 1.0
    vs_scalar = 1.0
    if SCALE_SCALAR:
        ks_scalar = tl.load(k_scale_ptr)
        vs_scalar = tl.load(v_scale_ptr)

    for b in range(ctx_start // BLOCK, tl.cdiv(ctx_end, BLOCK)):
        block_id = tl.load(blk_tbl_ptr + seq * bt_s0 + b).to(tl.int64)
        m_i, l_i, acc = _pa_shuffle_swa_block(
            q,
            m_i,
            l_i,
            acc,
            k_cache_ptr,
            v_cache_ptr,
            k_scale_ptr,
            v_scale_ptr,
            ks_scalar,
            vs_scalar,
            block_id,
            kvh,
            b * BLOCK,
            ctx_start,
            ctx_end,
            scale,
            ks_s0,
            ks_s1,
            ks_s2,
            vs_s0,
            vs_s1,
            vs_s2,
            k_blk_stride,
            k_head_stride,
            v_blk_stride,
            v_head_stride,
            HEAD,
            HEAD_POW2,
            BLOCK,
            X,
            SCALE_SCALAR,
        )

    tl.store(m_ws_ptr + m_off, m_i, mask=grp < GRP)
    tl.store(l_ws_ptr + m_off, l_i, mask=grp < GRP)
    acc_off = (
        seq * a_s0 + kvh * a_s1 + grp[:, None] * a_s2 + part * a_s3 + d[None, :] * a_s4
    )
    tl.store(acc_ws_ptr + acc_off, acc, mask=(grp[:, None] < GRP) & (d[None, :] < HEAD))


@triton.jit(repr=_reduce_repr)
def _pa_shuffle_swa_reduce_kernel(
    out_ptr,
    m_ws_ptr,
    l_ws_ptr,
    acc_ws_ptr,
    n_part,
    o_s0,
    o_s1,
    o_s2,
    m_s0,
    m_s1,
    m_s2,
    m_s3,
    a_s0,
    a_s1,
    a_s2,
    a_s3,
    a_s4,
    HEAD: tl.constexpr,
    HEAD_POW2: tl.constexpr,
    GRP: tl.constexpr,
    GRP_POW2: tl.constexpr,
    NPART_POW2: tl.constexpr,
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    grp = tl.arange(0, GRP_POW2)
    parts = tl.arange(0, NPART_POW2)
    d = tl.arange(0, HEAD_POW2)
    gmask = grp < GRP
    load_m = gmask[:, None] & (parts[None, :] < n_part)

    m_off = seq * m_s0 + kvh * m_s1 + grp[:, None] * m_s2 + parts[None, :] * m_s3
    m_part = tl.load(m_ws_ptr + m_off, mask=load_m, other=-float("inf"))
    l_part = tl.load(l_ws_ptr + m_off, mask=load_m, other=0.0)
    # empty partitions (l = 0) never wrote their acc: mask them out of the load
    # so stale or uninitialized scratch can't come back as 0 * NaN
    live = load_m & (l_part > 0.0)

    g_m = tl.max(tl.where(live, m_part, -float("inf")), axis=1)
    factor = tl.where(live, tl.exp(m_part - g_m[:, None]), 0.0)
    L = tl.sum(factor * l_part, axis=1)

    acc_off = (
        seq * a_s0
        + kvh * a_s1
        + grp[:, None, None] * a_s2
        + parts[None, :, None] * a_s3
        + d[None, None, :] * a_s4
    )
    acc_part = tl.load(
        acc_ws_ptr + acc_off,
        mask=live[:, :, None] & (d[None, None, :] < HEAD),
        other=0.0,
    )
    acc = tl.sum(acc_part * factor[:, :, None], axis=1)
    out = tl.where(L[:, None] > 0.0, acc / L[:, None], 0.0)

    q_head = kvh * GRP + grp
    o_mask = (grp[:, None] < GRP) & (d[None, :] < HEAD)
    tl.store(
        out_ptr + seq * o_s0 + q_head[:, None] * o_s1 + d[None, :] * o_s2,
        out.to(out_ptr.dtype.element_ty),
        mask=o_mask,
    )


def _pick_n_part(
    eff, partition_size, base_grid, n_cu, split_threshold=8, min_split_ctx=1024
):
    """Number of context partitions. Stay one-pass unless splitting actually
    helps: either the base grid underfills the device, or one program would have
    to walk a long key range serially (>= split_threshold partitions worth).

    The reduce launch + fp32 workspace traffic is a fixed cost per call, so it
    only pays when the serial key walk is long. Below ``min_split_ctx`` effective
    context we always stay one-pass even if the grid underfills (short decode
    contexts: the one-pass walk is already cheap and an extra launch loses).
    The caller sizes the partitions so that all of them together cover eff."""
    p_work = max(1, (eff + partition_size - 1) // partition_size)
    if p_work == 1 or eff < min_split_ctx:
        return 1
    underfilled = base_grid < n_cu
    if underfilled:
        # split enough to roughly hit ~2x CU occupancy, capped by available work
        want = max(1, (2 * n_cu + base_grid - 1) // base_grid)
        return min(p_work, want)
    # device already full: only split when the serial walk is long enough that
    # the extra parallelism beats the workspace round-trip
    if p_work >= split_threshold:
        return p_work
    return 1


def _partition_size(eff, n_part, partition_size, block_size):
    # smallest block-aligned partition, at least partition_size, with
    # n_part * part >= eff (a capped n_part must still cover every key)
    per = (eff + n_part - 1) // n_part
    per = (per + block_size - 1) // block_size * block_size
    return max(partition_size, per)


# Arches where the split path is measured to win end to end. On gfx950 the
# one-pass grid underfills the device at long context and bf16 decode is fast
# enough that attention is a real slice of the step, so the split recovers ~2x;
# on gfx942 the dense decode is projection/MLP bound and the split is parity at
# best, so the auto path stays one-pass there. Override with a comma list in
# AITER_PA_SHUFFLE_SPLIT_ARCH (e.g. "gfx950,gfx942") when tuning a new part.
_SPLIT_ARCH_DEFAULT = ("gfx950",)


def _arch_allows_split():
    env = os.environ.get("AITER_PA_SHUFFLE_SPLIT_ARCH")
    names = (
        tuple(a.strip() for a in env.split(",") if a.strip())
        if env
        else _SPLIT_ARCH_DEFAULT
    )
    return arch_info.get_arch() in names


# fp32 scratch for the split path, one set per (device, stream) so calls in
# flight on different streams never share it, and calls on one stream are
# ordered by the stream. Buffers only grow. A replaced buffer stays referenced:
# a captured cudagraph can still point at it.
_WS_CACHE = {}


def _get_workspace(workspace, dev, need_m, need_a):
    import torch

    if workspace is None:
        key = (dev.index, torch.cuda.current_stream(dev).cuda_stream)
        workspace = _WS_CACHE.setdefault(key, {})
    n_m = need_m[0] * need_m[1] * need_m[2] * need_m[3]
    n_a = n_m * need_a[4]
    bufs = workspace.get("bufs")
    if (
        bufs is None
        or bufs[0].device != dev
        or bufs[0].numel() < n_m
        or bufs[2].numel() < n_a
    ):
        if bufs is not None and bufs[0].device == dev:
            workspace.setdefault("retired", []).append(bufs)
            n_m = max(n_m, 2 * bufs[0].numel())
            n_a = max(n_a, 2 * bufs[2].numel())
        bufs = tuple(
            torch.empty(n, dtype=torch.float32, device=dev) for n in (n_m, n_m, n_a)
        )
        workspace["bufs"] = bufs
    nm = need_m[0] * need_m[1] * need_m[2] * need_m[3]
    return (
        bufs[0][:nm].view(need_m),
        bufs[1][:nm].view(need_m),
        bufs[2][: nm * need_a[4]].view(need_a),
    )


def paged_attention_decode_shuffle_swa_perf(
    out,
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    scale,
    k_scale,
    v_scale,
    sliding_window,
    partition_size: int = 256,
    num_warps=None,
    max_seq_len: int | None = None,
    workspace=None,
    force_split=None,
    split_threshold: int = 8,
    min_split_ctx: int = 1024,
):
    """Flash-decoding split-K decode over the shuffled fp8 KV cache with a window.

    Splits the windowed key range into partitions of at least ``partition_size``
    keys across the grid so long context fills the device; falls back to the
    one-pass kernel when splitting wouldn't pay for itself (short context, full
    grid, or an arch where the split doesn't help end to end).

    Args:
        out: output, same shape as query, written in place.
        query: 16-bit query, [num_seqs, num_q_heads, head_size].
        key_cache, value_cache: fp8 caches in the shuffled layout above.
        block_tables: int32 [num_seqs, max_num_blocks].
        seq_lens: int32 [num_seqs]. A row with seq_len 0 gets zeros.
        scale: softmax scale.
        k_scale, v_scale: one scalar each, or per token [num_blocks,
            num_kv_heads, block_size].
        sliding_window: keys attended per sequence, <= 0 means all.
        partition_size: smallest partition, in keys.
        max_seq_len: host int max of seq_lens, saves a device sync.
        workspace: optional dict that holds the fp32 scratch. Without it a
            scratch per (device, stream) is used. Don't share one dict between
            calls that run at the same time on different streams.
        force_split: True or False forces the split or the one-pass path
            (tests), None picks one.

    Returns:
        out.
    """

    num_seqs, num_q_heads, head = query.shape
    num_kv_heads = key_cache.shape[1]
    block_size = key_cache.shape[3]
    x = key_cache.shape[4]
    grp = num_q_heads // num_kv_heads
    grp_pow2 = max(16, triton.next_power_of_2(grp))
    head_pow2 = triton.next_power_of_2(head)

    scale_scalar = k_scale.numel() == 1
    if scale_scalar:
        ks_s0 = ks_s1 = ks_s2 = 0
        vs_s0 = vs_s1 = vs_s2 = 0
    else:
        ks_s0, ks_s1, ks_s2 = k_scale.stride()
        vs_s0, vs_s1, vs_s2 = v_scale.stride()

    # max_seq_len is a host int in the real call path (vLLM already tracks it),
    # so avoid a per-step .item() device sync unless the caller didn't pass it.
    max_len = max_seq_len if max_seq_len is not None else int(seq_lens.max().item())
    eff = max_len if sliding_window <= 0 else min(sliding_window, max_len)
    base_grid = num_seqs * num_kv_heads
    if force_split is True:
        n_part = max(2, (eff + partition_size - 1) // partition_size)
    elif force_split is False or not _arch_allows_split():
        # auto path: only split on arches where it's measured to help; force_split
        # still exercises both paths everywhere for correctness testing.
        n_part = 1
    else:
        n_part = _pick_n_part(
            eff,
            partition_size,
            base_grid,
            get_num_sms(),
            split_threshold,
            min_split_ctx,
        )
    part_size = partition_size
    if n_part > 1 and force_split is not True:
        part_size = _partition_size(eff, n_part, partition_size, block_size)
        n_part = (eff + part_size - 1) // part_size

    _LOGGER.info(
        "PA_DECODE_SHUFFLE_SWA_PERF: q=%s k_cache=%s window=%s n_part=%s part_size=%s",
        tuple(query.shape),
        tuple(key_cache.shape),
        sliding_window,
        n_part,
        part_size,
    )

    kw = {} if num_warps is None else {"num_warps": num_warps}
    if n_part == 1:
        grid = (num_seqs, num_kv_heads)
        _pa_shuffle_swa_onepass_kernel[grid](
            out,
            query,
            key_cache,
            value_cache,
            block_tables,
            seq_lens,
            k_scale,
            v_scale,
            scale,
            sliding_window,
            query.stride(0),
            query.stride(1),
            query.stride(2),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            block_tables.stride(0),
            ks_s0,
            ks_s1,
            ks_s2,
            vs_s0,
            vs_s1,
            vs_s2,
            key_cache.stride(0),
            key_cache.stride(1),
            value_cache.stride(0),
            value_cache.stride(1),
            HEAD=head,
            HEAD_POW2=head_pow2,
            BLOCK=block_size,
            X=x,
            GRP=grp,
            GRP_POW2=grp_pow2,
            SCALE_SCALAR=scale_scalar,
            **kw,
        )
        return out

    # the partition dim is rounded up to a power of two so a growing context
    # reuses the same scratch view and the same reduce kernel
    npart_pow2 = triton.next_power_of_2(n_part)
    need_m = (num_seqs, num_kv_heads, grp, npart_pow2)
    need_a = (num_seqs, num_kv_heads, grp, npart_pow2, head_pow2)
    m_ws, l_ws, acc_ws = _get_workspace(workspace, query.device, need_m, need_a)

    grid_p = (num_seqs, num_kv_heads, n_part)
    _pa_shuffle_swa_partition_kernel[grid_p](
        m_ws,
        l_ws,
        acc_ws,
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        k_scale,
        v_scale,
        scale,
        sliding_window,
        part_size,
        query.stride(0),
        query.stride(1),
        query.stride(2),
        block_tables.stride(0),
        ks_s0,
        ks_s1,
        ks_s2,
        vs_s0,
        vs_s1,
        vs_s2,
        key_cache.stride(0),
        key_cache.stride(1),
        value_cache.stride(0),
        value_cache.stride(1),
        m_ws.stride(0),
        m_ws.stride(1),
        m_ws.stride(2),
        m_ws.stride(3),
        acc_ws.stride(0),
        acc_ws.stride(1),
        acc_ws.stride(2),
        acc_ws.stride(3),
        acc_ws.stride(4),
        HEAD=head,
        HEAD_POW2=head_pow2,
        BLOCK=block_size,
        X=x,
        GRP=grp,
        GRP_POW2=grp_pow2,
        SCALE_SCALAR=scale_scalar,
        **kw,
    )
    grid_r = (num_seqs, num_kv_heads)
    _pa_shuffle_swa_reduce_kernel[grid_r](
        out,
        m_ws,
        l_ws,
        acc_ws,
        n_part,
        out.stride(0),
        out.stride(1),
        out.stride(2),
        m_ws.stride(0),
        m_ws.stride(1),
        m_ws.stride(2),
        m_ws.stride(3),
        acc_ws.stride(0),
        acc_ws.stride(1),
        acc_ws.stride(2),
        acc_ws.stride(3),
        acc_ws.stride(4),
        HEAD=head,
        HEAD_POW2=head_pow2,
        GRP=grp,
        GRP_POW2=grp_pow2,
        NPART_POW2=npart_pow2,
    )
    return out
