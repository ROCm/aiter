# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()

# Flash-decoding (split-K) sliding-window paged decode over the shuffled fp8 KV
# layout used by the asm pa_fwd path. Same cache layout and window semantics as
# pa_decode_shuffle, but the windowed key range is split across PARTITION_SIZE
# chunks so the long-context case fills the GPU instead of serialising one
# program per (seq, kv head). A second pass reduces the per-partition softmax
# state. For a single partition (small window) the host calls the one-pass
# kernel directly so we don't pay the workspace round-trip.
#
# Shuffled layout (x = 16 // elem_size, = 16 for fp8):
#   K: [num_blocks, num_kv_heads, head_size // x, block_size, x]
#   V: [num_blocks, num_kv_heads, block_size // x, head_size, x]
#   K[d, slot] -> (d // x) * block_size * x + slot * x + (d % x)
#   V[slot, d] -> (slot // x) * head_size * x + d * x + (slot % x)


@triton.jit
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
    s = tl.arange(0, BLOCK)
    n_blk = tl.cdiv(seq_len, BLOCK)
    start_blk = win_start // BLOCK
    if SCALE_SCALAR:
        ks_scalar = tl.load(k_scale_ptr)
        vs_scalar = tl.load(v_scale_ptr)

    for b in range(start_blk, n_blk):
        block_id = tl.load(blk_tbl_ptr + seq * bt_s0 + b).to(tl.int64)
        pos = b * BLOCK + s
        valid = (pos < seq_len) & (pos >= win_start)
        k_base = block_id * k_blk_stride + kvh * k_head_stride
        v_base = block_id * v_blk_stride + kvh * v_head_stride
        ld_mask = (s[:, None] < BLOCK) & (d[None, :] < HEAD)
        k_off = (
            k_base + (d[None, :] // X) * (BLOCK * X) + s[:, None] * X + (d[None, :] % X)
        )
        v_off = (
            v_base + (s[:, None] // X) * (HEAD * X) + d[None, :] * X + (s[:, None] % X)
        )
        k = tl.load(k_cache_ptr + k_off, mask=ld_mask, other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + v_off, mask=ld_mask, other=0.0).to(tl.float32)
        if SCALE_SCALAR:
            k = k * ks_scalar
            v = v * vs_scalar
        else:
            sc_off = block_id * ks_s0 + kvh * ks_s1 + s * ks_s2
            ks = tl.load(k_scale_ptr + sc_off, mask=s < BLOCK, other=0.0)
            vs = tl.load(v_scale_ptr + sc_off, mask=s < BLOCK, other=0.0)
            k = k * ks[:, None]
            v = v * vs[:, None]
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        m_i = m_new

    out = acc / l_i[:, None]
    tl.store(
        out_ptr + seq * o_s0 + q_head[:, None] * o_s1 + d[None, :] * o_s2,
        out.to(out_ptr.dtype.element_ty),
        mask=q_mask,
    )


@triton.jit
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
    q_s0,
    q_s1,
    q_s2,
    bt_s0,
    ks_s0,
    ks_s1,
    ks_s2,
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
    PART: tl.constexpr,
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    part = tl.program_id(2)
    seq_len = tl.load(seq_lens_ptr + seq)
    win_start = 0
    if sliding_window > 0:
        ws = seq_len - sliding_window
        win_start = tl.where(ws > 0, ws, 0)

    ctx_start = win_start + part * PART
    ctx_end = ctx_start + PART
    ctx_end = tl.where(ctx_end < seq_len, ctx_end, seq_len)

    grp = tl.arange(0, GRP_POW2)
    d = tl.arange(0, HEAD_POW2)
    m_off = seq * m_s0 + kvh * m_s1 + grp * m_s2 + part * m_s3

    if ctx_start >= seq_len:
        # empty partition: m=-inf, l=0 so the reduce ignores it
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
    s = tl.arange(0, BLOCK)
    start_blk = ctx_start // BLOCK
    end_blk = tl.cdiv(ctx_end, BLOCK)
    if SCALE_SCALAR:
        ks_scalar = tl.load(k_scale_ptr)
        vs_scalar = tl.load(v_scale_ptr)

    for b in range(start_blk, end_blk):
        block_id = tl.load(blk_tbl_ptr + seq * bt_s0 + b).to(tl.int64)
        pos = b * BLOCK + s
        valid = (pos < ctx_end) & (pos >= ctx_start)
        k_base = block_id * k_blk_stride + kvh * k_head_stride
        v_base = block_id * v_blk_stride + kvh * v_head_stride
        ld_mask = (s[:, None] < BLOCK) & (d[None, :] < HEAD)
        k_off = (
            k_base + (d[None, :] // X) * (BLOCK * X) + s[:, None] * X + (d[None, :] % X)
        )
        v_off = (
            v_base + (s[:, None] // X) * (HEAD * X) + d[None, :] * X + (s[:, None] % X)
        )
        k = tl.load(k_cache_ptr + k_off, mask=ld_mask, other=0.0).to(tl.float32)
        v = tl.load(v_cache_ptr + v_off, mask=ld_mask, other=0.0).to(tl.float32)
        if SCALE_SCALAR:
            k = k * ks_scalar
            v = v * vs_scalar
        else:
            sc_off = block_id * ks_s0 + kvh * ks_s1 + s * ks_s2
            ks = tl.load(k_scale_ptr + sc_off, mask=s < BLOCK, other=0.0)
            vs = tl.load(v_scale_ptr + sc_off, mask=s < BLOCK, other=0.0)
            k = k * ks[:, None]
            v = v * vs[:, None]
        scores = tl.dot(q, tl.trans(k)) * scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        m_new = tl.maximum(m_i, tl.max(scores, axis=1))
        p = tl.exp(scores - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None] + tl.dot(p.to(tl.float32), v)
        m_i = m_new

    tl.store(m_ws_ptr + m_off, m_i, mask=grp < GRP)
    tl.store(l_ws_ptr + m_off, l_i, mask=grp < GRP)
    acc_off = (
        seq * a_s0 + kvh * a_s1 + grp[:, None] * a_s2 + part * a_s3 + d[None, :] * a_s4
    )
    tl.store(acc_ws_ptr + acc_off, acc, mask=(grp[:, None] < GRP) & (d[None, :] < HEAD))


@triton.jit
def _pa_shuffle_swa_reduce_kernel(
    out_ptr,
    m_ws_ptr,
    l_ws_ptr,
    acc_ws_ptr,
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
    NPART: tl.constexpr,
    NPART_POW2: tl.constexpr,
):
    seq = tl.program_id(0)
    kvh = tl.program_id(1)
    grp = tl.arange(0, GRP_POW2)
    parts = tl.arange(0, NPART_POW2)
    d = tl.arange(0, HEAD_POW2)
    gmask = grp < GRP
    pmask = parts < NPART
    load_m = gmask[:, None] & pmask[None, :]

    m_off = seq * m_s0 + kvh * m_s1 + grp[:, None] * m_s2 + parts[None, :] * m_s3
    m_part = tl.load(m_ws_ptr + m_off, mask=load_m, other=-float("inf"))
    l_part = tl.load(l_ws_ptr + m_off, mask=load_m, other=0.0)

    g_m = tl.max(m_part, axis=1)
    factor = tl.where(load_m, tl.exp(m_part - g_m[:, None]), 0.0)
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
        mask=gmask[:, None, None] & pmask[None, :, None],
        other=0.0,
    )
    acc = tl.sum(acc_part * factor[:, :, None], axis=1)
    out = acc / L[:, None]

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
    contexts: the one-pass walk is already cheap and an extra launch loses)."""
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


_N_CU = {}

# Reused fp32 scratch for the split path. vLLM calls decode every step, so a
# fresh torch.empty per call (allocator + free) shows up as net e2e overhead;
# keep one buffer per (device, shape) and grow it as the decode context does.
_WS_CACHE = {}


def _cu_count(device):
    import torch

    idx = device.index if device.index is not None else torch.cuda.current_device()
    n = _N_CU.get(idx)
    if n is None:
        n = torch.cuda.get_device_properties(idx).multi_processor_count
        _N_CU[idx] = n
    return n


# Arches where the split path is measured to win end to end. On gfx950 the
# one-pass grid underfills the device at long context and bf16 decode is fast
# enough that attention is a real slice of the step, so the split recovers ~2x;
# on gfx942 the dense decode is projection/MLP bound and the split is parity at
# best, so the auto path stays one-pass there. Override with a comma list in
# AITER_PA_SHUFFLE_SPLIT_ARCH (e.g. "gfx950,gfx942") when tuning a new part.
_SPLIT_ARCH_DEFAULT = ("gfx950",)
_SPLIT_ARCH = {}


def _arch_allows_split(device):
    import os
    import torch

    idx = device.index if device.index is not None else torch.cuda.current_device()
    allow = _SPLIT_ARCH.get(idx)
    if allow is None:
        env = os.environ.get("AITER_PA_SHUFFLE_SPLIT_ARCH")
        names = (
            tuple(a.strip() for a in env.split(",") if a.strip())
            if env
            else _SPLIT_ARCH_DEFAULT
        )
        arch = torch.cuda.get_device_properties(idx).gcnArchName.split(":")[0]
        allow = arch in names
        _SPLIT_ARCH[idx] = allow
    return allow


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
    max_seq_len: int = None,
    workspace=None,
    force_split=None,
    split_threshold: int = 8,
    min_split_ctx: int = 1024,
):
    """Flash-decoding split-K decode over the shuffled fp8 KV cache with a window.

    Splits the windowed key range into ``partition_size`` chunks across the grid
    so long context fills the device; falls back to the one-pass kernel (the PR1
    path) when splitting wouldn't pay for itself (short context, full grid, or an
    arch where the split doesn't help end to end). ``workspace`` (a dict cache)
    can be passed to reuse the fp32 scratch tensors across calls. ``force_split``
    overrides the gating (used by the correctness tests to exercise both paths on
    any arch).
    """
    import torch

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
    else:
        ks_s0, ks_s1, ks_s2 = k_scale.stride()

    # max_seq_len is a host int in the real call path (vLLM already tracks it),
    # so avoid a per-step .item() device sync unless the caller didn't pass it.
    max_len = max_seq_len if max_seq_len is not None else int(seq_lens.max().item())
    eff = max_len if sliding_window <= 0 else min(sliding_window, max_len)
    base_grid = num_seqs * num_kv_heads
    n_cu = _cu_count(query.device)
    if force_split is True:
        n_part = max(2, (eff + partition_size - 1) // partition_size)
    elif force_split is False:
        n_part = 1
    elif not _arch_allows_split(query.device):
        # auto path: only split on arches where it's measured to help; force_split
        # still exercises both paths everywhere for correctness testing.
        n_part = 1
    else:
        n_part = _pick_n_part(
            eff, partition_size, base_grid, n_cu, split_threshold, min_split_ctx
        )

    if n_part == 1:
        grid = (num_seqs, num_kv_heads)
        kw = {} if num_warps is None else {"num_warps": num_warps}
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

    dev = query.device
    need_m = (num_seqs, num_kv_heads, grp, n_part)
    need_a = (num_seqs, num_kv_heads, grp, n_part, head_pow2)
    if workspace is None:
        workspace = _WS_CACHE.setdefault(dev.index, {})
    if workspace.get("shape_m") != need_m:
        m_ws = torch.empty(need_m, dtype=torch.float32, device=dev)
        l_ws = torch.empty(need_m, dtype=torch.float32, device=dev)
        acc_ws = torch.empty(need_a, dtype=torch.float32, device=dev)
        workspace.update(shape_m=need_m, m=m_ws, l=l_ws, acc=acc_ws)
    else:
        m_ws, l_ws, acc_ws = workspace["m"], workspace["l"], workspace["acc"]

    grid_p = (num_seqs, num_kv_heads, n_part)
    kw = {} if num_warps is None else {"num_warps": num_warps}
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
        query.stride(0),
        query.stride(1),
        query.stride(2),
        block_tables.stride(0),
        ks_s0,
        ks_s1,
        ks_s2,
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
        PART=partition_size,
        **kw,
    )
    grid_r = (num_seqs, num_kv_heads)
    _pa_shuffle_swa_reduce_kernel[grid_r](
        out,
        m_ws,
        l_ws,
        acc_ws,
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
        NPART=n_part,
        NPART_POW2=triton.next_power_of_2(n_part),
    )
    return out
