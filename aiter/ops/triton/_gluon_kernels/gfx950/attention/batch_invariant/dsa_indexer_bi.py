# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Batch-invariant DSA indexer kernels (gfx950, Gluon): paged FP8 MQA logits and
the exact top-k. The arithmetic of one logit depends only on its query row and key."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.language.core import PropagateNan

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_PROPAGATE_NAN_ALL = gl.constexpr(PropagateNan.ALL)

# Contract: one MFMA shape and one head-chunk size for every launch.
MFMA_INSTR_SHAPE = gl.constexpr((32, 32, 64))
HEAD_CHUNK = gl.constexpr(32)
WARP_SIZE = gl.constexpr(64)
K_WIDTH = gl.constexpr(16)


@gluon.constexpr_function
def _offset_bases_to_blocked(offset_bases, contiguity, num_warps, warp_size, shape):
    # lg2(C) bases to reg, lg2(WS) to lane, lg2(NW) to warp, the rest to reg:
    # the blocked layout that lets the async copy fold into the shared store.
    rank = len(shape)
    lg2_c = contiguity.bit_length() - 1
    lg2_nw = num_warps.bit_length() - 1
    lg2_ws = warp_size.bit_length() - 1
    i = 0
    reg_bases = offset_bases[i : i + lg2_c]
    i += lg2_c
    lane_bases = offset_bases[i : i + lg2_ws]
    i += lg2_ws
    warp_bases = offset_bases[i : i + lg2_nw]
    i += lg2_nw
    warp_bases = warp_bases + [[0] * rank] * (lg2_nw - len(warp_bases))
    reg_bases = reg_bases + offset_bases[i:]
    return gl.DistributedLinearLayout(
        reg_bases=reg_bases,
        lane_bases=lane_bases,
        warp_bases=warp_bases,
        block_bases=[],
        shape=shape,
    )


@gluon.constexpr_function
def _k_tile_layouts(HEAD_SIZE, BLOCK_KV, NUM_WARPS, PADDED_LDS):
    # K tile [HEAD_SIZE, BLOCK_KV] fp8: (blocked layout of the copy, LDS layout).
    contiguity = 16
    if PADDED_LDS:
        lg2_hs = HEAD_SIZE.bit_length() - 1
        lg2_ts = BLOCK_KV.bit_length() - 1
        hs_lane = lg2_hs - (contiguity.bit_length() - 1)
        bases = [[1 << i, 0] for i in range(lg2_hs)] + [
            [0, 1 << ((i + hs_lane) % lg2_ts)] for i in range(lg2_ts)
        ]
        shared = gl.PaddedSharedLayout(
            interval_padding_pairs=[[1024, 16]],
            offset_bases=bases,
            cga_layout=[],
            shape=[HEAD_SIZE, BLOCK_KV],
        )
        blocked = _offset_bases_to_blocked(
            bases, contiguity, NUM_WARPS, 64, [HEAD_SIZE, BLOCK_KV]
        )
    else:
        hs_div = HEAD_SIZE // contiguity
        blocked = gl.BlockedLayout(
            size_per_thread=[contiguity, 1],
            threads_per_warp=[hs_div, 64 // hs_div],
            warps_per_cta=[1, NUM_WARPS],
            order=[0, 1],
        )
        shared = gl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0, 1])
    return blocked, shared


@gluon.constexpr_function
def _head_fold_plan(linear_layout, num_heads, block_kv):
    # Head bits held in registers are FMA-folded in a fixed serial order; the
    # head bits held across lanes are summed after the fold.
    head_bits = num_heads.bit_length() - 1
    reg_bases = [tuple(b) for b in linear_layout.reg_bases]
    summed, folded = [], []
    for bit in range(head_bits):
        if (1 << (head_bits - 1 - bit), 0) in reg_bases:
            folded.append(bit)
        else:
            summed.append(bit)
    depth = len(folded)
    bit_shape = tuple([2] * head_bits + [block_kv])
    bit_order = tuple(summed + [head_bits] + folded)
    folded_shape = tuple([1 << len(summed), block_kv, 1] + [2] * depth)
    return bit_shape, bit_order, folded_shape, depth, 1 << depth


@gluon.jit
def _leaf(x, IDX: gl.constexpr, DEPTH: gl.constexpr):
    for bit in gl.static_range(0, DEPTH):
        lo, hi = x.split()
        if (IDX // (2**bit)) % 2 == 0:
            x = lo
        else:
            x = hi
    return x


@gluon.jit
def _fold_heads(s, w_col, acc, HAS_ACC: gl.constexpr, mfma_layout: gl.constexpr):
    # acc += sum_h s[h, k] * w[h] over the register head bits, one serial FMA chain.
    CH: gl.constexpr = s.shape[0]
    BKV: gl.constexpr = s.shape[1]
    ll: gl.constexpr = gl.to_linear_layout(mfma_layout, [CH, BKV])
    plan: gl.constexpr = _head_fold_plan(ll, CH, BKV)
    bit_shape: gl.constexpr = plan[0]
    bit_order: gl.constexpr = plan[1]
    folded_shape: gl.constexpr = plan[2]
    depth: gl.constexpr = plan[3]
    leaves: gl.constexpr = plan[4]
    w = w_col.broadcast_to([CH, BKV])
    s = s.reshape(bit_shape).permute(bit_order).reshape(folded_shape)
    w = w.reshape(bit_shape).permute(bit_order).reshape(folded_shape)
    if not HAS_ACC:
        acc = _leaf(s, 0, depth) * _leaf(w, 0, depth)
    for i in gl.static_range(0, leaves):
        if HAS_ACC or i > 0:
            acc = gl.fma(_leaf(s, i, depth), _leaf(w, i, depth), acc)
    return acc


_logits_repr = make_kernel_repr(
    "_dsa_indexer_bi_logits_kernel",
    ["NUM_HEADS", "HEAD_SIZE", "CACHE_BS", "BLOCK_KV", "NUM_WARPS", "PADDED_LDS"],
)


@gluon.jit(
    repr=_logits_repr,
    do_not_specialize=["tiles_per_split", "max_len"],
)
def _dsa_indexer_bi_logits_kernel(
    q_ptr,  # fp8 e4m3 [T, H, D]
    w_ptr,  # fp32 [T, H]
    k_ptr,  # fp8 view of the index cache, block-flat [NB, CACHE_BS * (D + 4)]
    s_ptr,  # fp32 view of the same bytes
    bt_ptr,  # int32 [R, max_blocks]
    req_ptr,  # int32 [T]: block-table row of each query row
    len_ptr,  # int32 [T]: keys [0, len) of each row
    out_ptr,  # fp32 [T, >= max_len]
    stride_q_s,
    stride_q_h,
    stride_w_s,
    stride_w_h,
    stride_k_blk,
    stride_s_blk,
    stride_bt,
    stride_out_s,
    tiles_per_split,
    max_len,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    CACHE_BS: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PADDED_LDS: gl.constexpr,
):
    """logits[t, j] = k_scale[j] * sum_h w[t, h] * relu(q[t, h] . k[j]), j < len[t].

    Grid (T, splits). A split walks a fixed range of BLOCK_KV-key tiles; the tile,
    split and block placement only partition the columns.
    """
    gl.static_assert(CACHE_BS % BLOCK_KV == 0, "cache block must hold whole tiles")
    gl.static_assert(NUM_HEADS % HEAD_CHUNK == 0, "heads come in chunks of 32")
    NCHUNK: gl.constexpr = NUM_HEADS // HEAD_CHUNK
    SCALE_OFF: gl.constexpr = CACHE_BS * HEAD_SIZE // 4

    row = gl.program_id(0)
    split = gl.program_id(1)
    n = gl.minimum(gl.load(len_ptr + row), max_len)
    n_tiles = (n + BLOCK_KV - 1) // BLOCK_KV
    tile_lo = split * tiles_per_split
    if tile_lo >= n_tiles:
        return
    tile_hi = gl.minimum(tile_lo + tiles_per_split, n_tiles)
    col_hi = gl.minimum(n, tile_hi * BLOCK_KV)

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[32, 32, 64],
        transposed=False,
        warps_per_cta=[1, NUM_WARPS],
    )
    dot_a: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=K_WIDTH
    )
    dot_b: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=K_WIDTH
    )
    q_inner: gl.constexpr = HEAD_SIZE // 16
    layout_q: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[64 // q_inner, q_inner],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    col_layout: gl.constexpr = gl.SliceLayout(0, mfma_layout)

    # Q and the head weights of this row, per 32-head chunk.
    q_row = q_ptr + row.to(gl.int64) * stride_q_s
    w_row = w_ptr + row.to(gl.int64) * stride_w_s
    hq = gl.arange(0, HEAD_CHUNK, layout=gl.SliceLayout(1, layout_q))
    dq = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, layout_q))
    hw = gl.arange(0, HEAD_CHUNK, layout=gl.SliceLayout(1, mfma_layout))
    qs = ()
    ws = ()
    for c in gl.static_range(0, NCHUNK):
        qc = gl.amd.cdna4.buffer_load(
            ptr=q_row,
            offsets=((hq + c * HEAD_CHUNK) * stride_q_h)[:, None] + dq[None, :],
        )
        wc = gl.amd.cdna4.buffer_load(
            ptr=w_row, offsets=((hw + c * HEAD_CHUNK) * stride_w_h)[:, None]
        )
        qs = qs + (gl.convert_layout(qc, dot_a),)
        ws = ws + (wc,)

    # Double-buffered K tiles: global -> LDS by async copy.
    blocked: gl.constexpr = _k_tile_layouts(HEAD_SIZE, BLOCK_KV, NUM_WARPS, PADDED_LDS)[
        0
    ]
    shared: gl.constexpr = _k_tile_layouts(HEAD_SIZE, BLOCK_KV, NUM_WARPS, PADDED_LDS)[
        1
    ]
    k_lds = gl.allocate_shared_memory(
        k_ptr.type.element_ty, [2, HEAD_SIZE, BLOCK_KV], layout=shared
    )
    offs_d = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, blocked))[:, None]
    offs_n = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, blocked))[None, :]
    k_offsets = offs_d + offs_n * HEAD_SIZE
    bt_row = bt_ptr + gl.load(req_ptr + row).to(gl.int64) * stride_bt
    cols = gl.arange(0, BLOCK_KV, layout=col_layout)
    out_row = out_ptr + row.to(gl.int64) * stride_out_s
    last = tile_hi - 1

    for p in gl.static_range(0, 2):
        t = gl.minimum(tile_lo + p, last)
        blk = gl.load(bt_row + (t * BLOCK_KV) // CACHE_BS).to(gl.int64)
        off = (t * BLOCK_KV) % CACHE_BS
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            k_lds.index(p),
            k_ptr + blk * stride_k_blk + off * HEAD_SIZE,
            k_offsets,
            mask=offs_n < (col_hi - (tile_lo + p) * BLOCK_KV),
        )
        gl.amd.cdna4.async_copy.commit_group()

    for i in range(tile_lo, tile_hi):
        buf = (i - tile_lo) % 2
        c0 = i * BLOCK_KV
        blk = gl.load(bt_row + c0 // CACHE_BS).to(gl.int64)
        off = c0 % CACHE_BS
        scales = gl.amd.cdna4.buffer_load(
            ptr=s_ptr + blk * stride_s_blk + SCALE_OFF + off,
            offsets=cols,
            mask=cols < (col_hi - c0),
        )
        gl.amd.cdna4.async_copy.wait_group(1)
        kt = k_lds.index(buf).load(layout=dot_b)
        # Prefetch tile i + 2 into the buffer just read (masked past the split).
        tn = gl.minimum(i + 2, last)
        nblk = gl.load(bt_row + (tn * BLOCK_KV) // CACHE_BS).to(gl.int64)
        noff = (tn * BLOCK_KV) % CACHE_BS
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            k_lds.index(buf),
            k_ptr + nblk * stride_k_blk + noff * HEAD_SIZE,
            k_offsets,
            mask=offs_n < (col_hi - (i + 2) * BLOCK_KV),
        )
        gl.amd.cdna4.async_copy.commit_group()

        acc = None
        for c in gl.static_range(0, NCHUNK):
            sc = gl.amd.cdna4.mfma_scaled(
                a=qs[c],
                a_scale=None,
                a_format="e4m3",
                b=kt,
                b_scale=None,
                b_format="e4m3",
                acc=gl.zeros(
                    [HEAD_CHUNK, BLOCK_KV], dtype=gl.float32, layout=mfma_layout
                ),
            )
            sc = gl.maximum(sc, 0, propagate_nan=_PROPAGATE_NAN_ALL)
            acc = _fold_heads(sc, ws[c], acc, c > 0, mfma_layout)
        acc = gl.sum(acc, axis=2)
        acc = gl.sum(acc, axis=0)
        logit = gl.convert_layout(acc, col_layout) * scales
        gl.amd.cdna4.buffer_store(
            logit, ptr=out_row + c0, offsets=cols, mask=cols < (col_hi - c0)
        )
    gl.amd.cdna4.async_copy.wait_group(0)


@gluon.jit
def _ordered_key(x):
    # fp32 bits -> uint32 in the total order of the bit patterns (-0.0 < +0.0).
    bits = x.to(gl.int32, bitcast=True)
    u = bits.to(gl.uint32, bitcast=True)
    return gl.where(bits < 0, ~u, u | 0x80000000)


@gluon.jit
def _digit_ge(u, m, thr, shift, DBITS: gl.constexpr, layout2d: gl.constexpr):
    # ge[i, d] = m[i] and u[i] >= thr | d << shift; its column sums fall as d grows.
    d = gl.arange(0, 1 << DBITS, layout=gl.SliceLayout(0, layout2d)).to(gl.uint32)
    cand = thr | (d << shift)
    return (m[:, None] & (u[:, None] >= cand[None, :])).to(gl.int32)


@gluon.jit
def _digit_counts(u, m, thr, shift, DBITS: gl.constexpr, layout2d: gl.constexpr):
    return gl.sum(_digit_ge(u, m, thr, shift, DBITS, layout2d), axis=0)


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _cumsum(x, flat: gl.constexpr):
    # Inclusive prefix sum; the scan runs in a plain 1-D blocked layout.
    y = gl.associative_scan(gl.convert_layout(x, flat), 0, _add)
    return gl.convert_layout(y, x.type.layout)


@gluon.constexpr_function
def _flat_layout(n, num_warps):
    return gl.BlockedLayout([n // (64 * num_warps)], [64], [num_warps], [0])


@gluon.jit
def _topk_resident(
    row_ptr,
    out_row,
    n,
    sel,
    RES: gl.constexpr,
    DBITS: gl.constexpr,
    layout: gl.constexpr,
    NW: gl.constexpr,
):
    flat: gl.constexpr = _flat_layout(RES, NW)
    # Row of at most RES columns: one load, the select in registers.
    cols = gl.arange(0, RES, layout=gl.SliceLayout(1, layout))
    m = cols < n
    u = _ordered_key(gl.load(row_ptr + cols, mask=m, other=0.0))
    thr = (n * 0).to(gl.uint32)
    for r in gl.static_range(32 // DBITS):
        shift = 32 - DBITS * (r + 1)
        cnt = _digit_counts(u, m, thr, shift, DBITS, layout)
        nd = gl.sum((cnt >= sel).to(gl.int32), axis=0)
        thr = thr | ((nd - 1).to(gl.uint32) << shift)
    gt = m & (u > thr)
    eq = m & (u == thr)
    n_gt = gl.sum(gt.to(gl.int32), axis=0)
    n_eq = gl.sum(eq.to(gl.int32), axis=0)
    # Of the n_eq tied columns take the (sel - n_gt) largest.
    skip_eq = n_eq - (sel - n_gt)
    eq_rank = _cumsum(eq.to(gl.int32), flat) - 1
    take = gt | (eq & (eq_rank >= skip_eq))
    pos = _cumsum(take.to(gl.int32), flat) - 1
    gl.store(out_row + pos, cols, mask=take)


@gluon.jit
def _topk_streamed(
    row_ptr,
    out_row,
    n,
    sel,
    BLOCK: gl.constexpr,
    DBITS: gl.constexpr,
    layout: gl.constexpr,
    NW: gl.constexpr,
):
    flat: gl.constexpr = _flat_layout(BLOCK, NW)
    # Longer row: every radix round and the final passes stream BLOCK tiles.
    # Counts stay per thread across the tiles and are reduced once per pass.
    jb = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, layout))
    thr = (n * 0).to(gl.uint32)
    for r in gl.static_range(32 // DBITS):
        shift = 32 - DBITS * (r + 1)
        acc = gl.zeros([BLOCK, 1 << DBITS], dtype=gl.int32, layout=layout)
        for c0 in range(0, n, BLOCK):
            cols = c0 + jb
            m = cols < n
            u = _ordered_key(gl.load(row_ptr + cols, mask=m, other=0.0))
            acc += _digit_ge(u, m, thr, shift, DBITS, layout)
        cnt = gl.sum(acc, axis=0)
        nd = gl.sum((cnt >= sel).to(gl.int32), axis=0)
        thr = thr | ((nd - 1).to(gl.uint32) << shift)
    jf = gl.arange(0, BLOCK, layout=flat)
    gt_acc = gl.zeros([BLOCK], dtype=gl.int32, layout=flat)
    eq_acc = gl.zeros([BLOCK], dtype=gl.int32, layout=flat)
    for c0 in range(0, n, BLOCK):
        cols = c0 + jf
        m = cols < n
        u = _ordered_key(gl.load(row_ptr + cols, mask=m, other=0.0))
        gt_acc += (m & (u > thr)).to(gl.int32)
        eq_acc += (m & (u == thr)).to(gl.int32)
    n_gt = gl.sum(gt_acc, axis=0)
    n_eq = gl.sum(eq_acc, axis=0)
    skip_eq = n_eq - (sel - n_gt)
    eq_seen = n * 0
    written = n * 0
    for c0 in range(0, n, BLOCK):
        cols = c0 + jf
        m = cols < n
        u = _ordered_key(gl.load(row_ptr + cols, mask=m, other=0.0))
        eq = (m & (u == thr)).to(gl.int32)
        eq_rank = eq_seen + gl.associative_scan(eq, 0, _add) - 1
        take = (m & (u > thr)) | ((eq != 0) & (eq_rank >= skip_eq))
        pos_in = gl.associative_scan(take.to(gl.int32), 0, _add)
        gl.store(out_row + written + pos_in - 1, cols, mask=take)
        eq_seen += gl.sum(eq, axis=0)
        written += gl.sum(take.to(gl.int32), axis=0)


_topk_repr = make_kernel_repr(
    "_dsa_indexer_bi_topk_kernel", ["TOPK", "BLOCK", "RES", "DBITS", "NUM_WARPS"]
)


@gluon.jit(repr=_topk_repr, do_not_specialize=["max_len"])
def _dsa_indexer_bi_topk_kernel(
    logits_ptr,  # fp32 [T, >= max_len]
    stride_l_s,
    len_ptr,  # int32 [T]
    out_ptr,  # int32 [T, >= TOPK]
    stride_o_s,
    max_len,
    TOPK: gl.constexpr,
    BLOCK: gl.constexpr,
    RES: gl.constexpr,
    DBITS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Exact top-k of logits[t, 0:len[t]]: by the key order, ties to the larger column.

    Writes the selected columns ascending, then -1 to TOPK. An MSB-first radix
    select finds the unique cutoff key; rows of at most RES columns stay in
    registers, longer rows stream BLOCK tiles. The result is the same for any
    BLOCK, RES or DBITS.
    """
    l_res: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[RES // (64 * NUM_WARPS), 1 << DBITS],
        threads_per_warp=[64, 1],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    l_blk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[BLOCK // (64 * NUM_WARPS), 1 << DBITS],
        threads_per_warp=[64, 1],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    row = gl.program_id(0)
    n = gl.minimum(gl.maximum(gl.load(len_ptr + row), 0), max_len)
    sel = gl.minimum(n, TOPK)
    row_ptr = logits_ptr + row.to(gl.int64) * stride_l_s
    out_row = out_ptr + row.to(gl.int64) * stride_o_s
    if n > TOPK:
        if n <= RES:
            _topk_resident(row_ptr, out_row, n, sel, RES, DBITS, l_res, NUM_WARPS)
        else:
            _topk_streamed(row_ptr, out_row, n, sel, BLOCK, DBITS, l_blk, NUM_WARPS)
    jb = gl.arange(0, BLOCK, layout=gl.SliceLayout(1, l_blk))
    for c0 in range(0, TOPK, BLOCK):
        j = c0 + jb
        # Rows of at most TOPK columns take them all; then the -1 tail.
        val = gl.where(j < sel, j, -1)
        gl.store(out_row + j, val, mask=(j >= sel) | (n <= TOPK))
