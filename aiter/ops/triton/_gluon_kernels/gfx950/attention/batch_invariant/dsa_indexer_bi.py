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


@gluon.jit
def _logits_rows(
    q_ptr,
    w_ptr,
    k_ptr,
    s_ptr,
    bt_ptr,
    req_ptr,
    len_ptr,
    out_ptr,
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
    row0,
    split,
    NR: gl.constexpr,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    CACHE_BS: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PADDED_LDS: gl.constexpr,
):
    # Rows row0 .. row0 + NR - 1 of one block-table row: each K tile is loaded
    # once and used by all NR rows. Each row's arithmetic is the one-row arithmetic.
    NCHUNK: gl.constexpr = NUM_HEADS // HEAD_CHUNK
    SCALE_OFF: gl.constexpr = CACHE_BS * HEAD_SIZE // 4
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

    lens = ()
    n = 0
    for r in gl.static_range(0, NR):
        n_r = gl.minimum(gl.load(len_ptr + row0 + r), max_len)
        lens = lens + (n_r,)
        n = gl.maximum(n, n_r)
    n_tiles = (n + BLOCK_KV - 1) // BLOCK_KV
    tile_lo = split * tiles_per_split
    if tile_lo < n_tiles:
        tile_hi = gl.minimum(tile_lo + tiles_per_split, n_tiles)
        col_hi = gl.minimum(n, tile_hi * BLOCK_KV)

        # Q and the head weights of each row, per 32-head chunk.
        hq = gl.arange(0, HEAD_CHUNK, layout=gl.SliceLayout(1, layout_q))
        dq = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(0, layout_q))
        hw = gl.arange(0, HEAD_CHUNK, layout=gl.SliceLayout(1, mfma_layout))
        qs = ()
        ws = ()
        for r in gl.static_range(0, NR):
            q_row = q_ptr + (row0 + r).to(gl.int64) * stride_q_s
            w_row = w_ptr + (row0 + r).to(gl.int64) * stride_w_s
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
        blocked: gl.constexpr = _k_tile_layouts(
            HEAD_SIZE, BLOCK_KV, NUM_WARPS, PADDED_LDS
        )[0]
        shared: gl.constexpr = _k_tile_layouts(
            HEAD_SIZE, BLOCK_KV, NUM_WARPS, PADDED_LDS
        )[1]
        k_lds = gl.allocate_shared_memory(
            k_ptr.type.element_ty, [2, HEAD_SIZE, BLOCK_KV], layout=shared
        )
        offs_d = gl.arange(0, HEAD_SIZE, layout=gl.SliceLayout(1, blocked))[:, None]
        offs_n = gl.arange(0, BLOCK_KV, layout=gl.SliceLayout(0, blocked))[None, :]
        k_offsets = offs_d + offs_n * HEAD_SIZE
        bt_row = bt_ptr + gl.load(req_ptr + row0).to(gl.int64) * stride_bt
        cols = gl.arange(0, BLOCK_KV, layout=col_layout)
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

            for r in gl.static_range(0, NR):
                acc = None
                for c in gl.static_range(0, NCHUNK):
                    sc = gl.amd.cdna4.mfma_scaled(
                        a=qs[r * NCHUNK + c],
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
                    acc = _fold_heads(sc, ws[r * NCHUNK + c], acc, c > 0, mfma_layout)
                acc = gl.sum(acc, axis=2)
                acc = gl.sum(acc, axis=0)
                logit = gl.convert_layout(acc, col_layout) * scales
                hi_r = gl.minimum(lens[r], tile_hi * BLOCK_KV)
                gl.amd.cdna4.buffer_store(
                    logit,
                    ptr=out_ptr + (row0 + r).to(gl.int64) * stride_out_s + c0,
                    offsets=cols,
                    mask=cols < (hi_r - c0),
                )
        gl.amd.cdna4.async_copy.wait_group(0)


_logits_repr = make_kernel_repr(
    "_dsa_indexer_bi_logits_kernel",
    [
        "NUM_HEADS",
        "HEAD_SIZE",
        "CACHE_BS",
        "BLOCK_KV",
        "ROWS",
        "NUM_WARPS",
        "PADDED_LDS",
    ],
)


@gluon.jit(
    repr=_logits_repr,
    do_not_specialize=["num_rows", "tiles_per_split", "max_len"],
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
    num_rows,
    tiles_per_split,
    max_len,
    NUM_HEADS: gl.constexpr,
    HEAD_SIZE: gl.constexpr,
    CACHE_BS: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    ROWS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    PADDED_LDS: gl.constexpr,
):
    """logits[t, j] = k_scale[j] * sum_h w[t, h] * relu(q[t, h] . k[j]), j < len[t].

    Grid (cdiv(T, ROWS), splits, ROWS). A group of ROWS rows of one block-table
    row runs in the z = 0 program, which loads each K tile once; the others exit.
    Rows of other groups run alone, row ROWS * i + z.
    A split walks a fixed range of BLOCK_KV-key tiles. Rows, tiles, splits and
    block placement only partition the work.
    """
    gl.static_assert(CACHE_BS % BLOCK_KV == 0, "cache block must hold whole tiles")
    gl.static_assert(NUM_HEADS % HEAD_CHUNK == 0, "heads come in chunks of 32")
    gl.static_assert(ROWS == 1 or ROWS == 2 or ROWS == 4, "1, 2 or 4 rows per program")
    lead = gl.program_id(0) * ROWS
    split = gl.program_id(1)
    row = lead + gl.program_id(2)
    if ROWS > 1:
        req0 = gl.load(req_ptr + lead)
        group = lead + ROWS <= num_rows
        for r in gl.static_range(1, ROWS):
            nxt = gl.minimum(lead + r, num_rows - 1)
            group = group & (gl.load(req_ptr + nxt) == req0)
        if group:
            if row == lead:
                _logits_rows(
                    q_ptr,
                    w_ptr,
                    k_ptr,
                    s_ptr,
                    bt_ptr,
                    req_ptr,
                    len_ptr,
                    out_ptr,
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
                    lead,
                    split,
                    ROWS,
                    NUM_HEADS,
                    HEAD_SIZE,
                    CACHE_BS,
                    BLOCK_KV,
                    NUM_WARPS,
                    PADDED_LDS,
                )
        elif row < num_rows:
            _logits_rows(
                q_ptr,
                w_ptr,
                k_ptr,
                s_ptr,
                bt_ptr,
                req_ptr,
                len_ptr,
                out_ptr,
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
                row,
                split,
                1,
                NUM_HEADS,
                HEAD_SIZE,
                CACHE_BS,
                BLOCK_KV,
                NUM_WARPS,
                PADDED_LDS,
            )
    else:
        _logits_rows(
            q_ptr,
            w_ptr,
            k_ptr,
            s_ptr,
            bt_ptr,
            req_ptr,
            len_ptr,
            out_ptr,
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
            row,
            split,
            1,
            NUM_HEADS,
            HEAD_SIZE,
            CACHE_BS,
            BLOCK_KV,
            NUM_WARPS,
            PADDED_LDS,
        )


# Top-k: MSB-first radix select with 8-bit digits (4 rounds) finds the unique
# cutoff key. All counts are exact integers, so the result does not depend on
# how a row is split into chunks or programs.
DIGIT_BITS = gl.constexpr(8)
NUM_BINS = gl.constexpr(256)
NUM_ROUNDS = gl.constexpr(4)


@gluon.jit
def _ordered_key(x):
    # fp32 bits -> uint32 in the total order of the bit patterns (-0.0 < +0.0).
    bits = x.to(gl.int32, bitcast=True)
    u = bits.to(gl.uint32, bitcast=True)
    return gl.where(bits < 0, ~u, u | 0x80000000)


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
    # 4 contiguous elements per thread, repeated: coalesced 16-byte loads.
    return gl.BlockedLayout(
        [max(1, min(4, n // (64 * num_warps)))], [64], [num_warps], [0]
    )


@gluon.jit
def _excl_prefix(v, tri: gl.constexpr):
    # Exclusive prefix sum of a short vector: a triangular masked sum.
    vc = gl.convert_layout(v, gl.SliceLayout(0, tri))
    i = gl.arange(0, v.shape[0], layout=gl.SliceLayout(1, tri))
    j = gl.arange(0, v.shape[0], layout=gl.SliceLayout(0, tri))
    s = gl.sum(gl.where(j[None, :] < i[:, None], vc[None, :], 0), axis=1)
    return gl.convert_layout(s, v.type.layout)


@gluon.jit
def _round_hist(u, m, thr, ROUND: gl.constexpr, hist_layout: gl.constexpr):
    # Histogram of digit ROUND of the keys whose higher digits equal thr's.
    shift: gl.constexpr = 32 - DIGIT_BITS * (ROUND + 1)
    if ROUND > 0:
        m = m & ((u >> (shift + DIGIT_BITS)) == (thr >> (shift + DIGIT_BITS)))
    digit = ((u >> shift) & (NUM_BINS - 1)).to(gl.int32)
    # Integer adds in LDS: the counts do not depend on the order of the adds.
    smem = gl.allocate_shared_memory(
        gl.int32, [NUM_BINS], layout=gl.SwizzledSharedLayout(1, 1, 1, [0])
    )
    smem.store(gl.zeros([NUM_BINS], dtype=gl.int32, layout=hist_layout))
    gl.barrier()
    ones = gl.full(digit.shape, 1, dtype=gl.int32, layout=digit.type.layout)
    smem.atomic_scatter_add(ones, digit, 0, mask=m)
    gl.barrier()
    return smem.load(hist_layout)


@gluon.jit
def _pick_digit(total, need, hist_layout: gl.constexpr):
    # Largest digit d with count(digit >= d) >= need; also count(digit > d).
    ge = gl.sum(total, axis=0) - _cumsum(total, hist_layout) + total
    d = gl.sum((ge >= need).to(gl.int32), axis=0) - 1
    bins = gl.arange(0, NUM_BINS, layout=hist_layout)
    n_gt = gl.sum(gl.where(bins > d, total, 0), axis=0)
    return d, n_gt


@gluon.jit
def _take_and_store(u, cols, m, thr, skip_eq, eq_before, base, out_row, flat):
    # Keys above thr, and the tied keys of global tie rank >= skip_eq. One scan
    # of packed (above, tied) counts; a tile has fewer than 2**16 columns.
    gt = (m & (u > thr)).to(gl.int32)
    eq = (m & (u == thr)).to(gl.int32)
    packed = gt + (eq << 16)
    excl = _cumsum(packed, flat) - packed
    eq_x = eq_before + (excl >> 16)
    taken_before = gl.maximum(eq_x - skip_eq, 0) - gl.maximum(eq_before - skip_eq, 0)
    take = (gt != 0) | ((eq != 0) & (eq_x >= skip_eq))
    gl.store(out_row + base + (excl & 0xFFFF) + taken_before, cols, mask=take)
    n_eq = gl.sum(eq, axis=0)
    taken = gl.maximum(eq_before + n_eq - skip_eq, 0) - gl.maximum(
        eq_before - skip_eq, 0
    )
    return gl.sum(gt, axis=0) + taken, n_eq


_topk_repr = make_kernel_repr(
    "_dsa_indexer_bi_topk_kernel", ["TOPK", "RES", "NUM_WARPS"]
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
    RES: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Exact top-k of rows of at most RES columns, one program per row.

    Order: the key order, ties to the larger column. Writes the selected columns
    ascending, then -1 to TOPK. Rows longer than RES are left to the chunked kernels.
    """
    flat: gl.constexpr = _flat_layout(RES, NUM_WARPS)
    hist_layout: gl.constexpr = _flat_layout(NUM_BINS, NUM_WARPS)
    row = gl.program_id(0)
    n = gl.minimum(gl.maximum(gl.load(len_ptr + row), 0), max_len)
    sel = gl.minimum(n, TOPK)
    row_ptr = logits_ptr + row.to(gl.int64) * stride_l_s
    out_row = out_ptr + row.to(gl.int64) * stride_o_s
    if (n > TOPK) & (n <= RES):
        cols = gl.arange(0, RES, layout=flat)
        m = cols < n
        u = _ordered_key(gl.load(row_ptr + cols, mask=m, other=0.0))
        thr = (n * 0).to(gl.uint32)
        need = sel
        for r in gl.static_range(NUM_ROUNDS):
            h = _round_hist(u, m, thr, r, hist_layout)
            d, n_gt = _pick_digit(h, need, hist_layout)
            need = need - n_gt
            thr = thr | (d.to(gl.uint32) << (32 - DIGIT_BITS * (r + 1)))
        n_eq = gl.sum((m & (u == thr)).to(gl.int32), axis=0)
        _take_and_store(u, cols, m, thr, n_eq - need, 0, 0, out_row, flat)
    j = gl.arange(0, TOPK, layout=_flat_layout(TOPK, NUM_WARPS))
    # Rows of at most TOPK columns take them all; then the -1 tail.
    gl.store(out_row + j, gl.where(j < sel, j, -1), mask=(j >= sel) | (n <= TOPK))


_hist_repr = make_kernel_repr(
    "_dsa_indexer_bi_topk_hist_kernel", ["ROUND", "RES", "CHUNK", "NUM_WARPS"]
)


@gluon.jit(repr=_hist_repr, do_not_specialize=["max_len", "stride_h_s", "stride_h_r"])
def _dsa_indexer_bi_topk_hist_kernel(
    logits_ptr,  # fp32 [T, >= max_len]
    stride_l_s,
    len_ptr,  # int32 [T]
    state_ptr,  # int32 [T, 4]: thr, need, skip_eq
    hist_ptr,  # int32 [T, NUM_ROUNDS, chunks, NUM_BINS]
    stride_h_s,
    stride_h_r,
    max_len,
    ROUND: gl.constexpr,
    RES: gl.constexpr,
    CHUNK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Digit histogram of one CHUNK of a row longer than RES, for radix round ROUND."""
    flat: gl.constexpr = _flat_layout(CHUNK, NUM_WARPS)
    hist_layout: gl.constexpr = _flat_layout(NUM_BINS, NUM_WARPS)
    row = gl.program_id(0)
    chunk = gl.program_id(1)
    c0 = chunk * CHUNK
    n = gl.minimum(gl.maximum(gl.load(len_ptr + row), 0), max_len)
    if (n > RES) & (c0 < n):
        cols = c0 + gl.arange(0, CHUNK, layout=flat)
        m = cols < n
        x = gl.load(
            logits_ptr + row.to(gl.int64) * stride_l_s + cols, mask=m, other=0.0
        )
        thr = gl.load(state_ptr + row * 4).to(gl.uint32, bitcast=True)
        h = _round_hist(_ordered_key(x), m, thr, ROUND, hist_layout)
        bins = gl.arange(0, NUM_BINS, layout=hist_layout)
        h_row = hist_ptr + row.to(gl.int64) * stride_h_s + ROUND * stride_h_r
        gl.store(h_row + chunk * NUM_BINS + bins, h)


_select_repr = make_kernel_repr(
    "_dsa_indexer_bi_topk_select_kernel", ["ROUND", "TOPK", "RES", "CHUNK"]
)


@gluon.jit
def _hist_tile(h_ptr, c, nc, bins):
    return gl.load(
        h_ptr + c[:, None] * NUM_BINS + bins[None, :], mask=(c < nc)[:, None], other=0
    )


@gluon.jit(
    repr=_select_repr,
    do_not_specialize=["max_len", "stride_h_s", "stride_h_r", "stride_a_s"],
)
def _dsa_indexer_bi_topk_select_kernel(
    len_ptr,  # int32 [T]
    state_ptr,  # int32 [T, 4]
    hist_ptr,  # int32 [T, NUM_ROUNDS, chunks, NUM_BINS]
    stride_h_s,
    stride_h_r,
    aux_ptr,  # int32 [T, 2, chunks]: output offset, then tie offset, of each chunk
    stride_a_s,
    max_len,
    ROUND: gl.constexpr,
    TOPK: gl.constexpr,
    RES: gl.constexpr,
    CHUNK: gl.constexpr,
):
    """Sum the chunk histograms of a row and pick the digit of ROUND.

    After the last round, it computes each chunk's output offset and tie offset
    from the histograms of all rounds, in chunk order.
    """
    NW: gl.constexpr = 4
    CB: gl.constexpr = 16
    # Chunks on the warps, bins on the lanes: per-chunk sums stay in a warp.
    tile: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [NW, 1], [1, 0])
    tri: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [1, NW], [1, 0])
    flat_b: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    shift: gl.constexpr = 32 - DIGIT_BITS * (ROUND + 1)
    row = gl.program_id(0)
    n = gl.minimum(gl.maximum(gl.load(len_ptr + row), 0), max_len)
    if n > RES:
        nc = (n + CHUNK - 1) // CHUNK
        h_row = hist_ptr + row.to(gl.int64) * stride_h_s
        bins = gl.arange(0, NUM_BINS, layout=gl.SliceLayout(0, tile))
        ci = gl.arange(0, CB, layout=gl.SliceLayout(1, tile))
        if ROUND == 0:
            thr = (n * 0).to(gl.uint32)
            need = n * 0 + TOPK
        else:
            thr = gl.load(state_ptr + row * 4).to(gl.uint32, bitcast=True)
            need = gl.load(state_ptr + row * 4 + 1)
        acc = gl.zeros([CB, NUM_BINS], dtype=gl.int32, layout=tile)
        for cb in range(0, nc, CB):
            acc += _hist_tile(h_row + ROUND * stride_h_r, cb + ci, nc, bins)
        total = gl.convert_layout(gl.sum(acc, 0), flat_b)
        d, n_gt = _pick_digit(total, need, flat_b)
        need = need - n_gt
        thr = thr | (d.to(gl.uint32) << shift)
        gl.store(state_ptr + row * 4, thr.to(gl.int32, bitcast=True))
        gl.store(state_ptr + row * 4 + 1, need)
        if ROUND == NUM_ROUNDS - 1:
            n_eq = gl.sum(
                gl.where(gl.arange(0, NUM_BINS, layout=flat_b) == d, total, 0), 0
            )
            skip_eq = n_eq - need
            gl.store(state_ptr + row * 4 + 2, skip_eq)
            a_row = aux_ptr + row.to(gl.int64) * stride_a_s
            gt_done = n * 0
            eq_done = n * 0
            for cb in range(0, nc, CB):
                c = cb + ci
                # Keys above thr: a higher digit in some round, equal before it.
                gt = gl.zeros([CB], dtype=gl.int32, layout=gl.SliceLayout(1, tile))
                for r in gl.static_range(NUM_ROUNDS - 1):
                    dr = ((thr >> (32 - DIGIT_BITS * (r + 1))) & (NUM_BINS - 1)).to(
                        gl.int32
                    )
                    t = _hist_tile(h_row + r * stride_h_r, c, nc, bins)
                    gt += gl.sum(gl.where(bins[None, :] > dr, t, 0), 1)
                t = _hist_tile(h_row + ROUND * stride_h_r, c, nc, bins)
                gt += gl.sum(gl.where(bins[None, :] > d, t, 0), 1)
                eq = gl.sum(gl.where(bins[None, :] == d, t, 0), 1)
                eq_before = eq_done + _excl_prefix(eq, tri)
                take = gt + gl.maximum(gl.minimum(eq, eq_before + eq - skip_eq), 0)
                base = gt_done + _excl_prefix(take, tri)
                cm = c < nc
                gl.store(a_row + c, base, mask=cm)
                gl.store(a_row + stride_a_s // 2 + c, eq_before, mask=cm)
                gt_done += gl.sum(take, axis=0)
                eq_done += gl.sum(eq, axis=0)


_write_repr = make_kernel_repr(
    "_dsa_indexer_bi_topk_write_kernel", ["RES", "CHUNK", "BLOCK", "NUM_WARPS"]
)


@gluon.jit(repr=_write_repr, do_not_specialize=["max_len", "stride_a_s"])
def _dsa_indexer_bi_topk_write_kernel(
    logits_ptr,  # fp32 [T, >= max_len]
    stride_l_s,
    len_ptr,  # int32 [T]
    state_ptr,  # int32 [T, 4]
    aux_ptr,  # int32 [T, 2, chunks]
    stride_a_s,
    out_ptr,  # int32 [T, >= TOPK]
    stride_o_s,
    max_len,
    RES: gl.constexpr,
    CHUNK: gl.constexpr,
    BLOCK: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Write the selected columns of one CHUNK of a row longer than RES, BLOCK at a time."""
    flat: gl.constexpr = _flat_layout(BLOCK, NUM_WARPS)
    row = gl.program_id(0)
    chunk = gl.program_id(1)
    c0 = chunk * CHUNK
    n = gl.minimum(gl.maximum(gl.load(len_ptr + row), 0), max_len)
    if (n > RES) & (c0 < n):
        thr = gl.load(state_ptr + row * 4).to(gl.uint32, bitcast=True)
        skip_eq = gl.load(state_ptr + row * 4 + 2)
        a_row = aux_ptr + row.to(gl.int64) * stride_a_s
        base = gl.load(a_row + chunk)
        eq_before = gl.load(a_row + stride_a_s // 2 + chunk)
        row_ptr = logits_ptr + row.to(gl.int64) * stride_l_s
        out_row = out_ptr + row.to(gl.int64) * stride_o_s
        jb = gl.arange(0, BLOCK, layout=flat)
        for b0 in range(c0, gl.minimum(c0 + CHUNK, n), BLOCK):
            cols = b0 + jb
            m = cols < n
            u = _ordered_key(gl.load(row_ptr + cols, mask=m, other=0.0))
            taken, n_eq = _take_and_store(
                u, cols, m, thr, skip_eq, eq_before, base, out_row, flat
            )
            base += taken
            eq_before += n_eq
