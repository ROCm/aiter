# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax sparse attention (MSA) over a k4v8 KV cache, Triton.

MSA is grouped-query attention restricted to the top-k 128-token blocks an
indexer picked per (query row, KV head). The sparse block is one KV page.

k4v8 KV cache, one record per (page, KV head), PAGE = 128 tokens, D = 128:

    bytes [0, 8192)        K, MXFP4 (e2m1, two per byte, the even element in
                           the low nibble), preshuffled for the scaled MFMA
                           B operand: token t, byte b at
                           (t // 16) * 1024 + (b // 32) * 512
                           + (b // 16 % 2) * 256 + (t % 16) * 16 + b % 16
                           (aiter's shuffle_weight (16, 16) layout per page)
    bytes [8192, 8704)     K scales, E8M0, one per 32 dims, row-major [t, D/32]
    bytes [8704, 25088)    V, fp8 e4m3 (OCP) of v / v_scale, as the PV MFMA's
                           preshuffled B: V^T [d, t] in shuffle_weight (16, 16)
                           tiles, dim d, token t at
                           (d // 16) * 2048 + (t // 32) * 512
                           + (t // 16 % 2) * 256 + (d % 16) * 16 + t % 16

Both preshuffles are the (16, 16) layout aiter's preshuffled Triton GEMMs read,
unshuffled in registers with the same reshape / permute, so the operands load
straight into the MFMA layout.

K is quantized along D, the QK dot dim, so QK^T runs on the scaled MFMA with
no dequant. V is quantized per tensor, so P V runs as an fp8 MFMA and v_scale
is applied once in the epilogue.
"""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.quant.quant import _mxfp4_quant_op

PAGE = 128
HEAD_DIM = 128
K_BYTES = PAGE * HEAD_DIM // 2
KS_BYTES = PAGE * HEAD_DIM // 32
V_BYTES = PAGE * HEAD_DIM
RECORD_BYTES = K_BYTES + KS_BYTES + V_BYTES
FP8_MAX = 448.0
# P is stored as fp8 scaled by this, so the small probabilities keep bits;
# l is summed from the same rounded values, so the factor cancels exactly.
P_SCALE = 256.0


@triton.jit
def _k_byte_offsets(t, b):
    return (t // 16) * 1024 + (b // 32) * 512 + (b // 16 % 2) * 256 + (t % 16) * 16 + b % 16


@triton.jit
def _v_offsets(t, d, D: tl.constexpr):
    return (d // 16) * 2048 + (t // 32) * 512 + (t // 16 % 2) * 256 + (d % 16) * 16 + t % 16


@triton.jit
def _mxfp8_quant(x, M: tl.constexpr, D: tl.constexpr):
    """fp32 [M, D] -> (e4m3 [M, D], E8M0 [M, D/32], the dequantized fp32).
    The scale is the smallest power of two that fits the block max in e4m3."""
    x3 = x.reshape(M, D // 32, 32)
    amax = tl.max(tl.abs(x3), axis=2)
    e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
    e = tl.clamp(e, -127.0, 127.0)
    x8 = (x3 * tl.exp2(-e)[:, :, None]).to(tl.float8e4nv)
    deq = (x8.to(tl.float32) * tl.exp2(e)[:, :, None]).reshape(M, D)
    return x8.reshape(M, D), (e + 127.0).to(tl.uint8), deq


@triton.jit
def msa_k4v8_store_kernel(
    k_ptr,
    v_ptr,
    slot_ptr,
    cache_u8_ptr,
    cache_f8_ptr,
    v_scale_ptr,
    num_tokens,
    stride_kt,
    stride_kh,
    stride_vt,
    stride_vh,
    stride_page,
    stride_head,
    PAGE: tl.constexpr,
    D: tl.constexpr,
    K_BYTES: tl.constexpr,
    KS_BYTES: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    """Quantize BLOCK_T tokens of one KV head into their slots (-1 = skip)."""
    pid = tl.program_id(0)
    h = tl.program_id(1)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    t_ok = offs_t < num_tokens
    slot = tl.load(slot_ptr + offs_t, mask=t_ok, other=-1).to(tl.int64)
    ok = t_ok & (slot >= 0)
    page = tl.where(ok, slot // PAGE, 0)
    t = tl.where(ok, slot % PAGE, 0).to(tl.int32)
    base = page * stride_page + h * stride_head
    offs_d = tl.arange(0, D)

    k = tl.load(
        k_ptr + offs_t[:, None] * stride_kt + h * stride_kh + offs_d[None, :],
        mask=ok[:, None],
        other=0.0,
    ).to(tl.float32)
    k_fp4, k_sc = _mxfp4_quant_op(k, D, BLOCK_T, 32)
    offs_b = tl.arange(0, D // 2)
    tl.store(
        cache_u8_ptr + base[:, None] + _k_byte_offsets(t[:, None], offs_b[None, :]),
        k_fp4,
        mask=ok[:, None],
    )
    offs_g = tl.arange(0, D // 32)
    tl.store(
        cache_u8_ptr + base[:, None] + K_BYTES + t[:, None] * (D // 32) + offs_g[None, :],
        k_sc,
        mask=ok[:, None],
    )

    inv_v = 1.0 / tl.load(v_scale_ptr)
    v = tl.load(
        v_ptr + offs_t[:, None] * stride_vt + h * stride_vh + offs_d[None, :],
        mask=ok[:, None],
        other=0.0,
    ).to(tl.float32)
    v = tl.clamp(v * inv_v, -448.0, 448.0).to(cache_f8_ptr.dtype.element_ty)
    tl.store(
        cache_f8_ptr
        + base[:, None]
        + K_BYTES
        + KS_BYTES
        + _v_offsets(t[:, None], offs_d[None, :], D),
        v,
        mask=ok[:, None],
    )


@triton.jit
def msa_gfx950_triton_kernel(
    q_ptr,
    cache_u8_ptr,
    cache_f8_ptr,
    topk_ptr,
    block_table_ptr,
    seq_lens_ptr,
    query_req_ptr,
    query_pos_ptr,
    out_ptr,
    part_o_ptr,
    part_ml_ptr,
    counter_ptr,
    v_scale_ptr,
    qk_scale_log2,
    stride_qt,
    stride_qh,
    stride_page,
    stride_head,
    stride_tk_h,
    stride_tk_t,
    stride_bt,
    stride_ot,
    stride_oh,
    HG: tl.constexpr,
    D: tl.constexpr,
    PAGE: tl.constexpr,
    TOPK: tl.constexpr,
    SPLITS: tl.constexpr,
    DECODE_QL: tl.constexpr,
    K_BYTES: tl.constexpr,
    KS_BYTES: tl.constexpr,
    Q_MODE: tl.constexpr,
    P_FP8: tl.constexpr,
    P_SCALE: tl.constexpr,
    NUM_STAGES: tl.constexpr,
):
    """One program = one query row x the HG query heads of one KV head x one
    split of the row's TOPK blocks.

    DECODE_QL > 0: row r is query row r % DECODE_QL of request r // DECODE_QL,
    at position seq_len - DECODE_QL + r % DECODE_QL. DECODE_QL == 0: the
    request and position come from query_req / query_pos (prefill).
    A row attends the tokens <= its position of its valid blocks; a row
    with none (seq_len 0 padding, all -1) gives 0.
    """
    row = tl.program_id(0)
    g = tl.program_id(1)
    sid = tl.program_id(2)
    if DECODE_QL > 0:
        req = row // DECODE_QL
        pos = tl.load(seq_lens_ptr + req) - DECODE_QL + row % DECODE_QL
    else:
        req = tl.load(query_req_ptr + row)
        pos = tl.load(query_pos_ptr + row)
    req = req.to(tl.int64)

    offs_h = tl.arange(0, HG)
    offs_d = tl.arange(0, D)
    offs_n = tl.arange(0, PAGE)
    q = tl.load(
        q_ptr
        + row.to(tl.int64) * stride_qt
        + (g * HG + offs_h)[:, None] * stride_qh
        + offs_d[None, :]
    ).to(tl.float32)
    # Softmax runs in base 2 with the scale folded into q.
    q = q * qk_scale_log2
    # Q_MODE 0: bf16 Q (K upcast to bf16); 1: MXFP8 Q; 2: MXFP8 Q as two terms
    # q_hi + q_lo, two chained scaled MFMAs (about bf16 accuracy).
    if Q_MODE == 0:
        q_op = q.to(tl.bfloat16)
        q_sc = None
        q_lo = q_op
        q_lo_sc = None
    else:
        q_op, q_sc, q_hi = _mxfp8_quant(q, HG, D)
        if Q_MODE == 2:
            q_lo, q_lo_sc, _ = _mxfp8_quant(q - q_hi, HG, D)
        else:
            q_lo = q_op
            q_lo_sc = q_sc

    m_i = tl.full([HG], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([HG], dtype=tl.float32)
    acc = tl.zeros([HG, D], dtype=tl.float32)
    PER: tl.constexpr = TOPK // SPLITS
    topk_row = topk_ptr + g * stride_tk_h + row.to(tl.int64) * stride_tk_t + sid * PER
    # Branch-free so the loop pipelines: an invalid block (-1, or past the
    # row's position) loads nothing and scores -inf.
    k_offs = (
        tl.arange(0, PAGE // 16)[:, None] * (D // 2 * 16)
        + tl.arange(0, D // 2 * 16)[None, :]
    )
    ks_offs = K_BYTES + offs_n[:, None] * (D // 32) + tl.arange(0, D // 32)[None, :]
    v_offs = (
        K_BYTES
        + KS_BYTES
        + tl.arange(0, D // 16)[:, None] * (PAGE * 16)
        + tl.arange(0, PAGE * 16)[None, :]
    )
    for j in tl.range(0, PER, num_stages=NUM_STAGES):
        blk = tl.load(topk_row + j)
        valid = (blk >= 0) & (blk * PAGE <= pos)
        blk = tl.where(valid, blk, 0)
        page = tl.load(block_table_ptr + req * stride_bt + blk, mask=valid, other=0)
        base = page.to(tl.int64) * stride_page + g * stride_head
        k = (
            tl.load(cache_u8_ptr + base + k_offs, mask=valid, other=0)
            .reshape(1, PAGE // 16, D // 64, 2, 16, 16)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(PAGE, D // 2)
            .trans(1, 0)
        )
        k_sc = tl.load(cache_u8_ptr + base + ks_offs, mask=valid, other=127)
        v = (
            tl.load(cache_f8_ptr + base + v_offs, mask=valid, other=0.0)
            .reshape(1, D // 16, PAGE // 32, 2, 16, 16)
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(D, PAGE)
            .trans(1, 0)
        )
        if Q_MODE == 0:
            s = tl.dot_scaled(q_op, None, "bf16", k, k_sc, "e2m1")
        else:
            s = tl.dot_scaled(q_op, q_sc, "e4m3", k, k_sc, "e2m1")
            if Q_MODE == 2:
                s = tl.dot_scaled(q_lo, q_lo_sc, "e4m3", k, k_sc, "e2m1", s)
        tok = blk * PAGE + offs_n
        s = tl.where(valid & (tok <= pos)[None, :], s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, axis=1))
        # Until a row has seen a key, m stays -inf: exponentiate against 0.
        m_use = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_use)
        p = tl.exp2(s - m_use[:, None])
        if P_FP8:
            p_op = (p * P_SCALE).to(tl.float8e4nv)
            pv = tl.dot(p_op, v)
        else:
            p_op = p.to(tl.bfloat16)
            pv = tl.dot(p_op, v.to(tl.bfloat16))
        l_i = l_i * alpha + tl.sum(p_op.to(tl.float32), axis=1)
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    out_rows = (
        out_ptr
        + row.to(tl.int64) * stride_ot
        + (g * HG + offs_h)[:, None] * stride_oh
        + offs_d[None, :]
    )
    if SPLITS == 1:
        v_scale = tl.load(v_scale_ptr)
        inv_l = tl.where(l_i > 0, v_scale / l_i, 0.0)
        tl.store(out_rows, (acc * inv_l[:, None]).to(out_ptr.dtype.element_ty))
    else:
        # Unnormalized partials: [rows, Hkv, SPLITS, HG, D] and [.., HG, 2].
        # The last split of a (row, KV head) to arrive merges them in split
        # order (so the result does not depend on arrival order) and resets
        # the counter for the next call.
        nkv = tl.num_programs(1)
        pair = row.to(tl.int64) * nkv + g
        part = pair * SPLITS + sid
        tl.store(
            part_o_ptr + part * (HG * D) + offs_h[:, None] * D + offs_d[None, :], acc
        )
        tl.store(part_ml_ptr + part * (HG * 2) + offs_h * 2, m_i)
        tl.store(part_ml_ptr + part * (HG * 2) + offs_h * 2 + 1, l_i)
        tl.debug_barrier()
        arrived = tl.atomic_add(counter_ptr + pair, 1, sem="acq_rel", scope="gpu")
        if arrived == SPLITS - 1:
            tl.store(counter_ptr + pair, 0)
            _merge_splits(
                part_o_ptr, part_ml_ptr, pair * SPLITS, out_rows, v_scale_ptr, HG, D, SPLITS
            )


@triton.jit
def _merge_splits(
    part_o_ptr,
    part_ml_ptr,
    part0,
    out_rows,
    v_scale_ptr,
    HG: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Merge the SPLITS partials starting at part0, in split order."""
    offs_s = tl.arange(0, SPLITS)
    offs_h = tl.arange(0, HG)
    offs_d = tl.arange(0, D)
    ml = part_ml_ptr + (part0 + offs_s)[:, None] * (HG * 2) + offs_h[None, :] * 2
    m = tl.load(ml, cache_modifier=".cg")
    l = tl.load(ml + 1, cache_modifier=".cg")
    m_max = tl.max(m, axis=0)
    w = tl.where(m > float("-inf"), tl.exp2(m - m_max[None, :]), 0.0)
    l_tot = tl.sum(w * l, axis=0)
    o = tl.zeros([HG, D], dtype=tl.float32)
    for s in tl.static_range(SPLITS):
        w_s = tl.sum(tl.where(offs_s[:, None] == s, w, 0.0), axis=0)
        o_s = tl.load(
            part_o_ptr + (part0 + s) * (HG * D) + offs_h[:, None] * D + offs_d[None, :],
            cache_modifier=".cg",
        )
        o += w_s[:, None] * o_s
    v_scale = tl.load(v_scale_ptr)
    inv_l = tl.where(l_tot > 0, v_scale / l_tot, 0.0)
    tl.store(out_rows, (o * inv_l[:, None]).to(out_rows.dtype.element_ty))


@triton.jit
def msa_q_mxfp8_kernel(
    q_ptr,
    q8_ptr,
    qs_ptr,
    qk_scale_log2,
    num_pairs,
    num_heads,
    stride_qt,
    stride_qh,
    D: tl.constexpr,
    QT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """Q * qk_scale_log2 of BLOCK (row, head) pairs as QT MXFP8 terms (hi, then
    lo = q - hi when QT == 2): q8 [rows, H, QT, D] e4m3, qs [rows, H, QT, D/32]
    E8M0 (the Gluon MSA's Q)."""
    pair = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = pair < num_pairs
    row = (pair // num_heads).to(tl.int64)
    h = pair % num_heads
    offs_d = tl.arange(0, D)
    q = tl.load(
        q_ptr + row[:, None] * stride_qt + h[:, None] * stride_qh + offs_d[None, :],
        mask=ok[:, None],
        other=0.0,
    )
    q = q.to(tl.float32) * qk_scale_log2
    hi, hi_sc, hi_deq = _mxfp8_quant(q, BLOCK, D)
    base = pair.to(tl.int64)[:, None]
    offs_s = tl.arange(0, D // 32)[None, :]
    tl.store(q8_ptr + base * (QT * D) + offs_d[None, :], hi, mask=ok[:, None])
    tl.store(qs_ptr + base * (QT * (D // 32)) + offs_s, hi_sc, mask=ok[:, None])
    if QT == 2:
        lo, lo_sc, _ = _mxfp8_quant(q - hi_deq, BLOCK, D)
        tl.store(q8_ptr + base * (2 * D) + D + offs_d[None, :], lo, mask=ok[:, None])
        tl.store(
            qs_ptr + base * (2 * (D // 32)) + D // 32 + offs_s, lo_sc, mask=ok[:, None]
        )


@triton.jit
def msa_merge_kernel(
    part_o_ptr,
    part_ml_ptr,
    out_ptr,
    v_scale_ptr,
    stride_ot,
    stride_oh,
    HG: tl.constexpr,
    D: tl.constexpr,
    SPLITS: tl.constexpr,
):
    """Merge the SPLITS partials of one (row, KV head) (Gluon MSA splits)."""
    row = tl.program_id(0)
    g = tl.program_id(1)
    pair = row.to(tl.int64) * tl.num_programs(1) + g
    offs_h = tl.arange(0, HG)
    offs_d = tl.arange(0, D)
    out_rows = (
        out_ptr
        + row.to(tl.int64) * stride_ot
        + (g * HG + offs_h)[:, None] * stride_oh
        + offs_d[None, :]
    )
    _merge_splits(part_o_ptr, part_ml_ptr, pair * SPLITS, out_rows, v_scale_ptr, HG, D, SPLITS)


@triton.jit
def msa_union_kernel(
    topk_ptr,
    block_table_ptr,
    query_req_ptr,
    query_pos_ptr,
    uniq_ptr,  # int32 [Hkv, groups, TQ * TOPK]: the group's distinct pages
    count_ptr,  # int32 [Hkv, groups]
    tokblk_ptr,  # int32 [Hkv, groups, TQ, TQ * TOPK]: row t's block for unique slot u, -1 = not picked
    num_rows,
    stride_tk_h,
    stride_tk_t,
    stride_bt,
    TOPK: tl.constexpr,
    TQ: tl.constexpr,
):
    """The union of the pages TQ consecutive query rows selected (one KV head),
    in first-pick order, for the Gluon MSA prefill walk."""
    grp = tl.program_id(0)
    g = tl.program_id(1)
    ngrp = tl.num_programs(0)
    C: tl.constexpr = TQ * TOPK
    c = tl.arange(0, C)
    t_c = c // TOPK
    row = grp * TQ + t_c
    r_ok = row < num_rows
    blk = tl.load(
        topk_ptr + g * stride_tk_h + row.to(tl.int64) * stride_tk_t + c % TOPK,
        mask=r_ok,
        other=-1,
    )
    req = tl.load(query_req_ptr + row, mask=r_ok, other=0).to(tl.int64)
    pos = tl.load(query_pos_ptr + row, mask=r_ok, other=-1)
    valid = r_ok & (blk >= 0) & (blk * 128 <= pos)
    page = tl.load(block_table_ptr + req * stride_bt + tl.where(valid, blk, 0), mask=valid, other=-1)
    page = tl.where(valid, page, -1)
    same = (page[:, None] == page[None, :]) & valid[:, None] & valid[None, :]
    earlier = c[None, :] < c[:, None]
    first = valid & (tl.sum(tl.where(same & earlier, 1, 0), axis=1) == 0)
    slot = tl.cumsum(first.to(tl.int32), axis=0) - 1
    base = (g * ngrp + grp).to(tl.int64)
    tl.store(uniq_ptr + base * C + slot, page, mask=first)
    tl.store(count_ptr + base, tl.sum(first.to(tl.int32), axis=0))
    # tb[t, c]: row t's block for candidate c's page (-1 when t did not pick it)
    for t in tl.static_range(TQ):
        mine = same & (t_c[:, None] == t)  # [c' of row t, c]
        tb = tl.max(tl.where(mine, blk[:, None], -1), axis=0)
        tl.store(tokblk_ptr + (base * TQ + t) * C + slot, tb, mask=first)
