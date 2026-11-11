# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""MiniMax sparse attention (MSA) over a k4v8 KV cache, Gluon, gfx950.

Same contract and cache layout as the Triton kernel
(aiter/ops/triton/_triton_kernels/attention/msa.py). Both K and V are stored
in aiter's (16, 16) preshuffle, which is the scaled MFMA's B-operand order, so
each block loads from HBM straight into dot-operand registers (no LDS):

  S^T-free form: S = Q K^T   mfma_scaled 16x16x128, A = MXFP8 Q, B = MXFP4 K
                 O += P V    mfma_scaled 16x16x128, A = fp8 P,   B = fp8 V

Q arrives pre-quantized as two MXFP8 terms (hi + lo, see msa_q_mxfp8x2_kernel)
so QK^T is two chained scaled MFMAs at about bf16 accuracy. One program is one
query row x the HG query heads of one KV head x one split of its top-k blocks.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

PAGE = gl.constexpr(128)
D = gl.constexpr(128)
HG = gl.constexpr(16)
K_BYTES = gl.constexpr(128 * 128 // 2)
KS_BYTES = gl.constexpr(128 * 128 // 32)
NUM_SCALES = gl.constexpr(128 // 32)


@gluon.jit
def _shuffled(offs_k, offs_n, K_EXTENT: gl.constexpr):
    """Byte of (k container, n) in a (16, 16)-preshuffled [N, K_EXTENT] tile."""
    return (
        (offs_k % 16)
        + (offs_k // 16) * 256
        + (offs_n % 16) * 16
        + (offs_n // 16) * (16 * K_EXTENT)
    )


@gluon.jit
def msa_gfx950_gluon_kernel(
    q8_ptr,  # e4m3 [rows, H, 2, D]: hi, lo terms (already scaled by sm_scale*log2e)
    qs_ptr,  # u8   [rows, H, 2, D/32]
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
    v_scale_ptr,
    stride_page,
    stride_head,
    stride_tk_h,
    stride_tk_t,
    stride_bt,
    stride_ot,
    stride_oh,
    NUM_HEADS: gl.constexpr,
    TOPK: gl.constexpr,
    SPLITS: gl.constexpr,
    DECODE_QL: gl.constexpr,
    P_SCALE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    MFMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, NUM_WARPS])
    A_OP: gl.constexpr = gl.DotOperandLayout(0, MFMA, 16)
    B_OP: gl.constexpr = gl.DotOperandLayout(1, MFMA, 16)
    Q_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(A_OP, [HG, NUM_SCALES])
    K_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(B_OP, [PAGE, NUM_SCALES])
    ROW: gl.constexpr = gl.SliceLayout(1, MFMA)
    COL: gl.constexpr = gl.SliceLayout(0, MFMA)
    PER: gl.constexpr = TOPK // SPLITS
    VEC: gl.constexpr = gl.BlockedLayout([1], [64], [NUM_WARPS], [0])

    row = gl.program_id(0)
    g = gl.program_id(1)
    sid = gl.program_id(2)
    if DECODE_QL > 0:
        req = row // DECODE_QL
        pos = gl.load(seq_lens_ptr + req) - DECODE_QL + row % DECODE_QL
    else:
        req = gl.load(query_req_ptr + row)
        pos = gl.load(query_pos_ptr + row)
    req = req.to(gl.int64)

    # Q terms, straight into the A operand and its scale layout.
    qh = gl.arange(0, HG, layout=gl.SliceLayout(1, A_OP))[:, None]
    qd = gl.arange(0, D, layout=gl.SliceLayout(0, A_OP))[None, :]
    q_row = q8_ptr + (row.to(gl.int64) * NUM_HEADS + g * HG) * (2 * D)
    q_offs = qh * (2 * D) + qd
    q_hi = gl.amd.cdna4.buffer_load(q_row, q_offs)
    q_lo = gl.amd.cdna4.buffer_load(q_row, q_offs + D)
    sh = gl.arange(0, HG, layout=gl.SliceLayout(1, Q_SCL))[:, None]
    ss = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, Q_SCL))[None, :]
    qs_row = qs_ptr + (row.to(gl.int64) * NUM_HEADS + g * HG) * (2 * NUM_SCALES)
    qs_offs = sh * (2 * NUM_SCALES) + ss
    qs_hi = gl.amd.cdna4.buffer_load(qs_row, qs_offs)
    qs_lo = gl.amd.cdna4.buffer_load(qs_row, qs_offs + NUM_SCALES)

    # Loop-invariant operand offsets inside one (page, KV head) record.
    kb = gl.arange(0, D // 2, layout=gl.SliceLayout(1, B_OP))[:, None]
    kn = gl.arange(0, PAGE, layout=gl.SliceLayout(0, B_OP))[None, :]
    k_offs = _shuffled(kb, kn, D // 2)
    st = gl.arange(0, PAGE, layout=gl.SliceLayout(1, K_SCL))[:, None]
    sg = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, K_SCL))[None, :]
    ks_offs = K_BYTES + st * NUM_SCALES + sg
    vt = gl.arange(0, PAGE, layout=gl.SliceLayout(1, B_OP))[:, None]
    vd = gl.arange(0, D, layout=gl.SliceLayout(0, B_OP))[None, :]
    v_offs = K_BYTES + KS_BYTES + _shuffled(vt, vd, PAGE)
    tok_n = gl.arange(0, PAGE, layout=COL)

    # This split's block ids and pages, read once.
    jj = gl.arange(0, PER, layout=VEC)
    blks = gl.load(topk_ptr + g * stride_tk_h + row.to(gl.int64) * stride_tk_t + sid * PER + jj)
    valids = (blks >= 0) & (blks * PAGE <= pos)
    blks = gl.where(valids, blks, 0)
    pages = gl.load(block_table_ptr + req * stride_bt + blks, mask=valids, other=0)

    m_i = gl.full([HG], float("-inf"), gl.float32, ROW)
    l_i = gl.zeros([HG], gl.float32, ROW)
    acc = gl.zeros([HG, D], gl.float32, MFMA)
    for j in range(PER):
        sel = jj == j
        blk = gl.sum(gl.where(sel, blks, 0), axis=0)
        valid = gl.sum(gl.where(sel & valids, 1, 0), axis=0) > 0
        page = gl.sum(gl.where(sel, pages, 0), axis=0)
        base = page.to(gl.int64) * stride_page + g * stride_head
        k = gl.amd.cdna4.buffer_load(cache_u8_ptr + base, k_offs)
        k_sc = gl.amd.cdna4.buffer_load(cache_u8_ptr + base, ks_offs)
        v = gl.amd.cdna4.buffer_load(cache_f8_ptr + base, v_offs)
        s = gl.zeros([HG, PAGE], gl.float32, MFMA)
        s = gl.amd.cdna4.mfma_scaled(q_hi, qs_hi, "e4m3", k, k_sc, "e2m1", s)
        s = gl.amd.cdna4.mfma_scaled(q_lo, qs_lo, "e4m3", k, k_sc, "e2m1", s)
        ok = valid & (blk * PAGE + tok_n <= pos)
        s = gl.where(ok[None, :], s, float("-inf"))
        m_new = gl.maximum(m_i, gl.max(s, axis=1))
        m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m_i - m_use)
        p8 = (gl.exp2(s - m_use[:, None]) * P_SCALE).to(gl.float8e4nv)
        l_i = l_i * alpha + gl.sum(p8.to(gl.float32), axis=1)
        acc = acc * alpha[:, None]
        acc = gl.amd.cdna4.mfma_scaled(
            gl.convert_layout(p8, A_OP), None, "e4m3", v, None, "e4m3", acc
        )
        m_i = m_new

    oh = gl.arange(0, HG, layout=ROW)[:, None]
    od = gl.arange(0, D, layout=COL)[None, :]
    if SPLITS == 1:
        v_scale = gl.load(v_scale_ptr)
        inv_l = gl.where(l_i > 0, v_scale / l_i, 0.0)
        o = (acc * inv_l[:, None]).to(gl.bfloat16)
        gl.store(
            out_ptr + row.to(gl.int64) * stride_ot + (g * HG + oh) * stride_oh + od, o
        )
    else:
        part = (row.to(gl.int64) * gl.num_programs(1) + g) * SPLITS + sid
        gl.store(part_o_ptr + part * (HG * D) + oh * D + od, acc)
        hh = gl.arange(0, HG, layout=ROW)
        gl.store(part_ml_ptr + part * (HG * 2) + hh * 2, m_i)
        gl.store(part_ml_ptr + part * (HG * 2) + hh * 2 + 1, l_i)


@gluon.jit
def msa_gfx950_gluon_prefill_kernel(
    q8_ptr,
    qs_ptr,
    cache_u8_ptr,
    cache_f8_ptr,
    uniq_ptr,
    count_ptr,
    tokblk_ptr,
    query_pos_ptr,
    out_ptr,
    v_scale_ptr,
    num_rows,
    stride_page,
    stride_head,
    stride_ot,
    stride_oh,
    NUM_HEADS: gl.constexpr,
    TOPK: gl.constexpr,
    TQ: gl.constexpr,
    P_SCALE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """TQ consecutive query rows x the HG heads of one KV head walk the union of
    the pages they selected (msa_union_kernel), so a page several rows share is
    loaded once. A row takes a page's keys only when it selected that page, up
    to its position."""
    M: gl.constexpr = TQ * HG
    C: gl.constexpr = TQ * TOPK
    MFMA: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, NUM_WARPS])
    A_OP: gl.constexpr = gl.DotOperandLayout(0, MFMA, 16)
    B_OP: gl.constexpr = gl.DotOperandLayout(1, MFMA, 16)
    Q_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(A_OP, [M, NUM_SCALES])
    K_SCL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(B_OP, [PAGE, NUM_SCALES])
    ROW: gl.constexpr = gl.SliceLayout(1, MFMA)
    COL: gl.constexpr = gl.SliceLayout(0, MFMA)

    grp = gl.program_id(0)
    g = gl.program_id(1)
    r0 = grp * TQ
    ubase = (g * gl.num_programs(0) + grp).to(gl.int64)
    count = gl.load(count_ptr + ubase)

    qm = gl.arange(0, M, layout=gl.SliceLayout(1, A_OP))[:, None]
    qd = gl.arange(0, D, layout=gl.SliceLayout(0, A_OP))[None, :]
    q_ok = (r0 + qm // HG) < num_rows
    q_offs = ((r0 + qm // HG).to(gl.int64) * NUM_HEADS + g * HG + qm % HG) * (2 * D) + qd
    q_hi = gl.load(q8_ptr + q_offs, mask=q_ok, other=0.0)
    q_lo = gl.load(q8_ptr + q_offs + D, mask=q_ok, other=0.0)
    sm = gl.arange(0, M, layout=gl.SliceLayout(1, Q_SCL))[:, None]
    ss = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, Q_SCL))[None, :]
    s_ok = (r0 + sm // HG) < num_rows
    qs_offs = ((r0 + sm // HG).to(gl.int64) * NUM_HEADS + g * HG + sm % HG) * (
        2 * NUM_SCALES
    ) + ss
    qs_hi = gl.load(qs_ptr + qs_offs, mask=s_ok, other=127)
    qs_lo = gl.load(qs_ptr + qs_offs + NUM_SCALES, mask=s_ok, other=127)

    kb = gl.arange(0, D // 2, layout=gl.SliceLayout(1, B_OP))[:, None]
    kn = gl.arange(0, PAGE, layout=gl.SliceLayout(0, B_OP))[None, :]
    k_offs = _shuffled(kb, kn, D // 2)
    st = gl.arange(0, PAGE, layout=gl.SliceLayout(1, K_SCL))[:, None]
    sg = gl.arange(0, NUM_SCALES, layout=gl.SliceLayout(0, K_SCL))[None, :]
    ks_offs = K_BYTES + st * NUM_SCALES + sg
    vt = gl.arange(0, PAGE, layout=gl.SliceLayout(1, B_OP))[:, None]
    vd = gl.arange(0, D, layout=gl.SliceLayout(0, B_OP))[None, :]
    v_offs = K_BYTES + KS_BYTES + _shuffled(vt, vd, PAGE)
    tok_n = gl.arange(0, PAGE, layout=COL)
    row_t = gl.arange(0, M, layout=ROW) // HG
    row_ok = (r0 + row_t) < num_rows
    row_pos = gl.load(query_pos_ptr + r0 + row_t, mask=row_ok, other=-1)
    tb_rows = tokblk_ptr + (ubase * TQ + row_t) * C

    m_i = gl.full([M], float("-inf"), gl.float32, ROW)
    l_i = gl.zeros([M], gl.float32, ROW)
    acc = gl.zeros([M, D], gl.float32, MFMA)
    for u in range(count):
        page = gl.load(uniq_ptr + ubase * C + u)
        row_blk = gl.load(tb_rows + u)
        base = page.to(gl.int64) * stride_page + g * stride_head
        k = gl.amd.cdna4.buffer_load(cache_u8_ptr + base, k_offs)
        k_sc = gl.amd.cdna4.buffer_load(cache_u8_ptr + base, ks_offs)
        v = gl.amd.cdna4.buffer_load(cache_f8_ptr + base, v_offs)
        s = gl.zeros([M, PAGE], gl.float32, MFMA)
        s = gl.amd.cdna4.mfma_scaled(q_hi, qs_hi, "e4m3", k, k_sc, "e2m1", s)
        s = gl.amd.cdna4.mfma_scaled(q_lo, qs_lo, "e4m3", k, k_sc, "e2m1", s)
        ok = (row_blk[:, None] >= 0) & (
            row_blk[:, None] * PAGE + tok_n[None, :] <= row_pos[:, None]
        )
        s = gl.where(ok, s, float("-inf"))
        m_new = gl.maximum(m_i, gl.max(s, axis=1))
        m_use = gl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = gl.exp2(m_i - m_use)
        p8 = (gl.exp2(s - m_use[:, None]) * P_SCALE).to(gl.float8e4nv)
        l_i = l_i * alpha + gl.sum(p8.to(gl.float32), axis=1)
        acc = acc * alpha[:, None]
        acc = gl.amd.cdna4.mfma_scaled(
            gl.convert_layout(p8, A_OP), None, "e4m3", v, None, "e4m3", acc
        )
        m_i = m_new

    v_scale = gl.load(v_scale_ptr)
    inv_l = gl.where(l_i > 0, v_scale / l_i, 0.0)
    o = (acc * inv_l[:, None]).to(gl.bfloat16)
    om = gl.arange(0, M, layout=ROW)[:, None]
    od = gl.arange(0, D, layout=COL)[None, :]
    o_row = r0 + om // HG
    gl.store(
        out_ptr + o_row.to(gl.int64) * stride_ot + (g * HG + om % HG) * stride_oh + od,
        o,
        mask=o_row < num_rows,
    )
