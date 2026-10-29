# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon (gfx950) sparse MLA for a whole-row fp8 cache with one per-tensor scale
("fp8_scalar") and a 512-wide head with no separate rope buffer (DeepSeek-V4
geometry: rope inside the row, V = the whole row), fp8 dots.

Two kernels, picked by the launcher (sparse_mla.py, _fp8_fast_path):

    _sparse_mla_fp8_wide    64 heads per program (each key gathered once per 64
        heads): 4 waves x 16 heads in QK (v_mfma_scale 16x16x128, unit scales),
        PV v_mfma_scale 32x32x64 with each wave owning 128 dims. QK runs in two
        32-key halves so one half's K operand is live at a time. For launches
        with at least one workgroup per CU.
    _sparse_mla_fp8_narrow  G = 8, 16 or 32 heads per program, waves split the
        keys in QK and the dims in PV (16x16x32 fp8 PV at G <= 16, 32x32x64 at
        32; G = 8 runs on 16-row tiles), deferred row sum, split-K with
        _sparse_mla_fp8_reduce. For small batches and few heads.

Shared by both:
  * a 2-slot ring of [64, 512] fp8 tiles filled by buffer_load ... lds (async
    direct-to-LDS) one tile ahead, in LDS images whose K (ds_read_b128) and V^T
    (ds_read_b64_tr_b8) reads are bank-conflict-free on gfx950 (_layouts);
    the gather layout's register order puts dims before keys, so the next
    gather's row ids never force a vmcnt(0) behind the gathers in flight;
  * online softmax in base 2 with a lagging max (moves only when beaten by
    2^LAZY) and a rare, wave-uniform rescale; log2(448 / 2^LAZY) is folded into
    the subtracted max so P comes out scaled for e4m3 (exp2 of one FMA);
  * q loaded straight into the MFMA operand layout (fp8 q) or quantized in the
    prologue with one scale per head (bf16 q);
  * the per-tensor kv scale folds into the row scale (K side) and the final
    normalization (V side); P is fp8 for the PV dot.
Measured design notes: asm_vs_gluon/asm_analysis/design/10_fp8.md.
"""

import functools
import math

import torch
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

LOG2E = 1.4426950408889634
LOG2_448 = 8.807354922057604


@gluon.jit
def _rows(idx_row, k, n, kg, HAS_INVALID: gl.constexpr):
    """Row ids of one tile in the gather layout; past-the-end keys clamp onto the
    row's last key (the score mask drops them), -1 sentinels onto row 0."""
    rid = gl.amd.cdna4.buffer_load(idx_row, gl.minimum(k + kg, n - 1))
    if HAS_INVALID:
        rid = gl.maximum(rid, 0)
    return rid


@gluon.jit
def _gather(dst, kv_ptr, rid, cg, BIG: gl.constexpr):
    if BIG:
        # pools past 2 GB: per-lane 64-bit addresses (buffer offsets are i32)
        gl.amd.cdna4.async_copy.global_load_to_shared(
            dst, kv_ptr + rid.to(gl.int64)[:, None] * 512 + cg[None, :])
    else:
        gl.amd.cdna4.async_copy.buffer_load_to_shared(dst, kv_ptr, rid[:, None] * 512 + cg[None, :])
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _mask(S, idx_row, k0, kb, end, kcol, HAS_INVALID: gl.constexpr):
    """-inf past the row's end (tail tile only) and on -1 sentinels."""
    BN: gl.constexpr = S.shape[1]
    if HAS_INVALID:
        kk = kb + kcol
        ok = (k0 + kcol < end) & (gl.amd.cdna4.buffer_load(idx_row, kk, mask=k0 + kcol < end, other=-1) >= 0)
        S = gl.where(ok[None, :], S, float("-inf"))
    elif k0 + BN > end:
        S = gl.where((k0 + kcol < end)[None, :], S, float("-inf"))
    return S


@gluon.jit
def _q_fp8(q_ptr, q_stride0, q_stride1, row, h0, G: gl.constexpr, GM: gl.constexpr,
           q_op: gl.constexpr):
    hq = gl.arange(0, GM, layout=gl.SliceLayout(1, q_op))
    cq = gl.arange(0, 512, layout=gl.SliceLayout(0, q_op))
    q = gl.amd.cdna4.buffer_load(q_ptr + row.to(gl.int64) * q_stride0 + h0 * q_stride1,
                                 hq[:, None] * q_stride1 + cq[None, :],
                                 mask=(hq < G)[:, None], other=0)
    return q.to(gl.float8e4nv, bitcast=True)


@gluon.jit
def _q_bf16(q_ptr, q_stride0, q_stride1, row, h0, G: gl.constexpr, GM: gl.constexpr,
            q_op: gl.constexpr, qk_l: gl.constexpr):
    """bf16 q quantized to e4m3 in the prologue, one scale per head."""
    NW: gl.constexpr = gl.num_warps()
    ql: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [NW, 1], [1, 0])
    hq = gl.arange(0, GM, layout=gl.SliceLayout(1, ql))
    cq = gl.arange(0, 512, layout=gl.SliceLayout(0, ql))
    q = gl.load(q_ptr + row.to(gl.int64) * q_stride0 + (h0 + hq)[:, None] * q_stride1 + cq[None, :],
                mask=(hq < G)[:, None], other=0.0).to(gl.float32)
    s = gl.maximum(gl.max(gl.abs(q), axis=1), 1e-30) / 448.0
    q8 = (q / s[:, None]).to(gl.float8e4nv)
    q8 = gl.convert_layout(q8, q_op)
    return q8, gl.convert_layout(s, gl.SliceLayout(1, qk_l))


@gluon.jit
def _softmax_tail(m_i, m_new, LOG2_PS: gl.constexpr):
    m_sub = gl.where(m_new > float("-inf"), m_new - LOG2_PS, 0.0)
    alpha = gl.where(m_new > float("-inf"), gl.exp2(m_i - m_new), 1.0)
    return m_sub, alpha


@gluon.jit
def _finish(acc, m_i, l_i, row, h0, num_heads, k_scale, sink_ptr, out_ptr, out_stride0, out_stride1,
            lse_ptr, G: gl.constexpr, HAS_SINK: gl.constexpr, HAS_LSE: gl.constexpr,
            LOG2_PS: gl.constexpr, qk_l: gl.constexpr, pv_l: gl.constexpr):
    """Normalize (sink as one more term in the denominator), store bf16 rows and
    the natural-log LSE. l_i carries PS = 448 / 2^LAZY, m_i is base 2."""
    NW: gl.constexpr = gl.num_warps()
    GM: gl.constexpr = acc.shape[0]
    hs = gl.arange(0, GM, layout=gl.SliceLayout(1, qk_l))
    hm = hs < G
    m_f = m_i
    if HAS_SINK:
        sink = gl.load(sink_ptr + h0 + hs, mask=hm, other=0.0) * 1.4426950408889634
        m_f = gl.maximum(m_i, sink)
    # sum over keys of 2^(x - m_f); an empty row has m_i = -inf and adds 0
    l_t = gl.where(m_i > float("-inf"), l_i * gl.exp2(m_i - m_f - LOG2_PS), 0.0)
    if HAS_SINK:
        l_t = l_t + gl.exp2(sink - m_f)
    if HAS_LSE:
        gl.store(lse_ptr + row * num_heads + h0 + hs, (m_f + gl.log2(l_t)) * 0.6931471805599453, mask=hm)
    # out = k_scale * acc / (l_t * PS * 2^(m_f - m_i)); an empty row comes out 0
    den = gl.where(m_i > float("-inf"), l_t * gl.exp2(m_f - m_i + LOG2_PS), float("inf"))
    inv = gl.convert_layout(k_scale / den, gl.SliceLayout(1, pv_l))
    o_l: gl.constexpr = gl.BlockedLayout([1, 8], [1, 64], [NW, 1], [1, 0])
    o = gl.convert_layout((acc * inv[:, None]).to(gl.bfloat16), o_l)
    ho = gl.arange(0, GM, layout=gl.SliceLayout(1, o_l))
    co = gl.arange(0, 512, layout=gl.SliceLayout(0, o_l))
    gl.amd.cdna4.buffer_store(o, out_ptr + row.to(gl.int64) * out_stride0 + h0 * out_stride1,
                              ho[:, None] * out_stride1 + co[None, :], mask=(ho < G)[:, None])


# --------------------------------------------------------------------------
# 64 heads per program
# --------------------------------------------------------------------------


@gluon.jit
def _qk_half(kvt, half: gl.constexpr, q_dot, q_sc, k_sc, k_op: gl.constexpr, qk_l: gl.constexpr):
    H: gl.constexpr = q_dot.shape[0]
    HB: gl.constexpr = kvt.shape[0] // 2
    k = kvt.slice(half * HB, HB, dim=0).permute([1, 0]).load(k_op)
    return gl.amd.cdna4.mfma_scaled(q_dot, q_sc, "e4m3", k.to(gl.float8e4nv, bitcast=True),
                                    k_sc, "e4m3", gl.zeros([H, HB], gl.float32, qk_l))


@gluon.jit
def _wide_step(t, start, end, kvt, kv_nx, p_smem, a_smem, q_dot, q_sc, k_sc, p_sc, v_sc, c_row,
               acc, m_i, l_i, rid, kv_ptr, idx_row, kg, cg, kcol,
               qk_l: gl.constexpr, pv_l: gl.constexpr, k_op: gl.constexpr,
               p_op: gl.constexpr, v_op: gl.constexpr, BN: gl.constexpr,
               LAZY: gl.constexpr, HAS_INVALID: gl.constexpr, BIG: gl.constexpr):
    HB: gl.constexpr = BN // 2
    LOG2_PS: gl.constexpr = 8.807354922057604 - LAZY
    gl.amd.cdna4.async_copy.wait_group(0)  # tile t landed (+ the barrier before its reads)
    _gather(kv_nx, kv_ptr, rid, cg, BIG)
    rid = _rows(idx_row, (t + 2) * BN, end - start, kg, HAS_INVALID)

    S0 = _qk_half(kvt, 0, q_dot, q_sc, k_sc, k_op, qk_l)
    gl.amd.hint.sched_barrier()  # keep one half's K operand live at a time
    S1 = _qk_half(kvt, 1, q_dot, q_sc, k_sc, k_op, qk_l)
    k0 = start + t * BN
    S0 = _mask(S0, idx_row, k0, t * BN, end, kcol, HAS_INVALID)
    S1 = _mask(S1, idx_row, k0 + HB, t * BN + HB, end, kcol, HAS_INVALID)
    # c_row > 0: scale the row max instead of every score
    m_cand = gl.maximum(m_i, gl.maximum(gl.max(S0, axis=1), gl.max(S1, axis=1)) * c_row)
    m_new = gl.where(m_cand > m_i + LAZY, m_cand, m_i)
    m_sub, alpha = _softmax_tail(m_i, m_new, LOG2_PS)
    c2 = gl.broadcast(c_row[:, None], S0)[0]
    nm = gl.broadcast(-m_sub[:, None], S0)[0]
    p0 = gl.exp2(gl.fma(S0, c2, nm))
    p1 = gl.exp2(gl.fma(S1, c2, nm))
    l_i = l_i * alpha + gl.sum(p0, axis=1) + gl.sum(p1, axis=1)
    m_i = m_new
    p_smem.slice(0, HB, dim=1).store(p0.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
    p_smem.slice(HB, HB, dim=1).store(p1.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
    a_smem.store(alpha)
    gl.barrier()  # P, alpha published
    alpha_pv = a_smem.load(gl.SliceLayout(1, pv_l))
    if gl.min(alpha_pv, axis=0) < 1.0:
        acc = acc * alpha_pv[:, None]
    v = kvt.load(v_op).to(gl.float8e4nv, bitcast=True)
    p = p_smem.load(p_op).to(gl.float8e4nv, bitcast=True)
    acc = gl.amd.cdna4.mfma_scaled(p, p_sc, "e4m3", v, v_sc, "e4m3", acc)
    return acc, m_i, l_i, rid


@gluon.jit
def _sparse_mla_fp8_wide(
    q_ptr, q_stride0, q_stride1, q_scale_ptr, kv_ptr, ks_ptr, out_ptr, out_stride0, out_stride1,
    lse_ptr, indptr_ptr, idx_ptr, sink_ptr, num_heads, sm_scale_log2,
    LAZY: gl.constexpr, Q_BF16: gl.constexpr, HAS_SINK: gl.constexpr, HAS_LSE: gl.constexpr,
    HAS_INVALID: gl.constexpr, BIG: gl.constexpr,
):
    NW: gl.constexpr = 4
    H: gl.constexpr = 64
    BN: gl.constexpr = 64
    HB: gl.constexpr = BN // 2
    LOG2_PS: gl.constexpr = 8.807354922057604 - LAZY
    row = gl.program_id(0)
    h0 = gl.program_id(1) * H

    qk_l: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [NW, 1])
    pv_l: gl.constexpr = gl.amd.AMDMFMALayout(4, [32, 32, 64], True, [1, NW], tiles_per_warp=[1, 4])
    q_op: gl.constexpr = gl.DotOperandLayout(0, qk_l, 16)
    k_op: gl.constexpr = gl.DotOperandLayout(1, qk_l, 16)
    p_op: gl.constexpr = gl.DotOperandLayout(0, pv_l, 16)
    v_op: gl.constexpr = gl.DotOperandLayout(1, pv_l, 16)
    qs_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(q_op, [H, 16])
    ks_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(k_op, [HB, 16])
    ps_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(p_op, [H, BN // 32])
    vs_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(v_op, [512, BN // 32])
    # Ring image (in-block offset bits 4..9 = dim4, key0, key1, key2, dim5, dim6;
    # block bits dim7, dim8, key4, key3, key5 with a 32 B pad per KB): the
    # 32-lane passes of the K ds_read_b128 (key0-3, dim4, over 512 B) and of the
    # V^T ds_read_b64_tr_b8 (key0-2, dim3, dim4, over 256 B) hit distinct banks.
    ga_l: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8], [0, 128], [0, 256], [8, 0]],
        lane_bases=[[0, 16], [1, 0], [2, 0], [4, 0], [0, 32], [0, 64]],
        warp_bases=[[16, 0], [32, 0]], block_bases=[], shape=[BN, 512])
    kv_sh: gl.constexpr = gl.PaddedSharedLayout(
        [[1024, 32]],
        [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32], [0, 64],
         [0, 128], [0, 256], [16, 0], [8, 0], [32, 0]], [], [BN, 512])
    p_sh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BN, 16]], [H, BN], [1, 0])
    a_sh: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    start = gl.load(indptr_ptr + row)
    end = gl.load(indptr_ptr + row + 1)
    n = end - start
    idx_row = idx_ptr + start
    k_scale = gl.load(ks_ptr)
    if Q_BF16:
        # quantized before the ring is live, so its LDS scratch aliases the ring
        q_dot, q_s = _q_bf16(q_ptr, q_stride0, q_stride1, row, h0, H, H, q_op, qk_l)
        c_row = q_s * (k_scale * sm_scale_log2)

    q_sc = gl.full([H, 16], 127, gl.uint8, qs_l)  # 2^0: unit block scales
    k_sc = gl.full([HB, 16], 127, gl.uint8, ks_l)
    p_sc = gl.full([H, BN // 32], 127, gl.uint8, ps_l)
    v_sc = gl.full([512, BN // 32], 127, gl.uint8, vs_l)

    kv0 = gl.allocate_shared_memory(gl.uint8, [BN, 512], kv_sh)
    kv1 = gl.allocate_shared_memory(gl.uint8, [BN, 512], kv_sh)
    p_smem = gl.allocate_shared_memory(gl.uint8, [H, BN], p_sh)
    a_smem = gl.allocate_shared_memory(gl.float32, [H], a_sh)

    kg = gl.arange(0, BN, layout=gl.SliceLayout(1, ga_l))
    cg = gl.arange(0, 512, layout=gl.SliceLayout(0, ga_l))
    kcol = gl.arange(0, HB, layout=gl.SliceLayout(0, qk_l))

    m_i = gl.full([H], float("-inf"), gl.float32, gl.SliceLayout(1, qk_l))
    l_i = gl.zeros([H], gl.float32, gl.SliceLayout(1, qk_l))
    acc = gl.zeros([H, 512], gl.float32, pv_l)

    n_tiles = gl.cdiv(n, BN)
    rid = _rows(idx_row, 0, n, kg, HAS_INVALID)
    _gather(kv0, kv_ptr, rid, cg, BIG)
    rid = _rows(idx_row, BN, n, kg, HAS_INVALID)
    if not Q_BF16:
        # behind the first gather instead of ahead of the row-id -> gather chain
        q_dot = _q_fp8(q_ptr, q_stride0, q_stride1, row, h0, H, H, q_op)
        c_row = gl.full([H], 1.0, gl.float32, gl.SliceLayout(1, qk_l)) * (gl.load(q_scale_ptr) * k_scale * sm_scale_log2)

    for it in range(0, gl.cdiv(n_tiles, 2)):
        t = it * 2
        acc, m_i, l_i, rid = _wide_step(
            t, start, end, kv0, kv1, p_smem, a_smem, q_dot, q_sc, k_sc, p_sc, v_sc, c_row,
            acc, m_i, l_i, rid, kv_ptr, idx_row, kg, cg, kcol,
            qk_l, pv_l, k_op, p_op, v_op, BN, LAZY, HAS_INVALID, BIG)
        acc, m_i, l_i, rid = _wide_step(
            t + 1, start, end, kv1, kv0, p_smem, a_smem, q_dot, q_sc, k_sc, p_sc, v_sc, c_row,
            acc, m_i, l_i, rid, kv_ptr, idx_row, kg, cg, kcol,
            qk_l, pv_l, k_op, p_op, v_op, BN, LAZY, HAS_INVALID, BIG)
    gl.amd.cdna4.async_copy.wait_group(0)
    _finish(acc, m_i, l_i, row, h0, num_heads, k_scale, sink_ptr, out_ptr, out_stride0, out_stride1,
            lse_ptr, H, HAS_SINK, HAS_LSE, LOG2_PS, qk_l, pv_l)


# --------------------------------------------------------------------------
# 8, 16 or 32 heads per program, split-K
# --------------------------------------------------------------------------


@gluon.jit
def _narrow_tile(kvt, p_smem, a_smem, q_dot, c_row, acc, m_i, l_acc, k0, kb, end, kcol, idx_row,
                 QK_L: gl.constexpr, PV_L: gl.constexpr, VV_SH: gl.constexpr, PV_SH: gl.constexpr,
                 GM: gl.constexpr, BN: gl.constexpr, KWP: gl.constexpr, PV_SCALED: gl.constexpr,
                 LAZY: gl.constexpr, HAS_INVALID: gl.constexpr):
    LOG2_PS: gl.constexpr = 8.807354922057604 - LAZY
    q_op: gl.constexpr = gl.DotOperandLayout(0, QK_L, 16)
    k_op: gl.constexpr = gl.DotOperandLayout(1, QK_L, 16)
    p_op: gl.constexpr = gl.DotOperandLayout(0, PV_L, KWP)
    v_op: gl.constexpr = gl.DotOperandLayout(1, PV_L, KWP)
    qs_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(q_op, [GM, 16])
    ks_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(k_op, [BN, 16])
    k = kvt.permute([1, 0]).load(k_op).to(gl.float8e4nv, bitcast=True)
    S = gl.amd.cdna4.mfma_scaled(q_dot, gl.full([GM, 16], 127, gl.uint8, qs_l), "e4m3",
                                 k, gl.full([BN, 16], 127, gl.uint8, ks_l), "e4m3",
                                 gl.zeros([GM, BN], gl.float32, QK_L))
    S = _mask(S, idx_row, k0, kb, end, kcol, HAS_INVALID)
    m_cand = gl.maximum(m_i, gl.max(S, axis=1) * c_row)
    m_new = gl.where(m_cand > m_i + LAZY, m_cand, m_i)
    m_sub, alpha = _softmax_tail(m_i, m_new, LOG2_PS)
    c2 = gl.broadcast(c_row[:, None], S)[0]
    nm = gl.broadcast(-m_sub[:, None], S)[0]
    p = gl.exp2(gl.fma(S, c2, nm))
    # deferred row sum: no cross-wave reduction per tile
    l_acc = l_acc * gl.broadcast(alpha[:, None], S)[0] + p
    m_i = m_new
    p_smem.store(p.to(gl.float8e4nv).to(gl.uint8, bitcast=True))
    a_smem.store(alpha)
    gl.barrier()  # P, alpha published
    alpha_pv = a_smem.load(gl.SliceLayout(1, PV_L))
    if gl.min(alpha_pv, axis=0) < 1.0:
        acc = acc * alpha_pv[:, None]
    v = kvt.reinterpret(gl.uint8, [BN, 512], VV_SH).load(v_op).to(gl.float8e4nv, bitcast=True)
    pd = p_smem.reinterpret(gl.uint8, [GM, BN], PV_SH).load(p_op).to(gl.float8e4nv, bitcast=True)
    if PV_SCALED:
        ps_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(p_op, [GM, BN // 32])
        vs_l: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(v_op, [512, BN // 32])
        acc = gl.amd.cdna4.mfma_scaled(pd, gl.full([GM, BN // 32], 127, gl.uint8, ps_l), "e4m3",
                                       v, gl.full([512, BN // 32], 127, gl.uint8, vs_l), "e4m3", acc)
    else:
        acc = gl.amd.cdna4.mfma(pd, v, acc)
    return acc, m_i, l_acc


@gluon.jit
def _sparse_mla_fp8_narrow(
    q_ptr, q_stride0, q_stride1, q_scale_ptr, kv_ptr, ks_ptr, out_ptr, out_stride0, out_stride1,
    lse_ptr, part_ptr, ml_ptr, indptr_ptr, idx_ptr, sink_ptr, num_heads, sm_scale_log2, NSPLIT,
    G: gl.constexpr, GM: gl.constexpr, SPLIT: gl.constexpr, LAZY: gl.constexpr,
    Q_BF16: gl.constexpr, HAS_SINK: gl.constexpr, HAS_LSE: gl.constexpr, HAS_INVALID: gl.constexpr,
    BIG: gl.constexpr, STAGES: gl.constexpr,
    QK_L: gl.constexpr, PV_L: gl.constexpr, GA_L: gl.constexpr, KV_SH: gl.constexpr,
    VV_SH: gl.constexpr, PV_SH: gl.constexpr, KWP: gl.constexpr, PV_SCALED: gl.constexpr,
):
    BN: gl.constexpr = 64
    LOG2_PS: gl.constexpr = 8.807354922057604 - LAZY
    row = gl.program_id(0)
    h0 = gl.program_id(1) * G
    split = gl.program_id(2)
    q_op: gl.constexpr = gl.DotOperandLayout(0, QK_L, 16)
    p_sh: gl.constexpr = gl.PaddedSharedLayout.with_identity_for([[BN, 16]], [GM, BN], [1, 0])
    a_sh: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [0])

    start = gl.load(indptr_ptr + row)
    end = gl.load(indptr_ptr + row + 1)
    n = end - start
    idx_row = idx_ptr + start
    n_tiles = gl.cdiv(n, BN)
    tps = gl.cdiv(n_tiles, NSPLIT)
    t0 = split * tps
    nt = gl.maximum(gl.minimum(n_tiles - t0, tps), 0)
    kb = t0 * BN  # first key of this split, relative to start
    # a split's keys end at its last tile: the masked tail step of an odd tile
    # count must not reach into the next split's keys
    end_s = gl.minimum(end, start + kb + nt * BN)

    k_scale = gl.load(ks_ptr)
    if Q_BF16:
        # quantized before the ring is live, so its LDS scratch aliases the ring
        q_dot, q_s = _q_bf16(q_ptr, q_stride0, q_stride1, row, h0, G, GM, q_op, QK_L)
        c_row = q_s * (k_scale * sm_scale_log2)

    kv0 = gl.allocate_shared_memory(gl.uint8, [BN, 512], KV_SH)
    if STAGES == 2:
        kv1 = gl.allocate_shared_memory(gl.uint8, [BN, 512], KV_SH)
    else:
        kv1 = kv0
    p_smem = gl.allocate_shared_memory(gl.uint8, [GM, BN], p_sh)
    a_smem = gl.allocate_shared_memory(gl.float32, [GM], a_sh)

    kg = gl.arange(0, BN, layout=gl.SliceLayout(1, GA_L))
    cg = gl.arange(0, 512, layout=gl.SliceLayout(0, GA_L))
    kcol = gl.arange(0, BN, layout=gl.SliceLayout(0, QK_L))

    m_i = gl.full([GM], float("-inf"), gl.float32, gl.SliceLayout(1, QK_L))
    l_acc = gl.zeros([GM, BN], gl.float32, QK_L)
    acc = gl.zeros([GM, 512], gl.float32, PV_L)

    rid = _rows(idx_row, kb, n, kg, HAS_INVALID)
    _gather(kv0, kv_ptr, rid, cg, BIG)
    if STAGES == 2:
        rid = _rows(idx_row, kb + BN, n, kg, HAS_INVALID)
    if not Q_BF16:
        q_dot = _q_fp8(q_ptr, q_stride0, q_stride1, row, h0, G, GM, q_op)
        c_row = gl.full([GM], 1.0, gl.float32, gl.SliceLayout(1, QK_L)) * (gl.load(q_scale_ptr) * k_scale * sm_scale_log2)

    if STAGES == 1:
        # one tile buffer: half the LDS, so more workgroups per CU hide the
        # gather latency instead of a prefetched tile (aiter's _async_buffers)
        for t in range(0, nt):
            gl.amd.cdna4.async_copy.wait_group(0)  # tile t landed (+ barrier)
            rid = _rows(idx_row, kb + (t + 1) * BN, n, kg, HAS_INVALID)
            acc, m_i, l_acc = _narrow_tile(
                kv0, p_smem, a_smem, q_dot, c_row, acc, m_i, l_acc, start + kb + t * BN, kb + t * BN,
                end_s, kcol, idx_row, QK_L, PV_L, VV_SH, PV_SH, GM, BN, KWP, PV_SCALED, LAZY, HAS_INVALID)
            gl.barrier()  # every wave is done reading the buffer
            _gather(kv0, kv_ptr, rid, cg, BIG)
    for it in range(0, gl.cdiv(nt, 2) if STAGES == 2 else 0):
        t = it * 2
        gl.amd.cdna4.async_copy.wait_group(0)  # tile t landed (+ barrier)
        _gather(kv1, kv_ptr, rid, cg, BIG)
        rid = _rows(idx_row, kb + (t + 2) * BN, n, kg, HAS_INVALID)
        acc, m_i, l_acc = _narrow_tile(
            kv0, p_smem, a_smem, q_dot, c_row, acc, m_i, l_acc, start + kb + t * BN, kb + t * BN,
            end_s, kcol, idx_row, QK_L, PV_L, VV_SH, PV_SH, GM, BN, KWP, PV_SCALED, LAZY, HAS_INVALID)
        gl.amd.cdna4.async_copy.wait_group(0)
        _gather(kv0, kv_ptr, rid, cg, BIG)
        rid = _rows(idx_row, kb + (t + 3) * BN, n, kg, HAS_INVALID)
        acc, m_i, l_acc = _narrow_tile(
            kv1, p_smem, a_smem, q_dot, c_row, acc, m_i, l_acc, start + kb + (t + 1) * BN,
            kb + (t + 1) * BN, end_s, kcol, idx_row,
            QK_L, PV_L, VV_SH, PV_SH, GM, BN, KWP, PV_SCALED, LAZY, HAS_INVALID)
    gl.amd.cdna4.async_copy.wait_group(0)

    l_i = gl.sum(l_acc, axis=1)
    if SPLIT:
        # fp32 partials, unnormalized; m base 2, l carries PS (as _finish expects)
        hp = gl.arange(0, GM, layout=gl.SliceLayout(1, PV_L))
        dp = gl.arange(0, 512, layout=gl.SliceLayout(0, PV_L))
        hs = gl.arange(0, GM, layout=gl.SliceLayout(1, QK_L))
        base = ((row * NSPLIT + split) * num_heads + h0).to(gl.int64)
        gl.amd.cdna4.buffer_store(acc, part_ptr + base * 512, hp[:, None] * 512 + dp[None, :],
                                  mask=(hp < G)[:, None])
        gl.store(ml_ptr + (base + hs) * 2, m_i, mask=hs < G)
        gl.store(ml_ptr + (base + hs) * 2 + 1, l_i, mask=hs < G)
    else:
        _finish(acc, m_i, l_i, row, h0, num_heads, k_scale, sink_ptr, out_ptr, out_stride0,
                out_stride1, lse_ptr, G, HAS_SINK, HAS_LSE, LOG2_PS, QK_L, PV_L)


@triton.jit
def _sparse_mla_fp8_reduce(part_ptr, ml_ptr, sink_ptr, ks_ptr, out_ptr, out_stride0, out_stride1,
                           lse_ptr, num_heads, NSPLIT, LOG2_PS: tl.constexpr, NS: tl.constexpr,
                           HAS_SINK: tl.constexpr, HAS_LSE: tl.constexpr):
    """Merge the split partials of one (query, head): same math as _finish."""
    row = tl.program_id(0)
    h = tl.program_id(1)
    s = tl.arange(0, NS)
    sm = s < NSPLIT
    base = (row * NSPLIT + s) * num_heads + h
    m = tl.load(ml_ptr + base * 2, mask=sm, other=float("-inf"))
    l = tl.load(ml_ptr + base * 2 + 1, mask=sm, other=0.0)
    m_star = tl.max(m, axis=0)
    w = tl.where(m > float("-inf"), tl.exp2(m - m_star), 0.0)
    l_i = tl.sum(w * l, axis=0)  # carries PS, relative to m_star
    m_f = m_star
    if HAS_SINK:
        sink = tl.load(sink_ptr + h) * 1.4426950408889634
        m_f = tl.maximum(m_star, sink)
    l_t = tl.where(m_star > float("-inf"), l_i * tl.exp2(m_star - m_f - LOG2_PS), 0.0)
    if HAS_SINK:
        l_t = l_t + tl.exp2(sink - m_f)
    if HAS_LSE:
        tl.store(lse_ptr + row * num_heads + h, (m_f + tl.log2(l_t)) * 0.6931471805599453)
    den = tl.where(m_star > float("-inf"), l_t * tl.exp2(m_f - m_star + LOG2_PS), float("inf"))
    d = tl.arange(0, 512)
    a = tl.load(part_ptr + base.to(tl.int64)[:, None] * 512 + d[None, :], mask=sm[:, None], other=0.0)
    o = tl.sum(a * w[:, None], axis=0) * (tl.load(ks_ptr) / den)
    tl.store(out_ptr + row.to(tl.int64) * out_stride0 + h * out_stride1 + d, o.to(tl.bfloat16))


# --------------------------------------------------------------------------
# host side
# --------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _narrow_layouts(G, NW):
    """Layouts of _sparse_mla_fp8_narrow for G heads per program on NW waves.
    Cached: rebuilding them per call made the launch slow enough to show up as
    kernel time (the GPU clocks down while it waits)."""
    D, BN = 512, 64
    GM = max(G, 16)
    HW = min(NW, GM // 16)  # head groups in QK; the rest of the waves split keys
    qk = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [HW, NW // HW])
    if GM == 32:
        pv = gl.amd.AMDMFMALayout(4, [32, 32, 64], True, [1, NW], tiles_per_warp=[1, D // 32 // NW])
        kwp, scaled = 16, True
        # in-block dim4, key0, key1, key2, dim5, dim6; key3 = block bit 3 (32 B
        # pad -> 256 B): K and the 32x32x64 V^T passes are conflict-free
        kv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32], [0, 64],
              [0, 128], [0, 256], [16, 0], [8, 0], [32, 0]]
        pad, vv, pp = 32, kv, None
    else:
        pv = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, NW], tiles_per_warp=[1, D // 16 // NW])
        kwp, scaled = 8, False
        # 16 B pad; key4 = block bit 0 (16 B), key3 = block bit 4 (256 B). The
        # 16x16x32 V^T pass varies key3, so V and P are read through views with
        # key bits 3 and 4 swapped (PV sums over keys: a common key permutation
        # of P and V is exact).
        kv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32], [0, 64],
              [16, 0], [0, 128], [0, 256], [32, 0], [8, 0]]
        vv = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 16], [1, 0], [2, 0], [4, 0], [0, 32], [0, 64],
              [8, 0], [0, 128], [0, 256], [32, 0], [16, 0]]
        pad = 16
        pp = [[0, 1], [0, 2], [0, 4], [0, 16], [0, 8], [0, 32]] + \
             [[1 << i, 0] for i in range(GM.bit_length() - 1)]
    kv_sh = gl.PaddedSharedLayout([[1024, pad]], kv, [], [BN, D])
    vv_sh = gl.PaddedSharedLayout([[1024, pad]], vv, [], [BN, D])
    p_sh = gl.PaddedSharedLayout.with_identity_for([[BN, 16]], [GM, BN], [1, 0])
    pv_sh = p_sh if pp is None else gl.PaddedSharedLayout([[BN, 16]], pp, [], [GM, BN])
    # gather: lanes are the image's in-block bits; the five block bits go to
    # registers (dims first: register order is issue order, and a key bit
    # early makes the 2nd gather wait on every gather in flight) and warps
    block = sorted(kv[10:], key=lambda b: b[0] != 0)
    nwb = NW.bit_length() - 1
    ga = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [0, 8]] + block[:5 - nwb],
        lane_bases=kv[4:10], warp_bases=block[5 - nwb:], block_bases=[], shape=[BN, D])
    return dict(GM=GM, QK_L=qk, PV_L=pv, GA_L=ga, KV_SH=kv_sh, VV_SH=vv_sh, PV_SH=pv_sh,
                KWP=kwp, PV_SCALED=scaled)


def _pow2_floor(x):
    return 1 << (max(1, int(x)).bit_length() - 1)


def fp8_narrow_plan(num_queries, num_heads, keys, cus):
    """(heads per program, splits) for _sparse_mla_fp8_narrow, from sweeps on
    MI355X decode (8..128 heads, 640 / 1152 / 2048 keys per row).

    More programs fill the machine; two ways to get them. Fewer heads per
    program re-gathers every key H/G times; split-K only adds a combine.
    Long rows (>= 24 tiles): the largest G and split, looking for a program per
    CU with >= 4 tiles per split, then >= 2, up to 16 splits (128 heads b1:
    G = 8 x 16 splits, 12.9 us vs 16.4 at G = 32 x 8).
    Short rows: shrink G first (largest G with a program per CU, else 8), then
    split up to 8, >= 2.5 tiles per split from 32 programs on."""
    tiles = max(1, math.ceil(keys / 64))
    gs = [g for g in (32, 16, 8) if num_heads % g == 0]
    if tiles >= 24:
        for floor in (4, 2):
            for G in gs:
                wg = num_queries * num_heads // G
                ns = min(16, _pow2_floor(max(1, cus // wg)), _pow2_floor(tiles // floor))
                if wg * ns >= cus:
                    return G, ns
        G = gs[-1]
        wg = num_queries * num_heads // G
        return G, min(16, _pow2_floor(max(1, cus // wg)), _pow2_floor(tiles // 2))
    fill = [g for g in gs if num_queries * num_heads // g >= cus]
    G = fill[0] if fill else gs[-1]
    wg = num_queries * num_heads // G
    if wg >= cus:
        return G, 1
    ns = min(8, _pow2_floor(cus // wg))
    floor = 2.5 if wg >= 32 else 0
    while ns > 1 and tiles / ns < floor:
        ns //= 2
    return G, max(1, min(ns, tiles))


_CUS = {}


def _num_cus(device):
    if device not in _CUS:
        _CUS[device] = torch.cuda.get_device_properties(device).multi_processor_count
    return _CUS[device]


def sparse_mla_fp8(q, q_scale, cache_u8, kv_scale, kv_indptr, kv_indices, softmax_scale, out,
                   lse=None, attn_sink=None, has_invalid=False, kv_splits=None, max_keys=None,
                   lazy=8, waves=4, heads_per_prog=None, stages=None, wpe=None):
    """q [C, H, 512] fp8 (+ scalar q_scale) or bf16; cache_u8 [rows, 512] fp8
    bytes; kv_scale [1] f32; out [C, H, >= 512] bf16; lse [C, H] f32 or None.
    H in {8, 16, 32, 64, 128}."""
    C, H, _ = q.shape
    cus = _num_cus(q.device)
    q_bf16 = q.dtype == torch.bfloat16
    qv = q if q_bf16 else q.view(torch.uint8)
    qs = q_scale if q_scale is not None else kv_scale  # unread for bf16 q
    sink = attn_sink if attn_sink is not None else kv_scale
    lse_t = lse if lse is not None else kv_scale
    sm2 = softmax_scale * LOG2E
    common = dict(LAZY=lazy, Q_BF16=q_bf16, HAS_SINK=attn_sink is not None,
                  HAS_LSE=lse is not None, HAS_INVALID=has_invalid,
                  BIG=cache_u8.numel() >= 2**31)
    if H % 64 == 0 and kv_splits in (None, 1) and C * (H // 64) >= cus:
        _sparse_mla_fp8_wide[(C, H // 64)](
            qv, q.stride(0), q.stride(1), qs, cache_u8, kv_scale, out, out.stride(0), out.stride(1),
            lse_t, kv_indptr, kv_indices, sink, H, sm2, num_warps=4, waves_per_eu=2, **common)
        return
    # the average row length is known on the host: reading the max off
    # kv_indptr would sync and launch two kernels per call
    keys = max_keys if max_keys is not None else kv_indices.numel() / max(1, C)
    G, ns = fp8_narrow_plan(C, H, keys, cus)
    if heads_per_prog is not None:
        G = heads_per_prog
    if kv_splits is None:
        kv_splits = ns
    L = _narrow_layouts(G, waves)
    hblk = H // G
    split = kv_splits > 1
    if split:
        part = torch.empty((C, kv_splits, H, 512), dtype=torch.float32, device=q.device)
        ml = torch.empty((C, kv_splits, H, 2), dtype=torch.float32, device=q.device)
    else:
        part = ml = kv_scale
    if stages is None:
        # 16-row tiles fit 4 waves/SIMD in registers (~144 VGPRs); with enough
        # workgroups, one tile buffer and 4 workgroups per CU beat a prefetched
        # tile at 2 (16 heads prefill 86.2 -> 82.0 us). 32-row tiles spill there.
        big = G <= 16 and C * hblk * kv_splits >= 4 * cus
        stages, wpe = (1, 4) if big else (2, 2)
    _sparse_mla_fp8_narrow[(C, hblk, kv_splits)](
        qv, q.stride(0), q.stride(1), qs, cache_u8, kv_scale, out, out.stride(0), out.stride(1),
        lse_t, part, ml, kv_indptr, kv_indices, sink, H, sm2, kv_splits,
        G=G, SPLIT=split, STAGES=stages, num_warps=waves, waves_per_eu=wpe or 2, **common, **L)
    if split:
        _sparse_mla_fp8_reduce[(C, H)](
            part, ml, sink, kv_scale, out, out.stride(0), out.stride(1), lse_t, H, kv_splits,
            LOG2_PS=LOG2_448 - lazy, NS=triton.next_power_of_2(kv_splits),
            HAS_SINK=attn_sink is not None, HAS_LSE=lse is not None, num_warps=4)
