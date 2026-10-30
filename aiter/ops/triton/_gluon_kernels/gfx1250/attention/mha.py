# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon MHA forward for gfx1250: bf16 varlen attention with a 192-wide QK head
(128 "nope" + 64 rope/PE columns) and a 128-wide V head -- the MLA prefill shape
(DeepSeek / Kimi) -- reached through `flash_attn_varlen_func` in
aiter/ops/triton/attention/mha.py.

Layout handled by `_attn_fwd`:
  q    [total_q, H, 192]                 (nope | pe), any token / head strides
  k    [total_k, H, >=128]               only the first 128 columns are read
  k_pe [total_k, H or 1, 64]             PE columns of K; head stride 0 = shared by
                                         all heads (MLA), or the k[..., 128:] view
                                         of a concatenated [nope | pe] K
  v    [total_k, H, 128 or 192]          a 192-wide V adds a 64-column high slice
  o    [total_q, H, 128 or 192]
  lse  any 2-D strides (head, token), natural log
The last dimension of every tensor must be contiguous.  K, K_PE and V are staged
into LDS by the Tensor Data Mover (TDM), so their bases and row/head strides must
be 16-byte aligned.

Scheduling (all chosen by `_get_config`):
  * non-persistent: one program per (q block, head); q blocks of a varlen batch are
    numbered `cu_seqlens_q[s] // BLOCK_M + s + i` so a program finds its sequence
    with a binary search over cu_seqlens_q.  Causal launches run longest-first.
  * persistent (causal): G = WG_PER_CU * #CUs programs walk the (q block, head)
    items longest-first in snake order for load balance.
  * split-KV (non-causal, few q blocks against a long K): NUM_SPLITS programs per
    (q block, head) each cover a slice of the K tiles and write fp32 partials that
    `_attn_fwd_combine` merges.
An optional initial state (o_init normalised, lse_init) continues the online softmax
from a previous partial result over another key set, so a chunked-context prefill
needs no separate merge kernel.
"""

import triton
import triton.experimental.gluon.language as gl
import triton.language as tl
from triton.experimental import gluon

from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import (
    AITER_TRITON_CONFIGS_PATH,
    load_config_json,
)

RCP_LN2 = gl.constexpr(1.4426950408889634)
BLOCK_DMODEL_NOPE = gl.constexpr(128)
BLOCK_DMODEL_PE = gl.constexpr(64)
BLOCK_DMODEL_V = gl.constexpr(128)
# extra V / O columns of a 192-wide V (v head dim == qk head dim)
BLOCK_DMODEL_V_HI = gl.constexpr(64)


@gluon.constexpr_function
def _warp_bases(num_warps):
    b = []
    i = 1
    while i < num_warps:
        b.append([i, 0])
        i *= 2
    return b


@gluon.jit
def _find_seq(cu_seqlens_q, qblk, num_seqs, BLOCK_M: gl.constexpr):
    lo = 0
    hi = num_seqs
    while lo < hi:
        mid = (lo + hi) // 2
        v = gl.load(cu_seqlens_q + mid)
        if v // BLOCK_M + mid <= qblk:
            lo = mid + 1
        else:
            hi = mid
    return lo - 1


@gluon.jit
def _tdm_load_kv(k_desc, k_pe_desc, v_desc, k_smem, k_pe_smem, v_smem, row, buf, pred):
    gl.amd.gfx1250.tdm.async_load(k_desc, [row, 0], k_smem.index(buf), pred=pred)
    gl.amd.gfx1250.tdm.async_load(k_pe_desc, [row, 0], k_pe_smem.index(buf), pred=pred)
    gl.amd.gfx1250.tdm.async_load(v_desc, [row, 0], v_smem.index(buf), pred=pred)


@gluon.jit
def _tdm_load_kv_v_hi(
    k_desc, k_pe_desc, v_desc, v_hi_desc, k_smem, k_pe_smem, v_smem, v_hi_smem, row, buf, pred
):
    _tdm_load_kv(k_desc, k_pe_desc, v_desc, k_smem, k_pe_smem, v_smem, row, buf, pred)
    gl.amd.gfx1250.tdm.async_load(v_hi_desc, [row, 0], v_hi_smem.index(buf), pred=pred)


@gluon.jit
def _attn_qk_softmax(
    q, q_pe, k_smem, k_pe_smem, buf, m_i, l_i, qk_scale, j, qpos, seqlen_k, causal_off,
    MASK: gl.constexpr, IS_CAUSAL: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    L: gl.constexpr, KL: gl.constexpr,
):
    """S = Q K^T of K tile j (+ masks), online-softmax update. Returns P (fp32), alpha."""
    k = k_smem.index(buf).permute([1, 0]).load(KL)
    k_pe = k_pe_smem.index(buf).permute([1, 0]).load(KL)
    s = gl.zeros([BLOCK_M, BLOCK_N], gl.float32, L)
    s = gl.amd.gfx1250.wmma(q, k, s)
    s = gl.amd.gfx1250.wmma(q_pe, k_pe, s)
    if MASK:
        kpos = j * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, L))
        valid = kpos[None, :] < seqlen_k
        if IS_CAUSAL:
            valid = valid & (kpos[None, :] <= (qpos + causal_off)[:, None])
        s = gl.where(valid, s, float("-inf"))
    m_new = gl.maximum(m_i, gl.max(s, 1))
    m_safe = gl.where(m_new == float("-inf"), 0.0, m_new)
    mc = m_safe * qk_scale
    p = gl.exp2(s * qk_scale - mc[:, None])
    alpha = gl.exp2(m_i * qk_scale - mc)
    l_i = l_i * alpha + gl.sum(p, 1)
    return m_new, l_i, p, alpha


@gluon.jit
def _attn_tile(
    q, q_pe, k_smem, k_pe_smem, v_smem, buf, m_i, l_i, acc, qk_scale, j, qpos, seqlen_k, causal_off,
    MASK: gl.constexpr, IS_CAUSAL: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    L: gl.constexpr, KL: gl.constexpr, PL: gl.constexpr,
):
    m_new, l_i, p, alpha = _attn_qk_softmax(
        q, q_pe, k_smem, k_pe_smem, buf, m_i, l_i, qk_scale, j, qpos, seqlen_k, causal_off,
        MASK, IS_CAUSAL, BLOCK_M, BLOCK_N, L, KL,
    )
    acc = acc * alpha[:, None]
    pb = gl.convert_layout(p.to(gl.bfloat16), PL)
    v = v_smem.index(buf).load(KL)
    acc = gl.amd.gfx1250.wmma(pb, v, acc)
    return m_new, l_i, acc


@gluon.jit
def _attn_tile_v_hi(
    q, q_pe, k_smem, k_pe_smem, v_smem, v_hi_smem, buf, m_i, l_i, acc, acc_hi, qk_scale, j, qpos,
    seqlen_k, causal_off,
    MASK: gl.constexpr, IS_CAUSAL: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
    L: gl.constexpr, KL: gl.constexpr, PL: gl.constexpr,
):
    """`_attn_tile` for a 192-wide V: P V over V[:, :128] and V[:, 128:192]."""
    m_new, l_i, p, alpha = _attn_qk_softmax(
        q, q_pe, k_smem, k_pe_smem, buf, m_i, l_i, qk_scale, j, qpos, seqlen_k, causal_off,
        MASK, IS_CAUSAL, BLOCK_M, BLOCK_N, L, KL,
    )
    acc = acc * alpha[:, None]
    acc_hi = acc_hi * alpha[:, None]
    pb = gl.convert_layout(p.to(gl.bfloat16), PL)
    v = v_smem.index(buf).load(KL)
    acc = gl.amd.gfx1250.wmma(pb, v, acc)
    v_hi = v_hi_smem.index(buf).load(KL)
    acc_hi = gl.amd.gfx1250.wmma(pb, v_hi, acc_hi)
    return m_new, l_i, acc, acc_hi


@gluon.jit
def _attn_fwd_block(
    qblk, h, split,
    q_ptr, k_ptr, k_pe_ptr, v_ptr, o_ptr, lse_ptr,
    cu_seqlens_q, cu_seqlens_k,
    stride_qt, stride_qh, stride_kt, stride_kh, kpe_offset, stride_kpe_t, stride_kpe_h, stride_vt, stride_vh,
    stride_ot, stride_oh, stride_lse_h, stride_lse_t, stride_os, stride_lse_s,
    o_init_ptr, lse_init_ptr, stride_oit, stride_oih, stride_lsei_h, stride_lsei_t,
    sm_scale, num_seqs,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, NUM_WARPS: gl.constexpr,
    IS_CAUSAL: gl.constexpr, RETURN_LSE: gl.constexpr, NUM_SPLITS: gl.constexpr,
    HAS_INIT: gl.constexpr, HAS_V_HI: gl.constexpr,
):
    """One (q block, head[, KV split]) work item."""
    NB: gl.constexpr = 2
    seq = _find_seq(cu_seqlens_q, qblk, num_seqs, BLOCK_M)
    q_start = gl.load(cu_seqlens_q + seq)
    q_end = gl.load(cu_seqlens_q + seq + 1)
    seqlen_q = q_end - q_start
    mblk = qblk - (q_start // BLOCK_M + seq)
    m0 = mblk * BLOCK_M
    if m0 >= seqlen_q:
        return
    k_start = gl.load(cu_seqlens_k + seq)
    seqlen_k = gl.load(cu_seqlens_k + seq + 1) - k_start
    causal_off = seqlen_k - seqlen_q  # bottom-right aligned causal mask

    # ---------------- layouts
    WB: gl.constexpr = _warp_bases(NUM_WARPS)
    L: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=WB, instr_shape=[16, 16, 32]
    )
    QL: gl.constexpr = gl.DotOperandLayout(0, L, 8)
    KL: gl.constexpr = gl.DotOperandLayout(1, L, 8)
    PL: gl.constexpr = gl.DotOperandLayout(0, L, 8)
    K_SL: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_DMODEL_NOPE, 8]], [BLOCK_N, BLOCK_DMODEL_NOPE], [1, 0]
    )
    KPE_SL: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_DMODEL_PE, 8]], [BLOCK_N, BLOCK_DMODEL_PE], [1, 0]
    )
    V_SL: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_DMODEL_V, 16]], [BLOCK_N, BLOCK_DMODEL_V], [1, 0]
    )

    # ---------------- K tile range
    n_tiles_k = gl.cdiv(seqlen_k, BLOCK_N)
    if IS_CAUSAL:
        last_row = gl.minimum(m0 + BLOCK_M, seqlen_q) - 1
        n_end = gl.minimum(gl.cdiv(last_row + causal_off + 1, BLOCK_N), n_tiles_k)
        n_end = gl.maximum(n_end, 0)
        n_full = gl.minimum(gl.maximum((m0 + causal_off + 1) // BLOCK_N, 0), seqlen_k // BLOCK_N)
        n_full = gl.minimum(n_full, n_end)
    else:
        n_end = n_tiles_k
        n_full = seqlen_k // BLOCK_N

    # ---------------- Q (registers)
    q_base = q_ptr + (q_start + m0).to(gl.int64) * stride_qt + h.to(gl.int64) * stride_qh
    offs_qm = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, QL))
    offs_qd = gl.arange(0, BLOCK_DMODEL_NOPE, layout=gl.SliceLayout(0, QL))
    offs_qpe = gl.arange(0, BLOCK_DMODEL_PE, layout=gl.SliceLayout(0, QL))
    qmask = (m0 + offs_qm)[:, None] < seqlen_q
    q = gl.load(q_base + offs_qm[:, None] * stride_qt + offs_qd[None, :], mask=qmask, other=0.0)
    q_pe = gl.load(
        q_base + offs_qm[:, None] * stride_qt + BLOCK_DMODEL_NOPE + offs_qpe[None, :],
        mask=qmask,
        other=0.0,
    )

    # ---------------- K/V TDM descriptors + double-buffered LDS
    k_base = k_ptr + k_start.to(gl.int64) * stride_kt + h.to(gl.int64) * stride_kh
    k_pe_base = (
        k_pe_ptr + kpe_offset + k_start.to(gl.int64) * stride_kpe_t + h.to(gl.int64) * stride_kpe_h
    )
    v_base = v_ptr + k_start.to(gl.int64) * stride_vt + h.to(gl.int64) * stride_vh
    k_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_base, shape=(seqlen_k, BLOCK_DMODEL_NOPE), strides=(stride_kt, 1),
        block_shape=(BLOCK_N, BLOCK_DMODEL_NOPE), layout=K_SL,
    )
    k_pe_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=k_pe_base, shape=(seqlen_k, BLOCK_DMODEL_PE), strides=(stride_kpe_t, 1),
        block_shape=(BLOCK_N, BLOCK_DMODEL_PE), layout=KPE_SL,
    )
    v_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=v_base, shape=(seqlen_k, BLOCK_DMODEL_V), strides=(stride_vt, 1),
        block_shape=(BLOCK_N, BLOCK_DMODEL_V), layout=V_SL,
    )
    k_smem = gl.allocate_shared_memory(k_ptr.type.element_ty, [NB, BLOCK_N, BLOCK_DMODEL_NOPE], K_SL)
    k_pe_smem = gl.allocate_shared_memory(k_pe_ptr.type.element_ty, [NB, BLOCK_N, BLOCK_DMODEL_PE], KPE_SL)
    v_smem = gl.allocate_shared_memory(v_ptr.type.element_ty, [NB, BLOCK_N, BLOCK_DMODEL_V], V_SL)
    if HAS_V_HI:
        VHI_SL: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_DMODEL_V_HI, 16]], [BLOCK_N, BLOCK_DMODEL_V_HI], [1, 0]
        )
        v_hi_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=v_base + BLOCK_DMODEL_V, shape=(seqlen_k, BLOCK_DMODEL_V_HI), strides=(stride_vt, 1),
            block_shape=(BLOCK_N, BLOCK_DMODEL_V_HI), layout=VHI_SL,
        )
        v_hi_smem = gl.allocate_shared_memory(
            v_ptr.type.element_ty, [NB, BLOCK_N, BLOCK_DMODEL_V_HI], VHI_SL
        )

    qk_scale = sm_scale * RCP_LN2
    offs_om = gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, L))
    offs_od = gl.arange(0, BLOCK_DMODEL_V, layout=gl.SliceLayout(0, L))
    offs_od_hi = gl.arange(0, BLOCK_DMODEL_V_HI, layout=gl.SliceLayout(0, L))
    if HAS_INIT:
        # continue the online softmax from a previous partial result (o_init normalised,
        # lse_init natural log): m = lse / sm_scale, l = 1, acc = o; rows with
        # lse = -inf start empty.
        imask = (m0 + offs_om) < seqlen_q
        li = gl.load(
            lse_init_ptr + h.to(gl.int64) * stride_lsei_h
            + (q_start + m0 + offs_om).to(gl.int64) * stride_lsei_t,
            mask=imask,
            other=float("-inf"),
        )
        has = li > float("-inf")
        m_i = gl.where(has, li / sm_scale, float("-inf"))
        l_i = gl.where(has, 1.0, 0.0)
        oi_base = o_init_ptr + (q_start + m0).to(gl.int64) * stride_oit + h.to(gl.int64) * stride_oih
        oi = gl.load(
            oi_base + offs_om[:, None] * stride_oit + offs_od[None, :], mask=imask[:, None], other=0.0
        )
        acc = gl.where(has[:, None], oi.to(gl.float32), 0.0)
        if HAS_V_HI:
            oi_hi = gl.load(
                oi_base + offs_om[:, None] * stride_oit + BLOCK_DMODEL_V + offs_od_hi[None, :],
                mask=imask[:, None],
                other=0.0,
            )
            acc_hi = gl.where(has[:, None], oi_hi.to(gl.float32), 0.0)
    else:
        m_i = gl.full([BLOCK_M], float("-inf"), gl.float32, gl.SliceLayout(1, L))
        l_i = gl.zeros([BLOCK_M], gl.float32, gl.SliceLayout(1, L))
        acc = gl.zeros([BLOCK_M, BLOCK_DMODEL_V], gl.float32, L)
        if HAS_V_HI:
            acc_hi = gl.zeros([BLOCK_M, BLOCK_DMODEL_V_HI], gl.float32, L)
    qpos = m0 + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, L))

    # this program covers K tiles [tb, te): all of them, or one KV split
    tb = 0
    te = n_end
    if NUM_SPLITS > 1:
        per = gl.cdiv(n_end, NUM_SPLITS)
        tb = gl.minimum(split * per, n_end)
        te = gl.minimum(tb + per, n_end)
    if HAS_V_HI:
        _tdm_load_kv_v_hi(k_desc, k_pe_desc, v_desc, v_hi_desc, k_smem, k_pe_smem, v_smem, v_hi_smem,
                          tb * BLOCK_N, 0, tb < te)
    else:
        _tdm_load_kv(k_desc, k_pe_desc, v_desc, k_smem, k_pe_smem, v_smem, tb * BLOCK_N, 0, tb < te)
    # unmasked tiles, then the tiles that need the ragged-K / causal mask
    for j in range(tb, gl.minimum(n_full, te)):
        buf = (j - tb) % NB
        gl.amd.gfx1250.tdm.async_wait(0)
        if HAS_V_HI:
            _tdm_load_kv_v_hi(k_desc, k_pe_desc, v_desc, v_hi_desc, k_smem, k_pe_smem, v_smem, v_hi_smem,
                              (j + 1) * BLOCK_N, 1 - buf, j + 1 < te)
            m_i, l_i, acc, acc_hi = _attn_tile_v_hi(
                q, q_pe, k_smem, k_pe_smem, v_smem, v_hi_smem, buf, m_i, l_i, acc, acc_hi, qk_scale, j,
                qpos, seqlen_k, causal_off, False, IS_CAUSAL, BLOCK_M, BLOCK_N, L, KL, PL,
            )
        else:
            _tdm_load_kv(k_desc, k_pe_desc, v_desc, k_smem, k_pe_smem, v_smem,
                         (j + 1) * BLOCK_N, 1 - buf, j + 1 < te)
            m_i, l_i, acc = _attn_tile(
                q, q_pe, k_smem, k_pe_smem, v_smem, buf, m_i, l_i, acc, qk_scale, j, qpos, seqlen_k,
                causal_off, False, IS_CAUSAL, BLOCK_M, BLOCK_N, L, KL, PL,
            )
    for j in range(gl.maximum(n_full, tb), te):
        buf = (j - tb) % NB
        gl.amd.gfx1250.tdm.async_wait(0)
        if HAS_V_HI:
            _tdm_load_kv_v_hi(k_desc, k_pe_desc, v_desc, v_hi_desc, k_smem, k_pe_smem, v_smem, v_hi_smem,
                              (j + 1) * BLOCK_N, 1 - buf, j + 1 < te)
            m_i, l_i, acc, acc_hi = _attn_tile_v_hi(
                q, q_pe, k_smem, k_pe_smem, v_smem, v_hi_smem, buf, m_i, l_i, acc, acc_hi, qk_scale, j,
                qpos, seqlen_k, causal_off, True, IS_CAUSAL, BLOCK_M, BLOCK_N, L, KL, PL,
            )
        else:
            _tdm_load_kv(k_desc, k_pe_desc, v_desc, k_smem, k_pe_smem, v_smem,
                         (j + 1) * BLOCK_N, 1 - buf, j + 1 < te)
            m_i, l_i, acc = _attn_tile(
                q, q_pe, k_smem, k_pe_smem, v_smem, buf, m_i, l_i, acc, qk_scale, j, qpos, seqlen_k,
                causal_off, True, IS_CAUSAL, BLOCK_M, BLOCK_N, L, KL, PL,
            )
    gl.amd.gfx1250.tdm.async_wait(0)

    # ---------------- epilogue (rows without any visible key: o = 0, lse = -inf)
    l_safe = gl.where(l_i == 0.0, 1.0, l_i)
    rcp_l = 1.0 / l_safe
    out = acc * rcp_l[:, None]
    o_base = o_ptr + (q_start + m0).to(gl.int64) * stride_ot + h.to(gl.int64) * stride_oh + split * stride_os
    omask = (m0 + offs_om)[:, None] < seqlen_q
    gl.store(o_base + offs_om[:, None] * stride_ot + offs_od[None, :], out.to(o_ptr.type.element_ty), mask=omask)
    if HAS_V_HI:
        out_hi = acc_hi * rcp_l[:, None]
        gl.store(
            o_base + offs_om[:, None] * stride_ot + BLOCK_DMODEL_V + offs_od_hi[None, :],
            out_hi.to(o_ptr.type.element_ty),
            mask=omask,
        )
    if RETURN_LSE:
        lse = gl.where(l_i == 0.0, float("-inf"), m_i * sm_scale + gl.log(l_safe))
        l_base = (
            lse_ptr + h.to(gl.int64) * stride_lse_h + (q_start + m0).to(gl.int64) * stride_lse_t
            + split * stride_lse_s
        )
        gl.store(l_base + offs_om * stride_lse_t, lse, mask=(m0 + offs_om) < seqlen_q)


_attn_fwd_repr = make_kernel_repr(
    "_attn_fwd_gluon_gfx1250",
    ["BLOCK_M", "BLOCK_N", "IS_CAUSAL", "RETURN_LSE", "PERSISTENT", "NUM_SPLITS", "HAS_INIT", "HAS_V_HI"],
)


@gluon.jit(repr=_attn_fwd_repr)
def _attn_fwd(
    q_ptr, k_ptr, k_pe_ptr, v_ptr, o_ptr, lse_ptr,
    cu_seqlens_q, cu_seqlens_k,
    stride_qt, stride_qh, stride_kt, stride_kh, kpe_offset, stride_kpe_t, stride_kpe_h, stride_vt, stride_vh,
    stride_ot, stride_oh, stride_lse_h, stride_lse_t,
    sm_scale, num_seqs,
    stride_os, stride_lse_s,
    o_init_ptr, lse_init_ptr, stride_oit, stride_oih, stride_lsei_h, stride_lsei_t,
    n_items, num_heads,
    BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, NUM_WARPS: gl.constexpr,
    IS_CAUSAL: gl.constexpr, RETURN_LSE: gl.constexpr,
    PERSISTENT: gl.constexpr, NUM_SPLITS: gl.constexpr, HAS_INIT: gl.constexpr, HAS_V_HI: gl.constexpr,
):
    if PERSISTENT:
        # work items ranked longest-first (reverse q-block order, heads interleaved),
        # dealt to the programs in snake (boustrophedon) order. n_items = n_qblk * H.
        G = gl.num_programs(0)
        p = gl.program_id(0)
        n_rounds = gl.cdiv(n_items, G)
        n_qblk = n_items // num_heads
        for r in range(0, n_rounds):
            if r % 2 == 0:
                item = r * G + p
            else:
                item = r * G + (G - 1 - p)
            if item < n_items:
                _attn_fwd_block(
                    n_qblk - 1 - item // num_heads, item % num_heads, 0,
                    q_ptr, k_ptr, k_pe_ptr, v_ptr, o_ptr, lse_ptr, cu_seqlens_q, cu_seqlens_k,
                    stride_qt, stride_qh, stride_kt, stride_kh, kpe_offset, stride_kpe_t, stride_kpe_h, stride_vt,
                    stride_vh, stride_ot, stride_oh, stride_lse_h, stride_lse_t, stride_os, stride_lse_s,
                    o_init_ptr, lse_init_ptr, stride_oit, stride_oih, stride_lsei_h, stride_lsei_t,
                    sm_scale, num_seqs,
                    BLOCK_M, BLOCK_N, NUM_WARPS, IS_CAUSAL, RETURN_LSE, NUM_SPLITS, HAS_INIT, HAS_V_HI,
                )
    else:
        qblk = gl.program_id(0)
        if IS_CAUSAL:
            # longest q blocks (most K tiles) first
            qblk = gl.num_programs(0) - 1 - qblk
        _attn_fwd_block(
            qblk, gl.program_id(1), gl.program_id(2),
            q_ptr, k_ptr, k_pe_ptr, v_ptr, o_ptr, lse_ptr, cu_seqlens_q, cu_seqlens_k,
            stride_qt, stride_qh, stride_kt, stride_kh, kpe_offset, stride_kpe_t, stride_kpe_h, stride_vt,
            stride_vh, stride_ot, stride_oh, stride_lse_h, stride_lse_t, stride_os, stride_lse_s,
            o_init_ptr, lse_init_ptr, stride_oit, stride_oih, stride_lsei_h, stride_lsei_t,
            sm_scale, num_seqs,
            BLOCK_M, BLOCK_N, NUM_WARPS, IS_CAUSAL, RETURN_LSE, NUM_SPLITS, HAS_INIT, HAS_V_HI,
        )


_attn_fwd_combine_repr = make_kernel_repr(
    "_attn_fwd_combine_gfx1250", ["HEAD_DIM", "NUM_SPLITS", "RETURN_LSE", "HAS_INIT"]
)


@triton.jit(repr=_attn_fwd_combine_repr)
def _attn_fwd_combine(
    o_part_ptr, lse_part_ptr, o_ptr, lse_ptr, o_init_ptr, lse_init_ptr,
    stride_ops, stride_opt, stride_oph, stride_lps, stride_lph,
    stride_ot, stride_oh, stride_lse_h, stride_lse_t,
    stride_oit, stride_oih, stride_lsei_h, stride_lsei_t,
    HEAD_DIM: tl.constexpr, NUM_SPLITS: tl.constexpr, BLOCK_D: tl.constexpr,
    RETURN_LSE: tl.constexpr, HAS_INIT: tl.constexpr,
):
    """Merge the split-KV partials (+ optional initial state) of one (token, head):
    o = sum_s exp(lse_s - lse) * o_s, lse = logsumexp_s lse_s."""
    t = tl.program_id(0).to(tl.int64)
    h = tl.program_id(1).to(tl.int64)
    offs = tl.arange(0, BLOCK_D)
    dmask = offs < HEAD_DIM
    sp = tl.arange(0, NUM_SPLITS)
    lse_s = tl.load(lse_part_ptr + sp * stride_lps + h * stride_lph + t)  # [NUM_SPLITS]
    m = tl.max(lse_s, 0)
    if HAS_INIT:
        lse_i = tl.load(lse_init_ptr + h * stride_lsei_h + t * stride_lsei_t)
        m = tl.maximum(m, lse_i)
    m_safe = tl.where(m == float("-inf"), 0.0, m)
    w = tl.exp(lse_s - m_safe)
    w_sum = tl.sum(w, 0)
    o = tl.load(
        o_part_ptr + sp[:, None] * stride_ops + t * stride_opt + h * stride_oph + offs[None, :],
        mask=dmask[None, :],
        other=0.0,
    )  # [NUM_SPLITS, BLOCK_D]
    o = tl.sum(o * w[:, None], 0)
    if HAS_INIT:
        w_i = tl.exp(lse_i - m_safe)
        o_i = tl.load(o_init_ptr + t * stride_oit + h * stride_oih + offs, mask=dmask, other=0.0)
        o = o + w_i * o_i.to(tl.float32)
        w_sum = w_sum + w_i
    o = o / tl.where(w_sum == 0.0, 1.0, w_sum)
    tl.store(o_ptr + t * stride_ot + h * stride_oh + offs, o.to(o_ptr.dtype.element_ty), mask=dmask)
    if RETURN_LSE:
        lse = tl.where(w_sum == 0.0, float("-inf"), m_safe + tl.log(w_sum))
        tl.store(lse_ptr + h * stride_lse_h + t * stride_lse_t, lse)


def _get_config(
    causal: bool = False,
    v_head_dim: int = 128,
    seqlen_k: int = 0,
    num_q_heads: int = 1,
    total_q: int = 0,
    num_cus: int = 1,
):
    """Launch config for one call: tile, waves, persistent schedule and KV split.

    `pe_narrow_v` is the 128-wide V (v head dim < qk head dim), `pe` the 192-wide
    V. The 192-wide V needs more registers, so it uses one wave per SIMD. Causal
    calls use the persistent schedule. Non-causal calls above
    `short_max_seqlen_k` keys split K when the q blocks cannot fill the GPU.
    """
    arch = arch_info.get_arch()
    fpath = f"{AITER_TRITON_CONFIGS_PATH}/{arch}/gluon/attention/mha/mha.json"
    fwd_cfg = load_config_json(fpath)["fwd"]
    family = "pe_narrow_v" if v_head_dim < BLOCK_DMODEL_NOPE + BLOCK_DMODEL_PE else "pe"
    if causal:
        return dict(fwd_cfg[f"{family}_causal"], NUM_SPLITS=1)
    if seqlen_k <= fwd_cfg["short_max_seqlen_k"]:
        return dict(fwd_cfg[f"{family}_short"], NUM_SPLITS=1)
    cfg = dict(fwd_cfg[family])
    # Split K when there are fewer than num_cus / 2 programs, while each split
    # keeps at least split_kv_min_tiles_per_split K tiles.
    progs = max(1, triton.cdiv(total_q, cfg["BLOCK_M"])) * num_q_heads
    ns = 1
    while (
        ns < fwd_cfg["split_kv_max_num_splits"]
        and progs * ns * 2 <= num_cus
        and seqlen_k // (ns * 2 * cfg["BLOCK_N"]) >= fwd_cfg["split_kv_min_tiles_per_split"]
    ):
        ns *= 2
    cfg["NUM_SPLITS"] = ns
    return cfg
