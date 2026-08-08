# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2026, Songlin Yang, Yu Zhang, Zhiyuan Li

"""
Intra-chunk attention kernels for chunk_delta_attn forward pass.

Three-step pipeline:
  1. Token-parallel kernel: computes diagonal Akk blocks + full Aqk row-by-row.
  2. Sub-chunk intra kernel (safe-gate path): alternative diagonal-block kernel.
  3. Fused inter + solve_tril kernel: computes off-diagonal Akk, then solves
     the full chunk triangular system in one pass → writes Akk_inv.

The Python dispatch ``chunk_delta_attn_fwd_intra`` orchestrates all three steps
and returns w, u, qg, kg, Aqk, Akk as required by the top-level forward.
"""

import torch
import triton
import triton.language as tl

from .utils import (
    IS_GATHER_SUPPORTED,
    IS_TF32_SUPPORTED,
    autotune_cache_kwargs,
    chunk_delta_attn_autotune_configs,
    exp2,
    input_guard,
    prepare_chunk_indices,
)
from .wy_fast import recompute_w_u_fwd

if IS_TF32_SUPPORTED:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("tf32")
else:
    SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")


# Fall back to a no-op gather if tl.gather is unavailable
if IS_GATHER_SUPPORTED:
    _gather = tl.gather
else:

    @triton.jit
    def _gather(src, index, axis, _builder=None):
        """Row-gather fallback: broadcasts the selected row across the output."""
        return tl.sum(src * (tl.arange(0, src.shape[0]) == index)[:, None], 0)[None, :]


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=chunk_delta_attn_autotune_configs(
        [
            triton.Config({"BH": BH, "BK": BK}, num_warps=nw)
            for BH in [1, 2, 4, 8]
            for BK in [32, 64]
            for nw in [1, 2, 4, 8]
        ],
        default_config=triton.Config({"BH": 1, "BK": 64}, num_warps=4),
    ),
    key=["K", "H", "HV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T", "N"])
def chunk_delta_attn_fwd_kernel_intra_token_parallel(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    N,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BH: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Token-parallel: each thread-block handles one output token (i_tg) and
    a BH-wide head stripe (i_hg).  Writes directly to Aqk and Akk.
    """
    i_tg, i_hg = tl.program_id(0).to(tl.int64), tl.program_id(1)

    if IS_VARLEN:
        i_n = 0
        left, right = 0, N
        for _ in range(20):
            if left < right:
                mid = (left + right) // 2
                if i_tg < tl.load(cu_seqlens + mid + 1).to(tl.int32):
                    right = mid
                else:
                    left = mid + 1
        i_n = left
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
        i_t = i_tg - bos
    else:
        bos = (i_tg // T) * T
        i_t = i_tg % T

    if i_t >= T:
        return

    i_c = i_t // BT
    i_s = (i_t % BT) // BC
    i_tc = i_c * BT
    i_ts = i_tc + i_s * BC

    G: tl.constexpr = HV // H

    q += bos * H * K
    k += bos * H * K
    g += bos * HV * K
    Aqk += bos * HV * BT
    Akk += bos * HV * BC
    beta += bos * HV

    o_hv = i_hg * BH + tl.arange(0, BH)
    o_h = o_hv // G
    m_hv = o_hv < HV

    p_beta = beta + i_t * HV + o_hv
    b_beta = tl.load(p_beta, mask=m_hv, other=0.0).to(tl.float32)

    # Accumulate dot products across K-tiles (BK ≤ 64 to avoid slow tl.dot / huge scatter loads)
    for j in range(i_ts, min(i_t + 1, min(T, i_ts + BC))):
        b_Aqk_j = tl.zeros([BH], dtype=tl.float32)
        b_Akk_j = tl.zeros([BH], dtype=tl.float32)
        for i_k in range(tl.cdiv(K, BK)):
            o_k = i_k * BK + tl.arange(0, BK)
            m_k = o_k < K
            m_hk = m_hv[:, None] & m_k[None, :]
            p_qk = o_h[:, None] * K + o_k[None, :]

            b_q = tl.load(q + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
            b_k = tl.load(k + i_t * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)
            b_kj = tl.load(k + j * H * K + p_qk, mask=m_hk, other=0).to(tl.float32)

            p_g = g + i_t * HV * K + o_hv[:, None] * K + o_k[None, :]
            p_gj = g + j * HV * K + o_hv[:, None] * K + o_k[None, :]
            b_g = tl.load(p_g, mask=m_hk, other=0.0).to(tl.float32)
            b_gj = tl.load(p_gj, mask=m_hk, other=0.0).to(tl.float32)

            b_kgj = tl.where(m_k[None, :], b_kj * exp2(b_g - b_gj), 0.0)
            b_Aqk_j += tl.sum(b_q * b_kgj, axis=1)
            b_Akk_j += tl.sum(b_k * b_beta[:, None] * b_kgj, axis=1)

        b_Aqk_j *= scale
        b_Akk_j *= tl.where(j < i_t, 1.0, 0.0)
        tl.store(
            Aqk + i_t * HV * BT + o_hv * BT + j % BT,
            b_Aqk_j.to(Aqk.dtype.element_ty),
            mask=m_hv,
        )
        tl.store(
            Akk + i_t * HV * BC + o_hv * BC + j - i_ts,
            b_Akk_j.to(Akk.dtype.element_ty),
            mask=m_hv,
        )


def _chunk_delta_attn_fwd_intra_token_parallel(
    q: torch.Tensor,
    k: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    Aqk: torch.Tensor,
    Akk: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    sub_chunk_size: int = 16,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, H, K = q.shape
    HV = gk.shape[2]
    N = len(cu_seqlens) - 1 if cu_seqlens is not None else B
    BT = chunk_size
    BC = sub_chunk_size

    def grid(meta):
        return (B * T, triton.cdiv(HV, meta["BH"]))

    chunk_delta_attn_fwd_kernel_intra_token_parallel[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        N=N,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
    )
    return Aqk, Akk


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=chunk_delta_attn_autotune_configs(
        [
            triton.Config({}, num_warps=nw, num_stages=ns)
            for nw in [1, 2, 4, 8]
            for ns in [2, 3, 4]
        ],
        default_config=triton.Config({}, num_warps=1, num_stages=3),
    ),
    key=["BT", "BC", "HV"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_delta_attn_fwd_kernel_intra_sub_chunk(
    q,
    k,
    g,
    beta,
    Aqk,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_GATHER: tl.constexpr,
    CACHE_QK: tl.constexpr = "",
    CACHE_G: tl.constexpr = "",
    CACHE_BETA: tl.constexpr = "",
    USE_REG_SUBSTITUTION: tl.constexpr = False,
):
    i_t, i_i, i_bh = (
        tl.program_id(0).to(tl.int64),
        tl.program_id(1),
        tl.program_id(2).to(tl.int64),
    )
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    i_ti = i_t * BT + i_i * BC
    if i_ti >= T:
        return

    o_c = i_ti + tl.arange(0, BC)
    m_c = o_c < T

    q = q + (bos * H + i_h) * K
    k = k + (bos * H + i_h) * K
    g = g + (bos * HV + i_hv) * K
    beta = beta + bos * HV + i_hv
    Aqk = Aqk + (bos * HV + i_hv) * BT
    Akk = Akk + (bos * HV + i_hv) * BC

    p_beta = beta + o_c * HV
    b_beta = tl.load(p_beta, mask=m_c, other=0.0, cache_modifier=CACHE_BETA)

    # Reference gate at the mid-point of this sub-chunk (same for all K tiles)
    i_gn = i_ti + min(BC // 2, T - i_ti - 1)

    # Accumulate Aqk and Akk over K tiles (BK ≤ 64 to keep tl.dot shape small)
    b_Aqk = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk = tl.zeros([BC, BC], dtype=tl.float32)
    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K
        m_ck = m_c[:, None] & m_k[None, :]
        p_q = q + o_c[:, None] * (H * K) + o_k[None, :]
        p_k = k + o_c[:, None] * (H * K) + o_k[None, :]
        p_g = g + o_c[:, None] * (HV * K) + o_k[None, :]
        b_q = tl.load(p_q, mask=m_ck, other=0.0, cache_modifier=CACHE_QK)
        b_k = tl.load(p_k, mask=m_ck, other=0.0, cache_modifier=CACHE_QK)
        b_g = tl.load(p_g, mask=m_ck, other=0.0, cache_modifier=CACHE_G)

        if USE_GATHER:
            b_gn = _gather(
                b_g,
                tl.full([1, BK], min(BC // 2, T - i_ti - 1), dtype=tl.int16),
                axis=0,
            )
        else:
            p_gn = g + i_gn * HV * K + o_k
            b_gn = tl.load(p_gn, mask=m_k, other=0.0, cache_modifier=CACHE_G)
            b_gn = b_gn[None, :]

        b_gm = (b_g - b_gn).to(tl.float32)
        b_gq = tl.where(m_c[:, None], exp2(b_gm), 0.0)
        b_gk = tl.where(m_c[:, None], exp2(-b_gm), 0.0)

        b_kgt = tl.trans(b_k * b_gk)
        b_Aqk += tl.dot(b_q * b_gq, b_kgt)
        b_Akk += tl.dot(b_k * b_gq, b_kgt)

    b_Aqk *= scale
    b_Akk *= b_beta[:, None]

    o_i = tl.arange(0, BC)
    m_Aqk = o_i[:, None] >= o_i[None, :]
    m_Akk = o_i[:, None] > o_i[None, :]
    m_I = o_i[:, None] == o_i[None, :]

    b_Aqk = tl.where(m_Aqk, b_Aqk, 0.0)
    b_Akk = tl.where(m_Akk, b_Akk, 0.0)

    m_Aqk_st = m_c[:, None] & (o_i[None, :] < BT)
    m_Akk_st = m_c[:, None] & (o_i[None, :] < BC)
    p_Aqk = Aqk + o_c[:, None] * (HV * BT) + (i_i * BC + o_i)[None, :]
    p_Akk = Akk + o_c[:, None] * (HV * BC) + o_i[None, :]
    tl.store(p_Aqk, b_Aqk.to(Aqk.dtype.element_ty), mask=m_Aqk_st)

    # Forward substitution
    if USE_REG_SUBSTITUTION:
        # Register-based: no global load-back
        b_Ai = -b_Akk
        for i in range(2, min(BC, T - i_ti)):
            b_a = tl.sum(b_Ai * (o_i == i)[:, None].to(b_Ai.dtype), 0)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I
        tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), mask=m_Akk_st)
    else:
        # Global load-back
        tl.store(p_Akk, b_Akk.to(Akk.dtype.element_ty), mask=m_Akk_st)
        tl.debug_barrier()

        b_Ai = -b_Akk
        for i in range(2, min(BC, T - i_ti)):
            b_a = -tl.load(Akk + (i_ti + i) * HV * BC + o_i)
            b_a = tl.where(o_i < i, b_a, 0.0)
            b_a += tl.sum(b_a[:, None] * b_Ai, 0)
            b_Ai = tl.where((o_i == i)[:, None], b_a, b_Ai)
        b_Ai += m_I
        tl.store(p_Akk, b_Ai.to(Akk.dtype.element_ty), mask=m_Akk_st)


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.autotune(
    configs=chunk_delta_attn_autotune_configs(
        [
            triton.Config({"BK": BK}, num_warps=nw)
            for BK in [32, 64]
            for nw in [1, 2, 4]
        ],
        default_config=triton.Config({"BK": 32}, num_warps=1),
    ),
    key=["H", "HV", "K", "BT", "BC", "NC"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def chunk_delta_attn_fwd_kernel_inter_solve_fused(
    q,
    k,
    g,
    beta,
    Aqk,
    Akkd,
    Akk,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    HV: tl.constexpr,
    K: tl.constexpr,
    BT: tl.constexpr,
    BC: tl.constexpr,
    NC: tl.constexpr,
    BK: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    USE_SAFE_GATE: tl.constexpr,
):
    """
    Fused kernel:
      1. Compute off-diagonal Aqk and Akk blocks (cross-sub-chunk).
      2. Load diagonal Akkd blocks (fp32) computed by the previous kernel.
      3. Forward-substitute diagonals → block-level Akk_inv.
      4. Compute merged cross-diagonal Akk_inv blocks.
      5. Write full Akk_inv to Akk.
    """
    i_t, i_bh = tl.program_id(0).to(tl.int64), tl.program_id(1).to(tl.int64)
    i_b, i_hv = i_bh // HV, i_bh % HV
    i_h = i_hv // (HV // H)

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int64),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int64),
            tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if i_t * BT >= T:
        return

    i_tc0 = i_t * BT
    i_tc1 = i_t * BT + BC
    i_tc2 = i_t * BT + 2 * BC
    i_tc3 = i_t * BT + 3 * BC

    q += (bos * H + i_h) * K
    k += (bos * H + i_h) * K
    g += (bos * HV + i_hv) * K
    Aqk += (bos * HV + i_hv) * BT
    Akk += (bos * HV + i_hv) * BT
    Akkd += (bos * HV + i_hv) * BC

    o_i = tl.arange(0, BC)
    m_tc1 = (i_tc1 + o_i) < T
    m_tc2 = (i_tc2 + o_i) < T
    m_tc3 = (i_tc3 + o_i) < T
    o_c0 = i_tc0 + o_i
    o_c1 = i_tc1 + o_i
    o_c2 = i_tc2 + o_i
    o_c3 = i_tc3 + o_i
    m_tc0 = o_c0 < T
    m_A0 = m_tc0[:, None] & (o_i[None, :] < BT)
    m_A1 = m_tc1[:, None] & (o_i[None, :] < BT)
    m_A2 = m_tc2[:, None] & (o_i[None, :] < BT)
    m_A3 = m_tc3[:, None] & (o_i[None, :] < BT)

    b_Aqk10 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk10 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk20 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk21 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk21 = tl.zeros([BC, BC], dtype=tl.float32)

    b_Aqk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk30 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk31 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Aqk32 = tl.zeros([BC, BC], dtype=tl.float32)
    b_Akk32 = tl.zeros([BC, BC], dtype=tl.float32)

    for i_k in range(tl.cdiv(K, BK)):
        o_k = i_k * BK + tl.arange(0, BK)
        m_k = o_k < K

        m_ck0 = m_tc0[:, None] & m_k[None, :]
        p_k0 = k + o_c0[:, None] * (H * K) + o_k[None, :]
        p_g0 = g + o_c0[:, None] * (HV * K) + o_k[None, :]
        b_k0 = tl.load(p_k0, mask=m_ck0, other=0.0).to(tl.float32)
        b_g0 = tl.load(p_g0, mask=m_ck0, other=0.0).to(tl.float32)

        if i_tc1 < T:
            m_ck1 = m_tc1[:, None] & m_k[None, :]
            p_q1 = q + o_c1[:, None] * (H * K) + o_k[None, :]
            p_k1 = k + o_c1[:, None] * (H * K) + o_k[None, :]
            p_g1 = g + o_c1[:, None] * (HV * K) + o_k[None, :]
            b_q1 = tl.load(p_q1, mask=m_ck1, other=0.0).to(tl.float32)
            b_k1 = tl.load(p_k1, mask=m_ck1, other=0.0).to(tl.float32)
            b_g1 = tl.load(p_g1, mask=m_ck1, other=0.0).to(tl.float32)
            b_gn1 = tl.load(g + i_tc1 * HV * K + o_k, mask=m_k, other=0).to(tl.float32)
            b_gqn = tl.where(m_tc1[:, None], exp2(b_g1 - b_gn1[None, :]), 0)
            b_kgt = tl.trans(b_k0 * exp2(b_gn1[None, :] - b_g0))
            b_Aqk10 += tl.dot(b_q1 * b_gqn, b_kgt)
            b_Akk10 += tl.dot(b_k1 * b_gqn, b_kgt)

            if NC >= 3 and i_tc2 < T:
                m_ck2 = m_tc2[:, None] & m_k[None, :]
                p_q2 = q + o_c2[:, None] * (H * K) + o_k[None, :]
                p_k2 = k + o_c2[:, None] * (H * K) + o_k[None, :]
                p_g2 = g + o_c2[:, None] * (HV * K) + o_k[None, :]
                b_q2 = tl.load(p_q2, mask=m_ck2, other=0.0).to(tl.float32)
                b_k2 = tl.load(p_k2, mask=m_ck2, other=0.0).to(tl.float32)
                b_g2 = tl.load(p_g2, mask=m_ck2, other=0.0).to(tl.float32)
                b_gn2 = tl.load(g + i_tc2 * HV * K + o_k, mask=m_k, other=0).to(
                    tl.float32
                )
                b_gqn2 = tl.where(m_tc2[:, None], exp2(b_g2 - b_gn2[None, :]), 0)
                b_qg2 = b_q2 * b_gqn2
                b_kg2 = b_k2 * b_gqn2
                b_kgt = tl.trans(b_k0 * exp2(b_gn2[None, :] - b_g0))
                b_Aqk20 += tl.dot(b_qg2, b_kgt)
                b_Akk20 += tl.dot(b_kg2, b_kgt)
                b_kgt = tl.trans(b_k1 * exp2(b_gn2[None, :] - b_g1))
                b_Aqk21 += tl.dot(b_qg2, b_kgt)
                b_Akk21 += tl.dot(b_kg2, b_kgt)

                if NC >= 4 and i_tc3 < T:
                    m_ck3 = m_tc3[:, None] & m_k[None, :]
                    p_q3 = q + o_c3[:, None] * (H * K) + o_k[None, :]
                    p_k3 = k + o_c3[:, None] * (H * K) + o_k[None, :]
                    p_g3 = g + o_c3[:, None] * (HV * K) + o_k[None, :]
                    b_q3 = tl.load(p_q3, mask=m_ck3, other=0.0).to(tl.float32)
                    b_k3 = tl.load(p_k3, mask=m_ck3, other=0.0).to(tl.float32)
                    b_g3 = tl.load(p_g3, mask=m_ck3, other=0.0).to(tl.float32)
                    b_gn3 = tl.load(g + i_tc3 * HV * K + o_k, mask=m_k, other=0).to(
                        tl.float32
                    )
                    b_gqn3 = tl.where(m_tc3[:, None], exp2(b_g3 - b_gn3[None, :]), 0)
                    b_qg3 = b_q3 * b_gqn3
                    b_kg3 = b_k3 * b_gqn3
                    b_kgt = tl.trans(b_k0 * exp2(b_gn3[None, :] - b_g0))
                    b_Aqk30 += tl.dot(b_qg3, b_kgt)
                    b_Akk30 += tl.dot(b_kg3, b_kgt)
                    b_kgt = tl.trans(b_k1 * exp2(b_gn3[None, :] - b_g1))
                    b_Aqk31 += tl.dot(b_qg3, b_kgt)
                    b_Akk31 += tl.dot(b_kg3, b_kgt)
                    b_kgt = tl.trans(b_k2 * exp2(b_gn3[None, :] - b_g2))
                    b_Aqk32 += tl.dot(b_qg3, b_kgt)
                    b_Akk32 += tl.dot(b_kg3, b_kgt)

    if i_tc1 < T:
        p_Aqk10 = Aqk + o_c1[:, None] * (HV * BT) + o_i[None, :]
        tl.store(p_Aqk10, (b_Aqk10 * scale).to(Aqk.dtype.element_ty), mask=m_A1)
        p_b1 = beta + bos * HV + i_hv + o_c1 * HV
        b_b1 = tl.load(p_b1, mask=m_tc1, other=0.0).to(tl.float32)
        b_Akk10 *= b_b1[:, None]
    if NC >= 3 and i_tc2 < T:
        p_Aqk20 = Aqk + o_c2[:, None] * (HV * BT) + o_i[None, :]
        p_Aqk21 = Aqk + o_c2[:, None] * (HV * BT) + (o_i + BC)[None, :]
        tl.store(p_Aqk20, (b_Aqk20 * scale).to(Aqk.dtype.element_ty), mask=m_A2)
        tl.store(p_Aqk21, (b_Aqk21 * scale).to(Aqk.dtype.element_ty), mask=m_A2)
        p_b2 = beta + bos * HV + i_hv + o_c2 * HV
        b_b2 = tl.load(p_b2, mask=m_tc2, other=0.0).to(tl.float32)
        b_Akk20 *= b_b2[:, None]
        b_Akk21 *= b_b2[:, None]
    if NC >= 4 and i_tc3 < T:
        p_Aqk30 = Aqk + o_c3[:, None] * (HV * BT) + o_i[None, :]
        p_Aqk31 = Aqk + o_c3[:, None] * (HV * BT) + (o_i + BC)[None, :]
        p_Aqk32 = Aqk + o_c3[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :]
        tl.store(p_Aqk30, (b_Aqk30 * scale).to(Aqk.dtype.element_ty), mask=m_A3)
        tl.store(p_Aqk31, (b_Aqk31 * scale).to(Aqk.dtype.element_ty), mask=m_A3)
        tl.store(p_Aqk32, (b_Aqk32 * scale).to(Aqk.dtype.element_ty), mask=m_A3)
        p_b3 = beta + bos * HV + i_hv + o_c3 * HV
        b_b3 = tl.load(p_b3, mask=m_tc3, other=0.0).to(tl.float32)
        b_Akk30 *= b_b3[:, None]
        b_Akk31 *= b_b3[:, None]
        b_Akk32 *= b_b3[:, None]

    p_Akk00 = Akkd + o_c0[:, None] * (HV * BC) + o_i[None, :]
    p_Akk11 = Akkd + o_c1[:, None] * (HV * BC) + o_i[None, :]
    b_Ai00 = tl.load(p_Akk00, mask=m_A0, other=0.0).to(tl.float32)
    b_Ai11 = tl.load(p_Akk11, mask=m_A1, other=0.0).to(tl.float32)
    if NC >= 3:
        p_Akk22 = Akkd + o_c2[:, None] * (HV * BC) + o_i[None, :]
        b_Ai22 = tl.load(p_Akk22, mask=m_A2, other=0.0).to(tl.float32)
    if NC >= 4:
        p_Akk33 = Akkd + o_c3[:, None] * (HV * BC) + o_i[None, :]
        b_Ai33 = tl.load(p_Akk33, mask=m_A3, other=0.0).to(tl.float32)

    if not USE_SAFE_GATE:
        m_A = o_i[:, None] > o_i[None, :]
        m_I = o_i[:, None] == o_i[None, :]

        b_Ai00 = -tl.where(m_A, b_Ai00, 0)
        b_Ai11 = -tl.where(m_A, b_Ai11, 0)
        if NC >= 3:
            b_Ai22 = -tl.where(m_A, b_Ai22, 0)
        if NC >= 4:
            b_Ai33 = -tl.where(m_A, b_Ai33, 0)

        for i in range(2, min(BC, T - i_tc0)):
            b_a00 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a00 = tl.where(o_i < i, b_a00, 0.0)
            b_a00 += tl.sum(b_a00[:, None] * b_Ai00, 0)
            b_Ai00 = tl.where((o_i == i)[:, None], b_a00, b_Ai00)
        for i in range(BC + 2, min(2 * BC, T - i_tc0)):
            b_a11 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
            b_a11 = tl.where(o_i < i - BC, b_a11, 0.0)
            b_a11 += tl.sum(b_a11[:, None] * b_Ai11, 0)
            b_Ai11 = tl.where((o_i == i - BC)[:, None], b_a11, b_Ai11)
        if NC >= 3:
            for i in range(2 * BC + 2, min(3 * BC, T - i_tc0)):
                b_a22 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
                b_a22 = tl.where(o_i < i - 2 * BC, b_a22, 0.0)
                b_a22 += tl.sum(b_a22[:, None] * b_Ai22, 0)
                b_Ai22 = tl.where((o_i == i - 2 * BC)[:, None], b_a22, b_Ai22)
        if NC >= 4:
            for i in range(3 * BC + 2, min(4 * BC, T - i_tc0)):
                b_a33 = -tl.load(Akkd + (i_tc0 + i) * HV * BC + o_i)
                b_a33 = tl.where(o_i < i - 3 * BC, b_a33, 0.0)
                b_a33 += tl.sum(b_a33[:, None] * b_Ai33, 0)
                b_Ai33 = tl.where((o_i == i - 3 * BC)[:, None], b_a33, b_Ai33)

        b_Ai00 += m_I
        b_Ai11 += m_I
        if NC >= 3:
            b_Ai22 += m_I
        if NC >= 4:
            b_Ai33 += m_I

    b_Ai10 = -tl.dot(
        tl.dot(b_Ai11, b_Akk10, input_precision=SOLVE_TRIL_DOT_PRECISION),
        b_Ai00,
        input_precision=SOLVE_TRIL_DOT_PRECISION,
    )
    if NC >= 3:
        b_Ai21 = -tl.dot(
            tl.dot(b_Ai22, b_Akk21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai11,
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai20 = -tl.dot(
            b_Ai22,
            tl.dot(b_Akk20, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk21, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
    if NC >= 4:
        b_Ai32 = -tl.dot(
            tl.dot(b_Ai33, b_Akk32, input_precision=SOLVE_TRIL_DOT_PRECISION),
            b_Ai22,
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai31 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk31, b_Ai11, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk32, b_Ai21, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )
        b_Ai30 = -tl.dot(
            b_Ai33,
            tl.dot(b_Akk30, b_Ai00, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk31, b_Ai10, input_precision=SOLVE_TRIL_DOT_PRECISION)
            + tl.dot(b_Akk32, b_Ai20, input_precision=SOLVE_TRIL_DOT_PRECISION),
            input_precision=SOLVE_TRIL_DOT_PRECISION,
        )

    p_Akk00 = Akk + o_c0[:, None] * (HV * BT) + o_i[None, :]
    p_Akk10 = Akk + o_c1[:, None] * (HV * BT) + o_i[None, :]
    p_Akk11 = Akk + o_c1[:, None] * (HV * BT) + (o_i + BC)[None, :]
    tl.store(p_Akk00, b_Ai00.to(Akk.dtype.element_ty), mask=m_A0)
    tl.store(p_Akk10, b_Ai10.to(Akk.dtype.element_ty), mask=m_A1)
    tl.store(p_Akk11, b_Ai11.to(Akk.dtype.element_ty), mask=m_A1)
    if NC >= 3:
        p_Akk20 = Akk + o_c2[:, None] * (HV * BT) + o_i[None, :]
        p_Akk21 = Akk + o_c2[:, None] * (HV * BT) + (o_i + BC)[None, :]
        p_Akk22 = Akk + o_c2[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :]
        tl.store(p_Akk20, b_Ai20.to(Akk.dtype.element_ty), mask=m_A2)
        tl.store(p_Akk21, b_Ai21.to(Akk.dtype.element_ty), mask=m_A2)
        tl.store(p_Akk22, b_Ai22.to(Akk.dtype.element_ty), mask=m_A2)
    if NC >= 4:
        p_Akk30 = Akk + o_c3[:, None] * (HV * BT) + o_i[None, :]
        p_Akk31 = Akk + o_c3[:, None] * (HV * BT) + (o_i + BC)[None, :]
        p_Akk32 = Akk + o_c3[:, None] * (HV * BT) + (o_i + 2 * BC)[None, :]
        p_Akk33 = Akk + o_c3[:, None] * (HV * BT) + (o_i + 3 * BC)[None, :]
        tl.store(p_Akk30, b_Ai30.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk31, b_Ai31.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk32, b_Ai32.to(Akk.dtype.element_ty), mask=m_A3)
        tl.store(p_Akk33, b_Ai33.to(Akk.dtype.element_ty), mask=m_A3)


@input_guard
def chunk_delta_attn_fwd_intra(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
    skip_recompute_wu: bool = False,
) -> tuple:
    """Intra-chunk attention (base Triton sub-chunk kernel)."""
    B, T, H, K = k.shape
    HV = gk.shape[2]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(
            f"chunk_delta_attn intra kernel only supports chunk_size 32 or 64, got {BT}."
        )
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)

    if safe_gate:
        BK = min(64, triton.next_power_of_2(K))
        grid = (NT, NC, B * HV)
        chunk_delta_attn_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=IS_GATHER_SUPPORTED,
        )
    else:
        _chunk_delta_attn_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    grid = (NT, B * HV)
    chunk_delta_attn_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        USE_SAFE_GATE=safe_gate,
    )

    if skip_recompute_wu:
        return None, None, None, None, Aqk, Akk

    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=gk,
        q=q if disable_recompute else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk


@input_guard
def chunk_delta_attn_fwd_intra_opt(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gk: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    cu_seqlens: torch.Tensor | None = None,
    chunk_size: int = 64,
    chunk_indices: torch.Tensor | None = None,
    safe_gate: bool = False,
    disable_recompute: bool = False,
    skip_recompute_wu: bool = False,
) -> tuple:
    """Intra-chunk attention (optimized single-tile Triton sub-chunk kernel)."""
    B, T, H, K = k.shape
    HV = gk.shape[2]
    BT = chunk_size
    if BT not in (32, 64):
        raise ValueError(
            f"chunk_delta_attn intra kernel only supports chunk_size 32 or 64, got {BT}."
        )
    BC = 16
    if chunk_indices is None and cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    NC = triton.cdiv(BT, BC)

    Aqk = torch.empty(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akk = torch.zeros(B, T, HV, BT, device=k.device, dtype=k.dtype)
    Akkd = torch.empty(B, T, HV, BC, device=k.device, dtype=torch.float32)

    if safe_gate:
        BK = triton.next_power_of_2(K)
        grid = (NT, NC, B * HV)
        chunk_delta_attn_fwd_kernel_intra_sub_chunk[grid](
            q=q,
            k=k,
            g=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            T=T,
            H=H,
            HV=HV,
            K=K,
            BT=BT,
            BC=BC,
            BK=BK,
            USE_GATHER=IS_GATHER_SUPPORTED,
            CACHE_QK=".cs",
            CACHE_G=".cs",
            USE_REG_SUBSTITUTION=True,
        )
    else:
        _chunk_delta_attn_fwd_intra_token_parallel(
            q=q,
            k=k,
            gk=gk,
            beta=beta,
            Aqk=Aqk,
            Akk=Akkd,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=BT,
            sub_chunk_size=BC,
        )

    grid = (NT, B * HV)
    chunk_delta_attn_fwd_kernel_inter_solve_fused[grid](
        q=q,
        k=k,
        g=gk,
        beta=beta,
        Aqk=Aqk,
        Akkd=Akkd,
        Akk=Akk,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        HV=HV,
        K=K,
        BT=BT,
        BC=BC,
        NC=NC,
        USE_SAFE_GATE=safe_gate,
    )

    if skip_recompute_wu:
        return None, None, None, None, Aqk, Akk

    w, u, qg, kg = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=Akk,
        gk=gk,
        q=q if disable_recompute else None,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
    )
    return w, u, qg, kg, Aqk, Akk
