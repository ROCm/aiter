# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# Adapted from flash-linear-attention: Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

"""
Fused triangular solve + recompute w, u in a single kernel.

Eliminates the intermediate Ai tensor (64x64 per chunk x head) global
memory round-trip by keeping the inverse blocks in registers.
"""

import torch
import triton
import triton.language as tl

from ..gated_delta_rule_utils import autotune_cache_kwargs, IS_AMD, maybe_autotune
from ..utils import prepare_chunk_indices
from ..utils.op import exp
from ..utils.solve_tril import FLA_TRIL_PRECISION


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@maybe_autotune(
    configs=[
        triton.Config({}, num_warps=nw, num_stages=ns)
        for nw in [2, 4, 8]
        for ns in ([2, 3] if IS_AMD else [2, 3, 4])
    ],
    key=["H", "K", "V", "BT", "BK", "BV", "IS_VARLEN"],
    **autotune_cache_kwargs,
)
@triton.jit(do_not_specialize=["T"])
def fused_solve_tril_recompute_w_u_kernel(
    A_raw,
    k,
    v,
    beta,
    g,
    w,
    u,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    T_flat = T

    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos = i_b * T

    # ================================================================
    # Phase 1: compute (I + A)^{-1} in registers (triangular solve)
    # ================================================================
    o_i = tl.arange(0, 16)
    m_lo = o_i[:, None] > o_i[None, :]
    m_id = o_i[:, None] == o_i[None, :]
    A_base = A_raw + (bos * H + i_h) * BT

    p11 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT, 0), (16, 16), (1, 0)
    )
    p22 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 16), (16, 16), (1, 0)
    )
    p33 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 32), (16, 16), (1, 0)
    )
    p44 = tl.make_block_ptr(
        A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 48), (16, 16), (1, 0)
    )
    b11 = -tl.where(m_lo, tl.load(p11, boundary_check=(0, 1)).to(tl.float32), 0)
    b22 = -tl.where(m_lo, tl.load(p22, boundary_check=(0, 1)).to(tl.float32), 0)
    b33 = -tl.where(m_lo, tl.load(p33, boundary_check=(0, 1)).to(tl.float32), 0)
    b44 = -tl.where(m_lo, tl.load(p44, boundary_check=(0, 1)).to(tl.float32), 0)

    for i in range(2, min(16, T - i_t * BT)):
        r = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i)
        r = r + tl.sum(r[:, None] * b11, 0)
        b11 = tl.where((o_i == i)[:, None], r, b11)
    for i in range(18, min(32, T - i_t * BT)):
        r = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 16)
        r = r + tl.sum(r[:, None] * b22, 0)
        b22 = tl.where((o_i == i - 16)[:, None], r, b22)
    for i in range(34, min(48, T - i_t * BT)):
        r = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 32)
        r = r + tl.sum(r[:, None] * b33, 0)
        b33 = tl.where((o_i == i - 32)[:, None], r, b33)
    for i in range(50, min(64, T - i_t * BT)):
        r = -tl.load(A_base + (i_t * BT + i) * H * BT + o_i + 48)
        r = r + tl.sum(r[:, None] * b44, 0)
        b44 = tl.where((o_i == i - 48)[:, None], r, b44)
    b11 += m_id
    b22 += m_id
    b33 += m_id
    b44 += m_id

    rA21 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 16, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA31 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA32 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 32, 16), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA41 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 0), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA42 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 16), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)
    rA43 = tl.load(
        tl.make_block_ptr(
            A_base, (T, BT), (H * BT, 1), (i_t * BT + 48, 32), (16, 16), (1, 0)
        ),
        boundary_check=(0, 1),
    ).to(tl.float32)

    b21 = -tl.dot(
        tl.dot(b22, rA21, input_precision=DOT_PRECISION),
        b11,
        input_precision=DOT_PRECISION,
    )
    b32 = -tl.dot(
        tl.dot(b33, rA32, input_precision=DOT_PRECISION),
        b22,
        input_precision=DOT_PRECISION,
    )
    b43 = -tl.dot(
        tl.dot(b44, rA43, input_precision=DOT_PRECISION),
        b33,
        input_precision=DOT_PRECISION,
    )
    b31 = -tl.dot(
        b33,
        tl.dot(rA31, b11, input_precision=DOT_PRECISION)
        + tl.dot(rA32, b21, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b42 = -tl.dot(
        b44,
        tl.dot(rA42, b22, input_precision=DOT_PRECISION)
        + tl.dot(rA43, b32, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )
    b41 = -tl.dot(
        b44,
        tl.dot(rA41, b11, input_precision=DOT_PRECISION)
        + tl.dot(rA42, b21, input_precision=DOT_PRECISION)
        + tl.dot(rA43, b31, input_precision=DOT_PRECISION),
        input_precision=DOT_PRECISION,
    )

    h11 = b11.to(tl.bfloat16)
    h22 = b22.to(tl.bfloat16)
    h33 = b33.to(tl.bfloat16)
    h44 = b44.to(tl.bfloat16)
    h21 = b21.to(tl.bfloat16)
    h31 = b31.to(tl.bfloat16)
    h32 = b32.to(tl.bfloat16)
    h41 = b41.to(tl.bfloat16)
    h42 = b42.to(tl.bfloat16)
    h43 = b43.to(tl.bfloat16)

    # ================================================================
    # Phase 2: u = Ai @ (v * beta), w = Ai @ (k * beta * exp(g))
    # ================================================================
    beta_base = beta + bos * H + i_h
    g_base = g + bos * H + i_h

    p_b0 = tl.make_block_ptr(beta_base, (T,), (H,), (i_t * BT,), (16,), (0,))
    p_b1 = tl.make_block_ptr(beta_base, (T,), (H,), (i_t * BT + 16,), (16,), (0,))
    p_b2 = tl.make_block_ptr(beta_base, (T,), (H,), (i_t * BT + 32,), (16,), (0,))
    p_b3 = tl.make_block_ptr(beta_base, (T,), (H,), (i_t * BT + 48,), (16,), (0,))
    bb0 = tl.load(p_b0, boundary_check=(0,))
    bb1 = tl.load(p_b1, boundary_check=(0,))
    bb2 = tl.load(p_b2, boundary_check=(0,))
    bb3 = tl.load(p_b3, boundary_check=(0,))

    p_g0 = tl.make_block_ptr(g_base, (T,), (H,), (i_t * BT,), (16,), (0,))
    p_g1 = tl.make_block_ptr(g_base, (T,), (H,), (i_t * BT + 16,), (16,), (0,))
    p_g2 = tl.make_block_ptr(g_base, (T,), (H,), (i_t * BT + 32,), (16,), (0,))
    p_g3 = tl.make_block_ptr(g_base, (T,), (H,), (i_t * BT + 48,), (16,), (0,))
    eg0 = exp(tl.load(p_g0, boundary_check=(0,)))
    eg1 = exp(tl.load(p_g1, boundary_check=(0,)))
    eg2 = exp(tl.load(p_g2, boundary_check=(0,)))
    eg3 = exp(tl.load(p_g3, boundary_check=(0,)))

    v_base = v + (bos * H + i_h) * V
    if IS_VARLEN:
        u_base = u + (i_h * T_flat + bos) * V
    else:
        u_base = u + (((i_b * H + i_h) * T_flat) * V)

    for i_v in range(tl.cdiv(V, BV)):
        pv0 = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT, i_v * BV), (16, BV), (1, 0)
        )
        pv1 = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0)
        )
        pv2 = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0)
        )
        pv3 = tl.make_block_ptr(
            v_base, (T, V), (H * V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0)
        )
        vb0 = (tl.load(pv0, boundary_check=(0, 1)) * bb0[:, None]).to(tl.bfloat16)
        vb1 = (tl.load(pv1, boundary_check=(0, 1)) * bb1[:, None]).to(tl.bfloat16)
        vb2 = (tl.load(pv2, boundary_check=(0, 1)) * bb2[:, None]).to(tl.bfloat16)
        vb3 = (tl.load(pv3, boundary_check=(0, 1)) * bb3[:, None]).to(tl.bfloat16)

        u0 = tl.dot(h11, vb0, allow_tf32=False)
        u1 = tl.dot(h21, vb0, allow_tf32=False) + tl.dot(h22, vb1, allow_tf32=False)
        u2 = (
            tl.dot(h31, vb0, allow_tf32=False)
            + tl.dot(h32, vb1, allow_tf32=False)
            + tl.dot(h33, vb2, allow_tf32=False)
        )
        u3 = (
            tl.dot(h41, vb0, allow_tf32=False)
            + tl.dot(h42, vb1, allow_tf32=False)
            + tl.dot(h43, vb2, allow_tf32=False)
            + tl.dot(h44, vb3, allow_tf32=False)
        )

        pu0 = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT, i_v * BV), (16, BV), (1, 0)
        )
        pu1 = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT + 16, i_v * BV), (16, BV), (1, 0)
        )
        pu2 = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT + 32, i_v * BV), (16, BV), (1, 0)
        )
        pu3 = tl.make_block_ptr(
            u_base, (T, V), (V, 1), (i_t * BT + 48, i_v * BV), (16, BV), (1, 0)
        )
        tl.store(pu0, u0.to(pu0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu1, u1.to(pu1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu2, u2.to(pu2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pu3, u3.to(pu3.dtype.element_ty), boundary_check=(0, 1))

    k_base = k + (bos * Hg + i_h // (H // Hg)) * K
    if IS_VARLEN:
        w_base = w + (i_h * T_flat + bos) * K
    else:
        w_base = w + (((i_b * H + i_h) * T_flat) * K)

    for i_k in range(tl.cdiv(K, BK)):
        pk0 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT, i_k * BK), (16, BK), (1, 0)
        )
        pk1 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0)
        )
        pk2 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0)
        )
        pk3 = tl.make_block_ptr(
            k_base, (T, K), (Hg * K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0)
        )
        kb0 = (tl.load(pk0, boundary_check=(0, 1)) * bb0[:, None] * eg0[:, None]).to(
            tl.bfloat16
        )
        kb1 = (tl.load(pk1, boundary_check=(0, 1)) * bb1[:, None] * eg1[:, None]).to(
            tl.bfloat16
        )
        kb2 = (tl.load(pk2, boundary_check=(0, 1)) * bb2[:, None] * eg2[:, None]).to(
            tl.bfloat16
        )
        kb3 = (tl.load(pk3, boundary_check=(0, 1)) * bb3[:, None] * eg3[:, None]).to(
            tl.bfloat16
        )

        w0 = tl.dot(h11, kb0)
        w1 = tl.dot(h21, kb0) + tl.dot(h22, kb1)
        w2 = tl.dot(h31, kb0) + tl.dot(h32, kb1) + tl.dot(h33, kb2)
        w3 = tl.dot(h41, kb0) + tl.dot(h42, kb1) + tl.dot(h43, kb2) + tl.dot(h44, kb3)

        pw0 = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT, i_k * BK), (16, BK), (1, 0)
        )
        pw1 = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT + 16, i_k * BK), (16, BK), (1, 0)
        )
        pw2 = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT + 32, i_k * BK), (16, BK), (1, 0)
        )
        pw3 = tl.make_block_ptr(
            w_base, (T, K), (K, 1), (i_t * BT + 48, i_k * BK), (16, BK), (1, 0)
        )
        tl.store(pw0, w0.to(pw0.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw1, w1.to(pw1.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw2, w2.to(pw2.dtype.element_ty), boundary_check=(0, 1))
        tl.store(pw3, w3.to(pw3.dtype.element_ty), boundary_check=(0, 1))


def fused_solve_tril_recompute_w_u(
    A_raw: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused triangular solve + recompute w, u in a single kernel.

    Args:
        A_raw: [B, T, H, BT=64], strictly lower triangular
        k: [B, T, Hg, K]
        v: [B, T, H, V]
        beta: [B, T, H]
        g_cumsum: [B, T, H] FP32, cumulative gate
        cu_seqlens: [N+1]

    Returns:
        w: [B, H, T, K], head-major contiguous layout
        u: [B, H, T, V], head-major contiguous layout
    """
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A_raw.shape[-1]
    BK = 64
    BV = 64

    if cu_seqlens is not None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, BT)
        NT = len(chunk_indices)
    else:
        chunk_indices = None
        NT = triton.cdiv(T, BT)

    u_out = v.new_empty(B, H, T, V)
    w_out = k.new_empty(B, H, T, K)

    fused_solve_tril_recompute_w_u_kernel[(NT, B * H)](
        A_raw,
        k,
        v,
        beta,
        g_cumsum,
        w_out,
        u_out,
        cu_seqlens,
        chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        DOT_PRECISION=FLA_TRIL_PRECISION,
    )
    return w_out, u_out
