# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# DSV4 sparse-MLA training kernels (forward with LSE + backward).
# Optimised for MI308X (gfx942, 64 KB LDS).
#
# Forward: online-softmax sparse attention with autotuning.
# Backward: 3-kernel pipeline (dQ+storeP/dP → dKV-interm → CSR gather).

import triton
import triton.language as tl

# =====================================================================
# Forward — autotune configs
# =====================================================================


def _fwd_configs():
    configs = []
    for BLOCK_H in [16, 32, 64]:
        for BLOCK_K in [16, 32]:
            for num_stages in [1, 2, 3, 4]:
                configs.append(
                    triton.Config(
                        {"BLOCK_H": BLOCK_H, "BLOCK_K": BLOCK_K},
                        num_warps=4,
                        num_stages=num_stages,
                    )
                )
    return configs


def _fwd_prune(configs, named_args, **kwargs):
    D = named_args["head_dim"]
    BLOCK_D = kwargs.get("BLOCK_D", D)
    BLOCK_D = max(BLOCK_D, D)
    pruned = []
    for c in configs:
        bk = c.kwargs["BLOCK_K"]
        ns = c.num_stages
        lds = BLOCK_D * bk * 2 * ns
        if lds <= 65536:
            pruned.append(c)
    return pruned


# =====================================================================
# Forward kernel — sparse MLA with LSE output
# =====================================================================


@triton.autotune(
    configs=_fwd_configs(),
    key=["num_heads", "topk", "head_dim"],
    prune_configs_by={"early_config_prune": _fwd_prune},
)
@triton.jit
def _sparse_mla_fwd_kernel(
    q_ptr,
    kv_ptr,
    indices_ptr,
    attn_sink_ptr,
    out_ptr,
    lse_ptr,
    q_stride_t: tl.int64,
    q_stride_h: tl.int64,
    kv_stride_n: tl.int64,
    out_stride_t: tl.int64,
    out_stride_h: tl.int64,
    lse_stride_t: tl.int64,
    idx_stride_t: tl.int64,
    num_heads: tl.int32,
    head_dim: tl.int32,
    num_kv: tl.int32,
    topk: tl.int32,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    head_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    dim_offsets = tl.arange(0, BLOCK_D)
    head_mask = head_offsets < num_heads
    dim_mask = dim_offsets < head_dim

    q = tl.load(
        q_ptr
        + query_idx * q_stride_t
        + head_offsets[:, None] * q_stride_h
        + dim_offsets[None, :],
        mask=head_mask[:, None] & dim_mask[None, :],
        other=0.0,
    )

    m_i = tl.full((BLOCK_H,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    k_offsets = tl.arange(0, BLOCK_K)

    for k_start in tl.range(0, topk, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < topk
        slot = tl.load(
            indices_ptr + query_idx * idx_stride_t + k_pos, mask=in_range, other=-1
        )
        valid = in_range & (slot >= 0) & (slot < num_kv)

        kv = tl.load(
            kv_ptr + slot[:, None] * kv_stride_n + dim_offsets[None, :],
            mask=valid[:, None] & dim_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv)) * scale
        scores = tl.where(head_mask[:, None] & valid[None, :], scores, float("-inf"))

        m_block = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_block)
        alpha = tl.where(m_new == float("-inf"), 0.0, tl.exp(m_i - m_new))
        p = tl.where(
            m_new[:, None] == float("-inf"), 0.0, tl.exp(scores - m_new[:, None])
        )
        p = tl.where(head_mask[:, None] & valid[None, :], p, 0.0)
        l_new = l_i * alpha + tl.sum(p, axis=1)

        acc = acc * alpha[:, None] + tl.dot(p.to(kv.dtype), kv)
        m_i = m_new
        l_i = l_new

    if HAS_ATTN_SINK:
        sink = tl.load(
            attn_sink_ptr + head_offsets, mask=head_mask, other=float("-inf")
        ).to(tl.float32)
        m_final = tl.maximum(m_i, sink)
        alpha = tl.where(m_final == float("-inf"), 0.0, tl.exp(m_i - m_final))
        exp_sink = tl.where(sink == float("-inf"), 0.0, tl.exp(sink - m_final))
        l_final = l_i * alpha + exp_sink
        denom = tl.maximum(l_final, 1e-30)
        out = tl.where(
            l_final[:, None] > 0.0, (acc * alpha[:, None]) / denom[:, None], 0.0
        )
        lse = tl.where(l_final > 0.0, tl.log(l_final) + m_final, float("-inf"))
    else:
        denom = tl.maximum(l_i, 1e-30)
        out = tl.where(l_i[:, None] > 0.0, acc / denom[:, None], 0.0)
        lse = tl.where(l_i > 0.0, tl.log(l_i) + m_i, float("-inf"))

    tl.store(
        out_ptr
        + query_idx * out_stride_t
        + head_offsets[:, None] * out_stride_h
        + dim_offsets[None, :],
        out,
        mask=head_mask[:, None] & dim_mask[None, :],
    )
    tl.store(lse_ptr + query_idx * lse_stride_t + head_offsets, lse, mask=head_mask)


# =====================================================================
# Backward — Kernel 1: dQ + store P/dP
# =====================================================================
# Grid: (N, num_hg). One CTA per (token, head-group).
# Reads pre-computed delta. Stores P and dP to buffers for kernel 2.


@triton.jit
def _bwd_dq_store_dp_kernel(
    q_ptr,
    kv_ptr,
    do_ptr,
    indices_ptr,
    lse_ptr,
    delta_ptr,
    attn_sink_ptr,
    dq_ptr,
    d_sink_ptr,
    p_buf_ptr,
    dp_buf_ptr,
    q_stride_t: tl.int64,
    q_stride_h: tl.int64,
    kv_stride_n: tl.int64,
    do_stride_t: tl.int64,
    do_stride_h: tl.int64,
    idx_stride_t: tl.int64,
    lse_stride_t: tl.int64,
    delta_stride_t: tl.int64,
    dq_stride_t: tl.int64,
    dq_stride_h: tl.int64,
    p_stride_t: tl.int64,
    p_stride_h: tl.int64,
    num_heads: tl.int32,
    head_dim: tl.int32,
    num_kv: tl.int32,
    topk: tl.int32,
    scale,
    HAS_ATTN_SINK: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    tok_idx = tl.program_id(0)
    pid_h = tl.program_id(1)

    h_offsets = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    d_offsets = tl.arange(0, BLOCK_D)
    k_offsets = tl.arange(0, BLOCK_K)
    h_mask = h_offsets < num_heads
    d_mask = d_offsets < head_dim

    scale_log2e = scale * 1.44269504

    q = tl.load(
        q_ptr
        + tok_idx * q_stride_t
        + h_offsets[:, None] * q_stride_h
        + d_offsets[None, :],
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    do = tl.load(
        do_ptr
        + tok_idx * do_stride_t
        + h_offsets[:, None] * do_stride_h
        + d_offsets[None, :],
        mask=h_mask[:, None] & d_mask[None, :],
        other=0.0,
    )
    delta = tl.load(
        delta_ptr + tok_idx * delta_stride_t + h_offsets, mask=h_mask, other=0.0
    )
    lse = tl.load(lse_ptr + tok_idx * lse_stride_t + h_offsets, mask=h_mask, other=0.0)
    lse_log2 = lse * 1.44269504

    acc_dq = tl.zeros((BLOCK_H, BLOCK_D), dtype=tl.float32)

    for k_start in tl.range(0, topk, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < topk
        slot = tl.load(
            indices_ptr + tok_idx * idx_stride_t + k_pos, mask=in_range, other=-1
        )
        valid = in_range & (slot >= 0) & (slot < num_kv)

        kv = tl.load(
            kv_ptr + slot[:, None] * kv_stride_n + d_offsets[None, :],
            mask=valid[:, None] & d_mask[None, :],
            other=0.0,
        )

        scores = tl.dot(q, tl.trans(kv))
        p = tl.exp2(scores * scale_log2e - lse_log2[:, None])
        p = tl.where(h_mask[:, None] & valid[None, :], p, 0.0)

        dp_val = tl.dot(do, tl.trans(kv))
        dp = p * (dp_val - delta[:, None]) * scale
        dp = tl.where(h_mask[:, None] & valid[None, :], dp, 0.0)

        acc_dq += tl.dot(dp.to(kv.dtype), kv)

        # Store P and dP for kernel 2
        p_base = tok_idx * p_stride_t + h_offsets[:, None] * p_stride_h + k_pos[None, :]
        p_mask = h_mask[:, None] & in_range[None, :]
        tl.store(p_buf_ptr + p_base, p.to(tl.bfloat16), mask=p_mask)
        tl.store(dp_buf_ptr + p_base, dp.to(tl.bfloat16), mask=p_mask)

    tl.store(
        dq_ptr
        + tok_idx * dq_stride_t
        + h_offsets[:, None] * dq_stride_h
        + d_offsets[None, :],
        acc_dq,
        mask=h_mask[:, None] & d_mask[None, :],
    )

    if HAS_ATTN_SINK:
        sink = tl.load(attn_sink_ptr + h_offsets, mask=h_mask, other=0.0).to(tl.float32)
        d_sink_val = -delta * tl.exp(sink - lse)
        d_sink_val = tl.where(h_mask, d_sink_val, 0.0)
        tl.atomic_add(d_sink_ptr + h_offsets, d_sink_val, mask=h_mask)


# =====================================================================
# Backward — Kernel 2: dKV intermediate (no atomics)
# =====================================================================
# Grid: (N,). One CTA per token, loops over head-groups.
# Reads P_buf, dP_buf, Q, dO → computes interm[tok, k, :D].


@triton.jit
def _bwd_dkv_interm_kernel(
    q_ptr,
    do_ptr,
    p_buf_ptr,
    dp_buf_ptr,
    interm_ptr,
    q_stride_t: tl.int64,
    q_stride_h: tl.int64,
    do_stride_t: tl.int64,
    do_stride_h: tl.int64,
    p_stride_t: tl.int64,
    p_stride_h: tl.int64,
    interm_stride_t: tl.int64,
    interm_stride_k: tl.int64,
    num_heads: tl.int32,
    head_dim: tl.int32,
    topk: tl.int32,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_HG: tl.constexpr,
):
    tok_idx = tl.program_id(0)

    d_offsets = tl.arange(0, BLOCK_D)
    k_offsets = tl.arange(0, BLOCK_K)
    d_mask = d_offsets < head_dim

    for k_start in tl.range(0, topk, BLOCK_K):
        k_pos = k_start + k_offsets
        in_range = k_pos < topk

        acc_interm = tl.zeros((BLOCK_K, BLOCK_D), dtype=tl.float32)

        for hg in tl.range(0, NUM_HG):
            h_offsets = hg * BLOCK_H + tl.arange(0, BLOCK_H)
            h_mask = h_offsets < num_heads

            q = tl.load(
                q_ptr
                + tok_idx * q_stride_t
                + h_offsets[:, None] * q_stride_h
                + d_offsets[None, :],
                mask=h_mask[:, None] & d_mask[None, :],
                other=0.0,
            )
            do = tl.load(
                do_ptr
                + tok_idx * do_stride_t
                + h_offsets[:, None] * do_stride_h
                + d_offsets[None, :],
                mask=h_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            p_base = (
                tok_idx * p_stride_t + h_offsets[:, None] * p_stride_h + k_pos[None, :]
            )
            p_mask = h_mask[:, None] & in_range[None, :]
            p = tl.load(p_buf_ptr + p_base, mask=p_mask, other=0.0)
            dp = tl.load(dp_buf_ptr + p_base, mask=p_mask, other=0.0)

            # interm += dP^T @ Q + P^T @ dO  [BLOCK_K, BLOCK_D]
            acc_interm += tl.dot(tl.trans(dp), q) + tl.dot(tl.trans(p), do)

        interm_base = (
            tok_idx * interm_stride_t
            + k_pos[:, None] * interm_stride_k
            + d_offsets[None, :]
        )
        interm_mask = in_range[:, None] & d_mask[None, :]
        tl.store(interm_ptr + interm_base, acc_interm.to(tl.float32), mask=interm_mask)


# =====================================================================
# Backward — dKV gather kernel (CSR-based, no atomics)
# =====================================================================


@triton.jit
def _bwd_dkv_gather_kernel(
    interm_ptr,
    inv_ptr_ptr,
    inv_data_ptr,
    dkv_ptr,
    interm_stride_t: tl.int64,
    interm_stride_k: tl.int64,
    dkv_stride_n: tl.int64,
    head_dim: tl.int32,
    topk: tl.int32,
    num_tokens: tl.int32,
    BLOCK_D: tl.constexpr,
    BLOCK_G: tl.constexpr,
):
    kv_idx = tl.program_id(0)
    d_offsets = tl.arange(0, BLOCK_D)
    d_mask = d_offsets < head_dim

    start = tl.load(inv_ptr_ptr + kv_idx)
    end = tl.load(inv_ptr_ptr + kv_idx + 1)

    acc = tl.zeros((BLOCK_D,), dtype=tl.float32)

    for g_start in tl.range(0, end - start, BLOCK_G):
        g_offsets = tl.arange(0, BLOCK_G)
        g_pos = start + g_start + g_offsets
        g_mask = g_pos < end

        flat_pos = tl.load(inv_data_ptr + g_pos, mask=g_mask, other=0)
        byte_offsets = (flat_pos // topk) * interm_stride_t + (
            flat_pos % topk
        ) * interm_stride_k

        rows = tl.load(
            interm_ptr + byte_offsets[:, None] + d_offsets[None, :],
            mask=g_mask[:, None] & d_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += tl.sum(rows, axis=0)

    tl.store(dkv_ptr + kv_idx * dkv_stride_n + d_offsets, acc, mask=d_mask)
