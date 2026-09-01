# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# DSV4 sparse-MLA training op: forward (with LSE) + backward (non-atomic).
# Optimised for MI308X (gfx942, 64 KB LDS).

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.sparse_mla_dsv4_train import (
    _bwd_dkv_gather_kernel,
    _bwd_dkv_interm_kernel,
    _bwd_dq_store_dp_kernel,
    _sparse_mla_fwd_kernel,
)


def _get_lds_limit():
    try:
        prop = torch.cuda.get_device_properties(torch.cuda.current_device())
        gcn_arch = getattr(prop, "gcnArchName", "")
        if "gfx950" in gcn_arch:
            return 163840
    except (OSError, RuntimeError, AttributeError):
        return 65536
    return 65536


def _select_bwd_tiles():
    lds = _get_lds_limit()
    if lds <= 65536:
        return 16, 32, 4
    return 32, 32, 4


# =====================================================================
# Host-side CSR inverted-topk builder
# =====================================================================


def _build_inverted_topk(indices, num_kv):
    """Build CSR index mapping KV tokens back to (token, rank) pairs.

    Args:
        indices: [N, topk] int32, -1 = invalid
        num_kv:  number of KV tokens

    Returns:
        inv_ptr:  [num_kv + 1] int32, CSR row pointers
        inv_data: [nnz] int32, flat positions (token * topk + rank)
    """
    N, topk = indices.shape
    flat = indices.reshape(-1).long()
    valid = flat >= 0

    positions = torch.arange(N * topk, device=indices.device, dtype=torch.int64)
    valid_pos = positions[valid]
    valid_kv = flat[valid]

    sort_idx = torch.argsort(valid_kv, stable=True)
    inv_data = valid_pos[sort_idx].to(torch.int32)
    sorted_kv = valid_kv[sort_idx]

    counts = torch.zeros(num_kv + 1, device=indices.device, dtype=torch.int32)
    if sorted_kv.numel() > 0:
        counts[1:].scatter_add_(
            0, sorted_kv.int(), torch.ones_like(sorted_kv, dtype=torch.int32)
        )
    inv_ptr = counts.cumsum(0).to(torch.int32)

    return inv_ptr, inv_data


# =====================================================================
# Forward wrapper
# =====================================================================


def sparse_mla_fwd(q, kv, attn_sink, indices, scale=None):
    N, H, D = q.shape
    N_kv = kv.shape[0]
    topk = indices.shape[1]

    if scale is None:
        scale = 1.0 / (D**0.5)

    BLOCK_D = triton.next_power_of_2(D)

    has_sink = attn_sink is not None
    if not has_sink:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)

    out = torch.empty(N, H, D, device=q.device, dtype=q.dtype)
    lse = torch.empty(N, H, device=q.device, dtype=torch.float32)

    grid = lambda META: (N, triton.cdiv(H, META["BLOCK_H"]))

    _sparse_mla_fwd_kernel[grid](
        q,
        kv,
        indices,
        attn_sink,
        out,
        lse,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        out.stride(0),
        out.stride(1),
        lse.stride(0),
        indices.stride(0),
        H,
        D,
        N_kv,
        topk,
        scale,
        HAS_ATTN_SINK=has_sink,
        BLOCK_D=BLOCK_D,
    )

    return out, lse


# =====================================================================
# Backward wrapper
# =====================================================================


def sparse_mla_bwd(q, kv, o, do, indices, lse, attn_sink, scale=None):
    N, H, D = q.shape
    N_kv = kv.shape[0]
    topk = indices.shape[1]

    if scale is None:
        scale = 1.0 / (D**0.5)

    BLOCK_H, BLOCK_K, num_warps = _select_bwd_tiles()
    BLOCK_D = triton.next_power_of_2(D)

    has_sink = attn_sink is not None
    if not has_sink:
        attn_sink = torch.empty(1, device=q.device, dtype=torch.float32)

    num_hg = triton.cdiv(H, BLOCK_H)

    # Pre-compute delta = sum_d(O * dO) — avoids loading O in the dQ kernel
    delta = (o.float() * do.float()).sum(dim=-1)  # [N, H] fp32

    dq = torch.zeros(N, H, D, device=q.device, dtype=torch.float32)
    d_sink = torch.zeros(H, device=q.device, dtype=torch.float32) if has_sink else None
    d_sink_buf = (
        d_sink if has_sink else torch.empty(1, device=q.device, dtype=torch.float32)
    )

    # P/dP buffers for kernel 2  [N, H, topk] bf16
    p_buf = torch.empty(N, H, topk, device=q.device, dtype=torch.bfloat16)
    dp_buf = torch.empty(N, H, topk, device=q.device, dtype=torch.bfloat16)

    # Kernel 1: dQ + store P/dP — grid (N, num_hg)
    _bwd_dq_store_dp_kernel[(N, num_hg)](
        q,
        kv,
        do,
        indices,
        lse,
        delta,
        attn_sink,
        dq,
        d_sink_buf,
        p_buf,
        dp_buf,
        q.stride(0),
        q.stride(1),
        kv.stride(0),
        do.stride(0),
        do.stride(1),
        indices.stride(0),
        lse.stride(0),
        delta.stride(0),
        dq.stride(0),
        dq.stride(1),
        p_buf.stride(0),
        p_buf.stride(1),
        H,
        D,
        N_kv,
        topk,
        scale,
        HAS_ATTN_SINK=has_sink,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
    )

    # Kernel 2: dKV intermediate — grid (N,)
    interm = torch.empty(N, topk, D, device=q.device, dtype=torch.float32)
    _bwd_dkv_interm_kernel[(N,)](
        q,
        do,
        p_buf,
        dp_buf,
        interm,
        q.stride(0),
        q.stride(1),
        do.stride(0),
        do.stride(1),
        p_buf.stride(0),
        p_buf.stride(1),
        interm.stride(0),
        interm.stride(1),
        H,
        D,
        topk,
        BLOCK_H=BLOCK_H,
        BLOCK_D=BLOCK_D,
        BLOCK_K=BLOCK_K,
        NUM_HG=num_hg,
        num_warps=num_warps,
    )

    # CSR gather into dkv
    inv_ptr, inv_data = _build_inverted_topk(indices, N_kv)
    dkv = torch.zeros(N_kv, D, device=q.device, dtype=torch.float32)

    BLOCK_G = 64
    _bwd_dkv_gather_kernel[(N_kv,)](
        interm,
        inv_ptr,
        inv_data,
        dkv,
        interm.stride(0),
        interm.stride(1),
        dkv.stride(0),
        D,
        topk,
        N,
        BLOCK_D=BLOCK_D,
        BLOCK_G=BLOCK_G,
        num_warps=num_warps,
    )

    return dq, dkv, d_sink


# =====================================================================
# Autograd Function
# =====================================================================


class SparseMLADSV4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, attn_sink, indices, scale):
        out, lse = sparse_mla_fwd(q, kv, attn_sink, indices, scale)
        ctx.save_for_backward(q, kv, out, indices, lse, attn_sink)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv, o, indices, lse, attn_sink = ctx.saved_tensors
        has_sink = attn_sink is not None and attn_sink.numel() > 1

        dq, dkv, d_sink = sparse_mla_bwd(
            q,
            kv,
            o,
            do.contiguous(),
            indices,
            lse,
            attn_sink if has_sink else None,
            ctx.scale,
        )
        return dq.to(q.dtype), dkv.to(kv.dtype), d_sink, None, None


def sparse_mla_dsv4_train(q, kv, attn_sink, indices, scale=None):
    if scale is None:
        scale = 1.0 / (q.shape[-1] ** 0.5)
    return SparseMLADSV4Function.apply(q, kv, attn_sink, indices, scale)
