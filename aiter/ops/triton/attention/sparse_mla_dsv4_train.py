# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# DSV4 sparse-MLA training op: forward (with LSE) + backward (chunked, non-atomic).
# Optimised for MI308X (gfx942, 64 KB LDS).

import torch
import triton

from aiter.ops.triton._triton_kernels.attention.sparse_mla_dsv4_train import (
    _FWD_FALLBACK,
    _bwd_dkv_interm_kernel,
    _bwd_dq_kernel,
    _sparse_mla_fwd_kernel,
)
from aiter.ops.triton.attention.sparse_attention_dsv4_bwd import (
    build_inverted_topk,
    delta_v4,
    dkv_gather_acc,
)
from aiter.ops.triton.utils.tuned_config_utils import get_tuned_kernel_config

_BWD_FALLBACK = triton.Config({"BLOCK_H": 16, "BLOCK_K": 16}, num_warps=4, num_stages=1)

# The backward splits the top-k axis into chunks of at most this many slots. This
# limits the dKV-interm workspace to [N, chunk, D].
_MAX_R_CHUNK = 512


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

    fwd_cfg = get_tuned_kernel_config(
        "attention", "SPARSE_MLA_DSV4_TRAIN", "_sparse_mla_fwd_kernel", _FWD_FALLBACK
    )
    BLOCK_H_fwd = fwd_cfg.kwargs["BLOCK_H"]
    BLOCK_K_fwd = fwd_cfg.kwargs["BLOCK_K"]
    grid = (N, triton.cdiv(H, BLOCK_H_fwd))

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
        BLOCK_H=BLOCK_H_fwd,
        BLOCK_K=BLOCK_K_fwd,
        num_warps=fwd_cfg.num_warps,
        num_stages=fwd_cfg.num_stages,
    )

    return out, lse


# =====================================================================
# Backward wrapper
# =====================================================================


def sparse_mla_bwd(q, kv, o, do, indices, lse, attn_sink, scale=None):
    """Backward of the DSV4 sparse-MLA training attention (K == V == kv).

    The op splits the top-k axis into chunks of at most ``_MAX_R_CHUNK`` slots.
    For each chunk, one kernel computes dQ and stores dS and P. A second kernel
    computes interm[t, k, :] = sum_h (dS * Q + P * dO), and a CSR gather adds
    interm into dKV. dQ, interm and dKV are fp32.

    Args:
        q: [N, H, D] query.
        kv: [N_kv, D] key and value.
        o: [N, H, D] forward output.
        do: [N, H, D] gradient of o.
        indices: [N, topk] int32 rows of kv. A value out of [0, N_kv) is masked.
        lse: [N, H] fp32 forward log-sum-exp, with the sink included.
        attn_sink: [H] fp32 sink logits, or None.
        scale: softmax scale. The default is 1 / sqrt(D).

    Returns:
        dq [N, H, D] fp32, dkv [N_kv, D] fp32, and d_sink [H] fp32 (None without
        a sink).

    D must be a power of two.
    """
    N, H, D = q.shape
    N_kv = kv.shape[0]
    topk = indices.shape[1]
    assert D & (D - 1) == 0, f"head_dim must be a power of two, got {D}"

    if scale is None:
        scale = 1.0 / (D**0.5)

    dq_cfg = get_tuned_kernel_config(
        "attention", "SPARSE_MLA_DSV4_TRAIN", "_bwd_dq_kernel", _BWD_FALLBACK
    )
    interm_cfg = get_tuned_kernel_config(
        "attention", "SPARSE_MLA_DSV4_TRAIN", "_bwd_dkv_interm_kernel", _BWD_FALLBACK
    )
    # tl.dot needs 16 rows. A head tile larger than the head count only adds masked rows.
    max_block_h = max(16, triton.next_power_of_2(H))
    dq_kwargs = dict(dq_cfg.kwargs)
    dq_kwargs["BLOCK_H"] = min(dq_kwargs["BLOCK_H"], max_block_h)
    interm_kwargs = dict(interm_cfg.kwargs)
    interm_kwargs["BLOCK_H"] = min(interm_kwargs["BLOCK_H"], max_block_h)

    num_chunks = triton.cdiv(topk, _MAX_R_CHUNK)
    r_chunk = triton.cdiv(topk, num_chunks)
    padded_topk = num_chunks * r_chunk
    if padded_topk != topk:
        # A CSR entry addresses interm by its slot in the chunk, so all chunks have one width.
        indices = torch.nn.functional.pad(indices, (0, padded_topk - topk), value=-1)

    delta = delta_v4(o.contiguous(), do.contiguous())
    q_t = q.transpose(1, 2).contiguous()
    do_t = do.transpose(1, 2).contiguous()

    dq = torch.empty(N, H, D, device=q.device, dtype=torch.float32)
    dkv = torch.empty(N_kv, D, device=q.device, dtype=torch.float32)
    chunk_ds = torch.empty(N, H, r_chunk, device=q.device, dtype=torch.bfloat16)
    chunk_p = torch.empty(N, H, r_chunk, device=q.device, dtype=torch.bfloat16)
    interm = torch.empty(N, r_chunk, D, device=q.device, dtype=torch.float32)

    for r_start in range(0, padded_topk, r_chunk):
        _bwd_dq_kernel[(N, triton.cdiv(H, dq_kwargs["BLOCK_H"]))](
            q,
            kv,
            do,
            indices,
            lse,
            delta,
            dq,
            chunk_ds,
            chunk_p,
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
            chunk_ds.stride(0),
            chunk_ds.stride(1),
            H,
            N_kv,
            r_start,
            scale,
            R_CHUNK=r_chunk,
            IS_FIRST_CHUNK=(r_start == 0),
            BLOCK_D=D,
            **dq_kwargs,
            num_warps=dq_cfg.num_warps,
            num_stages=dq_cfg.num_stages,
        )
        _bwd_dkv_interm_kernel[(N,)](
            q_t,
            do_t,
            chunk_ds,
            chunk_p,
            interm,
            q_t.stride(0),
            q_t.stride(1),
            do_t.stride(0),
            do_t.stride(1),
            chunk_ds.stride(0),
            chunk_ds.stride(1),
            interm.stride(0),
            interm.stride(1),
            H,
            R_CHUNK=r_chunk,
            BLOCK_D=D,
            NUM_HG=triton.cdiv(H, interm_kwargs["BLOCK_H"]),
            **interm_kwargs,
            num_warps=interm_cfg.num_warps,
            num_stages=interm_cfg.num_stages,
        )
        inv_ptr, inv_data = build_inverted_topk(
            indices[:, r_start : r_start + r_chunk], N_kv
        )
        # The first chunk writes every row of dkv, so dkv needs no zero fill.
        dkv_gather_acc(interm, inv_ptr, inv_data, dkv, accumulate=(r_start > 0))

    d_sink = None
    if attn_sink is not None:
        d_sink = -(torch.exp(attn_sink[None, :].float() - lse) * delta).sum(dim=0)

    return dq, dkv, d_sink


# =====================================================================
# Autograd Function
# =====================================================================


class SparseMLADSV4Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kv, attn_sink, indices, scale):
        out, lse = sparse_mla_fwd(q, kv, attn_sink, indices, scale)
        if attn_sink is None:
            attn_sink = torch.empty(0, device=q.device, dtype=torch.float32)
        ctx.save_for_backward(q, kv, out, indices, lse, attn_sink)
        ctx.scale = scale
        return out

    @staticmethod
    def backward(ctx, do):
        q, kv, o, indices, lse, attn_sink = ctx.saved_tensors
        has_sink = attn_sink.numel() > 0

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
