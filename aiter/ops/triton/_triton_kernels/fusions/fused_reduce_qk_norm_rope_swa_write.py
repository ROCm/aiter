# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused split-K reduce + per-head weighted RMSNorm + RoPE (tail) on Q,
per-row weighted RMSNorm + RoPE (tail) on KV (+ optional SWA KV write).

Grid: ``(cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)``. Each program tile
handles ``BLOCK_SIZE_M`` tokens. Programs with ``pid_h < num_local_heads``
load a query head tile ``[NUM_SPLITK, BLOCK_SIZE_M, HEAD_DIM]`` from
``q_in``, reduce over the split-K axis, apply per-head weighted batched
RMSNorm, store the (pre-RoPE) head into ``q_out``, then call the batched
RoPE on the last ``rope_head_dim`` elements. Programs with
``pid_h == num_local_heads`` load the full ``[BLOCK_SIZE_M, HEAD_DIM]``
kv tile, apply weighted batched RMSNorm over ``head_dim``, store the
normed nope part back into ``kv``, then extract the tail with the same
reshape+sum trick used for q, apply RoPE, write the result to the kv
tail, and optionally scatter both parts into ``swa_kv``.

``q_in`` layout (driven by API helper):
- 2D: ``[M, N]`` — ``q_in_splitk_stride`` = 0, ``NUM_SPLITK`` = 1.
- 3D: ``[num_splitk, M, N]`` — ``q_in_splitk_stride`` = ``q_in.stride(0)``,
  ``NUM_SPLITK`` = ``num_splitk``.
"""

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.rope.rope import _get_neox_rotated_x, _get_gptj_rotated_x


@triton.jit
def _batched_rmsnorm_op(row, weight, n_cols, epsilon):
    """Per-row RMSNorm over the last axis of a [BLOCK_M, N] tile (row in fp32)."""
    row_norm = row * row
    row_norm = tl.sum(row_norm, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
    if weight is not None:
        rms_norm = row * norm_factor[:, None] * weight[None, :]
    else:
        rms_norm = row * norm_factor[:, None]
    return rms_norm


@triton.jit
def _batched_unit_rope(
    x_pe,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    """RoPE on a [BLOCK_M, BLOCK_D_pe] tile; cos/sin are [BLOCK_M, BLOCK_D_pe]."""
    if IS_NEOX:
        x_rotated_mask = (d_pe_offs < BLOCK_D_HALF_pe)[None, :]
        x_pe_rotated = _get_neox_rotated_x(
            x_pe, x_rotated_mask, BLOCK_M, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = (d_pe_offs % 2 == 0)[None, :]
        x_pe_rotated = _get_gptj_rotated_x(
            x_pe, x_rotated_mask, BLOCK_M, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    return x_pe * cos + x_pe_rotated * sin


_fused_reduce_qk_norm_rope_swa_write_repr = make_kernel_repr(
    "_fused_reduce_qk_norm_rope_swa_write_kernel",
    [
        "BLOCK_SIZE_M",
        "HEAD_DIM",
        "ROPE_DIM",
        "NUM_LOCAL_HEADS",
        "NUM_SPLITK",
        "HAS_SWA",
        "IS_NEOX",
        "REUSE_FREQS_FRONT_PART",
    ],
)


@triton.jit(repr=_fused_reduce_qk_norm_rope_swa_write_repr)
def _fused_reduce_qk_norm_rope_swa_write_kernel(
    q_in_ptr,
    q_out_ptr,
    kv_ptr,
    q_norm_weight_ptr,
    kv_norm_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    swa_write_active_ptr,
    batch_id_per_token_ptr,
    state_slot_per_seq_ptr,
    swa_kv_ptr,
    M,
    q_in_splitk_stride,
    q_in_m_stride,
    q_in_d_stride,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kv_m,
    stride_kv_d,
    cos_stride_t,
    cos_stride_d,
    swa_kv_slot_stride,
    swa_kv_pos_stride,
    win,
    q_eps,
    kv_eps,
    BLOCK_SIZE_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NUM_LOCAL_HEADS: tl.constexpr,
    NUM_SPLITK: tl.constexpr,
    HAS_SWA: tl.constexpr,
    IS_NEOX: tl.constexpr,
    REUSE_FREQS_FRONT_PART: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM
    NUM_PE_CHUNKS: tl.constexpr = HEAD_DIM // ROPE_DIM

    m_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    m_mask = m_offs < M

    offs_d_full = tl.arange(0, HEAD_DIM)
    nope_d_mask = offs_d_full < NOPE_DIM

    d_pe_offs = tl.arange(0, ROPE_DIM).to(tl.int64)
    if REUSE_FREQS_FRONT_PART:
        if IS_NEOX:
            d_cos_offs = d_pe_offs
            d_cos_offs = tl.where(
                (d_cos_offs >= (ROPE_DIM // 2)) & (d_cos_offs < ROPE_DIM),
                d_cos_offs - (ROPE_DIM // 2),
                d_cos_offs,
            ).to(d_cos_offs.dtype)
        else:
            d_cos_offs = d_pe_offs // 2
    else:
        d_cos_offs = d_pe_offs

    if pid_h < NUM_LOCAL_HEADS:
        head_id = pid_h.to(tl.int32)
        offs_n = head_id * HEAD_DIM + offs_d_full

        splitk_offs = tl.arange(0, NUM_SPLITK).to(tl.int64)
        q_ptrs = (
            q_in_ptr
            + splitk_offs[:, None, None] * q_in_splitk_stride
            + m_offs[None, :, None] * q_in_m_stride
            + offs_n[None, None, :] * q_in_d_stride
        )
        q_tile = tl.load(
            q_ptrs,
            mask=m_mask[None, :, None],
            other=0.0,
        ).to(
            tl.float32
        )  # [NUM_SPLITK, BLOCK_SIZE_M, HEAD_DIM]
        q_acc = tl.sum(q_tile, axis=0)  # [BLOCK_SIZE_M, HEAD_DIM]

        if q_norm_weight_ptr is not None:
            w_q = tl.load(q_norm_weight_ptr + offs_d_full).to(tl.float32)
        else:
            w_q = None
        q_out_normed = _batched_rmsnorm_op(q_acc, w_q, HEAD_DIM, q_eps)

        q_base_ptrs = q_out_ptr + m_offs[:, None] * stride_qm + pid_h * stride_qh
        tl.store(
            q_base_ptrs + offs_d_full[None, :] * stride_qd,
            q_out_normed.to(q_out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & nope_d_mask[None, :],
        )

        # Slice the trailing ROPE_DIM elements: only the last chunk is nonzero.
        q_pe = tl.where(
            (offs_d_full >= NOPE_DIM)[None, :], q_out_normed, 0.0
        )  # [BLOCK_SIZE_M, HEAD_DIM]
        q_pe = q_pe.reshape(BLOCK_SIZE_M, NUM_PE_CHUNKS, ROPE_DIM)
        q_pe = tl.sum(q_pe, axis=1)  # [BLOCK_SIZE_M, ROPE_DIM]

        pos = tl.load(positions_ptr + m_offs, mask=m_mask, other=0)  # [BLOCK_SIZE_M]
        cos_offs = pos[:, None] * cos_stride_t + d_cos_offs[None, :] * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs, mask=m_mask[:, None], other=0)
        sin = tl.load(sin_ptr + cos_offs, mask=m_mask[:, None], other=0)

        q_pe_ptrs = q_base_ptrs + (NOPE_DIM + d_pe_offs[None, :]) * stride_qd
        q_pe = _batched_unit_rope(
            q_pe,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            BLOCK_SIZE_M,
            ROPE_DIM,
            ROPE_DIM // 2,
        )
        tl.store(
            q_pe_ptrs,
            q_pe.to(q_out_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )
        return

    if HAS_SWA:
        src_id = tl.load(
            swa_write_active_ptr + m_offs, mask=m_mask, other=-1
        )  # [BLOCK_SIZE_M]
    else:
        src_id = m_offs.to(tl.int32)
    src_mask = m_mask & (src_id >= 0)

    pos = tl.load(positions_ptr + src_id, mask=src_mask, other=0)
    cos_offs = pos[:, None] * cos_stride_t + d_cos_offs[None, :] * cos_stride_d
    cos = tl.load(cos_ptr + cos_offs, mask=src_mask[:, None], other=0)
    sin = tl.load(sin_ptr + cos_offs, mask=src_mask[:, None], other=0)

    kv_base_ptrs = kv_ptr + src_id[:, None].to(tl.int64) * stride_kv_m
    kv_full_ptrs = kv_base_ptrs + offs_d_full[None, :] * stride_kv_d
    kv_pe_ptrs = kv_base_ptrs + (NOPE_DIM + d_pe_offs[None, :]) * stride_kv_d

    # Load the entire kv row (nope + pe) so we can RMSNorm over head_dim.
    kv_full = tl.load(kv_full_ptrs, mask=src_mask[:, None], other=0.0).to(tl.float32)

    if kv_norm_weight_ptr is not None:
        w_kv = tl.load(kv_norm_weight_ptr + offs_d_full).to(tl.float32)
    else:
        w_kv = None
    kv_normed = _batched_rmsnorm_op(
        kv_full, w_kv, HEAD_DIM, kv_eps
    )  # [BLOCK_SIZE_M, HEAD_DIM]

    # Store the normed nope portion back into kv.
    tl.store(
        kv_full_ptrs,
        kv_normed.to(kv_ptr.dtype.element_ty),
        mask=src_mask[:, None] & nope_d_mask[None, :],
    )

    # Extract pe via the same reshape+sum trick used for q.
    kv_pe = tl.where(
        (offs_d_full >= NOPE_DIM)[None, :], kv_normed, 0.0
    )  # [BLOCK_SIZE_M, HEAD_DIM]
    kv_pe = kv_pe.reshape(BLOCK_SIZE_M, NUM_PE_CHUNKS, ROPE_DIM)
    kv_pe = tl.sum(kv_pe, axis=1)  # [BLOCK_SIZE_M, ROPE_DIM]

    kv_pe = _batched_unit_rope(
        kv_pe,
        cos,
        sin,
        d_pe_offs,
        IS_NEOX,
        BLOCK_SIZE_M,
        ROPE_DIM,
        ROPE_DIM // 2,
    )
    tl.store(
        kv_pe_ptrs,
        kv_pe.to(kv_ptr.dtype.element_ty),
        mask=src_mask[:, None],
    )

    if HAS_SWA:
        bid = tl.load(batch_id_per_token_ptr + src_id, mask=src_mask, other=0)
        slot = tl.load(state_slot_per_seq_ptr + bid, mask=src_mask, other=0)
        ring_idx = pos % win
        swa_kv_ptrs = (
            swa_kv_ptr
            + slot[:, None].to(tl.int64) * swa_kv_slot_stride
            + ring_idx[:, None].to(tl.int64) * swa_kv_pos_stride
        )
        tl.store(
            swa_kv_ptrs + offs_d_full[None, :],
            kv_normed.to(swa_kv_ptr.dtype.element_ty),
            mask=src_mask[:, None] & nope_d_mask[None, :],
        )
        tl.store(
            swa_kv_ptrs + NOPE_DIM + d_pe_offs[None, :],
            kv_pe.to(swa_kv_ptr.dtype.element_ty),
            mask=src_mask[:, None],
        )
