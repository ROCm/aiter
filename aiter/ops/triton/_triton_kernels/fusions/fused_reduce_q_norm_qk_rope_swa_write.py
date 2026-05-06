# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused split-K reduce + per-head weighted RMSNorm + RoPE (tail) + KV norm/RoPE
(+ optional SWA KV write).

Grid: ``(M, num_local_heads + 1)``. Programs with ``pid_h < num_local_heads``
load one query head row from ``q_in`` as a ``[NUM_SPLITK, HEAD_DIM]`` tile,
reduce over the split-K axis, apply per-head weighted ``_rmsmorm_op``, store
the (pre-RoPE) head into ``q_out``, then call ``_unit_rope`` to rotate the
last ``rope_head_dim`` elements in-place. Programs with
``pid_h == num_local_heads`` apply RoPE on the ``kv`` tail (also via
``_unit_rope``) and optionally scatter into ``swa_kv`` (same semantics as
``state_writes._swa_write_kernel``).

``q_in`` layout (driven by API helper):
- 2D: ``[M, N]`` — ``q_in_splitk_stride`` = 0, ``NUM_SPLITK`` = 1.
- 3D: ``[num_splitk, M, N]`` — ``q_in_splitk_stride`` = ``q_in.stride(0)``,
  ``NUM_SPLITK`` = ``num_splitk``.
"""

import triton
import triton.language as tl

from aiter.ops.triton._triton_kernels.fusions.fused_kv_cache import _unit_rope
from aiter.ops.triton._triton_kernels.quant.fused_mxfp4_quant import _rmsmorm_op
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.rope.rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D


@triton.jit
def _unit_rope(
    x_pe,
    cos,
    sin,
    d_pe_offs,
    IS_NEOX: tl.constexpr,
    BLOCK_D_pe: tl.constexpr,
    BLOCK_D_HALF_pe: tl.constexpr,
):
    if IS_NEOX:
        x_rotated_mask = d_pe_offs < BLOCK_D_HALF_pe
        x_pe_rotated = _get_neox_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )
    else:
        x_rotated_mask = d_pe_offs % 2 == 0
        x_pe_rotated = _get_gptj_rotated_x_1D(
            x_pe, x_rotated_mask, BLOCK_D_pe, BLOCK_D_HALF_pe
        )

    x_pe = x_pe * cos + x_pe_rotated * sin

    return x_pe


_fused_reduce_q_norm_qk_rope_swa_write_repr = make_kernel_repr(
    "_fused_reduce_q_norm_qk_rope_swa_write_kernel",
    [
        "HEAD_DIM",
        "ROPE_DIM",
        "NUM_LOCAL_HEADS",
        "NUM_SPLITK",
        "HAS_SWA",
        "IS_NEOX",
        "REUSE_FREQS_FRONT_PART",
    ],
)


@triton.jit(repr=_fused_reduce_q_norm_qk_rope_swa_write_repr)
def _fused_reduce_q_norm_qk_rope_swa_write_kernel(
    q_in_ptr,
    q_out_ptr,
    kv_ptr,
    q_norm_weight_ptr,
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

    offs_d_full = tl.arange(0, HEAD_DIM)
    mask_d = offs_d_full < HEAD_DIM

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
            + splitk_offs[:, None] * q_in_splitk_stride
            + pid_m * q_in_m_stride
            + offs_n[None, :] * q_in_d_stride
        )
        q_tile = tl.load(
            q_ptrs,
            mask=mask_d[None, :],
            other=0.0,
        ).to(
            tl.float32
        )  # [NUM_SPLITK, HEAD_DIM]
        q_acc = tl.sum(q_tile, axis=0)  # [HEAD_DIM]

        if q_norm_weight_ptr is not None:
            w_q = tl.load(
                q_norm_weight_ptr + offs_d_full,
                mask=mask_d,
                other=0.0,
            ).to(tl.float32)
        else:
            w_q = None
        q_out_normed = _rmsmorm_op(q_acc, w_q, HEAD_DIM, q_eps)

        q_base_ptrs = q_out_ptr + pid_m * stride_qm + pid_h * stride_qh
        tl.store(
            q_base_ptrs + offs_d_full * stride_qd,
            q_out_normed.to(q_out_ptr.dtype.element_ty),
            mask=offs_d_full < NOPE_DIM,
        )

        q_pe = tl.where(offs_d_full >= NOPE_DIM, q_out_normed, 0.0)
        q_pe = q_pe.reshape((HEAD_DIM // ROPE_DIM), ROPE_DIM)
        q_pe = tl.sum(q_pe, axis=0)

        pos = tl.load(positions_ptr + pid_m)
        cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
        cos = tl.load(cos_ptr + cos_offs)
        sin = tl.load(sin_ptr + cos_offs)

        q_pe_ptrs = q_base_ptrs + (NOPE_DIM + d_pe_offs) * stride_qd
        q_pe = _unit_rope(
            q_pe,
            cos,
            sin,
            d_pe_offs,
            IS_NEOX,
            ROPE_DIM,
            ROPE_DIM // 2,
        )
        tl.store(q_pe_ptrs, q_pe.to(q_out_ptr.dtype.element_ty))
        return

    if HAS_SWA:
        src_id = tl.load(swa_write_active_ptr + pid_m)
    else:
        src_id = pid_m

    pos = tl.load(positions_ptr + src_id)
    cos_offs = pos * cos_stride_t + d_cos_offs * cos_stride_d
    cos = tl.load(cos_ptr + cos_offs)
    sin = tl.load(sin_ptr + cos_offs)

    kv_base_ptrs = kv_ptr + src_id * stride_kv_m
    kv_nope_ptrs = kv_base_ptrs + offs_d_full * stride_kv_d
    kv_pe_ptrs = kv_base_ptrs + (NOPE_DIM + d_pe_offs) * stride_kv_d
    kv_nope = tl.load(kv_nope_ptrs, mask=offs_d_full < NOPE_DIM, other=0.0)
    kv_pe = tl.load(kv_pe_ptrs)
    kv_pe = _unit_rope(
        kv_pe,
        cos,
        sin,
        d_pe_offs,
        IS_NEOX,
        ROPE_DIM,
        ROPE_DIM // 2,
    )
    tl.store(kv_pe_ptrs, kv_pe.to(kv_ptr.dtype.element_ty))

    if HAS_SWA:
        if src_id >= 0:
            bid = tl.load(batch_id_per_token_ptr + src_id)
            slot = tl.load(state_slot_per_seq_ptr + bid)
            ring_idx = pos % win
            swa_kv_ptrs = (
                swa_kv_ptr + slot * swa_kv_slot_stride + ring_idx * swa_kv_pos_stride
            )
            tl.store(swa_kv_ptrs + offs_d_full, kv_nope, mask=offs_d_full < NOPE_DIM)
            tl.store(swa_kv_ptrs + NOPE_DIM + d_pe_offs, kv_pe)
