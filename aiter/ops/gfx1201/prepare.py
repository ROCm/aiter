# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid_3d
from aiter.ops.triton._triton_kernels.quant.sage_attention_quant import _general_quant_kernel

@triton.jit
def sage_quant_transposed_kernel(
    Q_Input,
    Q_Output,
    Q_Scale,
    K_Input,
    K_Output,
    K_Scale,
    V_Input,
    V_Output,
    V_Scale,
    stride_qz,
    stride_qh,
    stride_qn,
    stride_kz,
    stride_kh,
    stride_kn,
    stride_qsz,
    stride_qsh,
    stride_ksz,
    stride_ksh,
    stride_vsz,
    stride_vsh,
    sm_scale,
    q_task_count,
    k_task_count,
    BATCH,
    Q_HEAD,
    K_HEAD,
    Q_NUM_BLKS,
    K_NUM_BLKS,
    SEQLEN_Q,
    SEQLEN_K,
    STORAGE_SEQUENCE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    INT8_MAX: tl.constexpr,
    D: tl.constexpr,
    BLK_Q: tl.constexpr,
    BLK_K: tl.constexpr,
):
    pid = tl.program_id(0).to(tl.int64)

    offs_blk_q = tl.arange(0, BLK_Q)
    offs_blk_k = tl.arange(0, BLK_K)
    offs_d = tl.arange(0, D)

    if pid < q_task_count:
        # here we do Q
        off_blk, off_h, off_b = pid_grid_3d(pid, Q_NUM_BLKS, Q_HEAD, BATCH)
        offs_qn = off_blk * BLK_Q + offs_blk_q

        q_offs = (
            off_b * stride_qz
            + off_h * stride_qh
            + offs_qn[:, None] * stride_qn
            + offs_d[None, :]
        )

        q_input_ptrs = Q_Input + q_offs
        q_output_ptrs = Q_Output + ((off_b * STORAGE_SEQUENCE + offs_qn[:, None]) * Q_HEAD + off_h) * D + offs_d[None, :]
        q_scale_ptrs = Q_Scale + off_b * stride_qsz + off_h * stride_qsh + off_blk

        _general_quant_kernel(
            q_input_ptrs,
            q_output_ptrs,
            q_scale_ptrs,
            INT8_MAX,
            offs_qn[:, None] < SEQLEN_Q,
            sm_scale=sm_scale,
        )
        tl.store(q_output_ptrs, 0, mask=offs_qn[:, None] >= SEQLEN_Q)
    elif pid >= q_task_count and pid < q_task_count + k_task_count:
        # here we do K
        _pid = pid - q_task_count
        off_blk, off_h, off_b = pid_grid_3d(_pid, K_NUM_BLKS, K_HEAD, BATCH)

        offs_kn = off_blk * BLK_K + offs_blk_k

        k_offs = (
            off_b * stride_kz
            + off_h * stride_kh
            + offs_kn[:, None] * stride_kn
            + offs_d[None, :]
        )

        k_input_ptrs = K_Input + k_offs
        k_output_ptrs = K_Output + ((off_b * STORAGE_SEQUENCE + offs_kn[:, None]) * K_HEAD + off_h) * D + offs_d[None, :]
        k_scale_ptrs = K_Scale + off_b * stride_ksz + off_h * stride_ksh + off_blk

        _general_quant_kernel(
            k_input_ptrs,
            k_output_ptrs,
            k_scale_ptrs,
            INT8_MAX,
            offs_kn[:, None] < SEQLEN_K,
        )
        tl.store(k_output_ptrs, 0, mask=offs_kn[:, None] >= SEQLEN_K)
    else:
        # V
        _pid = pid - (q_task_count + k_task_count)
        off_blk, off_h, off_b = pid_grid_3d(_pid, K_NUM_BLKS, K_HEAD, BATCH)
        offs_kn = off_blk * BLK_K + offs_blk_k

        v_offs = (
            off_b * stride_kz
            + off_h * stride_kh
            + offs_kn[:, None] * stride_kn
            + offs_d[None, :]
        )

        v_input_ptrs = V_Input + v_offs
        v_output_ptrs = V_Output + ((off_b * K_HEAD + off_h) * D + offs_d[None, :]) * STORAGE_SEQUENCE + offs_kn[:, None]

        # just apply the per channel v_scales that have been computed outside
        v_scale_ptrs = (
            V_Scale + off_b * stride_vsz + off_h * stride_vsh + offs_d[None, :]
        )
        v = tl.load(v_input_ptrs, mask=offs_kn[:, None] < SEQLEN_K, other=0.0)
        v = v.to(tl.float32)
        v_scales = tl.load(v_scale_ptrs)
        v_quant = v / v_scales
        v_quant = v_quant.to(v_output_ptrs.dtype.element_ty)
        tl.store(v_output_ptrs, tl.where(offs_kn[:, None] < SEQLEN_K, v_quant, 0.0))


def prepare_sage(query, key, value):
    if query.shape != key.shape or query.shape != value.shape:
        raise ValueError("Expected equal Q/K/V shapes")
    if query.ndim != 4 or query.shape[-1] != 128 or any(size <= 0 for size in query.shape):
        raise ValueError("Expected positive [B,S,H,128]")
    if any(not tensor.is_contiguous() or tensor.dtype != torch.bfloat16 or not tensor.is_cuda or tensor.device != query.device for tensor in (query, key, value)):
        raise ValueError("Expected contiguous BF16 Q/K/V on the same GPU")
    batch, sequence, heads, dimension = query.shape
    padded_sequence = ((sequence + 31) // 32) * 32
    blocks = padded_sequence // 32
    query_int8 = torch.empty((batch, padded_sequence, heads, dimension), device=query.device, dtype=torch.int8)
    key_int8 = torch.empty_like(query_int8)
    value_fp8 = torch.empty((batch, heads, dimension, padded_sequence), device=value.device, dtype=torch.float8_e4m3fn)
    query_scale = torch.empty((batch, heads, blocks), device=query.device, dtype=torch.float32)
    key_scale = torch.empty_like(query_scale)
    value_scale = value.abs().amax(dim=1).to(torch.float32) / 448.0
    tasks = batch * heads * blocks
    sage_quant_transposed_kernel[(3 * tasks,)](
        query, query_int8, query_scale, key, key_int8, key_scale,
        value, value_fp8, value_scale,
        query.stride(0), query.stride(2), query.stride(1),
        key.stride(0), key.stride(2), key.stride(1),
        query_scale.stride(0), query_scale.stride(1),
        key_scale.stride(0), key_scale.stride(1),
        value_scale.stride(0), value_scale.stride(1),
        dimension**-0.5 * 1.4426950408889634,
        tasks, tasks, batch, heads, heads, blocks, blocks, sequence, sequence,
        padded_sequence, FP8_MAX=448.0, INT8_MAX=127,
        D=dimension, BLK_Q=32, BLK_K=32, num_stages=3, num_warps=8,
    )
    return query_int8, query_scale, key_int8, key_scale, value_fp8, value_scale
