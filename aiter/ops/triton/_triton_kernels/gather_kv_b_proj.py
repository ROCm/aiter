# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _shuffled_weight_offset(offs_n, offs_k, KV_CDim: tl.constexpr):
    """Compute linear offset into a weight matrix that has been preshuffled
    via shuffle_weight(layout=(16,16)) for fp8.

    shuffle_weight transforms [N, K] with tile (BN=16, BK=32, K_pack=16):
      view(N//16, 16, K//32, 2, 16)  ->  permute(0, 1, 3, 4, 2, 5)  ->  view(N, K)
    Permuted shape: [batch, n_blk, k_blk, k_half, n_local, k_local]

    flat offset = n_blk * (KV_CDim // 32) * 512
               + k_blk * 512 + k_half * 256 + n_local * 16 + k_local
    """
    n_blk = offs_n // 16
    n_local = offs_n % 16
    k_blk = offs_k // 32
    k_half = (offs_k % 32) // 16
    k_local = offs_k % 16
    return (
        n_blk[:, None] * ((KV_CDim // 32) * 512)
        + k_blk[None, :] * 512
        + k_half[None, :] * 256
        + n_local[:, None] * 16
        + k_local[None, :]
    )


@triton.jit
def _triton_gather_kv_b_proj(
    batch_size,
    k_buffer,  # [num_block, block_size, kv_c_dim + kv_pe_dim]
    k_scale,  # [1] or None
    kv_indptr,  # [batch_size + 1]
    kv_indices,  # [total_kv]
    kv_prefix_sum_context_lens,  # [batch_size + 1]
    kv_proj_weight,  # [tp_k_head_num * 2 * qk_nope_head_dim, kv_c_dim]
    kv_proj_scale,  # [tp_k_head_num * 2 * qk_nope_head_dim // 128, kv_c_dim // 128]
    k_prefix,  # [total_kv, tp_k_head_num * qk_nope_head_dim + kv_pe_dim]
    v_prefix,  # [total_kv, tp_k_head_num * qk_nope_head_dim]
    KBlockSize: tl.constexpr,
    TpNumHeads: tl.constexpr,
    QkNopeHeadDim: tl.constexpr,
    KV_CDim: tl.constexpr,
    KV_PeDim: tl.constexpr,
    ChunkK: tl.constexpr,
    WEIGHT_PRESHUFFLE: tl.constexpr = False,
):
    stride_k_buffer: tl.constexpr = KBlockSize * (KV_CDim + KV_PeDim)
    stride_k_prefix: tl.constexpr = TpNumHeads * (QkNopeHeadDim + KV_PeDim)
    stride_v_prefix: tl.constexpr = TpNumHeads * QkNopeHeadDim

    ScaleKGranularity: tl.constexpr = 128
    ScaleNGranularity: tl.constexpr = 128
    KBlocksPerChunkK: tl.constexpr = ChunkK // KBlockSize
    assert KV_CDim == 4 * ScaleKGranularity

    # ===---------------------------------------------------
    # Workload Partition
    # ===---------------------------------------------------
    pid = tl.program_id(0)
    pid_batch = pid // TpNumHeads
    pid_head = pid % TpNumHeads

    kv_block_start = tl.load(kv_indptr + pid_batch)
    kv_block_end = tl.load(kv_indptr + pid_batch + 1)

    context_start = tl.load(kv_prefix_sum_context_lens + pid_batch)
    context_end = tl.load(kv_prefix_sum_context_lens + pid_batch + 1)

    total_kv_block = kv_block_end - kv_block_start
    total_kv_chunk = (total_kv_block + KBlocksPerChunkK - 1) // KBlocksPerChunkK

    # ===---------------------------------------------------
    # Pipeline Start
    # ===---------------------------------------------------
    k_type = k_buffer.dtype.element_ty
    if k_type == tl.bfloat16:
        k_scalar_scale = 1.0
    else:
        k_scalar_scale = tl.load(k_scale)

    k_nope_scale_base_offset = (
        kv_proj_scale
        + pid_head
        * 2
        * QkNopeHeadDim
        * KV_CDim
        // ScaleKGranularity
        // ScaleNGranularity
        + tl.arange(0, QkNopeHeadDim // ScaleNGranularity)
        * (KV_CDim // ScaleKGranularity)
    )

    offs_n = tl.arange(0, QkNopeHeadDim)
    offs_k = tl.arange(0, ScaleKGranularity)
    k_head_base = kv_proj_weight + pid_head * 2 * QkNopeHeadDim * KV_CDim
    v_head_base = k_head_base + QkNopeHeadDim * KV_CDim

    if WEIGHT_PRESHUFFLE:
        k_nope_weight_0 = tl.load(k_head_base + _shuffled_weight_offset(offs_n, offs_k + 0 * ScaleKGranularity, KV_CDim)).to(k_type)
        k_nope_weight_1 = tl.load(k_head_base + _shuffled_weight_offset(offs_n, offs_k + 1 * ScaleKGranularity, KV_CDim)).to(k_type)
        k_nope_weight_2 = tl.load(k_head_base + _shuffled_weight_offset(offs_n, offs_k + 2 * ScaleKGranularity, KV_CDim)).to(k_type)
        k_nope_weight_3 = tl.load(k_head_base + _shuffled_weight_offset(offs_n, offs_k + 3 * ScaleKGranularity, KV_CDim)).to(k_type)

        v_nope_weight_0 = tl.load(v_head_base + _shuffled_weight_offset(offs_n, offs_k + 0 * ScaleKGranularity, KV_CDim)).to(k_type)
        v_nope_weight_1 = tl.load(v_head_base + _shuffled_weight_offset(offs_n, offs_k + 1 * ScaleKGranularity, KV_CDim)).to(k_type)
        v_nope_weight_2 = tl.load(v_head_base + _shuffled_weight_offset(offs_n, offs_k + 2 * ScaleKGranularity, KV_CDim)).to(k_type)
        v_nope_weight_3 = tl.load(v_head_base + _shuffled_weight_offset(offs_n, offs_k + 3 * ScaleKGranularity, KV_CDim)).to(k_type)
    else:
        k_nope_weight_base_offset = (
            k_head_base
            + offs_n[:, None] * KV_CDim
            + offs_k[None, :]
        )
        k_nope_weight_0 = tl.load(k_nope_weight_base_offset + 0 * ScaleKGranularity).to(k_type)
        k_nope_weight_1 = tl.load(k_nope_weight_base_offset + 1 * ScaleKGranularity).to(k_type)
        k_nope_weight_2 = tl.load(k_nope_weight_base_offset + 2 * ScaleKGranularity).to(k_type)
        k_nope_weight_3 = tl.load(k_nope_weight_base_offset + 3 * ScaleKGranularity).to(k_type)

        v_nope_weight_0 = tl.load(k_nope_weight_base_offset + QkNopeHeadDim * KV_CDim + 0 * ScaleKGranularity).to(k_type)
        v_nope_weight_1 = tl.load(k_nope_weight_base_offset + QkNopeHeadDim * KV_CDim + 1 * ScaleKGranularity).to(k_type)
        v_nope_weight_2 = tl.load(k_nope_weight_base_offset + QkNopeHeadDim * KV_CDim + 2 * ScaleKGranularity).to(k_type)
        v_nope_weight_3 = tl.load(k_nope_weight_base_offset + QkNopeHeadDim * KV_CDim + 3 * ScaleKGranularity).to(k_type)

    k_nope_scale_0 = tl.load(k_nope_scale_base_offset + 0)
    k_nope_scale_1 = tl.load(k_nope_scale_base_offset + 1)
    k_nope_scale_2 = tl.load(k_nope_scale_base_offset + 2)
    k_nope_scale_3 = tl.load(k_nope_scale_base_offset + 3)

    v_nope_scale_0 = tl.load(
        k_nope_scale_base_offset
        + QkNopeHeadDim * KV_CDim // ScaleNGranularity // ScaleKGranularity
        + 0
    )
    v_nope_scale_1 = tl.load(
        k_nope_scale_base_offset
        + QkNopeHeadDim * KV_CDim // ScaleNGranularity // ScaleKGranularity
        + 1
    )
    v_nope_scale_2 = tl.load(
        k_nope_scale_base_offset
        + QkNopeHeadDim * KV_CDim // ScaleNGranularity // ScaleKGranularity
        + 2
    )
    v_nope_scale_3 = tl.load(
        k_nope_scale_base_offset
        + QkNopeHeadDim * KV_CDim // ScaleNGranularity // ScaleKGranularity
        + 3
    )

    for chunk_id in range(total_kv_chunk):
        kv_block_idx = tl.load(
            kv_indices
            + kv_block_start
            + chunk_id * KBlocksPerChunkK
            + tl.arange(0, ChunkK) // KBlockSize,
            mask=chunk_id * KBlocksPerChunkK + tl.arange(0, ChunkK) // KBlockSize
            < total_kv_block,
        )
        kv_c_data_base_offset = (
            kv_block_idx[:, None] * stride_k_buffer
            + tl.arange(0, ChunkK)[:, None] % KBlockSize * (KV_CDim + KV_PeDim)
            + tl.arange(0, ScaleKGranularity)[None, :]
        )  # [ChunkK, kv_c_dim]

        accum_k = tl.zeros((ChunkK, QkNopeHeadDim), dtype=tl.float32)
        accum_v = tl.zeros((ChunkK, QkNopeHeadDim), dtype=tl.float32)

        kv_c_data_0 = tl.load(k_buffer + kv_c_data_base_offset + 0 * ScaleKGranularity)
        kv_c_data_1 = tl.load(k_buffer + kv_c_data_base_offset + 1 * ScaleKGranularity)
        kv_c_data_2 = tl.load(k_buffer + kv_c_data_base_offset + 2 * ScaleKGranularity)
        kv_c_data_3 = tl.load(k_buffer + kv_c_data_base_offset + 3 * ScaleKGranularity)
        kv_pe_data = tl.load(
            k_buffer
            + kv_block_idx[:, None] * stride_k_buffer
            + tl.arange(0, ChunkK)[:, None] % KBlockSize * (KV_CDim + KV_PeDim)
            + KV_CDim
            + tl.arange(0, KV_PeDim)[None, :],
        )

        accum_k += tl.dot(kv_c_data_0, k_nope_weight_0.T) * k_nope_scale_0
        accum_v += tl.dot(kv_c_data_0, v_nope_weight_0.T) * v_nope_scale_0
        accum_k += tl.dot(kv_c_data_1, k_nope_weight_1.T) * k_nope_scale_1
        accum_v += tl.dot(kv_c_data_1, v_nope_weight_1.T) * v_nope_scale_1
        accum_k += tl.dot(kv_c_data_2, k_nope_weight_2.T) * k_nope_scale_2
        accum_v += tl.dot(kv_c_data_2, v_nope_weight_2.T) * v_nope_scale_2
        accum_k += tl.dot(kv_c_data_3, k_nope_weight_3.T) * k_nope_scale_3
        accum_v += tl.dot(kv_c_data_3, v_nope_weight_3.T) * v_nope_scale_3

        accum_k *= k_scalar_scale
        accum_v *= k_scalar_scale
        kv_pe_data *= k_scalar_scale

        context_mask = (
            context_start + chunk_id * ChunkK + tl.arange(0, ChunkK) < context_end
        )
        tl.store(
            k_prefix
            + (context_start + chunk_id * ChunkK + tl.arange(0, ChunkK))[:, None]
            * stride_k_prefix
            + pid_head * (QkNopeHeadDim + KV_PeDim)
            + QkNopeHeadDim
            + tl.arange(0, KV_PeDim)[None, :],
            kv_pe_data,
            mask=context_mask[:, None],
        )
        tl.store(
            k_prefix
            + (context_start + chunk_id * ChunkK + tl.arange(0, ChunkK))[:, None]
            * stride_k_prefix
            + pid_head * (QkNopeHeadDim + KV_PeDim)
            + tl.arange(0, QkNopeHeadDim)[None, :],
            accum_k,
            mask=context_mask[:, None],
        )
        tl.store(
            v_prefix
            + (context_start + chunk_id * ChunkK + tl.arange(0, ChunkK))[:, None]
            * stride_v_prefix
            + pid_head * QkNopeHeadDim
            + tl.arange(0, QkNopeHeadDim)[None, :],
            accum_v,
            mask=context_mask[:, None],
        )
