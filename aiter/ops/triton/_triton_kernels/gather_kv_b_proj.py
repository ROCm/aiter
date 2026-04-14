# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


@triton.jit
def _load_unshuffle_segment(
    base_ptr,
    seg_idx,
    HeadDim: tl.constexpr,
    KV_CDim: tl.constexpr,
    ScaleKGranularity: tl.constexpr,
):
    """Load one [HeadDim, ScaleKGranularity] weight segment from a
    preshuffled weight matrix via coalesced row-major loads, then unshuffle
    in registers.

    Each n_blk (16 original N rows) occupies KV_CDim//32 shuffled rows.
    A ScaleK segment of 128 K values spans SegKBlocks=4 consecutive rows
    within each n_blk.  We gather these rows across all n_blks (with row
    stride KV_CDim), producing a [NumNBlk*SegKBlocks, KV_CDim] tensor,
    then reshape + permute to recover [HeadDim, ScaleKGranularity].
    """
    NumNBlk: tl.constexpr = HeadDim // 16
    SegKBlocks: tl.constexpr = ScaleKGranularity // 32
    NumKBlkTotal: tl.constexpr = KV_CDim // 32
    TotalRows: tl.constexpr = NumNBlk * SegKBlocks

    offs_nb = tl.arange(0, NumNBlk)
    offs_kb = tl.arange(0, SegKBlocks)
    row_indices = (
        offs_nb[:, None] * NumKBlkTotal + seg_idx * SegKBlocks + offs_kb[None, :]
    )
    row_indices_flat = tl.reshape(row_indices, (TotalRows,))

    offs_col = tl.arange(0, KV_CDim)
    raw = tl.load(base_ptr + row_indices_flat[:, None] * KV_CDim + offs_col[None, :])

    w = tl.reshape(
        tl.permute(
            tl.reshape(raw, (NumNBlk, SegKBlocks, 2, 16, 16)),
            (0, 3, 1, 2, 4),
        ),
        (HeadDim, ScaleKGranularity),
    )
    return w


@triton.jit
def _triton_gather_kv_b_proj(
    batch_size,
    k_buffer,  # [num_block, block_size, kv_c_dim + kv_pe_dim]
    k_scale,  # [1] or None
    kv_indptr,  # [batch_size + 1]
    kv_indices,  # [total_kv]
    kv_prefix_sum_context_lens,  # [batch_size + 1]
    kv_proj_weight,  # [tp_k_head_num * (qk_nope_head_dim + v_head_dim), kv_c_dim]
    kv_proj_scale,  # block: [n//128, k//128]; per-row: [weight_n] or [weight_n, 1]
    k_prefix,  # [total_kv, tp_k_head_num, qk_nope_head_dim + kv_pe_dim]
    v_prefix,  # [total_kv, tp_k_head_num, v_head_dim]
    KBlockSize: tl.constexpr,
    TpNumHeads: tl.constexpr,
    QkNopeHeadDim: tl.constexpr,
    VHeadDim: tl.constexpr,
    KV_CDim: tl.constexpr,
    KV_PeDim: tl.constexpr,
    ChunkK: tl.constexpr,
    WEIGHT_PRESHUFFLE: tl.constexpr = False,
    PER_ROW_SCALE: tl.constexpr = False,
):
    stride_k_buffer: tl.constexpr = KBlockSize * (KV_CDim + KV_PeDim)
    stride_k_prefix: tl.constexpr = TpNumHeads * (QkNopeHeadDim + KV_PeDim)
    stride_v_prefix: tl.constexpr = TpNumHeads * VHeadDim

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

    offs_n_k = tl.arange(0, QkNopeHeadDim)
    offs_n_v = tl.arange(0, VHeadDim)
    offs_k = tl.arange(0, ScaleKGranularity)
    k_head_base = kv_proj_weight + pid_head * (QkNopeHeadDim + VHeadDim) * KV_CDim
    v_head_base = k_head_base + QkNopeHeadDim * KV_CDim

    if PER_ROW_SCALE:
        k_row0 = pid_head * (QkNopeHeadDim + VHeadDim)
        k_nope_scale_vec = tl.load(kv_proj_scale + k_row0 + offs_n_k).to(tl.float32)
        v_nope_scale_vec = tl.load(
            kv_proj_scale + k_row0 + QkNopeHeadDim + offs_n_v
        ).to(tl.float32)
    else:
        num_scale_cols: tl.constexpr = KV_CDim // ScaleKGranularity
        k_abs_rows = pid_head * (QkNopeHeadDim + VHeadDim) + offs_n_k
        k_scale_n_idx = k_abs_rows // ScaleNGranularity
        v_abs_rows = pid_head * (QkNopeHeadDim + VHeadDim) + QkNopeHeadDim + offs_n_v
        v_scale_n_idx = v_abs_rows // ScaleNGranularity

    if WEIGHT_PRESHUFFLE:
        k_nope_weight_0 = _load_unshuffle_segment(
            k_head_base, 0, QkNopeHeadDim, KV_CDim, ScaleKGranularity
        ).to(k_type)
        k_nope_weight_1 = _load_unshuffle_segment(
            k_head_base, 1, QkNopeHeadDim, KV_CDim, ScaleKGranularity
        ).to(k_type)
        k_nope_weight_2 = _load_unshuffle_segment(
            k_head_base, 2, QkNopeHeadDim, KV_CDim, ScaleKGranularity
        ).to(k_type)
        k_nope_weight_3 = _load_unshuffle_segment(
            k_head_base, 3, QkNopeHeadDim, KV_CDim, ScaleKGranularity
        ).to(k_type)

        v_nope_weight_0 = _load_unshuffle_segment(
            v_head_base, 0, VHeadDim, KV_CDim, ScaleKGranularity
        ).to(k_type)
        v_nope_weight_1 = _load_unshuffle_segment(
            v_head_base, 1, VHeadDim, KV_CDim, ScaleKGranularity
        ).to(k_type)
        v_nope_weight_2 = _load_unshuffle_segment(
            v_head_base, 2, VHeadDim, KV_CDim, ScaleKGranularity
        ).to(k_type)
        v_nope_weight_3 = _load_unshuffle_segment(
            v_head_base, 3, VHeadDim, KV_CDim, ScaleKGranularity
        ).to(k_type)
    else:
        k_nope_weight_base_offset = (
            k_head_base + offs_n_k[:, None] * KV_CDim + offs_k[None, :]
        )
        k_nope_weight_0 = tl.load(k_nope_weight_base_offset + 0 * ScaleKGranularity).to(
            k_type
        )
        k_nope_weight_1 = tl.load(k_nope_weight_base_offset + 1 * ScaleKGranularity).to(
            k_type
        )
        k_nope_weight_2 = tl.load(k_nope_weight_base_offset + 2 * ScaleKGranularity).to(
            k_type
        )
        k_nope_weight_3 = tl.load(k_nope_weight_base_offset + 3 * ScaleKGranularity).to(
            k_type
        )

        v_nope_weight_base_offset = (
            v_head_base + offs_n_v[:, None] * KV_CDim + offs_k[None, :]
        )
        v_nope_weight_0 = tl.load(v_nope_weight_base_offset + 0 * ScaleKGranularity).to(
            k_type
        )
        v_nope_weight_1 = tl.load(v_nope_weight_base_offset + 1 * ScaleKGranularity).to(
            k_type
        )
        v_nope_weight_2 = tl.load(v_nope_weight_base_offset + 2 * ScaleKGranularity).to(
            k_type
        )
        v_nope_weight_3 = tl.load(v_nope_weight_base_offset + 3 * ScaleKGranularity).to(
            k_type
        )

    if not PER_ROW_SCALE:
        k_nope_scale_0 = tl.load(kv_proj_scale + k_scale_n_idx * num_scale_cols + 0).to(
            tl.float32
        )
        k_nope_scale_1 = tl.load(kv_proj_scale + k_scale_n_idx * num_scale_cols + 1).to(
            tl.float32
        )
        k_nope_scale_2 = tl.load(kv_proj_scale + k_scale_n_idx * num_scale_cols + 2).to(
            tl.float32
        )
        k_nope_scale_3 = tl.load(kv_proj_scale + k_scale_n_idx * num_scale_cols + 3).to(
            tl.float32
        )

        v_nope_scale_0 = tl.load(kv_proj_scale + v_scale_n_idx * num_scale_cols + 0).to(
            tl.float32
        )
        v_nope_scale_1 = tl.load(kv_proj_scale + v_scale_n_idx * num_scale_cols + 1).to(
            tl.float32
        )
        v_nope_scale_2 = tl.load(kv_proj_scale + v_scale_n_idx * num_scale_cols + 2).to(
            tl.float32
        )
        v_nope_scale_3 = tl.load(kv_proj_scale + v_scale_n_idx * num_scale_cols + 3).to(
            tl.float32
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
        accum_v = tl.zeros((ChunkK, VHeadDim), dtype=tl.float32)

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

        if PER_ROW_SCALE:
            accum_k += (
                tl.dot(kv_c_data_0, k_nope_weight_0.T) * k_nope_scale_vec[None, :]
            )
            accum_v += (
                tl.dot(kv_c_data_0, v_nope_weight_0.T) * v_nope_scale_vec[None, :]
            )
            accum_k += (
                tl.dot(kv_c_data_1, k_nope_weight_1.T) * k_nope_scale_vec[None, :]
            )
            accum_v += (
                tl.dot(kv_c_data_1, v_nope_weight_1.T) * v_nope_scale_vec[None, :]
            )
            accum_k += (
                tl.dot(kv_c_data_2, k_nope_weight_2.T) * k_nope_scale_vec[None, :]
            )
            accum_v += (
                tl.dot(kv_c_data_2, v_nope_weight_2.T) * v_nope_scale_vec[None, :]
            )
            accum_k += (
                tl.dot(kv_c_data_3, k_nope_weight_3.T) * k_nope_scale_vec[None, :]
            )
            accum_v += (
                tl.dot(kv_c_data_3, v_nope_weight_3.T) * v_nope_scale_vec[None, :]
            )
        else:
            accum_k += tl.dot(kv_c_data_0, k_nope_weight_0.T) * k_nope_scale_0[None, :]
            accum_v += tl.dot(kv_c_data_0, v_nope_weight_0.T) * v_nope_scale_0[None, :]
            accum_k += tl.dot(kv_c_data_1, k_nope_weight_1.T) * k_nope_scale_1[None, :]
            accum_v += tl.dot(kv_c_data_1, v_nope_weight_1.T) * v_nope_scale_1[None, :]
            accum_k += tl.dot(kv_c_data_2, k_nope_weight_2.T) * k_nope_scale_2[None, :]
            accum_v += tl.dot(kv_c_data_2, v_nope_weight_2.T) * v_nope_scale_2[None, :]
            accum_k += tl.dot(kv_c_data_3, k_nope_weight_3.T) * k_nope_scale_3[None, :]
            accum_v += tl.dot(kv_c_data_3, v_nope_weight_3.T) * v_nope_scale_3[None, :]

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
            + pid_head * VHeadDim
            + tl.arange(0, VHeadDim)[None, :],
            accum_v,
            mask=context_mask[:, None],
        )
