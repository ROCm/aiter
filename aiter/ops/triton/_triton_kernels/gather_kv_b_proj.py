# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl


def _next_pow2(n):
    """Return the smallest power of 2 >= n (Python-side helper, not a JIT function)."""
    return 1 << (n - 1).bit_length()


@triton.jit
def _load_unshuffle_segment(
    base_ptr,
    seg_idx,
    HeadDim: tl.constexpr,
    PaddedHeadDim: tl.constexpr,
    KV_CDim: tl.constexpr,
    ScaleKGranularity: tl.constexpr,
):
    """Load one [PaddedHeadDim, ScaleKGranularity] weight segment from a
    preshuffled weight matrix via coalesced row-major loads, then unshuffle
    in registers.  PaddedHeadDim is HeadDim rounded up to the next power of 2.
    Out-of-range rows are zero-filled so dot-products stay correct.
    """
    NumNBlk: tl.constexpr = HeadDim // 16
    PaddedNumNBlk: tl.constexpr = PaddedHeadDim // 16
    SegKBlocks: tl.constexpr = ScaleKGranularity // 32
    NumKBlkTotal: tl.constexpr = KV_CDim // 32
    PaddedTotalRows: tl.constexpr = PaddedNumNBlk * SegKBlocks

    offs_nb = tl.arange(0, PaddedNumNBlk)
    offs_kb = tl.arange(0, SegKBlocks)
    row_indices = (
        offs_nb[:, None] * NumKBlkTotal + seg_idx * SegKBlocks + offs_kb[None, :]
    )
    row_indices_flat = tl.reshape(row_indices, (PaddedTotalRows,))
    mask_flat = tl.reshape(
        (offs_nb[:, None] < NumNBlk).broadcast_to(PaddedNumNBlk, SegKBlocks),
        (PaddedTotalRows,),
    )

    offs_col = tl.arange(0, KV_CDim)
    raw = tl.load(
        base_ptr + row_indices_flat[:, None] * KV_CDim + offs_col[None, :],
        mask=mask_flat[:, None],
        other=0.0,
    )

    w = tl.reshape(
        tl.permute(
            tl.reshape(raw, (PaddedNumNBlk, SegKBlocks, 2, 16, 16)),
            (0, 3, 1, 2, 4),
        ),
        (PaddedHeadDim, ScaleKGranularity),
    )
    return w


@triton.jit
def _triton_gather_kv_b_proj_fp4(
    batch_size,
    k_buffer,  # [num_block, block_size, kv_c_dim + kv_pe_dim]
    k_scale,  # [1] or None
    kv_indptr,  # [batch_size + 1]
    kv_indices,  # [total_kv]
    kv_prefix_sum_context_lens,  # [batch_size + 1]
    kv_proj_weight,  # packed fp4: [tp_heads * (qk_nope_head_dim + v_head_dim), kv_c_dim // 2]
    kv_proj_scale,  # e8m0 per-1x32: [weight_n, kv_c_dim // 32]
    k_prefix,  # [total_kv, tp_k_head_num, qk_nope_head_dim + kv_pe_dim]
    v_prefix,  # [total_kv, tp_k_head_num, v_head_dim]
    KBlockSize: tl.constexpr,
    TpNumHeads: tl.constexpr,
    QkNopeHeadDim: tl.constexpr,
    VHeadDim: tl.constexpr,
    KV_CDim: tl.constexpr,
    KV_PeDim: tl.constexpr,
    ChunkK: tl.constexpr,
    PaddedK: tl.constexpr,
    PaddedV: tl.constexpr,
    ScaleCols: tl.constexpr,
    WEIGHT_PRESHUFFLE: tl.constexpr = False,
):
    """FP4/per-1x32 gather + kv_b_proj expansion for raw MXFP4 weights."""
    stride_k_buffer = tl.full([], KBlockSize * (KV_CDim + KV_PeDim), dtype=tl.int64)
    stride_k_prefix = tl.full(
        [], TpNumHeads * (QkNopeHeadDim + KV_PeDim), dtype=tl.int64
    )
    stride_v_prefix = tl.full([], TpNumHeads * VHeadDim, dtype=tl.int64)

    ScaleKGranularity: tl.constexpr = 32
    PackedScaleKGranularity: tl.constexpr = ScaleKGranularity // 2
    PackedKV_CDim: tl.constexpr = KV_CDim // 2
    KBlocksPerChunkK: tl.constexpr = ChunkK // KBlockSize
    NumKSegments: tl.constexpr = KV_CDim // ScaleKGranularity

    pid = tl.program_id(0)
    pid_batch = pid // TpNumHeads
    pid_head = pid % TpNumHeads

    kv_block_start = tl.load(kv_indptr + pid_batch)
    kv_block_end = tl.load(kv_indptr + pid_batch + 1)

    context_start = tl.load(kv_prefix_sum_context_lens + pid_batch)
    context_end = tl.load(kv_prefix_sum_context_lens + pid_batch + 1)

    total_kv_block = kv_block_end - kv_block_start
    total_kv_chunk = (total_kv_block + KBlocksPerChunkK - 1) // KBlocksPerChunkK

    k_type = k_buffer.dtype.element_ty
    if k_type == tl.bfloat16:
        k_scalar_scale = 1.0
    else:
        k_scalar_scale = tl.load(k_scale)

    offs_n_k = tl.arange(0, PaddedK)
    offs_n_v = tl.arange(0, PaddedV)
    mask_k = offs_n_k < QkNopeHeadDim
    mask_v = offs_n_v < VHeadDim
    offs_k = tl.arange(0, ScaleKGranularity)
    offs_k_packed = tl.arange(0, PackedScaleKGranularity)

    head_row_base = pid_head * (QkNopeHeadDim + VHeadDim)
    k_abs_rows = head_row_base + offs_n_k
    v_abs_rows = head_row_base + QkNopeHeadDim + offs_n_v

    for chunk_id in range(total_kv_chunk):
        block_lane_valid = (
            chunk_id * KBlocksPerChunkK + tl.arange(0, ChunkK) // KBlockSize
            < total_kv_block
        )
        kv_block_idx = tl.load(
            kv_indices
            + kv_block_start
            + chunk_id * KBlocksPerChunkK
            + tl.arange(0, ChunkK) // KBlockSize,
            mask=block_lane_valid,
            other=0,
        )

        accum_k = tl.zeros((ChunkK, PaddedK), dtype=tl.float32)
        accum_v = tl.zeros((ChunkK, PaddedV), dtype=tl.float32)
        row_mask = block_lane_valid[:, None]

        for seg in range(NumKSegments):
            kv_c_data = tl.load(
                k_buffer
                + kv_block_idx[:, None] * stride_k_buffer
                + tl.arange(0, ChunkK)[:, None] % KBlockSize * (KV_CDim + KV_PeDim)
                + seg * ScaleKGranularity
                + offs_k[None, :],
                mask=row_mask,
                other=0.0,
            ).to(tl.bfloat16)

            packed_cols = seg * PackedScaleKGranularity + offs_k_packed[:, None]

            if WEIGHT_PRESHUFFLE:
                # Inverse of aiter.ops.shuffle.shuffle_weight for uint8/packed-fp4
                # tensors with layout=(16, 16).  It maps logical [N, K//2]
                # coordinates back into the preshuffled storage.
                k_n0 = k_abs_rows[None, :] // 16
                k_bn = k_abs_rows[None, :] % 16
                k_k0 = packed_cols // 32
                k_r = (packed_cols % 32) // 16
                k_c = packed_cols % 16
                k_weight_offset = (
                    ((k_n0 * (PackedKV_CDim // 32) + k_k0) * 2 + k_r) * 16 + k_bn
                ) * 16 + k_c

                v_n0 = v_abs_rows[None, :] // 16
                v_bn = v_abs_rows[None, :] % 16
                v_k0 = packed_cols // 32
                v_r = (packed_cols % 32) // 16
                v_c = packed_cols % 16
                v_weight_offset = (
                    ((v_n0 * (PackedKV_CDim // 32) + v_k0) * 2 + v_r) * 16 + v_bn
                ) * 16 + v_c
            else:
                k_weight_offset = k_abs_rows[None, :] * PackedKV_CDim + packed_cols
                v_weight_offset = v_abs_rows[None, :] * PackedKV_CDim + packed_cols

            k_weight = tl.load(
                kv_proj_weight + k_weight_offset,
                mask=mask_k[None, :],
                other=0,
            )
            v_weight = tl.load(
                kv_proj_weight + v_weight_offset,
                mask=mask_v[None, :],
                other=0,
            )

            if WEIGHT_PRESHUFFLE:
                # Inverse of fp4_utils.e8m0_shuffle / shuffle_scale.
                k_scale_n = k_abs_rows[:, None]
                k_scale_g = seg
                k_scale_a = k_scale_n // 32
                k_scale_b = (k_scale_n % 32) // 16
                k_scale_c = k_scale_n % 16
                k_scale_d = k_scale_g // 8
                k_scale_e = (k_scale_g % 8) // 4
                k_scale_f = k_scale_g % 4
                k_scale_offset = (
                    (
                        ((k_scale_a * (ScaleCols // 8) + k_scale_d) * 4 + k_scale_f)
                        * 16
                        + k_scale_c
                    )
                    * 2
                    + k_scale_e
                ) * 2 + k_scale_b

                v_scale_n = v_abs_rows[:, None]
                v_scale_g = seg
                v_scale_a = v_scale_n // 32
                v_scale_b = (v_scale_n % 32) // 16
                v_scale_c = v_scale_n % 16
                v_scale_d = v_scale_g // 8
                v_scale_e = (v_scale_g % 8) // 4
                v_scale_f = v_scale_g % 4
                v_scale_offset = (
                    (
                        ((v_scale_a * (ScaleCols // 8) + v_scale_d) * 4 + v_scale_f)
                        * 16
                        + v_scale_c
                    )
                    * 2
                    + v_scale_e
                ) * 2 + v_scale_b
            else:
                k_scale_offset = k_abs_rows[:, None] * ScaleCols + seg
                v_scale_offset = v_abs_rows[:, None] * ScaleCols + seg

            k_weight_scale = tl.load(
                kv_proj_scale + k_scale_offset,
                mask=mask_k[:, None],
                other=0,
            )
            v_weight_scale = tl.load(
                kv_proj_scale + v_scale_offset,
                mask=mask_v[:, None],
                other=0,
            )

            accum_k = tl.dot_scaled(
                kv_c_data,
                None,
                "bf16",
                k_weight,
                k_weight_scale,
                "e2m1",
                acc=accum_k,
                fast_math=True,
            )
            accum_v = tl.dot_scaled(
                kv_c_data,
                None,
                "bf16",
                v_weight,
                v_weight_scale,
                "e2m1",
                acc=accum_v,
                fast_math=True,
            )

        kv_pe_data = tl.load(
            k_buffer
            + kv_block_idx[:, None] * stride_k_buffer
            + tl.arange(0, ChunkK)[:, None] % KBlockSize * (KV_CDim + KV_PeDim)
            + KV_CDim
            + tl.arange(0, KV_PeDim)[None, :],
            mask=row_mask,
            other=0.0,
        )

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
            + offs_n_k[None, :],
            accum_k,
            mask=context_mask[:, None] & mask_k[None, :],
        )
        tl.store(
            v_prefix
            + (context_start + chunk_id * ChunkK + tl.arange(0, ChunkK))[:, None]
            * stride_v_prefix
            + pid_head * VHeadDim
            + offs_n_v[None, :],
            accum_v,
            mask=context_mask[:, None] & mask_v[None, :],
        )


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
    PaddedK: tl.constexpr,
    PaddedV: tl.constexpr,
    WEIGHT_PRESHUFFLE: tl.constexpr = False,
    PER_ROW_SCALE: tl.constexpr = False,
    NO_SCALE: tl.constexpr = False,
):
    # All three strides are multiplied by runtime indices that can overflow
    # i32 at large scales. Promote the scalar (broadcast) side to i64 so the multiply
    # is i32-zext x i64 -> i64
    stride_k_buffer = tl.full([], KBlockSize * (KV_CDim + KV_PeDim), dtype=tl.int64)
    stride_k_prefix = tl.full(
        [], TpNumHeads * (QkNopeHeadDim + KV_PeDim), dtype=tl.int64
    )
    stride_v_prefix = tl.full([], TpNumHeads * VHeadDim, dtype=tl.int64)

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

    offs_n_k = tl.arange(0, PaddedK)
    offs_n_v = tl.arange(0, PaddedV)
    mask_k = offs_n_k < QkNopeHeadDim
    mask_v = offs_n_v < VHeadDim
    offs_k = tl.arange(0, ScaleKGranularity)
    k_head_base = kv_proj_weight + pid_head * (QkNopeHeadDim + VHeadDim) * KV_CDim
    v_head_base = k_head_base + QkNopeHeadDim * KV_CDim

    if NO_SCALE:
        # weight is not quantized; skip scale loading entirely
        pass
    elif PER_ROW_SCALE:
        k_row0 = pid_head * (QkNopeHeadDim + VHeadDim)
        k_nope_scale_vec = tl.load(
            kv_proj_scale + k_row0 + offs_n_k, mask=mask_k, other=1.0
        ).to(tl.float32)
        v_nope_scale_vec = tl.load(
            kv_proj_scale + k_row0 + QkNopeHeadDim + offs_n_v, mask=mask_v, other=1.0
        ).to(tl.float32)
    else:
        num_scale_cols: tl.constexpr = KV_CDim // ScaleKGranularity
        k_abs_rows = pid_head * (QkNopeHeadDim + VHeadDim) + offs_n_k
        k_scale_n_idx = k_abs_rows // ScaleNGranularity
        v_abs_rows = pid_head * (QkNopeHeadDim + VHeadDim) + QkNopeHeadDim + offs_n_v
        v_scale_n_idx = v_abs_rows // ScaleNGranularity

    if WEIGHT_PRESHUFFLE:
        # _load_unshuffle_segment returns [PaddedHeadDim, ScaleKGranularity]
        # with zero-filled rows beyond HeadDim
        k_nope_weight_0 = _load_unshuffle_segment(
            k_head_base, 0, QkNopeHeadDim, PaddedK, KV_CDim, ScaleKGranularity
        ).to(k_type)
        k_nope_weight_1 = _load_unshuffle_segment(
            k_head_base, 1, QkNopeHeadDim, PaddedK, KV_CDim, ScaleKGranularity
        ).to(k_type)
        k_nope_weight_2 = _load_unshuffle_segment(
            k_head_base, 2, QkNopeHeadDim, PaddedK, KV_CDim, ScaleKGranularity
        ).to(k_type)
        k_nope_weight_3 = _load_unshuffle_segment(
            k_head_base, 3, QkNopeHeadDim, PaddedK, KV_CDim, ScaleKGranularity
        ).to(k_type)

        v_nope_weight_0 = _load_unshuffle_segment(
            v_head_base, 0, VHeadDim, PaddedV, KV_CDim, ScaleKGranularity
        ).to(k_type)
        v_nope_weight_1 = _load_unshuffle_segment(
            v_head_base, 1, VHeadDim, PaddedV, KV_CDim, ScaleKGranularity
        ).to(k_type)
        v_nope_weight_2 = _load_unshuffle_segment(
            v_head_base, 2, VHeadDim, PaddedV, KV_CDim, ScaleKGranularity
        ).to(k_type)
        v_nope_weight_3 = _load_unshuffle_segment(
            v_head_base, 3, VHeadDim, PaddedV, KV_CDim, ScaleKGranularity
        ).to(k_type)
    else:
        k_nope_weight_base_offset = (
            k_head_base + offs_n_k[:, None] * KV_CDim + offs_k[None, :]
        )
        k_mask_2d = mask_k[:, None]
        k_nope_weight_0 = tl.load(
            k_nope_weight_base_offset + 0 * ScaleKGranularity,
            mask=k_mask_2d,
            other=0.0,
        ).to(k_type)
        k_nope_weight_1 = tl.load(
            k_nope_weight_base_offset + 1 * ScaleKGranularity,
            mask=k_mask_2d,
            other=0.0,
        ).to(k_type)
        k_nope_weight_2 = tl.load(
            k_nope_weight_base_offset + 2 * ScaleKGranularity,
            mask=k_mask_2d,
            other=0.0,
        ).to(k_type)
        k_nope_weight_3 = tl.load(
            k_nope_weight_base_offset + 3 * ScaleKGranularity,
            mask=k_mask_2d,
            other=0.0,
        ).to(k_type)

        v_nope_weight_base_offset = (
            v_head_base + offs_n_v[:, None] * KV_CDim + offs_k[None, :]
        )
        v_mask_2d = mask_v[:, None]
        v_nope_weight_0 = tl.load(
            v_nope_weight_base_offset + 0 * ScaleKGranularity,
            mask=v_mask_2d,
            other=0.0,
        ).to(k_type)
        v_nope_weight_1 = tl.load(
            v_nope_weight_base_offset + 1 * ScaleKGranularity,
            mask=v_mask_2d,
            other=0.0,
        ).to(k_type)
        v_nope_weight_2 = tl.load(
            v_nope_weight_base_offset + 2 * ScaleKGranularity,
            mask=v_mask_2d,
            other=0.0,
        ).to(k_type)
        v_nope_weight_3 = tl.load(
            v_nope_weight_base_offset + 3 * ScaleKGranularity,
            mask=v_mask_2d,
            other=0.0,
        ).to(k_type)

    if (not NO_SCALE) and (not PER_ROW_SCALE):
        k_nope_scale_0 = tl.load(
            kv_proj_scale + k_scale_n_idx * num_scale_cols + 0,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k_nope_scale_1 = tl.load(
            kv_proj_scale + k_scale_n_idx * num_scale_cols + 1,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k_nope_scale_2 = tl.load(
            kv_proj_scale + k_scale_n_idx * num_scale_cols + 2,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)
        k_nope_scale_3 = tl.load(
            kv_proj_scale + k_scale_n_idx * num_scale_cols + 3,
            mask=mask_k,
            other=0.0,
        ).to(tl.float32)

        v_nope_scale_0 = tl.load(
            kv_proj_scale + v_scale_n_idx * num_scale_cols + 0,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        v_nope_scale_1 = tl.load(
            kv_proj_scale + v_scale_n_idx * num_scale_cols + 1,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        v_nope_scale_2 = tl.load(
            kv_proj_scale + v_scale_n_idx * num_scale_cols + 2,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)
        v_nope_scale_3 = tl.load(
            kv_proj_scale + v_scale_n_idx * num_scale_cols + 3,
            mask=mask_v,
            other=0.0,
        ).to(tl.float32)

    for chunk_id in range(total_kv_chunk):
        block_lane_valid = (
            chunk_id * KBlocksPerChunkK + tl.arange(0, ChunkK) // KBlockSize
            < total_kv_block
        )
        kv_block_idx = tl.load(
            kv_indices
            + kv_block_start
            + chunk_id * KBlocksPerChunkK
            + tl.arange(0, ChunkK) // KBlockSize,
            mask=block_lane_valid,
            other=0,
        )
        kv_c_data_base_offset = (
            kv_block_idx[:, None] * stride_k_buffer
            + tl.arange(0, ChunkK)[:, None] % KBlockSize * (KV_CDim + KV_PeDim)
            + tl.arange(0, ScaleKGranularity)[None, :]
        )  # [ChunkK, kv_c_dim]

        accum_k = tl.zeros((ChunkK, PaddedK), dtype=tl.float32)
        accum_v = tl.zeros((ChunkK, PaddedV), dtype=tl.float32)

        row_mask = block_lane_valid[:, None]
        kv_c_data_0 = tl.load(
            k_buffer + kv_c_data_base_offset + 0 * ScaleKGranularity,
            mask=row_mask,
            other=0.0,
        )
        kv_c_data_1 = tl.load(
            k_buffer + kv_c_data_base_offset + 1 * ScaleKGranularity,
            mask=row_mask,
            other=0.0,
        )
        kv_c_data_2 = tl.load(
            k_buffer + kv_c_data_base_offset + 2 * ScaleKGranularity,
            mask=row_mask,
            other=0.0,
        )
        kv_c_data_3 = tl.load(
            k_buffer + kv_c_data_base_offset + 3 * ScaleKGranularity,
            mask=row_mask,
            other=0.0,
        )
        kv_pe_data = tl.load(
            k_buffer
            + kv_block_idx[:, None] * stride_k_buffer
            + tl.arange(0, ChunkK)[:, None] % KBlockSize * (KV_CDim + KV_PeDim)
            + KV_CDim
            + tl.arange(0, KV_PeDim)[None, :],
            mask=row_mask,
            other=0.0,
        )

        if NO_SCALE:
            accum_k = tl.dot(kv_c_data_0, k_nope_weight_0.T, acc=accum_k)
            accum_v = tl.dot(kv_c_data_0, v_nope_weight_0.T, acc=accum_v)
            accum_k = tl.dot(kv_c_data_1, k_nope_weight_1.T, acc=accum_k)
            accum_v = tl.dot(kv_c_data_1, v_nope_weight_1.T, acc=accum_v)
            accum_k = tl.dot(kv_c_data_2, k_nope_weight_2.T, acc=accum_k)
            accum_v = tl.dot(kv_c_data_2, v_nope_weight_2.T, acc=accum_v)
            accum_k = tl.dot(kv_c_data_3, k_nope_weight_3.T, acc=accum_k)
            accum_v = tl.dot(kv_c_data_3, v_nope_weight_3.T, acc=accum_v)
        elif PER_ROW_SCALE:
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
            + offs_n_k[None, :],
            accum_k,
            mask=context_mask[:, None] & mask_k[None, :],
        )
        tl.store(
            v_prefix
            + (context_start + chunk_id * ChunkK + tl.arange(0, ChunkK))[:, None]
            * stride_v_prefix
            + pid_head * VHeadDim
            + offs_n_v[None, :],
            accum_v,
            mask=context_mask[:, None] & mask_v[None, :],
        )
