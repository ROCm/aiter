# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import lru_cache
import torch
import aiter
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.types import torch_to_triton_dtype
from aiter.ops.attention import pa_reduce_v1

import triton
import triton.language as tl

GLUON_JIT_KERNEL_ENABLED = True
try:
    from triton.experimental import gluon
    from triton.experimental.gluon import language as gl
except ImportError:
    print(
        "Warning: triton.experimental.gluon or triton.experimental.gluon.language not exists, gluon can only be used in triton AOT mode!"
    )
    gluon = triton
    gl = tl
    GLUON_JIT_KERNEL_ENABLED = False


@lru_cache(maxsize=1)
def get_cdna_version():
    """Get CDNA version lazily to avoid CUDA initialization during import."""
    if arch_info.get_arch() in ["gfx950"]:
        return 4
    elif arch_info.get_arch() in ["gfx942"]:
        return 3
    else:
        return -1


def parse_triton_version(version_str):
    """Parse version string into comparable tuple format, handling possible development version suffixes"""
    # Remove potential suffixes like .dev, +git etc.
    version_str = version_str.split("+")[0].split("-")[0]

    # Split version number and convert to integers
    parts = []
    for part in version_str.split("."):
        try:
            parts.append(int(part))
        except ValueError:
            break
    return tuple(parts)


TRITON_VERSION = parse_triton_version(triton.__version__)
# Pre-compute version check as constexpr for use in JIT kernels
TRITON_VERSION_GE_3_6_0 = tl.constexpr(TRITON_VERSION >= (3, 6, 0))


@gluon.jit
def pa_ps_kernel(
    work_indptr,
    work_info,
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    split_lse_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [sum_qlen, num_query_heads, head_size]
    query_ptr,  # [sum_qlen, num_query_heads, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    qo_indptr,
    kv_indptr,
    kv_page_indices,
    context_lengths_ptr,  # [num_seqs]
    softmax_scale: float,
    query_scale,  # [num_seqs, query_length, num_kv_heads, query_group_size, 1](per-token) or [1](per-tensor) or None
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    stride_logits_0: int,
    stride_logits_1: int,
    stride_logits_2: int,
    stride_lse_0: int,
    # 5D output strides for [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    stride_query_0: int,
    stride_query_1: int,
    stride_key_0: int,
    stride_key_1: int,
    stride_key_2: int,
    stride_key_3: int,
    stride_value_0: int,
    stride_value_1: int,
    stride_value_2: int,
    stride_query_scale_0: int,
    stride_query_scale_1: int,
    stride_kv_scale_0: int,
    stride_kv_scale_1: int,
    max_qlen: int,
    query_group_size: int,
    head_size: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
    ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,  # [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    IS_CAUSAL: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr = 0,
    CDNA_VERSION: gl.constexpr = 3,
):
    """
    Paged attention persistent scheduling kernel (Gluon JIT).

    This kernel computes attention for variable-length queries against a paged KV
    cache, using metadata produced by get_pa_metadata_v1. It supports FP8/BF16/FP16
    with optional per-token or per-tensor quantization, causal masking, and optional
    sliding-window attention.

    Args:
        work_indptr/work_info: Worklist metadata (prefix + entries) from
            get_pa_metadata_v1 for persistent scheduling.
        logits_ptr/split_lse_ptr: Partial outputs and log-sum-exp buffers for
            reduce stage.
        output_ptr/query_ptr: [sum_qlen, num_query_heads, head_size].
        key_cache_ptr/value_cache_ptr: KV cache in block layout.
        qo_indptr/kv_indptr/kv_page_indices: Paged cache indices and prefix sums.
        context_lengths_ptr: Valid KV lengths per sequence.
        softmax_scale: Scaling factor for attention logits.
        query_scale/key_scale/value_scale: Quantization scales (optional).
        stride_*: Tensor strides for runtime layout.
        max_qlen/query_group_size/head_size: Runtime sizes for masking/layout.
        *constexpr args: Kernel configuration (block size, quant modes, etc.).

    Note:
        Uses AMD CDNA MFMA instructions; supported on gfx942/gfx950.
    """
    # ==================== VALIDATION CHECKS ====================
    # gl.static_assert(
    #     KV_BLOCK_SIZE == 16 or KV_BLOCK_SIZE == 64 or KV_BLOCK_SIZE == 1024,
    #     f"KV_BLOCK_SIZE={KV_BLOCK_SIZE}, Only support KV_BLOCK_SIZE in [16, 64, 1024]",
    # )
    # Data type validation
    gl.static_assert(
        query_ptr.dtype.is_fp8()
        or query_ptr.dtype.element_ty == gl.bfloat16
        or query_ptr.dtype.element_ty == gl.float16
    )
    gl.static_assert(
        key_cache_ptr.dtype.is_fp8()
        or key_cache_ptr.dtype.element_ty == gl.bfloat16
        or key_cache_ptr.dtype.element_ty == gl.float16
    )
    gl.static_assert(
        value_cache_ptr.dtype.is_fp8()
        or value_cache_ptr.dtype.element_ty == gl.bfloat16
        or value_cache_ptr.dtype.element_ty == gl.float16
    )

    if QUERY_QUANT_MODE >= 0:
        gl.static_assert(query_scale.dtype.element_ty == gl.float32)
    if KV_QUANT_MODE >= 0:
        gl.static_assert(key_scale.dtype.element_ty == gl.float32)
        gl.static_assert(value_scale.dtype.element_ty == gl.float32)

    # ==================== CONSTANTS AND CONFIGURATION ====================
    if COMPUTE_TYPE.is_fp8():
        MFMA_INSTR_K: gl.constexpr = 32
    else:
        MFMA_INSTR_K: gl.constexpr = 16
    if TRITON_VERSION_GE_3_6_0:
        QK_PV_MFMA_INSTR_SHAPE: gl.constexpr = [16, 16, MFMA_INSTR_K]
    else:
        QK_PV_MFMA_INSTR_SHAPE: gl.constexpr = [16, 16]

    if KV_QUANT_MODE >= 0:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 16
    else:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 8

    if COMPUTE_TYPE.is_fp8():
        OUTPUT_DTYPE: gl.constexpr = tl.bfloat16
    else:
        OUTPUT_DTYPE: gl.constexpr = COMPUTE_TYPE

    LOG2_E: gl.constexpr = 1.4426950408889634  # log2(e) for exponential conversion

    K_HEAD_SIZE_SPLITS: gl.constexpr = HEAD_SIZE_POW2 // KV_16B_ELEMENT_COUNT

    # ==================== MEMORY LAYOUT DEFINITIONS ====================
    # Query tensor layout - optimized for sequential access (2D)
    blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    QUERY_GROUP_SIZE_POW2: gl.constexpr = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2
    # MTP Query tensor layout (3D) [QUERY_SEQ_LEN_POW2, QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    if ONE_QUERY_GROUP_SIZE_POW2 <= 16:
        # QUERY_GROUP_SIZE_POW2 may be 4, 8, 16
        # corresponding Q_WARPS_PER_CTA_DIM1 should be 1, 2, 4
        # corresponding Q_WARPS_PER_CTA_DIM0 should be 4, 2, 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = ONE_QUERY_GROUP_SIZE_POW2 // 4
        Q_WARPS_PER_CTA_DIM0: gl.constexpr = 4 // Q_WARPS_PER_CTA_DIM1
    else:
        Q_WARPS_PER_CTA_DIM0: gl.constexpr = 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = 4
    mtp_blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 8],
        threads_per_warp=[1, 4, 16],
        warps_per_cta=[Q_WARPS_PER_CTA_DIM0, Q_WARPS_PER_CTA_DIM1, 1],
        order=[2, 1, 0],
    )

    # Key cache layout - optimized for block-wise access patterns
    key_warps_per_cta: gl.constexpr = (
        [4, 1, 1, 1] if KV_BLOCK_SIZE == 16 else [1, 1, 4, 1]
    )
    blocked_key_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=key_warps_per_cta,
        order=[3, 2, 1, 0],
    )

    # QK Matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[1, 4],
    )
    qk_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )
    qk_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )

    # Register allocation configuration based on group size and compute block size

    if QUERY_GROUP_SIZE_POW2 == 16:
        if KV_BLOCK_SIZE == 64:
            register_bases: gl.constexpr = ((0, 1), (0, 2))
        elif KV_BLOCK_SIZE == 256:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (0, 128))
    elif QUERY_GROUP_SIZE_POW2 == 32:
        if KV_BLOCK_SIZE == 64:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64))
        elif KV_BLOCK_SIZE == 256:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (0, 128), (16, 0))
    elif QUERY_GROUP_SIZE_POW2 == 64:
        if KV_BLOCK_SIZE == 64:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (16, 0))
        elif KV_BLOCK_SIZE == 256:
            register_bases: gl.constexpr = (
                (0, 1),
                (0, 2),
                (0, 64),
                (0, 128),
                (16, 0),
                (32, 0),
            )

    # Distributed layout for QK linear operations
    qk_linear_layout: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=register_bases,
        lane_bases=((1, 0), (2, 0), (4, 0), (8, 0), (0, 4), (0, 8)),
        warp_bases=((0, 16), (0, 32)),
        block_bases=[],
        shape=[QUERY_GROUP_SIZE_POW2, KV_BLOCK_SIZE],
    )

    # Value cache layout configuration based on transpose flag
    if VALUE_TRANSPOSED:
        # Transposed value layout for better memory access patterns
        value_threads_per_warp: gl.constexpr = (
            [4, 1, 16, 1] if KV_BLOCK_SIZE == 16 else [1, 4, 16, 1]
        )
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )

        value_dim1_offsets = gl.arange(
            0,
            KV_BLOCK_SIZE // KV_16B_ELEMENT_COUNT,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
            ),
        )
        value_dim2_offsets = gl.arange(
            0,
            HEAD_SIZE_POW2,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_value_layout))
            ),
        )
        value_dim3_offsets = gl.arange(
            0,
            KV_16B_ELEMENT_COUNT,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout))
            ),
        )
    else:
        # Standard value layout
        value_threads_per_warp: gl.constexpr = (
            [4, 16, 1] if KV_BLOCK_SIZE == 16 else [1, 16, 4]
        )
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 16],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 4, 1],
            order=[2, 1, 0],
        )

        value_dim1_offsets = gl.arange(
            0,
            HEAD_SIZE_POW2,
            layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_value_layout)),
        )
        value_dim2_offsets = gl.arange(
            0,
            KV_BLOCK_SIZE,
            layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_value_layout)),
        )

    # PV Matrix multiplication layout using AMD MFMA instructions
    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[1, 4],
    )
    pv_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=16
    )
    pv_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=16
    )

    # ==================== LAYOUT SLICE DEFINITIONS ====================

    # MTP Query layout slices (for 3D layout)
    mtp_query_len_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, mtp_blocked_query_layout)
    )
    mtp_query_group_size_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(2, mtp_blocked_query_layout)
    )
    mtp_head_size_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, mtp_blocked_query_layout)
    )

    head_size_split_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(2, gl.SliceLayout(3, blocked_key_layout))
    )
    block_element_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, gl.SliceLayout(3, blocked_key_layout))
    )
    contiguous_kv_elements_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_key_layout))
    )

    # Coordinate offsets for various dimensions
    # MTP offsets (for 3D layout)
    mtp_query_len_offsets = gl.arange(
        0, QUERY_SEQ_LEN_POW2, layout=mtp_query_len_layout
    )
    mtp_query_group_size_offsets = gl.arange(
        0, ONE_QUERY_GROUP_SIZE_POW2, layout=mtp_query_group_size_layout
    )
    mtp_head_size_offsets = gl.arange(0, HEAD_SIZE_POW2, layout=mtp_head_size_layout)

    head_size_split_offsets = gl.arange(
        0, K_HEAD_SIZE_SPLITS, layout=head_size_split_layout
    )
    block_element_offsets = gl.arange(0, KV_BLOCK_SIZE, layout=block_element_layout)
    kv_scale_offsets = gl.arange(
        0, KV_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
    )
    contiguous_kv_element_offsets = gl.arange(
        0, KV_16B_ELEMENT_COUNT, layout=contiguous_kv_elements_layout
    )
    qk_row_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    logits_query_group_size_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    logits_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )

    # ==================== PROGRAM ID AND INITIALIZATION ====================
    wg_idx = gl.program_id(0)
    work_start_idx = gl.load(work_indptr + wg_idx)
    work_end_idx = gl.load(work_indptr + wg_idx + 1)
    if QUERY_QUANT_MODE == 0:
        # Per-tensor quantization
        query_scale_value = tl.load(query_scale)
    
    if KV_QUANT_MODE == 0:
        # Per-tensor quantization
        key_scale_value = tl.load(key_scale)
        value_scale_value = tl.load(value_scale)

    for i in range(work_start_idx, work_end_idx):
        work_info_item = work_info + i * 8
        sequence_idx = gl.load(work_info_item)
        logits_idx = gl.load(work_info_item + 1)
        qo_start = gl.load(work_info_item + 2)
        qo_end = gl.load(work_info_item + 3)
        kv_block_start_idx = gl.load(work_info_item + 4)
        kv_block_end_idx = gl.load(work_info_item + 5)
        causal_offset = gl.load(work_info_item + 6)
        q_head_range = gl.load(work_info_item + 7)
        q_head_start = q_head_range & 0xFFFF
        q_head_end = (q_head_range >> 16) & 0xFFFF

        context_length = gl.load(context_lengths_ptr + sequence_idx)
        kv_start_idx = gl.load(kv_indptr + sequence_idx)
        max_logits = gl.full(
            (QUERY_GROUP_SIZE_POW2,),
            float("-inf"),
            dtype=gl.float32,
            layout=gl.SliceLayout(1, qk_linear_layout),
        )
        exp_sums = gl.full(
            (QUERY_GROUP_SIZE_POW2,),
            0.0,
            dtype=gl.float32,
            layout=gl.SliceLayout(1, qk_linear_layout),
        )
        attention_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2),
            dtype=gl.float32,
            layout=pv_mfma_layout,
        )

        mtp_query_offsets = (
            (qo_start + mtp_query_len_offsets[:, None, None]) * stride_query_0
            + (q_head_start + mtp_query_group_size_offsets[None, :, None])
            * stride_query_1
            + mtp_head_size_offsets[None, None, :]
        )
        query_row_mask_3d = (
            qo_start + mtp_query_len_offsets[:, None, None] < qo_end
        ) & ((q_head_start + mtp_query_group_size_offsets[None, :, None]) < q_head_end)

        mtp_query_mask = query_row_mask_3d & (
            mtp_head_size_offsets[None, None, :] < head_size
        )
        query_row_mask_1d = gl.reshape(query_row_mask_3d, [QUERY_GROUP_SIZE_POW2])
        qk_row_mask = gl.convert_layout(
            query_row_mask_1d, layout=gl.SliceLayout(1, qk_linear_layout)
        )
        mtp_query_tensor = gl.amd.cdna3.buffer_load(
            ptr=query_ptr,
            offsets=mtp_query_offsets,
            mask=mtp_query_mask,
        )
        mtp_query_tensor = gl.reshape(
            mtp_query_tensor,
            [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2],
        )

        query_tensor = gl.convert_layout(mtp_query_tensor, layout=blocked_query_layout)
        query_shared = gl.allocate_shared_memory(
            query_tensor.dtype, query_tensor.shape, shared_query_layout, query_tensor
        )
        query_converted = query_shared.load(qk_lhs_operand_layout)

        if QUERY_QUANT_MODE < 0 and COMPUTE_TYPE.is_fp8():
            # Quantize bf16 query to fp8
            # Convert query to float32 for computation
            query_f32 = query_converted.to(gl.float32)
            # Compute max absolute value for scaling
            query_abs = gl.abs(query_f32)
            query_max_abs = gl.max(query_abs, axis=1, keep_dims=True)
            # Compute scale factor: FP8_MAX_VALUE / max_abs_value
            # Add epsilon to avoid division by zero
            query_scale_value = query_max_abs / float(FP8_MAX_VALUE)
            # Quantize: scale query to fp8 range and convert to fp8 type
            query_converted = query_f32.to(COMPUTE_TYPE)
        else:
            query_converted = query_converted.to(COMPUTE_TYPE)

        kv_head_idx = q_head_start // query_group_size
        if SLIDING_WINDOW > 0:
            sequence_start_idx = (
                kv_start_idx * KV_BLOCK_SIZE + context_length - SLIDING_WINDOW
            )
            sequence_end_idx = kv_start_idx * KV_BLOCK_SIZE + context_length
        else:
            sequence_start_idx = kv_start_idx * KV_BLOCK_SIZE
            sequence_end_idx = kv_start_idx * KV_BLOCK_SIZE + context_length
        if QUERY_QUANT_MODE == 1:
            # Per-token quantization
            query_scale_offsets = (
                (qo_start + mtp_query_len_offsets[:, None, None]) * stride_query_scale_0
                + kv_head_idx * stride_query_scale_1
                + q_head_start
                + mtp_query_group_size_offsets[None, :, None]
            )
            query_scale_value = gl.amd.cdna3.buffer_load(
                ptr=query_scale,
                offsets=query_scale_offsets,
                mask=query_row_mask_3d,
            )
            query_scale_value = gl.reshape(
                query_scale_value, [QUERY_GROUP_SIZE_POW2, 1]
            )
            query_scale_value = gl.convert_layout(
                query_scale_value, layout=qk_linear_layout
            )
        for kv_block_idx in range(kv_block_start_idx, kv_block_end_idx):
            qk_column_offsets = kv_block_idx * KV_BLOCK_SIZE + kv_scale_offsets

            # Create mask for valid blocks
            kv_block_numbers = gl.load(
                kv_page_indices + kv_block_idx,
            ).to(gl.int64)

            # ==================== KEY LOADING AND PROCESSING ====================
            # Calculate key cache offsets and load keys
            key_block_offsets = (
                kv_block_numbers * stride_key_0
                + kv_head_idx * stride_key_1
                + head_size_split_offsets[None, :, None, None] * stride_key_2
                # Use runtime stride for KV block element (may be padded for large blocks).
                + block_element_offsets[None, None, :, None] * stride_key_3
                + contiguous_kv_element_offsets[None, None, None, :]
            )

            # Optimize: Start key load, then prepare QK MFMA accumulators/query (overlaps with key load)
            if KV_BLOCK_SIZE == 1024 and SLIDING_WINDOW > 0:
                kv_token_global = kv_block_idx * KV_BLOCK_SIZE + block_element_offsets
                kv_in_window_mask = kv_token_global >= sequence_start_idx
                key_tensor = gl.load(
                    key_cache_ptr + key_block_offsets,
                    mask=kv_in_window_mask[None, None, :, None],
                    other=0.0,
                )
            else:
                key_tensor = gl.load(key_cache_ptr + key_block_offsets)
            # Prepare QK MFMA while key loads (these don't depend on key data)
            qk_accumulator = gl.zeros(
                (QUERY_GROUP_SIZE_POW2, KV_BLOCK_SIZE),
                dtype=gl.float32,
                layout=qk_mfma_layout,
            )
            # Load key quantization scales if needed (overlaps with key tensor load)
            if KV_QUANT_MODE == 1:
                # Per-token quantization - prepare offsets while key loads
                key_scale_offsets = (
                    kv_block_numbers * stride_kv_scale_0
                    + kv_head_idx * stride_kv_scale_1
                    + kv_scale_offsets
                )
                # Optimize: Load both scales with VMEM scheduling, overlap with key reshape
                if KV_BLOCK_SIZE == 1024 and SLIDING_WINDOW > 0:
                    key_scale_value = gl.load(
                        key_scale + key_scale_offsets,
                        mask=kv_in_window_mask[None, None, :, None],
                        other=0.0,
                    )
                    value_scale_value = gl.load(
                        value_scale + key_scale_offsets,
                        mask=kv_in_window_mask[None, None, :, None],
                        other=0.0,
                    )
                else:
                    key_scale_value = gl.load(key_scale + key_scale_offsets)
                    value_scale_value = gl.load(value_scale + key_scale_offsets)

                key_scale_value = key_scale_value[None, :]

            # Reshape key tensor for matrix multiplication
            key_tensor = gl.permute(key_tensor, [1, 3, 0, 2])
            key_tensor = gl.reshape(key_tensor, [HEAD_SIZE_POW2, KV_BLOCK_SIZE])

            # ==================== VALUE LOADING WITH QK MFMA OVERLAP ====================
            # Convert key layout for MFMA (query_converted and qk_accumulator already prepared above)
            key_converted = gl.convert_layout(key_tensor, layout=qk_rhs_operand_layout)
            key_converted = key_converted.to(COMPUTE_TYPE)

            if VALUE_TRANSPOSED:
                # Load values from transposed cache layout

                value_block_offsets = (
                    kv_block_numbers * stride_value_0
                    + kv_head_idx * stride_value_1
                    + value_dim1_offsets[None, :, None, None] * stride_value_2
                    + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                    + value_dim3_offsets[None, None, None, :]
                )
                if KV_BLOCK_SIZE == 1024 and SLIDING_WINDOW > 0:
                    value_tensor = gl.load(
                        value_cache_ptr + value_block_offsets,
                        mask=kv_in_window_mask[None, None, :, None],
                        other=0.0,
                    )
                else:
                    value_tensor = gl.load(value_cache_ptr + value_block_offsets)
                # Compute QK attention scores using MFMA (overlaps with value load)
                attention_scores = gl.amd.cdna3.mfma(
                    query_converted, key_converted, qk_accumulator
                )

                # Permute and reshape for matrix multiplication
                value_tensor = gl.permute(value_tensor, [0, 1, 3, 2])
            else:
                # Load values from standard cache layout

                value_block_offsets = (
                    kv_block_numbers * stride_value_0
                    + kv_head_idx * stride_value_1
                    + value_dim1_offsets[None, :, None] * stride_value_2
                    + value_dim2_offsets[None, None, :]
                )

                # Schedule: Start value VMEM load, then QK MFMA
                if KV_BLOCK_SIZE == 1024 and SLIDING_WINDOW > 0:
                    value_token_global = (
                        kv_block_idx * KV_BLOCK_SIZE + value_dim2_offsets
                    )
                    value_in_window_mask = value_token_global >= sequence_start_idx
                    value_tensor = gl.load(
                        value_cache_ptr + value_block_offsets,
                        mask=value_in_window_mask[None, None, :],
                        other=0.0,
                    )
                else:
                    value_tensor = gl.load(value_cache_ptr + value_block_offsets)
                # Compute QK attention scores using MFMA (overlaps with value load)
                attention_scores = gl.amd.cdna3.mfma(
                    query_converted, key_converted, qk_accumulator
                )

                # Permute and resape for matrix multiplication
                value_tensor = gl.permute(value_tensor, [0, 2, 1])

            value_tensor = gl.reshape(value_tensor, [KV_BLOCK_SIZE, HEAD_SIZE_POW2])

            attention_scores = gl.reshape(
                attention_scores, [QUERY_GROUP_SIZE_POW2, KV_BLOCK_SIZE]
            )

            # Apply quantization scaling to attention scores
            if KV_QUANT_MODE >= 0:
                if QUERY_QUANT_MODE >= 0:
                    qk_scale_value = softmax_scale * query_scale_value * key_scale_value
                else:
                    qk_scale_value = softmax_scale * key_scale_value
            else:
                if QUERY_QUANT_MODE >= 0:
                    qk_scale_value = softmax_scale * query_scale_value
                else:
                    qk_scale_value = softmax_scale

            attention_scores = qk_scale_value * attention_scores
            # ==================== ATTENTION MASKING ====================
            query_token_idx = qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2

            # Apply causal masking if required
            if IS_CAUSAL:
                # Compute causal mask based on sequence positions
                causal_mask = (
                    causal_offset[:, None] + qk_column_offsets[None, :]
                    < sequence_end_idx
                )
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        causal_offset[:, None] + qk_column_offsets[None, :]
                        >= sequence_start_idx + query_token_idx[:, None] + 1
                    )
                else:
                    causal_mask = causal_mask & (
                        causal_offset[:, None] + qk_column_offsets[None, :]
                        >= sequence_start_idx
                    )
            else:
                causal_mask = qk_column_offsets[None, :] < sequence_end_idx
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        qk_column_offsets[None, :]
                        >= sequence_start_idx + query_token_idx[:, None] + 1
                    )
                else:
                    causal_mask = causal_mask & (
                        qk_column_offsets[None, :] >= sequence_start_idx
                    )

            boundary_mask = qk_row_mask[:, None] & causal_mask
            # Apply masking to attention scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)
            attention_scores = gl.where(boundary_mask, attention_scores, float(-3.4e38))

            # ==================== SOFTMAX COMPUTATION ====================
            # Update running maximum for numerical stability
            current_max_logits = gl.max(attention_scores, axis=1)
            new_max_logits = gl.maximum(max_logits, current_max_logits)
            accumulator_scale = tl.math.exp2((max_logits - new_max_logits) * LOG2_E)
            # Compute attention probabilities
            attention_probs = tl.math.exp2(
                (attention_scores - new_max_logits[:, None]) * LOG2_E
            )

            exp_sums = accumulator_scale * exp_sums + gl.sum(attention_probs, axis=1)
            # ==================== VALUE ACCUMULATION ====================
            # Handle value quantization scaling for FP8
            if KV_QUANT_MODE >= 0:
                if KV_QUANT_MODE == 1:
                    # Per-token quantization scaling
                    # Create mask for valid tokens
                    valid_token_mask = qk_column_offsets < sequence_end_idx
                    # Mask out value_scale of invalid tokens
                    value_scale_value = gl.where(
                        valid_token_mask, value_scale_value, float(0.0)
                    )
                    value_scale_max = gl.max(value_scale_value, axis=0)
                    # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                    value_scale_value = (
                        value_scale_value
                        * float(FP8_MAX_VALUE)
                        / (value_scale_max + 1e-8)
                    )
                    attention_probs = value_scale_value[None, :] * attention_probs
                    probability_scale = value_scale_max / float(FP8_MAX_VALUE)
                elif KV_QUANT_MODE == 0:
                    # Per-tensor quantization scaling
                    probability_scale = value_scale_value
                else:
                    raise ValueError(f"Invalid KV_QUANT_MODE: {KV_QUANT_MODE}")

            # Convert attention probabilities to compute type for MFMA operation
            # Convert layouts for PV MFMA operation
            attention_probs = attention_probs.to(COMPUTE_TYPE)
            probs_converted = gl.convert_layout(
                attention_probs, layout=pv_lhs_operand_layout
            )

            values_converted = gl.convert_layout(
                value_tensor, layout=pv_rhs_operand_layout
            )
            values_converted = values_converted.to(COMPUTE_TYPE)

            accumulator_scale_expanded = gl.convert_layout(
                accumulator_scale[:, None], layout=pv_mfma_layout
            )
            attention_accumulator *= accumulator_scale_expanded

            pv_accumulator = gl.zeros(
                (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2),
                dtype=gl.float32,
                layout=pv_mfma_layout,
            )
            attention_output = gl.amd.cdna3.mfma(
                probs_converted, values_converted, pv_accumulator
            )

            if KV_QUANT_MODE >= 0:
                attention_accumulator += probability_scale * attention_output
            else:
                attention_accumulator += attention_output

            max_logits = new_max_logits

        # ==================== OUTPUT NORMALIZATION AND STORING ====================

        # Normalize attention output by softmax denominator
        exp_sums_reciprocal = 1.0 / exp_sums
        exp_sums_reciprocal_cvt = gl.convert_layout(
            exp_sums_reciprocal[:, None], layout=pv_mfma_layout
        )
        attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt
        # Store results to global memory
        if logits_idx >= 0:
            # Compute log-sum-exp for reduce stage
            logits_offsets = (
                logits_idx * stride_logits_0
                + (q_head_start + logits_query_group_size_offsets[:, None])
                * stride_logits_2
                + logits_head_size_offsets[None, :]
            )
            logits_mask = (
                (q_head_start + logits_query_group_size_offsets[:, None]) < q_head_end
            ) & (logits_head_size_offsets[None, :] < head_size)
            split_lse_offsets = logits_idx * stride_lse_0 + (
                q_head_start + qk_row_offsets
            )
            split_lse_mask = (q_head_start + qk_row_offsets) < q_head_end

            lse = tl.math.log2(exp_sums) / LOG2_E + max_logits

            gl.amd.cdna3.buffer_store(
                stored_value=attention_accumulator,
                ptr=logits_ptr,
                offsets=logits_offsets,
                mask=logits_mask,
            )
            gl.amd.cdna3.buffer_store(
                stored_value=lse,
                ptr=split_lse_ptr,
                offsets=split_lse_offsets,
                mask=split_lse_mask,
            )

        else:
            output = gl.reshape(
                attention_accumulator,
                [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2],
            )
            output = gl.convert_layout(output, layout=mtp_blocked_query_layout)

            gl.amd.cdna3.buffer_store(
                stored_value=output.to(OUTPUT_DTYPE),
                ptr=output_ptr,
                offsets=mtp_query_offsets,
                mask=mtp_query_mask,
            )


def pa_ps_gluon(
    output: torch.Tensor,  # [sum_qlen, num_query_heads, head_size]
    query: torch.Tensor,  # [sum_qlen, num_query_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    qo_indptr: torch.Tensor,  # [num_seqs + 1], qo len prefix sum
    kv_indptr: torch.Tensor,  # [num_seqs + 1], kvlen prefix sum
    kv_indices: torch.Tensor,  # [num_seqs], packed kv ids
    context_lengths: torch.Tensor,  # [num_seqs]
    softmax_scale: float,
    max_qlen: int,
    compute_type: torch.dtype,
    work_indptr: torch.Tensor,
    work_info: torch.Tensor,
    reduce_indptr: torch.Tensor,
    reduce_final_map: torch.Tensor,
    reduce_partial_map: torch.Tensor,
    query_scale: torch.Tensor = None,  # [sum_qlen, num_query_heads, 1] or [1]
    key_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    logits: torch.Tensor = None,
    split_lse: torch.Tensor = None,
    final_lse: torch.Tensor = None,
    alibi_slopes: torch.Tensor = None,
    sinks: torch.Tensor = None,  # [num_query_heads]
    sliding_window: int = 0,
):
    """Paged-attention persistent scheduling via Gluon (JIT).

    Args:
        output: [sum_qlen, num_query_heads, head_size] output buffer.
        query: [sum_qlen, num_query_heads, head_size] query tensor.
        key_cache: [num_blocks, num_kv_heads, head_size//x, kv_block_size, x] KV cache.
        value_cache: [num_blocks, num_kv_heads, head_size, kv_block_size] KV cache.
        qo_indptr: [num_seqs + 1] prefix sum of query lengths.
        kv_indptr: [num_seqs + 1] prefix sum of KV blocks per sequence.
        kv_indices: [sum_kv_blocks] packed KV block indices.
        context_lengths: [num_seqs] valid KV lengths per sequence.
        softmax_scale: scale factor for attention logits.
        max_qlen: maximum query length per sequence (<= 4).
        compute_type: compute dtype (fp8/bf16/fp16).
        work_indptr/work_info/reduce_*: metadata buffers from get_pa_metadata_v1.
        query_scale/key_scale/value_scale: optional quantization scales.
        logits/split_lse/final_lse: optional intermediate buffers for PS reduce.
        alibi_slopes: optional ALiBi slopes.
        sinks: optional sinks tensor for attention sinks.
        sliding_window: window size for sliding-window attention.

    Notes:
        - Only supported on gfx942/gfx950.
        - kv_block_size must be one of [16, 64, 1024].
    """
    if not GLUON_JIT_KERNEL_ENABLED:
        raise RuntimeError(
            "This version triton is not support gluon jit mode, please upgrade to 3.5.0 or higher!"
        )
    cdna_version = get_cdna_version()
    assert cdna_version in [
        3,
        4,
    ], f"pa_decode_gluon only supports gfx942 (CDNA3) and gfx950 (CDNA4) now, but got {arch_info.get_arch()}"
    # Extract tensor dimensions from input tensors
    # context_partition_size = 256
    # if sliding_window > 0:
    # context_partition_size = 1024
    num_query_heads = query.shape[1]
    head_size = query.shape[-1]

    num_kv_heads = key_cache.shape[1]
    query_group_size = num_query_heads // num_kv_heads
    # Calculate equivalent group sizes for kernel configuration
    equivalent_query_group_size = max_qlen * query_group_size
    kv_block_size = key_cache.shape[-2]

    # Determine if causal masking is needed
    is_causal = max_qlen > 1
    props = torch.cuda.get_device_properties()
    num_sm = props.multi_processor_count

    grid = (num_sm, 1, 1)
    assert max_qlen <= 4, f"max_qlen == {max_qlen} exceeds maximum of 4"
    # Validate input params constraint
    assert query.dtype in [
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"query tensor only support dtype in [{aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16}], but got query.dtype == {query.dtype}"
    assert key_cache.dtype in [
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"key_cache tensor only support dtype in [{aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16}], but got key_cache.dtype == {key_cache.dtype}"
    assert value_cache.dtype in [
        aiter.dtypes.fp8,
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"value_cache tensor only support dtype in [{aiter.dtypes.fp8, aiter.dtypes.bf16, aiter.dtypes.fp16}], but got value_cache.dtype == {value_cache.dtype}"
    assert output.dtype in [
        aiter.dtypes.bf16,
        aiter.dtypes.fp16,
    ], f"output tensor only support dtype in [{aiter.dtypes.bf16, aiter.dtypes.fp16}], but got output.dtype == {output.dtype}"
    assert (
        equivalent_query_group_size <= 64
    ), f"equivalent_query_group_size={equivalent_query_group_size} exceeds maximum of 64"
    # assert kv_block_size in [
    #     16,
    #     64,
    #     1024,
    # ], f"kv_block_size == {kv_block_size} not in [16, 64, 1024]"
    assert (
        len(output.shape) == 3
    ), f"Expected 3D output tensor, but got shape {output.shape}"
    assert (
        len(query.shape) == 3
    ), f"Expected 3D query tensor, but got shape {query.shape}"
    assert (
        len(key_cache.shape) == 5
    ), f"Expected 5D key_cache tensor, but got shape {key_cache.shape}"
    if logits is None:
        logits = torch.empty(
            (reduce_partial_map.size(0) * max_qlen, 1, num_query_heads, head_size),
            dtype=aiter.dtypes.fp32,
            device=query.device,
        )
    if split_lse is None:
        split_lse = torch.empty(
            (reduce_partial_map.size(0) * max_qlen, 1, num_query_heads, 1),
            dtype=aiter.dtypes.fp32,
            device=query.device,
        )
    fp8_max_value = 1.0
    if value_cache.dtype == aiter.dtypes.fp8:
        fp8_max_value = torch.finfo(aiter.dtypes.fp8).max

    stride_query_scale_0 = 0
    stride_query_scale_1 = 0
    key_scale_stride_0 = 0
    key_scale_stride_1 = 0
    query_quant_mode = -1
    kv_quant_mode = -1

    if query_scale is not None:
        if query_scale.numel() == 1:
            # Per-tensor quantization
            query_quant_mode = 0
        else:
            # Per-token quantization
            assert (
                len(query_scale.shape) == 3
            ), f"Expected 3D query_scale tensor, but got shape {query_scale.shape}"
            assert (
                query_scale.shape[-1] == 1
            ), f"Expected query_scale.shape[-1] == 1, but got query_scale.shape[-1]={query_scale.shape[-1]}"
            query_quant_mode = 1
            stride_query_scale_0 = query_scale.stride(0)
            stride_query_scale_1 = query_scale.stride(1)

    if key_scale is not None and value_scale is not None:
        if key_scale.numel() == 1:
            # Per-tensor quantization
            kv_quant_mode = 0
        else:
            # Per-token quantization
            assert (
                len(key_scale.shape) == 4
            ), f"Expected 4D key_scale tensor, but got shape {key_scale.shape}"
            assert (
                key_scale.shape[-1] == 1
            ), f"Expected key_scale.shape[-1] == 1, but got key_scale.shape[-1]={key_scale.shape[-1]}"
            kv_quant_mode = 1
            key_scale_stride_0 = key_scale.stride(0)
            key_scale_stride_1 = key_scale.stride(1)
    if value_cache.ndim == 5:
        value_transposed = True
    else:
        value_transposed = False
    pa_ps_kernel[grid](
        work_indptr,
        work_info,
        logits,
        split_lse,
        output,
        query,
        key_cache,
        value_cache,
        qo_indptr,
        kv_indptr,
        kv_indices,
        context_lengths,
        softmax_scale,
        query_scale,
        key_scale,
        value_scale,
        logits.stride(0),
        logits.stride(1),
        logits.stride(2),
        split_lse.stride(0),
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        stride_query_scale_0,
        stride_query_scale_1,
        key_scale_stride_0,
        key_scale_stride_1,
        max_qlen=max_qlen,
        query_group_size=query_group_size,
        head_size=head_size,
        COMPUTE_TYPE=torch_to_triton_dtype[compute_type],
        QUERY_SEQ_LEN_POW2=triton.next_power_of_2(max_qlen),
        ONE_QUERY_GROUP_SIZE_POW2=max(16, triton.next_power_of_2(query_group_size)),
        HEAD_SIZE_POW2=triton.next_power_of_2(head_size),
        KV_BLOCK_SIZE=kv_block_size,
        QUERY_QUANT_MODE=query_quant_mode,
        KV_QUANT_MODE=kv_quant_mode,
        VALUE_TRANSPOSED=value_transposed,
        IS_CAUSAL=is_causal,
        FP8_MAX_VALUE=fp8_max_value,
        SLIDING_WINDOW=sliding_window,
        CDNA_VERSION=cdna_version,
    )

    pa_reduce_v1(
        logits,
        split_lse,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_qlen,
        output,
        final_lse,
    )
    return logits, final_lse
