# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import math
from typing import Optional, Dict, Tuple
import tempfile
import subprocess
import os
import sys

import triton
import triton.language as tl
from triton.compiler.compiler import compile
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
import torch
import aiter
from aiter.test_common import perftest


# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

_SEQUENCE_PARTITION_SIZE = 256


@gluon.jit
def paged_attention_decode_v2_gluon_large_block_fp8(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [num_seqs, num_kv_heads * query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size/x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    sequence_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, num_kv_heads * query_group_size, 1]
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    alibi_slopes_ptr,
    stride_max_logits_seq,
    stride_max_logits_head,
    stride_max_logits_part,
    stride_output_seq,
    stride_output_head,
    stride_output_part,
    stride_output_group,
    stride_query_seq,
    stride_query_head,
    stride_key_block,
    stride_key_head,
    stride_key_head_split,
    stride_key_block_elem,
    stride_value_block,
    stride_value_head,
    stride_value_head_size,
    stride_block_table_seq,
    query_scale_stride_0,
    kv_scale_stride_0,
    kv_scale_stride_1,
    QUERY_SEQ_LEN: tl.constexpr,
    COMPUTE_TYPE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_POW2: tl.constexpr,
    QUERY_GROUP_SIZE_ORIGINAL: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE_POW2: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE_POW2: tl.constexpr,
    SEQUENCE_PARTITION_SIZE: tl.constexpr,
    KV_16B_ELEMENT_COUNT: tl.constexpr,
    QUERY_QUANT_MODE: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
    KV_COMPUTE_BLOCK_SIZE: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    VALUE_TRANSPOSED: tl.constexpr,  # [num_blocks, num_kv_heads, kv_block_size/x, head_size, x]
    IS_CAUSAL: tl.constexpr,
):
    """
    Gluon-based paged attention decode kernel with FP8 support for large blocks.

    This kernel implements efficient attention computation for decoding scenarios with:
    - Paged key-value caches for handling long sequences
    - FP8 quantization support for both queries and key-value pairs
    - Blocked computation for memory efficiency
    - Support for ALiBi attention biases
    - Causal masking for autoregressive generation

    The kernel processes sequences in partitions and computes attention scores
    using matrix multiplication operations optimized for AMD CDNA3 architecture.

    Args:
        Various pointers to tensors and configuration parameters as described above.
    """

    # ==================== Validation Checks ====================
    gl.static_assert(
        QUERY_SEQ_LEN <= 4,
        f"QUERY_SEQ_LEN={QUERY_SEQ_LEN}, Do not support QUERY_SEQ_LEN > 4",
    )
    gl.static_assert(
        QUERY_GROUP_SIZE_POW2 <= 64,
        f"QUERY_GROUP_SIZE_POW2={QUERY_GROUP_SIZE_POW2}, Do not support QUERY_GROUP_SIZE_POW2 > 64",
    )
    gl.static_assert(
        SEQUENCE_PARTITION_SIZE == 256,
        f"SEQUENCE_PARTITION_SIZE={SEQUENCE_PARTITION_SIZE}, Only support SEQUENCE_PARTITION_SIZE == 256",
    )
    gl.static_assert(
        KV_BLOCK_SIZE == 1024,
        f"KV_BLOCK_SIZE={KV_BLOCK_SIZE}, Only support KV_BLOCK_SIZE == 1024",
    )

    # Data type validation
    gl.static_assert(query_ptr.dtype.element_ty == gl.float8e4b8)
    gl.static_assert(key_cache_ptr.dtype.element_ty == gl.float8e4b8)
    gl.static_assert(value_cache_ptr.dtype.element_ty == gl.float8e4b8)

    if QUERY_QUANT_MODE >= 0:
        gl.static_assert(query_scale.dtype.element_ty == gl.float32)
    if KV_QUANT_MODE >= 0:
        gl.static_assert(key_scale.dtype.element_ty == gl.float32)
        gl.static_assert(value_scale.dtype.element_ty == gl.float32)

    # ==================== Constants and Configuration ====================
    LOG2_E: gl.constexpr = 1.4426950408889634  # log2(e) for exponential calculations
    CONTIGUOUS_KV_ELEMENTS_16B_LOAD: gl.constexpr = KV_16B_ELEMENT_COUNT
    KEY_HEAD_SIZE_POW2_SPLIT: gl.constexpr = (
        HEAD_SIZE_POW2 // CONTIGUOUS_KV_ELEMENTS_16B_LOAD
    )

    # ==================== Memory Layout Definitions ====================
    # Query tensor layout - blocked for efficient memory access
    blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    # Key cache layout - optimized for CDNA3 architecture
    blocked_key_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 16],
        threads_per_warp=[4, 16, 1],
        warps_per_cta=[1, 4, 1],
        order=[2, 1, 0],
    )

    # QK matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    qk_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=16
    )
    qk_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=16
    )

    # Register allocation bases for different query group sizes
    if QUERY_GROUP_SIZE_POW2 == 16:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (0, 128))
    elif QUERY_GROUP_SIZE_POW2 == 32:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (16, 0))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (0, 128), (16, 0))
    elif QUERY_GROUP_SIZE_POW2 == 64:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (16, 0), (32, 0))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
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
        shape=[QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE],
    )

    # Value cache layout configuration based on transposition
    if VALUE_TRANSPOSED:
        # Transposed value cache layout
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 16],
            threads_per_warp=[4, 16, 1],
            warps_per_cta=[1, 4, 1],
            order=[2, 1, 0],
        )
        value_dim0_offsets = gl.arange(
            0,
            KV_COMPUTE_BLOCK_SIZE // CONTIGUOUS_KV_ELEMENTS_16B_LOAD,
            layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout)),
        )
        value_dim1_offsets = gl.arange(
            0,
            HEAD_SIZE_POW2,
            layout=gl.SliceLayout(0, gl.SliceLayout(2, blocked_value_layout)),
        )
        value_dim2_offsets = gl.arange(
            0,
            CONTIGUOUS_KV_ELEMENTS_16B_LOAD,
            layout=gl.SliceLayout(0, gl.SliceLayout(1, blocked_value_layout)),
        )
    else:
        # Standard value cache layout
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 16],
            threads_per_warp=[16, 4],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
        value_dim0_offsets = gl.arange(
            0, HEAD_SIZE_POW2, layout=gl.SliceLayout(1, blocked_value_layout)
        )
        value_dim1_offsets = gl.arange(
            0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, blocked_value_layout)
        )

    # PV matrix multiplication layout using AMD MFMA instructions
    pv_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    pv_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=16
    )
    pv_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=16
    )

    # ==================== Dimension Layout Definitions ====================
    # Query dimension layouts
    query_group_size_layout: gl.constexpr = gl.SliceLayout(1, blocked_query_layout)
    head_size_layout: gl.constexpr = gl.SliceLayout(0, blocked_query_layout)

    # Key cache dimension layouts
    head_size_split_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, blocked_key_layout)
    )
    block_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(2, blocked_key_layout)
    )
    contiguous_kv_elements_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, blocked_key_layout)
    )

    # Create offset arrays for various dimensions
    query_group_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=query_group_size_layout
    )
    head_size_offsets = gl.arange(0, HEAD_SIZE_POW2, layout=head_size_layout)

    kv_scale_column_offsets = gl.arange(
        0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
    )

    head_size_split_offsets = gl.arange(
        0, KEY_HEAD_SIZE_POW2_SPLIT, layout=head_size_split_layout
    )
    block_offsets = gl.arange(0, KV_COMPUTE_BLOCK_SIZE, layout=block_layout)
    contiguous_kv_elements_offsets = gl.arange(
        0, CONTIGUOUS_KV_ELEMENTS_16B_LOAD, layout=contiguous_kv_elements_layout
    )

    qk_row_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )

    # ==================== Program ID and Sequence Setup ====================
    sequence_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    sequence_partition_idx = gl.program_id(2)

    # Calculate page offset based on partition index
    page_offset = 0
    if sequence_partition_idx % 4 == 1:
        page_offset = 1 * SEQUENCE_PARTITION_SIZE
    elif sequence_partition_idx % 4 == 2:
        page_offset = 2 * SEQUENCE_PARTITION_SIZE
    elif sequence_partition_idx % 4 == 3:
        page_offset = 3 * SEQUENCE_PARTITION_SIZE

    # ==================== Query Loading ====================
    # Calculate query tensor offsets
    query_offsets_base = (
        sequence_idx * stride_query_seq
        + (kv_head_idx * QUERY_GROUP_SIZE + query_group_offsets[:, None])
        * stride_query_head
        + head_size_offsets[None, :]
    )

    # Create mask for valid query elements
    query_mask = (query_group_offsets[:, None] < QUERY_GROUP_SIZE) & (
        head_size_offsets[None, :] < HEAD_SIZE
    )

    # Load query tensor [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=query_offsets_base, mask=query_mask
    )

    # ==================== Query Quantization Scale Handling ====================
    if QUERY_QUANT_MODE == 0:
        # Per-tensor quantization
        query_scale_value = tl.load(query_scale)
    elif QUERY_QUANT_MODE == 1:
        # Per-token quantization
        query_scale_offsets = (
            sequence_idx * query_scale_stride_0
            + kv_head_idx * QUERY_GROUP_SIZE
            + qk_row_offsets[:, None]
        )
        # [QUERY_GROUP_SIZE_POW2, 1]
        query_scale_value = gl.amd.cdna3.buffer_load(
            ptr=query_scale,
            offsets=query_scale_offsets,
            mask=qk_row_offsets[:, None] < QUERY_GROUP_SIZE,
        )

    # ==================== Output Buffer Setup ====================
    max_logits_base_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    max_logits_offsets = (
        sequence_idx * stride_max_logits_seq
        + kv_head_idx * stride_max_logits_head
        + sequence_partition_idx * stride_max_logits_part
        + max_logits_base_offsets
    )
    max_logits_group_mask = max_logits_base_offsets < QUERY_GROUP_SIZE

    output_group_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )
    output_mask = (output_group_offsets[:, None] < QUERY_GROUP_SIZE) & (
        output_head_size_offsets[None, :] < HEAD_SIZE
    )

    output_offsets = sequence_idx * stride_output_seq
    output_offsets += kv_head_idx * stride_output_head
    output_offsets += (
        sequence_partition_idx * stride_output_part
        + output_group_offsets[:, None] * stride_output_group
        + output_head_size_offsets[None, :]
    )

    # ==================== Attention State Initialization ====================
    # Initialize attention computation state
    current_max_logits = max_logits_base_offsets.to(gl.float32) * float(0.0) - float(
        "inf"
    )
    current_exp_sums = max_logits_base_offsets.to(gl.float32) * float(0.0)
    attention_accumulator = gl.zeros(
        (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=gl.float32, layout=pv_mfma_layout
    )

    # ==================== Sequence Length Handling ====================
    kv_sequence_length = gl.load(sequence_lengths_ptr + sequence_idx)
    kv_sequence_start_index = sequence_partition_idx * SEQUENCE_PARTITION_SIZE

    # Early return if this partition is beyond sequence length
    if kv_sequence_start_index >= kv_sequence_length:
        return

    KV_COMPUTE_BLOCK_COUNT: gl.constexpr = (
        SEQUENCE_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE
    )

    # ==================== Main Attention Computation Loop ====================
    for kv_block_index in range(KV_COMPUTE_BLOCK_COUNT):
        kv_sub_sequence_start_index = (
            kv_sequence_start_index + kv_block_index * KV_COMPUTE_BLOCK_SIZE
        )
        block_table_id = kv_sub_sequence_start_index // KV_BLOCK_SIZE
        current_page_offset = page_offset + kv_block_index * KV_COMPUTE_BLOCK_SIZE

        # Calculate column offsets for QK computation
        qk_column_offsets = kv_sub_sequence_start_index + gl.arange(
            0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
        )

        # ==================== ALiBi Slopes Handling ====================
        if alibi_slopes_ptr is None:
            alibi_slope_values = gl.zeros([QUERY_GROUP_SIZE_POW2], dtype=gl.float32)
        else:
            alibi_slope_values = gl.amd.cdna3.buffer_load(
                ptr=alibi_slopes_ptr + kv_head_idx * QUERY_GROUP_SIZE,
                offsets=qk_row_offsets,
                mask=qk_row_offsets < QUERY_GROUP_SIZE,
            )

        # ==================== Block Table Lookup ====================
        block_tables_start_ptr = (
            block_tables_ptr + sequence_idx * stride_block_table_seq
        )
        kv_page_id = tl.load(block_tables_start_ptr + block_table_id)

        # ==================== Key Cache Loading ====================
        # Calculate key cache block offsets [KEY_HEAD_SIZE_POW2_SPLIT, KV_COMPUTE_BLOCK_SIZE, CONTIGUOUS_KV_ELEMENTS_16B_LOAD]
        key_block_offsets = (
            kv_page_id * stride_key_block
            + kv_head_idx * stride_key_head
            + head_size_split_offsets[:, None, None] * stride_key_head_split
            + (current_page_offset + block_offsets)[None, :, None]
            * CONTIGUOUS_KV_ELEMENTS_16B_LOAD
            + contiguous_kv_elements_offsets[None, None, :]
        )

        # Load key cache block
        key_block = gl.amd.cdna3.buffer_load(
            ptr=key_cache_ptr, offsets=key_block_offsets
        )

        # ==================== Key Quantization Scale Handling ====================
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 0:
                # Per-tensor quantization
                key_scale_value = tl.load(key_scale)
                value_scale_value = tl.load(value_scale)
            elif KV_QUANT_MODE == 1:
                # Per-token quantization
                key_scale_offsets = (
                    kv_page_id * kv_scale_stride_0
                    + kv_head_idx * kv_scale_stride_1
                    + current_page_offset
                    + kv_scale_column_offsets
                )
                key_scale_value = gl.amd.cdna3.buffer_load(
                    ptr=key_scale, offsets=key_scale_offsets
                )
                value_scale_value = gl.amd.cdna3.buffer_load(
                    ptr=value_scale, offsets=key_scale_offsets
                )

        # Reshape key block to [HEAD_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE]
        key_block = gl.permute(key_block, [0, 2, 1])
        key_block = gl.reshape(key_block, [HEAD_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE])

        # ==================== Value Cache Loading ====================
        if VALUE_TRANSPOSED:
            # Calculate offsets for transposed value cache
            value_block_offsets = (
                kv_page_id * stride_value_block
                + kv_head_idx * stride_value_head
                + (
                    current_page_offset // CONTIGUOUS_KV_ELEMENTS_16B_LOAD
                    + value_dim0_offsets
                )[:, None, None]
                * stride_value_head_size
                + value_dim1_offsets[None, :, None] * CONTIGUOUS_KV_ELEMENTS_16B_LOAD
                + value_dim2_offsets[None, None, :]
            )
            # Load transposed value block
            value_block = gl.amd.cdna3.buffer_load(
                ptr=value_cache_ptr, offsets=value_block_offsets
            )
            # Reshape to [KV_COMPUTE_BLOCK_SIZE, HEAD_SIZE_POW2]
            value_block = gl.permute(value_block, [0, 2, 1])
            value_block = gl.reshape(
                value_block, [KV_COMPUTE_BLOCK_SIZE, HEAD_SIZE_POW2]
            )
        else:
            # Calculate offsets for standard value cache [HEAD_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE]
            value_block_offsets = (
                kv_page_id * stride_value_block
                + kv_head_idx * stride_value_head
                + value_dim0_offsets[:, None] * stride_value_head_size
                + (current_page_offset + value_dim1_offsets)[None, :]
            )
            # Load standard value block
            value_block = gl.amd.cdna3.buffer_load(
                ptr=value_cache_ptr, offsets=value_block_offsets
            )
            # Transpose to [KV_COMPUTE_BLOCK_SIZE, HEAD_SIZE_POW2]
            value_block = gl.permute(value_block, [1, 0])

        # ==================== QK Matrix Multiplication ====================
        # Initialize QK accumulator
        qk_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE),
            dtype=gl.float32,
            layout=qk_mfma_layout,
        )

        # Convert layouts for MFMA operation
        query_converted = gl.convert_layout(query_tensor, layout=qk_lhs_layout)
        key_converted = gl.convert_layout(key_block, layout=qk_rhs_layout)

        # Perform matrix multiplication
        qk_matrix = gl.amd.cdna3.mfma(query_converted, key_converted, qk_accumulator)
        qk_matrix = gl.reshape(
            qk_matrix, [QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE]
        )

        # ==================== Scale QK Scores ====================
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # Expand key scale for broadcasting [1, KV_COMPUTE_BLOCK_SIZE]
                key_scale_value = key_scale_value[None, :]
            if QUERY_QUANT_MODE >= 0:
                qk_scale_value = softmax_scale * query_scale_value * key_scale_value
            else:
                qk_scale_value = softmax_scale * key_scale_value
        else:
            if QUERY_QUANT_MODE >= 0:
                qk_scale_value = softmax_scale * query_scale_value
            else:
                qk_scale_value = softmax_scale

        # Apply scaling to QK scores
        qk_matrix = qk_scale_value * qk_matrix

        # ==================== ALiBi Bias Application ====================
        if alibi_slopes_ptr is not None:
            alibi_bias = (
                alibi_slope_values[:, None]
                * (qk_column_offsets - kv_sequence_length + 1)[None, :]
            ).to(gl.float32)
            qk_matrix += alibi_bias

        # ==================== Attention Masking ====================
        # Create boundary mask for valid query groups
        boundary_mask = qk_row_offsets[:, None] < QUERY_GROUP_SIZE

        # Apply causal masking if required
        if IS_CAUSAL:
            sequence_extension = (
                QUERY_SEQ_LEN - 1 - qk_row_offsets // QUERY_GROUP_SIZE_ORIGINAL
            )
            causal_mask = (
                sequence_extension[:, None] + qk_column_offsets[None, :]
                < kv_sequence_length
            )
        else:
            causal_mask = qk_column_offsets[None, :] < kv_sequence_length

        # Combine masks
        combined_mask = boundary_mask & causal_mask

        # Apply masking to QK scores (if [0, SEQUENCE_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -1e38 other than -inf)
        qk_matrix = tl.where(combined_mask, qk_matrix, float(-1e38))

        # ==================== Softmax Computation ====================
        # Compute new maximum logits
        current_max_new = gl.max(qk_matrix, axis=1)
        updated_max_logits = gl.maximum(current_max_logits, current_max_new)

        # Compute scaling factor for numerical stability
        accumulator_scale = tl.math.exp2(
            (current_max_logits - updated_max_logits) * LOG2_E
        )

        # Compute attention probabilities
        attention_probs = tl.math.exp2(
            (qk_matrix - updated_max_logits[:, None]) * LOG2_E
        )
        current_exp_sums = accumulator_scale * current_exp_sums + gl.sum(
            attention_probs, axis=1
        )

        # ==================== Value Scaling for FP8 ====================
        probability_scale = 1.0
        if value_block.dtype.is_fp8():
            if KV_QUANT_MODE == 1:
                # Per-token quantization scaling
                value_scale_max = gl.max(value_scale_value, axis=0)
                value_scale_value = (
                    value_scale_value * float(FP8_MAX_VALUE) / value_scale_max
                )
                attention_probs = value_scale_value[None, :] * attention_probs
                probability_scale = value_scale_max / float(FP8_MAX_VALUE)

        # Convert attention probabilities to appropriate data type
        if CONTIGUOUS_KV_ELEMENTS_16B_LOAD == 16:
            # FP8 quantization
            attention_probs = attention_probs.to(value_cache_ptr.dtype.element_ty)
        else:
            # FP16/BF16 computation
            attention_probs = attention_probs.to(COMPUTE_TYPE)

        # ==================== PV Matrix Multiplication ====================
        # Convert layouts for MFMA operation
        attention_probs_converted = gl.convert_layout(
            attention_probs, layout=pv_lhs_layout
        )
        value_block_converted = gl.convert_layout(value_block, layout=pv_rhs_layout)

        # Scale previous accumulator
        accumulator_scale_expanded = gl.convert_layout(
            accumulator_scale[:, None], layout=pv_mfma_layout
        )
        attention_accumulator *= accumulator_scale_expanded

        # Compute new attention output
        pv_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2),
            dtype=gl.float32,
            layout=pv_mfma_layout,
        )
        attention_output = gl.amd.cdna3.mfma(
            attention_probs_converted, value_block_converted, pv_accumulator
        )
        attention_accumulator += probability_scale * attention_output

        # Update maximum logits for next iteration
        current_max_logits = updated_max_logits

    # ==================== Final Output Scaling and Storage ====================
    # Compute final exponential sums
    exponential_sums = 1.0 / current_exp_sums
    exponential_sums_converted = gl.convert_layout(
        exponential_sums[:, None], layout=pv_mfma_layout
    )

    # Apply final scaling to attention accumulator
    attention_accumulator = attention_accumulator * exponential_sums_converted
    attention_accumulator = attention_accumulator.to(COMPUTE_TYPE)

    # Store results to output buffers
    gl.amd.cdna3.buffer_store(
        stored_value=current_max_logits,
        ptr=max_logits_ptr,
        offsets=max_logits_offsets,
        mask=max_logits_group_mask,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=current_exp_sums,
        ptr=exp_sums_ptr,
        offsets=max_logits_offsets,
        mask=max_logits_group_mask,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=attention_accumulator,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=output_mask,
    )


# @triton.autotune(
#     configs=[
#         triton.Config({'matrix_instr_nonkdim' : dim, 'waves_per_eu' : wa}, num_stages=s, num_warps=w) \
#         for s in [1, 2, 3, 4, 5, 6, 7, 8] \
#         for w in [4] \
#         for wa in [1, 2, 3, 4] \
#         for dim in [16] \
#     ],
#     key = ['Q_SEQ_LEN', 'QUERY_GRP_SZ_POW2', 'KV_BLK_SZ'],
# )
@gluon.jit
def paged_attention_decode_v2_gluon_fp8(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [num_seqs, num_kv_heads * query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size/x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    sequence_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, num_kv_heads * query_group_size, 1]
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    alibi_slopes_ptr,
    stride_max_logits_seq,
    stride_max_logits_head,
    stride_max_logits_part,
    stride_output_seq,
    stride_output_head,
    stride_output_part,
    stride_output_group,
    stride_query_seq,
    stride_query_head,
    stride_key_block,
    stride_key_head,
    stride_key_head_split,
    stride_key_block_elem,
    stride_value_block,
    stride_value_head,
    stride_value_head_size,
    stride_block_table_seq,
    query_scale_stride_0,
    kv_scale_stride_0,
    kv_scale_stride_1,
    QUERY_SEQ_LEN: tl.constexpr,
    COMPUTE_TYPE: tl.constexpr,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_POW2: tl.constexpr,
    QUERY_GROUP_SIZE_ORIGINAL: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE_POW2: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE_POW2: tl.constexpr,
    SEQUENCE_PARTITION_SIZE: tl.constexpr,
    KV_16B_ELEMENT_COUNT: tl.constexpr,
    QUERY_QUANT_MODE: tl.constexpr,
    KV_QUANT_MODE: tl.constexpr,
    KV_COMPUTE_BLOCK_SIZE: tl.constexpr,
    FP8_MAX_VALUE: tl.constexpr,
    VALUE_TRANSPOSED: tl.constexpr,  # [num_blocks, num_kv_heads, kv_block_size/x, head_size, x]
    IS_CAUSAL: tl.constexpr,
):
    """
    Paged Attention Decode Kernel with FP8/BF16 support for AMD GPUs.

    This kernel implements the attention mechanism for decoding in transformer models
    with support for paged KV caches and FP8 quantization. It handles causal masking,
    ALiBi biases, and various quantization schemes.

    Args:
        exp_sums_ptr: Pointer to exponential sums output tensor
        max_logits_ptr: Pointer to maximum logits output tensor
        output_ptr: Pointer to attention output tensor
        query_ptr: Pointer to query tensor
        key_cache_ptr: Pointer to key cache in block layout
        value_cache_ptr: Pointer to value cache in block layout
        block_tables_ptr: Pointer to block tables mapping sequences to physical blocks
        sequence_lengths_ptr: Pointer to sequence lengths for each sequence
        softmax_scale: Scaling factor for softmax
        query_scale: Query quantization scales
        key_scale: Key quantization scales
        value_scale: Value quantization scales
        alibi_slopes_ptr: Pointer to ALiBi slopes for attention bias
        Various stride parameters for tensor access
        Compile-time constants for kernel configuration

    Note:
        This kernel uses AMD CDNA3 MFMA instructions for efficient matrix operations
        and supports both FP8 and BF16 data types with various quantization modes.
    """

    # ==================== VALIDATION CHECKS ====================
    gl.static_assert(
        QUERY_SEQ_LEN <= 4,
        f"QUERY_SEQ_LEN={QUERY_SEQ_LEN}, Only support QUERY_SEQ_LEN <= 4",
    )
    gl.static_assert(
        QUERY_GROUP_SIZE_POW2 <= 64,
        f"QUERY_GROUP_SIZE_POW2={QUERY_GROUP_SIZE_POW2}, Only support QUERY_GROUP_SIZE_POW2 <= 64",
    )
    gl.static_assert(
        KV_BLOCK_SIZE == 16 or KV_BLOCK_SIZE == 64,
        f"KV_BLOCK_SIZE={KV_BLOCK_SIZE}, Only support KV_BLOCK_SIZE in [16, 64]",
    )

    # Data type validation
    gl.static_assert(query_ptr.dtype.element_ty == gl.float8e4b8)
    gl.static_assert(key_cache_ptr.dtype.element_ty == gl.float8e4b8)
    gl.static_assert(value_cache_ptr.dtype.element_ty == gl.float8e4b8)

    if QUERY_QUANT_MODE >= 0:
        gl.static_assert(query_scale.dtype.element_ty == gl.float32)
    if KV_QUANT_MODE >= 0:
        gl.static_assert(key_scale.dtype.element_ty == gl.float32)
        gl.static_assert(value_scale.dtype.element_ty == gl.float32)

    # ==================== CONSTANTS AND CONFIGURATION ====================
    LOG2_E: gl.constexpr = 1.4426950408889634  # log2(e) for exponential conversion
    CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD: gl.constexpr = KV_16B_ELEMENT_COUNT
    K_HEAD_SIZE_SPLITS: gl.constexpr = (
        HEAD_SIZE_POW2 // CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD
    )
    MAX_NUM_KV_BLOCKS_PER_COMPUTE: gl.constexpr = KV_COMPUTE_BLOCK_SIZE // KV_BLOCK_SIZE

    # ==================== MEMORY LAYOUT DEFINITIONS ====================

    # Query tensor layout - optimized for sequential access
    blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )

    # Key cache layout - optimized for block-wise access patterns
    blocked_key_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, 16],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=[4, 1, 1, 1],
        order=[3, 2, 1, 0],
    )

    # QK Matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    qk_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=16
    )
    qk_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=16
    )

    # Register allocation configuration based on group size and compute block size
    if QUERY_GROUP_SIZE_POW2 == 16:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (0, 128))
    elif QUERY_GROUP_SIZE_POW2 == 32:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (16, 0))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (0, 128), (16, 0))
    elif QUERY_GROUP_SIZE_POW2 == 64:
        if KV_COMPUTE_BLOCK_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (16, 0), (32, 0))
        elif KV_COMPUTE_BLOCK_SIZE == 256:
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
        shape=[QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE],
    )

    # Value cache layout configuration based on transpose flag
    if VALUE_TRANSPOSED:
        # Transposed value layout for better memory access patterns
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, 16],
            threads_per_warp=[4, 1, 16, 1],
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )
        value_dim1_offsets = gl.arange(
            0,
            KV_BLOCK_SIZE // CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD,
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
            CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD,
            layout=gl.SliceLayout(
                0, gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout))
            ),
        )
    else:
        # Standard value layout
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 16],
            threads_per_warp=[4, 16, 1],
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
        version=3, instr_shape=[16, 16], transposed=True, warps_per_cta=[1, 4]
    )
    pv_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=pv_mfma_layout, k_width=16
    )
    pv_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=16
    )

    # ==================== LAYOUT SLICE DEFINITIONS ====================

    # Query layout slices
    query_group_size_layout: gl.constexpr = gl.SliceLayout(1, blocked_query_layout)
    head_size_layout: gl.constexpr = gl.SliceLayout(0, blocked_query_layout)

    # Key layout slices
    block_id_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_key_layout))
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
    query_group_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=query_group_size_layout
    )
    head_size_offsets = gl.arange(0, HEAD_SIZE_POW2, layout=head_size_layout)
    head_size_split_offsets = gl.arange(
        0, K_HEAD_SIZE_SPLITS, layout=head_size_split_layout
    )
    block_element_offsets = gl.arange(0, KV_BLOCK_SIZE, layout=block_element_layout)
    contiguous_kv_element_offsets = gl.arange(
        0, CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD, layout=contiguous_kv_elements_layout
    )
    qk_row_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )

    # ==================== PROGRAM ID AND INITIALIZATION ====================
    sequence_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    sequence_partition_idx = gl.program_id(2)

    # Load query tensor with appropriate masking
    query_offsets_base = (
        sequence_idx * stride_query_seq
        + (kv_head_idx * QUERY_GROUP_SIZE + query_group_offsets[:, None])
        * stride_query_head
        + head_size_offsets[None, :]
    )
    query_mask = (query_group_offsets[:, None] < QUERY_GROUP_SIZE) & (
        head_size_offsets[None, :] < HEAD_SIZE
    )
    query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=query_offsets_base, mask=query_mask
    )

    # Load query quantization scales if needed
    if QUERY_QUANT_MODE == 0:
        # Per-tensor quantization
        query_scale_value = tl.load(query_scale)
    elif QUERY_QUANT_MODE == 1:
        # Per-token quantization
        query_scale_offsets = (
            sequence_idx * query_scale_stride_0
            + kv_head_idx * QUERY_GROUP_SIZE
            + qk_row_offsets[:, None]
        )
        query_scale_value = gl.amd.cdna3.buffer_load(
            ptr=query_scale,
            offsets=query_scale_offsets,
            mask=qk_row_offsets[:, None] < QUERY_GROUP_SIZE,
        )

    # Initialize output pointers and accumulators
    max_logits_base_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    max_logits_offsets = (
        sequence_idx * stride_max_logits_seq
        + kv_head_idx * stride_max_logits_head
        + sequence_partition_idx * stride_max_logits_part
        + max_logits_base_offsets
    )
    max_logits_group_mask = max_logits_base_offsets < QUERY_GROUP_SIZE

    output_group_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )
    output_mask = (output_group_offsets[:, None] < QUERY_GROUP_SIZE) & (
        output_head_size_offsets[None, :] < HEAD_SIZE
    )

    output_offsets = sequence_idx * stride_output_seq
    output_offsets += kv_head_idx * stride_output_head
    output_offsets += (
        sequence_partition_idx * stride_output_part
        + output_group_offsets[:, None] * stride_output_group
        + output_head_size_offsets[None, :]
    )

    # Initialize attention state variables
    max_attention_scores = max_logits_base_offsets.to(gl.float32) * float(0.0) - float(
        "inf"
    )
    attention_denominators = max_logits_base_offsets.to(gl.float32) * float(0.0)
    attention_output_accumulator = gl.zeros(
        (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=gl.float32, layout=pv_mfma_layout
    )

    # ==================== SEQUENCE PROCESSING ====================
    kv_sequence_length = gl.load(sequence_lengths_ptr + sequence_idx)
    kv_sequence_start_idx = sequence_partition_idx * SEQUENCE_PARTITION_SIZE
    if kv_sequence_start_idx >= kv_sequence_length:
        return  # No computation needed for this partition

    KV_COMPUTE_BLOCK_COUNT: gl.constexpr = (
        SEQUENCE_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE
    )
    SEQUENCE_PARTITION_KV_BLOCKS: gl.constexpr = (
        SEQUENCE_PARTITION_SIZE // KV_BLOCK_SIZE
    )

    # Process KV sequence in compute blocks
    for kv_compute_idx in range(KV_COMPUTE_BLOCK_COUNT):
        kv_subsequence_start_idx = (
            kv_sequence_start_idx + kv_compute_idx * KV_COMPUTE_BLOCK_SIZE
        )
        kv_subsequence_end_idx = gl.minimum(
            kv_subsequence_start_idx + KV_COMPUTE_BLOCK_SIZE, kv_sequence_length
        )

        num_kv_blocks = gl.cdiv(
            kv_subsequence_end_idx - kv_subsequence_start_idx, KV_BLOCK_SIZE
        )
        kv_block_start_idx = (
            sequence_partition_idx * SEQUENCE_PARTITION_KV_BLOCKS
            + kv_compute_idx * MAX_NUM_KV_BLOCKS_PER_COMPUTE
        )
        qk_column_offsets = kv_block_start_idx * KV_BLOCK_SIZE + gl.arange(
            0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
        )

        # Load ALiBi slopes if provided
        if alibi_slopes_ptr is None:
            alibi_slope_values = gl.zeros([QUERY_GROUP_SIZE_POW2], dtype=gl.float32)
        else:
            alibi_slope_values = gl.amd.cdna3.buffer_load(
                ptr=alibi_slopes_ptr + kv_head_idx * QUERY_GROUP_SIZE,
                offsets=qk_row_offsets,
                mask=qk_row_offsets < QUERY_GROUP_SIZE,
            )

        # Load KV block indices from block table
        block_indices = gl.arange(
            0, MAX_NUM_KV_BLOCKS_PER_COMPUTE, layout=block_id_layout
        )
        masked_block_indices = gl.where(block_indices < num_kv_blocks, block_indices, 0)
        block_table_start_ptr = block_tables_ptr + sequence_idx * stride_block_table_seq
        kv_block_numbers = gl.amd.cdna3.buffer_load(
            ptr=block_table_start_ptr + kv_block_start_idx, offsets=masked_block_indices
        )

        # ==================== KEY LOADING AND PROCESSING ====================
        # Calculate key cache offsets and load keys
        key_block_offsets = (
            kv_block_numbers[:, None, None, None] * stride_key_block
            + kv_head_idx * stride_key_head
            + head_size_split_offsets[None, :, None, None] * stride_key_head_split
            + block_element_offsets[None, None, :, None]
            * CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD
            + contiguous_kv_element_offsets[None, None, None, :]
        )
        key_tensor = gl.amd.cdna3.buffer_load(
            ptr=key_cache_ptr, offsets=key_block_offsets
        )

        # Load key quantization scales if needed
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 0:
                # Per-tensor quantization
                key_scale_value = tl.load(key_scale)
                value_scale_value = tl.load(value_scale)
            elif KV_QUANT_MODE == 1:
                # Per-token quantization
                key_scale_offsets = (
                    kv_block_numbers[:, None, None, None] * kv_scale_stride_0
                    + kv_head_idx * kv_scale_stride_1
                    + block_element_offsets[None, None, :, None]
                )
                key_scale_offsets = gl.reshape(
                    key_scale_offsets, [KV_COMPUTE_BLOCK_SIZE]
                )
                key_scale_offsets = gl.convert_layout(
                    key_scale_offsets, layout=gl.SliceLayout(0, qk_linear_layout)
                )
                key_scale_value = gl.amd.cdna3.buffer_load(
                    ptr=key_scale, offsets=key_scale_offsets
                )
                value_scale_value = gl.amd.cdna3.buffer_load(
                    ptr=value_scale, offsets=key_scale_offsets
                )

        # Reshape key tensor for matrix multiplication
        key_tensor = gl.permute(key_tensor, [1, 3, 0, 2])
        key_tensor = gl.reshape(key_tensor, [HEAD_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE])

        # ==================== VALUE LOADING AND PROCESSING ====================
        if VALUE_TRANSPOSED:
            # Load values from transposed cache layout
            kv_block_numbers_reshaped = gl.convert_layout(
                kv_block_numbers,
                layout=gl.SliceLayout(
                    1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
                ),
            )
            value_block_offsets = (
                kv_block_numbers_reshaped[:, None, None, None] * stride_value_block
                + kv_head_idx * stride_value_head
                + value_dim1_offsets[None, :, None, None] * stride_value_head_size
                + value_dim2_offsets[None, None, :, None]
                * CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD
                + value_dim3_offsets[None, None, None, :]
            )
            value_tensor = gl.amd.cdna3.buffer_load(
                ptr=value_cache_ptr, offsets=value_block_offsets
            )
            # Permute and reshape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 1, 3, 2])
            value_tensor = gl.reshape(
                value_tensor, [KV_COMPUTE_BLOCK_SIZE, HEAD_SIZE_POW2]
            )
        else:
            # Load values from standard cache layout
            kv_block_numbers_reshaped = gl.convert_layout(
                kv_block_numbers,
                layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout)),
            )
            value_block_offsets = (
                kv_block_numbers_reshaped[:, None, None] * stride_value_block
                + kv_head_idx * stride_value_head
                + value_dim1_offsets[None, :, None] * stride_value_head_size
                + value_dim2_offsets[None, None, :]
            )
            value_tensor = gl.amd.cdna3.buffer_load(
                ptr=value_cache_ptr, offsets=value_block_offsets
            )
            # Permute and reshape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 2, 1])
            value_tensor = gl.reshape(
                value_tensor, [KV_COMPUTE_BLOCK_SIZE, HEAD_SIZE_POW2]
            )

        # ==================== ATTENTION SCORE COMPUTATION ====================
        # Initialize QK accumulator
        qk_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE),
            dtype=gl.float32,
            layout=qk_mfma_layout,
        )

        # Convert layouts for MFMA operation
        query_converted = gl.convert_layout(query_tensor, layout=qk_lhs_operand_layout)
        key_converted = gl.convert_layout(key_tensor, layout=qk_rhs_operand_layout)

        # Compute QK attention scores using MFMA
        attention_scores = gl.amd.cdna3.mfma(
            query_converted, key_converted, qk_accumulator
        )
        attention_scores = gl.reshape(
            attention_scores, [QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE]
        )

        # Apply quantization scaling to attention scores
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # Expand scale for broadcasting
                key_scale_value = key_scale_value[None, :]
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

        # Apply ALiBi biases if provided
        if alibi_slopes_ptr is not None:
            alibi_bias = (
                alibi_slope_values[:, None]
                * (qk_column_offsets - kv_sequence_length + 1)[None, :]
            ).to(gl.float32)
            attention_scores += alibi_bias

        # ==================== ATTENTION MASKING ====================
        # Create boundary mask for valid sequence positions
        boundary_mask = qk_row_offsets[:, None] < QUERY_GROUP_SIZE

        # Apply causal masking if required
        if IS_CAUSAL:
            # Compute causal mask based on sequence positions
            sequence_position_extension = (
                QUERY_SEQ_LEN - 1 - qk_row_offsets // QUERY_GROUP_SIZE_ORIGINAL
            )
            causal_mask = (
                sequence_position_extension[:, None] + qk_column_offsets[None, :]
                < kv_sequence_length
            )
        else:
            causal_mask = qk_column_offsets[None, :] < kv_sequence_length

        boundary_mask = boundary_mask & causal_mask

        # Apply masking to attention scores (if [0, SEQUENCE_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -1e38 other than -inf)
        attention_scores = tl.where(boundary_mask, attention_scores, float(-1e38))

        # ==================== SOFTMAX COMPUTATION ====================
        # Update running maximum for numerical stability
        current_max_scores = gl.max(attention_scores, axis=1)
        new_max_scores = gl.maximum(max_attention_scores, current_max_scores)

        # Compute scaling factor for previous accumulator
        accumulator_scale = tl.math.exp2(
            (max_attention_scores - new_max_scores) * LOG2_E
        )

        # Compute attention probabilities
        attention_probs = tl.math.exp2(
            (attention_scores - new_max_scores[:, None]) * LOG2_E
        )
        attention_denominators = accumulator_scale * attention_denominators + gl.sum(
            attention_probs, axis=1
        )

        # ==================== VALUE ACCUMULATION ====================
        # Handle value quantization scaling for FP8
        if value_tensor.dtype.is_fp8():
            if KV_QUANT_MODE == 1:
                # Per-token quantization scaling
                value_scale_max = gl.max(value_scale_value, axis=0)
                value_scale_value = (
                    value_scale_value * float(FP8_MAX_VALUE) / value_scale_max
                )
                attention_probs = value_scale_value[None, :] * attention_probs
                probability_scale = value_scale_max / float(FP8_MAX_VALUE)

        # Convert attention probabilities to appropriate data type
        if CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD == 16:
            # FP8 data type
            attention_probs = attention_probs.to(value_cache_ptr.dtype.element_ty)
        else:
            # BF16/FP16 data type
            attention_probs = attention_probs.to(COMPUTE_TYPE)

        # Convert layouts for PV MFMA operation
        probs_converted = gl.convert_layout(
            attention_probs, layout=pv_lhs_operand_layout
        )
        values_converted = gl.convert_layout(value_tensor, layout=pv_rhs_operand_layout)

        # Scale previous accumulator and compute new attention output
        accumulator_scale_expanded = gl.convert_layout(
            accumulator_scale[:, None], layout=pv_mfma_layout
        )
        attention_output_accumulator *= accumulator_scale_expanded

        pv_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2),
            dtype=gl.float32,
            layout=pv_mfma_layout,
        )
        attention_update = gl.amd.cdna3.mfma(
            probs_converted, values_converted, pv_accumulator
        )

        # Apply value quantization scaling
        if KV_QUANT_MODE == 0:
            attention_output_accumulator += value_scale_value * attention_update
        elif KV_QUANT_MODE == 1:
            attention_output_accumulator += probability_scale * attention_update

        # Update running maximum for next iteration
        max_attention_scores = new_max_scores

    # ==================== OUTPUT NORMALIZATION AND STORING ====================
    # Normalize attention output by softmax denominator
    exp_sum_reciprocal = 1.0 / attention_denominators
    exp_sum_reciprocal_expanded = gl.convert_layout(
        exp_sum_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_output_accumulator = (
        attention_output_accumulator * exp_sum_reciprocal_expanded
    )
    attention_output_accumulator = attention_output_accumulator.to(COMPUTE_TYPE)

    # Store results to global memory
    gl.amd.cdna3.buffer_store(
        stored_value=max_attention_scores,
        ptr=max_logits_ptr,
        offsets=max_logits_offsets,
        mask=max_logits_group_mask,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=attention_denominators,
        ptr=exp_sums_ptr,
        offsets=max_logits_offsets,
        mask=max_logits_group_mask,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=attention_output_accumulator,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=output_mask,
    )


@gluon.jit
def paged_attention_decode_v2_reduce_gluon(
    output_ptr,  # [num_seqs, num_kv_heads, query_group_size, head_size]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size, head_size]
    sequence_lengths_ptr,  # [num_seqs]
    sink_token_ptr,  # [num_query_heads]
    stride_output_seq,
    stride_output_head,
    stride_exp_sums_seq,
    stride_exp_sums_head,
    stride_exp_sums_part,
    stride_logits_seq,
    stride_logits_head,
    stride_logits_part,
    stride_logits_group,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_POW2: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE_POW2: tl.constexpr,
    SEQUENCE_PARTITION_SIZE: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
    USE_SINK_TOKENS: tl.constexpr,
):
    """
    Gluon-based reduction kernel for paged attention decode that combines partial results.

    This kernel performs the final reduction step by:
    1. Finding global maximum logits across all sequence partitions
    2. Rescaling exponential sums for numerical stability
    3. Computing normalized attention probabilities
    4. Weighted summation of partial logits to produce final output

    Args:
        output_ptr: Output tensor for final attention results
        exp_sums_ptr: Exponential sums from partial computations
        max_logits_ptr: Maximum logits from partial computations
        logits_ptr: Partial logit tensors from each sequence partition
        sequence_lengths_ptr: Sequence lengths for each sequence
        sink_token_ptr: Sink token values for attention (optional)
        Various stride parameters for tensor access
        Compile-time constants for kernel configuration
    """
    # ==================== INITIALIZATION AND LAYOUT CONFIGURATION ====================
    sequence_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    num_query_heads_total = gl.num_programs(1) * QUERY_GROUP_SIZE
    sequence_length = gl.load(sequence_lengths_ptr + sequence_idx)
    num_partitions = gl.cdiv(sequence_length, SEQUENCE_PARTITION_SIZE)

    # Select optimal memory layout based on maximum partition count
    if MAX_NUM_SEQ_PARTITIONS_POW2 >= 256:
        blocked_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 2, 4],
            threads_per_warp=[4, 4, 4],
            warps_per_cta=[4, 1, 1],
            order=[2, 1, 0],
        )
    else:
        blocked_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[4, 1, 2],
            threads_per_warp=[4, 4, 4],
            warps_per_cta=[1, 1, 4],
            order=[2, 1, 0],
        )

    # Define layout slices for different tensor dimensions
    query_group_size_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(2, blocked_layout)
    )
    head_size_layout: gl.constexpr = gl.SliceLayout(
        0, gl.SliceLayout(1, blocked_layout)
    )
    sequence_partition_layout: gl.constexpr = gl.SliceLayout(
        1, gl.SliceLayout(2, blocked_layout)
    )

    # Generate coordinate offsets for tensor access
    partition_offsets = gl.arange(
        0, MAX_NUM_SEQ_PARTITIONS_POW2, layout=sequence_partition_layout
    )
    query_group_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=query_group_size_layout
    )
    head_size_offsets = gl.arange(0, HEAD_SIZE_POW2, layout=head_size_layout)

    # ==================== GLOBAL MAXIMUM LOGIT COMPUTATION ====================
    # Calculate offsets for accessing exponential sums and max logits tensors
    exp_sums_offsets = (
        sequence_idx * stride_exp_sums_seq
        + kv_head_idx * stride_exp_sums_head
        + partition_offsets[:, None] * stride_exp_sums_part
        + query_group_offsets[None, :]
    )

    # Create mask for valid partitions and query groups
    exp_sums_mask = (partition_offsets[:, None] < num_partitions) & (
        query_group_offsets[None, :] < QUERY_GROUP_SIZE
    )

    # Load maximum logits from all partitions [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GROUP_SIZE_POW2]
    max_logits = gl.amd.cdna3.buffer_load(
        ptr=max_logits_ptr, offsets=exp_sums_offsets, mask=exp_sums_mask
    )

    # Compute global maximum logit across all partitions [QUERY_GROUP_SIZE_POW2]
    global_max_logits = gl.max(max_logits, axis=0)

    # ==================== EXPONENTIAL SUMS RESCALING ====================
    # Load exponential sums from all partitions [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GROUP_SIZE_POW2]
    exp_sums = gl.amd.cdna3.buffer_load(
        ptr=exp_sums_ptr, offsets=exp_sums_offsets, mask=exp_sums_mask
    )

    # Rescale exponential sums for numerical stability using global maximum
    exp_sums *= gl.math.exp(max_logits - global_max_logits[None, :])

    # Compute global exponential sum across all partitions [QUERY_GROUP_SIZE_POW2]
    global_exp_sum = gl.sum(exp_sums, axis=0)

    # Add sink token contributions if enabled
    if USE_SINK_TOKENS:
        sink_token_values = gl.load(
            sink_token_ptr + (kv_head_idx * QUERY_GROUP_SIZE + query_group_offsets),
            mask=(kv_head_idx * QUERY_GROUP_SIZE + query_group_offsets)
            < num_query_heads_total,
        )
        global_exp_sum += gl.math.exp(sink_token_values - global_max_logits)

    # ==================== ATTENTION PROBABILITY COMPUTATION ====================
    # Compute normalized attention probabilities [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GROUP_SIZE_POW2]
    attention_probs = exp_sums / global_exp_sum[None, :]

    # Reshape probabilities for broadcasting with logits
    attention_probs = gl.reshape(
        attention_probs, (MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GROUP_SIZE_POW2, 1)
    )

    # ==================== LOGITS LOADING AND WEIGHTED SUMMATION ====================
    # Calculate offsets for loading partial logits
    logits_offsets = (
        sequence_idx * stride_logits_seq
        + kv_head_idx * stride_logits_head
        + partition_offsets[:, None, None] * stride_logits_part
        + query_group_offsets[None, :, None] * stride_logits_group
        + head_size_offsets[None, None, :]
    )

    # Create mask for valid logits access
    logits_mask = (partition_offsets[:, None] < num_partitions) & (
        query_group_offsets[None, :] < QUERY_GROUP_SIZE
    )

    # Load partial logits from all partitions
    partial_logits = gl.amd.cdna3.buffer_load(
        ptr=logits_ptr, offsets=logits_offsets, mask=logits_mask[:, :, None]
    )

    # Convert probabilities to blocked layout for efficient computation
    probs_converted = gl.convert_layout(attention_probs, layout=blocked_layout)

    # Compute weighted sum of logits to produce final output [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    final_output = gl.sum(
        (partial_logits * probs_converted).to(gl.float32), axis=0, keep_dims=True
    ).to(output_ptr.dtype.element_ty)

    # ==================== FINAL OUTPUT STORING ====================
    # Calculate output tensor offsets
    output_offsets = (
        sequence_idx * stride_output_seq
        + (kv_head_idx * QUERY_GROUP_SIZE + query_group_offsets[None, :, None])
        * stride_output_head
        + head_size_offsets[None, None, :]
    )

    # Create mask for valid output storage
    output_mask = (query_group_offsets[None, :, None] < QUERY_GROUP_SIZE) & (
        head_size_offsets[None, None, :] < HEAD_SIZE
    )

    # Store final output to global memory
    gl.amd.cdna3.buffer_store(
        stored_value=final_output,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=output_mask,
    )


@triton.jit
def paged_attention_decode_v2_reduce_kernel(
    output_ptr,  # [num_seqs, num_kv_heads, query_group_size, head_size]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size, head_size]
    sequence_lengths_ptr,  # [num_seqs]
    stride_output_seq,
    stride_output_head,
    stride_exp_sums_seq,
    stride_exp_sums_head,
    stride_exp_sums_part,
    stride_logits_seq,
    stride_logits_head,
    stride_logits_part,
    stride_logits_group,
    HEAD_SIZE: tl.constexpr,
    HEAD_SIZE_POW2: tl.constexpr,
    QUERY_GROUP_SIZE: tl.constexpr,
    QUERY_GROUP_SIZE_POW2: tl.constexpr,
    SEQUENCE_PARTITION_SIZE: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS: tl.constexpr,
    MAX_NUM_SEQ_PARTITIONS_POW2: tl.constexpr,
):
    """
    Triton reduction kernel for paged attention decode that combines partial results.

    This kernel performs the final reduction by:
    1. Finding global maximum logits across partitions
    2. Rescaling exponential sums for numerical stability
    3. Computing normalized attention probabilities
    4. Weighted summation of partial logits

    Args:
        output_ptr: Output tensor for final attention results
        exp_sums_ptr: Exponential sums from partial computations
        max_logits_ptr: Maximum logits from partial computations
        logits_ptr: Partial logit tensors from each sequence partition
        sequence_lengths_ptr: Sequence lengths for each sequence
        Various stride parameters for tensor access
        Compile-time constants for kernel configuration
    """
    # Mathematical constant for exponential calculations
    LOG2_E: tl.constexpr = 1.4426950408889634

    # ==================== INITIALIZATION ====================
    sequence_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    kv_sequence_length = tl.load(sequence_lengths_ptr + sequence_idx)
    num_partitions = tl.cdiv(kv_sequence_length, SEQUENCE_PARTITION_SIZE)

    # Generate coordinate ranges
    partition_offsets = tl.arange(0, MAX_NUM_SEQ_PARTITIONS_POW2)
    query_group_offsets = tl.arange(0, QUERY_GROUP_SIZE_POW2)
    head_size_offsets = tl.arange(0, HEAD_SIZE_POW2)

    # ==================== GLOBAL MAXIMUM LOGIT COMPUTATION ====================
    # Calculate offsets for exponential sums and max logits
    exp_sums_offsets = (
        sequence_idx * stride_exp_sums_seq
        + kv_head_idx * stride_exp_sums_head
        + partition_offsets[:, None] * stride_exp_sums_part
        + query_group_offsets[None, :]
    )

    # Create mask for valid partitions and query groups
    exp_sums_mask = (partition_offsets[:, None] < num_partitions) & (
        query_group_offsets[None, :] < QUERY_GROUP_SIZE
    )

    # Load maximum logits from all partitions [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GROUP_SIZE_POW2]
    max_logits = tl.load(
        max_logits_ptr + exp_sums_offsets, mask=exp_sums_mask, other=float("-inf")
    )

    # Compute global maximum logit across all partitions [QUERY_GROUP_SIZE_POW2]
    global_max_logits = tl.max(max_logits, axis=0)

    # ==================== EXPONENTIAL SUMS RESCALING ====================
    # Load exponential sums from all partitions
    exp_sums = tl.load(exp_sums_ptr + exp_sums_offsets, mask=exp_sums_mask)

    # Rescale exponential sums using global maximum for numerical stability
    exp_sums *= tl.exp(max_logits - global_max_logits[None, :])

    # Compute global exponential sum across all partitions [QUERY_GROUP_SIZE_POW2]
    global_exp_sum = tl.sum(exp_sums, axis=0)

    # ==================== ATTENTION PROBABILITY COMPUTATION ====================
    # Compute normalized attention probabilities [MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GROUP_SIZE_POW2]
    attention_probs = exp_sums / global_exp_sum[None, :]

    # Reshape probabilities for broadcasting with logits
    attention_probs = tl.reshape(
        attention_probs, (MAX_NUM_SEQ_PARTITIONS_POW2, QUERY_GROUP_SIZE_POW2, 1)
    )

    # ==================== LOGITS LOADING AND WEIGHTED SUMMATION ====================
    # Calculate offsets for loading partial logits
    logits_offsets = (
        sequence_idx * stride_logits_seq
        + kv_head_idx * stride_logits_head
        + partition_offsets[:, None, None] * stride_logits_part
        + query_group_offsets[None, :, None] * stride_logits_group
        + head_size_offsets[None, None, :]
    )

    # Create mask for valid logits access
    logits_mask = (partition_offsets[:, None] < num_partitions) & (
        query_group_offsets[None, :] < QUERY_GROUP_SIZE
    )

    # Load partial logits from all partitions
    partial_logits = tl.load(logits_ptr + logits_offsets, mask=logits_mask[:, :, None])

    # Compute weighted sum of logits to produce final output [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    final_output = tl.sum((partial_logits * attention_probs).to(tl.float32), axis=0).to(
        output_ptr.dtype.element_ty
    )

    # ==================== FINAL OUTPUT STORING ====================
    # Calculate output tensor offsets
    output_offsets = (
        sequence_idx * stride_output_seq
        + (kv_head_idx * QUERY_GROUP_SIZE + query_group_offsets[:, None])
        * stride_output_head
        + head_size_offsets[None, :]
    )

    # Create mask for valid output storage
    output_mask = (query_group_offsets[:, None] < QUERY_GROUP_SIZE) & (
        head_size_offsets[None, :] < HEAD_SIZE
    )

    # Store final output to global memory
    tl.store(
        output_ptr + output_offsets,
        final_output,
        mask=output_mask,
    )


def compile_ttgir_with_triton(ttgir_content: str):
    """
    Compile TTGIR (Triton Tensor IR) content to executable artifact.

    This function takes TTGIR string content, writes it to a temporary file,
    and compiles it using the Triton compiler. The temporary file is cleaned up
    after compilation regardless of success or failure.

    Args:
        ttgir_content (str): The TTGIR (Triton Tensor IR) code as a string
                             to be compiled.

    Returns:
        object: The compiled artifact from the Triton compiler.

    Raises:
        Exception: Any exception raised during the compilation process
                  will be propagated to the caller.

    Note:
        This function uses a temporary file to work with the Triton compiler
        which expects file input. The file is automatically deleted after
        compilation to avoid leaving temporary files on disk.
    """
    # Create a temporary file to store the TTGIR content
    # Using NamedTemporaryFile with delete=False to control deletion manually
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".ttgir", delete=False
    ) as temp_file:
        # Write TTGIR content to temporary file
        temp_file.write(ttgir_content)
        ttgir_file_path = temp_file.name

    try:
        # Compile the TTGIR file using Triton compiler
        # This converts the intermediate representation to executable code
        compiled_artifact = compile(ttgir_file_path)
        return compiled_artifact

    finally:
        # Ensure temporary file is cleaned up even if compilation fails
        # This prevents leaving temporary files on the filesystem
        if os.path.exists(ttgir_file_path):
            os.unlink(ttgir_file_path)


@perftest()
def _paged_attention_decode_v2_with_dot_kernel_reshape_wrapper(
    grid,
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [num_seqs, num_kv_heads * query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size/x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    sequence_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, num_kv_heads * query_group_size, 1]
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    alibi_slopes_ptr,
    stride_max_logits_seq,
    stride_max_logits_head,
    stride_max_logits_part,
    stride_output_seq,
    stride_output_head,
    stride_output_part,
    stride_output_group,
    stride_query_seq,
    stride_query_head,
    stride_key_block,
    stride_key_head,
    stride_key_head_split,
    stride_key_block_elem,
    stride_value_block,
    stride_value_head_size,
    stride_value_block_elem,
    stride_block_table_seq,
    query_scale_stride_0,
    kv_scale_stride_0,
    kv_scale_stride_1,
    kv_type,
    QUERY_SEQ_LEN,
    COMPUTE_TYPE,
    HEAD_SIZE,
    HEAD_SIZE_POW2,
    QUERY_GROUP_SIZE_ORIGINAL,
    QUERY_GROUP_SIZE,
    QUERY_GROUP_SIZE_POW2,
    KV_BLOCK_SIZE,
    KV_BLOCK_SIZE_POW2,
    SEQUENCE_PARTITION_SIZE,
    KV_16B_ELEMENT_COUNT,
    QUERY_QUANT_MODE,
    KV_QUANT_MODE,
    FP8_MAX_VALUE,
    VALUE_TRANSPOSED,
    IS_CAUSAL,
):
    """
    Wrapper function for paged attention decode kernel with dynamic kernel selection.

    This wrapper selects between different kernel implementations based on the
    configuration parameters and launches the appropriate kernel.

    Args:
        All parameters from the paged_attention_decode function, plus kernel configuration
        parameters for Triton compilation and execution.
    """
    # Debug compilation path - kept for development and debugging purposes
    if 0:
        ttgir_file_path = os.path.join(
            os.path.dirname(__file__),
            "./thread_trace/triton_gen_asm/pa_decode_v2_fp8/pa_decode_v2_fp8.ttgir",
        )
        with open(ttgir_file_path, "r") as f:
            ttgir_content = f.read()
        try:
            compiled_kernel = compile_ttgir_with_triton(ttgir_content)
            compiled_kernel[grid](
                exp_sums_ptr,
                max_logits_ptr,
                output_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                sequence_lengths_ptr,
                softmax_scale,
                query_scale,
                key_scale,
                value_scale,
                stride_max_logits_seq,
                stride_max_logits_head,
                stride_max_logits_part,
                stride_output_seq,
                stride_output_head,
                stride_output_part,
                stride_output_group,
                stride_query_seq,
                stride_query_head,
                stride_key_block,
                stride_key_head,
                stride_key_head_split,
                stride_key_block_elem,
                stride_value_block,
                stride_value_head_size,
                stride_value_block_elem,
                stride_block_table_seq,
                query_scale_stride_0,
                kv_scale_stride_0,
                kv_scale_stride_1,
            )
        except Exception as e:
            print(f"Compilation failed: {e}")
    else:
        # Production path - select and launch appropriate kernel
        KV_COMPUTE_BLOCK_SIZE = 256
        waves_per_eu = 1

        # Select kernel implementation based on block size
        if KV_BLOCK_SIZE > SEQUENCE_PARTITION_SIZE:
            # Use big block kernel for large block sizes
            paged_attention_kernel = paged_attention_decode_v2_gluon_large_block_fp8
            if VALUE_TRANSPOSED:
                # Use smaller compute block size for better performance with transposed values
                KV_COMPUTE_BLOCK_SIZE = 128
        else:
            # Use standard kernel for normal block sizes
            paged_attention_kernel = paged_attention_decode_v2_gluon_fp8
            # Configure waves per EU based on query group size
            if QUERY_GROUP_SIZE_POW2 == 64:
                waves_per_eu = 3
            else:
                waves_per_eu = 4

        # Launch the selected kernel
        paged_attention_kernel[grid](
            exp_sums_ptr,
            max_logits_ptr,
            output_ptr,
            query_ptr,
            key_cache_ptr,
            value_cache_ptr,
            block_tables_ptr,
            sequence_lengths_ptr,
            softmax_scale,
            query_scale,
            key_scale,
            value_scale,
            alibi_slopes_ptr,
            stride_max_logits_seq,
            stride_max_logits_head,
            stride_max_logits_part,
            stride_output_seq,
            stride_output_head,
            stride_output_part,
            stride_output_group,
            stride_query_seq,
            stride_query_head,
            stride_key_block,
            stride_key_head,
            stride_key_head_split,
            stride_key_block_elem,
            stride_value_block,
            stride_value_head_size,
            stride_value_block_elem,
            stride_block_table_seq,
            query_scale_stride_0,
            kv_scale_stride_0,
            kv_scale_stride_1,
            QUERY_SEQ_LEN=QUERY_SEQ_LEN,
            COMPUTE_TYPE=COMPUTE_TYPE,
            HEAD_SIZE=HEAD_SIZE,
            HEAD_SIZE_POW2=HEAD_SIZE_POW2,
            QUERY_GROUP_SIZE_ORIGINAL=QUERY_GROUP_SIZE_ORIGINAL,
            QUERY_GROUP_SIZE=QUERY_GROUP_SIZE,
            QUERY_GROUP_SIZE_POW2=QUERY_GROUP_SIZE_POW2,
            KV_BLOCK_SIZE=KV_BLOCK_SIZE,
            KV_BLOCK_SIZE_POW2=KV_BLOCK_SIZE_POW2,
            SEQUENCE_PARTITION_SIZE=SEQUENCE_PARTITION_SIZE,
            KV_16B_ELEMENT_COUNT=KV_16B_ELEMENT_COUNT,
            QUERY_QUANT_MODE=QUERY_QUANT_MODE,
            KV_QUANT_MODE=KV_QUANT_MODE,
            KV_COMPUTE_BLOCK_SIZE=KV_COMPUTE_BLOCK_SIZE,
            FP8_MAX_VALUE=FP8_MAX_VALUE,
            VALUE_TRANSPOSED=VALUE_TRANSPOSED,
            IS_CAUSAL=IS_CAUSAL,
            waves_per_eu=waves_per_eu,
            num_stages=1,
        )


@perftest()
def _paged_attention_decode_v2_reduce_kernel_wrapper(
    grid,
    output_ptr,  # [num_seqs, num_kv_heads, query_group_size, head_size]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size, head_size]
    sequence_lengths_ptr,  # [num_seqs]
    stride_output_seq,
    stride_output_head,
    stride_exp_sums_seq,
    stride_exp_sums_head,
    stride_exp_sums_part,
    stride_logits_seq,
    stride_logits_head,
    stride_logits_part,
    stride_logits_group,
    HEAD_SIZE,
    HEAD_SIZE_POW2,
    QUERY_GROUP_SIZE,
    QUERY_GROUP_SIZE_POW2,
    SEQUENCE_PARTITION_SIZE,
    MAX_NUM_SEQ_PARTITIONS,
    MAX_NUM_SEQ_PARTITIONS_POW2,
):
    """
    Wrapper function for paged attention reduction kernel with kernel selection.

    This wrapper selects between Gluon and Triton kernel implementations
    based on configuration and launches the appropriate kernel.

    Args:
        All parameters from the reduction kernel plus execution grid configuration
    """
    # Configuration flag for kernel selection
    USE_GLUON_KERNEL = False

    if USE_GLUON_KERNEL:
        # Launch Gluon-based reduction kernel (optimized for AMD hardware)
        paged_attention_decode_v2_reduce_gluon[grid](
            output_ptr,
            exp_sums_ptr,
            max_logits_ptr,
            logits_ptr,
            sequence_lengths_ptr,
            None,  # sink_token_ptr not used in this configuration
            stride_output_seq,
            stride_output_head,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
            HEAD_SIZE=HEAD_SIZE,
            HEAD_SIZE_POW2=HEAD_SIZE_POW2,
            QUERY_GROUP_SIZE=QUERY_GROUP_SIZE,
            QUERY_GROUP_SIZE_POW2=QUERY_GROUP_SIZE_POW2,
            SEQUENCE_PARTITION_SIZE=SEQUENCE_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=MAX_NUM_SEQ_PARTITIONS,
            MAX_NUM_SEQ_PARTITIONS_POW2=MAX_NUM_SEQ_PARTITIONS_POW2,
            USE_SINK_TOKENS=False,
        )
    else:
        # Launch standard Triton reduction kernel
        paged_attention_decode_v2_reduce_kernel[grid](
            output_ptr,
            exp_sums_ptr,
            max_logits_ptr,
            logits_ptr,
            sequence_lengths_ptr,
            stride_output_seq,
            stride_output_head,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
            HEAD_SIZE=HEAD_SIZE,
            HEAD_SIZE_POW2=HEAD_SIZE_POW2,
            QUERY_GROUP_SIZE=QUERY_GROUP_SIZE,
            QUERY_GROUP_SIZE_POW2=QUERY_GROUP_SIZE_POW2,
            SEQUENCE_PARTITION_SIZE=SEQUENCE_PARTITION_SIZE,
            MAX_NUM_SEQ_PARTITIONS=MAX_NUM_SEQ_PARTITIONS,
            MAX_NUM_SEQ_PARTITIONS_POW2=MAX_NUM_SEQ_PARTITIONS_POW2,
        )


def paged_attention_decode(
    output: torch.Tensor,  # [num_seqs, num_kv_heads * query_group_size, head_size]
    query: torch.Tensor,  # [num_seqs, num_kv_heads * query_group_size, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size/x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size/x, head_size, x]
    sequence_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_sequence_length: int,
    max_sequence_length: int,
    compute_type,
    query_scale: torch.Tensor,  # [num_seqs, num_kv_heads * query_group_size, 1]
    key_scale: torch.Tensor,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    num_sequence_partitions: int = 0,
    alibi_slopes: torch.Tensor = None,
) -> None:
    """
    Paged Attention Decode Function with FP8/BF16 Support.

    This function implements the attention mechanism for transformer decoding with
    paged KV caches, supporting various quantization schemes and data types.

    Args:
        output: Output tensor for attention results
        query: Input query tensor
        key_cache: Paged key cache in block layout
        value_cache: Paged value cache in block layout
        sequence_lengths: Current sequence lengths for each sequence
        block_tables: Mapping from sequences to physical cache blocks
        softmax_scale: Scaling factor for attention scores
        query_sequence_length: Length of query sequences
        max_sequence_length: Maximum sequence length supported
        compute_type: Data type for computation (FP8, BF16, etc.)
        query_scale: Quantization scales for queries
        key_scale: Quantization scales for keys
        value_scale: Quantization scales for values
        num_sequence_partitions: Number of sequence partitions (future use)
        alibi_slopes: ALiBi attention bias slopes

    Returns:
        Dictionary containing timing information and intermediate tensors
    """
    # Extract tensor dimensions
    num_sequences = query.shape[0]
    num_query_heads_total = query.shape[1]
    num_query_heads_per_kv = num_query_heads_total // query_sequence_length
    num_kv_heads = key_cache.shape[1]
    max_num_partitions = int(
        (max_sequence_length + _SEQUENCE_PARTITION_SIZE - 1) // _SEQUENCE_PARTITION_SIZE
    )
    head_size = query.shape[-1]
    kv_block_size = key_cache.shape[-2]
    query_group_size = num_query_heads_per_kv // num_kv_heads

    # Calculate equivalent group sizes for kernel configuration
    equivalent_query_group_size = query_sequence_length * query_group_size
    equivalent_query_group_size_pow2 = triton.next_power_of_2(
        equivalent_query_group_size
    )
    kv_block_size_pow2 = triton.next_power_of_2(kv_block_size)
    head_size_pow2 = triton.next_power_of_2(head_size)

    # Determine if causal masking is needed
    is_causal = query_sequence_length > 1

    # Calculate elements per 16B load based on data type
    kv_elements_per_16b = 16 // key_cache.dtype.itemsize

    # Configure execution grid
    grid = (num_sequences, num_kv_heads, max_num_partitions)
    intermediate_shape = (
        num_sequences,
        num_kv_heads,
        max_num_partitions,
        equivalent_query_group_size,
    )

    # Initialize intermediate tensors for attention computation
    max_logits = torch.zeros(
        intermediate_shape, dtype=torch.float32, device=output.device
    )
    exp_sums = torch.zeros(
        intermediate_shape, dtype=torch.float32, device=output.device
    )
    temporary_output = torch.zeros(
        *intermediate_shape, head_size, dtype=output.dtype, device=output.device
    )

    # Adjust query group size to power of 2 with constraints
    if equivalent_query_group_size <= 16:
        equivalent_query_group_size_pow2 = 16
    else:
        equivalent_query_group_size_pow2 = triton.next_power_of_2(
            equivalent_query_group_size
        )

    # Validate input params constraint
    assert (
        query.dtype == aiter.dtypes.fp8
    ), f"query tensor only support dtype == {aiter.dtypes.fp8}, but got query.dtype == {query.dtype}"
    assert (
        key_cache.dtype == aiter.dtypes.fp8
    ), f"key_cache tensor only support dtype == {aiter.dtypes.fp8}, but got key_cache.dtype == {key_cache.dtype}"
    assert (
        value_cache.dtype == aiter.dtypes.fp8
    ), f"value_cache tensor only support dtype == {aiter.dtypes.fp8}, but got value_cache.dtype == {value_cache.dtype}"
    assert (
        output.dtype == aiter.dtypes.bf16
    ), f"output tensor only support dtype == {aiter.dtypes.bf16}, but got output.dtype == {output.dtype}"
    assert (
        equivalent_query_group_size_pow2 <= 64
    ), f"equivalent_query_group_size_pow2={equivalent_query_group_size_pow2} exceeds maximum of 64"
    assert kv_block_size in [
        16,
        64,
        1024,
    ], f"kv_block_size == {kv_block_size} not in [16, 64, 1024]"
    assert (
        len(output.shape) == 3
    ), f"Expected 3D output tensor, but got shape {output.shape}"
    assert (
        len(query.shape) == 3
    ), f"Expected 3D query tensor, but got shape {query.shape}"
    assert (
        len(key_cache.shape) == 5
    ), f"Expected 5D key_cache tensor, but got shape {key_cache.shape}"

    # ==================== QUANTIZATION MODE CONFIGURATION ====================
    query_scale_stride_0 = 0
    key_scale_stride_0 = 0
    key_scale_stride_1 = 0
    query_quant_mode = -1
    kv_quant_mode = -1

    # Configure query quantization
    if query_scale is not None:
        assert (
            isinstance(query_scale, torch.Tensor)
            and query_scale.dtype == aiter.dtypes.fp32
        ), f"query_scale tensor only support dtype == {aiter.dtypes.fp32}, but got query_scale.dtype == {query_scale.dtype}"

        if len(query_scale.shape) == 0:
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
            query_scale_stride_0 = query_scale.stride(0)

    # Configure KV quantization
    if key_scale is not None and value_scale is not None:
        assert (
            isinstance(key_scale, torch.Tensor) and key_scale.dtype == aiter.dtypes.fp32
        ), f"key_scale tensor only support dtype == {aiter.dtypes.fp32}, but got key_scale.dtype == {key_scale.dtype}"
        assert (
            isinstance(value_scale, torch.Tensor)
            and value_scale.dtype == aiter.dtypes.fp32
        ), f"value_scale tensor only support dtype == {aiter.dtypes.fp32}, but got value_scale.dtype == {value_scale.dtype}"

        if len(key_scale.shape) == 0:
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

        # Validate KV scale shape consistency
        assert (
            key_scale.shape == value_scale.shape
        ), f"Key and value scales must have same shape, but got key: {key_scale.shape}, value: {value_scale.shape}"

    # ==================== VALUE CACHE LAYOUT DETECTION ====================
    value_transposed = False
    if len(value_cache.shape) == 5:
        value_transposed = True
    elif len(value_cache.shape) == 4:
        value_transposed = False
    else:
        raise RuntimeError(f"Unsupported value cache shape: {value_cache.shape}")

    # ==================== FP8 CONFIGURATION ====================
    fp8_max_value = 1.0
    if value_cache.dtype == aiter.dtypes.fp8:
        fp8_max_value = torch.finfo(aiter.dtypes.fp8).max

    # ==================== ATTENTION DECODE KERNEL EXECUTION ====================
    _, decode_execution_time = (
        _paged_attention_decode_v2_with_dot_kernel_reshape_wrapper(
            grid,
            exp_sums,
            max_logits,
            temporary_output,
            query,
            key_cache,
            value_cache,
            block_tables,
            sequence_lengths,
            softmax_scale,
            query_scale,
            key_scale,
            value_scale,
            alibi_slopes,
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            query.stride(0),
            query.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            key_cache.stride(3),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            block_tables.stride(0),
            query_scale_stride_0,
            key_scale_stride_0,
            key_scale_stride_1,
            kv_type=compute_type,
            QUERY_SEQ_LEN=query_sequence_length,
            COMPUTE_TYPE=compute_type,
            HEAD_SIZE=head_size,
            HEAD_SIZE_POW2=head_size_pow2,
            QUERY_GROUP_SIZE_ORIGINAL=query_group_size,
            QUERY_GROUP_SIZE=equivalent_query_group_size,
            QUERY_GROUP_SIZE_POW2=equivalent_query_group_size_pow2,
            KV_BLOCK_SIZE=kv_block_size,
            KV_BLOCK_SIZE_POW2=kv_block_size_pow2,
            SEQUENCE_PARTITION_SIZE=_SEQUENCE_PARTITION_SIZE,
            KV_16B_ELEMENT_COUNT=kv_elements_per_16b,
            QUERY_QUANT_MODE=query_quant_mode,
            KV_QUANT_MODE=kv_quant_mode,
            FP8_MAX_VALUE=fp8_max_value,
            VALUE_TRANSPOSED=value_transposed,
            IS_CAUSAL=is_causal,
        )
    )

    # ==================== REDUCTION KERNEL EXECUTION ====================
    grid = (num_sequences, num_kv_heads, 1)
    _, reduce_execution_time = _paged_attention_decode_v2_reduce_kernel_wrapper(
        grid,
        output,
        exp_sums,
        max_logits,
        temporary_output,
        sequence_lengths,
        output.stride(0),
        output.stride(1),
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        temporary_output.stride(0),
        temporary_output.stride(1),
        temporary_output.stride(2),
        temporary_output.stride(3),
        HEAD_SIZE=head_size,
        HEAD_SIZE_POW2=head_size_pow2,
        QUERY_GROUP_SIZE=equivalent_query_group_size,
        QUERY_GROUP_SIZE_POW2=equivalent_query_group_size_pow2,
        SEQUENCE_PARTITION_SIZE=_SEQUENCE_PARTITION_SIZE,
        MAX_NUM_SEQ_PARTITIONS=int(max_num_partitions),
        MAX_NUM_SEQ_PARTITIONS_POW2=int(triton.next_power_of_2(max_num_partitions)),
    )

    # ==================== RETURN RESULTS AND TIMING INFORMATION ====================
    return {
        "triton_decode_time": decode_execution_time,
        "triton_reduce_time": reduce_execution_time,
        "temporary_output": temporary_output,
        "final_output": output,
        "exponential_sums": exp_sums,
        "maximum_logits": max_logits,
        "total_triton_time": decode_execution_time + reduce_execution_time,
    }
