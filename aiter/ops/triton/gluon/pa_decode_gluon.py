# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
# This code is derived from sglang and FLASHNN projects
# https://github.com/AlibabaPAI/FLASHNN/blob/main/flashnn/triton_kernels/paged_attn.py

from typing import Optional
from functools import lru_cache
import torch
import aiter
import aiter.ops.triton.utils._triton.arch_info as arch_info

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


@gluon.jit
def transpose_query_gluon_kernel(
    input_ptr,  # Input tensor pointer
    output_ptr,  # Output tensor pointer
    batch_size,
    seq_len,
    num_kv_heads,
    query_group_size,
    last_dim,  # head_size or 1 for scale
    stride_input_batch,
    stride_input_seq,
    stride_input_head,
    stride_input_group,
    stride_output_batch,
    stride_output_merged,
    grid_dim_0,  # Grid dimension 0 (batch_size)
    grid_dim_1,  # Grid dimension 1 (merged_blocks)
    grid_dim_2,  # Grid dimension 2 (last_blocks)
    MERGED_BLOCK_SIZE: gl.constexpr,  # Block size for merged dimension
    BLOCK_SIZE_LAST: gl.constexpr,  # Block size for last dimension
    STRIDE_LAST: gl.constexpr = 1,  # Stride for last dimension (always 1)
):
    """
    Gluon version: Transpose query tensor from [batch_size, seq_len, num_kv_heads, query_group_size, last_dim]
    to [batch_size, num_kv_heads * seq_len * query_group_size, last_dim]
    """
    # ==================== Memory Layout Definitions ====================
    # Define blocked layout for efficient memory access
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],  # Each thread processes 1x8 elements
        threads_per_warp=[4, 16],  # 4x16=64 threads per warp
        warps_per_cta=[4, 1],  # 4 warps per CTA
        order=[1, 0],  # Column-major order
    )

    # ==================== Program ID and Dimensions ====================
    # Get 3D program IDs: batch, merged_block, last_block
    batch_idx = gl.program_id(0)
    merged_block_idx = gl.program_id(1)
    last_block_idx = gl.program_id(2)

    # Calculate total merged dimension size
    merged_dim_size = num_kv_heads * seq_len * query_group_size

    # Check if this CTA is within valid range
    if batch_idx >= batch_size:
        return

    # ==================== Offset Calculation ====================
    # Create offsets for the merged and last dimensions
    merged_offsets = gl.arange(
        0, MERGED_BLOCK_SIZE, layout=gl.SliceLayout(1, blocked_layout)
    )
    last_offsets = gl.arange(
        0, BLOCK_SIZE_LAST, layout=gl.SliceLayout(0, blocked_layout)
    )

    # Global offsets for this block
    global_merged_offsets = merged_block_idx * MERGED_BLOCK_SIZE + merged_offsets
    global_last_offsets = last_block_idx * BLOCK_SIZE_LAST + last_offsets

    # ==================== Decompose merged index to input indices ====================
    # For each element in merged dimension, calculate corresponding indices in input tensor
    # Input layout: [batch_size, seq_len, num_kv_heads, query_group_size, last_dim]
    # Merged index = kv_head_idx * (seq_len * query_group_size) + seq_idx * query_group_size + group_idx

    kv_head_idx = global_merged_offsets // (seq_len * query_group_size)
    remainder = global_merged_offsets % (seq_len * query_group_size)
    seq_idx = remainder // query_group_size
    group_idx = remainder % query_group_size

    # ==================== Load from Input (with transpose) ====================
    # Calculate input offsets
    # Input layout: [batch_size, seq_len, num_kv_heads, query_group_size, last_dim]
    input_offsets = (
        batch_idx * stride_input_batch
        + seq_idx[:, None] * stride_input_seq
        + kv_head_idx[:, None] * stride_input_head
        + group_idx[:, None] * stride_input_group
        + global_last_offsets[None, :] * STRIDE_LAST
    )

    # Create mask for valid elements
    input_mask = (global_merged_offsets[:, None] < merged_dim_size) & (
        global_last_offsets[None, :] < last_dim
    )

    # Load data from global memory using AMD CDNA3 buffer load
    # Shape: [MERGED_BLOCK_SIZE, BLOCK_SIZE_LAST]
    data_tensor = gl.amd.cdna3.buffer_load(
        ptr=input_ptr,
        offsets=input_offsets,
        mask=input_mask,
    )

    # Calculate output offsets
    # Output layout: [batch_size, num_kv_heads * seq_len * query_group_size, last_dim]
    output_offsets = (
        batch_idx * stride_output_batch
        + global_merged_offsets[:, None] * stride_output_merged
        + global_last_offsets[None, :] * STRIDE_LAST
    )

    # Create mask for valid output elements
    output_mask = (global_merged_offsets[:, None] < merged_dim_size) & (
        global_last_offsets[None, :] < last_dim
    )

    # Store to global memory
    gl.amd.cdna3.buffer_store(
        stored_value=data_tensor,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=output_mask,
    )


def transpose_query_gluon(
    query: torch.Tensor,
    query_gluon: torch.Tensor,
    query_scale: Optional[torch.Tensor],
    query_scale_gluon: Optional[torch.Tensor],
    batch_size: int,
    query_sequence_length: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
) -> None:
    """
    Transpose query and optionally query_scale tensors for Gluon kernel.

    This version supports both the original Triton kernel and the new Gluon kernel
    with shared memory optimization.

    Args:
        query: [batch_size * query_sequence_length, num_query_heads, head_size]
        query_gluon: [batch_size, num_kv_heads * query_sequence_length * query_group_size, head_size] - output buffer
        query_scale: Optional[batch_size * query_sequence_length, num_query_heads, 1]
        query_scale_gluon: Optional[batch_size, num_kv_heads * query_sequence_length * query_group_size, 1] - output buffer
        batch_size: Batch size
        query_sequence_length: Query sequence length
        num_kv_heads: Number of KV heads
        query_group_size: Query group size (num_query_heads // num_kv_heads)
        head_size: Head dimension size

    Returns:
        None (results are written to query_gluon and query_scale_gluon in-place)
    """
    # Calculate strides for input tensors using actual tensor strides
    # Input query shape: [batch_size * seq_len, num_kv_heads * query_group_size, head_size]
    # We interpret it as 5D: [batch_size, seq_len, num_kv_heads, query_group_size, head_size]
    # Use actual tensor strides to support non-contiguous tensors
    query_stride_0 = query.stride(0)  # stride for dim 0 (batch*seq dimension)
    query_stride_1 = query.stride(1)  # stride for dim 1 (num_heads dimension)

    stride_input_batch = query_sequence_length * query_stride_0  # skip seq_len rows
    stride_input_seq = query_stride_0  # skip 1 row
    stride_input_head = query_group_size * query_stride_1  # skip query_group_size heads
    stride_input_group = query_stride_1  # skip 1 head

    # Calculate strides for output tensors using actual tensor strides
    # Output shape: [batch_size, num_kv_heads * seq_len * query_group_size, head_size]
    stride_output_batch = query_gluon.stride(0)
    stride_output_merged = query_gluon.stride(1)

    if GLUON_JIT_KERNEL_ENABLED:
        BLOCK_SIZE_LAST = triton.next_power_of_2(head_size)
        # Calculate merged block size
        max_merged_block_size = num_kv_heads * query_sequence_length * query_group_size
        MERGED_BLOCK_SIZE = triton.next_power_of_2(max_merged_block_size)
        # Calculate grid dimensions
        merged_dim_size = num_kv_heads * query_sequence_length * query_group_size
        grid = (
            batch_size,
            triton.cdiv(merged_dim_size, MERGED_BLOCK_SIZE),
            triton.cdiv(head_size, BLOCK_SIZE_LAST),
        )

        # Launch query transpose kernel
        transpose_query_gluon_kernel[grid](
            query,
            query_gluon,
            batch_size,
            query_sequence_length,
            num_kv_heads,
            query_group_size,
            head_size,
            stride_input_batch,
            stride_input_seq,
            stride_input_head,
            stride_input_group,
            stride_output_batch,
            stride_output_merged,
            grid[0],  # grid_dim_0
            grid[1],  # grid_dim_1
            grid[2],  # grid_dim_2
            MERGED_BLOCK_SIZE,
            BLOCK_SIZE_LAST,
        )

        # Handle query_scale if present
        if query_scale is not None and len(query_scale.shape) > 1:
            # For scale, last_dim = 1
            BLOCK_SIZE_LAST_SCALE = 1

            # Calculate strides for query_scale using actual tensor strides
            # Input shape: [batch_size * seq_len, num_kv_heads * query_group_size, 1]
            query_scale_stride_0 = query_scale.stride(0)  # stride for dim 0
            query_scale_stride_1 = query_scale.stride(1)  # stride for dim 1

            stride_input_batch_scale = query_sequence_length * query_scale_stride_0
            stride_input_seq_scale = query_scale_stride_0
            stride_input_head_scale = query_group_size * query_scale_stride_1
            stride_input_group_scale = query_scale_stride_1

            # Output strides using actual tensor strides
            # Output shape: [batch_size, num_kv_heads * seq_len * query_group_size, 1]
            stride_output_batch_scale = query_scale_gluon.stride(0)
            stride_output_merged_scale = query_scale_gluon.stride(1)

            grid_scale = (
                batch_size,
                triton.cdiv(merged_dim_size, MERGED_BLOCK_SIZE),
                1,  # last_dim = 1
            )

            transpose_query_gluon_kernel[grid_scale](
                query_scale,
                query_scale_gluon,
                batch_size,
                query_sequence_length,
                num_kv_heads,
                query_group_size,
                1,  # last_dim = 1 for scale
                stride_input_batch_scale,
                stride_input_seq_scale,
                stride_input_head_scale,
                stride_input_group_scale,
                stride_output_batch_scale,
                stride_output_merged_scale,
                grid_scale[0],  # grid_dim_0
                grid_scale[1],  # grid_dim_1
                grid_scale[2],  # grid_dim_2
                MERGED_BLOCK_SIZE,
                BLOCK_SIZE_LAST_SCALE,
            )


@gluon.jit
def transpose_output_gluon_kernel(
    input_ptr,  # Input tensor pointer
    output_ptr,  # Output tensor pointer
    batch_size,
    seq_len,
    num_kv_heads,
    query_group_size,
    last_dim,
    stride_input_batch,
    stride_input_kv_head,
    stride_input_seq,
    stride_input_group,
    stride_output_batch_seq,
    stride_output_merged,
    grid_dim_0,  # Grid dimension 0 (batch_seq_size)
    grid_dim_1,  # Grid dimension 1 (merged_blocks)
    grid_dim_2,  # Grid dimension 2 (last_blocks)
    MERGED_BLOCK_SIZE: gl.constexpr,  # Block size for merged dimension
    BLOCK_SIZE_LAST: gl.constexpr,  # Block size for last dimension
    STRIDE_LAST: gl.constexpr = 1,  # Stride for last dimension (always 1)
):
    """
    Gluon version: Transpose output tensor from [batch_size, num_kv_heads, seq_len, query_group_size, last_dim]
    to [batch_size * seq_len, num_query_heads, last_dim]

    This implements the transformation:
        output_final = output_gluon.reshape(batch_size, num_kv_heads, seq_len, query_group_size, head_size)
        output_final = output_final.transpose(1, 2).reshape(batch_size * seq_len, num_query_heads, head_size)
    """
    # ==================== Memory Layout Definitions ====================
    # Define blocked layout for efficient memory access
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],  # Each thread processes 1x8 elements
        threads_per_warp=[4, 16],  # 4x16=64 threads per warp
        warps_per_cta=[4, 1],  # 4 warps per CTA
        order=[1, 0],  # Column-major order
    )

    # ==================== Program ID and Dimensions ====================
    # Get 3D program IDs: batch_seq_block, merged_block, last_block
    batch_seq_block_idx = gl.program_id(0)
    merged_block_idx = gl.program_id(1)
    last_block_idx = gl.program_id(2)

    # Calculate total batch_seq dimension size and merged dimension
    batch_seq_size = batch_size * seq_len
    merged_dim_size = num_kv_heads * query_group_size

    # Check if this CTA is within valid range
    if batch_seq_block_idx >= batch_seq_size:
        return

    # ==================== Offset Calculation ====================
    # Create offsets for the merged and last dimensions
    merged_offsets = gl.arange(
        0, MERGED_BLOCK_SIZE, layout=gl.SliceLayout(1, blocked_layout)
    )
    last_offsets = gl.arange(
        0, BLOCK_SIZE_LAST, layout=gl.SliceLayout(0, blocked_layout)
    )

    # Global offsets for this block
    global_merged_offsets = merged_block_idx * MERGED_BLOCK_SIZE + merged_offsets
    global_last_offsets = last_block_idx * BLOCK_SIZE_LAST + last_offsets

    # ==================== Decompose batch_seq index to batch and seq ====================
    # Decompose batch_seq_block_idx to batch_idx and seq_idx
    batch_idx = batch_seq_block_idx // seq_len
    seq_idx = batch_seq_block_idx % seq_len

    # ==================== Decompose merged index to kv_head and group ====================
    # For output, merged index = kv_head_idx * query_group_size + group_idx
    # After transpose(1, 2), the layout becomes [batch, seq, kv_head, group, last_dim]
    kv_head_idx = global_merged_offsets // query_group_size
    group_idx = global_merged_offsets % query_group_size

    # ==================== Load from Input ====================
    # Input layout: [batch_size, num_kv_heads, seq_len, query_group_size, last_dim]
    input_offsets = (
        batch_idx * stride_input_batch
        + kv_head_idx[:, None] * stride_input_kv_head
        + seq_idx * stride_input_seq
        + group_idx[:, None] * stride_input_group
        + global_last_offsets[None, :] * STRIDE_LAST
    )

    # Create mask for valid elements
    input_mask = (global_merged_offsets[:, None] < merged_dim_size) & (
        global_last_offsets[None, :] < last_dim
    )

    # Load data from global memory using AMD CDNA3 buffer load
    # Shape: [MERGED_BLOCK_SIZE, BLOCK_SIZE_LAST]
    data_tensor = gl.amd.cdna3.buffer_load(
        ptr=input_ptr,
        offsets=input_offsets,
        mask=input_mask,
    )

    # Calculate output offsets
    # Output layout: [batch_size * seq_len, num_query_heads, last_dim]
    # where num_query_heads = num_kv_heads * query_group_size
    output_offsets = (
        batch_seq_block_idx * stride_output_batch_seq
        + global_merged_offsets[:, None] * stride_output_merged
        + global_last_offsets[None, :] * STRIDE_LAST
    )

    # Create mask for valid output elements
    output_mask = (global_merged_offsets[:, None] < merged_dim_size) & (
        global_last_offsets[None, :] < last_dim
    )

    # Store to global memory
    gl.amd.cdna3.buffer_store(
        stored_value=data_tensor,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=output_mask,
    )


def transpose_output_gluon(
    output_gluon: torch.Tensor,
    output: torch.Tensor,
    batch_size: int,
    query_sequence_length: int,
    num_kv_heads: int,
    query_group_size: int,
    head_size: int,
) -> None:
    """
    Transpose output tensor from Gluon format to standard format.

    This implements the transformation:
        output_final = output_gluon.reshape(batch_size, num_kv_heads, query_sequence_length, query_group_size, head_size)
        output_final = output_final.transpose(1, 2).reshape(batch_size * query_sequence_length, num_query_heads, head_size)

    Args:
        output_gluon: Input tensor with physical shape [batch_size, num_kv_heads * query_sequence_length * query_group_size, head_size] (3D),
                      interpreted as logical layout [batch_size, num_kv_heads, query_sequence_length, query_group_size, head_size] (5D)
        output: [batch_size * query_sequence_length, num_query_heads, head_size] - output buffer
        batch_size: Batch size
        query_sequence_length: Query sequence length
        num_kv_heads: Number of KV heads
        query_group_size: Query group size (num_query_heads // num_kv_heads)
        head_size: Head dimension size

    Returns:
        None (results are written to output in-place)
    """
    # Calculate strides for input tensors using actual tensor strides
    # Input output_gluon shape: [batch_size, num_kv_heads * query_sequence_length * query_group_size, head_size]
    # Logical layout: [batch_size, num_kv_heads, query_sequence_length, query_group_size, head_size] (5D view)
    # Merged dimension is organized as: kv_head * (seq_len * group_size) + seq * group_size + group
    output_gluon_stride_0 = output_gluon.stride(0)  # stride for batch dimension
    output_gluon_stride_1 = output_gluon.stride(1)  # stride for merged dimension

    stride_input_batch = output_gluon_stride_0
    stride_input_kv_head = (
        query_sequence_length * query_group_size * output_gluon_stride_1
    )
    stride_input_seq = query_group_size * output_gluon_stride_1
    stride_input_group = output_gluon_stride_1

    # Calculate strides for output tensors using actual tensor strides
    # Output shape: [batch_size * query_sequence_length, num_query_heads, head_size]
    stride_output_batch_seq = output.stride(0)
    stride_output_merged = output.stride(1)

    if GLUON_JIT_KERNEL_ENABLED:
        BLOCK_SIZE_LAST = triton.next_power_of_2(head_size)
        # Calculate merged block size (num_kv_heads * query_group_size)
        max_merged_block_size = num_kv_heads * query_group_size
        MERGED_BLOCK_SIZE = triton.next_power_of_2(max_merged_block_size)

        # Calculate grid dimensions
        merged_dim_size = num_kv_heads * query_group_size
        batch_seq_size = batch_size * query_sequence_length
        grid = (
            batch_seq_size,
            triton.cdiv(merged_dim_size, MERGED_BLOCK_SIZE),
            triton.cdiv(head_size, BLOCK_SIZE_LAST),
        )

        # Launch output transpose kernel
        transpose_output_gluon_kernel[grid](
            output_gluon,
            output,
            batch_size,
            query_sequence_length,
            num_kv_heads,
            query_group_size,
            head_size,
            stride_input_batch,
            stride_input_kv_head,
            stride_input_seq,
            stride_input_group,
            stride_output_batch_seq,
            stride_output_merged,
            grid[0],  # grid_dim_0
            grid[1],  # grid_dim_1
            grid[2],  # grid_dim_2
            MERGED_BLOCK_SIZE,
            BLOCK_SIZE_LAST,
        )


@gluon.jit
def paged_attention_decode_v2_gluon_large_block_dot_kernel(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, num_kv_heads * query_length * query_group_size, 1]
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
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
    query_seq_len: int,
    query_group_size_original,
    head_size: int,
    num_seqs: int,
    num_kv_heads: int,
    max_context_partition_num: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_GROUP_SIZE_POW2: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    KV_COMPUTE_BLOCK_SIZE: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,  # [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    IS_CAUSAL: gl.constexpr,
    CDNA_VERSION: gl.constexpr,
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
    if KV_QUANT_MODE >= 0:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 16
    else:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 8
    QUERY_GROUP_SIZE = query_seq_len * query_group_size_original
    # ==================== Validation Checks ====================
    gl.static_assert(
        CONTEXT_PARTITION_SIZE == 256,
        f"CONTEXT_PARTITION_SIZE={CONTEXT_PARTITION_SIZE}, Only support CONTEXT_PARTITION_SIZE == 256",
    )
    gl.static_assert(
        KV_BLOCK_SIZE == 1024,
        f"KV_BLOCK_SIZE={KV_BLOCK_SIZE}, Only support KV_BLOCK_SIZE == 1024",
    )

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

    # ==================== Constants and Configuration ====================
    if COMPUTE_TYPE.is_fp8():
        OUTPUT_DTYPE: gl.constexpr = tl.bfloat16
    else:
        OUTPUT_DTYPE: gl.constexpr = COMPUTE_TYPE
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
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])

    # Key cache layout - optimized for CDNA3 architecture
    blocked_key_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, CONTIGUOUS_KV_ELEMENTS_16B_LOAD],
        threads_per_warp=[4, 16, 1],
        warps_per_cta=[1, 4, 1],
        order=[2, 1, 0],
    )

    # QK matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
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
        version=CDNA_VERSION,
        instr_shape=[16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
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
        page_offset = 1 * CONTEXT_PARTITION_SIZE
    elif sequence_partition_idx % 4 == 2:
        page_offset = 2 * CONTEXT_PARTITION_SIZE
    elif sequence_partition_idx % 4 == 3:
        page_offset = 3 * CONTEXT_PARTITION_SIZE

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
        head_size_offsets[None, :] < head_size
    )
    # Load query tensor [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=query_offsets_base, mask=query_mask
    )
    query_shared = gl.allocate_shared_memory(
        query_tensor.dtype, query_tensor.shape, shared_query_layout, query_tensor
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
        output_head_size_offsets[None, :] < head_size
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
    max_logits = max_logits_base_offsets.to(gl.float32) * float(0.0) - float("inf")
    exp_sums = max_logits_base_offsets.to(gl.float32) * float(0.0)
    attention_accumulator = gl.zeros(
        (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=gl.float32, layout=pv_mfma_layout
    )

    # ==================== Sequence Length Handling ====================
    context_length = gl.load(context_lengths_ptr + sequence_idx)
    kv_sequence_start_index = sequence_partition_idx * CONTEXT_PARTITION_SIZE
    # Early return if this partition is beyond sequence length
    if kv_sequence_start_index >= context_length:
        return

    KV_COMPUTE_BLOCK_COUNT: gl.constexpr = (
        CONTEXT_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE
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

        # ==================== Block Table Lookup ====================
        block_tables_start_ptr = (
            block_tables_ptr + sequence_idx * stride_block_table_seq
        )
        kv_page_id = tl.load(block_tables_start_ptr + block_table_id)
        kv_page_id = kv_page_id.to(gl.int64)

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
        key_block = gl.load(key_cache_ptr + key_block_offsets)

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
                key_scale_value = gl.load(key_scale + key_scale_offsets)
                value_scale_value = gl.load(value_scale + key_scale_offsets)

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
            value_block = gl.load(value_cache_ptr + value_block_offsets)
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
            value_block = gl.load(value_cache_ptr + value_block_offsets)
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
        # query_converted = query_shared.load(qk_lhs_layout)
        key_converted = gl.convert_layout(key_block, layout=qk_rhs_layout)
        query_converted = query_converted.to(COMPUTE_TYPE)
        key_converted = key_converted.to(COMPUTE_TYPE)

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

        # ==================== Attention Masking ====================
        # Create boundary mask for valid query groups
        boundary_mask = qk_row_offsets[:, None] < QUERY_GROUP_SIZE

        # Apply causal masking if required
        if IS_CAUSAL:
            sequence_extension = (
                query_seq_len - 1 - qk_row_offsets // query_group_size_original
            )
            causal_mask = (
                sequence_extension[:, None] + qk_column_offsets[None, :]
                < context_length
            )
        else:
            causal_mask = qk_column_offsets[None, :] < context_length

        # Combine masks
        combined_mask = boundary_mask & causal_mask

        # Apply masking to QK scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)
        qk_matrix = tl.where(combined_mask, qk_matrix, float(-3.4e38))

        # ==================== Softmax Computation ====================
        # Compute new maximum logits
        current_max_logits = gl.max(qk_matrix, axis=1)
        new_max_logits = gl.maximum(max_logits, current_max_logits)

        # Compute scaling factor for numerical stability
        accumulator_scale = tl.math.exp2((max_logits - new_max_logits) * LOG2_E)

        # Compute attention probabilities
        attention_probs = tl.math.exp2((qk_matrix - new_max_logits[:, None]) * LOG2_E)
        exp_sums = accumulator_scale * exp_sums + gl.sum(attention_probs, axis=1)

        # ==================== Value Scaling for FP8 ====================
        if value_block.dtype.is_fp8():
            if KV_QUANT_MODE == 1:
                # Per-token quantization scaling
                # Create mask for valid tokens
                valid_token_mask = qk_column_offsets < context_length
                # Mask out value_scale of invalid tokens
                value_scale_value = tl.where(
                    valid_token_mask, value_scale_value, float(0.0)
                )
                value_scale_max = gl.max(value_scale_value, axis=0)
                # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                value_scale_value = (
                    value_scale_value * float(FP8_MAX_VALUE) / (value_scale_max + 1e-8)
                )
                attention_probs = value_scale_value[None, :] * attention_probs
                probability_scale = value_scale_max / float(FP8_MAX_VALUE)
            elif KV_QUANT_MODE == 0:
                attention_probs *= float(FP8_MAX_VALUE)
                probability_scale = value_scale_value / float(FP8_MAX_VALUE)
            else:
                raise ValueError(f"Invalid KV_QUANT_MODE: {KV_QUANT_MODE}")

        # Convert attention probabilities to compute type for MFMA operation
        attention_probs = attention_probs.to(COMPUTE_TYPE)

        # ==================== PV Matrix Multiplication ====================
        # Convert layouts for MFMA operation
        attention_probs_converted = gl.convert_layout(
            attention_probs, layout=pv_lhs_layout
        )
        values_converted = gl.convert_layout(value_block, layout=pv_rhs_layout)
        values_converted = values_converted.to(COMPUTE_TYPE)

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
            attention_probs_converted, values_converted, pv_accumulator
        )
        if KV_QUANT_MODE >= 0:
            attention_accumulator += probability_scale * attention_output
        else:
            attention_accumulator += attention_output

        # Update maximum logits for next iteration
        max_logits = new_max_logits

    # ==================== Final Output Scaling and Storage ====================
    # Compute final exponential sums
    exp_sums_reciprocal = 1.0 / exp_sums
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )

    # Apply final scaling to attention accumulator
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt
    attention_accumulator = attention_accumulator.to(OUTPUT_DTYPE)

    # Store results to output buffers
    gl.amd.cdna3.buffer_store(
        stored_value=max_logits,
        ptr=max_logits_ptr,
        offsets=max_logits_offsets,
        mask=max_logits_group_mask,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=exp_sums,
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


@gluon.jit
def paged_attention_decode_sliding_window(
    output_ptr,  # [num_seqs, num_kv_heads, query_group_size, head_size]
    query_ptr,  # [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale: float,
    query_scale,  # [num_seqs, num_kv_heads * query_length * query_group_size, 1]
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    sinks_ptr,  # [num_query_heads]
    stride_output_seq: int,
    stride_output_head: int,
    stride_query_seq: int,
    stride_query_head: int,
    stride_key_block: int,
    stride_key_head: int,
    stride_key_head_split: int,
    stride_key_block_elem: int,
    stride_value_block: int,
    stride_value_head: int,
    stride_value_head_size: int,
    stride_block_table_seq: int,
    query_scale_stride_0: int,
    kv_scale_stride_0: int,
    kv_scale_stride_1: int,
    query_seq_len: int,
    query_group_size_original: int,
    head_size: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_GROUP_SIZE_POW2: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,  # [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    IS_CAUSAL: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    SLIDING_WINDOW: gl.constexpr = 0,
    CDNA_VERSION: gl.constexpr = 3,
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
        context_lengths_ptr: Pointer to sequence lengths for each sequence
        softmax_scale: Scaling factor for softmax
        query_scale: Query quantization scales
        key_scale: Key quantization scales
        value_scale: Value quantization scales
        Various stride parameters for tensor access
        Compile-time constants for kernel configuration

    Note:
        This kernel uses AMD CDNA3 MFMA instructions for efficient matrix operations
        and supports both FP8 and BF16 data types with various quantization modes.
    """

    if KV_QUANT_MODE >= 0:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 16
    else:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 8

    query_group_size = query_seq_len * query_group_size_original
    # ==================== VALIDATION CHECKS ====================
    gl.static_assert(
        KV_BLOCK_SIZE == 16 or KV_BLOCK_SIZE == 64,
        f"KV_BLOCK_SIZE={KV_BLOCK_SIZE}, Only support KV_BLOCK_SIZE in [16, 64]",
    )

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
        OUTPUT_DTYPE: gl.constexpr = tl.bfloat16
    else:
        OUTPUT_DTYPE: gl.constexpr = COMPUTE_TYPE
    LOG2_E: gl.constexpr = 1.4426950408889634  # log2(e) for exponential conversion
    CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD: gl.constexpr = KV_16B_ELEMENT_COUNT

    K_HEAD_SIZE_SPLITS: gl.constexpr = (
        HEAD_SIZE_POW2 // CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD
    )
    MAX_NUM_KV_BLOCKS_PER_COMPUTE: gl.constexpr = (
        CONTEXT_PARTITION_SIZE // KV_BLOCK_SIZE
    )

    # ==================== MEMORY LAYOUT DEFINITIONS ====================
    # Query tensor layout - optimized for sequential access
    blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])

    # Key cache layout - optimized for block-wise access patterns
    blocked_key_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=[4, 1, 1, 1],
        order=[3, 2, 1, 0],
    )

    # QK Matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
    )
    qk_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=16
    )
    qk_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=16
    )

    # Register allocation configuration based on group size and compute block size
    if QUERY_GROUP_SIZE_POW2 == 16:
        if CONTEXT_PARTITION_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64))
        elif CONTEXT_PARTITION_SIZE == 256:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (0, 128))
    elif QUERY_GROUP_SIZE_POW2 == 32:
        if CONTEXT_PARTITION_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (16, 0))
        elif CONTEXT_PARTITION_SIZE == 256:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (0, 128), (16, 0))
    elif QUERY_GROUP_SIZE_POW2 == 64:
        if CONTEXT_PARTITION_SIZE == 128:
            register_bases: gl.constexpr = ((0, 1), (0, 2), (0, 64), (16, 0), (32, 0))
        elif CONTEXT_PARTITION_SIZE == 256:
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
        shape=[QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE],
    )

    # Value cache layout configuration based on transpose flag
    if VALUE_TRANSPOSED:
        # Transposed value layout for better memory access patterns
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD],
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
            size_per_thread=[1, 1, CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD],
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
        version=CDNA_VERSION,
        instr_shape=[16, 16],
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

    context_length = gl.load(context_lengths_ptr + sequence_idx)
    # Load query tensor with appropriate masking
    query_offsets_base = (
        sequence_idx * stride_query_seq
        + (kv_head_idx * query_group_size + query_group_offsets[:, None])
        * stride_query_head
        + head_size_offsets[None, :]
    )
    query_mask = (query_group_offsets[:, None] < query_group_size) & (
        head_size_offsets[None, :] < head_size
    )

    query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=query_offsets_base, mask=query_mask
    )
    query_shared = gl.allocate_shared_memory(
        query_tensor.dtype, query_tensor.shape, shared_query_layout, query_tensor
    )
    # Load query quantization scales if needed
    if QUERY_QUANT_MODE == 0:
        # Per-tensor quantization
        query_scale_value = gl.load(query_scale)
    elif QUERY_QUANT_MODE == 1:
        # Per-token quantization
        query_scale_offsets = (
            sequence_idx * query_scale_stride_0
            + kv_head_idx * query_group_size
            + qk_row_offsets[:, None]
        )
        query_scale_value = gl.amd.cdna3.buffer_load(
            ptr=query_scale,
            offsets=query_scale_offsets,
            mask=qk_row_offsets[:, None] < query_group_size,
        )

    output_group_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )
    output_mask = (output_group_offsets[:, None] < query_group_size) & (
        output_head_size_offsets[None, :] < head_size
    )

    output_offsets = (
        sequence_idx * stride_output_seq
        + (kv_head_idx * query_group_size + output_group_offsets[:, None])
        * stride_output_head
        + output_head_size_offsets[None, :]
    )

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
        (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=gl.float32, layout=pv_mfma_layout
    )

    # ==================== SEQUENCE PROCESSING ====================
    query_converted = query_shared.load(qk_lhs_operand_layout)

    if SLIDING_WINDOW > 0:
        sequence_partition_start_idx = (
            context_length - SLIDING_WINDOW
        ) // CONTEXT_PARTITION_SIZE
    else:
        sequence_partition_start_idx = 0
    sequence_partition_end_idx = gl.cdiv(context_length, CONTEXT_PARTITION_SIZE)

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

    for sequence_partition_idx in range(
        sequence_partition_start_idx, sequence_partition_end_idx
    ):
        kv_sequence_start_idx = sequence_partition_idx * CONTEXT_PARTITION_SIZE
        # Process KV sequence in compute blocks
        kv_sequence_end_idx = gl.minimum(
            kv_sequence_start_idx + CONTEXT_PARTITION_SIZE, context_length
        )

        num_kv_blocks = gl.cdiv(
            kv_sequence_end_idx - kv_sequence_start_idx, KV_BLOCK_SIZE
        )
        kv_block_start_idx = sequence_partition_idx * MAX_NUM_KV_BLOCKS_PER_COMPUTE
        qk_column_offsets = kv_block_start_idx * KV_BLOCK_SIZE + gl.arange(
            0, CONTEXT_PARTITION_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
        )
        # Load KV block indices from block table
        block_indices = gl.arange(
            0, MAX_NUM_KV_BLOCKS_PER_COMPUTE, layout=block_id_layout
        )
        # Create mask for valid blocks
        valid_block_mask = block_indices < num_kv_blocks
        masked_block_indices = gl.where(valid_block_mask, block_indices, 0)
        block_table_start_ptr = block_tables_ptr + sequence_idx * stride_block_table_seq
        kv_block_numbers = gl.amd.cdna3.buffer_load(
            ptr=block_table_start_ptr + kv_block_start_idx, offsets=masked_block_indices
        ).to(gl.int64)

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

        # Optimize: Start key load, then prepare QK MFMA accumulators/query (overlaps with key load)
        key_tensor = gl.load(key_cache_ptr + key_block_offsets)
        # Prepare QK MFMA while key loads (these don't depend on key data)
        qk_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE),
            dtype=gl.float32,
            layout=qk_mfma_layout,
        )
        # Load key quantization scales if needed (overlaps with key tensor load)
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 0:
                # Per-tensor quantization
                key_scale_value = tl.load(key_scale)
                value_scale_value = tl.load(value_scale)
            elif KV_QUANT_MODE == 1:
                # Per-token quantization - prepare offsets while key loads
                key_scale_offsets = (
                    kv_block_numbers[:, None, None, None] * kv_scale_stride_0
                    + kv_head_idx * kv_scale_stride_1
                    + block_element_offsets[None, None, :, None]
                )
                # Optimize: Load both scales with VMEM scheduling, overlap with key reshape
                key_scale_value_blocked = gl.amd.cdna3.buffer_load(
                    ptr=key_scale, offsets=key_scale_offsets
                )
                value_scale_value_blocked = gl.amd.cdna3.buffer_load(
                    ptr=value_scale, offsets=key_scale_offsets
                )

                # Convert to required distributed layout for computation
                key_scale_value_blocked = gl.reshape(
                    key_scale_value_blocked, [CONTEXT_PARTITION_SIZE]
                )
                key_scale_value = gl.convert_layout(
                    key_scale_value_blocked, layout=gl.SliceLayout(0, qk_linear_layout)
                )
                key_scale_value = key_scale_value[None, :]
                value_scale_value_blocked = gl.reshape(
                    value_scale_value_blocked, [CONTEXT_PARTITION_SIZE]
                )
                value_scale_value = gl.convert_layout(
                    value_scale_value_blocked,
                    layout=gl.SliceLayout(0, qk_linear_layout),
                )

        # Reshape key tensor for matrix multiplication
        key_tensor = gl.permute(key_tensor, [1, 3, 0, 2])
        key_tensor = gl.reshape(key_tensor, [HEAD_SIZE_POW2, CONTEXT_PARTITION_SIZE])

        # ==================== VALUE LOADING WITH QK MFMA OVERLAP ====================
        # Convert key layout for MFMA (query_converted and qk_accumulator already prepared above)
        key_converted = gl.convert_layout(key_tensor, layout=qk_rhs_operand_layout)
        key_converted = key_converted.to(COMPUTE_TYPE)

        if VALUE_TRANSPOSED:
            # Load values from transposed cache layout
            kv_block_numbers_reshaped = gl.convert_layout(
                kv_block_numbers,
                layout=gl.SliceLayout(
                    1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
                ),
            )
            valid_block_mask = gl.convert_layout(
                valid_block_mask,
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
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)
            # Compute QK attention scores using MFMA (overlaps with value load)
            attention_scores = gl.amd.cdna3.mfma(
                query_converted, key_converted, qk_accumulator
            )

            # Permute and reshape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 1, 3, 2])
            value_tensor = gl.reshape(
                value_tensor, [CONTEXT_PARTITION_SIZE, HEAD_SIZE_POW2]
            )
        else:
            # Load values from standard cache layout
            kv_block_numbers_reshaped = gl.convert_layout(
                kv_block_numbers,
                layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout)),
            )
            valid_block_mask = gl.convert_layout(
                valid_block_mask,
                layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout)),
            )
            value_block_offsets = (
                kv_block_numbers_reshaped[:, None, None] * stride_value_block
                + kv_head_idx * stride_value_head
                + value_dim1_offsets[None, :, None] * stride_value_head_size
                + value_dim2_offsets[None, None, :]
            )

            # Schedule: Start value VMEM load, then QK MFMA
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)
            # Compute QK attention scores using MFMA (overlaps with value load)
            attention_scores = gl.amd.cdna3.mfma(
                query_converted, key_converted, qk_accumulator
            )

            # Permute and resape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 2, 1])
            value_tensor = gl.reshape(
                value_tensor, [CONTEXT_PARTITION_SIZE, HEAD_SIZE_POW2]
            )

        attention_scores = gl.reshape(
            attention_scores, [QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE]
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
        # Create boundary mask for valid sequence positions
        boundary_mask = qk_row_offsets[:, None] < query_group_size

        # Compute query token index (0 to query_seq_len-1)
        query_token_idx = qk_row_offsets // query_group_size_original

        # Query positions: queries are AFTER the KV cache
        # query_pos = context_length + query_token_idx
        # kv_pos = qk_column_offsets

        # Apply causal masking if required
        if IS_CAUSAL:
            # Compute causal mask based on sequence positions
            sequence_position_extension = query_seq_len - 1 - query_token_idx
            causal_mask = (
                sequence_position_extension[:, None] + qk_column_offsets[None, :]
                < context_length
            )
        else:
            causal_mask = qk_column_offsets[None, :] < context_length

        boundary_mask = boundary_mask & causal_mask

        # Apply sliding window mask
        if SLIDING_WINDOW > 0:
            # Sliding window: keep only KV tokens within SLIDING_WINDOW distance from query position
            # query_pos - kv_pos < SLIDING_WINDOW
            # (context_length + query_token_idx) - qk_column_offsets < SLIDING_WINDOW
            # OR: qk_column_offsets > context_length + query_token_idx - SLIDING_WINDOW
            sliding_window_mask = (
                qk_column_offsets[None, :]
                > context_length + query_token_idx[:, None] - SLIDING_WINDOW
            )
            boundary_mask = boundary_mask & sliding_window_mask
        # Apply masking to attention scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)

        attention_scores = tl.where(boundary_mask, attention_scores, float(-3.4e38))
        # ==================== SOFTMAX COMPUTATION ====================
        # Update running maximum for numerical stability
        current_max_logits = gl.max(attention_scores, axis=1)
        new_max_logits = gl.maximum(max_logits, current_max_logits)
        accumulator_scale = tl.math.exp2((max_logits - new_max_logits) * LOG2_E)
        # Compute attention probabilities
        attention_probs = tl.math.exp2(
            (attention_scores - new_max_logits[:, None]) * LOG2_E
        )
        # exp_sums = gl.sum(attention_probs, axis=1)
        exp_sums = accumulator_scale * exp_sums + gl.sum(attention_probs, axis=1)
        # ==================== VALUE ACCUMULATION ====================
        # Handle value quantization scaling for FP8
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # Per-token quantization scaling
                # Create mask for valid tokens
                valid_token_mask = qk_column_offsets < context_length
                # Mask out value_scale of invalid tokens
                value_scale_value = gl.where(
                    valid_token_mask, value_scale_value, float(0.0)
                )
                value_scale_max = gl.max(value_scale_value, axis=0)
                # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                value_scale_value = (
                    value_scale_value * float(FP8_MAX_VALUE) / (value_scale_max + 1e-8)
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
        values_converted = gl.convert_layout(value_tensor, layout=pv_rhs_operand_layout)
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

    if sinks_ptr is not None:
        sinks_values = gl.load(
            sinks_ptr + (kv_head_idx * query_group_size + query_group_offsets),
            mask=query_group_offsets < query_group_size,
        )
        exp_sums += gl.exp(
            gl.convert_layout(sinks_values, layout=max_logits.type.layout) - max_logits
        )
    # ==================== OUTPUT NORMALIZATION AND STORING ====================
    # Normalize attention output by softmax denominator

    exp_sums_reciprocal = 1.0 / exp_sums
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt
    gl.amd.cdna3.buffer_store(
        stored_value=attention_accumulator.to(OUTPUT_DTYPE),
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
def paged_attention_decode_v2_gluon_dot_kernel(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, num_kv_heads * query_length * query_group_size, 1]
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    stride_max_logits_seq: int,
    stride_max_logits_head: int,
    stride_max_logits_part: int,
    stride_output_seq: int,
    stride_output_head: int,
    stride_output_part: int,
    stride_output_group: int,
    stride_query_seq: int,
    stride_query_head: int,
    stride_key_block: int,
    stride_key_head: int,
    stride_key_head_split: int,
    stride_key_block_elem: int,
    stride_value_block: int,
    stride_value_head: int,
    stride_value_head_size: int,
    stride_block_table_seq: int,
    query_scale_stride_0: int,
    kv_scale_stride_0: int,
    kv_scale_stride_1: int,
    query_seq_len: int,
    query_group_size_original: int,
    head_size: int,
    num_seqs: int,
    num_kv_heads: int,
    max_context_partition_num: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_GROUP_SIZE_POW2: gl.constexpr,
    HEAD_SIZE_POW2: gl.constexpr,
    KV_BLOCK_SIZE: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    KV_COMPUTE_BLOCK_SIZE: gl.constexpr,
    QUERY_QUANT_MODE: gl.constexpr,
    KV_QUANT_MODE: gl.constexpr,
    FP8_MAX_VALUE: gl.constexpr,
    VALUE_TRANSPOSED: gl.constexpr,  # [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    IS_CAUSAL: gl.constexpr,
    CDNA_VERSION: gl.constexpr = 3,
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
        context_lengths_ptr: Pointer to sequence lengths for each sequence
        softmax_scale: Scaling factor for softmax
        query_scale: Query quantization scales
        key_scale: Key quantization scales
        value_scale: Value quantization scales
        Various stride parameters for tensor access
        Compile-time constants for kernel configuration

    Note:
        This kernel uses AMD CDNA3 MFMA instructions for efficient matrix operations
        and supports both FP8 and BF16 data types with various quantization modes.
    """

    if KV_QUANT_MODE >= 0:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 16
    else:
        KV_16B_ELEMENT_COUNT: gl.constexpr = 8
    QUERY_GROUP_SIZE = query_seq_len * query_group_size_original
    # ==================== VALIDATION CHECKS ====================
    gl.static_assert(
        KV_BLOCK_SIZE == 16 or KV_BLOCK_SIZE == 64,
        f"KV_BLOCK_SIZE={KV_BLOCK_SIZE}, Only support KV_BLOCK_SIZE in [16, 64]",
    )

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
        OUTPUT_DTYPE: gl.constexpr = tl.bfloat16
    else:
        OUTPUT_DTYPE: gl.constexpr = COMPUTE_TYPE
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
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])

    # Key cache layout - optimized for block-wise access patterns
    blocked_key_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=[4, 1, 1, 1],
        order=[3, 2, 1, 0],
    )

    # QK Matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=[16, 16],
        transposed=True,
        warps_per_cta=[1, 4],
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
            size_per_thread=[1, 1, 1, CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD],
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
            size_per_thread=[1, 1, CONTIGUOUS_KV_ELEMENTS_PER_16B_LOAD],
            threads_per_warp=[4, 16, 1],
            warps_per_cta=[1, 4, 1],
            order=[2, 1, 0],
        )
        # blocked_value_layout: gl.constexpr = gl.DistributedLinearLayout(
        #     reg_bases=((0,0,1), (0,0,2), (0,0,4), (0,0,8), (4,0,0), (8,0,0), (0,64,0)),
        #     lane_bases=((0,1,0), (0,2,0), (0,4,0), (0,8,0), (1,0,0), (2,0,0)),
        #     warp_bases=((0,16,0), (0,32,0)),
        #     block_bases=[],
        #     shape=[16, 128, 16],
        # )
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
        instr_shape=[16, 16],
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
        head_size_offsets[None, :] < head_size
    )
    query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=query_offsets_base, mask=query_mask
    )
    query_shared = gl.allocate_shared_memory(
        query_tensor.dtype, query_tensor.shape, shared_query_layout, query_tensor
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
        output_head_size_offsets[None, :] < head_size
    )

    output_offsets = sequence_idx * stride_output_seq
    output_offsets += kv_head_idx * stride_output_head
    output_offsets += (
        sequence_partition_idx * stride_output_part
        + output_group_offsets[:, None] * stride_output_group
        + output_head_size_offsets[None, :]
    )

    # Initialize attention state variables
    max_logits = max_logits_base_offsets.to(gl.float32) * float(0.0) - float("inf")
    exp_sums = max_logits_base_offsets.to(gl.float32) * float(0.0)
    attention_accumulator = gl.zeros(
        (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=gl.float32, layout=pv_mfma_layout
    )

    # ==================== SEQUENCE PROCESSING ====================
    context_length = gl.load(context_lengths_ptr + sequence_idx)
    kv_sequence_start_idx = sequence_partition_idx * CONTEXT_PARTITION_SIZE
    if kv_sequence_start_idx >= context_length:
        return  # No computation needed for this partition

    KV_COMPUTE_BLOCK_COUNT: gl.constexpr = (
        CONTEXT_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE
    )
    SEQUENCE_PARTITION_KV_BLOCKS: gl.constexpr = CONTEXT_PARTITION_SIZE // KV_BLOCK_SIZE

    # Process KV sequence in compute blocks
    for kv_compute_idx in range(KV_COMPUTE_BLOCK_COUNT):
        kv_subsequence_start_idx = (
            kv_sequence_start_idx + kv_compute_idx * KV_COMPUTE_BLOCK_SIZE
        )
        kv_subsequence_end_idx = gl.minimum(
            kv_subsequence_start_idx + KV_COMPUTE_BLOCK_SIZE, context_length
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

        # Load KV block indices from block table
        block_indices = gl.arange(
            0, MAX_NUM_KV_BLOCKS_PER_COMPUTE, layout=block_id_layout
        )
        # Create mask for valid blocks
        valid_block_mask = block_indices < num_kv_blocks
        masked_block_indices = gl.where(valid_block_mask, block_indices, 0)
        block_table_start_ptr = block_tables_ptr + sequence_idx * stride_block_table_seq
        kv_block_numbers = gl.amd.cdna3.buffer_load(
            ptr=block_table_start_ptr + kv_block_start_idx, offsets=masked_block_indices
        )
        kv_block_numbers = kv_block_numbers.to(gl.int64)

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
        key_tensor = gl.load(key_cache_ptr + key_block_offsets)

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
                key_scale_value = gl.load(key_scale + key_scale_offsets)
                value_scale_value = gl.load(value_scale + key_scale_offsets)

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
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)
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
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)
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
        query_converted = query_shared.load(qk_lhs_operand_layout)
        key_converted = gl.convert_layout(key_tensor, layout=qk_rhs_operand_layout)

        query_converted = query_converted.to(COMPUTE_TYPE)
        key_converted = key_converted.to(COMPUTE_TYPE)

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

        # ==================== ATTENTION MASKING ====================
        # Create boundary mask for valid sequence positions
        boundary_mask = qk_row_offsets[:, None] < QUERY_GROUP_SIZE

        # Apply causal masking if required
        if IS_CAUSAL:
            # Compute causal mask based on sequence positions
            sequence_position_extension = (
                query_seq_len - 1 - qk_row_offsets // query_group_size_original
            )
            causal_mask = (
                sequence_position_extension[:, None] + qk_column_offsets[None, :]
                < context_length
            )
        else:
            causal_mask = qk_column_offsets[None, :] < context_length

        boundary_mask = boundary_mask & causal_mask

        # Apply masking to attention scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)
        attention_scores = tl.where(boundary_mask, attention_scores, float(-3.4e38))

        # ==================== SOFTMAX COMPUTATION ====================
        # Update running maximum for numerical stability
        current_max_logits = gl.max(attention_scores, axis=1)
        new_max_logits = gl.maximum(max_logits, current_max_logits)

        # Compute scaling factor for previous accumulator
        accumulator_scale = tl.math.exp2((max_logits - new_max_logits) * LOG2_E)

        # Compute attention probabilities
        attention_probs = tl.math.exp2(
            (attention_scores - new_max_logits[:, None]) * LOG2_E
        )
        exp_sums = accumulator_scale * exp_sums + gl.sum(attention_probs, axis=1)

        # ==================== VALUE ACCUMULATION ====================
        # Handle value quantization scaling for FP8
        if value_tensor.dtype.is_fp8():
            if KV_QUANT_MODE == 1:
                # Per-token quantization scaling
                # Create mask for valid tokens
                valid_token_mask = qk_column_offsets < context_length
                # Mask out value_scale of invalid tokens
                value_scale_value = tl.where(
                    valid_token_mask, value_scale_value, float(0.0)
                )
                value_scale_max = gl.max(value_scale_value, axis=0)
                # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                value_scale_value = (
                    value_scale_value * float(FP8_MAX_VALUE) / (value_scale_max + 1e-8)
                )
                attention_probs = value_scale_value[None, :] * attention_probs
                probability_scale = value_scale_max / float(FP8_MAX_VALUE)
            elif KV_QUANT_MODE == 0:
                # Per-tensor quantization scaling
                attention_probs *= float(FP8_MAX_VALUE)
                probability_scale = value_scale_value / float(FP8_MAX_VALUE)
            else:
                raise ValueError(f"Invalid KV_QUANT_MODE: {KV_QUANT_MODE}")

        # Convert attention probabilities to compute type for MFMA operation
        attention_probs = attention_probs.to(COMPUTE_TYPE)

        # Convert layouts for PV MFMA operation
        probs_converted = gl.convert_layout(
            attention_probs, layout=pv_lhs_operand_layout
        )
        values_converted = gl.convert_layout(value_tensor, layout=pv_rhs_operand_layout)
        values_converted = values_converted.to(COMPUTE_TYPE)

        # Scale previous accumulator and compute new attention output
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

        # Update running maximum for next iteration
        max_logits = new_max_logits

    # ==================== OUTPUT NORMALIZATION AND STORING ====================
    # Normalize attention output by softmax denominator
    exp_sums_reciprocal = 1.0 / exp_sums
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt
    attention_accumulator = attention_accumulator.to(OUTPUT_DTYPE)

    # Store results to global memory
    gl.amd.cdna3.buffer_store(
        stored_value=max_logits,
        ptr=max_logits_ptr,
        offsets=max_logits_offsets,
        mask=max_logits_group_mask,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=exp_sums,
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


@triton.jit
def paged_attention_decode_v2_reduce_kernel(
    output_ptr,  # [num_seqs, num_kv_heads, query_group_size, head_size]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size, head_size]
    context_lengths_ptr,  # [num_seqs]
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
    query_group_size,
    head_size,
    num_seqs,
    num_kv_heads,
    QUERY_GROUP_SIZE_POW2: tl.constexpr,
    HEAD_SIZE_POW2: tl.constexpr,
    CONTEXT_PARTITION_SIZE: tl.constexpr,
    USE_SINKS: tl.constexpr,
):
    """
    Triton reduction kernel for paged attention decode that combines partial results.

    This version uses a fixed MAX_CONTEXT_PARTITION_NUM=16 and loops through partitions
    in chunks to handle arbitrary numbers of context partitions.

    This kernel performs the final reduction by:
    1. Finding global maximum logits across partitions (first pass)
    2. Rescaling exponential sums for numerical stability (second pass)
    3. Computing normalized attention probabilities (second pass)
    4. Weighted summation of partial logits (second pass)

    Args:
        output_ptr: Output tensor for final attention results
        exp_sums_ptr: Exponential sums from partial computations
        max_logits_ptr: Maximum logits from partial computations
        logits_ptr: Partial logit tensors from each sequence partition
        context_lengths_ptr: Sequence lengths for each sequence
        Various stride parameters for tensor access
        Compile-time constants for kernel configuration (no MAX_CONTEXT_PARTITION_NUM needed)
    """
    # Mathematical constant for exponential calculations
    LOG2_E: tl.constexpr = 1.4426950408889634
    MAX_CONTEXT_PARTITION_NUM: tl.constexpr = 16

    # ==================== INITIALIZATION ====================
    sequence_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    context_length = tl.load(context_lengths_ptr + sequence_idx)
    context_partition_num = tl.cdiv(context_length, CONTEXT_PARTITION_SIZE)

    # Generate coordinate ranges
    query_group_offsets = tl.arange(0, QUERY_GROUP_SIZE_POW2)
    head_size_offsets = tl.arange(0, HEAD_SIZE_POW2)

    # Initialize global accumulation variables
    # if USE_SINKS:
    #     global_max = tl.load(
    #         sink_token_ptr + (kv_head_idx * query_group_size + query_group_offsets),
    #         mask=query_group_offsets < query_group_size,
    #         other=float("-inf"),
    #     ).to(tl.float32)
    # else:
    global_max = tl.full((QUERY_GROUP_SIZE_POW2,), float("-inf"), dtype=tl.float32)
    global_max_prev = global_max
    global_exp_sum = tl.zeros((QUERY_GROUP_SIZE_POW2,), dtype=tl.float32)
    final_output = tl.zeros((QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2), dtype=tl.float32)

    # Calculate number of iterations needed
    num_iterations = tl.cdiv(context_partition_num, MAX_CONTEXT_PARTITION_NUM)

    # ==================== FIRST PASS: FIND GLOBAL MAX ====================
    # Loop through partitions in chunks of MAX_CONTEXT_PARTITION_NUM
    for iter_idx in range(num_iterations):
        partition_base = iter_idx * MAX_CONTEXT_PARTITION_NUM
        partition_offsets = tl.arange(0, MAX_CONTEXT_PARTITION_NUM) + partition_base

        # Calculate offsets for exponential sums and max logits
        exp_sums_offsets = (
            sequence_idx * stride_exp_sums_seq
            + kv_head_idx * stride_exp_sums_head
            + partition_offsets[:, None] * stride_exp_sums_part
            + query_group_offsets[None, :]
        )

        # Create mask for valid partitions and query groups
        exp_sums_mask = (partition_offsets[:, None] < context_partition_num) & (
            query_group_offsets[None, :] < query_group_size
        )

        # Load maximum logits from current chunk of partitions
        max_logits = tl.load(
            max_logits_ptr + exp_sums_offsets, mask=exp_sums_mask, other=float("-inf")
        )
        exp_sums = tl.load(
            exp_sums_ptr + exp_sums_offsets, mask=exp_sums_mask, other=0.0
        )

        # Update global maximum logit
        chunk_max_logits = tl.max(max_logits, axis=0)
        global_max = tl.maximum(global_max, chunk_max_logits)
        # Compute update scale for exponential sums
        update_scale = tl.exp(global_max_prev - global_max)

        # Rescale exponential sums using global maximum for numerical stability
        exp_sums *= tl.exp(max_logits - global_max[None, :])
        # Update and accumulate global exponential sum
        global_exp_sum = update_scale * global_exp_sum + tl.sum(exp_sums, axis=0)
        global_max_prev = global_max

    if USE_SINKS:
        sink_token_values = gl.load(
            sink_token_ptr + (kv_head_idx * query_group_size + query_group_offsets),
            mask=query_group_offsets < query_group_size,
        )
        global_exp_sum += gl.exp(sink_token_values - global_max)
    # ==================== SECOND PASS: COMPUTE RESCALED EXP SUMS AND ACCUMULATE ====================
    for iter_idx in range(num_iterations):
        partition_base = iter_idx * MAX_CONTEXT_PARTITION_NUM
        partition_offsets = tl.arange(0, MAX_CONTEXT_PARTITION_NUM) + partition_base

        # Calculate offsets for exponential sums and max logits
        exp_sums_offsets = (
            sequence_idx * stride_exp_sums_seq
            + kv_head_idx * stride_exp_sums_head
            + partition_offsets[:, None] * stride_exp_sums_part
            + query_group_offsets[None, :]
        )

        # Create mask for valid partitions and query groups
        exp_sums_mask = (partition_offsets[:, None] < context_partition_num) & (
            query_group_offsets[None, :] < query_group_size
        )

        # Load maximum logits and exponential sums from current chunk
        max_logits = tl.load(
            max_logits_ptr + exp_sums_offsets, mask=exp_sums_mask, other=float("-inf")
        )
        # BUGFIX: Add other=0.0 to prevent loading undefined values for invalid partitions
        exp_sums = tl.load(
            exp_sums_ptr + exp_sums_offsets, mask=exp_sums_mask, other=0.0
        )

        # Rescale exponential sums using global maximum for numerical stability
        exp_sums *= tl.exp(max_logits - global_max[None, :])

        # ==================== ATTENTION PROBABILITY AND WEIGHTED SUMMATION ====================
        # Compute normalized attention probabilities for this chunk
        attention_probs = exp_sums / global_exp_sum[None, :]

        # Reshape probabilities for broadcasting with logits
        attention_probs = tl.reshape(
            attention_probs, (MAX_CONTEXT_PARTITION_NUM, QUERY_GROUP_SIZE_POW2, 1)
        )

        # Calculate offsets for loading partial logits
        logits_offsets = (
            sequence_idx * stride_logits_seq
            + kv_head_idx * stride_logits_head
            + partition_offsets[None, :, None] * stride_logits_part
            + query_group_offsets[:, None, None] * stride_logits_group
            + head_size_offsets[None, None, :]
        )

        # Create mask for valid logits access
        logits_mask = (partition_offsets[None, :] < context_partition_num) & (
            query_group_offsets[:, None] < query_group_size
        )

        # Load partial logits from current chunk of partitions
        partial_logits = tl.load(
            logits_ptr + logits_offsets, mask=logits_mask[:, :, None], other=0.0
        )

        # Permute to match the expected dimension order
        partial_logits = tl.permute(partial_logits, (1, 0, 2)).to(tl.float32)
        updated_output = partial_logits * attention_probs

        # Accumulate weighted sum of logits
        final_output += tl.sum(updated_output, axis=0)

    # ==================== FINAL OUTPUT STORING ====================
    # Calculate output tensor offsets
    output_offsets = (
        sequence_idx * stride_output_seq
        + (kv_head_idx * query_group_size + query_group_offsets[:, None])
        * stride_output_head
        + head_size_offsets[None, :]
    )

    # Create mask for valid output storage
    output_mask = (query_group_offsets[:, None] < query_group_size) & (
        head_size_offsets[None, :] < head_size
    )

    # Store final output to global memory
    tl.store(
        output_ptr + output_offsets,
        final_output.to(output_ptr.dtype.element_ty),
        mask=output_mask,
    )


def _paged_attention_decode_v2_with_dot_kernel_reshape_wrapper(
    grid,
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, num_kv_heads * query_length * query_group_size, 1]
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1]
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
    COMPUTE_TYPE,
    QUERY_SEQ_LEN,
    QUERY_GROUP_SIZE_ORIGINAL,
    HEAD_SIZE,
    KV_BLOCK_SIZE,
    KV_16B_ELEMENT_COUNT,
    CONTEXT_PARTITION_SIZE,
    QUERY_QUANT_MODE,
    KV_QUANT_MODE,
    FP8_MAX_VALUE,
    VALUE_TRANSPOSED,
    IS_CAUSAL,
    SLIDING_WINDOW,
    sinks_ptr,
    ONE_SHOT,
    CDNA_VERSION,
):
    """
    Wrapper function for paged attention decode kernel with dynamic kernel selection.

    This wrapper selects between different kernel implementations based on the
    configuration parameters and launches the appropriate kernel.

    Args:
        All parameters from the pa_decode_gluon function, plus kernel configuration
        parameters for Triton compilation and execution.
    """
    HEAD_SIZE_POW2 = triton.next_power_of_2(HEAD_SIZE)
    # Production path - select and launch appropriate kernel
    QUERY_GROUP_SIZE = QUERY_SEQ_LEN * QUERY_GROUP_SIZE_ORIGINAL
    KV_COMPUTE_BLOCK_SIZE = CONTEXT_PARTITION_SIZE
    waves_per_eu = 2
    if QUERY_GROUP_SIZE < 16:
        QUERY_GROUP_SIZE_POW2 = 16
    else:
        QUERY_GROUP_SIZE_POW2 = triton.next_power_of_2(QUERY_GROUP_SIZE)
    # Select kernel implementation based on block size
    if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE:
        # Use big block kernel for large block sizes
        paged_attention_kernel = paged_attention_decode_v2_gluon_large_block_dot_kernel
        if VALUE_TRANSPOSED:
            # Use smaller compute block size for better performance with transposed values
            KV_COMPUTE_BLOCK_SIZE = 128
    else:
        # Configure waves per EU based on query group size
        if QUERY_GROUP_SIZE_POW2 == 64:
            waves_per_eu = 3
        else:
            waves_per_eu = 4

        if ONE_SHOT:
            paged_attention_decode_sliding_window[(grid[0], grid[1], 1)](
                output_ptr,
                query_ptr,
                key_cache_ptr,
                value_cache_ptr,
                block_tables_ptr,
                context_lengths_ptr,
                softmax_scale,
                query_scale,
                key_scale,
                value_scale,
                sinks_ptr,
                stride_output_seq,
                stride_output_head,
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
                query_seq_len=QUERY_SEQ_LEN,
                query_group_size_original=QUERY_GROUP_SIZE_ORIGINAL,
                head_size=HEAD_SIZE,
                COMPUTE_TYPE=COMPUTE_TYPE,
                QUERY_GROUP_SIZE_POW2=QUERY_GROUP_SIZE_POW2,
                HEAD_SIZE_POW2=HEAD_SIZE_POW2,
                KV_BLOCK_SIZE=KV_BLOCK_SIZE,
                CONTEXT_PARTITION_SIZE=CONTEXT_PARTITION_SIZE,
                QUERY_QUANT_MODE=QUERY_QUANT_MODE,
                KV_QUANT_MODE=KV_QUANT_MODE,
                VALUE_TRANSPOSED=VALUE_TRANSPOSED,
                IS_CAUSAL=IS_CAUSAL,
                FP8_MAX_VALUE=FP8_MAX_VALUE,
                SLIDING_WINDOW=SLIDING_WINDOW,
                CDNA_VERSION=CDNA_VERSION,
                waves_per_eu=waves_per_eu,
                num_stages=1,
            )
            return

        # Use standard kernel for normal block sizes
        paged_attention_kernel = paged_attention_decode_v2_gluon_dot_kernel

    # Launch the selected kernel
    paged_attention_kernel[grid](
        exp_sums_ptr,
        max_logits_ptr,
        output_ptr,
        query_ptr,
        key_cache_ptr,
        value_cache_ptr,
        block_tables_ptr,
        context_lengths_ptr,
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
        query_seq_len=QUERY_SEQ_LEN,
        query_group_size_original=QUERY_GROUP_SIZE_ORIGINAL,
        head_size=HEAD_SIZE,
        num_seqs=grid[0],
        num_kv_heads=grid[1],
        max_context_partition_num=grid[2],
        COMPUTE_TYPE=COMPUTE_TYPE,
        QUERY_GROUP_SIZE_POW2=QUERY_GROUP_SIZE_POW2,
        HEAD_SIZE_POW2=HEAD_SIZE_POW2,
        KV_BLOCK_SIZE=KV_BLOCK_SIZE,
        CONTEXT_PARTITION_SIZE=CONTEXT_PARTITION_SIZE,
        KV_COMPUTE_BLOCK_SIZE=KV_COMPUTE_BLOCK_SIZE,
        QUERY_QUANT_MODE=QUERY_QUANT_MODE,
        KV_QUANT_MODE=KV_QUANT_MODE,
        FP8_MAX_VALUE=FP8_MAX_VALUE,
        VALUE_TRANSPOSED=VALUE_TRANSPOSED,
        IS_CAUSAL=IS_CAUSAL,
        CDNA_VERSION=CDNA_VERSION,
        waves_per_eu=waves_per_eu,
        num_stages=1,
    )


def _paged_attention_decode_v2_reduce_kernel_wrapper(
    grid,
    output_ptr,  # [num_seqs, num_kv_heads, query_group_size, head_size]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size, head_size]
    context_lengths_ptr,  # [num_seqs]
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
    QUERY_GROUP_SIZE,
    HEAD_SIZE,
    MAX_CONTEXT_PARTITION_NUM,
    CONTEXT_PARTITION_SIZE,
):
    """
    Wrapper function for paged attention reduction kernel with kernel selection.

    This wrapper selects between Gluon and Triton kernel implementations
    based on configuration and launches the appropriate kernel.

    Args:
        All parameters from the reduction kernel plus execution grid configuration
    """
    # Configuration flag for kernel selection
    if QUERY_GROUP_SIZE < 16:
        QUERY_GROUP_SIZE_POW2 = 16
    else:
        QUERY_GROUP_SIZE_POW2 = triton.next_power_of_2(QUERY_GROUP_SIZE)

    kernel = paged_attention_decode_v2_reduce_kernel
    # Launch standard Triton reduction kernel
    kernel[grid](
        output_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        logits_ptr,
        context_lengths_ptr,
        sink_token_ptr,
        stride_output_seq,
        stride_output_head,
        stride_exp_sums_seq,
        stride_exp_sums_head,
        stride_exp_sums_part,
        stride_logits_seq,
        stride_logits_head,
        stride_logits_part,
        stride_logits_group,
        query_group_size=QUERY_GROUP_SIZE,
        head_size=HEAD_SIZE,
        num_seqs=grid[0],
        num_kv_heads=grid[1],
        QUERY_GROUP_SIZE_POW2=QUERY_GROUP_SIZE_POW2,
        HEAD_SIZE_POW2=triton.next_power_of_2(HEAD_SIZE),
        CONTEXT_PARTITION_SIZE=CONTEXT_PARTITION_SIZE,
        USE_SINKS=sink_token_ptr is not None,
    )


def pa_decode_gluon(
    output: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    output_gluon: torch.Tensor,  # [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    query: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    query_gluon: torch.Tensor,  # [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
    query_scale_gluon: torch.Tensor,  # [num_seqs, num_kv_heads * query_length * query_group_size, 1] or [1]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_length: int,
    max_context_length: int,
    context_partition_size: int,
    compute_type: tl.dtype,
    query_scale: torch.Tensor,  # [num_seqs * query_length, num_query_heads, 1] or [1]
    key_scale: torch.Tensor,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    exp_sums: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    max_logits: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    temporary_output: torch.Tensor,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
    alibi_slopes: torch.Tensor = None,
    sinks: torch.Tensor = None,
    sliding_window: int = 0,
    one_shot=None,
) -> None:
    """
    Paged Attention Decode with FP8/BF16/FP16 Support.

    Implements the attention mechanism for transformer decoding with paged KV caches,
    supporting various quantization schemes and data types. This function performs
    attention computation in two phases: a partitioned attention kernel followed
    by a reduction kernel.

    Parameters
    ----------
    output : torch.Tensor
        Output tensor for final attention results.
        - Shape: [num_seqs * query_length, num_query_heads, head_size]
        - Dtype: torch.bfloat16, torch.float16

    output_gluon : torch.Tensor
        Intermediate output tensor in gluon layout for internal computation.
        - Shape: [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
        - Dtype: torch.bfloat16, torch.float16 (same as output)

    query : torch.Tensor
        Input query tensor in standard layout.
        - Shape: [num_seqs * query_length, num_query_heads, head_size]
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16

    query_gluon : torch.Tensor
        Query tensor in gluon layout for internal computation.
        - Shape: [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16 (same as query)

    query_scale_gluon : torch.Tensor
        Quantization scales for query in gluon layout.
        - Shape: [1] (per-tensor) or [num_seqs, num_kv_heads * query_length * query_group_size, 1] (per-token)
        - Dtype: torch.float32

    key_cache : torch.Tensor
        Paged key cache in block layout with interleaved head dimension.
        - Shape: [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
          where x = 16 // dtype.itemsize (e.g., x=16 for fp8, x=8 for bf16/fp16)
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16

    value_cache : torch.Tensor
        Paged value cache in block layout. Supports two layouts:
        - Non-transposed shape: [num_blocks, num_kv_heads, head_size, kv_block_size]
        - Transposed shape: [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
          where x = 16 // dtype.itemsize
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16

    context_lengths : torch.Tensor
        Current context lengths (KV cache lengths) for each sequence.
        - Shape: [num_seqs]
        - Dtype: torch.int32

    block_tables : torch.Tensor
        Mapping from sequences to physical cache block indices.
        - Shape: [num_seqs, max_num_blocks_per_seq]
        - Dtype: torch.int32

    softmax_scale : float
        Scaling factor for attention scores, typically 1/sqrt(head_size).

    query_length : int
        Length of query sequences. Must be <= 4.

    max_context_length : int
        Maximum sequence length supported in the KV cache.

    context_partition_size : int
        Size of each context partition for partitioned attention computation.

    compute_type : tl.dtype
        Triton data type for computation.
        - Supported: tl.float8e4b8, tl.bfloat16, tl.float16

    query_scale : torch.Tensor
        Quantization scales for queries in standard layout. Required for FP8 queries.
        - Shape: [1] (per-tensor) or [num_seqs * query_length, num_query_heads, 1] (per-token)
        - Dtype: torch.float32

    key_scale : torch.Tensor
        Quantization scales for keys. Required for FP8 keys.
        - Shape: [1] (per-tensor) or [num_blocks, num_kv_heads, kv_block_size, 1] (per-token)
        - Dtype: torch.float32

    value_scale : torch.Tensor
        Quantization scales for values. Must have same shape as key_scale.
        - Shape: [1] (per-tensor) or [num_blocks, num_kv_heads, kv_block_size, 1] (per-token)
        - Dtype: torch.float32

    exp_sums : torch.Tensor
        Buffer for exponential sums used in online softmax computation.
        - Shape: [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
          where max_context_partition_num = ceil(max_context_length / context_partition_size)
        - Dtype: torch.float32

    max_logits : torch.Tensor
        Buffer for maximum logits used in online softmax computation.
        - Shape: [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
        - Dtype: torch.float32

    temporary_output : torch.Tensor
        Buffer for partial attention outputs from each context partition.
        - Shape: [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
        - Dtype: torch.float32

    alibi_slopes : torch.Tensor, optional
        ALiBi (Attention with Linear Biases) slopes for positional encoding.
        - Shape: [num_query_heads]
        - Dtype: torch.float32
        - Default: None (no ALiBi)

    Returns
    -------
    None
        Results are written directly to the output tensor.

    Notes
    -----
    - query_length * query_group_size must be <= 64
    - kv_block_size must be one of [16, 64, 1024]
    - When query_length > 1, automatic transpose operations are performed
      between standard and gluon layouts
    - For FP8 computation, query_scale and key_scale/value_scale are required
    - For BF16/FP16 computation, scales can be None
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
    num_query_heads = query.shape[1]
    head_size = query.shape[-1]
    batch_size = query.shape[0] // query_length
    num_kv_heads = key_cache.shape[1]
    query_group_size = num_query_heads // num_kv_heads

    if query_length > 1:
        # Transpose query and query_scale from [num_seqs * query_length, num_query_heads, head_size]
        # to [num_seqs, num_kv_heads * query_length * query_group_size, head_size]
        transpose_query_gluon(
            query=query,
            query_gluon=query_gluon,
            query_scale=query_scale,
            query_scale_gluon=query_scale_gluon,
            batch_size=batch_size,
            query_sequence_length=query_length,
            num_kv_heads=num_kv_heads,
            query_group_size=query_group_size,
            head_size=head_size,
        )

    num_sequences = batch_size
    num_query_heads_total = num_query_heads
    max_context_partition_num = int(
        (max_context_length + context_partition_size - 1) // context_partition_size
    )
    head_size = query.shape[-1]
    kv_block_size = key_cache.shape[-2]
    query_group_size = num_query_heads_total // num_kv_heads

    # Calculate equivalent group sizes for kernel configuration
    equivalent_query_group_size = query_length * query_group_size

    # Determine if causal masking is needed
    is_causal = query_length > 1

    # Calculate elements per 16B load based on data type
    kv_elements_per_16b = 16 // key_cache.dtype.itemsize

    # Configure execution grid
    grid = (num_sequences, num_kv_heads, max_context_partition_num)

    assert query_length <= 4, f"query_length == {query_length} exceeds maximum of 4"
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
    if one_shot is None:
        one_shot = sliding_window > 0
    _paged_attention_decode_v2_with_dot_kernel_reshape_wrapper(
        grid,
        exp_sums,
        max_logits,
        output_gluon if one_shot else temporary_output,
        query_gluon,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        softmax_scale,
        query_scale_gluon,
        key_scale,
        value_scale,
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        (output_gluon if one_shot else temporary_output).stride(0),
        (output_gluon if one_shot else temporary_output).stride(1),
        temporary_output.stride(2),
        temporary_output.stride(3),
        query_gluon.stride(0),
        query_gluon.stride(1),
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
        COMPUTE_TYPE=compute_type,
        QUERY_SEQ_LEN=query_length,
        HEAD_SIZE=head_size,
        QUERY_GROUP_SIZE_ORIGINAL=query_group_size,
        KV_BLOCK_SIZE=kv_block_size,
        KV_16B_ELEMENT_COUNT=kv_elements_per_16b,
        CONTEXT_PARTITION_SIZE=context_partition_size,
        QUERY_QUANT_MODE=query_quant_mode,
        KV_QUANT_MODE=kv_quant_mode,
        FP8_MAX_VALUE=fp8_max_value,
        VALUE_TRANSPOSED=value_transposed,
        IS_CAUSAL=is_causal,
        SLIDING_WINDOW=sliding_window,
        sinks_ptr=sinks,
        ONE_SHOT=one_shot,
        CDNA_VERSION=cdna_version,
    )
    if not one_shot:
        # ==================== REDUCTION KERNEL EXECUTION ====================
        grid = (num_sequences, num_kv_heads, 1)
        _paged_attention_decode_v2_reduce_kernel_wrapper(
            grid,
            output_gluon,
            exp_sums,
            max_logits,
            temporary_output,
            context_lengths,
            sinks,
            output_gluon.stride(0),
            output_gluon.stride(1),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            QUERY_GROUP_SIZE=equivalent_query_group_size,
            HEAD_SIZE=head_size,
            MAX_CONTEXT_PARTITION_NUM=max_context_partition_num,
            CONTEXT_PARTITION_SIZE=context_partition_size,
        )

    if query_length > 1:
        # Transpose output from [num_seqs, num_kv_heads, query_length, query_group_size, head_size]
        # back to [num_seqs * query_length, num_query_heads, head_size]
        transpose_output_gluon(
            output_gluon=output_gluon,
            output=output,
            batch_size=batch_size,
            query_sequence_length=query_length,
            num_kv_heads=num_kv_heads,
            query_group_size=query_group_size,
            head_size=head_size,
        )
