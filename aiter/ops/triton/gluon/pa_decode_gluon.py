# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import lru_cache

import torch

import aiter
import aiter.ops.triton.gluon.triton_version as tv
import aiter.ops.triton.utils._triton.arch_info as arch_info
import triton
import triton.language as tl
from triton.language.extra.hip import libdevice as hip_libdevice

CXX_PS_REDUCE_AVAILABLE = True
try:
    from csrc.cpp_itfs.pa.pa_ps import (
        launch_pa_decode_ps_reduce as launch_pa_decode_ps_reduce_cxx,
    )
except Exception:
    CXX_PS_REDUCE_AVAILABLE = False
    launch_pa_decode_ps_reduce_cxx = None

FLYDSL_PS_REDUCE_AVAILABLE = True
try:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.expr import arith, gpu, rocdl, buffer_ops, range_constexpr
    from flydsl.expr.typing import T, Int32
    from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
    from flydsl.runtime.device import get_rocm_arch as get_hip_arch
    from flydsl._mlir import ir
    from flydsl.compiler.kernel_function import CompilationContext
    from flydsl._mlir.dialects import arith as _mlir_arith
except Exception:
    FLYDSL_PS_REDUCE_AVAILABLE = False
    flyc = None
    fx = None
    arith = None
    gpu = None
    rocdl = None
    buffer_ops = None
    range_constexpr = None
    T = None
    Int32 = None
    SmemAllocator = None
    SmemPtr = None
    get_hip_arch = None
    ir = None
    CompilationContext = None
    _mlir_arith = None

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

try:
    from triton.experimental.gluon.language.amd.cdna3 import (
        sched_barrier as _amd_iglp_sched_barrier,
        sched_group_barrier as _amd_iglp_sched_group_barrier,
        set_prio as _amd_set_prio,
    )
except ImportError:
    # ignore iglp hint
    @gluon.jit
    def _amd_iglp_sched_barrier(tensor, inst_mask):
        return tensor

    @gluon.jit
    def _amd_iglp_sched_group_barrier(tensor, inst_mask, cnt, _):
        return tensor

    @gluon.jit
    def _amd_set_prio(value, prio):
        pass


@gluon.jit
def _fused_max_combine(a1, a2, b1, b2):
    return tl.maximum(a1, b1), tl.maximum(a2, b2)


@lru_cache(maxsize=1)
def get_cdna_version():
    """Get CDNA version lazily to avoid CUDA initialization during import."""
    if arch_info.get_arch() in ["gfx950"]:
        return 4
    elif arch_info.get_arch() in ["gfx942"]:
        return 3
    else:
        return -1


def get_occupancy():
    return 2


def get_recommended_splits(num_sequences, num_kv_heads, split_kv_blocks=1):
    props = torch.cuda.get_device_properties()
    num_sm = props.multi_processor_count * get_occupancy()
    max_context_partition_num = triton.cdiv(
        num_sm, num_sequences * num_kv_heads * split_kv_blocks
    )
    max_context_partition_num *= split_kv_blocks
    return min(max_context_partition_num, 8)


DS_WRITE = gl.constexpr(0x200)
DS_READ = gl.constexpr(0x100)
VMEM_LOAD = gl.constexpr(0x020)
MFMA = gl.constexpr(0x008)
COMPUTE = gl.constexpr(0x001)
# Pre-compute version check as constexpr for use in JIT kernels
TRITON_VERSION_GE_3_6_0 = tl.constexpr(tv.TRITON_VERSION_GE_3_6_0)


def define_layout(
    QUERY_GROUP_SIZE_POW2: gl.constexpr,
    CONTEXT_PARTITION_SIZE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
) -> gl.constexpr:
    # Register allocation configuration based on group size and compute block size
    if QUERY_GROUP_SIZE_POW2 >= 16:
        if QUERY_GROUP_SIZE_POW2 == 16:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64]]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64], [0, 128]]
        elif QUERY_GROUP_SIZE_POW2 == 32:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [[0, 1], [0, 2], [0, 64], [16, 0]]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [0, 128],
                    [16, 0],
                ]
        elif QUERY_GROUP_SIZE_POW2 == 64:
            if CONTEXT_PARTITION_SIZE == 128:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [16, 0],
                    [32, 0],
                ]
            elif CONTEXT_PARTITION_SIZE == 256:
                register_bases: gl.constexpr = [
                    [0, 1],
                    [0, 2],
                    [0, 64],
                    [0, 128],
                    [16, 0],
                    [32, 0],
                ]

        # Distributed layout for QK linear operations
        qk_linear_layout: gl.constexpr = gl.DistributedLinearLayout(
            reg_bases=register_bases,
            lane_bases=[[1, 0], [2, 0], [4, 0], [8, 0], [0, 4], [0, 8]],
            warp_bases=[[0, 16], [0, 32]],
            block_bases=[],
            shape=[QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE],
        )
    else:
        VGPRS0: gl.constexpr = QUERY_SEQ_LEN_POW2
        THREADS0: gl.constexpr = triton.cdiv(QUERY_GROUP_SIZE_POW2, (4 * VGPRS0))
        THREADS1: gl.constexpr = 64 // THREADS0

        qk_linear_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[VGPRS0, CONTEXT_PARTITION_SIZE // THREADS1],
            threads_per_warp=[THREADS0, THREADS1],
            warps_per_cta=[4, 1],
            order=[1, 0],
        )
    return qk_linear_layout


define_layout.__triton_builtin__ = True


def store_temporary_result(
    max_logits,
    exp_sums,
    attention_accumulator,
    max_logits_ptr,
    exp_sums_ptr,
    output_ptr,
    max_logits_offsets,
    output_offsets,
    qk_row_mask,
    output_mask,
    _semantic=None,
) -> None:
    gl.amd.cdna3.buffer_store(
        stored_value=max_logits,
        ptr=max_logits_ptr,
        offsets=max_logits_offsets,
        mask=qk_row_mask,
        _semantic=_semantic,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=exp_sums,
        ptr=exp_sums_ptr,
        offsets=max_logits_offsets,
        mask=qk_row_mask,
        _semantic=_semantic,
    )
    gl.amd.cdna3.buffer_store(
        stored_value=attention_accumulator,
        ptr=output_ptr,
        offsets=output_offsets,
        mask=output_mask,
        _semantic=_semantic,
    )


store_temporary_result.__triton_builtin__ = True


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"waves_per_eu": wa}, maxnreg=512 // wa if wa > 0 else None, num_stages=1
#         )
#         for wa in range(5)
#     ],
#     key=[
#         "KV_BLOCK_SIZE",
#         "SLIDING_WINDOW",
#         "KV_QUANT_MODE",
#         "QUERY_QUANT_MODE",
#         "ONE_QUERY_GROUP_SIZE_POW2",
#         "HEAD_SIZE_POW2",
#         "COMPUTE_TYPE",
#     ],
#     cache_results=True,
# )
@gluon.jit
def paged_attention_decode_v2_gluon_large_block_dot_kernel(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, query_length, num_kv_heads, query_group_size, 1](per-token) or [1](per-tensor) or None
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    stride_max_logits_seq,
    stride_max_logits_head,
    stride_max_logits_part,
    stride_output_seq,
    stride_output_head,
    stride_output_part,
    stride_output_group,
    # 5D query strides for [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    stride_query_bs: int,
    stride_query_qlen: int,
    stride_query_kv_head: int,
    stride_query_group_size: int,
    stride_key_block,
    stride_key_head,
    stride_key_head_split,
    stride_key_block_elem,
    stride_value_block,
    stride_value_head,
    stride_value_head_size,
    stride_block_table_seq,
    stride_query_scale_bs: int,
    stride_query_scale_qlen: int,
    stride_query_scale_kv_head: int,
    kv_scale_stride_0,
    kv_scale_stride_1,
    head_size: int,
    num_seqs: int,
    num_kv_heads: int,
    max_context_partition_num: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN: gl.constexpr,
    ONE_QUERY_GROUP_SIZE: gl.constexpr,
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
    SLIDING_WINDOW: gl.constexpr = 0,
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
    if COMPUTE_TYPE.is_fp8() or CDNA_VERSION == 4:
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
    LOG2_E: gl.constexpr = 1.4426950408889634  # log2(e) for exponential calculations
    CONTIGUOUS_KV_ELEMENTS_16B_LOAD: gl.constexpr = KV_16B_ELEMENT_COUNT

    KEY_HEAD_SIZE_POW2_SPLIT: gl.constexpr = (
        HEAD_SIZE_POW2 // CONTIGUOUS_KV_ELEMENTS_16B_LOAD
    )

    # Calculate MTP (Multi-Token Prefill) layout parameters
    QUERY_SEQ_LEN_POW2: gl.constexpr = triton.next_power_of_2(QUERY_SEQ_LEN)
    if ONE_QUERY_GROUP_SIZE <= 16 // QUERY_SEQ_LEN_POW2:
        ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr = 16 // QUERY_SEQ_LEN_POW2
    else:
        ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr = triton.next_power_of_2(
            ONE_QUERY_GROUP_SIZE
        )
    QUERY_GROUP_SIZE_POW2: gl.constexpr = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2

    # ==================== Memory Layout Definitions ====================
    if COMPUTE_TYPE.is_fp8():
        SHARED_LAYOUT_WIDTH: gl.constexpr = 8
    else:
        SHARED_LAYOUT_WIDTH: gl.constexpr = 4
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(
        SHARED_LAYOUT_WIDTH, 1, 16, order=[1, 0]
    )
    shared_value_scale_layout: gl.constexpr = gl.SwizzledSharedLayout(
        1, 1, 8, order=[0]
    )
    shared_key_scale_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 8, order=[0])

    # MTP Query tensor layout (3D) [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    if ONE_QUERY_GROUP_SIZE_POW2 <= 16:
        # ONE_QUERY_GROUP_SIZE_POW2 may be 4, 8, 16
        # corresponding Q_WARPS_PER_CTA_DIM1 should be 1, 2, 4
        # corresponding Q_WARPS_PER_CTA_DIM0 should be 4, 2, 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = triton.cdiv(ONE_QUERY_GROUP_SIZE_POW2, 4)
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
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[1, 4],
    )
    qk_lhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=16
    )
    qk_rhs_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=16
    )

    qk_linear_layout: gl.constexpr = define_layout(
        QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE, QUERY_SEQ_LEN_POW2
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
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
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
    # MTP offsets (for 3D layout)
    mtp_query_len_offsets = gl.arange(
        0, QUERY_SEQ_LEN_POW2, layout=mtp_query_len_layout
    )
    mtp_query_group_size_offsets = gl.arange(
        0, ONE_QUERY_GROUP_SIZE_POW2, layout=mtp_query_group_size_layout
    )
    mtp_head_size_offsets = gl.arange(0, HEAD_SIZE_POW2, layout=mtp_head_size_layout)

    kv_scale_column_offsets = gl.arange(
        0,
        KV_COMPUTE_BLOCK_SIZE,
        layout=gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[64],
            warps_per_cta=[4],
            order=[0],
        ),
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
    query_row_mask_3d = (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN) & (
        mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE
    )
    query_row_mask_1d = gl.reshape(query_row_mask_3d, [QUERY_GROUP_SIZE_POW2])
    qk_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    pv_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, pv_mfma_layout)
    )

    # ==================== Program ID and Sequence Setup ====================
    sequence_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    output_partition_idx = gl.program_id(2)
    context_length = gl.load(context_lengths_ptr + sequence_idx)

    # Compute KV partition index (adjusted for sliding window)
    if SLIDING_WINDOW > 0:
        sequence_start_idx = context_length - SLIDING_WINDOW
        sequence_partition_idx = (
            sequence_start_idx // CONTEXT_PARTITION_SIZE + output_partition_idx
        )
    else:
        sequence_start_idx = 0
        sequence_partition_idx = output_partition_idx

    # Calculate page offset based on KV partition index
    page_offset = 0
    if sequence_partition_idx % 4 == 1:
        page_offset = 1 * CONTEXT_PARTITION_SIZE
    elif sequence_partition_idx % 4 == 2:
        page_offset = 2 * CONTEXT_PARTITION_SIZE
    elif sequence_partition_idx % 4 == 3:
        page_offset = 3 * CONTEXT_PARTITION_SIZE

    # ==================== Query Loading ====================
    # Load query tensor with 3D MTP layout
    # Query shape: [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    mtp_query_offsets = (
        sequence_idx * stride_query_bs
        + mtp_query_len_offsets[:, None, None] * stride_query_qlen
        + kv_head_idx * stride_query_kv_head
        + mtp_query_group_size_offsets[None, :, None] * stride_query_group_size
        + mtp_head_size_offsets[None, None, :]
    )
    mtp_query_mask = (
        (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN)
        & (mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE)
        & (mtp_head_size_offsets[None, None, :] < head_size)
    )
    mtp_query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=mtp_query_offsets, mask=mtp_query_mask
    )
    mtp_query_tensor = gl.reshape(
        mtp_query_tensor,
        [QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2],
    )

    # ==================== Query Quantization Scale Handling ====================
    if QUERY_QUANT_MODE == 0:
        # Per-tensor quantization
        query_scale_value = gl.load(query_scale)
    elif QUERY_QUANT_MODE == 1:
        # Per-token quantization
        query_scale_offsets = (
            sequence_idx * stride_query_scale_bs
            + mtp_query_len_offsets[:, None, None] * stride_query_scale_qlen
            + kv_head_idx * stride_query_scale_kv_head
            + mtp_query_group_size_offsets[None, :, None]
        )
        query_scale_mask = (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN) & (
            mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE
        )
        query_scale_value = gl.amd.cdna3.buffer_load(
            ptr=query_scale,
            offsets=query_scale_offsets,
            mask=query_scale_mask,
        )
        query_scale_value = gl.reshape(query_scale_value, [QUERY_GROUP_SIZE_POW2, 1])
        query_scale_value = gl.convert_layout(
            query_scale_value, layout=qk_linear_layout
        )

    # ==================== Output Buffer Setup ====================
    # Create MTP layout indices for max_logits/exp_sums
    max_logits_base_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    # Convert MTP layout indices to continuous indices for exp_sums/max_logits
    max_logits_query_len_idx = max_logits_base_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    max_logits_group_idx_in_len = (
        max_logits_base_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    )
    max_logits_base_offsets = (
        max_logits_query_len_idx * ONE_QUERY_GROUP_SIZE + max_logits_group_idx_in_len
    )
    max_logits_offsets = (
        sequence_idx * stride_max_logits_seq
        + kv_head_idx * stride_max_logits_head
        + output_partition_idx * stride_max_logits_part
        + max_logits_base_offsets
    )

    # Create MTP layout indices for output
    output_group_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    # Convert MTP layout indices to continuous indices for temporary_output
    output_query_len_idx = output_group_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    output_group_idx_in_len = output_group_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    output_group_offsets = (
        output_query_len_idx * ONE_QUERY_GROUP_SIZE + output_group_idx_in_len
    )
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )
    # Use pv_row_mask for output mask (consistent with paged_attention_decode_v2_gluon_dot_kernel)
    output_mask = pv_row_mask[:, None] & (output_head_size_offsets[None, :] < head_size)

    output_offsets = sequence_idx * stride_output_seq
    output_offsets += kv_head_idx * stride_output_head
    output_offsets += (
        output_partition_idx * stride_output_part
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
    kv_sequence_start_index = sequence_partition_idx * CONTEXT_PARTITION_SIZE

    # Early return if partition is entirely before sliding window
    if SLIDING_WINDOW > 0:
        if (
            (kv_sequence_start_index + CONTEXT_PARTITION_SIZE) <= sequence_start_idx
            or kv_sequence_start_index < 0
            or kv_sequence_start_index >= context_length
        ):
            store_temporary_result(
                max_logits,
                exp_sums,
                attention_accumulator,
                max_logits_ptr,
                exp_sums_ptr,
                output_ptr,
                max_logits_offsets,
                output_offsets,
                qk_row_mask,
                output_mask,
            )
            return
    else:
        # Early return if this partition is beyond sequence length
        if kv_sequence_start_index >= context_length:
            return

    query_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2], shared_query_layout
    )
    mtp_query_tensor = mtp_query_tensor.to(COMPUTE_TYPE)
    query_shared.store(mtp_query_tensor)

    if KV_QUANT_MODE == 1:
        value_scale_shared = gl.allocate_shared_memory(
            gl.float32, [KV_COMPUTE_BLOCK_SIZE], shared_value_scale_layout
        )
        key_scale_shared = gl.allocate_shared_memory(
            gl.float32, [KV_COMPUTE_BLOCK_SIZE], shared_key_scale_layout
        )

    KV_COMPUTE_BLOCK_COUNT: gl.constexpr = (
        CONTEXT_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE
    )
    block_table_id = kv_sequence_start_index // KV_BLOCK_SIZE
    # ==================== Block Table Lookup ====================
    block_tables_start_ptr = block_tables_ptr + sequence_idx * stride_block_table_seq
    kv_page_id = gl.load(block_tables_start_ptr + block_table_id)
    # ==================== Key Quantization Scale Handling ====================
    if KV_QUANT_MODE == 0:
        # Per-tensor quantization
        key_scale_value = gl.load(key_scale)
        value_scale_value = gl.load(value_scale)

    # ==================== Main Attention Computation Loop ====================
    for kv_block_index in gl.static_range(KV_COMPUTE_BLOCK_COUNT):
        current_page_offset = page_offset + kv_block_index * KV_COMPUTE_BLOCK_SIZE
        kv_sub_sequence_start_index = (
            kv_sequence_start_index + kv_block_index * KV_COMPUTE_BLOCK_SIZE
        )
        if KV_QUANT_MODE == 1:
            # Per-token quantization: load scales in a distributed block layout,
            # then remap to the QK linear layout used by the compute path.
            key_scale_offsets = (
                kv_page_id * kv_scale_stride_0
                + kv_head_idx * kv_scale_stride_1
                + current_page_offset
                + kv_scale_column_offsets
            )
            kv_scale_mask = (
                kv_sub_sequence_start_index + kv_scale_column_offsets
                >= sequence_start_idx
            )
            key_scale_value = gl.load(
                key_scale + key_scale_offsets,
                mask=kv_scale_mask,
                other=0.0,
            )
            key_scale_shared.store(key_scale_value)
            value_scale_value = gl.load(
                value_scale + key_scale_offsets,
                mask=kv_scale_mask,
                other=0.0,
            )
            value_scale_shared.store(value_scale_value)

        # Calculate column offsets for QK computation
        qk_column_offsets = kv_sub_sequence_start_index + gl.arange(
            0, KV_COMPUTE_BLOCK_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
        )

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
        # Reshape key block to [HEAD_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE]
        key_block = gl.permute(key_block, [0, 2, 1])
        key_block = gl.reshape(key_block, [HEAD_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE])
        for _ in gl.static_range(8):
            key_block = _amd_iglp_sched_group_barrier(key_block, VMEM_LOAD, 1, 0)
            key_block = _amd_iglp_sched_group_barrier(key_block, COMPUTE, 4, 0)
        key_block = _amd_iglp_sched_barrier(key_block, 0x0)

        # ==================== QK Matrix Multiplication ====================
        # Initialize QK accumulator
        qk_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE),
            dtype=gl.float32,
            layout=qk_mfma_layout,
        )

        # Convert layouts for MFMA operation
        query_converted = query_shared.load(qk_lhs_layout)
        key_converted = gl.convert_layout(key_block, layout=qk_rhs_layout)
        key_converted = key_converted.to(COMPUTE_TYPE)
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
        # Perform matrix multiplication
        qk_matrix = gl.amd.cdna3.mfma(query_converted, key_converted, qk_accumulator)
        qk_matrix = gl.reshape(
            qk_matrix, [QUERY_GROUP_SIZE_POW2, KV_COMPUTE_BLOCK_SIZE]
        )
        for _ in gl.static_range(8):
            qk_matrix = _amd_iglp_sched_group_barrier(qk_matrix, VMEM_LOAD, 1, 1)
            qk_matrix = _amd_iglp_sched_group_barrier(qk_matrix, COMPUTE, 4, 1)
        qk_matrix = _amd_iglp_sched_barrier(qk_matrix, 0x0)
        # ==================== Scale QK Scores ====================
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                key_scale_value = key_scale_shared.load(
                    gl.SliceLayout(0, qk_linear_layout)
                )
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

        # ==================== Attention Masking ====================
        # Apply causal masking if required
        if IS_CAUSAL:
            sequence_extension = (
                QUERY_SEQ_LEN - 1 - qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2
            )
            causal_mask = (
                sequence_extension[:, None] + qk_column_offsets[None, :]
                < context_length
            )
            if SLIDING_WINDOW > 0:
                causal_mask = causal_mask & (
                    sequence_extension[:, None] + qk_column_offsets[None, :]
                    >= sequence_start_idx + QUERY_SEQ_LEN
                )
        else:
            causal_mask = qk_column_offsets[None, :] < context_length
            if SLIDING_WINDOW > 0:
                query_token_idx = qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2
                causal_mask = causal_mask & (
                    qk_column_offsets[None, :]
                    >= sequence_start_idx + query_token_idx[:, None] + 1
                )

        # Combine masks
        combined_mask = qk_row_mask[:, None] & causal_mask
        qk_matrix = gl.convert_layout(qk_matrix, layout=qk_linear_layout)
        # Apply scaling to QK scores
        qk_matrix = qk_scale_value * qk_matrix
        # Apply masking to QK scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)
        qk_matrix = gl.where(combined_mask, qk_matrix, float(-3.4e38))

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
                value_scale_value = value_scale_shared.load(
                    gl.SliceLayout(0, qk_linear_layout)
                )
                # Per-token quantization scaling
                # Create mask for valid tokens
                valid_token_mask = qk_column_offsets < context_length
                # Mask out value_scale of invalid tokens
                value_scale_value = tl.where(
                    valid_token_mask, value_scale_value, float(0.0)
                )
                value_scale_max = gl.max(value_scale_value, axis=0)
                # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                # Use fast reciprocal plus multiplies instead of a full divide.
                inv_value_scale_max = hip_libdevice.fast_dividef(
                    1.0, value_scale_max + 1e-8
                )
                fp8_inv_scale = float(FP8_MAX_VALUE) * inv_value_scale_max
                value_scale_value = value_scale_value * fp8_inv_scale
                attention_probs = value_scale_value[None, :] * attention_probs
                probability_scale = value_scale_max * (1.0 / float(FP8_MAX_VALUE))
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

    # Store results to output buffers
    store_temporary_result(
        max_logits,
        exp_sums,
        attention_accumulator,
        max_logits_ptr,
        exp_sums_ptr,
        output_ptr,
        max_logits_offsets,
        output_offsets,
        qk_row_mask,
        output_mask,
    )


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"waves_per_eu": wa}, maxnreg=512 // wa if wa > 0 else None, num_stages=1
#         )
#         for wa in range(5)
#     ],
#     key=[
#         "KV_BLOCK_SIZE",
#         "SLIDING_WINDOW",
#         "KV_QUANT_MODE",
#         "QUERY_QUANT_MODE",
#         "ONE_QUERY_GROUP_SIZE_POW2",
#         "HEAD_SIZE_POW2",
#         "COMPUTE_TYPE",
#         "ONE_SHOT",
#     ],
#     cache_results=True,
# )
@gluon.jit
def paged_attention_decode_sliding_window_head_1(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    query_ptr,  # [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale: float,
    query_scale,  # [num_seqs, query_length, num_kv_heads, query_group_size, 1](per-token) or [1](per-tensor) or None
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    sinks_ptr,  # [num_query_heads]
    stride_max_logits_seq: int,
    stride_max_logits_head: int,
    stride_max_logits_part: int,
    # 5D output strides for [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    stride_output_bs: int,
    stride_output_len: int,
    stride_output_kv_head: int,
    stride_output_group_size: int,
    # 5D query strides for [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    stride_query_bs: int,
    stride_query_qlen: int,
    stride_query_kv_head: int,
    stride_query_group_size: int,
    stride_key_block: int,
    stride_key_head: int,
    stride_key_head_split: int,
    stride_key_block_elem: int,
    stride_value_block: int,
    stride_value_head: int,
    stride_value_head_size: int,
    stride_block_table_seq: int,
    stride_query_scale_bs: int,
    stride_query_scale_qlen: int,
    stride_query_scale_kv_head: int,
    kv_scale_stride_0: int,
    kv_scale_stride_1: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
    ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr,
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
    ONE_SHOT: gl.constexpr = False,
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
    # ==================== VALIDATION CHECKS ====================
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

    sequence_idx = gl.program_id(0)
    mtp_idx = gl.program_id(1)
    split_idx = gl.program_id(2)

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
    MAX_NUM_KV_BLOCKS_PER_COMPUTE: gl.constexpr = triton.cdiv(
        CONTEXT_PARTITION_SIZE, KV_BLOCK_SIZE
    )

    CONTEXT_PARTITION_SIZE_PER_BLOCK: gl.constexpr = triton.cdiv(
        KV_BLOCK_SIZE, CONTEXT_PARTITION_SIZE
    )
    sequence_split_idx = split_idx // CONTEXT_PARTITION_SIZE_PER_BLOCK
    block_split_idx = split_idx % CONTEXT_PARTITION_SIZE_PER_BLOCK
    # ==================== MEMORY LAYOUT DEFINITIONS ====================
    # Query tensor layout - optimized for sequential access (2D)
    if COMPUTE_TYPE.is_fp8():
        SHARED_LAYOUT_WIDTH: gl.constexpr = 8
    else:
        SHARED_LAYOUT_WIDTH: gl.constexpr = 4
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(
        SHARED_LAYOUT_WIDTH, 1, 16, order=[1, 0]
    )
    shared_probs_layout: gl.constexpr = gl.SwizzledSharedLayout(
        SHARED_LAYOUT_WIDTH, 1, 8, order=[1, 0]
    )
    shared_value_scale_layout: gl.constexpr = gl.SwizzledSharedLayout(
        1, 1, 8, order=[0]
    )
    shared_key_scale_layout: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 8, order=[0])
    QUERY_GROUP_SIZE_POW2: gl.constexpr = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2
    if ONE_QUERY_GROUP_SIZE_POW2 <= 16:
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = triton.cdiv(ONE_QUERY_GROUP_SIZE_POW2, 4)
        Q_WARPS_PER_CTA_DIM0: gl.constexpr = 4 // Q_WARPS_PER_CTA_DIM1
    else:
        Q_WARPS_PER_CTA_DIM0: gl.constexpr = 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = 4
    if QUERY_SEQ_LEN_POW2 == 1:
        blocked_query_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 8],
            threads_per_warp=[4, 16],
            warps_per_cta=[Q_WARPS_PER_CTA_DIM0, Q_WARPS_PER_CTA_DIM1],
            order=[1, 0],
        )
    else:
        mtp_blocked_query_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 8],
            threads_per_warp=[1, 4, 16],
            warps_per_cta=[Q_WARPS_PER_CTA_DIM0, Q_WARPS_PER_CTA_DIM1, 1],
            order=[2, 1, 0],
        )
    # Key cache layout - optimized for block-wise access patterns
    if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE:
        KV_COMPUTE_BLOCK_SIZE: gl.constexpr = CONTEXT_PARTITION_SIZE
    else:
        KV_COMPUTE_BLOCK_SIZE: gl.constexpr = KV_BLOCK_SIZE

    key_warps_per_cta: gl.constexpr = (
        [4, 1, 1, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 1, 4, 1]
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

    qk_linear_layout: gl.constexpr = define_layout(
        QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE, QUERY_SEQ_LEN_POW2
    )
    if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
        kv_scale_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1],
            threads_per_warp=[64],
            warps_per_cta=[4],
            order=[0],
        )
        kv_scale_indices = gl.arange(0, CONTEXT_PARTITION_SIZE, layout=kv_scale_layout)

    context_length = gl.load(context_lengths_ptr + sequence_idx)
    # Value cache layout configuration based on transpose flag
    if VALUE_TRANSPOSED:
        # Transposed value layout for better memory access patterns
        value_threads_per_warp: gl.constexpr = (
            [4, 1, 16, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 4, 16, 1]
        )

        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )

        value_dim1_offsets = gl.arange(
            0,
            KV_COMPUTE_BLOCK_SIZE // KV_16B_ELEMENT_COUNT,
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
            [4, 16, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 16, 4]
        )
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, KV_16B_ELEMENT_COUNT],
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
            KV_COMPUTE_BLOCK_SIZE,
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
        operand_index=0, parent=pv_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )
    pv_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )

    # ==================== LAYOUT SLICE DEFINITIONS ====================

    if QUERY_SEQ_LEN_POW2 == 1:
        query_group_size_layout: gl.constexpr = gl.SliceLayout(1, blocked_query_layout)
        head_size_layout: gl.constexpr = gl.SliceLayout(0, blocked_query_layout)
    else:
        mtp_query_len_layout: gl.constexpr = gl.SliceLayout(
            1, gl.SliceLayout(2, mtp_blocked_query_layout)
        )
        mtp_query_group_size_layout: gl.constexpr = gl.SliceLayout(
            0, gl.SliceLayout(2, mtp_blocked_query_layout)
        )
        mtp_head_size_layout: gl.constexpr = gl.SliceLayout(
            0, gl.SliceLayout(1, mtp_blocked_query_layout)
        )

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
    if QUERY_SEQ_LEN_POW2 == 1:
        query_group_size_offsets = gl.arange(
            0, ONE_QUERY_GROUP_SIZE_POW2, layout=query_group_size_layout
        )
        head_size_offsets = gl.arange(0, HEAD_SIZE_POW2, layout=head_size_layout)
    else:
        mtp_query_len_offsets = gl.arange(
            0, QUERY_SEQ_LEN_POW2, layout=mtp_query_len_layout
        )
        mtp_query_group_size_offsets = gl.arange(
            0, ONE_QUERY_GROUP_SIZE_POW2, layout=mtp_query_group_size_layout
        )
        mtp_head_size_offsets = gl.arange(
            0, HEAD_SIZE_POW2, layout=mtp_head_size_layout
        )

    head_size_split_offsets = gl.arange(
        0, K_HEAD_SIZE_SPLITS, layout=head_size_split_layout
    )
    block_element_offsets = gl.arange(
        0, KV_COMPUTE_BLOCK_SIZE, layout=block_element_layout
    )
    contiguous_kv_element_offsets = gl.arange(
        0, KV_16B_ELEMENT_COUNT, layout=contiguous_kv_elements_layout
    )
    max_logits_base_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    if QUERY_SEQ_LEN_POW2 == 1:
        query_row_mask_1d = query_group_size_offsets < query_group_size
    else:
        query_row_mask_3d = (
            mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None]
            < query_seq_len
        ) & (mtp_query_group_size_offsets[None, :, None] < query_group_size)
        query_row_mask_1d = gl.reshape(query_row_mask_3d, [QUERY_GROUP_SIZE_POW2])
    output_group_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    if QUERY_SEQ_LEN_POW2 == 1:
        output_group_offsets = mtp_idx * query_group_size + output_group_offsets
    else:
        output_query_len_idx = output_group_offsets // ONE_QUERY_GROUP_SIZE_POW2
        output_group_idx_in_len = output_group_offsets % ONE_QUERY_GROUP_SIZE_POW2
        output_group_offsets = (
            mtp_idx * QUERY_SEQ_LEN_POW2 + output_query_len_idx
        ) * query_group_size + output_group_idx_in_len
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )

    # ==================== PROGRAM ID AND INITIALIZATION ====================

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
    if QUERY_SEQ_LEN_POW2 == 1:
        query_offsets = (
            sequence_idx * stride_query_bs
            + mtp_idx * stride_query_qlen
            + query_group_size_offsets[:, None] * stride_query_group_size
            + head_size_offsets[None, :]
        )
        query_mask = (query_group_size_offsets[:, None] < query_group_size) & (
            head_size_offsets[None, :] < head_size
        )
        query_tensor = gl.amd.cdna3.buffer_load(
            ptr=query_ptr, offsets=query_offsets, mask=query_mask
        )
    else:
        mtp_query_offsets = (
            sequence_idx * stride_query_bs
            + (mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None])
            * stride_query_qlen
            + mtp_query_group_size_offsets[None, :, None] * stride_query_group_size
            + mtp_head_size_offsets[None, None, :]
        )
        mtp_query_mask = (
            (
                mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None]
                < query_seq_len
            )
            & (mtp_query_group_size_offsets[None, :, None] < query_group_size)
            & (mtp_head_size_offsets[None, None, :] < head_size)
        )
        mtp_query_tensor = gl.amd.cdna3.buffer_load(
            ptr=query_ptr, offsets=mtp_query_offsets, mask=mtp_query_mask
        )
    if SLIDING_WINDOW > 0:
        sequence_end_idx = context_length
        sequence_start_idx = context_length - SLIDING_WINDOW
        window_partition_start_idx = gl.maximum(
            0, sequence_start_idx // CONTEXT_PARTITION_SIZE
        )
        window_partition_end_idx = gl.cdiv(sequence_end_idx, CONTEXT_PARTITION_SIZE)
        window_partition_count = gl.maximum(
            0, window_partition_end_idx - window_partition_start_idx
        )
        sequence_split_count = gl.maximum(
            1, gl.num_programs(2) // CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        partitions_per_sequence_split = gl.cdiv(
            window_partition_count, sequence_split_count
        )
        local_partition_start_idx = (
            window_partition_start_idx
            + partitions_per_sequence_split * sequence_split_idx
        )
        local_partition_end_idx = gl.minimum(
            window_partition_end_idx,
            window_partition_start_idx
            + partitions_per_sequence_split * (sequence_split_idx + 1),
        )
        sequence_partition_start_idx = local_partition_start_idx + block_split_idx
        sequence_partition_end_idx = local_partition_end_idx + block_split_idx
    else:
        page_size = gl.cdiv(
            context_length, gl.num_programs(2) // CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        sequence_start_idx = page_size * sequence_split_idx

        sequence_end_idx = gl.minimum(context_length, page_size + sequence_start_idx)
        sequence_partition_start_idx = (
            sequence_start_idx // CONTEXT_PARTITION_SIZE + block_split_idx
        )
        sequence_partition_end_idx = (
            gl.cdiv(sequence_end_idx, CONTEXT_PARTITION_SIZE) + block_split_idx
        )
    block_indices = gl.arange(0, MAX_NUM_KV_BLOCKS_PER_COMPUTE, layout=block_id_layout)
    max_num_kv_blocks = gl.cdiv(context_length, KV_COMPUTE_BLOCK_SIZE)
    kv_block_start_idx = sequence_partition_start_idx * MAX_NUM_KV_BLOCKS_PER_COMPUTE
    if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
        if kv_block_start_idx < max_num_kv_blocks:
            kv_block_numbers = gl.load(
                block_tables_ptr
                + sequence_idx * stride_block_table_seq
                + kv_block_start_idx // CONTEXT_PARTITION_SIZE_PER_BLOCK
            )
        else:
            kv_block_numbers = 0
    else:
        kv_block_numbers = gl.amd.cdna3.buffer_load(
            ptr=block_tables_ptr
            + sequence_idx * stride_block_table_seq
            + kv_block_start_idx // CONTEXT_PARTITION_SIZE_PER_BLOCK,
            offsets=block_indices,
            mask=block_indices < (max_num_kv_blocks - kv_block_start_idx),
        )
    if QUERY_SEQ_LEN_POW2 == 1:
        max_logits_group_idx_in_len = max_logits_base_offsets
        query_token_idx = mtp_idx
        max_logits_base_offsets = mtp_idx * query_group_size + max_logits_base_offsets
    else:
        max_logits_query_len_idx = max_logits_base_offsets // ONE_QUERY_GROUP_SIZE_POW2
        max_logits_group_idx_in_len = (
            max_logits_base_offsets % ONE_QUERY_GROUP_SIZE_POW2
        )
        # Keep the per-row query token index in the original query sequence space.
        # `max_logits_base_offsets` below already includes the mtp split offset, so
        # recomputing from it later would double-count `mtp_idx` for the second split.
        query_token_idx = mtp_idx * QUERY_SEQ_LEN_POW2 + max_logits_query_len_idx
        max_logits_base_offsets = (
            query_token_idx
        ) * query_group_size + max_logits_group_idx_in_len
    if ONE_SHOT:
        if QUERY_SEQ_LEN_POW2 == 1:
            output_offsets = (
                sequence_idx * stride_output_bs
                + mtp_idx * stride_output_len
                + query_group_size_offsets[:, None] * stride_output_group_size
                + head_size_offsets[None, :]
            )
        else:
            output_offsets = (
                sequence_idx * stride_output_bs
                + (mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None])
                * stride_output_len
                + mtp_query_group_size_offsets[None, :, None] * stride_output_group_size
                + mtp_head_size_offsets[None, None, :]
            )
    else:
        output_offsets = sequence_idx * stride_output_bs
        output_offsets += (
            split_idx * stride_output_kv_head
            + output_group_offsets[:, None] * stride_output_group_size
            + output_head_size_offsets[None, :]
        )

        max_logits_offsets = (
            sequence_idx * stride_max_logits_seq
            + split_idx * stride_max_logits_part
            + max_logits_base_offsets
        )
    qk_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    pv_row_mask = gl.convert_layout(
        qk_row_mask, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    if ONE_SHOT:
        if QUERY_SEQ_LEN_POW2 == 1:
            output_mask = query_mask
        else:
            output_mask = mtp_query_mask
    else:
        output_mask = pv_row_mask[:, None] & (
            output_head_size_offsets[None, :] < head_size
        )

    if (sequence_start_idx >= context_length) | (
        sequence_partition_start_idx >= sequence_partition_end_idx
    ):
        if not ONE_SHOT:
            store_temporary_result(
                max_logits,
                exp_sums,
                attention_accumulator,
                max_logits_ptr,
                exp_sums_ptr,
                output_ptr,
                max_logits_offsets,
                output_offsets,
                qk_row_mask,
                output_mask,
            )
        return  # No computation needed for this partition

    if QUERY_SEQ_LEN_POW2 > 1:
        mtp_query_tensor = gl.reshape(
            mtp_query_tensor, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
        )

    query_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2], shared_query_layout
    )
    probs_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE,
        [QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE],
        shared_probs_layout,
    )
    qk_column_indices = gl.arange(
        0, CONTEXT_PARTITION_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
    )
    value_scale_shared = gl.allocate_shared_memory(
        gl.float32,
        [CONTEXT_PARTITION_SIZE],
        shared_value_scale_layout,
    )
    key_scale_shared = gl.allocate_shared_memory(
        gl.float32,
        [CONTEXT_PARTITION_SIZE],
        shared_key_scale_layout,
    )
    if QUERY_QUANT_MODE == 0:
        query_scale_value = gl.load(query_scale)
    elif QUERY_QUANT_MODE == 1:
        if QUERY_SEQ_LEN_POW2 == 1:
            query_scale_offsets = (
                sequence_idx * stride_query_scale_bs
                + mtp_idx * stride_query_scale_qlen
                + query_group_size_offsets[:, None]
            )
            query_scale_mask = query_group_size_offsets[:, None] < query_group_size
            query_scale_value = gl.amd.cdna3.buffer_load(
                ptr=query_scale,
                offsets=query_scale_offsets,
                mask=query_scale_mask,
            )
        else:
            query_scale_offsets = (
                sequence_idx * stride_query_scale_bs
                + (mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None])
                * stride_query_scale_qlen
                + mtp_query_group_size_offsets[None, :, None]
            )
            query_scale_mask = (
                mtp_idx * QUERY_SEQ_LEN_POW2 + mtp_query_len_offsets[:, None, None]
                < query_seq_len
            ) & (mtp_query_group_size_offsets[None, :, None] < query_group_size)
            query_scale_value = gl.amd.cdna3.buffer_load(
                ptr=query_scale,
                offsets=query_scale_offsets,
                mask=query_scale_mask,
            )
            query_scale_value = gl.reshape(
                query_scale_value, [QUERY_GROUP_SIZE_POW2, 1]
            )
        query_scale_value = gl.convert_layout(
            query_scale_value, layout=qk_linear_layout
        )

    # ==================== SEQUENCE PROCESSING ====================

    if ONE_SHOT and sinks_ptr is not None:
        # sinks_ptr is per-query-head: [num_query_heads] where
        # num_query_heads = num_kv_heads * query_group_size.
        # It is shared across query positions (query_seq_len).
        sinks_values = gl.load(
            sinks_ptr + max_logits_group_idx_in_len,
            mask=qk_row_mask,
            other=float("-inf"),
        )

    if KV_QUANT_MODE == 0:
        # Per-tensor quantization
        key_scale_value = gl.load(key_scale)
        value_scale_value = gl.load(value_scale)

    if QUERY_SEQ_LEN_POW2 == 1:
        mtp_query_tensor = query_tensor.to(COMPUTE_TYPE)
    else:
        mtp_query_tensor = mtp_query_tensor.to(COMPUTE_TYPE)
    query_shared.store(mtp_query_tensor)
    page_offset = (
        kv_block_start_idx % CONTEXT_PARTITION_SIZE_PER_BLOCK
    ) * CONTEXT_PARTITION_SIZE
    kv_block_numbers = kv_block_numbers.to(gl.int64)
    if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
        key_block_offsets = (
            kv_block_numbers * stride_key_block
            + head_size_split_offsets[None, :, None, None] * stride_key_head_split
            # Use runtime stride for KV block element (may be padded for large blocks).
            + (page_offset + block_element_offsets)[None, None, :, None]
            * stride_key_block_elem
            + contiguous_kv_element_offsets[None, None, None, :]
        )
    else:
        key_block_offsets = (
            kv_block_numbers[:, None, None, None] * stride_key_block
            + head_size_split_offsets[None, :, None, None] * stride_key_head_split
            # Use runtime stride for KV block element (may be padded for large blocks).
            + (page_offset + block_element_offsets)[None, None, :, None]
            * stride_key_block_elem
            + contiguous_kv_element_offsets[None, None, None, :]
        )

    key_tensor = gl.load(key_cache_ptr + key_block_offsets)
    query_converted = query_shared.load(qk_lhs_operand_layout)
    for sequence_partition_idx in range(
        sequence_partition_start_idx,
        sequence_partition_end_idx,
        CONTEXT_PARTITION_SIZE_PER_BLOCK,
    ):
        kv_block_start_idx2 = (
            kv_block_start_idx
            + MAX_NUM_KV_BLOCKS_PER_COMPUTE * CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        # Prepare QK MFMA while key loads (these don't depend on key data)
        qk_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE),
            dtype=gl.float32,
            layout=qk_mfma_layout,
        )
        # Load key quantization scales if needed (overlaps with key tensor load)
        key_tensor = _amd_iglp_sched_barrier(key_tensor, 0x0)
        if KV_QUANT_MODE == 1:
            # Per-token quantization - prepare offsets while key loads
            if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
                key_scale_offsets = kv_block_numbers * kv_scale_stride_0 + (
                    page_offset + kv_scale_indices
                )
            else:
                key_scale_offsets = (
                    kv_block_numbers[:, None, None, None] * kv_scale_stride_0
                    + (page_offset + block_element_offsets)[None, None, :, None]
                )
            # Optimize: Load both scales with VMEM scheduling, overlap with key reshape
            key_scale_value_blocked = gl.load(key_scale + key_scale_offsets)
            # Shared memory is used as a 1D staging buffer for layout conversion.
            key_scale_value_blocked = gl.reshape(
                key_scale_value_blocked, [CONTEXT_PARTITION_SIZE]
            )
            key_scale_shared.store(key_scale_value_blocked)
            value_scale_value_blocked = gl.load(value_scale + key_scale_offsets)
            value_scale_value_blocked = gl.reshape(
                value_scale_value_blocked, [CONTEXT_PARTITION_SIZE]
            )
            value_scale_shared.store(value_scale_value_blocked)

        # Reshape key tensor for matrix multiplication
        key_converted = gl.permute(key_tensor, [1, 3, 0, 2])
        key_converted = gl.reshape(
            key_converted, [HEAD_SIZE_POW2, CONTEXT_PARTITION_SIZE]
        )

        # ==================== VALUE LOADING WITH QK MFMA OVERLAP ====================
        # Convert key layout for MFMA (query_converted and qk_accumulator already prepared above)
        key_converted = gl.convert_layout(key_converted, layout=qk_rhs_operand_layout)
        key_converted = key_converted.to(COMPUTE_TYPE)
        if VALUE_TRANSPOSED:
            # Load values from transposed cache layout
            if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
                kv_block_numbers_reshaped = kv_block_numbers
                value_block_offsets = (
                    kv_block_numbers_reshaped * stride_value_block
                    + (page_offset // KV_16B_ELEMENT_COUNT + value_dim1_offsets)[
                        None, :, None, None
                    ]
                    * stride_value_head_size
                    + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                    + value_dim3_offsets[None, None, None, :]
                )

            else:
                kv_block_numbers_reshaped = gl.convert_layout(
                    kv_block_numbers,
                    layout=gl.SliceLayout(
                        1, gl.SliceLayout(2, gl.SliceLayout(3, blocked_value_layout))
                    ),
                )

                value_block_offsets = (
                    kv_block_numbers_reshaped[:, None, None, None] * stride_value_block
                    + (page_offset // KV_16B_ELEMENT_COUNT + value_dim1_offsets)[
                        None, :, None, None
                    ]
                    * stride_value_head_size
                    + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                    + value_dim3_offsets[None, None, None, :]
                )
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)

            # Permute and reshape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 1, 3, 2])
        else:
            # Load values from standard cache layout
            if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
                kv_block_numbers_reshaped = kv_block_numbers
                value_block_offsets = (
                    kv_block_numbers_reshaped * stride_value_block
                    + value_dim1_offsets[None, :, None] * stride_value_head_size
                    + (page_offset + value_dim2_offsets)[None, None, :]
                )
            else:
                kv_block_numbers_reshaped = gl.convert_layout(
                    kv_block_numbers,
                    layout=gl.SliceLayout(1, gl.SliceLayout(2, blocked_value_layout)),
                )
                value_block_offsets = (
                    kv_block_numbers_reshaped[:, None, None] * stride_value_block
                    + value_dim1_offsets[None, :, None] * stride_value_head_size
                    + (page_offset + value_dim2_offsets)[None, None, :]
                )

            # Schedule: Start value VMEM load, then QK MFMA
            value_tensor = gl.load(value_cache_ptr + value_block_offsets)

            # Permute and resape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 2, 1])
        value_tensor = gl.reshape(
            value_tensor, [CONTEXT_PARTITION_SIZE, HEAD_SIZE_POW2]
        )
        # Compute QK attention scores using MFMA (overlaps with value load)
        attention_scores = gl.amd.cdna3.mfma(
            query_converted, key_converted, qk_accumulator
        )
        attention_scores = gl.reshape(
            attention_scores, [QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE]
        )
        if KV_QUANT_MODE == 1:
            value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 1, 0)
            value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
            value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 1, 0)
            value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, VMEM_LOAD, 2, 0)
        value_tensor = _amd_iglp_sched_group_barrier(value_tensor, MFMA, 2, 0)

        value_tensor = _amd_iglp_sched_barrier(value_tensor, 0x0)
        qk_column_offsets = (
            kv_block_start_idx * KV_COMPUTE_BLOCK_SIZE + qk_column_indices
        )

        # Apply quantization scaling to attention scores
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                key_scale_value = key_scale_shared.load(
                    gl.SliceLayout(0, qk_linear_layout)
                )
                # key_scale_value = gl.convert_layout(key_scale_value_blocked, layout=gl.SliceLayout(0, qk_linear_layout))
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

        # ==================== ATTENTION MASKING ====================

        if QUERY_SEQ_LEN_POW2 == 1:
            if IS_CAUSAL:
                sequence_position_extension = query_seq_len - 1 - mtp_idx
                causal_mask = (
                    sequence_position_extension + qk_column_offsets[None, :]
                    < sequence_end_idx
                )
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        sequence_position_extension + qk_column_offsets[None, :]
                        >= sequence_start_idx + query_seq_len
                    )
                else:
                    causal_mask = causal_mask & (
                        sequence_position_extension + qk_column_offsets[None, :]
                        >= sequence_start_idx
                    )
            else:
                causal_mask = qk_column_offsets[None, :] < sequence_end_idx
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        qk_column_offsets[None, :] >= sequence_start_idx + mtp_idx + 1
                    )
                else:
                    causal_mask = causal_mask & (
                        qk_column_offsets[None, :] >= sequence_start_idx
                    )
        else:

            if IS_CAUSAL:
                sequence_position_extension = query_seq_len - 1 - query_token_idx
                causal_mask = (
                    sequence_position_extension[:, None] + qk_column_offsets[None, :]
                    < sequence_end_idx
                )
                if SLIDING_WINDOW > 0:
                    causal_mask = causal_mask & (
                        sequence_position_extension[:, None]
                        + qk_column_offsets[None, :]
                        >= sequence_start_idx + query_seq_len
                    )
                else:
                    causal_mask = causal_mask & (
                        sequence_position_extension[:, None]
                        + qk_column_offsets[None, :]
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
        if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
            if kv_block_start_idx2 < max_num_kv_blocks:
                kv_block_numbers2 = gl.load(
                    block_tables_ptr
                    + sequence_idx * stride_block_table_seq
                    + kv_block_start_idx2 // CONTEXT_PARTITION_SIZE_PER_BLOCK
                )
            else:
                kv_block_numbers2 = 0
        else:
            kv_block_numbers2 = gl.amd.cdna3.buffer_load(
                ptr=block_tables_ptr
                + sequence_idx * stride_block_table_seq
                + kv_block_start_idx2 // CONTEXT_PARTITION_SIZE_PER_BLOCK,
                offsets=block_indices,
                mask=block_indices < (max_num_kv_blocks - kv_block_start_idx2),
            )

        kv_block_numbers2 = _amd_iglp_sched_barrier(kv_block_numbers2, 0x0)
        boundary_mask = qk_row_mask[:, None] & causal_mask

        attention_scores = gl.convert_layout(attention_scores, layout=qk_linear_layout)
        attention_scores = qk_scale_value * attention_scores
        # Apply masking to attention scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)
        attention_scores = gl.where(boundary_mask, attention_scores, float(-3.4e38))

        # ==================== SOFTMAX COMPUTATION ====================
        # Optimization: For per-token quant mode, load value_scale early and fuse its reduction
        # with softmax max reduction to share the same barrier synchronization
        if KV_QUANT_MODE == 1:
            # Load value_scale for fused reduction (reduces one barrier)
            value_scale_value = value_scale_shared.load(
                gl.SliceLayout(0, qk_linear_layout)
            )
            valid_token_mask = qk_column_offsets < context_length
            # Mask out value_scale of invalid tokens
            value_scale_value = gl.where(
                valid_token_mask, value_scale_value, float(0.0)
            )
            value_scale_broadcast = tl.broadcast_to(
                value_scale_value[None, :], attention_scores.shape
            )
            current_max_logits, value_scale_max_2d = gl.reduce(
                (attention_scores, value_scale_broadcast),
                axis=1,
                combine_fn=_fused_max_combine,
            )
            # All rows are identical (broadcast from 1D); collapse to scalar.
            # This is a cheap intra-wave reduction, no extra cross-wave barrier.
            value_scale_max = gl.max(value_scale_max_2d, axis=0)
        else:
            # Update running maximum for numerical stability
            current_max_logits = gl.max(attention_scores, axis=1)

        new_max_logits = gl.maximum(max_logits, current_max_logits)

        accumulator_scale = tl.math.exp2((max_logits - new_max_logits) * LOG2_E)
        # Compute attention probabilities
        attention_probs = tl.math.exp2(
            (attention_scores - new_max_logits[:, None]) * LOG2_E
        )

        exp_sums = accumulator_scale * exp_sums + gl.sum(attention_probs, axis=1)
        kv_block_numbers2 = kv_block_numbers2.to(gl.int64)
        if MAX_NUM_KV_BLOCKS_PER_COMPUTE == 1:
            key_block_offsets2 = (
                kv_block_numbers2 * stride_key_block
                + head_size_split_offsets[None, :, None, None] * stride_key_head_split
                # Use runtime stride for KV block element (may be padded for large blocks).
                + (page_offset + block_element_offsets)[None, None, :, None]
                * stride_key_block_elem
                + contiguous_kv_element_offsets[None, None, None, :]
            )
        else:
            key_block_offsets2 = (
                kv_block_numbers2[:, None, None, None] * stride_key_block
                + head_size_split_offsets[None, :, None, None] * stride_key_head_split
                # Use runtime stride for KV block element (may be padded for large blocks).
                + (page_offset + block_element_offsets)[None, None, :, None]
                * stride_key_block_elem
                + contiguous_kv_element_offsets[None, None, None, :]
            )

        # ==================== VALUE ACCUMULATION ====================
        # Handle value quantization scaling for FP8
        if KV_QUANT_MODE >= 0:
            if KV_QUANT_MODE == 1:
                # Per-token quantization scaling
                # value_scale_value and value_scale_max already computed above (fused with softmax max)
                # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                # Optimization: use fast reciprocal (v_rcp_f32) instead of full IEEE-754 division
                # Use HIP libdevice fast_dividef which generates v_rcp_f32 + one Newton-Raphson step
                # This reduces ~12 instructions to ~3-4 instructions with acceptable precision loss
                # The relative error is < 2^-22 which is sufficient for FP8 quantization scaling
                inv_value_scale_max = hip_libdevice.fast_dividef(
                    1.0, value_scale_max + 1e-8
                )
                fp8_inv_scale = float(FP8_MAX_VALUE) * inv_value_scale_max
                value_scale_value = value_scale_value * fp8_inv_scale
                attention_probs = value_scale_value[None, :] * attention_probs
                # Use multiply by reciprocal instead of divide (precomputed 1/FP8_MAX_VALUE)
                probability_scale = value_scale_max * (1.0 / float(FP8_MAX_VALUE))
            elif KV_QUANT_MODE == 0:
                # Per-tensor quantization scaling
                probability_scale = value_scale_value
            else:
                raise ValueError(f"Invalid KV_QUANT_MODE: {KV_QUANT_MODE}")

        # Convert attention probabilities to compute type for MFMA operation
        # Convert layouts for PV MFMA operation
        attention_probs = attention_probs.to(COMPUTE_TYPE)
        probs_shared.store(attention_probs)
        values_converted = value_tensor.to(COMPUTE_TYPE)
        values_converted = gl.convert_layout(
            values_converted, layout=pv_rhs_operand_layout
        )

        accumulator_scale_expanded = gl.convert_layout(
            accumulator_scale[:, None], layout=pv_mfma_layout
        )
        attention_accumulator *= accumulator_scale_expanded

        pv_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2),
            dtype=gl.float32,
            layout=pv_mfma_layout,
        )
        probs_converted = probs_shared.load(pv_lhs_operand_layout)
        key_block_offsets2 = _amd_iglp_sched_barrier(key_block_offsets2, 0x0)
        key_tensor2 = gl.load(key_cache_ptr + key_block_offsets2)

        attention_output = gl.amd.cdna3.mfma(
            probs_converted, values_converted, pv_accumulator
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, VMEM_LOAD, 2, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, MFMA, 4, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, VMEM_LOAD, 2, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, MFMA, 4, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, VMEM_LOAD, 2, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, MFMA, 4, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, VMEM_LOAD, 2, 1
        )
        key_block_offsets2 = _amd_iglp_sched_group_barrier(
            key_block_offsets2, MFMA, 4, 1
        )
        key_block_offsets2 = _amd_iglp_sched_barrier(key_block_offsets2, 0x0)
        if KV_QUANT_MODE >= 0:
            attention_accumulator += probability_scale * attention_output
        else:
            attention_accumulator += attention_output
        max_logits = new_max_logits
        if (
            sequence_partition_idx + CONTEXT_PARTITION_SIZE_PER_BLOCK
            < sequence_partition_end_idx
        ):
            kv_block_numbers = kv_block_numbers2
            key_tensor = key_tensor2
            kv_block_start_idx = kv_block_start_idx2

    # ==================== SINKS HANDLING ====================
    # Add sinks contribution to exp_sums (does not contribute to attention output)
    if ONE_SHOT and sinks_ptr is not None:
        exp_sums += tl.math.exp2((sinks_values.to(gl.float32) - max_logits) * LOG2_E)

    # ==================== OUTPUT NORMALIZATION AND STORING ====================
    # Normalize attention output by softmax denominator
    # Guard against division by zero when all tokens are masked for a query position
    exp_sums_safe = tl.where(exp_sums > 0, exp_sums, 1.0)
    exp_sums_reciprocal = 1.0 / exp_sums_safe
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt

    if not ONE_SHOT:
        # Store results to global memory
        store_temporary_result(
            max_logits,
            exp_sums,
            attention_accumulator,
            max_logits_ptr,
            exp_sums_ptr,
            output_ptr,
            max_logits_offsets,
            output_offsets,
            qk_row_mask,
            output_mask,
        )
    else:
        attention_accumulator = attention_accumulator.to(OUTPUT_DTYPE)
        if QUERY_SEQ_LEN_POW2 == 1:
            attention_accumulator = gl.reshape(
                attention_accumulator,
                [ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2],
            )
            attention_accumulator = gl.convert_layout(
                attention_accumulator, layout=blocked_query_layout
            )
        else:
            attention_accumulator = gl.reshape(
                attention_accumulator,
                [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2],
            )
            attention_accumulator = gl.convert_layout(
                attention_accumulator, layout=mtp_blocked_query_layout
            )

        gl.amd.cdna3.buffer_store(
            stored_value=attention_accumulator,
            ptr=output_ptr,
            offsets=output_offsets,
            mask=output_mask,
        )


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"waves_per_eu": wa}, maxnreg=512 // wa if wa > 0 else None, num_stages=1
#         )
#         for wa in range(5)
#     ],
#     key=[
#         "KV_BLOCK_SIZE",
#         "SLIDING_WINDOW",
#         "KV_QUANT_MODE",
#         "QUERY_QUANT_MODE",
#         "ONE_QUERY_GROUP_SIZE_POW2",
#         "QUERY_SEQ_LEN_POW2",
#         "HEAD_SIZE_POW2",
#         "COMPUTE_TYPE",
#         "ONE_SHOT",
#     ],
#     cache_results=True,
# )
@gluon.jit
def paged_attention_decode_sliding_window(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    query_ptr,  # [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale: float,
    query_scale,  # [num_seqs, query_length, num_kv_heads, query_group_size, 1](per-token) or [1](per-tensor) or None
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    sinks_ptr,  # [num_query_heads]
    stride_max_logits_seq: int,
    stride_max_logits_head: int,
    stride_max_logits_part: int,
    # 5D output strides for [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    stride_output_bs: int,
    stride_output_len: int,
    stride_output_kv_head: int,
    stride_output_group_size: int,
    # 5D query strides for [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    stride_query_bs: int,
    stride_query_qlen: int,
    stride_query_kv_head: int,
    stride_query_group_size: int,
    stride_key_block: int,
    stride_key_head: int,
    stride_key_head_split: int,
    stride_key_block_elem: int,
    stride_value_block: int,
    stride_value_head: int,
    stride_value_head_size: int,
    stride_block_table_seq: int,
    stride_query_scale_bs: int,
    stride_query_scale_qlen: int,
    stride_query_scale_kv_head: int,
    kv_scale_stride_0: int,
    kv_scale_stride_1: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN_POW2: gl.constexpr,
    ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr,
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
    ONE_SHOT: gl.constexpr = False,
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
    # ==================== VALIDATION CHECKS ====================
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
    sequence_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    split_idx = gl.program_id(2)

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
    MAX_NUM_KV_BLOCKS_PER_COMPUTE: gl.constexpr = triton.cdiv(
        CONTEXT_PARTITION_SIZE, KV_BLOCK_SIZE
    )

    CONTEXT_PARTITION_SIZE_PER_BLOCK: gl.constexpr = triton.cdiv(
        KV_BLOCK_SIZE, CONTEXT_PARTITION_SIZE
    )
    sequence_split_idx = split_idx // CONTEXT_PARTITION_SIZE_PER_BLOCK
    block_split_idx = split_idx % CONTEXT_PARTITION_SIZE_PER_BLOCK
    # ==================== MEMORY LAYOUT DEFINITIONS ====================
    # Query tensor layout - optimized for sequential access (2D)
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, order=[1, 0])
    shared_probs_layout: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 8, order=[1, 0])
    QUERY_GROUP_SIZE_POW2: gl.constexpr = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2
    # MTP Query tensor layout (3D) [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    if ONE_QUERY_GROUP_SIZE_POW2 <= 16:
        # ONE_QUERY_GROUP_SIZE_POW2 may be 4, 8, 16
        # corresponding Q_WARPS_PER_CTA_DIM1 should be 1, 2, 4
        # corresponding Q_WARPS_PER_CTA_DIM0 should be 4, 2, 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = triton.cdiv(ONE_QUERY_GROUP_SIZE_POW2, 4)
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
    if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE:
        KV_COMPUTE_BLOCK_SIZE: gl.constexpr = CONTEXT_PARTITION_SIZE
    else:
        KV_COMPUTE_BLOCK_SIZE: gl.constexpr = KV_BLOCK_SIZE

    key_warps_per_cta: gl.constexpr = (
        [4, 1, 1, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 1, 4, 1]
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

    qk_linear_layout: gl.constexpr = define_layout(
        QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE, QUERY_SEQ_LEN_POW2
    )

    context_length = gl.load(context_lengths_ptr + sequence_idx)
    # Value cache layout configuration based on transpose flag
    if VALUE_TRANSPOSED:
        # Transposed value layout for better memory access patterns
        value_threads_per_warp: gl.constexpr = (
            [4, 1, 16, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 4, 16, 1]
        )

        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )

        value_dim1_offsets = gl.arange(
            0,
            KV_COMPUTE_BLOCK_SIZE // KV_16B_ELEMENT_COUNT,
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
            [4, 16, 1] if KV_COMPUTE_BLOCK_SIZE == 16 else [1, 16, 4]
        )
        blocked_value_layout: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, KV_16B_ELEMENT_COUNT],
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
            KV_COMPUTE_BLOCK_SIZE,
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
        operand_index=0, parent=pv_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
    )
    pv_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=pv_mfma_layout, k_width=KV_16B_ELEMENT_COUNT
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
    block_element_offsets = gl.arange(
        0, KV_COMPUTE_BLOCK_SIZE, layout=block_element_layout
    )
    contiguous_kv_element_offsets = gl.arange(
        0, KV_16B_ELEMENT_COUNT, layout=contiguous_kv_elements_layout
    )
    qk_row_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    query_row_mask_3d = (mtp_query_len_offsets[:, None, None] < query_seq_len) & (
        mtp_query_group_size_offsets[None, :, None] < query_group_size
    )
    query_row_mask_1d = gl.reshape(query_row_mask_3d, [QUERY_GROUP_SIZE_POW2])

    # For sinks handling
    # Convert MTP layout indices to continuous indices for exp_sums/max_logits
    output_group_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    output_query_len_idx = output_group_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    output_group_idx_in_len = output_group_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2

    output_group_offsets = (
        output_query_len_idx * query_group_size + output_group_idx_in_len
    )
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )

    # ==================== PROGRAM ID AND INITIALIZATION ====================

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
    mtp_query_offsets = (
        sequence_idx * stride_query_bs
        + mtp_query_len_offsets[:, None, None] * stride_query_qlen
        + kv_head_idx * stride_query_kv_head
        + mtp_query_group_size_offsets[None, :, None] * stride_query_group_size
        + mtp_head_size_offsets[None, None, :]
    )
    mtp_query_mask = (
        (mtp_query_len_offsets[:, None, None] < query_seq_len)
        & (mtp_query_group_size_offsets[None, :, None] < query_group_size)
        & (mtp_head_size_offsets[None, None, :] < head_size)
    )
    mtp_query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=mtp_query_offsets, mask=mtp_query_mask
    )
    if SLIDING_WINDOW > 0:
        sequence_end_idx = context_length
        sequence_start_idx = context_length - SLIDING_WINDOW
        window_partition_start_idx = gl.maximum(
            0, sequence_start_idx // CONTEXT_PARTITION_SIZE
        )
        window_partition_end_idx = gl.cdiv(sequence_end_idx, CONTEXT_PARTITION_SIZE)
        window_partition_count = gl.maximum(
            0, window_partition_end_idx - window_partition_start_idx
        )
        sequence_split_count = gl.maximum(
            1, gl.num_programs(2) // CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        partitions_per_sequence_split = gl.cdiv(
            window_partition_count, sequence_split_count
        )
        local_partition_start_idx = (
            window_partition_start_idx
            + partitions_per_sequence_split * sequence_split_idx
        )
        local_partition_end_idx = gl.minimum(
            window_partition_end_idx,
            window_partition_start_idx
            + partitions_per_sequence_split * (sequence_split_idx + 1),
        )
        sequence_partition_start_idx = local_partition_start_idx + block_split_idx
        sequence_partition_end_idx = local_partition_end_idx + block_split_idx
    else:
        page_size = gl.cdiv(
            context_length, gl.num_programs(2) // CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        sequence_start_idx = page_size * sequence_split_idx

        sequence_end_idx = gl.minimum(
            context_length, page_size * (sequence_split_idx + 1)
        )
        sequence_partition_start_idx = (
            sequence_start_idx // CONTEXT_PARTITION_SIZE + block_split_idx
        )
        sequence_partition_end_idx = (
            gl.cdiv(sequence_end_idx, CONTEXT_PARTITION_SIZE) + block_split_idx
        )
    block_indices = gl.arange(0, MAX_NUM_KV_BLOCKS_PER_COMPUTE, layout=block_id_layout)
    max_num_kv_blocks = gl.cdiv(context_length, KV_COMPUTE_BLOCK_SIZE)
    kv_block_start_idx = sequence_partition_start_idx * MAX_NUM_KV_BLOCKS_PER_COMPUTE
    kv_block_numbers = gl.amd.cdna3.buffer_load(
        ptr=block_tables_ptr
        + sequence_idx * stride_block_table_seq
        + kv_block_start_idx // CONTEXT_PARTITION_SIZE_PER_BLOCK,
        offsets=block_indices,
        mask=block_indices < (max_num_kv_blocks - kv_block_start_idx),
        cache=".cg",
    )
    max_logits_base_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    max_logits_query_len_idx = max_logits_base_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    max_logits_group_idx_in_len = (
        max_logits_base_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    )

    max_logits_base_offsets = (
        max_logits_query_len_idx * query_group_size + max_logits_group_idx_in_len
    )
    # Output shape: [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    if ONE_SHOT:
        output_offsets = (
            sequence_idx * stride_output_bs
            + mtp_query_len_offsets[:, None, None] * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + mtp_query_group_size_offsets[None, :, None] * stride_output_group_size
            + mtp_head_size_offsets[None, None, :]
        )
    else:
        output_offsets = sequence_idx * stride_output_bs
        output_offsets += kv_head_idx * stride_output_len
        output_offsets += (
            split_idx * stride_output_kv_head
            + output_group_offsets[:, None] * stride_output_group_size
            + output_head_size_offsets[None, :]
        )

        max_logits_offsets = (
            sequence_idx * stride_max_logits_seq
            + kv_head_idx * stride_max_logits_head
            + split_idx * stride_max_logits_part
            + max_logits_base_offsets
        )
    qk_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    pv_row_mask = gl.convert_layout(
        qk_row_mask, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    if ONE_SHOT:
        output_mask = mtp_query_mask
    else:
        output_mask = pv_row_mask[:, None] & (
            output_head_size_offsets[None, :] < head_size
        )

    if (sequence_start_idx >= context_length) | (
        sequence_partition_start_idx >= sequence_partition_end_idx
    ):
        if not ONE_SHOT:
            store_temporary_result(
                max_logits,
                exp_sums,
                attention_accumulator,
                max_logits_ptr,
                exp_sums_ptr,
                output_ptr,
                max_logits_offsets,
                output_offsets,
                qk_row_mask,
                output_mask,
            )
        return  # No computation needed for this partition

    # Load query tensor with 3D MTP layout
    # Query shape: [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    mtp_query_tensor = gl.reshape(
        mtp_query_tensor, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    )
    query_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE, mtp_query_tensor.shape, shared_query_layout
    )

    probs_shared = gl.allocate_shared_memory(
        COMPUTE_TYPE,
        [QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE],
        shared_probs_layout,
    )

    # Load query quantization scales if needed
    if QUERY_QUANT_MODE == 0:
        # Per-tensor quantization
        query_scale_value = gl.load(query_scale)
    elif QUERY_QUANT_MODE == 1:
        # Per-token quantization
        query_scale_offsets = (
            sequence_idx * stride_query_scale_bs
            + mtp_query_len_offsets[:, None, None] * stride_query_scale_qlen
            + kv_head_idx * stride_query_scale_kv_head
            + mtp_query_group_size_offsets[None, :, None]
        )
        query_scale_mask = (mtp_query_len_offsets[:, None, None] < query_seq_len) & (
            mtp_query_group_size_offsets[None, :, None] < query_group_size
        )
        query_scale_value = gl.amd.cdna3.buffer_load(
            ptr=query_scale,
            offsets=query_scale_offsets,
            mask=query_scale_mask,
        )
        query_scale_value = gl.reshape(query_scale_value, [QUERY_GROUP_SIZE_POW2, 1])
        query_scale_value = gl.convert_layout(
            query_scale_value, layout=qk_linear_layout
        )

    # ==================== SEQUENCE PROCESSING ====================

    if ONE_SHOT and sinks_ptr is not None:
        # sinks_ptr is per-query-head: [num_query_heads] where
        # num_query_heads = num_kv_heads * query_group_size.
        # It is shared across query positions (query_seq_len).
        sinks_values = gl.load(
            sinks_ptr + kv_head_idx * query_group_size + max_logits_group_idx_in_len,
            mask=qk_row_mask,
            other=float("-inf"),
            cache_modifier=".cg",
        )

    if KV_QUANT_MODE == 0:
        # Per-tensor quantization
        key_scale_value = gl.load(key_scale)
        value_scale_value = gl.load(value_scale)

    if QUERY_QUANT_MODE < 0 and COMPUTE_TYPE.is_fp8():
        # Quantize bf16 query to fp8
        # Convert query to float32 for computation
        query_f32 = mtp_query_tensor.to(gl.float32)
        # Compute max absolute value for scaling
        query_abs = gl.abs(query_f32)
        query_max_abs = gl.max(query_abs, axis=1, keep_dims=True)
        # Compute scale factor: FP8_MAX_VALUE / max_abs_value
        # Add epsilon to avoid division by zero
        query_scale_value = query_max_abs / float(FP8_MAX_VALUE)
        # Quantize: scale query to fp8 range and convert to fp8 type
        mtp_query_tensor = query_f32.to(COMPUTE_TYPE)
    else:
        mtp_query_tensor = mtp_query_tensor.to(COMPUTE_TYPE)

    query_shared.store(mtp_query_tensor)

    page_offset = (
        kv_block_start_idx % CONTEXT_PARTITION_SIZE_PER_BLOCK
    ) * CONTEXT_PARTITION_SIZE

    kv_block_numbers = kv_block_numbers.to(gl.int64)
    key_block_offsets = (
        kv_block_numbers[:, None, None, None] * stride_key_block
        + kv_head_idx * stride_key_head
        + head_size_split_offsets[None, :, None, None] * stride_key_head_split
        # Use runtime stride for KV block element (may be padded for large blocks).
        + (page_offset + block_element_offsets)[None, None, :, None]
        * stride_key_block_elem
        + contiguous_kv_element_offsets[None, None, None, :]
    )
    key_tensor = gl.load(key_cache_ptr + key_block_offsets, cache_modifier=".cg")
    query_converted = query_shared.load(qk_lhs_operand_layout)
    for sequence_partition_idx in range(
        sequence_partition_start_idx,
        sequence_partition_end_idx,
        CONTEXT_PARTITION_SIZE_PER_BLOCK,
    ):

        # ==================== KEY LOADING AND PROCESSING ====================
        # Calculate key cache offsets and load keys
        kv_block_start_idx2 = (
            kv_block_start_idx
            + MAX_NUM_KV_BLOCKS_PER_COMPUTE * CONTEXT_PARTITION_SIZE_PER_BLOCK
        )
        kv_block_numbers2 = gl.amd.cdna3.buffer_load(
            ptr=block_tables_ptr
            + sequence_idx * stride_block_table_seq
            + kv_block_start_idx2 // CONTEXT_PARTITION_SIZE_PER_BLOCK,
            offsets=block_indices,
            mask=block_indices < (max_num_kv_blocks - kv_block_start_idx2),
            cache=".cg",
        )

        page_offset2 = (
            kv_block_start_idx2 % CONTEXT_PARTITION_SIZE_PER_BLOCK
        ) * CONTEXT_PARTITION_SIZE

        # kv_block_numbers2_i32 = gl.where(valid_block_mask, kv_block_numbers2_i32, 0)
        kv_block_numbers2 = kv_block_numbers2.to(gl.int64)
        key_block_offsets2 = (
            kv_block_numbers2[:, None, None, None] * stride_key_block
            + kv_head_idx * stride_key_head
            + head_size_split_offsets[None, :, None, None] * stride_key_head_split
            # Use runtime stride for KV block element (may be padded for large blocks).
            + (page_offset2 + block_element_offsets)[None, None, :, None]
            * stride_key_block_elem
            + contiguous_kv_element_offsets[None, None, None, :]
        )

        # Prepare QK MFMA while key loads (these don't depend on key data)
        qk_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE),
            dtype=gl.float32,
            layout=qk_mfma_layout,
        )
        # Load key quantization scales if needed (overlaps with key tensor load)
        if KV_QUANT_MODE == 1:
            # Per-token quantization - prepare offsets while key loads
            key_scale_offsets = (
                kv_block_numbers[:, None, None, None] * kv_scale_stride_0
                + kv_head_idx * kv_scale_stride_1
                + (page_offset + block_element_offsets)[None, None, :, None]
            )
            # Optimize: Load both scales with VMEM scheduling, overlap with key reshape
            if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE and SLIDING_WINDOW > 0:
                kv_token_global = (
                    kv_block_start_idx * KV_COMPUTE_BLOCK_SIZE + block_element_offsets
                )
                kv_in_window_mask = kv_token_global >= sequence_start_idx

                key_scale_value_blocked = gl.load(
                    key_scale + key_scale_offsets,
                    mask=kv_in_window_mask[None, None, :, None],
                    other=0.0,
                )
                value_scale_value_blocked = gl.load(
                    value_scale + key_scale_offsets,
                    mask=kv_in_window_mask[None, None, :, None],
                    other=0.0,
                )
            else:
                key_scale_value_blocked = gl.load(key_scale + key_scale_offsets)
                value_scale_value_blocked = gl.load(value_scale + key_scale_offsets)

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
        key_converted = gl.permute(key_tensor, [1, 3, 0, 2])
        key_converted = gl.reshape(
            key_converted, [HEAD_SIZE_POW2, CONTEXT_PARTITION_SIZE]
        )

        # ==================== VALUE LOADING WITH QK MFMA OVERLAP ====================
        # Convert key layout for MFMA (query_converted and qk_accumulator already prepared above)
        key_converted = gl.convert_layout(key_converted, layout=qk_rhs_operand_layout)
        key_converted = key_converted.to(COMPUTE_TYPE)
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
                + (page_offset // KV_16B_ELEMENT_COUNT + value_dim1_offsets)[
                    None, :, None, None
                ]
                * stride_value_head_size
                + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
                + value_dim3_offsets[None, None, None, :]
            )
            if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE and SLIDING_WINDOW > 0:
                value_token_global = (
                    kv_block_start_idx * KV_COMPUTE_BLOCK_SIZE
                    + value_dim1_offsets[None, :, None, None] * KV_16B_ELEMENT_COUNT
                    + value_dim3_offsets[None, None, None, :]
                )
                value_in_window_mask = value_token_global >= sequence_start_idx

                value_tensor = gl.load(
                    value_cache_ptr + value_block_offsets,
                    cache_modifier=".cg",
                )
                value_tensor = gl.where(value_in_window_mask, value_tensor, 0.0)
            else:
                value_tensor = gl.load(
                    value_cache_ptr + value_block_offsets, cache_modifier=".cg"
                )

            # Permute and reshape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 1, 3, 2])
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
                + (page_offset + value_dim2_offsets)[None, None, :]
            )

            # Schedule: Start value VMEM load, then QK MFMA
            if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE and SLIDING_WINDOW > 0:
                value_token_global = (
                    kv_block_start_idx * KV_COMPUTE_BLOCK_SIZE + value_dim2_offsets
                )
                value_in_window_mask = value_token_global >= sequence_start_idx
                value_tensor = gl.load(
                    value_cache_ptr + value_block_offsets,
                    cache_modifier=".cg",
                )
                value_tensor = gl.where(
                    value_in_window_mask[None, None, :], value_tensor, 0.0
                )
            else:
                value_tensor = gl.load(
                    value_cache_ptr + value_block_offsets, cache_modifier=".cg"
                )

            # Permute and resape for matrix multiplication
            value_tensor = gl.permute(value_tensor, [0, 2, 1])

        # Compute QK attention scores using MFMA (overlaps with value load)
        attention_scores = gl.amd.cdna3.mfma(
            query_converted, key_converted, qk_accumulator
        )
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

        # ==================== ATTENTION MASKING ====================
        # Compute query token index (0 to query_seq_len-1)
        query_token_idx = qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2
        qk_column_offsets = kv_block_start_idx * KV_COMPUTE_BLOCK_SIZE + gl.arange(
            0, CONTEXT_PARTITION_SIZE, layout=gl.SliceLayout(0, qk_linear_layout)
        )
        # Apply causal masking if required
        if IS_CAUSAL:
            # Compute causal mask based on sequence positions
            sequence_position_extension = query_seq_len - 1 - query_token_idx
            causal_mask = (
                sequence_position_extension[:, None] + qk_column_offsets[None, :]
                < sequence_end_idx
            )
            if SLIDING_WINDOW > 0:
                causal_mask = causal_mask & (
                    sequence_position_extension[:, None] + qk_column_offsets[None, :]
                    >= sequence_start_idx + query_seq_len
                )
            else:
                causal_mask = causal_mask & (
                    sequence_position_extension[:, None] + qk_column_offsets[None, :]
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
        attention_scores = gl.convert_layout(attention_scores, layout=qk_linear_layout)
        attention_scores = qk_scale_value * attention_scores
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
                valid_token_mask = qk_column_offsets < context_length
                # Mask out value_scale of invalid tokens
                value_scale_value = gl.where(
                    valid_token_mask, value_scale_value, float(0.0)
                )
                value_scale_max = gl.max(value_scale_value, axis=0)
                # Scale the maximum value of value_scale to FP8_MAX_VALUE to improve the precision of P * V
                # Optimization: compute reciprocal once and reuse, use multiply instead of divide for FP8_MAX_VALUE
                inv_value_scale_max = 1.0 / (value_scale_max + 1e-8)
                value_scale_value = (
                    value_scale_value * float(FP8_MAX_VALUE) * inv_value_scale_max
                )
                attention_probs = value_scale_value[None, :] * attention_probs
                # Use multiply by reciprocal instead of divide (precomputed 1/FP8_MAX_VALUE)
                probability_scale = value_scale_max * (1.0 / float(FP8_MAX_VALUE))
            elif KV_QUANT_MODE == 0:
                # Per-tensor quantization scaling
                probability_scale = value_scale_value
            else:
                raise ValueError(f"Invalid KV_QUANT_MODE: {KV_QUANT_MODE}")

        # Convert attention probabilities to compute type for MFMA operation
        # Convert layouts for PV MFMA operation
        attention_probs = attention_probs.to(COMPUTE_TYPE)
        probs_shared.store(attention_probs)
        values_converted = value_tensor.to(COMPUTE_TYPE)
        values_converted = gl.convert_layout(
            values_converted, layout=pv_rhs_operand_layout
        )

        accumulator_scale_expanded = gl.convert_layout(
            accumulator_scale[:, None], layout=pv_mfma_layout
        )
        attention_accumulator *= accumulator_scale_expanded

        pv_accumulator = gl.zeros(
            (QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2),
            dtype=gl.float32,
            layout=pv_mfma_layout,
        )
        probs_converted = probs_shared.load(pv_lhs_operand_layout)
        if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE and SLIDING_WINDOW > 0:
            kv_token_global2 = (
                kv_block_start_idx2 * KV_COMPUTE_BLOCK_SIZE + block_element_offsets
            )
            kv_in_window_mask2 = kv_token_global2 >= sequence_start_idx

            key_tensor2 = gl.load(
                key_cache_ptr + key_block_offsets2,
                mask=kv_in_window_mask2[None, None, :, None],
                other=0.0,
                cache_modifier=".cg",
            )
        else:
            key_tensor2 = gl.load(
                key_cache_ptr + key_block_offsets2, cache_modifier=".cg"
            )

        attention_output = gl.amd.cdna3.mfma(
            probs_converted, values_converted, pv_accumulator
        )
        if KV_QUANT_MODE >= 0:
            attention_accumulator += probability_scale * attention_output
        else:
            attention_accumulator += attention_output
        max_logits = new_max_logits
        if (
            sequence_partition_idx + CONTEXT_PARTITION_SIZE_PER_BLOCK
            < sequence_partition_end_idx
        ):
            kv_block_numbers = kv_block_numbers2
            key_tensor = key_tensor2
            kv_block_start_idx = kv_block_start_idx2
            page_offset = page_offset2

    # ==================== SINKS HANDLING ====================
    # Add sinks contribution to exp_sums (does not contribute to attention output)
    if ONE_SHOT and sinks_ptr is not None:
        exp_sums += tl.math.exp2((sinks_values.to(gl.float32) - max_logits) * LOG2_E)

    # ==================== OUTPUT NORMALIZATION AND STORING ====================
    # Normalize attention output by softmax denominator
    # Guard against division by zero when all tokens are masked for a query position
    exp_sums_safe = tl.where(exp_sums > 0, exp_sums, 1.0)
    exp_sums_reciprocal = 1.0 / exp_sums_safe
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt

    if not ONE_SHOT:
        # Store results to global memory
        store_temporary_result(
            max_logits,
            exp_sums,
            attention_accumulator,
            max_logits_ptr,
            exp_sums_ptr,
            output_ptr,
            max_logits_offsets,
            output_offsets,
            qk_row_mask,
            output_mask,
        )
    else:
        attention_accumulator = attention_accumulator.to(OUTPUT_DTYPE)
        # Reshape to 3D and store
        # attention_accumulator is [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
        # Reshape to [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]

        attention_accumulator = gl.reshape(
            attention_accumulator,
            [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2],
        )
        attention_accumulator = gl.convert_layout(
            attention_accumulator, layout=mtp_blocked_query_layout
        )

        gl.amd.cdna3.buffer_store(
            stored_value=attention_accumulator,
            ptr=output_ptr,
            offsets=output_offsets,
            mask=output_mask,
        )


# @triton.autotune(
#     configs=[
#         triton.Config(
#             {"waves_per_eu": wa}, maxnreg=512 // wa if wa > 0 else None, num_stages=1
#         )
#         for wa in range(5)
#     ],
#     key=[
#         "KV_BLOCK_SIZE",
#         "SLIDING_WINDOW",
#         "KV_QUANT_MODE",
#         "QUERY_QUANT_MODE",
#         "ONE_QUERY_GROUP_SIZE_POW2",
#         "QUERY_SEQ_LEN_POW2",
#         "HEAD_SIZE_POW2",
#         "COMPUTE_TYPE",
#         "ONE_SHOT",
#     ],
#     cache_results=True,
# )
@gluon.jit
def paged_attention_decode_v2_gluon_dot_kernel(
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size]
    output_ptr,  # [num_seqs, num_kv_heads, max_parts, q_group_size, head_size]
    query_ptr,  # [num_seqs, query_length, num_kv_heads, query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, query_length, num_kv_heads, query_group_size, 1](per-token) or [1](per-tensor) or None
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    stride_max_logits_seq: int,
    stride_max_logits_head: int,
    stride_max_logits_part: int,
    stride_output_seq: int,
    stride_output_head: int,
    stride_output_part: int,
    stride_output_group: int,
    stride_query_bs: int,
    stride_query_qlen: int,
    stride_query_kv_head: int,
    stride_query_group_size: int,
    stride_key_block: int,
    stride_key_head: int,
    stride_key_head_split: int,
    stride_key_block_elem: int,
    stride_value_block: int,
    stride_value_head: int,
    stride_value_head_size: int,
    stride_block_table_seq: int,
    stride_query_scale_bs: int,
    stride_query_scale_qlen: int,
    stride_query_scale_kv_head: int,
    kv_scale_stride_0: int,
    kv_scale_stride_1: int,
    head_size: int,
    num_seqs: int,
    num_kv_heads: int,
    max_context_partition_num: int,
    COMPUTE_TYPE: gl.constexpr,
    QUERY_SEQ_LEN: gl.constexpr,
    ONE_QUERY_GROUP_SIZE: gl.constexpr,
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
    SLIDING_WINDOW: gl.constexpr = 0,
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
    if COMPUTE_TYPE.is_fp8() or CDNA_VERSION == 4:
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

    # Calculate MTP (Multi-Token Prefill) layout parameters
    QUERY_SEQ_LEN_POW2: gl.constexpr = triton.next_power_of_2(QUERY_SEQ_LEN)
    if ONE_QUERY_GROUP_SIZE <= 16 // QUERY_SEQ_LEN_POW2:
        ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr = 16 // QUERY_SEQ_LEN_POW2
    else:
        ONE_QUERY_GROUP_SIZE_POW2: gl.constexpr = triton.next_power_of_2(
            ONE_QUERY_GROUP_SIZE
        )
    QUERY_GROUP_SIZE_POW2: gl.constexpr = QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2

    K_HEAD_SIZE_SPLITS: gl.constexpr = HEAD_SIZE_POW2 // KV_16B_ELEMENT_COUNT
    MAX_NUM_KV_BLOCKS_PER_COMPUTE: gl.constexpr = KV_COMPUTE_BLOCK_SIZE // KV_BLOCK_SIZE

    # ==================== MEMORY LAYOUT DEFINITIONS ====================
    # MTP Query tensor layout - 3D [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    if ONE_QUERY_GROUP_SIZE_POW2 <= 16:
        # ONE_QUERY_GROUP_SIZE_POW2 may be 4, 8, 16
        # corresponding Q_WARPS_PER_CTA_DIM1 should be 1, 2, 4
        # corresponding Q_WARPS_PER_CTA_DIM0 should be 4, 2, 1
        Q_WARPS_PER_CTA_DIM1: gl.constexpr = triton.cdiv(ONE_QUERY_GROUP_SIZE_POW2, 4)
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
    # [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    blocked_query_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 16],
        warps_per_cta=[4, 1],
        order=[1, 0],
    )
    shared_query_layout: gl.constexpr = gl.SwizzledSharedLayout(
        KV_16B_ELEMENT_COUNT, 1, 16, order=[1, 0]
    )

    # Key cache layout - optimized for block-wise access patterns
    blocked_key_layout_fp8: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=[4, 1, 1, 1],
        order=[3, 2, 1, 0],
    )
    key_warps_per_cta_f16: gl.constexpr = (
        [4, 1, 1, 1] if KV_BLOCK_SIZE == 16 else [1, 1, 4, 1]
    )
    blocked_key_layout_f16: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 1, 1, KV_16B_ELEMENT_COUNT],
        threads_per_warp=[1, 4, 16, 1],
        warps_per_cta=key_warps_per_cta_f16,
        order=[3, 2, 1, 0],
    )
    blocked_key_layout: gl.constexpr = (
        blocked_key_layout_fp8 if KV_16B_ELEMENT_COUNT == 16 else blocked_key_layout_f16
    )

    DOT_QK_K_WIDTH: gl.constexpr = KV_16B_ELEMENT_COUNT
    # QK Matrix multiplication layout using AMD MFMA instructions
    qk_mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=CDNA_VERSION,
        instr_shape=QK_PV_MFMA_INSTR_SHAPE,
        transposed=True,
        warps_per_cta=[1, 4],
    )
    qk_lhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=qk_mfma_layout, k_width=DOT_QK_K_WIDTH
    )
    qk_rhs_operand_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=qk_mfma_layout, k_width=DOT_QK_K_WIDTH
    )

    # Register allocation configuration based on group size and compute block size
    qk_linear_layout: gl.constexpr = define_layout(
        QUERY_GROUP_SIZE_POW2, CONTEXT_PARTITION_SIZE, QUERY_SEQ_LEN_POW2
    )

    # Value cache layout configuration based on transpose flag
    if VALUE_TRANSPOSED:
        # Transposed value layout for better memory access patterns
        value_threads_per_warp: gl.constexpr = (
            [4, 1, 16, 1] if KV_BLOCK_SIZE == 16 else [1, 4, 16, 1]
        )
        blocked_value_layout_f16: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, 8],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )
        blocked_value_layout_fp8: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[1, 1, 1, 16],
            threads_per_warp=value_threads_per_warp,
            warps_per_cta=[1, 1, 4, 1],
            order=[3, 2, 1, 0],
        )
        blocked_value_layout: gl.constexpr = (
            blocked_value_layout_fp8
            if KV_16B_ELEMENT_COUNT == 16
            else blocked_value_layout_f16
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
    contiguous_kv_element_offsets = gl.arange(
        0, KV_16B_ELEMENT_COUNT, layout=contiguous_kv_elements_layout
    )
    qk_row_offsets = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    query_row_mask_3d = (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN) & (
        mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE
    )
    query_row_mask_1d = gl.reshape(query_row_mask_3d, [QUERY_GROUP_SIZE_POW2])
    qk_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    pv_row_mask = gl.convert_layout(
        query_row_mask_1d, layout=gl.SliceLayout(1, pv_mfma_layout)
    )

    # ==================== PROGRAM ID AND INITIALIZATION ====================
    sequence_idx = gl.program_id(0)
    kv_head_idx = gl.program_id(1)
    output_partition_idx = gl.program_id(2)
    context_length = gl.load(context_lengths_ptr + sequence_idx)
    if SLIDING_WINDOW > 0:
        sequence_start_idx = context_length - SLIDING_WINDOW
        partition_offset = gl.maximum(0, sequence_start_idx // CONTEXT_PARTITION_SIZE)
        sequence_partition_idx = partition_offset + output_partition_idx
    else:
        sequence_start_idx = 0
        sequence_partition_idx = output_partition_idx

    # Load query tensor with 3D MTP layout
    # Query shape: [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    mtp_query_offsets = (
        sequence_idx * stride_query_bs
        + mtp_query_len_offsets[:, None, None] * stride_query_qlen
        + kv_head_idx * stride_query_kv_head
        + mtp_query_group_size_offsets[None, :, None] * stride_query_group_size
        + mtp_head_size_offsets[None, None, :]
    )
    mtp_query_mask = (
        (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN)
        & (mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE)
        & (mtp_head_size_offsets[None, None, :] < head_size)
    )
    # [QUERY_SEQ_LEN_POW2, ONE_QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    mtp_query_tensor = gl.amd.cdna3.buffer_load(
        ptr=query_ptr, offsets=mtp_query_offsets, mask=mtp_query_mask
    )
    mtp_query_tensor = gl.reshape(
        mtp_query_tensor, [QUERY_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    )
    query_tensor = gl.convert_layout(mtp_query_tensor, layout=blocked_query_layout)
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
            sequence_idx * stride_query_scale_bs
            + mtp_query_len_offsets[:, None, None] * stride_query_scale_qlen
            + kv_head_idx * stride_query_scale_kv_head
            + mtp_query_group_size_offsets[None, :, None]
        )
        query_scale_mask = (mtp_query_len_offsets[:, None, None] < QUERY_SEQ_LEN) & (
            mtp_query_group_size_offsets[None, :, None] < ONE_QUERY_GROUP_SIZE
        )
        query_scale_value = gl.amd.cdna3.buffer_load(
            ptr=query_scale,
            offsets=query_scale_offsets,
            mask=query_scale_mask,
        )
        query_scale_value = gl.reshape(query_scale_value, [QUERY_GROUP_SIZE_POW2, 1])
        query_scale_value = gl.convert_layout(
            query_scale_value, layout=qk_linear_layout
        )

    # Initialize output pointers and accumulators
    max_logits_base_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, qk_linear_layout)
    )
    # Convert MTP layout indices to continuous indices for exp_sums/max_logits
    max_logits_query_len_idx = max_logits_base_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    max_logits_group_idx_in_len = (
        max_logits_base_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    )
    max_logits_base_offsets = (
        max_logits_query_len_idx * ONE_QUERY_GROUP_SIZE + max_logits_group_idx_in_len
    )
    max_logits_offsets = (
        sequence_idx * stride_max_logits_seq
        + kv_head_idx * stride_max_logits_head
        + output_partition_idx * stride_max_logits_part
        + max_logits_base_offsets
    )

    output_group_offsets_mtp = gl.arange(
        0, QUERY_GROUP_SIZE_POW2, layout=gl.SliceLayout(1, pv_mfma_layout)
    )
    # Convert MTP layout indices to continuous indices for temporary_output
    # MTP layout: [QUERY_SEQ_LEN_POW2 * ONE_QUERY_GROUP_SIZE_POW2] with valid indices at non-contiguous positions
    # Continuous layout: [QUERY_SEQ_LEN * ONE_QUERY_GROUP_SIZE] with valid indices at contiguous positions
    output_query_len_idx = output_group_offsets_mtp // ONE_QUERY_GROUP_SIZE_POW2
    output_group_idx_in_len = output_group_offsets_mtp % ONE_QUERY_GROUP_SIZE_POW2
    output_group_offsets = (
        output_query_len_idx * ONE_QUERY_GROUP_SIZE + output_group_idx_in_len
    )
    output_head_size_offsets = gl.arange(
        0, HEAD_SIZE_POW2, layout=gl.SliceLayout(0, pv_mfma_layout)
    )
    output_mask = pv_row_mask[:, None] & (output_head_size_offsets[None, :] < head_size)

    output_offsets = sequence_idx * stride_output_seq
    output_offsets += kv_head_idx * stride_output_head
    output_offsets += (
        output_partition_idx * stride_output_part
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

    KV_COMPUTE_BLOCK_COUNT: gl.constexpr = (
        CONTEXT_PARTITION_SIZE // KV_COMPUTE_BLOCK_SIZE
    )
    SEQUENCE_PARTITION_KV_BLOCKS: gl.constexpr = CONTEXT_PARTITION_SIZE // KV_BLOCK_SIZE

    if SLIDING_WINDOW > 0:
        # Only program 0 processes the full sliding window in a single FP32
        # accumulation pass, avoiding BF16 intermediate quantization loss from
        # multi-partition PS combine. All other programs early-return with defaults.
        if output_partition_idx > 0:
            store_temporary_result(
                max_logits,
                exp_sums,
                attention_accumulator,
                max_logits_ptr,
                exp_sums_ptr,
                output_ptr,
                max_logits_offsets,
                output_offsets,
                qk_row_mask,
                output_mask,
            )
            return
        window_start_partition = gl.maximum(
            0, sequence_start_idx // CONTEXT_PARTITION_SIZE
        )
        window_end_partition = gl.cdiv(context_length, CONTEXT_PARTITION_SIZE)
        MAX_WINDOW_BLOCKS: gl.constexpr = (
            SLIDING_WINDOW // CONTEXT_PARTITION_SIZE + 2
        ) * KV_COMPUTE_BLOCK_COUNT
    else:
        kv_sequence_start_idx = sequence_partition_idx * CONTEXT_PARTITION_SIZE
        if kv_sequence_start_idx >= context_length:
            return
        MAX_WINDOW_BLOCKS: gl.constexpr = KV_COMPUTE_BLOCK_COUNT

    # Process KV sequence in compute blocks
    for block_idx in gl.static_range(MAX_WINDOW_BLOCKS):
        if SLIDING_WINDOW > 0:
            partition_in_window = block_idx // KV_COMPUTE_BLOCK_COUNT
            kv_compute_idx = block_idx % KV_COMPUTE_BLOCK_COUNT
            raw_partition = window_start_partition + partition_in_window
            is_valid_partition = raw_partition < window_end_partition
            current_partition = gl.minimum(raw_partition, window_end_partition - 1)
            kv_sequence_start_idx = current_partition * CONTEXT_PARTITION_SIZE
        else:
            kv_compute_idx = block_idx
            current_partition = sequence_partition_idx
            is_valid_partition = True

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
            current_partition * SEQUENCE_PARTITION_KV_BLOCKS
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
            + block_element_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
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
                + value_dim2_offsets[None, None, :, None] * KV_16B_ELEMENT_COUNT
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

        # ==================== ATTENTION MASKING ====================
        # Create boundary mask for valid sequence positions
        # Apply causal masking if required
        if IS_CAUSAL:
            # Compute causal mask based on sequence positions
            sequence_position_extension = (
                QUERY_SEQ_LEN - 1 - qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2
            )
            causal_mask = (
                sequence_position_extension[:, None] + qk_column_offsets[None, :]
                < context_length
            )
            if SLIDING_WINDOW > 0:
                causal_mask = causal_mask & (
                    sequence_position_extension[:, None] + qk_column_offsets[None, :]
                    >= sequence_start_idx
                )
        else:
            causal_mask = qk_column_offsets[None, :] < context_length
            if SLIDING_WINDOW > 0:
                query_token_idx = qk_row_offsets // ONE_QUERY_GROUP_SIZE_POW2
                causal_mask = causal_mask & (
                    qk_column_offsets[None, :]
                    >= sequence_start_idx + query_token_idx[:, None]
                )

        boundary_mask = qk_row_mask[:, None] & causal_mask
        if SLIDING_WINDOW > 0:
            boundary_mask = boundary_mask & is_valid_partition

        attention_scores = gl.convert_layout(attention_scores, layout=qk_linear_layout)
        attention_scores = qk_scale_value * attention_scores

        # Apply masking to attention scores (if [0, CONTEXT_PARTITION_SIZE) are all -inf, the result will be NaN, so we use -3.4e38 other than -inf)
        attention_scores = gl.where(boundary_mask, attention_scores, float(-3.4e38))

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
        if KV_QUANT_MODE >= 0:
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
    exp_sums_safe = tl.where(exp_sums > 0, exp_sums, 1.0)
    exp_sums_reciprocal = 1.0 / exp_sums_safe
    exp_sums_reciprocal_cvt = gl.convert_layout(
        exp_sums_reciprocal[:, None], layout=pv_mfma_layout
    )
    attention_accumulator = attention_accumulator * exp_sums_reciprocal_cvt

    # Store results to global memory
    store_temporary_result(
        max_logits,
        exp_sums,
        attention_accumulator,
        max_logits_ptr,
        exp_sums_ptr,
        output_ptr,
        max_logits_offsets,
        output_offsets,
        qk_row_mask,
        output_mask,
    )


@triton.jit
def paged_attention_decode_ps_reduce_kernel(
    output_ptr,  # [num_seqs, query_seq_len, num_kv_heads, query_group_size, head_size]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_seq_len * query_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_seq_len * query_group_size]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_seq_len * query_group_size, head_size]
    sink_token_ptr,  # [num_query_heads]
    stride_output_bs,
    stride_output_len,
    stride_output_kv_head,
    stride_output_group_size,
    stride_exp_sums_seq,
    stride_exp_sums_head,
    stride_exp_sums_part,
    stride_logits_seq,
    stride_logits_head,
    stride_logits_part,
    stride_logits_group,
    head_size,
    context_partition_num,
    query_group_size,
    HEAD_SIZE_POW2: tl.constexpr,
    USE_SINKS: tl.constexpr,
    MAX_CONTEXT_PARTITION_NUM: tl.constexpr,
):
    """
    Triton port of FlyDSL `compile_pa_decode_sw_reduce`.

    Grid = (num_seqs, num_kv_heads, query_seq_len * query_group_size).
    Each program reduces one flattened `(query_idx, group_idx)` slice across all
    partition slots, then accumulates the corresponding head vector.
    """
    sequence_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)
    eqgs_idx = tl.program_id(2)

    LOG2_E: tl.constexpr = 1.4426950408889634
    partition_offsets = tl.arange(0, MAX_CONTEXT_PARTITION_NUM)
    head_offsets = tl.arange(0, HEAD_SIZE_POW2)
    partition_mask = partition_offsets < context_partition_num
    head_mask = head_offsets < head_size

    query_idx = eqgs_idx // query_group_size
    group_idx = eqgs_idx % query_group_size

    exp_sums_offsets = (
        sequence_idx * stride_exp_sums_seq
        + kv_head_idx * stride_exp_sums_head
        + partition_offsets * stride_exp_sums_part
        + eqgs_idx
    )
    partition_max_logits = tl.load(
        max_logits_ptr + exp_sums_offsets, mask=partition_mask, other=float("-inf")
    )
    partition_exp_sums = tl.load(
        exp_sums_ptr + exp_sums_offsets, mask=partition_mask, other=0.0
    )

    global_max = tl.max(partition_max_logits, axis=0)
    partition_scales = tl.math.exp2((partition_max_logits - global_max) * LOG2_E)
    partition_scales = tl.where(partition_mask, partition_scales, 0.0)
    scaled_exp_sums = partition_exp_sums * partition_scales
    global_exp_sum = tl.sum(scaled_exp_sums, axis=0)

    # Preserve existing sink semantics for multi-partition PS reduce.
    if USE_SINKS:
        sink_token_value = tl.load(
            sink_token_ptr + kv_head_idx * query_group_size + group_idx
        )
        global_exp_sum += tl.math.exp2(
            (sink_token_value.to(tl.float32) - global_max) * LOG2_E
        )

    safe_global_exp_sum = tl.where(global_exp_sum > 0, global_exp_sum, 1.0)
    partition_weights = scaled_exp_sums / safe_global_exp_sum

    logits_offsets = (
        sequence_idx * stride_logits_seq
        + kv_head_idx * stride_logits_head
        + partition_offsets[:, None] * stride_logits_part
        + eqgs_idx * stride_logits_group
        + head_offsets[None, :]
    )
    logits_mask = partition_mask[:, None] & head_mask[None, :]
    partial_logits = tl.load(logits_ptr + logits_offsets, mask=logits_mask, other=0.0)
    final_output = tl.sum(
        partial_logits.to(tl.float32) * partition_weights[:, None], axis=0
    )

    output_offsets = (
        sequence_idx * stride_output_bs
        + query_idx * stride_output_len
        + kv_head_idx * stride_output_kv_head
        + group_idx * stride_output_group_size
        + head_offsets
    )
    tl.store(
        output_ptr + output_offsets,
        final_output.to(output_ptr.dtype.element_ty),
        mask=head_mask,
    )


@triton.jit
def paged_attention_decode_v2_reduce_kernel(
    output_ptr,  # [num_seqs, num_kv_heads, query_group_size, head_size]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size, head_size]
    context_lengths_ptr,  # [num_seqs]
    sink_token_ptr,  # [num_query_heads]
    stride_output_bs,
    stride_output_len,
    stride_output_kv_head,
    stride_output_group_size,
    stride_exp_sums_seq,
    stride_exp_sums_head,
    stride_exp_sums_part,
    stride_logits_seq,
    stride_logits_head,
    stride_logits_part,
    stride_logits_group,
    head_size,
    num_seqs,
    num_kv_heads,
    OUTPUT_SEQ_LEN: tl.constexpr,
    ONE_OUTPUT_GROUP_SIZE: tl.constexpr,
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
    MAX_CONTEXT_PARTITION_NUM: tl.constexpr = 16

    # Calculate output layout parameters
    OUTPUT_SEQ_LEN_POW2: tl.constexpr = triton.next_power_of_2(OUTPUT_SEQ_LEN)
    if ONE_OUTPUT_GROUP_SIZE <= 16 // OUTPUT_SEQ_LEN_POW2:
        ONE_OUTPUT_GROUP_SIZE_POW2: gl.constexpr = 16 // OUTPUT_SEQ_LEN_POW2
    else:
        ONE_OUTPUT_GROUP_SIZE_POW2: gl.constexpr = triton.next_power_of_2(
            ONE_OUTPUT_GROUP_SIZE
        )
    QUERY_GROUP_SIZE_POW2: gl.constexpr = (
        OUTPUT_SEQ_LEN_POW2 * ONE_OUTPUT_GROUP_SIZE_POW2
    )

    # ==================== INITIALIZATION ====================
    sequence_idx = tl.program_id(0)
    kv_head_idx = tl.program_id(1)

    context_length = tl.load(context_lengths_ptr + sequence_idx)
    context_partition_num = tl.cdiv(context_length, CONTEXT_PARTITION_SIZE)

    # Generate coordinate ranges
    output_len_offsets = tl.arange(0, OUTPUT_SEQ_LEN_POW2)
    output_group_offsets = tl.arange(0, ONE_OUTPUT_GROUP_SIZE_POW2)
    query_group_offsets_mtp = tl.arange(0, QUERY_GROUP_SIZE_POW2)
    # Convert MTP layout indices to continuous indices for reading from temporary_output
    query_len_idx = query_group_offsets_mtp // ONE_OUTPUT_GROUP_SIZE_POW2
    group_idx_in_len = query_group_offsets_mtp % ONE_OUTPUT_GROUP_SIZE_POW2
    query_group_offsets = query_len_idx * ONE_OUTPUT_GROUP_SIZE + group_idx_in_len
    head_size_offsets = tl.arange(0, HEAD_SIZE_POW2)

    query_group_mask = (output_len_offsets[:, None] < OUTPUT_SEQ_LEN) & (
        output_group_offsets[None, :] < ONE_OUTPUT_GROUP_SIZE
    )
    query_group_mask = tl.reshape(query_group_mask, [QUERY_GROUP_SIZE_POW2])

    # Initialize global reduction variables
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
        exp_sums_mask = (
            partition_offsets[:, None] < context_partition_num
        ) & query_group_mask[None, :]

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
        sink_token_values = tl.load(
            sink_token_ptr
            + (
                kv_head_idx * OUTPUT_SEQ_LEN * ONE_OUTPUT_GROUP_SIZE
                + query_group_offsets
            ),
            mask=query_group_mask,
        )
        global_exp_sum += tl.exp(sink_token_values - global_max)
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
        exp_sums_mask = (
            partition_offsets[:, None] < context_partition_num
        ) & query_group_mask[None, :]

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

        # Calculate offsets and mask for loading partial logits
        if TRITON_VERSION_GE_3_6_0:
            logits_offsets = (
                sequence_idx * stride_logits_seq
                + kv_head_idx * stride_logits_head
                + partition_offsets[:, None, None] * stride_logits_part
                + query_group_offsets[None, :, None] * stride_logits_group
                + head_size_offsets[None, None, :]
            )
            logits_mask = (
                partition_offsets[:, None] < context_partition_num
            ) & query_group_mask[None, :]
        else:
            logits_offsets = (
                sequence_idx * stride_logits_seq
                + kv_head_idx * stride_logits_head
                + partition_offsets[None, :, None] * stride_logits_part
                + query_group_offsets[:, None, None] * stride_logits_group
                + head_size_offsets[None, None, :]
            )
            logits_mask = (
                partition_offsets[None, :] < context_partition_num
            ) & query_group_mask[:, None]

        # Load partial logits from current chunk of partitions
        partial_logits = tl.load(
            logits_ptr + logits_offsets, mask=logits_mask[:, :, None], other=0.0
        )

        # Permute to match the expected dimension order
        if not TRITON_VERSION_GE_3_6_0:
            partial_logits = tl.permute(partial_logits, (1, 0, 2)).to(tl.float32)

        updated_output = partial_logits * attention_probs

        # Accumulate weighted sum of logits
        final_output += tl.sum(updated_output, axis=0)

    # ==================== FINAL OUTPUT STORING ====================
    # 3D output path
    # Output shape: [batch_size, query_length, num_kv_heads, query_group_size, head_size]
    final_output = tl.reshape(
        final_output, [OUTPUT_SEQ_LEN_POW2, ONE_OUTPUT_GROUP_SIZE_POW2, HEAD_SIZE_POW2]
    )
    # Calculate output tensor offsets
    output_offsets = (
        sequence_idx * stride_output_bs
        + output_len_offsets[:, None, None] * stride_output_len
        + kv_head_idx * stride_output_kv_head
        + output_group_offsets[None, :, None] * stride_output_group_size
        + head_size_offsets[None, None, :]
    )

    # Create mask for valid output storage
    output_mask = (
        (output_len_offsets[:, None, None] < OUTPUT_SEQ_LEN)
        & (output_group_offsets[None, :, None] < ONE_OUTPUT_GROUP_SIZE)
        & (head_size_offsets[None, None, :] < head_size)
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
    query_ptr,  # [num_seqs, query_length, num_kv_heads, query_group_size, head_size]
    key_cache_ptr,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache_ptr,  # [num_blocks, num_kv_heads, head_size, kv_block_size]
    block_tables_ptr,  # [num_seqs, max_num_blocks_per_seq]
    context_lengths_ptr,  # [num_seqs]
    softmax_scale,
    query_scale,  # [num_seqs, query_length, num_kv_heads, query_group_size, 1](per-token) or [1](per-tensor) or None
    key_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    value_scale,  # [num_blocks, num_kv_heads, kv_block_size, 1](per-token) or [1](per-tensor) or None
    stride_max_logits_seq,
    stride_max_logits_head,
    stride_max_logits_part,
    stride_output_seq,
    stride_output_head,
    stride_output_part,
    stride_output_group,
    stride_query_bs,
    stride_query_qlen,
    stride_query_kv_head,
    stride_query_group_size,
    stride_key_block,
    stride_key_head,
    stride_key_head_split,
    stride_key_block_elem,
    stride_value_block,
    stride_value_head_size,
    stride_value_block_elem,
    stride_block_table_seq,
    stride_query_scale_bs,
    stride_query_scale_qlen,
    stride_query_scale_kv_head,
    kv_scale_stride_0,
    kv_scale_stride_1,
    COMPUTE_TYPE,
    query_seq_len,
    query_group_size,
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
    PS,
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
    num_sequences, num_kv_heads, num_splits = grid
    HEAD_SIZE_POW2 = triton.next_power_of_2(HEAD_SIZE)
    QUERY_SEQ_LEN_POW2 = triton.next_power_of_2(query_seq_len)
    ONE_QUERY_GROUP_SIZE_POW2 = triton.next_power_of_2(query_group_size)
    KV_COMPUTE_BLOCK_SIZE = CONTEXT_PARTITION_SIZE
    # Select kernel implementation based on block size

    # PS path uses the sliding-window kernel for all KV_BLOCK_SIZE values.
    # This is required for KV_BLOCK_SIZE==1024 support in PS mode.
    if PS and not (SLIDING_WINDOW > 0 and KV_BLOCK_SIZE == 1024):
        ONE_SHOT = num_splits <= 1
        if num_kv_heads == 1:
            paged_attention_kernel = paged_attention_decode_sliding_window_head_1
            if ONE_QUERY_GROUP_SIZE_POW2 >= 16:
                grid = (num_sequences, query_seq_len * num_kv_heads, num_splits)
                QUERY_SEQ_LEN_POW2 = 1
            else:
                mtp_splits = triton.cdiv(
                    query_seq_len * num_kv_heads,
                    triton.cdiv(16, ONE_QUERY_GROUP_SIZE_POW2),
                )
                grid = (num_sequences, mtp_splits, num_splits)
                QUERY_SEQ_LEN_POW2 = triton.cdiv(QUERY_SEQ_LEN_POW2, mtp_splits)
        else:
            paged_attention_kernel = paged_attention_decode_sliding_window
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
            sinks_ptr if ONE_SHOT else None,
            stride_max_logits_seq,
            stride_max_logits_head,
            stride_max_logits_part,
            # 5D output strides: [batch_size, query_length, num_kv_heads, query_group_size, head_size]
            stride_output_seq,  # stride_output_bs
            stride_output_head,  # stride_output_len
            stride_output_part,  # stride_output_kv_head
            stride_output_group,  # stride_output_group_size
            # 5D query strides
            stride_query_bs,
            stride_query_qlen,
            stride_query_kv_head,
            stride_query_group_size,
            stride_key_block,
            stride_key_head,
            stride_key_head_split,
            stride_key_block_elem,
            stride_value_block,
            stride_value_head_size,
            stride_value_block_elem,
            stride_block_table_seq,
            stride_query_scale_bs,
            stride_query_scale_qlen,
            stride_query_scale_kv_head,
            kv_scale_stride_0,
            kv_scale_stride_1,
            query_seq_len=query_seq_len,
            query_group_size=query_group_size,
            head_size=HEAD_SIZE,
            COMPUTE_TYPE=COMPUTE_TYPE,
            QUERY_SEQ_LEN_POW2=QUERY_SEQ_LEN_POW2,
            ONE_QUERY_GROUP_SIZE_POW2=ONE_QUERY_GROUP_SIZE_POW2,
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
            ONE_SHOT=ONE_SHOT,
        )
        return

    if KV_BLOCK_SIZE > CONTEXT_PARTITION_SIZE:
        # Use big block kernel for large block sizes
        paged_attention_kernel = paged_attention_decode_v2_gluon_large_block_dot_kernel

    else:
        # Use standard kernel for normal block sizes
        paged_attention_kernel = paged_attention_decode_v2_gluon_dot_kernel

    # Launch the dot kernel
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
        stride_query_bs,
        stride_query_qlen,
        stride_query_kv_head,
        stride_query_group_size,
        stride_key_block,
        stride_key_head,
        stride_key_head_split,
        stride_key_block_elem,
        stride_value_block,
        stride_value_head_size,
        stride_value_block_elem,
        stride_block_table_seq,
        stride_query_scale_bs,
        stride_query_scale_qlen,
        stride_query_scale_kv_head,
        kv_scale_stride_0,
        kv_scale_stride_1,
        head_size=HEAD_SIZE,
        num_seqs=grid[0],
        num_kv_heads=grid[1],
        max_context_partition_num=grid[2],
        COMPUTE_TYPE=COMPUTE_TYPE,
        QUERY_SEQ_LEN=query_seq_len,
        ONE_QUERY_GROUP_SIZE=query_group_size,
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
        SLIDING_WINDOW=SLIDING_WINDOW,
    )


def _flydsl_dtype_str(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"Unsupported FlyDSL dtype: {dtype!r}")


@lru_cache(maxsize=256)
def compile_pa_decode_ps_reduce_flydsl(
    *,
    max_context_partition_num: int,
    query_seq_len: int,
    query_group_size: int,
    head_size: int,
    output_dtype_str: str,
    logits_dtype_str: str,
    sink_dtype_str: str,
    use_sinks: bool,
):
    if not FLYDSL_PS_REDUCE_AVAILABLE:
        raise ImportError("FlyDSL is unavailable for pa_decode PS reduce")

    FLYDSL_WARP_SIZE = 64
    FLYDSL_LOG2E = 1.4426950408889634

    block_threads = head_size
    assert block_threads > 0, "head_size must be positive"
    assert block_threads <= 1024, "head_size must fit in one workgroup"
    reduce_width = (
        1
        if max_context_partition_num <= 1
        else 1 << ((max_context_partition_num - 1).bit_length())
    )
    reduce_shuffle_offsets = [off for off in [32, 16, 8, 4, 2, 1] if off < reduce_width]
    red_slots = max(1, (block_threads + FLYDSL_WARP_SIZE - 1) // FLYDSL_WARP_SIZE)
    arch = get_hip_arch()
    allocator = SmemAllocator(None, arch=arch, global_sym_name="pa_ps_sw_reduce_smem")
    red_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = red_off + red_slots * 4
    part_weights_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = part_weights_off + max_context_partition_num * 4

    @flyc.kernel(known_block_size=(block_threads, 1, 1))
    def pa_decode_ps_reduce_flydsl_kernel(
        output_ptr: fx.Tensor,
        exp_sums_ptr: fx.Tensor,
        max_logits_ptr: fx.Tensor,
        logits_ptr: fx.Tensor,
        sink_token_ptr: fx.Tensor,
        stride_output_bs: Int32,
        stride_output_len: Int32,
        stride_output_kv_head: Int32,
        stride_output_group_size: Int32,
        stride_exp_sums_seq: Int32,
        stride_exp_sums_head: Int32,
        stride_exp_sums_part: Int32,
        stride_logits_seq: Int32,
        stride_logits_head: Int32,
        stride_logits_part: Int32,
        stride_logits_group: Int32,
    ):
        tid = gpu.thread_idx.x
        batch_idx = gpu.block_idx.x
        kv_head_idx = gpu.block_idx.y
        eqgs_idx = gpu.block_idx.z

        smem_base = allocator.get_base()
        red_scratch = SmemPtr(smem_base, red_off, T.f32, shape=(red_slots,))
        red_scratch.get()
        if max_context_partition_num > FLYDSL_WARP_SIZE:
            part_weights_lds = SmemPtr(
                smem_base, part_weights_off, T.f32, shape=(max_context_partition_num,)
            )
            part_weights_lds.get()

        out_rsrc = buffer_ops.create_buffer_resource(output_ptr, max_size=True)
        es_rsrc = buffer_ops.create_buffer_resource(exp_sums_ptr, max_size=True)
        ml_rsrc = buffer_ops.create_buffer_resource(max_logits_ptr, max_size=True)
        logits_rsrc = buffer_ops.create_buffer_resource(logits_ptr, max_size=True)
        if use_sinks:
            sink_rsrc = buffer_ops.create_buffer_resource(sink_token_ptr, max_size=True)

        c_zero_f = arith.constant(0.0, type=T.f32)
        c_one_f = arith.constant(1.0, type=T.f32)
        c_neg_inf = arith.constant(float("-inf"), type=T.f32)
        c_log2e = arith.constant(FLYDSL_LOG2E, type=T.f32)
        fm_fast = arith.FastMathFlags.fast
        c_zero_i = arith.constant(0, type=T.i32)
        c_w = arith.constant(FLYDSL_WARP_SIZE, type=T.i32)
        c_wave_mask = arith.constant(FLYDSL_WARP_SIZE - 1, type=T.i32)
        c_wave_shift = arith.constant(6, type=T.i32)
        c_red_slots = arith.constant(red_slots, type=T.i32)
        lane = tid & c_wave_mask
        wave = tid >> c_wave_shift
        c_qgs = arith.constant(query_group_size, type=T.i32)
        group_idx = eqgs_idx % c_qgs

        def _wave_reduce_max_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.maximumf(red.shuffle_xor(arith.constant(sh, type=T.i32), c_w))
            return red

        def _wave_reduce_sum_full(val):
            red = val
            for sh in [32, 16, 8, 4, 2, 1]:
                red = red.addf(
                    red.shuffle_xor(arith.constant(sh, type=T.i32), c_w),
                    fastmath=fm_fast,
                )
            return red

        def _block_reduce(val, mode):
            if red_slots == 1:
                return (
                    _wave_reduce_max_full(val)
                    if mode == "max"
                    else _wave_reduce_sum_full(val)
                )

            neutral = c_neg_inf if mode == "max" else c_zero_f
            w = (
                _wave_reduce_max_full(val)
                if mode == "max"
                else _wave_reduce_sum_full(val)
            )

            if arith.cmpi(arith.CmpIPredicate.eq, lane, c_zero_i):
                wave_idx = arith.index_cast(T.index, wave)
                red_scratch.store(w, [wave_idx])
            gpu.barrier()

            if arith.cmpi(arith.CmpIPredicate.eq, wave, c_zero_i):
                in_range = arith.cmpi(arith.CmpIPredicate.slt, lane, c_red_slots)
                lane_safe = arith.select(in_range, lane, c_zero_i)
                lane_safe_idx = arith.index_cast(T.index, lane_safe)
                red_val = red_scratch.load([lane_safe_idx])
                red_val = arith.select(in_range, red_val, neutral)
                red_val = (
                    _wave_reduce_max_full(red_val)
                    if mode == "max"
                    else _wave_reduce_sum_full(red_val)
                )
                if arith.cmpi(arith.CmpIPredicate.eq, lane, c_zero_i):
                    red_scratch.store(red_val, [arith.constant(0, index=True)])
            gpu.barrier()

            return red_scratch.load([arith.constant(0, index=True)])

        if max_context_partition_num <= FLYDSL_WARP_SIZE:
            c_part_num = arith.constant(max_context_partition_num, type=T.i32)
            c_reduce_width = arith.constant(reduce_width, type=T.i32)
            c_four = arith.constant(4, type=T.i32)

            def _wave_reduce_max(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.maximumf(
                        red.shuffle_xor(arith.constant(sh, type=T.i32), c_w)
                    )
                return red

            def _wave_reduce_sum(val):
                red = val
                for sh in reduce_shuffle_offsets:
                    red = red.addf(
                        red.shuffle_xor(arith.constant(sh, type=T.i32), c_w),
                        fastmath=fm_fast,
                    )
                return red

            lane_in_range = arith.cmpi(arith.CmpIPredicate.slt, lane, c_part_num)
            lane_in_reduce = arith.cmpi(arith.CmpIPredicate.slt, lane, c_reduce_width)
            part_sum = c_zero_f
            part_max = c_neg_inf
            if lane_in_reduce:
                part_i32 = arith.select(lane_in_range, lane, c_zero_i)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = buffer_ops.buffer_load(
                    es_rsrc, es_off, vec_width=1, dtype=T.f32
                )
                part_max_raw = buffer_ops.buffer_load(
                    ml_rsrc, es_off, vec_width=1, dtype=T.f32
                )
                part_sum = arith.select(lane_in_range, part_sum_raw, c_zero_f)
                part_max = arith.select(lane_in_range, part_max_raw, c_neg_inf)

            global_max = _wave_reduce_max(part_max)
            safe_global_max = arith.select(
                global_max > c_neg_inf,
                global_max,
                c_zero_f,
            )
            part_scale = arith.select(
                part_max > c_neg_inf,
                ((part_max - safe_global_max) * c_log2e).exp2(fastmath=fm_fast),
                c_zero_f,
            )
            scaled_sum = part_sum * part_scale
            global_exp_sum = _wave_reduce_sum(scaled_sum)
            if use_sinks:
                sink_off = kv_head_idx * c_qgs + group_idx
                if sink_dtype_str == "f32":
                    sink_value = buffer_ops.buffer_load(
                        sink_rsrc, sink_off, vec_width=1, dtype=T.f32
                    )
                elif sink_dtype_str == "f16":
                    sink_value_raw = buffer_ops.buffer_load(
                        sink_rsrc, sink_off, vec_width=1, dtype=T.f16
                    )
                    sink_value = _mlir_arith.ExtFOp(T.f32, sink_value_raw).result
                else:
                    sink_value_raw = buffer_ops.buffer_load(
                        sink_rsrc, sink_off, vec_width=1, dtype=T.bf16
                    )
                    sink_value = _mlir_arith.ExtFOp(T.f32, sink_value_raw).result
                sink_scale = arith.select(
                    global_max > c_neg_inf,
                    ((sink_value - safe_global_max) * c_log2e).exp2(fastmath=fm_fast),
                    c_zero_f,
                )
                global_exp_sum = global_exp_sum + sink_scale
            safe_global_exp_sum = arith.select(
                global_exp_sum > c_zero_f,
                global_exp_sum,
                c_one_f,
            )
            weight_local = scaled_sum / safe_global_exp_sum
            weight_local_i32 = arith.bitcast(T.i32, weight_local)

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = arith.constant(part_idx, type=T.i32)
                bcast_addr = part_i32 * c_four
                weight_i32 = rocdl.ds_bpermute(
                    T.i32, arith.unwrap(bcast_addr), arith.unwrap(weight_local_i32)
                )
                weight = arith.bitcast(T.f32, weight_i32)
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                if logits_dtype_str == "f32":
                    part_logits = buffer_ops.buffer_load(
                        logits_rsrc, logits_off, vec_width=1, dtype=T.f32
                    )
                elif logits_dtype_str == "f16":
                    part_logits_raw = buffer_ops.buffer_load(
                        logits_rsrc, logits_off, vec_width=1, dtype=T.f16
                    )
                    part_logits = _mlir_arith.ExtFOp(T.f32, part_logits_raw).result
                else:
                    part_logits_raw = buffer_ops.buffer_load(
                        logits_rsrc, logits_off, vec_width=1, dtype=T.bf16
                    )
                    part_logits = _mlir_arith.ExtFOp(T.f32, part_logits_raw).result
                acc = acc + part_logits * weight
        else:
            global_max = c_neg_inf
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = arith.constant(chunk_size, type=T.i32)
                c_chunk_base = arith.constant(chunk_base, type=T.i32)
                in_chunk = arith.cmpi(arith.CmpIPredicate.slt, tid, c_chunk_size)
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, c_zero_i)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_max_raw = buffer_ops.buffer_load(
                    ml_rsrc, es_off, vec_width=1, dtype=T.f32
                )
                part_max = arith.select(in_chunk, part_max_raw, c_neg_inf)
                chunk_max = _block_reduce(part_max, "max")
                global_max = global_max.maximumf(chunk_max)

            safe_global_max = arith.select(
                global_max > c_neg_inf,
                global_max,
                c_zero_f,
            )
            global_exp_sum = c_zero_f
            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = arith.constant(chunk_size, type=T.i32)
                c_chunk_base = arith.constant(chunk_base, type=T.i32)
                in_chunk = arith.cmpi(arith.CmpIPredicate.slt, tid, c_chunk_size)
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, c_zero_i)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = buffer_ops.buffer_load(
                    es_rsrc, es_off, vec_width=1, dtype=T.f32
                )
                part_max_raw = buffer_ops.buffer_load(
                    ml_rsrc, es_off, vec_width=1, dtype=T.f32
                )
                part_sum = arith.select(in_chunk, part_sum_raw, c_zero_f)
                part_max = arith.select(in_chunk, part_max_raw, c_neg_inf)
                part_scale = arith.select(
                    part_max > c_neg_inf,
                    ((part_max - safe_global_max) * c_log2e).exp2(fastmath=fm_fast),
                    c_zero_f,
                )
                chunk_sum = _block_reduce(part_sum * part_scale, "sum")
                global_exp_sum = global_exp_sum + chunk_sum

            if use_sinks:
                sink_off = kv_head_idx * c_qgs + group_idx
                if sink_dtype_str == "f32":
                    sink_value = buffer_ops.buffer_load(
                        sink_rsrc, sink_off, vec_width=1, dtype=T.f32
                    )
                elif sink_dtype_str == "f16":
                    sink_value_raw = buffer_ops.buffer_load(
                        sink_rsrc, sink_off, vec_width=1, dtype=T.f16
                    )
                    sink_value = _mlir_arith.ExtFOp(T.f32, sink_value_raw).result
                else:
                    sink_value_raw = buffer_ops.buffer_load(
                        sink_rsrc, sink_off, vec_width=1, dtype=T.bf16
                    )
                    sink_value = _mlir_arith.ExtFOp(T.f32, sink_value_raw).result
                sink_scale = arith.select(
                    global_max > c_neg_inf,
                    ((sink_value - safe_global_max) * c_log2e).exp2(fastmath=fm_fast),
                    c_zero_f,
                )
                global_exp_sum = global_exp_sum + sink_scale

            safe_global_exp_sum = arith.select(
                global_exp_sum > c_zero_f,
                global_exp_sum,
                c_one_f,
            )

            for chunk_base in range(0, max_context_partition_num, block_threads):
                chunk_size = min(block_threads, max_context_partition_num - chunk_base)
                c_chunk_size = arith.constant(chunk_size, type=T.i32)
                c_chunk_base = arith.constant(chunk_base, type=T.i32)
                in_chunk = arith.cmpi(arith.CmpIPredicate.slt, tid, c_chunk_size)
                part_i32 = arith.select(in_chunk, tid + c_chunk_base, c_zero_i)
                es_off = (
                    batch_idx * stride_exp_sums_seq
                    + kv_head_idx * stride_exp_sums_head
                    + part_i32 * stride_exp_sums_part
                    + eqgs_idx
                )
                part_sum_raw = buffer_ops.buffer_load(
                    es_rsrc, es_off, vec_width=1, dtype=T.f32
                )
                part_max_raw = buffer_ops.buffer_load(
                    ml_rsrc, es_off, vec_width=1, dtype=T.f32
                )
                if in_chunk:
                    part_sum = part_sum_raw
                    part_max = part_max_raw
                    part_scale = arith.select(
                        part_max > c_neg_inf,
                        ((part_max - safe_global_max) * c_log2e).exp2(fastmath=fm_fast),
                        c_zero_f,
                    )
                    weight = (part_sum * part_scale) / safe_global_exp_sum
                    part_idx_idx = arith.index_cast(T.index, part_i32)
                    part_weights_lds.store(weight, [part_idx_idx])

            gpu.barrier()

            acc = c_zero_f
            for part_idx in range_constexpr(max_context_partition_num):
                part_i32 = arith.constant(part_idx, type=T.i32)
                part_idx_idx = arith.constant(part_idx, index=True)
                weight = part_weights_lds.load([part_idx_idx])
                logits_off = (
                    batch_idx * stride_logits_seq
                    + kv_head_idx * stride_logits_head
                    + part_i32 * stride_logits_part
                    + eqgs_idx * stride_logits_group
                    + tid
                )
                if logits_dtype_str == "f32":
                    part_logits = buffer_ops.buffer_load(
                        logits_rsrc, logits_off, vec_width=1, dtype=T.f32
                    )
                elif logits_dtype_str == "f16":
                    part_logits_raw = buffer_ops.buffer_load(
                        logits_rsrc, logits_off, vec_width=1, dtype=T.f16
                    )
                    part_logits = _mlir_arith.ExtFOp(T.f32, part_logits_raw).result
                else:
                    part_logits_raw = buffer_ops.buffer_load(
                        logits_rsrc, logits_off, vec_width=1, dtype=T.bf16
                    )
                    part_logits = _mlir_arith.ExtFOp(T.f32, part_logits_raw).result
                acc = acc + part_logits * weight

        query_idx = eqgs_idx // c_qgs
        group_idx = eqgs_idx % c_qgs
        out_off = (
            batch_idx * stride_output_bs
            + query_idx * stride_output_len
            + kv_head_idx * stride_output_kv_head
            + group_idx * stride_output_group_size
            + tid
        )
        if output_dtype_str == "f32":
            out_val = acc
        elif output_dtype_str == "f16":
            out_val = arith.trunc_f(T.f16, acc)
        else:
            out_val = arith.trunc_f(T.bf16, acc)
        buffer_ops.buffer_store(out_val, out_rsrc, out_off)

    @flyc.jit
    def launch_pa_decode_ps_reduce_flydsl(
        output,
        exp_sums,
        max_logits,
        logits,
        sink_token,
        stride_output_bs,
        stride_output_len,
        stride_output_kv_head,
        stride_output_group_size,
        stride_exp_sums_seq,
        stride_exp_sums_head,
        stride_exp_sums_part,
        stride_logits_seq,
        stride_logits_head,
        stride_logits_part,
        stride_logits_group,
        batch_size,
        num_kv_heads,
        stream: fx.Stream = fx.Stream(None),
    ):
        allocator.finalized = False
        ctx = CompilationContext.get_current()
        with ir.InsertionPoint(ctx.gpu_module_body):
            allocator.finalize()
        pa_decode_ps_reduce_flydsl_kernel(
            output,
            exp_sums,
            max_logits,
            logits,
            sink_token,
            stride_output_bs,
            stride_output_len,
            stride_output_kv_head,
            stride_output_group_size,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
        ).launch(
            grid=(batch_size, num_kv_heads, query_seq_len * query_group_size),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return {
        "launch": launch_pa_decode_ps_reduce_flydsl,
        "kernel": pa_decode_ps_reduce_flydsl_kernel,
        "allocator": allocator,
    }


def launch_pa_decode_ps_reduce_flydsl(
    output_ptr,
    exp_sums_ptr,
    max_logits_ptr,
    logits_ptr,
    sink_token_ptr,
    stride_output_bs,
    stride_output_len,
    stride_output_kv_head,
    stride_output_group_size,
    stride_exp_sums_seq,
    stride_exp_sums_head,
    stride_exp_sums_part,
    stride_logits_seq,
    stride_logits_head,
    stride_logits_part,
    stride_logits_group,
    query_seq_len,
    query_group_size,
    head_size,
    context_partition_num,
):
    if logits_ptr.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ImportError(
            "FlyDSL PS reduce fallback: only bf16/fp16/fp32 logits supported"
        )

    compiled = compile_pa_decode_ps_reduce_flydsl(
        max_context_partition_num=context_partition_num,
        query_seq_len=query_seq_len,
        query_group_size=query_group_size,
        head_size=head_size,
        output_dtype_str=_flydsl_dtype_str(output_ptr.dtype),
        logits_dtype_str=_flydsl_dtype_str(logits_ptr.dtype),
        sink_dtype_str=_flydsl_dtype_str(
            output_ptr.dtype if sink_token_ptr is None else sink_token_ptr.dtype
        ),
        use_sinks=sink_token_ptr is not None,
    )

    if sink_token_ptr is None:
        sink_token_ptr = torch.empty(
            0, dtype=output_ptr.dtype, device=output_ptr.device
        )
    compiled["launch"](
        output_ptr,
        exp_sums_ptr,
        max_logits_ptr,
        logits_ptr,
        sink_token_ptr,
        stride_output_bs,
        stride_output_len,
        stride_output_kv_head,
        stride_output_group_size,
        stride_exp_sums_seq,
        stride_exp_sums_head,
        stride_exp_sums_part,
        stride_logits_seq,
        stride_logits_head,
        stride_logits_part,
        stride_logits_group,
        output_ptr.shape[0],
        output_ptr.shape[2],
        torch.cuda.current_stream(output_ptr.device),
    )


def _paged_attention_decode_v2_reduce_kernel_wrapper(
    grid,
    output_ptr,  # [num_seqs, num_kv_heads, query_group_size, head_size]
    exp_sums_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    max_logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size]
    logits_ptr,  # [num_seqs, num_kv_heads, max_parts, query_group_size, head_size]
    context_lengths_ptr,  # [num_seqs]
    sink_token_ptr,  # [num_query_heads]
    stride_output_bs,
    stride_output_len,
    stride_output_kv_head,
    stride_output_group_size,
    stride_exp_sums_seq,
    stride_exp_sums_head,
    stride_exp_sums_part,
    stride_logits_seq,
    stride_logits_head,
    stride_logits_part,
    stride_logits_group,
    query_seq_len,
    query_group_size,
    head_size,
    CONTEXT_PARTITION_SIZE,
    PS=False,
    context_partition_num=1,
):
    """
    Wrapper function for paged attention reduction kernel with kernel selection.

    This wrapper selects between Gluon and Triton kernel implementations
    based on configuration and launches the appropriate kernel.

    Args:
        All parameters from the reduction kernel plus execution grid configuration
    """
    if PS:
        if CXX_PS_REDUCE_AVAILABLE:
            try:
                launch_pa_decode_ps_reduce_cxx(
                    output_ptr,
                    exp_sums_ptr,
                    max_logits_ptr,
                    logits_ptr,
                    sink_token_ptr,
                    stride_output_bs,
                    stride_output_len,
                    stride_output_kv_head,
                    stride_output_group_size,
                    stride_exp_sums_seq,
                    stride_exp_sums_head,
                    stride_exp_sums_part,
                    stride_logits_seq,
                    stride_logits_head,
                    stride_logits_part,
                    stride_logits_group,
                    query_seq_len=query_seq_len,
                    query_group_size=query_group_size,
                    head_size=head_size,
                    context_partition_num=context_partition_num,
                )
                return
            except ImportError:
                pass
        try:
            launch_pa_decode_ps_reduce_flydsl(
                output_ptr,
                exp_sums_ptr,
                max_logits_ptr,
                logits_ptr,
                sink_token_ptr,
                stride_output_bs,
                stride_output_len,
                stride_output_kv_head,
                stride_output_group_size,
                stride_exp_sums_seq,
                stride_exp_sums_head,
                stride_exp_sums_part,
                stride_logits_seq,
                stride_logits_head,
                stride_logits_part,
                stride_logits_group,
                query_seq_len=query_seq_len,
                query_group_size=query_group_size,
                head_size=head_size,
                context_partition_num=context_partition_num,
            )
            return
        except ImportError:
            ps_reduce_grid = (grid[0], grid[1], query_seq_len * query_group_size)
            paged_attention_decode_ps_reduce_kernel[ps_reduce_grid](
                output_ptr,
                exp_sums_ptr,
                max_logits_ptr,
                logits_ptr,
                sink_token_ptr,
                stride_output_bs,
                stride_output_len,
                stride_output_kv_head,
                stride_output_group_size,
                stride_exp_sums_seq,
                stride_exp_sums_head,
                stride_exp_sums_part,
                stride_logits_seq,
                stride_logits_head,
                stride_logits_part,
                stride_logits_group,
                query_group_size=query_group_size,
                head_size=head_size,
                context_partition_num=context_partition_num,
                HEAD_SIZE_POW2=triton.next_power_of_2(head_size),
                USE_SINKS=sink_token_ptr is not None,
                MAX_CONTEXT_PARTITION_NUM=triton.next_power_of_2(context_partition_num),
            )
    else:
        paged_attention_decode_v2_reduce_kernel[grid](
            output_ptr,
            exp_sums_ptr,
            max_logits_ptr,
            logits_ptr,
            context_lengths_ptr,
            sink_token_ptr,
            stride_output_bs,
            stride_output_len,
            stride_output_kv_head,
            stride_output_group_size,
            stride_exp_sums_seq,
            stride_exp_sums_head,
            stride_exp_sums_part,
            stride_logits_seq,
            stride_logits_head,
            stride_logits_part,
            stride_logits_group,
            head_size=head_size,
            num_seqs=grid[0],
            num_kv_heads=grid[1],
            OUTPUT_SEQ_LEN=query_seq_len,
            ONE_OUTPUT_GROUP_SIZE=query_group_size,
            HEAD_SIZE_POW2=triton.next_power_of_2(head_size),
            CONTEXT_PARTITION_SIZE=CONTEXT_PARTITION_SIZE,
            USE_SINKS=sink_token_ptr is not None,
        )


def pa_decode_gluon(
    output: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    query: torch.Tensor,  # [num_seqs * query_length, num_query_heads, head_size]
    key_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size // x, kv_block_size, x]
    value_cache: torch.Tensor,  # [num_blocks, num_kv_heads, head_size, kv_block_size] or [num_blocks, num_kv_heads, kv_block_size // x, head_size, x]
    context_lengths: torch.Tensor,  # [num_seqs]
    block_tables: torch.Tensor,  # [num_seqs, max_num_blocks_per_seq]
    softmax_scale: float,
    query_length: int,
    max_context_partition_num: int,
    context_partition_size: int = 256,
    compute_type: torch.dtype = torch.bfloat16,
    query_scale: torch.Tensor = None,  # [num_seqs * query_length, num_query_heads, 1] or [1]
    key_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    value_scale: torch.Tensor = None,  # [num_blocks, num_kv_heads, kv_block_size, 1]
    exp_sums: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    max_logits: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
    temporary_output: torch.Tensor = None,  # [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
    alibi_slopes: torch.Tensor = None,
    sinks: torch.Tensor = None,
    sliding_window: int = 0,
    ps: bool = True,
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

    query : torch.Tensor
        Input query tensor in standard layout.
        - Shape: [num_seqs * query_length, num_query_heads, head_size]
        - Dtype: torch.float8_e4m3fnuz (fp8), torch.bfloat16, torch.float16

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

    max_context_partition_num : int
        Caller-provided upper bound for context partitions/splits. In PS mode,
        this is typically from get_recommended_splits(); otherwise it is based
        on the active context length or sliding window.

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
          where max_context_partition_num is the caller-provided partition/split count
        - Dtype: torch.float32

    max_logits : torch.Tensor
        Buffer for maximum logits used in online softmax computation.
        - Shape: [num_seqs, num_kv_heads, max_context_partition_num, query_group_size]
        - Dtype: torch.float32

    temporary_output : torch.Tensor
        Buffer for partial attention outputs from each context partition.
        - Shape: [num_seqs, num_kv_heads, max_context_partition_num, query_group_size, head_size]
        - Dtype: torch.float32 for multi-partition decode; output dtype for one-shot decode

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
    from aiter.ops.triton.utils.types import torch_to_triton_dtype

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
    # Calculate equivalent group sizes for kernel configuration
    equivalent_query_group_size = query_length * query_group_size
    kv_block_size = key_cache.shape[-2]

    # Determine if causal masking is needed
    is_causal = query_length > 1
    # Calculate elements per 16B load based on data type
    kv_elements_per_16b = 16 // key_cache.dtype.itemsize

    # if sliding_window > 0 and kv_block_size != 1024:
    #     max_context_partition_num = 1
    grid = (batch_size, num_kv_heads, max_context_partition_num)

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

    assert (
        len(output.shape) == 3
    ), f"Expected 3D output tensor, but got shape {output.shape}"
    assert (
        len(query.shape) == 3
    ), f"Expected 3D query tensor, but got shape {query.shape}"
    assert (
        len(key_cache.shape) == 5
    ), f"Expected 5D key_cache tensor, but got shape {key_cache.shape}"

    one_shot = max_context_partition_num <= 1

    if exp_sums is None:
        exp_sums = torch.empty(
            batch_size,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
            device=query.device,
            dtype=aiter.dtypes.fp32,
        )
    if max_logits is None:
        max_logits = torch.empty(
            batch_size,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
            device=query.device,
            dtype=aiter.dtypes.fp32,
        )
    if temporary_output is None:
        temporary_output = torch.empty(
            batch_size,
            num_kv_heads,
            max_context_partition_num,
            equivalent_query_group_size,
            head_size,
            device=query.device,
            dtype=aiter.dtypes.fp32,
        )

    # ==================== QUANTIZATION MODE CONFIGURATION ====================
    stride_query_scale_bs = 0
    stride_query_scale_qlen = 0
    stride_query_scale_kv_head = 0
    key_scale_stride_0 = 0
    key_scale_stride_1 = 0
    query_quant_mode = -1
    kv_quant_mode = -1

    # Configure query quantization
    query_scale_5d = None
    if query_scale is not None:
        assert (
            isinstance(query_scale, torch.Tensor)
            and query_scale.dtype == aiter.dtypes.fp32
        ), f"query_scale tensor only support dtype == {aiter.dtypes.fp32}, but got query_scale.dtype == {query_scale.dtype}"

        if query_scale.numel() == 1:
            # Per-tensor quantization
            query_quant_mode = 0
            query_scale_5d = query_scale
        else:
            # Per-token quantization
            assert (
                len(query_scale.shape) == 3
            ), f"Expected 3D query_scale tensor, but got shape {query_scale.shape}"
            assert (
                query_scale.shape[-1] == 1
            ), f"Expected query_scale.shape[-1] == 1, but got query_scale.shape[-1]={query_scale.shape[-1]}"
            query_quant_mode = 1
            # Reshape query_scale to 5D: [num_seqs, query_length, num_kv_heads, query_group_size, 1]
            query_scale_5d = query_scale.reshape(
                batch_size, query_length, num_kv_heads, query_group_size, 1
            )
            stride_query_scale_bs = query_scale_5d.stride(0)
            stride_query_scale_qlen = query_scale_5d.stride(1)
            stride_query_scale_kv_head = query_scale_5d.stride(2)

    # Configure KV quantization
    if (
        key_scale is not None
        and value_scale is not None
        and key_cache.dtype == aiter.dtypes.fp8
    ):
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

    # Reshape query to 5D for direct read access
    query_5d = query.reshape(
        batch_size, query_length, num_kv_heads, query_group_size, head_size
    )
    # Reshape output to 5D for direct write access
    output_5d = output.reshape(
        batch_size, query_length, num_kv_heads, query_group_size, head_size
    )
    # ==================== ATTENTION DECODE KERNEL EXECUTION ====================
    # Determine output tensor and strides based on one_shot mode
    output_for_kernel = output_5d if one_shot else temporary_output
    ps = ps or one_shot
    _paged_attention_decode_v2_with_dot_kernel_reshape_wrapper(
        grid,
        exp_sums,
        max_logits,
        output_for_kernel,
        query_5d,
        key_cache,
        value_cache,
        block_tables,
        context_lengths,
        softmax_scale,
        query_scale_5d,
        key_scale,
        value_scale,
        exp_sums.stride(0),
        exp_sums.stride(1),
        exp_sums.stride(2),
        output_for_kernel.stride(0),
        output_for_kernel.stride(1),
        output_for_kernel.stride(2),
        output_for_kernel.stride(3),
        query_5d.stride(0),
        query_5d.stride(1),
        query_5d.stride(2),
        query_5d.stride(3),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_cache.stride(3),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        stride_query_scale_bs,
        stride_query_scale_qlen,
        stride_query_scale_kv_head,
        key_scale_stride_0,
        key_scale_stride_1,
        COMPUTE_TYPE=torch_to_triton_dtype[compute_type],
        query_seq_len=query_length,
        HEAD_SIZE=head_size,
        query_group_size=query_group_size,
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
        PS=ps,
        CDNA_VERSION=cdna_version,
    )
    # output is already reshaped via output_5d view
    if not one_shot:
        # ==================== REDUCTION KERNEL EXECUTION ====================
        grid = (batch_size, num_kv_heads, 1)
        _paged_attention_decode_v2_reduce_kernel_wrapper(
            grid,
            output_5d,
            exp_sums,
            max_logits,
            temporary_output,
            context_lengths,
            sinks,
            output_5d.stride(0),
            output_5d.stride(1),
            output_5d.stride(2),
            output_5d.stride(3),
            exp_sums.stride(0),
            exp_sums.stride(1),
            exp_sums.stride(2),
            temporary_output.stride(0),
            temporary_output.stride(1),
            temporary_output.stride(2),
            temporary_output.stride(3),
            query_seq_len=query_length,
            query_group_size=query_group_size,
            head_size=head_size,
            CONTEXT_PARTITION_SIZE=context_partition_size,
            PS=ps,
            context_partition_num=max_context_partition_num,
        )
