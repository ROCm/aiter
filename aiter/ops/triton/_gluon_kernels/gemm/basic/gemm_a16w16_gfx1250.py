# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import math
from typing import Optional, Dict
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton._triton_kernels.activation import _get_activation_from_str
from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config
from aiter.ops.triton._gluon_kernels.utils.prefetch import gemm_l2_prefetch, gemm_l2_prefetch_prologue
from aiter.ops.triton.utils.logger import AiterTritonLogger
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_LOGGER = AiterTritonLogger()

_GLUON_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "NUM_BUFFERS",
    "PHYSICAL_MK",
    "PHYSICAL_KN",
    "USE_ACTIVATION",
    "ADD_BIAS",
    "L2_PREFETCH_DISTANCE",
]

_gemm_a16w16_basic_repr = make_kernel_repr(
    "_gemm_a16w16_gfx1250_basic_kernel", _GLUON_REPR_KEYS
)

_gemm_a16w16_warp_priority_repr = make_kernel_repr(
    "_gemm_a16w16_gfx1250_warp_priority_kernel", _GLUON_REPR_KEYS
)

_gemm_a16w16_k_subtiling_repr = make_kernel_repr(
    "_gemm_a16w16_gfx1250_k_subtiling_kernel", _GLUON_REPR_KEYS
)


def _get_config(M: int, N: int, K: int):
    config, is_tuned = get_gemm_config("GEMM-A16W16", M, N, K)
    return config, is_tuned


def create_shared_layouts(BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr,
                          BLOCK_K: gl.constexpr, PHYSICAL_MK: gl.constexpr,
                          PHYSICAL_KN: gl.constexpr):
    if PHYSICAL_MK:
        SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
        )
    else:
        SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_M, 8]], [BLOCK_K, BLOCK_M], [1, 0]
        )

    if PHYSICAL_KN:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
        )
    else:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
        )

    return (SHARED_LAYOUT_A, SHARED_LAYOUT_B)


@gluon.jit(repr=_gemm_a16w16_basic_repr)
def _gemm_a16w16_basic_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    L2_PREFETCH_DISTANCE: gl.constexpr,
):
    USE_L2_PREFETCH: gl.constexpr = L2_PREFETCH_DISTANCE > 0

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(K, M),
            strides=(stride_ak, stride_am),
            block_shape=(BLOCK_K, BLOCK_M),
            layout=SHARED_LAYOUT_A
        )

    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B
        )

    if PHYSICAL_MK:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A
        )
    else:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A
        )

    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B
        )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    if USE_L2_PREFETCH:
        gemm_l2_prefetch_prologue(L2_PREFETCH_DISTANCE, load_idx, a_desc, b_desc, off_am, off_bn,
                                      BLOCK_K, NUM_BUFFERS, not PHYSICAL_MK, not PHYSICAL_KN)

    # Fill the pipeline
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [load_idx * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )

        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [load_idx * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, load_idx * BLOCK_K],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )

        load_idx += 1

    # Main pipeline loop
    num_k_tiles = gl.cdiv(K, BLOCK_K)

    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [load_idx * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )

        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [load_idx * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, load_idx * BLOCK_K],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )

        load_idx += 1

        if USE_L2_PREFETCH:
            gemm_l2_prefetch(L2_PREFETCH_DISTANCE - 1, load_idx, a_desc, b_desc,
                                  off_am, off_bn, BLOCK_K, not PHYSICAL_MK, not PHYSICAL_KN)

        if PHYSICAL_MK:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A)
        else:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_A)

        if PHYSICAL_KN:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B)
        else:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B)

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

        compute_idx += 1

    # Epilogue: no more loads
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

        if PHYSICAL_MK:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A)
        else:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_A)

        if PHYSICAL_KN:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B)
        else:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B)

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # Bias
    if ADD_BIAS:
        offs_bias = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
        bias_vals = gl.load(bias_ptr + offs_bias)
        accumulator = accumulator + bias_vals[None, :]

    # Activation
    if USE_ACTIVATION:
        accumulator = activation(accumulator)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))

    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Store
    gl.amd.gfx1250.buffer_store(
        accumulator.to(c_ptr.type.element_ty),
        c_ptr,
        offs_c,
        mask=mask_c
    )


@gluon.jit(repr=_gemm_a16w16_warp_priority_repr)
def _gemm_a16w16_warp_priority_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    L2_PREFETCH_DISTANCE: gl.constexpr,
):
    USE_L2_PREFETCH: gl.constexpr = L2_PREFETCH_DISTANCE > 0
    gl.static_assert(NUM_BUFFERS >= 3, "Warp priority kernel requires NUM_BUFFERS >= 3")

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(K, M),
            strides=(stride_ak, stride_am),
            block_shape=(BLOCK_K, BLOCK_M),
            layout=SHARED_LAYOUT_A
        )

    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B
        )

    if PHYSICAL_MK:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A
        )
    else:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A
        )

    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B
        )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    if USE_L2_PREFETCH:
        gemm_l2_prefetch_prologue(L2_PREFETCH_DISTANCE, load_idx, a_desc, b_desc, off_am, off_bn,
                                      BLOCK_K, NUM_BUFFERS, not PHYSICAL_MK, not PHYSICAL_KN)

    # Fill the pipeline
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [load_idx * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(load_idx % NUM_BUFFERS)
            )

        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [load_idx * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, load_idx * BLOCK_K],
                b_buffer.index(load_idx % NUM_BUFFERS)
            )

        load_idx += 1

    # Main pipeline loop with warp pipelining
    num_k_tiles = gl.cdiv(K, BLOCK_K)

    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

        with gl.amd.warp_pipeline_stage("stage0", priority=1):
            if PHYSICAL_MK:
                cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A)
            else:
                cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_A)

            if PHYSICAL_KN:
                cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B)
            else:
                cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B)

            if PHYSICAL_MK:
                gl.amd.gfx1250.tdm.async_load(
                    a_desc,
                    [pid_m * BLOCK_M, load_idx * BLOCK_K],
                    a_buffer.index(load_idx % NUM_BUFFERS)
                )
            else:
                gl.amd.gfx1250.tdm.async_load(
                    a_desc,
                    [load_idx * BLOCK_K, pid_m * BLOCK_M],
                    a_buffer.index(load_idx % NUM_BUFFERS)
                )

            if PHYSICAL_KN:
                gl.amd.gfx1250.tdm.async_load(
                    b_desc,
                    [load_idx * BLOCK_K, pid_n * BLOCK_N],
                    b_buffer.index(load_idx % NUM_BUFFERS)
                )
            else:
                gl.amd.gfx1250.tdm.async_load(
                    b_desc,
                    [pid_n * BLOCK_N, load_idx * BLOCK_K],
                    b_buffer.index(load_idx % NUM_BUFFERS)
                )

            load_idx += 1

            if USE_L2_PREFETCH:
                gemm_l2_prefetch(L2_PREFETCH_DISTANCE - 1, load_idx, a_desc, b_desc,
                                      off_am, off_bn, BLOCK_K, not PHYSICAL_MK, not PHYSICAL_KN)

        with gl.amd.warp_pipeline_stage("stage1", priority=0):
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            compute_idx += 1

    # Epilogue: drain remaining tiles
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

        if PHYSICAL_MK:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A)
        else:
            cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_A)

        if PHYSICAL_KN:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B)
        else:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B)

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # Bias
    if ADD_BIAS:
        offs_bias = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
        bias_vals = gl.load(bias_ptr + offs_bias)
        accumulator = accumulator + bias_vals[None, :]

    # Activation
    if USE_ACTIVATION:
        accumulator = activation(accumulator)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))

    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Store
    gl.amd.gfx1250.buffer_store(
        accumulator.to(c_ptr.type.element_ty),
        c_ptr,
        offs_c,
        mask=mask_c
    )


@gluon.jit(repr=_gemm_a16w16_k_subtiling_repr)
def _gemm_a16w16_k_subtiling_kernel(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    PHYSICAL_MK: gl.constexpr,
    PHYSICAL_KN: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    L2_PREFETCH_DISTANCE: gl.constexpr,
):
    USE_L2_PREFETCH: gl.constexpr = L2_PREFETCH_DISTANCE > 0
    SUBTILE_LEN: gl.constexpr = 32
    NUM_SUBTILES: gl.constexpr = BLOCK_K // SUBTILE_LEN
    gl.static_assert(NUM_SUBTILES >= 2, "BLOCK_K must be >= 64 for k-subtiling")

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    if PHYSICAL_MK:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(M, K),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_LAYOUT_A
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr,
            shape=(K, M),
            strides=(stride_ak, stride_am),
            block_shape=(BLOCK_K, BLOCK_M),
            layout=SHARED_LAYOUT_A
        )

    if PHYSICAL_KN:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B
        )

    if PHYSICAL_MK:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A
        )
    else:
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_M],
            layout=SHARED_LAYOUT_A
        )

    if PHYSICAL_KN:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B
        )

    producer = 0
    consumer = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    off_am = pid_m * BLOCK_M
    off_bn = pid_n * BLOCK_N

    if USE_L2_PREFETCH:
        gemm_l2_prefetch_prologue(L2_PREFETCH_DISTANCE, producer, a_desc, b_desc, off_am, off_bn,
                                      BLOCK_K, NUM_BUFFERS, not PHYSICAL_MK, not PHYSICAL_KN)

    # Prologue: fill pipeline with NUM_BUFFERS - 1 tiles
    for _ in gl.static_range(NUM_BUFFERS - 1):
        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, producer * BLOCK_K],
                a_buffer.index(producer % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [producer * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(producer % NUM_BUFFERS)
            )

        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [producer * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(producer % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, producer * BLOCK_K],
                b_buffer.index(producer % NUM_BUFFERS)
            )

        producer += 1

    num_k_tiles = gl.cdiv(K, BLOCK_K)

    # Main loop: issue TDM load for next tile, then process current tile via subtile loop
    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

        if PHYSICAL_MK:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [pid_m * BLOCK_M, producer * BLOCK_K],
                a_buffer.index(producer % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [producer * BLOCK_K, pid_m * BLOCK_M],
                a_buffer.index(producer % NUM_BUFFERS)
            )

        if PHYSICAL_KN:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [producer * BLOCK_K, pid_n * BLOCK_N],
                b_buffer.index(producer % NUM_BUFFERS)
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [pid_n * BLOCK_N, producer * BLOCK_K],
                b_buffer.index(producer % NUM_BUFFERS)
            )

        producer += 1

        if USE_L2_PREFETCH:
            gemm_l2_prefetch(L2_PREFETCH_DISTANCE - 1, producer, a_desc, b_desc,
                                  off_am, off_bn, BLOCK_K, not PHYSICAL_MK, not PHYSICAL_KN)

        # Subtile loop: load subtile s+1 while computing subtile s
        idx = consumer % NUM_BUFFERS
        if PHYSICAL_MK:
            cur_a = a_buffer.index(idx).slice(0, SUBTILE_LEN, 1).load(layout=OPERAND_LAYOUT_A)
        else:
            cur_a = a_buffer.index(idx).slice(0, SUBTILE_LEN, 0).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
        if PHYSICAL_KN:
            cur_b = b_buffer.index(idx).slice(0, SUBTILE_LEN, 0).load(layout=OPERAND_LAYOUT_B)
        else:
            cur_b = b_buffer.index(idx).slice(0, SUBTILE_LEN, 1).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)

        for s in gl.static_range(1, NUM_SUBTILES):
            if PHYSICAL_MK:
                next_a = a_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 1).load(layout=OPERAND_LAYOUT_A)
            else:
                next_a = a_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 0).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
            if PHYSICAL_KN:
                next_b = b_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 0).load(layout=OPERAND_LAYOUT_B)
            else:
                next_b = b_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 1).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            cur_a = next_a
            cur_b = next_b
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

        consumer += 1

    # Epilogue: drain remaining pipeline stages (no new TDM loads)
    for i in gl.static_range(NUM_BUFFERS - 2):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)

        idx = consumer % NUM_BUFFERS
        if PHYSICAL_MK:
            cur_a = a_buffer.index(idx).slice(0, SUBTILE_LEN, 1).load(layout=OPERAND_LAYOUT_A)
        else:
            cur_a = a_buffer.index(idx).slice(0, SUBTILE_LEN, 0).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
        if PHYSICAL_KN:
            cur_b = b_buffer.index(idx).slice(0, SUBTILE_LEN, 0).load(layout=OPERAND_LAYOUT_B)
        else:
            cur_b = b_buffer.index(idx).slice(0, SUBTILE_LEN, 1).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)

        for s in gl.static_range(1, NUM_SUBTILES):
            if PHYSICAL_MK:
                next_a = a_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 1).load(layout=OPERAND_LAYOUT_A)
            else:
                next_a = a_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 0).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
            if PHYSICAL_KN:
                next_b = b_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 0).load(layout=OPERAND_LAYOUT_B)
            else:
                next_b = b_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 1).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
            cur_a = next_a
            cur_b = next_b
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

        consumer += 1

    # Final tile
    gl.amd.gfx1250.tdm.async_wait(0)

    idx = consumer % NUM_BUFFERS
    if PHYSICAL_MK:
        cur_a = a_buffer.index(idx).slice(0, SUBTILE_LEN, 1).load(layout=OPERAND_LAYOUT_A)
    else:
        cur_a = a_buffer.index(idx).slice(0, SUBTILE_LEN, 0).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
    if PHYSICAL_KN:
        cur_b = b_buffer.index(idx).slice(0, SUBTILE_LEN, 0).load(layout=OPERAND_LAYOUT_B)
    else:
        cur_b = b_buffer.index(idx).slice(0, SUBTILE_LEN, 1).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)

    for s in gl.static_range(1, NUM_SUBTILES):
        if PHYSICAL_MK:
            next_a = a_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 1).load(layout=OPERAND_LAYOUT_A)
        else:
            next_a = a_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 0).permute([1, 0]).load(layout=OPERAND_LAYOUT_A)
        if PHYSICAL_KN:
            next_b = b_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 0).load(layout=OPERAND_LAYOUT_B)
        else:
            next_b = b_buffer.index(idx).slice(s * SUBTILE_LEN, SUBTILE_LEN, 1).permute([1, 0]).load(layout=OPERAND_LAYOUT_B)
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        cur_a = next_a
        cur_b = next_b
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

    # Bias
    if ADD_BIAS:
        offs_bias = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))
        bias_vals = gl.load(bias_ptr + offs_bias)
        accumulator = accumulator + bias_vals[None, :]

    # Activation
    if USE_ACTIVATION:
        accumulator = activation(accumulator)

    offs_cm = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT))
    offs_cn = pid_n * BLOCK_N + gl.arange(0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT))

    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]

    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # Store
    gl.amd.gfx1250.buffer_store(
        accumulator.to(c_ptr.type.element_ty),
        c_ptr,
        offs_c,
        mask=mask_c
    )

_KERNEL_MAP = {
    "basic": _gemm_a16w16_basic_kernel,
    "warp_priority": _gemm_a16w16_warp_priority_kernel,
    "k_subtiling": _gemm_a16w16_k_subtiling_kernel,
}


def gemm_a16w16_gfx1250(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[Dict] = None,
    activation: Optional[str] = None,
    kernel_type: str = "basic",
):
    """
    Compute 16 bit gemm y = x @ w^T + bias using gluon (gfx1250).

    Args:
        x: Input tensor of shape (M, K)
        w: Weight tensor of shape (N, K), internally transposed
        bias: Optional bias tensor of shape (N,)
        dtype: Output data type
        y: Optional pre-allocated output tensor
        config: Kernel tuning parameters:
            - BLOCK_M: Tile size in M dimension (default: 128)
            - BLOCK_N: Tile size in N dimension (default: 128)
            - BLOCK_K: Tile size in K dimension (default: 32)
            - NUM_BUFFERS: Pipeline stages (default: 2)
            - num_warps: Warps per block (default: 8)
        activation: Activation function ("gelu", "gelu_tanh", "silu", "silu_exp2", "relu")
        kernel_type: Kernel variant to use:
            - "basic": Simple pipelining with async TDM loads (default)
            - "warp_priority": Warp priority pipelining (requires NUM_BUFFERS >= 3)
            - "k_subtiling": K-dimension subtiling for LDS latency hiding (requires BLOCK_K >= 64)

    Returns:
        Output tensor of shape (M, N)
    """

    assert kernel_type in _KERNEL_MAP, \
        f"Unknown kernel_type '{kernel_type}', must be one of {list(_KERNEL_MAP.keys())}"

    _LOGGER.info(f"GEMM_A16W16 [gluon/gfx1250]: x={tuple(x.shape)} w={tuple(w.shape)} kernel={kernel_type}")

    assert x.dtype in (torch.float16, torch.bfloat16), \
        f"Activations (x) must be fp16 or bf16, got {x.dtype}"
    assert w.dtype in (torch.float16, torch.bfloat16), f"Weights (w) must be fp16 or bf16, got {w.dtype}"
    assert x.shape[1] == w.shape[1], "Incompatible matrix shapes."

    M, K = x.shape
    N, _ = w.shape

    if config is None:
        config, _ = _get_config(M, N, K)

    BLOCK_M = config["BLOCK_M"]
    BLOCK_N = config["BLOCK_N"]
    BLOCK_K = config["BLOCK_K"]
    NUM_BUFFERS = config.get("NUM_BUFFERS", 2)
    num_warps = config["num_warps"]
    L2_PREFETCH_DISTANCE = config.get("L2_PREFETCH_DISTANCE", 0)

    # Pad K to be divisible by block k so tdm loads never read out of bounds
    K_padded = triton.cdiv(K, BLOCK_K) * BLOCK_K
    if K_padded != K:
        pad_size = K_padded - K
        x = torch.nn.functional.pad(x, (0, pad_size))
        w = torch.nn.functional.pad(w, (0, pad_size))
        K = K_padded

    w = w.T

    if x.stride(1) == 1:
        physical_mk = True
    elif x.stride(0) == 1:
        physical_mk = False
    else:
        raise ValueError(f"x must be contiguous in at least one dimension, got strides {x.stride()}")

    if w.stride(1) == 1:
        physical_kn = True
    elif w.stride(0) == 1:
        physical_kn = False
    else:
        raise ValueError(f"w must be contiguous in at least one dimension, got strides {w.stride()}")

    if y is None:
        y = torch.empty((M, N), device=x.device, dtype=dtype)

    warp_bases = [(0, 1)]
    for i in range(int(math.log2(num_warps // 2))):
        warp_bases.append((1 << i, 0))
    warp_bases = tuple(warp_bases)

    wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        instr_shape=[16, 16, 32]
    )

    operand_a = gl.DotOperandLayout(operand_index=0, parent=wmma_layout, k_width=8)
    operand_b = gl.DotOperandLayout(operand_index=1, parent=wmma_layout, k_width=8)

    shared_layouts = create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, physical_mk, physical_kn)
    shared_a, shared_b = shared_layouts[0], shared_layouts[1]

    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    grid = (num_tiles_m * num_tiles_n, 1)

    kernel_fn = _KERNEL_MAP[kernel_type]

    kernel_fn[grid](
        x, w, y, bias,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        NUM_BUFFERS=NUM_BUFFERS,
        PHYSICAL_MK=physical_mk,
        PHYSICAL_KN=physical_kn,
        SHARED_LAYOUT_A=shared_a,
        SHARED_LAYOUT_B=shared_b,
        WMMA_LAYOUT=wmma_layout,
        OPERAND_LAYOUT_A=operand_a,
        OPERAND_LAYOUT_B=operand_b,
        activation=_get_activation_from_str(activation) if activation else None,
        USE_ACTIVATION=activation is not None,
        ADD_BIAS=(bias is not None),
        L2_PREFETCH_DISTANCE=L2_PREFETCH_DISTANCE,
        num_warps=num_warps,
    )

    return y
