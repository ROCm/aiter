# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon a8w8 GEMM kernels for gfx1250.

    Y = (X @ W^T) * x_scale[:, None] * w_scale[None, :]  (+ bias)

    X       : (M, K) fp8, row major
    W       : (N, K) fp8, K-contiguous (transposed inside the WMMA)
    x_scale : (M,) fp32  per-token
    w_scale : (N,) fp32  per-channel
    Y       : (M, N) bf16/fp16/fp32

Structure follows the gfx1250 gluon a16w16 kernel: TDM async copies into a
NUM_BUFFERS-deep LDS ring, WMMA over the K tiles.  Because the a8w8 scales are
per-row / per-column they are loop invariant, so unlike the blockscale kernel
they are applied ONCE in the epilogue instead of per K tile.

This module exports the kernels plus the layout helpers; the host-side
launch logic lives in ``aiter.ops.triton.gemm.basic.gemm_a8w8``.
"""

import math

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_GLUON_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "NUM_BUFFERS",
    "GROUP_SIZE_M",
    "NUM_KSPLIT",
    "ADD_BIAS",
]

_gemm_a8w8_bandwidth_bound_repr = make_kernel_repr(
    "_gemm_a8w8_gfx1250_bandwidth_bound_kernel", _GLUON_REPR_KEYS
)

_gemm_a8w8_compute_bound_repr = make_kernel_repr(
    "_gemm_a8w8_gfx1250_compute_bound_kernel", _GLUON_REPR_KEYS
)


_MAX_PAD_INTERVAL_DWORDS = 256


def _pad_interval(extent, elem_bits):
    """Largest encodable pad interval, in elements, not exceeding `extent`."""
    return min(extent, _MAX_PAD_INTERVAL_DWORDS * 32 // elem_bits)


def create_wmma_layouts(num_warps, warps_n=2, instr_k=128, k_width=8):
    """WMMA + dot-operand layouts.

    warps_n: how many warps tile the N dimension; the rest tile M.  The default
    (2) reproduces aiter's create_wmma_layouts.  Skinny-M shapes want all warps
    on N so that a BLOCK_M=16 tile is not replicated across warps.
    """
    warps_n = min(warps_n, num_warps)
    warps_m = num_warps // warps_n
    warp_bases = []
    for i in range(int(math.log2(warps_n))):
        warp_bases.append((0, 1 << i))
    for j in range(int(math.log2(warps_m))):
        warp_bases.append((1 << j, 0))
    warp_bases = tuple(warp_bases)

    wmma_layout = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=warp_bases,
        instr_shape=[16, 16, instr_k],
    )
    operand_a = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=k_width
    )
    operand_b = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=k_width
    )
    return (wmma_layout, operand_a, operand_b)


def create_shared_layouts(BLOCK_M, BLOCK_N, BLOCK_K, elem_bits=8):
    """LDS layouts for the A (M,K) and B (N,K) tiles -- both K-contiguous."""
    shared_a = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BLOCK_K, elem_bits), 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    shared_b = gl.PaddedSharedLayout.with_identity_for(
        [[_pad_interval(BLOCK_K, elem_bits), 8]], [BLOCK_N, BLOCK_K], [1, 0]
    )
    return (shared_a, shared_b)


@gluon.jit
def _pid_to_mn(pid, num_pid_m, num_pid_n, GROUP_SIZE_M: gl.constexpr):
    """Grouped (L2-friendly) program-id ordering; GROUP_SIZE_M == 1 is
    plain column-major (pid_m fastest)."""
    if GROUP_SIZE_M == 1:
        pid_m = pid % num_pid_m
        pid_n = pid // num_pid_m
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m
    return pid_m, pid_n


@gluon.jit
def _epilogue(
    accumulator,
    c_ptr,
    x_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    M,
    N,
    stride_ck,
    stride_cm,
    stride_cn,
    pid_m,
    pid_n,
    pid_k,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    SHARED_LAYOUT_C: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    USE_TDM_STORE: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
):
    offs_m = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
    )
    offs_n = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
    )

    x_scale = gl.load(x_scale_ptr + offs_m, mask=offs_m < M, other=0.0)
    w_scale = gl.load(w_scale_ptr + offs_n, mask=offs_n < N, other=0.0)

    accumulator = accumulator * x_scale[:, None] * w_scale[None, :]

    if ADD_BIAS:
        bias = gl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator = accumulator + bias[None, :]


    if NUM_KSPLIT > 1:
        c_ptr = c_ptr + pid_k * stride_ck

    out = accumulator.to(c_ptr.type.element_ty)

    if USE_TDM_STORE and NUM_KSPLIT == 1:
        c_buffer = gl.allocate_shared_memory(
            c_ptr.type.element_ty,
            shape=[BLOCK_M, BLOCK_N],
            layout=SHARED_LAYOUT_C,
        )
        c_buffer.store(out)
        gl.barrier()
        c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=c_ptr,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            block_shape=(BLOCK_M, BLOCK_N),
            layout=SHARED_LAYOUT_C,
        )
        gl.amd.gfx1250.tdm.async_store(
            c_desc, [pid_m * BLOCK_M, pid_n * BLOCK_N], c_buffer
        )
        gl.amd.gfx1250.tdm.async_wait(0)
    else:
        offs_c = stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
        mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        gl.amd.gfx1250.buffer_store(out, c_ptr, offs_c, mask=mask_c)


# ---------------------------------------------------------------------------
# bandwidth_bound: ds_read immediately before the wmma
# ---------------------------------------------------------------------------
@gluon.jit(repr=_gemm_a8w8_bandwidth_bound_repr)
def _gemm_a8w8_gfx1250_bandwidth_bound_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    x_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    SHARED_LAYOUT_C: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    USE_TDM_STORE: gl.constexpr,
):
    pid_unified = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_KSPLIT == 1:
        pid_k = 0
        pid = pid_unified
    else:
        grid_mn = num_pid_m * num_pid_n
        pid_k = pid_unified // grid_mn
        pid = pid_unified % grid_mn
    pid_m, pid_n = _pid_to_mn(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # K range owned by this split-K partition.
    if NUM_KSPLIT == 1:
        k_start = 0
        K_local = K
    else:
        k_start = pid_k * SPLITK_BLOCK_SIZE
        K_local = min(SPLITK_BLOCK_SIZE, K - k_start)

    a_base = a_ptr + pid_m * BLOCK_M * stride_am + k_start * stride_ak
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn + k_start * stride_bk

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_base,
        shape=(M - pid_m * BLOCK_M, K_local),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_base,
        shape=(N - pid_n * BLOCK_N, K_local),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )

    a_buffer = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
        layout=SHARED_LAYOUT_A,
    )
    b_buffer = gl.allocate_shared_memory(
        b_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
        layout=SHARED_LAYOUT_B,
    )

    load_idx = 0
    compute_idx = 0
    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Fill the pipeline
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )
        load_idx += 1

    num_k_tiles = gl.cdiv(K_local, BLOCK_K)

    for _ in range(num_k_tiles - (NUM_BUFFERS - 1) - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )
        load_idx += 1

        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
            OPERAND_LAYOUT_B,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # ---- Peeled final K tile (bounds-checked, K need not divide BLOCK_K) ----
    k_main = (num_k_tiles - 1) * BLOCK_K
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, set_bounds=[M - pid_m * BLOCK_M, K_local - k_main]
    )
    b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        b_desc, set_bounds=[N - pid_n * BLOCK_N, K_local - k_main]
    )
    gl.amd.gfx1250.tdm.async_load(
        a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_load(
        b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
    load_idx += 1

    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
    )
    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B
    )
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
    compute_idx += 1

    # Epilogue: drain
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
            OPERAND_LAYOUT_B,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    _epilogue(
        accumulator,
        c_ptr,
        x_scale_ptr,
        w_scale_ptr,
        bias_ptr,
        M,
        N,
        stride_ck,
        stride_cm,
        stride_cn,
        pid_m,
        pid_n,
        pid_k,
        BLOCK_M,
        BLOCK_N,
        WMMA_LAYOUT,
        SHARED_LAYOUT_C,
        ADD_BIAS,
        USE_TDM_STORE,
        NUM_KSPLIT,
    )


# ---------------------------------------------------------------------------
# compute_bound: ds_read for tile i+1 issued before the wmma of tile i
# ---------------------------------------------------------------------------
@gluon.jit(repr=_gemm_a8w8_compute_bound_repr)
def _gemm_a8w8_gfx1250_compute_bound_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    x_scale_ptr,
    w_scale_ptr,
    bias_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bn,
    stride_bk,
    stride_ck,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
    SHARED_LAYOUT_C: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    USE_TDM_STORE: gl.constexpr,
):
    gl.static_assert(NUM_BUFFERS >= 2, "compute_bound kernel requires NUM_BUFFERS >= 2")

    pid_unified = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    num_pid_n = gl.cdiv(N, BLOCK_N)
    if NUM_KSPLIT == 1:
        pid_k = 0
        pid = pid_unified
    else:
        grid_mn = num_pid_m * num_pid_n
        pid_k = pid_unified // grid_mn
        pid = pid_unified % grid_mn
    pid_m, pid_n = _pid_to_mn(pid, num_pid_m, num_pid_n, GROUP_SIZE_M)

    # K range owned by this split-K partition.
    if NUM_KSPLIT == 1:
        k_start = 0
        K_local = K
    else:
        k_start = pid_k * SPLITK_BLOCK_SIZE
        K_local = min(SPLITK_BLOCK_SIZE, K - k_start)

    a_base = a_ptr + pid_m * BLOCK_M * stride_am + k_start * stride_ak
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn + k_start * stride_bk

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_base,
        shape=(M - pid_m * BLOCK_M, K_local),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_base,
        shape=(N - pid_n * BLOCK_N, K_local),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )

    a_buffer = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
        layout=SHARED_LAYOUT_A,
    )
    b_buffer = gl.allocate_shared_memory(
        b_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
        layout=SHARED_LAYOUT_B,
    )

    load_idx = 0
    compute_idx = 0
    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )
        load_idx += 1

    num_k_tiles = gl.cdiv(K_local, BLOCK_K)

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
    )
    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]), OPERAND_LAYOUT_B
    )

    # ---- Peeled first iteration ----
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
    gl.amd.gfx1250.tdm.async_load(
        a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_load(
        b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
    )
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, add_offsets=[0, BLOCK_K]
    )
    b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        b_desc, add_offsets=[0, BLOCK_K]
    )
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
    load_idx += 1

    next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index((compute_idx + 1) % NUM_BUFFERS), OPERAND_LAYOUT_A
    )
    next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
        OPERAND_LAYOUT_B,
    )
    cur_a = next_a
    cur_b = next_b
    compute_idx += 1

    # ---- Remaining main-loop iterations ----
    for _ in range(num_k_tiles - NUM_BUFFERS - 1):
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        load_idx += 1

        next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index((compute_idx + 1) % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
            OPERAND_LAYOUT_B,
        )
        cur_a = next_a
        cur_b = next_b
        compute_idx += 1

    # ---- Peeled final K tile (bounds-checked) ----
    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

    k_main = (num_k_tiles - 1) * BLOCK_K
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, set_bounds=[M - pid_m * BLOCK_M, K_local - k_main]
    )
    b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        b_desc, set_bounds=[N - pid_n * BLOCK_N, K_local - k_main]
    )
    gl.amd.gfx1250.tdm.async_load(
        a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_load(
        b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
    load_idx += 1

    next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index((compute_idx + 1) % NUM_BUFFERS), OPERAND_LAYOUT_A
    )
    next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
        OPERAND_LAYOUT_B,
    )
    cur_a = next_a
    cur_b = next_b
    compute_idx += 1

    for i in gl.static_range(NUM_BUFFERS - 2):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)
        next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index((compute_idx + 1) % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
            OPERAND_LAYOUT_B,
        )
        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        cur_a = next_a
        cur_b = next_b
        compute_idx += 1

    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

    _epilogue(
        accumulator,
        c_ptr,
        x_scale_ptr,
        w_scale_ptr,
        bias_ptr,
        M,
        N,
        stride_ck,
        stride_cm,
        stride_cn,
        pid_m,
        pid_n,
        pid_k,
        BLOCK_M,
        BLOCK_N,
        WMMA_LAYOUT,
        SHARED_LAYOUT_C,
        ADD_BIAS,
        USE_TDM_STORE,
        NUM_KSPLIT,
    )


_KERNEL_MAP = {
    "bandwidth_bound": _gemm_a8w8_gfx1250_bandwidth_bound_kernel,
    "compute_bound": _gemm_a8w8_gfx1250_compute_bound_kernel,
}

# Minimum pipeline depth each variant needs, and how much reach beyond the
# main loop it consumes (prologue pre-load + peeled final tile).
# NUM_BUFFERS == 1 is NOT safe: the next TDM async_load targets the same LDS
# buffer the current ds_read is still draining (WAR hazard), which silently
# corrupts results.  Both variants therefore need at least double buffering.
_MIN_BUFFERS = {"bandwidth_bound": 2, "compute_bound": 2}
_DEPTH_SLACK = {"bandwidth_bound": 0, "compute_bound": 2}

# Fallback used when no tuned config file matches (mirrors DEFAULT.json).
DEFAULT_CONFIG = {
    "BLOCK_M": 64,
    "BLOCK_N": 64,
    "BLOCK_K": 256,
    "NUM_BUFFERS": 3,
    "GROUP_SIZE_M": 1,
    "num_warps": 4,
    "WARPS_N": 2,
    "INSTR_K": 128,
    "kernel_type": "compute_bound",
    "USE_TDM_STORE": True,
    "NUM_KSPLIT": 1,
}
