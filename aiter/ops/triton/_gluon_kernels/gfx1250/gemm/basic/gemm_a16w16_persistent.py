# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Gluon persistent Triton GEMM.

"""

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd

_GLUON_REPR_KEYS = [
    "BLOCK_M",
    "BLOCK_N",
    "BLOCK_K",
    "GROUP_SIZE_M",
    "NUM_BUFFERS",
    "NUM_KSPLIT",
    "SPLITK_BLOCK_SIZE",
    "TRANSPOSE",
    "USE_ACTIVATION",
    "ADD_BIAS",
    "SKIP_REDUCE",
    "NUM_SMS",
    "num_warps",
    "num_stages",
    "waves_per_eu",
]

_gemm_a16w16_persistent_repr = make_kernel_repr(
    "gemm_a16w16_persistent_gfx1250_kernel_", _GLUON_REPR_KEYS
)

_gemm_a16w16_persistent_compute_bound_repr = make_kernel_repr(
    "gemm_a16w16_persistent_compute_bound_gfx1250_kernel_", _GLUON_REPR_KEYS
)


@gluon.jit(repr=_gemm_a16w16_persistent_repr)
def gemm_a16w16_persistent_kernel_(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    num_tiles,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    WARP_BASES: gl.constexpr,
    TRANSPOSE: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    SKIP_REDUCE: gl.constexpr,
    NUM_SMS: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr = 0,
    waves_per_eu: gl.constexpr = 0,
):

    WRITES_FINAL: gl.constexpr = NUM_KSPLIT == 1

    SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    if TRANSPOSE:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
        )
    else:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
        )

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES,
        instr_shape=[16, 16, 32],
    )
    OPERAND_LAYOUT_A: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    OPERAND_LAYOUT_B: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )

    start_pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    a_buffer = gl.allocate_shared_memory(
        a_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
        layout=SHARED_LAYOUT_A,
    )

    if TRANSPOSE:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_buffer = gl.allocate_shared_memory(
            b_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
            layout=SHARED_LAYOUT_B,
        )

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )

    if TRANSPOSE:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B,
        )

    # Persistent loop
    for tile_id in range(start_pid, num_tiles, NUM_SMS):

        SHARED_LAYOUT_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
                [[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0]
            )
        c_buffer = gl.allocate_shared_memory(
                c_ptr.type.element_ty,
                shape=[BLOCK_M, BLOCK_N],
                layout=SHARED_LAYOUT_C,
            )
        # remap tile index
        t = remap_xcd(tile_id, num_tiles, NUM_XCDS=8)
        pid_k = t % NUM_KSPLIT
        pid = t // NUM_KSPLIT

        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

        m_off = pid_m * BLOCK_M
        n_off = pid_n * BLOCK_N

        split_k_start = pid_k * SPLITK_BLOCK_SIZE
        if split_k_start < K:
            split_k_end = gl.minimum(split_k_start + SPLITK_BLOCK_SIZE, K)
            k_span = split_k_end - split_k_start
            num_k_tiles = gl.cdiv(k_span, BLOCK_K)

            load_idx = 0
            compute_idx = 0

            accumulator = gl.zeros(
                (BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT
            )

            if ADD_BIAS and pid_k == 0:
                offs_bias = n_off + gl.arange(
                    0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
                )
                bias_vals = gl.load(bias_ptr + offs_bias, mask=offs_bias < N, other=0.0)
                accumulator = accumulator + bias_vals[None, :]

            # prologue: fill buffers with tiles
            for _ in gl.static_range(NUM_BUFFERS - 1):
                gl.amd.gfx1250.tdm.async_load(
                    a_desc,
                    [m_off, split_k_start + load_idx * BLOCK_K],
                    a_buffer.index(load_idx % NUM_BUFFERS),
                )
                if TRANSPOSE:
                    gl.amd.gfx1250.tdm.async_load(
                        b_desc,
                        [split_k_start + load_idx * BLOCK_K, n_off],
                        b_buffer.index(load_idx % NUM_BUFFERS),
                    )
                else:
                    gl.amd.gfx1250.tdm.async_load(
                        b_desc,
                        [n_off, split_k_start + load_idx * BLOCK_K],
                        b_buffer.index(load_idx % NUM_BUFFERS),
                    )
                load_idx += 1

            # main loop: produce and consume k tiles
            for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
                gl.amd.gfx1250.tdm.async_load(
                    a_desc,
                    [m_off, split_k_start + load_idx * BLOCK_K],
                    a_buffer.index(load_idx % NUM_BUFFERS),
                )
                if TRANSPOSE:
                    gl.amd.gfx1250.tdm.async_load(
                        b_desc,
                        [split_k_start + load_idx * BLOCK_K, n_off],
                        b_buffer.index(load_idx % NUM_BUFFERS),
                    )
                else:
                    gl.amd.gfx1250.tdm.async_load(
                        b_desc,
                        [n_off, split_k_start + load_idx * BLOCK_K],
                        b_buffer.index(load_idx % NUM_BUFFERS),
                    )
                # leave the NUM_BUFFERS-1 most recent pairs in flight
                gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

                load_idx += 1

                cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
                )
                if TRANSPOSE:
                    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                        b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
                    )
                else:
                    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                        b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                        OPERAND_LAYOUT_B,
                    )
                accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
                compute_idx += 1

            # epilogue: drain remaining loads
            for i in gl.static_range(NUM_BUFFERS - 1):
                gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

                cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
                )
                if TRANSPOSE:
                    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                        b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
                    )
                else:
                    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                        b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                        OPERAND_LAYOUT_B,
                    )
                accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
                compute_idx += 1

            if USE_ACTIVATION and WRITES_FINAL:
                accumulator = activation(accumulator)

            c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
                base=c_ptr + pid_k.to(gl.int64) * stride_ck,
                shape=(M, N),
                strides=(stride_cm, stride_cn),
                block_shape=(BLOCK_M, BLOCK_N),
                layout=SHARED_LAYOUT_C,
            )
            c_buffer.store(accumulator.to(c_ptr.type.element_ty))
            gl.barrier()
            gl.amd.gfx1250.tdm.async_store(c_desc, [m_off, n_off], c_buffer)
            gl.amd.gfx1250.tdm.async_wait(0)

        gl.barrier()


@gluon.jit(repr=_gemm_a16w16_persistent_compute_bound_repr)
def gemm_a16w16_persistent_compute_bound_kernel_(
    a_ptr,
    b_ptr,
    bias_ptr,
    c_ptr,
    M,
    N,
    K,
    num_tiles,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_ck,
    stride_cm,
    stride_cn,
    # Meta-parameters
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    WARP_BASES: gl.constexpr,
    TRANSPOSE: gl.constexpr,
    activation: gl.constexpr,
    USE_ACTIVATION: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    SKIP_REDUCE: gl.constexpr,
    NUM_SMS: gl.constexpr,
    num_warps: gl.constexpr,
    num_stages: gl.constexpr = 0,
    waves_per_eu: gl.constexpr = 0,
):

    WRITES_FINAL: gl.constexpr = NUM_KSPLIT == 1
    # prefetch depth is derived from the buffer count
    PD: gl.constexpr = NUM_BUFFERS - 1

    SHARED_LAYOUT_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 8]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    if TRANSPOSE:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_N, 16]], [BLOCK_K, BLOCK_N], [1, 0]
        )
    else:
        SHARED_LAYOUT_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BLOCK_K, 8]], [BLOCK_N, BLOCK_K], [1, 0]
        )

    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3,
        transposed=True,
        warp_bases=WARP_BASES,
        instr_shape=[16, 16, 32],
    )
    OPERAND_LAYOUT_A: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=8
    )
    OPERAND_LAYOUT_B: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=8
    )

    start_pid = gl.program_id(axis=0)
    num_pid_n = gl.cdiv(N, BLOCK_N)

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr,
        shape=(M, K),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )

    if TRANSPOSE:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(K, N),
            strides=(stride_bk, stride_bn),
            block_shape=(BLOCK_K, BLOCK_N),
            layout=SHARED_LAYOUT_B,
        )
    else:
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr,
            shape=(N, K),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_N, BLOCK_K),
            layout=SHARED_LAYOUT_B,
        )

    SHARED_LAYOUT_C: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_N, 8]], [BLOCK_M, BLOCK_N], [1, 0]
    )

    tile_id = start_pid
    while tile_id < num_tiles:
        # allocations done within the loop, this is very slow but allows prefetching
        a_buffer = gl.allocate_shared_memory(
            a_ptr.type.element_ty,
            shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
            layout=SHARED_LAYOUT_A,
        )
        if TRANSPOSE:
            b_buffer = gl.allocate_shared_memory(
                b_ptr.type.element_ty,
                shape=[NUM_BUFFERS, BLOCK_K, BLOCK_N],
                layout=SHARED_LAYOUT_B,
            )
        else:
            b_buffer = gl.allocate_shared_memory(
                b_ptr.type.element_ty,
                shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
                layout=SHARED_LAYOUT_B,
            )

        t = remap_xcd(tile_id, num_tiles, NUM_XCDS=8)
        pid_k = t % NUM_KSPLIT
        pid = t // NUM_KSPLIT
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
        m_off = pid_m * BLOCK_M
        n_off = pid_n * BLOCK_N
        sk_start = pid_k * SPLITK_BLOCK_SIZE
        cur_num_k_tiles = gl.cdiv(
            gl.minimum(sk_start + SPLITK_BLOCK_SIZE, K) - sk_start, BLOCK_K
        )

        # prologue: prefetch this tile's leading PD k-tiles at the top of the loop
        for pf in gl.static_range(PD):
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [m_off, sk_start + pf * BLOCK_K],
                a_buffer.index(pf % NUM_BUFFERS),
            )
            if TRANSPOSE:
                gl.amd.gfx1250.tdm.async_load(
                    b_desc,
                    [sk_start + pf * BLOCK_K, n_off],
                    b_buffer.index(pf % NUM_BUFFERS),
                )
            else:
                gl.amd.gfx1250.tdm.async_load(
                    b_desc,
                    [n_off, sk_start + pf * BLOCK_K],
                    b_buffer.index(pf % NUM_BUFFERS),
                )

        load_idx = PD
        compute_idx = 0

        accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
        if ADD_BIAS and pid_k == 0:
            offs_bias = n_off + gl.arange(
                0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
            )
            bias_vals = gl.load(bias_ptr + offs_bias, mask=offs_bias < N, other=0.0)
            accumulator = accumulator + bias_vals[None, :]

        gl.amd.gfx1250.tdm.async_wait((PD - 1) * 2)

        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        if TRANSPOSE:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_B
            )
        else:
            cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
                OPERAND_LAYOUT_B,
            )

        # main loop
        for _ in range(cur_num_k_tiles - PD):

            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [m_off, sk_start + load_idx * BLOCK_K],
                a_buffer.index(load_idx % NUM_BUFFERS),
            )
            if TRANSPOSE:
                gl.amd.gfx1250.tdm.async_load(
                    b_desc,
                    [sk_start + load_idx * BLOCK_K, n_off],
                    b_buffer.index(load_idx % NUM_BUFFERS),
                )
            else:
                gl.amd.gfx1250.tdm.async_load(
                    b_desc,
                    [n_off, sk_start + load_idx * BLOCK_K],
                    b_buffer.index(load_idx % NUM_BUFFERS),
                )

            gl.amd.gfx1250.tdm.async_wait((PD - 1) * 2)

            load_idx += 1

            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

            next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index((compute_idx + 1) % NUM_BUFFERS), OPERAND_LAYOUT_A
            )
            if TRANSPOSE:
                next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buffer.index((compute_idx + 1) % NUM_BUFFERS),
                    OPERAND_LAYOUT_B,
                )
            else:
                next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
                    OPERAND_LAYOUT_B,
                )

            cur_a = next_a
            cur_b = next_b
            compute_idx += 1

        # epilogue: drain remaining prefetched k-tiles
        for i in gl.static_range(PD - 1):
            gl.amd.gfx1250.tdm.async_wait((PD - 2 - i) * 2)

            next_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_buffer.index((compute_idx + 1) % NUM_BUFFERS), OPERAND_LAYOUT_A
            )
            if TRANSPOSE:
                next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buffer.index((compute_idx + 1) % NUM_BUFFERS),
                    OPERAND_LAYOUT_B,
                )
            else:
                next_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_buffer.index((compute_idx + 1) % NUM_BUFFERS).permute([1, 0]),
                    OPERAND_LAYOUT_B,
                )
            accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

            cur_a = next_a
            cur_b = next_b
            compute_idx += 1

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

        if USE_ACTIVATION and WRITES_FINAL:
            accumulator = activation(accumulator)

        # release a and b buffers, counterintuitive name
        a_buffer._keep_alive()
        b_buffer._keep_alive()

        c_buffer = gl.allocate_shared_memory(
            c_ptr.type.element_ty,
            shape=[BLOCK_M, BLOCK_N],
            layout=SHARED_LAYOUT_C,
        )

        c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=c_ptr + pid_k.to(gl.int64) * stride_ck,
            shape=(M, N),
            strides=(stride_cm, stride_cn),
            block_shape=(BLOCK_M, BLOCK_N),
            layout=SHARED_LAYOUT_C,
        )
        c_buffer.store(accumulator.to(c_ptr.type.element_ty))
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(c_desc, [m_off, n_off], c_buffer)
        gl.amd.gfx1250.tdm.async_wait(0)

        # release c buffer
        c_buffer._keep_alive()

        gl.barrier()

        tile_id = tile_id + NUM_SMS


_KERNEL_MAP = {
    "bandwidth_bound": gemm_a16w16_persistent_kernel_,
    "compute_bound": gemm_a16w16_persistent_compute_bound_kernel_,
}
