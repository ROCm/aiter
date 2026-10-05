# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.common.utils import mx_e8m0_scale
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_gluon_dynamic_mxfp4_quant_kernel_gfx1250_repr = make_kernel_repr(
    "gluon_dynamic_mxfp4_quant_kernel_gfx1250",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "NUM_ITER",
        "num_warps",
        "MXFP4_QUANT_BLOCK_SIZE",
        "EVEN_M_N",
        "NUM_BUFFERS",
    ],
)


@gluon.jit
def _mxfp4_downcast_op(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
):
    """x (fp32) -> packed mxfp4 via scaled_downcast, plus its e8m0 scale."""
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_grouped = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    amax = gl.max(gl.abs(x_grouped), axis=-1, keep_dims=True)
    bs_e8m0 = mx_e8m0_scale(amax, 2)
    bs_e8m0 = bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

    x_fp4 = gl.amd.cdna5.scaled_downcast(x, bs_e8m0, "e2m1", axis=1)

    return x_fp4, bs_e8m0


@gluon.jit
def _mxfp4_quant_tile(
    x_buffer,
    out_smem,
    out_desc,
    bs_ptr,
    compute_idx,
    start_n,
    pid_m,
    M,
    N,
    stride_bs_m,
    stride_bs_n,
    blocked_layout: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    EVEN_M_N: gl.constexpr,
):
    """Quantize the LDS tile at compute_idx, then store its payload and scales."""
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_reg = (
        x_buffer.index(compute_idx % NUM_BUFFERS)
        .load(layout=blocked_layout)
        .to(gl.float32)
    )

    pid_n = start_n + compute_idx
    if not EVEN_M_N:
        # Tail tile: mask stale ring-buffer columns past N.
        gLayoutN: gl.constexpr = gl.SliceLayout(0, blocked_layout)
        col_valid = (
            pid_n * BLOCK_SIZE_N + gl.arange(0, BLOCK_SIZE_N, layout=gLayoutN)
        ) < N
        x_reg = gl.where(col_valid[None, :], x_reg, 0.0)
    out_fp4, bs_e8m0 = _mxfp4_downcast_op(
        x_reg, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
    )

    out_smem.store(out_fp4)
    gl.barrier()
    gl.amd.gfx1250.tdm.async_store(
        out_desc,
        [pid_m * BLOCK_SIZE_M, pid_n * (BLOCK_SIZE_N // 2)],
        out_smem,
    )
    gl.amd.gfx1250.tdm.async_wait(0)

    bs_offs_m = pid_m * BLOCK_SIZE_M + gl.arange(0, BLOCK_SIZE_M)
    bs_offs_n = pid_n * NUM_QUANT_BLOCKS + gl.arange(0, NUM_QUANT_BLOCKS)
    bs_offs = bs_offs_m[:, None] * stride_bs_m + bs_offs_n[None, :] * stride_bs_n
    if EVEN_M_N:
        gl.store(bs_ptr + bs_offs, bs_e8m0)
    else:
        gl.store(
            bs_ptr + bs_offs,
            bs_e8m0,
            mask=(bs_offs_m < M)[:, None]
            & (bs_offs_n < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE)[
                None, :
            ],
        )


@gluon.jit(repr=_gluon_dynamic_mxfp4_quant_kernel_gfx1250_repr)
def gluon_dynamic_mxfp4_quant_kernel_gfx1250(
    x_ptr,
    x_fp4_ptr,
    bs_ptr,
    stride_x_m_in,
    stride_x_n_in,
    stride_x_fp4_m_in,
    stride_x_fp4_n_in,
    stride_bs_m_in,
    stride_bs_n_in,
    M,
    N,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    NUM_ITER: gl.constexpr,
    num_warps: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    EVEN_M_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr = 2,
):
    gl.static_assert(NUM_BUFFERS >= 2, "LDS kernel requires NUM_BUFFERS >= 2")

    pid_m = gl.program_id(0)
    start_n = gl.program_id(1) * NUM_ITER

    stride_x_m = gl.cast(stride_x_m_in, gl.int64)
    stride_x_n = gl.cast(stride_x_n_in, gl.int64)
    stride_x_fp4_m = gl.cast(stride_x_fp4_m_in, gl.int64)
    stride_x_fp4_n = gl.cast(stride_x_fp4_n_in, gl.int64)
    stride_bs_m = gl.cast(stride_bs_m_in, gl.int64)
    stride_bs_n = gl.cast(stride_bs_n_in, gl.int64)

    # Padded row-major LDS, 128-bit vectors.
    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )

    # N-fastest, matching the LDS layout.
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 8],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )

    # LDS ring buffer.
    x_buffer = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=SHARED_LAYOUT_X,
    )
    SHARED_LAYOUT_O: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N // 2, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N // 2], [1, 0]
    )
    out_smem = gl.allocate_shared_memory(
        x_fp4_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N // 2],
        layout=SHARED_LAYOUT_O,
    )

    # TDM descriptor: base at this CTA's (M, N) origin
    x_base = (
        x_ptr + pid_m * BLOCK_SIZE_M * stride_x_m + start_n * BLOCK_SIZE_N * stride_x_n
    )
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_base,
        shape=(M - pid_m * BLOCK_SIZE_M, N - start_n * BLOCK_SIZE_N),
        strides=(stride_x_m, stride_x_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=SHARED_LAYOUT_X,
    )
    out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_fp4_ptr,
        shape=(M, N // 2),
        strides=(stride_x_fp4_m, stride_x_fp4_n),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N // 2),
        layout=SHARED_LAYOUT_O,
    )

    load_idx = 0
    compute_idx = 0
    num_tiles = min(NUM_ITER, gl.cdiv(N, BLOCK_SIZE_N) - start_n)
    # Prologue: fill NUM_BUFFERS-1 slots.
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

    # Main loop: load ahead, wait for the oldest tile, quantize, store.
    for _ in range(num_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 1)  # 1 TDM op/tile
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

        _mxfp4_quant_tile(
            x_buffer,
            out_smem,
            out_desc,
            bs_ptr,
            compute_idx,
            start_n,
            pid_m,
            M,
            N,
            stride_bs_m,
            stride_bs_n,
            blocked_layout,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            MXFP4_QUANT_BLOCK_SIZE,
            NUM_BUFFERS,
            EVEN_M_N,
        )
        compute_idx += 1

    # Epilogue: drain the remaining NUM_BUFFERS-1 tiles.
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 2 - i)

        _mxfp4_quant_tile(
            x_buffer,
            out_smem,
            out_desc,
            bs_ptr,
            compute_idx,
            start_n,
            pid_m,
            M,
            N,
            stride_bs_m,
            stride_bs_n,
            blocked_layout,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            MXFP4_QUANT_BLOCK_SIZE,
            NUM_BUFFERS,
            EVEN_M_N,
        )
        compute_idx += 1


_gluon_dynamic_mxfp8_quant_kernel_gfx1250_repr = make_kernel_repr(
    "gluon_dynamic_mxfp8_quant_kernel_gfx1250",
    [
        "BLOCK_SIZE_M",
        "BLOCK_SIZE_N",
        "NUM_ITER",
        "num_warps",
        "MXFP8_QUANT_BLOCK_SIZE",
        "NUM_BUFFERS",
    ],
)


@gluon.jit
def _mxfp8_downcast_op(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP8_QUANT_BLOCK_SIZE: gl.constexpr,
    num_warps: gl.constexpr,
):
    """x (fp32) -> fp8 e4m3 with a per-block e8m0 scale."""
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP8_QUANT_BLOCK_SIZE
    x_grouped = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP8_QUANT_BLOCK_SIZE)
    amax = gl.max(gl.abs(x_grouped), axis=-1, keep_dims=True)
    bs_e8m0 = mx_e8m0_scale(amax, 8)
    bs_e8m0 = bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

    # scaled_downcast requires the scale in this compact per-thread layout.
    compact_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, NUM_QUANT_BLOCKS],
        threads_per_warp=[8, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    bs_e8m0 = gl.convert_layout(bs_e8m0, compact_layout)
    x_fp8 = gl.amd.cdna5.scaled_downcast(x, bs_e8m0, "e4m3", axis=1)

    return x_fp8, bs_e8m0


@gluon.jit
def _mxfp8_quant_tile(
    x_buffer,
    out_smem,
    bs_smem,
    out_desc,
    bs_desc,
    compute_idx,
    start_n,
    pid_m,
    blocked_layout: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    MXFP8_QUANT_BLOCK_SIZE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    num_warps: gl.constexpr,
):
    """Quantize the LDS tile at compute_idx, then start its payload and scale stores."""
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP8_QUANT_BLOCK_SIZE
    # 2 stores per tile; TDM ops complete in issue order.
    STORE_WAIT: gl.constexpr = 2 * (NUM_BUFFERS - 1)
    x_reg = (
        x_buffer.index(compute_idx % NUM_BUFFERS)
        .load(layout=blocked_layout)
        .to(gl.float32)
    )
    out_fp8, bs_e8m0 = _mxfp8_downcast_op(
        x_reg, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP8_QUANT_BLOCK_SIZE, num_warps
    )

    pid_n = start_n + compute_idx
    store_slot = compute_idx % NUM_BUFFERS
    gl.amd.gfx1250.tdm.async_wait(STORE_WAIT)
    out_smem.index(store_slot).store(out_fp8)
    bs_smem.index(store_slot).store(bs_e8m0)
    gl.barrier()  # One barrier covers both LDS writes.
    gl.amd.gfx1250.tdm.async_store(
        out_desc,
        [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        out_smem.index(store_slot),
    )
    gl.amd.gfx1250.tdm.async_store(
        bs_desc,
        [pid_m * BLOCK_SIZE_M, pid_n * NUM_QUANT_BLOCKS],
        bs_smem.index(store_slot),
    )


@gluon.jit(repr=_gluon_dynamic_mxfp8_quant_kernel_gfx1250_repr)
def gluon_dynamic_mxfp8_quant_kernel_gfx1250(
    x_ptr,
    x_fp8_ptr,
    bs_ptr,
    stride_x_m_in: gl.constexpr,
    stride_x_n_in: gl.constexpr,
    stride_x_fp8_m_in: gl.constexpr,
    stride_x_fp8_n_in: gl.constexpr,
    stride_bs_m_in: gl.constexpr,
    stride_bs_n_in: gl.constexpr,
    M,
    N,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    NUM_ITER: gl.constexpr,
    num_warps: gl.constexpr,
    MXFP8_QUANT_BLOCK_SIZE: gl.constexpr,
    NUM_BUFFERS: gl.constexpr = 2,
):
    # NUM_BUFFERS=1: synchronous, no prefetch.
    gl.static_assert(NUM_BUFFERS >= 1, "LDS kernel requires NUM_BUFFERS >= 1")
    gl.static_assert(
        BLOCK_SIZE_N <= 1024,
        "BLOCK_SIZE_N > 1024 not yet verified safe for fp8 TDM async_store",
    )

    pid_m = gl.program_id(0)
    start_n = gl.program_id(1) * NUM_ITER

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP8_QUANT_BLOCK_SIZE

    # Padded row-major LDS, 128-bit vectors.
    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )

    # order=[1,0]: N fastest; threads_per_warp=[8,4] required by scaled_downcast
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    x_buffer = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=SHARED_LAYOUT_X,
    )
    # Unpadded: padding corrupts fp8 async_store at BLOCK_SIZE_N >= 256.
    SHARED_LAYOUT_OUT: gl.constexpr = gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
    out_smem = gl.allocate_shared_memory(
        x_fp8_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=SHARED_LAYOUT_OUT,
    )
    SHARED_LAYOUT_BS: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[NUM_QUANT_BLOCKS, 8]], [BLOCK_SIZE_M, NUM_QUANT_BLOCKS], [1, 0]
    )
    bs_smem = gl.allocate_shared_memory(
        bs_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, NUM_QUANT_BLOCKS],
        layout=SHARED_LAYOUT_BS,
    )

    x_base = (
        x_ptr
        + pid_m * BLOCK_SIZE_M * stride_x_m_in
        + start_n * BLOCK_SIZE_N * stride_x_n_in
    )
    x_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_base,
        shape=(M - pid_m * BLOCK_SIZE_M, N - start_n * BLOCK_SIZE_N),
        strides=(stride_x_m_in, stride_x_n_in),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=SHARED_LAYOUT_X,
    )
    out_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=x_fp8_ptr,
        shape=(M, N),
        strides=(stride_x_fp8_m_in, stride_x_fp8_n_in),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=SHARED_LAYOUT_OUT,
    )
    bs_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=bs_ptr,
        shape=(M, (N + MXFP8_QUANT_BLOCK_SIZE - 1) // MXFP8_QUANT_BLOCK_SIZE),
        strides=(stride_bs_m_in, stride_bs_n_in),
        block_shape=(BLOCK_SIZE_M, NUM_QUANT_BLOCKS),
        layout=SHARED_LAYOUT_BS,
    )

    load_idx = 0
    compute_idx = 0
    num_tiles = min(NUM_ITER, gl.cdiv(N, BLOCK_SIZE_N) - start_n)
    # Prologue: fill NUM_BUFFERS-1 slots.
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

    # Main loop: load ahead, wait for the oldest tile, quantize, store.
    for _ in range(num_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 1)
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

        _mxfp8_quant_tile(
            x_buffer,
            out_smem,
            bs_smem,
            out_desc,
            bs_desc,
            compute_idx,
            start_n,
            pid_m,
            blocked_layout,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            MXFP8_QUANT_BLOCK_SIZE,
            NUM_BUFFERS,
            num_warps,
        )
        compute_idx += 1

    # Epilogue: drain the remaining NUM_BUFFERS-1 tiles.
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 2 - i)

        _mxfp8_quant_tile(
            x_buffer,
            out_smem,
            bs_smem,
            out_desc,
            bs_desc,
            compute_idx,
            start_n,
            pid_m,
            blocked_layout,
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            MXFP8_QUANT_BLOCK_SIZE,
            NUM_BUFFERS,
            num_warps,
        )
        compute_idx += 1

    # The CTA must not retire with a TDM store still outstanding.
    gl.amd.gfx1250.tdm.async_wait(0)
