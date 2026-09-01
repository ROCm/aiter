# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _mxfp4_quant_op(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
):
    """
    Converts x (fp32) [BLOCK_SIZE_M, BLOCK_SIZE_N] to packed mxfp4 bytes via
    gl.amd.cdna5.scaled_downcast, computing the per-32-element e8m0 scale
    ourselves.
    """
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_grouped = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE)
    amax = gl.max(gl.abs(x_grouped), axis=-1, keep_dims=True)
    amax = amax.to(gl.int32, bitcast=True)
    amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(gl.float32, bitcast=True)
    scale_e8m0_unbiased = gl.log2(amax).floor() - 2
    scale_e8m0_unbiased = gl.maximum(-127, gl.minimum(scale_e8m0_unbiased, 127))
    bs_e8m0 = scale_e8m0_unbiased.to(gl.uint8) + 127
    bs_e8m0 = bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

    x_fp4 = gl.amd.cdna5.scaled_downcast(x, bs_e8m0, "e2m1", axis=1)

    return x_fp4, bs_e8m0


@gluon.jit
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
    NUM_STAGES: gl.constexpr,
    num_warps: gl.constexpr,
    MXFP4_QUANT_BLOCK_SIZE: gl.constexpr,
    EVEN_M_N: gl.constexpr,
    SCALING_MODE: gl.constexpr,
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

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE

    # LDS layout: row-major, vec=8 elements = 128-bit stores, padded to avoid bank conflicts
    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )

    # Register layout for LDS reads: order=[1,0] = N fastest, matches row-major LDS
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[4, 8],
        warps_per_cta=[1, num_warps],
        order=[1, 0],
    )

    # LDS ring buffer
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
    # ---- Prologue: fill NUM_BUFFERS-1 slots ----
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

    # ---- Main loop: load tile ahead, wait for oldest, quant, store ----
    for _ in range(num_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 1)  # 1 TDM op/tile
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

        x_reg = (
            x_buffer.index(compute_idx % NUM_BUFFERS)
            .load(layout=blocked_layout)
            .to(gl.float32)
        )

        out_fp4, bs_e8m0 = _mxfp4_quant_op(
            x_reg, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        pid_n = start_n + compute_idx
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
                & (
                    bs_offs_n
                    < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
                )[None, :],
            )
        compute_idx += 1

    # ---- Epilogue: drain remaining NUM_BUFFERS-1 tiles ----
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 2 - i)

        x_reg = (
            x_buffer.index(compute_idx % NUM_BUFFERS)
            .load(layout=blocked_layout)
            .to(gl.float32)
        )

        out_fp4, bs_e8m0 = _mxfp4_quant_op(
            x_reg, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
        )

        pid_n = start_n + compute_idx
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
                & (
                    bs_offs_n
                    < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
                )[None, :],
            )
        compute_idx += 1


@gluon.jit
def _mxfp8_quant_op(
    x,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    MXFP8_QUANT_BLOCK_SIZE: gl.constexpr,
    num_warps: gl.constexpr,
):
    """
    Converts x (fp32) [BLOCK_SIZE_M, BLOCK_SIZE_N] to fp8 e4m3 via
    gl.amd.cdna5.scaled_downcast, computing the per-32-element e8m0 scale
    ourselves. Unlike mxfp4, fp8 downcast is elementwise (no packing), so
    x_fp8 keeps the input's shape.

    scaled_downcast requires bs_e8m0 in a compact layout (size_per_thread
    reduced to NUM_QUANT_BLOCKS on the scaled axis), only expressible as a
    plain BlockedLayout when threads_per_warp[axis] == 32 // size_per_thread
    [axis] -- hence caller must use threads_per_warp=[8, 4], and we
    convert_layout since gl.max's native output layout doesn't match.
    """
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP8_QUANT_BLOCK_SIZE
    x_grouped = x.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP8_QUANT_BLOCK_SIZE)
    amax = gl.max(gl.abs(x_grouped), axis=-1, keep_dims=True)
    amax = amax.to(gl.int32, bitcast=True)
    amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(gl.float32, bitcast=True)
    # e4m3 dtypeMax = 448 = 2**8 * 1.75 -> unbiased exponent offset -8 (mxfp4: -2, dtypeMax 6)
    scale_e8m0_unbiased = gl.log2(amax).floor() - 8
    scale_e8m0_unbiased = gl.maximum(-127, gl.minimum(scale_e8m0_unbiased, 127))
    bs_e8m0 = scale_e8m0_unbiased.to(gl.uint8) + 127
    bs_e8m0 = bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

    # compact layout scaled_downcast requires for the scale (see docstring)
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
    NUM_STAGES: gl.constexpr,
    num_warps: gl.constexpr,
    MXFP8_QUANT_BLOCK_SIZE: gl.constexpr,
    EVEN_M_N: gl.constexpr,
    NUM_BUFFERS: gl.constexpr = 2,
):
    gl.static_assert(NUM_BUFFERS >= 2, "LDS kernel requires NUM_BUFFERS >= 2")

    pid_m = gl.program_id(0)
    start_n = gl.program_id(1) * NUM_ITER

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP8_QUANT_BLOCK_SIZE

    # LDS layout: row-major, vec=8 elements = 128-bit stores, padded to avoid bank conflicts
    SHARED_LAYOUT_X: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )

    # Register layout for LDS reads (order=[1,0]: N fastest, matches row-major LDS);
    # threads_per_warp=[8,4] required by _mxfp8_quant_op's scaled_downcast.
    blocked_layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 4],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )

    # LDS ring buffer
    x_buffer = gl.allocate_shared_memory(
        x_ptr.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=SHARED_LAYOUT_X,
    )
    out_smem = gl.allocate_shared_memory(
        x_fp8_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=SHARED_LAYOUT_X,
    )
    SHARED_LAYOUT_BS: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[NUM_QUANT_BLOCKS, 8]], [BLOCK_SIZE_M, NUM_QUANT_BLOCKS], [1, 0]
    )
    bs_smem = gl.allocate_shared_memory(
        bs_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, NUM_QUANT_BLOCKS],
        layout=SHARED_LAYOUT_BS,
    )

    # TDM descriptor: base at this CTA's (M, N) origin
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
        layout=SHARED_LAYOUT_X,
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
    # ---- Prologue: fill NUM_BUFFERS-1 slots ----
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

    # ---- Main loop: load tile ahead, wait for oldest, quant, store ----
    for _ in range(num_tiles - (NUM_BUFFERS - 1)):
        gl.amd.gfx1250.tdm.async_load(
            x_desc, [0, 0], x_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 1)  # 1 TDM op/tile
        x_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            x_desc, add_offsets=[0, BLOCK_SIZE_N]
        )
        load_idx += 1

        x_reg = (
            x_buffer.index(compute_idx % NUM_BUFFERS)
            .load(layout=blocked_layout)
            .to(gl.float32)
        )

        out_fp8, bs_e8m0 = _mxfp8_quant_op(
            x_reg, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP8_QUANT_BLOCK_SIZE, num_warps
        )

        pid_n = start_n + compute_idx
        out_smem.store(out_fp8)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(
            out_desc,
            [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
            out_smem,
        )
        gl.amd.gfx1250.tdm.async_wait(0)

        bs_smem.store(bs_e8m0)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(
            bs_desc,
            [pid_m * BLOCK_SIZE_M, pid_n * NUM_QUANT_BLOCKS],
            bs_smem,
        )
        gl.amd.gfx1250.tdm.async_wait(0)
        compute_idx += 1

    # ---- Epilogue: drain remaining NUM_BUFFERS-1 tiles ----
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 2 - i)

        x_reg = (
            x_buffer.index(compute_idx % NUM_BUFFERS)
            .load(layout=blocked_layout)
            .to(gl.float32)
        )

        out_fp8, bs_e8m0 = _mxfp8_quant_op(
            x_reg, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP8_QUANT_BLOCK_SIZE, num_warps
        )

        pid_n = start_n + compute_idx
        out_smem.store(out_fp8)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(
            out_desc,
            [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
            out_smem,
        )
        gl.amd.gfx1250.tdm.async_wait(0)

        bs_smem.store(bs_e8m0)
        gl.barrier()
        gl.amd.gfx1250.tdm.async_store(
            bs_desc,
            [pid_m * BLOCK_SIZE_M, pid_n * NUM_QUANT_BLOCKS],
            bs_smem,
        )
        gl.amd.gfx1250.tdm.async_wait(0)
        compute_idx += 1

