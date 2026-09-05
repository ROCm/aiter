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
    Converts x (bf16) [BLOCK_SIZE_M, BLOCK_SIZE_N] to packed mxfp4 bytes via
    gl.amd.cdna4.scaled_downcast, computing the per-32-element e8m0 scale
    ourselves. Reduces over split evens/odds, not a plain reshape+max, so
    amax's layout stays compatible with the split tensors it's compared to.
    """
    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    x_grouped = x.reshape(
        BLOCK_SIZE_M, NUM_QUANT_BLOCKS, MXFP4_QUANT_BLOCK_SIZE // 2, 2
    )
    evens, odds = gl.split(x_grouped)
    amax = gl.maximum(gl.abs(evens), gl.abs(odds)).to(gl.float32)
    amax = gl.max(amax, axis=-1, keep_dims=True)
    amax = amax.to(gl.int32, bitcast=True)
    amax = (amax + 0x200000).to(gl.uint32, bitcast=True) & 0xFF800000
    amax = amax.to(gl.float32, bitcast=True)
    scale_e8m0_unbiased = gl.log2(amax).floor() - 2
    scale_e8m0_unbiased = gl.maximum(-127, gl.minimum(scale_e8m0_unbiased, 127))
    bs_e8m0 = scale_e8m0_unbiased.to(gl.uint8) + 127
    bs_e8m0 = bs_e8m0.reshape(BLOCK_SIZE_M, NUM_QUANT_BLOCKS)

    x_fp4 = gl.amd.cdna4.scaled_downcast(x, bs_e8m0, "e2m1", axis=1)

    return x_fp4, bs_e8m0


@gluon.jit
def gluon_dynamic_mxfp4_quant_kernel_gfx950(
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
):
    pid_m = gl.program_id(0)
    start_n = gl.program_id(1) * NUM_ITER
    # cast strides to int64, in case M*N > max int32
    stride_x_m = gl.cast(stride_x_m_in, gl.int64)
    stride_x_n = gl.cast(stride_x_n_in, gl.int64)
    stride_x_fp4_m = gl.cast(stride_x_fp4_m_in, gl.int64)
    stride_x_fp4_n = gl.cast(stride_x_fp4_n_in, gl.int64)
    stride_bs_m = gl.cast(stride_bs_m_in, gl.int64)
    stride_bs_n = gl.cast(stride_bs_n_in, gl.int64)

    NUM_QUANT_BLOCKS: gl.constexpr = BLOCK_SIZE_N // MXFP4_QUANT_BLOCK_SIZE
    # Each warp's nominal per-axis tile (given size_per_thread=[1,8],
    # threads_per_warp=[8,8]) is 8 rows along M or 64 cols along N. Putting all
    # `num_warps` along an axis whose BLOCK_SIZE can't fit them (nominal tile >
    # BLOCK_SIZE) leaves most warps idle -- e.g. BLOCK_SIZE_M=8 narrow-N configs.
    # Prefer the M axis (matches the large-BLOCK_SIZE_N configs this was tuned
    # for) and only fall back to the N axis when M can't fit all the warps.
    WARPS_M: gl.constexpr = num_warps if (BLOCK_SIZE_M // 8) >= num_warps else 1
    WARPS_N: gl.constexpr = num_warps // WARPS_M
    # N (dim 1) is memory-contiguous; vectorize 8 elements/thread there for dwordx4 loads.
    layout: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 8],
        threads_per_warp=[8, 8],
        warps_per_cta=[WARPS_M, WARPS_N],
        order=[1, 0],
    )

    end_n = min(start_n + NUM_ITER, N)

    # Per-lane intra-block offsets don't depend on pid_n -- compute once so the
    # buffer_load vaddr operand is loop-invariant (compiler hoists it), like
    # the non-prefetch kernel already does.
    local_m = gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, layout))
    local_n = gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, layout))
    x_offs = local_m[:, None] * stride_x_m_in + local_n[None, :] * stride_x_n_in
    # Loop-invariant per-block-column stride: an affine accumulator added to
    # the scalar base pointer each iteration, so the compiler can strength-
    # reduce it to one cheap add/iteration instead of re-deriving from an index.
    x_block_stride = BLOCK_SIZE_N * stride_x_n
    if not EVEN_M_N:
        x_offs_m = pid_m * BLOCK_SIZE_M + local_m

    # NUM_STAGES==1 (skinny shapes): plain loop, no cross-iteration overlap.
    # NUM_STAGES==2 (larger shapes): double-buffered, warp_pipeline_stage-staged
    # loop below. Gluon has no closures, so the body is duplicated per branch.
    if NUM_STAGES == 1:
        for pid_n in range(start_n, end_n):
            x_block_ptr = (
                x_ptr
                + (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_x_m
                + (pid_n * BLOCK_SIZE_N).to(gl.int64) * stride_x_n
            )
            if EVEN_M_N:
                x = gl.amd.cdna4.buffer_load(x_block_ptr, x_offs, cache=".cg")
            else:
                x_offs_n = pid_n * BLOCK_SIZE_N + local_n
                x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
                x = gl.amd.cdna4.buffer_load(
                    x_block_ptr, x_offs, mask=x_mask, cache=".cg"
                )

            out_tensor, bs_e8m0 = _mxfp4_quant_op(
                x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
            )

            out_m_local = gl.arange(0, BLOCK_SIZE_M)
            out_n_local = gl.arange(0, BLOCK_SIZE_N // 2)
            out_offs_m = pid_m * BLOCK_SIZE_M + out_m_local
            out_offs_n = pid_n * (BLOCK_SIZE_N // 2) + out_n_local
            out_block_offset = (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_x_fp4_m + (
                pid_n * (BLOCK_SIZE_N // 2)
            ).to(gl.int64) * stride_x_fp4_n
            out_block_ptr = x_fp4_ptr + out_block_offset
            out_offs = (
                out_m_local[:, None] * stride_x_fp4_m_in
                + out_n_local[None, :] * stride_x_fp4_n_in
            )

            if EVEN_M_N:
                gl.amd.cdna4.buffer_store(out_tensor, out_block_ptr, out_offs)
            else:
                out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[None, :]
                gl.amd.cdna4.buffer_store(
                    out_tensor, out_block_ptr, out_offs, mask=out_mask
                )

            bs_m_local = gl.arange(0, BLOCK_SIZE_M)
            bs_n_local = gl.arange(0, NUM_QUANT_BLOCKS)
            bs_offs_m = pid_m * BLOCK_SIZE_M + bs_m_local
            bs_offs_n = pid_n * NUM_QUANT_BLOCKS + bs_n_local
            bs_block_offset = (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_bs_m + (
                pid_n * NUM_QUANT_BLOCKS
            ).to(gl.int64) * stride_bs_n
            bs_block_ptr = bs_ptr + bs_block_offset
            bs_offs = (
                bs_m_local[:, None] * stride_bs_m_in
                + bs_n_local[None, :] * stride_bs_n_in
            )
            if EVEN_M_N:
                gl.amd.cdna4.buffer_store(bs_e8m0, bs_block_ptr, bs_offs)
            else:
                bs_mask = (bs_offs_m < M)[:, None] & (
                    bs_offs_n
                    < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
                )[None, :]
                gl.amd.cdna4.buffer_store(
                    bs_e8m0,
                    bs_block_ptr,
                    bs_offs,
                    mask=bs_mask,
                )
    else:
        # Prologue: load the first iteration unconditionally (start_n < end_n
        # is guaranteed by the launch grid).
        pid_n = start_n
        x_block_ptr = (
            x_ptr
            + (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_x_m
            + (pid_n * BLOCK_SIZE_N).to(gl.int64) * stride_x_n
        )
        if EVEN_M_N:
            x = gl.amd.cdna4.buffer_load(x_block_ptr, x_offs, cache=".cg")
        else:
            x_offs_n = pid_n * BLOCK_SIZE_N + local_n
            x_mask = (x_offs_m < M)[:, None] & (x_offs_n < N)[None, :]
            x = gl.amd.cdna4.buffer_load(x_block_ptr, x_offs, mask=x_mask, cache=".cg")

        for pid_n in range(start_n, end_n):
            # Warp-pipeline staging: marks the load as a separate cluster from
            # compute+store, letting the backend overlap them across warps.
            with gl.amd.warp_pipeline_stage("load", priority=1):
                # Uniform (warp-synchronous) scalar check -- not a per-lane mask --
                # so this is a plain branch, no divergence.
                has_next = pid_n + 1 < end_n
                x_block_ptr_next = x_block_ptr + x_block_stride
                if EVEN_M_N:
                    if has_next:
                        x_next = gl.amd.cdna4.buffer_load(
                            x_block_ptr_next, x_offs, cache=".cg"
                        )
                    else:
                        x_next = x
                else:
                    x_offs_n_next = (pid_n + 1) * BLOCK_SIZE_N + local_n
                    x_mask_next = (
                        (x_offs_m < M)[:, None]
                        & (x_offs_n_next < N)[None, :]
                        & has_next
                    )
                    x_next = gl.amd.cdna4.buffer_load(
                        x_block_ptr_next, x_offs, mask=x_mask_next, cache=".cg"
                    )

            with gl.amd.warp_pipeline_stage("compute_store", priority=0):
                out_tensor, bs_e8m0 = _mxfp4_quant_op(
                    x, BLOCK_SIZE_N, BLOCK_SIZE_M, MXFP4_QUANT_BLOCK_SIZE
                )

                out_m_local = gl.arange(0, BLOCK_SIZE_M)
                out_n_local = gl.arange(0, BLOCK_SIZE_N // 2)
                out_offs_m = pid_m * BLOCK_SIZE_M + out_m_local
                out_offs_n = pid_n * (BLOCK_SIZE_N // 2) + out_n_local
                out_block_offset = (pid_m * BLOCK_SIZE_M).to(
                    gl.int64
                ) * stride_x_fp4_m + (pid_n * (BLOCK_SIZE_N // 2)).to(
                    gl.int64
                ) * stride_x_fp4_n
                out_block_ptr = x_fp4_ptr + out_block_offset
                out_offs = (
                    out_m_local[:, None] * stride_x_fp4_m_in
                    + out_n_local[None, :] * stride_x_fp4_n_in
                )

                if EVEN_M_N:
                    gl.amd.cdna4.buffer_store(out_tensor, out_block_ptr, out_offs)
                else:
                    out_mask = (out_offs_m < M)[:, None] & (out_offs_n < (N // 2))[
                        None, :
                    ]
                    gl.amd.cdna4.buffer_store(
                        out_tensor, out_block_ptr, out_offs, mask=out_mask
                    )

                bs_m_local = gl.arange(0, BLOCK_SIZE_M)
                bs_n_local = gl.arange(0, NUM_QUANT_BLOCKS)
                bs_offs_m = pid_m * BLOCK_SIZE_M + bs_m_local
                bs_offs_n = pid_n * NUM_QUANT_BLOCKS + bs_n_local
                bs_block_offset = (pid_m * BLOCK_SIZE_M).to(gl.int64) * stride_bs_m + (
                    pid_n * NUM_QUANT_BLOCKS
                ).to(gl.int64) * stride_bs_n
                bs_block_ptr = bs_ptr + bs_block_offset
                bs_offs = (
                    bs_m_local[:, None] * stride_bs_m_in
                    + bs_n_local[None, :] * stride_bs_n_in
                )
                if EVEN_M_N:
                    gl.amd.cdna4.buffer_store(bs_e8m0, bs_block_ptr, bs_offs)
                else:
                    bs_mask = (bs_offs_m < M)[:, None] & (
                        bs_offs_n
                        < (N + MXFP4_QUANT_BLOCK_SIZE - 1) // MXFP4_QUANT_BLOCK_SIZE
                    )[None, :]
                    gl.amd.cdna4.buffer_store(
                        bs_e8m0,
                        bs_block_ptr,
                        bs_offs,
                        mask=bs_mask,
                    )

            x = x_next
            x_block_ptr = x_block_ptr_next
