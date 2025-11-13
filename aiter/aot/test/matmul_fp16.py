# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl
from triton.experimental import gluon
from triton.tools.compile import compile_kernel, CompileArgs
from triton.experimental.gluon import language as gl

@gluon.jit
def matmul_fp16(
    # C,
    A,
    B,
    # M,
    # N,
    # K,
    # stride_cm,
    # stride_cn,
    # stride_am,
    # stride_ak,
    # stride_bk,
    # stride_bn,
    # BLOCK_M: gl.constexpr,
    # BLOCK_N: gl.constexpr,
    # BLOCK_K: gl.constexpr,
):
    a = gl.load(A)
    gl.store(B, a)
    # NUM_WARPS: gl.constexpr = 4
    # pid_m = gl.program_id(axis=0)
    # pid_n = gl.program_id(axis=1)

    # # Define layouts
    # threads_per_elem_mk: gl.constexpr = triton.cdiv(
    #     BLOCK_M * BLOCK_K // (NUM_WARPS * 64), 16
    # )
    # threads_per_elem_kn: gl.constexpr = triton.cdiv(
    #     BLOCK_K * BLOCK_N // (NUM_WARPS * 64), 16
    # )
    # blocked_kn: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[16, threads_per_elem_kn],
    #     threads_per_warp=[8, 8],
    #     warps_per_cta=[1, NUM_WARPS],
    #     order=[0, 1],
    # )
    # mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
    #     version=3,
    #     instr_shape=[16, 16],
    #     transposed=True,
    #     warps_per_cta=[NUM_WARPS // 2, 2],
    # )

    # shared_a: gl.constexpr = gl.SwizzledSharedLayout(

    #     vec=16, per_phase=2, max_phase=8, order=[1, 0]
    # )
    # shared_b: gl.constexpr = gl.SwizzledSharedLayout(
    #     vec=16, per_phase=2, max_phase=8, order=[0, 1]
    # )
    # dot_a_layout: gl.constexpr = gl.DotOperandLayout(
    #     operand_index=0, parent=mfma_layout, k_width=16
    # )
    # dot_b_layout: gl.constexpr = gl.DotOperandLayout(
    #     operand_index=1, parent=mfma_layout, k_width=16
    # )

    # # Allocate shared memory
    # smem_a = gl.allocate_shared_memory(
    #     A.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a
    # )
    # smem_b = gl.allocate_shared_memory(
    #     B.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b
    # )

    # # Create offsets for first block of A and B
    # offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_mk))
    # offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked_kn))
    # offs_am = pid_m * BLOCK_M + gl.arange(
    #     0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk)
    # )
    # offs_bn = pid_n * BLOCK_N + gl.arange(
    #     0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kn)
    # )

    # offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    # offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # # Load first block
    # a = gl.amd.cdna3.buffer_load(
    #     ptr=A,
    #     offsets=offs_a,
    #     mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M),
    #     cache=".ca",
    # )
    # b = gl.amd.cdna3.buffer_load(
    #     ptr=B,
    #     offsets=offs_b,
    #     mask=(offs_bk[:, None] < K) & (offs_bn[None, :] < N),
    #     cache=".ca",
    # )
    # smem_a.store(a)

    # # Initialize accumulator
    # accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    # zeros = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    # num_k_iter = gl.cdiv(K, BLOCK_K)

    # for k in range(num_k_iter - 1):
    #     # Advance the offsets to the next K block
    #     offs_a += BLOCK_K * stride_ak
    #     offs_b += BLOCK_K * stride_bk

    #     # Load the next block of A and B
    #     a = gl.amd.cdna3.buffer_load(
    #         ptr=A,
    #         offsets=offs_a,
    #         mask=(offs_ak[None, :] < K - (k + 1) * BLOCK_K) & (offs_am[:, None] < M),
    #         cache=".ca",
    #     )

    #     # Store current B and load from shared memory
    #     smem_b.store(b)
    #     cur_a = smem_a.load(layout=dot_a_layout)

    #     b = gl.amd.cdna3.buffer_load(
    #         ptr=B,
    #         offsets=offs_b,
    #         mask=(offs_bk[:, None] < K - (k + 1) * BLOCK_K) & (offs_bn[None, :] < N),
    #         cache=".ca",
    #     )
    #     cur_b = smem_b.load(layout=dot_b_layout)

    #     # Perform MFMA
    #     mfma_out = gl.amd.cdna3.mfma(cur_a, cur_b, zeros)
    #     accumulator += mfma_out

    #     smem_a.store(a)

    # # Epilogue - final iteration
    # smem_b.store(b)
    # cur_a = smem_a.load(layout=dot_a_layout)
    # cur_b = smem_b.load(layout=dot_b_layout)

    # mfma_out = gl.amd.cdna3.mfma(cur_a, cur_b, zeros)
    # accumulator += mfma_out

    # c = accumulator

    # # Write back the block of the output matrix C with masks
    # offs_cm = pid_m * BLOCK_M + gl.arange(
    #     0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout)
    # )
    # offs_cn = pid_n * BLOCK_N + gl.arange(
    #     0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
    # )
    # c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # gl.amd.cdna3.buffer_store(
    #     stored_value=c, ptr=C, offsets=c_offs, mask=c_mask
    # )

    # blocked_mk: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[threads_per_elem_mk, 16],
    #     threads_per_warp=[8, 8],
    #     warps_per_cta=[NUM_WARPS, 1],
    #     order=[1, 0],
    # )
    # blocked_kn: gl.constexpr = gl.BlockedLayout(
    #     size_per_thread=[16, threads_per_elem_kn],
    #     threads_per_warp=[8, 8],
    #     warps_per_cta=[1, NUM_WARPS],
    #     order=[0, 1],
    # )
    # mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
    #     version=3,
    #     instr_shape=[16, 16],
    #     transposed=True,
    #     warps_per_cta=[NUM_WARPS // 2, 2],
    # )

    # shared_a: gl.constexpr = gl.SwizzledSharedLayout(

    #     vec=16, per_phase=2, max_phase=8, order=[1, 0]
    # )
    # shared_b: gl.constexpr = gl.SwizzledSharedLayout(
    #     vec=16, per_phase=2, max_phase=8, order=[0, 1]
    # )
    # dot_a_layout: gl.constexpr = gl.DotOperandLayout(
    #     operand_index=0, parent=mfma_layout, k_width=16
    # )
    # dot_b_layout: gl.constexpr = gl.DotOperandLayout(
    #     operand_index=1, parent=mfma_layout, k_width=16
    # )

    # # Allocate shared memory
    # smem_a = gl.allocate_shared_memory(
    #     A.type.element_ty, [BLOCK_M, BLOCK_K], layout=shared_a
    # )
    # smem_b = gl.allocate_shared_memory(
    #     B.type.element_ty, [BLOCK_K, BLOCK_N], layout=shared_b
    # )

    # # Create offsets for first block of A and B
    # offs_ak = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(0, blocked_mk))
    # offs_bk = gl.arange(0, BLOCK_K, layout=gl.SliceLayout(1, blocked_kn))
    # offs_am = pid_m * BLOCK_M + gl.arange(
    #     0, BLOCK_M, layout=gl.SliceLayout(1, blocked_mk)
    # )
    # offs_bn = pid_n * BLOCK_N + gl.arange(
    #     0, BLOCK_N, layout=gl.SliceLayout(0, blocked_kn)
    # )

    # offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    # offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    # # Load first block
    # a = gl.amd.cdna3.buffer_load(
    #     ptr=A,
    #     offsets=offs_a,
    #     mask=(offs_ak[None, :] < K) & (offs_am[:, None] < M),
    #     cache=".ca",
    # )
    # b = gl.amd.cdna3.buffer_load(
    #     ptr=B,
    #     offsets=offs_b,
    #     mask=(offs_bk[:, None] < K) & (offs_bn[None, :] < N),
    #     cache=".ca",
    # )
    # smem_a.store(a)

    # # Initialize accumulator
    # accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)
    # zeros = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=mfma_layout)

    # num_k_iter = gl.cdiv(K, BLOCK_K)

    # for k in range(num_k_iter - 1):
    #     # Advance the offsets to the next K block
    #     offs_a += BLOCK_K * stride_ak
    #     offs_b += BLOCK_K * stride_bk

    #     # Load the next block of A and B
    #     a = gl.amd.cdna3.buffer_load(
    #         ptr=A,
    #         offsets=offs_a,
    #         mask=(offs_ak[None, :] < K - (k + 1) * BLOCK_K) & (offs_am[:, None] < M),
    #         cache=".ca",
    #     )

    #     # Store current B and load from shared memory
    #     smem_b.store(b)
    #     cur_a = smem_a.load(layout=dot_a_layout)

    #     b = gl.amd.cdna3.buffer_load(
    #         ptr=B,
    #         offsets=offs_b,
    #         mask=(offs_bk[:, None] < K - (k + 1) * BLOCK_K) & (offs_bn[None, :] < N),
    #         cache=".ca",
    #     )
    #     cur_b = smem_b.load(layout=dot_b_layout)

    #     # Perform MFMA
    #     mfma_out = gl.amd.cdna3.mfma(cur_a, cur_b, zeros)
    #     accumulator += mfma_out

    #     smem_a.store(a)

    # # Epilogue - final iteration
    # smem_b.store(b)
    # cur_a = smem_a.load(layout=dot_a_layout)
    # cur_b = smem_b.load(layout=dot_b_layout)

    # mfma_out = gl.amd.cdna3.mfma(cur_a, cur_b, zeros)
    # accumulator += mfma_out

    # c = accumulator

    # # Write back the block of the output matrix C with masks
    # offs_cm = pid_m * BLOCK_M + gl.arange(
    #     0, BLOCK_M, layout=gl.SliceLayout(1, mfma_layout)
    # )
    # offs_cn = pid_n * BLOCK_N + gl.arange(
    #     0, BLOCK_N, layout=gl.SliceLayout(0, mfma_layout)
    # )
    # c_offs = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    # c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    # gl.amd.cdna3.buffer_store(
    #     stored_value=c, ptr=C, offsets=c_offs, mask=c_mask
    # )


if __name__ == "__main__":
    compile_args = CompileArgs(
        path=__file__,
        kernel_name="matmul_fp16",
        signature="*fp16:16,*fp16:16",
        grid="(M+16-1)/16,(N+16-1)/16,1",
        num_warps=4,
        num_stages=2,
        out_name="matmul_fp16",
    )
    compile_kernel(compile_args)
