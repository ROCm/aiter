from typing import Optional
import functools
import json
import os
import triton
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from triton import language as tl

_LOGGER = AiterTritonLogger()
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@triton.jit
def _gemm_a8w8_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    a_scale_ptr,
    b_scale_ptr,
    bias_ptr,
    c_ptr,
    # Matrix dimensions
    M,
    N,
    K,
    # The stride variables represent how much to increase the ptr by when
    # moving by 1 element in a particular dimension. E.g. `stride_am` is
    # how much to increase `a_ptr` by to get the element one row down
    # (A has M rows).
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # Meta-parameters
    HAS_BIAS: gl.constexpr,
    BLOCK_SIZE_M: gl    .constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8 function
    below

    Computes the 8 bit matmul C = A x B, applies a conversion scale and optionally adds a bias to
    the result.
    The conversion scale is received in the form of two 1D tensors that are multiplied to form a
    2D one before being applied.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N).
    - A_scale: First scale tensor with shape (M, 1).
    - B_scale: Second scale tensor with shape (1, N).
    - Bias: Bias tensor with shape (1, N).
    """

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    pid = remap_xcd(pid, GRID_MN)

    pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)

    threads_per_elem_mk: gl.constexpr = triton.cdiv(
        BLOCK_SIZE_M * BLOCK_SIZE_K // (NUM_WARPS * 64), 16
    )
    threads_per_elem_kn: gl.constexpr = triton.cdiv(
        BLOCK_SIZE_K * BLOCK_SIZE_N // (NUM_WARPS * 64), 16
    )
    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_mk, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, threads_per_elem_kn],
        threads_per_warp=[8, 8],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16],
        transposed=True,
        warps_per_cta=[NUM_WARPS // 2, 2],
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )
   
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    
    #create shared memories
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )

    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    # Create offsets for first block of A and B input matrices
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))
    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
    )
    offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
    )
    
    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    # buffer load for A
    if EVEN_K:
        a = gl.amd.cdna4.buffer_load(
            ptr=a_ptr,
            offsets=offs_a,
            mask=offs_am[:, None] < M,
        )
    else:
        a = gl.amd.cdna4.buffer_load(
            ptr=a_ptr,
            offsets=offs_a,
            mask=(offs_ak[None, :] < K)
            & (offs_am[:, None] < M),
        )


    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    # buffer load for B
    if EVEN_K:
        b = gl.amd.cdna4.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
            mask=offs_bn[None, :] < N,
        )
    else:
        b = gl.amd.cdna4.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
            mask=(offs_bk[:, None] < K)
            & (offs_bn[None, :] < N),
        )


    # store first block of A to shared memory
    smem_a.store(a)

    #accumulator
    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
    )
    zeros = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
    )

    
    #loop over K
    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K) - 1):
        
        #advance block pointers for A and B
        offs_a += BLOCK_SIZE_K * stride_ak
        offs_b += BLOCK_SIZE_K * stride_bk
        
        # buffer load next block of A
        if EVEN_K:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=offs_am[:, None] < M,
            )
        else:
            a = gl.amd.cdna4.buffer_load(
                ptr=a_ptr,
                offsets=offs_a,
                mask=(offs_ak[None, :] < K - (k + 1) * BLOCK_SIZE_K) 
                & (offs_am[:, None] < M),
            )
        
        # store curr block of B to shared memory
        smem_b.store(b)
        
        # load current block of A from shared memory
        cur_a = smem_a.load(layout=dot_a_layout)
        
        # buffer load next block of B
        if EVEN_K:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=offs_bn[None, :] < N,
            )
        else:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=(offs_bk[:, None] < K - (k + 1) * BLOCK_SIZE_K)
                & (offs_bn[None, :] < N),
            )
        
        #load current block of B from shared memory
        cur_b = smem_b.load(layout=dot_b_layout)

        mfma_out = gl.amd.cdna4.mfma(cur_a, cur_b, zeros)
        acc += mfma_out
        
        # store next block of A to shared memory
        smem_a.store(a)

    # ======= Epilogue ========
    #store last block of B to shared memory
    smem_b.store(b)

    #load last blocks of A and B from shared memory
    cur_a = smem_a.load(layout=dot_a_layout)
    cur_b = smem_b.load(layout=dot_b_layout)

    zeros = gl.amd.cdna4.mfma(cur_a, cur_b, zeros)
    acc += zeros

    
    #create offsets for scales
    offs_a_scale = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_b_scale = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )

    #load scales from global memory
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        mask=offs_a_scale < M,
    )
    b_scale = gl.amd.cdna4.buffer_load(
        ptr=b_scale_ptr,
        offsets=offs_b_scale,
        mask=offs_b_scale < N,
    )

    # apply scales to accumulator
    acc *= a_scale[:, None] * b_scale[None, :]

    # add bias
    if HAS_BIAS:
        bias = gl.amd.cdna4.buffer_load(
            ptr=bias_ptr,
            offsets=offs_b_scale,
            mask=offs_b_scale < N,
        )
        acc = acc.to(bias_ptr.type.element_ty) + bias[None, :]

    c = acc.to(c_ptr.type.element_ty)


    # # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    c_offs = (
        stride_cm * offs_cm[:, None]
        + stride_cn * offs_cn[None, :]
    )
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)

    gl.amd.cdna4.buffer_store(
        stored_value=c, ptr=c_ptr, offsets=c_offs, mask=c_mask
    )