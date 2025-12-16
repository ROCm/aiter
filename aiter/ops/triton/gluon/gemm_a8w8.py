from typing import Optional
import functools
import json
import os
import triton
import torch
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
import aiter.ops.triton.utils._triton.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
from aiter.ops.triton.utils.logger import AiterTritonLogger
from triton import language as tl
from aiter.ops.triton.utils.device_info import get_num_xcds
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
@gluon.jit
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
    FP8_FORMAT: gl.constexpr,
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

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS

    # Calculate new pid based on the new grouping
    # (for some reason triton compiler is smart enough to optimize the 
    # if xcd < tall_xcds condition during compile-time but gluon doesn't
    # perform the optimization so we manually do it) 
    if not GRID_MN % NUM_XCDS: 
        pid = xcd * pids_per_xcd + local_pid
    else:
        # Note that we need to consider the following two cases:
        # 1. the current pid is on a tall xcd
        # 2. the current pid is on a short xcd
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid
            )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[4, 16],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )
    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, 1],
        threads_per_warp=[16, 4],
        warps_per_cta=[1, NUM_WARPS],
        order=[0, 1],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[2, NUM_WARPS // 2],
    )
    
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=16, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=16, order=[0, 1]
    )
   
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    
    # Create offsets for first block of A and B input matrices
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
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

    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))
    offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
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

    #create shared memories
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )

    # store first block of A to shared memory
    smem_a.store(a)

    acc_dtype = gl.float32 if a_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
    )

    #num_stages:2
    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K) - 1):
        
        #advance block pointers for A and B
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * stride_bk
        
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

        # store curr block of B to shared mem    
        smem_b.store(b)
        
        # load curr block of A from shared mem
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

        if FP8_FORMAT is None: # in_dtype is int8
            acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)
        else:
            acc = gl.amd.cdna4.mfma_scaled(a=cur_a, a_scale=None, a_format=FP8_FORMAT, b=cur_b, b_scale=None, b_format=FP8_FORMAT, acc=acc)

        # store next block of A to shared memory
        smem_a.store(a)

    # ======= Epilogue ========
    
    #store last block of B to shared memory
    smem_b.store(b)
    
    #load last blocks of A and B from shared memory
    cur_a = smem_a.load(layout=dot_a_layout)
    cur_b = smem_b.load(layout=dot_b_layout)

    if FP8_FORMAT is None: # in_dtype is int8
        acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)
    else:
        acc = gl.amd.cdna4.mfma_scaled(a=cur_a, a_scale=None, a_format=FP8_FORMAT, b=cur_b, b_scale=None, b_format=FP8_FORMAT, acc=acc)

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

@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit
def _gemm_a8w8_preshuffled_kernel(
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
    stride_bn,
    stride_bk,
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
    FP8_FORMAT: gl.constexpr,
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

    ## pid remapping on xcds
    # Number of pids per XCD in the new arrangement
    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    # When GRID_MN cannot divide NUM_XCDS, some xcds will have
    # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
    # We calculate the number of xcds that have pids_per_xcd pids as
    # tall_xcds
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    # Compute current XCD and local pid within the XCD
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS

    # Calculate new pid based on the new grouping
    # (for some reason triton compiler is smart enough to optimize the 
    # if xcd < tall_xcds condition during compile-time but gluon doesn't
    # perform the optimization so we manually do it) 
    if not GRID_MN % NUM_XCDS: 
        pid = xcd * pids_per_xcd + local_pid
    else:
        # Note that we need to consider the following two cases:
        # 1. the current pid is on a tall xcd
        # 2. the current pid is on a short xcd
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid
            )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + (pid % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    blocked_mk: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[1, 16],
        threads_per_warp=[4, 16],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    linear_nk: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases = [[0, 1], [0, 2], [0, 4], [0, 8], [0, 1024], [0, 2048], [8, 0]],
        lane_bases = [[0, 16], [0, 32], [0, 64], [0, 128], [0, 256], [0, 512]],
        warp_bases= [[1, 0], [2, 0], [4, 0]],
        block_bases= [],
        shape= [BLOCK_SIZE_N//16, BLOCK_SIZE_K*16],
    )

    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 128],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=1, max_phase=16, order=[1, 0]
    )
   
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )
    
    # Create offsets for first block of A and B input matrices
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
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

    offs_bk = gl.arange(0, BLOCK_SIZE_K * 16, layout=gl.SliceLayout(0, linear_nk))
    offs_bn = pid_n * BLOCK_SIZE_N//16 + gl.arange(
        0, BLOCK_SIZE_N//16, layout=gl.SliceLayout(1, linear_nk)
    )
    offs_b = offs_bn[:, None] * stride_bn + offs_bk[None, :] * stride_bk
    # buffer load for B
    if EVEN_K:
        b = gl.amd.cdna4.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
            mask=offs_bn[:, None] < N,
        )
    else:
        b = gl.amd.cdna4.buffer_load(
            ptr=b_ptr,
            offsets=offs_b,
            mask=(offs_bk[None, :] < K)
            & (offs_bn[:, None] < N),
        )
    
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

    #create shared memories
    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )

    # store first block of A to shared memory
    smem_a.store(a)

    acc_dtype = gl.float32 if a_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=mfma_layout
    )

    cur_b = b

    #num_stages:2
    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K) - 1):
        
        #advance block pointers for A and B
        a_ptr += BLOCK_SIZE_K * stride_ak
        b_ptr += BLOCK_SIZE_K * 16 * stride_bk
        
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
        
        # load curr block of A from shared mem
        cur_a = smem_a.load(layout=dot_a_layout)
        
        # buffer load next block of B
        if EVEN_K:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=offs_bn[:, None] < N,
            )
        else:
            b = gl.amd.cdna4.buffer_load(
                ptr=b_ptr,
                offsets=offs_b,
                mask=(offs_bk[None, :] < K - (k + 1) * BLOCK_SIZE_K)
                & (offs_bn[:, None] < N),
            )

        cur_b = (
            cur_b.reshape(
                1,
                BLOCK_SIZE_N // 16,
                BLOCK_SIZE_K // 32,
                2,
                16,
                16,
            )
            .permute(0, 1, 4, 2, 3, 5)
            .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
            .trans(1, 0)
        )
        cur_b = gl.convert_layout(value=cur_b, layout=dot_b_layout, assert_trivial=True)
        
        if FP8_FORMAT is None: # in_dtype is int8
            acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)
        else:
            acc = gl.amd.cdna4.mfma_scaled(a=cur_a, a_scale=None, a_format=FP8_FORMAT, b=cur_b, b_scale=None, b_format=FP8_FORMAT, acc=acc)

        # store next block of A to shared memory
        smem_a.store(a)
        cur_b = b

    # ======= Epilogue ========
    
    #load last blocks of A and B from shared memory
    cur_a = smem_a.load(layout=dot_a_layout)
    cur_b = (
        cur_b.reshape(
            1,
            BLOCK_SIZE_N // 16,
            BLOCK_SIZE_K // 32,
            2,
            16,
            16,
        )
        .permute(0, 1, 4, 2, 3, 5)
        .reshape(BLOCK_SIZE_N, BLOCK_SIZE_K)
        .trans(1, 0)
    )
    cur_b = gl.convert_layout(value=cur_b, layout=dot_b_layout, assert_trivial=True)

    if FP8_FORMAT is None: # in_dtype is int8
        acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)
    else:
        acc = gl.amd.cdna4.mfma_scaled(a=cur_a, a_scale=None, a_format=FP8_FORMAT, b=cur_b, b_scale=None, b_format=FP8_FORMAT, acc=acc)

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

@functools.lru_cache(maxsize=1024)
def _get_config(
    M: int,
    N: int,
    K: int,
):
    if not hasattr(_get_config, "_config_dict"):
        dev = arch_info.get_device()
        fpath = f"{AITER_TRITON_CONFIGS_PATH}/gemm/{dev}-GEMM-A8W8.json"
        with open(fpath, "r") as file:
            config = json.load(file)
        _get_config._config_dict = config

    return _get_config._config_dict["any"]

def gemm_a8w8(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias.
    INT8 inputs are scaled back to higher precision using per-tensor scale factors.

    Args:
        x (torch.Tensor): INT8 input matrix with shape (M, K).
        w (torch.Tensor): INT8 weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Scale factor for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Scale factor for w with shape (1, N) or (N,).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).

    Returns:
        torch.Tensor: Output with shape (M, N) in higher precision format.
    """

    _LOGGER.info(
        f"GEMM_A8W8: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    # Check constraints.
    assert x.shape[1] == w.shape[1], "Incompatible dimensions!!!"
    assert x.dtype == w.dtype, "Input types must be the same"

    M, K = x.shape
    N, K = w.shape

    # Transpose w (kernel expects (K, N))
    w = w.T

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    if x.dtype == torch.float8_e4m3fn:
        fp8_format = "e4m3"
    elif x.dtype == torch.float8_e5m2:
        fp8_format = "e5m2"
    else:
        fp8_format = None # int8 case
    
    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gemm_a8w8_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        bias is not None,
        NUM_XCDS=get_num_xcds(),
        NUM_WARPS=config["num_warps"],
        **config,
        FP8_FORMAT=fp8_format
    )

    return y

def gemm_a8w8_preshuffle(
    x: torch.Tensor,
    w: torch.Tensor,
    x_scale: torch.Tensor,
    w_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    dtype: Optional[float] = torch.bfloat16,
    y: Optional[torch.Tensor] = None,
    config: Optional[dict] = None,
):
    """
    Computes 8 bit matrix multiplication Y = (X @ W^T) * (x_scale * w_scale) with optional bias.
    INT8 inputs are scaled back to higher precision using per-tensor scale factors.

    Args:
        x (torch.Tensor): INT8 input matrix with shape (M, K).
        w (torch.Tensor): INT8 weight matrix with shape (N, K), internally transposed.
        x_scale (torch.Tensor): Scale factor for x with shape (M, 1) or (M,).
        w_scale (torch.Tensor): Scale factor for w with shape (1, N) or (N,).
        bias (Optional[torch.Tensor]): Bias vector with shape (N,).
        dtype (Optional[torch.dtype]): Output datatype (BF16 or FP16).
        y (Optional[torch.Tensor]): Pre-allocated output tensor with shape (M, N).
        config (Optional[dict]): Kernel tuning parameters (BLOCK_SIZE_M, BLOCK_SIZE_N,
            BLOCK_SIZE_K, GROUP_SIZE_M).

    Returns:
        torch.Tensor: Output with shape (M, N) in higher precision format.
    """

    _LOGGER.info(
        f"GEMM_A8W8: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape)}"
    )

    M, K = x.shape
    N, K = w.shape
    N = N * 16
    K = K // 16

    if y is None:
        y = torch.empty((M, N), dtype=dtype, device=x.device)

    if config is None:
        config = _get_config(M, N, K)

    if x.dtype == torch.float8_e4m3fn:
        fp8_format = "e4m3"
    elif x.dtype == torch.float8_e5m2:
        fp8_format = "e5m2"
    else:
        fp8_format = None # int8 case
    
    grid = (
        triton.cdiv(M, config["BLOCK_SIZE_M"]) * triton.cdiv(N, config["BLOCK_SIZE_N"]),
    )
    _gemm_a8w8_preshuffled_kernel[grid](
        x,
        w,
        x_scale,
        w_scale,
        bias,
        y,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        w.stride(0),
        w.stride(1),
        y.stride(0),
        y.stride(1),
        bias is not None,
        NUM_XCDS=get_num_xcds(),
        NUM_WARPS=config["num_warps"],
        **config,
        FP8_FORMAT=fp8_format
    )

    return y