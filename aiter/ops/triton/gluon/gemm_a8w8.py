from typing import Optional
import functools
import json
import os
import triton

from aiter.ops.triton.utils.logger import AiterTritonLogger

from triton import language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

_LOGGER = AiterTritonLogger()


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
    c_ptr,
    bias_ptr,
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
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    NUM_WARPS: gl.constexpr,
    cache_modifier: gl.constexpr,
    cdna_version: gl.constexpr,
):
    pid = gl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)

    pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
    tall_xcds = GRID_MN % NUM_XCDS
    tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
    xcd = pid % NUM_XCDS
    local_pid = pid // NUM_XCDS
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
        order=[0, 1],
    )

    blocked_kn: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_kn, 16],
        threads_per_warp=[8, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[0, 1],
    )
    
    mfma_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=cdna_version,
        instr_shape=[16, 16],
        transposed=True,
        warps_per_cta=[1, NUM_WARPS],
    )
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[1, 0]
    )
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0, 1]
    )
    offs_ak = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(0, blocked_mk))
    offs_bk = gl.arange(0, BLOCK_SIZE_K, layout=gl.SliceLayout(1, blocked_kn))

    smem_a = gl.allocate_shared_memory(
        a_ptr.type.element_ty, [BLOCK_SIZE_M, BLOCK_SIZE_K], layout=shared_a
    )
    smem_b = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [BLOCK_SIZE_K, BLOCK_SIZE_N], layout=shared_b
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfma_layout, k_width=16
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfma_layout, k_width=16
    )

    offs_am = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, blocked_mk)
    )
    offs_bn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, blocked_kn)
    )

    offs_a = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    offs_b = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn

    offs_a_scale = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_b_scale = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    a_scale = gl.amd.cdna4.buffer_load(ptr=a_scale_ptr, offsets=offs_a_scale)
    b_scale = gl.amd.cdna4.buffer_load(ptr=b_scale_ptr, offsets=offs_b_scale)

    
    acc = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout
    )
    zeros = gl.zeros(
        (BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=gl.float32, layout=mfma_layout
    )
    
    for k in range(0, gl.cdiv(K, BLOCK_SIZE_K)):
        a = gl.amd.cdna4.buffer_load(ptr=a_ptr, offsets=offs_a)
        b = gl.amd.cdna4.buffer_load(ptr=b_ptr, offsets=offs_b)
        
        smem_a.store(a)
        smem_b.store(b)
        
        cur_a = smem_a.load(layout=dot_a_layout)
        cur_b = smem_b.load(layout=dot_b_layout)
        
        acc = gl.amd.cdna4.mfma(cur_a, cur_b, acc)
        
        offs_a += BLOCK_SIZE_K * stride_ak
        offs_b += BLOCK_SIZE_K * stride_bk
        
    acc = acc * a_scale[:, None] * b_scale[None, :]
        
    # ====== Epilogue ======
    if HAS_BIAS:
        offs_bias = pid_n * BLOCK_SIZE_N + gl.arange(
            0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
        )
        bias = gl.amd.cdna4.buffer_load(ptr=bias_ptr, offsets=offs_bias)
        acc = acc + bias[None, :]

    c = acc.to(c_ptr.type.element_ty)
    # Write back the block of the output matrix C with masks
    offs_cm = pid_m * BLOCK_SIZE_M + gl.arange(
        0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, mfma_layout)
    )
    offs_cn = pid_n * BLOCK_SIZE_N + gl.arange(
        0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, mfma_layout)
    )
    offs_c = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    
    gl.amd.cdna4.buffer_store(stored_value=c, ptr=c_ptr, offsets=offs_c)