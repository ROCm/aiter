# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon BF16/FP16 GEMM for gfx950, compute-bound variant.

Ported from ROCm/gfx950-gluon-tutorials (ef48b5f),
kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop: one wave per SIMD (4 warps), a
256x256x64 tile split into four 128x128 quadrants, a 2-deep async-copy LDS pipeline
with local prefetch, the K loop unrolled by 2, MFMA accumulators pinned to AGPRs with
``cd_regclass="a"``, and an XCD-aware PID remap. Its instruction schedule comes from
the llirSched plugin (``aiter/ops/triton/llir_sched``); without the plugin the kernel
still compiles and runs, just slower. The tutorial explains every step:
https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/intra_wave/a16w16

The tile is fixed and nothing is masked, so the kernel needs a TN problem (x
row-major, w row-major (N, K)) with M and N multiples of 256 and K a multiple of 128
of at least 256. ``unsupported_reason`` says why a problem is outside that.
"""

import inspect

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton import llir_sched

BLOCK_M, BLOCK_N, BLOCK_K = 256, 256, 64
NUM_WARPS = 4
NUM_XCDS = 8
GROUP_SIZE_M = 4


@gluon.jit
def _get_pids(
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """XCD-aware PID remapping + GROUP_SIZE_M swizzle. Active at any grid_mn."""
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
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
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def _gemm_a16w16_compute_bound_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
):
    """
    Beyond the hot loop: L2 locality + interleaved epilogue on top of v8_sliceMN.

    Builds on v8's slice-both-M-and-N design (4 quadrant accumulators, unrolled by 2,
    pre-computed _next offsets) and adds:

    1. XCD-aware PID remapping + GROUP_SIZE_M workgroup swizzling for L2 cache locality.
    2. Interleaved epilogue using extract_slice: each 128x128 accumulator is split into
       two 64x128 sub-tiles along M. Stores for sub-tile i are pipelined with the MFMA
       computing sub-tile i+1, spreading write traffic over time.
    """

    pid_m, pid_n = _get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # Half-M global load layout
    gLoadLayoutA: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[0, 1], [0, 2], [0, 4], [4, 0], [8, 0]],
        lane_bases=[[0, 8], [0, 16], [0, 32], [16, 0], [32, 0], [64, 0]],
        warp_bases=[[1, 0], [2, 0]],
        block_bases=[],
        shape=[BLOCK_M // 2, BLOCK_K],
    )
    # Half-N global load layout
    gLoadLayoutB: gl.constexpr = gl.DistributedLinearLayout(
        reg_bases=[[1, 0], [2, 0], [4, 0], [0, 4], [0, 8]],
        lane_bases=[[8, 0], [16, 0], [32, 0], [0, 16], [0, 32], [0, 64]],
        warp_bases=[[0, 1], [0, 2]],
        block_bases=[],
        shape=[BLOCK_K, BLOCK_N // 2],
    )

    # Half-M padded shared layout
    sharedLayoutA: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
            [0, 16],
            [0, 32],
            [16, 0],
            [32, 0],
            [64, 0],
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
        ],
        [],
        [BLOCK_M // 2, BLOCK_K],
    )
    # Half-N padded shared layout
    sharedLayoutB: gl.constexpr = gl.PaddedSharedLayout(
        [[512, 16]],
        [
            [1, 0],
            [2, 0],
            [4, 0],
            [8, 0],
            [16, 0],
            [32, 0],
            [0, 16],
            [0, 32],
            [0, 64],
            [0, 1],
            [0, 2],
            [0, 4],
            [0, 8],
        ],
        [],
        [BLOCK_K, BLOCK_N // 2],
    )

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BLOCK_M // 2, BLOCK_K], sharedLayoutA
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, BLOCK_N // 2], sharedLayoutB
    )

    offs_am = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))

    offs_bn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))

    a_base = a_ptr + pid_m * BLOCK_M * stride_am
    b_base = b_ptr + pid_n * BLOCK_N * stride_bn

    # Two sets of offsets: base (even K-steps) and _next (odd K-steps).
    a_top_offsets = offs_am[:, None] * stride_am + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_top_offsets + BLOCK_M * stride_am // 2
    b_left_offsets = offs_bk[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    b_right_offsets = b_left_offsets + BLOCK_N * stride_bn // 2

    a_top_offsets_next = a_top_offsets + BLOCK_K * stride_ak
    a_bot_offsets_next = a_bot_offsets + BLOCK_K * stride_ak
    b_left_offsets_next = b_left_offsets + BLOCK_K * stride_bk
    b_right_offsets_next = b_right_offsets + BLOCK_K * stride_bk

    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )

    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )

    acc_tl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_bl = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_tr = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)
    acc_br = gl.zeros((BLOCK_M // 2, BLOCK_N // 2), gl.float32, mfmaLayout)

    iterMax = gl.cdiv(K, BLOCK_K)

    ## Prologue — same as v8
    g_idx = 0
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base, a_bot_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base, b_right_offsets
    )
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_left.index(g_idx), b_base, b_left_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_top.index(g_idx), a_base, a_top_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemA_bot.index(g_idx), a_base, a_bot_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        smemB_right.index(g_idx), b_base, b_right_offsets_next
    )
    gl.amd.cdna4.async_copy.commit_group()

    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = smemB_left.index(0).load(dotOpLayoutB)
    a_top = smemA_top.index(0).load(dotOpLayoutA)

    gl.assume(iterMax > 3)

    ## Main loop — same as v8
    for k in range(0, iterMax - 2, 2):

        ## =============================================================
        ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(0), b_base, b_left_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(0), a_base, a_top_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(0), a_base, a_bot_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(0), b_base, b_right_offsets
        )
        gl.amd.cdna4.async_copy.commit_group()

        ## =============================================================
        ## Loop unroll: Sub-iteration 1: consume buffer 1, prefetch
        ## into buffer 1. AC uses _next offsets (odd K-step).
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = smemA_bot.index(1).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_left.index(1), b_base, b_left_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = smemB_right.index(1).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_top.index(1), a_base, a_top_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = smemB_left.index(0).load(dotOpLayoutB)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemA_bot.index(1), a_base, a_bot_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = smemA_top.index(0).load(dotOpLayoutA)

        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            smemB_right.index(1), b_base, b_right_offsets_next
        )
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    ## Epilogue: 4-quadrant stores with natural-pipeline ordering (matches v8).
    ## v9's contribution lives in the prologue / pid remapping (§3); the epilogue
    ## is unchanged from v8 because the sub-tile variant produced only ~200 cycles
    ## of additional savings — within noise relative to the full kernel.

    gStoreLayoutC: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])

    offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
    offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
    c_base = c_ptr + pid_m * BLOCK_M * stride_cm + pid_n * BLOCK_N * stride_cn
    c_tl_offsets = stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
    c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
    c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2

    if ADD_BIAS:
        # bias[N] is added in fp32 before the downcast; loaded here so its latency
        # overlaps the last MFMAs.
        offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, mfmaLayout))
        bias_l = gl.load(bias_ptr + pid_n * BLOCK_N + offs_bias).to(gl.float32)
        bias_r = gl.load(bias_ptr + pid_n * BLOCK_N + BLOCK_N // 2 + offs_bias).to(
            gl.float32
        )

    ## Iter iterMax - 2: same 4-region pattern as main loop, no AC
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = smemA_bot.index(l_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = smemB_right.index(l_idx).load(dotOpLayoutB)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = smemB_left.index(g_idx).load(dotOpLayoutB)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = smemA_top.index(g_idx).load(dotOpLayoutA)

    ## Iter iterMax - 1
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving.
    acc_tl = gl.amd.cdna3.mfma(a_top, b_left, acc_tl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = smemA_bot.index(g_idx).load(dotOpLayoutA)

    acc_bl = gl.amd.cdna3.mfma(a_bot, b_left, acc_bl, cd_regclass="a")
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = smemB_right.index(g_idx).load(dotOpLayoutB)

    if ADD_BIAS:
        acc_tl = acc_tl + bias_l[None, :]
    c_tl = acc_tl.to(c_ptr.dtype.element_ty)
    c_tl = gl.convert_layout(c_tl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tl_offsets, stored_value=c_tl)

    acc_tr = gl.amd.cdna3.mfma(a_top, b_right, acc_tr, cd_regclass="a")

    if ADD_BIAS:
        acc_bl = acc_bl + bias_l[None, :]
    c_bl = acc_bl.to(c_ptr.dtype.element_ty)
    c_bl = gl.convert_layout(c_bl, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_bl_offsets, stored_value=c_bl)

    acc_br = gl.amd.cdna3.mfma(a_bot, b_right, acc_br, cd_regclass="a")

    if ADD_BIAS:
        acc_tr = acc_tr + bias_r[None, :]
    c_tr = acc_tr.to(c_ptr.dtype.element_ty)
    c_tr = gl.convert_layout(c_tr, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_tr_offsets, stored_value=c_tr)

    if ADD_BIAS:
        acc_br = acc_br + bias_r[None, :]
    c_br = acc_br.to(c_ptr.dtype.element_ty)
    c_br = gl.convert_layout(c_br, layout=gStoreLayoutC)
    gl.amd.cdna3.buffer_store(ptr=c_base, offsets=c_br_offsets, stored_value=c_br)


# The kernel pins its accumulators with mfma(..., cd_regclass="a")
# (triton-lang/triton#11792); older Triton releases cannot compile it.
_TRITON_HAS_CD_REGCLASS = (
    "cd_regclass" in inspect.signature(gl.amd.cdna3.mfma).parameters
)


def unsupported_reason(M, N, K, x=None, w=None, bias=None, activation=None):
    """Why this kernel cannot run the problem, or None if it can."""
    if not _TRITON_HAS_CD_REGCLASS:
        return (
            f"Triton {triton.__version__} has no mfma cd_regclass "
            "(needs triton-lang/triton#11792)"
        )
    if activation:
        return "no activation support yet"
    if M % BLOCK_M or N % BLOCK_N:
        return f"M and N must be multiples of {BLOCK_M} (got M={M}, N={N})"
    if K % (2 * BLOCK_K) or K < 4 * BLOCK_K:
        return f"K must be a multiple of {2 * BLOCK_K} and at least {4 * BLOCK_K} (got K={K})"
    if x is not None and x.stride(1) != 1:
        return "x must be row-major (M, K)"
    if w is not None and w.stride(1) != 1:
        return "w must be row-major (N, K)"
    if x is not None and x.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {x.dtype} is not fp16/bf16"
    return None


def gemm_a16w16_compute_bound(x, w, y, bias=None):
    """y = x @ w.T (+ bias) with x (M, K) and w (N, K), both row-major; y (M, N)."""
    M, K = x.shape
    N, _ = w.shape
    reason = unsupported_reason(M, N, K, x, w)
    if reason is not None:
        raise ValueError(f"gfx950 gluon compute_bound a16w16: {reason}")
    b = w.T  # (K, N) view, K contiguous: the layout the kernel loads
    grid_mn = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    _gemm_a16w16_compute_bound_kernel[(grid_mn, 1)](
        x,
        b,
        y,
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        GRID_MN=grid_mn,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ADD_BIAS=bias is not None,
        num_warps=NUM_WARPS,
        **llir_sched.compile_options(),
    )
    return y


_KERNEL_MAP = {"compute_bound": gemm_a16w16_compute_bound}
