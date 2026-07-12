# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
from aiter.ops.triton.utils.logger import AiterTritonLogger  # debug
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_LOGGER = AiterTritonLogger()

_GLUON_REPR_KEYS = [
    "GROUP_K",
    "GROUP_N",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "NUM_KSPLIT",
    "SPLITK_BLOCK_SIZE",
    "EVEN_K",
    "GRID_MN",
    "num_warps",
    "cache_modifier",
    "NUM_BUFFERS",
]

_gemm_a8w8_blockscale_bandwidth_bound_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_gfx1250_bandwidth_bound_kernel", _GLUON_REPR_KEYS
)

_gemm_a8w8_blockscale_compute_bound_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_gfx1250_compute_bound_kernel", _GLUON_REPR_KEYS
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit(repr=_gemm_a8w8_blockscale_bandwidth_bound_repr)
def _gemm_a8w8_blockscale_bandwidth_bound_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
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
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    num_warps: gl.constexpr,
    warp_bases: gl.constexpr,
    cache_modifier: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """
    Note: this is Triton jited function and not meant to be called directly. Call gemm_a8w8_blockscale function
    below

    Computes the 8 bit matmul C = A x B using the block-scale quantization approach, with block shape assumed to be the same as BLOCK_SIZE_N/K.

    Key parameters:
    - A: Matrix A with shape (M, K).
    - B: Matrix B with shape (K, N).
    - C: Matrix C with shape (M, N) or (NUM_KSPLIT, M, N) when split-K.
    - A_scale: Scale tensor for A with shape (M, *scale_k).
    - B_scale: Scale tensor for B with shape (*scale_k, **scale_n).

    *scale_k = (K + GROUP_K - 1) // GROUP_K
    **scale_n = (N + GROUP_N - 1) // GROUP_N
    """

    # program setup — split-K decomposition
    pid_unified = gl.program_id(axis=0)
    pid_k = pid_unified // GRID_MN
    pid = pid_unified % GRID_MN
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    # K-split range for this partition
    k_split_offset = pid_k * SPLITK_BLOCK_SIZE
    K_local = K - k_split_offset
    if NUM_KSPLIT > 1:
        K_local = SPLITK_BLOCK_SIZE

    # acc layout
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3, True, warp_bases, [], [16, 16, 128]
    )

    # TDM Shared Layouts
    tdm_shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_K], [1, 0]
    )
    tdm_shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_N, BLOCK_SIZE_K], [1, 0]
    )

    shared_a_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    shared_b_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )

    # scales shared mem and offsets -- offsets in wmma layout to match tdm
    smem_scale_a = gl.allocate_shared_memory(
        gl.float32, [BLOCK_SIZE_M], layout=shared_a_scale
    )

    smem_scale_b = gl.allocate_shared_memory(
        gl.float32, [BLOCK_SIZE_N], layout=shared_b_scale
    )

    offs_am = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout))
    ) % M
    offs_bn = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout))
    ) % N

    offs_a_scale = offs_am * stride_ascale_m

    offs_b_scale_n = offs_bn // GROUP_N
    offs_b_scale = offs_b_scale_n * stride_bscale_n
    # Fast path: when a single scale group spans the whole tile in both N and K,
    # every column shares one b_scale, so load it with a single scalar global
    # load (folded into the a-scale at the multiply) instead of a BLOCK_SIZE_N
    # vector load of identical values.
    SCALAR_B_SCALE: gl.constexpr = (GROUP_N >= BLOCK_SIZE_N) and (
        GROUP_K >= BLOCK_SIZE_K
    )
    b_scale_scalar_off = ((pid_n * BLOCK_SIZE_N) // GROUP_N) * stride_bscale_n

    # Offset scale pointers to this split-K partition
    k_scale_offset = k_split_offset // GROUP_K
    a_scale_ptr += k_scale_offset * stride_ascale_k
    b_scale_ptr += k_scale_offset * stride_bscale_k

    # tdm offsets
    off_am_tdm = pid_m * BLOCK_SIZE_M
    off_bm_tdm = pid_n * BLOCK_SIZE_N

    # TDM tensor descriptors — offset base pointers by k_split_offset
    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + k_split_offset * stride_ak,
        shape=(M, K_local),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        layout=tdm_shared_a,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + k_split_offset * stride_bk,
        shape=(N, K_local),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        layout=tdm_shared_b,
    )
    tdm_smem_a = gl.allocate_shared_memory(
        a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=tdm_shared_a
    )
    tdm_smem_b = gl.allocate_shared_memory(
        b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=tdm_shared_b
    )

    # loads/computes indexes/counters
    num_loads = 0
    num_computes = 0

    # acc setup
    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)
    zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)

    # ------------ Prologue ---------------

    # load scales
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        cache=cache_modifier,
    )
    if SCALAR_B_SCALE:
        b_scale = gl.load(
            b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
        )
    else:
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
        )

    # TDM prologue
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc,
            [off_am_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_a.index(num_loads % NUM_BUFFERS),
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc,
            [off_bm_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_b.index(num_loads % NUM_BUFFERS),
        )
        num_loads += 1

    # wait for the buffers to finish
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

    cur_a = tdm_smem_a.index(num_computes % NUM_BUFFERS).load(layout=dot_a_layout)
    cur_b = (
        tdm_smem_b.index(num_computes % NUM_BUFFERS)
        .permute((1, 0))
        .load(layout=dot_b_layout)
    )

    # store scales
    smem_scale_a.store(a_scale)
    if not SCALAR_B_SCALE:
        smem_scale_b.store(b_scale)
    # Scalar case carries b_scale in a register (folded into the a-scale below).
    cur_b_scale = b_scale

    # setup for loop — iterate over this partition's K tiles
    k_tiles_count = gl.cdiv(K_local, BLOCK_SIZE_K)

    # ----- Main Loop --------

    for k in range(k_tiles_count - (NUM_BUFFERS - 1)):
        # Loading a scale and curr A scale
        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
        if not SCALAR_B_SCALE:
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

        # wmma
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        if SCALAR_B_SCALE:
            acc += res * cur_a_scale[:, None] * cur_b_scale
        else:
            acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]
        # load into tdm
        gl.amd.gfx1250.tdm.async_load(
            a_desc,
            [off_am_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_a.index(num_loads % NUM_BUFFERS),
            pred=1,
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc,
            [off_bm_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_b.index(num_loads % NUM_BUFFERS),
            pred=1,
        )

        # wait for loads before proceeding
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        num_loads += 1

        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = (
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=dot_b_layout)
        )

        # scales -- ptrs, load from global
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )
        if SCALAR_B_SCALE:
            b_scale = gl.load(
                b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
            )
        else:
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
            )
        smem_scale_a.store(a_scale)
        if not SCALAR_B_SCALE:
            smem_scale_b.store(b_scale)
        cur_b_scale = b_scale

        cur_a = next_a
        cur_b = next_b
        num_computes += 1

    # ======= Epilogue ========

    # scale from last store
    cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
    if not SCALAR_B_SCALE:
        cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    for i in gl.static_range(NUM_BUFFERS - 2):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )
        if SCALAR_B_SCALE:
            b_scale = gl.load(
                b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
            )
        else:
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
            )

        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = (
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=dot_b_layout)
        )
        # wmma
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        if SCALAR_B_SCALE:
            acc += res * cur_a_scale[:, None] * cur_b_scale
        else:
            acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]
        cur_a = next_a
        cur_b = next_b
        num_computes += 1

        # scale store in smem and load for next iteration
        smem_scale_a.store(a_scale)
        if not SCALAR_B_SCALE:
            smem_scale_b.store(b_scale)
        cur_b_scale = b_scale

        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
        if not SCALAR_B_SCALE:
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    # wmma remaining tile
    res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
    if SCALAR_B_SCALE:
        acc += res * cur_a_scale[:, None] * cur_b_scale
    else:
        acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]

    # Store — offset c_ptr by pid_k * stride_ck for split-K
    tdm_shared_c: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )
    tdm_smem_c = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=tdm_shared_c,
    )
    tdm_smem_c.store(acc.to(c_ptr.type.element_ty))

    # wait for all wavefronts before write
    gl.barrier()

    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + pid_k * stride_ck,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=tdm_shared_c,
    )
    gl.amd.gfx1250.tdm.async_store(
        c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], tdm_smem_c
    )
    gl.amd.gfx1250.tdm.async_wait(0)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit(repr=_gemm_a8w8_blockscale_compute_bound_repr)
def _gemm_a8w8_blockscale_compute_bound_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
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
    stride_ck,
    stride_cm,
    stride_cn,
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    num_warps: gl.constexpr,
    warp_bases: gl.constexpr,
    cache_modifier: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """
    this is currently a copy of the bandwidth_bound kernel
    """

    # program setup — split-K decomposition
    pid_unified = gl.program_id(axis=0)
    pid_k = pid_unified // GRID_MN
    pid = pid_unified % GRID_MN
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    k_split_offset = pid_k * SPLITK_BLOCK_SIZE
    K_local = K - k_split_offset
    if NUM_KSPLIT > 1:
        K_local = SPLITK_BLOCK_SIZE

    # acc layout
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3, True, warp_bases, [], [16, 16, 128]
    )

    # TDM Shared Layouts
    tdm_shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_K], [1, 0]
    )
    tdm_shared_b: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_N, BLOCK_SIZE_K], [1, 0]
    )

    shared_a_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    shared_b_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )

    smem_scale_a = gl.allocate_shared_memory(
        gl.float32, [BLOCK_SIZE_M], layout=shared_a_scale
    )
    smem_scale_b = gl.allocate_shared_memory(
        gl.float32, [BLOCK_SIZE_N], layout=shared_b_scale
    )

    offs_am = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout))
    ) % M
    offs_bn = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout))
    ) % N

    offs_a_scale = offs_am * stride_ascale_m
    offs_b_scale_n = offs_bn // GROUP_N
    offs_b_scale = offs_b_scale_n * stride_bscale_n
    # Fast path: when a single scale group spans the whole tile in both N and K,
    # every column shares one b_scale, so load it with a single scalar global
    # load (folded into the a-scale at the multiply) instead of a BLOCK_SIZE_N
    # vector load of identical values.
    SCALAR_B_SCALE: gl.constexpr = (GROUP_N >= BLOCK_SIZE_N) and (
        GROUP_K >= BLOCK_SIZE_K
    )
    b_scale_scalar_off = ((pid_n * BLOCK_SIZE_N) // GROUP_N) * stride_bscale_n

    k_scale_offset = k_split_offset // GROUP_K
    a_scale_ptr += k_scale_offset * stride_ascale_k
    b_scale_ptr += k_scale_offset * stride_bscale_k

    off_am_tdm = pid_m * BLOCK_SIZE_M
    off_bm_tdm = pid_n * BLOCK_SIZE_N

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_ptr + k_split_offset * stride_ak,
        shape=(M, K_local),
        strides=(stride_am, stride_ak),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
        layout=tdm_shared_a,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + k_split_offset * stride_bk,
        shape=(N, K_local),
        strides=(stride_bn, stride_bk),
        block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K),
        layout=tdm_shared_b,
    )
    tdm_smem_a = gl.allocate_shared_memory(
        a_desc.dtype, shape=[NUM_BUFFERS] + a_desc.block_shape, layout=tdm_shared_a
    )
    tdm_smem_b = gl.allocate_shared_memory(
        b_desc.dtype, shape=[NUM_BUFFERS] + b_desc.block_shape, layout=tdm_shared_b
    )

    num_loads = 0
    num_computes = 0

    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)
    zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)

    # ------------ Prologue ---------------

    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        cache=cache_modifier,
    )
    if SCALAR_B_SCALE:
        b_scale = gl.load(
            b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
        )
    else:
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
        )

    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            a_desc,
            [off_am_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_a.index(num_loads % NUM_BUFFERS),
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc,
            [off_bm_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_b.index(num_loads % NUM_BUFFERS),
        )
        num_loads += 1

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

    cur_a = tdm_smem_a.index(num_computes % NUM_BUFFERS).load(layout=dot_a_layout)
    cur_b = (
        tdm_smem_b.index(num_computes % NUM_BUFFERS)
        .permute((1, 0))
        .load(layout=dot_b_layout)
    )

    smem_scale_a.store(a_scale)
    if not SCALAR_B_SCALE:
        smem_scale_b.store(b_scale)
    # Scalar case carries b_scale in a register (folded into the a-scale below).
    cur_b_scale = b_scale

    k_tiles_count = gl.cdiv(K_local, BLOCK_SIZE_K)

    # ----- Main Loop --------

    for k in range(k_tiles_count - (NUM_BUFFERS - 1)):
        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
        if not SCALAR_B_SCALE:
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        if SCALAR_B_SCALE:
            acc += res * cur_a_scale[:, None] * cur_b_scale
        else:
            acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]

        gl.amd.gfx1250.tdm.async_load(
            a_desc,
            [off_am_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_a.index(num_loads % NUM_BUFFERS),
            pred=1,
        )
        gl.amd.gfx1250.tdm.async_load(
            b_desc,
            [off_bm_tdm, num_loads * BLOCK_SIZE_K],
            tdm_smem_b.index(num_loads % NUM_BUFFERS),
            pred=1,
        )

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        num_loads += 1

        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = (
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=dot_b_layout)
        )

        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )
        if SCALAR_B_SCALE:
            b_scale = gl.load(
                b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
            )
        else:
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
            )
        smem_scale_a.store(a_scale)
        if not SCALAR_B_SCALE:
            smem_scale_b.store(b_scale)
        cur_b_scale = b_scale

        cur_a = next_a
        cur_b = next_b
        num_computes += 1

    # ======= Epilogue ========

    cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
    if not SCALAR_B_SCALE:
        cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    for i in gl.static_range(NUM_BUFFERS - 2):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )
        if SCALAR_B_SCALE:
            b_scale = gl.load(
                b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
            )
        else:
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
            )

        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = (
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS)
            .permute((1, 0))
            .load(layout=dot_b_layout)
        )

        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        if SCALAR_B_SCALE:
            acc += res * cur_a_scale[:, None] * cur_b_scale
        else:
            acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]
        cur_a = next_a
        cur_b = next_b
        num_computes += 1

        smem_scale_a.store(a_scale)
        if not SCALAR_B_SCALE:
            smem_scale_b.store(b_scale)
        cur_b_scale = b_scale

        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
        if not SCALAR_B_SCALE:
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    # wmma remaining tile
    res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
    if SCALAR_B_SCALE:
        acc += res * cur_a_scale[:, None] * cur_b_scale
    else:
        acc += res * cur_a_scale[:, None] * cur_b_scale[None, :]

    # Store — offset c_ptr by pid_k * stride_ck for split-K
    tdm_shared_c: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )
    tdm_smem_c = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=tdm_shared_c,
    )
    tdm_smem_c.store(acc.to(c_ptr.type.element_ty))

    gl.barrier()

    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + pid_k * stride_ck,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=tdm_shared_c,
    )
    gl.amd.gfx1250.tdm.async_store(
        c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], tdm_smem_c
    )
    gl.amd.gfx1250.tdm.async_wait(0)


_PRESHUFFLE_GLUON_REPR_KEYS = [
    "GROUP_K",
    "GROUP_N",
    "BLOCK_SIZE_M",
    "BLOCK_SIZE_N",
    "BLOCK_SIZE_K",
    "GROUP_SIZE_M",
    "NUM_KSPLIT",
    "SPLITK_BLOCK_SIZE",
    "EVEN_K",
    "GRID_MN",
    "num_warps",
    "cache_modifier",
    "NUM_BUFFERS",
    "N_CONST",
    "K_CONST",
]

_gemm_a8w8_blockscale_preshuffle_bandwidth_bound_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_preshuffle_gfx1250_bandwidth_bound_kernel",
    _PRESHUFFLE_GLUON_REPR_KEYS,
)


@gluon.jit
def depreshuffle_b(
    smem_b_raw,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
):
    """Unshuffle preshuffled weight tile in shared memory.

    Host shuffle: (N//16, 16, K//32, 2, 16) -> permute(0, 2, 3, 1, 4)
                  -> stored as (N//16, K*16)
    Inverse:      (N//16, K//32, 2, 16, 16)
                  -> permute(0, 3, 1, 2, 4)
                  -> (N, K) then transpose to (K, N)
    """
    return (
        smem_b_raw.reshape((BLOCK_SIZE_N // 16, BLOCK_SIZE_K // 32, 2, 16, 16))
        .permute((0, 3, 1, 2, 4))
        .reshape((BLOCK_SIZE_N, BLOCK_SIZE_K))
        .permute((1, 0))
    )


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit(repr=_gemm_a8w8_blockscale_preshuffle_bandwidth_bound_repr)
def _gemm_a8w8_blockscale_preshuffle_bandwidth_bound_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
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
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    num_warps: gl.constexpr,
    warp_bases: gl.constexpr,
    cache_modifier: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    N_CONST: gl.constexpr,
    K_CONST: gl.constexpr,
    MAYBE_LOOP_UNROLL: gl.constexpr,
):
    """
    Gluon gfx1250 kernel for a8w8 blockscale GEMM with preshuffled weights.

    Weight B is preshuffled on the host into shape (N//16, K*16).
    The kernel unshuffles in shared memory before the WMMA dot.

    A_scale strides are already adjusted by the wrapper for the
    transposed vs non-transposed case.
    """
    # Fast path: when a single scale group spans the whole tile in both N and K,
    # every column shares one b_scale, so load it with a single scalar global
    # load (folded into the a-scale at the multiply) instead of a BLOCK_SIZE_N
    # vector load of identical values.
    LDS_A_SCALE: gl.constexpr = False
    SCALAR_B_SCALE: gl.constexpr = (GROUP_N >= BLOCK_SIZE_N) and (
        GROUP_K >= BLOCK_SIZE_K
    )

    # Step the A/B TDM descriptors along K with update_tensor_descriptor
    # (add_offsets) and issue async_load at a fixed [0, 0], instead of passing
    # absolute [row, num_loads*BLOCK_K] offsets every iteration. The absolute
    # form forces the 8-SGPR tile descriptor to be rebuilt from VGPRs
    # (v_readfirstlane) each load; stepping keeps it in SGPRs (s_add), removing
    # the per-iter scalarization that dominates low-K shapes. The M/N block
    # offset is baked into the base pointer and the descriptor shape is shrunk
    # (M - off_am, K_local) so TDM boundary clamping stays correct.
    # Fully unroll the K loop using a constexpr trip count derived from
    # SPLITK_BLOCK_SIZE (like the Triton kernel), so per-iter counters/offsets
    # become compile-time -- removing the rolled-loop overhead and the runtime
    # descriptor address math (v_readfirstlane) that dominate low-K shapes.

    # program setup — split-K decomposition
    pid_unified = gl.program_id(axis=0)
    pid_k = pid_unified // GRID_MN
    pid = pid_unified % GRID_MN
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    k_split_offset = pid_k * SPLITK_BLOCK_SIZE
    K_local = K - k_split_offset
    if NUM_KSPLIT > 1:
        K_local = SPLITK_BLOCK_SIZE

    # Descriptor stepping (update_tensor_descriptor + fixed [0,0] async_load) vs
    # absolute [row, k*BK] offsets. Toggle for testing.
    USE_DESC_STEP: gl.constexpr = False

    if MAYBE_LOOP_UNROLL:
        LOOP_UNROLL: gl.constexpr = (
            (SPLITK_BLOCK_SIZE + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        ) < 32
    else:
        LOOP_UNROLL: gl.constexpr = False
    if LOOP_UNROLL:
        NUM_K_ITER: gl.constexpr = (
            SPLITK_BLOCK_SIZE + BLOCK_SIZE_K - 1
        ) // BLOCK_SIZE_K
    else:
        NUM_K_ITER = gl.cdiv(K_local, BLOCK_SIZE_K)

    # acc layout
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3, True, warp_bases, [], [16, 16, 128]
    )

    # Shared memory layouts
    tdm_shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_K], [1, 0]
    )
    tdm_shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )

    shared_a_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    shared_b_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )

    if LDS_A_SCALE:
        smem_scale_a = gl.allocate_shared_memory(
            gl.float32, [BLOCK_SIZE_M], layout=shared_a_scale
        )
    if not SCALAR_B_SCALE:
        smem_scale_b = gl.allocate_shared_memory(
            gl.float32, [BLOCK_SIZE_N], layout=shared_b_scale
        )

    offs_am = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout))
    ) % M
    offs_bn = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout))
    ) % N

    offs_a_scale = offs_am * stride_ascale_m
    offs_b_scale_n = offs_bn // GROUP_N
    offs_b_scale = offs_b_scale_n * stride_bscale_n

    b_scale_scalar_off = ((pid_n * BLOCK_SIZE_N) // GROUP_N) * stride_bscale_n

    # Offset scale pointers to this split-K partition
    k_scale_offset = k_split_offset // GROUP_K
    a_scale_ptr += k_scale_offset * stride_ascale_k
    b_scale_ptr += k_scale_offset * stride_bscale_k

    # TDM offsets
    off_am_tdm = pid_m * BLOCK_SIZE_M
    off_bn_tdm = pid_n * (BLOCK_SIZE_N // 16)

    # TDM tensor descriptors: bake the (M, N) block offset into the base pointer
    # and shrink the descriptor shape (M - off_am, K_local) so async_load uses a
    # fixed [0, 0] and steps along K via update_tensor_descriptor (the boundary
    # clamp stays correct, and the tile descriptor stays in SGPRs).
    if USE_DESC_STEP:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr + k_split_offset * stride_ak + off_am_tdm * stride_am,
            shape=(M - off_am_tdm, K_local),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            layout=tdm_shared_a,
        )
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr + k_split_offset * 16 * stride_bk + off_bn_tdm * stride_bn,
            shape=(gl.cdiv(N, 16) - off_bn_tdm, K_local * 16),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16),
            layout=tdm_shared_b,
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr + k_split_offset * stride_ak,
            shape=(M, K_local),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            layout=tdm_shared_a,
        )
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr + k_split_offset * 16 * stride_bk,
            shape=(gl.cdiv(N, 16), K_local * 16),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16),
            layout=tdm_shared_b,
        )

    tdm_smem_a = gl.allocate_shared_memory(
        a_desc.dtype,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_K],
        layout=tdm_shared_a,
    )
    tdm_smem_b = gl.allocate_shared_memory(
        b_desc.dtype,
        shape=[NUM_BUFFERS, BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16],
        layout=tdm_shared_b,
    )

    num_loads = 0
    num_computes = 0

    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)
    zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)

    # ------------ Prologue ---------------

    if SCALAR_B_SCALE:
        b_scale = gl.load(
            b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
        )
    else:
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
        )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        cache=cache_modifier,
    )

    for _ in gl.static_range(NUM_BUFFERS - 1):
        if USE_DESC_STEP:
            if not EVEN_K:
                # Ragged K: clamp each tile to its own remaining K extent
                # (add_offsets leaves the bound stale). Full tiles get
                # remaining >= BLOCK_SIZE_K (no clamp); the last, partial tile
                # gets clamped so TDM zero-fills past K.
                a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    a_desc,
                    set_bounds=[M - off_am_tdm, K_local - num_loads * BLOCK_SIZE_K],
                )
                b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    b_desc,
                    set_bounds=[
                        gl.cdiv(N, 16) - off_bn_tdm,
                        K_local * 16 - num_loads * BLOCK_SIZE_K * 16,
                    ],
                )
            gl.amd.gfx1250.tdm.async_load(
                a_desc, [0, 0], tdm_smem_a.index(num_loads % NUM_BUFFERS)
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc, [0, 0], tdm_smem_b.index(num_loads % NUM_BUFFERS)
            )
            # Advance to the next K tile.
            a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, BLOCK_SIZE_K]
            )
            b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, BLOCK_SIZE_K * 16]
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [off_am_tdm, num_loads * BLOCK_SIZE_K],
                tdm_smem_a.index(num_loads % NUM_BUFFERS),
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [off_bn_tdm, num_loads * BLOCK_SIZE_K * 16],
                tdm_smem_b.index(num_loads % NUM_BUFFERS),
            )
        num_loads += 1

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)

    cur_a = tdm_smem_a.index(num_computes % NUM_BUFFERS).load(layout=dot_a_layout)
    cur_b = depreshuffle_b(
        tdm_smem_b.index(num_computes % NUM_BUFFERS),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    ).load(layout=dot_b_layout)

    if not SCALAR_B_SCALE:
        smem_scale_b.store(b_scale)
    else:
        cur_b_scale = b_scale
    if LDS_A_SCALE:
        smem_scale_a.store(a_scale)
    else:
        cur_a_scale = a_scale

    # ----- Main Loop --------

    for k in (gl.static_range if LOOP_UNROLL else range)(
        NUM_K_ITER - (NUM_BUFFERS - 1)
    ):
        # Advance scales
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        if not SCALAR_B_SCALE:
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))
        if LDS_A_SCALE:
            cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))

        if SCALAR_B_SCALE:
            cur_ab_scale = cur_a_scale[:, None] * cur_b_scale
        else:
            cur_ab_scale = cur_a_scale[:, None] * cur_b_scale[None, :]

        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        acc += res * cur_ab_scale

        if SCALAR_B_SCALE:
            b_scale = gl.load(
                b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
            )
        else:
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
            )
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )

        # TDM load next tile
        if USE_DESC_STEP:
            if not EVEN_K:
                a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    a_desc,
                    set_bounds=[M - off_am_tdm, K_local - num_loads * BLOCK_SIZE_K],
                )
                b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    b_desc,
                    set_bounds=[
                        gl.cdiv(N, 16) - off_bn_tdm,
                        K_local * 16 - num_loads * BLOCK_SIZE_K * 16,
                    ],
                )
            gl.amd.gfx1250.tdm.async_load(
                a_desc, [0, 0], tdm_smem_a.index(num_loads % NUM_BUFFERS), pred=1
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc, [0, 0], tdm_smem_b.index(num_loads % NUM_BUFFERS), pred=1
            )
            # Advance to the next K tile.
            a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, BLOCK_SIZE_K]
            )
            b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, BLOCK_SIZE_K * 16]
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [off_am_tdm, num_loads * BLOCK_SIZE_K],
                tdm_smem_a.index(num_loads % NUM_BUFFERS),
                pred=1,
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [off_bn_tdm, num_loads * BLOCK_SIZE_K * 16],
                tdm_smem_b.index(num_loads % NUM_BUFFERS),
                pred=1,
            )

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2) * 2)
        num_loads += 1

        # Load next tile from shared (unshuffle B)
        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = depreshuffle_b(
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        ).load(layout=dot_b_layout)

        if not SCALAR_B_SCALE:
            smem_scale_b.store(b_scale)
        else:
            cur_b_scale = b_scale
        if LDS_A_SCALE:
            smem_scale_a.store(a_scale)
        else:
            cur_a_scale = a_scale

        cur_a = next_a
        cur_b = next_b
        num_computes += 1

    # ======= Epilogue ========

    for i in gl.static_range(NUM_BUFFERS - 2):
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        if not SCALAR_B_SCALE:
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))
        if LDS_A_SCALE:
            cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))

        if SCALAR_B_SCALE:
            cur_ab_scale = cur_a_scale[:, None] * cur_b_scale
        else:
            cur_ab_scale = cur_a_scale[:, None] * cur_b_scale[None, :]
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        acc += res * cur_ab_scale

        if SCALAR_B_SCALE:
            b_scale = gl.load(
                b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
            )
        else:
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
            )
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 3 - i) * 2)
        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = depreshuffle_b(
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        ).load(layout=dot_b_layout)

        cur_a = next_a
        cur_b = next_b
        num_computes += 1

        if not SCALAR_B_SCALE:
            smem_scale_b.store(b_scale)
        else:
            cur_b_scale = b_scale
        if LDS_A_SCALE:
            smem_scale_a.store(a_scale)
        else:
            cur_a_scale = a_scale

    # Final WMMA
    if LDS_A_SCALE:
        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
    if not SCALAR_B_SCALE:
        cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    if SCALAR_B_SCALE:
        cur_ab_scale = cur_a_scale[:, None] * cur_b_scale
    else:
        cur_ab_scale = cur_a_scale[:, None] * cur_b_scale[None, :]
    res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
    acc += res * cur_ab_scale

    # Store — offset c_ptr by pid_k * stride_ck for split-K
    tdm_shared_c: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )
    tdm_smem_c = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=tdm_shared_c,
    )
    tdm_smem_c.store(acc.to(c_ptr.type.element_ty))

    gl.barrier()

    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + pid_k * stride_ck,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=tdm_shared_c,
    )
    gl.amd.gfx1250.tdm.async_store(
        c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], tdm_smem_c
    )
    gl.amd.gfx1250.tdm.async_wait(0)


_gemm_a8w8_blockscale_preshuffle_compute_bound_repr = make_kernel_repr(
    "_gemm_a8w8_blockscale_preshuffle_gfx1250_compute_bound_kernel",
    _PRESHUFFLE_GLUON_REPR_KEYS,
)


@triton.heuristics(
    {
        "EVEN_K": lambda args: args["K"] % args["BLOCK_SIZE_K"] == 0,
        "GRID_MN": lambda args: triton.cdiv(args["M"], args["BLOCK_SIZE_M"])
        * triton.cdiv(args["N"], args["BLOCK_SIZE_N"]),
    }
)
@gluon.jit(repr=_gemm_a8w8_blockscale_preshuffle_compute_bound_repr)
def _gemm_a8w8_blockscale_preshuffle_compute_bound_kernel(
    # Pointers to matrices
    a_ptr,
    b_ptr,
    c_ptr,
    a_scale_ptr,
    b_scale_ptr,
    # Matrix dimensions
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
    stride_ascale_m,
    stride_ascale_k,
    stride_bscale_k,
    stride_bscale_n,
    # Meta-parameters
    GROUP_K: gl.constexpr,
    GROUP_N: gl.constexpr,
    BLOCK_SIZE_M: gl.constexpr,
    BLOCK_SIZE_N: gl.constexpr,
    BLOCK_SIZE_K: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    NUM_KSPLIT: gl.constexpr,
    SPLITK_BLOCK_SIZE: gl.constexpr,
    EVEN_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    num_warps: gl.constexpr,
    warp_bases: gl.constexpr,
    cache_modifier: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    N_CONST: gl.constexpr,
    K_CONST: gl.constexpr,
    MAYBE_LOOP_UNROLL: gl.constexpr,
):
    """
    Compute-bound variant with ds_read/wmma pipelining.

    Issues load_shared_relaxed (ds_read) for tile i+1 *before* the wmma
    for tile i so the LDS read latency is hidden behind the matrix op.
    """
    LDS_A_SCALE: gl.constexpr = False
    SCALAR_B_SCALE: gl.constexpr = (GROUP_N >= BLOCK_SIZE_N) and (
        GROUP_K >= BLOCK_SIZE_K
    )

    USE_DESC_STEP: gl.constexpr = False

    # program setup — split-K decomposition
    pid_unified = gl.program_id(axis=0)
    pid_k = pid_unified // GRID_MN
    pid = pid_unified % GRID_MN
    num_pid_m = gl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = gl.cdiv(N, BLOCK_SIZE_N)

    if NUM_KSPLIT == 1:
        pid_m, pid_n = pid_grid(pid, num_pid_m, num_pid_n, GROUP_SIZE_M=GROUP_SIZE_M)
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    k_split_offset = pid_k * SPLITK_BLOCK_SIZE
    K_local = K - k_split_offset
    if NUM_KSPLIT > 1:
        K_local = SPLITK_BLOCK_SIZE

    if MAYBE_LOOP_UNROLL:
        LOOP_UNROLL: gl.constexpr = (
            (SPLITK_BLOCK_SIZE + BLOCK_SIZE_K - 1) // BLOCK_SIZE_K
        ) < 32
    else:
        LOOP_UNROLL: gl.constexpr = False
    if LOOP_UNROLL:
        NUM_K_ITER: gl.constexpr = (
            SPLITK_BLOCK_SIZE + BLOCK_SIZE_K - 1
        ) // BLOCK_SIZE_K
    else:
        NUM_K_ITER = gl.cdiv(K_local, BLOCK_SIZE_K)

    # acc layout
    wmma_layout: gl.constexpr = gl.amd.AMDWMMALayout(
        3, True, warp_bases, [], [16, 16, 128]
    )

    # Shared memory layouts
    tdm_shared_a: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_K, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_K], [1, 0]
    )
    tdm_shared_b: gl.constexpr = gl.SwizzledSharedLayout(
        vec=1, per_phase=1, max_phase=1, order=[1, 0]
    )

    shared_a_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )
    shared_b_scale: gl.constexpr = gl.SwizzledSharedLayout(
        vec=16, per_phase=2, max_phase=8, order=[0]
    )

    dot_a_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=wmma_layout, k_width=8
    )
    dot_b_layout: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=wmma_layout, k_width=8
    )

    if LDS_A_SCALE:
        smem_scale_a = gl.allocate_shared_memory(
            gl.float32, [BLOCK_SIZE_M], layout=shared_a_scale
        )
    if not SCALAR_B_SCALE:
        smem_scale_b = gl.allocate_shared_memory(
            gl.float32, [BLOCK_SIZE_N], layout=shared_b_scale
        )

    offs_am = (
        pid_m * BLOCK_SIZE_M
        + gl.arange(0, BLOCK_SIZE_M, layout=gl.SliceLayout(1, wmma_layout))
    ) % M
    offs_bn = (
        pid_n * BLOCK_SIZE_N
        + gl.arange(0, BLOCK_SIZE_N, layout=gl.SliceLayout(0, wmma_layout))
    ) % N

    offs_a_scale = offs_am * stride_ascale_m
    offs_b_scale_n = offs_bn // GROUP_N
    offs_b_scale = offs_b_scale_n * stride_bscale_n

    b_scale_scalar_off = ((pid_n * BLOCK_SIZE_N) // GROUP_N) * stride_bscale_n

    k_scale_offset = k_split_offset // GROUP_K
    a_scale_ptr += k_scale_offset * stride_ascale_k
    b_scale_ptr += k_scale_offset * stride_bscale_k

    # TDM offsets
    off_am_tdm = pid_m * BLOCK_SIZE_M
    off_bn_tdm = pid_n * (BLOCK_SIZE_N // 16)

    if USE_DESC_STEP:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr + k_split_offset * stride_ak + off_am_tdm * stride_am,
            shape=(M - off_am_tdm, K_local),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            layout=tdm_shared_a,
        )
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr + k_split_offset * 16 * stride_bk + off_bn_tdm * stride_bn,
            shape=(gl.cdiv(N, 16) - off_bn_tdm, K_local * 16),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16),
            layout=tdm_shared_b,
        )
    else:
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a_ptr + k_split_offset * stride_ak,
            shape=(M, K_local),
            strides=(stride_am, stride_ak),
            block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_K),
            layout=tdm_shared_a,
        )
        b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=b_ptr + k_split_offset * 16 * stride_bk,
            shape=(gl.cdiv(N, 16), K_local * 16),
            strides=(stride_bn, stride_bk),
            block_shape=(BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16),
            layout=tdm_shared_b,
        )

    tdm_smem_a = gl.allocate_shared_memory(
        a_desc.dtype,
        shape=[NUM_BUFFERS, BLOCK_SIZE_M, BLOCK_SIZE_K],
        layout=tdm_shared_a,
    )
    tdm_smem_b = gl.allocate_shared_memory(
        b_desc.dtype,
        shape=[NUM_BUFFERS, BLOCK_SIZE_N // 16, BLOCK_SIZE_K * 16],
        layout=tdm_shared_b,
    )

    num_loads = 0
    num_computes = 0

    acc_dtype = gl.float32 if c_ptr.type.element_ty != gl.int8 else gl.int32
    acc = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)
    zeros = gl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=acc_dtype, layout=wmma_layout)

    # ------------ Prologue ---------------

    # Load first scales
    if SCALAR_B_SCALE:
        b_scale = gl.load(
            b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
        )
    else:
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
        )
    a_scale = gl.amd.cdna4.buffer_load(
        ptr=a_scale_ptr,
        offsets=offs_a_scale,
        cache=cache_modifier,
    )

    # TDM prologue: fill pipeline with NUM_BUFFERS tiles (one more than bw-bound)
    for _ in gl.static_range(NUM_BUFFERS):
        if USE_DESC_STEP:
            if not EVEN_K:
                a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    a_desc,
                    set_bounds=[M - off_am_tdm, K_local - num_loads * BLOCK_SIZE_K],
                )
                b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    b_desc,
                    set_bounds=[
                        gl.cdiv(N, 16) - off_bn_tdm,
                        K_local * 16 - num_loads * BLOCK_SIZE_K * 16,
                    ],
                )
            gl.amd.gfx1250.tdm.async_load(
                a_desc, [0, 0], tdm_smem_a.index(num_loads % NUM_BUFFERS)
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc, [0, 0], tdm_smem_b.index(num_loads % NUM_BUFFERS)
            )
            a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, BLOCK_SIZE_K]
            )
            b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, BLOCK_SIZE_K * 16]
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [off_am_tdm, num_loads * BLOCK_SIZE_K],
                tdm_smem_a.index(num_loads % NUM_BUFFERS),
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [off_bn_tdm, num_loads * BLOCK_SIZE_K * 16],
                tdm_smem_b.index(num_loads % NUM_BUFFERS),
            )
        num_loads += 1

    # Wait for tile 0 to land in LDS
    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

    # Pre-load tile 0 operands from LDS into registers (ds_read)
    cur_a = tdm_smem_a.index(num_computes % NUM_BUFFERS).load(layout=dot_a_layout)
    cur_b = depreshuffle_b(
        tdm_smem_b.index(num_computes % NUM_BUFFERS),
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    ).load(layout=dot_b_layout)

    # Stage first scales
    if not SCALAR_B_SCALE:
        smem_scale_b.store(b_scale)
    else:
        cur_b_scale = b_scale
    if LDS_A_SCALE:
        smem_scale_a.store(a_scale)
    else:
        cur_a_scale = a_scale

    # ----- Main Loop --------
    # Pipelining: ds_read(next) is issued BEFORE wmma(current), so the LDS
    # read latency is hidden behind the matrix multiply execution.

    for k in (gl.static_range if LOOP_UNROLL else range)(NUM_K_ITER - NUM_BUFFERS):
        # -- Compute scales for current tile --
        if not SCALAR_B_SCALE:
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))
        if LDS_A_SCALE:
            cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))

        if SCALAR_B_SCALE:
            cur_ab_scale = cur_a_scale[:, None] * cur_b_scale
        else:
            cur_ab_scale = cur_a_scale[:, None] * cur_b_scale[None, :]

        # -- WMMA for current tile (operands already in registers) --
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        acc += res * cur_ab_scale

        # -- Issue TDM load for tile (num_loads) --
        if USE_DESC_STEP:
            if not EVEN_K:
                a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    a_desc,
                    set_bounds=[M - off_am_tdm, K_local - num_loads * BLOCK_SIZE_K],
                )
                b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                    b_desc,
                    set_bounds=[
                        gl.cdiv(N, 16) - off_bn_tdm,
                        K_local * 16 - num_loads * BLOCK_SIZE_K * 16,
                    ],
                )
            gl.amd.gfx1250.tdm.async_load(
                a_desc, [0, 0], tdm_smem_a.index(num_loads % NUM_BUFFERS), pred=1
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc, [0, 0], tdm_smem_b.index(num_loads % NUM_BUFFERS), pred=1
            )
            a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                a_desc, add_offsets=[0, BLOCK_SIZE_K]
            )
            b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                b_desc, add_offsets=[0, BLOCK_SIZE_K * 16]
            )
        else:
            gl.amd.gfx1250.tdm.async_load(
                a_desc,
                [off_am_tdm, num_loads * BLOCK_SIZE_K],
                tdm_smem_a.index(num_loads % NUM_BUFFERS),
                pred=1,
            )
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [off_bn_tdm, num_loads * BLOCK_SIZE_K * 16],
                tdm_smem_b.index(num_loads % NUM_BUFFERS),
                pred=1,
            )

        # Wait for next compute tile to land
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        num_loads += 1

        # -- ds_read NEXT tile into registers (issued before next wmma) --
        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = depreshuffle_b(
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        ).load(layout=dot_b_layout)

        # -- Advance and load next scales --
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        if SCALAR_B_SCALE:
            b_scale = gl.load(
                b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
            )
        else:
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
            )
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )

        if not SCALAR_B_SCALE:
            smem_scale_b.store(b_scale)
        else:
            cur_b_scale = b_scale
        if LDS_A_SCALE:
            smem_scale_a.store(a_scale)
        else:
            cur_a_scale = a_scale

        cur_a = next_a
        cur_b = next_b
        num_computes += 1

    # ======= Epilogue: drain remaining NUM_BUFFERS tiles, no more TDM loads ========

    for i in gl.static_range(NUM_BUFFERS - 1):
        if not SCALAR_B_SCALE:
            cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))
        if LDS_A_SCALE:
            cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))

        if SCALAR_B_SCALE:
            cur_ab_scale = cur_a_scale[:, None] * cur_b_scale
        else:
            cur_ab_scale = cur_a_scale[:, None] * cur_b_scale[None, :]

        # Wait for next tile
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

        # ds_read NEXT tile (issued before wmma of current)
        next_a = tdm_smem_a.index((num_computes + 1) % NUM_BUFFERS).load(
            layout=dot_a_layout
        )
        next_b = depreshuffle_b(
            tdm_smem_b.index((num_computes + 1) % NUM_BUFFERS),
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
        ).load(layout=dot_b_layout)

        # WMMA for current tile
        res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
        acc += res * cur_ab_scale

        cur_a = next_a
        cur_b = next_b
        num_computes += 1

        # Advance scales for next iteration
        a_scale_ptr += stride_ascale_k
        b_scale_ptr += stride_bscale_k
        if SCALAR_B_SCALE:
            b_scale = gl.load(
                b_scale_ptr + b_scale_scalar_off, cache_modifier=cache_modifier
            )
        else:
            b_scale = gl.amd.cdna4.buffer_load(
                ptr=b_scale_ptr, offsets=offs_b_scale, cache=cache_modifier
            )
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=a_scale_ptr,
            offsets=offs_a_scale,
            cache=cache_modifier,
        )
        if not SCALAR_B_SCALE:
            smem_scale_b.store(b_scale)
        else:
            cur_b_scale = b_scale
        if LDS_A_SCALE:
            smem_scale_a.store(a_scale)
        else:
            cur_a_scale = a_scale

    # Final WMMA — last tile, operands already in registers
    if LDS_A_SCALE:
        cur_a_scale = smem_scale_a.load(layout=gl.SliceLayout(1, wmma_layout))
    if not SCALAR_B_SCALE:
        cur_b_scale = smem_scale_b.load(layout=gl.SliceLayout(0, wmma_layout))

    if SCALAR_B_SCALE:
        cur_ab_scale = cur_a_scale[:, None] * cur_b_scale
    else:
        cur_ab_scale = cur_a_scale[:, None] * cur_b_scale[None, :]
    res = gl.amd.gfx1250.wmma(cur_a, cur_b, zeros)
    acc += res * cur_ab_scale

    # Store — offset c_ptr by pid_k * stride_ck for split-K
    tdm_shared_c: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_SIZE_N, 8]], [BLOCK_SIZE_M, BLOCK_SIZE_N], [1, 0]
    )
    tdm_smem_c = gl.allocate_shared_memory(
        c_ptr.type.element_ty,
        shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        layout=tdm_shared_c,
    )
    tdm_smem_c.store(acc.to(c_ptr.type.element_ty))

    gl.barrier()

    c_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=c_ptr + pid_k * stride_ck,
        shape=(M, N),
        strides=(stride_cm, stride_cn),
        block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N),
        layout=tdm_shared_c,
    )
    gl.amd.gfx1250.tdm.async_store(
        c_desc, [pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N], tdm_smem_c
    )
    gl.amd.gfx1250.tdm.async_wait(0)


_KERNEL_MAP = {
    "bandwidth_bound": _gemm_a8w8_blockscale_bandwidth_bound_kernel,
    "compute_bound": _gemm_a8w8_blockscale_compute_bound_kernel,
}

_PRESHUFFLE_KERNEL_MAP = {
    "bandwidth_bound": _gemm_a8w8_blockscale_preshuffle_bandwidth_bound_kernel,
    "compute_bound": _gemm_a8w8_blockscale_preshuffle_compute_bound_kernel,
}
