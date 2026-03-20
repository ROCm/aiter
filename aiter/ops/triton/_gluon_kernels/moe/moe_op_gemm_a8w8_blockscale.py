# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
# adapted from triton_kernels package

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
from aiter.ops.triton.utils._triton.arch_info import get_arch
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
from aiter.ops.triton._triton_kernels.moe.quant_moe import _compute_static_fp8_quant


def matmul_launch_metadata(grid, kernel, args):
    ret = dict()
    M, N, K = None, args["N"], args["K"]
    Y, X, W = args["Y"], args["X"], args["W"]
    hist = args["ExptHist"]
    if hist is not None:
        n_rows = int(hist.float().mean())
        n_tokens = float(hist.sum())
        n_w_bytes = (W.numel() * W.element_size() // hist.numel()) * (hist > 0).sum()
    else:
        n_tokens = None
        n_w_bytes = W.numel() * W.element_size()

    def repr(s, x):
        return f"{s}={x}" if x is not None else f"E_{len(hist)}({s})={n_rows}"

    nbits = X.dtype.itemsize * 8
    ret["name"] = f"{kernel.name} [{repr('M', M)}, {repr('N', N)}, {repr('K', K)}]"
    gindx = args.get("GatherIndx", None)
    # sindx = args.get("WriteBackIndx", None)
    if gindx is not None:
        ret["name"] += "_layer1"
    else:
        ret["name"] += "_layer2"
    if args["B"] is not None:
        ret["name"] += "_bias"
    if args["APPLY_SWIGLU"]:
        ret["name"] += "_swiglu"
    if args["Quant_static_scale"] is not None:
        ret["name"] += "_quant"

    fM = n_tokens
    fK = K if K is not None else n_tokens
    ret[f"flops{nbits}"] = 2.0 * fM * N * fK

    gindx = args.get("GatherIndx", None)
    # sindx = args.get("WriteBackIndx", None)
    n_x_bytes = X.numel() * X.element_size()
    n_y_bytes = Y.numel() * Y.element_size()
    if hist is not None:
        assert n_tokens is not None
        n_expts_act = args["N_EXPTS_ACT"]

        if gindx is not None:
            # recreate inverse GatherIndx.
            dst = torch.full_like(gindx, -1)
            idx = torch.arange(len(gindx), device=gindx.device, dtype=torch.int32)
            mask = gindx != -1
            dst[gindx[mask]] = idx[mask]
            n_read_rows = (dst.view((-1, n_expts_act)) != -1).any(dim=1).sum()
        else:
            n_read_rows = n_tokens
        n_x_bytes = n_read_rows * X.shape[-1] * X.element_size()
        n_y_bytes = n_tokens * Y.shape[-1] * Y.element_size()
    ret["bytes"] = int(n_x_bytes + n_y_bytes + n_w_bytes)

    return ret


@gluon.jit
def unshuffle_b_to_kn(
    b,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
):
    """converts preshuffled (BLOCK_N//16, BLOCK_K*16) to logical (BLOCK_K, BLOCK_N)."""
    return (
        b.reshape((1, BLOCK_N // 16, BLOCK_K // 32, 2, 16, 16))
        .permute((0, 1, 4, 2, 3, 5))
        .reshape((BLOCK_N, BLOCK_K))
        .permute((1, 0))
    )


@gluon.jit
def xcd_swizzle(pid, domain_size, XCD_SWIZZLE: gl.constexpr):
    """
    Swizzle the program id based on integer XCD_SWIZZLE.
    This is useful for reording how blocks are ordered. A scheduler may, for example,
    assign sequential blocks 0, 1, 2, 3, ..., 8, 9, 10.. to its 8 hardware units 0, 1, 2, 3, ..., 0, 1, 2.
    This pattern may not be ideal for memory access, and it may be better to swizzle so the assignment
    becomes 0, 0, 0, 0, ..., 1, 1, 1, ... In the swizzled arrangement, sequential blocks are assigned to
    the same hardware unit.
    """
    # Number of pids per group in the new arrangement
    pids_per_group = domain_size // XCD_SWIZZLE
    extra_pid_groups = domain_size % XCD_SWIZZLE

    # Compute current current and local pid within the group
    group = pid % XCD_SWIZZLE
    local_pid = pid // XCD_SWIZZLE

    # Calculate new pid based on the new grouping
    new_pid = group * pids_per_group + min(group, extra_pid_groups) + local_pid
    return new_pid


@gluon.jit
def clip(x, limit, clip_lower: gl.constexpr):
    res = gl.minimum(x, limit)
    if clip_lower:
        res = gl.maximum(-limit, res)
    return res


@gluon.jit
def _swiglu(input, alpha, limit):
    gelu, linear = gl.split(gl.reshape(input, (input.shape[0], input.shape[1] // 2, 2)))
    gelu = gelu.to(gl.float32)
    if limit is not None:
        gelu = clip(gelu, limit, clip_lower=False)
    linear = linear.to(gl.float32)
    if limit is not None:
        linear = clip(linear, limit, clip_lower=True)
    s = gelu / (1 + gl.exp2(-1.44269504089 * alpha * gelu))
    return gl.fma(s, linear, s)  # (s * (linear + 1))


@gluon.jit
def _reduce_grouped(
    X,
    stride_xb: gl.uint64,
    stride_xm: gl.uint64,
    stride_xn,  #
    Out,
    stride_om: gl.uint64,
    stride_on,  # output tensor
    InIndx,
    B,
    N,  #
    # fused activation function
    APPLY_SWIGLU: gl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    K: gl.constexpr,
    BLOCK_N: gl.constexpr,
    EVEN_N: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    pid_t = gl.program_id(1)
    pid_n = gl.program_id(0)

    BLOCK_N_OUT: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    start = pid_t * K

    threads_per_elem_n: gl.constexpr = gl.cdiv(BLOCK_N // (NUM_WARPS * 32), 16)
    threads_per_elem_n_out: gl.constexpr = gl.cdiv(BLOCK_N_OUT // (NUM_WARPS * 32), 16)

    blocked_n: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_n, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    blocked_n_out: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_n_out, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[NUM_WARPS, 1],
        order=[1, 0],
    )

    # load indices into a tuple
    if InIndx is None:
        indxs = (pid_t,)
    else:
        indxs = ()
        for i in gl.static_range(0, K):
            indxs = indxs + (gl.load(InIndx + start + i),)

    # Setup offsets
    offs_n = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, blocked_n)
    )
    offs_n_out = pid_n * BLOCK_N_OUT + gl.arange(
        0, BLOCK_N_OUT, layout=gl.SliceLayout(0, blocked_n_out)
    )

    acc = gl.zeros(
        [BLOCK_N_OUT], dtype=gl.float32, layout=gl.SliceLayout(0, blocked_n_out)
    )
    x_n_mask = offs_n < N
    XPtrs = X + offs_n * stride_xn

    for i in gl.static_range(0, K):
        curr = gl.zeros(
            [BLOCK_N], dtype=gl.float32, layout=gl.SliceLayout(0, blocked_n)
        )

        # Iterate over split_k partial values
        for b in range(0, B):
            x_row_ptr = XPtrs + indxs[i] * stride_xm + b * stride_xb

            if EVEN_N:
                vals = gl.load(x_row_ptr)
            else:
                vals = gl.load(x_row_ptr, mask=x_n_mask, other=0.0)
            vals = vals.to(gl.float32)
            curr += vals

        # apply nonlinearity to split-k output
        if APPLY_SWIGLU:
            curr = _swiglu(curr[None, :], alpha, limit)
        curr = gl.reshape(curr, [curr.shape[-1]])
        # Convert curr to match acc's layout before adding
        curr = gl.convert_layout(curr, gl.SliceLayout(0, blocked_n_out))
        # update final accumulator
        acc += curr
    Nrem = N // ACTIVATION_REDUCTION_N

    # Write-back
    offs_out = pid_t * stride_om + offs_n_out * stride_on
    if EVEN_N:
        gl.store(Out + offs_out, acc)
    else:
        out_n_mask = offs_n_out < Nrem
        gl.store(Out + offs_out, acc, mask=out_n_mask)


@gluon.jit
def issue_async_tile_loads(
    a_desc,
    b_desc,
    GatherIndx,
    gathered_m,
    block_id,
    pid_n,
    k_offset_start,
    k,
    a_buffer,
    b_buffer,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
):
    """
    Load A, B, and scale tiles asynchronously into shared memory buffers.
    B is loaded in preshuffled layout into b_buffer.
    """
    buffer_idx = k % NUM_BUFFERS

    # Load A tile
    if GatherIndx is None:
        gl.amd.gfx1250.tdm.async_load(
            a_desc,
            [block_id * BLOCK_M, k_offset_start + k * BLOCK_K],
            a_buffer.index(buffer_idx),
        )
    else:
        col_offset = k_offset_start + k * BLOCK_K
        gl.amd.gfx1250.tdm.async_gather(
            a_desc, gathered_m, col_offset, a_buffer.index(buffer_idx)
        )

    # Load B tile (N//16, K*16)
    gl.amd.gfx1250.tdm.async_load(
        b_desc,
        [
            pid_n * (BLOCK_N // 16),
            (k_offset_start + k * BLOCK_K) * 16,
        ],
        b_buffer.index(buffer_idx),
    )


@gluon.jit
def precompute_scale_offsets(
    stride_x_bs_m,
    WBlockScale,
    stride_w_bs_e,
    stride_w_bs_n,
    gathered_m,
    expt_id,
    pid_n,
    N,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCKSCALE_M: gl.constexpr,
    BLOCKSCALE_N: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    PER_ROW_X_SCALE: gl.constexpr,
    is_x_blockscale: gl.constexpr,
    is_w_blockscale: gl.constexpr,
):
    if is_x_blockscale:
        if PER_ROW_X_SCALE:
            a_scale_m_offs = gathered_m * stride_x_bs_m
        else:
            a_scale_m_offs = (gathered_m // BLOCKSCALE_M) * stride_x_bs_m
    else:
        a_scale_m_offs = gl.zeros(
            (BLOCK_M,), dtype=gl.int32, layout=gl.SliceLayout(1, WMMA_LAYOUT)
        )

    if is_w_blockscale:
        b_scale_base = WBlockScale + expt_id * stride_w_bs_e
        offs_w_n = pid_n * BLOCK_N + gl.arange(
            0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        offs_w_n = gl.max_contiguous(
            gl.multiple_of(offs_w_n % N, BLOCK_N),
            BLOCK_N,
        )
        b_scale_n_offs = (offs_w_n // BLOCKSCALE_N) * stride_w_bs_n
    else:
        b_scale_base = WBlockScale
        b_scale_n_offs = gl.zeros(
            (BLOCK_N,), dtype=gl.int32, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )

    return a_scale_m_offs, b_scale_base, b_scale_n_offs


@gluon.jit
def buffer_load_scales(
    offs_k_scale,
    XBlockScale,
    stride_x_bs_k,
    b_scale_base,
    stride_w_bs_k,
    a_scale_m_offs,
    b_scale_n_offs,
    is_x_blockscale: gl.constexpr,
    is_w_blockscale: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
):
    if is_x_blockscale:
        a_scale = gl.amd.cdna4.buffer_load(
            ptr=XBlockScale,
            offsets=a_scale_m_offs + offs_k_scale * stride_x_bs_k,
            cache="",
        )
    else:
        a_scale = gl.full(
            (BLOCK_M,), 1.0, dtype=gl.float32, layout=gl.SliceLayout(1, WMMA_LAYOUT)
        )
    if is_w_blockscale:
        b_scale = gl.amd.cdna4.buffer_load(
            ptr=b_scale_base,
            offsets=offs_k_scale * stride_w_bs_k + b_scale_n_offs,
            cache="",
        )
    else:
        b_scale = gl.full(
            (BLOCK_N,), 1.0, dtype=gl.float32, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
    return a_scale, b_scale


@gluon.jit
def consume_scaled_tile(
    a_buffer,
    b_buffer,
    m,
    XBlockScale,
    stride_x_bs_k,
    stride_w_bs_k,
    a_scale_m_offs,
    b_scale_base,
    b_scale_n_offs,
    k_offset_start,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    BLOCKSCALE_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    DOT_A_LAYOUT: gl.constexpr,
    DOT_B_LAYOUT: gl.constexpr,
    is_x_blockscale: gl.constexpr,
    is_w_blockscale: gl.constexpr,
):
    N_SUBTILES: gl.constexpr = gl.cdiv(BLOCK_K, BLOCKSCALE_K)
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    a_tile = a_buffer.index(m % NUM_BUFFERS)
    b_shuffled_tile = b_buffer.index(m % NUM_BUFFERS)

    offs_k_scale = (k_offset_start // BLOCKSCALE_K) + m * N_SUBTILES
    cur_a_scale, cur_b_scale = buffer_load_scales(
        offs_k_scale,
        XBlockScale, stride_x_bs_k, b_scale_base, stride_w_bs_k,
        a_scale_m_offs, b_scale_n_offs,
        is_x_blockscale, is_w_blockscale,
        BLOCK_M, BLOCK_N, WMMA_LAYOUT,
    )

    for sub in gl.static_range(N_SUBTILES):
        if sub < N_SUBTILES - 1:
            offs_k_scale += sub + 1
            next_a_scale, next_b_scale = buffer_load_scales(
                offs_k_scale,
                XBlockScale, stride_x_bs_k, b_scale_base, stride_w_bs_k,
                a_scale_m_offs, b_scale_n_offs,
                is_x_blockscale, is_w_blockscale,
                BLOCK_M, BLOCK_N, WMMA_LAYOUT,
            )

        sub_a_smem = a_tile.slice(sub * BLOCKSCALE_K, BLOCKSCALE_K, dim=1)
        sub_a = sub_a_smem.load(layout=DOT_A_LAYOUT)

        b_tile = unshuffle_b_to_kn(b_shuffled_tile, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K)
        sub_b_smem = b_tile.slice(sub * BLOCKSCALE_K, BLOCKSCALE_K, dim=0)
        sub_b = sub_b_smem.load(layout=DOT_B_LAYOUT)

        zeros = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
        partial = gl.amd.gfx1250.wmma(sub_a, sub_b, zeros)
        acc += partial * cur_a_scale[:, None] * cur_b_scale[None, :]

        if sub < N_SUBTILES - 1:
            cur_a_scale = next_a_scale
            cur_b_scale = next_b_scale

    return acc


@gluon.jit
def create_descriptor(
    X,
    stride_x_m,
    stride_x_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    GatherIndx,
    start_m,
    block_id,
    pid_n,
    expt_id,
    M,
    N,
    K,
    grid_m,
    index_type,
    # Constexprs
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    SPLIT_K: gl.constexpr,
    N_EXPTS_ACT: gl.constexpr,
    BLOCKED_MK: gl.constexpr,
    BLOCKED_KN: gl.constexpr,
    SHARED_A: gl.constexpr,
    SHARED_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
):
    """
    Create tensor descriptors for X, W, and their scales.
    W is in preshuffled layout (N//16, K*16).

    Returns:
        a_desc: Tensor descriptor or AsyncCopyDescriptor for X
        b_desc: Tensor descriptor for W
        gathered_m_idx: Row indices in blocked layout for async_gather
        gathered_m_scale: Row indices in WMMA layout for scale buffer_load

    Notes:
        For this kernel implementation, BLOCKSCALE_K must equal BLOCK_K.
    """
    # A descriptor
    in_m = grid_m * BLOCK_M
    if GatherIndx is None:
        # blocked layout for async_gather
        gathered_m_idx = start_m + BLOCK_M * block_id + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, BLOCKED_MK)
        )
        # wmma layout for buffer_load
        gathered_m_scale = start_m + BLOCK_M * block_id + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
        )
        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X + start_m * stride_x_m,
            shape=(in_m, K),
            strides=(stride_x_m, stride_x_k),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_A,
        )
    else:
        GatherIndx += start_m
        num_warps: gl.constexpr = gl.num_warps()
        IDX_BASE_LAYOUT: gl.constexpr = gl.BlockedLayout(
            size_per_thread=[BLOCK_M, 1],
            threads_per_warp=[1, 32],
            warps_per_cta=[1, num_warps],
            order=[1, 0],
        )
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(1, IDX_BASE_LAYOUT)
        idx_offs = BLOCK_M * block_id + gl.arange(0, BLOCK_M, layout=IDX_LAYOUT)
        gathered_m_idx = gl.load(GatherIndx + idx_offs) // N_EXPTS_ACT

        wmma_idx_offs = BLOCK_M * block_id + gl.arange(
            0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
        )
        gathered_m_scale = gl.amd.cdna4.buffer_load(
            ptr=GatherIndx,
            offsets=wmma_idx_offs,
            cache=".cg",
        ) // N_EXPTS_ACT

        a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=X,
            shape=(in_m, K),
            strides=(stride_x_m, stride_x_k),
            block_shape=(BLOCK_M, BLOCK_K),
            layout=SHARED_A,
        )

    # B descriptor
    offs_w_n = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, BLOCKED_KN)
    )
    offs_w_n = gl.max_contiguous(
        gl.multiple_of(offs_w_n % N, BLOCK_N),
        BLOCK_N,
    )
    W += expt_id * stride_w_e

    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=W,
        shape=(N // 16, K * 16),
        strides=(stride_w_n, stride_w_k),
        block_shape=(BLOCK_N // 16, BLOCK_K * 16),
        layout=SHARED_B,
    )

    return a_desc, b_desc, gathered_m_idx, gathered_m_scale


@gluon.jit(launch_metadata=matmul_launch_metadata)
def _moe_gemm_a8w8_blockscale(
    Y,
    stride_y_k,
    stride_y_m,
    stride_y_n,
    X,
    stride_x_m,
    stride_x_k,
    XBlockScale,  # [M, K_blocks] or [M_blocks, K_blocks]
    stride_x_bs_m,
    stride_x_bs_k,
    W,
    stride_w_e,
    stride_w_k,
    stride_w_n,
    WBlockScale,  # [K_blocks, N_blocks]
    stride_w_bs_e,
    stride_w_bs_k,
    stride_w_bs_n,
    X_static_scale,
    W_static_scale,
    Quant_static_scale,
    B,
    stride_b_e,  # Bias
    Gammas,
    N,
    K,  # shapes
    # expt data
    GatherIndx,
    ExptHist,
    ExptOffs,
    ExptOffsSum,
    ExptData,
    # true grid size
    grid_m,
    grid_n,
    # fused activation function
    APPLY_SWIGLU: gl.constexpr,
    alpha,
    limit,
    ACTIVATION_REDUCTION_N: gl.constexpr,
    # MoE config
    N_EXPTS_ACT: gl.constexpr,
    # optimization config
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GROUP_M: gl.constexpr,
    BLOCKSCALE_M: gl.constexpr,
    BLOCKSCALE_N: gl.constexpr,
    BLOCKSCALE_K: gl.constexpr,
    XCD_SWIZZLE: gl.constexpr,
    EVEN_K: gl.constexpr,
    MASK_K_LIMIT: gl.constexpr,
    SPLIT_K: gl.constexpr,
    W_CACHE_MODIFIER: gl.constexpr,
    UPCAST_INDICES: gl.constexpr = False,
    # Use per-row or 2D blockscale on X
    PER_ROW_X_SCALE: gl.constexpr = False,
    # Number of buffers to use for async_load
    NUM_BUFFERS: gl.constexpr = 2,
):
    """
    Computes the 8 bit matmul C = A x B using the block-scale quantization approach.

    Key parameters:
    - X: Matrix X with shape (M, K).
    - W: Matrix E with shape (E, K, N).
    - Y: Matrix C with shape (E, M, N).
    - x_scale: Scale tensor for A with shape (M // blockscale_m, K // blockscale_k) or (M, K // blockscale_k)
    - w_scale: Scale tensor for B with shape (K // blockscale_k, N // blockscale_n)
    - PER_ROW_X_SCALE: Determines whether we use per-row or 2D blockscale on X
    - NUM_BUFFERS: Determines the number of buffers to use for async_load
    """
    is_x_blockscale: gl.constexpr = XBlockScale is not None
    is_w_blockscale: gl.constexpr = WBlockScale is not None

    OUT_BLOCK_N: gl.constexpr = BLOCK_N // ACTIVATION_REDUCTION_N
    yN = N // ACTIVATION_REDUCTION_N

    pid = gl.program_id(0)
    if ExptOffsSum is not None and XCD_SWIZZLE > 1:
        # Determine how much padding there is on the expert data. This allows us to
        # know the true grid size and avoid processing padding tiles.
        padding_m = grid_m - gl.load(ExptOffsSum)
    else:
        padding_m: gl.constexpr = 0

    index_type: gl.constexpr = gl.int64 if UPCAST_INDICES else gl.int32

    unpadded_m = grid_m - padding_m
    gl.assume(unpadded_m >= 0)
    total_actual_tiles = unpadded_m * grid_n * SPLIT_K
    if padding_m > 0 and pid >= total_actual_tiles:
        return

    pid_emnk = pid
    if XCD_SWIZZLE != 1:
        pid_emnk = xcd_swizzle(pid_emnk, total_actual_tiles, XCD_SWIZZLE)
    # pid_e = pid_emnk // (unpadded_m * grid_n * SPLIT_K)
    pid_mnk = pid_emnk % (unpadded_m * grid_n * SPLIT_K)
    pid_k = pid_mnk % SPLIT_K
    pid_mn = pid_mnk // SPLIT_K
    pid_m, pid_n = pid_grid(pid_mn, unpadded_m, grid_n, GROUP_M)
    # For split-k, advance to the output k slice
    if SPLIT_K > 1:
        Y += pid_k.to(index_type) * stride_y_k
    # unpack expert data
    expt_data = gl.load(ExptData + pid_m)
    if expt_data == -1:
        return
    expt_id = expt_data & 0x0000FFFF
    block_id = expt_data >> 16
    M = gl.load(ExptHist + expt_id)
    start_m = gl.load(ExptOffs + expt_id)
    expt_id, block_id = expt_id.to(index_type), block_id.to(index_type)
    start_m = start_m.to(index_type)
    pid_n, pid_k = pid_n.to(index_type), pid_k.to(index_type)

    # Create layouts
    num_warps: gl.constexpr = gl.num_warps()
    threads_per_elem_mk: gl.constexpr = gl.cdiv(BLOCK_M * BLOCK_K // (num_warps * 32), 16)
    threads_per_elem_kn: gl.constexpr = gl.cdiv(BLOCK_K * BLOCK_N // (num_warps * 32), 16)
    BLOCKED_MK: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[threads_per_elem_mk, 16],
        threads_per_warp=[4, 8],
        warps_per_cta=[num_warps, 1],
        order=[1, 0],
    )
    BLOCKED_KN: gl.constexpr = gl.BlockedLayout(
        size_per_thread=[16, threads_per_elem_kn],
        threads_per_warp=[4, 8],
        warps_per_cta=[1, num_warps],
        order=[0, 1],
    )
    if num_warps == 2:
        warp_bases: gl.constexpr = [[0, 1]]
    elif num_warps == 4:
        warp_bases: gl.constexpr = [[0, 1], [1, 0]]
    else:
        warp_bases: gl.constexpr = [[0, 1], [0, 2], [1, 0]]
    WMMA_LAYOUT: gl.constexpr = gl.amd.AMDWMMALayout(
        version=3, transposed=True, warp_bases=warp_bases, instr_shape=[16, 16, 128]
    )
    SHARED_A: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 4]], [BLOCK_M, BLOCK_K], [1, 0]
    )
    SHARED_B: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BLOCK_K, 4]], [BLOCK_N // 16, BLOCK_K * 16], [1, 0]
    )
    DOT_A_LAYOUT: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=WMMA_LAYOUT, k_width=16
    )
    DOT_B_LAYOUT: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=WMMA_LAYOUT, k_width=16
    )

    # Create tensor descriptors
    a_desc, b_desc, gathered_m_idx, gathered_m_scale = create_descriptor(
        X,
        stride_x_m,
        stride_x_k,
        W,
        stride_w_e,
        stride_w_k,
        stride_w_n,
        GatherIndx,
        start_m,
        block_id,
        pid_n,
        expt_id,
        M,
        N,
        K,
        grid_m,
        index_type,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        SPLIT_K,
        N_EXPTS_ACT,
        BLOCKED_MK,
        BLOCKED_KN,
        SHARED_A,
        SHARED_B,
        WMMA_LAYOUT,
    )

    # Allocate shared memory buffers
    a_buffer = gl.allocate_shared_memory(
        a_desc.dtype,
        [NUM_BUFFERS] + a_desc.block_shape,
        layout=a_desc.layout,
    )
    b_buffer = gl.allocate_shared_memory(
        b_desc.dtype,
        [NUM_BUFFERS] + b_desc.block_shape,
        layout=b_desc.layout,
    )

    k = 0
    m = 0
    acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    splitk_block_size = gl.cdiv(K, SPLIT_K)
    num_k_tiles = gl.cdiv(splitk_block_size, BLOCK_K)
    k_offset_start = pid_k * splitk_block_size

    # Precompute scale offsets that are constant across K iterations
    a_scale_m_offs, b_scale_base, b_scale_n_offs = precompute_scale_offsets(
        stride_x_bs_m,
        WBlockScale,
        stride_w_bs_e,
        stride_w_bs_n,
        gathered_m_scale,
        expt_id,
        pid_n,
        N,
        BLOCK_M,
        BLOCK_N,
        BLOCKSCALE_M,
        BLOCKSCALE_N,
        WMMA_LAYOUT,
        PER_ROW_X_SCALE,
        is_x_blockscale,
        is_w_blockscale,
    )

    N_SUBTILES: gl.constexpr = gl.cdiv(BLOCK_K, BLOCKSCALE_K)

    # prologue
    for _ in gl.static_range(NUM_BUFFERS - 1):
        issue_async_tile_loads(
            a_desc,
            b_desc,
            GatherIndx,
            gathered_m_idx,
            block_id,
            pid_n,
            k_offset_start,
            k,
            a_buffer,
            b_buffer,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_BUFFERS,
        )
        k += 1

    # cur_a_scale, cur_b_scale = prefetch_scales(
    #     (k_offset_start // BLOCKSCALE_K) + m * N_SUBTILES,
    #     XBlockScale, stride_x_bs_k, b_scale_base, stride_w_bs_k,
    #     a_scale_m_offs, b_scale_n_offs,
    #     is_x_blockscale, is_w_blockscale,
    #     BLOCK_M, BLOCK_N, WMMA_LAYOUT,
    # )

    # Main loop
    for _ in range(num_k_tiles - (NUM_BUFFERS - 1)):
        issue_async_tile_loads(
            a_desc,
            b_desc,
            GatherIndx,
            gathered_m_idx,
            block_id,
            pid_n,
            k_offset_start,
            k,
            a_buffer,
            b_buffer,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            NUM_BUFFERS,
        )
        k += 1

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)
        acc += consume_scaled_tile(
            a_buffer,
            b_buffer,
            m,
            XBlockScale,
            stride_x_bs_k,
            stride_w_bs_k,
            a_scale_m_offs,
            b_scale_base,
            b_scale_n_offs,
            k_offset_start,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCKSCALE_K,
            NUM_BUFFERS,
            WMMA_LAYOUT,
            DOT_A_LAYOUT,
            DOT_B_LAYOUT,
            is_x_blockscale,
            is_w_blockscale,
        )
        m += 1

        # cur_a_scale, cur_b_scale = prefetch_scales(
        #     (k_offset_start // BLOCKSCALE_K) + m * N_SUBTILES,
        #     XBlockScale, stride_x_bs_k, b_scale_base, stride_w_bs_k,
        #     a_scale_m_offs, b_scale_n_offs,
        #     is_x_blockscale, is_w_blockscale,
        #     BLOCK_M, BLOCK_N, WMMA_LAYOUT,
        # )

    # Epilogue
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)
        acc += consume_scaled_tile(
            a_buffer,
            b_buffer,
            m,
            XBlockScale,
            stride_x_bs_k,
            stride_w_bs_k,
            a_scale_m_offs,
            b_scale_base,
            b_scale_n_offs,
            k_offset_start,
            BLOCK_M,
            BLOCK_N,
            BLOCK_K,
            BLOCKSCALE_K,
            NUM_BUFFERS,
            WMMA_LAYOUT,
            DOT_A_LAYOUT,
            DOT_B_LAYOUT,
            is_x_blockscale,
            is_w_blockscale,
        )
        m += 1

    offs_m = block_id * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, WMMA_LAYOUT)
    )
    offs_n = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
    )
    offs_cm = start_m + offs_m
    offs_cn = offs_n
    mask_m = offs_m < M
    mask_n = offs_cn < N

    # scalar fp8 scale
    if X_static_scale is not None:
        acc = acc * gl.load(X_static_scale)
    if W_static_scale is not None:
        acc = acc * gl.load(W_static_scale)

    # bias
    if B is not None:
        offs_bias = expt_id * stride_b_e + offs_cn
        if pid_k == 0:
            bias = gl.load(B + offs_bias, mask=mask_n, cache_modifier=W_CACHE_MODIFIER)
        else:
            bias = gl.full(
                [BLOCK_N], 0, dtype=gl.float32, layout=gl.SliceLayout(0, WMMA_LAYOUT)
            )
        acc = acc + bias[None, :]

    if APPLY_SWIGLU and SPLIT_K == 1:
        out = _swiglu(acc, alpha, limit)
        gl.static_assert(
            out.shape[1] == OUT_BLOCK_N,
            f"Activation fn out.shape[1] ({out.shape[1]}) doesn't match computed OUT_BLOCK_N ({OUT_BLOCK_N})",
        )
        out = gl.convert_layout(out, WMMA_LAYOUT)
        offs_y_n = OUT_BLOCK_N * pid_n + gl.arange(
            0, OUT_BLOCK_N, layout=gl.SliceLayout(0, WMMA_LAYOUT)
        )
        offs_cn = offs_y_n
        mask_n = offs_y_n < yN
    else:
        gl.static_assert(
            ACTIVATION_REDUCTION_N == 1,
            "Activation reduction must be 1 if no activation fn is provided",
        )
        out = acc

    if Gammas is not None:
        offs_gammas = start_m + offs_m
        gammas = gl.load(Gammas + offs_gammas, mask=mask_m)
        out *= gammas[:, None]
    # quant
    if Quant_static_scale is not None:
        out = _compute_static_fp8_quant(out, gl.load(Quant_static_scale))
    else:
        out = out.to(gl.bfloat16)
    # write-back
    offs_c = stride_y_m * offs_cm[:, None] + stride_y_n * offs_cn[None, :]
    mask_c = mask_m[:, None] & mask_n[None, :]
    gl.amd.gfx1250.buffer_store(out, Y, offs_c, mask=mask_c)
