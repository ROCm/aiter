# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

# Y = cat(norm_0(A_0), norm_1(A_1)) @ W^T: an a16w16 GEMM with a Gemma RMSNorm
# prologue on A, for small M. Each program loads its whole A rows into
# registers, computes their rstd as the norm kernel does (bit for bit), and
# stores the normed bf16 rows into LDS. The K loop then reads A from LDS and
# streams only W. With K_SPLIT each warp takes one K group of every tile and
# the warps' sums are added at the end; without it every warp runs the whole
# K chain, in the order of the a16w16 GEMM (bit for bit).

import triton.experimental.gluon.language as gl
from triton.experimental import gluon

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_fused_rmsnorm_gemm_a16w16_repr = make_kernel_repr(
    "_fused_rmsnorm_gemm_a16w16_kernel",
    [
        "BLOCK_M",
        "BLOCK_N",
        "BLOCK_K",
        "NUM_BUFFERS",
        "NUM_SEG",
        "GATHER_0",
        "NORM_MODE",
        "NORM_0",
        "K_SPLIT",
    ],
)

# NORM_MODE: the norm kernel whose rstd the prologue reproduces bit for bit.
# 0: aiter's HIP fused_qk_rmsnorm_group_quant, 2048 wide: each of 128 lanes
#    sums the squares of 16 contiguous elements with FMA, then the 128 partials
#    meet in a balanced binary tree in lane order.
# 1: the Gluon _gluon_fused_rms_kernel, 2048 wide: gl.sum over its
#    [1, 8] x 32 lanes x 4 warps layout.


@gluon.jit
def _fma_chain(s_acc, x_acc, s_cur, x_cur):
    # Scan step: acc = fma(x, x, acc), in element order.
    return gl.fma(x_cur, x_cur, s_acc), x_cur


@gluon.jit
def _take_last(a, b):
    return b


@gluon.jit
def _add_first(a0, a1, b0, b1):
    return a0 + b0, a1


@gluon.jit
def _bitrev5(c):
    return ((c & 1) << 4) | ((c & 2) << 2) | (c & 4) | ((c & 8) >> 2) | ((c & 16) >> 4)


@gluon.jit
def _segment_layout(NORM_MODE: gl.constexpr, NUM_WARPS: gl.constexpr):
    # NORM_MODE 0: [rows, 128, 16], a lane holds 16 contiguous elements (one
    # of aiter's 128 lanes). NORM_MODE 1: [rows, SEG_K] in the layout of
    # _gluon_fused_rms_kernel.
    # Both: 4 warps along a row, the others split the rows.
    if NORM_MODE == 0:
        return gl.BlockedLayout(
            [1, 1, 16], [1, 32, 1], [NUM_WARPS // 4, 4, 1], [2, 1, 0]
        )
    else:
        return gl.BlockedLayout([1, 8], [1, 32], [NUM_WARPS // 4, 4], [1, 0])


@gluon.jit
def _segment_cols(SEG_K: gl.constexpr, layout: gl.constexpr, NORM_MODE: gl.constexpr):
    """The column of each element of a row in the segment layout. NORM_MODE 0:
    position c of axis 1 holds chunk perm(c), with the lane bits and the warp
    bits reversed, so that the cross-lane butterfly of the reduce (highest
    lane bit first) and its cross-warp step add neighbouring chunks first."""
    if NORM_MODE == 0:
        col_layout: gl.constexpr = gl.SliceLayout(0, layout)
        c = gl.arange(0, SEG_K // 16, layout=gl.SliceLayout(1, col_layout))
        hi = c >> 5
        c = _bitrev5(c & 31) | ((((hi & 1) << 1) | (hi >> 1)) << 5)
        e = gl.arange(0, 16, layout=gl.SliceLayout(0, col_layout))
        return c[:, None] * 16 + e[None, :]
    else:
        return gl.arange(0, SEG_K, layout=gl.SliceLayout(0, layout))


@gluon.jit
def _unpermute_chunks(y, R: gl.constexpr):
    # [R, 128, 16] with position c holding chunk perm(c) -> chunk order.
    y = gl.reshape(y, [R, 2, 2, 2, 2, 2, 2, 2, 16])
    y = gl.permute(y, [0, 2, 1, 7, 6, 5, 4, 3, 8])
    return gl.reshape(y, [R, 128, 16])


@gluon.jit
def _load_segment(
    x_ptr,
    stride_xm,
    rows,
    row_ok,
    nw_ptr,
    SEG_K: gl.constexpr,
    NORM_MODE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """Rows rows[i] of a [*, SEG_K] tensor (zeros where not row_ok), and the
    Gemma norm weight 1 + w, in the segment layout."""
    layout: gl.constexpr = _segment_layout(NORM_MODE, NUM_WARPS)
    cols = _segment_cols(SEG_K, layout, NORM_MODE)
    # Unmasked loads of clamped rows: a mask would branch row by row.
    rows = gl.where(row_ok, rows, 0)
    if NORM_MODE == 0:
        x = gl.load(x_ptr + rows[:, None, None] * stride_xm + cols[None, :, :])
        x = gl.where(row_ok[:, None, None], x, 0.0)
    else:
        x = gl.load(x_ptr + rows[:, None] * stride_xm + cols[None, :])
        x = gl.where(row_ok[:, None], x, 0.0)
    w = gl.load(nw_ptr + cols).to(gl.float32) + 1.0
    return x, w


@gluon.jit
def _norm_segment_to_lds(
    x,
    w,
    eps,
    a_buffer,
    SEG_K: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_K: gl.constexpr,
    K_GROUPS: gl.constexpr,
    NORM_MODE: gl.constexpr,
    NUM_WARPS: gl.constexpr,
):
    """bf16(x * rstd * (1 + w)) of the BLOCK_M rows x, stored into a_buffer
    ([SEG_K // BLOCK_K, K_GROUPS, BLOCK_M, BLOCK_K // K_GROUPS])."""
    layout: gl.constexpr = _segment_layout(NORM_MODE, NUM_WARPS)
    xf = x.to(gl.float32)
    if NORM_MODE == 0:
        gl.static_assert(SEG_K == 2048, "NORM_MODE 0 reproduces 128 lanes of 16")
        # Each lane's chain of 16 FMAs, then the tree over the 128 partials
        # of a row across the lanes and the warps. A reduce of two operands
        # is neither vectorized nor sent to the DPP path.
        s, _ = gl.associative_scan((xf * xf, xf), 2, _fma_chain)
        part = gl.reduce(s, 2, _take_last)
        total, _ = gl.reduce((part, part), 1, _add_first)
        rstd = gl.convert_layout(
            gl.rsqrt(total / SEG_K + eps),
            gl.SliceLayout(1, gl.SliceLayout(2, layout)),
        )
        y = ((xf * rstd[:, None, None]) * w[None, :, :]).to(x.dtype)
        y = _unpermute_chunks(y, BLOCK_M)
    else:
        gl.static_assert(SEG_K == 2048, "NORM_MODE 1 reproduces a 2048-wide row")
        rstd = gl.rsqrt(gl.sum(xf * xf, axis=1) / SEG_K + eps)
        y = ((xf * rstd[:, None]) * w[None, :]).to(x.dtype)
    SEG_TILES: gl.constexpr = SEG_K // BLOCK_K
    y = gl.reshape(y, [BLOCK_M, SEG_TILES, K_GROUPS, BLOCK_K // K_GROUPS])
    a_buffer.store(gl.permute(y, [1, 2, 0, 3]))


@gluon.jit(repr=_fused_rmsnorm_gemm_a16w16_repr)
def _fused_rmsnorm_gemm_a16w16_kernel(
    a0_ptr,
    a1_ptr,
    idx_ptr,
    nw0_ptr,
    nw1_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    A0_ROWS,
    stride_a0m,
    stride_a1m,
    stride_bn,
    stride_cm,
    eps0,
    eps1,
    SEG_K: gl.constexpr,
    NUM_SEG: gl.constexpr,
    GATHER_0: gl.constexpr,
    NORM_MODE: gl.constexpr,
    NORM_0: gl.constexpr,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    K_SPLIT: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
):
    """A row-major [M, K] as NUM_SEG segments of SEG_K columns: segment 0 from
    a0 (row idx[m] of a0 when GATHER_0), segment 1 from a1. B is the [N, K]
    weight. Segment s is normed with nw<s> and eps<s>, Gemma style:
    bf16(x * rstd * (1 + w)), then multiplied with fp32 accumulation.
    WMMA_LAYOUT is rank 3: its first axis is the K group of a warp."""
    NUM_WARPS: gl.constexpr = gl.num_warps()
    gl.static_assert(NUM_WARPS % 4 == 0, "4 warps along a row of the norm")
    # K groups: one a warp (the warps' sums are added at the end), or one in
    # all (every warp runs the whole K chain, as the a16w16 GEMM does).
    if K_SPLIT:
        K_GROUPS: gl.constexpr = NUM_WARPS
    else:
        K_GROUPS: gl.constexpr = 1
    BKG: gl.constexpr = BLOCK_K // K_GROUPS
    K: gl.constexpr = SEG_K * NUM_SEG
    K_TILES: gl.constexpr = K // BLOCK_K
    SEG_TILES: gl.constexpr = SEG_K // BLOCK_K
    gl.static_assert(SEG_K % BLOCK_K == 0, "BLOCK_K must divide SEG_K")
    gl.static_assert(BKG % 32 == 0, "a K group is whole WMMA steps")
    gl.static_assert(K_TILES >= NUM_BUFFERS, "NUM_BUFFERS must not exceed K tiles")
    OPERAND_A: gl.constexpr = gl.DotOperandLayout(0, WMMA_LAYOUT, 8)
    OPERAND_B: gl.constexpr = gl.DotOperandLayout(1, WMMA_LAYOUT, 8)

    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BLOCK_M)
    pid_m = pid % num_pid_m
    pid_n = pid // num_pid_m

    # B tiles by TDM, as [K_GROUPS, BLOCK_N, BKG] blocks of B viewed as
    # [K // BKG, N, BKG]. The first NUM_BUFFERS - 1 tiles fly during the
    # prologue.
    b_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BKG, 8]], [K_GROUPS, BLOCK_N, BKG], [2, 1, 0]
    )
    b_buffer = gl.allocate_shared_memory(
        b_ptr.type.element_ty, [NUM_BUFFERS, K_GROUPS, BLOCK_N, BKG], b_layout
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_ptr + pid_n * BLOCK_N * stride_bn,
        shape=(K // BKG, N - pid_n * BLOCK_N, BKG),
        strides=(BKG, stride_bn, 1),
        block_shape=(K_GROUPS, BLOCK_N, BKG),
        layout=b_layout,
    )
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(b_desc, [i * K_GROUPS, 0, 0], b_buffer.index(i))

    # A in LDS: one buffer a segment, [SEG_TILES, K_GROUPS, BLOCK_M, BKG].
    # The K loop reads it through a rank-3 view with the same padding.
    a_store_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BKG, 8]], [SEG_TILES, K_GROUPS, BLOCK_M, BKG], [3, 2, 1, 0]
    )
    a_tile_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
        [[BKG, 8]], [K_GROUPS, BLOCK_M, BKG], [2, 1, 0]
    )
    a0_buffer = gl.allocate_shared_memory(
        a0_ptr.type.element_ty, [SEG_TILES, K_GROUPS, BLOCK_M, BKG], a_store_layout
    )
    if NUM_SEG == 2:
        a1_buffer = gl.allocate_shared_memory(
            a0_ptr.type.element_ty,
            [SEG_TILES, K_GROUPS, BLOCK_M, BKG],
            a_store_layout,
        )

    # Prologue: both segments' rows and norm weights in flight at once, then
    # the normed rows into LDS. A row past M, or an id outside [0, A0_ROWS),
    # is a zero row.
    seg_layout: gl.constexpr = _segment_layout(NORM_MODE, NUM_WARPS)
    if NORM_MODE == 0:
        row_layout: gl.constexpr = gl.SliceLayout(1, gl.SliceLayout(2, seg_layout))
    else:
        row_layout: gl.constexpr = gl.SliceLayout(1, seg_layout)
    offs_m = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=row_layout)
    m_ok = offs_m < M
    if NUM_SEG == 2:
        x1, w1 = _load_segment(
            a1_ptr,
            stride_a1m,
            offs_m.to(gl.int64),
            m_ok,
            nw1_ptr,
            SEG_K,
            NORM_MODE,
            NUM_WARPS,
        )
    if NORM_0:
        if GATHER_0:
            rows0 = gl.load(idx_ptr + gl.minimum(offs_m, M - 1)).to(gl.int64)
            ok0 = m_ok & (rows0 >= 0) & (rows0 < A0_ROWS)
        else:
            rows0 = offs_m.to(gl.int64)
            ok0 = m_ok
        x0, w0 = _load_segment(
            a0_ptr, stride_a0m, rows0, ok0, nw0_ptr, SEG_K, NORM_MODE, NUM_WARPS
        )
        _norm_segment_to_lds(
            x0,
            w0,
            eps0,
            a0_buffer,
            SEG_K,
            BLOCK_M,
            BLOCK_K,
            K_GROUPS,
            NORM_MODE,
            NUM_WARPS,
        )
    else:
        # Segment 0 is used as is (rows of a table that is normed already):
        # its rows go to LDS by TDM gather, in [BLOCK_M, BKG] blocks. A row
        # past M or an id outside [0, A0_ROWS) reads as zeros (out of bounds).
        gl.static_assert(GATHER_0, "NORM_0 = False is for a gathered segment")
        IDX_LAYOUT: gl.constexpr = gl.SliceLayout(
            1, gl.BlockedLayout([BLOCK_M, 1], [1, 32], [1, NUM_WARPS], [1, 0])
        )
        offs_i = pid_m * BLOCK_M + gl.arange(0, BLOCK_M, layout=IDX_LAYOUT)
        row_idx = gl.load(idx_ptr + offs_i, mask=offs_i < M, other=A0_ROWS)
        g_layout: gl.constexpr = gl.PaddedSharedLayout.with_identity_for(
            [[BKG, 8]], [BLOCK_M, BKG], [1, 0]
        )
        a0_blocks = a0_buffer.reinterpret(
            shape=[SEG_TILES * K_GROUPS, BLOCK_M, BKG], layout=g_layout
        )
        g_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
            base=a0_ptr,
            shape=(A0_ROWS, SEG_K),
            strides=(stride_a0m, 1),
            block_shape=(BLOCK_M, BKG),
            layout=g_layout,
        )
        for i in gl.static_range(SEG_TILES * K_GROUPS):
            d = gl.amd.gfx1250.tdm.update_tensor_descriptor(
                g_desc, add_offsets=[0, i * BKG]
            )
            gl.amd.gfx1250.tdm.async_gather(d, row_idx, a0_blocks.index(i))
    if NUM_SEG == 2:
        _norm_segment_to_lds(
            x1,
            w1,
            eps1,
            a1_buffer,
            SEG_K,
            BLOCK_M,
            BLOCK_K,
            K_GROUPS,
            NORM_MODE,
            NUM_WARPS,
        )

    a0_tiles = a0_buffer.reinterpret(layout=a_tile_layout)
    if NUM_SEG == 2:
        a1_tiles = a1_buffer.reinterpret(layout=a_tile_layout)

    # The K loop, software pipelined: tile kt + 1 is read from LDS before the
    # WMMAs of tile kt. With K_SPLIT, warp g multiplies K group g of each tile.
    acc = gl.zeros((K_GROUPS, BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)
    if NORM_0:
        gl.amd.gfx1250.tdm.async_wait(NUM_BUFFERS - 2)
    else:
        gl.amd.gfx1250.tdm.async_wait(0)
    gl.barrier()
    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a0_tiles.index(0), OPERAND_A)
    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_buffer.index(0).permute([0, 2, 1]), OPERAND_B
    )
    for kt in gl.static_range(K_TILES):
        # B tile kt + NUM_BUFFERS - 1 goes to the buffer of tile kt - 1, read
        # in the previous iteration.
        if kt + NUM_BUFFERS - 1 < K_TILES:
            # Every warp is past its reads of that buffer.
            gl.barrier()
            gl.amd.gfx1250.tdm.async_load(
                b_desc,
                [(kt + NUM_BUFFERS - 1) * K_GROUPS, 0, 0],
                b_buffer.index((kt + NUM_BUFFERS - 1) % NUM_BUFFERS),
            )
        if kt + 1 < K_TILES:
            # B tile kt + 1 in LDS; the tiles issued after it may be in flight.
            gl.amd.gfx1250.tdm.async_wait(min(K_TILES, kt + NUM_BUFFERS) - kt - 2)
            if kt + 1 < SEG_TILES:
                a_tile = a0_tiles.index(kt + 1)
            else:
                a_tile = a1_tiles.index(kt + 1 - SEG_TILES)
            nxt_a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_tile, OPERAND_A)
            nxt_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_buffer.index((kt + 1) % NUM_BUFFERS).permute([0, 2, 1]), OPERAND_B
            )
        acc = gl.amd.gfx1250.wmma(cur_a, cur_b, acc)
        if kt + 1 < K_TILES:
            cur_a = nxt_a
            cur_b = nxt_b

    # The K groups' sums, across the warps.
    out = gl.sum(acc, axis=0)
    out_layout: gl.constexpr = gl.SliceLayout(0, WMMA_LAYOUT)
    offs_cm = pid_m * BLOCK_M + gl.arange(
        0, BLOCK_M, layout=gl.SliceLayout(1, out_layout)
    )
    offs_cn = pid_n * BLOCK_N + gl.arange(
        0, BLOCK_N, layout=gl.SliceLayout(0, out_layout)
    )
    c_base = (
        c_ptr
        + (pid_m * BLOCK_M).to(gl.int64) * stride_cm
        + (pid_n * BLOCK_N).to(gl.int64)
    )
    offs_c = stride_cm * (offs_cm[:, None] - pid_m * BLOCK_M) + (
        offs_cn[None, :] - pid_n * BLOCK_N
    )
    mask_c = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    gl.amd.gfx1250.buffer_store(
        out.to(c_ptr.type.element_ty), c_base, offs_c, mask=mask_c
    )
