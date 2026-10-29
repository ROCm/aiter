# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The bf16 GEMM of gemm_a16w16 + all-reduce + residual add + RMSNorm over 4 GPUs in one
kernel. y = x W^T (bf16 x and W) of a rank is a partial sum: the norm needs the sum of
the 4. A GEMM program posts its tile of y to the inbox of each rank in the units of the
fused all-reduce, so y never goes to memory, and a receiver adds and normalizes 3 rows
as the fused all-reduce does."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton._gluon_kernels.gfx1250.comms.all_reduce import (
    _unit_words,
    _values,
)
from aiter.ops.triton._gluon_kernels.gfx1250.comms.fused.all_reduce_rmsnorm_quant import (
    _columns,
)
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _gemm_post(
    x,
    w,
    inboxes,
    calls,
    tokens,
    blocks,
    pid,
    RANK: gl.constexpr,
    HIDDEN: gl.constexpr,
    K: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    N_FIRST: gl.constexpr,
    SLOT: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
):
    """Computes tile pid of y = x W^T and writes it to the inbox of each rank. The GEMM
    is the one of _gemm_a16w16_bandwidth_bound_kernel with layout TN. The tile has
    BLOCK_M = 4 ROWS rows and columns BLOCK_N n to BLOCK_N n + BLOCK_N - 1. Block m
    has rows 3 ROWS m to 3 ROWS m + 3 ROWS - 1 of x: the 4th row tile is 0 and is not
    posted, because a unit of the fused all-reduce has 3 rows."""
    ROW: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    BLOCK_M: gl.constexpr = 4 * ROWS
    if N_FIRST:
        # The programs of a block take its column tiles in turn: the divisor is a
        # constant, so the first load does not wait for a division.
        pid_n = pid % (HIDDEN // BLOCK_N)
        pid_m = pid // (HIDDEN // BLOCK_N)
    else:
        pid_m = pid % blocks
        pid_n = pid // blocks
    row = 3 * ROWS * pid_m
    rows = gl.minimum(tokens - row, 3 * ROWS)

    # Descriptors start at this block's (M, N) offset by biasing the base
    # pointer — subsequent async_loads use [0, 0] and step only along K.
    a_base = x + row * K
    b_base = w + pid_n * BLOCK_N * K

    a_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=a_base,
        shape=(rows, K),
        strides=(K, 1),
        block_shape=(BLOCK_M, BLOCK_K),
        layout=SHARED_LAYOUT_A,
    )
    b_desc = gl.amd.gfx1250.tdm.make_tensor_descriptor(
        base=b_base,
        shape=(HIDDEN - pid_n * BLOCK_N, K),
        strides=(K, 1),
        block_shape=(BLOCK_N, BLOCK_K),
        layout=SHARED_LAYOUT_B,
    )

    a_buffer = gl.allocate_shared_memory(
        x.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_M, BLOCK_K],
        layout=SHARED_LAYOUT_A,
    )
    b_buffer = gl.allocate_shared_memory(
        w.type.element_ty,
        shape=[NUM_BUFFERS, BLOCK_N, BLOCK_K],
        layout=SHARED_LAYOUT_B,
    )

    load_idx = 0
    compute_idx = 0

    accumulator = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=WMMA_LAYOUT)

    # Fill the pipeline. The W tile first: its descriptor does not depend on the
    # token count.
    for _ in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )

        # Walk the descriptors forward one K tile.
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )

        load_idx += 1

    # The compiler cannot move the address math of the K loop above the loads.
    gl.amd.hint.sched_barrier()

    # Line i of row j of a row tile has the units of columns 2i and 2i + 1.
    LINES: gl.constexpr = ROWS * BLOCK_N // 2
    line = gl.arange(0, LINES, layout=gl.SliceLayout(1, ROW))[:, None]
    word = gl.arange(0, 4, layout=gl.SliceLayout(0, ROW))[None, :]
    j = line // (BLOCK_N // 2)
    # The call of a triple is in the counter of its receiver. The receiver sets it only
    # after it has the lines of this program, so this load gets the call before it.
    call = gl.load(calls + ROWS * pid_m + j) % (1 << 30) + 1  # 1 to 2^30

    # Main pipeline loop
    num_k_tiles: gl.constexpr = (K + BLOCK_K - 1) // BLOCK_K

    # The last K tile is peeled out of the loop: set_bounds gives the TDM engine the
    # true remaining extent, so it zero-fills past K. K is a constexpr here, so the
    # loop is unrolled.
    for _ in gl.static_range(num_k_tiles - (NUM_BUFFERS - 1) - 1):
        gl.amd.gfx1250.tdm.async_load(
            b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
        )
        gl.amd.gfx1250.tdm.async_load(
            a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
        )

        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

        # Walk the descriptors forward one K tile.
        a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            a_desc, add_offsets=[0, BLOCK_K]
        )
        b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
            b_desc, add_offsets=[0, BLOCK_K]
        )

        load_idx += 1

        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
            OPERAND_LAYOUT_B,
        )

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

        compute_idx += 1

    # ---- Peeled final K tile (bounds-checked) ----
    k_main = (num_k_tiles - 1) * BLOCK_K
    a_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        a_desc, set_bounds=[rows, K - k_main]
    )
    b_desc = gl.amd.gfx1250.tdm.update_tensor_descriptor(
        b_desc, set_bounds=[HIDDEN - pid_n * BLOCK_N, K - k_main]
    )

    gl.amd.gfx1250.tdm.async_load(
        b_desc, [0, 0], b_buffer.index(load_idx % NUM_BUFFERS)
    )
    gl.amd.gfx1250.tdm.async_load(
        a_desc, [0, 0], a_buffer.index(load_idx % NUM_BUFFERS)
    )

    gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 1) * 2)

    load_idx += 1

    cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
        a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
    )
    cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
        b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
        OPERAND_LAYOUT_B,
    )

    accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)

    compute_idx += 1

    # Epilogue: no more loads
    for i in gl.static_range(NUM_BUFFERS - 1):
        gl.amd.gfx1250.tdm.async_wait((NUM_BUFFERS - 2 - i) * 2)

        cur_a = gl.amd.cdna4.async_copy.load_shared_relaxed(
            a_buffer.index(compute_idx % NUM_BUFFERS), OPERAND_LAYOUT_A
        )
        cur_b = gl.amd.cdna4.async_copy.load_shared_relaxed(
            b_buffer.index(compute_idx % NUM_BUFFERS).permute([1, 0]),
            OPERAND_LAYOUT_B,
        )

        accumulator = gl.amd.gfx1250.wmma(cur_a, cur_b, accumulator)
        compute_idx += 1

    # Post the tile: a thread holds 2 columns of row j of each row tile, and a unit
    # has rows j, ROWS + j and 2 ROWS + j of a column.
    Y: gl.constexpr = gl.BlockedLayout(
        [1, 2], [ROWS, 32 // ROWS], [1, gl.num_warps()], [1, 0]
    )
    y = gl.convert_layout(accumulator.to(gl.bfloat16), Y)
    y = gl.permute(gl.reshape(y, [4, ROWS, BLOCK_N]), (1, 2, 0))
    y02, y13 = gl.split(gl.reshape(y, [ROWS, BLOCK_N, 2, 2]))
    y0, y2 = gl.split(y02)
    y1, _ = gl.split(y13)
    b0 = y0.to(gl.int16, bitcast=True).to(gl.int32) & 65535
    b1 = y1.to(gl.int16, bitcast=True).to(gl.int32) << 16
    b2 = y2.to(gl.int16, bitcast=True).to(gl.int32) & 65535
    lines = gl.convert_layout(gl.reshape(gl.join(b1 | b0, b2), [LINES, 4]), ROW)
    # The tag of the call in the high half of the 2nd word of each unit.
    tag = call % 32768 + 1  # as in the fused all-reduce
    lines = lines | word % 2 * (tag << 16)
    pair = line % (BLOCK_N // 2)
    lane = pid_n * (BLOCK_N // 4) + pair // 2
    # The place of the line in the triple, as in the fused all-reduce: a group of 32
    # lanes has their first 2 columns, then their last 2.
    pos = 64 * (lane // 32) + 32 * (pair % 2) + lane % 32
    pos = 2 * HIDDEN * (ROWS * pid_m + j) + 4 * pos + word
    slot = (call % 2 * 4 + RANK) * SLOT
    for r in gl.static_range(4):  # this rank too: its receivers read it there
        # 16-byte aligned, so that a line moves in one store.
        inbox = gl.multiple_of(inboxes[r].to(gl.pointer_type(gl.int32)), 16)
        gl.amd.cdna5.buffer_store(
            lines, inbox, slot + pos, mask=row + j < tokens, cache=".wt"
        )


@gluon.jit
def _receive_all(
    inboxes,
    pos,
    call,
    err,
    RANK: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
):
    """Waits until the lines at pos of the 4 ranks carry the tag of the call. Returns
    the 3 values of each unit, added over the 4 ranks in fp32."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    tag = call % 32768 + 1
    bank = call % 2 * 4 * SLOT
    inbox = gl.multiple_of(inboxes[RANK].to(gl.pointer_type(gl.int32)), 16)
    l0 = pos
    l1 = pos
    l2 = pos
    l3 = pos
    late = 1
    it = 0
    while (late > 0) & (it < MAXIT):
        # All loads first, then the checks: the loads of the 4 ranks overlap.
        l0 = gl.amd.cdna5.buffer_load(inbox, bank + pos, cache=".cv")
        l1 = gl.amd.cdna5.buffer_load(inbox, bank + SLOT + pos, cache=".cv")
        l2 = gl.amd.cdna5.buffer_load(inbox, bank + 2 * SLOT + pos, cache=".cv")
        l3 = gl.amd.cdna5.buffer_load(inbox, bank + 3 * SLOT + pos, cache=".cv")
        old = gl.zeros([pos.shape[0], 2], gl.int32, WORDS)
        for r in gl.static_range(4):
            _, hi = _unit_words((l0, l1, l2, l3)[r])
            old += (hi & -65536) != tag << 16
        # Keeps the loads in the loop.
        gl.inline_asm("", "~{memory}", [], [], is_pure=False)
        late = gl.sum(gl.sum(old, axis=1), axis=0)
        it += 1

    if late > 0:  # the poll timed out
        gl.atomic_add(err, 1)
    a0, a1, a2 = _values(l0)
    b0, b1, b2 = _values(l1)
    c0, c1, c2 = _values(l2)
    d0, d1, d2 = _values(l3)
    # Rank order, so that all ranks get the same bits.
    return a0 + b0 + c0 + d0, a1 + b1 + c1 + d1, a2 + b2 + c2 + d2


@gluon.jit
def _receive_norm(
    residual,
    weight,
    out,
    residual_out,
    inboxes,
    calls,
    err,
    tokens,
    blocks,
    first,
    programs,
    eps,
    RANK: gl.constexpr,
    HIDDEN: gl.constexpr,
    ROWS: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
    GEMMA: gl.constexpr,
):
    """Receives triples first, first + programs, ... of all blocks: adds the 4 partial
    sums and the residual, and normalizes the rows."""
    ROW: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    N: gl.constexpr = HIDDEN // 4
    lane = gl.arange(0, N, layout=gl.SliceLayout(1, ROW))[:, None]
    col = 4 * lane + gl.arange(0, 4, layout=gl.SliceLayout(0, ROW))[None, :]
    line = gl.arange(0, 2 * N, layout=gl.SliceLayout(1, ROW))[:, None]
    i = line % N
    pos = 64 * (i // 32) + 32 * (line // N) + i % 32
    pos = 4 * pos + gl.arange(0, 4, layout=gl.SliceLayout(0, ROW))[None, :]
    w = gl.amd.cdna5.buffer_load(weight, col).to(gl.float32)  # once for all rows
    if GEMMA:
        w += 1.0
    for t in range(first, blocks * ROWS, programs):
        row = 3 * ROWS * (t // ROWS) + t % ROWS
        if row < tokens:
            # The counter of triple t counts the calls with its rows.
            call = gl.load(calls + t) % (1 << 30) + 1
            # The residual is local: load it before the wait.
            res = residual + row * HIDDEN
            r0 = gl.amd.cdna5.buffer_load(res, col)
            r1 = gl.amd.cdna5.buffer_load(
                res, col + ROWS * HIDDEN, mask=row + ROWS < tokens, other=0
            )
            r2 = gl.amd.cdna5.buffer_load(
                res, col + 2 * ROWS * HIDDEN, mask=row + 2 * ROWS < tokens, other=0
            )
            s0, s1, s2 = _receive_all(
                inboxes, 2 * HIDDEN * t + pos, call, err, RANK, SLOT, MAXIT
            )
            # All GEMM programs of this rank have read the call: they posted its lines.
            gl.barrier()  # all waves have read the old call number
            gl.store(calls + t, call)
            # The residual last, in fp32: all ranks get the same bits.
            s0 = _columns(s0) + r0.to(gl.float32)
            s1 = _columns(s1) + r1.to(gl.float32)
            s2 = _columns(s2) + r2.to(gl.float32)
            # One sum over the waves for the 3 rows: the joins put row r at index r.
            sq = gl.join(gl.join(s0 * s0, s2 * s2), gl.join(s1 * s1, s2 * 0))
            sq = gl.sum(gl.sum(gl.reshape(sq, [N, 4, 4]), axis=1), axis=0)
            idx = gl.arange(0, 4, layout=sq.type.layout)
            for r in gl.static_range(3):
                if row + r * ROWS < tokens:
                    at = (row + r * ROWS) * HIDDEN + col
                    s = (s0, s1, s2)[r]
                    gl.amd.cdna5.buffer_store(s.to(gl.bfloat16), residual_out, at)
                    square_sum = gl.sum(gl.where(idx == r, sq, 0.0), axis=0)
                    y = s * gl.rsqrt(square_sum / HIDDEN + eps) * w
                    gl.amd.cdna5.buffer_store(y.to(gl.bfloat16), out, at)


_gemm_a16w16_all_reduce_rmsnorm_repr = make_kernel_repr(
    "_gemm_a16w16_all_reduce_rmsnorm_kernel",
    [
        "RANK",
        "HIDDEN",
        "K",
        "ROWS",
        "BLOCK_N",
        "BLOCK_K",
        "NUM_BUFFERS",
        "N_FIRST",
        "GEMMA",
    ],
)


@gluon.jit(
    repr=_gemm_a16w16_all_reduce_rmsnorm_repr, do_not_specialize=["tokens", "programs"]
)
def _gemm_a16w16_all_reduce_rmsnorm_kernel(
    x,
    w,
    residual,
    weight,
    out,
    residual_out,
    inboxes,
    calls,
    err,
    tokens,
    programs,
    eps,
    RANK: gl.constexpr,
    HIDDEN: gl.constexpr,
    K: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    NUM_BUFFERS: gl.constexpr,
    N_FIRST: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
    GEMMA: gl.constexpr,
    SHARED_LAYOUT_A: gl.constexpr,
    SHARED_LAYOUT_B: gl.constexpr,
    WMMA_LAYOUT: gl.constexpr,
    OPERAND_LAYOUT_A: gl.constexpr,
    OPERAND_LAYOUT_B: gl.constexpr,
):
    """A block has 3 ROWS rows of x. Triple j of block m has rows 3 ROWS m + j, + ROWS and
    + 2 ROWS. Programs 0 to G - 1 compute the G tiles of y. Programs G to G + P - 1
    receive triples p, p + P, ... (P = programs). The GPU starts programs in order, so a
    waiting receiver cannot block a GEMM program."""
    blocks = gl.cdiv(tokens, 3 * ROWS)
    tiles = blocks * (HIDDEN // BLOCK_N)
    pid = gl.program_id(0)
    if pid < tiles:
        _gemm_post(
            x,
            w,
            inboxes,
            calls,
            tokens,
            blocks,
            pid,
            RANK,
            HIDDEN,
            K,
            ROWS,
            BLOCK_N,
            BLOCK_K,
            NUM_BUFFERS,
            N_FIRST,
            SLOT,
            SHARED_LAYOUT_A,
            SHARED_LAYOUT_B,
            WMMA_LAYOUT,
            OPERAND_LAYOUT_A,
            OPERAND_LAYOUT_B,
        )
    else:
        _receive_norm(
            residual,
            weight,
            out,
            residual_out,
            inboxes,
            calls,
            err,
            tokens,
            blocks,
            pid - tiles,
            programs,
            eps,
            RANK,
            HIDDEN,
            ROWS,
            SLOT,
            MAXIT,
            GEMMA,
        )
