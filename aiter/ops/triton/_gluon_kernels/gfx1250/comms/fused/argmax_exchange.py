# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The greedy pick of a TP-sharded LM head over 4 GPUs in one kernel: each rank
reduces its tile keys (gemm_a16w16_argmax_keys) to one (max, token) a row, sends it
to the 3 peers as a 16-byte line [max, tag, token, tag] (each 8-byte half carries
the tag of the call, as the units of all_reduce do), and every rank picks the row's
winner over the 4 ranks in rank order: the larger max, the lower rank on a tie, a
NaN above all (torch.argmax over the ranks' maxima). Even and odd calls use 2 banks
of the inbox, as in all_reduce."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _row_best(keys, stride_keys, rows, num_tiles, BLOCK_T: gl.constexpr):
    """The largest key of each row in rows over its num_tiles tile keys, as the
    (fp32 max, column) it encodes (see _argmax_key)."""
    KEYS: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [gl.num_warps(), 1], [1, 0])
    r = gl.convert_layout(rows, gl.SliceLayout(1, KEYS))[:, None]
    t = gl.arange(0, BLOCK_T, layout=gl.SliceLayout(0, KEYS))[None, :]
    best = gl.full([rows.shape[0], BLOCK_T], -(2**63), gl.int64, KEYS)
    for start in range(0, num_tiles, BLOCK_T):
        k = gl.load(
            keys + r.to(gl.int64) * stride_keys + start + t,
            mask=(start + t < num_tiles) & (r >= 0),
            other=-(2**63),
        )
        best = gl.maximum(best, k)
    best = gl.max(best, axis=1)
    mono = (best >> 32).to(gl.int32)
    bits = mono ^ ((mono >> 31) & 2147483647)
    col = (4294967295 - (best & 4294967295)).to(gl.int32)
    return bits, col


@gluon.jit
def _word(lines, k: gl.constexpr):
    """Word k of each line ([R, 4] int32) as [R]."""
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    col = gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    return gl.sum(gl.where(col == k, lines, 0), axis=1)


_argmax_exchange_repr = make_kernel_repr("_argmax_exchange_pick_kernel", ["RANK"])


@gluon.jit(
    repr=_argmax_exchange_repr, do_not_specialize=["M", "num_tiles", "vocab_start"]
)
def _argmax_exchange_pick_kernel(
    keys,
    out,
    inboxes,
    calls,
    err,
    M,
    num_tiles,
    stride_keys,
    vocab_start,
    RANK: gl.constexpr,
    ROWS: gl.constexpr,
    BLOCK_T: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
):
    """out[m] = the token of row m's maximum over the 4 ranks. Programs 0 to P-1
    send rows pR to pR + R - 1 (R = ROWS), programs P to 2P-1 receive them and
    pick: the GPU starts programs in order, so a waiting receiver cannot block a
    sender."""
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    num_blocks = gl.cdiv(M, ROWS)
    pid = gl.program_id(0)
    block = pid % num_blocks
    row = block * ROWS + gl.arange(0, ROWS, layout=gl.SliceLayout(1, LINE))
    live = row < M
    bits, col = _row_best(
        keys, stride_keys, gl.where(live, row, -1), num_tiles, BLOCK_T
    )
    bits = gl.convert_layout(bits, gl.SliceLayout(1, LINE))
    token = gl.convert_layout(col, gl.SliceLayout(1, LINE)) + vocab_start
    counter = calls + 2 * block + (pid >= num_blocks).to(gl.int32)
    call = gl.load(counter) % (1 << 30) + 1  # 1 to 2^30: never 0, an empty inbox
    gl.barrier()  # all waves have read the old call number
    gl.store(counter, call)
    c = gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    pos = 4 * row[:, None] + c
    mask = live[:, None] & (c < 4)
    bank = call % 2 * 4 * SLOT
    if pid < num_blocks:
        lines = gl.where(c == 0, bits[:, None], gl.where(c == 2, token[:, None], call))
        for peer in gl.static_range(4):
            if peer != RANK:
                inbox = gl.multiple_of(inboxes[peer].to(gl.pointer_type(gl.int32)), 16)
                gl.amd.cdna5.buffer_store(
                    lines, inbox, bank + RANK * SLOT + pos, mask=mask, cache=".wt"
                )
    else:
        inbox = gl.multiple_of(inboxes[RANK].to(gl.pointer_type(gl.int32)), 16)
        # A line out of the mask reads as ready.
        empty = gl.where((pos - 4 * row[:, None]) % 2 == 1, call, 0)
        l0 = gl.zeros([ROWS, 4], gl.int32, LINE)
        l1 = l0
        l2 = l0
        l3 = l0
        late = 1
        it = 0
        while (late > 0) & (it < MAXIT):
            # All loads first, then the checks: the loads of the 3 peers overlap.
            for r in gl.static_range(4):
                if r != RANK:
                    lines = gl.amd.cdna5.buffer_load(
                        inbox,
                        bank + r * SLOT + pos,
                        mask=mask,
                        other=empty,
                        cache=".cv",
                    )
                    if r == 0:
                        l0 = lines
                    elif r == 1:
                        l1 = lines
                    elif r == 2:
                        l2 = lines
                    else:
                        l3 = lines
            old = gl.zeros([ROWS, 4], gl.int32, LINE)
            for r in gl.static_range(4):
                if r != RANK:
                    stale = ((l0, l1, l2, l3)[r] != call) & (c % 2 == 1)
                    old += stale.to(gl.int32)
            # Keeps the loads in the loop.
            gl.inline_asm("", "~{memory}", [], [], is_pure=False)
            late = gl.sum(gl.sum(old, axis=1), axis=0)
            it += 1
        if late > 0:  # the poll timed out
            gl.atomic_add(err, 1)
        best_val = bits.to(gl.float32, bitcast=True)
        best_tok = token
        for r in gl.static_range(4):
            if r == RANK:
                val = bits.to(gl.float32, bitcast=True)
                tok = token
            else:
                lines = (l0, l1, l2, l3)[r]
                val = _word(lines, 0).to(gl.float32, bitcast=True)
                tok = _word(lines, 2)
            if r == 0:
                best_val = val
                best_tok = tok
            else:
                nan = val != val  # noqa: PLR0124
                best_nan = best_val != best_val  # noqa: PLR0124
                take = (val > best_val) | (nan & ~best_nan)
                best_val = gl.where(take, val, best_val)
                best_tok = gl.where(take, tok, best_tok)
        gl.store(out + row, best_tok, mask=live)
