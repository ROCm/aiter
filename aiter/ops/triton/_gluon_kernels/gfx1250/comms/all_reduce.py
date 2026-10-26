"""All-reduce of bf16 data over 4 GPUs. A sender writes a chunk of x into the inbox
of each peer as 16-byte lines [word, call, word, call]. A receiver waits until its
lines carry the current call number, then adds the 4 chunks. Even and odd calls
use different banks, because a GPU can be one call ahead of a peer."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _lo(word):  # the low bf16 value of a word, as fp32
    return (word << 16).to(gl.float32, bitcast=True)


@gluon.jit
def _hi(word):
    return (word & -65536).to(gl.float32, bitcast=True)


@gluon.jit
def _pack(lo, hi):  # two fp32 values as one word of two bf16 values
    lo = lo.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 65535
    hi = hi.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) << 16
    return hi | lo


@gluon.jit
def _send(
    x,
    inboxes,
    calls,
    c,
    n_words,
    RANK: gl.constexpr,
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
):
    """Write chunk c of x, tagged with the call number, into the inbox of each peer."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    line = c * LINES + gl.arange(0, LINES, layout=gl.SliceLayout(1, WORDS))[:, None]
    word = 2 * line + gl.arange(0, 2, layout=gl.SliceLayout(0, WORDS))[None, :]
    call = gl.load(calls + 2 * c) + 1
    data = gl.load(x + word, mask=2 * line < n_words)  # one mask for each line
    gl.barrier()  # all waves have read the old call number
    gl.store(calls + 2 * c, call)
    lines = gl.reshape(gl.join(data, gl.full_like(data, call)), [LINES, 4])
    lines = gl.convert_layout(lines, LINE, assert_trivial=True)

    line = c * LINES + gl.arange(0, LINES, layout=gl.SliceLayout(1, LINE))[:, None]
    pos = 4 * line + gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    slot = (call % 2 * 4 + RANK) * SLOT  # this rank's slot in the bank of the call
    for peer in gl.static_range(4):
        if peer != RANK:
            # 16-byte aligned, so that a line moves in one store
            inbox = gl.multiple_of(inboxes[peer].to(gl.pointer_type(gl.int32)), 16)
            gl.amd.cdna5.buffer_store(
                lines, inbox, slot + pos, mask=2 * line < n_words, cache=".wt"
            )


@gluon.jit
def _receive(
    x,
    inboxes,
    calls,
    err,
    c,
    n_words,
    RANK: gl.constexpr,
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
):
    """Wait for chunk c of the 3 peers. Returns the words of ranks 0 to 3, their
    x offsets and the line mask."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    line = c * LINES + gl.arange(0, LINES, layout=gl.SliceLayout(1, WORDS))[:, None]
    word = 2 * line + gl.arange(0, 2, layout=gl.SliceLayout(0, WORDS))[None, :]
    call = gl.load(calls + 2 * c + 1) + 1
    word_ok = 2 * line < n_words
    mine = gl.load(x + word, mask=word_ok)
    gl.barrier()  # all waves have read the old call number
    gl.store(calls + 2 * c + 1, call)

    line = c * LINES + gl.arange(0, LINES, layout=gl.SliceLayout(1, LINE))[:, None]
    pos = 4 * line + gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    bank = call % 2 * 4 * SLOT
    inbox = gl.multiple_of(inboxes[RANK].to(gl.pointer_type(gl.int32)), 16)
    w0 = mine
    w1 = mine
    w2 = mine
    w3 = mine
    late = 1
    it = 0
    while (late > 0) & (it < MAXIT):
        old = gl.zeros_like(mine)
        for r in gl.static_range(4):
            if r != RANK:
                lines = gl.amd.cdna5.buffer_load(
                    inbox,
                    bank + r * SLOT + pos,
                    mask=2 * line < n_words,
                    other=call,
                    cache=".cv",
                )
                words, tags = gl.split(gl.reshape(lines, [LINES, 2, 2]))
                words = gl.convert_layout(words, WORDS, assert_trivial=True)
                old += gl.convert_layout(tags, WORDS, assert_trivial=True) != call
                if r == 0:
                    w0 = words
                elif r == 1:
                    w1 = words
                elif r == 2:
                    w2 = words
                else:
                    w3 = words
        # Keeps the loads in the loop.
        gl.inline_asm("", "~{memory}", [], [], is_pure=False)
        late = gl.sum(gl.sum(old, axis=1), axis=0)
        it += 1

    if it >= MAXIT:
        gl.atomic_add(err, 1)
    return w0, w1, w2, w3, word, word_ok


_all_reduce_repr = make_kernel_repr("_all_reduce_kernel", ["RANK", "LINES"])


@gluon.jit(repr=_all_reduce_repr, do_not_specialize=["n_words"])
def _all_reduce_kernel(
    x,
    y,
    inboxes,
    calls,
    err,
    n_words,
    RANK: gl.constexpr,
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
):
    """y = the sum of x over 4 ranks, as n_words words of two bf16 values.
    Programs 0 to C-1 send the C chunks and programs C to 2C-1 receive them: the
    GPU starts programs in order, so a waiting receiver cannot block a sender."""
    num_chunks = gl.cdiv(n_words, 2 * LINES)
    pid = gl.program_id(0)
    if pid < num_chunks:
        _send(x, inboxes, calls, pid, n_words, RANK, LINES, SLOT)
    else:
        w0, w1, w2, w3, word, word_ok = _receive(
            x, inboxes, calls, err, pid - num_chunks, n_words, RANK, LINES, SLOT, MAXIT
        )
        # Rank order in fp32, so that all ranks get the same bits.
        lo = _lo(w0) + _lo(w1) + _lo(w2) + _lo(w3)
        hi = _hi(w0) + _hi(w1) + _hi(w2) + _hi(w3)
        gl.store(y + word, _pack(lo, hi), mask=word_ok)
