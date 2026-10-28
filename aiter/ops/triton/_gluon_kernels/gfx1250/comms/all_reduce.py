"""All-reduce of bf16 data over 4 GPUs in one kernel. A sender writes its chunk of x
into the inbox of each peer as 16-byte lines of 2 units [value, value, value, tag]. A
receiver waits until its lines carry the tag of the call, then adds the 4 chunks. Even
and odd calls use 2 banks of the inbox: a GPU can be one call ahead of a peer."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _lines(x, value, mask0, mask1, mask2, PLANE: gl.constexpr, counter):
    """Advances the call counter. Returns the call and the lines of 2 units [x[value],
    x[value + PLANE], x[value + 2 PLANE], tag], with 0 for the values out of a mask."""
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    call = gl.load(counter) % (1 << 30) + 1  # 1 to 2^30
    # Buffer loads: a mask does not make a branch.
    v0 = gl.amd.cdna5.buffer_load(x, value, mask=mask0, other=0)
    v1 = gl.amd.cdna5.buffer_load(x, value + PLANE, mask=mask1, other=0)
    v2 = gl.amd.cdna5.buffer_load(x, value + 2 * PLANE, mask=mask2, other=0)
    gl.barrier()  # all waves have read the old call number
    gl.store(counter, call)
    b0 = v0.to(gl.int16, bitcast=True).to(gl.int32) & 65535
    b1 = v1.to(gl.int16, bitcast=True).to(gl.int32) << 16
    b2 = v2.to(gl.int16, bitcast=True).to(gl.int32) & 65535
    # Never 0 (an empty inbox). Before the data comes, a unit holds the call of 2 uses
    # of the chunk ago: 15 bits tell the 2 calls apart.
    tag = call % 32768 + 1
    N: gl.constexpr = value.shape[0]
    H: gl.constexpr = value.shape[1] // 2  # lines of a row
    # Line h * N + i has the units of columns 2h and 2h + 1 of row i.
    lines = gl.reshape(gl.join(b1 | b0, tag << 16 | b2), [N, H, 4])
    lines = gl.reshape(gl.permute(lines, (1, 0, 2)), [H * N, 4])
    return call, gl.convert_layout(lines, LINE, assert_trivial=True)


@gluon.jit
def _unit_words(lines):
    """The 2 words of each unit: values 0 and 1, and value 2 and the tag."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    lo, hi = gl.split(gl.reshape(lines, [lines.shape[0], 2, 2]))
    lo = gl.convert_layout(lo, WORDS, assert_trivial=True)
    return lo, gl.convert_layout(hi, WORDS, assert_trivial=True)


@gluon.jit
def _values(lines):
    """The 3 values of each unit as fp32. A word has 2 bf16 values, the low one first."""
    lo, hi = _unit_words(lines)
    v0 = (lo << 16).to(gl.float32, bitcast=True)
    v1 = (lo & -65536).to(gl.float32, bitcast=True)
    return v0, v1, (hi << 16).to(gl.float32, bitcast=True)


@gluon.jit
def _post(lines, inboxes, pos, mask, call, RANK: gl.constexpr, SLOT: gl.constexpr):
    """Writes the lines in the mask to pos in the slot of this rank in each peer."""
    slot = (call % 2 * 4 + RANK) * SLOT
    for peer in gl.static_range(4):
        if peer != RANK:
            # 16-byte aligned, so that a line moves in one store.
            inbox = gl.multiple_of(inboxes[peer].to(gl.pointer_type(gl.int32)), 16)
            gl.amd.cdna5.buffer_store(lines, inbox, slot + pos, mask=mask, cache=".wt")


@gluon.jit
def _receive(
    inboxes,
    pos,
    mask,
    call,
    mine,
    err,
    RANK: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
):
    """Waits until the lines at pos in the mask of the 3 peers carry the tag of the
    call. Returns the 3 values of each unit, added over the 4 ranks in fp32."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    col = gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    tag = call % 32768 + 1
    empty = col % 2 * (tag << 16)  # a line out of the mask reads as 0 with the tag
    bank = call % 2 * 4 * SLOT
    inbox = gl.multiple_of(inboxes[RANK].to(gl.pointer_type(gl.int32)), 16)
    l0 = mine
    l1 = mine
    l2 = mine
    l3 = mine
    late = 1
    it = 0
    while (late > 0) & (it < MAXIT):
        # All loads first, then the checks: the loads of the 3 peers overlap.
        for r in gl.static_range(4):
            if r != RANK:
                lines = gl.amd.cdna5.buffer_load(
                    inbox, bank + r * SLOT + pos, mask=mask, other=empty, cache=".cv"
                )
                if r == 0:
                    l0 = lines
                elif r == 1:
                    l1 = lines
                elif r == 2:
                    l2 = lines
                else:
                    l3 = lines
        old = gl.zeros([pos.shape[0], 2], gl.int32, WORDS)
        for r in gl.static_range(4):
            if r != RANK:
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


_all_reduce_repr = make_kernel_repr("_all_reduce_kernel", ["RANK", "LINES"])


@gluon.jit(repr=_all_reduce_repr, do_not_specialize=["n", "programs"])
def _all_reduce_kernel(
    x,
    y,
    inboxes,
    calls,
    err,
    n,
    programs,
    RANK: gl.constexpr,
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
):
    """y = the sum of the n values of x over the 4 ranks. Programs 0 to C-1 send the C
    chunks, one each: a load after a remote store waits for its ack. Programs C to
    C+P-1 receive chunks p, p + P, ... (P = programs). The GPU starts programs in
    order, so a waiting receiver cannot block a sender."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    num_chunks = gl.cdiv(n, 6 * LINES)
    pid = gl.program_id(0)
    # Unit u of a chunk has the values u, U + u and 2U + u (U = 2 LINES units), so that
    # the loads of x are coalesced.
    line = gl.arange(0, LINES, layout=gl.SliceLayout(1, WORDS))[:, None]
    pair = gl.arange(0, 2, layout=gl.SliceLayout(0, WORDS))[None, :]  # 2 units a line
    line4 = gl.arange(0, LINES, layout=gl.SliceLayout(1, LINE))[:, None]
    col4 = gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    if pid < num_chunks:
        first = 6 * pid * LINES + 2 * line  # one mask for the 2 units of a line
        ok1 = first + 2 * LINES < n
        ok2 = first + 4 * LINES < n
        call, lines = _lines(
            x, first + pair, first < n, ok1, ok2, 2 * LINES, calls + 2 * pid
        )
        # All lines of the chunk, also past n: no old tag can stay.
        pos = 4 * (pid * LINES + line4) + col4
        _post(lines, inboxes, pos, line4 < LINES, call, RANK, SLOT)
    else:
        for c in range(pid - num_chunks, num_chunks, programs):
            first = 6 * c * LINES + 2 * line
            value = first + pair
            ok1 = first + 2 * LINES < n
            ok2 = first + 4 * LINES < n
            call, lines = _lines(
                x, value, first < n, ok1, ok2, 2 * LINES, calls + 2 * c + 1
            )
            pos = 4 * (c * LINES + line4) + col4
            needed = 6 * c * LINES + 2 * line4 < n  # the line has values of x
            s0, s1, s2 = _receive(
                inboxes, pos, needed, call, lines, err, RANK, SLOT, MAXIT
            )
            s0 = s0.to(gl.bfloat16)
            s1 = s1.to(gl.bfloat16)
            s2 = s2.to(gl.bfloat16)
            gl.amd.cdna5.buffer_store(s0, y, value, mask=first < n)
            gl.amd.cdna5.buffer_store(s1, y, value + 2 * LINES, mask=ok1)
            gl.amd.cdna5.buffer_store(s2, y, value + 4 * LINES, mask=ok2)
