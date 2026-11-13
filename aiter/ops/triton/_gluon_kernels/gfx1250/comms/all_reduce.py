"""All-reduce of bf16 data over 4 GPUs in one kernel. A sender writes its chunk of x
into the inbox of each peer as 16-byte lines of 2 units [value, value, value, tag]. A
receiver waits until its lines carry the tag of the call, then adds the 4 chunks. Even
and odd calls use 2 banks of the inbox: a GPU can be one call ahead of a peer.

With NORM, the kernel also adds the residual and applies RMSNorm (and per-token FP8
quant) to its rows. A row is then cdiv(HIDDEN, 384) chunks; the last one is partial and
its lines past the row carry 0. A program runs W waves (a power of 2, at least
HIDDEN / 768): a sender sends one chunk a wave, and a receiver takes one row, 2 chunks
a wave. For HIDDEN 6144 (W 8), these 768 values are those of one wave of the HIP
add_rmsnorm_quant kernel, and the waves add and compare their results through LDS in
the order of that kernel. So the results have the same bits."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.constexpr_function
def _row_chunks(hidden, lines):
    """The chunks of a row of hidden values: 6 values a line."""
    return -(-hidden // (6 * lines))


@gluon.constexpr_function
def _log2(w):
    return w.bit_length() - 1


@gluon.jit
def _lines(x, value, mask0, mask1, mask2, PLANE: gl.constexpr, counter, live=None):
    """Advances the call counter. Returns the call and the lines of 2 units [x[value],
    x[value + PLANE], x[value + 2 PLANE], tag], with 0 for the values out of a mask.
    With live, counter is a tensor and only its live counters advance."""
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    if live is None:
        call = gl.load(counter) % (1 << 30) + 1  # 1 to 2^30
    else:
        call = gl.load(counter, mask=live, other=0) % (1 << 30) + 1
    # Buffer loads: a mask does not make a branch.
    v0 = gl.amd.cdna5.buffer_load(x, value, mask=mask0, other=0)
    v1 = gl.amd.cdna5.buffer_load(x, value + PLANE, mask=mask1, other=0)
    v2 = gl.amd.cdna5.buffer_load(x, value + 2 * PLANE, mask=mask2, other=0)
    gl.barrier()  # all waves have read the old call number
    if live is None:
        gl.store(counter, call)
    else:
        gl.store(counter, call, mask=live)
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
    return _sum_ranks(l0, l1, l2, l3)


@gluon.jit
def _sum_ranks(l0, l1, l2, l3):
    """The 3 values of each unit, added over the 4 ranks in fp32. Rank order, so that
    all ranks get the same bits."""
    a0, a1, a2 = _values(l0)
    b0, b1, b2 = _values(l1)
    c0, c1, c2 = _values(l2)
    d0, d1, d2 = _values(l3)
    return a0 + b0 + c0 + d0, a1 + b1 + c1 + d1, a2 + b2 + c2 + d2


@gluon.jit
def _receive_pair(
    inboxes,
    pos,
    call_a,
    call_b,
    mine_a,
    mine_b,
    ok_a,
    ok_b,
    err,
    RANK: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
):
    """_receive for 2 chunks in one poll loop: the lines at pos and pos + 4 LINES
    words, with the call tensors call_a and call_b of their rows. Only the chunks in
    ok_a and ok_b exist; the others read as 0."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINES: gl.constexpr = 64
    N: gl.constexpr = pos.shape[0]
    tag_a = call_a % 32768 + 1
    tag_b = call_b % 32768 + 1
    col = gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    empty_a = col % 2 * gl.convert_layout(tag_a << 16, LINE, True)
    empty_b = col % 2 * gl.convert_layout(tag_b << 16, LINE, True)
    at_a = gl.convert_layout(call_a % 2 * 4 * SLOT, LINE, True) + pos
    at_b = gl.convert_layout(call_b % 2 * 4 * SLOT, LINE, True) + pos + 4 * LINES
    inbox = gl.multiple_of(inboxes[RANK].to(gl.pointer_type(gl.int32)), 16)
    a0 = mine_a
    a1 = mine_a
    a2 = mine_a
    a3 = mine_a
    b0 = mine_b
    b1 = mine_b
    b2 = mine_b
    b3 = mine_b
    late = 1
    it = 0
    while (late > 0) & (it < MAXIT):
        for r in gl.static_range(4):
            if r != RANK:
                la = gl.amd.cdna5.buffer_load(
                    inbox, at_a + r * SLOT, mask=ok_a, other=empty_a, cache=".cv"
                )
                lb = gl.amd.cdna5.buffer_load(
                    inbox, at_b + r * SLOT, mask=ok_b, other=empty_b, cache=".cv"
                )
                if r == 0:
                    a0 = la
                    b0 = lb
                elif r == 1:
                    a1 = la
                    b1 = lb
                elif r == 2:
                    a2 = la
                    b2 = lb
                else:
                    a3 = la
                    b3 = lb
        old = gl.zeros([N, 2], gl.int32, WORDS)
        for r in gl.static_range(4):
            if r != RANK:
                _, hi = _unit_words((a0, a1, a2, a3)[r])
                old += (hi & -65536) != tag_a << 16
                _, hi = _unit_words((b0, b1, b2, b3)[r])
                old += (hi & -65536) != tag_b << 16
        # Keeps the loads in the loop.
        gl.inline_asm("", "~{memory}", [], [], is_pure=False)
        late = gl.sum(gl.sum(old, axis=1), axis=0)
        it += 1

    if late > 0:  # the poll timed out
        gl.atomic_add(err, 1)
    sa0, sa1, sa2 = _sum_ranks(a0, a1, a2, a3)
    sb0, sb1, sb2 = _sum_ranks(b0, b1, b2, b3)
    return sa0, sa1, sa2, sb0, sb1, sb2


@gluon.jit
def _asm(op: gl.constexpr, v):
    """One VALU instruction on fp32 v, as the HIP kernel uses it (rcp or rsq)."""
    return gl.inline_asm_elementwise(
        op + " $0, $1", "=v,v", [v], dtype=gl.float32, is_pure=True, pack=1
    )


@gluon.jit
def _wave_units(p):
    """Plane p [rows, 2] of the waves as [wave, unit] (see _wave_lines)."""
    W: gl.constexpr = gl.num_warps()
    p = gl.permute(gl.reshape(p, [2, W, 32, 2]), (1, 0, 2, 3))
    return gl.reshape(p, [W, 128])


@gluon.jit
def _square_sum(s0, s1, s2, s3, s4, s5):
    """The sum of the squares of a row, as add_rmsnorm_quant_kernel<256, 24> adds them.
    s0 to s5 are the 6 planes of 128 values of the slab of each wave. Lane L of HIP
    wave w has values 256 j + 8 L + k (k < 8) of slab w: it adds their squares in the
    order 8 j + k. Then lane pairs at distance 1 to 16 and wave pairs at distance 1 to
    W / 2 are added. A sum over a dimension of 2 is one add, so for W 8 the order is
    that of HIP."""
    W: gl.constexpr = gl.num_warps()
    HIP: gl.constexpr = gl.BlockedLayout([1, 8], [1, 32], [W, 1], [1, 0])
    planes = (s0, s1, s2, s3, s4, s5)
    sq = None
    for j in gl.static_range(3):
        # HIP plane j is slab planes 2j and 2j + 1: [wave, 256] in the HIP lane layout.
        t = gl.join(_wave_units(planes[2 * j]), _wave_units(planes[2 * j + 1]))
        t = gl.reshape(gl.permute(t, (0, 2, 1)), [W, 256])
        t = gl.reshape(gl.convert_layout(t, HIP), [W, 32, 2, 2, 2])  # k = 4a + 2b + c
        c0, c1 = gl.split(t)
        c0b0, c0b1 = gl.split(c0)
        c1b0, c1b1 = gl.split(c1)
        e0, e4 = gl.split(c0b0)
        e2, e6 = gl.split(c0b1)
        e1, e5 = gl.split(c1b0)
        e3, e7 = gl.split(c1b1)
        for k in gl.static_range(8):
            e = (e0, e1, e2, e3, e4, e5, e6, e7)[k]
            if j == 0 and k == 0:
                sq = e * e
            else:
                sq = sq + e * e
    sq = gl.reshape(sq, [W, 2, 2, 2, 2, 2])  # lane bit 0 last
    sq = gl.sum(gl.sum(gl.sum(gl.sum(gl.sum(sq, axis=5), axis=4), axis=3), axis=2), 1)
    for _ in gl.static_range(_log2(W)):  # wave bit 0 first
        sq = gl.sum(gl.reshape(sq, [sq.shape[0] // 2, 2]), axis=1)
    return gl.sum(sq, axis=0)


@gluon.jit
def _wave_lines(LINES: gl.constexpr, LAYOUT: gl.constexpr):
    """Row r of a [W LINES, ...] tensor in LAYOUT: its wave and its line in the chunk
    of that wave (lane l of a wave has lines l and l + 32, as with one wave)."""
    W: gl.constexpr = gl.num_warps()
    r = gl.arange(0, W * LINES, layout=gl.SliceLayout(1, LAYOUT))[:, None]
    return r // 32 % W, r % 32 + 32 * (r // (32 * W))


@gluon.jit
def _planes(k, line, HIDDEN: gl.constexpr, LINES: gl.constexpr):
    """Masks of the 3 value planes of the lines of chunk k of a row: in the row."""
    at = k * 6 * LINES + 2 * line
    return at < HIDDEN, at + 2 * LINES < HIDDEN, at + 4 * LINES < HIDDEN


@gluon.jit
def _send_norm(
    x,
    inboxes,
    calls,
    pid,
    tokens,
    RANK: gl.constexpr,
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
    HIDDEN: gl.constexpr,
):
    """Sends chunks W pid to W pid + W - 1 of the rows, one a wave, as a plain sender.
    All lines of a chunk are sent, with 0 past the row: no old tag can stay."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    ROW: gl.constexpr = _row_chunks(HIDDEN, LINES)
    pair = gl.arange(0, 2, layout=gl.SliceLayout(0, WORDS))[None, :]
    col4 = gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    wave, line = _wave_lines(LINES, WORDS)
    wave4, line4 = _wave_lines(LINES, LINE)
    c = gl.num_warps() * pid + wave
    c4 = gl.num_warps() * pid + wave4
    k = c % ROW
    value = c // ROW * HIDDEN + k * 6 * LINES + 2 * line + pair
    if HIDDEN == 12 * LINES * gl.num_warps():
        # A program sends half a row: no mask, as the masked counter is slower.
        live = None
        live4 = None
        ok = line >= 0
        m0, m1, m2 = ok, ok, ok
    else:
        live = c < tokens * ROW
        live4 = c4 < tokens * ROW
        m0, m1, m2 = _planes(k, line, HIDDEN, LINES)
        m0, m1, m2 = live & m0, live & m1, live & m2
    call, lines = _lines(x, value, m0, m1, m2, 2 * LINES, calls + 2 * c, live)
    pos = 4 * (c4 * LINES + line4) + col4
    call = gl.convert_layout(call, LINE, True)
    _post(lines, inboxes, pos, live4, call, RANK, SLOT)


@gluon.jit
def _receive_norm(
    x,
    residual,
    weight,
    out,
    residual_out,
    scale,
    inboxes,
    calls,
    err,
    row,
    RANK: gl.constexpr,
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
    NORM: gl.constexpr,
    HIDDEN: gl.constexpr,
    GEMMA: gl.constexpr,
    EPS: gl.constexpr,
):
    """Receives one row: wave w takes slab w (chunks 2w and 2w + 1 of the row; the
    values past the row are masked). Then stores residual_out = the sum + residual and
    out = RMSNorm(residual_out) as bf16 (NORM 1) or as per-token FP8 with its fp32
    scale (NORM 2). The arithmetic is that of add_rmsnorm_quant_kernel<256, 24>
    (round_before_quant for FP8)."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    ROW: gl.constexpr = _row_chunks(HIDDEN, LINES)
    gl.static_assert(2 * gl.num_warps() >= ROW and gl.num_warps() <= 32)
    pair = gl.arange(0, 2, layout=gl.SliceLayout(0, WORDS))[None, :]
    col4 = gl.arange(0, 4, layout=gl.SliceLayout(0, LINE))[None, :]
    wave, line = _wave_lines(LINES, WORDS)
    k = 2 * wave  # the first chunk of the slab of the wave
    c = ROW * row + k
    wa = k * 6 * LINES + 2 * line + pair  # plane j of chunk k is at wa + 2 LINES j
    wb = wa + 6 * LINES  # and of chunk k + 1 at wb + 2 LINES j
    va = row * HIDDEN + wa
    vb = row * HIDDEN + wb
    m0, m1, m2 = _planes(k, line, HIDDEN, LINES)
    m3, m4, m5 = _planes(k + 1, line, HIDDEN, LINES)
    # The residual and the weight are local: load them before the wait.
    r0 = gl.amd.cdna5.buffer_load(residual, va, mask=m0, other=0)
    r1 = gl.amd.cdna5.buffer_load(residual, va + 2 * LINES, mask=m1, other=0)
    r2 = gl.amd.cdna5.buffer_load(residual, va + 4 * LINES, mask=m2, other=0)
    r3 = gl.amd.cdna5.buffer_load(residual, vb, mask=m3, other=0)
    r4 = gl.amd.cdna5.buffer_load(residual, vb + 2 * LINES, mask=m4, other=0)
    r5 = gl.amd.cdna5.buffer_load(residual, vb + 4 * LINES, mask=m5, other=0)
    w0 = gl.amd.cdna5.buffer_load(weight, wa, mask=m0, other=0)
    w1 = gl.amd.cdna5.buffer_load(weight, wa + 2 * LINES, mask=m1, other=0)
    w2 = gl.amd.cdna5.buffer_load(weight, wa + 4 * LINES, mask=m2, other=0)
    w3 = gl.amd.cdna5.buffer_load(weight, wb, mask=m3, other=0)
    w4 = gl.amd.cdna5.buffer_load(weight, wb + 2 * LINES, mask=m4, other=0)
    w5 = gl.amd.cdna5.buffer_load(weight, wb + 4 * LINES, mask=m5, other=0)
    ok_a = k < ROW
    ok_b = k + 1 < ROW
    call_a, mine_a = _lines(x, va, m0, m1, m2, 2 * LINES, calls + 2 * c + 1, ok_a)
    call_b, mine_b = _lines(x, vb, m3, m4, m5, 2 * LINES, calls + 2 * c + 3, ok_b)
    wave4, line4 = _wave_lines(LINES, LINE)
    pos = 4 * ((ROW * row + 2 * wave4) * LINES + line4) + col4
    s0, s1, s2, s3, s4, s5 = _receive_pair(
        inboxes,
        pos,
        call_a,
        call_b,
        mine_a,
        mine_b,
        2 * wave4 < ROW,
        2 * wave4 + 1 < ROW,
        err,
        RANK,
        SLOT,
        MAXIT,
    )
    # As the split path: the all-reduce result is bf16, the residual add is fp32.
    s0 = s0.to(gl.bfloat16).to(gl.float32) + r0.to(gl.float32)
    s1 = s1.to(gl.bfloat16).to(gl.float32) + r1.to(gl.float32)
    s2 = s2.to(gl.bfloat16).to(gl.float32) + r2.to(gl.float32)
    s3 = s3.to(gl.bfloat16).to(gl.float32) + r3.to(gl.float32)
    s4 = s4.to(gl.bfloat16).to(gl.float32) + r4.to(gl.float32)
    s5 = s5.to(gl.bfloat16).to(gl.float32) + r5.to(gl.float32)
    gl.amd.cdna5.buffer_store(s0.to(gl.bfloat16), residual_out, va, mask=m0)
    gl.amd.cdna5.buffer_store(s1.to(gl.bfloat16), residual_out, va + 2 * LINES, mask=m1)
    gl.amd.cdna5.buffer_store(s2.to(gl.bfloat16), residual_out, va + 4 * LINES, mask=m2)
    gl.amd.cdna5.buffer_store(s3.to(gl.bfloat16), residual_out, vb, mask=m3)
    gl.amd.cdna5.buffer_store(s4.to(gl.bfloat16), residual_out, vb + 2 * LINES, mask=m4)
    gl.amd.cdna5.buffer_store(s5.to(gl.bfloat16), residual_out, vb + 4 * LINES, mask=m5)

    sq = _square_sum(s0, s1, s2, s3, s4, s5)
    # rsqrtf(sum * rcp(n) + eps), with eps added in fp64, as add_rmsnorm_quant_kernel.
    mean = sq * _asm("v_rcp_f32", gl.full([], HIDDEN, gl.float32))
    rstd = _asm("v_rsq_f32", (mean.to(gl.float64) + EPS).to(gl.float32))
    offset: gl.constexpr = 1.0 if GEMMA else 0.0
    y0 = s0 * rstd * (w0.to(gl.float32) + offset)
    y1 = s1 * rstd * (w1.to(gl.float32) + offset)
    y2 = s2 * rstd * (w2.to(gl.float32) + offset)
    y3 = s3 * rstd * (w3.to(gl.float32) + offset)
    y4 = s4 * rstd * (w4.to(gl.float32) + offset)
    y5 = s5 * rstd * (w5.to(gl.float32) + offset)
    if NORM == 1:
        gl.amd.cdna5.buffer_store(y0.to(gl.bfloat16), out, va, mask=m0)
        gl.amd.cdna5.buffer_store(y1.to(gl.bfloat16), out, va + 2 * LINES, mask=m1)
        gl.amd.cdna5.buffer_store(y2.to(gl.bfloat16), out, va + 4 * LINES, mask=m2)
        gl.amd.cdna5.buffer_store(y3.to(gl.bfloat16), out, vb, mask=m3)
        gl.amd.cdna5.buffer_store(y4.to(gl.bfloat16), out, vb + 2 * LINES, mask=m4)
        gl.amd.cdna5.buffer_store(y5.to(gl.bfloat16), out, vb + 4 * LINES, mask=m5)
    else:
        gl.static_assert(NORM == 2)
        # Quantize the bf16 norm output, as the split norm -> per-token quant path.
        y0 = y0.to(gl.bfloat16).to(gl.float32)
        y1 = y1.to(gl.bfloat16).to(gl.float32)
        y2 = y2.to(gl.bfloat16).to(gl.float32)
        y3 = y3.to(gl.bfloat16).to(gl.float32)
        y4 = y4.to(gl.bfloat16).to(gl.float32)
        y5 = y5.to(gl.bfloat16).to(gl.float32)
        top = gl.maximum(gl.abs(y0), gl.abs(y1))
        top = gl.maximum(top, gl.maximum(gl.abs(y2), gl.abs(y3)))
        top = gl.maximum(top, gl.maximum(gl.abs(y4), gl.abs(y5)))
        top = gl.maximum(gl.max(gl.max(top, axis=1), axis=0), 0.0)  # max is exact
        row_scale = top * (1.0 / 448.0)  # 448 is the fp8 maximum
        gl.store(scale + row, row_scale)
        inv = _asm("v_rcp_f32", row_scale)
        q = gl.clamp(y0 * inv, -448.0, 448.0).to(gl.float8e4nv)
        gl.amd.cdna5.buffer_store(q, out, va, mask=m0)
        q = gl.clamp(y1 * inv, -448.0, 448.0).to(gl.float8e4nv)
        gl.amd.cdna5.buffer_store(q, out, va + 2 * LINES, mask=m1)
        q = gl.clamp(y2 * inv, -448.0, 448.0).to(gl.float8e4nv)
        gl.amd.cdna5.buffer_store(q, out, va + 4 * LINES, mask=m2)
        q = gl.clamp(y3 * inv, -448.0, 448.0).to(gl.float8e4nv)
        gl.amd.cdna5.buffer_store(q, out, vb, mask=m3)
        q = gl.clamp(y4 * inv, -448.0, 448.0).to(gl.float8e4nv)
        gl.amd.cdna5.buffer_store(q, out, vb + 2 * LINES, mask=m4)
        q = gl.clamp(y5 * inv, -448.0, 448.0).to(gl.float8e4nv)
        gl.amd.cdna5.buffer_store(q, out, vb + 4 * LINES, mask=m5)


_plain_repr = make_kernel_repr("_all_reduce_kernel", ["RANK", "LINES"])
_norm_repr = make_kernel_repr(
    "_all_reduce_kernel", ["RANK", "LINES", "NORM", "GEMMA", "HIDDEN"]
)


def _all_reduce_repr(specialization):
    """The name of the plain kernel stays the same; NORM kernels add their keys."""
    if specialization.constants.get("NORM", 0):
        return _norm_repr(specialization)
    return _plain_repr(specialization)


@gluon.jit(repr=_all_reduce_repr, do_not_specialize=["n", "programs"])
def _all_reduce_kernel(
    x,
    y,
    inboxes,
    calls,
    err,
    n,
    programs,
    residual,
    weight,
    out,
    scale,
    RANK: gl.constexpr,
    LINES: gl.constexpr,
    SLOT: gl.constexpr,
    MAXIT: gl.constexpr,
    NORM: gl.constexpr,
    HIDDEN: gl.constexpr,
    GEMMA: gl.constexpr,
    EPS: gl.constexpr,
):
    """y = the sum of the n values of x over the 4 ranks. Programs 0 to C-1 send the C
    chunks, one each: a load after a remote store waits for its ack. Programs C to
    C+P-1 receive chunks p, p + P, ... (P = programs). The GPU starts programs in
    order, so a waiting receiver cannot block a sender.

    NORM 0 ignores residual, weight, out and scale. With NORM 1 or 2, y is
    residual_out, there are P rows of HIDDEN values, and the W-wave programs 0 to
    S - 1 send W chunks each (S = cdiv(P cdiv(HIDDEN, 6 LINES), W)); the next programs
    receive one row each."""
    WORDS: gl.constexpr = gl.BlockedLayout([1, 2], [32, 1], [gl.num_warps(), 1], [1, 0])
    LINE: gl.constexpr = gl.BlockedLayout([1, 4], [32, 1], [gl.num_warps(), 1], [1, 0])
    num_chunks = gl.cdiv(n, 6 * LINES)
    pid = gl.program_id(0)
    if NORM > 0:
        ROW: gl.constexpr = _row_chunks(HIDDEN, LINES)
        senders = gl.cdiv(programs * ROW, gl.num_warps())
        if pid < senders:
            _send_norm(x, inboxes, calls, pid, programs, RANK, LINES, SLOT, HIDDEN)
        else:
            _receive_norm(
                x,
                residual,
                weight,
                out,
                y,
                scale,
                inboxes,
                calls,
                err,
                pid - senders,
                RANK,
                LINES,
                SLOT,
                MAXIT,
                NORM,
                HIDDEN,
                GEMMA,
                EPS,
            )
    else:
        # Unit u of a chunk has the values u, U + u and 2U + u (U = 2 LINES units), so that
        # the loads of x are coalesced.
        line = gl.arange(0, LINES, layout=gl.SliceLayout(1, WORDS))[:, None]
        pair = gl.arange(0, 2, layout=gl.SliceLayout(0, WORDS))[
            None, :
        ]  # 2 units a line
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
