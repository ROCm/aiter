"""Gated-delta prefill with FP32 chunk algebra, for AMD MI355.

All sequence schedules are computed from runtime GPU sequence lengths. Small
diagonal solves are wave-local; state propagation never truncates history or
quantizes the recurrent state. The convolution, Q/K normalization, core output,
and normalized output retain the reference's BF16 rounding boundaries.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _add(a, b):
    return a + b


@gluon.jit
def _packed_channel(channel):
    return gl.where(
        channel < 512,
        (channel // 128) * 768 + channel % 128,
        gl.where(
            channel < 1024,
            ((channel - 512) // 128) * 768 + 128 + channel % 128,
            ((channel - 1024) // 256) * 768 + 256 + channel % 256,
        ),
    )


_prepare_inputs_tiled_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_prepare_inputs_tiled",
    ["M", "BATCH", "ROWS", "CHANNELS_PER_LANE"],
)


@gluon.jit(repr=_prepare_inputs_tiled_repr)
def _prepare_inputs_tiled(
    projected,
    ba,
    conv_state,
    indices,
    starts,
    initial,
    conv_weight,
    conv_bias,
    prepared,
    gates,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    ROWS: gl.constexpr,
    CHANNELS_PER_LANE: gl.constexpr,
):
    """Several independent width-128 reductions per wave, sharing weights."""
    group = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, CHANNELS_PER_LANE], [4, 16], [4, 1], [1, 0]
    )
    row_layout: gl.constexpr = gl.SliceLayout(1, layout)
    col_layout: gl.constexpr = gl.SliceLayout(0, layout)
    token = gl.program_id(0) * ROWS + gl.arange(0, ROWS, row_layout)
    col = gl.arange(0, 128, col_layout)
    channel = group * 128 + col
    packed = _packed_channel(channel)
    lo = gl.full((ROWS,), 0, gl.int32, row_layout)
    hi = gl.full((ROWS,), BATCH, gl.int32, row_layout)
    for depth in gl.static_range(0, 7):
        if (1 << depth) < BATCH:
            mid = (lo + hi) // 2
            before = token < gl.load(starts + mid)
            hi = gl.where(before, mid, hi)
            lo = gl.where(before, lo, mid)
    begin = gl.load(starts + lo)
    slot = gl.load(indices + lo)
    cached = gl.load(initial + lo)
    local = token - begin
    valid = token < M
    x0 = gl.load(
        projected + (token[:, None] - 3) * 3072 + packed[None, :],
        valid[:, None] & (local[:, None] >= 3),
        0,
    ).to(gl.float32)
    x1 = gl.load(
        projected + (token[:, None] - 2) * 3072 + packed[None, :],
        valid[:, None] & (local[:, None] >= 2),
        0,
    ).to(gl.float32)
    x2 = gl.load(
        projected + (token[:, None] - 1) * 3072 + packed[None, :],
        valid[:, None] & (local[:, None] >= 1),
        0,
    ).to(gl.float32)
    x3 = gl.load(
        projected + token[:, None] * 3072 + packed[None, :], valid[:, None], 0
    ).to(gl.float32)
    history = conv_state + (slot[:, None] * 2048 + channel[None, :]) * 3
    h0 = gl.load(
        history + local[:, None],
        valid[:, None] & (local[:, None] < 3) & cached[:, None],
        0,
    ).to(gl.float32)
    h1 = gl.load(
        history + local[:, None] + 1,
        valid[:, None] & (local[:, None] < 2) & cached[:, None],
        0,
    ).to(gl.float32)
    h2 = gl.load(
        history + local[:, None] + 2,
        valid[:, None] & (local[:, None] < 1) & cached[:, None],
        0,
    ).to(gl.float32)
    x0 = gl.where(local[:, None] < 3, h0, x0)
    x1 = gl.where(local[:, None] < 2, h1, x1)
    x2 = gl.where(local[:, None] < 1, h2, x2)
    w0 = gl.load(conv_weight + channel * 4).to(gl.float32)
    w1 = gl.load(conv_weight + channel * 4 + 1).to(gl.float32)
    w2 = gl.load(conv_weight + channel * 4 + 2).to(gl.float32)
    w3 = gl.load(conv_weight + channel * 4 + 3).to(gl.float32)
    bias = gl.load(conv_bias + channel).to(gl.float32)
    convolved = (
        bias[None, :]
        + x0 * w0[None, :]
        + x1 * w1[None, :]
        + x2 * w2[None, :]
        + x3 * w3[None, :]
    )
    value = gl.div_rn(convolved, 1.0 + libdevice.exp(-convolved)).to(gl.bfloat16)
    if group < 8:
        value32 = value.to(gl.float32)
        value = (value32 * gl.rsqrt(gl.sum(value32 * value32, 1)[:, None] + 1.0e-6)).to(
            gl.bfloat16
        )
    else:
        head = group - 8
        ba_base = token * 16 + (head // 2) * 4 + head % 2
        bv = gl.load(ba + ba_base, valid, 0).to(gl.float32)
        beta = (1.0 / (1.0 + gl.exp(-bv))).to(gl.bfloat16).to(gl.float32)
        gl.store(gates + token * 8 + head, beta, valid)
    gl.store(prepared + token[:, None] * 2048 + channel[None, :], value, valid[:, None])


_update_conv_state_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_update_conv_state", []
)


@gluon.jit(repr=_update_conv_state_repr)
def _update_conv_state(projected, state, indices, starts, initial):
    seq = gl.program_id(0)
    channel = gl.program_id(1) * 256 + gl.arange(
        0, 256, layout=gl.BlockedLayout([1], [64], [4], [0])
    )
    begin = gl.load(starts + seq)
    end = gl.load(starts + seq + 1)
    slot = gl.load(indices + seq)
    cached = gl.load(initial + seq)
    length = end - begin
    packed = _packed_channel(channel)
    history = state + (slot * 2048 + channel) * 3
    # Read every old history member before any store, including short sequences.
    h0 = gl.load(history + length, (length < 3) & cached, 0)
    h1 = gl.load(history + length + 1, (length < 2) & cached, 0)
    h2 = gl.load(history + length + 2, (length < 1) & cached, 0)
    x0 = gl.load(projected + (end - 3) * 3072 + packed, length >= 3, 0)
    x1 = gl.load(projected + (end - 2) * 3072 + packed, length >= 2, 0)
    x2 = gl.load(projected + (end - 1) * 3072 + packed, length >= 1, 0)
    gl.store(history, gl.where(length < 3, h0, x0))
    gl.store(history + 1, gl.where(length < 2, h1, x1))
    gl.store(history + 2, gl.where(length < 1, h2, x2))


@gluon.jit
def _matrix_product(a, b, WM: gl.constexpr = 2):
    """BF16 for logical BF16 Gram products; native FP32 for state algebra."""
    if a.dtype == gl.bfloat16:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 32],
            transposed=True,
            warps_per_cta=[WM, 4 // WM],
        )
        kw: gl.constexpr = 8
    else:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 4],
            transposed=True,
            warps_per_cta=[WM, 4 // WM],
        )
        kw: gl.constexpr = 1
    aa = gl.convert_layout(a, gl.DotOperandLayout(0, mma, kw))
    bb = gl.convert_layout(b, gl.DotOperandLayout(1, mma, kw))
    acc = gl.zeros((a.shape[0], b.shape[1]), gl.float32, mma)
    return gl.amd.cdna4.mfma(aa, bb, acc)


@gluon.jit
def _store_packed_operand(pointer, value):
    """Pack [M,K] as [K/16, M/16, K-lane, M-lane, K-register]."""
    M: gl.constexpr = value.shape[0]
    K: gl.constexpr = value.shape[1]
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    flat: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    x = gl.convert_layout(value, layout)
    x = (
        x.reshape((M // 16, 16, K // 16, 4, 4))
        .permute((2, 0, 4, 1, 3))
        .reshape((M * K,))
    )
    x = gl.convert_layout(x, flat)
    gl.store(pointer + gl.arange(0, M * K, flat), x)


@gluon.jit
def _load_packed_operand(pointer, M: gl.constexpr, K: gl.constexpr, WM: gl.constexpr):
    flat: gl.constexpr = gl.SliceLayout(
        0, gl.BlockedLayout([1, 4], [1, 64], [4 // WM, WM], [1, 0])
    )
    x = gl.load(pointer + gl.arange(0, M * K, flat))
    x = x.reshape((K // 16, M // 16, 4, 16, 4)).permute((1, 3, 0, 4, 2)).reshape((M, K))
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[WM, 4 // WM]
    )
    return gl.convert_layout(x, gl.DotOperandLayout(0, mma, 1))


@gluon.jit
def _inverse16(power):
    """Inverse of I-P for a strictly lower triangular 16-by-16 P."""
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    p = gl.convert_layout(power, layout)
    r = gl.arange(0, 16, gl.SliceLayout(1, layout))
    c = gl.arange(0, 16, gl.SliceLayout(0, layout))
    inverse = p + gl.where(r[:, None] == c[None, :], 1.0, 0.0)
    for level in gl.static_range(1, 4):
        p = gl.convert_layout(_matrix_product(p, p), layout)
        inverse = inverse + gl.convert_layout(_matrix_product(p, inverse), layout)
    return inverse


@gluon.jit
def _assemble_inverse(top, bottom, lower):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    N: gl.constexpr = top.shape[0]
    zero = gl.full((N, N), 0.0, gl.float32, layout)
    upper_rows = gl.join(top, zero).permute((0, 2, 1)).reshape((N, 2 * N))
    lower_rows = gl.join(lower, bottom).permute((0, 2, 1)).reshape((N, 2 * N))
    combined = (
        gl.join(upper_rows, lower_rows).permute((2, 0, 1)).reshape((2 * N, 2 * N))
    )
    return gl.convert_layout(combined, layout)


@gluon.jit
def _inverse32(power):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    p = gl.convert_layout(power, layout)
    top = _inverse16(gl.amd.slice(p, (16, 16), (0, 0)))
    bottom = _inverse16(gl.amd.slice(p, (16, 16), (16, 16)))
    lower = gl.amd.slice(p, (16, 16), (16, 0))
    lower = gl.convert_layout(_matrix_product(lower, top), layout)
    lower = gl.convert_layout(_matrix_product(bottom, lower), layout)
    return _assemble_inverse(top, bottom, lower)


@gluon.jit
def _four_diagonal_inverses(power):
    """Each wave solves one 16-by-16 diagonal block without CTA exchanges."""
    small: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    d0 = gl.amd.slice(power, (16, 16), (0, 0))
    d1 = gl.amd.slice(power, (16, 16), (16, 16))
    d2 = gl.amd.slice(power, (16, 16), (32, 32))
    d3 = gl.amd.slice(power, (16, 16), (48, 48))
    pair0 = gl.join(d0, d1).permute((2, 0, 1))
    pair1 = gl.join(d2, d3).permute((2, 0, 1))
    diagonal = gl.join(pair0, pair1).permute((3, 0, 1, 2)).reshape((4, 16, 16))
    layout: gl.constexpr = gl.BlockedLayout([1, 1, 4], [1, 16, 4], [4, 1, 1], [2, 1, 0])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[4, 1, 1]
    )
    diagonal = gl.convert_layout(diagonal, layout)
    eye2d = gl.where(
        gl.arange(0, 16, gl.SliceLayout(1, small))[:, None]
        == gl.arange(0, 16, gl.SliceLayout(0, small))[None, :],
        1.0,
        0.0,
    )
    eye3d = gl.convert_layout(eye2d, gl.SliceLayout(0, layout))[None, :, :]
    inverse = diagonal + eye3d
    p = diagonal
    for level in gl.static_range(1, 4):
        p = gl.amd.cdna4.mfma(
            gl.convert_layout(p, gl.DotOperandLayout(0, mma, 1)),
            gl.convert_layout(p, gl.DotOperandLayout(1, mma, 1)),
            gl.zeros((4, 16, 16), gl.float32, mma),
        )
        inverse = inverse + gl.convert_layout(
            gl.amd.cdna4.mfma(
                gl.convert_layout(p, gl.DotOperandLayout(0, mma, 1)),
                gl.convert_layout(inverse, gl.DotOperandLayout(1, mma, 1)),
                gl.zeros((4, 16, 16), gl.float32, mma),
            ),
            layout,
        )
        p = gl.convert_layout(p, layout)
    gather_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [1, 4, 16], [1, 4, 1], [2, 1, 0]
    )
    inverse = gl.convert_layout(inverse, gather_layout)
    out0 = gl.amd.slice(inverse, (1, 16, 16), (0, 0, 0)).reshape((16, 16))
    out1 = gl.amd.slice(inverse, (1, 16, 16), (1, 0, 0)).reshape((16, 16))
    out2 = gl.amd.slice(inverse, (1, 16, 16), (2, 0, 0)).reshape((16, 16))
    out3 = gl.amd.slice(inverse, (1, 16, 16), (3, 0, 0)).reshape((16, 16))
    return (
        gl.convert_layout(out0, small),
        gl.convert_layout(out1, small),
        gl.convert_layout(out2, small),
        gl.convert_layout(out3, small),
    )


@gluon.jit
def _block_inverse(power, C: gl.constexpr):
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    if C == 16:
        return _inverse16(power)
    elif C == 32:
        return _inverse32(power)
    else:
        p = gl.convert_layout(power, layout)
        d0, d1, d2, d3 = _four_diagonal_inverses(p)
        p10 = gl.amd.slice(p, (16, 16), (16, 0))
        p32 = gl.amd.slice(p, (16, 16), (48, 32))
        low0 = gl.convert_layout(_matrix_product(p10, d0), layout)
        low0 = gl.convert_layout(_matrix_product(d1, low0), layout)
        low1 = gl.convert_layout(_matrix_product(p32, d2), layout)
        low1 = gl.convert_layout(_matrix_product(d3, low1), layout)
        top = _assemble_inverse(d0, d1, low0)
        bottom = _assemble_inverse(d2, d3, low1)
        lower = gl.amd.slice(p, (32, 32), (32, 0))
        lower = gl.convert_layout(_matrix_product(lower, top), layout)
        lower = gl.convert_layout(_matrix_product(bottom, lower), layout)
        return _assemble_inverse(top, bottom, lower)


_chunk_offsets_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_chunk_offsets", ["BATCH", "BLOCK", "C"]
)


@gluon.jit(repr=_chunk_offsets_repr)
def _chunk_offsets(
    starts, offsets, BATCH: gl.constexpr, BLOCK: gl.constexpr, C: gl.constexpr
):
    seq = gl.program_id(0)
    i = gl.arange(0, BLOCK, layout=gl.BlockedLayout([1], [64], [1], [0]))
    lo = gl.load(starts + i, i < BATCH, 0)
    hi = gl.load(starts + i + 1, i < BATCH, 0)
    n = gl.cdiv(hi - lo, C)
    gl.store(offsets + seq, gl.sum(gl.where(i < seq, n, 0), 0))


@gluon.jit
def _find_sequence(offsets, item, BATCH: gl.constexpr):
    lo = 0
    hi = BATCH
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        before = item < gl.load(offsets + mid)
        hi = gl.where(before, mid, hi)
        lo = gl.where(before, lo, mid)
    return lo


@gluon.jit
def _chunk_location(starts, offsets, chunk, BATCH: gl.constexpr, C: gl.constexpr):
    seq = _find_sequence(offsets, chunk, BATCH)
    base = gl.load(starts + seq) + (chunk - gl.load(offsets + seq)) * C
    return base, gl.load(starts + seq + 1)


_prepare_chunk_factors_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_prepare_chunk_factors", ["BATCH", "C", "PLANES"]
)


@gluon.jit(repr=_prepare_chunk_factors_repr)
def _prepare_chunk_factors(
    prepared,
    ba,
    gates,
    a_log,
    dt_bias,
    starts,
    offsets,
    W,
    U,
    G,
    BATCH: gl.constexpr,
    C: gl.constexpr,
    PLANES: gl.constexpr = 0,
):
    """Form chunk updates as U-W@H by inverting a unit-lower delta system.

    G contains C prefix decays followed by C*C interval decays. Optional packed
    planes contain W, the final-decayed transposed keys, and prefix-scaled Q.
    """
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    total = gl.load(offsets + BATCH)
    if chunk < total:
        begin, end = _chunk_location(starts, offsets, chunk, BATCH, C)
        layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        t = gl.arange(0, C, gl.SliceLayout(1, layout))
        k = gl.arange(0, 128, gl.SliceLayout(0, layout))
        tokens = begin + t
        key = gl.load(
            prepared + tokens[:, None] * 2048 + 512 + (head // 2) * 128 + k[None, :],
            tokens[:, None] < end,
            0,
        )
        beta = gl.load(gates + tokens * 8 + head, tokens < end, 0)
        ba_idx = tokens * 16 + (head // 2) * 4 + head % 2
        av = gl.load(ba + ba_idx + 2, tokens < end, 0).to(gl.float32) + gl.load(
            dt_bias + head
        ).to(gl.float32)
        aw = gl.exp(gl.load(a_log + head))
        softplus = gl.where(av <= 20.0, gl.log(1.0 + gl.exp(av)), av)
        g = gl.where(tokens < end, -aw * softplus, 0.0)
        scan_g = gl.convert_layout(g, gl.BlockedLayout([1], [64], [4], [0]))
        cumulative = gl.convert_layout(
            gl.associative_scan(scan_g, 0, _add), gl.SliceLayout(1, layout)
        )
        gl.store(G + (chunk * 8 + head) * (C * C + C) + t, gl.exp(cumulative))
        gram = _matrix_product(key, gl.permute(key, (1, 0)))
        ii = gl.arange(0, C, gl.SliceLayout(1, layout))
        jj = gl.arange(0, C, gl.SliceLayout(0, layout))
        sl: gl.constexpr = gl.BlockedLayout([1, 1], [32, 2], [1, 4], [0, 1])
        si = gl.arange(0, C, gl.SliceLayout(1, sl))
        sj = gl.arange(0, C, gl.SliceLayout(0, sl))
        sg = gl.convert_layout(g, gl.SliceLayout(1, sl))
        # Direct interval scans also handle zero decay without inf-inf or 0/0.
        intervals = gl.where(si[:, None] > sj[None, :], sg[:, None], 0.0)
        decay = gl.convert_layout(
            gl.exp(gl.associative_scan(intervals, 0, _add)), layout
        )
        bi = gl.convert_layout(beta, gl.SliceLayout(1, layout))
        lower = ii[:, None] > jj[None, :]
        gl.store(
            G + (chunk * 8 + head) * (C * C + C) + C + ii[:, None] * C + jj[None, :],
            decay,
        )
        if PLANES >= 2:
            last_decay = gl.sum(gl.where(ii[:, None] == C - 1, decay, 0.0), 0)
            last_decay = gl.convert_layout(last_decay, gl.SliceLayout(1, layout))
        power = gl.where(
            lower, -bi[:, None] * gl.convert_layout(gram, layout) * decay, 0.0
        )
        inverse = gl.convert_layout(_block_inverse(power, C), layout)
        w = _matrix_product(
            inverse, (beta * gl.exp(cumulative))[:, None] * key.to(gl.float32)
        )
        if PLANES:
            packed_base = W + (chunk * 8 + head) * PLANES * C * 128
            _store_packed_operand(packed_base, w)
        else:
            gl.store(
                W + ((chunk * 8 + head) * C + t[:, None]) * 128 + k[None, :],
                gl.convert_layout(w, layout),
            )
        val = gl.load(
            prepared + tokens[:, None] * 2048 + 1024 + head * 128 + k[None, :],
            tokens[:, None] < end,
            0,
        ).to(gl.float32)
        u = _matrix_product(inverse, beta[:, None] * val)
        gl.store(
            U + ((chunk * 8 + head) * C + t[:, None]) * 128 + k[None, :],
            gl.convert_layout(u, layout),
        )

        if PLANES >= 2:
            key_tail = gl.load(
                prepared
                + tokens[:, None] * 2048
                + 512
                + (head // 2) * 128
                + k[None, :],
                tokens[:, None] < end,
                0,
            ).to(gl.float32)
            _store_packed_operand(
                packed_base + C * 128,
                gl.permute(key_tail * last_decay[:, None], (1, 0)),
            )
        if PLANES == 3:
            query = gl.load(
                prepared + tokens[:, None] * 2048 + (head // 2) * 128 + k[None, :],
                tokens[:, None] < end,
                0,
            ).to(gl.float32)
            _store_packed_operand(
                packed_base + 2 * C * 128, query * gl.exp(cumulative)[:, None]
            )


_propagate_chunks_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_propagate_chunks", ["C", "BV", "WM", "HISTORY", "PLANES"]
)


@gluon.jit(repr=_propagate_chunks_repr)
def _propagate_chunks(
    prepared,
    states,
    indices,
    starts,
    offsets,
    W,
    U,
    G,
    updates,
    carry_buffer,
    C: gl.constexpr,
    BV: gl.constexpr,
    WM: gl.constexpr,
    HISTORY: gl.constexpr = False,
    PLANES: gl.constexpr = 0,
):
    VWIDTH: gl.constexpr = max(16, BV)
    seq = gl.program_id(0)
    head = gl.program_id(1)
    tile = gl.program_id(2)
    slot = gl.load(indices + seq)
    first = gl.load(offsets + seq)
    last = gl.load(offsets + seq + 1)
    seq_begin = gl.load(starts + seq)
    seq_end = gl.load(starts + seq + 1)
    lm: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    lv: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WM, 4 // WM], [0, 1])
    k = gl.arange(0, 128, gl.SliceLayout(1, lv))
    v = tile * BV + gl.arange(0, VWIDTH, gl.SliceLayout(0, lv))
    state_ptr = states + ((slot * 8 + head) * 128 + v[None, :]) * 128 + k[:, None]
    h = gl.load(state_ptr, (v[None, :] < (tile + 1) * BV) & (v[None, :] < 128), 0)
    t = gl.arange(0, C, gl.SliceLayout(1, lm))
    kk = gl.arange(0, 128, gl.SliceLayout(0, lm))
    tv = gl.arange(0, C, gl.SliceLayout(1, lv))
    for chunk in range(first, last):
        token = seq_begin + (chunk - first) * C + t
        if PLANES:
            packed_base = W + (chunk * 8 + head) * PLANES * C * 128
            w = _load_packed_operand(packed_base, C, 128, WM)
        else:
            w = gl.load(W + ((chunk * 8 + head) * C + t[:, None]) * 128 + kk[None, :])
        u = gl.load(
            U + ((chunk * 8 + head) * C + tv[:, None]) * 128 + v[None, :],
            v[None, :] < (tile + 1) * BV,
            0,
        )
        du = u - gl.convert_layout(_matrix_product(w, h, WM), lv)
        gl.store(
            updates + ((chunk * 8 + head) * C + tv[:, None]) * 128 + v[None, :],
            du,
            v[None, :] < (tile + 1) * BV,
        )
        if HISTORY:
            gl.store(
                carry_buffer
                + ((chunk * 8 + head) * 128 + k[:, None]) * 128
                + v[None, :],
                h,
                v[None, :] < (tile + 1) * BV,
            )
        else:
            if PLANES == 3:
                qp = _load_packed_operand(packed_base + 2 * C * 128, C, 128, WM)
            else:
                q = gl.load(
                    prepared + token[:, None] * 2048 + (head // 2) * 128 + kk[None, :],
                    token[:, None] < seq_end,
                    0,
                ).to(gl.float32)
                p = gl.load(G + (chunk * 8 + head) * (C * C + C) + t)
                qp = q * p[:, None]
            base = gl.convert_layout(_matrix_product(qp, h, WM), lv)
            gl.store(
                carry_buffer
                + ((chunk * 8 + head) * C + tv[:, None]) * 128
                + v[None, :],
                base,
                v[None, :] < (tile + 1) * BV,
            )
        if PLANES >= 2:
            key_t = _load_packed_operand(packed_base + C * 128, 128, C, WM)
        else:
            key = gl.load(
                prepared
                + token[:, None] * 2048
                + 512
                + (head // 2) * 128
                + kk[None, :],
                token[:, None] < seq_end,
                0,
            ).to(gl.float32)
            last_decay = gl.load(
                G + (chunk * 8 + head) * (C * C + C) + C + (C - 1) * C + t
            )
            key_t = gl.permute(key * last_decay[:, None], (1, 0))
        final_decay = gl.load(G + (chunk * 8 + head) * (C * C + C) + C - 1)
        step = gl.convert_layout(_matrix_product(key_t, du, WM), lv)
        h = final_decay * h + step
    gl.store(state_ptr, h, (v[None, :] < (tile + 1) * BV) & (v[None, :] < 128))


_output_norm_quant_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_output_norm_quant", ["BATCH", "C", "HISTORY"]
)


@gluon.jit(repr=_output_norm_quant_repr)
def _output_norm_quant(
    prepared,
    projected,
    norm_weight,
    starts,
    offsets,
    G,
    updates,
    carry_buffer,
    normalized,
    quantized,
    scales,
    scale,
    eps,
    BATCH: gl.constexpr,
    C: gl.constexpr,
    HISTORY: gl.constexpr = False,
):
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    total = gl.load(offsets + BATCH)
    if chunk < total:
        begin, end = _chunk_location(starts, offsets, chunk, BATCH, C)
        lm: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        t = gl.arange(0, C, gl.SliceLayout(1, lm))
        k = gl.arange(0, 128, gl.SliceLayout(0, lm))
        token = begin + t
        q = gl.load(
            prepared + token[:, None] * 2048 + (head // 2) * 128 + k[None, :],
            token[:, None] < end,
            0,
        )
        key = gl.load(
            prepared + token[:, None] * 2048 + 512 + (head // 2) * 128 + k[None, :],
            token[:, None] < end,
            0,
        )
        gram = gl.convert_layout(_matrix_product(q, gl.permute(key, (1, 0))), lm)
        j = gl.arange(0, C, gl.SliceLayout(0, lm))
        decay = gl.load(
            G + (chunk * 8 + head) * (C * C + C) + C + t[:, None] * C + j[None, :]
        )
        factor = gl.where(t[:, None] >= j[None, :], gram * decay, 0.0)
        u = gl.load(updates + ((chunk * 8 + head) * C + t[:, None]) * 128 + k[None, :])
        correction = gl.convert_layout(_matrix_product(factor, u), lm)
        if HISTORY:
            hk = gl.arange(0, 128, gl.SliceLayout(1, lm))
            h = gl.load(
                carry_buffer
                + ((chunk * 8 + head) * 128 + hk[:, None]) * 128
                + k[None, :]
            )
            p = gl.load(G + (chunk * 8 + head) * (C * C + C) + t)
            base = gl.convert_layout(
                _matrix_product(q.to(gl.float32) * p[:, None], h), lm
            )
        else:
            base = gl.load(
                carry_buffer + ((chunk * 8 + head) * C + t[:, None]) * 128 + k[None, :]
            )
        y = (base + correction) * scale
        # Preserve both BF16 materialization boundaries before FP8 quantization.
        x = y.to(gl.bfloat16).to(gl.float32)
        gate = gl.load(
            projected
            + token[:, None] * 3072
            + (head // 2) * 768
            + 512
            + (head % 2) * 128
            + k[None, :],
            token[:, None] < end,
            0,
        ).to(gl.float32)
        weight = gl.load(norm_weight + k).to(gl.float32)
        rms = gl.rsqrt(gl.sum(x * x, 1) / 128 + eps)
        sigmoid = 1.0 / (1.0 + gl.exp(-gate))
        norm = (x * rms[:, None] * weight[None, :] * gate * sigmoid).to(gl.bfloat16)
        out_ptr = (token[:, None] * 8 + head) * 128 + k[None, :]
        gl.store(normalized + out_ptr, norm, token[:, None] < end)
        rounded = norm.to(gl.float32)
        maximum: gl.constexpr = (
            448.0 if quantized.dtype.element_ty == gl.float8e4nv else 224.0
        )
        quant_scale = gl.maximum(gl.max(gl.abs(rounded), 1), 1.0e-10) * (1.0 / maximum)
        encoded = gl.clamp(rounded * (1.0 / quant_scale[:, None]), -maximum, maximum)
        gl.store(quantized + out_ptr, encoded, token[:, None] < end)
        gl.store(scales + token * 8 + head, quant_scale, token < end)


_prepare_segment_maps_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_prepare_segment_maps", ["BATCH", "C", "S", "BV", "WM"]
)


@gluon.jit(repr=_prepare_segment_maps_repr)
def _prepare_segment_maps(
    prepared,
    starts,
    offsets,
    segment_offsets,
    W,
    U,
    G,
    A,
    B,
    BATCH: gl.constexpr,
    C: gl.constexpr,
    S: gl.constexpr,
    BV: gl.constexpr = 16,
    WM: gl.constexpr = 4,
):
    segment = gl.program_id(0)
    head = gl.program_id(1)
    tile = gl.program_id(2)
    total = gl.load(segment_offsets + BATCH)
    if segment < total:
        seq = _find_sequence(segment_offsets, segment, BATCH)
        seq_first = gl.load(offsets + seq)
        first = seq_first + (segment - gl.load(segment_offsets + seq)) * S
        last = gl.minimum(gl.load(offsets + seq + 1), first + S)
        begin = gl.load(starts + seq)
        end = gl.load(starts + seq + 1)
        lm: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        lv: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WM, 4 // WM], [0, 1])
        k = gl.arange(0, 128, gl.SliceLayout(1, lv))
        v = tile * BV + gl.arange(0, BV, gl.SliceLayout(0, lv))
        ha = gl.where(k[:, None] == v[None, :], 1.0, 0.0)
        hb = gl.full((128, BV), 0.0, gl.float32, lv)
        t = gl.arange(0, C, gl.SliceLayout(1, lm))
        kk = gl.arange(0, 128, gl.SliceLayout(0, lm))
        tv = gl.arange(0, C, gl.SliceLayout(1, lv))
        for chunk in range(first, last):
            token = begin + (chunk - seq_first) * C + t
            w = gl.load(W + ((chunk * 8 + head) * C + t[:, None]) * 128 + kk[None, :])
            u = gl.load(U + ((chunk * 8 + head) * C + tv[:, None]) * 128 + v[None, :])
            da = -gl.convert_layout(_matrix_product(w, ha, WM), lv)
            db = u - gl.convert_layout(_matrix_product(w, hb, WM), lv)
            key = gl.load(
                prepared
                + token[:, None] * 2048
                + 512
                + (head // 2) * 128
                + kk[None, :],
                token[:, None] < end,
                0,
            ).to(gl.float32)
            last_decay = gl.load(
                G + (chunk * 8 + head) * (C * C + C) + C + (C - 1) * C + t
            )
            final_decay = gl.load(G + (chunk * 8 + head) * (C * C + C) + C - 1)
            key_t = gl.permute(key * last_decay[:, None], (1, 0))
            ha = final_decay * ha + gl.convert_layout(
                _matrix_product(key_t, da, WM), lv
            )
            hb = final_decay * hb + gl.convert_layout(
                _matrix_product(key_t, db, WM), lv
            )
        loc = (segment * 8 + head) * 128 * 128 + k[:, None] * 128 + v[None, :]
        gl.store(A + loc, ha)
        gl.store(B + loc, hb)


_propagate_segments_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_propagate_segments", []
)


@gluon.jit(repr=_propagate_segments_repr)
def _propagate_segments(states, indices, segment_offsets, A, B, H):
    seq = gl.program_id(0)
    head = gl.program_id(1)
    tile = gl.program_id(2)
    slot = gl.load(indices + seq)
    first = gl.load(segment_offsets + seq)
    last = gl.load(segment_offsets + seq + 1)
    lm: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    lv: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [4, 1], [0, 1])
    k = gl.arange(0, 128, gl.SliceLayout(1, lv))
    v = tile * 16 + gl.arange(0, 16, gl.SliceLayout(0, lv))
    state_ptr = states + ((slot * 8 + head) * 128 + v[None, :]) * 128 + k[:, None]
    h = gl.load(state_ptr)
    r = gl.arange(0, 128, gl.SliceLayout(1, lm))
    c = gl.arange(0, 128, gl.SliceLayout(0, lm))
    for segment in range(first, last):
        loc = (segment * 8 + head) * 128 * 128
        gl.store(H + loc + k[:, None] * 128 + v[None, :], h)
        a = gl.load(A + loc + r[:, None] * 128 + c[None, :])
        b = gl.load(B + loc + k[:, None] * 128 + v[None, :])
        h = gl.convert_layout(_matrix_product(a, h, 4), lv) + b
    gl.store(state_ptr, h)


_render_segment_chunks_repr = make_kernel_repr(
    "gdn_prefill_m1024_3071_render_segment_chunks", ["BATCH", "C", "S", "BV", "WM"]
)


@gluon.jit(repr=_render_segment_chunks_repr)
def _render_segment_chunks(
    prepared,
    starts,
    offsets,
    segment_offsets,
    W,
    U,
    G,
    H,
    updates,
    carry_buffer,
    BATCH: gl.constexpr,
    C: gl.constexpr,
    S: gl.constexpr,
    BV: gl.constexpr,
    WM: gl.constexpr = 4,
):
    segment = gl.program_id(0)
    head = gl.program_id(1)
    tile = gl.program_id(2)
    total = gl.load(segment_offsets + BATCH)
    if segment < total:
        seq = _find_sequence(segment_offsets, segment, BATCH)
        seq_first = gl.load(offsets + seq)
        first = seq_first + (segment - gl.load(segment_offsets + seq)) * S
        last = gl.minimum(gl.load(offsets + seq + 1), first + S)
        begin = gl.load(starts + seq)
        end = gl.load(starts + seq + 1)
        lm: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        lv: gl.constexpr = gl.BlockedLayout([1, 4], [16, 4], [WM, 4 // WM], [0, 1])
        k = gl.arange(0, 128, gl.SliceLayout(1, lv))
        v = tile * BV + gl.arange(0, max(16, BV), gl.SliceLayout(0, lv))
        h = gl.load(
            H + (segment * 8 + head) * 128 * 128 + k[:, None] * 128 + v[None, :],
            v[None, :] < (tile + 1) * BV,
            0,
        )
        t = gl.arange(0, C, gl.SliceLayout(1, lm))
        kk = gl.arange(0, 128, gl.SliceLayout(0, lm))
        tv = gl.arange(0, C, gl.SliceLayout(1, lv))
        for chunk in range(first, last):
            token = begin + (chunk - seq_first) * C + t
            w = gl.load(W + ((chunk * 8 + head) * C + t[:, None]) * 128 + kk[None, :])
            u = gl.load(
                U + ((chunk * 8 + head) * C + tv[:, None]) * 128 + v[None, :],
                v[None, :] < (tile + 1) * BV,
                0,
            )
            du = u - gl.convert_layout(_matrix_product(w, h, WM), lv)
            gl.store(
                updates + ((chunk * 8 + head) * C + tv[:, None]) * 128 + v[None, :],
                du,
                v[None, :] < (tile + 1) * BV,
            )
            q = gl.load(
                prepared + token[:, None] * 2048 + (head // 2) * 128 + kk[None, :],
                token[:, None] < end,
                0,
            ).to(gl.float32)
            p = gl.load(G + (chunk * 8 + head) * (C * C + C) + t)
            base = gl.convert_layout(_matrix_product(q * p[:, None], h, WM), lv)
            gl.store(
                carry_buffer
                + ((chunk * 8 + head) * C + tv[:, None]) * 128
                + v[None, :],
                base,
                v[None, :] < (tile + 1) * BV,
            )
            key = gl.load(
                prepared
                + token[:, None] * 2048
                + 512
                + (head // 2) * 128
                + kk[None, :],
                token[:, None] < end,
                0,
            ).to(gl.float32)
            last_decay = gl.load(
                G + (chunk * 8 + head) * (C * C + C) + C + (C - 1) * C + t
            )
            final_decay = gl.load(G + (chunk * 8 + head) * (C * C + C) + C - 1)
            step = gl.convert_layout(
                _matrix_product(gl.permute(key * last_decay[:, None], (1, 0)), du, WM),
                lv,
            )
            h = final_decay * h + step
