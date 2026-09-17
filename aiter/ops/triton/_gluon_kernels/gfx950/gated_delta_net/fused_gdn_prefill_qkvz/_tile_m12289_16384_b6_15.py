"""GDN prefill using BF16-aware preprocessing and FP32 chunk recurrences.

Each device sequence owns its cache slot. Chunk-local triangular solves expose
matrix work. Low-batch sequences use independent affine groups and a short,
ordered state propagation; larger batches consume compact chunk factors.
Required BF16 materialization boundaries are retained, and matrix/state
arithmetic uses FP32. Every consumed scratch element is initialized within
the invocation, including ragged tails.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@gluon.jit
def _projection_channel(group, column):
    return gl.where(
        group < 8,
        (group % 4) * 768 + (group // 4) * 128 + column,
        ((group - 8) // 2) * 768 + 256 + (group % 2) * 128 + column,
    )


@gluon.jit
def _load_history(qkvz, state, token, begin, slot, cached, channel, packed, valid):
    from_input = token >= begin
    x = gl.load(qkvz + token * 3072 + packed, from_input & valid, 0)
    history = gl.load(
        state + (slot * 2048 + channel) * 3 + token - begin + 3,
        (~from_input) & cached & valid,
        0,
    )
    return gl.where(from_input, x, history).to(gl.float32)


_prepare_tokens_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_prepare_tokens",
    ["M", "LOG_BATCH", "BATCH", "ROWS", "LANES", "PACK"],
)


@gluon.jit(repr=_prepare_tokens_repr)
def _prepare_tokens(
    qkvz,
    ba,
    state,
    indices,
    starts,
    initial,
    weight,
    bias,
    a_log,
    dt_bias,
    prepared,
    gates,
    M: gl.constexpr,
    LOG_BATCH: gl.constexpr,
    BATCH: gl.constexpr,
    ROWS: gl.constexpr,
    LANES: gl.constexpr,
    PACK: gl.constexpr,
):
    group = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, PACK],
        [64 // LANES, LANES],
        [gl.num_warps(), 1],
        [1, 0],
    )
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    col = gl.arange(0, 128, gl.SliceLayout(0, layout))
    lo = gl.full((ROWS,), 0, gl.int32, gl.SliceLayout(1, layout))
    hi = gl.full((ROWS,), BATCH, gl.int32, gl.SliceLayout(1, layout))
    for _ in gl.static_range(LOG_BATCH):
        mid = (lo + hi) // 2
        boundary = gl.load(starts + mid)
        right = row >= boundary
        lo = gl.where(right, mid, lo)
        hi = gl.where(right, hi, mid)
    begin = gl.load(starts + lo)
    slot = gl.load(indices + lo)
    cached = gl.load(initial + lo)
    channel = group * 128 + col
    packed = _projection_channel(group, col)
    valid = row[:, None] < M
    h0 = _load_history(
        qkvz,
        state,
        row[:, None] - 3,
        begin[:, None],
        slot[:, None],
        cached[:, None],
        channel[None, :],
        packed[None, :],
        valid,
    )
    h1 = _load_history(
        qkvz,
        state,
        row[:, None] - 2,
        begin[:, None],
        slot[:, None],
        cached[:, None],
        channel[None, :],
        packed[None, :],
        valid,
    )
    h2 = _load_history(
        qkvz,
        state,
        row[:, None] - 1,
        begin[:, None],
        slot[:, None],
        cached[:, None],
        channel[None, :],
        packed[None, :],
        valid,
    )
    x = gl.load(qkvz + row[:, None] * 3072 + packed[None, :], valid, 0).to(gl.float32)
    w0 = gl.load(weight + channel * 4).to(gl.float32)
    w1 = gl.load(weight + channel * 4 + 1).to(gl.float32)
    w2 = gl.load(weight + channel * 4 + 2).to(gl.float32)
    w3 = gl.load(weight + channel * 4 + 3).to(gl.float32)
    b = gl.load(bias + channel).to(gl.float32)
    conv = (
        b[None, :]
        + h0 * w0[None, :]
        + h1 * w1[None, :]
        + h2 * w2[None, :]
        + x * w3[None, :]
    )
    activated = gl.div_rn(conv, 1.0 + libdevice.exp(-conv)).to(gl.bfloat16)
    result = activated
    if group < 8:
        z = activated.to(gl.float32)
        result = (z * gl.rsqrt(gl.sum(z * z, 1) + 1.0e-6)[:, None]).to(gl.bfloat16)
    else:
        head = group - 8
        av = gl.load(ba + row * 16 + head // 2 * 4 + 2 + head % 2, row < M, 0).to(
            gl.float32
        )
        av += gl.load(dt_bias + head).to(gl.float32)
        bv = gl.load(ba + row * 16 + head // 2 * 4 + head % 2, row < M, 0).to(
            gl.float32
        )
        decay_weight = gl.exp(gl.load(a_log + head).to(gl.float32))
        softplus = gl.where(av <= 20.0, gl.log(1.0 + gl.exp(av)), av)
        decay = gl.exp(-decay_weight * softplus)
        beta = (1.0 / (1.0 + gl.exp(-bv))).to(gl.bfloat16).to(gl.float32)
        gl.store(gates + (head * M + row) * 2, decay, row < M)
        gl.store(gates + (head * M + row) * 2 + 1, beta, row < M)
    gl.store(prepared + (group * M + row[:, None]) * 128 + col[None, :], result, valid)


_update_conv_state_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_update_conv_state", []
)


@gluon.jit(repr=_update_conv_state_repr)
def _update_conv_state(qkvz, state, indices, starts, initial):
    seq = gl.program_id(0)
    group = gl.program_id(1)
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [1], [0])
    column = gl.arange(0, 128, layout=layout)
    begin = gl.load(starts + seq)
    end = gl.load(starts + seq + 1)
    slot = gl.load(indices + seq)
    cached = gl.load(initial + seq)
    channel = group * 128 + column
    packed = _projection_channel(group, column)
    h0 = _load_history(qkvz, state, end - 3, begin, slot, cached, channel, packed, True)
    h1 = _load_history(qkvz, state, end - 2, begin, slot, cached, channel, packed, True)
    h2 = _load_history(qkvz, state, end - 1, begin, slot, cached, channel, packed, True)
    gl.store(state + (slot * 2048 + channel) * 3, h0)
    gl.store(state + (slot * 2048 + channel) * 3 + 1, h1)
    gl.store(state + (slot * 2048 + channel) * 3 + 2, h2)


@gluon.jit
def _chunk_dot(a, b, accumulator, WIDTH: gl.constexpr):
    # WIDTH=8 uses already-rounded BF16 Q/K; WIDTH=1 is native FP32 MFMA.
    layout: gl.constexpr = accumulator.type.layout
    return gl.amd.cdna3.mfma(
        gl.convert_layout(a, gl.DotOperandLayout(0, layout, WIDTH)),
        gl.convert_layout(b, gl.DotOperandLayout(1, layout, WIDTH)),
        accumulator,
    )


@gluon.jit
def _multiply(left, right):
    return left * right


@gluon.jit
def _chunk_coefficients(
    prepared,
    gates,
    starts,
    chunk_c,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
):
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    lo = 0
    hi = BATCH
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        first = gl.load(starts + mid) // BT + mid
        right = chunk >= first
        lo = gl.where(right, mid, lo)
        hi = gl.where(right, hi, mid)
    begin = gl.load(starts + lo)
    end = gl.load(starts + lo + 1)
    # floor(begin / BT) + sequence reserves enough space for every ragged tail.
    token_begin = begin + (chunk - (begin // BT + lo)) * BT
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 16], [gl.num_warps(), 1], [1, 0]
    )
    gram_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=True,
        warps_per_cta=[1, gl.num_warps()],
    )
    tri_layout: gl.constexpr = gl.BlockedLayout(
        [1, BT // 16], [4, 16], [1, gl.num_warps()], [1, 0]
    )
    t = gl.arange(0, BT, layout=gl.SliceLayout(1, load_layout))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, load_layout))
    token = token_begin + t
    valid = token < end
    q = gl.load(
        prepared + (head // 2 * M + token[:, None]) * 128 + k[None, :],
        valid[:, None],
        0,
    )
    key = gl.load(
        prepared + ((4 + head // 2) * M + token[:, None]) * 128 + k[None, :],
        valid[:, None],
        0,
    )
    kk = _chunk_dot(
        key, gl.permute(key, (1, 0)), gl.zeros((BT, BT), gl.float32, gram_layout), 8
    )
    qk = _chunk_dot(
        q, gl.permute(key, (1, 0)), gl.zeros((BT, BT), gl.float32, gram_layout), 8
    )
    kk = gl.convert_layout(kk, tri_layout)
    qk = gl.convert_layout(qk, tri_layout)
    row = gl.arange(0, BT, layout=gl.SliceLayout(1, tri_layout))
    col = gl.arange(0, BT, layout=gl.SliceLayout(0, tri_layout))
    gt = token_begin + row
    decay = gl.load(gates + (head * M + gt) * 2, gt < end, 1)
    beta = gl.load(gates + (head * M + gt) * 2 + 1, gt < end, 0)
    # Product scans avoid both a cubic expansion and division by a zero prefix.
    scan_layout: gl.constexpr = gl.BlockedLayout([1], [64], [gl.num_warps()], [0])
    prefix = gl.convert_layout(
        gl.associative_scan(gl.convert_layout(decay, scan_layout), 0, _multiply),
        gl.SliceLayout(1, tri_layout),
    )
    between = gl.associative_scan(
        gl.where(row[:, None] > col[None, :], decay[:, None], 1.0), 0, _multiply
    )
    # Store QK coefficients before the register-heavy triangular solve.
    c = gl.where(row[:, None] >= col[None, :], qk * between, 0.0)
    gl.store(chunk_c + ((chunk * 8 + head) * BT + row[:, None]) * BT + col[None, :], c)
    inverse = gl.where(row[:, None] > col[None, :], -beta[:, None] * between * kk, 0.0)
    # Forward substitution forms the inverse of the unit-lower delta system.
    for i in gl.static_range(BT):
        index_r = gl.full((1, BT), i, gl.int32, tri_layout)
        index_c = gl.full((BT, 1), i, gl.int32, tri_layout)
        inverse_row = gl.sum(gl.gather(inverse, index_r, 0), 0)
        inverse_col = gl.sum(gl.gather(inverse, index_c, 1), 1)
        inverse += gl.where(
            (row[:, None] > i) & (col[None, :] < i),
            inverse_col[:, None] * inverse_row[None, :],
            0.0,
        )
    inverse += (row[:, None] == col[None, :]).to(gl.float32)
    final_decay_row = gl.sum(
        gl.gather(between, gl.full((1, BT), BT - 1, gl.int32, tri_layout), 0),
        0,
    )
    return q, key, inverse, prefix, final_decay_row, beta, token_begin, end


_prepare_chunks_full_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_prepare_chunks_full", ["M", "BATCH", "BT"]
)


@gluon.jit(repr=_prepare_chunks_full_repr)
def _prepare_chunks_full(
    prepared,
    gates,
    starts,
    chunk_w,
    chunk_u,
    chunk_q,
    chunk_k,
    chunk_c,
    chunk_g,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
):
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    q, key, inverse, prefix, suffix, beta, token_begin, end = _chunk_coefficients(
        prepared, gates, starts, chunk_c, M, BATCH, BT
    )
    layout: gl.constexpr = q.type.layout
    matrix_layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[1, gl.num_warps()],
    )
    t = gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    token = token_begin + t
    value = gl.load(
        prepared + ((8 + head) * M + token[:, None]) * 128 + k[None, :],
        token[:, None] < end,
        0,
    ).to(gl.float32)
    beta_r = gl.convert_layout(beta, gl.SliceLayout(1, layout))
    prefix_r = gl.convert_layout(prefix, gl.SliceLayout(1, layout))
    suffix_r = gl.convert_layout(suffix, gl.SliceLayout(1, layout))
    w = _chunk_dot(
        inverse,
        (beta_r * prefix_r)[:, None] * key.to(gl.float32),
        gl.zeros((BT, 128), gl.float32, matrix_layout),
        1,
    )
    u = _chunk_dot(
        inverse,
        beta_r[:, None] * value,
        gl.zeros((BT, 128), gl.float32, matrix_layout),
        1,
    )
    offsets = ((chunk * 8 + head) * BT + t[:, None]) * 128 + k[None, :]
    gl.store(chunk_w + offsets, gl.convert_layout(w, layout))
    gl.store(chunk_u + offsets, gl.convert_layout(u, layout))
    gl.store(chunk_q + offsets, prefix_r[:, None] * q.to(gl.float32))
    gl.store(chunk_k + offsets, suffix_r[:, None] * key.to(gl.float32))
    final_decay = gl.sum(
        gl.gather(prefix, gl.full((1,), BT - 1, gl.int32, prefix.type.layout), 0),
        0,
    )
    gl.store(chunk_g + chunk * 8 + head, final_decay)


_prepare_chunks_compact_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_prepare_chunks_compact", ["M", "BATCH", "BT"]
)


@gluon.jit(repr=_prepare_chunks_compact_repr)
def _prepare_chunks_compact(
    prepared,
    gates,
    starts,
    chunk_inverse,
    chunk_c,
    chunk_decay,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
):
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    q, key, inverse, prefix, suffix, beta, token_begin, end = _chunk_coefficients(
        prepared, gates, starts, chunk_c, M, BATCH, BT
    )
    layout: gl.constexpr = inverse.type.layout
    row = gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
    col = gl.arange(0, BT, layout=gl.SliceLayout(0, layout))
    offsets = ((chunk * 8 + head) * BT + row[:, None]) * BT + col[None, :]
    gl.store(chunk_inverse + offsets, inverse)
    gl.store(chunk_decay + (chunk * 8 + head) * 2 * BT + row, prefix)
    gl.store(chunk_decay + ((chunk * 8 + head) * 2 + 1) * BT + col, suffix)


_recurrence_full_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_recurrence_full", ["BT", "BV"]
)


@gluon.jit(repr=_recurrence_full_repr)
def _recurrence_full(
    chunk_w,
    chunk_u,
    chunk_q,
    chunk_k,
    chunk_c,
    chunk_g,
    state,
    indices,
    starts,
    out,
    scale,
    BT: gl.constexpr,
    BV: gl.constexpr,
):
    seq = gl.program_id(0)
    head = gl.program_id(1)
    value_tile = gl.program_id(2)
    begin = gl.load(starts + seq)
    end = gl.load(starts + seq + 1)
    slot = gl.load(indices + seq).to(gl.int64)
    # One reserved chunk per sequence also covers empty and unaligned starts.
    first_chunk = begin // BT + seq
    chunks = gl.cdiv(end - begin, BT)
    layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[1, gl.num_warps()],
    )
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 16], [gl.num_warps(), 1], [1, 0]
    )
    v = value_tile * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    state_offset = ((slot * 8 + head) * 128 + v[:, None]) * 128 + k[None, :]
    h = gl.load(state + state_offset)
    t = gl.arange(0, BT, layout=gl.SliceLayout(0, layout))
    load_t = gl.arange(0, BT, layout=gl.SliceLayout(1, load_layout))
    load_k = gl.arange(0, 128, layout=gl.SliceLayout(0, load_layout))
    load_c = gl.arange(0, BT, layout=gl.SliceLayout(0, load_layout))
    for local_chunk in range(chunks):
        chunk = first_chunk + local_chunk
        base = (chunk * 8 + head) * BT * 128
        offsets = base + load_t[:, None] * 128 + load_k[None, :]
        w = gl.load(chunk_w + offsets)
        q = gl.load(chunk_q + offsets)
        key = gl.load(chunk_k + offsets)
        u = gl.load(chunk_u + base + t[None, :] * 128 + v[:, None])
        decay = gl.load(chunk_g + chunk * 8 + head)
        c = gl.load(
            chunk_c + ((chunk * 8 + head) * BT + load_t[:, None]) * BT + load_c[None, :]
        )
        projected = _chunk_dot(
            h, gl.permute(w, (1, 0)), gl.zeros((BV, BT), gl.float32, layout), 1
        )
        delta = u - projected
        y = _chunk_dot(
            h, gl.permute(q, (1, 0)), gl.zeros((BV, BT), gl.float32, layout), 1
        )
        y = _chunk_dot(delta, gl.permute(c, (1, 0)), y, 1)
        h = _chunk_dot(delta, key, h * decay, 1)
        token = begin + local_chunk * BT + t
        gl.store(
            out + (token[None, :] * 8 + head) * 128 + v[:, None],
            y * scale,
            token[None, :] < end,
        )
    gl.store(state + state_offset, h)


_recurrence_compact_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_recurrence_compact",
    ["M", "BT", "BV", "ROW_WARPS", "TRANSPOSED"],
)


@gluon.jit(repr=_recurrence_compact_repr)
def _recurrence_compact(
    prepared,
    gates,
    chunk_inverse,
    chunk_c,
    chunk_decay,
    state,
    indices,
    starts,
    out,
    scale,
    M: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    ROW_WARPS: gl.constexpr,
    TRANSPOSED: gl.constexpr,
):
    seq = gl.program_id(0)
    head = gl.program_id(1)
    value_tile = gl.program_id(2)
    begin = gl.load(starts + seq)
    end = gl.load(starts + seq + 1)
    slot = gl.load(indices + seq).to(gl.int64)
    # One reserved chunk per sequence also covers empty and unaligned starts.
    first_chunk = begin // BT + seq
    chunks = gl.cdiv(end - begin, BT)
    layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=TRANSPOSED,
        warps_per_cta=[ROW_WARPS, gl.num_warps() // ROW_WARPS],
    )
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 16], [gl.num_warps(), 1], [1, 0]
    )
    v = value_tile * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    state_offset = ((slot * 8 + head) * 128 + v[:, None]) * 128 + k[None, :]
    h = gl.load(state + state_offset)
    t = gl.arange(0, BT, layout=gl.SliceLayout(0, layout))
    load_t = gl.arange(0, BT, layout=gl.SliceLayout(1, load_layout))
    load_k = gl.arange(0, 128, layout=gl.SliceLayout(0, load_layout))
    load_c = gl.arange(0, BT, layout=gl.SliceLayout(0, load_layout))
    for local_chunk in range(chunks):
        chunk = first_chunk + local_chunk
        token_start = begin + local_chunk * BT
        token_load = token_start + load_t
        key = gl.load(
            prepared
            + ((4 + head // 2) * M + token_load[:, None]) * 128
            + load_k[None, :],
            token_load[:, None] < end,
            0,
        )
        q = gl.load(
            prepared + (head // 2 * M + token_load[:, None]) * 128 + load_k[None, :],
            token_load[:, None] < end,
            0,
        )
        token = token_start + t
        value = gl.load(
            prepared + ((8 + head) * M + token[None, :]) * 128 + v[:, None],
            token[None, :] < end,
            0,
        ).to(gl.float32)
        beta = gl.load(gates + (head * M + token) * 2 + 1, token < end, 0)
        prefix = gl.load(chunk_decay + (chunk * 8 + head) * 2 * BT + t)
        suffix = gl.load(chunk_decay + ((chunk * 8 + head) * 2 + 1) * BT + t)
        final_decay = gl.load(chunk_decay + (chunk * 8 + head) * 2 * BT + BT - 1)
        offsets = ((chunk * 8 + head) * BT + load_t[:, None]) * BT + load_c[None, :]
        inverse = gl.load(chunk_inverse + offsets)
        c = gl.load(chunk_c + offsets)
        projected = _chunk_dot(
            h,
            gl.permute(key.to(gl.float32), (1, 0)),
            gl.zeros((BV, BT), gl.float32, layout),
            1,
        )
        y = _chunk_dot(
            h,
            gl.permute(q.to(gl.float32), (1, 0)),
            gl.zeros((BV, BT), gl.float32, layout),
            1,
        )
        rhs = beta[None, :] * (value - prefix[None, :] * projected)
        delta = _chunk_dot(
            rhs,
            gl.permute(inverse, (1, 0)),
            gl.zeros((BV, BT), gl.float32, layout),
            1,
        )
        y = _chunk_dot(delta, gl.permute(c, (1, 0)), y * prefix[None, :], 1)
        h = _chunk_dot(delta * suffix[None, :], key.to(gl.float32), h * final_decay, 1)
        gl.store(
            out + (token[None, :] * 8 + head) * 128 + v[:, None],
            y * scale,
            token[None, :] < end,
        )
    gl.store(state + state_offset, h)


@gluon.jit
def _group_bounds(starts, group, BATCH: gl.constexpr, GROUP: gl.constexpr):
    lo = 0
    hi = BATCH
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        first = gl.load(starts + mid) // GROUP + mid
        right = group >= first
        lo = gl.where(right, mid, lo)
        hi = gl.where(right, hi, mid)
    begin = gl.load(starts + lo)
    end = gl.load(starts + lo + 1)
    local_group = group - (begin // GROUP + lo)
    return lo, begin, end, local_group


_build_group_maps_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_build_group_maps",
    ["BATCH", "BT", "BV", "GROUP", "ROW_WARPS"],
)


@gluon.jit(repr=_build_group_maps_repr)
def _build_group_maps(
    chunk_w,
    chunk_u,
    chunk_k,
    chunk_g,
    starts,
    maps,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    GROUP: gl.constexpr,
    ROW_WARPS: gl.constexpr,
):
    """Propagate the identity and zero-state response through independent groups."""
    group = gl.program_id(2)
    head = gl.program_id(1)
    tile = gl.program_id(0)
    seq, begin, end, local_group = _group_bounds(starts, group, BATCH, GROUP)
    group_begin = begin + local_group * GROUP
    first_chunk = begin // BT + seq + local_group * (GROUP // BT)
    chunks = gl.minimum(gl.cdiv(end - group_begin, BT), GROUP // BT)
    layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[ROW_WARPS, gl.num_warps() // ROW_WARPS],
    )
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 16], [gl.num_warps(), 1], [1, 0]
    )
    v = (tile % (128 // BV)) * BV + gl.arange(0, BV, gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, gl.SliceLayout(0, layout))
    identity = tile < 128 // BV
    h = ((v[:, None] == k[None, :]) & identity).to(gl.float32)
    t = gl.arange(0, BT, gl.SliceLayout(0, layout))
    load_t = gl.arange(0, BT, gl.SliceLayout(1, load_layout))
    load_k = gl.arange(0, 128, gl.SliceLayout(0, load_layout))
    for local_chunk in range(chunks):
        chunk = first_chunk + local_chunk
        base = (chunk * 8 + head) * BT * 128
        offsets = base + load_t[:, None] * 128 + load_k[None, :]
        w = gl.load(chunk_w + offsets)
        key = gl.load(chunk_k + offsets)
        u = gl.load(chunk_u + base + t[None, :] * 128 + v[:, None], ~identity, 0)
        decay = gl.load(chunk_g + chunk * 8 + head)
        projected = _chunk_dot(
            h, gl.permute(w, (1, 0)), gl.zeros((BV, BT), gl.float32, layout), 1
        )
        h = _chunk_dot(u - projected, key, h * decay, 1)
    map_row = tile * BV + gl.arange(0, BV, gl.SliceLayout(1, layout))
    gl.store(maps + ((group * 8 + head) * 256 + map_row[:, None]) * 128 + k[None, :], h)


_propagate_group_maps_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_propagate_group_maps", ["GROUP", "BV"]
)


@gluon.jit(repr=_propagate_group_maps_repr)
def _propagate_group_maps(
    maps,
    boundaries,
    state,
    indices,
    starts,
    GROUP: gl.constexpr,
    BV: gl.constexpr,
):
    seq = gl.program_id(0)
    head = gl.program_id(1)
    value_tile = gl.program_id(2)
    begin = gl.load(starts + seq)
    end = gl.load(starts + seq + 1)
    slot = gl.load(indices + seq).to(gl.int64)
    first_group = begin // GROUP + seq
    groups = gl.cdiv(end - begin, GROUP)
    layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[1, gl.num_warps()],
    )
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 16], [gl.num_warps(), 1], [1, 0]
    )
    v = value_tile * BV + gl.arange(0, BV, gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, gl.SliceLayout(0, layout))
    state_offsets = ((slot * 8 + head) * 128 + v[:, None]) * 128 + k[None, :]
    h = gl.load(state + state_offsets)
    load_r = gl.arange(0, 128, gl.SliceLayout(1, load_layout))
    load_k = gl.arange(0, 128, gl.SliceLayout(0, load_layout))
    for local_group in range(groups):
        group = first_group + local_group
        boundary_offsets = ((group * 8 + head) * 128 + v[:, None]) * 128 + k[None, :]
        gl.store(boundaries + boundary_offsets, h)
        map_base = (group * 8 + head) * 256 * 128
        transition = gl.load(maps + map_base + load_r[:, None] * 128 + load_k[None, :])
        response = gl.load(maps + map_base + (128 + v[:, None]) * 128 + k[None, :])
        h = _chunk_dot(h, transition, response, 1)
    gl.store(state + state_offsets, h)


_evaluate_groups_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_evaluate_groups",
    ["BATCH", "BT", "BV", "GROUP", "ROW_WARPS"],
)


@gluon.jit(repr=_evaluate_groups_repr)
def _evaluate_groups(
    chunk_w,
    chunk_u,
    chunk_q,
    chunk_k,
    chunk_c,
    chunk_g,
    boundaries,
    starts,
    out,
    scale,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    GROUP: gl.constexpr,
    ROW_WARPS: gl.constexpr,
):
    group = gl.program_id(2)
    head = gl.program_id(1)
    value_tile = gl.program_id(0)
    seq, begin, end, local_group = _group_bounds(starts, group, BATCH, GROUP)
    group_begin = begin + local_group * GROUP
    first_chunk = begin // BT + seq + local_group * (GROUP // BT)
    chunks = gl.minimum(gl.cdiv(end - group_begin, BT), GROUP // BT)
    layout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[ROW_WARPS, gl.num_warps() // ROW_WARPS],
    )
    load_layout: gl.constexpr = gl.BlockedLayout(
        [1, 4], [4, 16], [gl.num_warps(), 1], [1, 0]
    )
    v = value_tile * BV + gl.arange(0, BV, gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, gl.SliceLayout(0, layout))
    boundary_offsets = ((group * 8 + head) * 128 + v[:, None]) * 128 + k[None, :]
    h = gl.load(boundaries + boundary_offsets, group_begin < end, 0)
    t = gl.arange(0, BT, gl.SliceLayout(0, layout))
    load_t = gl.arange(0, BT, gl.SliceLayout(1, load_layout))
    load_k = gl.arange(0, 128, gl.SliceLayout(0, load_layout))
    load_c = gl.arange(0, BT, gl.SliceLayout(0, load_layout))
    for local_chunk in range(chunks):
        chunk = first_chunk + local_chunk
        base = (chunk * 8 + head) * BT * 128
        offsets = base + load_t[:, None] * 128 + load_k[None, :]
        w = gl.load(chunk_w + offsets)
        q = gl.load(chunk_q + offsets)
        key = gl.load(chunk_k + offsets)
        u = gl.load(chunk_u + base + t[None, :] * 128 + v[:, None])
        decay = gl.load(chunk_g + chunk * 8 + head)
        c = gl.load(
            chunk_c + ((chunk * 8 + head) * BT + load_t[:, None]) * BT + load_c[None, :]
        )
        projected = _chunk_dot(
            h, gl.permute(w, (1, 0)), gl.zeros((BV, BT), gl.float32, layout), 1
        )
        delta = u - projected
        y = _chunk_dot(
            h, gl.permute(q, (1, 0)), gl.zeros((BV, BT), gl.float32, layout), 1
        )
        y = _chunk_dot(delta, gl.permute(c, (1, 0)), y, 1)
        h = _chunk_dot(delta, key, h * decay, 1)
        token = group_begin + local_chunk * BT + t
        gl.store(
            out + (token[None, :] * 8 + head) * 128 + v[:, None],
            y * scale,
            token[None, :] < end,
        )


_gated_rms_quant_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_gated_rms_quant", ["TOTAL", "ROWS", "LANES", "PACK"]
)


@gluon.jit(repr=_gated_rms_quant_repr)
def _gated_rms_quant(
    core,
    qkvz,
    weight,
    normalized,
    quantized,
    scales,
    eps,
    quant_max,
    inverse_quant_max,
    TOTAL: gl.constexpr,
    ROWS: gl.constexpr,
    LANES: gl.constexpr,
    PACK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout(
        [1, PACK],
        [64 // LANES, LANES],
        [gl.num_warps(), 1],
        [1, 0],
    )
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    col = gl.arange(0, 128, gl.SliceLayout(0, layout))
    valid = row[:, None] < TOTAL
    x = gl.load(core + row[:, None] * 128 + col[None, :], valid, 0).to(gl.float32)
    z_offset = row // 8 * 3072 + (row % 8) // 2 * 768 + 512 + row % 2 * 128
    gate = gl.load(qkvz + z_offset[:, None] + col[None, :], valid, 0).to(gl.float32)
    w = gl.load(weight + col).to(gl.float32)
    inverse_rms = gl.rsqrt(gl.sum(x * x, 1) / 128 + eps)
    sigmoid = 1.0 / (1.0 + gl.exp(-gate))
    result = (x * inverse_rms[:, None] * w[None, :] * gate * sigmoid).to(gl.bfloat16)
    values = result.to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(values), 1), 1.0e-10) * inverse_quant_max
    encoded = gl.clamp(values * (1.0 / scale[:, None]), -quant_max, quant_max)
    gl.store(normalized + row[:, None] * 128 + col[None, :], result, valid)
    gl.store(quantized + row[:, None] * 128 + col[None, :], encoded, valid)
    gl.store(scales + row, scale, row < TOTAL)
