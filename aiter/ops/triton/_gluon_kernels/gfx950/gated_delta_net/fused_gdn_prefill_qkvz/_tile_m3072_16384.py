"""Post-projection GDN prefill on CDNA4, implemented entirely in Gluon.

Convolution and Q/K normalization are token-parallel.  A unit-triangular
chunk solve converts the delta rule into FP32 matrix operations.  Small
batches additionally use affine block summaries to shorten the time-serial
path.  BF16 rounding boundaries, both state pools, and the final groupwise
FP8 quantization are preserved.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.language.core import range as loop_range
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_convolve_sliding_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_convolve_sliding",
    [
        "M",
        "BATCH",
        "SEARCH_STEPS",
        "PAD_BATCH",
        "BT",
        "GROUP",
        "HIERARCHICAL",
        "TOKENS",
    ],
)


@gluon.jit(repr=_convolve_sliding_repr)
def _convolve_sliding(
    packed,
    pool,
    starts,
    indices,
    initial,
    weight,
    bias,
    prepared,
    chunk_offsets,
    block_offsets,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    SEARCH_STEPS: gl.constexpr,
    PAD_BATCH: gl.constexpr,
    BT: gl.constexpr,
    GROUP: gl.constexpr,
    HIERARCHICAL: gl.constexpr,
    TOKENS: gl.constexpr,
):
    # Two waves own either Q/K or the paired value heads.
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [1, 64], [2, 1], [1, 0])
    role = gl.program_id(2)
    component = role * 2 + gl.arange(0, 2, gl.SliceLayout(1, layout))
    d = gl.arange(0, 128, gl.SliceLayout(0, layout))
    kh = gl.program_id(1)
    token = gl.program_id(0) * TOKENS
    low, high = 0, BATCH
    for _ in gl.static_range(SEARCH_STEPS):
        middle = (low + high) // 2
        right = gl.load(starts + middle) <= token
        low = gl.where(right, middle + 1, low)
        high = gl.where(right, high, middle)
    sequence = gl.minimum(low - 1, BATCH - 1)
    begin = gl.load(starts + sequence)
    end = gl.load(starts + sequence + 1)
    slot = gl.load(indices + sequence)
    cached = gl.load(initial + sequence)
    channel_base = gl.where(
        component == 0,
        kh * 128,
        gl.where(
            component == 1, 512 + kh * 128, 1024 + kh * 256 + (component - 2) * 128
        ),
    )
    channel = channel_base[:, None] + d[None, :]
    packed_channel = kh * 768 + component[:, None] * 128 + d[None, :]
    weight_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2, 4], [1, 64, 1], [2, 1, 1], [2, 1, 0]
    )
    wc = gl.convert_layout(
        channel_base, gl.SliceLayout(1, gl.SliceLayout(2, weight_layout))
    )
    wd = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, weight_layout)))
    wt = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, weight_layout)))
    taps = gl.load(
        weight + (wc[:, None, None] + wd[None, :, None]) * 4 + wt[None, None, :]
    )
    even, odd = gl.split(taps.reshape((2, 128, 2, 2)))
    w0, w2 = gl.split(even)
    w1, w3 = gl.split(odd)
    w0 = gl.convert_layout(w0, layout).to(gl.float32)
    w1 = gl.convert_layout(w1, layout).to(gl.float32)
    w2 = gl.convert_layout(w2, layout).to(gl.float32)
    w3 = gl.convert_layout(w3, layout).to(gl.float32)
    b = gl.load(bias + channel).to(gl.float32)

    # Most tiles stay within one sequence and need no history or tail masks.
    # The boundary path also handles tiles crossing arbitrarily short sequences.
    interior = (token >= begin + 3) & (token + TOKENS <= end)
    if interior:
        x0 = gl.load(packed + (token - 3) * 3072 + packed_channel).to(gl.float32)
        x1 = gl.load(packed + (token - 2) * 3072 + packed_channel).to(gl.float32)
        x2 = gl.load(packed + (token - 1) * 3072 + packed_channel).to(gl.float32)
        for step in gl.static_range(TOKENS):
            current_token = token + step
            x3 = gl.load(packed + current_token * 3072 + packed_channel).to(gl.float32)
            conv = b + x0 * w0
            conv = conv + x1 * w1
            conv = conv + x2 * w2
            conv = conv + x3 * w3
            activated = (
                gl.div_rn(conv, 1.0 + libdevice.exp(-conv))
                .to(gl.bfloat16)
                .to(gl.float32)
            )
            result = activated
            if role == 0:
                inverse_norm = gl.rsqrt(gl.sum(activated * activated, 1) + 1.0e-6)
                result = activated * inverse_norm[:, None]
            gl.store(prepared + current_token * 2048 + channel, result)
            x0, x1, x2 = x1, x2, x3
    else:
        x0 = gl.load(
            packed + (token - 3) * 3072 + packed_channel, token - 3 >= begin, other=0
        ).to(gl.float32)
        x1 = gl.load(
            packed + (token - 2) * 3072 + packed_channel, token - 2 >= begin, other=0
        ).to(gl.float32)
        x2 = gl.load(
            packed + (token - 1) * 3072 + packed_channel, token - 1 >= begin, other=0
        ).to(gl.float32)
        h0 = gl.load(
            pool + (slot * 2048 + channel) * 3 + token - begin,
            (token - 3 < begin) & cached,
            other=0,
        ).to(gl.float32)
        h1 = gl.load(
            pool + (slot * 2048 + channel) * 3 + token - begin + 1,
            (token - 2 < begin) & cached,
            other=0,
        ).to(gl.float32)
        h2 = gl.load(
            pool + (slot * 2048 + channel) * 3 + token - begin + 2,
            (token - 1 < begin) & cached,
            other=0,
        ).to(gl.float32)
        x0 = gl.where(token - 3 >= begin, x0, h0)
        x1 = gl.where(token - 2 >= begin, x1, h1)
        x2 = gl.where(token - 1 >= begin, x2, h2)
        for step in gl.static_range(TOKENS):
            current_token = token + step
            if BATCH > 1 and step > 0:
                if current_token >= end:
                    low, high = 0, BATCH
                    for _ in gl.static_range(SEARCH_STEPS):
                        middle = (low + high) // 2
                        right = gl.load(starts + middle) <= current_token
                        low = gl.where(right, middle + 1, low)
                        high = gl.where(right, high, middle)
                    sequence = gl.minimum(low - 1, BATCH - 1)
                    end = gl.load(starts + sequence + 1)
                    slot = gl.load(indices + sequence)
                    cached = gl.load(initial + sequence)
                    x0 = gl.load(
                        pool + (slot * 2048 + channel) * 3,
                        cached & (current_token < M),
                        other=0,
                    ).to(gl.float32)
                    x1 = gl.load(
                        pool + (slot * 2048 + channel) * 3 + 1,
                        cached & (current_token < M),
                        other=0,
                    ).to(gl.float32)
                    x2 = gl.load(
                        pool + (slot * 2048 + channel) * 3 + 2,
                        cached & (current_token < M),
                        other=0,
                    ).to(gl.float32)
            x3 = gl.load(
                packed + current_token * 3072 + packed_channel,
                current_token < M,
                other=0,
            ).to(gl.float32)
            conv = b + x0 * w0
            conv = conv + x1 * w1
            conv = conv + x2 * w2
            conv = conv + x3 * w3
            activated = (
                gl.div_rn(conv, 1.0 + libdevice.exp(-conv))
                .to(gl.bfloat16)
                .to(gl.float32)
            )
            result = activated
            if role == 0:
                inverse_norm = gl.rsqrt(gl.sum(activated * activated, 1) + 1.0e-6)
                result = activated * inverse_norm[:, None]
            gl.store(
                prepared + current_token * 2048 + channel, result, current_token < M
            )
            x0, x1, x2 = x1, x2, x3
    if (gl.program_id(0) == 0) & (gl.program_id(1) == 0) & (role == 0):
        _write_offsets(
            starts,
            chunk_offsets,
            block_offsets,
            BATCH,
            PAD_BATCH,
            BT,
            GROUP,
            HIERARCHICAL,
            2,
        )


@gluon.jit
def _update_conv_pool(
    packed, pool, starts, indices, initial, sequence, channel_block, BLOCK: gl.constexpr
):
    c = channel_block * BLOCK + gl.arange(
        0, BLOCK, layout=gl.BlockedLayout([1], [64], [4], [0])
    )
    begin = gl.load(starts + sequence)
    end = gl.load(starts + sequence + 1)
    slot = gl.load(indices + sequence)
    cached = gl.load(initial + sequence)
    packed_c = gl.where(
        c < 512,
        (c // 128) * 768 + c % 128,
        gl.where(
            c < 1024,
            ((c - 512) // 128) * 768 + 128 + c % 128,
            ((c - 1024) // 256) * 768 + 256 + c % 256,
        ),
    )
    h0 = gl.load(
        pool + (slot * 2048 + c) * 3 + (end - begin),
        (end - 3 < begin) & cached,
        other=0,
    )
    h1 = gl.load(
        pool + (slot * 2048 + c) * 3 + (end - begin + 1),
        (end - 2 < begin) & cached,
        other=0,
    )
    h2 = gl.load(
        pool + (slot * 2048 + c) * 3 + (end - begin + 2),
        (end - 1 < begin) & cached,
        other=0,
    )
    x0 = gl.load(packed + (end - 3) * 3072 + packed_c, end - 3 >= begin, other=0)
    x1 = gl.load(packed + (end - 2) * 3072 + packed_c, end - 2 >= begin, other=0)
    x2 = gl.load(packed + (end - 1) * 3072 + packed_c, end - 1 >= begin, other=0)
    gl.store(pool + (slot * 2048 + c) * 3, gl.where(end - 3 >= begin, x0, h0))
    gl.store(pool + (slot * 2048 + c) * 3 + 1, gl.where(end - 2 >= begin, x1, h1))
    gl.store(pool + (slot * 2048 + c) * 3 + 2, gl.where(end - 1 >= begin, x2, h2))


@gluon.jit
def _add_prefix(a, b):
    return a + b


@gluon.jit
def _multiply_prefix(a, b):
    return a * b


@gluon.jit
def _write_offsets(
    starts,
    chunk_offsets,
    block_offsets,
    BATCH: gl.constexpr,
    PAD_BATCH: gl.constexpr,
    BT: gl.constexpr,
    GROUP: gl.constexpr,
    HIERARCHICAL: gl.constexpr,
    NW: gl.constexpr,
):
    # One convolution CTA publishes the offsets. The next launch consumes them.
    layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
    sequence = gl.arange(0, PAD_BATCH, layout=layout)
    begin = gl.load(starts + sequence, sequence < BATCH, other=0)
    end = gl.load(starts + sequence + 1, sequence < BATCH, other=0)
    counts = gl.cdiv(end - begin, BT)
    prefix = gl.associative_scan(counts, 0, _add_prefix)
    gl.store(chunk_offsets + sequence + 1, prefix, sequence < BATCH)
    gl.store(chunk_offsets, 0)
    if HIERARCHICAL:
        block_counts = gl.cdiv(counts, GROUP)
        block_prefix = gl.associative_scan(block_counts, 0, _add_prefix)
        gl.store(block_offsets + sequence + 1, block_prefix, sequence < BATCH)
        gl.store(block_offsets, 0)


@gluon.jit
def _float_matrix_product(a, b, accumulator):
    mma: gl.constexpr = accumulator.type.layout
    aa = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 1))
    bb = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 1))
    return gl.amd.cdna4.mfma(aa, bb, accumulator)


@gluon.jit
def _store_chunk_rhs(pointer, value, K: gl.constexpr, N: gl.constexpr):
    # Each 16x16 tile puts a lane's four reduction registers together.
    packed = (
        value.reshape((K // 16, 4, 4, N // 16, 16))
        .permute((0, 3, 2, 4, 1))
        .reshape((K * N,))
    )
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    index = gl.arange(0, K * N, layout)
    gl.store(pointer + index, gl.convert_layout(packed, layout))


@gluon.jit
def _chunk_rhs_offset(row, col, N: gl.constexpr):
    tile = (row // 16) * (N // 16) + col // 16
    return tile * 256 + (row % 4) * 64 + (col % 16) * 4 + (row % 16) // 4


@gluon.jit
def _accumulator_offset(row, col, N: gl.constexpr):
    """Four adjacent registers per lane in a transposed 16x16 accumulator."""
    tile = (row // 16) * (N // 16) + col // 16
    return tile * 256 + ((col % 16) // 4) * 64 + (row % 16) * 4 + col % 4


@gluon.jit
def _store_chunk_accumulator(pointer, value, M: gl.constexpr, N: gl.constexpr):
    packed = (
        value.reshape((M // 16, 16, N // 16, 4, 4))
        .permute((0, 2, 3, 1, 4))
        .reshape((M * N,))
    )
    layout: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    index = gl.arange(0, M * N, layout)
    gl.store(pointer + index, gl.convert_layout(packed, layout))


@gluon.jit
def _load_chunk_u(base, row, col, BT: gl.constexpr, NATIVE: gl.constexpr):
    # Keep the record base scalar for both dense and accumulator-packed U.
    if NATIVE:
        return gl.amd.cdna4.buffer_load(
            base + BT * 128, _accumulator_offset(row, col, BT)
        )
    return gl.amd.cdna4.buffer_load(base + BT * 128, row * BT + col)


@gluon.jit
def _load_matrix_rhs(
    pointer,
    K: gl.constexpr,
    N: gl.constexpr,
    mma: gl.constexpr,
    PACKED: gl.constexpr = True,
):
    if not PACKED:
        standard_layout: gl.constexpr = gl.DotOperandLayout(1, mma, 1)
        row = gl.arange(0, K, gl.SliceLayout(1, standard_layout))
        col = gl.arange(0, N, gl.SliceLayout(0, standard_layout))
        return gl.amd.cdna4.buffer_load(pointer, row[:, None] * N + col[None, :])
    WM: gl.constexpr = mma.warps_per_cta[0]
    WN: gl.constexpr = mma.warps_per_cta[1]
    # Recurrence uses two column waves. The output emitter uses one and
    # consumes only 32-column matrices.
    n_registers: gl.constexpr = (
        [[512], [1024]] if N == 128 else ([[256]] if WN == 1 else [])
    )
    k_registers: gl.constexpr = (
        [[N * 16], [N * 32], [N * 64]] if K == 128 else [[N * 16]]
    )
    m_warps: gl.constexpr = [] if WM == 1 else [[0]] if WM == 2 else [[0], [0]]
    n_warps: gl.constexpr = [] if WN == 1 else [[256]]
    layout: gl.constexpr = gl.DistributedLinearLayout(
        [[1], [2]] + n_registers + k_registers,
        [[4], [8], [16], [32], [64], [128]],
        n_warps + m_warps,
        [],
        [K * N],
    )
    index = gl.arange(0, K * N, layout)
    packed = gl.amd.cdna4.buffer_load(pointer, index)
    value = (
        packed.reshape((K // 16, N // 16, 4, 16, 4))
        .permute((0, 4, 2, 1, 3))
        .reshape((K, N))
    )
    # MFMA assigns column waves first. Replicated row waves follow them.
    return gl.convert_layout(value, gl.DotOperandLayout(1, mma, 1), assert_trivial=True)


@gluon.jit
def _load_scaled_qk(
    base,
    raw_qk,
    qk_factors,
    record,
    BT: gl.constexpr,
    mma: gl.constexpr,
    PACKED: gl.constexpr,
    COMPACT_QK: gl.constexpr,
    IS_Q: gl.constexpr,
):
    if COMPACT_QK:
        # Re-form the FP32 coefficients exactly from shared BF16 Q/K.
        raw_base = raw_qk + (record // 2) * 2 * BT * 128
        rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 1)
        if IS_Q:
            value = _load_matrix_rhs(raw_base + BT * 128, 128, BT, mma)
            token = gl.arange(0, BT, gl.SliceLayout(0, rhs))
            factor = gl.load(qk_factors + record * 2 * BT + token)
            return value.to(gl.float32) * factor[None, :]
        else:
            value = _load_matrix_rhs(raw_base, BT, 128, mma)
            token = gl.arange(0, BT, gl.SliceLayout(1, rhs))
            factor = gl.load(qk_factors + record * 2 * BT + BT + token)
            return value.to(gl.float32) * factor[:, None]
    elif IS_Q:
        return _load_matrix_rhs(base + 3 * BT * 128, 128, BT, mma, PACKED=PACKED)
    else:
        return _load_matrix_rhs(base + 2 * BT * 128, BT, 128, mma, PACKED=PACKED)


@gluon.jit
def _solve_diagonal_blocks(triangular, BT: gl.constexpr):
    """One wave solves each independent 8-by-8 diagonal block."""
    IB: gl.constexpr = BT // 4
    gather_layout: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [4, 1], [1, 0])
    matrix = gl.convert_layout(triangular, gather_layout)
    r = gl.arange(0, BT, gl.SliceLayout(1, gather_layout))
    c = gl.arange(0, IB, gl.SliceLayout(0, gather_layout))
    columns = (r // IB)[:, None] * IB + c[None, :]
    diagonal = gl.gather(matrix, columns, 1)
    local_layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [1, 8, 8], [4, 1, 1], [2, 1, 0]
    )
    diagonal = gl.convert_layout(diagonal.reshape((4, IB, IB)), local_layout)
    for pivot in gl.static_range(1, IB - 1):
        row_index = gl.full((4, 1, IB), pivot, gl.int32, local_layout)
        col_index = gl.full((4, IB, 1), pivot, gl.int32, local_layout)
        pivot_row = gl.gather(diagonal, row_index, 1)
        pivot_col = gl.gather(diagonal, col_index, 2)
        diagonal = diagonal + pivot_col * pivot_row
    diagonal = gl.convert_layout(diagonal.reshape((BT, IB)), gather_layout)
    c_full = gl.arange(0, BT, gl.SliceLayout(0, gather_layout))
    repeat = r[:, None] * 0 + (c_full % IB)[None, :]
    expanded = gl.gather(diagonal, repeat, 1)
    expanded = gl.where((r // IB)[:, None] == (c_full // IB)[None, :], expanded, 0.0)
    return gl.convert_layout(expanded, triangular.type.layout)


_prepare_chunk_matrices_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_prepare_chunk_matrices",
    ["BATCH", "SEARCH_STEPS", "BT", "PACKED", "COMPACT_QK", "NATIVE_U"],
)


@gluon.jit(repr=_prepare_chunk_matrices_repr)
def _prepare_chunk_matrices(
    prepared,
    packed_ba,
    a_log,
    dt_bias,
    starts,
    offsets,
    matrices,
    output_weights,
    chunk_decay,
    chunk_begin,
    BATCH: gl.constexpr,
    SEARCH_STEPS: gl.constexpr,
    BT: gl.constexpr,
    PACKED: gl.constexpr = True,
    COMPACT_QK: gl.constexpr = False,
    raw_qk=None,
    qk_factors=None,
    NATIVE_U: gl.constexpr = False,
):
    # Each chunk stores W.T, U.T, scaled K, and scaled Q.T. The recurrence is
    # delta = U.T - H @ W.T; H_next = decay * H + delta @ scaled_K.
    # Its output is H @ scaled_Q.T + delta @ output_weights.T.
    head, chunk = gl.program_id(0), gl.program_id(1)
    count = gl.load(offsets + BATCH)
    if chunk < count:
        low, high = 0, BATCH
        for _ in gl.static_range(SEARCH_STEPS):
            middle = (low + high) // 2
            boundary = gl.load(offsets + middle)
            right = boundary <= chunk
            low = gl.where(right, middle + 1, low)
            high = gl.where(right, high, middle)
        sequence = low - 1
        sequence_chunk = gl.load(offsets + sequence)
        begin = gl.load(starts + sequence) + (chunk - sequence_chunk) * BT
        end = gl.load(starts + sequence + 1)
        if head == 0:
            gl.store(chunk_begin + chunk, begin)

        load_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
        row = gl.arange(0, BT, gl.SliceLayout(1, load_layout))
        col = gl.arange(0, BT, gl.SliceLayout(0, load_layout))
        k = gl.arange(0, 128, gl.SliceLayout(0, load_layout))
        token = begin + row
        valid = token < end
        q = gl.load(
            prepared + token[:, None] * 2048 + (head // 2) * 128 + k[None, :],
            valid[:, None],
            other=0,
        )
        key = gl.load(
            prepared + token[:, None] * 2048 + 512 + (head // 2) * 128 + k[None, :],
            valid[:, None],
            other=0,
        )
        value = gl.load(
            prepared + token[:, None] * 2048 + 1024 + head * 128 + k[None, :],
            valid[:, None],
            other=0,
        ).to(gl.float32)
        ba_offset = token * 16 + (head // 2) * 4 + head % 2
        b = gl.load(packed_ba + ba_offset, valid, other=0).to(gl.float32)
        a = gl.load(packed_ba + ba_offset + 2, valid, other=0).to(gl.float32)
        av = a + gl.load(dt_bias + head).to(gl.float32)
        softplus = gl.where(av <= 20.0, gl.log(1.0 + gl.exp(av)), av)
        log_decay = -gl.exp(gl.load(a_log + head)) * softplus
        decay = gl.where(valid, gl.exp(log_decay), 1.0)
        beta = (1.0 / (1.0 + gl.exp(-b))).to(gl.bfloat16).to(gl.float32)
        beta = gl.where(valid, beta, 0.0)
        scan_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
        scan_input = gl.convert_layout(decay, scan_layout)
        scan_row = gl.arange(0, BT, scan_layout)
        prefix = gl.convert_layout(
            gl.associative_scan(scan_input, 0, _multiply_prefix),
            gl.SliceLayout(1, load_layout),
        )
        shifted = gl.gather(scan_input, gl.minimum(scan_row + 1, BT - 1), 0)
        shifted = gl.where(scan_row + 1 < BT, shifted, 1.0)
        suffix = gl.convert_layout(
            gl.associative_scan(shifted, 0, _multiply_prefix, reverse=True),
            gl.SliceLayout(1, load_layout),
        )
        end_decay = gl.sum(gl.where(row == BT - 1, prefix, 0.0), 0)
        gl.store(chunk_decay + chunk * 8 + head, end_decay)

        bf_mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [2, 2])
        fp_mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 4], True, [2, 2])
        key_a = gl.convert_layout(key, gl.DotOperandLayout(0, bf_mma, 8))
        key_b = gl.convert_layout(key.T, gl.DotOperandLayout(1, bf_mma, 8))
        q_a = gl.convert_layout(q, gl.DotOperandLayout(0, bf_mma, 8))
        zero_bf = gl.zeros((BT, BT), gl.float32, bf_mma)
        # Retain Gram products and the triangular solve in MFMA ownership.
        kk = gl.convert_layout(gl.amd.cdna4.mfma(key_a, key_b, zero_bf), fp_mma)
        qk = gl.convert_layout(gl.amd.cdna4.mfma(q_a, key_b, zero_bf), fp_mma)
        lower = row[:, None] > col[None, :]
        causal_decay = gl.associative_scan(
            gl.where(lower, decay[:, None], 1.0), 0, _multiply_prefix
        )
        mma_row = gl.arange(0, BT, gl.SliceLayout(1, fp_mma))
        mma_col = gl.arange(0, BT, gl.SliceLayout(0, fp_mma))
        causal_mma = gl.convert_layout(causal_decay, fp_mma)
        beta_mma = gl.convert_layout(beta, gl.SliceLayout(1, fp_mma))
        triangular = gl.where(
            mma_row[:, None] > mma_col[None, :],
            -beta_mma[:, None] * kk * causal_mma,
            0.0,
        )

        IB: gl.constexpr = BT // 4
        diagonal_block = mma_row[:, None] // IB == mma_col[None, :] // IB
        diagonal = _solve_diagonal_blocks(triangular, BT)
        diagonal_inverse = diagonal + (mma_row[:, None] == mma_col[None, :]).to(
            gl.float32
        )
        off_diagonal = gl.where(diagonal_block, 0.0, triangular)
        # Publish both operands together; retain the small diagonal tile
        # for its later use as a right-hand operand.
        diagonal_shared = gl.allocate_shared_memory(
            gl.float32,
            (BT, BT),
            gl.SwizzledSharedLayout(4, 1, 8, [1, 0]),
            diagonal_inverse,
        )
        off_diagonal_shared = gl.allocate_shared_memory(
            gl.float32, (BT, BT), gl.SwizzledSharedLayout(4, 1, 8, [1, 0]), off_diagonal
        )
        diagonal_a = diagonal_shared.load(gl.DotOperandLayout(0, fp_mma, 1))
        off_diagonal_b = off_diagonal_shared.load(gl.DotOperandLayout(1, fp_mma, 1))
        block_system = gl.amd.cdna4.mfma(
            diagonal_a, off_diagonal_b, gl.zeros((BT, BT), gl.float32, fp_mma)
        )
        # Contract only the active eight-column pivot block.
        pivot_shared = gl.allocate_shared_memory(
            gl.float32, (BT, BT), gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
        )
        for pivot in gl.static_range(1, 3):
            pivot_shared.store(block_system)
            left = pivot_shared.slice(pivot * IB, IB, dim=1).load(
                gl.DotOperandLayout(0, fp_mma, 1)
            )
            right = pivot_shared.slice(pivot * IB, IB, dim=0).load(
                gl.DotOperandLayout(1, fp_mma, 1)
            )
            block_system = gl.amd.cdna4.mfma(left, right, block_system)
        pivot_shared.store(block_system)
        diagonal_b = diagonal_shared.load(gl.DotOperandLayout(1, fp_mma, 1))
        inverse = gl.amd.cdna4.mfma(
            pivot_shared.load(gl.DotOperandLayout(0, fp_mma, 1)),
            diagonal_b,
            diagonal_inverse,
        )
        # W and U share the inverse and reuse a weighted-input staging tile.
        inverse_shared = gl.allocate_shared_memory(
            gl.float32, (BT, BT), gl.SwizzledSharedLayout(4, 1, 8, [1, 0]), inverse
        )
        weighted_shared = gl.allocate_shared_memory(
            gl.float32,
            (BT, 128),
            gl.SwizzledSharedLayout(4, 1, 8, [1, 0]),
            key.to(gl.float32) * (beta * prefix)[:, None],
        )
        inverse_a = inverse_shared.load(gl.DotOperandLayout(0, fp_mma, 1))
        zero_fp = gl.zeros((BT, 128), gl.float32, fp_mma)
        w = gl.amd.cdna4.mfma(
            inverse_a, weighted_shared.load(gl.DotOperandLayout(1, fp_mma, 1)), zero_fp
        )
        weighted_shared.store(value * beta[:, None])
        u = gl.amd.cdna4.mfma(
            inverse_a, weighted_shared.load(gl.DotOperandLayout(1, fp_mma, 1)), zero_fp
        )
        scaled_key = key.to(gl.float32) * suffix[:, None]
        scaled_q = q.to(gl.float32) * prefix[:, None]
        base = matrices + (chunk * 8 + head) * (2 if COMPACT_QK else 4) * BT * 128
        address = row[:, None] * 128 + k[None, :]
        if COMPACT_QK:
            factor_base = qk_factors + (chunk * 8 + head) * 2 * BT
            gl.store(factor_base + row, prefix)
            gl.store(factor_base + BT + row, suffix)
            if head % 2 == 0:
                raw_base = raw_qk + (chunk * 4 + head // 2) * 2 * BT * 128
                _store_chunk_rhs(raw_base, key, BT, 128)
                _store_chunk_rhs(raw_base + BT * 128, q.T, 128, BT)
        elif PACKED:
            _store_chunk_rhs(base + 2 * BT * 128, scaled_key, BT, 128)
        else:
            gl.store(base + 2 * BT * 128 + address, scaled_key)
        store_layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
        store_k = gl.arange(0, 128, gl.SliceLayout(1, store_layout))
        store_t = gl.arange(0, BT, gl.SliceLayout(0, store_layout))
        transposed_offset = store_k[:, None] * BT + store_t[None, :]
        if PACKED:
            _store_chunk_rhs(base, w.T, 128, BT)
        else:
            gl.store(base + transposed_offset, gl.convert_layout(w.T, store_layout))
        if NATIVE_U:
            _store_chunk_accumulator(base + BT * 128, u.T, 128, BT)
        else:
            gl.store(
                base + BT * 128 + transposed_offset,
                gl.convert_layout(u.T, store_layout),
            )
        if not COMPACT_QK:
            if PACKED:
                _store_chunk_rhs(base + 3 * BT * 128, scaled_q.T, 128, BT)
            else:
                gl.store(
                    base + 3 * BT * 128 + transposed_offset,
                    gl.convert_layout(scaled_q.T, store_layout),
                )
        output_matrix = gl.where(
            mma_row[:, None] >= mma_col[None, :], qk * causal_mma, 0.0
        )
        if PACKED:
            _store_chunk_rhs(
                output_weights + (chunk * 8 + head) * BT * BT, output_matrix.T, BT, BT
            )
        else:
            gl.store(
                output_weights
                + (chunk * 8 + head) * BT * BT
                + col[None, :] * BT
                + row[:, None],
                gl.convert_layout(output_matrix, load_layout),
            )


@gluon.jit
def _store_core_output(
    core,
    output,
    scale,
    begin,
    end,
    head,
    value_block,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [NW, 1], [1, 0])
    token = begin + gl.arange(0, BT, gl.SliceLayout(1, layout))
    value = value_block * BV + gl.arange(0, BV, gl.SliceLayout(0, layout))
    rounded = gl.convert_layout((output * scale).to(gl.bfloat16).T, layout)
    gl.store(
        core + (token[:, None] * 8 + head) * 128 + value[None, :],
        rounded,
        token[:, None] < end,
    )


_chunk_delta_recurrence_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_chunk_delta_recurrence",
    ["BT", "BV", "NW", "WM", "PACKED", "TRANSPOSED", "PRELOAD", "COMPACT_QK"],
)


@gluon.jit(repr=_chunk_delta_recurrence_repr)
def _chunk_delta_recurrence(
    matrices,
    output_weights,
    chunk_decay,
    chunk_begin,
    offsets,
    states,
    indices,
    starts,
    core,
    scale,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    WM: gl.constexpr = 1,
    PACKED: gl.constexpr = True,
    TRANSPOSED: gl.constexpr = True,
    PRELOAD: gl.constexpr = 0,
    COMPACT_QK: gl.constexpr = False,
    raw_qk=None,
    qk_factors=None,
):
    sequence, head = gl.program_id(0), gl.program_id(1)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 4], TRANSPOSED, [WM, NW // WM])
    v = gl.program_id(2) * BV + gl.arange(0, BV, gl.SliceLayout(1, mma))
    k = gl.arange(0, 128, gl.SliceLayout(0, mma))
    slot = gl.load(indices + sequence)
    state_address = ((slot * 8 + head) * 128 + v[:, None]) * 128 + k[None, :]
    state = gl.load(states + state_address)
    first_chunk = gl.load(offsets + sequence)
    last_chunk = gl.load(offsets + sequence + 1)
    end_token = gl.load(starts + sequence + 1)
    out_t = gl.arange(0, BT, gl.SliceLayout(0, mma))
    if BV >= 64:
        # Put each MFMA operand's four K registers next to each other.
        # The high-batch scan reuses this full tile for state and delta.
        operand_layout: gl.constexpr = gl.SharedLinearLayout(
            [
                [0, 4],
                [0, 8],
                [1, 0],
                [2, 0],
                [4, 0],
                [8, 0],
                [0, 1],
                [0, 2],
                [0, 16],
                [0, 32],
                [0, 64],
                [16, 0],
                [32, 0],
            ]
        )
        operand_shared = gl.allocate_shared_memory(
            gl.float32, (BV, 128), operand_layout
        )
    for chunk in range(first_chunk, last_chunk):
        base = matrices + (chunk * 8 + head) * (2 if COMPACT_QK else 4) * BT * 128
        w = _load_matrix_rhs(base, 128, BT, mma, PACKED=PACKED)
        u = gl.load(base + BT * 128 + (v[:, None] * BT + out_t[None, :]))
        if PRELOAD == 1:
            scaled_key = _load_scaled_qk(
                base,
                raw_qk,
                qk_factors,
                chunk * 8 + head,
                BT,
                mma,
                PACKED,
                COMPACT_QK,
                False,
            )
            decay = gl.load(chunk_decay + chunk * 8 + head)
        zero_delta = gl.zeros((BV, BT), gl.float32, mma)
        if BV >= 64:
            operand_shared.store(state)
            state_a = operand_shared.load(gl.DotOperandLayout(0, mma, 1))
        else:
            state_a = state
        projected = _float_matrix_product(state_a, w, zero_delta)
        delta = u - projected
        scaled_q = _load_scaled_qk(
            base,
            raw_qk,
            qk_factors,
            chunk * 8 + head,
            BT,
            mma,
            PACKED,
            COMPACT_QK,
            True,
        )
        y = _float_matrix_product(state_a, scaled_q, zero_delta)
        output_matrix = _load_matrix_rhs(
            output_weights + (chunk * 8 + head) * BT * BT, BT, BT, mma, PACKED=PACKED
        )
        if PRELOAD == 3:
            scaled_key = _load_scaled_qk(
                base,
                raw_qk,
                qk_factors,
                chunk * 8 + head,
                BT,
                mma,
                PACKED,
                COMPACT_QK,
                False,
            )
            decay = gl.load(chunk_decay + chunk * 8 + head)
        if BV >= 64:
            delta_shared = operand_shared.slice(0, BT, dim=1)
            delta_shared.store(delta)
            delta_a = delta_shared.load(gl.DotOperandLayout(0, mma, 1))
        else:
            delta_a = delta
        y = _float_matrix_product(delta_a, output_matrix, y)
        begin_token = gl.load(chunk_begin + chunk)
        if PACKED:
            gl.store(
                core + ((begin_token + out_t[None, :]) * 8 + head) * 128 + v[:, None],
                y * scale,
                begin_token + out_t[None, :] < end_token,
            )
        else:
            _store_core_output(
                core,
                y,
                scale,
                begin_token,
                end_token,
                head,
                gl.program_id(2),
                BT,
                BV,
                NW,
            )
        if PRELOAD == 0:
            scaled_key = _load_scaled_qk(
                base,
                raw_qk,
                qk_factors,
                chunk * 8 + head,
                BT,
                mma,
                PACKED,
                COMPACT_QK,
                False,
            )
            decay = gl.load(chunk_decay + chunk * 8 + head)
        state = _float_matrix_product(delta_a, scaled_key, state * decay)
    gl.store(states + state_address, state)


_scan_chunk_states_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_scan_chunk_states", ["BT", "BV", "NW"]
)


@gluon.jit(repr=_scan_chunk_states_repr)
def _scan_chunk_states(
    matrices,
    chunk_decay,
    offsets,
    states,
    indices,
    snapshots,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
):
    """Scan only the state dependency; independent CTAs later emit outputs."""
    sequence, head = gl.program_id(0), gl.program_id(1)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 4], True, [1, NW])
    v = gl.program_id(2) * BV + gl.arange(0, BV, gl.SliceLayout(1, mma))
    k = gl.arange(0, 128, gl.SliceLayout(0, mma))
    t = gl.arange(0, BT, gl.SliceLayout(0, mma))
    slot = gl.load(indices + sequence)
    address = ((slot * 8 + head) * 128 + v[:, None]) * 128 + k[None, :]
    state = gl.load(states + address)
    first = gl.load(offsets + sequence)
    last = gl.load(offsets + sequence + 1)
    for chunk in range(first, last):
        gl.store(
            snapshots
            + (chunk * 8 + head) * 16384
            + _accumulator_offset(v[:, None], k[None, :], 128),
            state,
        )
        base = matrices + (chunk * 8 + head) * 4 * BT * 128
        w = _load_matrix_rhs(base, 128, BT, mma)
        u_address = base + BT * 128 + _accumulator_offset(v[:, None], t[None, :], BT)
        u = gl.load(u_address)
        delta = u - _float_matrix_product(state, w, gl.zeros((BV, BT), gl.float32, mma))
        # U is dead after this scan; reuse its storage for solved updates.
        gl.store(u_address, delta)
        scaled_key = _load_matrix_rhs(base + 2 * BT * 128, BT, 128, mma)
        decay = gl.load(chunk_decay + chunk * 8 + head)
        state = _float_matrix_product(delta, scaled_key, state * decay)
    gl.store(states + address, state)


_emit_chunk_outputs_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_emit_chunk_outputs", ["BATCH", "SEARCH_STEPS", "BT"]
)


@gluon.jit(repr=_emit_chunk_outputs_repr)
def _emit_chunk_outputs(
    matrices,
    output_weights,
    snapshots,
    chunk_begin,
    offsets,
    packed,
    weight,
    normalized,
    quantized,
    scales,
    pool,
    starts,
    indices,
    initial,
    scale,
    eps,
    maximum,
    inverse_maximum,
    BATCH: gl.constexpr,
    SEARCH_STEPS: gl.constexpr,
    BT: gl.constexpr,
):
    chunk, head = gl.program_id(0), gl.program_id(1)
    if chunk < BATCH:
        _update_conv_pool(packed, pool, starts, indices, initial, chunk, head, 256)
    count = gl.load(offsets + BATCH)
    if chunk < count:
        low, high = 0, BATCH
        for _ in gl.static_range(SEARCH_STEPS):
            middle = (low + high) // 2
            right = gl.load(offsets + middle) <= chunk
            low = gl.where(right, middle + 1, low)
            high = gl.where(right, high, middle)
        sequence = low - 1
        begin = gl.load(chunk_begin + chunk)
        end = gl.load(starts + sequence + 1)
        mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 4], True, [4, 1])
        v = gl.arange(0, 128, gl.SliceLayout(1, mma))
        k = gl.arange(0, 128, gl.SliceLayout(0, mma))
        t = gl.arange(0, BT, gl.SliceLayout(0, mma))
        base = matrices + (chunk * 8 + head) * 4 * BT * 128
        state = gl.load(
            snapshots
            + (chunk * 8 + head) * 16384
            + _accumulator_offset(v[:, None], k[None, :], 128)
        )
        scaled_q = _load_matrix_rhs(base + 3 * BT * 128, 128, BT, mma)
        y = _float_matrix_product(state, scaled_q, gl.zeros((128, BT), gl.float32, mma))
        delta = _load_chunk_u(base, v[:, None], t[None, :], BT, True)
        output_matrix = _load_matrix_rhs(
            output_weights + (chunk * 8 + head) * BT * BT, BT, BT, mma
        )
        y = _float_matrix_product(delta, output_matrix, y)

        layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
        token = begin + gl.arange(0, BT, gl.SliceLayout(1, layout))
        d = gl.arange(0, 128, gl.SliceLayout(0, layout))
        x = gl.convert_layout((y * scale).to(gl.bfloat16).T, layout).to(gl.float32)
        z = gl.load(
            packed
            + token[:, None] * 3072
            + (head // 2) * 768
            + 512
            + (head % 2) * 128
            + d[None, :],
            token[:, None] < end,
            other=0,
        ).to(gl.float32)
        norm_weight = gl.load(weight + d).to(gl.float32)
        rms = gl.rsqrt(gl.sum(x * x, 1) / 128 + eps)
        sigmoid = 1.0 / (1.0 + gl.exp(-z))
        result = (x * rms[:, None] * norm_weight[None, :] * z * sigmoid).to(gl.bfloat16)
        address = (token[:, None] * 8 + head) * 128 + d[None, :]
        gl.store(normalized + address, result, token[:, None] < end)
        result_f32 = result.to(gl.float32)
        quant_scale = (
            gl.maximum(gl.max(gl.abs(result_f32), 1), 1.0e-10) * inverse_maximum
        )
        codes = gl.clamp(
            result_f32 * gl.div_rn(1.0, quant_scale[:, None]), -maximum, maximum
        )
        gl.store(quantized + address, codes, token[:, None] < end)
        gl.store(scales + token * 8 + head, quant_scale, token < end)


@gluon.jit
def _store_packed_linear(pointer, value, ROWS: gl.constexpr, NW: gl.constexpr):
    # Pack each 16x16 tile as [K_low_2, N_low_4, K_middle_2].
    n_registers: gl.constexpr = [[512], [1024]] if NW == 2 else [[1024]]
    k_registers: gl.constexpr = (
        [[2048], [4096], [8192]]
        if ROWS == 128
        else [[2048], [4096]] if ROWS == 64 else [[2048]] if ROWS == 32 else []
    )
    warp_bases: gl.constexpr = [[256]] if NW == 2 else [[256], [512]]
    layout: gl.constexpr = gl.DistributedLinearLayout(
        [[1], [2]] + n_registers + k_registers,
        [[4], [8], [16], [32], [64], [128]],
        warp_bases,
        [],
        [ROWS * 128],
    )
    packed = (
        value.reshape((ROWS // 16, 4, 4, 8, 16))
        .permute((0, 3, 2, 4, 1))
        .reshape((ROWS * 128,))
    )
    packed = gl.convert_layout(packed, layout)
    index = gl.arange(0, ROWS * 128, layout)
    gl.store(pointer + index, packed)


@gluon.jit
def _load_packed_linear(pointer, mma: gl.constexpr):
    # Affine checkpoint propagation uses four column waves.
    n_registers: gl.constexpr = [[1024]]
    warp_bases: gl.constexpr = [[256], [512]]
    layout: gl.constexpr = gl.DistributedLinearLayout(
        [[1], [2]] + n_registers + [[2048], [4096], [8192]],
        [[4], [8], [16], [32], [64], [128]],
        warp_bases,
        [],
        [16384],
    )
    index = gl.arange(0, 16384, layout)
    packed = gl.load(pointer + index)
    value = (
        packed.reshape((8, 8, 4, 16, 4)).permute((0, 4, 2, 1, 3)).reshape((128, 128))
    )
    return gl.convert_layout(value, gl.DotOperandLayout(1, mma, 1))


_block_transforms_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_block_transforms",
    [
        "BATCH",
        "SEARCH_STEPS",
        "BT",
        "GROUP",
        "BV",
        "NW",
        "WM",
        "DIRECT_FIRST",
        "NATIVE_U",
    ],
)


@gluon.jit(repr=_block_transforms_repr)
def _block_transforms(
    matrices,
    chunk_decay,
    chunk_offsets,
    block_offsets,
    transforms,
    block_info,
    states,
    indices,
    BATCH: gl.constexpr,
    SEARCH_STEPS: gl.constexpr,
    BT: gl.constexpr,
    GROUP: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    WM: gl.constexpr = 1,
    DIRECT_FIRST: gl.constexpr = True,
    NATIVE_U: gl.constexpr = False,
):
    # Zero-state and identity-state passes produce bias and linear maps.
    block, head = gl.program_id(0), gl.program_id(1)
    blocks = gl.load(block_offsets + BATCH)
    if block < blocks:
        low, high = 0, BATCH
        for _ in gl.static_range(SEARCH_STEPS):
            middle = (low + high) // 2
            boundary = gl.load(block_offsets + middle)
            right = boundary <= block
            low = gl.where(right, middle + 1, low)
            high = gl.where(right, high, middle)
        sequence = low - 1
        sequence_first = gl.load(chunk_offsets + sequence)
        end = gl.load(chunk_offsets + sequence + 1)
        first = sequence_first + (block - gl.load(block_offsets + sequence)) * GROUP
        last = gl.minimum(first + GROUP, end)
        value_block = gl.program_id(2) % (128 // BV)
        is_linear = gl.program_id(2) >= 128 // BV
        if (head == 0) & (gl.program_id(2) == 0):
            gl.store(block_info + block * 3, first)
            gl.store(block_info + block * 3 + 1, last)
            gl.store(block_info + block * 3 + 2, sequence)

        # A terminal block has no successor checkpoint. A first block can
        # propagate the supplied state directly, without a linear map.
        if (last < end) & (
            (not DIRECT_FIRST) | (first != sequence_first) | (~is_linear)
        ):
            mma: gl.constexpr = gl.amd.AMDMFMALayout(
                4, [16, 16, 4], True, [WM, NW // WM]
            )
            v = value_block * BV + gl.arange(0, BV, gl.SliceLayout(1, mma))
            k = gl.arange(0, 128, gl.SliceLayout(0, mma))
            out_t = gl.arange(0, BT, gl.SliceLayout(0, mma))

            if DIRECT_FIRST:
                base = matrices + (first * 8 + head) * 4 * BT * 128
                seed = gl.zeros((BV, 128), gl.float32, mma)
                if first == sequence_first:
                    slot = gl.load(indices + sequence)
                    seed = gl.load(
                        states
                        + ((slot * 8 + head) * 128 + v[:, None]) * 128
                        + k[None, :]
                    )
                    w = _load_matrix_rhs(base, 128, BT, mma)
                    initial_delta = _load_chunk_u(
                        base, v[:, None], out_t[None, :], BT, NATIVE_U
                    )
                    initial_delta = initial_delta - _float_matrix_product(
                        seed, w, gl.zeros((BV, BT), gl.float32, mma)
                    )
                elif is_linear:
                    initial_delta = -gl.load(
                        base + _chunk_rhs_offset(v[:, None], out_t[None, :], BT)
                    )
                    seed = (v[:, None] == k[None, :]).to(gl.float32)
                else:
                    initial_delta = _load_chunk_u(
                        base, v[:, None], out_t[None, :], BT, NATIVE_U
                    )
                scaled_key = _load_matrix_rhs(base + 2 * BT * 128, BT, 128, mma)
                decay = gl.load(chunk_decay + first * 8 + head)
                state = _float_matrix_product(initial_delta, scaled_key, seed * decay)
            else:
                base = matrices + (first * 8 + head) * 4 * BT * 128
                if is_linear:
                    initial_delta = -gl.load(
                        base + _chunk_rhs_offset(v[:, None], out_t[None, :], BT)
                    )
                else:
                    initial_delta = _load_chunk_u(
                        base, v[:, None], out_t[None, :], BT, NATIVE_U
                    )
                scaled_key = _load_matrix_rhs(base + 2 * BT * 128, BT, 128, mma)
                decay = gl.load(chunk_decay + first * 8 + head)
                identity = ((v[:, None] == k[None, :]) & is_linear).to(gl.float32)
                state = _float_matrix_product(
                    initial_delta, scaled_key, identity * decay
                )
            for chunk in range(first + 1, last):
                base = matrices + (chunk * 8 + head) * 4 * BT * 128
                w = _load_matrix_rhs(base, 128, BT, mma)
                if BV == 64:
                    scaled_key = _load_matrix_rhs(base + 2 * BT * 128, BT, 128, mma)
                    decay = gl.load(chunk_decay + chunk * 8 + head)
                    u = gl.zeros((BV, BT), gl.float32, mma)
                    if not is_linear:
                        u = _load_chunk_u(
                            base, v[:, None], out_t[None, :], BT, NATIVE_U
                        )
                projected = _float_matrix_product(
                    state, w, gl.zeros((BV, BT), gl.float32, mma)
                )
                delta = -projected
                if not is_linear:
                    if BV != 64:
                        u = _load_chunk_u(
                            base, v[:, None], out_t[None, :], BT, NATIVE_U
                        )
                    delta = delta + u
                if BV != 64:
                    scaled_key = _load_matrix_rhs(base + 2 * BT * 128, BT, 128, mma)
                    decay = gl.load(chunk_decay + chunk * 8 + head)
                state = _float_matrix_product(delta, scaled_key, state * decay)
            address = ((block * 8 + head) * 2 + is_linear.to(gl.int32)) * 16384
            if is_linear:
                _store_packed_linear(
                    transforms + address + value_block * BV * 128, state, BV, NW
                )
            else:
                gl.store(transforms + address + v[:, None] * 128 + k[None, :], state)


_propagate_block_states_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_propagate_block_states", ["BV", "NW", "DIRECT_FIRST"]
)


@gluon.jit(repr=_propagate_block_states_repr)
def _propagate_block_states(
    transforms,
    block_offsets,
    states,
    indices,
    BV: gl.constexpr,
    NW: gl.constexpr,
    DIRECT_FIRST: gl.constexpr = True,
):
    # Publish each incoming checkpoint over its now-dead bias tile.
    sequence, head = gl.program_id(0), gl.program_id(1)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 4], True, [1, NW])
    v = gl.program_id(2) * BV + gl.arange(0, BV, gl.SliceLayout(1, mma))
    k = gl.arange(0, 128, gl.SliceLayout(0, mma))
    slot = gl.load(indices + sequence)
    state = gl.load(states + ((slot * 8 + head) * 128 + v[:, None]) * 128 + k[None, :])
    first = gl.load(block_offsets + sequence)
    last = gl.load(block_offsets + sequence + 1)
    element = v[:, None] * 128 + k[None, :]
    if DIRECT_FIRST:
        if first + 1 < last:
            base = transforms + (first * 8 + head) * 2 * 16384
            outgoing = gl.load(base + element)
            gl.store(transforms + (first * 8 + head) * 2 * 16384 + element, state)
            state = outgoing
            for block in range(first + 1, last - 1):
                base = transforms + (block * 8 + head) * 2 * 16384
                linear = _load_packed_linear(base + 16384, mma)
                bias = gl.load(base + element)
                gl.store(transforms + (block * 8 + head) * 2 * 16384 + element, state)
                state = _float_matrix_product(state, linear, bias)
    else:
        for block in range(first, last - 1):
            base = transforms + (block * 8 + head) * 2 * 16384
            linear = _load_packed_linear(base + 16384, mma)
            bias = gl.load(base + element)
            gl.store(transforms + (block * 8 + head) * 2 * 16384 + element, state)
            state = _float_matrix_product(state, linear, bias)
    if first < last:
        gl.store(transforms + ((last - 1) * 8 + head) * 2 * 16384 + element, state)


_finish_blocks_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_finish_blocks",
    ["BATCH", "BT", "BV", "NW", "WM", "UNROLL", "NATIVE_U"],
)


@gluon.jit(repr=_finish_blocks_repr)
def _finish_blocks(
    matrices,
    output_weights,
    chunk_decay,
    chunk_begin,
    chunk_offsets,
    block_offsets,
    block_info,
    checkpoints,
    states,
    indices,
    starts,
    core,
    scale,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    WM: gl.constexpr = 1,
    UNROLL: gl.constexpr = 4,
    NATIVE_U: gl.constexpr = False,
):
    block, head = gl.program_id(0), gl.program_id(1)
    count = gl.load(block_offsets + BATCH)
    if block < count:
        first = gl.load(block_info + block * 3)
        last = gl.load(block_info + block * 3 + 1)
        sequence = gl.load(block_info + block * 3 + 2)
        end_token = gl.load(starts + sequence + 1)
        mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 4], True, [WM, NW // WM])
        v = gl.program_id(2) * BV + gl.arange(0, BV, gl.SliceLayout(1, mma))
        k = gl.arange(0, 128, gl.SliceLayout(0, mma))
        state = gl.load(
            checkpoints + (block * 8 + head) * 2 * 16384 + v[:, None] * 128 + k[None, :]
        )
        out_t = gl.arange(0, BT, gl.SliceLayout(0, mma))
        if BV >= 64:
            # A full state tile avoids the split conversion's extra barrier.
            operand_shared = gl.allocate_shared_memory(
                gl.float32, (BV, 128), gl.SwizzledSharedLayout(4, 1, 8, [1, 0])
            )
        for chunk in loop_range(first, last, loop_unroll_factor=UNROLL):
            base = matrices + (chunk * 8 + head) * 4 * BT * 128
            w = _load_matrix_rhs(base, 128, BT, mma)
            u = _load_chunk_u(base, v[:, None], out_t[None, :], BT, NATIVE_U)
            zero_delta = gl.zeros((BV, BT), gl.float32, mma)
            if BV >= 64:
                operand_shared.store(state)
                state_a = operand_shared.load(gl.DotOperandLayout(0, mma, 1))
            else:
                state_a = state
            projected = _float_matrix_product(state_a, w, zero_delta)
            delta = u - projected
            scaled_q = _load_matrix_rhs(base + 3 * BT * 128, 128, BT, mma)
            y = _float_matrix_product(state_a, scaled_q, zero_delta)
            output_matrix = _load_matrix_rhs(
                output_weights + (chunk * 8 + head) * BT * BT, BT, BT, mma
            )
            scaled_key = _load_matrix_rhs(base + 2 * BT * 128, BT, 128, mma)
            decay = gl.load(chunk_decay + chunk * 8 + head)
            if BV >= 64:
                delta_shared = operand_shared.slice(0, BT, dim=1)
                delta_shared.store(delta)
                delta_a = delta_shared.load(gl.DotOperandLayout(0, mma, 1))
            else:
                delta_a = delta
            y = _float_matrix_product(delta_a, output_matrix, y)
            begin_token = gl.load(chunk_begin + chunk)
            if BV == 16 and NW == 2:
                # Narrow strips need only lane-local offsets from a scalar base.
                core_base = core + (begin_token * 8 + head) * 128
                core_offset = out_t[None, :] * 1024 + v[:, None]
                gl.amd.cdna4.buffer_store(
                    (y * scale).to(core.dtype.element_ty),
                    core_base,
                    core_offset,
                    begin_token + out_t[None, :] < end_token,
                )
            else:
                gl.store(
                    core
                    + ((begin_token + out_t[None, :]) * 8 + head) * 128
                    + v[:, None],
                    y * scale,
                    begin_token + out_t[None, :] < end_token,
                )
            state = _float_matrix_product(delta_a, scaled_key, state * decay)
        if last == gl.load(chunk_offsets + sequence + 1):
            slot = gl.load(indices + sequence)
            gl.store(
                states + ((slot * 8 + head) * 128 + v[:, None]) * 128 + k[None, :],
                state,
            )


_norm_and_quantize_repr = make_kernel_repr(
    "gdn_prefill_m3072_16384_norm_and_quantize", ["M", "ROWS", "BATCH"]
)


@gluon.jit(repr=_norm_and_quantize_repr)
def _norm_and_quantize(
    core,
    packed,
    weight,
    normalized,
    quantized,
    scales,
    pool,
    starts,
    indices,
    initial,
    eps,
    maximum,
    inverse_maximum,
    M: gl.constexpr,
    ROWS: gl.constexpr,
    BATCH: gl.constexpr,
):
    if gl.program_id(0) < BATCH * 8:
        _update_conv_pool(
            packed,
            pool,
            starts,
            indices,
            initial,
            gl.program_id(0) // 8,
            gl.program_id(0) % 8,
            256,
        )
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [4, 1], [1, 0])
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, gl.SliceLayout(1, layout))
    d = gl.arange(0, 128, gl.SliceLayout(0, layout))
    token, head = row // 8, row % 8
    x = gl.load(
        core + row[:, None] * 128 + d[None, :], row[:, None] < M * 8, other=0
    ).to(gl.float32)
    z = gl.load(
        packed
        + token[:, None] * 3072
        + (head // 2)[:, None] * 768
        + 512
        + (head % 2)[:, None] * 128
        + d[None, :],
        row[:, None] < M * 8,
        other=0,
    ).to(gl.float32)
    w = gl.load(weight + d).to(gl.float32)
    rms = gl.rsqrt(gl.sum(x * x, 1) / 128 + eps)
    sigmoid = 1.0 / (1.0 + gl.exp(-z))
    result = (x * rms[:, None] * w[None, :] * z * sigmoid).to(gl.bfloat16)
    gl.store(normalized + row[:, None] * 128 + d[None, :], result, row[:, None] < M * 8)
    result_f32 = result.to(gl.float32)
    quant_scale = gl.maximum(gl.max(gl.abs(result_f32), 1), 1.0e-10) * inverse_maximum
    codes = gl.clamp(
        result_f32 * gl.div_rn(1.0, quant_scale[:, None]), -maximum, maximum
    )
    gl.store(quantized + row[:, None] * 128 + d[None, :], codes, row[:, None] < M * 8)
    gl.store(scales + row, quant_scale, row < M * 8)
