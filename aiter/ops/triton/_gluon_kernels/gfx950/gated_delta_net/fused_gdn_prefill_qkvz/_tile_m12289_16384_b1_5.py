"""Gated-delta prefill using FP32 chunk solves and segmented recurrence.

Wave-local convolution and compact 8x8 inverses preserve the BF16 materialization
boundaries. Narrow FP32 prefixes propagate segment entry states; 64-column reverse
summaries stack the transform and bias products to share operand conversions.
Long, moderately batched inputs use 1024-token segments to reduce prefix work.
Tiny transforms traverse heads first; direct preweighted inputs separate QKV
head planes by eight rows. Preparation uses one wave above 2K tokens.
Direct scans use shape-specific MFMA ownership. Time-major W factors keep their
original sign until operand preparation; feature-major factors are pre-negated.
All ragged boundaries and cache ownership remain runtime device data. Consumed
scratch is initialized on every invocation and reused only after its final reader.
"""

from dataclasses import dataclass

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr


@dataclass(frozen=True)
class FP8PrecisionConfig:
    dtype: torch.dtype
    max_finite: float
    group_quant_max: float


FP8_E4M3_FN = FP8PrecisionConfig(torch.float8_e4m3fn, 448.0, 448.0)


@gluon.jit
def _gate_values(a, b, decay_weight, valid=None):
    """FP32 decay and beta with the required intermediate BF16 rounding."""
    softplus = gl.where(a <= 20.0, gl.log(1.0 + gl.exp(a)), a)
    decay = gl.exp(-decay_weight * softplus)
    if valid is not None:
        decay = gl.where(valid, decay, 1.0)
    beta = (1.0 / (1.0 + gl.exp(-b))).to(gl.bfloat16).to(gl.float32)
    if valid is not None:
        beta = gl.where(valid, beta, 0.0)
    return decay, beta


_prepare_qkv_window_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_prepare_qkv_window", ["M", "BATCH", "OUT", "NW"]
)


@gluon.jit(repr=_prepare_qkv_window_repr)
def _prepare_qkv_window(
    Projected,
    ConvState,
    Indices,
    Starts,
    Initial,
    Weight,
    Bias,
    QKV,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    OUT: gl.constexpr,
    NW: gl.constexpr,
    Bounds=None,
):
    work = gl.program_id(0)
    group = gl.program_id(1)
    BT: gl.constexpr = OUT * NW
    WIN: gl.constexpr = 1 << ((OUT + 2).bit_length())
    gl.static_assert(32 % BT == 0)
    tile = work // (32 // BT)
    part = work % (32 // BT)
    seq, first, end = _chunk_bounds(Starts, tile, BATCH, 32)
    if Bounds is not None:
        if (group == 0) & (part == 0):
            gl.store(Bounds + tile * 2, first)
            gl.store(Bounds + tile * 2 + 1, end)
    first += part * BT
    begin = gl.load(Starts + seq)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2], [1, 1, 64], [NW, 1, 1], [2, 1, 0]
    )
    wave = gl.arange(0, NW, layout=gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    ti = gl.arange(0, WIN, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    to = gl.arange(0, OUT, layout=gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    channel = group * 128 + c
    packed = (
        gl.where(
            group < 4,
            group * 768,
            gl.where(
                group < 8,
                (group - 4) * 768 + 128,
                ((group - 8) // 2) * 768 + 256 + (group % 2) * 128,
            ),
        )
        + c
    )
    # Each wave loads its convolution window once; gathers are register-local.
    previous = first + wave[:, None] * OUT + ti[None, :] - 3
    x = gl.load(
        Projected + previous[:, :, None] * 3072 + packed[None, None, :],
        (previous[:, :, None] >= begin) & (previous[:, :, None] < end),
        0,
    )
    # Interior windows never need history, including for ragged sequences.
    if first == begin:
        slot = gl.load(Indices + seq)
        cached = gl.load(Initial + seq)
        history = gl.load(
            ConvState
            + (slot * 2048 + channel[None, None, :]) * 3
            + previous[:, :, None]
            - begin
            + 3,
            (previous[:, :, None] < begin)
            & (previous[:, :, None] >= begin - 3)
            & cached,
            0,
        )
        x = gl.where(previous[:, :, None] >= begin, x, history)
    x = x.to(gl.float32)
    conv = gl.load(Bias + channel)[None, None, :].to(gl.float32)
    for tap in gl.static_range(4):
        indices = to[None, :, None] + tap + gl.full((NW, OUT, 128), 0, gl.int32, layout)
        shifted = gl.gather(x, indices, 1)
        weight = gl.load(Weight + channel * 4 + tap).to(gl.float32)
        # Adjacent channel pairs can use native packed FP32 FMAs.
        conv = gl.fma(shifted, weight[None, None, :], conv)
    activated = gl.div_rn(conv, 1.0 + libdevice.exp(-conv)).to(gl.bfloat16)
    if group < 8:
        f = activated.to(gl.float32)
        inverse_norm = gl.rsqrt(gl.sum(f * f, 2) + 1.0e-6)
        activated = (f * inverse_norm[:, :, None]).to(gl.bfloat16)
    token = first + wave[:, None] * OUT + to[None, :]
    # Initialize padding on every invocation so all downstream loads are safe.
    scratch_token = tile * 32 + part * BT + wave[:, None] * OUT + to[None, :]
    gl.store(
        QKV + (group * M + scratch_token[:, :, None]) * 128 + c[None, None, :],
        gl.where(token[:, :, None] < end, activated, 0),
    )
    gl.static_assert(BT >= 3)
    if first == begin:
        slot = gl.load(Indices + seq)
        cached = gl.load(Initial + seq)
        # Only this CTA reads history for this sequence/channel group.
        # Finish its readers before updating any cache element.
        if NW > 1:
            gl.barrier()
        t0 = end - 3
        t1 = end - 2
        t2 = end - 1
        cache = ConvState + (slot * 2048 + channel) * 3
        h0 = gl.load(cache + t0 - begin + 3, (t0 < begin) & cached, 0)
        h1 = gl.load(cache + t1 - begin + 3, (t1 < begin) & cached, 0)
        h2 = gl.load(cache + t2 - begin + 3, (t2 < begin) & cached, 0)
        x0 = gl.load(Projected + t0 * 3072 + packed, t0 >= begin, 0)
        x1 = gl.load(Projected + t1 * 3072 + packed, t1 >= begin, 0)
        x2 = gl.load(Projected + t2 * 3072 + packed, t2 >= begin, 0)
        gl.store(cache, gl.where(t0 >= begin, x0, h0))
        gl.store(cache + 1, gl.where(t1 >= begin, x1, h1))
        gl.store(cache + 2, gl.where(t2 >= begin, x2, h2))


_prepare_gates_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_prepare_gates", ["M", "BATCH", "BT"]
)


@gluon.jit(repr=_prepare_gates_repr)
def _prepare_gates(
    BA,
    Starts,
    ALog,
    DTBias,
    Gates,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
    Bounds=None,
):
    chunk = gl.program_id(0)
    if Bounds is None:
        seq, first, end = _chunk_bounds(Starts, chunk, BATCH, BT)
    else:
        first = gl.load(Bounds + chunk * 2)
        end = gl.load(Bounds + chunk * 2 + 1)
    layout: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [4, 1], [1, 0])
    head = gl.arange(0, 8, layout=gl.SliceLayout(1, layout))
    i = gl.arange(0, BT, layout=gl.SliceLayout(0, layout))
    token = first + i
    base = token[None, :] * 16 + (head[:, None] // 2) * 4 + head[:, None] % 2
    av = gl.load(BA + base + 2, token[None, :] < end, 0).to(gl.float32)
    bv = gl.load(BA + base, token[None, :] < end, 0).to(gl.float32)
    av = av + gl.load(DTBias + head).to(gl.float32)[:, None]
    decay_weight = gl.exp(gl.load(ALog + head).to(gl.float32))
    decay, beta = _gate_values(av, bv, decay_weight[:, None])
    scratch_token = chunk * BT + i
    gl.store(
        Gates + (head[:, None] * 2) * M + scratch_token[None, :],
        gl.where(token[None, :] < end, decay, 1.0),
    )
    gl.store(
        Gates + (head[:, None] * 2 + 1) * M + scratch_token[None, :],
        gl.where(token[None, :] < end, beta, 0.0),
    )


@gluon.jit
def _multiply(a, b):
    return a * b


@gluon.jit
def _dot_f32(
    a,
    b,
    initial=None,
    WARPS: gl.constexpr = (2, 2),
    MMA_SIZE: gl.constexpr = 16,
    TRANSPOSED: gl.constexpr = True,
):
    """FP32 MFMA, optionally accumulating into an existing state tile."""
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[MMA_SIZE, MMA_SIZE, 64 // MMA_SIZE],
        transposed=TRANSPOSED,
        warps_per_cta=WARPS,
    )
    aa = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 1))
    bb = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 1))
    if initial is None:
        acc = gl.zeros((a.shape[0], b.shape[1]), gl.float32, layout=mma)
    else:
        acc = gl.convert_layout(initial, mma)
    return gl.amd.cdna3.mfma(aa, bb, acc)


@gluon.jit
def _dot_bf16(a, b):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    aa = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
    bb = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 8))
    acc = gl.zeros((a.shape[0], b.shape[1]), gl.float32, layout=mma)
    return gl.amd.cdna3.mfma(aa, bb, acc)


@gluon.jit
def _chunk_bounds(Starts, tile, BATCH: gl.constexpr, BT: gl.constexpr):
    lo = 0
    hi = BATCH
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        boundary = gl.load(Starts + mid) // BT + mid
        right = tile >= boundary
        lo = gl.where(right, mid, lo)
        hi = gl.where(right, hi, mid)
    begin = gl.load(Starts + lo)
    end = gl.load(Starts + lo + 1)
    first = begin + (tile - begin // BT - lo) * BT
    return lo, first, end


@gluon.jit
def _invert_diagonal_sixteen(lower, NATIVE: gl.constexpr):
    """Return the two diagonal inverses without expanding a sparse 32x32 tile."""
    gl.static_assert(lower.shape[0] == 32 and lower.shape[1] == 32)
    # One wave owns each complete 8x8 diagonal block. Forward substitution
    # uses only wave-local gathers; off-diagonal blocks are joined by MFMA.
    blocks: gl.constexpr = 4
    square: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [4, 1], [1, 0])
    four: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1, 1], [1, 8, 1, 8], [4, 1, 1, 1], [3, 1, 2, 0]
    )
    three: gl.constexpr = gl.BlockedLayout([1, 1, 1], [1, 8, 8], [4, 1, 1], [2, 1, 0])
    matrix = gl.convert_layout(lower, square).reshape((blocks, 8, blocks, 8))
    matrix = gl.convert_layout(matrix, four)
    br = gl.arange(
        0, blocks, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, four)))
    )
    bc = gl.arange(
        0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, four)))
    )
    pick = br[:, None, None, None] + gl.full((blocks, 8, 1, 8), 0, gl.int32, four)
    diagonal = gl.gather(matrix, pick, 2).reshape((blocks, 8, 8))
    diagonal = gl.convert_layout(diagonal, three)
    r = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(2, three)))
    c = gl.arange(0, 8, layout=gl.SliceLayout(0, gl.SliceLayout(1, three)))
    inverse = diagonal + gl.where(r[None, :, None] == c[None, None, :], 1.0, 0.0)
    for pivot in gl.static_range(1, 7):
        index = gl.full((blocks, 8, 8), pivot, gl.int32, three)
        column = gl.gather(diagonal, index, 2)
        row = gl.gather(inverse, index, 1)
        update = column * row
        inverse += gl.where(
            (r[None, :, None] > pivot) & (c[None, None, :] < pivot), update, 0.0
        )
    # Join each neighboring pair while each 8x8 product stays wave-local.
    link_pick = (br ^ 1)[:, None, None, None] + gl.full(
        (blocks, 8, 1, 8), 0, gl.int32, four
    )
    link = gl.convert_layout(
        gl.gather(matrix, link_pick, 2).reshape((blocks, 8, 8)), three
    )
    block_id = gl.arange(0, blocks, layout=gl.SliceLayout(1, gl.SliceLayout(2, three)))
    partner_pick = (block_id ^ 1)[:, None, None] + gl.full(
        (blocks, 8, 8), 0, gl.int32, three
    )
    partner = gl.gather(inverse, partner_pick, 0)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[4, 1, 1]
    )
    lhs: gl.constexpr = gl.DotOperandLayout(0, mma, 1)
    rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 1)
    zero = gl.zeros((blocks, 8, 8), gl.float32, mma)
    joined = gl.amd.cdna3.mfma(
        gl.convert_layout(inverse, lhs), gl.convert_layout(link, rhs), zero
    )
    joined = gl.amd.cdna3.mfma(
        gl.convert_layout(joined, lhs), gl.convert_layout(partner, rhs), zero
    )
    joined = gl.convert_layout(joined, gl.SliceLayout(2, four))
    inverse = gl.convert_layout(inverse, gl.SliceLayout(2, four))
    result = gl.where(
        br[:, None, None, None] % 2 == bc[None, None, :, None],
        inverse[:, :, None, :],
        0.0,
    )
    result += gl.where(
        (br[:, None, None, None] % 2 == 1) & (bc[None, None, :, None] == 0),
        joined[:, :, None, :],
        0.0,
    )
    if NATIVE:
        factor_mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[1, 1, 4]
        )
        compact: gl.constexpr = gl.DotOperandLayout(0, factor_mma, 1)
    else:
        compact: gl.constexpr = gl.BlockedLayout(
            [1, 1, 1], [1, 8, 8], [1, 2, 2], [2, 1, 0]
        )
    return gl.convert_layout(result.reshape((2, 16, 16)), compact)


@gluon.jit
def _solve_two_halves(left, right, link, rhs):
    """Apply a block-triangular solve without building the off-diagonal inverse."""
    top_rhs = gl.amd.slice(rhs, [16, 128], [0, 0])
    bottom_rhs = gl.amd.slice(rhs, [16, 128], [16, 0])
    top = _dot_f32(left, top_rhs, WARPS=(1, 4))
    bottom_rhs = _dot_f32(link, top, bottom_rhs, (1, 4))
    bottom = _dot_f32(right, bottom_rhs, WARPS=(1, 4))
    return top, bottom


_chunk_transform_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_chunk_transform",
    ["M", "BT", "TIME_MAJOR", "BATCH", "FUSED_GATES", "HEAD_MAJOR"],
)


@gluon.jit(repr=_chunk_transform_repr)
def _chunk_transform(
    QKV,
    Gates,
    U,
    W,
    Scores,
    Coeff,
    M: gl.constexpr,
    BT: gl.constexpr,
    TIME_MAJOR: gl.constexpr = False,
    BA=None,
    Starts=None,
    ALog=None,
    DTBias=None,
    BATCH: gl.constexpr = 1,
    FUSED_GATES: gl.constexpr = False,
    TailKey=None,
    Bounds=None,
    HEAD_MAJOR: gl.constexpr = False,
):
    chunk = gl.program_id(1) if HEAD_MAJOR else gl.program_id(0)
    head = gl.program_id(0) if HEAD_MAJOR else gl.program_id(1)
    if Bounds is None:
        active = True
    else:
        first = gl.load(Bounds + chunk * 2)
        end = gl.load(Bounds + chunk * 2 + 1)
        active = first < end
    if active:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
        i = gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
        j = gl.arange(0, BT, layout=gl.SliceLayout(0, layout))
        k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        t = chunk * BT + i
        key = gl.load(QKV + ((4 + head // 2) * M + t[:, None]) * 128 + k[None, :])
        if FUSED_GATES:
            if Bounds is None:
                seq, first, end = _chunk_bounds(Starts, chunk, BATCH, BT)
            token = first + i
            ba_base = token * 16 + (head // 2) * 4 + head % 2
            av = gl.load(BA + ba_base + 2, token < end, 0).to(gl.float32)
            bv = gl.load(BA + ba_base, token < end, 0).to(gl.float32)
            av += gl.load(DTBias + head).to(gl.float32)
            decay_weight = gl.exp(gl.load(ALog + head).to(gl.float32))
            decay, beta = _gate_values(av, bv, decay_weight, token < end)
        else:
            decay = gl.load(Gates + (2 * head) * M + t)
            beta = gl.load(Gates + (2 * head + 1) * M + t)
        scan_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 2], [1, 4], [0, 1])
        prefix = gl.associative_scan(
            gl.convert_layout(decay, gl.BlockedLayout([1], [64], [4], [0])),
            0,
            _multiply,
        )
        prefix = gl.convert_layout(prefix, gl.SliceLayout(1, layout))
        pair_decay = gl.associative_scan(
            gl.convert_layout(
                gl.where(i[:, None] > j[None, :], decay[:, None], 1.0), scan_layout
            ),
            0,
            _multiply,
        )
        pair_decay = gl.convert_layout(pair_decay, layout)
        tail = gl.sum(gl.where(i[:, None] == BT - 1, pair_decay, 0.0), 0)
        tail = gl.convert_layout(tail, gl.SliceLayout(1, layout))
        gram = gl.convert_layout(_dot_bf16(key, key.trans()), layout)
        lower = gl.where(
            i[:, None] > j[None, :], -beta[:, None] * pair_decay * gram, 0.0
        )
        inverse = _invert_diagonal_sixteen(lower, FUSED_GATES)
        factor_mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[1, 4]
        )
        if FUSED_GATES:
            block_layout: gl.constexpr = gl.DotOperandLayout(0, factor_mma, 1)
        else:
            block_layout: gl.constexpr = gl.BlockedLayout(
                [1, 1], [8, 8], [2, 2], [1, 0]
            )
        left = gl.amd.slice(inverse, [1, 16, 16], [0, 0, 0]).reshape((16, 16))
        right = gl.amd.slice(inverse, [1, 16, 16], [1, 0, 0]).reshape((16, 16))
        left = gl.convert_layout(left, block_layout)
        right = gl.convert_layout(right, block_layout)
        link = gl.amd.slice(gl.convert_layout(lower, block_layout), [16, 16], [16, 0])
        base = (chunk * 8 + head) * BT
        value = gl.load(QKV + ((8 + head) * M + t[:, None]) * 128 + k[None, :]).to(
            gl.float32
        )
        fi = gl.arange(0, 16, layout=gl.SliceLayout(1, factor_mma))
        fk = gl.arange(0, 128, layout=gl.SliceLayout(0, factor_mma))
        if TIME_MAJOR:
            factor_offset = base * 128 + fi[:, None] * 128 + fk[None, :]
            second_offset: gl.constexpr = 16 * 128
        else:
            factor_offset = base * 128 + fk[None, :] * BT + fi[:, None]
            second_offset: gl.constexpr = 16
        u0, u1 = _solve_two_halves(left, right, link, beta[:, None] * value)
        gl.store(U + factor_offset, u0)
        gl.store(U + factor_offset + second_offset, u1)
        w0, w1 = _solve_two_halves(
            left, right, link, (beta * prefix)[:, None] * key.to(gl.float32)
        )
        # Feature-major consumers share a pre-negated factor. Time-major
        # consumers fold the negation into their operand preparation instead.
        gl.store(W + factor_offset, w0 if TIME_MAJOR else -w0)
        gl.store(W + factor_offset + second_offset, w1 if TIME_MAJOR else -w1)
        query = gl.load(QKV + ((head // 2) * M + t[:, None]) * 128 + k[None, :])
        score = gl.convert_layout(_dot_bf16(query, key.trans()), layout)
        score = gl.where(i[:, None] >= j[None, :], score * pair_decay, 0.0)
        gl.store(Scores + (base + i[:, None]) * BT + j[None, :], score)
        gl.store(Coeff + (chunk * 8 + head) * 2 * BT + i, prefix)
        gl.store(Coeff + (chunk * 8 + head) * 2 * BT + BT + i, tail)
        if TailKey is not None:
            gl.store(
                TailKey + (base + i[:, None]) * 128 + k[None, :],
                key.to(gl.float32) * tail[:, None],
            )


@gluon.jit
def _load_recurrent_factors(
    U,
    W,
    base,
    vblock,
    BT: gl.constexpr,
    BV: gl.constexpr,
    WARPS: gl.constexpr,
    TIME_MAJOR: gl.constexpr = False,
    TRANSPOSED: gl.constexpr = True,
):
    """Load U.T and -W.T in the recurrent consumer's operand ownership."""
    if TIME_MAJOR:
        layout: gl.constexpr = gl.BlockedLayout(
            [1, 2], [4, 16], [WARPS[0] * WARPS[1], 1], [1, 0]
        )
        ti = gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
        vr = vblock * BV + gl.arange(0, BV, layout=gl.SliceLayout(0, layout))
        kr = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        u = gl.amd.cdna3.buffer_load(
            U, (base + ti[:, None]) * 128 + vr[None, :]
        ).trans()
        w = -gl.amd.cdna3.buffer_load(
            W, (base + ti[:, None]) * 128 + kr[None, :]
        ).trans()
    else:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 4],
            transposed=TRANSPOSED,
            warps_per_cta=WARPS,
        )
        rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 1)
        vr = vblock * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, mma))
        ti = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        u = gl.amd.cdna3.buffer_load(U, base * 128 + vr[:, None] * BT + ti[None, :])
        kr = gl.arange(0, 128, layout=gl.SliceLayout(1, rhs))
        tj = gl.arange(0, BT, layout=gl.SliceLayout(0, rhs))
        w = gl.amd.cdna3.buffer_load(W, base * 128 + kr[:, None] * BT + tj[None, :])
    return u, w


_chunk_state_rows_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_chunk_state_rows", ["M", "BT", "BV", "NW", "WM"]
)


@gluon.jit(repr=_chunk_state_rows_repr)
def _chunk_state_rows(
    QKV,
    U,
    W,
    Coeff,
    State,
    Indices,
    Starts,
    ChunkState,
    M: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    WM: gl.constexpr,
    TailKey=None,
):
    seq = gl.program_id(0)
    head = gl.program_id(1)
    vblock = gl.program_id(2)
    begin = gl.load(Starts + seq)
    end = gl.load(Starts + seq + 1)
    slot = gl.load(Indices + seq)
    first_chunk = begin // BT + seq
    count = gl.cdiv(end - begin, BT)
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 4],
        transposed=True,
        warps_per_cta=[WM, NW // WM],
    )
    vr = vblock * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, mma))
    kc = gl.arange(0, 128, layout=gl.SliceLayout(0, mma))
    state_offset = ((slot * 8 + head) * 128 + vr[:, None]) * 128 + kc[None, :]
    h = gl.load(State + state_offset)
    for local in range(count):
        chunk = first_chunk + local
        state_base = (chunk * 8 + head) * 128 * 128
        gl.store(ChunkState + state_base + vr[:, None] * 128 + kc[None, :], h)
        h = _advance_chunk(
            h,
            QKV,
            U,
            W,
            Coeff,
            chunk,
            head,
            vblock,
            False,
            M,
            BT,
            BV,
            NW,
            WM,
            TailKey,
            SAVE_DELTA=True,
        )
    gl.store(State + state_offset, h)


@gluon.jit
def _load_key_decay(QKV, Coeff, chunk, head, i, k, M: gl.constexpr, BT: gl.constexpr):
    """Load keys, suffix weights, and the whole-chunk state decay."""
    key = gl.amd.cdna3.buffer_load(
        QKV, ((4 + head // 2) * M + chunk * BT + i[:, None]) * 128 + k[None, :]
    ).to(gl.float32)
    tail = gl.load(Coeff + (chunk * 8 + head) * 2 * BT + BT + i)
    decay = gl.load(Coeff + (chunk * 8 + head) * 2 * BT + BT - 1)
    return key, tail, decay


@gluon.jit
def _load_tail_key(
    TailKey,
    Coeff,
    chunk,
    head,
    BT: gl.constexpr,
    WARPS: gl.constexpr,
    TRANSPOSED: gl.constexpr = True,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 4], transposed=TRANSPOSED, warps_per_cta=WARPS
    )
    rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 1)
    ti = gl.arange(0, BT, layout=gl.SliceLayout(1, rhs))
    kc = gl.arange(0, 128, layout=gl.SliceLayout(0, rhs))
    base = (chunk * 8 + head) * BT
    key = gl.amd.cdna3.buffer_load(TailKey, (base + ti[:, None]) * 128 + kc[None, :])
    decay = gl.load(Coeff + (chunk * 8 + head) * 2 * BT + BT - 1)
    return key, decay


@gluon.jit
def _advance_chunk(
    h,
    QKV,
    U,
    W,
    Coeff,
    chunk,
    head,
    vblock,
    homogeneous,
    M: gl.constexpr,
    BT: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    WM: gl.constexpr,
    TailKey=None,
    SAVE_DELTA: gl.constexpr = False,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [NW, 1], [1, 0])
    i = gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    base = (chunk * 8 + head) * BT
    u, w = _load_recurrent_factors(U, W, base, vblock, BT, BV, (WM, NW // WM))
    u = gl.where(homogeneous, 0.0, u)
    delta = _dot_f32(h, w, u, (WM, NW // WM))
    if SAVE_DELTA:
        vr = vblock * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, delta.type.layout))
        ti = gl.arange(0, BT, layout=gl.SliceLayout(0, delta.type.layout))
        offset = base * 128 + vr[:, None] * BT + ti[None, :]
        gl.store(U + offset, delta)
    if TailKey is None:
        key, tail, decay = _load_key_decay(QKV, Coeff, chunk, head, i, k, M, BT)
        return _dot_f32(delta, key * tail[:, None], h * decay, (WM, NW // WM))
    else:
        key, decay = _load_tail_key(TailKey, Coeff, chunk, head, BT, (WM, NW // WM))
        return _dot_f32(delta, key, h * decay, (WM, NW // WM))


@gluon.jit
def _recurrence_ids(UNIT_MAJOR: gl.constexpr):
    if UNIT_MAJOR:
        return gl.program_id(2), gl.program_id(0), gl.program_id(1)
    else:
        return gl.program_id(0), gl.program_id(1), gl.program_id(2)


_build_segments_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_build_segments",
    ["M", "BATCH", "BT", "SEG", "BV", "NW", "WM"],
)


@gluon.jit(repr=_build_segments_repr)
def _build_segments(
    QKV,
    U,
    W,
    Coeff,
    Starts,
    Affine,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
    SEG: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    WM: gl.constexpr,
    TailKey=None,
):
    segment, head, plane_block = _recurrence_ids(True)
    plane = plane_block // (128 // BV)
    vblock = plane_block % (128 // BV)
    seq, first, seq_end = _chunk_bounds(Starts, segment, BATCH, SEG)
    if first + SEG < seq_end:
        begin = gl.load(Starts + seq)
        first_chunk = begin // BT + seq + (first - begin) // BT
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 4],
            transposed=True,
            warps_per_cta=[WM, NW // WM],
        )
        vr = vblock * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, mma))
        kc = gl.arange(0, 128, layout=gl.SliceLayout(0, mma))
        ti = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        base = (first_chunk * 8 + head) * BT
        # The homogeneous plane uses the already signed W factor.
        factors = gl.where(plane == 0, W, U)
        delta = gl.load(factors + base * 128 + vr[:, None] * BT + ti[None, :])
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [NW, 1], [1, 0])
        i = gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
        k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        if TailKey is None:
            key, tail, decay = _load_key_decay(
                QKV, Coeff, first_chunk, head, i, k, M, BT
            )
            h = gl.where((plane == 0) & (vr[:, None] == kc[None, :]), decay, 0.0)
            h = _dot_f32(delta, key * tail[:, None], h, (WM, NW // WM))
        else:
            key, decay = _load_tail_key(
                TailKey, Coeff, first_chunk, head, BT, (WM, NW // WM)
            )
            h = gl.where((plane == 0) & (vr[:, None] == kc[None, :]), decay, 0.0)
            h = _dot_f32(delta, key, h, (WM, NW // WM))
        for local in range(1, SEG // BT):
            h = _advance_chunk(
                h,
                QKV,
                U,
                W,
                Coeff,
                first_chunk + local,
                head,
                vblock,
                plane == 0,
                M,
                BT,
                BV,
                NW,
                WM,
                TailKey,
            )
        base = ((segment * 8 + head) * 2 + plane) * 128 * 128
        gl.store(Affine + base + vr[:, None] * 128 + kc[None, :], h)


_build_segments_reverse_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_build_segments_reverse",
    ["M", "BATCH", "BT", "SEG", "BC", "WM", "MMA_SIZE"],
)


@gluon.jit(repr=_build_segments_reverse_repr)
def _build_segments_reverse(
    QKV,
    U,
    W,
    Coeff,
    Starts,
    Affine,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
    SEG: gl.constexpr,
    BC: gl.constexpr,
    WM: gl.constexpr,
    MMA_SIZE: gl.constexpr = 16,
):
    """Compose from the right, sharing K @ T between transform and bias."""
    segment, head, cblock = _recurrence_ids(True)
    seq, first, seq_end = _chunk_bounds(Starts, segment, BATCH, SEG)
    if first + SEG < seq_end:
        begin = gl.load(Starts + seq)
        first_chunk = begin // BT + seq + (first - begin) // BT
        last_chunk = first_chunk + SEG // BT - 1
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[MMA_SIZE, MMA_SIZE, 64 // MMA_SIZE],
            transposed=True,
            warps_per_cta=[WM, 4 // WM],
        )
        r = gl.arange(0, 128, layout=gl.SliceLayout(1, mma))
        c = cblock * BC + gl.arange(0, BC, layout=gl.SliceLayout(0, mma))
        factors: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
        fr = gl.arange(0, 128, layout=gl.SliceLayout(1, factors))
        ft = gl.arange(0, BT, layout=gl.SliceLayout(0, factors))
        keys: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
        ti = gl.arange(0, BT, layout=gl.SliceLayout(1, keys))
        kc = cblock * BC + gl.arange(0, BC, layout=gl.SliceLayout(0, keys))
        k = gl.arange(0, 128, layout=gl.SliceLayout(0, keys))
        base = (last_chunk * 8 + head) * BT
        u = gl.amd.cdna3.buffer_load(U, base * 128 + fr[:, None] * BT + ft[None, :])
        w = gl.amd.cdna3.buffer_load(W, base * 128 + fr[:, None] * BT + ft[None, :])
        decay = gl.load(Coeff + (last_chunk * 8 + head) * 2 * BT + BT - 1)
        stripe_key = gl.amd.cdna3.buffer_load(
            QKV,
            ((4 + head // 2) * M + last_chunk * BT + ti[:, None]) * 128 + kc[None, :],
        ).to(gl.float32)
        tail = gl.load(Coeff + (last_chunk * 8 + head) * 2 * BT + BT + ti)
        stripe_key = stripe_key * tail[:, None]
        identity = gl.where(r[:, None] == c[None, :], decay, 0.0)
        transform = _dot_f32(w, stripe_key, identity, (WM, 4 // WM), MMA_SIZE)
        bias = _dot_f32(u, stripe_key, WARPS=(WM, 4 // WM), MMA_SIZE=MMA_SIZE)
        for local in range(SEG // BT - 2, -1, -1):
            chunk = first_chunk + local
            base = (chunk * 8 + head) * BT
            key, tail, decay = _load_key_decay(QKV, Coeff, chunk, head, ti, k, M, BT)
            key = key * tail[:, None]
            # Both loads are independent of the shared key projection below.
            u = gl.amd.cdna3.buffer_load(U, base * 128 + fr[:, None] * BT + ft[None, :])
            w = gl.amd.cdna3.buffer_load(W, base * 128 + fr[:, None] * BT + ft[None, :])
            projected_key = _dot_f32(key, transform, WARPS=(2, 2))
            transform = _dot_f32(
                w, projected_key, transform * decay, (WM, 4 // WM), MMA_SIZE
            )
            bias = _dot_f32(u, projected_key, bias, (WM, 4 // WM), MMA_SIZE)
        output = (segment * 8 + head) * 2 * 128 * 128 + r[:, None] * 128 + c[None, :]
        gl.store(Affine + output, transform)
        gl.store(Affine + output + 128 * 128, bias)


_build_segments_stacked_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_build_segments_stacked",
    ["M", "BATCH", "BT", "SEG", "BC", "WM", "MMA_SIZE"],
)


@gluon.jit(repr=_build_segments_stacked_repr)
def _build_segments_stacked(
    QKV,
    U,
    W,
    Coeff,
    Starts,
    Affine,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
    SEG: gl.constexpr,
    BC: gl.constexpr,
    WM: gl.constexpr,
    MMA_SIZE: gl.constexpr = 16,
):
    """Stack two 128-row products for the wide, well-populated summary grids."""
    gl.static_assert(BC == 64)
    segment, head, cblock = _recurrence_ids(True)
    seq, first, seq_end = _chunk_bounds(Starts, segment, BATCH, SEG)
    if first + SEG < seq_end:
        begin = gl.load(Starts + seq)
        first_chunk = begin // BT + seq + (first - begin) // BT
        last_chunk = first_chunk + SEG // BT - 1
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[MMA_SIZE, MMA_SIZE, 64 // MMA_SIZE],
            transposed=True,
            warps_per_cta=[WM, 4 // WM],
        )
        r = gl.arange(0, 256, layout=gl.SliceLayout(1, mma))
        c = cblock * BC + gl.arange(0, BC, layout=gl.SliceLayout(0, mma))
        factors: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [4, 1], [1, 0])
        fr = gl.arange(0, 128, layout=gl.SliceLayout(1, factors))
        ft = gl.arange(0, BT, layout=gl.SliceLayout(0, factors))
        keys: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
        ti = gl.arange(0, BT, layout=gl.SliceLayout(1, keys))
        kc = cblock * BC + gl.arange(0, BC, layout=gl.SliceLayout(0, keys))
        k = gl.arange(0, 128, layout=gl.SliceLayout(0, keys))
        base = (last_chunk * 8 + head) * BT
        u = gl.amd.cdna3.buffer_load(U, base * 128 + fr[:, None] * BT + ft[None, :])
        w = gl.amd.cdna3.buffer_load(W, base * 128 + fr[:, None] * BT + ft[None, :])
        decay = gl.load(Coeff + (last_chunk * 8 + head) * 2 * BT + BT - 1)
        stripe_key = gl.amd.cdna3.buffer_load(
            QKV,
            ((4 + head // 2) * M + last_chunk * BT + ti[:, None]) * 128 + kc[None, :],
        ).to(gl.float32)
        tail = gl.load(Coeff + (last_chunk * 8 + head) * 2 * BT + BT + ti)
        stripe_key = stripe_key * tail[:, None]
        identity = gl.where(r[:, None] == c[None, :], decay, 0.0)
        # The upper half is the transform; the lower half is the affine bias.
        stacked = gl.join(w, u).permute((2, 0, 1)).reshape((256, BT))
        affine = _dot_f32(stacked, stripe_key, identity, (WM, 4 // WM), MMA_SIZE)
        for local in range(SEG // BT - 2, -1, -1):
            chunk = first_chunk + local
            base = (chunk * 8 + head) * BT
            key, tail, decay = _load_key_decay(QKV, Coeff, chunk, head, ti, k, M, BT)
            key = key * tail[:, None]
            # Both loads are independent of the shared key projection below.
            u = gl.amd.cdna3.buffer_load(U, base * 128 + fr[:, None] * BT + ft[None, :])
            w = gl.amd.cdna3.buffer_load(W, base * 128 + fr[:, None] * BT + ft[None, :])
            transform = gl.amd.slice(affine, [128, BC], [0, 0])
            stacked = gl.join(w, u).permute((2, 0, 1)).reshape((256, BT))
            projected_key = _dot_f32(key, transform, WARPS=(2, 2))
            initial = gl.where(r[:, None] < 128, affine * decay, affine)
            affine = _dot_f32(stacked, projected_key, initial, (WM, 4 // WM), MMA_SIZE)
        output = (segment * 8 + head) * 2 * 128 * 128 + r[:, None] * 128 + c[None, :]
        gl.store(Affine + output, affine)


_prefix_segments_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_prefix_segments", ["SEG", "BV"]
)


@gluon.jit(repr=_prefix_segments_repr)
def _prefix_segments(
    Affine, State, Indices, Starts, SEG: gl.constexpr, BV: gl.constexpr
):
    """Propagate entry states with complete, unpadded narrow MFMA row tiles."""
    seq, head, vblock = _recurrence_ids(True)
    begin = gl.load(Starts + seq)
    end = gl.load(Starts + seq + 1)
    slot = gl.load(Indices + seq)
    first_segment = begin // SEG + seq
    count = gl.cdiv(end - begin, SEG)
    # Both narrow layouts cover all 128 output columns with two column waves.
    if BV == 4:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[4, 64, 16], transposed=False, warps_per_cta=[1, 2]
        )
    elif BV == 8:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[4, 64, 16], transposed=False, warps_per_cta=[2, 2]
        )
    else:
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4, instr_shape=[16, 16, 4], transposed=True, warps_per_cta=[1, 4]
        )
    vr = vblock * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, mma))
    kc = gl.arange(0, 128, layout=gl.SliceLayout(0, mma))
    rhs: gl.constexpr = gl.DotOperandLayout(1, mma, 1)
    kr = gl.arange(0, 128, layout=gl.SliceLayout(1, rhs))
    kn = gl.arange(0, 128, layout=gl.SliceLayout(0, rhs))
    h = gl.load(State + ((slot * 8 + head) * 128 + vr[:, None]) * 128 + kc[None, :])
    for local in range(count):
        segment = first_segment + local
        matrix_base = (segment * 8 + head) * 2 * 128 * 128
        state_offset = matrix_base + 128 * 128 + vr[:, None] * 128 + kc[None, :]
        if local + 1 < count:
            transform = gl.load(Affine + matrix_base + kr[:, None] * 128 + kn[None, :])
            bias = gl.load(Affine + state_offset)
            gl.store(Affine + state_offset, h)
            if BV <= 8:
                h = gl.amd.cdna3.mfma(
                    gl.convert_layout(h, gl.DotOperandLayout(0, mma, 1)),
                    transform,
                    bias,
                )
            else:
                h = _dot_f32(h, transform, bias, (1, 4))
        else:
            gl.store(Affine + state_offset, h)


@gluon.jit
def _gated_rms(core, gate, weight, eps):
    """The epilogue's BF16 materialization boundary, shared by both readouts."""
    inverse_rms = gl.rsqrt(gl.sum(core * core, 1) / 128 + eps)
    sigmoid = 1.0 / (1.0 + gl.exp(-gate))
    return (core * inverse_rms[:, None] * weight[None, :] * gate * sigmoid).to(
        gl.bfloat16
    )


@gluon.jit
def _quantize_group(normalized, maximum, inverse_maximum):
    """Scale each 128-value group using rounded BF16, then encode FP8."""
    f = normalized.to(gl.float32)
    group_scale = gl.maximum(gl.max(gl.abs(f), 1), 1.0e-10) * inverse_maximum
    encoded = gl.clamp(f * (1.0 / group_scale[:, None]), -maximum, maximum)
    return encoded, group_scale


_chunk_output_quant_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_chunk_output_quant", ["M", "BATCH", "BT"]
)


@gluon.jit(repr=_chunk_output_quant_repr)
def _chunk_output_quant(
    QKV,
    Updates,
    Scores,
    Coeff,
    Starts,
    ChunkState,
    Projected,
    Weight,
    Normalized,
    Quantized,
    Scales,
    scale,
    eps,
    maximum,
    inverse_maximum,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
):
    chunk = gl.program_id(0)
    head = gl.program_id(1)
    seq, first, end = _chunk_bounds(Starts, chunk, BATCH, BT)
    if first < end:
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [4, 1], [1, 0])
        state_layout: gl.constexpr = gl.BlockedLayout([2, 1], [64, 1], [1, 4], [0, 1])
        r = gl.arange(0, 128, layout=gl.SliceLayout(1, state_layout))
        c = gl.arange(0, 128, layout=gl.SliceLayout(0, state_layout))
        state_base = (chunk * 8 + head) * 128 * 128
        i = gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
        j = gl.arange(0, BT, layout=gl.SliceLayout(0, layout))
        k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        base = (chunk * 8 + head) * BT
        fk = gl.arange(0, 128, layout=gl.SliceLayout(1, layout))
        ft = gl.arange(0, BT, layout=gl.SliceLayout(0, layout))
        factor_offset = base * 128 + fk[:, None] * BT + ft[None, :]
        updates = gl.load(Updates + factor_offset).trans()
        query = gl.amd.cdna3.buffer_load(
            QKV, ((head // 2) * M + chunk * BT + i[:, None]) * 128 + k[None, :]
        ).to(gl.float32)
        prefix = gl.load(Coeff + (chunk * 8 + head) * 2 * BT + i)
        scores = gl.amd.cdna3.buffer_load(Scores, (base + i[:, None]) * BT + j[None, :])
        weighted_query = query * prefix[:, None]
        local_output = _dot_f32(scores, updates)
        h = gl.load(ChunkState + state_base + c[None, :] * 128 + r[:, None])
        out = _dot_f32(weighted_query, h, local_output) * scale
        core = gl.convert_layout(out, layout).to(gl.bfloat16).to(gl.float32)
        token = first + i
        gate = gl.load(
            Projected
            + token[:, None] * 3072
            + (head // 2) * 768
            + 512
            + (head % 2) * 128
            + k[None, :],
            token[:, None] < end,
            0,
        ).to(gl.float32)
        weight = gl.load(Weight + k).to(gl.float32)
        normalized = _gated_rms(core, gate, weight, eps)
        offset = (token[:, None] * 8 + head) * 128 + k[None, :]
        gl.store(Normalized + offset, normalized, token[:, None] < end)
        encoded, group_scale = _quantize_group(normalized, maximum, inverse_maximum)
        gl.store(Quantized + offset, encoded, token[:, None] < end)
        gl.store(Scales + token * 8 + head, group_scale, token < end)


_state_and_core_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_state_and_core",
    [
        "M",
        "BATCH",
        "BT",
        "SEG",
        "BV",
        "NW",
        "WM",
        "TIME_MAJOR",
        "UNIT_MAJOR",
        "TRANSPOSED",
    ],
)


@gluon.jit(repr=_state_and_core_repr)
def _state_and_core(
    QKV,
    U,
    W,
    Scores,
    Coeff,
    State,
    Indices,
    Starts,
    SegmentState,
    Core,
    scale,
    M: gl.constexpr,
    BATCH: gl.constexpr,
    BT: gl.constexpr,
    SEG: gl.constexpr,
    BV: gl.constexpr,
    NW: gl.constexpr,
    WM: gl.constexpr,
    TIME_MAJOR: gl.constexpr = False,
    UNIT_MAJOR: gl.constexpr = False,
    TailKey=None,
    TRANSPOSED: gl.constexpr = True,
):
    unit, head, vblock = _recurrence_ids(UNIT_MAJOR)
    if SEG == 0:
        seq = unit
        begin = gl.load(Starts + seq)
        first = begin
        seq_end = gl.load(Starts + seq + 1)
        end = seq_end
    else:
        seq, first, seq_end = _chunk_bounds(Starts, unit, BATCH, SEG)
        begin = gl.load(Starts + seq)
        end = gl.minimum(seq_end, first + SEG)
    if first < seq_end:
        slot = gl.load(Indices + seq)
        first_chunk = begin // BT + seq + (first - begin) // BT
        count = gl.cdiv(end - first, BT)
        mma: gl.constexpr = gl.amd.AMDMFMALayout(
            version=4,
            instr_shape=[16, 16, 4],
            transposed=TRANSPOSED,
            warps_per_cta=[WM, NW // WM],
        )
        layout: gl.constexpr = gl.BlockedLayout([1, 2], [4, 16], [NW, 1], [1, 0])
        vr = vblock * BV + gl.arange(0, BV, layout=gl.SliceLayout(1, mma))
        kc = gl.arange(0, 128, layout=gl.SliceLayout(0, mma))
        if SEG == 0:
            h = gl.load(
                State + ((slot * 8 + head) * 128 + vr[:, None]) * 128 + kc[None, :]
            )
        else:
            h = gl.load(
                SegmentState
                + (((unit * 8 + head) * 2 + 1) * 128 + vr[:, None]) * 128
                + kc[None, :]
            )
        i = gl.arange(0, BT, layout=gl.SliceLayout(1, layout))
        j = gl.arange(0, BT, layout=gl.SliceLayout(0, layout))
        k = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
        output_t = gl.arange(0, BT, layout=gl.SliceLayout(0, mma))
        for local in range(count):
            chunk = first_chunk + local
            base = (chunk * 8 + head) * BT
            # Key traffic is independent of the solve and can arrive early.
            if TailKey is None:
                key, tail, decay = _load_key_decay(QKV, Coeff, chunk, head, i, k, M, BT)
            else:
                key, decay = _load_tail_key(
                    TailKey, Coeff, chunk, head, BT, (WM, NW // WM), TRANSPOSED
                )
            # These narrow, preweighted-key grids benefit from overlapping the
            # independent readout loads with the recurrent factor projection.
            if TailKey is not None:
                query = gl.amd.cdna3.buffer_load(
                    QKV, ((head // 2) * M + chunk * BT + i[:, None]) * 128 + k[None, :]
                ).to(gl.float32)
                prefix = gl.load(Coeff + (chunk * 8 + head) * 2 * BT + i)
                scores = gl.amd.cdna3.buffer_load(
                    Scores, (base + i[:, None]) * BT + j[None, :]
                )
            u, w = _load_recurrent_factors(
                U, W, base, vblock, BT, BV, (WM, NW // WM), TIME_MAJOR, TRANSPOSED
            )
            delta = _dot_f32(h, w, u, (WM, NW // WM), TRANSPOSED=TRANSPOSED)
            if TailKey is None:
                query = gl.amd.cdna3.buffer_load(
                    QKV, ((head // 2) * M + chunk * BT + i[:, None]) * 128 + k[None, :]
                ).to(gl.float32)
                prefix = gl.load(Coeff + (chunk * 8 + head) * 2 * BT + i)
                scores = gl.amd.cdna3.buffer_load(
                    Scores, (base + i[:, None]) * BT + j[None, :]
                )
            local_output = _dot_f32(
                delta, scores.trans(), WARPS=(WM, NW // WM), TRANSPOSED=TRANSPOSED
            )
            y = (
                _dot_f32(
                    h,
                    (query * prefix[:, None]).trans(),
                    local_output,
                    (WM, NW // WM),
                    TRANSPOSED=TRANSPOSED,
                )
                * scale
            )
            output_token = first + local * BT + output_t
            gl.store(
                Core + (output_token[None, :] * 8 + head) * 128 + vr[:, None],
                y.to(gl.bfloat16),
                output_token[None, :] < end,
            )
            if TailKey is None:
                h = _dot_f32(
                    delta,
                    key * tail[:, None],
                    h * decay,
                    (WM, NW // WM),
                    TRANSPOSED=TRANSPOSED,
                )
            else:
                h = _dot_f32(
                    delta, key, h * decay, (WM, NW // WM), TRANSPOSED=TRANSPOSED
                )
        if end == seq_end:
            gl.store(
                State + ((slot * 8 + head) * 128 + vr[:, None]) * 128 + kc[None, :], h
            )


_normalize_quantize_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b1_5_normalize_quantize", ["M", "ROWS", "NW"]
)


@gluon.jit(repr=_normalize_quantize_repr)
def _normalize_quantize(
    Core,
    Projected,
    Weight,
    Normalized,
    Quantized,
    Scales,
    eps,
    maximum,
    inverse_maximum,
    M: gl.constexpr,
    ROWS: gl.constexpr,
    NW: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
    row = gl.program_id(0) * ROWS + gl.arange(0, ROWS, layout=gl.SliceLayout(1, layout))
    c = gl.arange(0, 128, layout=gl.SliceLayout(0, layout))
    token = row // 8
    head = row % 8
    x = gl.load(Core + row[:, None] * 128 + c[None, :], row[:, None] < M * 8, 0).to(
        gl.float32
    )
    gate = gl.load(
        Projected
        + token[:, None] * 3072
        + (head[:, None] // 2) * 768
        + 512
        + (head[:, None] % 2) * 128
        + c[None, :],
        row[:, None] < M * 8,
        0,
    ).to(gl.float32)
    weight = gl.load(Weight + c).to(gl.float32)
    normalized = _gated_rms(x, gate, weight, eps)
    gl.store(
        Normalized + row[:, None] * 128 + c[None, :], normalized, row[:, None] < M * 8
    )
    encoded, group_scale = _quantize_group(normalized, maximum, inverse_maximum)
    gl.store(Quantized + row[:, None] * 128 + c[None, :], encoded, row[:, None] < M * 8)
    gl.store(Scales + row, group_scale, row < M * 8)


@dataclass(frozen=True)
class _Schedule:
    """Shape-only tuning; sequence boundaries and cache ownership stay on-device."""

    segment_tokens: int
    build_rows: int
    reverse_columns: int
    prefix_rows: int
    core_rows: int
    core_transposed: bool
    prep_warps: int
    prep_tokens: int
    norm_rows: int
    norm_warps: int
    fused_gates: bool
    fused_core: bool
    time_major: bool
    unit_major: bool
    preweight_key: bool
    prepare_bounds: bool


def _schedule(m: int, batch: int) -> _Schedule:
    wide = batch == 2 and m >= 16384
    if 2 < batch < 8 and m <= 4096:
        segment = 0
    elif batch <= 4 and m <= 4096:
        segment = 128
    elif m <= 2048 * batch:
        segment = 0
    elif batch == 1:
        if m >= 32768:
            segment = 1024
        elif m <= 12288:
            segment = ((m + 511) // 512) * 32
        else:
            segment = 512
    elif 3 <= batch <= 4 and m >= 24576:
        # Fewer prefix matrices, balanced by wider column-wise summaries.
        segment = 1024
    elif 3 <= batch <= 5 and 12288 <= m <= 24576:
        segment = 704
    elif wide or (batch > 4 and m > 4096 * batch):
        segment = 512
    else:
        segment = 256
    core_rows = (
        (64 if wide or m >= 32768 else 32) if segment else (32 if batch < 32 else 64)
    )
    fused_core = segment != 0 or batch >= 8
    build_rows = (
        128 if wide or (batch == 1 and m >= 32768) else (64 if m >= 16384 else 32)
    )
    reverse_columns = 0
    if segment and m // segment + batch >= 16:
        reverse_columns = (
            64 if build_rows == 128 or (3 <= batch <= 4 and m >= 24576) else 32
        )
    return _Schedule(
        segment_tokens=segment,
        build_rows=build_rows,
        reverse_columns=reverse_columns,
        prefix_rows=4 if batch == 1 else (8 if batch == 2 else 16),
        core_rows=core_rows,
        # Keep the established ownership of each recurrent path.
        core_transposed=(segment != 0 or batch >= 32)
        and not (batch == 1 and m <= 1024),
        prep_warps=4 if m <= 2048 else 1,
        prep_tokens=4 if m <= 12288 else 8,
        norm_rows=16 if m > 8192 else 32,
        norm_warps=1 if m > 8192 else 4,
        fused_gates=m <= 8192,
        fused_core=fused_core,
        time_major=batch >= 32,
        unit_major=segment != 0 and (m < 32768 or batch >= 3),
        preweight_key=(
            not fused_core or (batch == 1 and m <= 1024) or (batch == 8 and m <= 8192)
        ),
        prepare_bounds=batch >= 8 or (2 < batch < 8 and m <= 4096),
    )


def gdn_prefill_group_fp8_quant(
    projected_qkvz: torch.Tensor,
    projected_ba: torch.Tensor,
    conv_state: torch.Tensor,
    delta_state: torch.Tensor,
    cache_indices: torch.Tensor,
    cu_seqlens: torch.Tensor,
    has_initial_state: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor,
    a_log: torch.Tensor,
    dt_bias: torch.Tensor,
    norm_weight: torch.Tensor,
    *,
    scale: float,
    eps: float = 1.0e-6,
    precision_config: FP8PrecisionConfig = FP8_E4M3_FN,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    m = projected_qkvz.shape[0]
    batch = cache_indices.numel()
    device = projected_qkvz.device
    schedule = _schedule(m, batch)
    bt = 32
    # A single sequence begins at zero, so no inter-sequence reserve is needed.
    # Multi-sequence bounds remain entirely on-device and use the safe capacity.
    chunks = (m + bt - 1) // bt if batch == 1 else m // bt + batch
    # Separate direct, preweighted head planes by eight unused rows. Consumers
    # address only initialized chunk rows; the gap changes physical head pitch.
    qkv_gap = (
        8
        if schedule.fused_core
        and not schedule.segment_tokens
        and schedule.preweight_key
        else 0
    )
    qkv_rows = chunks * bt + qkv_gap
    qkv = torch.empty((16, qkv_rows, 128), device=device, dtype=torch.bfloat16)
    bounds = (
        torch.empty((chunks, 2), device=device, dtype=torch.int32)
        if schedule.prepare_bounds
        else None
    )
    fused_gates = schedule.fused_gates
    gates = (
        projected_ba
        if fused_gates
        else torch.empty((8, 2, qkv_rows), device=device, dtype=torch.float32)
    )
    normalized = torch.empty((m, 8, 128), device=device, dtype=torch.bfloat16)
    quantized = torch.empty((m, 1024), device=device, dtype=precision_config.dtype)
    scales = torch.empty((m, 8), device=device, dtype=torch.float32)
    prep_warps = schedule.prep_warps
    prep_tokens = schedule.prep_tokens
    _prepare_qkv_window[(chunks * 32 // (prep_tokens * prep_warps), 16)](
        projected_qkvz,
        conv_state,
        cache_indices,
        cu_seqlens,
        has_initial_state,
        conv_weight,
        conv_bias,
        qkv,
        qkv_rows,
        batch,
        prep_tokens,
        prep_warps,
        Bounds=bounds,
        num_warps=prep_warps,
        enable_fp_fusion=False,
    )
    if not fused_gates:
        _prepare_gates[(chunks,)](
            projected_ba,
            cu_seqlens,
            a_log,
            dt_bias,
            gates,
            qkv_rows,
            batch,
            bt,
            Bounds=bounds,
            num_warps=4,
            enable_fp_fusion=False,
        )
    seg = schedule.segment_tokens
    fused_core = schedule.fused_core
    time_major = schedule.time_major
    u = torch.empty((chunks, 8, 128, bt), device=device, dtype=torch.float32)
    w = torch.empty_like(u)
    scores = torch.empty((chunks, 8, bt, bt), device=device, dtype=torch.float32)
    coeff = torch.empty((chunks, 8, 2, bt), device=device, dtype=torch.float32)
    tail_key = (
        torch.empty((chunks, 8, bt, 128), device=device, dtype=torch.float32)
        if schedule.preweight_key
        else None
    )
    if fused_core:
        core = normalized
    else:
        chunk_state = torch.empty(
            (chunks, 8, 128, 128), device=device, dtype=torch.float32
        )
    # Adjacent heads share the tiny-input factorization traversal.
    head_major = m <= 1024 and batch == 1
    factor_grid = (8, chunks) if head_major else (chunks, 8)
    _chunk_transform[factor_grid](
        qkv,
        gates,
        u,
        w,
        scores,
        coeff,
        qkv_rows,
        bt,
        num_warps=4,
        enable_fp_fusion=False,
        TIME_MAJOR=time_major,
        BA=projected_ba,
        Starts=cu_seqlens,
        ALog=a_log,
        DTBias=dt_bias,
        BATCH=batch,
        FUSED_GATES=fused_gates,
        TailKey=tail_key,
        Bounds=bounds,
        HEAD_MAJOR=head_major,
    )
    if seg:
        segments = (m + seg - 1) // seg if batch == 1 else m // seg + batch
        affine = torch.empty(
            (segments, 8, 2, 128, 128), device=device, dtype=torch.float32
        )
        segment_state = affine
        build_bv = schedule.build_rows
        build_wm = 4 if build_bv == 128 else 2
        if schedule.reverse_columns:
            columns = schedule.reverse_columns
            build_kernel = (
                _build_segments_stacked if columns == 64 else _build_segments_reverse
            )
            build_kernel[(8, 128 // columns, segments)](
                qkv,
                u,
                w,
                coeff,
                cu_seqlens,
                affine,
                qkv_rows,
                batch,
                bt,
                seg,
                columns,
                4,
                MMA_SIZE=32 if columns == 32 else 16,
                num_warps=4,
                enable_fp_fusion=False,
            )
        else:
            _build_segments[(8, 2 * 128 // build_bv, segments)](
                qkv,
                u,
                w,
                coeff,
                cu_seqlens,
                affine,
                qkv_rows,
                batch,
                bt,
                seg,
                build_bv,
                4,
                build_wm,
                TailKey=tail_key,
                num_warps=4,
                enable_fp_fusion=False,
            )
        prefix_bv = schedule.prefix_rows
        _prefix_segments[(8, 128 // prefix_bv, batch)](
            affine,
            delta_state,
            cache_indices,
            cu_seqlens,
            seg,
            prefix_bv,
            num_warps=2 if prefix_bv == 4 else 4,
            enable_fp_fusion=False,
        )
    if fused_core:
        units = segments if seg else batch
        bv = schedule.core_rows
        core_grid = (
            (8, 128 // bv, units) if schedule.unit_major else (units, 8, 128 // bv)
        )
        _state_and_core[core_grid](
            qkv,
            u,
            w,
            scores,
            coeff,
            delta_state,
            cache_indices,
            cu_seqlens,
            segment_state if seg else delta_state,
            core,
            scale,
            qkv_rows,
            batch,
            bt,
            seg,
            bv,
            4,
            2,
            num_warps=4,
            enable_fp_fusion=False,
            TIME_MAJOR=time_major,
            UNIT_MAJOR=schedule.unit_major,
            TailKey=tail_key,
            TRANSPOSED=schedule.core_transposed,
        )
    else:
        bv, wm = 16, 1
        _chunk_state_rows[(batch, 8, 128 // bv)](
            qkv,
            u,
            w,
            coeff,
            delta_state,
            cache_indices,
            cu_seqlens,
            chunk_state,
            qkv_rows,
            bt,
            bv,
            4,
            wm,
            num_warps=4,
            enable_fp_fusion=False,
            TailKey=tail_key,
        )
    maximum = precision_config.group_quant_max
    inverse_maximum = 1.0 / maximum
    if fused_core:
        norm_rows = schedule.norm_rows
        _normalize_quantize[((m * 8 + norm_rows - 1) // norm_rows,)](
            core,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            eps,
            maximum,
            inverse_maximum,
            m,
            norm_rows,
            schedule.norm_warps,
            num_warps=schedule.norm_warps,
            enable_fp_fusion=False,
        )
    else:
        _chunk_output_quant[(chunks, 8)](
            qkv,
            u,
            scores,
            coeff,
            cu_seqlens,
            chunk_state,
            projected_qkvz,
            norm_weight,
            normalized,
            quantized,
            scales,
            scale,
            eps,
            maximum,
            inverse_maximum,
            qkv_rows,
            batch,
            bt,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return normalized, conv_state, delta_state, quantized, scales
