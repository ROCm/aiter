# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon kernels for the fused Qwen3-Next GDN *prefill* step (gfx950/CDNA4).

A fused Gluon op replaces the prefill chain ``causal_conv1d_split_qkv ->
fused_gdn_gating -> chunk_gated_delta_rule -> gated_rmsnorm_fp8_group_quant`` with
six tight launches that fuse the work intra-kernel and drop the intermediate HBM
round-trips; the epilogue launch emits the per-head group-128 FP8 activations a
block-FP8 ``out_proj`` consumes directly (no separate quant kernel).

Prefill covers a wide token range, so unlike the single decode kernel this ships
four M-tile specializations (each is the winning autotuned schedule for its
token/sequence band). The public wrapper in
``aiter/ops/triton/gated_delta_net/fused_gdn_prefill_qkvz.py`` selects one via a
runtime (tokens, batch) dispatch (``_select_tile_key``); the mapping mirrors the
Artemis MI355 v1 kernel-pack profile it was ported from:

    tokens 1024..3071 , batch 1..64   -> m1024_3071
    tokens 3072..12288, batch 1..64   -> m3072_16384
    tokens 12289..16384, batch 1..5   -> m12289_16384_b1_5
    tokens 12289..16384, batch 6..15  -> m12289_16384_b6_15
    tokens 12289..16384, batch 16..64 -> m3072_16384

All four schedules live here side by side, one canonical place to look, each as
its own set of ``@gluon.jit`` kernels (torch-free). They are ported autotuner
output, not four hand-derived rewrites of the math, so they are pinned to one
contract: for any covered shape the result must match the single
``ref_gdn_prefill`` in ``test_fused_gdn_prefill_qkvz.py`` that every tile is
parametrized against. The schedules are otherwise independent; where two would
have shared a helper name (``_multiply``, ``_update_conv_state``) the m12289_16384
b6_15 copy carries a ``_b6_15`` suffix so each schedule binds its own kernels.

The torch/triton host orchestration lives next to the public wrapper in
``aiter/ops/triton/gated_delta_net/_gdn_prefill_launch.py``.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.extra import libdevice
from triton.language.core import range as loop_range

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

##############################################################################
# schedule m1024_3071
##############################################################################


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


##############################################################################
# schedule m12289_16384_b1_5
##############################################################################


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
    if Bounds is not None and (group == 0) & (part == 0):
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
        _seq, first, end = _chunk_bounds(Starts, chunk, BATCH, BT)
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
                _seq, first, end = _chunk_bounds(Starts, chunk, BATCH, BT)
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
    _seq, first, end = _chunk_bounds(Starts, chunk, BATCH, BT)
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


##############################################################################
# schedule m12289_16384_b6_15
##############################################################################


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


_update_conv_state_b6_15_repr = make_kernel_repr(
    "gdn_prefill_m12289_16384_b6_15_update_conv_state", []
)


@gluon.jit(repr=_update_conv_state_b6_15_repr)
def _update_conv_state_b6_15(qkvz, state, indices, starts, initial):
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
def _multiply_b6_15(left, right):
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
        gl.associative_scan(gl.convert_layout(decay, scan_layout), 0, _multiply_b6_15),
        gl.SliceLayout(1, tri_layout),
    )
    between = gl.associative_scan(
        gl.where(row[:, None] > col[None, :], decay[:, None], 1.0), 0, _multiply_b6_15
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
    _q, _key, inverse, prefix, suffix, _beta, _token_begin, _end = _chunk_coefficients(
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


##############################################################################
# schedule m3072_16384
##############################################################################

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
            if BATCH > 1 and step > 0 and current_token >= end:
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
