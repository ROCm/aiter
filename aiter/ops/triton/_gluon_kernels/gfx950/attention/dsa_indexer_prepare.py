# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fused DSA indexer prepare for decode, gfx950 (Gluon).

Two launches. ``_projections`` computes the three indexer projections as fp32
split-K partials: ``q = q_lora @ wq_b.T`` and ``k = x @ wk.T`` from block-FP8
weights with the 1x128 activation quant fused in, ``gate = x @ weights_proj.T``
in bf16. One program is one 16-row x BN-column x (128*bundle)-K tile; it runs a
single wave. ``_finish`` merges the partials per (row, head) and runs the
epilogue: K LayerNorm, interleaved RoPE, Hadamard(128), ue8m0 FP8 quant, the
preshuffled index-K cache write, and the gate weights. ``_packed_key`` is the
key-only projection: four 128-wide K groups of up to four rows share one
16x(BN*4)x128 MFMA.

Weights are AITER ``shuffle_weight(layout=(16, 16))`` bytes, read directly in
that order. Offsets are 32-bit buffer loads when every operand is below 2 GiB
(``BUFFER``), int64 otherwise.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_INV_SQRT_128 = gl.constexpr(0.08838834764831845)


@gluon.jit
def _load(P, off, mask, other, BUFFER: gl.constexpr):
    if BUFFER:
        return gl.amd.cdna4.buffer_load(P, off, mask, other)
    return gl.load(P + off, mask, other)


@gluon.jit
def _store(v, P, off, mask, BUFFER: gl.constexpr):
    if BUFFER:
        gl.amd.cdna4.buffer_store(v, P, off, mask)
    else:
        gl.store(P + off, v, mask)


@gluon.jit
def _quantize_activation(x):
    """Per-row 1x128 FP8 quant: scale = amax / 448 (fp32), codes e4m3."""
    x = x.to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(x), 1), 1e-10) * (1.0 / 448.0)
    return ((x * gl.div_rn(1.0, scale)[:, None]).to(gl.float8e4nv), scale)


@gluon.jit
def _weight_offsets(n, k, K: gl.constexpr, SW: gl.constexpr, PACKED: gl.constexpr):
    """Element offset of W[n, k]; PACKED = AITER (16, 16) preshuffle of a 1-byte weight."""
    if PACKED:
        return (
            n // 16 * (K // 32) * 512
            + k // 32 * 512
            + k // 16 % 2 * 256
            + n % 16 * 16
            + k % 16
        )
    return n * SW + k


@gluon.jit
def _partial(
    X,
    W,
    S,
    P,
    task,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PACKED: gl.constexpr,
    BUNDLE: gl.constexpr,
    BN: gl.constexpr,
    LOCAL: gl.constexpr,
    BUFFER: gl.constexpr,
    COMPACT: gl.constexpr,
):
    """One 16 x BN x (128*BUNDLE) tile of X @ W.T into the fp32 partials P.

    FP8 (S is the 128x128 block scale table): the activation tile is quantized
    per 128-group and the MFMA runs on e4m3; the fp32 result is scaled by the
    activation and weight scales. COMPACT (M < 16, FP8): the M live rows are
    loaded once and broadcast to the 16-row MFMA tile. LOCAL lays the partials
    out per (row, head, part); otherwise per (part, row, n)."""
    FP8: gl.constexpr = S is not None
    PARTS: gl.constexpr = gl.cdiv(K, 128 * BUNDLE)
    if not BUFFER:
        task = task.to(gl.int64)
    part = task % PARTS
    tile = task // PARTS
    col = tile % gl.cdiv(N, BN) * BN
    row = tile // gl.cdiv(N, BN) * 16
    AM: gl.constexpr = M if COMPACT and FP8 and (M < 16) else 16
    expanded: gl.constexpr = gl.BlockedLayout(
        [1, 1, 2 * AM], [1, AM, 64 // AM], [1, 1, 1], [2, 1, 0]
    )
    if COMPACT and FP8:
        la: gl.constexpr = gl.BlockedLayout([1, 2 * AM], [AM, 64 // AM], [1, 1], [1, 0])
    elif FP8:
        la: gl.constexpr = gl.BlockedLayout([1, 32], [16, 4], [1, 1], [0, 1])
    else:
        la: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [1, 1], [1, 0])
    if FP8:
        lb: gl.constexpr = gl.BlockedLayout([32, 1], [4, 16], [1, 1], [1, 0])
        mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, 1])
        kw: gl.constexpr = 32
    else:
        lb: gl.constexpr = gl.BlockedLayout([4, 1], [16, 4], [1, 1], [0, 1])
        mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 1])
        kw: gl.constexpr = 8
    rm = row + gl.arange(0, AM, gl.SliceLayout(1, la))
    ak = gl.arange(0, 128, gl.SliceLayout(0, la))
    bk = gl.arange(0, 128, gl.SliceLayout(1, lb))
    cn = col + gl.arange(0, BN, gl.SliceLayout(0, lb))
    mr = row + gl.arange(0, 16, gl.SliceLayout(1, mma))
    nr = col + gl.arange(0, BN, gl.SliceLayout(0, mma))
    zero = gl.zeros((16, BN), gl.float32, mma)
    value = zero
    for offset in gl.static_range(BUNDLE):
        group = part * BUNDLE + offset
        ka, kb = (group * 128 + ak, group * 128 + bk)
        if K % (128 * BUNDLE) == 0:
            if M % 16 == 0 or (COMPACT and FP8):
                a = _load(X, rm[:, None] * SX + ka[None, :], None, None, BUFFER)
            else:
                a = _load(
                    X, rm[:, None] * SX + ka[None, :], rm[:, None] < M, 0.0, BUFFER
                )
        else:
            a = _load(
                X,
                rm[:, None] * SX + ka[None, :],
                (rm[:, None] < M) & (ka[None, :] < K),
                0.0,
                BUFFER,
            )
        if FP8:
            a, scale_a = _quantize_activation(a)
            if AM < 16:
                ae = gl.convert_layout(a, gl.SliceLayout(0, expanded))[None, :, :]
                se = gl.convert_layout(
                    scale_a, gl.SliceLayout(0, gl.SliceLayout(2, expanded))
                )[None, :]
                a = gl.reshape(
                    gl.broadcast(
                        ae, gl.full((16 // AM, AM, 128), 0, a.dtype, ae.type.layout)
                    )[0],
                    (16, 128),
                )
                scale_a = gl.reshape(
                    gl.broadcast(
                        se, gl.full((16 // AM, AM), 0, scale_a.dtype, se.type.layout)
                    )[0],
                    (16,),
                )
        bw = _weight_offsets(cn[None, :], kb[:, None], K, SW, PACKED and FP8)
        if N % BN == 0 and K % (128 * BUNDLE) == 0:
            b = _load(W, bw, None, None, BUFFER)
        else:
            b = _load(W, bw, (cn[None, :] < N) & (kb[:, None] < K), 0.0, BUFFER)
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, kw))
        b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, kw))
        if FP8:
            dot = gl.amd.cdna4.mfma_scaled(a, None, "e4m3", b, None, "e4m3", zero)
            sa = gl.convert_layout(scale_a, gl.SliceLayout(1, mma))
            if N % BN == 0 and K % (128 * BUNDLE) == 0:
                sw = _load(S, nr // 128 * gl.cdiv(K, 128) + group, None, None, BUFFER)
            else:
                sw = _load(
                    S,
                    nr // 128 * gl.cdiv(K, 128) + group,
                    (nr < N) & (group < gl.cdiv(K, 128)),
                    0.0,
                    BUFFER,
                )
            value += dot * sa[:, None] * sw[None, :]
        else:
            value = gl.amd.cdna4.mfma(a, b, value)
    if LOCAL:
        dst = (
            mr[:, None] * N * PARTS
            + nr[None, :] // 128 * PARTS * 128
            + part * 128
            + nr[None, :] % 128
        )
    else:
        dst = part * M * N + mr[:, None] * N + nr[None, :]
    _store(value, P, dst, (mr[:, None] < M) & (nr[None, :] < N), BUFFER)


_projections_repr = make_kernel_repr(
    "_dsa_indexer_projections",
    ["M", "K", "QK", "HEADS", "KPG", "QPG", "GPG", "KN", "QN", "BUFFER", "COMPACT"],
)


@gluon.jit(repr=_projections_repr)
def _projections(
    X,
    QL,
    WK,
    WQ,
    WG,
    SK,
    SQ,
    KP,
    QP,
    GP,
    M: gl.constexpr,
    K: gl.constexpr,
    QK: gl.constexpr,
    HEADS: gl.constexpr,
    SX: gl.constexpr,
    SL: gl.constexpr,
    SWK: gl.constexpr,
    SWQ: gl.constexpr,
    SWG: gl.constexpr,
    PACKED: gl.constexpr,
    KPG: gl.constexpr,
    QPG: gl.constexpr,
    GPG: gl.constexpr,
    KN: gl.constexpr,
    QN: gl.constexpr,
    BUFFER: gl.constexpr,
    COMPACT: gl.constexpr,
):
    """Task order: key tiles, query tiles, gate tiles (one wave each)."""
    task = gl.program_id(0)
    KT: gl.constexpr = gl.cdiv(M, 16) * gl.cdiv(128, KN) * gl.cdiv(K, 128 * KPG)
    QT: gl.constexpr = (
        gl.cdiv(M, 16) * gl.cdiv(HEADS * 128, QN) * gl.cdiv(QK, 128 * QPG)
    )
    if task < KT:
        _partial(
            X,
            WK,
            SK,
            KP,
            task,
            M,
            128,
            K,
            SX,
            SWK,
            PACKED,
            KPG,
            KN,
            True,
            BUFFER,
            COMPACT,
        )
    elif task < KT + QT:
        _partial(
            QL,
            WQ,
            SQ,
            QP,
            task - KT,
            M,
            HEADS * 128,
            QK,
            SL,
            SWQ,
            PACKED,
            QPG,
            QN,
            True,
            BUFFER,
            COMPACT,
        )
    else:
        _partial(
            X,
            WG,
            None,
            GP,
            task - KT - QT,
            M,
            HEADS,
            K,
            SX,
            SWG,
            False,
            GPG,
            32,
            False,
            BUFFER,
            False,
        )


_packed_key_repr = make_kernel_repr(
    "_dsa_indexer_packed_key", ["M", "K", "RT", "PARTS", "BUFFER", "BN", "ROW_FIRST"]
)


@gluon.jit(repr=_packed_key_repr)
def _packed_key(
    X,
    W,
    S,
    P,
    M: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    PACKED: gl.constexpr,
    RT: gl.constexpr,
    PARTS: gl.constexpr,
    BUFFER: gl.constexpr,
    BN: gl.constexpr,
    ROW_FIRST: gl.constexpr,
):
    """Key-only projection: RT rows x 4 K-groups share one 16 x (BN*4) x 128 MFMA;
    the block-diagonal of the result is reduced into P[row, part, col]."""
    gl.static_assert(RT <= 4 and K % 512 == 0)
    if ROW_FIRST:
        gl.static_assert(RT == 4 and M % RT == 0)
        task = gl.program_id(0)
        if not BUFFER:
            task = task.to(gl.int64)
        row = task % (M // RT) * RT
        tile = task // (M // RT)
        col = tile % (128 // BN) * BN
        part = tile // (128 // BN)
    else:
        task = gl.program_id(0).to(gl.uint32 if BUFFER else gl.int64)
        part = task % PARTS
        tile = task // PARTS
        col = tile % (128 // BN) * BN
        row = tile // (128 // BN) * RT
    la: gl.constexpr = gl.BlockedLayout([1, 32], [16, 4], [1, 1], [0, 1])
    lb: gl.constexpr = gl.BlockedLayout([32, 1], [4, 16], [1, 1], [1, 0])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], True, [1, 1])
    vr = gl.arange(0, 16, gl.SliceLayout(1, la))
    ak = gl.arange(0, 128, gl.SliceLayout(0, la))
    bk = gl.arange(0, 128, gl.SliceLayout(1, lb))
    vc = gl.arange(0, BN * 4, gl.SliceLayout(0, lb))
    vm = gl.arange(0, 16, gl.SliceLayout(1, mma))
    vn = gl.arange(0, BN * 4, gl.SliceLayout(0, mma))
    ka = (part * 4 + vr[:, None] // RT) * 128 + ak[None, :]
    kb = (part * 4 + vc[None, :] // BN) * 128 + bk[:, None]
    rm, cn = (row + vr % RT, col + vc % BN)
    if ROW_FIRST:
        a = _load(X, rm[:, None] * SX + ka, None, None, BUFFER)
    else:
        a = _load(
            X,
            rm[:, None] * SX + ka,
            (vr[:, None] < RT * 4) & (rm[:, None] < M) & (ka < K),
            0.0,
            BUFFER,
        )
    a, scale_a = _quantize_activation(a)
    bw = _weight_offsets(cn[None, :], kb, K, SW, PACKED)
    b = _load(W, bw, None, None, BUFFER)
    a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 32))
    b = gl.convert_layout(b, gl.DotOperandLayout(1, mma, 32))
    dot = gl.amd.cdna4.mfma_scaled(
        a, None, "e4m3", b, None, "e4m3", gl.zeros((16, BN * 4), gl.float32, mma)
    )
    if ROW_FIRST:
        diagonal = gl.where(vm[:, None] // RT == vn[None, :] // BN, dot, 0.0)
    else:
        diagonal = gl.where(
            (vm[:, None] // RT == vn[None, :] // BN) & (vm[:, None] < RT * 4), dot, 0.0
        )
    useful = gl.sum(gl.reshape(diagonal, (16, 4, BN)), 1)
    sl: gl.constexpr = gl.SliceLayout(1, useful.type.layout)
    sa = gl.convert_layout(scale_a, sl)
    r = gl.arange(0, 16, sl)
    if ROW_FIRST:
        sw = _load(S, part * 4 + r // RT, None, None, BUFFER)
    else:
        sw = _load(S, part * 4 + r // RT, r < RT * 4, 0.0, BUFFER)
    scaled = useful * sa[:, None] * sw[:, None]
    value = gl.sum(gl.reshape(scaled, (16 // RT, RT, BN)), 0)
    rr = row + gl.arange(0, RT, gl.SliceLayout(1, value.type.layout))
    cc = col + gl.arange(0, BN, gl.SliceLayout(0, value.type.layout))
    dst = rr[:, None] * PARTS * 128 + part * 128 + cc[None, :]
    if ROW_FIRST:
        _store(value, P, dst, None, BUFFER)
    else:
        _store(value, P, dst, rr[:, None] < M, BUFFER)


@gluon.jit
def _merge(
    P,
    row,
    head,
    HEADS: gl.constexpr,
    PARTS: gl.constexpr,
    PARTS_P2: gl.constexpr,
    VECTOR: gl.constexpr,
    BUFFER: gl.constexpr,
):
    """Sum the PARTS fp32 partials of one (row, head) and round to bf16."""
    layout: gl.constexpr = gl.BlockedLayout(
        [1, VECTOR], [VECTOR // 2, 128 // VECTOR], [1, 1], [1, 0]
    )
    p = gl.arange(0, PARTS_P2, gl.SliceLayout(1, layout))
    d = gl.arange(0, 128, gl.SliceLayout(0, layout))
    if not BUFFER:
        row, head = (gl.cast(row, gl.int64), gl.cast(head, gl.int64))
    off = (row * HEADS + head) * PARTS * 128 + p[:, None] * 128 + d[None, :]
    x = gl.sum(_load(P, off, p[:, None] < PARTS, 0.0, BUFFER), 0)
    return (
        gl.convert_layout(x, gl.BlockedLayout([2], [64], [1], [0]))
        .to(gl.bfloat16)
        .to(gl.float32)
    )


@gluon.jit
def _butterfly(x, BIT: gl.constexpr):
    x3 = gl.reshape(x, (128 >> BIT + 1, 2, 1 << BIT))
    xp = gl.reshape(gl.permute(x3, (0, 2, 1)), (64, 2))
    xp = gl.convert_layout(xp, gl.BlockedLayout([1, 2], [64, 1], [1, 1], [0, 1]))
    lo, hi = gl.split(xp)
    y3 = gl.reshape(gl.join(lo + hi, lo - hi), (128 >> BIT + 1, 1 << BIT, 2))
    return gl.reshape(gl.permute(y3, (0, 2, 1)), (128,))


@gluon.jit
def _load_rope(
    Cos, Sin, Positions, row, SC: gl.constexpr, SS: gl.constexpr, BUFFER: gl.constexpr
):
    """cos/sin of the row's position, 32 pairs; lanes >= 32 get (1, 0) = identity."""
    pair = gl.arange(0, 64, gl.BlockedLayout([1], [64], [1], [0]))
    pos = gl.load(Positions + row)
    if BUFFER:
        pos = pos.to(gl.int32)
    c = _load(Cos, pos * SC + pair, pair < 32, 1.0, BUFFER).to(gl.float32)
    s = _load(Sin, pos * SS + pair, pair < 32, 0.0, BUFFER).to(gl.float32)
    return (c, s)


@gluon.jit
def _rotate_hadamard(x, c, s):
    """Interleaved RoPE on the first 64 lanes (bf16 rounded), then the normalized
    Hadamard(128) transform (bf16 rounded)."""
    lo, hi = gl.split(gl.reshape(x, (64, 2)))
    c = gl.convert_layout(c, lo.type.layout)
    s = gl.convert_layout(s, lo.type.layout)
    pair = gl.arange(0, 64, lo.type.layout)
    rlo = gl.where(pair < 32, lo * c + -hi * s, lo).to(gl.bfloat16).to(gl.float32)
    rhi = gl.where(pair < 32, hi * c + lo * s, hi).to(gl.bfloat16).to(gl.float32)
    x = gl.reshape(gl.join(rlo + rhi, rlo - rhi), (128,))
    for bit in gl.static_range(1, 7):
        x = _butterfly(x, bit)
    return (x * _INV_SQRT_128).to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _quantize_head(x):
    """ue8m0 head quant: scale = 2^ceil(log2(amax / 448)), codes e4m3."""
    scale = gl.exp2(gl.ceil(gl.log2(gl.maximum(gl.max(gl.abs(x), 0), 0.0001) / 448.0)))
    return ((x * gl.div_rn(1.0, scale)).to(gl.float8e4nv), scale)


@gluon.jit
def _key_finish(
    P,
    Gamma,
    Beta,
    Cos,
    Sin,
    Positions,
    Slots,
    Cache,
    row,
    KG: gl.constexpr,
    KG_P2: gl.constexpr,
    SC: gl.constexpr,
    SS: gl.constexpr,
    EPS: gl.constexpr,
    PAGE: gl.constexpr,
    BUFFER: gl.constexpr,
    KV: gl.constexpr,
):
    """LayerNorm -> RoPE -> Hadamard -> ue8m0 quant -> preshuffled index-K cache
    write (codes in 16x16 tiles, fp32 scale tail). Rows with slot < 0 skip the write."""
    x = _merge(P, row, 0, 1, KG, KG_P2, KV, BUFFER)
    d = gl.arange(0, 128, x.type.layout)
    c, s = _load_rope(Cos, Sin, Positions, row, SC, SS, BUFFER)
    raw_slot = gl.load(Slots + row)
    active = raw_slot >= 0
    slot = raw_slot.to(gl.int32 if BUFFER else gl.int64)
    gamma = _load(Gamma, d, None, None, BUFFER).to(gl.float32)
    beta = _load(Beta, d, None, None, BUFFER).to(gl.float32)
    delta = x - gl.sum(x, 0) * (1.0 / 128.0)
    inv = gl.rsqrt(gl.sum(delta * delta, 0) * (1.0 / 128.0) + EPS)
    x = (delta * inv * gamma + beta).to(gl.bfloat16).to(gl.float32)
    x = _rotate_hadamard(x, c, s)
    q, scale = _quantize_head(x)
    d = gl.arange(0, 128, q.type.layout)
    page, offset = (slot // PAGE, slot % PAGE)
    dst = (
        page * PAGE * 132
        + offset // 16 * 2048
        + d // 16 * 256
        + offset % 16 * 16
        + d % 16
    )
    _store(q.to(gl.uint8, bitcast=True), Cache, dst, active, BUFFER)
    si = gl.arange(0, 1, gl.BlockedLayout([1], [64], [1], [0]))
    sp = page * PAGE * 33 + PAGE * 32 + offset
    _store(scale, gl.cast(Cache, gl.pointer_type(gl.float32)), sp + si, active, BUFFER)


@gluon.jit
def _gate(
    GP,
    row,
    head,
    M: gl.constexpr,
    HEADS: gl.constexpr,
    GG: gl.constexpr,
    GG_P2: gl.constexpr,
    BUFFER: gl.constexpr,
):
    """Sum the gate partials of one (row, head); bf16 rounded like a bf16 GEMM output."""
    p = gl.arange(0, GG_P2, gl.BlockedLayout([1], [64], [1], [0]))
    if not BUFFER:
        p = p.to(gl.int64)
    gate = gl.sum(_load(GP, p * M * HEADS + row * HEADS + head, p < GG, 0.0, BUFFER), 0)
    return gate.to(gl.bfloat16).to(gl.float32)


@gluon.jit
def _query_finish(
    QP,
    GP,
    Cos,
    Sin,
    Positions,
    Out,
    Weights,
    row,
    head,
    M: gl.constexpr,
    HEADS: gl.constexpr,
    QG: gl.constexpr,
    QG_P2: gl.constexpr,
    GG: gl.constexpr,
    GG_P2: gl.constexpr,
    SC: gl.constexpr,
    SS: gl.constexpr,
    WEIGHTS_SCALE: gl.constexpr,
    BUFFER: gl.constexpr,
):
    """RoPE -> Hadamard -> ue8m0 quant of one query head; weights = gate * q_scale * WEIGHTS_SCALE."""
    gate = _gate(GP, row, head, M, HEADS, GG, GG_P2, BUFFER)
    x = _merge(QP, row, head, HEADS, QG, QG_P2, 2, BUFFER)
    c, s = _load_rope(Cos, Sin, Positions, row, SC, SS, BUFFER)
    x = _rotate_hadamard(x, c, s)
    q, scale = _quantize_head(x)
    d = gl.arange(0, 128, q.type.layout)
    _store(q, Out, (row * HEADS + head) * 128 + d, None, BUFFER)
    gl.store(Weights + row * HEADS + head, gate * scale * WEIGHTS_SCALE)


_finish_repr = make_kernel_repr(
    "_dsa_indexer_finish", ["M", "HEADS", "KG", "QG", "GG", "KEYONLY", "BUFFER", "KV"]
)


@gluon.jit(repr=_finish_repr)
def _finish(
    KP,
    QP,
    GP,
    Gamma,
    Beta,
    Cos,
    Sin,
    Positions,
    Slots,
    Cache,
    Out,
    Weights,
    M: gl.constexpr,
    HEADS: gl.constexpr,
    KG: gl.constexpr,
    QG: gl.constexpr,
    GG: gl.constexpr,
    KG_P2: gl.constexpr,
    QG_P2: gl.constexpr,
    GG_P2: gl.constexpr,
    SC: gl.constexpr,
    SS: gl.constexpr,
    EPS: gl.constexpr,
    PAGE: gl.constexpr,
    WEIGHTS_SCALE: gl.constexpr,
    KEYONLY: gl.constexpr,
    BUFFER: gl.constexpr,
    KV: gl.constexpr,
):
    """Grid (M, HEADS + 1): program (row, HEADS) finishes the key, (row, h) query head h."""
    row = gl.program_id(0)
    head = gl.program_id(1)
    if not BUFFER:
        row, head = (row.to(gl.int64), head.to(gl.int64))
    if KEYONLY or head == HEADS:
        _key_finish(
            KP,
            Gamma,
            Beta,
            Cos,
            Sin,
            Positions,
            Slots,
            Cache,
            row,
            KG,
            KG_P2,
            SC,
            SS,
            EPS,
            PAGE,
            BUFFER,
            KV,
        )
    else:
        _query_finish(
            QP,
            GP,
            Cos,
            Sin,
            Positions,
            Out,
            Weights,
            row,
            head,
            M,
            HEADS,
            QG,
            QG_P2,
            GG,
            GG_P2,
            SC,
            SS,
            WEIGHTS_SCALE,
            BUFFER,
        )
