# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon kernels for the fused MLA qkv_a projection + q/kv RMSNorm (gfx950/CDNA4).

Each call is two launches. A producer launch quantizes the BF16 activation per
1x128 group to FP8 e4m3, multiplies it with the (16, 16)-preshuffled FP8 weight
(128x128 block scales) and writes split-K partial sums in FP32. A finish launch
adds the partials, rounds the projection to BF16, applies RMSNorm to the q and
kv slices and copies the rope slice.

The split-K partials and the intra-CTA wave exchange are kept in FP32: storing
them in BF16/FP16 loses accuracy when the partial sums cancel.

The host code is in ``aiter/ops/triton/fusions/fused_mla_qkv_a_proj_rmsnorm.py``.
"""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

##############################################################################
# shared device helpers
##############################################################################


@gluon.jit
def _axes3(D0: gl.constexpr, D1: gl.constexpr, D2: gl.constexpr, LAYOUT: gl.constexpr):
    d0 = gl.arange(0, D0, gl.SliceLayout(1, gl.SliceLayout(2, LAYOUT)))
    d1 = gl.arange(0, D1, gl.SliceLayout(0, gl.SliceLayout(2, LAYOUT)))
    d2 = gl.arange(0, D2, gl.SliceLayout(0, gl.SliceLayout(1, LAYOUT)))
    return (d0, d1, d2)


@gluon.jit
def _quantize_block(x, DTYPE: gl.constexpr):
    # Dynamic per-group FP8 e4m3 quant: scale = amax / 448.
    axis: gl.constexpr = len(x.type.shape) - 1
    scale = gl.maximum(gl.max(gl.abs(x), axis), 1e-10) * (1.0 / 448.0)
    return ((x * gl.expand_dims(gl.div_rn(1.0, scale), axis)).to(DTYPE), scale)


@gluon.jit
def _packed_offset(k, n, K: gl.constexpr, group=0):
    # Byte offset of W[n, group * 128 + k] in the (16, 16)-preshuffled weight.
    return n // 16 * K * 16 + group * 2048 + k // 16 * 256 + n % 16 * 16 + k % 16


@gluon.jit
def _store_partial(
    value,
    P,
    tile,
    split,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    BN: gl.constexpr,
    BUFFER: gl.constexpr = True,
    CACHE: gl.constexpr = None,
):
    layout: gl.constexpr = value.type.layout
    row = gl.arange(0, M, gl.SliceLayout(1, layout))
    col = gl.arange(0, BN, gl.SliceLayout(0, layout))
    if BUFFER:
        gl.amd.cdna4.buffer_store(
            value.to(P.dtype.element_ty),
            P + split * M * PITCH + tile * BN,
            row[:, None] * PITCH + col[None, :],
            cache=CACHE,
        )
    else:
        gl.store(
            P + split * M * PITCH + row[:, None] * PITCH + tile * BN + col[None, :],
            value,
            cache_modifier=CACHE,
        )


##############################################################################
# small decode (M = 1, 2, 4, 8, 16): producers
##############################################################################


@gluon.jit
def _diagonal_two(dot, M: gl.constexpr, BN: gl.constexpr, WAVES: gl.constexpr):
    blocks = dot.reshape((WAVES, 2, M, 2, BN)).permute((0, 1, 2, 4, 3))
    a, b = gl.split(blocks.reshape((WAVES, 2, M, BN, 2)))
    group = gl.arange(
        0, 2, gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, a.type.layout)))
    )
    return gl.where(group[None, :, None, None] == 0, a, b)


@gluon.jit
def _diagonal_eight(dot, M: gl.constexpr, BN: gl.constexpr):
    blocks = dot.reshape((8, M, 8, BN)).permute((0, 1, 3, 2))
    even, odd = gl.split(blocks.reshape((8, M, BN, 4, 2)))
    ae, cg = gl.split(even.reshape((8, M, BN, 2, 2)))
    bf, dh = gl.split(odd.reshape((8, M, BN, 2, 2)))
    a, e = gl.split(ae)
    c, g = gl.split(cg)
    b, f = gl.split(bf)
    d, h = gl.split(dh)
    layout: gl.constexpr = a.type.layout
    group = gl.arange(0, 8, gl.SliceLayout(1, gl.SliceLayout(2, layout)))[:, None, None]
    if M == 2:
        bit0, bit1, bit2 = (group & 1 != 0, group & 2 != 0, group & 4 != 0)
        low = gl.where(bit1, gl.where(bit0, d, c), gl.where(bit0, b, a))
        high = gl.where(bit1, gl.where(bit0, h, g), gl.where(bit0, f, e))
        selected = gl.where(bit2, high, low)
    else:
        choices = (a, b, c, d, e, f, g)
        selected = h
        for index in gl.static_range(6, -1, -1):
            selected = gl.where(group == index, choices[index], selected)
    return selected


_project_native_repr = make_kernel_repr(
    "fused_mla_qkv_a_proj_rmsnorm_project_native", ["M", "BN", "PACK", "AV"]
)


@gluon.jit(repr=_project_native_repr)
def _project_native(
    X,
    W,
    WS,
    P,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    BN: gl.constexpr,
    PACK: gl.constexpr,
    WAVES: gl.constexpr,
    AV: gl.constexpr,
):
    # One wave, eight K groups in one 16x16x128 scaled MFMA (rows = group x M).
    gl.static_assert(WAVES == 1 and PACK == 8 and (M <= 2))
    BM: gl.constexpr = M * PACK
    al: gl.constexpr = gl.BlockedLayout([1, AV], [BM, 64 // BM], [1, 1], [1, 0])
    bl: gl.constexpr = gl.BlockedLayout([32, 1], [4, 16], [1, 1], [1, 0])
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 128], False, [1, 1])
    ra = gl.arange(0, BM, gl.SliceLayout(1, al))
    ka = gl.arange(0, 128, gl.SliceLayout(0, al))
    kb = gl.arange(0, 128, gl.SliceLayout(1, bl))
    nb = gl.arange(0, PACK * BN, gl.SliceLayout(0, bl))
    rc = gl.arange(0, BM, gl.SliceLayout(1, ml))
    tile, split = (gl.program_id(0), gl.program_id(1))
    first = split * PACK
    xo = ra[:, None] % M * SX + ra[:, None] // M * 128 + ka[None, :]
    wo = _packed_offset(nb[None, :] // BN * 128 + kb[:, None], nb[None, :] % BN, K)
    x = gl.amd.cdna4.buffer_load(X + first * 128, xo).to(gl.float32)
    a, sa = _quantize_block(x, W.dtype.element_ty)
    b = gl.amd.cdna4.buffer_load(W + tile * BN * K + first * 2048, wo)
    if M == 1:
        sw = gl.amd.cdna4.buffer_load(
            WS + tile * BN // 128 * (K // 128) + first, rc // M
        )
    else:
        sw = gl.load(WS + tile * BN // 128 * (K // 128) + first + rc // M)
    dot = gl.amd.cdna4.mfma_scaled(
        gl.convert_layout(a, gl.DotOperandLayout(0, ml, 32)).to(gl.uint8, bitcast=True),
        None,
        "e4m3",
        gl.convert_layout(b, gl.DotOperandLayout(1, ml, 32)).to(gl.uint8, bitcast=True),
        None,
        "e4m3",
        gl.full((BM, PACK * BN), 0, gl.float32, ml),
    )
    dot = _diagonal_eight(dot, M, BN)
    dl: gl.constexpr = dot.type.layout
    sa = gl.convert_layout(sa.reshape((PACK, M)), gl.SliceLayout(2, dl))
    sw = gl.convert_layout(sw.reshape((PACK, M)), gl.SliceLayout(2, dl))
    value = gl.sum(dot * sa[:, :, None] * sw[:, :, None], 0)
    _store_partial(value, P, tile, split, M, PITCH, BN, CACHE=".wt" if M == 2 else None)


_project_two_repr = make_kernel_repr(
    "fused_mla_qkv_a_proj_rmsnorm_project_two", ["M", "BN", "WAVES", "AV", "FN"]
)


@gluon.jit(repr=_project_two_repr)
def _project_two(
    X,
    W,
    WS,
    P,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    BN: gl.constexpr,
    PACK: gl.constexpr,
    WAVES: gl.constexpr,
    AV: gl.constexpr,
    FN: gl.constexpr,
):
    # Eight waves, two K groups per wave packed in the MFMA rows; LDS wave reduce.
    gl.static_assert(
        M == 4 and PACK == 2 and (WAVES == 8) and (BN == 32) and (FN == 16)
    )
    BM: gl.constexpr = M * PACK
    al: gl.constexpr = gl.BlockedLayout(
        [1, 1, AV], [1, BM, 64 // BM], [WAVES, 1, 1], [2, 1, 0]
    )
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [WAVES, 1, 1])
    bl: gl.constexpr = gl.BlockedLayout(
        [1, 16, 1], [1, 4, 16], [WAVES, 1, 1], [2, 1, 0]
    )
    rl: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [1, 2, 32], [1, WAVES // 2, 2], [2, 1, 0]
    )
    ga, ra, ka = _axes3(WAVES, BM, 128, al)
    gb, kb, nb = _axes3(WAVES, 128, PACK * FN, bl)
    gc = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
    rc = gl.arange(0, BM, gl.SliceLayout(0, gl.SliceLayout(2, ml)))
    tile, split = (gl.program_id(0).to(gl.uint32), gl.program_id(1).to(gl.uint32))
    first = split * WAVES * PACK
    ag = ga[:, None, None] * PACK + ra[None, :, None] // M
    x = gl.amd.cdna4.buffer_load(
        X + first * 128, ra[None, :, None] % M * SX + ag * 128 + ka[None, None, :]
    ).to(gl.float32)
    a, sa = _quantize_block(x, W.dtype.element_ty)
    a = gl.convert_layout(a, gl.DotOperandLayout(0, ml, 16))
    sa = gl.convert_layout(sa, gl.SliceLayout(2, ml))
    sw = gl.amd.cdna4.buffer_load(
        WS + tile * BN // 128 * (K // 128) + first,
        gc[:, None] * PACK + rc[None, :] // M,
    )
    factor = sa * sw
    shared = gl.allocate_shared_memory(
        gl.float32, (WAVES, M, BN), gl.SwizzledSharedLayout(4, 2, 4, [2, 1, 0])
    )
    for frag in gl.static_range(BN // FN):
        bk = (gb[:, None, None] * PACK + nb[None, None, :] // FN) * 128 + kb[
            None, :, None
        ]
        wo = _packed_offset(bk, nb[None, None, :] % FN, K)
        b = gl.convert_layout(
            gl.amd.cdna4.buffer_load(
                W + (tile * BN + frag * FN) * K + first * 2048, wo
            ),
            gl.DotOperandLayout(1, ml, 16),
        )
        dot = gl.amd.cdna4.mfma(
            a, b, gl.full((WAVES, BM, PACK * FN), 0, gl.float32, ml)
        )
        dot = _diagonal_two(dot, M, FN, WAVES).reshape((WAVES, BM, FN))
        factors = gl.convert_layout(factor, gl.SliceLayout(2, dot.type.layout))
        value = gl.sum((dot * factors[:, :, None]).reshape((WAVES, PACK, M, FN)), 1)
        shared.slice(frag * FN, FN, 2).store(value)
    value = gl.sum(shared.load(rl), 0)
    _store_partial(value, P, tile, split, M, PITCH, BN, CACHE=".wt")


_project_stream_repr = make_kernel_repr(
    "fused_mla_qkv_a_proj_rmsnorm_project_stream", ["M", "BN", "WAVES", "AV", "FN"]
)


@gluon.jit(repr=_project_stream_repr)
def _project_stream(
    X,
    W,
    WS,
    P,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    BN: gl.constexpr,
    PACK: gl.constexpr,
    WAVES: gl.constexpr,
    AV: gl.constexpr,
    FN: gl.constexpr,
):
    # One K group per wave; the waves reduce through an FP32 LDS exchange.
    gl.static_assert(PACK == 1 and (M == 8 or M == 16))
    gl.static_assert(FN % 16 == 0 and BN % FN == 0 and (BN <= 128))
    al: gl.constexpr = gl.BlockedLayout(
        [1, 1, AV], [1, M, 64 // M], [WAVES, 1, 1], [2, 1, 0]
    )
    ml: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [WAVES, 1, 1])
    if M == 8:
        bl: gl.constexpr = gl.DotOperandLayout(1, ml, 16)
        rl: gl.constexpr = gl.BlockedLayout(
            [1, 1, 1], [1, 2, 32], [1, WAVES // 2, 2], [2, 1, 0]
        )
    else:
        bl: gl.constexpr = gl.BlockedLayout(
            [1, 16, 1], [1, 4, 16], [WAVES, 1, 1], [2, 1, 0]
        )
        rl: gl.constexpr = gl.BlockedLayout(
            [1, 1, 2], [1, 4, 16], [1, WAVES // 2, 2], [2, 1, 0]
        )
    ga, ra, ka = _axes3(WAVES, M, 128, al)
    gb, kb, nb = _axes3(WAVES, 128, FN, bl)
    gc = gl.arange(0, WAVES, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
    tile, split = (gl.program_id(0), gl.program_id(1))
    if M == 16:
        tile, split = (tile.to(gl.uint32), split.to(gl.uint32))
    first = split * WAVES
    sw = gl.amd.cdna4.buffer_load(WS + tile * BN // 128 * (K // 128) + first, gc)[
        :, None
    ]
    if M == 16:
        xo = ra[None, :, None] * SX + ga[:, None, None] * 128 + ka[None, None, :]
        x = gl.amd.cdna4.buffer_load(X + first * 128, xo).to(gl.float32)
    else:
        xo = (
            ra[None, :, None] * SX
            + (first + ga[:, None, None]) * 128
            + ka[None, None, :]
        )
        x = gl.amd.cdna4.buffer_load(X, xo).to(gl.float32)
    a, sa = _quantize_block(x, W.dtype.element_ty)
    a = gl.convert_layout(a, gl.DotOperandLayout(0, ml, 16))
    sa = gl.convert_layout(sa, gl.SliceLayout(2, ml))
    factor = sa * sw
    shared = gl.allocate_shared_memory(
        gl.float32, (WAVES, M, BN), gl.SwizzledSharedLayout(4, 1, 8, [2, 1, 0])
    )
    for frag in gl.static_range(BN // FN):
        if M == 16:
            bk = gb[:, None, None] * 128 + kb[None, :, None]
            bn = nb[None, None, :]
            wo = (
                bn // 16 * K * 16
                + bk // 32 * 512
                + bk // 16 % 2 * 256
                + bn % 16 * 16
                + bk % 16
            )
        else:
            wo = _packed_offset(
                kb[None, :, None], nb[None, None, :], K, gb[:, None, None]
            )
        b = gl.convert_layout(
            gl.amd.cdna4.buffer_load(
                W + (tile * BN + frag * FN) * K + first * 2048, wo
            ),
            gl.DotOperandLayout(1, ml, 16),
        )
        dot = gl.amd.cdna4.mfma(a, b, gl.full((WAVES, M, FN), 0, gl.float32, ml))
        shared.slice(frag * FN, FN, 2).store(dot * factor[:, :, None])
    value = gl.sum(shared.load(rl), 0)
    _store_partial(value, P, tile, split, M, PITCH, BN, BUFFER=M != 8, CACHE=".wt")


##############################################################################
# small decode (M = 1, 2, 4, 8, 16): split-K merge + RMSNorm
##############################################################################


@gluon.jit
def _finish_row(
    P,
    G,
    Y,
    RO,
    row,
    shard,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    OFFSET: gl.constexpr,
    WIDTH: gl.constexpr,
    TAIL: gl.constexpr,
    EPS: gl.constexpr,
    PARTS: gl.constexpr,
    SHARDS: gl.constexpr,
    VEC: gl.constexpr,
    BLOCK: gl.constexpr,
    TAIL_BLOCK: gl.constexpr,
    NARROW_TAIL: gl.constexpr = False,
    BUFFER_GAMMA: gl.constexpr = False,
    BUFFER_OUTPUT: gl.constexpr = False,
):
    LOAD_WIDTH: gl.constexpr = WIDTH if NARROW_TAIL else WIDTH + TAIL
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [4], [0])
    d = gl.arange(0, BLOCK, layout)
    acc = gl.full((BLOCK,), 0, gl.float32, layout)
    if NARROW_TAIL:
        rope_layout: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
        rd = gl.arange(0, TAIL_BLOCK, rope_layout)
        racc = gl.full((TAIL_BLOCK,), 0, gl.float32, rope_layout)
    for part in gl.static_range(PARTS):
        acc += gl.amd.cdna4.buffer_load(
            P + part * M * PITCH + row * PITCH + OFFSET, d, d < LOAD_WIDTH, 0
        ).to(gl.float32)
        if NARROW_TAIL:
            racc += gl.amd.cdna4.buffer_load(
                P + part * M * PITCH + row * PITCH + OFFSET + WIDTH, rd, rd < TAIL, 0
            ).to(gl.float32)
    # The projection is rounded to BF16 before the norm, as an unfused GEMM output.
    x = acc.to(gl.bfloat16).to(gl.float32)
    if not NARROW_TAIL:
        if BUFFER_GAMMA:
            gamma = gl.amd.cdna4.buffer_load(G, d, d < WIDTH, 0).to(gl.float32)
        else:
            gamma = gl.load(G + d, d < WIDTH, 0).to(gl.float32)
    norm_x = gl.where(d < WIDTH, x, 0.0)
    inv = gl.rsqrt(gl.sum(norm_x * norm_x, 0) / WIDTH + EPS)
    owned = d < WIDTH
    if SHARDS > 1:
        owned = owned & (d // gl.cdiv(WIDTH, SHARDS) == shard)
    if NARROW_TAIL:
        gamma = gl.amd.cdna4.buffer_load(G, d, owned, 0).to(gl.float32)
        gl.amd.cdna4.buffer_store(
            (x * inv * gamma).to(gl.bfloat16), Y + row * WIDTH, d, owned, cache=".wt"
        )
        gl.amd.cdna4.buffer_store(
            racc.to(gl.bfloat16),
            RO + row * TAIL,
            rd,
            (rd < TAIL) & (shard == 0),
            cache=".wt",
        )
    else:
        if BUFFER_OUTPUT:
            gl.amd.cdna4.buffer_store(
                (x * inv * gamma).to(gl.bfloat16),
                Y + row * WIDTH,
                d,
                owned,
                cache=".wt",
            )
        else:
            gl.store(
                Y + row * WIDTH + d,
                (x * inv * gamma).to(gl.bfloat16),
                owned,
                cache_modifier=".wt",
            )
        if TAIL:
            rope_owned = (d >= WIDTH) & (d < WIDTH + TAIL)
            if SHARDS > 1:
                rope_owned = rope_owned & (shard == 0)
            gl.store(
                RO + row * TAIL + d - WIDTH,
                x.to(gl.bfloat16),
                rope_owned,
                cache_modifier=".wt",
            )


_finish_repr = make_kernel_repr(
    "fused_mla_qkv_a_proj_rmsnorm_finish", ["M", "PARTS", "QS", "KS", "NARROW_ROPE"]
)


@gluon.jit(repr=_finish_repr)
def _finish(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    M: gl.constexpr,
    PITCH: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    EPS: gl.constexpr,
    PARTS: gl.constexpr,
    QS: gl.constexpr,
    KS: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_KV: gl.constexpr,
    BLOCK_R: gl.constexpr,
    NARROW_ROPE: gl.constexpr,
    BUFFER_Q_GAMMA: gl.constexpr,
    BUFFER_Q_OUTPUT: gl.constexpr,
    KV_VEC: gl.constexpr,
):
    # grid (M, QS + KS): q shards first, then kv shards (kv shard 0 also writes rope).
    row, role = (gl.program_id(0), gl.program_id(1))
    if M == 1:
        row = 0
    if role < QS:
        _finish_row(
            P,
            QG,
            QO,
            RO,
            row,
            role,
            M,
            PITCH,
            0,
            Q,
            0,
            EPS,
            PARTS,
            QS,
            8,
            BLOCK_Q,
            BLOCK_R,
            False,
            BUFFER_Q_GAMMA,
            BUFFER_Q_OUTPUT,
        )
    else:
        _finish_row(
            P,
            KG,
            KO,
            RO,
            row,
            role - QS,
            M,
            PITCH,
            Q,
            KV,
            R,
            EPS,
            PARTS,
            KS,
            KV_VEC,
            BLOCK_KV,
            BLOCK_R,
            NARROW_ROPE,
        )


##############################################################################
# decode M = 32: chained-K producer + finish
##############################################################################


@gluon.jit
def _m32_indices(
    g: gl.constexpr, rows: gl.constexpr, cols: gl.constexpr, layout: gl.constexpr
):
    gi = gl.arange(0, g, gl.SliceLayout(1, gl.SliceLayout(2, layout)))
    ri = gl.arange(0, rows, gl.SliceLayout(0, gl.SliceLayout(2, layout)))
    ci = gl.arange(0, cols, gl.SliceLayout(0, gl.SliceLayout(1, layout)))
    return (gi[:, None, None], ri[None, :, None], ci[None, None, :])


@gluon.jit
def _m32_load_record(X, W, WS, offsets, start):
    ao, wo, gs = offsets
    x = gl.amd.cdna4.buffer_load(X + start * 128, ao)
    w = gl.amd.cdna4.buffer_load(W + start * 2048, wo)
    sw = gl.amd.cdna4.buffer_load(WS + start, gs)
    return (x.to(gl.float32), w, sw)


@gluon.jit
def _m32_apply_record(record, zeros):
    x, w, sw = record
    ml: gl.constexpr = zeros.type.layout
    ad: gl.constexpr = gl.DotOperandLayout(0, ml, 16)
    x = x.to(gl.float32)
    scale = gl.maximum(gl.max(gl.abs(x), 2), 1e-10) * (1.0 / 448.0)
    a = gl.convert_layout((x * gl.div_rn(1.0, scale)[:, :, None]).to(w.dtype), ad)
    sa = gl.convert_layout(scale, gl.SliceLayout(2, ml))
    return gl.amd.cdna4.mfma(a, w, zeros) * (sa * sw[:, None])[:, :, None]


_m32_project_repr = make_kernel_repr(
    "fused_mla_qkv_a_proj_rmsnorm_m32_project", ["M", "BN", "SPLIT", "CHAINS"]
)


@gluon.jit(repr=_m32_project_repr)
def _m32_project(
    X,
    W,
    WS,
    P,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    BN: gl.constexpr,
    SPLIT: gl.constexpr,
    CHAINS: gl.constexpr,
    FOLD_VEC: gl.constexpr,
    FOLD_ROW_LANES: gl.constexpr,
    FOLD_ORDER: gl.constexpr,
):
    # grid (SPLIT, N // BN); CHAINS waves each own one K group per step.
    gl.static_assert(M == 32)
    gl.static_assert(BN == 64 and N % BN == 0)
    gl.static_assert(K % (128 * CHAINS * SPLIT) == 0)
    si, ni = (gl.program_id(0).to(gl.uint32), gl.program_id(1).to(gl.uint32))
    gl.assume(si < SPLIT)
    gl.assume(ni < N // BN)
    si = (si + ni) % SPLIT
    row_waves: gl.constexpr = gl.num_warps() // CHAINS
    ml: gl.constexpr = gl.amd.AMDMFMALayout(
        4, [16, 16, 32], True, [CHAINS, row_waves, 1]
    )
    bd: gl.constexpr = gl.DotOperandLayout(1, ml, 16)
    al: gl.constexpr = gl.BlockedLayout(
        [1, 1, 16], [1, 16, 4], [CHAINS, row_waves, 1], [2, 1, 0]
    )
    ga, ra, ka = _m32_indices(CHAINS, M, 128, al)
    gb, kb, nb = _m32_indices(CHAINS, 128, BN, bd)
    gs = gl.arange(0, CHAINS, gl.SliceLayout(1, gl.SliceLayout(2, ml)))
    ao = ra * SX + ka + ga * 128
    wo = nb // 16 * K * 16 + kb // 32 * 512
    wo += kb // 16 % 2 * 256 + nb % 16 * 16 + kb % 16 + gb * 2048
    W += ni * BN * K
    WS += ni * BN // 128 * (K // 128)
    offsets = (ao, wo, gs)
    ITERS: gl.constexpr = K // (128 * CHAINS * SPLIT)
    first = si * CHAINS * ITERS
    zeros = gl.zeros((CHAINS, M, BN), gl.float32, ml)
    acc = zeros
    for it in gl.static_range(ITERS):
        record = _m32_load_record(X, W, WS, offsets, first + it * CHAINS)
        acc += _m32_apply_record(record, zeros)
    fold: gl.constexpr = gl.BlockedLayout(
        [CHAINS, 1, FOLD_VEC],
        [1, FOLD_ROW_LANES, 64 // FOLD_ROW_LANES],
        [1, gl.num_warps(), 1],
        FOLD_ORDER,
    )
    merged = gl.sum(gl.convert_layout(acc, fold), 0)
    cl: gl.constexpr = merged.type.layout
    row = gl.arange(0, M, gl.SliceLayout(1, cl))
    col = ni * BN + gl.arange(0, BN, gl.SliceLayout(0, cl))
    gl.amd.cdna4.buffer_store(
        merged.to(P.dtype.element_ty),
        P + si * N,
        row[:, None] * (SPLIT * N) + col[None, :],
        cache=".wt",
    )


@gluon.jit
def _m32_finish_role(
    P,
    Gamma,
    Out,
    Rope,
    row,
    N: gl.constexpr,
    OFFSET: gl.constexpr,
    WIDTH: gl.constexpr,
    R: gl.constexpr,
    EPS: gl.constexpr,
    SPLIT: gl.constexpr,
    VEC: gl.constexpr,
    BLOCK: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([VEC], [64], [gl.num_warps()], [0])
    d = gl.arange(0, BLOCK, layout)
    P += row * (SPLIT * N) + OFFSET
    x = gl.amd.cdna4.buffer_load(P, d, d < WIDTH + R, 0.0).to(gl.float32)
    for split in gl.static_range(1, SPLIT):
        x += gl.amd.cdna4.buffer_load(P + split * N, d, d < WIDTH + R, 0.0).to(
            gl.float32
        )
    x = x.to(gl.bfloat16).to(gl.float32)
    gamma = gl.load(Gamma + d, d < WIDTH, 0).to(gl.float32)
    xn = gl.where(d < WIDTH, x, 0.0)
    variance = gl.sum(xn * xn, 0) / WIDTH
    y = x * gl.rsqrt(variance + EPS) * gamma
    gl.store(Out + row * WIDTH + d, y.to(gl.bfloat16), d < WIDTH, cache_modifier=".wt")
    if R:
        gl.store(
            Rope + row * R + d - WIDTH,
            x.to(gl.bfloat16),
            (d >= WIDTH) & (d < WIDTH + R),
            cache_modifier=".wt",
        )


_m32_finish_repr = make_kernel_repr(
    "fused_mla_qkv_a_proj_rmsnorm_m32_finish", ["SPLIT", "Q_VEC", "KV_VEC"]
)


@gluon.jit(repr=_m32_finish_repr)
def _m32_finish(
    P,
    QG,
    KG,
    QO,
    KO,
    RO,
    N: gl.constexpr,
    Q: gl.constexpr,
    KV: gl.constexpr,
    R: gl.constexpr,
    EPS: gl.constexpr,
    SPLIT: gl.constexpr,
    Q_VEC: gl.constexpr,
    KV_VEC: gl.constexpr,
    BLOCK_Q: gl.constexpr,
    BLOCK_KV: gl.constexpr,
):
    # grid (M, 2): role 0 = q norm, role 1 = kv norm + rope copy.
    row = gl.program_id(0)
    if gl.program_id(1) == 0:
        _m32_finish_role(P, QG, QO, RO, row, N, 0, Q, 0, EPS, SPLIT, Q_VEC, BLOCK_Q)
    else:
        _m32_finish_role(P, KG, KO, RO, row, N, Q, KV, R, EPS, SPLIT, KV_VEC, BLOCK_KV)
