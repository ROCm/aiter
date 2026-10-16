"""Selected kda layer decode schedules and shared helpers."""

import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr

_CONV = gl.constexpr(gl.BlockedLayout([1, 1], [1, 64], [4, 2], [1, 0]))
_OUTPUT = gl.constexpr(gl.BlockedLayout([1, 2], [1, 64], [8, 1], [1, 0]))
_VECTOR = gl.constexpr(gl.BlockedLayout([2], [64], [8], [0]))


@gluon.jit
def _m1_sigmoid(x):
    return 1.0 / (1.0 + gl.exp2(x * -1.4426950408889634))


@gluon.jit
def _m1_split_vector(v, ST: gl.constexpr):
    left, right = gl.split(gl.permute(gl.reshape(v, (2, 64)), (1, 0)))
    v0, v1 = gl.split(gl.permute(gl.reshape(left, (2, 32)), (1, 0)))
    v2, v3 = gl.split(gl.permute(gl.reshape(right, (2, 32)), (1, 0)))
    layout: gl.constexpr = gl.SliceLayout(0, ST)
    return (
        gl.convert_layout(v0, layout, assert_trivial=True),
        gl.convert_layout(v1, layout, assert_trivial=True),
        gl.convert_layout(v2, layout, assert_trivial=True),
        gl.convert_layout(v3, layout, assert_trivial=True),
    )


@gluon.jit
def _m1_contract(packets, vectors, ORDER: gl.constexpr):
    partial = packets[ORDER[0]] * vectors[ORDER[0]][None, :]
    for j in gl.static_range(1, 4):
        partial = gl.fma(packets[ORDER[j]], vectors[ORDER[j]][None, :], partial)
    return gl.sum(partial, 1)


@gluon.jit
def _m128_stage_masked_panel(sa, sb, X, W, ao, bo, am, bm, offset):
    la = sa.reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    lb = sb.reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(la, X + offset, ao, am, 0)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(lb, W + offset, bo, bm, 0)
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m128_eight_parts(values):
    rows: gl.constexpr = values.shape[0]
    even, odd = gl.split(gl.reshape(values, (rows, 16, 4, 2)))
    e04, e26 = gl.split(gl.reshape(even, (rows, 16, 2, 2)))
    o15, o37 = gl.split(gl.reshape(odd, (rows, 16, 2, 2)))
    p0, p4 = gl.split(e04)
    p2, p6 = gl.split(e26)
    p1, p5 = gl.split(o15)
    p3, p7 = gl.split(o37)
    return (p0, p1, p2, p3, p4, p5, p6, p7)


_m256_fallback_gemm_repr = make_kernel_repr(
    "_m256_fallback_gemm",
    ["M", "N", "K"],
)


@gluon.jit(repr=_m256_fallback_gemm_repr)
def _m256_fallback_gemm(
    X,
    W,
    Y,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SXM: gl.constexpr,
    SXK: gl.constexpr,
    SWN: gl.constexpr,
    SWK: gl.constexpr,
    SY: gl.constexpr,
):
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [1, 4])
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [4, 1], [1, 0])
    rows = gl.program_id(1) * 16 + gl.arange(0, 16, gl.SliceLayout(1, layout))
    cols = gl.program_id(0) * 64 + gl.arange(0, 64, gl.SliceLayout(1, layout))
    ks = gl.arange(0, 64, gl.SliceLayout(0, layout))
    acc = gl.zeros((16, 64), gl.float32, mma)

    for block in range(gl.cdiv(K, 64)):
        kk = block * 64 + ks
        a = gl.load(
            X + rows[:, None] * SXM + kk[None, :] * SXK,
            (rows[:, None] < M) & (kk[None, :] < K),
            0,
        )
        b = gl.load(
            W + cols[:, None] * SWN + kk[None, :] * SWK,
            (cols[:, None] < N) & (kk[None, :] < K),
            0,
        )
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 8))
        b = gl.convert_layout(gl.permute(b, (1, 0)), gl.DotOperandLayout(1, mma, 8))
        acc = gl.amd.cdna4.mfma(a, b, acc)
    row = gl.program_id(1) * 16 + gl.arange(0, 16, gl.SliceLayout(1, mma))
    col = gl.program_id(0) * 64 + gl.arange(0, 64, gl.SliceLayout(0, mma))
    gl.store(
        Y + row[:, None] * SY + col[None, :],
        acc.to(gl.bfloat16),
        (row[:, None] < M) & (col[None, :] < N),
    )


@gluon.jit
def _m256_masked_dense_tile(
    X,
    W,
    Y,
    tile_m,
    tile_n,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SX: gl.constexpr,
    SW: gl.constexpr,
    RM: gl.constexpr,
    RN: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    NW: gl.constexpr,
    DEPTH: gl.constexpr,
    WT: gl.constexpr,
    SY: gl.constexpr,
):
    phase: gl.constexpr = BK // 8
    ca: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
    cb: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, NW], [0, 1])
    wm: gl.constexpr = 1 if NW == 1 or BM == 16 else 2
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [wm, NW // wm])
    da: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    db: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    sa = ()
    sb = ()
    for i in gl.static_range(DEPTH):
        sa += (
            gl.allocate_shared_memory(
                gl.bfloat16, [BM, BK], gl.SwizzledSharedLayout(8, 2, phase, [1, 0])
            ),
        )
        sb += (
            gl.allocate_shared_memory(
                gl.bfloat16, [BK, BN], gl.SwizzledSharedLayout(8, 2, phase, [0, 1])
            ),
        )
    ar = gl.arange(0, BM, gl.SliceLayout(1, ca))
    ak = gl.arange(0, BK, gl.SliceLayout(0, ca))
    bk = gl.arange(0, BK, gl.SliceLayout(1, cb))
    bc = gl.arange(0, BN, gl.SliceLayout(0, cb))
    ao = ar[:, None] * SX + (ak[None, :] ^ (ar[:, None] // 2 % phase * 8))
    bo = bc[None, :] * SW + (bk[:, None] ^ (bc[None, :] // 2 % phase * 8))
    ao = gl.max_contiguous(gl.multiple_of(ao, [1, 8]), [1, 8])
    bo = gl.max_contiguous(gl.multiple_of(bo, [8, 1]), [8, 1])
    if M % RM == 0:
        am = ar[:, None] < RM
    else:
        am = (ar[:, None] < RM) & (tile_m * RM + ar[:, None] < M)
    if N % RN == 0:
        bm = bc[None, :] < RN
    else:
        bm = (bc[None, :] < RN) & (tile_n * RN + bc[None, :] < N)
    X += tile_m * RM * SX
    W += tile_n * RN * SW
    for i in gl.static_range(DEPTH - 1):
        _m128_stage_masked_panel(sa[i], sb[i], X, W, ao, bo, am, bm, i * BK)
    acc = gl.zeros((BM, BN), gl.float32, mma)
    for step in gl.static_range(K // BK):
        if step < K // BK - DEPTH + 1:
            gl.amd.cdna4.async_copy.wait_group(DEPTH - 2)
            gl.barrier()
            _m128_stage_masked_panel(
                sa[(step + DEPTH - 1) % DEPTH],
                sb[(step + DEPTH - 1) % DEPTH],
                X,
                W,
                ao,
                bo,
                am,
                bm,
                (step + DEPTH - 1) * BK,
            )
        else:
            gl.amd.cdna4.async_copy.wait_group(K // BK - step - 1)
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(sa[step % DEPTH], da)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(sb[step % DEPTH], db)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    out_layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [NW, 1], [1, 0])
    out = gl.convert_layout(acc.to(gl.bfloat16), out_layout)
    r = gl.arange(0, BM, gl.SliceLayout(1, out_layout))
    c = gl.arange(0, BN, gl.SliceLayout(0, out_layout))
    offset = (tile_m * RM + r[:, None]) * SY + tile_n * RN + c[None, :]
    mask = (
        (r[:, None] < RM)
        & (tile_m * RM + r[:, None] < M)
        & (c[None, :] < RN)
        & (tile_n * RN + c[None, :] < N)
    )
    gl.amd.cdna4.buffer_store(out, Y, offset, mask, cache=".wt" if WT else "")


@gluon.jit
def _m256_stage_dense_panel(
    a_shared, b_shared, x_base, w_base, a_offsets, b_offsets, CACHE: gl.constexpr
):
    a_linear = a_shared.reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0]))
    b_linear = b_shared.reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
    gl.amd.cdna4.async_copy.buffer_load_to_shared(a_linear, x_base, a_offsets)
    gl.amd.cdna4.async_copy.buffer_load_to_shared(
        b_linear, w_base, b_offsets, cache_modifier=CACHE
    )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m256_consume_dense_panel(
    a_shared,
    b_shared,
    acc,
    DA: gl.constexpr,
    DB: gl.constexpr,
    BK: gl.constexpr,
    FRAGMENTS: gl.constexpr,
):
    if FRAGMENTS:
        for k in gl.static_range(0, BK, 32):
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_shared.slice(k, 32, 0), DB
            )
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_shared.slice(k, 32, 1), DA
            )
            acc = gl.amd.cdna4.mfma(a, b, acc)
    else:
        a = gl.amd.cdna4.async_copy.load_shared_relaxed(a_shared, DA)
        b = gl.amd.cdna4.async_copy.load_shared_relaxed(b_shared, DB)
        acc = gl.amd.cdna4.mfma(a, b, acc)
    return acc


@gluon.jit
def _m256_dense_tile(
    X,
    W,
    Y,
    M: gl.constexpr,
    N: gl.constexpr,
    K: gl.constexpr,
    SXM: gl.constexpr,
    SWN: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
    tile_m,
    tile_n,
    WT: gl.constexpr,
    SY: gl.constexpr,
    DEPTH: gl.constexpr = 4,
    BM: gl.constexpr = 64,
    WM: gl.constexpr = 2,
    BK: gl.constexpr = 64,
):
    gl.static_assert(M % BM == 0 and N % BN == 0 and (K % BK == 0))
    copy_a: gl.constexpr = gl.BlockedLayout(
        [1, 8], [512 // BK, BK // 8], [NW, 1], [1, 0]
    )
    copy_b: gl.constexpr = gl.BlockedLayout(
        [8, 1], [BK // 8, 512 // BK], [1, NW], [0, 1]
    )
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [WM, NW // WM])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 2, BK // 8, [1, 0])
    shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 2, BK // 8, [0, 1])
    ASYMMETRIC: gl.constexpr = M == 256 and BK == 128 and N >= 6144
    a_slots = ()
    b_slots = ()
    for slot in gl.static_range(DEPTH):
        if not ASYMMETRIC or slot < 2:
            a_slots += (gl.allocate_shared_memory(gl.bfloat16, [BM, BK], shared_a),)
        b_slots += (gl.allocate_shared_memory(gl.bfloat16, [BK, BN], shared_b),)
    rows = gl.arange(0, BM, gl.SliceLayout(1, copy_a))
    cols = gl.arange(0, BN, gl.SliceLayout(0, copy_b))
    a_k = gl.arange(0, BK, gl.SliceLayout(0, copy_a))
    b_k = gl.arange(0, BK, gl.SliceLayout(1, copy_b))
    a_offsets = rows[:, None] * SXM + (
        a_k[None, :] ^ rows[:, None] // 2 % (BK // 8) * 8
    )
    b_offsets = cols[None, :] * SWN + (
        b_k[:, None] ^ cols[None, :] // 2 % (BK // 8) * 8
    )
    a_offsets = gl.max_contiguous(gl.multiple_of(a_offsets, [1, 8]), [1, 8])
    b_offsets = gl.max_contiguous(gl.multiple_of(b_offsets, [8, 1]), [8, 1])
    x_base = X + tile_m * BM * SXM
    w_base = W + tile_n * BN * SWN
    if ASYMMETRIC:

        _m256_stage_dense_panel(
            a_slots[0], b_slots[0], x_base, w_base, a_offsets, b_offsets, ".cs"
        )
        b_next = b_slots[1].reinterpret(layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1]))
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            b_next, w_base + BK, b_offsets, cache_modifier=".cs"
        )
        gl.amd.cdna4.async_copy.commit_group()
        acc = gl.zeros((BM, BN), gl.float32, mma)
        for step in gl.static_range(K // BK):
            gl.amd.cdna4.async_copy.wait_group(1 if step < K // BK - 1 else 0)
            gl.barrier()
            for fragment in gl.static_range(0, BK, 32):
                b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    b_slots[step % 3].slice(fragment, 32, 0), dot_b
                )
                a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                    a_slots[step % 2].slice(fragment, 32, 1), dot_a
                )
                acc = gl.amd.cdna4.mfma(a, b, acc)
                if fragment == 32:
                    if step < K // BK - 1:
                        a_spare = a_slots[(step + 1) % 2].reinterpret(
                            layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
                        )
                        gl.amd.cdna4.async_copy.buffer_load_to_shared(
                            a_spare, x_base + (step + 1) * BK, a_offsets
                        )
                        gl.amd.cdna4.async_copy.commit_group()
                    if step < K // BK - 2:
                        b_spare = b_slots[(step + 2) % 3].reinterpret(
                            layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
                        )
                        gl.amd.cdna4.async_copy.buffer_load_to_shared(
                            b_spare,
                            w_base + (step + 2) * BK,
                            b_offsets,
                            cache_modifier=".cs",
                        )
                        gl.amd.cdna4.async_copy.commit_group()
    else:
        for slot in gl.static_range(DEPTH - 1):
            _m256_stage_dense_panel(
                a_slots[slot],
                b_slots[slot],
                x_base + slot * BK,
                w_base + slot * BK,
                a_offsets,
                b_offsets,
                ".cs" if N >= 6144 else "",
            )
        acc = gl.zeros((BM, BN), gl.float32, mma)
        for step in gl.static_range(K // BK - DEPTH + 1):
            gl.amd.cdna4.async_copy.wait_group(DEPTH - 2)
            gl.barrier()
            future = (step + DEPTH - 1) * BK
            _m256_stage_dense_panel(
                a_slots[(step + DEPTH - 1) % DEPTH],
                b_slots[(step + DEPTH - 1) % DEPTH],
                x_base + future,
                w_base + future,
                a_offsets,
                b_offsets,
                ".cs" if N >= 6144 else "",
            )
            acc = _m256_consume_dense_panel(
                a_slots[step % DEPTH],
                b_slots[step % DEPTH],
                acc,
                dot_a,
                dot_b,
                BK,
                BK == 128,
            )
        for tail in gl.static_range(DEPTH - 1):
            gl.amd.cdna4.async_copy.wait_group(DEPTH - 2 - tail)
            acc = _m256_consume_dense_panel(
                a_slots[(K // BK - DEPTH + 1 + tail) % DEPTH],
                b_slots[(K // BK - DEPTH + 1 + tail) % DEPTH],
                acc,
                dot_a,
                dot_b,
                BK,
                BK == 128,
            )
    store_layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [NW, 1], [1, 0])
    result = gl.convert_layout(acc.to(gl.bfloat16), store_layout)
    out_m = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, store_layout))
    out_n = tile_n * BN + gl.arange(0, BN, gl.SliceLayout(0, store_layout))
    gl.amd.cdna4.buffer_store(
        result,
        Y,
        out_m[:, None] * SY + out_n[None, :],
        cache=".cs" if M == 256 and N == 7168 else ".wt" if WT else "",
    )


@gluon.jit
def _m256_state_row_order(
    IDX,
    ORDER,
    M: gl.constexpr,
    SI: gl.constexpr,
    NW: gl.constexpr,
    BLOCK: gl.constexpr,
    LOG: gl.constexpr,
):
    row = gl.arange(0, BLOCK, gl.BlockedLayout([BLOCK // 64], [64], [NW], [0]))
    slot = gl.load(IDX + row * SI, row < M, 0).to(gl.uint32)
    slot = gl.where(row < M, slot, 0xFFFFFFFF)
    key = (slot.to(gl.uint64) << 32) | row.to(gl.uint64)
    for outer in gl.static_range(1, LOG + 1):
        for inner in gl.static_range(outer - 1, -1, -1):
            other = gl.gather(key, row ^ (1 << inner), 0)
            ascending = (row & (1 << outer)) == 0
            lower = (row & (1 << inner)) == 0
            take_min = ascending == lower
            key = gl.where(take_min, gl.minimum(key, other), gl.maximum(key, other))
    gl.store(ORDER + row, key, row < M)


_m256_project_qkvg_beta_repr = make_kernel_repr(
    "_m256_project_qkvg_beta",
    ["M", "BM", "BN", "NW", "GROUP", "SORT"],
)


@gluon.jit(repr=_m256_project_qkvg_beta_repr)
def _m256_project_qkvg_beta(
    X,
    WQ,
    WB,
    Q,
    B,
    M: gl.constexpr,
    SX: gl.constexpr,
    SQ: gl.constexpr,
    SB: gl.constexpr,
    BN: gl.constexpr,
    NW: gl.constexpr,
    GROUP: gl.constexpr,
    BR: gl.constexpr,
    BC: gl.constexpr,
    BD: gl.constexpr,
    WT: gl.constexpr,
    QS: gl.constexpr,
    BS: gl.constexpr,
    IDX,
    ORDER,
    SI: gl.constexpr,
    SORT: gl.constexpr,
    OB: gl.constexpr,
    LOG: gl.constexpr,
    BM: gl.constexpr = 64,
    WM: gl.constexpr = 2,
    QDEPTH: gl.constexpr = 4,
    QK: gl.constexpr = 64,
):
    qtiles: gl.constexpr = M // BM * (6144 // BN)
    pid = gl.program_id(0)
    if pid < qtiles:
        tile = pid % GROUP * (qtiles // GROUP) + pid // GROUP
        tile_m = tile // (6144 // BN)
        tile_n = tile % (6144 // BN)
        _m256_dense_tile(
            X,
            WQ,
            Q,
            M,
            6144,
            7168,
            SX,
            SQ,
            BN,
            NW,
            tile_m,
            tile_n,
            WT,
            QS,
            QDEPTH,
            BM,
            WM,
            QK,
        )
    elif pid < qtiles + gl.cdiv(M, 64) * gl.cdiv(144, BC):
        bpid = pid - qtiles
        tm = bpid // gl.cdiv(144, BC)
        tn = bpid % gl.cdiv(144, BC)
        if BC == 16:
            _m256_dense_tile(
                X, WB, B, M, 144, 7168, SX, SB, BC, NW, tm, tn, False, BS, BD
            )
        else:
            _m256_masked_dense_tile(
                X,
                WB,
                B,
                tm,
                tn,
                M,
                144,
                7168,
                SX,
                SB,
                BR,
                BC,
                BR,
                BC,
                64,
                NW,
                BD,
                False,
                BS,
            )
    elif SORT:
        _m256_state_row_order(IDX, ORDER, M, SI, NW, OB, LOG)


_m256_project_ragged_repr = make_kernel_repr(
    "_m256_project_ragged",
    ["M", "SORT"],
)


@gluon.jit(repr=_m256_project_ragged_repr)
def _m256_project_ragged(
    X,
    WQ,
    WB,
    Q,
    B,
    M: gl.constexpr,
    SX: gl.constexpr,
    SQ: gl.constexpr,
    SB: gl.constexpr,
    QS: gl.constexpr,
    BS: gl.constexpr,
    IDX,
    ORDER,
    SI: gl.constexpr,
    SORT: gl.constexpr,
    OB: gl.constexpr,
    LOG: gl.constexpr,
):
    qtiles: gl.constexpr = gl.cdiv(M, 64) * 96
    pid = gl.program_id(0)
    if pid < qtiles:
        tile = pid % 16 * (qtiles // 16) + pid // 16
        tm = tile // 96
        tn = tile % 96
        _m256_masked_dense_tile(
            X,
            WQ,
            Q,
            tm,
            tn,
            M,
            6144,
            7168,
            SX,
            SQ,
            64,
            64,
            64,
            64,
            64,
            4,
            4,
            M > 128,
            QS,
        )
    elif pid < qtiles + gl.cdiv(M, 64) * 9:
        tile = pid - qtiles
        _m256_masked_dense_tile(
            X,
            WB,
            B,
            tile // 9,
            tile % 9,
            M,
            144,
            7168,
            SX,
            SB,
            64,
            16,
            64,
            16,
            64,
            4,
            5,
            False,
            BS,
        )
    elif SORT:
        _m256_state_row_order(IDX, ORDER, M, SI, 4, OB, LOG)


_m256_output_ragged_repr = make_kernel_repr(
    "_m256_output_ragged",
    ["M"],
)


@gluon.jit(repr=_m256_output_ragged_repr)
def _m256_output_ragged(
    X, W, Y, M: gl.constexpr, SX: gl.constexpr, SW: gl.constexpr, SY: gl.constexpr
):
    tile = gl.program_id(0)
    _m256_masked_dense_tile(
        X,
        W,
        Y,
        tile // 112,
        tile % 112,
        M,
        7168,
        1536,
        SX,
        SW,
        64,
        64,
        64,
        64,
        64,
        4,
        4,
        True,
        SY,
    )


@gluon.jit
def _m256_stage_output_a(a_slots, X, offsets, PANEL: gl.constexpr):

    for segment in gl.static_range(3):
        linear = a_slots[PANEL % 2 * 3 + segment].reinterpret(
            layout=gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            linear, X + PANEL * 192, offsets + segment * 64
        )
    gl.amd.cdna4.async_copy.commit_group()


@gluon.jit
def _m256_stage_output_b(b_slots, W, offsets, PANEL: gl.constexpr):
    for segment in gl.static_range(3):
        linear = b_slots[PANEL % 2 * 3 + segment].reinterpret(
            layout=gl.SwizzledSharedLayout(1, 1, 1, [0, 1])
        )
        gl.amd.cdna4.async_copy.buffer_load_to_shared(
            linear, W + PANEL * 192, offsets + segment * 64, cache_modifier=".cs"
        )
    gl.amd.cdna4.async_copy.commit_group()


_m256_output_m128_repr = make_kernel_repr(
    "_m256_output_m128",
    [],
)


@gluon.jit(repr=_m256_output_m128_repr)
def _m256_output_m128(X, W, Y, SX: gl.constexpr, SW: gl.constexpr, SY: gl.constexpr):
    pid = gl.program_id(0).to(gl.uint32)
    tile_m = pid >> 3 & 1
    tile_n = (pid & 7) * 14 + pid // 16
    copy_a: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [4, 1], [1, 0])
    copy_b: gl.constexpr = gl.BlockedLayout([8, 1], [8, 8], [1, 4], [0, 1])
    mma: gl.constexpr = gl.amd.AMDMFMALayout(4, [16, 16, 32], True, [2, 2])
    dot_a: gl.constexpr = gl.DotOperandLayout(0, mma, 8)
    dot_b: gl.constexpr = gl.DotOperandLayout(1, mma, 8)
    a_slots = ()
    b_slots = ()
    for segment in gl.static_range(6):
        a_slots += (
            gl.allocate_shared_memory(
                gl.bfloat16, [64, 64], gl.SwizzledSharedLayout(8, 2, 8, [1, 0])
            ),
        )
    for segment in gl.static_range(6):
        b_slots += (
            gl.allocate_shared_memory(
                gl.bfloat16, [64, 64], gl.SwizzledSharedLayout(8, 2, 8, [0, 1])
            ),
        )
    rows = gl.arange(0, 64, gl.SliceLayout(1, copy_a))
    cols = gl.arange(0, 64, gl.SliceLayout(0, copy_b))
    ak = gl.arange(0, 64, gl.SliceLayout(0, copy_a))
    bk = gl.arange(0, 64, gl.SliceLayout(1, copy_b))
    ao = rows[:, None] * SX + (ak[None, :] ^ rows[:, None] // 2 % 8 * 8)
    bo = cols[None, :] * SW + (bk[:, None] ^ cols[None, :] // 2 % 8 * 8)
    ao = gl.max_contiguous(gl.multiple_of(ao, [1, 8]), [1, 8])
    bo = gl.max_contiguous(gl.multiple_of(bo, [8, 1]), [8, 1])
    X += tile_m * 64 * SX
    W += tile_n * 64 * SW
    _m256_stage_output_a(a_slots, X, ao, 0)
    _m256_stage_output_b(b_slots, W, bo, 0)
    _m256_stage_output_b(b_slots, W, bo, 1)
    acc = gl.zeros((64, 64), gl.float32, mma)
    for panel in gl.static_range(8):
        gl.amd.cdna4.async_copy.wait_group(1 if panel < 7 else 0)
        gl.barrier()
        for segment in gl.static_range(3):
            b = gl.amd.cdna4.async_copy.load_shared_relaxed(
                b_slots[panel % 2 * 3 + segment], dot_b
            )
            a = gl.amd.cdna4.async_copy.load_shared_relaxed(
                a_slots[panel % 2 * 3 + segment], dot_a
            )
            acc = gl.amd.cdna4.mfma(a, b, acc)
            if segment == 0 and panel < 7:

                _m256_stage_output_a(a_slots, X, ao, panel + 1)
        if panel < 7:

            acc = gl.inline_asm_elementwise(
                "", "=v,0,~{memory}", [acc], dtype=gl.float32, is_pure=False, pack=1
            )
            gl.inline_asm_elementwise(
                "s_waitcnt lgkmcnt(0)\n v_mov_b32 $0, 0",
                "=v,~{memory}",
                [],
                dtype=gl.int32,
                is_pure=False,
                pack=1,
            )
            gl.barrier()
            if panel < 6:
                _m256_stage_output_b(b_slots, W, bo, panel + 2)
    result = gl.convert_layout(acc.to(gl.bfloat16), copy_a)
    out_m = gl.arange(0, 64, gl.SliceLayout(1, copy_a))
    out_n = gl.arange(0, 64, gl.SliceLayout(0, copy_a))
    y_base = Y + tile_m * 64 * SY + tile_n * 64
    gl.amd.cdna4.buffer_store(
        result, y_base, out_m[:, None] * SY + out_n[None, :], cache=".wt"
    )


_m256_output_m256_repr = make_kernel_repr(
    "_m256_output_m256",
    ["M"],
)


@gluon.jit(repr=_m256_output_m256_repr)
def _m256_output_m256(
    X, W, Y, M: gl.constexpr, SX: gl.constexpr, SW: gl.constexpr, SY: gl.constexpr
):
    pid = gl.program_id(0).to(gl.uint32)
    tile_m = pid >> 3 & 1
    tile_n = (pid & 7) * 14 + pid // 16
    _m256_dense_tile(
        X, W, Y, M, 7168, 1536, SX, SW, 64, 4, tile_m, tile_n, True, SY, 3, 128, 2, 128
    )


@gluon.jit
def _m256_bf16_pair_sum(w0, w1, x0, x1, NW: gl.constexpr):
    rows: gl.constexpr = w0.shape[0]
    layout: gl.constexpr = gl.BlockedLayout(
        [1, 1, 1], [16, 4, 1], [1, NW, 1], [0, 1, 2]
    )
    weights = gl.permute(gl.join(w0, w1), (1, 0, 2))
    values = gl.permute(gl.join(x0, x1), (1, 2, 0))
    weights = gl.convert_layout(weights, gl.DotOperandLayout(0, layout, 0))
    values = gl.convert_layout(values, gl.DotOperandLayout(1, layout, 0))
    dots = gl.dot_fma(weights, values, gl.zeros((16, rows, 1), gl.float32, layout))
    return gl.permute(gl.reshape(dots, (16, rows)), (1, 0))


@gluon.jit
def _m256_project_forget_head(
    FA,
    FW,
    m,
    h,
    SFA: gl.constexpr,
    SW: gl.constexpr,
    ROWS: gl.constexpr,
    OFFSET: gl.constexpr,
    NW: gl.constexpr,
):
    pl: gl.constexpr = gl.BlockedLayout([1, 8], [4, 16], [NW, 1], [1, 0])
    n = gl.arange(0, ROWS, gl.SliceLayout(1, pl)) + OFFSET
    k = gl.arange(0, 128, gl.SliceLayout(0, pl))
    x = gl.amd.cdna4.buffer_load(FA + m * SFA, k)
    w = gl.amd.cdna4.buffer_load(FW + h * 128 * SW, n[:, None] * SW + k[None, :])
    w0, w1, w2, w3, w4, w5, w6, w7 = _m128_eight_parts(w)
    x0, x1, x2, x3, x4, x5, x6, x7 = _m128_eight_parts(x[None, :])
    even = _m256_bf16_pair_sum(w0, w2, x0, x2, NW) + _m256_bf16_pair_sum(
        w4, w6, x4, x6, NW
    )
    odd = _m256_bf16_pair_sum(w1, w3, x1, x3, NW) + _m256_bf16_pair_sum(
        w5, w7, x5, x7, NW
    )
    partial = even + odd
    return gl.sum(partial, 1).to(gl.bfloat16)


@gluon.jit
def _m256_convolve_head(
    X,
    CW,
    CS,
    m,
    h,
    slot,
    H: gl.constexpr,
    SX: gl.constexpr,
    SC0: gl.constexpr,
    SC1: gl.constexpr,
    SC2: gl.constexpr,
    LAYOUT: gl.constexpr,
    PREP,
    NW: gl.constexpr,
):
    cl: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [NW // 2, 2], [1, 0])
    plane = gl.arange(0, 4, gl.SliceLayout(1, cl))
    channel = gl.arange(0, 128, gl.SliceLayout(0, cl))
    c = plane[:, None] * H * 128 + h * 128 + channel[None, :]
    valid = plane[:, None] < 3
    history_base = CS + slot * SC0 + h * 128 * SC2
    history_offsets = (plane[:, None] * H * 128 + channel[None, :]) * SC2
    old0 = gl.amd.cdna4.buffer_load(history_base, history_offsets, valid, 0)
    old1 = gl.amd.cdna4.buffer_load(history_base, SC1 + history_offsets, valid, 0)
    old2 = gl.amd.cdna4.buffer_load(history_base, 2 * SC1 + history_offsets, valid, 0)
    x = gl.load(X + m * SX + c, valid, 0)
    wl: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 64, 1], [NW // 2, 2, 1], [2, 1, 0]
    )
    wp = gl.arange(0, 4, gl.SliceLayout(1, gl.SliceLayout(2, wl)))
    wc = gl.arange(0, 128, gl.SliceLayout(0, gl.SliceLayout(2, wl)))
    taps = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    weight_c = wp[:, None] * H * 128 + h * 128 + wc[None, :]
    weights = gl.load(
        CW + weight_c[:, :, None] * 4 + taps[None, None, :],
        weight_c[:, :, None] < 3 * H * 128,
        0,
    )
    even, odd = gl.split(gl.reshape(weights, (4, 128, 2, 2)))
    w0, w2 = gl.split(even)
    w1, w3 = gl.split(odd)
    w0 = gl.convert_layout(w0, cl)
    w1 = gl.convert_layout(w1, cl)
    w2 = gl.convert_layout(w2, cl)
    w3 = gl.convert_layout(w3, cl)
    z = (
        old0.to(gl.float32) * w0
        + old1.to(gl.float32) * w1
        + old2.to(gl.float32) * w2
        + x.to(gl.float32) * w3
    )
    z = (z * _m1_sigmoid(z)).to(gl.bfloat16)
    changed0 = old0.to(gl.uint16, bitcast=True) != old1.to(gl.uint16, bitcast=True)
    changed1 = old1.to(gl.uint16, bitcast=True) != old2.to(gl.uint16, bitcast=True)
    changed2 = old2.to(gl.uint16, bitcast=True) != x.to(gl.uint16, bitcast=True)
    gl.amd.cdna4.buffer_store(old1, history_base, history_offsets, valid & changed0)
    gl.amd.cdna4.buffer_store(
        old2, history_base, SC1 + history_offsets, valid & changed1
    )
    gl.amd.cdna4.buffer_store(
        x, history_base, 2 * SC1 + history_offsets, valid & changed2
    )
    handoff = PREP.slice(0, 512)
    handoff.store(gl.reshape(z, (512,)))
    q = handoff.slice(0, 128).load(gl.SliceLayout(0, LAYOUT)).to(gl.float32)
    key = handoff.slice(128, 128).load(gl.SliceLayout(0, LAYOUT)).to(gl.float32)
    return (q, key)


@gluon.jit
def _m256_or_reduce(a, b):
    return a | b


@gluon.jit
def _m256_store_changed_vectors(
    updated,
    previous,
    base,
    offsets,
    WT: gl.constexpr,
    LINE_VECTORS: gl.constexpr,
    STREAM: gl.constexpr = False,
):

    changed_bits = updated.to(gl.uint32, bitcast=True) ^ previous.to(
        gl.uint32, bitcast=True
    )
    rows: gl.constexpr = updated.shape[0]
    even, odd = gl.split(gl.reshape(changed_bits, (rows, 32, 2, 2)))
    b0, b2 = gl.split(even)
    b1, b3 = gl.split(odd)
    diff = gl.inline_asm_elementwise(
        "v_or3_b32 $0, $1, $2, $3\nv_or_b32 $0, $0, $4",
        constraints="=&v,v,v,v,v",
        args=[b0, b1, b2, b3],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )
    if LINE_VECTORS > 1:
        grouped = gl.reshape(diff, (rows, 32 // LINE_VECTORS, LINE_VECTORS))
        line_diff = gl.reduce(grouped, 2, _m256_or_reduce)
        line_mask, _ = gl.broadcast(line_diff[:, :, None], grouped)
        diff = gl.convert_layout(
            gl.reshape(line_mask, (rows, 32)), b0.type.layout, assert_trivial=True
        )
    mask, _ = gl.broadcast(
        (diff != 0)[:, :, None, None], gl.reshape(changed_bits, (rows, 32, 2, 2))
    )
    store_mask = gl.reshape(mask, (rows, 128))
    gl.amd.cdna4.buffer_store(
        updated,
        base,
        offsets,
        store_mask,
        cache=".cs" if STREAM else ".wt" if WT else "",
    )


_m256_kda_decode_repr = make_kernel_repr(
    "_m256_kda_decode",
    ["H", "GROUP", "NW", "LARGE_BATCH", "SORT", "STATE_STREAM"],
)


@gluon.jit(repr=_m256_kda_decode_repr)
def _m256_kda_decode(
    X,
    Gate,
    FA,
    Beta,
    FW,
    CW,
    A,
    DT,
    W,
    CS,
    S,
    IDX,
    OUT,
    H: gl.constexpr,
    SX: gl.constexpr,
    SG: gl.constexpr,
    SFA: gl.constexpr,
    SB: gl.constexpr,
    SW: gl.constexpr,
    SC0: gl.constexpr,
    SC1: gl.constexpr,
    SC2: gl.constexpr,
    SS: gl.constexpr,
    SI: gl.constexpr,
    lower,
    eps,
    GROUP: gl.constexpr,
    LARGE_BATCH: gl.constexpr,
    NW: gl.constexpr,
    WT: gl.constexpr,
    SO: gl.constexpr,
    ORDER,
    SORT: gl.constexpr,
    STATE_LINE_VECTORS: gl.constexpr,
    STATE_STREAM: gl.constexpr,
):
    state_cache: gl.constexpr = ".cg"
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 16], [NW, 1], [1, 0])
    pid = gl.program_id(0)
    if not LARGE_BATCH:
        pid = pid.to(gl.uint32)
    m = gl.program_id(1) * GROUP + pid % GROUP
    h = pid // GROUP
    if LARGE_BATCH:
        h = h ^ (gl.program_id(1) & 3)
    else:
        group_id = gl.program_id(1)
        h = h ^ ((group_id ^ (group_id >> 2)) & 3)
    if SORT:
        record = gl.load(ORDER + m)
        m = record.to(gl.int32)
        if ORDER.dtype.element_ty == gl.int64:
            slot = (record.to(gl.uint64) >> 32).to(gl.int32).to(gl.int64)
        else:
            slot = gl.load(IDX + m * SI).to(gl.int64)
    else:
        slot = gl.load(IDX + m * SI).to(gl.int64)
    v = gl.arange(0, 128, gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, gl.SliceLayout(0, layout))
    if slot < 0:
        gl.store(OUT + m * SO + h * 128 + v, 0)
    else:
        prep = gl.allocate_shared_memory(
            gl.bfloat16, (512,), gl.SwizzledSharedLayout(1, 1, 1, [0])
        )
        auxiliary = gl.allocate_shared_memory(
            gl.bfloat16, (256,), gl.SwizzledSharedLayout(1, 1, 1, [0])
        )
        if not LARGE_BATCH:
            projection = gl.allocate_shared_memory(
                gl.bfloat16, (128,), gl.SwizzledSharedLayout(1, 1, 1, [0])
            )
        if LARGE_BATCH:
            for offset in gl.static_range(0, 128, 32):
                projected_part = _m256_project_forget_head(
                    FA, FW, m, h, SFA, SW, 32, offset, NW
                )
                auxiliary.slice(offset, 32).store(projected_part)
        else:
            projected = _m256_project_forget_head(FA, FW, m, h, SFA, SW, 128, 0, NW)
            projection = projection.reshape((32, 4)).permute((1, 0)).reshape((128,))
            projection.store(projected)
        if LARGE_BATCH:
            base = S + slot * SS + h * 128 * 128
        else:
            base = S + slot * SS
        slab_v = gl.arange(0, 32, gl.SliceLayout(1, layout))
        if LARGE_BATCH:
            slab_offsets = slab_v[:, None] * 128 + k[None, :]
        else:
            slab_offsets = (h * 128 + slab_v[:, None]) * 128 + k[None, :]
        row0: gl.constexpr = 96 if LARGE_BATCH else 0
        row1: gl.constexpr = 32
        row2: gl.constexpr = 64
        row3: gl.constexpr = 0 if LARGE_BATCH else 96
        old0 = gl.amd.cdna4.buffer_load(
            base, slab_offsets + row0 * 128, cache=state_cache
        )
        if LARGE_BATCH:
            old1 = gl.amd.cdna4.buffer_load(
                base, slab_offsets + row1 * 128, cache=state_cache
            )
        q, key = _m256_convolve_head(
            X, CW, CS, m, h, slot, H, SX, SC0, SC1, SC2, layout, prep, NW
        )
        if not LARGE_BATCH:
            old1 = gl.amd.cdna4.buffer_load(
                base, slab_offsets + row1 * 128, cache=state_cache
            )
        decay_layout: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
        dk = gl.arange(0, 128, decay_layout)
        if LARGE_BATCH:
            projected = auxiliary.slice(0, 128).load(decay_layout).to(gl.float32)
        else:
            projected = projection.load(decay_layout).to(gl.float32)
        raw = projected + gl.load(DT + h * 128 + dk)
        decay = gl.exp(lower * _m1_sigmoid(gl.exp(gl.load(A + h)) * raw))
        decay = gl.convert_layout(decay, gl.SliceLayout(0, layout))
        q = q * gl.rsqrt(gl.sum(q * q, 0) + 1e-06) * 128 ** (-0.5)
        key = key * gl.rsqrt(gl.sum(key * key, 0) + 1e-06)
        beta = _m1_sigmoid(gl.load(Beta + m * SB + h).to(gl.float32))
        norm_layout: gl.constexpr = gl.BlockedLayout([2], [64], [NW], [0])
        nv = gl.arange(0, 128, norm_layout)
        weight = gl.amd.cdna4.buffer_load(W, nv).to(gl.float32)
        gate = gl.amd.cdna4.buffer_load(Gate + m * SG + h * 128, nv).to(gl.float32)
        value0 = (
            prep.slice(256 + row0, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
        )
        updated0 = old0 * decay[None, :]
        delta0 = (value0 - gl.sum(updated0 * key[None, :], 1)) * beta
        updated0 = gl.fma(delta0[:, None], key[None, :], updated0)
        if LARGE_BATCH:
            _m256_store_changed_vectors(
                updated0,
                old0,
                base,
                slab_offsets + row0 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM,
            )
        core0 = gl.sum(updated0 * q[None, :], 1)
        if not LARGE_BATCH:
            _m256_store_changed_vectors(
                updated0,
                old0,
                base,
                slab_offsets + row0 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM,
            )
        if LARGE_BATCH:
            auxiliary.slice(row0, 32).store(core0.to(gl.bfloat16))
        value1 = (
            prep.slice(256 + row1, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
        )
        updated1 = old1 * decay[None, :]
        delta1 = (value1 - gl.sum(updated1 * key[None, :], 1)) * beta
        updated1 = gl.fma(delta1[:, None], key[None, :], updated1)
        if LARGE_BATCH:
            _m256_store_changed_vectors(
                updated1,
                old1,
                base,
                slab_offsets + row1 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM,
            )
        core1 = gl.sum(updated1 * q[None, :], 1)
        if not LARGE_BATCH:
            _m256_store_changed_vectors(
                updated1,
                old1,
                base,
                slab_offsets + row1 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM,
            )
        if LARGE_BATCH:
            auxiliary.slice(row1, 32).store(core1.to(gl.bfloat16))
        if not LARGE_BATCH:
            pair0 = gl.reshape(gl.permute(gl.join(core0, core1), (1, 0)), (64,))
            auxiliary.slice(0, 64).store(pair0.to(gl.bfloat16))
        core1 = gl.inline_asm_elementwise(
            "", "=v,0,~{memory}", [core1], dtype=gl.float32, is_pure=False, pack=1
        )

        old2 = gl.amd.cdna4.buffer_load(
            base, slab_offsets + row2 * 128, cache=state_cache
        )
        old3 = gl.amd.cdna4.buffer_load(
            base, slab_offsets + row3 * 128, cache=state_cache
        )
        value2 = (
            prep.slice(256 + row2, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
        )
        updated2 = old2 * decay[None, :]
        delta2 = (value2 - gl.sum(updated2 * key[None, :], 1)) * beta
        updated2 = gl.fma(delta2[:, None], key[None, :], updated2)
        if LARGE_BATCH:
            _m256_store_changed_vectors(
                updated2,
                old2,
                base,
                slab_offsets + row2 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM,
            )
        core2 = gl.sum(updated2 * q[None, :], 1)
        if not LARGE_BATCH:
            _m256_store_changed_vectors(
                updated2,
                old2,
                base,
                slab_offsets + row2 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM,
            )
        if LARGE_BATCH:
            auxiliary.slice(row2, 32).store(core2.to(gl.bfloat16))
        if not LARGE_BATCH:
            core2 = gl.inline_asm_elementwise(
                "", "=v,0,~{memory}", [core2], dtype=gl.float32, is_pure=False, pack=1
            )
        value3 = (
            prep.slice(256 + row3, 32).load(gl.SliceLayout(1, layout)).to(gl.float32)
        )
        updated3 = old3 * decay[None, :]
        delta3 = (value3 - gl.sum(updated3 * key[None, :], 1)) * beta
        updated3 = gl.fma(delta3[:, None], key[None, :], updated3)
        if LARGE_BATCH:
            _m256_store_changed_vectors(
                updated3,
                old3,
                base,
                slab_offsets + row3 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM,
            )
        core3 = gl.sum(updated3 * q[None, :], 1)
        if not LARGE_BATCH:
            _m256_store_changed_vectors(
                updated3,
                old3,
                base,
                slab_offsets + row3 * 128,
                WT,
                STATE_LINE_VECTORS,
                STATE_STREAM,
            )
        if LARGE_BATCH:
            auxiliary.slice(row3, 32).store(core3.to(gl.bfloat16))
        if not LARGE_BATCH:
            pair1 = gl.reshape(gl.permute(gl.join(core2, core3), (1, 0)), (64,))
            auxiliary.slice(64, 64).store(pair1.to(gl.bfloat16))
        core = auxiliary.slice(0, 128).load(norm_layout).to(gl.float32)
        out = (
            core
            * gl.rsqrt(gl.sum(core * core, 0) / 128 + eps)
            * weight
            * _m1_sigmoid(gate)
        )
        gl.store(OUT + m * SO + h * 128 + nv, out)


@gluon.jit
def _m32_convolve(
    X,
    CW,
    CS,
    row,
    head,
    slot,
    SC0: gl.constexpr,
    SC1: gl.constexpr,
    SC2: gl.constexpr,
    NW: gl.constexpr,
):
    conv_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, NW // 4], [1, 0])
    group = gl.arange(0, 4, gl.SliceLayout(1, conv_layout))
    channel = gl.arange(0, 128, gl.SliceLayout(0, conv_layout))
    offset = (group[:, None] * 12 + head) * 128 + channel[None, :]
    valid = group[:, None] < 3
    history = CS + slot * SC0
    old0 = gl.amd.cdna4.buffer_load(history, offset * SC2, valid, 0).to(gl.float32)
    old1 = gl.amd.cdna4.buffer_load(history, SC1 + offset * SC2, valid, 0)
    old2 = gl.amd.cdna4.buffer_load(history, 2 * SC1 + offset * SC2, valid, 0)
    x = gl.load(X + row * 6336 + offset)
    wl: gl.constexpr = gl.BlockedLayout(
        [1, 1, 4], [1, 64, 1], [4, NW // 4, 1], [2, 1, 0]
    )
    wo = gl.convert_layout(offset, gl.SliceLayout(2, wl), assert_trivial=True)
    wvalid = gl.convert_layout(valid, gl.SliceLayout(2, wl), assert_trivial=True)
    tap = gl.arange(0, 4, gl.SliceLayout(0, gl.SliceLayout(1, wl)))
    weights = gl.amd.cdna4.buffer_load(
        CW, wo[:, :, None] * 4 + tap[None, None, :], wvalid[:, :, None], 0
    )
    even, odd = gl.split(gl.reshape(weights, (4, 128, 2, 2)))
    w0, w2 = gl.split(even)
    w1, w3 = gl.split(odd)
    w0 = gl.convert_layout(w0, conv_layout)
    w1 = gl.convert_layout(w1, conv_layout)
    w2 = gl.convert_layout(w2, conv_layout)
    w3 = gl.convert_layout(w3, conv_layout)
    z = (
        old0 * w0
        + old1.to(gl.float32) * w1
        + old2.to(gl.float32) * w2
        + x.to(gl.float32) * w3
    )
    z = gl.where(group[:, None] == 3, x.to(gl.float32), z)
    sigmoid = _m1_sigmoid(z)
    z = gl.where(
        group[:, None] == 3, sigmoid, (z * sigmoid).to(gl.bfloat16).to(gl.float32)
    )
    return z, (old1, old2, x, history, offset * SC2, valid)


@gluon.jit
def _m32_publish_decay(
    X,
    FW,
    A,
    DT,
    shared,
    m,
    h,
    LOWER,
    SW: gl.constexpr,
    NW: gl.constexpr,
    STAGED: gl.constexpr,
):

    width: gl.constexpr = min(8, SW & -SW)
    lanes_k: gl.constexpr = min(64, 128 // width)
    waves_k: gl.constexpr = 128 // (width * lanes_k)
    layout: gl.constexpr = gl.BlockedLayout(
        [1, width], [64 // lanes_k, lanes_k], [NW // waves_k, waves_k], [1, 0]
    )
    rows = gl.arange(0, 32, gl.SliceLayout(1, layout))
    k = gl.arange(0, 128, gl.SliceLayout(0, layout))
    a = gl.amd.cdna4.buffer_load(X + m * 6336 + 6144, k).to(gl.float32)
    if not STAGED:
        rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
    current = gl.amd.cdna4.buffer_load(
        FW + h * 128 * SW, rows[:, None] * SW + k[None, :]
    ).to(gl.float32)
    paired: gl.constexpr = SW % 8 == 0
    if STAGED:
        staging = gl.allocate_shared_memory(
            gl.bfloat16, (128,), gl.SwizzledSharedLayout(1, 1, 1, [0])
        )
    if not STAGED:
        contiguous: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
        channels = gl.arange(0, 64 if paired else 32, contiguous)
        bias = gl.load(DT + h * 128 + channels)
    for tile in gl.static_range(4):
        with gl.amd.warp_pipeline_stage("forget_decay"):
            if tile < 3:
                following = gl.amd.cdna4.buffer_load(
                    FW + (h * 128 + (tile + 1) * 32) * SW,
                    rows[:, None] * SW + k[None, :],
                ).to(gl.float32)
            if not STAGED:
                if paired and tile == 1:
                    next_bias = gl.load(DT + h * 128 + 64 + channels)
                elif not paired and tile < 3:
                    next_bias = gl.load(DT + h * 128 + (tile + 1) * 32 + channels)
            f = gl.sum(current * a[None, :], 1).to(gl.bfloat16)
            if STAGED:
                staging.slice(tile * 32, 32).store(f)
            elif paired:
                if tile % 2 == 0:
                    first_half = f
                else:
                    joined = gl.reshape(
                        gl.permute(gl.join(first_half, f), (1, 0)), (64,)
                    )
                    pair = gl.convert_layout(joined, contiguous).to(gl.float32)
                    decay = gl.exp2(
                        LOWER * _m1_sigmoid(rate * (pair + bias)) * 1.4426950408889634
                    )
                    shared.slice(4, 1).slice((tile - 1) * 32, 64, dim=1).store(
                        gl.reshape(decay, (1, 64))
                    )
            else:
                panel = gl.convert_layout(f.to(gl.float32), contiguous)
                decay = gl.exp2(
                    LOWER * _m1_sigmoid(rate * (panel + bias)) * 1.4426950408889634
                )
                shared.slice(4, 1).slice(tile * 32, 32, dim=1).store(
                    gl.reshape(decay, (1, 32))
                )
            if tile < 3:
                current = following
                if not STAGED and ((paired and tile == 1) or not paired):
                    bias = next_bias
    if STAGED:
        rate = gl.exp2(gl.load(A + h) * 1.4426950408889634)
        vl: gl.constexpr = gl.BlockedLayout([1], [64], [NW], [0])
        channel = gl.arange(0, 128, vl)
        dt = gl.load(DT + h * 128 + channel)
        f = staging.load(vl).to(gl.float32)
        decay = gl.exp2(LOWER * _m1_sigmoid(rate * (f + dt)) * 1.4426950408889634)
        shared.slice(4, 1).store(gl.reshape(decay, (1, 128)))


@gluon.jit
def _m32_project_stream(
    X,
    weight,
    ao,
    bo,
    amask,
    acc,
    sa,
    sb,
    K: gl.constexpr,
    BK: gl.constexpr,
    MMA: gl.constexpr,
    AD: gl.constexpr,
    BD: gl.constexpr,
    FRAG: gl.constexpr,
    COUPLED: gl.constexpr,
    FIXED_WEIGHT_BASE: gl.constexpr = False,
):

    aa = ()
    bb = ()
    for block in gl.static_range(AD):
        aa += (gl.amd.cdna4.buffer_load(X + block * BK, ao, amask, 0),)
    for block in gl.static_range(BD):
        if FIXED_WEIGHT_BASE:
            bb += (gl.amd.cdna4.buffer_load(weight, bo + block * BK, cache=".cg"),)
        else:
            bb += (gl.amd.cdna4.buffer_load(weight + block * BK, bo, cache=".cg"),)
    for block in gl.static_range(K // BK):
        a = aa[0]
        b = bb[0]
        if not COUPLED:
            if block + BD < K // BK:
                if FIXED_WEIGHT_BASE:
                    bb = bb[1:] + (
                        gl.amd.cdna4.buffer_load(
                            weight, bo + (block + BD) * BK, cache=".cg"
                        ),
                    )
                else:
                    bb = bb[1:] + (
                        gl.amd.cdna4.buffer_load(
                            weight + (block + BD) * BK, bo, cache=".cg"
                        ),
                    )
            else:
                bb = bb[1:]
        sa.store(a)
        sb.store(b)
        if COUPLED:
            if block + BD < K // BK:
                if FIXED_WEIGHT_BASE:
                    bb = bb[1:] + (
                        gl.amd.cdna4.buffer_load(
                            weight, bo + (block + BD) * BK, cache=".cg"
                        ),
                    )
                else:
                    bb = bb[1:] + (
                        gl.amd.cdna4.buffer_load(
                            weight + (block + BD) * BK, bo, cache=".cg"
                        ),
                    )
            else:
                bb = bb[1:]
        else:
            if block + AD < K // BK:
                aa = aa[1:] + (
                    gl.amd.cdna4.buffer_load(X + (block + AD) * BK, ao, amask, 0),
                )
            else:
                aa = aa[1:]
        for fragment in gl.static_range(BK // FRAG):
            a = sa.slice(fragment * FRAG, FRAG, dim=1).load(
                gl.DotOperandLayout(0, MMA, 8)
            )
            b = sb.slice(fragment * FRAG, FRAG, dim=0).load(
                gl.DotOperandLayout(1, MMA, 8)
            )
            if COUPLED and fragment == 0:
                if block + AD < K // BK:
                    aa = aa[1:] + (
                        gl.amd.cdna4.buffer_load(X + (block + AD) * BK, ao, amask, 0),
                    )
                else:
                    aa = aa[1:]
            acc = gl.amd.cdna4.mfma(a, b, acc)
    return acc


_m32_projection_repr = make_kernel_repr(
    "_m32_projection",
    ["M", "K", "BM", "NW", "BK", "INPUT"],
)


@gluon.jit(repr=_m32_projection_repr)
def _m32_projection(
    X,
    WQ,
    WB,
    Y,
    M: gl.constexpr,
    K: gl.constexpr,
    XM: gl.constexpr,
    WN: gl.constexpr,
    WBN: gl.constexpr,
    YS: gl.constexpr,
    INPUT: gl.constexpr,
    BM: gl.constexpr,
    NW: gl.constexpr,
    BK: gl.constexpr,
):
    BN: gl.constexpr = 32
    MT: gl.constexpr = triton.cdiv(M, BM)
    pid = gl.program_id(0)
    if M <= 32:
        pid = pid.to(gl.uint32)
    tile_n = pid // MT
    tile_m = pid % MT
    if M == 64:
        tile_n = (tile_n * (149 if INPUT else 11)) % (197 if INPUT else 224)
    if INPUT:
        auxiliary = tile_n >= 6144 // BN
        weight = gl.where(auxiliary, WB, WQ)
        stride = gl.where(auxiliary, WBN, WN)
        first_n = gl.where(auxiliary, tile_n * BN - 6144, tile_n * BN)
    else:
        weight = WQ
        stride = WN
        first_n = tile_n * BN
    mma: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4,
        instr_shape=[16, 16, 32],
        transposed=False,
        warps_per_cta=[BM // 16, NW // (BM // 16)],
    )
    la: gl.constexpr = gl.BlockedLayout([1, 8], [512 // BK, BK // 8], [NW, 1], [1, 0])
    lb: gl.constexpr = gl.BlockedLayout([8, 1], [BK // 8, 512 // BK], [1, NW], [0, 1])
    am = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, la))
    ak = gl.arange(0, BK, gl.SliceLayout(0, la))
    bk = gl.arange(0, BK, gl.SliceLayout(1, lb))
    bn = first_n + gl.arange(0, BN, gl.SliceLayout(0, lb))
    if INPUT:
        bn = gl.where(auxiliary & (bn >= 144), 0, bn)
    ao = am[:, None] * XM + ak[None, :]
    bo = bn[None, :] * stride + bk[:, None]
    amask = am[:, None] < M
    acc = gl.zeros((BM, BN), gl.float32, mma)
    if INPUT and BM == 32:
        shared_a: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [1, 0])
        shared_b: gl.constexpr = gl.SwizzledSharedLayout(8, 1, 16, [0, 1])
    else:
        shared_a: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
            gl.DotOperandLayout(0, mma, 8), (BM, BK), gl.bfloat16
        )
        shared_b: gl.constexpr = gl.amd.cdna4.compute_efficient_padded_shared_layout(
            gl.DotOperandLayout(1, mma, 8), (BK, BN), gl.bfloat16
        )
    sa = gl.allocate_shared_memory(gl.bfloat16, (BM, BK), shared_a)
    sb = gl.allocate_shared_memory(gl.bfloat16, (BK, BN), shared_b)
    if not INPUT and BM == 64:
        acc = _m32_project_stream(
            X, weight, ao, bo, amask, acc, sa, sb, K, BK, mma, 1, 1, 256, False
        )
    else:
        depth_a: gl.constexpr = (6 if INPUT else 3) if BM == 32 else 2
        depth_b: gl.constexpr = 3 if BM == 32 and not INPUT else 1
        acc = _m32_project_stream(
            X,
            weight,
            ao,
            bo,
            amask,
            acc,
            sa,
            sb,
            K,
            BK,
            mma,
            depth_a,
            depth_b,
            512 if BM == 32 and INPUT else 128,
            BM == 64 and INPUT,
            BM == 32 and not INPUT,
        )
    om = tile_m * BM + gl.arange(0, BM, gl.SliceLayout(1, mma))
    on = tile_n * BN + gl.arange(0, BN, gl.SliceLayout(0, mma))
    gl.store(
        Y + om[:, None] * YS + on[None, :],
        acc.to(gl.bfloat16),
        (om[:, None] < M) & (on[None, :] < YS),
    )


_m64_recurrent_head_repr = make_kernel_repr(
    "_m64_recurrent_head",
    ["M", "NW", "GROUP", "NARROW_SLOT"],
)


@gluon.jit(repr=_m64_recurrent_head_repr)
def _m64_recurrent_head(
    X,
    CW,
    FW,
    CS,
    S,
    IDX,
    A,
    DT,
    NORM,
    OUT,
    SW: gl.constexpr,
    SI: gl.constexpr,
    SC0: gl.constexpr,
    SC1: gl.constexpr,
    SC2: gl.constexpr,
    SS: gl.constexpr,
    LOWER,
    EPS,
    NW: gl.constexpr = 8,
    M: gl.constexpr = 32,
    GROUP: gl.constexpr = 1,
    CORE_STRIDE: gl.constexpr = 1536,
    NARROW_SLOT: gl.constexpr = False,
):

    st: gl.constexpr = gl.BlockedLayout([1, 4], [8, 8], [NW, 1], [1, 0])
    st_t: gl.constexpr = gl.BlockedLayout([4, 1], [8, 8], [1, NW], [0, 1])
    output_layout: gl.constexpr = gl.BlockedLayout(
        [1, 2], [1, 64], [4, NW // 4], [1, 0]
    )
    vector_layout: gl.constexpr = gl.BlockedLayout([2], [64], [NW], [0])
    pid = gl.program_id(0)
    group = pid // (12 * GROUP)
    remainder = pid % (12 * GROUP)
    if M % GROUP == 0:
        group_size = GROUP
    else:
        group_size = gl.minimum(GROUP, M - group * GROUP)
    m = group * GROUP + remainder % group_size
    h = remainder // group_size
    i = gl.arange(0, 128, vector_layout)
    slot = gl.load(IDX + m * SI).to(gl.int64)
    if slot < 0:
        gl.store(OUT + m * CORE_STRIDE + h * 128 + i, 0)
    else:

        if NARROW_SLOT:
            storage_slot = slot.to(gl.int32)
        else:
            storage_slot = slot
        r = gl.arange(0, 128, gl.SliceLayout(1, st))
        c = gl.arange(0, 32, gl.SliceLayout(0, st))
        sb = S + storage_slot * SS + h * 16384
        so = r[:, None] * 128 + c[None, :]
        shared = gl.allocate_shared_memory(
            gl.float32, (8, 128), gl.SwizzledSharedLayout(1, 1, 1, [1, 0])
        )
        if M <= 32:
            with gl.amd.warp_pipeline_stage("state_prefetch"):
                state_k3 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=".cg")
                state_k2 = gl.amd.cdna4.buffer_load(sb, so + 64, cache=".cg")
                state_k1 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=".cg")
                state_k0 = gl.amd.cdna4.buffer_load(sb, so, cache=".cg")
        else:
            with gl.amd.warp_pipeline_stage("state_prefetch"):
                state_k2 = gl.amd.cdna4.buffer_load(sb, so + 64, cache=".cg")
                state_k0 = gl.amd.cdna4.buffer_load(sb, so, cache=".cg")
                state_k1 = gl.amd.cdna4.buffer_load(sb, so + 32, cache=".cg")
                state_k3 = gl.amd.cdna4.buffer_load(sb, so + 96, cache=".cg")
        _m32_publish_decay(X, FW, A, DT, shared, m, h, LOWER, SW, NW, M <= 32)
        z, history = _m32_convolve(X, CW, CS, m, h, storage_slot, SC0, SC1, SC2, NW)
        partial_norm = gl.sum(gl.reshape(z * z, (4, 2, 64)), 2)
        shared.slice(7, 1).slice(0, 8, dim=1).store(gl.reshape(partial_norm, (1, 8)))
        shared.slice(0, 4).store(z)
        key = gl.sum(shared.slice(1, 1).load(st), 0)
        value = gl.sum(shared.slice(2, 1).load(st_t), 0)
        value = gl.convert_layout(value, gl.SliceLayout(1, st))
        decay = gl.sum(shared.slice(4, 1).load(st), 0)
        nl: gl.constexpr = gl.BlockedLayout([1, 2], [64, 1], [NW, 1], [0, 1])
        qp = shared.slice(7, 1).slice(0, 2, dim=1).load(nl)
        kp = shared.slice(7, 1).slice(2, 2, dim=1).load(nl)
        qnorm = gl.rsqrt(gl.sum(gl.sum(qp, 1), 0) + 1e-6)
        knorm = gl.rsqrt(gl.sum(gl.sum(kp, 1), 0) + 1e-6)
        keys = _m1_split_vector(key, st)
        decays = _m1_split_vector(decay, st)
        decayed = ()
        for p in gl.static_range(4):
            if p == 2:
                packet = state_k2
            elif p == 1:
                packet = state_k1
            elif p == 0:
                packet = state_k0
            else:
                packet = state_k3
            decayed += (packet * decays[p][None, :],)
        prediction = _m1_contract(decayed, keys, (0, 2, 1, 3))
        q = gl.sum(shared.slice(0, 1).load(st), 0)
        queries = _m1_split_vector(q, st)
        beta = _m1_sigmoid(gl.load(X + m * 6336 + 6272 + h).to(gl.float32))
        delta = gl.fma(-prediction, knorm, value) * (beta * knorm)
        update_order: gl.constexpr = (3, 2, 1, 0)
        updates = ()
        for p in gl.static_range(4):
            updated = gl.fma(
                delta[:, None], keys[update_order[p]][None, :], decayed[update_order[p]]
            )
            if p == 0:
                partial = updated * queries[update_order[p]][None, :]
            else:
                partial = gl.fma(updated, queries[update_order[p]][None, :], partial)
            updates += (updated,)
        out = gl.sum(partial, 1) * qnorm * 128 ** (-0.5)
        retire: gl.constexpr = (3, 1, 2, 0)
        for p in gl.static_range(4):
            if M > 32:
                gl.amd.cdna4.buffer_store(
                    updates[3 - retire[p]], sb, so + retire[p] * 32, cache=".cs"
                )
            else:
                gl.amd.cdna4.buffer_store(
                    updates[p], sb, so + update_order[p] * 32, cache=".cs"
                )
        out = out.to(gl.bfloat16).to(gl.float32)
        out = gl.convert_layout(out, gl.SliceLayout(0, st_t))
        shared.slice(6, 1).store(out[None, :])
        out = gl.sum(shared.slice(6, 1).load(output_layout), 0)
        out = gl.convert_layout(out, vector_layout)
        gate = gl.sum(shared.slice(3, 1).load(output_layout), 0)
        gate = gl.convert_layout(gate, vector_layout)
        nw = gl.load(NORM + i).to(gl.float32)
        scale = gl.rsqrt(gl.sum(out * out, 0) / 128 + EPS)
        out = out * scale * (nw * gate)
        old1, old2, x, hb, ho, valid = history
        gl.amd.cdna4.buffer_store(old1, hb, ho, valid)
        gl.amd.cdna4.buffer_store(old2, hb, SC1 + ho, valid)
        gl.amd.cdna4.buffer_store(x, hb, 2 * SC1 + ho, valid)
        gl.store(OUT + m * CORE_STRIDE + h * 128 + i, out)
