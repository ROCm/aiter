# SPDX-License-Identifier: MIT
"""The DS V4.1 MoE's routed gate/up stage (ug) in Gluon: ``stages/moe.py``
``stage_ug`` / ``ug_pass`` / ``ug_epilogue`` (vLLM 8321d11) on batched scaled
MFMA (patched Triton, DESIGN.md).

An ug task: slot u of U (expert e) and part ``part0 / 2`` of its intermediate:
w13 rows 64 part0 .. + 128 -- in aiter's GUGU order four (gate, up) pairs of
16-row tiles, i.e. four 32-row quarters, a pair each -- times every token's MXFP8
row (S <= 16: the x8r tile in LDS, tokens as the MFMA's 16 columns; a pick reads
its token's column). Waves split K 8 ways (wave w: steps w, w + 8, ..), all
four quarters on one B load a step; the waves' partials are summed in wave
order a quarter at a time through LDS. Epilogue: clamp, silu x up, bf16, the
stage-2 input quant (amax floored 1e-10, the ceil code) -> MID, its code ->
MIDS, then the task's UGF flag. FP4 weights in aiter's (16, 16) shuffle, E8M0
scales in its shuffle_scale order, as vLLM loads them."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from ..common.build import NOINLINE

from ..common.mx import clamp_fp8, code_ceil, fp8_pack4, rcp_pow2
from ..common.ops import bf16_round, hw_exp2, hw_rcp, lane_gather, xmax
from ..common.sync import publish, store32
from .moe_shared import AD, BD, MMA_S, SH3, _ar3

LIMIT = gl.constexpr(10.0)
LOG2E = gl.constexpr(1.4426950408889634)
# a quarter's weights [wave][32 rows][64 bytes]: 16 bytes a thread, lane row + 16 kq
# (the MFMA's own A order: no shuffle to convert)
A_LD = gl.constexpr(gl.BlockedLayout([1, 1, 16], [1, 16, 4], [8, 1, 1], [1, 2, 0]))
# its scales as stored ([wave][col e][row b][row half a], aiter's shuffle_scale):
# the two halves' codes are adjacent, 2 bytes a lane; permuted to [wave][32][4]
# it is the MFMA scale operand's layout
S_MEM = gl.constexpr(gl.BlockedLayout([1, 1, 1, 2], [1, 4, 16, 1], [8, 1, 1, 1], [3, 2, 1, 0]))
RED_Q = gl.constexpr(gl.BlockedLayout([1, 1, 1], [1, 4, 16], [8, 1, 1], [2, 1, 0]))


@gluon.jit
def scale_index(row, col, COLS: gl.constexpr):
    """Byte of e8m0 scale (row, col) in aiter's shuffle_scale layout."""
    r32 = row // 32
    a = (row // 16) % 2
    b = row % 16
    c8 = col // 8
    d = (col // 4) % 2
    e = col % 4
    return ((((r32 * (COLS // 8) + c8) * 4 + e) * 16 + b) * 2 + d) * 2 + a


@gluon.jit
def fp4_quarter(w13, w13_s, e, row0, st, N_ROWS: gl.constexpr, K: gl.constexpr):
    """Expert e's w13 rows row0 .. + 32, K step st (wave w: st + w): the packed
    FP4 bytes [wave][32][64] and their E8M0 codes [wave][32][4]."""
    wa, ra, kb = _ar3(8, 32, 64, A_LD)
    row = row0 + ra
    stw = st + wa
    # (16, 16) shuffle: a 16-row group's step is 1 KB, lane (j, r) 16 bytes at
    # (16 j + r) 16 holding row r's K bytes 16 j ..
    off = (e.to(gl.int64) * (N_ROWS * K // 2) + (row // 16) * (16 * K // 2) + stw * 1024
           + ((kb // 16) * 16 + row % 16) * 16 + kb % 16)  # fmt: skip
    a = gl.load(w13.to(gl.pointer_type(gl.uint8)) + off, cache_modifier=".cg")
    # scale_index of (row 32 r32 + 16 a + b, col 4 stw + g), unrolled: c8 = stw / 2,
    # d = stw % 2, e = g
    COLS: gl.constexpr = K // 32
    sw = gl.arange(0, 8, layout=gl.SliceLayout(1, gl.SliceLayout(2, gl.SliceLayout(3, S_MEM))))
    sg = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(2, gl.SliceLayout(3, S_MEM))))
    sb = gl.arange(0, 16, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(3, S_MEM))))
    sh = gl.arange(0, 2, layout=gl.SliceLayout(0, gl.SliceLayout(1, gl.SliceLayout(2, S_MEM))))
    stw4 = st + sw[:, None, None, None]
    r32 = (e * N_ROWS + row0) // 32
    sidx = (((((r32 * (COLS // 8) + stw4 // 2) * 4 + sg[None, :, None, None]) * 16
              + sb[None, None, :, None]) * 2 + stw4 % 2) * 2 + sh[None, None, None, :])  # fmt: skip
    sa = gl.load(w13_s.to(gl.pointer_type(gl.uint8)) + sidx, cache_modifier=".cg")
    return a, gl.reshape(gl.permute(sa, (0, 3, 2, 1)), (8, 32, 4))


@gluon.jit
def ug_gemv(w13, w13_s, e, part0, xa, xb, ca8, cb8, N_ROWS: gl.constexpr,
            K: gl.constexpr):  # fmt: skip
    """The task's four quarters' [8][32][16] partials (wave w: K steps w + 8 i)."""
    gl.static_assert(K == 5120)
    SAL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(AD, [8, 32, 4])
    SBL: gl.constexpr = gl.amd.cdna4.get_mfma_scale_layout(BD, [8, 16, 4])
    row0 = part0 * 64
    acc0 = gl.zeros((8, 32, 16), gl.float32, MMA_S)
    acc1 = gl.zeros((8, 32, 16), gl.float32, MMA_S)
    acc2 = gl.zeros((8, 32, 16), gl.float32, MMA_S)
    acc3 = gl.zeros((8, 32, 16), gl.float32, MMA_S)
    for i in gl.static_range(5):
        if i < 4:
            b = xa._reinterpret(gl.uint8, [4, 16, 8, 128], SH3).index(i).permute((1, 2, 0)).load(BD)
            sb = ca8._reinterpret(gl.uint8, [8, 16, 8, 4], SH3).index(i).permute((1, 0, 2)).load(SBL)
        else:
            b = xb.permute((1, 2, 0)).load(BD)
            sb = cb8._reinterpret(gl.uint8, [2, 16, 8, 4], SH3).index(0).permute((1, 0, 2)).load(SBL)
        a, sa = fp4_quarter(w13, w13_s, e, row0, 8 * i, N_ROWS, K)
        acc0 = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, AD), gl.convert_layout(sa, SAL), "e2m1",
                                        b, sb, "e4m3", acc0)  # fmt: skip
        a, sa = fp4_quarter(w13, w13_s, e, row0 + 32, 8 * i, N_ROWS, K)
        acc1 = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, AD), gl.convert_layout(sa, SAL), "e2m1",
                                        b, sb, "e4m3", acc1)  # fmt: skip
        a, sa = fp4_quarter(w13, w13_s, e, row0 + 64, 8 * i, N_ROWS, K)
        acc2 = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, AD), gl.convert_layout(sa, SAL), "e2m1",
                                        b, sb, "e4m3", acc2)  # fmt: skip
        a, sa = fp4_quarter(w13, w13_s, e, row0 + 96, 8 * i, N_ROWS, K)
        acc3 = gl.amd.cdna4.mfma_scaled(gl.convert_layout(a, AD), gl.convert_layout(sa, SAL), "e2m1",
                                        b, sb, "e4m3", acc3)  # fmt: skip
    return acc0, acc1, acc2, acc3


@gluon.jit
def quarter_sums(acc, red, r, col):
    """Gate (row r) and up (row 16 + r) of one quarter at MFMA column ``col``,
    the waves summed in order."""
    red.store(gl.reshape(gl.convert_layout(acc, RED_Q), (4096,)))
    gl.barrier()
    g = red.gather(r * 16 + col, 0)
    u = red.gather((16 + r) * 16 + col, 0)
    for w in gl.static_range(1, 8):
        g = g + red.gather(w * 512 + r * 16 + col, 0)
        u = u + red.gather(w * 512 + (16 + r) * 16 + col, 0)
    gl.barrier()
    return g, u


@gluon.jit
def act_bf16(g, up):
    gc = gl.minimum(g, LIMIT)
    uc = gl.maximum(gl.minimum(up, LIMIT), -LIMIT)
    sig = hw_rcp(1.0 + hw_exp2(gc * (-LOG2E)))
    return bf16_round(gc * sig * uc)


@gluon.jit
def at(tab, i, tid):
    """Route table word i (uniform), as a scalar."""
    return gl.max(tab.gather(gl.zeros_like(tid) + i, 0), axis=0)


@gluon.jit
def slot_unit(tab, u, tid, S: gl.constexpr, TOPK: gl.constexpr, FIRST0: gl.constexpr):
    """Slot u of U: its expert, its first pick in the route order, its picks."""
    first = at(tab, FIRST0 + u, tid)
    return at(tab, 2 * S * TOPK + 1 + u, tid), first, at(tab, FIRST0 + u + 1, tid) - first


@gluon.jit
def tile_unit(tab, tile, tid, S: gl.constexpr, TOPK: gl.constexpr, FIRST0: gl.constexpr,
              TILE_FIRST0: gl.constexpr, TILE_SLOT0: gl.constexpr):  # fmt: skip
    """Pick tile ``tile`` (past one token tile; upstream ``ug_tile_task``): its
    slot's expert, its first pick (``moe_plan.tile_pick0``), the slot's picks
    from it (an ug task takes 16 of them at most)."""
    u = at(tab, TILE_SLOT0 + tile, tid)
    e, sf, sc = slot_unit(tab, u, tid, S, TOPK, FIRST0)
    first = sf + (tile - at(tab, TILE_FIRST0 + u, tid)) * 16
    return e, first, sf + sc - first


@gluon.jit(noinline=NOINLINE)
def stage_ug(task, e, first, count, tab, w13, w13_s, mid, mids, ugf, xa, xb, ca8, cb8, red,
             tag, tid, lane, TOPK: gl.constexpr, UG_PARTS: gl.constexpr,
             ORDER0: gl.constexpr, N_ROWS: gl.constexpr, INTER: gl.constexpr,
             K: gl.constexpr, GATHERED: gl.constexpr):  # fmt: skip
    """Ug task ``task``: expert e's intermediate groups 2 (task % UG_PARTS) .., + 2
    for the picks first .. first + min(count, 16) of the route order -> their MID
    rows, then UGF ``task``. The x tile in LDS: every token's (one token tile: a
    pick reads its token's column), or ``GATHERED`` the picks' own rows (column
    j: pick first + j, ``load_x_tile``'s gather)."""
    MID_WORDS: gl.constexpr = INTER // 4
    STEPS: gl.constexpr = INTER // 128
    zero = gl.zeros_like(tid)
    part0 = (task % UG_PARTS) * 2
    q0, q1, q2, q3 = ug_gemv(w13, w13_s, e, part0, xa, xb, ca8, cb8, N_ROWS, K)
    # thread (j, i): the slot's pick j, column i of the pass's group
    j = tid // 32
    i = tid % 32
    r = i % 16
    live = j < count
    if GATHERED:
        col = gl.minimum(j, gl.maximum(count - 1, 0))
    else:
        pick = tab.gather(ORDER0 + first + gl.minimum(j, gl.maximum(count - 1, 0)), 0)
        col = pick // TOPK  # the pick's token: its MFMA column
    for p in gl.static_range(2):
        if p == 0:
            ga, ua = quarter_sums(q0, red, r, col)
            gb, ub = quarter_sums(q1, red, r, col)
        else:
            ga, ua = quarter_sums(q2, red, r, col)
            gb, ub = quarter_sums(q3, red, r, col)
        v = act_bf16(gl.where(i < 16, ga, gb), gl.where(i < 16, ua, ub))
        amax = gl.abs(v)
        amax = xmax(amax, lane, 1)
        amax = xmax(amax, lane, 2)
        amax = xmax(amax, lane, 4)
        amax = xmax(amax, lane, 8)
        amax = xmax(amax, lane, 16)
        code = code_ceil(gl.maximum(amax, 1e-10) * (1.0 / 448.0))
        q8 = clamp_fp8(v * rcp_pow2(code))
        word = fp8_pack4(q8, lane_gather(q8, gl.minimum(lane + 1, 63)),
                         lane_gather(q8, gl.minimum(lane + 2, 63)),
                         lane_gather(q8, gl.minimum(lane + 3, 63)))  # fmt: skip
        row = first + j
        part = part0 + p
        mw = mid.to(gl.uint64) + ((row * MID_WORDS + part * 8 + i // 4) * 4).to(gl.uint64)
        store32(mw, word, live & (i % 4 == 0))
        ci = row * (INTER // 32) + (part % 4) * STEPS + part // 4
        store32(mids.to(gl.uint64) + (ci * 4).to(gl.uint64), code, live & (i == 0))
    publish(ugf, zero + task, zero + 1, tag, tid == 0)
    gl.barrier()
