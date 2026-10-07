# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2a's decode sparse attention: the original path's Triton split + reduce
(``_paged_decode_split_kernel`` / ``_paged_decode_reduce_kernel``, a 16-head
tile, D = 512, bf16 pool: 64 splits, 16-key tiles, D_CHUNK 128), bit for
bit. Both orders were read off the kernels' ISA and LLVM IR and are pinned by
``harness/p32_split.py`` and ``harness/p32_reduce_emul.py``. The original picks
its splits (``_kv_splits_heuristic``: 64 up to T = 8, 32 at T = 12) and its
reduce's D chunk from the step's rows and heads; this keeps the order of a
``ORDER_ROWS`` (6)-row call over one 16-head tile at every S and TP size, so a
row's sums never depend on the other rows of its step or the rank's head count
(``check._attention_ref`` runs its reference ``ORDER_ROWS`` rows and a tile at
a time).

The rank's heads (``Dims.heads``) are tiles of 16, one a task: its 16 heads the
MFMA's N, as the original's ``block_h`` 16. A ragged tile's (TP8: 8 heads) dead
heads read the last live head and write nothing.

A token has at most ``ATTENTION_KEYS_MAX`` = 1024 keys (``config_refusal``), so
every split is one 16-key tile: split s is keys 16 s .., there are
cdiv(kv_len, 16) of them, and a split's (m, l, acc) is its tile's alone
(acc = MFMA(kv^T, bf16(p), 0): the split kernel's rn(0 x alpha)).

score (S x tiles x 8 tasks, task (t, tile, group sg), wave w: split 8 sg + w):
scores C[key][head] = 16 chained 16x16x32 bf16 MFMAs over D (A = kv rows, B = q), x
qk_scale, NEG past kv_len; m = max; p = exp2(s - m) (v_exp_f32); l = the row sum
(p0 + p2) + (p1 + p3) in lane, + lane^32, + lane^16 -> AM, AL, and bf16 p in the
PV operand's lane order -> AP.

pv (S x tiles x 16 tasks, task (t, tile, 32 columns dc)): wave w' holds the
splits w' + 8 r, which are the reduce's thread (w = w' // 2, hb = w' % 2) group, each split's
acc[d][head] (16x16x16 bf16 MFMA, the 32 columns) in registers. m_max, alpha =
exp2(m - m_max) and l_comb (the reduce's DPP tree) per head; per element
t_i = fma(a_2i, alpha_2i, rn(a_2i+1 alpha_2i+1)), T = (t0 + t1) + (t2 + t3),
X_w = T(2 w) + T(2 w + 1), (X0 + X2) + (X1 + X3); the sink fold,
rn(acc alpha_kv) / max(l_final, 1e-30), bf16 -> ``att["out"]`` LDS [16][32],
which the same task's inverse-RoPE quant reads.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.dsv41_mono.atomv41 import attention_plan as aplan
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41 import index_plan as ip
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.config import ATTENTION_KEYS_MAX
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.dims import HEAD_DIM, HEAD_TILE
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.step_meta import KMETA
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ops import (
    bf16_round,
    butterfly,
    hw_exp2,
    lane_gather,
    ld_dev,
    ld_i32,
    mfma_bf16,
    row_shr,
    rsrc,
    traced,
    xshfl,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.execution import BLOCKS

SPLITS, BK = aplan.SPLITS, aplan.BK  # a split is a 16-key tile
# the original decode call whose split and reduce order this keeps: T rows
# (``_kv_splits_heuristic`` and the reduce's D chunk follow T x heads)
ORDER_ROWS = 6
assert SPLITS * BK == ATTENTION_KEYS_MAX
SCORE_INFLIGHT = 2  # a score wave's splits whose key loads are in flight together
COLS = 32  # the pv task's columns: one FP8 quant group
GROUPS = HEAD_DIM // COLS  # a head's column groups


def score_tasks(s, tiles) -> int:
    """Score tasks a (token, head tile) (``stage_attn_score``): the most, up to
    a split a wave, that keep a step's tasks to one round of the CTAs. Past it
    (48 rows: 8 a token) 128 CTAs ran two in series, the stage's end 14 us;
    four a token, a wave's two splits' loads together: attention 34 -> 32 us."""
    per = SPLITS // 8
    while s * tiles * per > BLOCKS and per > 1:
        per //= 2
    return per


def pv_groups(s, tiles) -> int:
    """Column groups a PV task covers (``stage_attn_pv``): the fewest that keep
    a step's tasks to one round of the CTAs. A task's cost is its latency (the
    score outputs' poll, the key rows' loads), shared by its groups."""
    groups = 1
    while s * tiles * GROUPS // groups > BLOCKS and groups < GROUPS:
        groups *= 2
    return groups


NEG = -3.4028234663852886e38
LOG2E = 1.4426950408889634
ROW_BYTES = HEAD_DIM * 2


def scratch_pairs(tokens: int, d) -> dict[str, int]:
    return {
        "am": tokens * SPLITS * d.heads,
        "al": tokens * SPLITS * d.heads,
        "ap": tokens * d.head_tiles.count * SPLITS * 64 * 2,
    }


def tile_task(task, per, tiles):
    """Task ``task`` of a stage with ``per`` tasks a (token, head tile) -> (token,
    tile, the task's index in its tile)."""
    if tiles.count == 1:
        return task // per, 0, task % per
    return task // (per * tiles.count), task // per % tiles.count, task % per


def tile_head(d, tile, h):
    """Head h of ``tile`` -> (the rank's head it reads, live): a ragged tile's
    dead heads read the last live one."""
    tiles = d.head_tiles
    if tiles.count == 1 and not tiles.ragged:
        return h, True
    head = tile * HEAD_TILE + h
    if not tiles.ragged:
        return head, True
    return fx.min(head, d.heads - 1), head < d.heads


def _row_ptr(pool, row, byte):
    """Pool row ``row`` at ``byte``: 64-bit, a row index times the 1 KB row
    passing 2^32 (``pool_index.row_offset``), so neither a buffer offset nor an
    i32 product can carry it."""
    return pool + fx.Int64(row) * ROW_BYTES + fx.Int64(byte)


def _ld_row4(pool, row, byte):
    ptr = fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16), _row_ptr(pool, row, byte)
    )
    return fx.Vector(fx.ptr_load(ptr, result_type=fx.Vector.make_type(4, fx.Int32)))


def _keys(a, t):
    """Token t's key count and what ``_key_row`` needs: its selected count (the
    selection's valid ids, ``index_plan.selected`` of its scorer's bound), its
    first window position and its slot's row base (word 2 of its metadata, the
    block table offset, is the indexer's: ``index_score._selected_rows``)."""
    first, count, base = [ld_i32(a["kmeta"], t * KMETA + i) for i in (0, 1, 3)]
    n_sel = ip.selected(fx.Int32(ld_dev(a["sbound"], t)))
    return aplan.kv_len(n_sel, count), (n_sel, first, base)


@traced
def await_selection(c, t):
    """A layer that selects: its token t's selection is written this launch. A
    build without the indexer (``index_bound_max`` 0: the DSpark draft) has no
    selection to wait for."""
    if const_expr("irdy" in c):  # noqa: SIM102
        if c["args"]["iact"] != 0:
            c["poll"]([(c["irdy"], t, 1)])


def _key_row(a, t, meta, key):
    """Pool row of token t's key ``key``, in the original index build's order
    (``indices._indices``): the selected compressed rows ascending (``sel``: the
    pool rows the selecting layer's indexer wrote with its selection,
    ``index_score._selected_rows``), then the window rows oldest first."""
    n_sel, first, base = meta
    key = fx.max(key, 0)
    picked = key < n_sel
    # a layer without a selection passes a dummy ``sel``: its row is never used
    selected = fx.Int32(
        picked.select(ld_dev(a["sel"], t * a["topk"] + fx.min(key, a["topk"] - 1)), 0)
    )
    window = base + a["ring_off"] + (first + key - n_sel) % a["ring_slots"]
    return fx.Int32(picked.select(selected, window))


def only_live(cond, live):
    """``cond`` and ``live`` (``tile_head``'s; True: every head of the tile is)."""
    return cond if live is True else cond & live


def _ms(d, t, s, h):
    return (t * SPLITS + s) * d.heads + h


def _ap(d, t, tile, s):
    """Split s of token t's head tile in AP: its bf16 p, a pair a lane."""
    if d.head_tiles.count == 1:
        return (t * SPLITS + s) * 64
    return ((t * d.head_tiles.count + tile) * SPLITS + s) * 64


@traced
def stage_attn_score(c, task, per):
    """Score task ``task``, ``per`` a (token, head tile) (``score_tasks``): wave
    w its splits (sg + per k) 8 + w, SCORE_INFLIGHT splits' key loads at a
    time (q's once, first) issued before those splits' MFMAs."""
    lane, wave, d = c["lane"], c["wave"], c["d"]
    a = c["args"]
    g = lane // 16
    head = lane % 16
    t, tile, sg = tile_task(task, per, d.head_tiles)
    qh, live = tile_head(d, tile, head)
    await_selection(c, t)
    kv_len, meta = _keys(a, t)
    ss = [
        aplan.score_split(sg, per, k, wave)
        for k in range(aplan.score_splits_a_wave(per))
    ]
    slots = [_key_row(a, t, meta, fx.min(s * BK + head, kv_len - 1)) for s in ss]
    qws = [
        fx.Vector(
            bo.buffer_load(
                rsrc(a["q"]),
                ((t * d.heads + qh) * HEAD_DIM + ch * 32 + 8 * g) // 2,
                vec_width=4,
                dtype=T.i32,
            )
        )
        for ch in range(HEAD_DIM // 32)
    ]
    # SCORE_INFLIGHT splits' keys in flight at a time: a split's are 64
    # registers a lane, and TP2's two head tiles at 40+ rows give a wave four
    # splits, whose loads together spilled (424 B of scratch a lane)
    for b0 in range_constexpr(0, len(ss), SCORE_INFLIGHT):
        batch = ss[b0 : b0 + SCORE_INFLIGHT]
        kws = [
            [
                _ld_row4(a["pool"], slot, (ch * 32 + 8 * g) * 2)
                for ch in range(HEAD_DIM // 32)
            ]
            for slot in slots[b0 : b0 + SCORE_INFLIGHT]
        ]
        for s, kw in zip(batch, kws):
            _score_split(c, t, tile, s, qh, live, kv_len, kw, qws)


@traced
def _score_split(c, t, tile, s, qh, live, kv_len, kws, qws):
    """Split s of token t, head tile ``tile``, on this wave from its key and q
    operands: the scores' max, sum and bf16 probabilities -> AM / AL / AP."""
    lane, d = c["lane"], c["d"]
    a = c["args"]
    g = lane // 16
    if s < aplan.written_splits(kv_len):
        key = s * BK + lane % 16
        ok = key < kv_len
        sc = fx.Vector.filled(4, 0.0, fx.Float32)
        for ch in range_constexpr(HEAD_DIM // 32):
            kw = fx.Vector.from_elements(
                [ok.select(kws[ch][e], fx.Int32(0)) for e in range(4)], fx.Int32
            )
            qw = qws[ch]
            sc = mfma_bf16(kw.bitcast(fx.BFloat16), qw.bitcast(fx.BFloat16), sc)
        scale = fx.Int32(a["qk_scale"]).bitcast(fx.Float32)
        sv = [
            fx.Float32(
                (s * BK + 4 * g + i < kv_len).select(sc[i] * scale, fx.Float32(NEG))
            )
            for i in range(4)
        ]
        m = butterfly(
            fx.max(fx.max(sv[0], sv[1]), fx.max(sv[2], sv[3])), (32, 16), fx.max
        )
        p = [hw_exp2(sv[i] - m) for i in range(4)]
        ssum = (p[0] + p[2]) + (p[1] + p[3])
        ssum = ssum + xshfl(ssum, 32)
        ssum = ssum + xshfl(ssum, 16)
        if only_live(g == 0, live):
            c["put"](c["am"], _ms(d, t, s, qh), m)
            c["put"](c["al"], _ms(d, t, s, qh), ssum)
        pw = fx.Vector.from_elements(p, fx.Float32).to(fx.BFloat16).bitcast(fx.Int32)
        c["put_words"](c["ap"], (_ap(d, t, tile, s) + lane) * 2, [pw[0], pw[1]])


def _l_tree(l, alpha, lane):
    """The reduce's sum of l x alpha over lanes = splits (read at lane 63)."""
    x = l * alpha
    v = fx.Float32(fmath.fma(l, alpha, row_shr(x, lane, 8)))
    for k in (4, 2, 1):
        v = row_shr(v, lane, k) + v
    row = lane // 16
    b15 = lane_gather(v.bitcast(fx.Int32), fx.max(row * 16 - 1, 0)).bitcast(fx.Float32)
    v = fx.Float32((row % 2 == 1).select(b15 + v, v))
    b31 = lane_gather(v.bitcast(fx.Int32), 31).bitcast(fx.Float32)
    v = fx.Float32((row >= 2).select(b31 + v, v))
    return lane_gather(v.bitcast(fx.Int32), 63).bitcast(fx.Float32)


@traced
def _stat_specs(c, t, tile, last, lane, wave):
    """Lane = split (``attention_plan.polled_split``), wave w: (m, l) of the tile's heads 2 w,
    2 w + 1."""
    d = c["d"]
    src = aplan.polled_split(lane, last)
    heads = [tile_head(d, tile, 2 * wave + k)[0] for k in range(2)]
    return [(c["am"], _ms(d, t, src, heads[k]), 1) for k in range(2)] + [
        (c["al"], _ms(d, t, src, heads[k]), 1) for k in range(2)
    ]


@traced
def _pv_stats(got, act, lane, wave, att):
    """``_stat_specs``' words -> per head m_max, alpha [split][head] and l_comb
    in LDS."""
    live = lane < act
    for k in range_constexpr(2):
        h = 2 * wave + k
        m = fx.Float32(live.select(got[k][0].bitcast(fx.Float32), fx.Float32(NEG)))
        l = fx.Float32(live.select(got[2 + k][0].bitcast(fx.Float32), fx.Float32(0.0)))
        m_max = butterfly(m, (32, 16, 8, 4, 2, 1), fx.max)
        alpha = hw_exp2(m - m_max)
        l_comb = _l_tree(l, alpha, lane)
        fx.ptr_store(alpha, att["alpha"] + (lane * HEAD_TILE + h))
        if lane == 0:
            fx.ptr_store(m_max, att["stat"] + h)
            fx.ptr_store(l_comb, att["stat"] + (HEAD_TILE + h))


@traced
def stage_attn_pv(c, task, groups):
    """Token t, ``groups`` column groups of 32 (from 32 ``groups`` g0) of every
    head of a tile -> ``att["out"]`` [group][head][column] (f32 of bf16),
    barriers included. The score outputs' poll serves every group; a group's
    pool loads issue under the previous group's MFMAs."""
    lane, wave, d = c["lane"], c["wave"], c["d"]
    att = c["att"]
    a = c["args"]
    t, tile, g0 = tile_task(task, GROUPS // groups, d.head_tiles)
    await_selection(c, t)
    kv_len, meta = _keys(a, t)
    act = (kv_len + BK - 1) // BK
    last = aplan.last_split(kv_len)
    splits = [wave + 8 * r for r in range(8)]
    # each split's 16 keys x 32 columns, a lane a quarter row (16 B), all issued
    # before the first is used
    rows = [
        _key_row(a, t, meta, fx.min(sp * BK + lane // 4, kv_len - 1)) for sp in splits
    ]

    def group_tiles(i):
        dc = g0 * groups + i
        return [
            _ld_row4(a["pool"], rows[r], (dc * COLS + 8 * (lane % 4)) * 2)
            for r in range(8)
        ]

    tiles = group_tiles(0)
    # thread (head, column)'s sink: one load a task, not a group
    sink = fx.Float32(
        bo.buffer_load(
            rsrc(a["sink"]), tile_head(d, tile, c["tid"] // COLS)[0], vec_width=1,
            dtype=T.f32,
        )
    )  # fmt: skip
    stats = _stat_specs(c, t, tile, last, lane, wave)
    # wave 0 waits (its lane = split: every split's stats) before the CTA polls:
    # 512 spinning threads load the memory the score tasks read (S=48 -1 us)
    if wave == 0:
        c["poll"](stats)
    gpu.barrier()
    # the score stage's outputs, one poll batch: stats, then each split's p
    got = c["poll"](
        stats
        + [
            (c["ap"], (_ap(d, t, tile, aplan.polled_split(sp, last)) + lane) * 2, 2)
            for sp in splits
        ],
        batch=12,
    )
    _pv_stats(got[:4], act, lane, wave, att)
    pws = got[4:]
    gpu.barrier()
    al = [fx.ptr_load(att["alpha"] + (sp * HEAD_TILE + lane % 16)) for sp in splits]
    for i in range_constexpr(groups):
        nxt = group_tiles(i + 1) if i + 1 < groups else None
        _pv_group(c, kv_len, act, splits, tiles, pws, al, sink, i)
        tiles = nxt


@traced
def _pv_group(c, kv_len, act, splits, tiles, pws, al, sink, i):
    """``stage_attn_pv``'s group i: each split's PV MFMAs on its key tile
    ``tiles``, their alpha-weighted tree, the combine (with thread (head,
    column)'s ``sink``) -> ``att["out"]`` group i."""
    lane, wave, tid = c["lane"], c["wave"], c["tid"]
    att = c["att"]
    g = lane // 16
    head = lane % 16
    kt = att["kv"] + wave * (BK * COLS // 2)
    accs = []
    for r in range_constexpr(8):
        sp = splits[r]
        pb = fx.Vector.from_elements([pws[r][0], pws[r][1]], fx.Int32).bitcast(fx.Int16)
        ok = sp * BK + lane // 4 < kv_len
        # this wave's LDS tile: [key][32 columns], this lane's quarter row
        fx.ptr_store(
            fx.Vector.from_elements(
                [ok.select(tiles[r][e], fx.Int32(0)) for e in range(4)], fx.Int32
            ),
            kt + (lane // 4 * (COLS // 2) + lane % 4 * 4),
        )
        for tt in range_constexpr(2):
            col = tt * 16 + head
            half = [
                fx.Int32(fx.ptr_load(kt + ((4 * g + i4) * (COLS // 2) + col // 2)))
                >> (col % 2 * 16)
                & 0xFFFF
                for i4 in range(4)
            ]
            av = fx.Vector.from_elements(
                [half[0] | (half[1] << 16), half[2] | (half[3] << 16)], fx.Int32
            )
            acc = fx.Vector(
                rocdl.mfma_f32_16x16x16bf16_1k(
                    T.vec(4, T.f32),
                    [av.bitcast(fx.Int16), pb, fx.Vector.filled(4, 0.0, fx.Float32)],
                )
            )
            live = sp < act
            accs.append(
                [fx.Float32(live.select(acc[j], fx.Float32(0.0))) for j in range(4)]
            )
    for k in range_constexpr(8):
        tt, j = k // 4, k % 4
        av = [accs[2 * r + tt][j] for r in range(8)]
        pr = [
            fx.Float32(fmath.fma(av[2 * m], al[2 * m], av[2 * m + 1] * al[2 * m + 1]))
            for m in range(4)
        ]
        node = (pr[0] + pr[1]) + (pr[2] + pr[3])
        fx.ptr_store(node, att["tree"] + ((wave * 64 + lane) * 8 + k))
    gpu.barrier()
    # thread (head, column): the lane that holds the element, its node k
    h = tid // COLS
    col = tid % COLS
    owner = col % 16 // 4 * 16 + h
    k = col // 16 * 4 + col % 4
    w = [fx.ptr_load(att["tree"] + ((ww * 64 + owner) * 8 + k)) for ww in range(8)]
    x = [w[2 * m] + w[2 * m + 1] for m in range(4)]
    tot = (x[0] + x[2]) + (x[1] + x[3])
    m_max = fx.ptr_load(att["stat"] + h)
    l_comb = fx.ptr_load(att["stat"] + (HEAD_TILE + h))
    log2e = fx.Float32(LOG2E)
    m_final = fx.max(m_max, sink * log2e)
    alpha_kv = hw_exp2(m_max - m_final)
    alpha_sink = hw_exp2(fx.Float32(fmath.fma(sink, log2e, -m_final)))
    l_final = fx.Float32(fmath.fma(l_comb, alpha_kv, alpha_sink))
    o = (tot * alpha_kv) / fx.max(l_final, fx.Float32(1e-30))
    o = fx.Float32((l_final > fx.Float32(0.0)).select(o, fx.Float32(0.0)))
    fx.ptr_store(bf16_round(o), att["out"] + ((i * HEAD_TILE + h) * COLS + col))
    gpu.barrier()
