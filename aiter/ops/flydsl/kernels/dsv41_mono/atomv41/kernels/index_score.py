# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2a's indexer (a layer that selects: FULL, and REINDEX within layer 20's
candidates): the scores of every visible index row, the top-512 of them and, on
the candidate-producing layer, the top-2048 blocks -- the answer of the
original ``score_topk_paged`` (logits: gluon ``deepgemm_fp8_paged_mqa_logits``;
selection: ``top_k_per_row_decode(stable=True)``; blocks: ``pick_candidate_blocks``)
at any context length. Which cells each stage writes and polls is decided per
step only in ``index_plan`` (DESIGN_v3 4.9); the CPU test there checks it.

    iscore  (a task: a run of a request's tokens x up to TASK_CHUNKS chunks
            of 128 columns, ``index_plan.score_tasks``; a wave a tile pair):
            every key load first, then each token's scores (4 chained
            16x16x32 fp8 MFMAs, relu x k-scale, gluon's head sum;
            ``harness/p31_score.py``) -> global ``ilog``; a candidate
            producer's 8-row block maxima (NaN ignored, capped at the largest
            finite, the newest block +inf) -> global ``ibmax``; then, once a
            task, the flags (ISF) of its (token, chunk)s. On the FP4 plane
            ``index_score_fp4.run_scores4`` instead (the row-group scorer's
            arithmetic), writing the same cells
    ilist   (a level's lists, per token): ``topk.lds_topk`` of a segment's
            values (level 0, after its chunks' flags) or of the lists it merges
            (level l > 0, from ISEL<l-1> / IBLK<l-1>), in index order. A list
            that is its token's last -> global (``isel``, lifted to real rows on
            a REINDEX layer; ``icout``) and, for the selection, the ready flag
            (IRDY) the attention waits on; else -> ISEL<l> / IBLK<l>.

Data too large for tagged pairs (the logits, up to max_model_len a token) goes
through global memory at device scope (SC1: past the per-XCD L2): stored,
``s_waitcnt vmcnt(0)``, then the flag: a task's drain and barrier, hence
several chunks a task.
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, T

from aiter.ops.flydsl.kernels.dsv41_mono.atomv41 import index_plan as ip
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.index_score_fp4 import run_scores4
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.topk import key_bits, lds_topk
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.step_meta import KMETA
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ops import (
    CM_DEV,
    butterfly,
    ld_dev,
    ld_i32,
    mfma_fp8,
    rsrc,
    traced,
    xred,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.sync import publish
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.arith import pick, sel
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.execution import BLOCKS, THREADS, first_task
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.trace import enter_stage

TILE = ip.BLOCK_ROWS
# a score task's chunks at most: 1 << this, loads in flight together
TASK_CHUNKS_LOG2 = 3
TASK_CHUNKS = 1 << TASK_CHUNKS_LOG2
TILE_BYTES = TILE * ip.DIM + TILE * 4  # 8 preshuffled rows, then 8 fp32 scales
PER_THREAD = ip.SEG // THREADS
NEG = float("-inf")
FLT_MAX = 3.4028234663852886e38


def scratch_pairs(tokens: int, bound_max: int) -> dict[str, int]:
    """The indexer's mailbox regions. The last level's list is always a token's
    final one (global), so only levels below it have a region."""
    out = {"isf": tokens * ip.chunks(bound_max), "irdy": tokens}
    for kind, k in ip.KINDS.items():
        for level in range(ip.top_level(kind, bound_max)):
            slots = ip.list_slots(kind, level, bound_max)
            out[f"i{kind}{level}"] = tokens * slots * k * 2
    return out


def _st_dev(v, ptr, i):
    bo.buffer_store(v, rsrc(ptr), i, cache_modifier=CM_DEV)


def _head_sum(terms):
    """gluon's: k = 4 h + e; pairs (0,2)(1,3)(4,6)(5,7) fma(w_a r_a, w_b r_b);
    (s0 + s2) + (s1 + s3); lane groups xor 32, then xor 16."""
    rw = [terms[k // 4, k % 4] for k in range(8)]
    firsts = [
        fx.Float32(fmath.fma(rw[a][0], rw[a][1], rw[b][0] * rw[b][1]))
        for a, b in ((0, 2), (1, 3), (4, 6), (5, 7))
    ]
    v = (firsts[0] + firsts[2]) + (firsts[1] + firsts[3])
    for off in (32, 16):
        v = xred(v, off, lambda x, y: x + y)
    return v


def _plane_ptr(plane, tile, byte):
    return fx.inttoptr(
        fx.PointerType.get(T.i32, fx.AddressSpace.Global, 16),
        plane + fx.Int64(tile) * TILE_BYTES + fx.Int64(byte),
    )


def _chunk_keys(a, t, ch0, width, wave, lane):
    """This lane's keys of chunks [ch0, ch0 + width) through token t's tile
    table (wave: a tile pair a chunk): every table load, then every key load,
    none waited on. A slot past ``width`` re-reads the last chunk (cache hits);
    a chunk past the table reads its last tile."""
    j = lane // 16
    r8 = lane % 8
    blks = [
        (ch0 + fx.min(width - 1, m)) * (ip.CHUNK // TILE) + 2 * wave + lane % 16 // 8
        for m in range(TASK_CHUNKS)
    ]
    tiles = [
        fx.Int32(
            bo.buffer_load(
                rsrc(a["itab"]),
                t * a["itab_stride"] + fx.min(blk, a["itab_len"] - 1),
                vec_width=1,
                dtype=T.i32,
            )
        )
        for blk in blks
    ]
    keys = []
    for blk, tile in zip(blks, tiles):
        ks = fx.Int32(
            fx.ptr_load(_plane_ptr(a["plane"], tile, TILE * ip.DIM + r8 * 4))
        ).bitcast(fx.Float32)
        kv = [
            fx.Vector(
                fx.ptr_load(
                    _plane_ptr(a["plane"], tile, ((4 * p + j) * TILE + r8) * 16),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            for p in range(2)
        ]
        b64 = fx.Vector.from_elements(
            [kv[p][d] for p in range(2) for d in range(4)], fx.Int32
        ).bitcast(fx.Int64)
        keys.append((blk, ks, b64))
    return keys


def _token_query(a, t, lane):
    """Token t's index q (this lane's MFMA A operand) and head weights, per
    16-head half."""
    j = lane // 16
    query = []
    for h in range(2):
        qw = [
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["iq"]),
                    ((t * ip.HEADS + h * 16 + lane % 16) * ip.DIM + (4 * p + j) * 16)
                    // 4,
                    vec_width=4,
                    dtype=T.i32,
                )
            )
            for p in range(2)
        ]
        a64 = fx.Vector.from_elements(
            [qw[p][d] for p in range(2) for d in range(4)], fx.Int32
        ).bitcast(fx.Int64)
        wts = [
            fx.Float32(
                bo.buffer_load(
                    rsrc(a["iw"]),
                    t * ip.HEADS + h * 16 + 4 * j + e,
                    vec_width=1,
                    dtype=T.f32,
                )
            )
            for e in range(4)
        ]
        query.append((a64, wts))
    return query


@traced
def _score_chunk(c, t, bound, blocks, query, blk, ks, b64):
    """Token t's scores (``bound`` its columns, ``blocks`` its candidate
    blocks) of the tile holding ``blk``, and its block's best on a candidate
    producer; see the module."""
    lane = c["lane"]
    a = c["args"]
    r8 = lane % 8
    col = blk * TILE + r8
    terms = {}
    for h in range_constexpr(2):
        a64, wts = query[h]
        acc = fx.Vector.filled(4, 0.0, fx.Float32)
        for x in range_constexpr(4):
            acc = mfma_fp8(a64[x], b64[x], acc)
        for e in range_constexpr(4):
            terms[h, e] = (fx.max(acc[e] * ks, fx.Float32(0.0)), wts[e])
    live = col < bound
    score = fx.Float32(live.select(_head_sum(terms), fx.Float32(NEG)))
    if lane < 16:
        _st_dev(score, a["ilog"], t * a["lstride"] + col)
    if a["iprod"] != 0:
        # a block's best visible score: its 8 rows are 8 lanes
        best = fx.Float32(fx.isnan(score).select(fx.Float32(NEG), score))
        best = fx.Float32(live.select(best, fx.Float32(NEG)))
        best = butterfly(best, (1, 2, 4), fx.max)
        best = fx.min(best, fx.Float32(FLT_MAX))
        newest = blk == blocks - 1
        best = fx.Float32(newest.select(fx.Float32(float("inf")), best))
        if (lane < 16) & (r8 == 0) & (blk < blocks):
            _st_dev(best, a["ibmax"], t * a["bstride"] + blk)


@traced
def stage_iscore(c, lead, span, ch0, width, max_chunks):
    """Chunks [ch0, ch0 + width) of the ``span`` tokens from ``lead``, which
    read lead's tile table (``index_plan.score_tasks``): the keys loaded once,
    every (token, chunk) scored, then one drain and the chunks' flags."""
    tid, lane, wave = c["tid"], c["lane"], c["wave"]
    a = c["args"]
    # a thread a (token, chunk) flag; its bound loaded with the keys
    row = lead + tid // TASK_CHUNKS
    m = tid % TASK_CHUNKS
    row_bound = ld_i32(a["ibound"], fx.min(row, c["S"] - 1))
    keys = _chunk_keys(a, lead, ch0, width, wave, lane)
    for r in range(span):
        tok = lead + r
        tok_bound = ld_i32(a["ibound"], tok)
        query = _token_query(a, tok, lane)
        tok_blocks = ip.blocks(tok_bound)
        for k in range_constexpr(TASK_CHUNKS):
            if ip.task_writes(tok_bound, ch0, width, k):
                _score_chunk(c, tok, tok_bound, tok_blocks, query, *keys[k])
    who = (tid < span * TASK_CHUNKS) & ip.task_writes(row_bound, ch0, width, m)
    publish(c["put"], c["isf"], row * max_chunks + ch0 + m, 1, who)


@traced
def stage_ilist(c, t, j, raw, kind, level, bound_max, max_chunks):
    """List j of ``level`` for token t (``raw`` its column bound); see the
    module. Only the lists ``index_plan.list_count`` gives a token run."""
    tid = c["tid"]
    a = c["args"]
    k = ip.KINDS[kind]
    slots = ip.list_slots(kind, level, bound_max)
    bound = ip.kind_bound(kind, raw)
    if j < ip.list_count(bound, level, k):
        n = ip.list_inputs(bound, level, j, k)
        final = ip.is_final(bound, level, k)
        # the top level has one list, a token's last: it has no region above it
        to_final = True if level == ip.top_level(kind, bound_max) else final
        got = []
        if const_expr(level == 0):
            if const_expr(kind == "sel"):
                first, count = ip.CHUNKS_PER_SEG * j, ip.seg_chunks(bound, j)
            else:
                first, count = ip.block_chunks(raw, j)
            if tid < count:
                c["poll"]([(c["isf"], t * max_chunks + first + tid, 1)])
            gpu.barrier()
        else:
            below = ip.list_slots(kind, level - 1, bound_max)
            base = tid * PER_THREAD
            got = c["poll"](
                [
                    (
                        c[f"i{kind}{level - 1}"],
                        (
                            (t * below + j * ip.fan(k) + fx.min(base + d, n - 1) // k)
                            * k
                            + fx.min(base + d, n - 1) % k
                        )
                        * 2,
                        2,
                    )
                    for d in range(PER_THREAD)
                ]
            )
        src, stride = (
            (a["ilog"], a["lstride"]) if kind == "sel" else (a["ibmax"], a["bstride"])
        )

        def load(q, i):
            if const_expr(level == 0):
                # a quad stays in the row: strides are whole chunks / blocks
                v = fx.Vector(ld_dev(src, t * stride + j * ip.SEG + i, 4))
                return [fx.Int32(v[e]) for e in range(4)]
            return [got[4 * q + e][0] for e in range(4)]

        keys = c["sel_lds"]["keys"]
        first_i = tid * PER_THREAD

        def ident_of(d, i):
            return j * ip.SEG + i if level == 0 else got[d][1]

        # a selection's rows (``_selected_rows``), in flight during the passes
        def selected_rows():
            return [
                _selected_rows(c, t, ident_of(d, fx.min(first_i + d, n - 1)))
                for d in range(PER_THREAD)
            ]

        def emit(pos, d, i, rows):
            if const_expr(d is None):
                ident, bits = fx.Int32(-1), fx.Float32(NEG).bitcast(fx.Int32)
                picked = (fx.Int32(-1), fx.Int32(0))
            else:
                ident = ident_of(d, i)
                bits = key_bits(fx.ptr_load(keys + i))
                picked = rows[d] if kind == "sel" else None
            if to_final:
                _final(c, kind, t, pos, ident, picked)
            else:
                c["put_words"](
                    c[f"i{kind}{level}"], ((t * slots + j) * k + pos) * 2, [bits, ident]
                )

        lds_topk(
            n, k, load, emit, c["sel_lds"], tid, PER_THREAD,
            selected_rows if kind == "sel" else None,
        )  # fmt: skip
        if const_expr(kind == "sel"):  # noqa: SIM102
            if final:
                publish(c["put"], c["irdy"], t, 1, tid == 0)


def _selected_rows(c, t, ident):
    """Column ``ident`` of token t as a selection holds it (a REINDEX layer's
    compacted column lifted back to the row it stands for) and as the
    attention reads it: its pool row, through the block table (``attention.
    _key_row``'s arithmetic; every layer sharing this selection shares its KV
    owner, so its row). Loaded for every element before the top-k
    (``stage_ilist``): inside the winners' branches each load was a round trip
    in series, and the attention's keys wait on none."""
    a = c["args"]
    cand = fx.Int32(
        bo.buffer_load(
            rsrc(a["icand"]),
            t * ip.TOPK_BLOCKS + fx.max(ident, 0) // TILE,
            vec_width=1,
            dtype=T.i32,
        )
    )
    lifted = cand * TILE + fx.max(ident, 0) % TILE
    row = fx.Int32(((a["ilift"] != 0) & (ident >= 0)).select(lifted, ident))
    toff = fx.Int32(
        bo.buffer_load(rsrc(a["kmeta"]), t * KMETA + 2, vec_width=1, dtype=T.i32)
    )
    rpp = a["rows_per_page"]
    live = fx.max(row, 0)
    page = fx.Int32(
        bo.buffer_load(rsrc(a["table"]), toff + live // rpp, vec_width=1, dtype=T.i32)
    )
    return row, page * a["page_rows"] + a["main_off"] + live % rpp


def _final(c, kind, t, pos, ident, rows=None):
    a = c["args"]
    if const_expr(kind == "sel"):
        row, pool_row = rows
        _st_dev(row, a["isel"], t * ip.TOPK + pos)
        _st_dev(pool_row, a["irow"], t * ip.TOPK + pos)
    else:
        _st_dev(Int32(ident), a["icout"], t * ip.TOPK_BLOCKS + pos)


@traced
def run_indexer(c, s, bid, bound_max):
    """Every indexer stage of a selecting layer, the tasks of each laid out from
    the step's bounds (``index_plan``): only live tasks. The bounds were written
    before this launch: plain loads."""
    a = c["args"]
    max_chunks = ip.chunks(bound_max)
    bounds = [ld_i32(a["ibound"], t) for t in range(s)]
    if const_expr(c["index_fp4"]):
        run_scores4(c, s, bid, max_chunks)
    else:
        run_scores(c, s, bid, bounds, max_chunks)
    _run_tree(c, s, bid, "sel", bounds, bound_max, max_chunks)
    if a["iprod"] != 0:
        _run_tree(c, s, bid, "blk", bounds, bound_max, max_chunks)


@traced
def run_scores(c, s, bid, bounds, max_chunks):
    """The score stage's tasks (``index_plan.score_tasks``). A request's tokens
    share its tile table (key metadata word 2: its block table offset); a
    REINDEX layer's are each token's own candidates."""
    assert s * TASK_CHUNKS <= THREADS, s  # a thread a (token, chunk) flag
    a = c["args"]
    reindex = a["ilift"] != 0
    owners = [sel(reindex, t, ld_i32(a["kmeta"], t * KMETA + 2)) for t in range(s)]
    counts, spans, width_log2 = ip.score_tasks(bounds, owners, BLOCKS, TASK_CHUNKS_LOG2)
    width = Int32(1) << width_log2
    enter_stage("k2a.iscore")
    for task in range(first_task(bid, 0), ip.flat_total(counts), BLOCKS):
        lead, i = ip.flat_task(counts, task)
        stage_iscore(c, lead, pick(spans, lead), i << width_log2, width, max_chunks)


@traced
def _run_tree(c, s, bid, kind, bounds, bound_max, max_chunks):
    k = ip.KINDS[kind]
    for level in range_constexpr(ip.top_level(kind, bound_max) + 1):
        counts = [ip.list_count(ip.kind_bound(kind, b), level, k) for b in bounds]
        enter_stage(f"k2a.i{kind}{level}")
        for task in range(first_task(bid, 0), ip.flat_total(counts), BLOCKS):
            t, j = ip.flat_task(counts, task)
            stage_ilist(c, t, j, pick(bounds, t), kind, level, bound_max, max_chunks)
