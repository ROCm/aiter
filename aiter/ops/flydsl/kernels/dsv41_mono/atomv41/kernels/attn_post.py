# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K2a ``attn_post``: one V4.1 layer from the sparse decode attention to the MoE
input.

One launch a layer and rank, ``BLOCKS`` x ``THREADS``, S = ``tokens`` rows:

Task counts are TP4's (``Dims``: the rank's widths at the build's ``tp``):

    score    (S x 8 a head tile, 8 splits a task): the decode attention's scores
             and softmax statistics (``attention.py``) -> AM, AL, AP
    irq      (S x 16 a head tile, 32 columns of its heads a task): the attention's PV and
             split combine (``attention.py``), then the inverse RoPE of each
             head's tail, group-32 FP8 of the fp32 result (aiter
             ``inverse_rope_group_quant``: amax floored at 1e-8, the ceil code,
             the hardware scaled convert) -> mailbox XO
    wo_a     (64, 32 rows: two tiles, K split 4 each): the grouped FP8 GEMV (8
             heads a group) -> bf16 -> ATOM ``quantize_fp8`` of the 32 (one
             group: the ceil code, floor 1e-4) -> X8B, X8BS
    wo_b     (320, 16 rows): FP8 GEMV -> the rank's bf16 partial -> pushed to
             every rank's ATTN region (system scope)
    slice    (160): the TP partials of its 32 columns summed as the all-reduce
             the original path runs sums them (fp32, this rank's first, then
             rank + 1, + 2, ..; bf16) -- the attention output -- folded into the
             residual; then the FFN seam as in K1
    gate, norm: the FFN gates and ffn_norm -> ``normed`` (the MoE input)

A selecting layer's indexer (``index_score.py``) runs first in this launch;
before it stay the compressor and each token's window metadata
(``step_meta.write_step_meta``).
"""

from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import Int32, Int64

from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels import attn_pre as k1
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.attention import (
    COLS,
    GROUPS,
    only_live,
    pv_groups,
    score_tasks,
    stage_attn_pv,
    stage_attn_score,
    tile_head,
    tile_task,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.attention import (
    scratch_pairs as attention_pairs,
)

# helpers the kernel calls are imported by name: FlyDSL keys a build by the
# source of the same-directory functions it names, never through a module
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.attn_pre import (
    MAX_TOKENS,
    TILE,
    gemv_fp8_loads,
    gemv_fp8_mfmas,
    ld_bf,
    lds_row,
    plus,
    poll_copy,
    stage_gate,
    stage_norm,
    stage_slice,
    tile_rows,
    tile_tasks,
    token_tiles,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.debug import mailbox
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.dims import (
    HEAD_TILE,
    WOA_ROWS,
    Dims,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.index_score import (
    PER_THREAD as SEL_PER_THREAD,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.index_score import run_indexer
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.index_score import (
    scratch_pairs as index_pairs,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.topk import lds_words
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.sources import SOURCES
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.mx import FP8_MAX, clamp_fp8, code_ceil, pow2
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ops import (
    bf16_round,
    butterfly,
    fp8_pack4,
    kernel_symbol,
    lane_gather,
    ld_i32,
    row_sum,
    traced,
    xshfl,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ranks import peer_bases, sum_partials
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.stamps import stamp, stamp_begin, stamp_flush
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.sync import preg, sreg
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.build_key import key_tuple
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.execution import BLOCKS, THREADS, WAVES, first_task
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.layout import pair_layout
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.trace import enter_stage
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.runtime.abi import KernelAbi

HEAD_DIM, ROPE = k1.HEAD_DIM, k1.ROPE
HIDDEN = k1.HIDDEN
ROWS = k1.ROWS
# wo_a's K order: the first WOA_SPLIT_K waves of a 32-row task's two tiles take
# one contiguous K range each, summed in wave order. Picked for speed: the
# original's own split need not be matched
WOA_SPLIT_K = WAVES * ROWS // WOA_ROWS
_SEL = lds_words(SEL_PER_THREAD)
SEL_KEYS, SEL_HIST, SEL_BUF = _SEL["keys"], _SEL["hist"], _SEL["buf"]
WOB_TASKS = HIDDEN // ROWS  # 320
IRQ_FLOOR = 1e-8
QUANT_FLOOR = 1e-4

SCORE0 = 0
IRQ0 = 64  # past the score tasks (S x ``score_tasks`` = S x 8, S <= 8)
WOA0 = 160  # past the pv tasks (S x 16, S <= 6): its weights load during them
GATE0 = k1.GATE0
NORM0 = k1.NORM0
_CURRENT_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class AttnPostBuild:
    tokens: int
    # the indexer's largest column bound (max_model_len): its mailbox regions and
    # merge levels (``index_plan``); 0 builds no indexer
    index_bound_max: int = 0
    # the index plane is FP4: its scorer is ``index_score_fp4`` (``iqs``,
    # ``pscale`` and the FULL layers' task plan ``iplan``)
    index_fp4: bool = False
    # the TP size: every width a rank holds (``Dims``)
    tp: int = 4
    timeline: bool = field(default=False, metadata={"sym": "tl"})
    # ``ATOM_MONO_DEBUG``: bounded mailbox waits recording at this scratch
    # offset (``debug``); -1 is the normal build
    diag_off: int = -1


# ``timeline`` stamps: kernel start, then the end of each stage's task loop
TL_STAGES = (
    "index",
    "attn_score",
    "irq",
    "wo_a",
    "wo_b",
    "slice",
    "gate",
    "norm",
)
TL_POINTS = 1 + len(TL_STAGES)


ABI = KernelAbi(
    (
        "iact", "iq", "iw", "plane", "iqs", "pscale", "iplan", "itab", "itab_stride",
        "itab_len", "ibatch", "ishift", "ibound",
        "ilog", "lstride", "isel", "icand", "ilift", "iprod", "ibmax", "bstride",
        "icout", "irow", "sbound",
        "q", "pool", "sel", "topk", "kmeta", "table", "rows_per_page", "page_rows",
        "main_off", "ring_off", "ring_slots", "sink", "qk_scale", "pos", "cos", "sin",
        "woa", "woa_s", "wob", "wob_s",
        "res_in", "post_in", "comb_in", "pre_in", "hc_fn", "hc_scale", "hc_base",
        "ffn_w", "res_out", "post_out", "comb_out", "pre_out", "normed",
        "scratch", "sym", "peers", "rank", "layer", "tl",
    )
)  # fmt: skip


def scratch_layout(key: AttnPostBuild) -> dict[str, tuple[int, int]]:
    tokens, index_bound_max, d = key.tokens, key.index_bound_max, Dims(key.tp)
    base = k1.scratch_layout(tokens)
    pairs = {
        "xo": tokens * d.xo_words,
        "xos": tokens * d.heads * HEAD_DIM // 32,
        **attention_pairs(tokens, d),
        "x8b": tokens * d.yb_words,
        "x8bs": tokens * d.o_rows // 32,
        **(index_pairs(tokens, index_bound_max) if index_bound_max else {}),
    }
    out = {name: base[name] for name in ("lin", "pmix", "x8", "x8s")}
    start = max(o + n for o, n in base.values())
    return out | pair_layout(pairs.items(), start=start)


def scratch_bytes(key: AttnPostBuild) -> int:
    return max(o + n for o, n in scratch_layout(key).values())


def peer_bytes(tokens: int, tp: int) -> int:
    """ATTN: [source rank][token][hidden / 2] bf16 pairs."""
    return tp * tokens * HIDDEN // 2 * 8


def attn_region_pair(src, t, s, col):
    return (src * s + t) * (HIDDEN // 2) + col // 2


# ---------------------------------------------------------------- irq
@traced
def stage_attn_irq(c, task, groups):
    """Token t, ``groups`` column groups of 32 of a head tile's heads
    (``tile_task``): the decode attention's PV and combine (``stage_attn_pv``),
    then its inverse RoPE and group-32 FP8 quant, 256 threads (head, a pair of
    dims) each, a group at a time. Every group's RoPE table loads issue before
    the PV: they do not wait on it, and a group's would sit between groups."""
    t, tile, g0 = tile_task(task, GROUPS // groups, c["d"].head_tiles)
    ropes = irq_rope_loads(c, t, [g0 * groups + i for i in range(groups)])
    stage_attn_pv(c, task, groups)
    for i in range_constexpr(groups):
        irq_group(
            c, t, tile, g0 * groups + i, c["att"]["out"] + i * HEAD_TILE * COLS,
            ropes[i],
        )  # fmt: skip


def irq_rope_loads(c, t, dcs):
    """This thread's (cos, sin) of its dim pair in each of column groups
    ``dcs`` of token t (1, 0 off the rotated tail): loads only."""
    a = c["args"]
    tt = c["tid"] % (COLS // 2)
    pos = ld_i32(a["pos"], 2 * t)
    out = []
    for dc in dcs:
        d = dc * COLS + tt * 2
        live = d >= HEAD_DIM - ROPE
        jj = fx.max((d - (HEAD_DIM - ROPE)) // 2, 0)
        out.append(
            (
                live.select(ld_bf(a["cos"], pos * k1.HALF + jj), fx.Float32(1.0)),
                live.select(ld_bf(a["sin"], pos * k1.HALF + jj), fx.Float32(0.0)),
            )
        )
    return out


@traced
def irq_group(c, t, tile, dc, out, rope):
    """Token t, columns 32 dc .. of a head tile's heads (PV out [head][column]
    at ``out``, the dims' ``rope`` (cos, sin): ``irq_rope_loads``) -> inverse
    RoPE, group-32 FP8 quant -> XO / XOS."""
    tid, dims = c["tid"], c["d"]
    mine = tid < HEAD_TILE * COLS // 2
    h = fx.min(tid, HEAD_TILE * COLS // 2 - 1) // (COLS // 2)
    head, live = tile_head(dims, tile, h)
    mine = only_live(mine, live)
    tt = tid % (COLS // 2)
    d = dc * COLS + tt * 2
    out = out + h * COLS
    e = fx.ptr_load(out + tt * 2)
    o = fx.ptr_load(out + (tt * 2 + 1))
    cs, sn = rope
    # inverse rotation, kept fp32 (bf16 products are exact: any contraction agrees)
    e1 = e * cs + o * sn
    o1 = o * cs - e * sn
    amax = butterfly(fx.max(abs(e1), abs(o1)), (1, 2, 4, 8), fx.max)
    code = code_ceil(fx.max(amax, fx.Float32(IRQ_FLOOR)) * fx.Float32(1.0 / FP8_MAX))
    inv = pow2(254 - code)
    # this thread's two bytes and its neighbour's make a word (4 dims)
    q0, q1 = clamp_fp8(e1 * inv), clamp_fp8(o1 * inv)
    n0, n1 = xshfl(q0, 1), xshfl(q1, 1)
    w = fp8_pack4(q0, q1, n0, n1)
    if mine & (tt % 2 == 0):
        c["put_words"](c["xo"], t * dims.xo_words + (head * HEAD_DIM + d) // 4, [w])
    if mine & (tt % 16 == 0):
        c["put"](
            c["xos"],
            t * (dims.heads * HEAD_DIM // 32) + (head * HEAD_DIM + d) // 32,
            code,
        )


@traced
def load_rows(
    c, words_mb, codes_mb, stride_w, off_w, words, stride_g, off_g, groups, xl, xsl,
    t0=0, n=None,
):  # fmt: skip
    """Tokens t0 .. t0 + n's (every token's by default) slice ``[off, off +
    words)`` of a mailbox of rows ``stride`` words apart (and its codes), into
    LDS rows at the ``lds_row`` stride. ``n`` traced (a tile a CTA picks at run
    time): a TILE-row copy of which only n rows are polled."""
    n = c["S"] if n is None else n
    rows = n if isinstance(n, int) else TILE
    poll_copy(
        c,
        c["tid"],
        THREADS,
        [
            (
                words_mb,
                lambda u: plus(t0, u // words) * stride_w + off_w + u % words,
                rows * words,
                xl,
                words,
            ),
            (
                codes_mb,
                lambda u: plus(t0, u // groups) * stride_g + off_g + u % groups,
                rows * groups,
                xsl,
                groups,
            ),
        ],
        live=None if isinstance(n, int) else [n * words, n * groups],
    )
    gpu.barrier()


@traced
def stage_woa(c, task, ops, t0=0, n=None):
    """Rows 32 task .. of wo_a (two tiles, ``woa_loads``' operands) -> bf16 ->
    ATOM ``quantize_fp8`` of the 32 (one group: amax floored 1e-4, the ceil
    code) -> X8B / X8BS, wo_b's input; the token tile from ``t0`` (its x in
    LDS)."""
    tid, red, d = c["tid"], c["red"], c["d"]
    n = c["S"] if n is None else n
    split = WOA_SPLIT_K
    gemv_fp8_mfmas(c, d.group_k, c["xl"], c["xsl"], red, ops, split, tiled=True, rows=n)
    gpu.barrier()
    # thread (t, row): a token's 32 rows are 32 lanes of one wave
    mine = tid < WOA_ROWS * n
    tl = fx.min(tid // WOA_ROWS, n - 1)
    t = plus(t0, tl)
    row = tid % WOA_ROWS
    tile = row // ROWS
    y = fx.Float32(
        row_sum(red, row % ROWS, tl, [tile * split + w for w in range(split)]).to(
            fx.BFloat16
        )
    )
    amax = fx.max(butterfly(abs(y), (1, 2, 4, 8, 16), fx.max), fx.Float32(QUANT_FLOOR))
    code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
    q = clamp_fp8(y / pow2(code))
    # rows row .. row + 3 (row a multiple of 4) make a word
    lane = c["lane"]
    w = fp8_pack4(
        q,
        *[
            lane_gather(q.bitcast(fx.Int32), fx.min(lane + k, 63)).bitcast(fx.Float32)
            for k in (1, 2, 3)
        ],
    )
    if mine & (row % 4 == 0):
        c["put_words"](c["x8b"], t * d.yb_words + (task * WOA_ROWS + row) // 4, [w])
    if mine & (row == 0):
        c["put"](c["x8bs"], t * (d.o_rows // 32) + task, code)
    gpu.barrier()


@traced
def woa_tile(c, task, ops, t0, n):
    """wo_a task ``task`` (``woa_loads``' ``ops``) on tokens t0 .. t0 + n: its
    group's attention output rows into LDS, then ``stage_woa``."""
    d = c["d"]
    g = task // (d.woa_tasks // d.groups)
    load_rows(
        c, c["xo"], c["xos"], d.xo_words, g * d.group_k // 4, d.group_k // 4,
        d.heads * HEAD_DIM // 32, g * d.group_k // 32, d.group_k // 32,
        c["xl"], c["xsl"], t0, n,
    )  # fmt: skip
    stage_woa(c, task, ops, t0, n)


def woa_loads(c, task):
    a = c["args"]
    split = WOA_SPLIT_K
    return gemv_fp8_loads(
        c, a["woa"], a["woa_s"], c["d"].group_k, task * (WOA_ROWS // ROWS), split,
        tiled=True,
    )  # fmt: skip


def wob_loads(c, task):
    a = c["args"]
    return gemv_fp8_loads(c, a["wob"], a["wob_s"], c["d"].o_rows, task)


@traced
def stage_wob(c, task, ops, t0=0, n=None):
    """Rows 16 task .. of wo_b (``wob_loads``' operands) -> this rank's bf16
    partial, pushed to every rank; the token tile from ``t0`` (its x in LDS)."""
    s, tid, red, d = c["S"], c["tid"], c["red"], c["d"]
    n = s if n is None else n
    gemv_fp8_mfmas(c, d.o_rows, c["xl"], c["xsl"], red, ops, rows=n)
    gpu.barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        col = task * ROWS + 2 * rp
        v0 = row_sum(red, 2 * rp, tl)
        v1 = row_sum(red, 2 * rp + 1, tl)
        for p in range_constexpr(d.tp):
            dst = preg(c["peer_addr"](p), 0, "attn")
            c["put_bf"](dst, 2 * attn_region_pair(c["rank"], t, s, col), [v0, v1])
    gpu.barrier()


@traced
def stage_reduce(c, task):
    """The attention output at this slice's 32 columns: the TP ranks' partials
    in the all-reduce's order (``sum_partials``), in fp32 -> bf16 ->
    ``pend_lds``."""
    s, tid = c["S"], c["tid"]
    for pas in range_constexpr(0, s * k1.COLS // 2, THREADS):
        reduce_attn_pair(c, task, plus(pas, tid))
    gpu.barrier()


@traced
def reduce_attn_pair(c, task, idx):
    """Thread ``idx``'s column pair of token idx / 16 (``stage_reduce``)."""
    s = c["S"]
    c0 = task * k1.COLS
    if idx < s * k1.COLS // 2:
        t = idx // (k1.COLS // 2)
        col = c0 + 2 * (idx % (k1.COLS // 2))
        own = preg(c["sym"], 0, "attn")
        acc0, acc1 = sum_partials(
            c["poll"],
            own,
            lambda src: attn_region_pair(src, t, s, col),
            c["d"].tp,
        )
        lds = c["pend_lds"] + (t * k1.COLS + col - c0)
        fx.ptr_store(bf16_round(acc0), lds)
        fx.ptr_store(bf16_round(acc1), lds + 1)


def attn_post_args(*values) -> dict:
    """K2a's kernel arguments by name (``ABI`` order, less the shared tail)."""
    names = ABI.names[: ABI.names.index("scratch")]
    assert len(values) == len(names)
    out = dict(zip(names, values))
    out["attn_w"] = out.pop("ffn_w")
    return out


def lds_struct(name, fields):
    """An ``fx.struct`` of ``fields`` (name -> array type), in that order."""
    return fx.struct(
        type(name, (), {"__annotations__": fields, "__module__": __name__})
    )


def lds_union(name, members):
    """An ``fx.union`` of ``members`` (name -> struct)."""
    return fx.union(
        type(name, (), {"__annotations__": members, "__module__": __name__})
    )


def attn_post_smem(s, timeline, bound_max, d) -> dict:
    """K2a's LDS, a struct a stage: the members of a union, so only one stage's
    buffers take space at a time (``run_attn_post`` fences each hand-over).
    Each opens with what every stage shares at the same offsets -- ``red`` and
    the stamp record (``timeline``)."""
    shared = {
        "red": fx.Array[fx.Float32, k1.WAVES * 2 * 64 * 4, 16],
        "tls": fx.Array[fx.Int64, TL_POINTS if timeline else 1, 16],
    }
    stages = {
        "sel": {
            "sk": fx.Array[fx.Int32, SEL_KEYS if bound_max else 1, 16],
            "sh": fx.Array[fx.Int32, SEL_HIST, 16],
            "sb": fx.Array[fx.Int32, SEL_BUF, 16],
        },
        "attn": {
            "at_tree": fx.Array[fx.Float32, 8 * 64 * 8, 16],
            "at_alpha": fx.Array[fx.Float32, 64 * HEAD_TILE, 16],
            "at_stat": fx.Array[fx.Float32, 2 * HEAD_TILE, 16],
            "at_out": fx.Array[
                fx.Float32, pv_groups(s, d.head_tiles.count) * HEAD_TILE * COLS, 16
            ],
            "at_kv": fx.Array[fx.Int32, k1.WAVES * 16 * COLS // 2, 16],
        },
        # wo_a's group row, and wo_b's input (``yb_words``, not longer)
        "gemv": {
            "xl": fx.Array[fx.Int32, tile_rows(s) * lds_row(d.group_k // 4), 16],
            "xsl": fx.Array[fx.Int32, tile_rows(s) * lds_row(d.group_k // 32), 16],
        },
        "slice": {
            "rl": fx.Array[fx.Float32, s * k1.KT, 16],
            "fl": fx.Array[fx.Float32, k1.MIX * k1.KT, 16],
            "pl": fx.Array[fx.Float32, s * k1.COLS, 16],
        },
    }
    return {
        f"k2a_{name}": lds_struct(f"K2a{name.title()}Lds", {**shared, **fields})
        for name, fields in stages.items()
    }


def attn_post_context(key, s, lds, mb, bases, layout, scratch, args, rank, sym):
    """K2a's stage context: LDS (``attn_post_smem``'s member views by name),
    mailbox, scratch regions, arguments."""
    tid = fx.thread_idx.x
    sl, gm, at, se = (lds[f"k2a_{n}"] for n in ("slice", "gemv", "attn", "sel"))
    c = {
        "S": s, "tid": tid, "bid": fx.block_idx.x, "lane": tid % 64,
        "wave": tid // 64,
        "rl": sl.rl.ptr, "fl": sl.fl.ptr, "red": gm.red.ptr,
        "xl": gm.xl.ptr, "xsl": gm.xsl.ptr, "pend_lds": sl.pl.ptr,
        "att": {
            "tree": at.at_tree.ptr, "alpha": at.at_alpha.ptr,
            "stat": at.at_stat.ptr, "out": at.at_out.ptr, "kv": at.at_kv.ptr,
        },
        "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words,
        "poll": mb.poll, "peer_addr": lambda p: bases[p], "rank": rank, "sym": sym,
        "fold": True, "index": False, "aux": False, "ffn": True,
        "index_fp4": key.index_fp4,
        "args": args, "d": Dims(key.tp),
    }  # fmt: skip
    for region, (off, _) in layout.items():
        c[region] = sreg(scratch, off, region)
    c["sel_lds"] = {"keys": se.sk.ptr, "hist": se.sh.ptr, "buf": se.sb.ptr}
    return c


@traced
def run_attn_post(c, key, bid, tls, tl0, after_norm=None):
    """K2a's stages, in order; ``timeline`` stamps from point ``tl0`` + 1.
    ``after_norm(t)``: once token t's ``normed`` row is written (a launch that
    also reads it: ``layer_post``)."""
    s, timeline, tid, d = key.tokens, key.timeline, c["tid"], c["d"]
    bound_max = key.index_bound_max
    tiles = d.head_tiles.count
    if const_expr(bound_max > 0):  # noqa: SIM102
        if c["args"]["iact"] != 0:
            run_indexer(c, s, bid, bound_max)
    # the stages' LDS is one union (``attn_post_smem``): a barrier hands it over
    gpu.barrier()
    stamp(timeline, tls, tid, tl0 + 1)
    enter_stage("k2a.attn_score")
    per = score_tasks(s, tiles)
    for task in range(first_task(bid, SCORE0), s * tiles * per, BLOCKS):
        stage_attn_score(c, task, per)
    stamp(timeline, tls, tid, tl0 + 2)
    enter_stage("k2a.irq")
    groups = pv_groups(s, tiles)
    for task in range(first_task(bid, IRQ0), s * tiles * GROUPS // groups, BLOCKS):
        stage_attn_irq(c, task, groups)
    gpu.barrier()
    stamp(timeline, tls, tid, tl0 + 3)
    enter_stage("k2a.wo_a")
    # a CTA's wo_a task (at most one): its weights in flight before the
    # attention output's wait
    assert d.woa_tasks <= BLOCKS
    tiles = token_tiles(s)
    if const_expr(len(tiles) == 1):
        woa_task = first_task(bid, WOA0)
        if woa_task < d.woa_tasks:
            woa_ops = woa_loads(c, woa_task)
            woa_tile(c, woa_task, woa_ops, 0, s)
    else:
        # a (task, token tile) a unit, a CTA each: a task's tiles in series on
        # one CTA left three quarters of the CTAs idle through wo_a (its weights
        # read again a tile: L2 hits)
        for unit in range(first_task(bid, WOA0), d.woa_tasks * len(tiles), BLOCKS):
            woa_task = unit % d.woa_tasks
            t0 = unit // d.woa_tasks * TILE
            woa_ops = woa_loads(c, woa_task)
            woa_tile(c, woa_task, woa_ops, t0, fx.min(fx.Int32(TILE), s - t0))
    stamp(timeline, tls, tid, tl0 + 4)
    enter_stage("k2a.wo_b")
    assert d.yb_words <= d.group_k // 4  # xl holds it
    assert BLOCKS < WOB_TASKS <= 2 * BLOCKS
    if const_expr(len(tiles) == 1):
        # every CTA has one or two wo_b tasks: both tasks' weights in flight
        # before y's wait (a CTA with one loads its own again: L2 hits)
        wob_task = first_task(bid, WOA0 + d.woa_tasks)
        wob_second = wob_task + BLOCKS
        has_second = wob_second < WOB_TASKS
        wob_ops = wob_loads(c, wob_task)
        wob_ops2 = wob_loads(c, has_second.select(wob_second, wob_task))
        load_rows(
            c, c["x8b"], c["x8bs"], d.yb_words, 0, d.yb_words,
            d.o_rows // 32, 0, d.o_rows // 32, c["xl"], c["xsl"], 0, s,
        )  # fmt: skip
        stage_wob(c, wob_task, wob_ops, 0, s)
        if has_second:
            stage_wob(c, wob_second, wob_ops2, 0, s)
    else:
        tile, wob_tasks = tile_tasks(
            bid, WOA0 + d.woa_tasks * len(tiles), len(tiles), WOB_TASKS
        )
        wob_ops = [wob_loads(c, fx.min(t, WOB_TASKS - 1)) for t in wob_tasks]
        t0 = tile * TILE
        n = fx.min(fx.Int32(TILE), s - t0)
        load_rows(
            c, c["x8b"], c["x8bs"], d.yb_words, 0, d.yb_words,
            d.o_rows // 32, 0, d.o_rows // 32, c["xl"], c["xsl"], t0, n,
        )  # fmt: skip
        for k in range_constexpr(len(wob_tasks)):
            if wob_tasks[k] < WOB_TASKS:
                stage_wob(c, wob_tasks[k], wob_ops[k], t0, n)
    stamp(timeline, tls, tid, tl0 + 5)
    enter_stage("k2a.slice")
    for task in range(first_task(bid, 0), k1.SLICES, BLOCKS):
        stage_reduce(c, task)
        stage_slice(c, task)
    stamp(timeline, tls, tid, tl0 + 6)
    enter_stage("k2a.gate")
    for t in range(first_task(bid, GATE0), s, BLOCKS):
        stage_gate(c, t)
    stamp(timeline, tls, tid, tl0 + 7)
    enter_stage("k2a.norm")
    for t in range(first_task(bid, NORM0), s, BLOCKS):
        stage_norm(c, t)
        if const_expr(after_norm is not None):
            after_norm(t)
    stamp(timeline, tls, tid, tl0 + 8)


def build_attn_post(key: AttnPostBuild):
    s = key.tokens
    timeline = key.timeline
    assert 1 <= s <= MAX_TOKENS
    bound_max = key.index_bound_max
    layout = scratch_layout(key)
    keyed = key_tuple(key, SOURCES)
    members = attn_post_smem(s, timeline, bound_max, Dims(key.tp))
    Smem = lds_union("K2aLds", members)

    name = kernel_symbol(
        "v41_attn_post",
        s=s,
        tl=int(timeline),
        ix=int(bound_max > 0),
        i4=int(key.index_fp4),
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_post(
        iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64,
        pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64,
        lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32,
        ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, sbound: Int64,
        q: Int64, pool: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32,
        ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64,
        pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        normed: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
        layer: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        smem = fx.SharedAllocator().allocate(Smem)
        lds = {name: getattr(smem, name).peek() for name in members}
        args = attn_post_args(
            iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len,
            ibatch, ishift, ibound, ilog, lstride,
            isel, icand, ilift, iprod, ibmax, bstride, icout, irow, sbound,
            q, pool, sel, topk, kmeta, table, rows_per_page, page_rows, main_off,
            ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob,
            wob_s, res_in, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, ffn_w,
            res_out, post_out, comb_out, pre_out, normed,
        )  # fmt: skip
        c = attn_post_context(
            key, s, lds, mailbox(layer, scratch, key.diag_off, layout),
            peer_bases(peers, key.tp), layout, scratch, args,
            rank, sym,
        )  # fmt: skip
        tls = lds["k2a_gemv"].tls.ptr
        stamp_begin(timeline, tls, tid, TL_POINTS)
        run_attn_post(c, key, bid, tls, 0)
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)
        _ = keyed

    @flyc.jit
    def launch(
        iact: Int32, iq: Int64, iw: Int64, plane: Int64, iqs: Int64,
        pscale: Int64, iplan: Int64, itab: Int64,
        itab_stride: Int32, itab_len: Int32, ibatch: Int64, ishift: Int32,
        ibound: Int64, ilog: Int64,
        lstride: Int32, isel: Int64, icand: Int64, ilift: Int32, iprod: Int32,
        ibmax: Int64, bstride: Int32, icout: Int64, irow: Int64, sbound: Int64,
        q: Int64, pool: Int64, sel: Int64, topk: Int32, kmeta: Int64, table: Int64,
        rows_per_page: Int32, page_rows: Int32, main_off: Int32, ring_off: Int32,
        ring_slots: Int32, sink: Int64,
        qk_scale: Int32, pos: Int64, cos: Int64, sin: Int64, woa: Int64, woa_s: Int64,
        wob: Int64, wob_s: Int64, res_in: Int64, post_in: Int64, comb_in: Int64,
        pre_in: Int64, hc_fn: Int64, hc_scale: Int64, hc_base: Int64, ffn_w: Int64,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        normed: Int64, scratch: Int64, sym: Int64, peers: Int64, rank: Int32,
        layer: Int32, tl: Int64, stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        attn_post(
            iact, iq, iw, plane, iqs, pscale, iplan, itab, itab_stride, itab_len,
            ibatch, ishift, ibound, ilog, lstride,
            isel, icand, ilift, iprod, ibmax, bstride, icout, irow, sbound,
            q, pool, sel, topk, kmeta, table, rows_per_page, page_rows, main_off,
            ring_off, ring_slots, sink, qk_scale, pos, cos, sin, woa, woa_s, wob,
            wob_s, res_in, post_in, comb_in,
            pre_in, hc_fn, hc_scale, hc_base, ffn_w, res_out, post_out, comb_out,
            pre_out, normed, scratch, sym, peers, rank, layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(attn_post, launch)
    return launch
