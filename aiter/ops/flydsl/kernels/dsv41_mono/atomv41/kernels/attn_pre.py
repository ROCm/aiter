# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K1 ``attn_pre``: one V4.1 layer from the attention seam to the rotated query.

One launch a layer and rank, ``BLOCKS`` CTAs x ``THREADS`` threads, S = ``tokens``
rows (a DSpark verify step). Stages, each a task table placed round the CTAs and
handed off through the tagged mailbox (device scope; nothing crosses GPUs here):

    slice   (160, 32 hidden columns x 4 streams each): the owed post folded into
            the residual (R_new, bf16 out), the collapsed layer input (mailbox
            LIN) and the partial mix projection / sum of squares (mailbox PMIX)
    gate    (S): the 160 partials -> post / comb (Sinkhorn) and the next pre
    norm    (S): attn_norm -> bf16, then quantize_fp8 of it (mailbox X8)
    wqkv_a  (112, 16 rows): FP8 GEMV -> the q latent | kv row (mailbox QKV)
    qkv     (S): q_norm -> MXFP8 (mailbox QX8); kv_norm -> RoPE -> FP8 QAT ->
            the window ring row
    wq_b    (the rank's heads x 512 / 16, 16 rows: 512 at TP4): FP8 GEMV -> bf16
            -> RoPE on each head's tail -> q out

Every elementwise step follows the original seam's rounding (aiter's
``mhc_fused_post_pre_delayed_rmsnorm``, then ATOM's ``quantize_fp8``); the mix
projection and the sums of squares, split-K reductions, differ only in their
split and order.
"""

import math
from dataclasses import dataclass, field

import flydsl.compiler as flyc
import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from aiter.ops.flydsl.kernels.dsv41_mono.atomv41 import index_plan as ip
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.debug import mailbox
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.dims import HEAD_DIM, Dims
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.index_query import (
    IDX_ROPE,
    IDX_ROWS,
    IW_TASKS,
    stage_iquant,
    stage_iquant_fp4,
    stage_iw,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.index_query import (
    scratch_pairs as indexer_pairs,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.kernels.mx import code_round, fp8_value
from aiter.ops.flydsl.kernels.dsv41_mono.atomv41.sources import SOURCES
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.mx import (
    FP8_MAX,
    UNIT_SCALE,
    clamp_fp8,
    code_ceil,
    mfma_scaled,
    pow2,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ops import (
    CM_NT,
    batched_rounds,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    div_rn,
    fp8_pack4,
    fresh,
    hw_exp2,
    hw_rcp,
    hw_rsq,
    kernel_symbol,
    lane_gather,
    ld_i32,
    row_sum,
    rsrc,
    traced,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.stamps import stamp, stamp_begin, stamp_flush
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.sync import POLL_MAX, sreg
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.build_key import key_tuple
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.execution import BLOCKS, THREADS, WAVES, first_task
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.layout import pair_layout
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.trace import enter_stage
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.runtime.abi import KernelAbi

HC = 4
HIDDEN = 5120
K_MIX = HC * HIDDEN
MIX = HC * (HC + 2)  # 24 projections: pre 4 | post 4 | comb 16
Q_RANK, KV_DIM = 1280, 512
QKV_ROWS = Q_RANK + KV_DIM  # 1792
ROPE = 64
HALF = ROPE // 2
# the index query rotates with the attention's RoPE table (``stage_iq``)
assert IDX_ROPE == ROPE
EPS = 1e-20
HC_EPS = 1e-6
SINKHORN = 20
POST_MULT = 2.0
LOG2E = 1.0 / math.log(2.0)

COLS = 32  # hidden columns a slice task
SLICES = HIDDEN // COLS  # 160
KT = HC * COLS  # projection columns a slice task
PART = MIX + 1  # a slice's partials a token: the mixes, then the sum of squares
ROWS = 16  # GEMV rows a task
TILE = 16  # an MFMA's N: the tokens a GEMV pass takes (``token_tiles``)
MAX_TOKENS = 48  # a step's rows at most
WQKV_TASKS = QKV_ROWS // ROWS  # 112
X8_WORDS = HIDDEN // 4  # fp8 words of a normed row
QX8_WORDS = Q_RANK // 4

# every stage's first CTA; a CTA runs its tasks of each stage in this order
GATE0 = SLICES
NORM0 = GATE0 + 8
WQKV0 = NORM0 + 8
QKV0 = (WQKV0 + WQKV_TASKS) % BLOCKS
WQB0 = QKV0 + 8
IW0 = 0
IQ_TASKS = IDX_ROWS // ROWS  # 256
IQ0 = IW_TASKS
IQUANT0 = 0

_CURRENT_STREAM = fx.Stream(None)


@dataclass(frozen=True)
class AttnPreBuild:
    tokens: int
    # the seam folds the owed post of the previous sublayer (every layer but 0 and
    # the Engram ones, whose residual arrives settled)
    fold: bool = True
    # FULL / REINDEX layers, whose normed row and q latent feed the indexer and
    # the compressor: the bf16 ``normed`` row out (and NB), the q latent's MXFP8
    # pair out, and the indexer's q / weights projections
    feeds_index: bool = False
    # the index plane is FP4: the indexer query quantized as its scorer reads
    # it (``stage_iquant_fp4``)
    index_fp4: bool = False
    # DSpark tap: the mean over the hc copies of the settled residual -> ``aux``
    aux: bool = False
    # the TP size: every width a rank holds (``Dims``)
    tp: int = 4
    timeline: bool = field(default=False, metadata={"sym": "tl"})
    # ``ATOM_MONO_DEBUG``: bounded mailbox waits recording at this scratch
    # offset (``debug``); -1 is the normal build
    diag_off: int = -1


# ``timeline`` stamps: kernel start, then the end of each stage's task loop
TL_STAGES = ("slice", "gate", "norm", "wqkv_a", "qkv", "wq_b", "index")
TL_POINTS = 1 + len(TL_STAGES)


ABI = KernelAbi(
    (
        "res_in", "pend", "post_in", "comb_in", "pre_in",
        "hc_fn", "hc_scale", "hc_base", "attn_w", "wqkv", "wqkv_s", "qn_w", "kvn_w",
        "wqb", "wqb_s", "cos", "sin", "pos", "ring", "ring_rows", "ring_off",
        "res_out", "post_out", "comb_out", "pre_out", "q_out", "normed", "qr_out",
        "qrs_out", "aux", "iq_w", "iq_ws", "iw_w", "iq_out", "iqs_out", "iw_out",
        "scratch", "layer", "tl",
    )
)  # fmt: skip


def scratch_layout(tokens: int) -> dict[str, tuple[int, int]]:
    """Mailbox region -> (byte offset, bytes) in K1's scratch; 8 B a pair."""
    pairs = {
        "lin": tokens * HIDDEN // 2,  # bf16 pairs
        "pmix": SLICES * tokens * PART,
        "x8": tokens * X8_WORDS,
        "x8s": tokens * HIDDEN // 32,
        "qkv": tokens * QKV_ROWS,
        "qx8": tokens * QX8_WORDS,
        "qx8s": tokens * Q_RANK // 32,
        **indexer_pairs(tokens, HIDDEN),
    }
    return pair_layout(pairs.items())


def scratch_bytes(tokens: int) -> int:
    return sum(n for _, n in scratch_layout(tokens).values())


def fma(a, b, c):
    return fx.Float32(fmath.fma(a, b, c))


def sigmoid(z):
    """Triton's ``tl.sigmoid``: 1 / (1 + exp2(-z log2(e)))."""
    return fx.Float32(1.0) / (fx.Float32(1.0) + hw_exp2(z * fx.Float32(-LOG2E)))


def ld_f32(ptr, i):
    return fx.Float32(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.f32))


def ld_bf(ptr, i):
    return fx.Float32(
        fx.BFloat16(bo.buffer_load(rsrc(ptr), i, vec_width=1, dtype=T.bf16))
    )


def lane_f32(v, src):
    return lane_gather(v.bitcast(fx.Int32), src).bitcast(fx.Float32)


def nth_task(first, i):
    """A CTA's i-th task of a stage whose first is ``first`` (every BLOCKS-th)."""
    return first if i == 0 else first + BLOCKS * i


def token_tiles(s):
    """A step's tokens in GEMV passes: [(first token, tokens)]. A GEMV loads its
    weights once and runs every pass on them, an x tile in LDS at a time."""
    return [(t0, min(TILE, s - t0)) for t0 in range(0, s, TILE)]


def tile_rows(s):
    """The LDS rows an x tile holds."""
    return min(s, TILE)


# a token's LDS row of MXFP8 words / E8M0 codes, 16 of which a MFMA B operand
# gathers, is padded by 16 B: at a multiple of 64 words they share a bank group
LDS_PAD = 4


def lds_row(words):
    return words + LDS_PAD


def lds_at(u, row):
    """Element u of a span of rows ``row`` wide, at the ``lds_row`` stride."""
    return u + u // row * LDS_PAD


def lds_tail(k):
    """Words a K = ``k`` row's last K step reads past its ``lds_row`` (selected
    out): the last row's buffer must hold them."""
    return max(0, 32 * -(-k // 128) - lds_row(k // 4))


def plus(t0, v):
    """Token ``v`` of the tile from ``t0`` (a first tile adds nothing; ``t0``
    traced: a tile a CTA picks at run time)."""
    return v if isinstance(t0, int) and t0 == 0 else v + t0


def tile_tasks(bid, base, ntiles, ntasks):
    """Past one token tile, a stage of ``ntasks`` tasks each run on a token
    tile: this CTA's tile (CTAs placed from ``base``) and its tasks, every
    ``per_tile``-th of those the tile's CTAs share -- a tile's x polled once a
    CTA and every task's weights in flight together, where a (task, tile) a
    unit polled a tile again for each of a CTA's units. A task from past
    ``ntasks`` is not run (the caller's ``< ntasks``)."""
    b = first_task(bid, base)
    tile = b % ntiles
    per_tile = (BLOCKS - tile + ntiles - 1) // ntiles
    most = -(-ntasks // (BLOCKS // ntiles))
    return tile, [b // ntiles + k * per_tile for k in range(most)]


def pend_value(c, t, col, cc):
    """The owed post's sublayer output at (token t, column col): read from
    ``pend``, or where a caller staged it (``pend_lds``, [token][32] of this
    task's columns: a reduction it did first)."""
    if const_expr(c.get("pend_lds") is None):
        return ld_bf(c["args"]["pend"], t * HIDDEN + col)
    return fx.ptr_load(c["pend_lds"] + (t * COLS + cc))


# ---------------------------------------------------------------- slice
@traced
def slice_input(c, task, idx):
    """Thread ``idx``'s pair of the layer input (token idx / 16, columns
    32 task + 2 (idx % 16) ..) and the drafter's tap."""
    s, rl = c["S"], c["rl"]
    a = c["args"]
    c0 = task * COLS
    if idx < s * COLS // 2:
        t = idx // (COLS // 2)
        cc = 2 * (idx % (COLS // 2))
        outs = []
        for d in range_constexpr(2):
            acc = fx.Float32(0.0)
            for h in range_constexpr(HC):
                rb = fx.ptr_load(rl + (t * KT + h * COLS + cc + d))
                acc = acc + ld_f32(a["pre_in"], t * HC + h) * rb
            outs.append(acc)
        c["put_bf"](c["lin"], t * HIDDEN + c0 + cc, outs)
        if const_expr(c["aux"]):
            # the drafter's tap: torch mean over the bf16 streams (sum in order, / 4)
            for d in range_constexpr(2):
                acc = fx.Float32(0.0)
                for h in range_constexpr(HC):
                    acc = acc + fx.ptr_load(rl + (t * KT + h * COLS + cc + d))
                bo.buffer_store(
                    (acc * fx.Float32(1.0 / HC)).to(fx.BFloat16),
                    rsrc(a["aux"]),
                    t * HIDDEN + c0 + cc + d,
                )


def slice_at(c0, e):
    """Element e of a slice task at column c0: (token, stream, hidden column)."""
    return e // KT, (e % KT) // COLS, c0 + e % COLS


def slice_loads(c, c0, e):
    """Element e's loads of ``stage_slice`` (e clamped into the step's)."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(not c["fold"]):
        # a settled residual: read as it is, and left where it is
        return ld_bf(a["res_in"], (t * HC + o) * HIDDEN + col)
    r = [ld_bf(a["res_in"], (t * HC + h) * HIDDEN + col) for h in range(HC)]
    cm = [ld_f32(a["comb_in"], (t * HC + h) * HC + o) for h in range(HC)]
    return r, cm, ld_f32(a["post_in"], t * HC + o)


def slice_row(c, c0, e, got):
    """Element e of ``stage_slice``'s R_new from ``slice_loads``' values."""
    a = c["args"]
    t, o, col = slice_at(c0, e)
    if const_expr(c["fold"]):
        r, cm, post = got
        # aiter's mhc_fused_post_pre_delayed_rmsnorm: ((post x + c0 r0)
        # + c1 r1) + c2 r2, each product rounded, then fma(c3, r3, ..)
        v = post * pend_value(c, t, col, e % COLS) + (r[0] * cm[0])
        for h in range_constexpr(1, HC - 1):
            v = v + cm[h] * r[h]
        # bf16 R_new is what the collapse, the sums of squares and the
        # mix projection all read
        v = bf16_round(fma(cm[HC - 1], r[HC - 1], v))
        bo.buffer_store(
            v.to(fx.BFloat16), rsrc(a["res_out"]), (t * HC + o) * HIDDEN + col
        )
    else:
        v = got
    fx.ptr_store(v, c["rl"] + (t * KT + o * COLS + e % COLS))


@traced
def stage_slice(c, task):
    """R_new, the layer input and the partial projections of hidden columns
    32 task .. +32, all four streams, every token."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl = c["rl"], c["fl"]
    a = c["args"]
    c0 = task * COLS
    batched_rounds(
        tid, s * KT, lambda e: slice_loads(c, c0, e),
        lambda e, got: slice_row(c, c0, e, got),
    )  # fmt: skip
    for i in range_constexpr((MIX * KT + THREADS - 1) // THREADS):
        e = tid + THREADS * i
        if e < MIX * KT:
            j = e // KT
            o = (e % KT) // COLS
            # the original's two bf16 MFMAs take fn as hi = bf16(fn) and lo =
            # bf16(fn - hi): their sum, exact in fp32, is the weight it applies
            f = ld_f32(a["hc_fn"], j * K_MIX + o * HIDDEN + c0 + e % COLS)
            hi = bf16_round(f)
            fx.ptr_store(hi + bf16_round(f - hi), fl + e)
    gpu.barrier()
    # the layer input: bf16 R_new of the four streams by the incoming pre-mix,
    # products rounded, summed in order (the Triton collapse, no contraction);
    # two columns a thread, one bf16 pair a mailbox store
    for pas in range_constexpr(0, s * COLS // 2, THREADS):
        slice_input(c, task, plus(pas, tid))
    # sum of squares: token t on wave t % WAVES, two columns a lane
    for t in range_constexpr(s):
        if wave == t % WAVES:
            x0 = fx.ptr_load(rl + (t * KT + 2 * lane))
            x1 = fx.ptr_load(rl + (t * KT + 2 * lane + 1))
            sq = butterfly(x0 * x0 + x1 * x1, (32, 16, 8, 4, 2, 1))
            if lane == 0:
                c["put"](c["pmix"], (task * s + t) * PART + MIX, sq)
    # mixes: fp32 MFMA 16x16x4, A = fn rows (two 16-row tiles), B = R_new^T,
    # a token tile at a time
    for t0, n in token_tiles(s):
        slice_mix(c, task, t0, n)


@traced
def slice_mix(c, task, t0, n):
    """The mix projection partials of tokens t0 .. t0 + n (``stage_slice``)."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    rl, fl, red = c["rl"], c["fl"], c["red"]
    tok = plus(t0, fx.min(lane % 16, n - 1))
    kw = KT // WAVES
    cs = [fx.Vector.filled(4, 0.0, fx.Float32) for _ in range(2)]
    for kk in range_constexpr(kw // 4):
        colk = wave * kw + lane // 16 + 4 * kk
        b = fx.ptr_load(rl + (tok * KT + colk))
        for h in range_constexpr(2):
            row = h * 16 + lane % 16
            av = fx.ptr_load(fl + (fx.min(row, MIX - 1) * KT + colk))
            av = (row < MIX).select(av, fx.Float32(0.0))
            ops = rocdl._split_mfma_operands([av, b, cs[h], 0, 0, 0])
            cs[h] = fx.Vector(rocdl.mfma_f32_16x16x4f32(T.vec(4, T.f32), *ops).result)
    for h in range_constexpr(2):
        fx.ptr_store(cs[h], red + ((wave * 2 + h) * 64 + lane) * 4)
    gpu.barrier()
    if tid < MIX * n:
        tl = tid // MIX
        t = plus(t0, tl)
        j = tid % MIX
        h = j // 16
        ln = 16 * ((j % 16) // 4) + tl
        tot = fx.Float32(0.0)
        for w in range_constexpr(WAVES):
            tot = tot + fx.ptr_load(red + (((w * 2 + h) * 64 + ln) * 4 + j % 4))
        c["put"](c["pmix"], (task * s + t) * PART + j, tot)
    gpu.barrier()


# ---------------------------------------------------------------- gate
GATE_BLOCK = 8  # slices a gate thread sums


def _row(v, lane, q):
    """comb lanes 8..23 hold comb[r][c] at lane 8 + 4 r + c: element q of the row."""
    return lane_f32(v, (lane - 8) // 4 * 4 + 8 + q)


def _col(v, lane, q):
    return lane_f32(v, 8 + 4 * q + (lane - 8) % 4)


@traced
def stage_gate(c, t):
    """Token t's 160 slices, in slice order, -> post / comb / next pre."""
    s, tid, lane, wave = c["S"], c["tid"], c["lane"], c["wave"]
    red = c["red"]
    a = c["args"]
    groups = SLICES // GATE_BLOCK  # 20
    if tid < groups * PART:
        g = tid // PART
        j = tid % PART
        vals = c["poll"](
            [
                (c["pmix"], ((g * GATE_BLOCK + k) * s + t) * PART + j, 1)
                for k in range(GATE_BLOCK)
            ]
        )
        acc = vals[0][0].bitcast(fx.Float32)
        for k in range_constexpr(1, GATE_BLOCK):
            acc = acc + vals[k][0].bitcast(fx.Float32)
        fx.ptr_store(acc, red + (g * PART + j))
    gpu.barrier()
    if wave == 0:
        j = fx.min(lane, MIX - 1)
        mix = fx.ptr_load(red + j)
        sq = fx.ptr_load(red + MIX)
        for g in range_constexpr(1, groups):
            mix = mix + fx.ptr_load(red + (g * PART + j))
            sq = sq + fx.ptr_load(red + (g * PART + MIX))
        sc = [ld_f32(a["hc_scale"], i) for i in range(3)]
        base = ld_f32(a["hc_base"], j)
        # aiter's mhc_fused_post_pre_delayed_rmsnorm reduce: one rstd =
        # rsq(fma(sum, 1/K, eps)), then each gate fma(mix rstd, scale, base)
        v = mix * hw_rsq(fma(sq, fx.Float32(1.0 / K_MIX), fx.Float32(EPS)))
        if lane < HC:
            pre = sigmoid(fma(v, sc[0], base))
            bo.buffer_store(pre + fx.Float32(HC_EPS), rsrc(a["pre_out"]), t * HC + lane)
        if (lane >= HC) & (lane < 2 * HC):
            post = sigmoid(fma(v, sc[1], base))
            bo.buffer_store(
                post * fx.Float32(POST_MULT), rsrc(a["post_out"]), t * HC + lane - HC
            )
        cv = fma(v, sc[2], base)
        row = [_row(cv, lane, q) for q in range(4)]
        # Triton's exp: exp2 of (x - max) log2(e)
        m = fx.max(fx.max(row[0], row[1]), fx.max(row[2], row[3]))
        cv = hw_exp2((cv - m) * fx.Float32(LOG2E))
        # fast_dividef is x rcp(y); its first "+ eps" contracts into the fma
        row = [_row(cv, lane, q) for q in range(4)]
        cv = fma(cv, hw_rcp((row[0] + row[1]) + (row[2] + row[3])), fx.Float32(HC_EPS))
        col = [_col(cv, lane, q) for q in range(4)]
        cv = cv * hw_rcp(((col[0] + col[1]) + (col[2] + col[3])) + fx.Float32(HC_EPS))
        for _ in range_constexpr(SINKHORN - 1):
            row = [_row(cv, lane, q) for q in range(4)]
            cv = cv * hw_rcp(
                ((row[0] + row[1]) + (row[2] + row[3])) + fx.Float32(HC_EPS)
            )
            col = [_col(cv, lane, q) for q in range(4)]
            cv = cv * hw_rcp(
                ((col[0] + col[1]) + (col[2] + col[3])) + fx.Float32(HC_EPS)
            )
        if (lane >= 2 * HC) & (lane < MIX):
            bo.buffer_store(cv, rsrc(a["comb_out"]), t * HC * HC + lane - 2 * HC)
    gpu.barrier()


# ---------------------------------------------------------------- norm
@traced
def stage_norm(c, t):
    """The seam's RMSNorm (rstd = rsq(fma(sum, 1/H, eps)), bf16((x rstd) w); 256
    threads, 3 chunks of 8 at 8 t + 2048 c), then ATOM's quantize_fp8 of the bf16
    row (amax floored at 1e-4, the ceil code, x / scale) -> x8 words and codes;
    the FFN seam (``ffn``, whose x8 K2b quantizes itself) stores the bf16 row as
    ``normed`` instead. A layer that feeds its indexer (``index``) stores it as
    ``normed`` too and puts it in NB. Threads 256.. mirror thread 255 and store
    nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    # the row's store policy: device scope when this launch also reads it
    normed_cm = c.get("normed_cm", 0)
    tt = fx.min(tid, 255)
    mine = tid < 256
    cols = [tt * 8 + 2048 * ch for ch in range(3)]
    xs = []
    for ch in range_constexpr(3):
        base = t * HIDDEN + fx.min(cols[ch], HIDDEN - 8)
        words = c["poll"]([(c["lin"], base // 2 + k, 1) for k in range(4)])
        v = []
        for k in range_constexpr(4):
            v += [bf_lo(words[k][0]), bf_hi(words[k][0])]
        xs.append([(cols[ch] < HIDDEN).select(x, fx.Float32(0.0)) for x in v])
    acc = fx.Float32(0.0)
    for ch in range_constexpr(3):
        for x in xs[ch]:
            acc = acc + x * x
    acc = butterfly(acc, (1, 2, 4, 8, 16, 32))
    if (lane == 0) & mine:
        fx.ptr_store(acc, red + wave)
    gpu.barrier()
    tot = butterfly(fx.ptr_load(red + lane % 4), (1, 2))
    r = hw_rsq(fma(tot, fx.Float32(1.0 / HIDDEN), fx.Float32(EPS)))
    for ch in range_constexpr(3):
        col = fx.min(cols[ch], HIDDEN - 8)
        ys = [
            bf16_round((xs[ch][j] * r) * ld_bf(a["attn_w"], col + j)) for j in range(8)
        ]
        amax = fx.Float32(0.0)
        for y in ys:
            amax = fx.max(amax, abs(y))
        amax = fx.max(butterfly(amax, (1, 2), fx.max), fx.Float32(1e-4))
        code = code_ceil(amax * fx.Float32(1.0 / FP8_MAX))
        step = pow2(code)
        if mine & (cols[ch] < HIDDEN):
            if const_expr(c["index"] or c["ffn"]):
                for j in range_constexpr(8):
                    bo.buffer_store(
                        ys[j].to(fx.BFloat16),
                        rsrc(a["normed"]),
                        t * HIDDEN + col + j,
                        cache_modifier=normed_cm,
                    )
            if const_expr(c["index"]):
                # the indexer's weights_proj reads the bf16 row
                c["put_bf"](c["nb"], t * HIDDEN + col, ys[0:4])
                c["put_bf"](c["nb"], t * HIDDEN + col + 4, ys[4:8])
            if const_expr(not c["ffn"]):
                w0 = fp8_pack4(*[clamp_fp8(ys[i] / step) for i in range(4)])
                w1 = fp8_pack4(*[clamp_fp8(ys[4 + i] / step) for i in range(4)])
                c["put_words"](c["x8"], t * X8_WORDS + col // 4, [w0, w1])
                if tt % 4 == 0:
                    c["put"](c["x8s"], t * (HIDDEN // 32) + col // 32, code)
    gpu.barrier()


# ---------------------------------------------------------------- GEMV
def _fp8_step_load(c, r_w, r_ws, k, rg, st, row_major=False):
    """K step st's global operands of rows 16 rg ..: the two 64-column weight
    chunks (the upper clamped into the row) and the raw scale word. The (16, 16)
    shuffle lays a row group out in 32-column blocks of 16 rows, a chunk two of
    them (lanes 0-31, 32-63); ``k`` % 64 = 32 (a last quarter step): the lanes
    of a block past the row read its last block. ``row_major`` (vLLM's dense
    layout, k % 64 == 0): the same registers from a row-major [N, K] weight --
    lane l row 16 rg + l % 16, K bytes 128 st + 16 (l // 16) .. and + 64."""
    lane = c["lane"]
    j = lane // 16
    kg = k // 32
    if const_expr(row_major):
        assert k % 64 == 0
        row = rg * 16 + lane % 16
        lo_at = row * (k // 4) + st * 32 + j * 4
        hi_at = row * (k // 4) + fx.min(2 * st + 1, k // 64 - 1) * 16 + j * 4
        lo = fx.Vector(bo.buffer_load(r_w, lo_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT))
        hi = fx.Vector(bo.buffer_load(r_w, hi_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT))
        wsi = (rg // 2) * kg + fx.min(st * 4 + j, kg - 1)
        word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
        return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF
    if const_expr(k % 64 == 0):
        lo_at = (rg * (k // 64) + 2 * st) * 256 + lane * 4
    else:
        lo_at = rg * (k * 4) + fx.min(4 * st + lane // 32, kg - 1) * 128 + lane % 32 * 4
    lo = fx.Vector(
        bo.buffer_load(r_w, lo_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    if const_expr(k % 64 == 0):
        hi_at = (rg * (k // 64) + fx.min(2 * st + 1, k // 64 - 1)) * 256 + lane * 4
    else:
        hi_at = rg * (k * 4) + fx.min(2 * st + 1, k // 64 - 1) * 256 + lane * 4
    hi = fx.Vector(
        bo.buffer_load(r_w, hi_at, vec_width=4, dtype=T.i32, cache_modifier=CM_NT)
    )
    wsi = (rg // 2) * kg + fx.min(st * 4 + j, kg - 1)
    word = fx.Int32(bo.buffer_load(r_ws, wsi // 4, vec_width=1, dtype=T.i32))
    return lo, hi, (word >> (wsi % 4 * 8)) & 0xFF


def _fp8_step_mfma(c, k, xl, xsl, st, ops, acc, live=None, rows=None):
    """``acc`` through K step st's scaled MFMA: ``_fp8_step_load``'s operands
    against every token's MXFP8 row in LDS. ``live`` false: a step past the end
    (its operands those of the last step), zero at unit scale. ``k`` % 64 = 32:
    a lane's lower block past the row is zero on both sides."""
    s, lane = c["S"] if rows is None else rows, c["lane"]
    lo, hi, sa = ops
    kg = k // 32
    x_row, s_row = lds_row(k // 4), lds_row(kg)
    j = lane // 16
    col = fx.min(lane % 16, s - 1)
    if const_expr(live is not None):
        st = fx.min(st, (k + 127) // 128 - 1)
        lo = [live.select(lo[d], fx.Int32(0)) for d in range(4)]
    past = (2 * st + 1) * 64 >= k
    if const_expr(live is not None):
        past = past | ~live
    av = fx.Vector.from_elements(
        [lo[d] for d in range(4)] + [past.select(fx.Int32(0), hi[d]) for d in range(4)],
        fx.Int32,
    )
    grp = st * 4 + j
    xlo = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    xhi = fx.Vector(
        fx.ptr_load(
            xl + (col * x_row + st * 32 + 16 + j * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
    )
    if const_expr(k % 64 != 0):
        over = 4 * st + j // 2 >= kg
        av = fx.Vector.from_elements(
            [over.select(fx.Int32(0), av[d]) for d in range(4)]
            + [av[4 + d] for d in range(4)],
            fx.Int32,
        )
        xlo = [over.select(fx.Int32(0), xlo[d]) for d in range(4)]
    xv = fx.Vector.from_elements(
        [xlo[d] for d in range(4)]
        + [past.select(fx.Int32(0), xhi[d]) for d in range(4)],
        fx.Int32,
    )
    sb = fx.ptr_load(xsl + (col * s_row + fx.min(grp, kg - 1)))
    unit = grp >= kg
    if const_expr(live is not None):
        unit = unit | ~live
    sa = unit.select(fx.Int32(UNIT_SCALE), sa)
    sb = unit.select(fx.Int32(UNIT_SCALE), sb)
    return mfma_scaled(av, xv, acc, sa, sb)


def gemv_fp8(c, w, ws, k, rg, xl, xsl, red, split=None):
    """Rows 16 rg .. +16 of the (16, 16)-preshuffled FP8 weight ``w`` (32x32 e8m0
    ``ws``) times every token's MXFP8 row in LDS (``xl`` words, ``xsl`` codes as
    i32) -> red[wave][lane] partial C: ``gemv_fp8_loads`` then
    ``gemv_fp8_mfmas``. A stage whose input it must wait for issues the loads
    itself first, so the weights' latency overlaps the wait."""
    ops = gemv_fp8_loads(c, w, ws, k, rg, split)
    gemv_fp8_mfmas(c, k, xl, xsl, red, ops, split)


def _gemv_steps(c, k, split, tiled=False):
    """This wave's K steps (a step past the end marked, ``split`` waves past
    split repeat the last one's; ``tiled``: wave w takes part w % split of tile
    w / split): [(step, live)]."""
    wave = c["wave"]
    steps = (k + 127) // 128
    if const_expr(split is None):
        per = (steps + WAVES - 1) // WAVES
        ragged = steps % WAVES != 0
        return [
            (wave + WAVES * i, (wave + WAVES * i < steps) if ragged else None)
            for i in range(per)
        ]
    assert split <= WAVES and steps % split == 0
    per = steps // split
    part = wave % split if tiled else fx.min(wave, split - 1)
    return [(part * per + i, None) for i in range(per)]


def gemv_fp8_loads(c, w, ws, k, rg, split=None, tiled=False, row_major=False):
    """The weight operands of ``gemv_fp8``'s rows for this wave, every one in
    flight at once (a step's latency each would put them in series): loads
    only. A step past the end loads the last step. ``tiled`` (split x tiles =
    WAVES): row groups rg .., one a group of ``split`` waves."""
    steps = (k + 127) // 128
    r_w, r_ws = rsrc(w), rsrc(ws)
    if const_expr(tiled):
        assert WAVES % split == 0
        rg = rg + c["wave"] // split
    return [
        _fp8_step_load(
            c, r_w, r_ws, k, rg, st if live is None else fx.min(st, steps - 1),
            row_major,
        )
        for st, live in _gemv_steps(c, k, split, tiled)
    ]


def gemv_fp8_mfmas(c, k, xl, xsl, red, ops, split=None, tiled=False, rows=None):
    """``gemv_fp8_loads``' operands against the rows in LDS (``rows``: a token
    tile's, every token's by default) ->
    red[wave][lane]; the scaled 16x16x128 MFMA, chunk kc in dwords 0-3 and kc +
    1 in 4-7. ``k`` a multiple of 32: a last half step has its upper chunk zeroed
    on both sides at unit scale (a NaN byte past the row must not reach the
    MFMA), its loads clamped into the row: the last row group's would read past
    the tensor; a last quarter step its lower chunk's upper block too. A step
    past the end adds zero at unit scale.

    The K order is the original kernel's, which ``row_sum`` completes:
    wave w takes steps w, w + 8, ... (``split`` None: aiter's preshuffled group32
    GEMM, P0.5 microbenchmark 1), or waves 0 .. split - 1 one contiguous K range
    each, the other waves zero (aiter's split-K bmm, whose last split sums the
    partials in split order)."""
    lane, wave = c["lane"], c["wave"]
    acc = fx.Vector.filled(4, 0.0, fx.Float32)
    for i, (st, live) in enumerate(_gemv_steps(c, k, split, tiled)):
        acc = _fp8_step_mfma(c, k, xl, xsl, st, ops[i], acc, live, rows)
    if const_expr(split is not None and not tiled):
        acc = fx.Vector.from_elements(
            [(wave < split).select(acc[e], fx.Float32(0.0)) for e in range(4)],
            fx.Float32,
        )
    fx.ptr_store(acc, red + (wave * 64 + lane) * 4)


@traced
def poll_copy(c, idx, width, spans, live=None):
    """Mailbox spans [(region, pair_of, n, dst, row)] into LDS: element u (pair
    ``pair_of(u)``) to dst + u, or ``lds_at(u, row)`` for a span of rows
    ``row`` words wide. Thread ``idx`` of ``width`` takes u = idx, idx +
    width, ..., POLL_MAX a round trip, each batch stored before the next is
    polled (holding them all spills past S = 16); a thread past a span's end
    re-reads its last element and stores nothing.

    ``live``: a span's elements the reader needs (traced, <= n), a span each;
    a batch past them is not polled and an element past them neither polled
    nor written (one past a step's rows is never written)."""
    idx = fresh(idx)  # its elements' indices not hoisted out of a task loop
    if const_expr(live is None):
        _poll_store_batches(c, [
            (region, pair_of, n, (dst, row), idx + width * i)
            for region, pair_of, n, dst, row in spans
            for i in range((n + width - 1) // width)
        ])  # fmt: skip
        return
    for (region, pair_of, n, dst, row), need in zip(spans, live):
        chunks = [
            (region, pair_of, need, (dst, row), idx + width * i)
            for i in range((n + width - 1) // width)
        ]
        for b0 in range_constexpr(0, len(chunks), POLL_MAX):
            if chunks[b0][4] < need:
                _poll_store_batches(c, chunks[b0 : b0 + POLL_MAX])


@traced
def _poll_store_batches(c, chunks):
    """``poll_copy``'s chunks [(region, pair_of, n, (dst, row), u)], POLL_MAX a
    round trip, each batch stored before the next is polled."""
    for b0 in range_constexpr(0, len(chunks), POLL_MAX):
        batch = chunks[b0 : b0 + POLL_MAX]
        got = c["poll"]([(r, p(fx.min(u, n - 1)), 1) for r, p, n, _, u in batch])
        for (_, _, n, (dst, row), u), v in zip(batch, got):
            at = u if row is None else lds_at(u, row)
            if u < n:
                fx.ptr_store(v[0], dst + at)


@traced
def load_x8(c, words_mb, codes_mb, words, groups, xl, xsl, t0=0, n=None):
    """Tokens t0 .. t0 + n's MXFP8 rows (every token's by default) from the
    mailbox into LDS rows 0 .. (codes as i32). ``n`` traced (a tile a CTA picks
    at run time): a TILE-row copy of which only n rows are polled."""
    n = c["S"] if n is None else n
    rows = n if isinstance(n, int) else TILE
    poll_copy(
        c,
        c["tid"],
        THREADS,
        [
            (words_mb, lambda u: plus(t0 * words, u), rows * words, xl, words),
            (codes_mb, lambda u: plus(t0 * groups, u), rows * groups, xsl, groups),
        ],
        live=None if isinstance(n, int) else [n * words, n * groups],
    )
    gpu.barrier()


@traced
def stage_wqkv(c, task, ops, t0, n):
    """Rows 16 task .. of wqkv_a (weights ``ops``) against tokens t0 .. t0 + n,
    their x in LDS (``load_x8``)."""
    tid, red = c["tid"], c["red"]
    gemv_fp8_mfmas(c, HIDDEN, c["xl"], c["xsl"], red, ops, rows=n)
    gpu.barrier()
    if tid < ROWS * n:
        r = tid % ROWS
        t = plus(t0, tid // ROWS)
        v = row_sum(red, r, tid // ROWS)
        c["put"](c["qkv"], t * QKV_ROWS + task * ROWS + r, bf16_round(v))
    gpu.barrier()


# ---------------------------------------------------------------- q / kv norms
@traced
def stage_qkv(c, t):
    """Token t: q_norm -> MXFP8 (the Triton dual norm, 4 warps x 8 columns) and
    kv_norm -> GPT-J RoPE of the tail -> bf16 -> group-32 FP8 QAT -> the ring.

    Threads 256.. mirror thread 255's loads (so every wave reaches the barrier)
    and store nothing."""
    tid, lane, wave, red = c["tid"], c["lane"], c["wave"], c["red"]
    a = c["args"]
    tt = fx.min(tid, 255)
    mine = tid < 256
    qc = [fx.min(tt * 8 + j, Q_RANK - 1) for j in range(8)]
    qv = c["poll"]([(c["qkv"], t * QKV_ROWS + qc[j], 1) for j in range(8)])
    xq = [
        ((tt * 8 + j) < Q_RANK).select(qv[j][0].bitcast(fx.Float32), fx.Float32(0.0))
        for j in range(8)
    ]
    kvv = c["poll"](
        [(c["qkv"], t * QKV_ROWS + Q_RANK + tt * 2 + j, 1) for j in range(2)]
    )
    xk = [kvv[j][0].bitcast(fx.Float32) for j in range(2)]
    ssq = xq[0] * xq[0]
    for j in range_constexpr(1, 8):
        ssq = ssq + xq[j] * xq[j]
    ssq = butterfly(ssq, (8, 4, 2, 1, 16, 32))
    ssk = butterfly(xk[0] * xk[0] + xk[1] * xk[1], (8, 4, 2, 1, 16, 32))
    if (lane == 63) & mine:
        fx.ptr_store(ssq, red + wave)
        fx.ptr_store(ssk, red + 4 + wave)
    gpu.barrier()
    tq = butterfly(fx.ptr_load(red + lane % 4), (2, 1))
    tk = butterfly(fx.ptr_load(red + 4 + lane % 4), (2, 1))
    nq = hw_rsq(
        div_rn(tq, fx.Float32(float(Q_RANK)), fx.Float32(1.0 / Q_RANK))
        + fx.Float32(EPS)
    )
    nk = hw_rsq(
        div_rn(tk, fx.Float32(float(KV_DIM)), fx.Float32(1.0 / KV_DIM))
        + fx.Float32(EPS)
    )
    # q: MXFP8 by _mxfp8_quant_op, a group of 32 = 4 threads
    ys = [(xq[j] * nq) * ld_bf(a["qn_w"], qc[j]) for j in range(8)]
    amax = fx.Float32(0.0)
    for y in ys:
        amax = fx.max(amax, abs(y))
    code = code_round(butterfly(amax, (1, 2), fx.max))
    mul = pow2(254 - code)
    if mine & (tt * 8 < Q_RANK):
        w0 = fp8_pack4(*[ys[i] * mul for i in range(4)])
        w1 = fp8_pack4(*[ys[4 + i] * mul for i in range(4)])
        c["put_words"](c["qx8"], t * QX8_WORDS + 2 * tt, [w0, w1])
        if tt % 4 == 0:
            c["put"](c["qx8s"], t * (Q_RANK // 32) + tt // 4, code)
        if const_expr(c["index"]):
            # the (qr, qr_scale) pair the original indexer reads
            bo.buffer_store(
                fx.Vector.from_elements([w0, w1], fx.Int32),
                rsrc(a["qr_out"]),
                t * QX8_WORDS + 2 * tt,
            )
            if tt % 4 == 0:
                bo.buffer_store(
                    fx.Int8(code), rsrc(a["qrs_out"]), t * (Q_RANK // 32) + tt // 4
                )
    # kv: this thread's pair (2 tt, 2 tt + 1), RoPE on the last 64 columns
    kn = [bf16_round((xk[j] * nk) * ld_bf(a["kvn_w"], tt * 2 + j)) for j in range(2)]
    pos = ld_i32(a["pos"], 2 * t)
    jr = tt - (KV_DIM - ROPE) // 2
    live = jr >= 0
    jj = fx.max(jr, 0)
    cs = live.select(ld_bf(a["cos"], pos * HALF + jj), fx.Float32(1.0))
    sn = live.select(ld_bf(a["sin"], pos * HALF + jj), fx.Float32(0.0))
    e0 = bf16_round(kn[0] * cs - kn[1] * sn)
    e1 = bf16_round(kn[0] * sn + kn[1] * cs)
    amax = butterfly(fx.max(abs(e0), abs(e1)), (1, 2, 4, 8), fx.max)
    step = pow2(code_ceil(fx.max(amax, fx.Float32(1e-4)) * fx.Float32(1.0 / FP8_MAX)))
    qat = [fp8_value(clamp_fp8(v / step)) * step for v in (e0, e1)]
    # the ring row of a [rows, 512] bf16 pool: the step's row of this token (int32,
    # -1: none) past the layer's ring start
    rel = ld_i32(a["ring_rows"], t)
    row = rel + a["ring_off"]
    if mine & (rel >= 0):
        dst = rsrc(a["ring"] + fx.Int64(row) * (KV_DIM * 2))
        for j in range_constexpr(2):
            bo.buffer_store(qat[j].to(fx.BFloat16), dst, tt * 2 + j)
    gpu.barrier()


# ---------------------------------------------------------------- wq_b
@traced
def stage_wqb(c, task, ops, t0=0, n=None):
    """Rows 16 task .. +16 of the rank's heads x 512: FP8 GEMV -> bf16 -> the
    GPT-J RoPE of a head's last 64 dims (the Triton seam: bf16 products are exact,
    so its contraction does not show) -> q out; the token tile from ``t0`` (its
    x in LDS)."""
    tid, red = c["tid"], c["red"]
    n = c["S"] if n is None else n
    a = c["args"]
    gemv_fp8_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], red, ops, rows=n)
    gpu.barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        r0 = task * ROWS + 2 * rp  # global row: head r0 // 512, dim r0 % 512
        v0 = bf16_round(row_sum(red, 2 * rp, tl))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tl))
        d = r0 % HEAD_DIM
        jr = (d - (HEAD_DIM - ROPE)) // 2
        live = d >= HEAD_DIM - ROPE
        pos = ld_i32(a["pos"], 2 * t)
        jj = fx.max(jr, 0)
        cs = ld_bf(a["cos"], pos * HALF + jj)
        sn = ld_bf(a["sin"], pos * HALF + jj)
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        out = t * c["d"].heads * HEAD_DIM + r0
        bo.buffer_store(e0.to(fx.BFloat16), rsrc(a["q_out"]), out)
        bo.buffer_store(e1.to(fx.BFloat16), rsrc(a["q_out"]), out + 1)
    gpu.barrier()


@traced
def stage_iq(c, task, ops, t0=0, n=None):
    """Rows 16 task .. of the indexer's wq_b [4096, 1280]: FP8 GEMV of the q
    latent -> bf16 -> the GPT-J RoPE of a head's dims 64..127 -> IQ; the token
    tile from ``t0`` (its x in LDS)."""
    tid, red = c["tid"], c["red"]
    n = c["S"] if n is None else n
    a = c["args"]
    gemv_fp8_mfmas(c, Q_RANK, c["qxl"], c["qxsl"], red, ops, rows=n)
    gpu.barrier()
    if tid < ROWS * n // 2:
        rp = tid % (ROWS // 2)
        tl = tid // (ROWS // 2)
        t = plus(t0, tl)
        r0 = task * ROWS + 2 * rp
        v0 = bf16_round(row_sum(red, 2 * rp, tl))
        v1 = bf16_round(row_sum(red, 2 * rp + 1, tl))
        d = r0 % ip.DIM
        live = d >= ip.DIM - IDX_ROPE
        jj = fx.max((d - (ip.DIM - IDX_ROPE)) // 2, 0)
        pos = ld_i32(a["pos"], 2 * t)
        cs = ld_bf(a["cos"], pos * HALF + jj)
        sn = ld_bf(a["sin"], pos * HALF + jj)
        # bf16 x bf16 products are exact: the original's contraction cannot show
        e0 = fx.Float32(live.select(v0 * cs - v1 * sn, v0))
        e1 = fx.Float32(live.select(v0 * sn + v1 * cs, v1))
        c["put_bf"](c["iq"], t * IDX_ROWS + r0, [e0, e1])
    gpu.barrier()


def build_attn_pre(key: AttnPreBuild):
    s = key.tokens
    assert 1 <= s <= MAX_TOKENS
    layout = scratch_layout(s)
    tiles, rows = token_tiles(s), tile_rows(s)
    keyed = key_tuple(key, SOURCES)
    dims = Dims(key.tp)
    assert dims.wqb_tasks % BLOCKS == 0
    wqb_per_cta = dims.wqb_tasks // BLOCKS

    # LDS: what several stages read, then one union of what only one stage
    # does -- slice's, wqkv_a's and index_w's buffers; a barrier ends each
    @fx.struct
    class Smem:
        red: fx.Array[fx.Float32, WAVES * 2 * 64 * 4, 16]
        qxl: fx.Array[fx.Int32, rows * lds_row(QX8_WORDS), 16]
        qxsl: fx.Array[fx.Int32, rows * lds_row(Q_RANK // 32), 16]
        tls: fx.Array[fx.Int64, TL_POINTS if key.timeline else 1, 16]

    @fx.struct
    class SliceLds:
        rl: fx.Array[fx.Float32, s * KT, 16]
        fl: fx.Array[fx.Float32, MIX * KT, 16]

    @fx.struct
    class WqkvLds:
        xl: fx.Array[fx.Int32, rows * lds_row(X8_WORDS), 16]
        xsl: fx.Array[fx.Int32, rows * lds_row(HIDDEN // 32), 16]

    @fx.struct
    class IwLds:
        # a group of WAVES tokens' normed rows (``stage_iw``)
        nbl: fx.Array[
            fx.Int32, min(s, WAVES) * HIDDEN // 2 if key.feeds_index else 1, 16
        ]

    @fx.union
    class StageLds:
        slice: SliceLds
        wqkv: WqkvLds
        iw: IwLds

    name = kernel_symbol(
        "v41_attn_pre",
        s=s,
        f=int(key.fold),
        ix=int(key.feeds_index),
        i4=int(key.index_fp4),
        a=int(key.aux),
        tl=int(key.timeline),
    )

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def attn_pre(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64, wqkv: Int64,
        wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos: Int64, sin: Int64, pos: Int64, ring: Int64, ring_rows: Int64, ring_off: Int32,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        q_out: Int64, normed: Int64, qr_out: Int64, qrs_out: Int64, aux: Int64,
        iq_w: Int64, iq_ws: Int64, iw_w: Int64, iq_out: Int64, iqs_out: Int64,
        iw_out: Int64, scratch: Int64, layer: Int32, tl: Int64,
    ):  # fmt: skip
        tid = fx.thread_idx.x
        bid = fx.block_idx.x
        alloc = fx.SharedAllocator()
        lds = alloc.allocate(Smem).peek()
        stage_lds = alloc.allocate(StageLds)
        sl, wl, il = stage_lds.slice.peek(), stage_lds.wqkv.peek(), stage_lds.iw.peek()
        mb = mailbox(layer, scratch, key.diag_off, layout)
        c = {
            "S": s, "tid": tid, "bid": bid, "lane": tid % 64, "wave": tid // 64,
            "rl": sl.rl.ptr, "fl": sl.fl.ptr, "red": lds.red.ptr,
            "xl": wl.xl.ptr, "xsl": wl.xsl.ptr, "qxl": lds.qxl.ptr, "qxsl": lds.qxsl.ptr,
            "put": mb.put, "put_bf": mb.put_bf, "put_words": mb.put_words, "poll": mb.poll,
            "args": {
                "res_in": res_in, "pend": pend, "post_in": post_in, "comb_in": comb_in,
                "pre_in": pre_in, "hc_fn": hc_fn, "hc_scale": hc_scale, "hc_base": hc_base,
                "attn_w": attn_w, "wqkv": wqkv, "wqkv_s": wqkv_s, "qn_w": qn_w,
                "kvn_w": kvn_w, "wqb": wqb, "wqb_s": wqb_s, "cos": cos, "sin": sin,
                "pos": pos, "ring": ring, "ring_rows": ring_rows, "ring_off": ring_off,
                "res_out": res_out,
                "post_out": post_out, "comb_out": comb_out, "pre_out": pre_out,
                "q_out": q_out, "normed": normed, "qr_out": qr_out, "qrs_out": qrs_out,
                "aux": aux, "iq_w": iq_w, "iq_ws": iq_ws, "iw_w": iw_w,
                "iq_out": iq_out, "iqs_out": iqs_out, "iw_out": iw_out,
            },
            "fold": key.fold, "index": key.feeds_index, "aux": key.aux, "ffn": False,
            "d": dims,
        }  # fmt: skip
        for region, (off, _) in layout.items():
            c[region] = sreg(scratch, off, region)
        timeline = key.timeline
        tls = lds.tls.ptr
        stamp_begin(timeline, tls, tid, TL_POINTS)
        enter_stage("k1.slice")
        for task in range(first_task(bid, 0), SLICES, BLOCKS):
            stage_slice(c, task)
        stamp(timeline, tls, tid, 1)
        enter_stage("k1.gate")
        for t in range(first_task(bid, GATE0), s, BLOCKS):
            stage_gate(c, t)
        stamp(timeline, tls, tid, 2)
        enter_stage("k1.norm")
        for t in range(first_task(bid, NORM0), s, BLOCKS):
            stage_norm(c, t)
        stamp(timeline, tls, tid, 3)
        enter_stage("k1.wqkv_a")
        # a CTA's wqkv task (at most one): its weights in flight before the
        # normed row's wait
        assert WQKV_TASKS <= BLOCKS
        if const_expr(len(tiles) == 1):
            wqkv_task = first_task(bid, WQKV0)
            if wqkv_task < WQKV_TASKS:
                wqkv_ops = gemv_fp8_loads(c, wqkv, wqkv_s, HIDDEN, wqkv_task)
                load_x8(
                    c, c["x8"], c["x8s"], X8_WORDS, HIDDEN // 32, c["xl"], c["xsl"],
                    0, rows,
                )  # fmt: skip
                stage_wqkv(c, wqkv_task, wqkv_ops, 0, s)
        else:
            # a task's tiles in series on one CTA left more than half the CTAs
            # idle through it (the weights read again a tile: L2 hits)
            tile, wqkv_tasks = tile_tasks(bid, WQKV0, len(tiles), WQKV_TASKS)
            wqkv_ops = [
                gemv_fp8_loads(c, wqkv, wqkv_s, HIDDEN, fx.min(t, WQKV_TASKS - 1))
                for t in wqkv_tasks
            ]
            t0 = tile * TILE
            n = fx.min(fx.Int32(TILE), s - t0)
            load_x8(
                c, c["x8"], c["x8s"], X8_WORDS, HIDDEN // 32, c["xl"], c["xsl"],
                t0, n,
            )  # fmt: skip
            for k in range_constexpr(len(wqkv_tasks)):
                if wqkv_tasks[k] < WQKV_TASKS:
                    stage_wqkv(c, wqkv_tasks[k], wqkv_ops[k], t0, n)
        stamp(timeline, tls, tid, 4)
        enter_stage("k1.qkv")
        for t in range(first_task(bid, QKV0), s, BLOCKS):
            stage_qkv(c, t)
        stamp(timeline, tls, tid, 5)
        enter_stage("k1.wq_b")
        # every CTA's wq_b tasks (and an indexer layer's wq task), two at a
        # time: the first two's weights in flight before the q latent's wait;
        # the weights stay in registers for the later token tiles (and the iq
        # task runs in each: held on past them, its weights spill in index_w)
        assert IQ_TASKS == BLOCKS
        wqb_task = first_task(bid, WQB0)
        wqb_ops = [
            gemv_fp8_loads(c, wqb, wqb_s, Q_RANK, nth_task(wqb_task, i))
            for i in range(min(2, wqb_per_cta))
        ]
        iq_task = first_task(bid, IQ0)
        if const_expr(key.feeds_index):
            iq_ops = gemv_fp8_loads(c, iq_w, iq_ws, Q_RANK, iq_task)
        wqb_all = list(wqb_ops)
        for t0, n in tiles:
            load_x8(
                c, c["qx8"], c["qx8s"], QX8_WORDS, Q_RANK // 32, c["qxl"], c["qxsl"],
                t0, n,
            )  # fmt: skip
            for b in range_constexpr(0, wqb_per_cta, 2):
                if const_expr(b > 0 and t0 == 0):
                    wqb_all += [
                        gemv_fp8_loads(c, wqb, wqb_s, Q_RANK, nth_task(wqb_task, b + i))
                        for i in range(min(2, wqb_per_cta - b))
                    ]
                for i in range_constexpr(min(2, wqb_per_cta - b)):
                    stage_wqb(c, nth_task(wqb_task, b + i), wqb_all[b + i], t0, n)
            if const_expr(key.feeds_index):
                stage_iq(c, iq_task, iq_ops, t0, n)
        stamp(timeline, tls, tid, 6)
        if const_expr(key.feeds_index):
            enter_stage("k1.index_w")

            def load_group(g, n):
                poll_copy(
                    c, tid, THREADS,
                    [(c["nb"], lambda u: plus(g * HIDDEN // 2, u), n * HIDDEN // 2,
                      il.nbl.ptr, None)],
                )  # fmt: skip
                gpu.barrier()

            for task in range(first_task(bid, IW0), IW_TASKS, BLOCKS):
                stage_iw(c, task, HIDDEN, il.nbl.ptr, load_group)
            enter_stage("k1.index_quant")
            for task in range(first_task(bid, IQUANT0), s * ip.HEADS, BLOCKS):
                if const_expr(key.index_fp4):
                    stage_iquant_fp4(c, task)
                else:
                    stage_iquant(c, task)
        stamp(timeline, tls, tid, TL_POINTS - 1)
        stamp_flush(timeline, tls, tl, tid, bid, TL_POINTS)
        _ = keyed

    @flyc.jit
    def launch(
        res_in: Int64, pend: Int64, post_in: Int64, comb_in: Int64, pre_in: Int64,
        hc_fn: Int64, hc_scale: Int64, hc_base: Int64, attn_w: Int64, wqkv: Int64,
        wqkv_s: Int64, qn_w: Int64, kvn_w: Int64, wqb: Int64, wqb_s: Int64,
        cos: Int64, sin: Int64, pos: Int64, ring: Int64, ring_rows: Int64, ring_off: Int32,
        res_out: Int64, post_out: Int64, comb_out: Int64, pre_out: Int64,
        q_out: Int64, normed: Int64, qr_out: Int64, qrs_out: Int64, aux: Int64,
        iq_w: Int64, iq_ws: Int64, iw_w: Int64, iq_out: Int64, iqs_out: Int64,
        iw_out: Int64, scratch: Int64, layer: Int32, tl: Int64,
        stream: fx.Stream = _CURRENT_STREAM,
    ):  # fmt: skip
        _ = keyed
        attn_pre(
            res_in, pend, post_in, comb_in, pre_in, hc_fn, hc_scale, hc_base, attn_w,
            wqkv, wqkv_s, qn_w, kvn_w, wqb, wqb_s, cos, sin, pos, ring, ring_rows, ring_off,
            res_out, post_out, comb_out, pre_out, q_out, normed, qr_out, qrs_out, aux,
            iq_w, iq_ws, iw_w, iq_out, iqs_out, iw_out, scratch, layer, tl,
        ).launch(grid=(BLOCKS,), block=(THREADS,), stream=stream)  # fmt: skip

    ABI.check(attn_pre, launch)
    return launch
