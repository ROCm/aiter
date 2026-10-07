# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""K1's indexer projections (FULL / REINDEX layers): the index query and the
head weights the scorer reads, bit for bit the original chain
(``Indexer.project`` then ``quantize_query_rows`` and ``scale_indexer_weights``,
``p31_indexer_plan.md``). The indexer is replicated: every rank computes all 32
heads, as the original does.

    iw      (32 tasks, a row of weights_proj [32, 5120] each, a wave a token): aiter
            ``wv_splitk_small<bf16, 64, 1, 1, 8, 4, 6>`` -- a lane's 8-element
            chunks at k = k1 + 512 k2 + 8 lane in (k1, k2) order, each
            (a0 b0, a1 b1) then fma chains on the even and odd elements, the
            chunk's x + y added to the lane's sum; across lanes v_add_f32 dpp
            row_shr 8, 4, 2, wave_shr 1, row_bcast 15, row_bcast 31, lane 63
            (``harness/p31_wvsplitk_emul.py``) -> bf16 -> IW
    iq      (256 tasks, 16 rows of wq_b [4096, 1280], ``attn_pre.stage_iq``):
            the FP8 GEMV of the q latent -> bf16 -> GPT-J RoPE on each head's
            dims 64..127 -> IQ
    iquant  (S x 32 tasks, a (token, head) row of 128): aiter
            ``dynamic_per_token_scaled_quant``: amax (seeded 1e-10),
            scale = rn(amax x f32(1/448)), inv = 1 / scale (IEEE), e4m3 of
            clamp(rn(x inv), +-448); then the head weight rn(rn(w q_scale) x
            weights_scale) -> global ``iq_out`` / ``iqs_out`` / ``iw_out``;
            on the FP4 plane (``stage_iquant_fp4``) ``quantize_fp4``'s group-32
            E8M0 E2M1 instead, the scales in the row-group scorer's packed
            order and the weight rn(w x weights_scale)
"""

import flydsl.expr as fx
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.dsv41_mono.atomv41 import index_plan as ip
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.mx import FP8_MAX, clamp_fp8
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.device.ops import (
    CM_NT,
    bf2_f32,
    bf16_round,
    bf_hi,
    bf_lo,
    butterfly,
    fp8_pack4,
    lane_gather,
    row_shr,
    rsrc,
    traced,
)
from aiter.ops.flydsl.kernels.dsv41_mono.atomfw.plan.execution import WAVES

IDX_ROWS = ip.HEADS * ip.DIM  # 4096
IDX_ROPE = 64  # a head's rotated tail
IDX_WEIGHTS_SCALE = ip.DIM**-0.5 * ip.HEADS**-0.5  # 1/64: exact
IW_TASKS = ip.HEADS  # a row a task, a wave a token
AMAX_SEED = 1e-10


def scratch_pairs(tokens: int, hidden: int) -> dict[str, int]:
    return {
        "nb": tokens * hidden // 2,  # the normed row, bf16 pairs
        "iq": tokens * IDX_ROWS // 2,  # the rotated index query, bf16 pairs
        "iw": tokens * ip.HEADS,  # weights_proj out, f32 of bf16
    }


def _wv_reduce(v, lane):
    """wv_splitk's cross-lane v_add_f32 dpp chain, the value lane 63 ends with."""
    for k in (8, 4, 2):
        v = row_shr(v, lane, k) + v
    moved = lane_gather(v.bitcast(fx.Int32), fx.max(lane - 1, 0)).bitcast(fx.Float32)
    v = fx.Float32((lane >= 1).select(moved + v, v))  # wave_shr:1
    row = lane // 16
    b15 = lane_gather(v.bitcast(fx.Int32), fx.max(row * 16 - 1, 0)).bitcast(fx.Float32)
    v = fx.Float32((row % 2 == 1).select(b15 + v, v))
    b31 = lane_gather(v.bitcast(fx.Int32), 31).bitcast(fx.Float32)
    v = fx.Float32((row >= 2).select(b31 + v, v))
    return lane_gather(v.bitcast(fx.Int32), 63).bitcast(fx.Float32)


@traced
def stage_iw(c, task, hidden, nbl, load_group):
    """Row ``task`` of weights_proj; wave w its tokens w, w + WAVES, ..., a group
    of WAVES tokens' normed rows at a time in LDS ``nbl`` (bf16 pairs,
    row-packed; ``load_group(first token, tokens)`` puts them there) -> IW. The
    weights load once; each (row, token) sum keeps wv_splitk's per-lane order,
    so spreading the tokens over waves changes nothing but the latency."""
    s, lane, wave = c["S"], c["lane"], c["wave"]
    a = c["args"]
    m = task
    ks = [
        k1 + k2 * 512
        for k1 in range(0, hidden, 2048)
        for k2 in range(4)
        if k1 + k2 * 512 < hidden
    ]
    wvs = []
    for k0 in ks:
        kk = fx.min(k0 + lane * 8, hidden - 8)
        wvs.append(
            fx.Vector(
                bo.buffer_load(
                    rsrc(a["iw_w"]),
                    (m * hidden + kk) // 2,
                    vec_width=4,
                    dtype=T.i32,
                    cache_modifier=CM_NT,
                )
            )
        )
    for g in range_constexpr(0, s, WAVES):
        n = min(WAVES, s - g)
        load_group(g, n)
        tl = fx.min(wave, n - 1)
        t = tl if g == 0 else tl + g
        total = fx.Float32(0.0)
        for i in range_constexpr(len(ks)):
            k = ks[i] + lane * 8
            live = k < hidden
            kk = fx.min(k, hidden - 8)
            xv = fx.Vector(
                fx.ptr_load(
                    nbl + ((tl * hidden + kk) // 2),
                    result_type=fx.Vector.make_type(4, fx.Int32),
                )
            )
            wv = wvs[i]
            ax = [bf_lo(xv[j]) for j in range(4)]
            ay = [bf_hi(xv[j]) for j in range(4)]
            bx = [bf_lo(wv[j]) for j in range(4)]
            by = [bf_hi(wv[j]) for j in range(4)]
            accx = ax[0] * bx[0]
            accy = ay[0] * by[0]
            for j in range_constexpr(1, 4):
                accx = fx.Float32(fmath.fma(ax[j], bx[j], accx))
                accy = fx.Float32(fmath.fma(ay[j], by[j], accy))
            total = fx.Float32(live.select(total + (accx + accy), total))
        w = _wv_reduce(total, lane)
        if (lane == 0) & (wave < n):
            c["put"](c["iw"], t * ip.HEADS + m, bf16_round(w))
        gpu.barrier()


@traced
def stage_iquant(c, task):
    """Token task // 32, head task % 32: a wave, 2 dims a lane (wave 0)."""
    lane, wave = c["lane"], c["wave"]
    a = c["args"]
    t = task // ip.HEADS
    h = task % ip.HEADS
    if wave == 0:
        got = c["poll"](
            [
                (c["iq"], (t * IDX_ROWS + h * ip.DIM) // 2 + lane, 1),
                (c["iw"], t * ip.HEADS + h, 1),
            ]
        )
        x0, x1 = bf2_f32(got[0][0])
        amax = butterfly(
            fx.max(fx.max(abs(x0), abs(x1)), fx.Float32(AMAX_SEED)),
            (1, 2, 4, 8, 16, 32),
            fx.max,
        )
        scale = amax * fx.Float32(1.0 / FP8_MAX)
        inv = fx.Float32(1.0) / scale
        q = fp8_pack4(
            clamp_fp8(x0 * inv), clamp_fp8(x1 * inv), fx.Float32(0.0), fx.Float32(0.0)
        )
        row = t * ip.HEADS + h
        bo.buffer_store(
            fx.Int16(q & 0xFFFF), rsrc(a["iq_out"]), row * (ip.DIM // 2) + lane
        )
        if lane == 0:
            w = got[1][0].bitcast(fx.Float32)
            bo.buffer_store(scale, rsrc(a["iqs_out"]), row)
            bo.buffer_store(
                (w * scale) * fx.Float32(IDX_WEIGHTS_SCALE), rsrc(a["iw_out"]), row
            )


# the FP4 query's E8M0 scales a token, as the row-group scorer packs them:
# [32-dim chunk][head % 16][head // 16, the M tiles padded to a dword]
QS_M_TILES = -(-(ip.HEADS // 16) // 4) * 4
QS_BYTES = ip.DIM // 32 * 16 * QS_M_TILES
FP4_MIN_AMAX = 6.0 * 2.0**-126  # the smallest normal times E2M1's max


def _e2m1(x):
    """The E2M1 nibble of ``x`` (a value over its scale): nearest-even with
    ``quantize_fp4``'s asymmetric midpoints, the sign bit x's."""
    m = abs(x)
    code = fx.Int32(0)
    for edge, value, inclusive in (
        (0.25, 1, False), (0.75, 2, True), (1.25, 3, False), (1.75, 4, True),
        (2.5, 5, False), (3.5, 6, True), (5.0, 7, False),
    ):  # fmt: skip
        past = (m >= fx.Float32(edge)) if inclusive else (m > fx.Float32(edge))
        code = fx.Int32(past.select(fx.Int32(value), code))
    sign = (x.bitcast(fx.Int32) >> 31) & 1
    return code | (sign << 3)


@traced
def stage_iquant_fp4(c, task):
    """``stage_iquant`` on the FP4 plane: token task // 32, head task % 32, a
    wave (wave 0), 2 dims a lane and a 32-dim group 16 lanes. ``quantize_fp4``
    to the bit: code = ceil(log2(max(amax, FP4_MIN_AMAX) x f32(1/6))), the
    value times 2^(127 - code) (its scale's exact inverse) to E2M1, the even
    dim the low nibble."""
    lane, wave = c["lane"], c["wave"]
    a = c["args"]
    t = task // ip.HEADS
    h = task % ip.HEADS
    if wave == 0:
        got = c["poll"](
            [
                (c["iq"], (t * IDX_ROWS + h * ip.DIM) // 2 + lane, 1),
                (c["iw"], t * ip.HEADS + h, 1),
            ]
        )
        x0, x1 = bf2_f32(got[0][0])
        amax = butterfly(fx.max(abs(x0), abs(x1)), (1, 2, 4, 8), fx.max)
        bits = (fx.max(amax, fx.Float32(FP4_MIN_AMAX)) * fx.Float32(1.0 / 6.0)).bitcast(
            fx.Int32
        )
        code = (bits >> 23) + fx.Int32(
            ((bits & 0x7FFFFF) != 0).select(fx.Int32(1), fx.Int32(0))
        )
        inv = ((fx.Int32(254) - code) << 23).bitcast(fx.Float32)
        row = t * ip.HEADS + h
        byte = _e2m1(x0 * inv) | (_e2m1(x1 * inv) << 4)
        bo.buffer_store(fx.Int8(byte), rsrc(a["iq_out"]), row * (ip.DIM // 2) + lane)
        if lane % 16 == 0:
            bo.buffer_store(
                fx.Int8(code),
                rsrc(a["iqs_out"]),
                t * QS_BYTES
                + lane // 16 * (16 * QS_M_TILES)
                + h % 16 * QS_M_TILES
                + h // 16,
            )
        if lane == 0:
            w = got[1][0].bitcast(fx.Float32)
            bo.buffer_store(w * fx.Float32(IDX_WEIGHTS_SCALE), rsrc(a["iw_out"]), row)
