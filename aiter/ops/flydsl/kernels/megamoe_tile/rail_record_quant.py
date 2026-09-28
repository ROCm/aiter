# SPDX-License-Identifier: MIT
"""rail_soa 的发送源:1x32 MXFP4 量化直接写成按 token 连续的 rail record。

record(每 token REC 字节,位于注册窗口 dispatch_staging 的 parity-0 平面):
    [q  H/2 B | scale H/32 B | topk_ids TOPK*4 B | route_weights TOPK*4 B | pad]
k1 的 T0 按 QP 切 token 段,每段一个 PUT 发出,不再有零散的 scale/ids/weights PUT
(每个 WQE 的投递成本 ~15µs,与大小无关)。量化数学与 mega_moe/quant.py 的 fp4
路径逐条相同,q/scale 逐位一致。
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp, T

from .. import buffer_ops
from ..mega_moe.quant import BLOCK, GROUP, _FP4_INV_MAX_POS_BITS


def rail_record_layout(hidden: int, topk: int):
    """(REC_BYTES, off_q, off_scale, off_ids, off_weights),全部 16B 对齐。"""
    off_q = 0
    off_s = off_q + hidden // 2
    off_i = off_s + hidden // 32
    # ids 段补齐到 16B:topk*4 不是 16 的倍数时(如 topk=6),weights 起点仍对齐。
    off_w = off_i + (topk * 4 + 15) // 16 * 16
    end = off_w + topk * 4
    rec = (end + 255) // 256 * 256
    for v in (off_s, off_i, off_w, rec):
        assert v % 16 == 0, (hidden, topk, v)
    return rec, off_q, off_s, off_i, off_w


def build_rail_record_quant(hidden: int, topk: int, side_off: int = 0):
    assert hidden % GROUP == 0
    scale_n = hidden // GROUP
    assert scale_n >= 2 * topk, "ids/weights 由每 token 前 2*topk 个 group 线程顺带写"
    rec, off_q, off_s, off_i, off_w = rail_record_layout(hidden, topk)

    # side_off>0(rail_ids_first):另在 out+side_off 处写连续的 [ids|weights],每 token topk*8 B,
    # k1 先单独 PUT 这段,接收端拿到 ids 就能开始领行。
    side_rec = topk * 8
    assert side_off % 16 == 0

    @flyc.kernel(
        name=f"per_1x32_mx_quant_fp4_n{hidden}_railrec_k{topk}_r{rec}"
        + (f"_s{side_off}" if side_off else "")
    )
    def rec_quant_kernel(
        x: fx.Int64,
        topk_ids: fx.Int64,
        route_weights: fx.Int64,
        out: fx.Int64,
        m: fx.Int32,
    ):
        group_id = fx.Int32(fx.block_idx.x) * fx.Int32(BLOCK) + fx.Int32(
            fx.thread_idx.x
        )
        if group_id < m * fx.Int32(scale_n):
            token = group_id // fx.Int32(scale_n)
            gi = group_id % fx.Int32(scale_n)
            x_rs = buffer_ops.create_buffer_resource_from_addr(x)
            out_rs = buffer_ops.create_buffer_resource_from_addr(out)
            # 与 quant.py fp4 路径相同的数学。
            in_vec = group_id * fx.Int32(GROUP * 2 // 16)
            act = []
            local_max = fx.Float32(1e-10)
            for chunk in range_constexpr(GROUP // 8):
                raw = buffer_ops.buffer_load(
                    x_rs,
                    (in_vec + fx.Int32(chunk)) * fx.Int32(4),
                    vec_width=4,
                    dtype=T.i32,
                )
                values = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                local_max = local_max.maximumf(
                    fmath.absf(values).reduce(ReductionOp.MAX)
                )
                for elem in range_constexpr(8):
                    act.append(values[elem])
            working = (
                local_max * fx.Int32(_FP4_INV_MAX_POS_BITS).bitcast(fx.Float32)
            ).bitcast(fx.Int32)
            mantissa = working & fx.Int32(0x7FFFFF)
            biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
            e8m0 = (mantissa != fx.Int32(0)).select(
                biased_exp + fx.Int32(1), biased_exp
            )
            e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
            rec_base = token * fx.Int32(rec)
            buffer_ops.buffer_store(
                e8m0.to(fx.Uint8),
                out_rs,
                rec_base + fx.Int32(off_s) + gi,
                offset_is_bytes=True,
            )
            dequant_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
            words = []
            for word in range_constexpr(GROUP // 8):
                packed = fx.Int32(0)
                for pair in range_constexpr(4):
                    idx = word * 8 + pair * 2
                    packed = rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32, packed, act[idx], act[idx + 1], dequant_scale, pair
                    )
                words.append(packed)
            buffer_ops.buffer_store(
                fx.Vector.from_elements(words, fx.Int32),
                out_rs,
                rec_base + fx.Int32(off_q) + gi * fx.Int32(16),
                offset_is_bytes=True,
            )
            # 每 token 前 2*topk 个 group 线程顺带把 ids / weights 写进 record。
            if gi < fx.Int32(topk):
                ids_rs = buffer_ops.create_buffer_resource_from_addr(topk_ids)
                v = buffer_ops.buffer_load(
                    ids_rs, token * fx.Int32(topk) + gi, vec_width=1, dtype=T.i32
                )
                buffer_ops.buffer_store(
                    v, out_rs, rec_base + fx.Int32(off_i) + gi * fx.Int32(4),
                    offset_is_bytes=True,
                )
                if const_expr(side_off):
                    buffer_ops.buffer_store(
                        v,
                        out_rs,
                        fx.Int32(side_off) + token * fx.Int32(side_rec) + gi * fx.Int32(4),
                        offset_is_bytes=True,
                    )
            if (gi >= fx.Int32(topk)) & (gi < fx.Int32(2 * topk)):
                w_rs = buffer_ops.create_buffer_resource_from_addr(route_weights)
                v = buffer_ops.buffer_load(
                    w_rs,
                    token * fx.Int32(topk) + gi - fx.Int32(topk),
                    vec_width=1,
                    dtype=T.i32,
                )
                buffer_ops.buffer_store(
                    v,
                    out_rs,
                    rec_base + fx.Int32(off_w) + (gi - fx.Int32(topk)) * fx.Int32(4),
                    offset_is_bytes=True,
                )
                if const_expr(side_off):
                    buffer_ops.buffer_store(
                        v,
                        out_rs,
                        fx.Int32(side_off)
                        + token * fx.Int32(side_rec)
                        + fx.Int32(topk * 4)
                        + (gi - fx.Int32(topk)) * fx.Int32(4),
                        offset_is_bytes=True,
                    )

    @flyc.jit
    def launch(
        x: fx.Int64,
        topk_ids: fx.Int64,
        route_weights: fx.Int64,
        out: fx.Int64,
        m: fx.Int32,
        grid_blocks: fx.Int32,
        stream: fx.Stream,
    ):
        rec_quant_kernel(x, topk_ids, route_weights, out, m).launch(
            grid=(fx.Int64(grid_blocks), 1, 1), block=(BLOCK, 1, 1), stream=stream
        )

    launch.rec_bytes = rec
    launch.layout = (rec, off_q, off_s, off_i, off_w)
    return launch


_CACHE = {}


def get_rail_record_quant(hidden: int, topk: int, side_off: int = 0):
    key = (int(hidden), int(topk), int(side_off))
    if key not in _CACHE:
        _CACHE[key] = build_rail_record_quant(*key)
    return _CACHE[key]
