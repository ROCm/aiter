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

from ..mega_moe.quant import (
    _FP4_INV_MAX_POS_BITS,
    _FP8_E4M3_INV_MAX_POS_BITS,
    BLOCK,
    GROUP,
)
from ..tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor


def rail_record_layout(hidden: int, topk: int, activation_dtype: str = "fp4"):
    """(REC_BYTES, off_q, off_scale, off_ids, off_weights),全部 16B 对齐。"""
    if activation_dtype not in ("fp4", "fp8"):
        raise ValueError("activation_dtype must be fp4 or fp8")
    off_q = 0
    off_s = off_q + hidden // (1 if activation_dtype == "fp8" else 2)
    off_i = off_s + hidden // 32
    # ids 段补齐到 16B:topk*4 不是 16 的倍数时(如 topk=6),weights 起点仍对齐。
    off_w = off_i + (topk * 4 + 15) // 16 * 16
    end = off_w + topk * 4
    rec = (end + 255) // 256 * 256
    for v in (off_s, off_i, off_w, rec):
        assert v % 16 == 0, (hidden, topk, v)
    return rec, off_q, off_s, off_i, off_w


def build_rail_record_quant(
    hidden: int, topk: int, activation_dtype: str = "fp4", pad_tokens: int = 0
):
    assert hidden % GROUP == 0
    if pad_tokens < 0:
        raise ValueError("pad_tokens must be nonnegative")
    scale_n = hidden // GROUP
    assert scale_n >= 2 * topk, "ids/weights 由每 token 前 2*topk 个 group 线程顺带写"
    rec, off_q, off_s, off_i, off_w = rail_record_layout(hidden, topk, activation_dtype)

    inv_max_bits = (
        _FP8_E4M3_INV_MAX_POS_BITS
        if activation_dtype == "fp8"
        else _FP4_INV_MAX_POS_BITS
    )

    @flyc.kernel(
        name=(
            f"per_1x32_mx_quant_{activation_dtype}_n{hidden}_railrec_k{topk}_r{rec}"
            + (f"_pad{pad_tokens}" if pad_tokens else "")
        )
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
        rows = fx.Int32(pad_tokens) if const_expr(pad_tokens > 0) else m
        if group_id < rows * fx.Int32(scale_n):
            token = group_id // fx.Int32(scale_n)
            gi = group_id % fx.Int32(scale_n)

            @flyc.jit
            def write_live_record():
                x_buf = ptr_buf_tensor(x, fx.Int32, unit_elems=4)
                out_q = ptr_buf_tensor(out, fx.Int32, unit_elems=4)
                out_scale = ptr_buf_tensor(out, fx.Uint8)
                out_meta = ptr_buf_tensor(out, fx.Int32)
                # 与 quant.py fp4 路径相同的数学。
                in_vec = group_id * fx.Int32(GROUP * 2 // 16)
                act = []
                local_max = fx.Float32(1e-10)
                for chunk in range_constexpr(GROUP // 8):
                    raw = buf_copy_load(
                        x_buf, in_vec + fx.Int32(chunk), fx.Int32, unit_elems=4
                    )
                    values = fx.Vector(raw).bitcast(fx.BFloat16).to(fx.Float32)
                    local_max = local_max.maximumf(
                        fmath.absf(values).reduce(ReductionOp.MAX)
                    )
                    for elem in range_constexpr(8):
                        act.append(values[elem])
                working = (
                    local_max * fx.Int32(inv_max_bits).bitcast(fx.Float32)
                ).bitcast(fx.Int32)
                mantissa = working & fx.Int32(0x7FFFFF)
                biased_exp = (working >> fx.Int32(23)) & fx.Int32(0xFF)
                e8m0 = (mantissa != fx.Int32(0)).select(
                    biased_exp + fx.Int32(1), biased_exp
                )
                e8m0 = (e8m0 > fx.Int32(255)).select(fx.Int32(255), e8m0)
                rec_base = token * fx.Int32(rec)
                out_scale[rec_base + fx.Int32(off_s) + gi] = e8m0.to(fx.Uint8)
                if const_expr(activation_dtype == "fp4"):
                    dequant_scale = (e8m0 << fx.Int32(23)).bitcast(fx.Float32)
                    words = []
                    for word in range_constexpr(GROUP // 8):
                        packed = fx.Int32(0)
                        for pair in range_constexpr(4):
                            idx = word * 8 + pair * 2
                            packed = rocdl.cvt_scalef32_pk_fp4_f32(
                                T.i32,
                                packed,
                                act[idx],
                                act[idx + 1],
                                dequant_scale,
                                pair,
                            )
                        words.append(packed)
                    buf_copy_store(
                        out_q,
                        (rec_base + fx.Int32(off_q)) // fx.Int32(16) + gi,
                        fx.Vector.from_elements(words, fx.Int32),
                        fx.Int32,
                        unit_elems=4,
                    )
                else:
                    # Match the shared BF16 -> MXFP8 converter, including reciprocal scale.
                    quant_scale = ((fx.Int32(254) - e8m0) << fx.Int32(23)).bitcast(
                        fx.Float32
                    )
                    scaled = [v * quant_scale for v in act]
                    out_vec = (rec_base + fx.Int32(off_q)) // fx.Int32(
                        16
                    ) + gi * fx.Int32(2)
                    for half in range_constexpr(2):
                        words = []
                        for word in range_constexpr(4):
                            idx = (half * 4 + word) * 4
                            packed = rocdl.cvt_pk_fp8_f32(
                                T.i32, scaled[idx], scaled[idx + 1], fx.Int32(0), 0
                            )
                            packed = rocdl.cvt_pk_fp8_f32(
                                T.i32, scaled[idx + 2], scaled[idx + 3], packed, 1
                            )
                            words.append(packed)
                        buf_copy_store(
                            out_q,
                            out_vec + fx.Int32(half),
                            fx.Vector.from_elements(words, fx.Int32),
                            fx.Int32,
                            unit_elems=4,
                        )
                # 每 token 前 2*topk 个 group 线程顺带把 ids / weights 写进 record。
                if gi < fx.Int32(topk):
                    ids_buf = ptr_buf_tensor(topk_ids, fx.Int32)
                    out_meta[(rec_base + fx.Int32(off_i)) // fx.Int32(4) + gi] = (
                        ids_buf[token * fx.Int32(topk) + gi]
                    )
                if (gi >= fx.Int32(topk)) & (gi < fx.Int32(2 * topk)):
                    weights_buf = ptr_buf_tensor(route_weights, fx.Int32)
                    weight_slot = gi - fx.Int32(topk)
                    out_meta[
                        (rec_base + fx.Int32(off_w)) // fx.Int32(4) + weight_slot
                    ] = weights_buf[token * fx.Int32(topk) + weight_slot]

            if const_expr(pad_tokens > 0):
                if token < m:
                    write_live_record()
                else:
                    # Empty routes keep the fixed-capacity rail protocol intact.
                    out_q = ptr_buf_tensor(out, fx.Int32, unit_elems=4)
                    out_scale = ptr_buf_tensor(out, fx.Uint8)
                    out_meta = ptr_buf_tensor(out, fx.Int32)
                    rec_base = token * fx.Int32(rec)
                    out_scale[rec_base + fx.Int32(off_s) + gi] = fx.Uint8(127)
                    qvecs = 2 if activation_dtype == "fp8" else 1
                    for half in range_constexpr(qvecs):
                        buf_copy_store(
                            out_q,
                            rec_base // fx.Int32(16)
                            + gi * fx.Int32(qvecs)
                            + fx.Int32(half),
                            fx.Vector.filled(4, 0, fx.Int32),
                            fx.Int32,
                            unit_elems=4,
                        )
                    if gi < fx.Int32(topk):
                        out_meta[
                            rec_base // fx.Int32(4) + fx.Int32(off_i // 4) + gi
                        ] = fx.Int32(-1)
                    if (gi >= fx.Int32(topk)) & (gi < fx.Int32(2 * topk)):
                        out_meta[
                            rec_base // fx.Int32(4)
                            + fx.Int32(off_w // 4)
                            + gi
                            - fx.Int32(topk)
                        ] = fx.Int32(0)
            else:
                write_live_record()

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


def get_rail_record_quant(
    hidden: int, topk: int, activation_dtype: str = "fp4", pad_tokens: int = 0
):
    key = (int(hidden), int(topk), activation_dtype, int(pad_tokens))
    if key not in _CACHE:
        _CACHE[key] = build_rail_record_quant(*key)
    return _CACHE[key]
