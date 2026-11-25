# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Biased sigmoid top-4 of 128 experts over the router's split-K partials (one token per wave).

Ported from amd-kernels' ``moe/mxfp4_h6144_i768_e128_top4/router_topk_sorted.py`` (Apache-2.0, "Copyright contributors to the
amd-kernels project"); the kernel bodies are unchanged, only imports, the warm-up ``@launcher`` wrapper and the
compile-cache helper were replaced by aiter's own (``aiter.ops.flydsl.kernels.buffer_ops`` / ``tensor_shim``).
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir as _mlir_ir
from flydsl._mlir.dialects import arith as _arith
from flydsl._mlir.dialects import rocdl as _rocdl_ops
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from aiter.ops.flydsl.kernels import buffer_ops

from .decode_router_common import (
    DPP_TREE,
    NEG_LOG2E,
    NW,
    PRELOAD_HINTS,
    _dpp,
    _ir,
)

H = 6144
assert NW == 1


# DPP steps whose source lane is valid in every lane (quad_perm, row_half_mirror, row_mirror)
FULL_DPP = (0xB1, 0x4E, 0x141, 0x140)


def _dpp_full(v, ctrl):
    """``_dpp`` for a step with a valid source in every lane: with bound_ctrl the old value is
    not read, so LLVM needs no copy of ``v`` before the v_mov_b32_dpp (same result)."""
    return fx.Int32(_rocdl_ops.update_dpp(T.i32, _ir(v), _ir(v), ctrl, 0xF, 0xF, True))


def _maxf_nn(a, b):
    """v_max_f32 with nnan (the queue holds no NaN): no canonicalization, and the value does not wait for the index."""
    fm = _mlir_ir.Attribute.parse("#arith.fastmath<nnan>")
    return fx.Float32(
        _arith.MaxNumFOp(fx.as_ir_value(a), fx.as_ir_value(b), fastmath=fm).result
    )


def compile_router_topk_sorted(
    num_experts: int, topk: int, split_k: int, token_major: bool = False
):
    """Each lane sorts its four choices once; the winning lane pops its head after each pick.

    Strict adjacent swaps keep the lower component first on ties. The tree's sixth DPP step only copies lane 31's
    result to lane 63, so the tree stops at five steps and reads lane 31.

    Args:
        token_major: contiguous [m, S, 128] partials, read at constant split offsets.
    """
    assert num_experts == 128 and topk == 4 and split_k % 8 == 0 and 8 <= split_k <= 96
    E, K, S = num_experts, topk, split_k
    name = (
        f"router_splitk_biased_top{K}_sorted_h{H}_e{E}_s{S}"
        + "_half_tree_maxnum_direct_weight"
        + ("_token_partials" if token_major else "")
    )

    @flyc.kernel(name=name, known_block_size=[64 * NW, 1, 1])
    def kernel(
        arg_part: fx.Int64,  # split-K partials, fp32, [M, S, E] by strides
        arg_bias: fx.Int64,  # correction bias [E] fp32
        arg_w: fx.Int64,  # top-k weights fp32, row stride i32_stk
        arg_ids: fx.Int64,  # top-k ids int32, row stride i32_stk
        i64_part_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sm: fx.Int32,  # partial strides (elements): token, split
        i32_sk: fx.Int32,
        i32_stk: fx.Int32,
        i32_scale: fx.Int32,  # routed_scaling_factor (fp32 bits)
    ):
        tx, pid = fx.thread_idx.x, fx.block_idx.x
        lane = tx % 64
        wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, tx // 64))
        t = pid * NW + wave
        if t < i32_m:
            pr = buffer_ops.create_buffer_resource_from_addr(
                arg_part, num_records_bytes=i64_part_bytes
            )
            br = buffer_ops.create_buffer_resource_from_addr(
                arg_bias, num_records_bytes=E * 4
            )
            out_bytes = fx.Int64(i32_m) * fx.Int64(i32_stk) * 4
            wr = buffer_ops.create_buffer_resource_from_addr(
                arg_w, num_records_bytes=out_bytes
            )
            ir = buffer_ops.create_buffer_resource_from_addr(
                arg_ids, num_records_bytes=out_bytes
            )
            has = lane < E // 4
            el = lane % 32 * 4 + lane // 32 * 2
            base = t * (S * E if const_expr(token_major) else i32_sm) + el
            parts = [
                Vec(
                    buffer_ops.buffer_load(
                        pr,
                        base + k * (E if const_expr(token_major) else i32_sk),
                        vec_width=2,
                        dtype=fx.Float32,
                    )
                )
                for k in range_constexpr(S)
            ]
            bias = Vec(
                buffer_ops.buffer_load(
                    br, lane * 4, vec_width=4, dtype=fx.Float32, mask=has
                )
            )
            neg_inf = fx.Float32(float("-inf"))
            ss = []
            for i in range_constexpr(2):
                x = fx.Float32(0.0)
                for k in range_constexpr(S):
                    x = x + fx.Float32(parts[k][i])
                y = (x.to(fx.Float64) * fx.Float64(NEG_LOG2E)).to(fx.Float32)
                ss.append(fx.Float32(fx.rocdl.rcp(T.f32, fx.Float32(1.0) + y.exp2())))
            # components 2 / 3 back to the quartet's lower lane, for the same argmax tree and ties
            sig = [
                ss[0],
                ss[1],
                fx.Int32(
                    fx.rocdl.ds_bpermute(
                        T.i32, (lane ^ 32) * 4, ss[0].bitcast(fx.Int32)
                    )
                ).bitcast(fx.Float32),
                fx.Int32(
                    fx.rocdl.ds_bpermute(
                        T.i32, (lane ^ 32) * 4, ss[1].bitcast(fx.Int32)
                    )
                ).bitcast(fx.Float32),
            ]
            choice = [
                has.select(sig[i] + fx.Float32(bias[i]), neg_inf)
                for i in range_constexpr(4)
            ]
            # Strict argmax ignores NaNs and -inf; keep that behavior in the queue.
            choice = [(v > neg_inf).select(v, neg_inf) for v in choice]
            local_ids = [lane * 4 + i for i in range_constexpr(4)]
            # stable insertion network: ties keep the component order
            for a, b in ((0, 1), (1, 2), (0, 1), (2, 3), (1, 2), (0, 1)):
                take = choice[b] > choice[a]
                va, vb = choice[a], choice[b]
                ia, ib = local_ids[a], local_ids[b]
                choice[a], choice[b] = take.select(vb, va), take.select(va, vb)
                local_ids[a], local_ids[b] = take.select(ib, ia), take.select(ia, ib)
            total = fx.Float32(0.0)
            out_id = fx.Int32(0)
            out_val = fx.Float32(0.0)
            for k in range_constexpr(K):
                mv = choice[0]
                mi = (mv > neg_inf).select(local_ids[0], fx.Int32(k))
                for ctrl in DPP_TREE[:-1]:
                    dpp = _dpp_full if ctrl in FULL_DPP else _dpp
                    rv = dpp(mv.bitcast(fx.Int32), ctrl).bitcast(fx.Float32)
                    ri = dpp(mi, ctrl)
                    take = rv > mv
                    mv = _maxf_nn(rv, mv)
                    mi = take.select(ri, mi)
                idx = fx.Int32(fx.rocdl.readlane(T.i32, mi, fx.Int32(31)))
                owner, comp = idx >> 2, idx & 3
                # expert idx: element comp % 2 of lane owner + 32 (comp // 2), also the strict argmax's fallback index
                source_lane = owner + ((comp & 2) << 4)
                local_w = ((comp & 1) == 1).select(ss[1], ss[0])
                w = fx.Int32(
                    fx.rocdl.readlane(T.i32, local_w.bitcast(fx.Int32), source_lane)
                ).bitcast(fx.Float32)
                # Only the lane that supplied the global pick advances its queue.
                pick = lane == owner
                for i in range_constexpr(3):
                    choice[i] = pick.select(choice[i + 1], choice[i])
                    local_ids[i] = pick.select(local_ids[i + 1], local_ids[i])
                choice[3] = pick.select(neg_inf, choice[3])
                total = total + w
                out_id = (lane == k).select(idx, out_id)
                out_val = (lane == k).select(w, out_val)
            scale = i32_scale.bitcast(fx.Float32)
            factor = scale / total
            off = t * i32_stk + lane
            buffer_ops.buffer_store(out_val * factor, wr, off, mask=lane < K)
            buffer_ops.buffer_store(out_id, ir, off, mask=lane < K)

    @flyc.jit
    def launch_router_topk_sorted(
        arg_part: fx.Int64,
        arg_bias: fx.Int64,
        arg_w: fx.Int64,
        arg_ids: fx.Int64,
        i64_part_bytes: fx.Int64,
        i32_m: fx.Int32,
        i32_sm: fx.Int32,
        i32_sk: fx.Int32,
        i32_stk: fx.Int32,
        i32_scale: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            arg_part,
            arg_bias,
            arg_w,
            arg_ids,
            i64_part_bytes,
            i32_m,
            i32_sm,
            i32_sk,
            i32_stk,
            i32_scale,
        ).launch(
            grid=(fx.Int64(i32_m), 1, 1),
            block=(64 * NW, 1, 1),
            stream=stream,
        )

    launch_router_topk_sorted.kernel_name = name
    launch_router_topk_sorted.compile_hints = PRELOAD_HINTS
    return launch_router_topk_sorted


@functools.cache
def _get_kernel(num_experts: int, topk: int, split_k: int, token_major: bool = False):
    return compile_router_topk_sorted(num_experts, topk, split_k, token_major)
