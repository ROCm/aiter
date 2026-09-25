# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 FlyDSL Project Contributors

"""gfx950 FP8 qlen8/GQA16 page64 decode with a short-context precision path.

Four waves own all 128 query/head rows. KV storage remains FP8 throughout:
contexts through 512 use BF16 Q/P and native FP8-to-BF16 operand conversion;
longer contexts use FP8 Q/P with a 128-token tile. Both branches accumulate
softmax and output in f32 and preserve the normalized-partial reducer ABI.
The branch is uniform per sequence and requires no host synchronization.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, const_expr, gpu, range_constexpr
from flydsl.expr.typing import ReductionOp, T

from aiter.ops.flydsl.kernels.kernels_common import LOG2E
from aiter.ops.flydsl.kernels.pa_decode_common import (
    async_load_lds_nt,
    cdiv,
    exp2_amdgcn_scalar,
    exp2_f32_fast,
    load_lds_words,
    make_flat_loader,
    page_resource,
    rcp_f32,
    reduce_lane_pair,
    swap_lane_pair,
)
from aiter.ops.flydsl.kernels.tensor_shim import buf_scalar_load

BF16_CONTEXT_LIMIT = 512


@functools.cache
def compile_pa_decode_fp8_wave(head_dim, num_partitions, softmax_scale):
    D = head_dim
    NP = num_partitions
    SCALE = softmax_scale if softmax_scale is not None else D**-0.5
    KG = D // 64
    BKG = D // 16
    BKV_BYTES = (D + 128) * 64
    KV_BYTES = (D + 128) * 128

    @fx.struct
    class SharedStorage:
        data: fx.Array[fx.Int32, 2 * KV_BYTES // 4, 16]

    @flyc.kernel(known_block_size=(256, 1, 1))
    def pa_decode_fp8_wave32_kernel(
        output_ptr: fx.Tensor,
        pmax_ptr: fx.Tensor,
        psum_ptr: fx.Tensor,
        pout_ptr: fx.Tensor,
        query_ptr: fx.Tensor,
        key_ptr: fx.Tensor,
        value_ptr: fx.Tensor,
        tables_ptr: fx.Tensor,
        lengths_ptr: fx.Tensor,
        key_scale_ptr: fx.Tensor,
        value_scale_ptr: fx.Tensor,
        max_blocks: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
    ):
        shared = fx.SharedAllocator().allocate(SharedStorage).peek()
        lds_base = fx.recast_iter(fx.Int8, shared.data.ptr)

        # Writable scalar outputs must be explicit DSL helper arguments.
        def _fp8_path(pmax_ptr, psum_ptr):
            tid = fx.Int32(gpu.thread_id("x"))
            wave = tid // 64
            lane = tid % 64
            col = lane % 32
            half = lane // 32
            seq = fx.Int32(gpu.block_id("x"))
            kv_h = fx.Int32(gpu.block_id("y"))
            part = fx.Int32(gpu.block_id("z"))
            n_kv = fx.Int32(gpu.grid_dim.y)
            row = wave * 32 + col
            qi = row // 16
            qh = kv_h * 16 + row % 16

            q_load = make_flat_loader(query_ptr, fx.BFloat16, 8, fx.UniversalCopy128b())

            tables = fx.rocdl.make_buffer_tensor(
                tables_ptr,
                max_size=False,
                num_records_bytes=fx.Int64(gpu.grid_dim.x) * fx.Int64(max_blocks) * 4,
            )
            lengths = fx.rocdl.make_buffer_tensor(lengths_ptr, max_size=True)
            ks = fx.rocdl.make_buffer_tensor(key_scale_ptr, max_size=True)
            vs = fx.rocdl.make_buffer_tensor(value_scale_ptr, max_size=True)
            context = fx.Int32(buf_scalar_load(lengths, seq))
            key_scale = fx.Int32(buf_scalar_load(ks, fx.Int32(0))).bitcast(fx.Float32)
            value_scale = fx.Int32(buf_scalar_load(vs, fx.Int32(0))).bitcast(fx.Float32)

            def _page(tt):
                return fx.Int64(
                    fx.Int32(buf_scalar_load(tables, seq * max_blocks + tt))
                )

            def _stage_kv(tt, buf):
                for pg in range_constexpr(2):
                    page = _page(fx.min(tt * 2 + pg, cdiv(context, 64) - 1))
                    k_src = page_resource(
                        key_ptr, (page * n_kv + kv_h) * (D * 64), D * 64
                    )
                    v_src = page_resource(value_ptr, (page * n_kv + kv_h) * 8192, 8192)
                    for u in range_constexpr(KG):
                        local = tid * 16 + u * 4096
                        async_load_lds_nt(
                            k_src,
                            local,
                            lds_base,
                            buf * KV_BYTES + pg * D * 64 + wave * 1024 + u * 4096,
                        )
                    for u in range_constexpr(2):
                        local = tid * 16 + u * 4096
                        async_load_lds_nt(
                            v_src,
                            local,
                            lds_base,
                            buf * KV_BYTES
                            + D * 128
                            + pg * 8192
                            + wave * 1024
                            + u * 4096,
                        )
                fx.rocdl.asyncmark()

            def _k_ops(buf, nt, kg):
                ops = []
                for u in range_constexpr(2):
                    off = (
                        buf * KV_BYTES
                        + (nt // 2) * D * 64
                        + ((kg * 4 + half * 2 + u) * 64 + (nt % 2) * 32 + col) * 16
                    )
                    words = load_lds_words(lds_base, off)
                    ops.extend([words[i] for i in range_constexpr(4)])
                return fx.Vector.from_elements(ops, dtype=fx.Int32)

            def _v_tile(buf, vh, step):
                words = []
                for u in range_constexpr(2):
                    token_group = half * 2 + u
                    off = (
                        buf * KV_BYTES
                        + D * 128
                        + step * 8192
                        + (token_group * 128 + vh * 32 + col) * 16
                    )
                    chunk = load_lds_words(lds_base, off)
                    words.extend([chunk[i] for i in range_constexpr(4)])
                return fx.Vector.from_elements(words, dtype=fx.Int32)

            def _pack(values):
                words = []
                for g in range_constexpr(len(values) // 4):
                    j = 4 * g
                    lo = fx.rocdl.cvt_pk_fp8_f32(
                        T.i32, values[j], values[j + 1], 0, False
                    )
                    words.append(
                        fx.Int32(
                            fx.rocdl.cvt_pk_fp8_f32(
                                T.i32, values[j + 2], values[j + 3], lo, True
                            )
                        )
                    )
                return fx.Vector.from_elements(words, dtype=fx.Int32)

            def _pack_p(values):
                words = []
                for g in range_constexpr(4):
                    j = 4 * g
                    word = fx.Vector.filled(2, 0, fx.Int16).ir_value()
                    word = fx.rocdl.cvt_scalef32_pk_fp8_f32(
                        T.vec(2, T.i16),
                        word,
                        values[j].ir_value(),
                        values[j + 1].ir_value(),
                        fx.Float32(1.0 / 256.0).ir_value(),
                        False,
                    )
                    word = fx.rocdl.cvt_scalef32_pk_fp8_f32(
                        T.vec(2, T.i16),
                        word,
                        values[j + 2].ir_value(),
                        values[j + 3].ir_value(),
                        fx.Float32(1.0 / 256.0).ir_value(),
                        True,
                    )
                    words.append(fx.Vector(word).bitcast(fx.Int32)[0])
                return fx.Vector.from_elements(words, dtype=fx.Int32)

            def _mma(a, b, acc):
                return fx.Vector(
                    fx.rocdl.mfma_scale_f32_32x32x64_f8f6f4(
                        T.vec(16, T.f32),
                        [
                            a,
                            b,
                            acc,
                            0,
                            0,
                            0,
                            fx.Int32(0x7F7F7F7F),
                            0,
                            fx.Int32(0x7F7F7F7F),
                        ],
                    )
                )

            q_units = []
            q_base = (seq * 8 + qi) * stride_q_row + qh * stride_q_head
            for kg in range_constexpr(KG):
                for u in range_constexpr(4):
                    q_units.append(q_load(q_base + kg * 64 + half * 32 + u * 8))
            amax = fx.Float32(0.0)
            for u in range_constexpr(KG * 4):
                amax = fx.maxnumf(
                    amax, fx.absf(q_units[u].to(fx.Float32)).reduce(ReductionOp.MAX)
                )
            amax_lo, amax_hi = reduce_lane_pair(amax)
            amax = fx.maxnumf(amax_lo, amax_hi)
            q_scale = fx.maxnumf(amax * fx.Float32(1.0 / 448.0), fx.Float32(1e-20))
            inv_q = fx.Float32(rcp_f32(q_scale))
            q_ops = []
            for kg in range_constexpr(KG):
                values = []
                for u in range_constexpr(4):
                    for i in range_constexpr(8):
                        values.append(q_units[kg * 4 + u][i].to(fx.Float32) * inv_q)
                q_ops.append(_pack(values))
            scale = fx.Float32(SCALE * LOG2E) * key_scale * q_scale
            zero = fx.Float32(0.0)
            neg_inf = fx.Float32(float("-inf"))
            num_tiles = cdiv(context, 128)
            per_part = cdiv(num_tiles, NP)
            begin = part * per_part
            end = fx.min(begin + per_part, num_tiles)
            if begin < end:
                _stage_kv(begin, fx.Int32(0))
                fx.rocdl.wait_asyncmark(0)
            gpu.barrier()
            out_zero = fx.Vector.filled(16, 0.0, fx.Float32)
            init = [neg_inf, zero, out_zero, out_zero, out_zero, out_zero]
            for tt, state in range(begin, end, 1, init=init):
                tt = fx.Int32(tt)
                buf = (tt - begin) & fx.Int32(1)
                if tt + 1 < end:
                    _stage_kv(tt + 1, buf ^ fx.Int32(1))
                scores = []
                for nt in range_constexpr(4):
                    acc = fx.Vector.filled(16, 0.0, fx.Float32)
                    for kg in range_constexpr(KG):
                        a = _k_ops(buf, nt, kg)
                        acc = _mma(a, q_ops[kg], acc)
                    score = acc
                    if (tt + 1) * 128 > context - 7:
                        tokens = fx.Vector.from_elements(
                            [
                                tt * 128 + nt * 32 + half * 4 + (i % 4) + (i // 4) * 8
                                for i in range_constexpr(16)
                            ],
                            dtype=fx.Int32,
                        )
                        bound = fx.Vector.from_elements(
                            [context - 7 + qi], dtype=fx.Int32
                        ).broadcast_to(16)
                        score = (tokens < bound).select(
                            acc, fx.Vector.filled(16, float("-inf"), fx.Float32)
                        )
                    scores.append(score)

                local_max = fx.maxnumf(
                    scores[0].reduce(ReductionOp.MAX), scores[1].reduce(ReductionOp.MAX)
                )
                for nt in range_constexpr(2, 4):
                    local_max = fx.maxnumf(
                        local_max, scores[nt].reduce(ReductionOp.MAX)
                    )
                max_lo, max_hi = reduce_lane_pair(local_max)
                tile_max = fx.maxnumf(max_lo, max_hi) * scale
                new_max = fx.maxnumf(state[0], tile_max)
                safe_max = (new_max > neg_inf).select(new_max, zero)
                max_vec = fx.Vector.from_elements(
                    [safe_max], dtype=fx.Float32
                ).broadcast_to(16)
                probs = [
                    fx.Vector(
                        exp2_f32_fast(
                            scores[nt]
                            * fx.Vector.from_elements(
                                [scale], dtype=fx.Float32
                            ).broadcast_to(16)
                            - max_vec
                        )
                    )
                    for nt in range_constexpr(4)
                ]
                local_sum = probs[0].reduce(ReductionOp.ADD) + probs[1].reduce(
                    ReductionOp.ADD
                )
                for nt in range_constexpr(2, 4):
                    local_sum = local_sum + probs[nt].reduce(ReductionOp.ADD)
                sum_lo, sum_hi = reduce_lane_pair(local_sum)
                tile_sum = sum_lo + sum_hi
                corr = fx.Float32(exp2_amdgcn_scalar(state[0] - safe_max))
                new_sum = state[1] * corr + tile_sum
                p_words = [
                    _pack_p([probs[nt][i] for i in range_constexpr(16)])
                    for nt in range_constexpr(4)
                ]
                p = []
                for step in range_constexpr(2):
                    packed = []
                    for g in range_constexpr(4):
                        low, high = swap_lane_pair(
                            p_words[2 * step][g], p_words[2 * step + 1][g]
                        )
                        packed.extend([low, high])
                    p.append(fx.Vector.from_elements(packed, dtype=fx.Int32))
                updated = []
                corr_vec = fx.Vector.from_elements(
                    [corr], dtype=fx.Float32
                ).broadcast_to(16)
                for vh in range_constexpr(4):
                    out = state[2 + vh] * corr_vec
                    for step in range_constexpr(2):
                        out = _mma(_v_tile(buf, vh, step), p[step], out)
                    updated.append(out)
                fx.rocdl.wait_asyncmark(0)
                gpu.barrier()
                results = yield [new_max, new_sum, *updated]

            denom = results[1]
            safe_denom = (denom > zero).select(denom, fx.Float32(1.0))
            out_scale = (
                fx.Float32(rcp_f32(safe_denom)) * value_scale * fx.Float32(1.0 / 256.0)
            )
            for vh in range_constexpr(4):
                for g in range_constexpr(4):
                    norm = fx.Vector.from_elements(
                        [
                            results[2 + vh][g * 4 + i] * out_scale
                            for i in range_constexpr(4)
                        ],
                        dtype=fx.Float32,
                    ).to(fx.BFloat16)
                    sub = (vh * 32 + half * 4 + g * 8) // 4
                    if const_expr(NP == 1):
                        out_row = output_ptr[seq * 8 + qi, qh, None]
                        fx.slice(
                            fx.logical_divide(out_row, fx.make_layout(4, 1)),
                            (None, sub),
                        ).store(norm)
                    else:
                        base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                        div = fx.logical_divide(pout_ptr, fx.make_layout(4, 1))
                        fx.slice(div, (None, base * 32 + sub)).store(norm)
            if const_expr(NP > 1):  # noqa: SIM102 - Static NP, dynamic lane mask.
                if half == 0:
                    base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                    pmax_ptr[base] = results[0] * fx.Float32(1.0 / LOG2E)
                    psum_ptr[base] = denom

        def _bf16_path(pmax_ptr, psum_ptr):
            tid = fx.Int32(gpu.thread_id("x"))
            wave = tid // 64
            lane = tid % 64
            col = lane % 32
            half = lane // 32
            seq = fx.Int32(gpu.block_id("x"))
            kv_h = fx.Int32(gpu.block_id("y"))
            part = fx.Int32(gpu.block_id("z"))
            n_kv = fx.Int32(gpu.grid_dim.y)
            row = wave * 32 + col
            qi = row // 16
            qh = kv_h * 16 + row % 16

            q_load = make_flat_loader(query_ptr, fx.BFloat16, 8, fx.UniversalCopy128b())

            tables = fx.rocdl.make_buffer_tensor(
                tables_ptr,
                max_size=False,
                num_records_bytes=fx.Int64(gpu.grid_dim.x) * fx.Int64(max_blocks) * 4,
            )
            lengths = fx.rocdl.make_buffer_tensor(lengths_ptr, max_size=True)
            context = fx.Int32(buf_scalar_load(lengths, seq))
            ks = fx.rocdl.make_buffer_tensor(key_scale_ptr, max_size=True)
            vs = fx.rocdl.make_buffer_tensor(value_scale_ptr, max_size=True)
            key_scale = fx.Int32(buf_scalar_load(ks, fx.Int32(0))).bitcast(fx.Float32)
            value_scale = fx.Int32(buf_scalar_load(vs, fx.Int32(0))).bitcast(fx.Float32)

            def _page(tt):
                return fx.Int64(
                    fx.Int32(buf_scalar_load(tables, seq * max_blocks + tt))
                )

            def _stage_kv(tt, buf):
                page = _page(tt)
                # Rebase each bounded descriptor at a 64-bit physical-page address.
                k_src = page_resource(key_ptr, (page * n_kv + kv_h) * (D * 64), D * 64)
                v_src = page_resource(value_ptr, (page * n_kv + kv_h) * 8192, 8192)
                for u in range_constexpr(D // 64):
                    local = tid * 16 + u * 4096
                    async_load_lds_nt(
                        k_src, local, lds_base, buf * BKV_BYTES + wave * 1024 + u * 4096
                    )
                for u in range_constexpr(2):
                    local = tid * 16 + u * 4096
                    async_load_lds_nt(
                        v_src,
                        local,
                        lds_base,
                        buf * BKV_BYTES + D * 64 + wave * 1024 + u * 4096,
                    )
                fx.rocdl.asyncmark()

            def _decode(words):
                values = []
                for word in range_constexpr(2):
                    for hi in range_constexpr(2):
                        pair = fx.Vector(
                            fx.rocdl.cvt_scalef32_pk_bf16_fp8(
                                T.vec(2, T.bf16),
                                words[word].ir_value(),
                                fx.Float32(1.0).ir_value(),
                                bool(hi),
                            )
                        )
                        values.extend([pair[0], pair[1]])
                return fx.Vector.from_elements(values, dtype=fx.BFloat16)

            def _k_ops(buf, nt, kg):
                off = buf * BKV_BYTES + (kg * 64 + nt * 32 + col) * 16 + half * 8
                return _decode(load_lds_words(lds_base, off, words=2))

            def _v_tile(buf, vh, step):
                off = (
                    buf * BKV_BYTES
                    + D * 64
                    + (step * 128 + vh * 32 + col) * 16
                    + half * 8
                )
                return _decode(load_lds_words(lds_base, off, words=2))

            def _mma(a, b, acc):
                return fx.Vector(
                    fx.rocdl.mfma_f32_32x32x16_bf16(
                        T.vec(16, T.f32), [a, b, acc, 0, 0, 0]
                    )
                )

            q_base = (seq * 8 + qi) * stride_q_row + qh * stride_q_head
            q_ops = [q_load(q_base + kg * 16 + half * 8) for kg in range_constexpr(BKG)]
            scale = fx.Float32(SCALE * LOG2E) * key_scale
            zero = fx.Float32(0.0)
            neg_inf = fx.Float32(float("-inf"))
            num_tiles = cdiv(context, 64)
            per_part = cdiv(num_tiles, NP)
            begin = part * per_part
            end = fx.min(begin + per_part, num_tiles)
            if begin < end:
                _stage_kv(begin, fx.Int32(0))
                fx.rocdl.wait_asyncmark(0)
            gpu.barrier()
            out_zero = fx.Vector.filled(16, 0.0, fx.Float32)
            init = [neg_inf, zero, out_zero, out_zero, out_zero, out_zero]
            for tt, state in range(begin, end, 1, init=init):
                tt = fx.Int32(tt)
                buf = (tt - begin) & fx.Int32(1)
                if tt + 1 < end:
                    _stage_kv(tt + 1, buf ^ fx.Int32(1))
                qk = [
                    fx.Vector.filled(16, 0.0, fx.Float32) for nt in range_constexpr(2)
                ]
                for kg in range_constexpr(BKG):
                    operands = [_k_ops(buf, nt, kg) for nt in range_constexpr(2)]
                    qk = [
                        _mma(operands[nt], q_ops[kg], qk[nt])
                        for nt in range_constexpr(2)
                    ]
                scores = []
                for nt in range_constexpr(2):
                    acc = qk[nt]
                    score = acc
                    if (tt + 1) * 64 > context - 7:
                        tokens = fx.Vector.from_elements(
                            [
                                tt * 64 + nt * 32 + half * 4 + (i % 4) + (i // 4) * 8
                                for i in range_constexpr(16)
                            ],
                            dtype=fx.Int32,
                        )
                        bound = fx.Vector.from_elements(
                            [context - 7 + qi], dtype=fx.Int32
                        ).broadcast_to(16)
                        score = (tokens < bound).select(
                            acc, fx.Vector.filled(16, float("-inf"), fx.Float32)
                        )
                    scores.append(score)

                local_max = fx.maxnumf(
                    scores[0].reduce(ReductionOp.MAX), scores[1].reduce(ReductionOp.MAX)
                )
                max_lo, max_hi = reduce_lane_pair(local_max)
                tile_max = fx.maxnumf(max_lo, max_hi) * scale
                new_max = fx.maxnumf(state[0], tile_max)
                safe_max = (new_max > neg_inf).select(new_max, zero)
                max_vec = fx.Vector.from_elements(
                    [safe_max], dtype=fx.Float32
                ).broadcast_to(16)
                probs = [
                    fx.Vector(
                        exp2_f32_fast(
                            scores[nt]
                            * fx.Vector.from_elements(
                                [scale], dtype=fx.Float32
                            ).broadcast_to(16)
                            - max_vec
                        )
                    )
                    for nt in range_constexpr(2)
                ]
                local_sum = probs[0].reduce(ReductionOp.ADD) + probs[1].reduce(
                    ReductionOp.ADD
                )
                sum_lo, sum_hi = reduce_lane_pair(local_sum)
                tile_sum = sum_lo + sum_hi
                # Keep the initial previous max at -inf, including for negative logits.
                corr = fx.Float32(exp2_amdgcn_scalar(state[0] - safe_max))
                new_sum = state[1] * corr + tile_sum
                p_words = [
                    probs[nt].to(fx.BFloat16).bitcast(fx.Int32)
                    for nt in range_constexpr(2)
                ]
                p = []
                for step in range_constexpr(4):
                    nt = step // 2
                    group = (step % 2) * 2
                    low = []
                    high = []
                    for word in range_constexpr(2):
                        a = p_words[nt][group * 2 + word]
                        b = p_words[nt][(group + 1) * 2 + word]
                        low_word, high_word = swap_lane_pair(a, b)
                        low.append(low_word)
                        high.append(high_word)
                    p.append(
                        fx.Vector.from_elements([*low, *high], dtype=fx.Int32).bitcast(
                            fx.BFloat16
                        )
                    )
                corr_vec = fx.Vector.from_elements(
                    [corr], dtype=fx.Float32
                ).broadcast_to(16)
                updated = [state[2 + vh] * corr_vec for vh in range_constexpr(4)]
                for step in range_constexpr(4):
                    v_ops = [_v_tile(buf, vh, step) for vh in range_constexpr(4)]
                    updated = [
                        _mma(v_ops[vh], p[step], updated[vh])
                        for vh in range_constexpr(4)
                    ]
                    # Finish next-tile DMA and all current-tile reads before reusing LDS.
                fx.rocdl.wait_asyncmark(0)
                gpu.barrier()
                results = yield [new_max, new_sum, *updated]

            denom = results[1]
            safe_denom = (denom > zero).select(denom, fx.Float32(1.0))
            out_scale = fx.Float32(rcp_f32(safe_denom)) * value_scale
            for vh in range_constexpr(4):
                for g in range_constexpr(4):
                    norm = fx.Vector.from_elements(
                        [
                            results[2 + vh][g * 4 + i] * out_scale
                            for i in range_constexpr(4)
                        ],
                        dtype=fx.Float32,
                    ).to(fx.BFloat16)
                    sub = (vh * 32 + half * 4 + g * 8) // 4
                    if const_expr(NP == 1):
                        out_row = output_ptr[seq * 8 + qi, qh, None]
                        fx.slice(
                            fx.logical_divide(out_row, fx.make_layout(4, 1)),
                            (None, sub),
                        ).store(norm)
                    else:
                        base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                        div = fx.logical_divide(pout_ptr, fx.make_layout(4, 1))
                        fx.slice(div, (None, base * 32 + sub)).store(norm)
            if const_expr(NP > 1):  # noqa: SIM102 - Static NP, dynamic lane mask.
                if half == 0:
                    base = ((seq * n_kv + kv_h) * NP + part) * 128 + row
                    pmax_ptr[base] = results[0] * fx.Float32(1.0 / LOG2E)
                    psum_ptr[base] = denom

        lengths_resource = fx.rocdl.make_buffer_tensor(lengths_ptr, max_size=True)
        sequence_context = fx.Int32(
            buf_scalar_load(lengths_resource, fx.Int32(gpu.block_id("x")))
        )
        # Short contexts amplify Q/P quantization error; keep KV storage FP8.
        if sequence_context <= fx.Int32(BF16_CONTEXT_LIMIT):
            _bf16_path(pmax_ptr, psum_ptr)
        else:
            _fp8_path(pmax_ptr, psum_ptr)

    @flyc.jit
    def launch(
        output: fx.Tensor,
        pmax: fx.Tensor,
        psum: fx.Tensor,
        pout: fx.Tensor,
        query: fx.Tensor,
        key: fx.Tensor,
        value: fx.Tensor,
        tables: fx.Tensor,
        lengths: fx.Tensor,
        key_scale: fx.Tensor,
        value_scale: fx.Tensor,
        max_blocks: fx.Int32,
        num_seqs: fx.Int32,
        num_kv: fx.Int32,
        stride_ks_block: fx.Int32,
        stride_ks_head: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL ABI default.
    ):
        with CompilationContext.compile_hints(
            {"fastmath": arith.FastMathFlags.contract}
        ):
            pa_decode_fp8_wave32_kernel(
                output,
                pmax,
                psum,
                pout,
                query,
                key,
                value,
                tables,
                lengths,
                key_scale,
                value_scale,
                max_blocks,
                stride_q_row,
                stride_q_head,
            ).launch(grid=(num_seqs, num_kv, NP), block=(256, 1, 1), stream=stream)

    return {"launch": launch, "kernel": pa_decode_fp8_wave32_kernel}
