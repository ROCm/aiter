# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx950 sparse-MLA decode producer with 64-key split granularity."""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import math as fly_math
from flydsl.expr.typing import T

# Hardware and instruction shape. Both MFMA atoms below are 16x16x*, and the
# lane map ties one head to one MFMA row, so the kernel always runs 16 head
# slots -- `H` is that tile width, not the model's head count. A model with
# fewer heads is read in place through the `q_heads` argument.
WAVE_SIZE = 64
MFMA_M = 16
MFMA_N = 16
H = MFMA_M
PARTIAL_WAVES = 4
PARTIAL_THREADS = WAVE_SIZE * PARTIAL_WAVES
# Each wave owns one 16-key QK tile, so a block covers `PARTIAL_WAVES` of them.
BLOCK_I = PARTIAL_WAVES * MFMA_N

# Model shape (GLM-5.2 absorbed MLA).
DV = 512
DT = 64
DIM = DV + DT
FP8_MAX = 448.0

# Derived layout. `LDS_BANK_PAD` keeps the transposed V reads off a single
# bank; it is a byte pad, unrelated to the MFMA tile that happens to match it.
LDS_BANK_PAD = 16
PITCH = DV + LDS_BANK_PAD
LANE_GROUPS = WAVE_SIZE // H
DV_CHUNKS = DV // 128
DV_TILES_PER_WAVE = (DV // MFMA_N) // PARTIAL_WAVES
Q_LANE_BYTES = DIM // LANE_GROUPS
assert WAVE_SIZE % H == 0, "the lane map splits a wave into whole head groups"
assert DV % (MFMA_N * PARTIAL_WAVES) == 0 and DV % 128 == 0
assert DIM % LANE_GROUPS == 0 and DT == DIM - DV


def _exp2(value):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(value).ir_value()))


# The key is (ng, inner_iter, split_major), not ng alone: 81 combinations are
# reachable over seq 1..96 and ng 1..33 at 256 CUs, 79 at 304.
@functools.lru_cache(maxsize=128)
def compile_sparse_mla_partial(
    ng: int,
    inner_iter: int = 1,
    waves_per_eu: int = 1,
    split_major: bool = False,
):
    """Compile the 64-key BF16-partial, log2-LSE producer."""
    if not 1 <= ng <= 33:
        raise ValueError(f"sparse MLA decode needs 1..33 splits, got {ng}")
    if inner_iter < 1 or inner_iter & (inner_iter - 1) or ng % inner_iter != 0:
        raise ValueError(
            f"inner_iter={inner_iter} must be a power-of-two divisor of ng={ng}"
        )
    n_groups = ng // inner_iter

    @fx.struct
    class PartialStorage:
        vlds: fx.Array[fx.Uint8, BLOCK_I * PITCH, 16]
        rmax: fx.Array[fx.Float32, PARTIAL_WAVES * H, 16]
        rsum: fx.Array[fx.Float32, PARTIAL_WAVES * H, 16]
        plds: fx.Array[fx.Uint8, BLOCK_I * H, 16]
        ilds: fx.Array[fx.Int32, BLOCK_I, 16]
        qlds: fx.Array[fx.Uint8, WAVE_SIZE * Q_LANE_BYTES, 16]

    attrs = {"rocdl.waves_per_eu": int(waves_per_eu)}

    @flyc.kernel(
        name=(
            f"flydsl_sparse_mla_partial_ng{ng}_ii{inner_iter}_xor_partner"
            + ("_split_major" if split_major else "")
        ),
        known_block_size=[PARTIAL_THREADS, 1, 1],
    )
    def kernel(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        index_ptr: fx.Pointer,
        partial_ptr: fx.Pointer,
        lse_ptr: fx.Pointer,
        scale_log2e: fx.Float32,
        seq: fx.Int32,
        num_kv_rows: fx.Int32,
        q_heads: fx.Int32,
    ):
        v16u8_t = fx.Vector.make_type(16, fx.Uint8)
        v2i32_t = fx.Vector.make_type(2, fx.Int32)
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(WAVE_SIZE)
        lane = tid % fx.Int32(WAVE_SIZE)
        group = lane // fx.Int32(H)
        head = lane % fx.Int32(H)
        owner = fx.Int32(fx.block_idx.x)
        if fx.const_expr(split_major):
            split = owner // seq
            tok = owner % seq
        else:
            tok = owner // fx.Int32(n_groups)
            split = owner % fx.Int32(n_groups)
        lds = fx.SharedAllocator().allocate(PartialStorage).peek()
        # Same two instructions the prefill kernel drives through atoms: the
        # gfx950 full-rate scaled FP8 MFMA for QK (unity scales, ABID=0) and
        # the plain 16x16x32 FP8 MFMA for PV.
        qk_mma = fx.make_mma_atom(
            fx.rocdl.cdna4.MFMA_Scale(16, 16, 128, fx.Float8E4M3FN)
        )
        pv_mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FN))

        def load16(ptr, offset):
            return fx.ptr_load(ptr + fx.Int64(offset), result_type=v16u8_t).bitcast(
                fx.Int32
            )

        def join8(lo, hi):
            return fx.Vector.from_elements(
                [lo[i] for i in fx.range_constexpr(4)]
                + [hi[i] for i in fx.range_constexpr(4)],
                fx.Int32,
            )

        # Wave zero publishes Q once; every wave reuses its lane-major LDS view.
        # A model with fewer heads than the 16-row MFMA tile is read in place
        # rather than staged into a padded copy: `q_heads` is the caller's row
        # stride and rows past it re-read head 0, which is in bounds and finite.
        # The partials stay 16 wide because the shared reducer only takes 16.
        safe_head = (head < q_heads).select(head, fx.Int32(0))
        q_base = (fx.Int64(tok) * fx.Int64(q_heads) + fx.Int64(safe_head)) * DIM
        qlane = lds.qlds.ptr + lane * fx.Int32(Q_LANE_BYTES)
        if wave == fx.Int32(0):
            for cc in fx.range_constexpr(4):
                lo = load16(q_ptr, q_base + cc * 128 + fx.Int64(group) * 16)
                hi = load16(q_ptr, q_base + cc * 128 + 64 + fx.Int64(group) * 16)
                fx.ptr_store(lo.bitcast(fx.Uint8), qlane + fx.Int32(cc * 32))
                fx.ptr_store(hi.bitcast(fx.Uint8), qlane + fx.Int32(cc * 32 + 16))
            tail = load16(q_ptr, q_base + DV + fx.Int64(group) * 16)
            fx.ptr_store(tail.bitcast(fx.Uint8), qlane + fx.Int32(128))
        fx.gpu.barrier()

        bq = [None] * 5
        for cc in fx.range_constexpr(4):
            lo = fx.ptr_load(qlane + fx.Int32(cc * 32), result_type=v16u8_t).bitcast(
                fx.Int32
            )
            hi = fx.ptr_load(
                qlane + fx.Int32(cc * 32 + 16), result_type=v16u8_t
            ).bitcast(fx.Int32)
            bq[cc] = fx.make_rmem_tensor(8, fx.Int32)
            bq[cc].store(join8(lo, hi))
        tail = fx.ptr_load(qlane + fx.Int32(128), result_type=v16u8_t).bitcast(fx.Int32)
        bq[4] = fx.make_rmem_tensor(8, fx.Int32)
        bq[4].store(
            fx.Vector.from_elements(
                [tail[i] for i in fx.range_constexpr(4)] + [fx.Int32(0)] * 4,
                fx.Int32,
            )
        )

        # QK-to-PV lane permutation for one 64-key tile.
        slot = (
            fx.Int32(32) * (wave // fx.Int32(2))
            + fx.Int32(8) * (head // fx.Int32(4))
            + fx.Int32(4) * (wave % fx.Int32(2))
            + head % fx.Int32(4)
        )
        out_record = (fx.Int64(tok) * n_groups + fx.Int64(split)) * H + fx.Int64(head)
        running_max = fx.Float32(float("-inf"))
        running_denom = fx.Float32(0.0)
        running_acc = [
            fx.Vector.filled(4, 0.0, fx.Float32)
            for _ in fx.range_constexpr(DV_TILES_PER_WAVE)
        ]

        for k_i in fx.range_constexpr(inner_iter):
            # For ng=32 and inner_iter=2 this merges (0,16), (1,17), ... rather
            # than adjacent rows, removing one real partial row per pair. The
            # pairing is otherwise free: the reducer's max and sum trees are an
            # unconditional six-level butterfly over all 64 lanes, masked only
            # on their inputs, so which tiles a producer pre-paired changes
            # rounding order and nothing else. What adjacent pairing would buy
            # is contiguous index entries per CTA instead of 1024 apart.
            tile = split + fx.Int32(k_i * n_groups)
            index_offset = (
                fx.Int64(tok) * (ng * BLOCK_I)
                + fx.Int64(tile * fx.Int32(BLOCK_I))
                + fx.Int64(slot)
            )
            raw_row = fx.Int32(fx.ptr_load(index_ptr + index_offset))
            # KV is a bare pointer, so nothing downstream bounds the row: an
            # out-of-range index reads unmapped memory and faults the queue.
            # Fold the upper bound into the existing sentinel -- the score
            # mask below already drops anything negative.
            in_range = (raw_row >= fx.Int32(0)) & (raw_row < num_kv_rows)
            row = in_range.select(raw_row, fx.Int32(-1))
            lds.ilds[slot] = row
            safe_row = in_range.select(raw_row, fx.Int32(0))
            kv_base = fx.Int64(safe_row) * DIM

            alo = [None] * 5
            ahi = [None] * 4
            for cc in fx.range_constexpr(4):
                alo[cc] = load16(kv_ptr, kv_base + cc * 128 + fx.Int64(group) * 16)
                ahi[cc] = load16(kv_ptr, kv_base + cc * 128 + 64 + fx.Int64(group) * 16)
            alo[4] = load16(kv_ptr, kv_base + DV + fx.Int64(group) * 16)

            score_fragment = fx.make_rmem_tensor(4, fx.Float32)
            score_fragment.store(fx.Vector.filled(4, 0.0, fx.Float32))
            for cc in fx.range_constexpr(4):
                kv_fragment = fx.make_rmem_tensor(8, fx.Int32)
                kv_fragment.store(join8(alo[cc], ahi[cc]))
                fx.gemm(qk_mma, score_fragment, kv_fragment, bq[cc], score_fragment)
            kv_fragment = fx.make_rmem_tensor(8, fx.Int32)
            kv_fragment.store(
                fx.Vector.from_elements(
                    [alo[4][i] for i in fx.range_constexpr(4)] + [fx.Int32(0)] * 4,
                    fx.Int32,
                )
            )
            fx.gemm(qk_mma, score_fragment, kv_fragment, bq[4], score_fragment)
            score = fx.Vector(score_fragment.load().ir_value())

            for cc in fx.range_constexpr(4):
                vbase = (
                    lds.vlds.ptr
                    + slot * fx.Int32(PITCH)
                    + fx.Int32(cc * 128)
                    + group * fx.Int32(16)
                )
                fx.ptr_store(alo[cc].bitcast(fx.Uint8), vbase)
                fx.ptr_store(ahi[cc].bitcast(fx.Uint8), vbase + fx.Int32(64))

            ids = fx.ptr_load(
                lds.ilds.ptr
                + fx.Int32(32) * (wave // fx.Int32(2))
                + fx.Int32(8) * group
                + fx.Int32(4) * (wave % fx.Int32(2)),
                result_type=fx.Vector.make_type(4, fx.Int32),
            )
            qk = [None] * 4
            for r in fx.range_constexpr(4):
                qk[r] = (ids[r] >= fx.Int32(0)).select(
                    fx.Float32(score[r]) * scale_log2e,
                    fx.Float32(float("-inf")),
                )
            local_max = fx.Float32(float("-inf"))
            for r in fx.range_constexpr(4):
                local_max = local_max.maximumf(qk[r])
            local_max = local_max.maximumf(
                local_max.shuffle_xor(fx.Int32(H), fx.Int32(WAVE_SIZE))
            )
            local_max = local_max.maximumf(
                local_max.shuffle_xor(fx.Int32(2 * H), fx.Int32(WAVE_SIZE))
            )
            if lane < fx.Int32(H):
                lds.rmax[wave * fx.Int32(H) + head] = local_max
            fx.gpu.barrier()

            tile_max = fx.Float32(float("-inf"))
            for ww in fx.range_constexpr(PARTIAL_WAVES):
                tile_max = tile_max.maximumf(fx.Float32(lds.rmax[ww * H + head]))
            max_safe = (tile_max == fx.Float32(float("-inf"))).select(
                fx.Float32(0.0), tile_max
            )
            probs = [None] * 4
            prob_sum = fx.Float32(0.0)
            for r in fx.range_constexpr(4):
                probs[r] = _exp2(qk[r] - max_safe)
                prob_sum = prob_sum + probs[r]
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                probs[0] * fx.Float32(FP8_MAX),
                probs[1] * fx.Float32(FP8_MAX),
                fx.Int32(0),
                False,
            )
            packed = fx.rocdl.cvt_pk_fp8_f32(
                T.i32,
                probs[2] * fx.Float32(FP8_MAX),
                probs[3] * fx.Float32(FP8_MAX),
                packed,
                True,
            )
            fx.ptr_store(
                fx.Vector.from_elements([packed], fx.Int32).bitcast(fx.Uint8),
                lds.plds.ptr + lane * fx.Int32(H) + wave * fx.Int32(4),
            )
            prob_sum = prob_sum + prob_sum.shuffle_xor(fx.Int32(H), fx.Int32(WAVE_SIZE))
            prob_sum = prob_sum + prob_sum.shuffle_xor(
                fx.Int32(2 * H), fx.Int32(WAVE_SIZE)
            )
            if lane < fx.Int32(H):
                lds.rsum[wave * fx.Int32(H) + head] = prob_sum
            fx.gpu.barrier()

            tile_denom = fx.Float32(0.0)
            for ww in fx.range_constexpr(PARTIAL_WAVES):
                tile_denom = tile_denom + fx.Float32(lds.rsum[ww * H + head])
            if fx.const_expr(inner_iter == 1):
                output_scale = (tile_denom == fx.Float32(0.0)).select(
                    fx.Float32(0.0),
                    fx.Float32(
                        fx.rocdl.rcp(
                            T.f32,
                            (tile_denom * fx.Float32(FP8_MAX)).ir_value(),
                        )
                    ),
                )
            else:
                next_max = running_max.maximumf(tile_max)
                alpha = (running_denom == fx.Float32(0.0)).select(
                    fx.Float32(0.0), _exp2(running_max - next_max)
                )
                beta = (tile_denom == fx.Float32(0.0)).select(
                    fx.Float32(0.0), _exp2(tile_max - next_max)
                )
                next_denom = running_denom * alpha + tile_denom * beta
                output_scale = beta * fx.Float32(1.0 / FP8_MAX)
                if fx.const_expr(k_i + 1 == inner_iter):
                    final_inv_denom = (next_denom == fx.Float32(0.0)).select(
                        fx.Float32(0.0),
                        fx.Float32(fx.rocdl.rcp(T.f32, next_denom.ir_value())),
                    )

            p4 = fx.ptr_load(
                lds.plds.ptr + lane * fx.Int32(H),
                result_type=fx.Vector.make_type(16, fx.Uint8),
            ).bitcast(fx.Int32)
            pvec = fx.Vector.from_elements(
                [p4[i] for i in fx.range_constexpr(4)] + [fx.Int32(0)] * 4,
                fx.Int32,
            )
            trbase = (fx.Int32(8) * group + head // fx.Int32(2)) * PITCH + fx.Int32(
                8
            ) * (head % fx.Int32(2))
            for j in fx.range_constexpr(DV_TILES_PER_WAVE):
                dv_base = (wave * fx.Int32(DV_TILES_PER_WAVE) + fx.Int32(j)) * MFMA_N
                acc_fragment = fx.make_rmem_tensor(4, fx.Float32)
                acc_fragment.store(fx.Vector.filled(4, 0.0, fx.Float32))
                for half in fx.range_constexpr(2):
                    vptr = lds.vlds.ptr + trbase + dv_base + fx.Int32(half * 32 * PITCH)
                    value_fragment = fx.make_rmem_tensor(2, fx.Int32)
                    value_fragment.store(
                        fx.Vector(
                            fx.rocdl.ds_read_tr8_b64(
                                v2i32_t, fx.to_llvm_ptr(vptr)
                            ).result
                        )
                    )
                    probability_fragment = fx.make_rmem_tensor(2, fx.Int32)
                    probability_fragment.store(
                        fx.Vector.from_elements(
                            [pvec[2 * half], pvec[2 * half + 1]], fx.Int32
                        )
                    )
                    fx.gemm(
                        pv_mma,
                        acc_fragment,
                        value_fragment,
                        probability_fragment,
                        acc_fragment,
                    )
                acc = fx.Vector(acc_fragment.load().ir_value())
                if fx.const_expr(inner_iter == 1):
                    out_col = dv_base + fx.Int32(4) * group
                    fx.ptr_store(
                        (fx.Vector(acc) * output_scale).to(fx.BFloat16),
                        partial_ptr + out_record * DV + fx.Int64(out_col),
                    )
                else:
                    next_acc = (
                        fx.Vector(running_acc[j]) * alpha
                        + fx.Vector(acc) * output_scale
                    )
                    if fx.const_expr(k_i + 1 == inner_iter):
                        out_col = dv_base + fx.Int32(4) * group
                        fx.ptr_store(
                            (fx.Vector(next_acc) * final_inv_denom).to(fx.BFloat16),
                            partial_ptr + out_record * DV + fx.Int64(out_col),
                        )
                    else:
                        running_acc[j] = next_acc

            if fx.const_expr(inner_iter > 1):
                running_max = next_max
                running_denom = next_denom
                if fx.const_expr(k_i + 1 < inner_iter):
                    fx.gpu.barrier()

        if (wave == fx.Int32(0)) & (lane < fx.Int32(H)):
            if fx.const_expr(inner_iter == 1):
                lse = (tile_denom == fx.Float32(0.0)).select(
                    fx.Float32(-(2**30)), fly_math.log2(tile_denom) + tile_max
                )
            else:
                lse = (running_denom == fx.Float32(0.0)).select(
                    fx.Float32(-(2**30)),
                    fly_math.log2(running_denom) + running_max,
                )
            fx.ptr_store(lse, lse_ptr + out_record)

    @flyc.jit
    def launch(
        q_ptr: fx.Pointer,
        kv_ptr: fx.Pointer,
        index_ptr: fx.Pointer,
        partial_ptr: fx.Pointer,
        lse_ptr: fx.Pointer,
        scale_log2e: fx.Float32,
        seq: fx.Int32,
        num_kv_rows: fx.Int32,
        q_heads: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(
            q_ptr,
            kv_ptr,
            index_ptr,
            partial_ptr,
            lse_ptr,
            scale_log2e,
            seq,
            num_kv_rows,
            q_heads,
        ).launch(
            grid=(seq * fx.Int32(n_groups), 1, 1),
            block=(PARTIAL_THREADS, 1, 1),
            stream=stream,
            value_attrs=attrs,
        )

    return launch
