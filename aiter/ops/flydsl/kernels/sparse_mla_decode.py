# SPDX-License-Identifier: MIT

"""gfx950 sparse-MLA decode producer with 64-key split granularity."""

# FlyDSL kernel annotations must be evaluated eagerly.
import functools
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import arith
from flydsl.expr import math as fly_math
from flydsl.expr.arith import CmpIPredicate
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

H = 16
DV = 512
DT = 64
DIM = DV + DT
BLOCK_I = 64
FP8_MAX = 448.0
PARTIAL_THREADS = 256
PARTIAL_WAVES = 4
PITCH = DV + 16


@contextmanager
def _if_then(if_op):
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            block = if_op.then_block
            if (not block.operations) or not isinstance(
                block.operations[-1], scf.YieldOp
            ):
                scf.YieldOp([])


def _exp2(value):
    return fx.Float32(fx.rocdl.exp2(T.f32, fx.Float32(value).ir_value()))


def _pack_i32x2(lo, hi):
    return fx.Vector.from_elements([lo, hi], fx.Int32).bitcast(fx.Int64)[0]


def _mfma128(a, b, c):
    """gfx950 full-rate FP8 MFMA with unity scales (ABID=0)."""
    return fx.rocdl.mfma_scale_f32_16x16x128_f8f6f4(
        T.f32x4,
        [a, b, c, 0, 0, 0, fx.Int32(0), 0, fx.Int32(0)],
    )


def _mfma32(a, b, c):
    return fx.rocdl.mfma_f32_16x16x32_fp8_fp8(
        T.f32x4, [_pack_i32x2(a[0], a[1]), _pack_i32x2(b[0], b[1]), c, 0, 0, 0]
    )


# ng ranges over 1..33, so 32 entries evict at the top of the domain and a
# re-entry costs about 31 ms with the disk cache warm. Size to the domain.
@functools.lru_cache(maxsize=64)
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
        qlds: fx.Array[fx.Uint8, 64 * 144, 16]

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
    ):
        v4u8_t = fx.Vector.make_type(16, fx.Uint8)
        v2i32_t = fx.Vector.make_type(2, fx.Int32)
        tid = fx.Int32(fx.thread_idx.x)
        wave = tid // fx.Int32(64)
        lane = tid % fx.Int32(64)
        group = lane // fx.Int32(16)
        head = lane % fx.Int32(16)
        owner = fx.Int32(fx.block_idx.x)
        if fx.const_expr(split_major):
            split = owner // seq
            tok = owner % seq
        else:
            tok = owner // fx.Int32(n_groups)
            split = owner % fx.Int32(n_groups)
        lds = fx.SharedAllocator().allocate(PartialStorage).peek()

        def load16(ptr, offset):
            return fx.ptr_load(ptr + fx.Int64(offset), result_type=v4u8_t).bitcast(
                fx.Int32
            )

        def join8(lo, hi):
            return fx.Vector.from_elements(
                [lo[i] for i in fx.range_constexpr(4)]
                + [hi[i] for i in fx.range_constexpr(4)],
                fx.Int32,
            )

        # Keep this as a compile-time region. A runtime guard here makes the
        # FlyDSL rewriter capture LDS handles as branch state.
        if fx.const_expr(True):
            # Wave zero publishes Q once; every wave reuses its lane-major LDS view.
            q_base = (fx.Int64(tok) * H + fx.Int64(head)) * DIM
            qlane = lds.qlds.ptr + lane * fx.Int32(144)
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
                lo = fx.ptr_load(qlane + fx.Int32(cc * 32), result_type=v4u8_t).bitcast(
                    fx.Int32
                )
                hi = fx.ptr_load(
                    qlane + fx.Int32(cc * 32 + 16), result_type=v4u8_t
                ).bitcast(fx.Int32)
                bq[cc] = join8(lo, hi)
            tail = fx.ptr_load(qlane + fx.Int32(128), result_type=v4u8_t).bitcast(
                fx.Int32
            )
            bq[4] = fx.Vector.from_elements(
                [tail[i] for i in fx.range_constexpr(4)] + [fx.Int32(0)] * 4,
                fx.Int32,
            )

            # QK-to-PV lane permutation for one 64-key tile.
            slot = (
                fx.Int32(32) * (wave // fx.Int32(2))
                + fx.Int32(8) * (head // fx.Int32(4))
                + fx.Int32(4) * (wave % fx.Int32(2))
                + head % fx.Int32(4)
            )
            out_record = (fx.Int64(tok) * n_groups + fx.Int64(split)) * H + fx.Int64(
                head
            )
            running_max = fx.Float32(float("-inf"))
            running_denom = fx.Float32(0.0)
            running_acc = [
                fx.Vector.filled(4, 0.0, fx.Float32) for _ in fx.range_constexpr(8)
            ]

            for k_i in fx.range_constexpr(inner_iter):
                # Preserve the direct reducer's XOR tree: for ng=32 and
                # inner_iter=2, merge (0,16), (1,17), ... rather than adjacent
                # rows.  This removes one real partial row per pair while
                # matching the reducer's first active shuffle level.
                tile = split + fx.Int32(k_i * n_groups)
                index_offset = (
                    fx.Int64(tok) * (ng * BLOCK_I)
                    + fx.Int64(tile * fx.Int32(BLOCK_I))
                    + fx.Int64(slot)
                )
                row = fx.Int32(fx.ptr_load(index_ptr + index_offset))
                lds.ilds[slot] = row
                safe_row = (row >= fx.Int32(0)).select(row, fx.Int32(0))
                kv_base = fx.Int64(safe_row) * DIM

                alo = [None] * 5
                ahi = [None] * 4
                for cc in fx.range_constexpr(4):
                    alo[cc] = load16(kv_ptr, kv_base + cc * 128 + fx.Int64(group) * 16)
                    ahi[cc] = load16(
                        kv_ptr, kv_base + cc * 128 + 64 + fx.Int64(group) * 16
                    )
                alo[4] = load16(kv_ptr, kv_base + DV + fx.Int64(group) * 16)

                score = fx.Vector.filled(4, 0.0, fx.Float32)
                for cc in fx.range_constexpr(4):
                    score = _mfma128(join8(alo[cc], ahi[cc]), bq[cc], score)
                score = _mfma128(
                    fx.Vector.from_elements(
                        [alo[4][i] for i in fx.range_constexpr(4)] + [fx.Int32(0)] * 4,
                        fx.Int32,
                    ),
                    bq[4],
                    score,
                )

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
                    local_max.shuffle_xor(fx.Int32(16), fx.Int32(64))
                )
                local_max = local_max.maximumf(
                    local_max.shuffle_xor(fx.Int32(32), fx.Int32(64))
                )
                with _if_then(
                    scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.slt,
                            _raw(lane),
                            arith.constant(16, type=T.i32),
                        )
                    )
                ):
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
                prob_sum = prob_sum + prob_sum.shuffle_xor(fx.Int32(16), fx.Int32(64))
                prob_sum = prob_sum + prob_sum.shuffle_xor(fx.Int32(32), fx.Int32(64))
                with _if_then(
                    scf.IfOp(
                        arith.cmpi(
                            CmpIPredicate.slt,
                            _raw(lane),
                            arith.constant(16, type=T.i32),
                        )
                    )
                ):
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
                for j in fx.range_constexpr(8):
                    dv_base = (wave * fx.Int32(8) + fx.Int32(j)) * 16
                    acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for half in fx.range_constexpr(2):
                        vptr = (
                            lds.vlds.ptr
                            + trbase
                            + dv_base
                            + fx.Int32(half * 32 * PITCH)
                        )
                        vaddr = fx.Int64(fx.ptrtoint(vptr))
                        llvm_vptr = llvm.inttoptr(
                            ir.Type.parse("!llvm.ptr<3>"), _raw(vaddr)
                        )
                        avec = fx.Vector(
                            fx.rocdl.ds_read_tr8_b64(v2i32_t, llvm_vptr).result
                        )
                        bvec = fx.Vector.from_elements(
                            [pvec[2 * half], pvec[2 * half + 1]], fx.Int32
                        )
                        acc = _mfma32(avec, bvec, acc)
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

            with _if_then(
                scf.IfOp(
                    arith.andi(
                        arith.cmpi(
                            CmpIPredicate.eq,
                            _raw(wave),
                            arith.constant(0, type=T.i32),
                        ),
                        arith.cmpi(
                            CmpIPredicate.slt,
                            _raw(lane),
                            arith.constant(16, type=T.i32),
                        ),
                    )
                )
            ):
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
        ).launch(
            grid=(seq * fx.Int32(n_groups), 1, 1),
            block=(PARTIAL_THREADS, 1, 1),
            stream=stream,
            value_attrs=attrs,
        )

    return launch
