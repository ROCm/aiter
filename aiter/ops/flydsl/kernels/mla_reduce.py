# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the MLA decode reduce/combine epilogue.

Faithful port of the HIP kernel ``kn_mla_reduce_v1`` / ``kn_mla_reduce_v1_ps``
(``csrc/kernels/mla/reduce.cu``). Stage-2 of split-KV MLA decode: merges per-split
partial outputs ``O_i`` (fp32) weighted by ``exp(LSE_i - LSE_max)`` (online softmax)
into the final output (bf16/fp16), and optionally the merged LSE.

Layout / contract (matches the HIP kernel):
  partial_output : fp32 [max_partial_row, H, Dv]  contiguous
  partial_lse    : fp32 [max_partial_row, H]       contiguous
  reduce_indptr  : i32  [num_reduce_tile + 1]      CSR over splits
  reduce_partial_map : i32 [reduce_indptr[-1]]     split -> partial row (gather)
  reduce_final_map   : i32 [num_reduce_tile, 2]    {q_start, q_end} (optional)
  final_output   : bf16/fp16 [bs, H, Dv]           runtime strides
  final_lse      : fp32 [bs, H]                    (optional)

Each work item = (head, q-pos-group, reduce-tile). A block of 128 threads (2 waves)
owns one (seq, head) output row; thread t owns ``VEC = Dv // 128`` contiguous floats.

Two launch modes (mirrors the HIP kernel):
  * grid-launch (default): 3-D grid (H, NTG, num_reduce_tile), one block per work item.
  * persistent (``persistent=True``): 1-D grid of ``num_cu * OCC * 2`` blocks that
    grid-stride over the flat work index, matching ``kn_mla_reduce_v1_ps``. This
    collapses the dispatch when ``num_reduce_tile`` is large but few tiles are
    active (sparse serving grids), eliminating launch/dispatch latency.
"""

import enum
import functools
import math
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, math as mlir_math, gpu as mlir_gpu
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch

from .tensor_shim import GTensor, STensor, _to_raw

_LOG2E = math.log2(math.e)
fm_fast = fx.arith.FastMathFlags.fast

# Matches MlaReduceKernelV1Traits (reduce.cu:13)
NUM_THREADS = 128
WARP = 64
NUM_WAVES = NUM_THREADS // WARP
OCC = 8
MASSIVE_THR = 4  # kMassiveThreshold


def _out_t(out_dtype: str):
    """Resolve the output MLIR element type. Call only inside an MLIR context."""
    if out_dtype in ("bf16", "bfloat16"):
        return T.bf16
    if out_dtype in ("fp16", "f16", "float16", "half"):
        return T.f16
    raise ValueError(f"Unsupported out_dtype: {out_dtype}")


def _exp(x, use_exp2=True):
    """exp(x) via the hardware v_exp_f32 (exp2(x*log2e))."""
    if fx.const_expr(use_exp2):
        return fx.rocdl.exp2(T.f32, x * _LOG2E)
    return mlir_math.exp(x, fastmath=fm_fast)


def _log(x):
    """natural log; mlir math dialect lowers to the device log."""
    return mlir_math.log(x, fastmath=fm_fast)


# Massive-path sub-tiers (mirror MlaReduceProblemSize, reduce.cu:121).
class Tier(enum.Enum):
    SIMPLE = "simple"  # register online-softmax, num_splits in {2,3}
    M64 = "m64"  # <=64 splits, 1 LSE per lane in warp0
    M256 = "m256"  # <=256 splits, up to 4 LSE per lane in warp0
    MLDS = "mlds"  # >256 splits, 4 in regs + overflow in LDS


# LSE registers per warp0 lane (None = LDS-backed overflow).
_TIER_NLSE = {Tier.SIMPLE: 0, Tier.M64: 1, Tier.M256: 4, Tier.MLDS: None}
LDS_MAX_SPLITS = 304  # >= MI300X CU count; compile-time LDS cap

# Production default (opt4 sweep: 4 ~= 6 < 8 on H=16 Dv=512 tiles=8 splits=32).
_DEFAULT_WAVES_PER_EU = 4


def waves_per_eu_from_env(default: int = _DEFAULT_WAVES_PER_EU) -> int:
    """Read ``AITER_MLA_REDUCE_WAVES_PER_EU`` (sweep/tuning knob)."""
    raw = os.environ.get("AITER_MLA_REDUCE_WAVES_PER_EU")
    if raw is None:
        return default
    return int(raw)


def select_tier(num_splits: int) -> Tier:
    if num_splits < MASSIVE_THR:
        return Tier.SIMPLE
    if num_splits <= 64:
        return Tier.M64
    if num_splits <= 256:
        return Tier.M256
    return Tier.MLDS


def should_use_persistent_launch(
    *,
    H: int,
    max_seqlen_q: int,
    num_reduce_tile: int,
    num_cu: int,
) -> bool:
    """Match HIP dispatch_mla_reduce_v1 grid-launch vs persistent heuristic.

    HIP uses the persistent kernel when the flattened work count exceeds the
    persistent grid size (``num_cu * OCC * 2``); otherwise a plain grid launch.
    """
    tot_work = H * max_seqlen_q * num_reduce_tile
    ps_grid_size = num_cu * OCC * 2
    return tot_work > ps_grid_size


@functools.lru_cache(maxsize=256)
def compile_mla_reduce(
    *,
    H: int,
    Dv: int,
    out_dtype: str = "bf16",
    tier: Tier = Tier.SIMPLE,
    persistent: bool = False,
    output_lse: bool = False,
    use_reduce_final_map: bool = True,
    prefetch_depth: int = 2,
    waves_per_eu: int = _DEFAULT_WAVES_PER_EU,
    use_exp2: bool = True,
    use_packed_cvt: bool = False,
    use_packed_f32_fma: bool = False,
    disable_guards: bool = False,
):
    """Compile an MLA reduce kernel for fixed (H, Dv, out_dtype, tier).

    tier selects the per-work-item algorithm (compile-time); the host picks it
    from the observed num_splits via ``select_tier``. ``persistent`` selects the
    grid-stride launch (kn_mla_reduce_v1_ps); otherwise a 3-D grid launch.

    disable_guards: test-only compile-time knob that skips gather/store bounds
    guards so the suite can run a pre-fix kernel in-process. The production
    wrapper (mla_reduce_kernels.py) never threads this (defaults False).
    """
    assert Dv % NUM_THREADS == 0, "Dv must be divisible by 128"
    assert tier in _TIER_NLSE, f"bad tier {tier}"
    VEC = Dv // NUM_THREADS
    is_massive = tier != Tier.SIMPLE
    if tier == Tier.MLDS:
        NLSE = (LDS_MAX_SPLITS + WARP - 1) // WARP
    else:
        NLSE = _TIER_NLSE[tier]

    # ---- LDS layout (massive tiers only) ----
    # [ lse_scale : LDS_MAX_SPLITS f32 ] ++ [ local_lse overflow : (NLSE-4 lanes) f32 ]
    GPU_ARCH = get_rocm_arch()
    allocator = None
    lse_scale_off = 0
    local_lse_off = 0
    if is_massive:
        allocator = SmemAllocator(
            None,
            arch=GPU_ARCH,
            global_sym_name=f"mla_reduce_smem_{tier.value}_{Dv}_{H}",
        )
        lse_scale_off = allocator._align(allocator.ptr, 16)
        allocator.ptr = lse_scale_off + LDS_MAX_SPLITS * 4
        if tier == Tier.MLDS:
            local_lse_off = allocator._align(allocator.ptr, 16)
            overflow_slots = max(0, LDS_MAX_SPLITS - 256)
            allocator.ptr = local_lse_off + overflow_slots * 4

    @flyc.kernel(known_block_size=[NUM_THREADS, 1, 1])
    def mla_reduce_kernel(
        partial_output: fx.Pointer,  # fp32 [row, H, Dv]
        partial_lse: fx.Pointer,  # fp32 [row, H]
        reduce_indptr: fx.Pointer,  # i32  [tiles+1]
        reduce_partial_map: fx.Pointer,  # i32  [nsplits_total]
        reduce_final_map: fx.Pointer,  # i32  [tiles, 2]
        final_output: fx.Pointer,  # out  [bs, H, Dv]
        final_lse: fx.Pointer,  # fp32 [bs, H]
        stride_s_o: fx.Int32,  # final_output row stride (elements)
        stride_h_o: fx.Int32,  # final_output head stride (elements)
        max_splits: fx.Int32,  # = num_cu (LSE distribution stride)
        num_reduce_tile: fx.Int32,  # CSR sentinel at indptr[num_reduce_tile]
        num_partial_rows: fx.Int32,  # partial_output row count (gather bounds)
        num_final_rows: fx.Int32,  # final_output row count (store q-range)
        num_thread_group: fx.Int32,  # NTG (persistent unflatten / seq stride)
    ):
        out_t = _out_t(out_dtype)
        c_VEC = fx.Index(VEC)
        out_vt = T.vec(VEC, out_t)

        def load_o_elems(g, row_idx, head_idx, col_idx):
            """Load VEC fp32 elements as a python list of scalars."""
            if fx.const_expr(VEC == 1):
                return [g[row_idx, head_idx, col_idx]]
            v = g.vec_load((row_idx, head_idx, col_idx), VEC)
            return [
                fx.vector.extract(v, static_position=[i], dynamic_position=[])
                for i in fx.range_constexpr(VEC)
            ]

        def store_o_elems(g, off_idx, elems_f32):
            """Cast VEC fp32 scalars to out_t and emit one vector buffer store.

            Truncate each element to out_t first, then pack into the out_t vector
            so the store lowers to buffer_store_dwordx2/4 instead of a per-element
            (scalarized) buffer_store_short.
            """
            if fx.const_expr(VEC == 1):
                g[off_idx] = elems_f32[0].truncf(out_t)
                return
            out_vec = fx.vector.from_elements(
                out_vt,
                [
                    _to_raw(elems_f32[i].truncf(out_t))
                    for i in fx.range_constexpr(VEC)
                ],
            )
            g.vec_store((off_idx,), out_vec, VEC)

        tid = fx.thread_idx.x
        col = tid * c_VEC
        lane = tid % fx.Int32(WARP)
        wave = tid // fx.Int32(WARP)
        width_i32 = _to_raw(fx.arith.constant(WARP, type=T.i32))

        g_po = GTensor(partial_output, dtype=T.f32, shape=(-1, H, Dv))
        g_pl = GTensor(partial_lse, dtype=T.f32, shape=(-1, H))
        g_indptr = GTensor(reduce_indptr, dtype=T.i32, shape=(-1,))
        g_pmap = GTensor(reduce_partial_map, dtype=T.i32, shape=(-1,))
        g_fmap = GTensor(reduce_final_map, dtype=T.i32, shape=(-1, 2))
        g_fo = GTensor(final_output, dtype=out_t, shape=(-1,))
        g_flse = GTensor(final_lse, dtype=T.f32, shape=(-1,))

        c_H = fx.Int32(H)
        last = fx.Int32(
            g_indptr[fx.arith.index_cast(T.index, _to_raw(num_reduce_tile))]
        )

        if fx.const_expr(is_massive):
            base = allocator.get_base()
            lds_scale = STensor(
                SmemPtr(base, lse_scale_off, T.f32, shape=(LDS_MAX_SPLITS,)),
                dtype=T.f32,
                shape=(LDS_MAX_SPLITS,),
            )

        def process_work_item(head, block_idx, tile, ntg):
            """Reduce one (head, q-pos-group, tile) work item into final_output.

            Self-guards: tiles at the CSR sentinel, n_splits<=1, or out-of-range
            q-start run zero seq iterations (ub_seq clamp) and never touch the
            gather/store paths, so callers may invoke this for any work index.
            """
            t0 = fx.Int32(g_indptr[tile])
            t1 = fx.Int32(g_indptr[tile + fx.Index(1)])
            n_splits = t1 - t0

            # HIP kn_mla_reduce_v1_ps: skip tiles at the CSR sentinel
            # (reduce.cu:692) or with n_splits<=1. Collapse the seq-loop upper
            # bound instead of an scf.IfOp body-wrap so inactive tiles never
            # enter gather/store paths even if reduce_final_map holds garbage
            # q-ranges (serving tail).
            has_work = fx.arith.andi(
                fx.arith.cmpi(fx.arith.CmpIPredicate.sgt, n_splits, fx.Int32(1)),
                fx.arith.cmpi(fx.arith.CmpIPredicate.ne, t0, last),
            )

            if fx.const_expr(use_reduce_final_map):
                q_start = fx.Int32(g_fmap[tile, fx.Index(0)])
                q_end = fx.Int32(g_fmap[tile, fx.Index(1)])
            else:
                row0_idx0 = fx.Int32(g_pmap[fx.Index(t0)])
                row1_idx0 = fx.Int32(g_pmap[fx.Index(t0) + fx.Index(1)])
                qo_len = row1_idx0 - row0_idx0
                q_start = fx.Int32(tile) * qo_len
                q_end = (fx.Int32(tile) + fx.Int32(1)) * qo_len

            if fx.const_expr(not disable_guards):
                q_valid = fx.arith.andi(
                    fx.arith.cmpi(
                        fx.arith.CmpIPredicate.sge, q_start, fx.Int32(0)
                    ),
                    fx.arith.cmpi(
                        fx.arith.CmpIPredicate.slt, q_start, num_final_rows
                    ),
                )
                has_work = fx.arith.andi(has_work, q_valid)
                q_end_oob = fx.arith.cmpi(
                    fx.arith.CmpIPredicate.sgt, q_end, num_final_rows
                )
                q_end = q_end_oob.select(num_final_rows, q_end)

            seq0 = q_start + fx.Int32(block_idx)
            ub_seq = has_work.select(q_end, seq0)

            def gather_row(split_i32, local_seq):
                pmap = fx.Int32(g_pmap[fx.Index(t0 + split_i32)])
                row_i32 = pmap + local_seq
                if fx.const_expr(not disable_guards):
                    in_bounds = fx.arith.andi(
                        fx.arith.cmpi(
                            fx.arith.CmpIPredicate.sge, row_i32, fx.Int32(0)
                        ),
                        fx.arith.cmpi(
                            fx.arith.CmpIPredicate.slt,
                            row_i32,
                            num_partial_rows,
                        ),
                    )
                    safe_row_i32 = in_bounds.select(row_i32, fx.Int32(0))
                else:
                    in_bounds = fx.arith.cmpi(
                        fx.arith.CmpIPredicate.eq, fx.Int32(0), fx.Int32(0)
                    )
                    safe_row_i32 = row_i32
                row_idx = fx.arith.index_cast(
                    ir.IndexType.get(), _to_raw(safe_row_i32)
                )
                return row_idx, in_bounds

            def load_split_o(split_i32, local_seq):
                row_idx, in_bounds = gather_row(split_i32, local_seq)
                loaded = load_o_elems(g_po, row_idx, fx.Index(head), col)
                zero = fx.arith.constant(0.0, type=T.f32)
                return [in_bounds.select(v, zero) for v in loaded]

            def load_split_lse(split_i32, local_seq):
                row_idx, in_bounds = gather_row(split_i32, local_seq)
                lse = g_pl[row_idx, fx.Index(head)]
                neg_inf = fx.arith.constant(float("-inf"), type=T.f32)
                return in_bounds.select(lse, neg_inf)

            def store_result(seq, out_elems):
                store_off = (
                    fx.Int32(seq) * stride_s_o
                    + fx.Int32(head) * stride_h_o
                    + fx.Int32(tid) * fx.Int32(VEC)
                )
                store_o_elems(g_fo, fx.Index(store_off), out_elems)

            def store_lse(seq, max_lse, sum_e):
                if fx.const_expr(output_lse):
                    bad = fx.arith.ori(
                        fx.arith.cmpf(
                            fx.arith.CmpFPredicate.OEQ,
                            sum_e,
                            fx.arith.constant(0.0, type=T.f32),
                        ),
                        fx.arith.cmpf(fx.arith.CmpFPredicate.UNO, sum_e, sum_e),
                    )
                    lse_val = _log(sum_e) + max_lse
                    inf = fx.arith.constant(float("inf"), type=T.f32)
                    final_lse_val = bad.select(inf, lse_val)
                    is_lane0 = fx.arith.cmpi(
                        fx.arith.CmpIPredicate.eq, tid, fx.Int32(0)
                    )
                    if_lane0 = scf.IfOp(is_lane0, results_=[], has_else=False)
                    with ir.InsertionPoint(if_lane0.then_block):
                        lse_off = fx.Int32(seq) * fx.Int32(H) + fx.Int32(head)
                        g_flse[fx.Index(lse_off)] = final_lse_val
                        scf.YieldOp([])

            lb = fx.arith.index_cast(ir.IndexType.get(), _to_raw(seq0))
            ub = fx.arith.index_cast(ir.IndexType.get(), _to_raw(ub_seq))
            st = fx.arith.index_cast(ir.IndexType.get(), _to_raw(ntg))
            for_seq = scf.ForOp(lb, ub, st)
            with ir.InsertionPoint(for_seq.body):
                seq = fx.Int32(for_seq.induction_variable)
                local_seq = seq - q_start

                if fx.const_expr(not is_massive):
                    o0 = load_split_o(fx.Int32(0), local_seq)
                    lse0 = load_split_lse(fx.Int32(0), local_seq)

                    init = [_to_raw(o0[i]) for i in fx.range_constexpr(VEC)]
                    init += [
                        _to_raw(lse0),
                        _to_raw(fx.arith.constant(1.0, type=T.f32)),
                    ]

                    results = init
                    for s, state in range(
                        fx.Index(1), fx.Index(n_splits), fx.Index(1), init=init
                    ):
                        regs = [state[i] for i in fx.range_constexpr(VEC)]
                        max_lse = state[VEC]
                        sum_e = state[VEC + 1]
                        os = load_split_o(fx.Int32(s), local_seq)
                        lse = load_split_lse(fx.Int32(s), local_seq)
                        new_max = fx.arith.maximumf(max_lse, lse)
                        old = _exp(max_lse - new_max, use_exp2)
                        new = _exp(lse - new_max, use_exp2)
                        new_regs = [
                            _to_raw(regs[i] * old + os[i] * new)
                            for i in fx.range_constexpr(VEC)
                        ]
                        results = yield new_regs + [
                            _to_raw(new_max),
                            _to_raw(sum_e * old + new),
                        ]

                    regs = [results[i] for i in fx.range_constexpr(VEC)]
                    max_lse = results[VEC]
                    sum_e = results[VEC + 1]
                    inv = fx.rocdl.rcp(T.f32, sum_e)
                    out_elems = [regs[i] * inv for i in fx.range_constexpr(VEC)]
                    store_result(seq, out_elems)
                    store_lse(seq, max_lse, sum_e)
                else:
                    neg_inf = fx.arith.constant(float("-inf"), type=T.f32)
                    is_wave0 = fx.arith.cmpi(
                        fx.arith.CmpIPredicate.eq, wave, fx.Int32(0)
                    )
                    if_w0 = scf.IfOp(is_wave0, results_=[], has_else=False)
                    with ir.InsertionPoint(if_w0.then_block):
                        local_lses = []
                        max_lse = neg_inf
                        for j in fx.range_constexpr(NLSE):
                            split_idx = lane + fx.Int32(j * WARP)
                            in_rng = fx.arith.cmpi(
                                fx.arith.CmpIPredicate.slt, split_idx, n_splits
                            )
                            safe = in_rng.select(split_idx, fx.Int32(0))
                            lse_j = load_split_lse(safe, local_seq)
                            lse_j = in_rng.select(lse_j, neg_inf)
                            local_lses.append(lse_j)
                            max_lse = fx.arith.maximumf(max_lse, lse_j)
                        for off in [32, 16, 8, 4, 2, 1]:
                            peer = mlir_gpu.ShuffleOp(
                                _to_raw(max_lse),
                                _to_raw(fx.arith.constant(off, type=T.i32)),
                                width_i32,
                                mode="xor",
                            ).shuffleResult
                            max_lse = fx.arith.maximumf(max_lse, peer)
                        sum_e = fx.arith.constant(0.0, type=T.f32)
                        for j in fx.range_constexpr(NLSE):
                            sum_e = sum_e + _exp(
                                local_lses[j] - max_lse, use_exp2
                            )
                        for off in [32, 16, 8, 4, 2, 1]:
                            peer = mlir_gpu.ShuffleOp(
                                _to_raw(sum_e),
                                _to_raw(fx.arith.constant(off, type=T.i32)),
                                width_i32,
                                mode="xor",
                            ).shuffleResult
                            sum_e = sum_e + peer
                        bad = fx.arith.ori(
                            fx.arith.cmpf(
                                fx.arith.CmpFPredicate.OEQ,
                                sum_e,
                                fx.arith.constant(0.0, type=T.f32),
                            ),
                            fx.arith.cmpf(
                                fx.arith.CmpFPredicate.UNO, sum_e, sum_e
                            ),
                        )
                        inf = fx.arith.constant(float("inf"), type=T.f32)
                        global_lse = bad.select(inf, _log(sum_e) + max_lse)
                        for j in fx.range_constexpr(NLSE):
                            split_idx = lane + fx.Int32(j * WARP)
                            in_rng = fx.arith.cmpi(
                                fx.arith.CmpIPredicate.slt, split_idx, n_splits
                            )
                            if_s = scf.IfOp(in_rng, results_=[], has_else=False)
                            with ir.InsertionPoint(if_s.then_block):
                                sc = _exp(local_lses[j] - global_lse, use_exp2)
                                lds_scale[fx.Index(split_idx)] = sc
                                scf.YieldOp([])
                        if fx.const_expr(output_lse):
                            is_l0 = fx.arith.cmpi(
                                fx.arith.CmpIPredicate.eq, lane, fx.Int32(0)
                            )
                            if_l0 = scf.IfOp(is_l0, results_=[], has_else=False)
                            with ir.InsertionPoint(if_l0.then_block):
                                lse_off = (
                                    fx.Int32(seq) * fx.Int32(H) + fx.Int32(head)
                                )
                                g_flse[fx.Index(lse_off)] = global_lse
                                scf.YieldOp([])
                        scf.YieldOp([])

                    fx.gpu.barrier()

                    acc0 = tuple(
                        _to_raw(fx.arith.constant(0.0, type=T.f32))
                        for _ in fx.range_constexpr(VEC)
                    )
                    for s, state in range(
                        fx.Index(0), fx.Index(n_splits), fx.Index(1), init=acc0
                    ):
                        regs = tuple(
                            state[i] for i in fx.range_constexpr(VEC)
                        )
                        os = load_split_o(fx.Int32(s), local_seq)
                        sc = lds_scale[fx.Index(s)]
                        new_regs = [
                            _to_raw(regs[i] + os[i] * sc)
                            for i in fx.range_constexpr(VEC)
                        ]
                        accr = yield new_regs
                    if fx.const_expr(VEC == 1):
                        out_elems = [accr]
                    else:
                        out_elems = [accr[i] for i in fx.range_constexpr(VEC)]
                    store_result(seq, out_elems)
                scf.YieldOp([])

        if fx.const_expr(persistent):
            # Grid-stride persistent launch (kn_mla_reduce_v1_ps, reduce.cu:669).
            # 1-D grid; each block grid-strides over the flat work index and
            # terminates once it reaches a tile at the CSR sentinel.
            tot_work = c_H * num_thread_group * num_reduce_tile
            grid_stride = fx.Int32(fx.grid_dim.x)
            init_work = fx.Int32(fx.block_idx.x)

            w = scf.WhileOp([T.i32], [_to_raw(init_work)])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                work_idx = fx.Int32(before.arguments[0])
                in_range = fx.arith.cmpi(
                    fx.arith.CmpIPredicate.slt, work_idx, tot_work
                )
                scf.ConditionOp(in_range, [before.arguments[0]])
            with ir.InsertionPoint(after):
                work_idx = fx.Int32(after.arguments[0])
                head = work_idx % c_H
                temp_idx = work_idx // c_H
                block_idx = temp_idx % num_thread_group
                tile = temp_idx // num_thread_group

                tile_start = fx.Int32(g_indptr[tile])
                is_past_end = fx.arith.cmpi(
                    fx.arith.CmpIPredicate.eq, tile_start, last
                )
                not_past_end = fx.arith.cmpi(
                    fx.arith.CmpIPredicate.ne, tile_start, last
                )
                if_has_work = scf.IfOp(
                    not_past_end, results_=[], has_else=False
                )
                with ir.InsertionPoint(if_has_work.then_block):
                    process_work_item(head, block_idx, tile, num_thread_group)
                    # LDS (lds_scale) is reused per work item; fence before the
                    # next iteration's warp0 overwrites it.
                    fx.gpu.barrier()
                    scf.YieldOp([])

                next_work = is_past_end.select(
                    tot_work, work_idx + grid_stride
                )
                scf.YieldOp([_to_raw(next_work)])
        else:
            head = fx.block_idx.x
            block_idx = fx.block_idx.y  # q-pos group (NTG)
            tile = fx.block_idx.z
            ntg = fx.Int32(fx.grid_dim.y)
            process_work_item(head, block_idx, tile, ntg)
        return

    @flyc.jit
    def launch_mla_reduce(
        partial_output: fx.Pointer,
        partial_lse: fx.Pointer,
        reduce_indptr: fx.Pointer,
        reduce_partial_map: fx.Pointer,
        reduce_final_map: fx.Pointer,
        final_output: fx.Pointer,
        final_lse: fx.Pointer,
        stride_s_o: fx.Int32,
        stride_h_o: fx.Int32,
        max_splits: fx.Int32,
        num_reduce_tile: fx.Int32,
        max_seqlen_q: fx.Int32,
        num_partial_rows: fx.Int32,
        num_final_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        ctx = CompilationContext.get_current()
        if fx.const_expr(is_massive):
            allocator.finalized = False
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()

        # opt4: pin occupancy via rocdl.waves_per_eu on the emitted gpu.func.
        if fx.const_expr(waves_per_eu >= 1):
            _wpe = int(waves_per_eu)
            for op in ctx.gpu_module_body.operations:
                if getattr(op, "OPERATION_NAME", None) == "gpu.func":
                    op.attributes["rocdl.waves_per_eu"] = ir.IntegerAttr.get(
                        T.i32, _wpe
                    )

        idx_tiles = fx.arith.index_cast(T.index, num_reduce_tile)
        idx_H = fx.Index(H)
        idx_ntg = fx.arith.index_cast(T.index, _to_raw(max_seqlen_q))
        if fx.const_expr(persistent):
            ps_grid = fx.arith.muli(
                _to_raw(max_splits), fx.arith.constant(OCC * 2, type=T.i32)
            )
            idx_grid = fx.arith.index_cast(T.index, ps_grid)
            grid = (idx_grid, fx.Index(1), fx.Index(1))
        else:
            grid = (idx_H, idx_ntg, idx_tiles)
        mla_reduce_kernel(
            partial_output,
            partial_lse,
            reduce_indptr,
            reduce_partial_map,
            reduce_final_map,
            final_output,
            final_lse,
            stride_s_o,
            stride_h_o,
            max_splits,
            num_reduce_tile,
            num_partial_rows,
            num_final_rows,
            max_seqlen_q,
        ).launch(
            grid=grid,
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    return launch_mla_reduce
