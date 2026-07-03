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
from contextlib import contextmanager

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, math as mlir_math, gpu as mlir_gpu
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import rocdl
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch

from .tensor_shim import GTensor, STensor, _to_raw

_LOG2E = math.log2(math.e)
fm_fast = fx.arith.FastMathFlags.fast

# Matches MlaReduceKernelV1Traits (reduce.cu:13)
# NUM_THREADS is an env-overridable JOINT-SEARCH knob (MLA_NUM_THREADS ∈
# {64,128,256}). It sets the block size (wave count = NT/64) and, via
# VEC = Dv/NT, the per-thread output-accumulator width. Larger NT -> smaller VEC
# -> smaller accumulate register live-set (the main VGPR lever for occupancy),
# at the cost of more waves per block. Must divide Dv (512).
NUM_THREADS = int(os.environ.get("MLA_NUM_THREADS", "256"))
WARP = 64
NUM_WAVES = NUM_THREADS // WARP
OCC = 8
MASSIVE_THR = 4  # kMassiveThreshold

# GRP (splits processed per accumulate-loop iteration / loads-in-flight) is a
# tier-dependent JOINT-SEARCH knob. GRP_M256 drives the long-loop M256/MLDS path
# (nlse>=4), GRP_M64 the M64 path (nlse<4). Must be powers of two (the loop uses
# a shift for num_iters). Overridable via MLA_GRP_M256 / MLA_GRP_M64 for sweeps.
GRP_M256 = int(os.environ.get("MLA_GRP_M256", "16"))
GRP_M64 = int(os.environ.get("MLA_GRP_M64", "8"))

# Runtime M64 sub-split (JOINT-SEARCH knob). When M64_HI_THR > 0, M64 tiles with
# n_splits > M64_HI_THR use GRP=M64_HI_GRP (deeper pipeline, wins high-split
# shapes b8_s32/s26/s13) while low-split tiles keep GRP_M64 (fewer wasted masked
# lanes, holds b8_s6/s5). Device-side branch -> capture-safe. 0 disables.
M64_HI_THR = int(os.environ.get("MLA_M64_HI_THR", "0"))
M64_HI_GRP = int(os.environ.get("MLA_M64_HI_GRP", "16"))
# Persistent-launch grid = num_cu * PS_GRID_MULT. HIP uses num_cu*kOccupancy*2
# (=16 here), but the FlyDSL Tier.ALL kernel runs at occupancy 1 wave/SIMD
# (193 VGPR from the shared massive path), so that 16x grid is ~8x oversubscribed
# on the sparse serving profile: thousands of blocks do one sentinel s_load then
# terminate, and at occupancy 1 that latency cannot hide. grid=num_cu (mult=1)
# trims the wasted blocks (grid-stride still covers any input; dense MLDS is
# bandwidth-bound and unchanged). Overridable via MLA_PS_GRID_MULT for sweeps.
PS_GRID_MULT = 1


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
    ALL = "all"  # all bodies; device-side runtime branch on n_splits (production)


# LSE registers per warp0 lane (None = LDS-backed overflow).
_TIER_NLSE = {
    Tier.SIMPLE: 0,
    Tier.M64: 1,
    Tier.M256: 4,
    Tier.MLDS: None,
    Tier.ALL: None,
}
LDS_MAX_SPLITS = 304  # >= MI300X CU count; compile-time LDS cap
NLSE_MLDS = (LDS_MAX_SPLITS + WARP - 1) // WARP  # ceil(304/64) = 5

# Production default (opt4 sweep: 4 ~= 6 < 8 on H=16 Dv=512 tiles=8 splits=32).
_DEFAULT_WAVES_PER_EU = 4


def waves_per_eu_from_env(default: int = _DEFAULT_WAVES_PER_EU) -> int:
    """Read ``AITER_MLA_REDUCE_WAVES_PER_EU`` (sweep/tuning knob)."""
    raw = os.environ.get("AITER_MLA_REDUCE_WAVES_PER_EU")
    if raw is None:
        return default
    return int(raw)


@contextmanager
def _if_then(if_op):
    """SCF IfOp then-region; auto-insert scf.yield if missing."""
    with ir.InsertionPoint(if_op.then_block):
        try:
            yield if_op.then_block
        finally:
            blk = if_op.then_block
            if (not blk.operations) or not isinstance(
                blk.operations[-1], scf.YieldOp
            ):
                scf.YieldOp([])


@contextmanager
def _if_else(if_op):
    """SCF IfOp else-region; auto-insert scf.yield if missing."""
    if getattr(if_op, "else_block", None) is None:
        raise RuntimeError("IfOp has no else block")
    with ir.InsertionPoint(if_op.else_block):
        try:
            yield if_op.else_block
        finally:
            blk = if_op.else_block
            if (not blk.operations) or not isinstance(
                blk.operations[-1], scf.YieldOp
            ):
                scf.YieldOp([])


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

    tier selects the per-work-item algorithm. ``Tier.ALL`` (production) emits
    every body and branches on device ``n_splits`` per tile (mirrors HIP).
    Other tiers compile a single body for isolated tests. ``persistent`` selects
    the grid-stride launch (kn_mla_reduce_v1_ps); otherwise a 3-D grid launch.

    disable_guards: test-only compile-time knob that skips gather/store bounds
    guards so the suite can run a pre-fix kernel in-process. The production
    wrapper (mla_reduce_kernels.py) never threads this (defaults False).
    """
    assert Dv % NUM_THREADS == 0, "Dv must be divisible by 128"
    assert tier in _TIER_NLSE, f"bad tier {tier}"
    VEC = Dv // NUM_THREADS
    is_runtime_tier = tier == Tier.ALL
    is_massive = tier != Tier.SIMPLE
    if tier == Tier.MLDS:
        NLSE = NLSE_MLDS
    elif tier == Tier.ALL:
        NLSE = 0  # unused; runtime dispatch picks NLSE per branch
    else:
        NLSE = _TIER_NLSE[tier]

    # ---- LDS layout (all tiers: pmap; massive adds scale + overflow) ----
    # [ pmap : LDS_MAX_SPLITS i32 ] ++ [ lse_scale : LDS_MAX_SPLITS f32 ]
    # ++ [ local_lse overflow : (NLSE-4 lanes) f32 ]  (MLDS/ALL only)
    GPU_ARCH = get_rocm_arch()
    allocator = SmemAllocator(
        None,
        arch=GPU_ARCH,
        global_sym_name=f"mla_reduce_smem_{tier.value}_{Dv}_{H}",
    )
    pmap_off = allocator._align(allocator.ptr, 16)
    allocator.ptr = pmap_off + LDS_MAX_SPLITS * 4
    lse_scale_off = 0
    local_lse_off = 0
    if is_massive:
        lse_scale_off = allocator._align(allocator.ptr, 16)
        allocator.ptr = lse_scale_off + LDS_MAX_SPLITS * 4
        if tier in (Tier.MLDS, Tier.ALL):
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

        _glb_ptr_ty = ir.Type.parse("!llvm.ptr<1>")

        def _scalar_indptr(idx):
            """Uniform ``reduce_indptr[idx]`` as a scalar SGPR value.

            Mirrors HIP ``__builtin_amdgcn_readfirstlane(p_reduce_indptr[tile])``
            (reduce.cu:688). Uses a raw pointer deref + readfirstlane rather than
            the GTensor buffer_load path: a ``buffer_load`` is inherently a vector
            memory op (voffset addressing) and never lowers to ``s_load_dword``,
            so wrapping it in readfirstlane leaves the vector load + lgkmcnt(0)
            floor stall in place. A raw ``llvm.load`` from the uniform address is
            scalarizable, so the sentinel becomes an ``s_load_dword``.
            """
            idx_index = fx.arith.index_cast(T.index, _to_raw(idx))
            byte_off = fx.arith.muli(
                idx_index, fx.arith.constant(4, index=True)
            )
            addr_i64 = g_indptr.get_llvm_ptr(reduce_indptr, byte_off)
            addr_ptr = llvm_dialect.inttoptr(_glb_ptr_ty, _to_raw(addr_i64))
            # invariant=True -> !invariant.load: reduce_indptr is read-only in
            # this kernel, so the uniform-address load can be scalarized by the
            # AMDGPU backend into s_load_dword (SMEM) instead of a per-lane
            # global_load + s_waitcnt vmcnt(0), matching HIP's scalar sentinel.
            val = llvm_dialect.load(T.i32, addr_ptr, invariant=True)
            return fx.Int32(rocdl.readfirstlane(T.i32, val))

        c_H = fx.Int32(H)
        last = _scalar_indptr(_to_raw(num_reduce_tile))

        base = allocator.get_base()
        lds_pmap = STensor(
            SmemPtr(base, pmap_off, T.i32, shape=(LDS_MAX_SPLITS,)),
            dtype=T.i32,
            shape=(LDS_MAX_SPLITS,),
        )
        if fx.const_expr(is_massive):
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

            # Stage reduce_partial_map[t0:t1] to LDS once per work item
            # (mirrors reduce.cu:431-438; removes per-split global pmap loads).
            for split_i in range(
                fx.Int32(tid), fx.Index(n_splits), fx.Int32(NUM_THREADS), init=None
            ):
                split_i32 = fx.Int32(split_i)
                lds_pmap[fx.Index(split_i32)] = fx.Int32(
                    g_pmap[fx.Index(t0 + split_i32)]
                )
            fx.gpu.barrier()

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
                row0_idx0 = fx.Int32(lds_pmap[fx.Index(0)])
                row1_idx0 = fx.Int32(lds_pmap[fx.Index(1)])
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

            def row_from_pmap(pmap_i32, local_seq):
                """Bounds-guarded row index from an already-loaded pmap value.

                Shared by the scalar gather_row (LDS read per split) and the
                vectorized group loader (one ds_read_b128 per group), so both
                paths apply the identical clamp/mask.
                """
                row_i32 = pmap_i32 + local_seq
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

            def gather_row(split_i32, local_seq):
                pmap = fx.Int32(lds_pmap[fx.Index(split_i32)])
                return row_from_pmap(pmap, local_seq)

            def load_split_o(split_i32, local_seq):
                row_idx, in_bounds = gather_row(split_i32, local_seq)
                loaded = load_o_elems(g_po, row_idx, fx.Index(head), col)
                zero = fx.arith.constant(0.0, type=T.f32)
                return [in_bounds.select(v, zero) for v in loaded]

            def load_split_o_raw(split_i32, local_seq):
                """Like load_split_o but without consuming the loaded value.

                Returns the raw loaded VEC scalars plus an OOB float mask
                (1.0 in-bounds / 0.0 OOB). Deferring the bounds guard to the
                point of use lets the prefetched load stay in flight (vmcnt(1))
                instead of draining (vmcnt(0)) in the same loop iteration.
                """
                row_idx, in_bounds = gather_row(split_i32, local_seq)
                loaded = load_o_elems(g_po, row_idx, fx.Index(head), col)
                one = fx.arith.constant(1.0, type=T.f32)
                zero = fx.arith.constant(0.0, type=T.f32)
                mask = in_bounds.select(one, zero)
                return loaded, mask

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

            # opt5: FlyDSL range (init=None -> scf_range without iter_args) so
            # hot_loop_scheduler can interleave the inner split-loop VMEM loads
            # with compute. Strided over the seq positions this block owns.
            def emit_simple_body(seq_i32, local_seq):
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
                store_result(seq_i32, out_elems)
                store_lse(seq_i32, max_lse, sum_e)

            def emit_massive_body(seq_i32, local_seq, nlse: int, grp_override=None):
                """Warp0 LSE reduce -> lds_scale -> barrier -> accumulate.

                ``grp_override`` forces the accumulate GRP (loads-in-flight)
                independent of the nlse-derived default, so the runtime M64
                sub-split can give high-split M64 tiles a deeper pipeline
                (GRP=16) while low-split tiles keep GRP=8 (fewer wasted masked
                lanes) -- a per-shape choice the single GRP_M64 constant can't
                make.
                """
                neg_inf = fx.arith.constant(float("-inf"), type=T.f32)
                is_wave0 = fx.arith.cmpi(
                    fx.arith.CmpIPredicate.eq, wave, fx.Int32(0)
                )
                if_w0 = scf.IfOp(is_wave0, results_=[], has_else=False)
                with ir.InsertionPoint(if_w0.then_block):
                    local_lses = []
                    max_lse = neg_inf
                    for j in fx.range_constexpr(nlse):
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
                    for j in fx.range_constexpr(nlse):
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
                    for j in fx.range_constexpr(nlse):
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
                                seq_i32 * fx.Int32(H) + fx.Int32(head)
                            )
                            g_flse[fx.Index(lse_off)] = global_lse
                            scf.YieldOp([])
                    scf.YieldOp([])

                # Lever #1/#5: GRP-wide double-rate software pipeline. Process
                # GRP splits per iteration and keep GRP output buffer_loads in
                # flight (generalizes HIP oaccu_0/oaccu_1 depth-2). Each group's
                # carried loads and the next group's prefetch are distinct SSA
                # values so the compiler allocates separate VGPRs -> genuine
                # vmcnt(>0) overlap (depth-1 aliased the single carried os into
                # the next load dest, forcing vmcnt(0) drain).
                #
                # Bounds are carried as float masks (deferred-guard); the split
                # index is also clamped so OOB prefetches read lds_pmap[0]/
                # lds_scale[0] (always written) -> no NaN*0 pollution, mask=0
                # zeroes the contribution.
                # GRP = splits processed per iteration (loads in flight), tier-
                # dependent. The long-loop M256/MLDS path (nlse>=4, e.g. b1_s128
                # @128 splits) wins ~8% in graph mode at GRP=16 (deeper vmcnt
                # overlap over many iterations, 13.1->12.1us, 0.88x HIP). The M64
                # path (nlse=1) keeps GRP=8: at GRP=16 the low-split tail (b8_s5/
                # s6 @5-6 splits) regresses +11% because a group wastes 10 masked
                # lanes instead of 2, and b8_s32 sees no gain beyond noise.
                if fx.const_expr(grp_override is not None):
                    GRP = grp_override
                else:
                    GRP = GRP_M256 if nlse >= 4 else GRP_M64
                _grp_shift = GRP.bit_length() - 1
                zero_f = fx.arith.constant(0.0, type=T.f32)

                one_f = fx.arith.constant(1.0, type=T.f32)

                def load_os_group(base_i32):
                    """pmap-only phase of a group load: gather rows + issue the
                    GRP os buffer_loads + compute the float OOB masks. Depends on
                    lds_pmap (staged + barriered at the top of the work item),
                    NOT on lds_scale, so it can be hoisted ahead of the LSE-reduce
                    scale barrier (lever #6) to overlap the VMEM load latency with
                    the warp0 LSE reduce + the barrier wait.

                    Vectorized LDS: base = i*GRP is 16B-aligned so the pmap read
                    is a wide ds_read_b128. Returns (os_list, masks, valids); the
                    scale fold happens later in load_scales.

                    Tail safety: OOB lanes (slot >= n_splits) read stale slots, so
                    the pmap value is substituted with slot-0's (always staged,
                    since the massive body only runs when n_splits > 1) and
                    row-clamped -- no stale value ever reaches the gather address,
                    even with guards disabled.
                    """
                    base_idx = fx.Index(base_i32)
                    pmap_v = lds_pmap.load(base_idx, vec_size=GRP)
                    pmap0 = fx.Int32(
                        fx.vector.extract(
                            pmap_v, static_position=[0], dynamic_position=[]
                        )
                    )
                    os_list = []
                    masks = []
                    valids = []
                    for j in fx.range_constexpr(GRP):
                        split_j = base_i32 + fx.Int32(j)
                        in_split = fx.arith.cmpi(
                            fx.arith.CmpIPredicate.slt, split_j, n_splits
                        )
                        pmap_raw = fx.Int32(
                            fx.vector.extract(
                                pmap_v, static_position=[j], dynamic_position=[]
                            )
                        )
                        pmap_j = fx.Int32(in_split.select(pmap_raw, pmap0))
                        row_idx, in_bounds = row_from_pmap(pmap_j, local_seq)
                        os_raw = load_o_elems(g_po, row_idx, fx.Index(head), col)
                        valid = fx.arith.andi(in_split, in_bounds)
                        mask = valid.select(one_f, zero_f)
                        os_list.append(os_raw)
                        masks.append(mask)
                        valids.append(valid)
                    return os_list, masks, valids

                def load_scales(base_i32, valids):
                    """scale phase: wide ds_read_b128 of lds_scale + select-to-0
                    on invalid splits (prevents stale-NaN * mask(0) pollution).
                    Must run after the scale barrier."""
                    scale_v = lds_scale.load(fx.Index(base_i32), vec_size=GRP)
                    scs = []
                    for j in fx.range_constexpr(GRP):
                        sc_j = fx.vector.extract(
                            scale_v, static_position=[j], dynamic_position=[]
                        )
                        scs.append(valids[j].select(sc_j, zero_f))
                    return scs

                def load_group(base_i32):
                    """Combined pmap+scale load for the steady-state loop (both
                    LDS regions are valid past the barrier)."""
                    os_list, masks, valids = load_os_group(base_i32)
                    scs = load_scales(base_i32, valids)
                    return os_list, masks, scs

                # Prologue + lever #6: hoist group-0's os buffer_loads ahead of
                # the scale barrier. They depend only on the already-staged
                # lds_pmap, so issuing them here overlaps GRP VMEM loads with the
                # warp0 LSE reduce and the barrier wait; the barrier-protected
                # scales are folded in afterwards.
                os_g0, mask_g0, valid_g0 = load_os_group(fx.Int32(0))

                fx.gpu.barrier()

                sc_g0 = load_scales(fx.Int32(0), valid_g0)
                init_os = []
                for j in fx.range_constexpr(GRP):
                    init_os += [
                        _to_raw(os_g0[j][k]) for k in fx.range_constexpr(VEC)
                    ]
                init_regs = [_to_raw(zero_f) for _ in fx.range_constexpr(VEC)]
                init_ms = []
                for j in fx.range_constexpr(GRP):
                    init_ms += [_to_raw(mask_g0[j]), _to_raw(sc_g0[j])]
                init_state = init_os + init_regs + init_ms

                # N = floor((n_splits - 1) / GRP) loop iterations; the loop
                # consumes groups 0..N-1 and the epilogue the final carried
                # group N. stop = N * GRP (step GRP).
                n_minus1 = n_splits - fx.Int32(1)
                num_iters = fx.Int32(
                    fx.arith.shrsi(
                        _to_raw(n_minus1),
                        _to_raw(fx.arith.constant(_grp_shift, type=T.i32)),
                    )
                )
                stop_i32 = num_iters * fx.Int32(GRP)
                loop_stop = fx.arith.index_cast(T.index, _to_raw(stop_i32))
                _mbase = (GRP + 1) * VEC

                def unpack(state):
                    os_g = [
                        [state[j * VEC + k] for k in fx.range_constexpr(VEC)]
                        for j in fx.range_constexpr(GRP)
                    ]
                    regs = [
                        state[GRP * VEC + k] for k in fx.range_constexpr(VEC)
                    ]
                    mask_g = [state[_mbase + 2 * j] for j in fx.range_constexpr(GRP)]
                    sc_g = [state[_mbase + 2 * j + 1] for j in fx.range_constexpr(GRP)]
                    return os_g, regs, mask_g, sc_g

                def accumulate(regs, os_g, mask_g, sc_g):
                    new_regs = list(regs)
                    for j in fx.range_constexpr(GRP):
                        sc_eff = sc_g[j] * mask_g[j]
                        new_regs = [
                            new_regs[k] + os_g[j][k] * sc_eff
                            for k in fx.range_constexpr(VEC)
                        ]
                    return new_regs

                for i, state in range(
                    fx.Index(0), loop_stop, fx.Index(GRP), init=init_state
                ):
                    os_g, regs, mask_g, sc_g = unpack(state)
                    i_i32 = fx.Int32(fx.arith.index_cast(T.i32, _to_raw(i)))
                    # Issue the next group's loads FIRST (stay in flight), then
                    # FMA the carried group. Vectorized LDS reads for the group.
                    n_os, n_mask, n_sc = load_group(i_i32 + fx.Int32(GRP))
                    new_regs = accumulate(regs, os_g, mask_g, sc_g)
                    next_os_flat = []
                    for j in fx.range_constexpr(GRP):
                        next_os_flat += [
                            _to_raw(n_os[j][k]) for k in fx.range_constexpr(VEC)
                        ]
                    next_ms = []
                    for j in fx.range_constexpr(GRP):
                        next_ms += [_to_raw(n_mask[j]), _to_raw(n_sc[j])]
                    results = yield (
                        next_os_flat
                        + [_to_raw(r) for r in new_regs]
                        + next_ms
                    )

                os_g, regs, mask_g, sc_g = unpack(results)
                out_elems = accumulate(regs, os_g, mask_g, sc_g)
                store_result(seq_i32, out_elems)

            def dispatch_tier_body(seq_i32, local_seq):
                """Select algorithm from n_splits (runtime for Tier.ALL)."""
                if fx.const_expr(is_runtime_tier):
                    is_lt_thr = fx.arith.cmpi(
                        fx.arith.CmpIPredicate.slt,
                        n_splits,
                        fx.Int32(MASSIVE_THR),
                    )
                    if_lt = scf.IfOp(is_lt_thr, results_=[], has_else=True)
                    with _if_then(if_lt):
                        emit_simple_body(seq_i32, local_seq)
                    with _if_else(if_lt):
                        is_le_64 = fx.arith.cmpi(
                            fx.arith.CmpIPredicate.sle,
                            n_splits,
                            fx.Int32(64),
                        )
                        if_le64 = scf.IfOp(
                            is_le_64, results_=[], has_else=True
                        )
                        with _if_then(if_le64):
                            if fx.const_expr(M64_HI_THR > 0):
                                # Runtime M64 sub-split: high-split M64 tiles
                                # (n_splits > thr) take the deeper GRP=M64_HI_GRP
                                # pipeline; low-split tiles keep GRP_M64 (fewer
                                # wasted masked lanes). Capture-safe (device-side
                                # branch, no host tier baking).
                                is_hi = fx.arith.cmpi(
                                    fx.arith.CmpIPredicate.sgt,
                                    n_splits,
                                    fx.Int32(M64_HI_THR),
                                )
                                if_hi = scf.IfOp(
                                    is_hi, results_=[], has_else=True
                                )
                                with _if_then(if_hi):
                                    emit_massive_body(
                                        seq_i32, local_seq, 1,
                                        grp_override=M64_HI_GRP,
                                    )
                                with _if_else(if_hi):
                                    emit_massive_body(seq_i32, local_seq, 1)
                            else:
                                emit_massive_body(seq_i32, local_seq, 1)
                        with _if_else(if_le64):
                            is_le_256 = fx.arith.cmpi(
                                fx.arith.CmpIPredicate.sle,
                                n_splits,
                                fx.Int32(256),
                            )
                            if_le256 = scf.IfOp(
                                is_le_256, results_=[], has_else=True
                            )
                            with _if_then(if_le256):
                                emit_massive_body(seq_i32, local_seq, 4)
                            with _if_else(if_le256):
                                emit_massive_body(
                                    seq_i32, local_seq, NLSE_MLDS
                                )
                elif fx.const_expr(not is_massive):
                    emit_simple_body(seq_i32, local_seq)
                else:
                    emit_massive_body(seq_i32, local_seq, NLSE)

            for seq in range(seq0, ub_seq, ntg, init=None):
                seq_i32 = fx.Int32(seq)
                local_seq = seq_i32 - q_start
                dispatch_tier_body(seq_i32, local_seq)

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

                # Uniform sentinel -> scalar s_load_dword (reduce.cu:688), not a
                # per-lane buffer_load + s_waitcnt lgkmcnt(0) (the traversal floor).
                tile_start = _scalar_indptr(tile)
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
                    # LDS (lds_pmap, lds_scale) is reused per work item; fence
                    # before the next iteration overwrites it.
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
            _ps_mult = int(os.environ.get("MLA_PS_GRID_MULT", PS_GRID_MULT))
            ps_grid = fx.arith.muli(
                _to_raw(max_splits), fx.arith.constant(_ps_mult, type=T.i32)
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
