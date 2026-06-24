# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL port of the MLA decode reduce/combine epilogue.

Faithful port of the HIP kernel ``kn_mla_reduce_v1`` / ``kn_mla_reduce_v1_ps``
(``csrc/kernels/mla/reduce.cu``). Stage-2 of split-KV MLA decode: merges per-split
partial outputs ``O_i`` (fp32) weighted by ``exp(LSE_i - LSE_max)`` (online softmax)
into the final output (bf16/fp16), and optionally the merged LSE.

This is a pure reduction (no MFMA), HBM-bandwidth bound. See the design dissection
docs for the algorithm and the build/verify plan.

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
"""

import functools
import math

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, gpu, vector, rocdl
from flydsl.expr import range_constexpr, const_expr
from flydsl.expr.typing import T
from flydsl._mlir import ir
from flydsl._mlir.dialects import scf, math as mlir_math, gpu as mlir_gpu
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch

from .tensor_shim import GTensor, STensor, _to_raw

_LOG2E = math.log2(math.e)
fm_fast = arith.FastMathFlags.fast

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
    if const_expr(use_exp2):
        return rocdl.exp2(T.f32, x * _LOG2E)
    return mlir_math.exp(x, fastmath=fm_fast)


def _log(x):
    """natural log; mlir math dialect lowers to the device log."""
    return mlir_math.log(x, fastmath=fm_fast)


def _as_list(state, n):
    """Normalize a loop-carried state to a list of n values.

    FlyDSL's range(init=...) unwraps a single-element carry to a bare scalar;
    re-wrap so element indexing works uniformly for VEC==1.
    """
    if n == 1:
        try:
            return [state[0]]
        except (TypeError, KeyError):
            return [state]
    return [state[i] for i in range(n)]


# Massive-path sub-tiers (mirror MlaReduceProblemSize, reduce.cu:121).
#   simple : register online-softmax, num_splits in {2,3}
#   m64    : <=64 splits, 1 LSE per lane in warp0
#   m256   : <=256 splits, up to 4 LSE per lane in warp0
#   mlds   : >256 splits, 4 in regs + overflow in LDS
_TIER_NLSE = {"simple": 0, "m64": 1, "m256": 4, "mlds": None}
LDS_MAX_SPLITS = 304  # >= MI300X CU count; compile-time LDS cap


def select_tier(num_splits: int) -> str:
    if num_splits < MASSIVE_THR:
        return "simple"
    if num_splits <= 64:
        return "m64"
    if num_splits <= 256:
        return "m256"
    return "mlds"


@functools.lru_cache(maxsize=256)
def compile_mla_reduce(
    *,
    H: int,
    Dv: int,
    out_dtype: str = "bf16",
    tier: str = "simple",
    persistent: bool = False,
    output_lse: bool = False,
    use_reduce_final_map: bool = True,
    prefetch_depth: int = 2,
    waves_per_eu: int = 8,
    use_exp2: bool = True,
    use_packed_cvt: bool = False,
    use_packed_f32_fma: bool = False,
):
    """Compile an MLA reduce kernel for fixed (H, Dv, out_dtype, tier).

    tier selects the per-work-item algorithm (compile-time); the host picks it
    from the observed num_splits via ``select_tier``. Grid-launch, NTG=1.
    """
    assert Dv % NUM_THREADS == 0, "Dv must be divisible by 128"
    assert tier in _TIER_NLSE, f"bad tier {tier}"
    VEC = Dv // NUM_THREADS
    is_massive = tier != "simple"
    if tier == "mlds":
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
            None, arch=GPU_ARCH, global_sym_name=f"mla_reduce_smem_{tier}_{Dv}_{H}"
        )
        lse_scale_off = allocator._align(allocator.ptr, 16)
        allocator.ptr = lse_scale_off + LDS_MAX_SPLITS * 4
        if tier == "mlds":
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
    ):
        out_t = _out_t(out_dtype)
        c_VEC = fx.Index(VEC)
        acc_vt = T.vec(VEC, T.f32)
        out_vt = T.vec(VEC, out_t)

        def load_o_elems(g, row_idx, head_idx, col_idx):
            """Load VEC fp32 elements as a python list of scalars."""
            if const_expr(VEC == 1):
                return [g[row_idx, head_idx, col_idx]]
            v = g.vec_load((row_idx, head_idx, col_idx), VEC)
            return [
                vector.extract(v, static_position=[i], dynamic_position=[])
                for i in range_constexpr(VEC)
            ]

        def store_o_elems(g, off_idx, elems_f32):
            """Cast VEC fp32 scalars to out_t and store."""
            if const_expr(VEC == 1):
                g[off_idx] = elems_f32[0].truncf(out_t)
                return
            ov = vector.from_elements(acc_vt, [_to_raw(e) for e in elems_f32])
            g.vec_store((off_idx,), ov.truncf(out_vt), VEC)

        tid = fx.thread_idx.x
        head = fx.block_idx.x
        block_idx = fx.block_idx.y  # q-pos group (NTG)
        tile = fx.block_idx.z

        g_po = GTensor(partial_output, dtype=T.f32, shape=(-1, H, Dv))
        g_pl = GTensor(partial_lse, dtype=T.f32, shape=(-1, H))
        g_indptr = GTensor(reduce_indptr, dtype=T.i32, shape=(-1,))
        g_pmap = GTensor(reduce_partial_map, dtype=T.i32, shape=(-1,))
        g_fmap = GTensor(reduce_final_map, dtype=T.i32, shape=(-1, 2))
        g_fo = GTensor(final_output, dtype=out_t, shape=(-1,))
        g_flse = GTensor(final_lse, dtype=T.f32, shape=(-1,))

        t0 = fx.Int32(g_indptr[tile])
        t1 = fx.Int32(g_indptr[tile + fx.Index(1)])
        n_splits = t1 - t0

        if const_expr(use_reduce_final_map):
            q_start = fx.Int32(g_fmap[tile, fx.Index(0)])
            q_end = fx.Int32(g_fmap[tile, fx.Index(1)])
        else:
            row0_idx0 = fx.Int32(g_pmap[fx.Index(t0)])
            row1_idx0 = fx.Int32(g_pmap[fx.Index(t0) + fx.Index(1)])
            qo_len = row1_idx0 - row0_idx0
            q_start = fx.Int32(tile) * qo_len
            q_end = (fx.Int32(tile) + fx.Int32(1)) * qo_len

        col = tid * c_VEC
        lane = tid % fx.Int32(WARP)
        wave = tid // fx.Int32(WARP)
        width_i32 = _to_raw(arith.constant(WARP, type=T.i32))

        if const_expr(is_massive):
            base = allocator.get_base()
            lds_scale = STensor(
                SmemPtr(base, lse_scale_off, T.f32, shape=(LDS_MAX_SPLITS,)),
                dtype=T.f32,
                shape=(LDS_MAX_SPLITS,),
            )

        # NTG (number of q-position groups) = grid-y, read at runtime so one
        # compiled kernel is correct for any grid-y in [1, max_seqlen_q]. Mirrors
        # the HIP kernel's `seq += kNumThreadGroupPerBh` stride (reduce.cu:496).
        ntg = fx.Int32(fx.grid_dim.y)
        seq0 = q_start + fx.Int32(block_idx)

        # Skip empty/degenerate tiles (n_splits <= 1). The real decode metadata
        # can hand us reduce_indptr rows of width 0 (no partials for this tile)
        # whose reduce_final_map q-range is uninitialized garbage; HIP guards this
        # with `reduce_tile_start == last` and `num_splits > 1` (reduce.cu:691,743).
        # Clamp the loop's upper bound to its lower bound so a no-work tile runs
        # zero iterations and never dereferences that garbage q-range / stores OOB.
        has_work = arith.cmpi(arith.CmpIPredicate.sgt, n_splits, fx.Int32(1))
        ub_seq = has_work.select(q_end, seq0)

        def gather_row(split_i32, local_seq):
            pmap = fx.Int32(g_pmap[fx.Index(t0 + split_i32)])
            return fx.Index(pmap + local_seq)

        def store_result(seq, out_elems):
            store_off = (
                fx.Int32(seq) * stride_s_o
                + fx.Int32(head) * stride_h_o
                + fx.Int32(tid) * fx.Int32(VEC)
            )
            store_o_elems(g_fo, fx.Index(store_off), out_elems)

        def store_lse(seq, max_lse, sum_e):
            if const_expr(output_lse):
                bad = arith.ori(
                    arith.cmpf(
                        arith.CmpFPredicate.OEQ, sum_e, arith.constant(0.0, type=T.f32)
                    ),
                    arith.cmpf(arith.CmpFPredicate.UNO, sum_e, sum_e),
                )
                lse_val = _log(sum_e) + max_lse
                inf = arith.constant(float("inf"), type=T.f32)
                final_lse_val = bad.select(inf, lse_val)
                is_lane0 = arith.cmpi(arith.CmpIPredicate.eq, tid, fx.Int32(0))
                if_lane0 = scf.IfOp(is_lane0, results_=[], has_else=False)
                with ir.InsertionPoint(if_lane0.then_block):
                    lse_off = fx.Int32(seq) * fx.Int32(H) + fx.Int32(head)
                    g_flse[fx.Index(lse_off)] = final_lse_val
                    scf.YieldOp([])

        # Strided loop over the seq positions this block owns: for seq in
        # range(q_start + blockIdx.y, q_end, ntg). Built as a raw scf.ForOp (not
        # the AST `for` rewriter) to match this file's raw-builder idiom and to
        # avoid auto iter_args inference on a store-only (no-carry) body.
        lb = arith.index_cast(ir.IndexType.get(), _to_raw(seq0))
        ub = arith.index_cast(ir.IndexType.get(), _to_raw(ub_seq))
        st = arith.index_cast(ir.IndexType.get(), _to_raw(ntg))
        for_seq = scf.ForOp(lb, ub, st)
        with ir.InsertionPoint(for_seq.body):
            seq = fx.Int32(for_seq.induction_variable)
            local_seq = seq - q_start

            if const_expr(not is_massive):
                row0 = gather_row(fx.Int32(0), local_seq)
                o0 = load_o_elems(g_po, row0, fx.Index(head), col)
                lse0 = g_pl[row0, fx.Index(head)]

                init = [_to_raw(o0[i]) for i in range_constexpr(VEC)]
                init += [_to_raw(lse0), _to_raw(arith.constant(1.0, type=T.f32))]

                results = init
                for s, state in range(
                    fx.Index(1), fx.Index(n_splits), fx.Index(1), init=init
                ):
                    regs = [state[i] for i in range_constexpr(VEC)]
                    max_lse = state[VEC]
                    sum_e = state[VEC + 1]
                    rs = gather_row(fx.Int32(s), local_seq)
                    os = load_o_elems(g_po, rs, fx.Index(head), col)
                    lse = g_pl[rs, fx.Index(head)]
                    new_max = arith.maximumf(max_lse, lse)
                    old = _exp(max_lse - new_max, use_exp2)
                    new = _exp(lse - new_max, use_exp2)
                    new_regs = [
                        _to_raw(regs[i] * old + os[i] * new)
                        for i in range_constexpr(VEC)
                    ]
                    results = yield new_regs + [
                        _to_raw(new_max),
                        _to_raw(sum_e * old + new),
                    ]

                regs = [results[i] for i in range_constexpr(VEC)]
                max_lse = results[VEC]
                sum_e = results[VEC + 1]
                inv = rocdl.rcp(T.f32, sum_e)
                out_elems = [regs[i] * inv for i in range_constexpr(VEC)]
                store_result(seq, out_elems)
                store_lse(seq, max_lse, sum_e)
            else:
                neg_inf = arith.constant(float("-inf"), type=T.f32)
                is_wave0 = arith.cmpi(arith.CmpIPredicate.eq, wave, fx.Int32(0))
                if_w0 = scf.IfOp(is_wave0, results_=[], has_else=False)
                with ir.InsertionPoint(if_w0.then_block):
                    local_lses = []
                    max_lse = neg_inf
                    for j in range_constexpr(NLSE):
                        split_idx = lane + fx.Int32(j * WARP)
                        in_rng = arith.cmpi(
                            arith.CmpIPredicate.slt, split_idx, n_splits
                        )
                        safe = in_rng.select(split_idx, fx.Int32(0))
                        rs = gather_row(safe, local_seq)
                        lse_j = g_pl[rs, fx.Index(head)]
                        lse_j = in_rng.select(lse_j, neg_inf)
                        local_lses.append(lse_j)
                        max_lse = arith.maximumf(max_lse, lse_j)
                    for off in [32, 16, 8, 4, 2, 1]:
                        peer = mlir_gpu.ShuffleOp(
                            _to_raw(max_lse),
                            _to_raw(arith.constant(off, type=T.i32)),
                            width_i32,
                            mode="xor",
                        ).shuffleResult
                        max_lse = arith.maximumf(max_lse, peer)
                    sum_e = arith.constant(0.0, type=T.f32)
                    for j in range_constexpr(NLSE):
                        sum_e = sum_e + _exp(local_lses[j] - max_lse, use_exp2)
                    for off in [32, 16, 8, 4, 2, 1]:
                        peer = mlir_gpu.ShuffleOp(
                            _to_raw(sum_e),
                            _to_raw(arith.constant(off, type=T.i32)),
                            width_i32,
                            mode="xor",
                        ).shuffleResult
                        sum_e = sum_e + peer
                    bad = arith.ori(
                        arith.cmpf(
                            arith.CmpFPredicate.OEQ,
                            sum_e,
                            arith.constant(0.0, type=T.f32),
                        ),
                        arith.cmpf(arith.CmpFPredicate.UNO, sum_e, sum_e),
                    )
                    inf = arith.constant(float("inf"), type=T.f32)
                    global_lse = bad.select(inf, _log(sum_e) + max_lse)
                    for j in range_constexpr(NLSE):
                        split_idx = lane + fx.Int32(j * WARP)
                        in_rng = arith.cmpi(
                            arith.CmpIPredicate.slt, split_idx, n_splits
                        )
                        if_s = scf.IfOp(in_rng, results_=[], has_else=False)
                        with ir.InsertionPoint(if_s.then_block):
                            sc = _exp(local_lses[j] - global_lse, use_exp2)
                            lds_scale[fx.Index(split_idx)] = sc
                            scf.YieldOp([])
                    if const_expr(output_lse):
                        is_l0 = arith.cmpi(arith.CmpIPredicate.eq, lane, fx.Int32(0))
                        if_l0 = scf.IfOp(is_l0, results_=[], has_else=False)
                        with ir.InsertionPoint(if_l0.then_block):
                            lse_off = fx.Int32(seq) * fx.Int32(H) + fx.Int32(head)
                            g_flse[fx.Index(lse_off)] = global_lse
                            scf.YieldOp([])
                    scf.YieldOp([])

                gpu.barrier()

                acc0 = [
                    _to_raw(arith.constant(0.0, type=T.f32))
                    for _ in range_constexpr(VEC)
                ]
                accr = acc0
                for s, state in range(
                    fx.Index(0), fx.Index(n_splits), fx.Index(1), init=acc0
                ):
                    regs = _as_list(state, VEC)
                    rs = gather_row(fx.Int32(s), local_seq)
                    os = load_o_elems(g_po, rs, fx.Index(head), col)
                    sc = lds_scale[fx.Index(s)]
                    new_regs = [
                        _to_raw(regs[i] + os[i] * sc) for i in range_constexpr(VEC)
                    ]
                    accr = yield new_regs
                accr = _as_list(accr, VEC)
                out_elems = [accr[i] for i in range_constexpr(VEC)]
                store_result(seq, out_elems)
            scf.YieldOp([])
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
        stream: fx.Stream = fx.Stream(None),
    ):
        if const_expr(is_massive):
            allocator.finalized = False
            ctx = CompilationContext.get_current()
            with ir.InsertionPoint(ctx.gpu_module_body):
                allocator.finalize()

        idx_tiles = arith.index_cast(T.index, num_reduce_tile)
        idx_H = fx.Index(H)
        idx_ntg = arith.index_cast(T.index, _to_raw(max_seqlen_q))
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
        ).launch(
            grid=(idx_H, idx_ntg, idx_tiles),
            block=(NUM_THREADS, 1, 1),
            stream=stream,
        )

    return launch_mla_reduce
