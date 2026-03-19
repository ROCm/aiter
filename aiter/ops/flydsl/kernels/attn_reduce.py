# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL attention reduce kernel: kernel definitions and launchers."""

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import gpu, buffer_ops, vector, arith
from flydsl.expr.typing import T

from flydsl._mlir import ir
from flydsl._mlir.dialects import scf
from flydsl._mlir.dialects import arith as _mlir_arith
from flydsl._mlir.dialects import gpu as _mlir_gpu
from flydsl._mlir.dialects import math as mlir_math
from flydsl._mlir.dialects import vector as _mlir_vector
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.expr.arith import _to_raw as _raw

WARP_SIZE = 64  # warp_size = 64 on AMD MI3xx GPU
NUM_WARPS = 2
NUM_THREADS = NUM_WARPS * WARP_SIZE
OCCUPANCY = 8
MASSIVE_THRESHOLD = max(
    4, 2
)  # at least 2. never handle the case that #splits = 0 or 1.


def _to_mlir(val, index=False):
    """Convert a Python int/float, ArithValue, or ir.Value to a raw MLIR Value.

    For Python ``int``, produces an ``i32`` constant by default.
    Pass ``index=True`` to get an ``index``-typed constant instead.
    For Python ``float``, produces an ``f32`` constant.
    ArithValue is unwrapped via ``_raw``; bare ``ir.Value`` is returned as-is.
    """
    if isinstance(val, int):
        return _raw(arith.constant(val, index=index))
    if isinstance(val, float):
        return _raw(arith.constant(val))
    if isinstance(val, ir.Value):
        return val
    return _raw(val)


def _scf_for(start, stop, step):
    """Pythonic SCF for-loop usable in non-AST-rewritten helper functions.

    Handles index-type conversion and auto-emits ``scf.YieldOp``.
    Usage::

        for i in _scf_for(0, n, stride):
            # loop body -- no scf.YieldOp needed
    """
    for_op = scf.ForOp(
        _to_mlir(start, index=True),
        _to_mlir(stop, index=True),
        _to_mlir(step, index=True),
    )
    with ir.InsertionPoint(for_op.body):
        yield for_op.induction_variable
        scf.YieldOp([])


def _warp_reduce_max(val):
    """Butterfly warp reduce for max (f32). All lanes get the result."""
    width = _mlir_arith.constant(T.i32, WARP_SIZE)
    w = _to_mlir(val)
    for sh in [32, 16, 8, 4, 2, 1]:
        offset = _mlir_arith.constant(T.i32, sh)
        peer = _mlir_gpu.ShuffleOp(
            w, offset, width, _mlir_gpu.ShuffleMode.XOR
        ).shuffleResult
        w = _mlir_arith.maximumf(w, peer)
    return w


def _warp_reduce_add(val):
    """Butterfly warp reduce for sum (f32). All lanes get the result."""
    width = _mlir_arith.constant(T.i32, WARP_SIZE)
    w = _to_mlir(val)
    for sh in [32, 16, 8, 4, 2, 1]:
        offset = _mlir_arith.constant(T.i32, sh)
        peer = _mlir_gpu.ShuffleOp(
            w, offset, width, _mlir_gpu.ShuffleMode.XOR
        ).shuffleResult
        w = _mlir_arith.addf(w, peer)
    return w


def _load_lse_for_lane(
    local_idx,
    lane_idx,
    reduce_tile_start,
    reduce_tile_end,
    lds_i32,
    partial_lse_rsrc,
    local_seq,
    num_heads_q,
    head_idx,
):
    """Load one LSE value for local_idx-th iteration of the per-lane loop.

    split_idx = local_idx * WARP_SIZE + lane_idx
    If out of bounds, returns -inf.
    local_idx is a raw i32 ir.Value, lane_idx is raw i32 ir.Value.
    """
    warp_sz = _mlir_arith.constant(T.i32, WARP_SIZE)
    split_idx = _mlir_arith.addi(_mlir_arith.muli(local_idx, warp_sz), lane_idx)
    num_splits = _mlir_arith.subi(_raw(reduce_tile_end), _raw(reduce_tile_start))
    in_bounds = _mlir_arith.cmpi(
        _mlir_arith.CmpIPredicate.slt,
        split_idx,
        num_splits,
    )
    neg_inf = _mlir_arith.constant(T.f32, float("-inf"))

    # Conditional load via scf.IfOp (returns f32)
    if_op = scf.IfOp(in_bounds, [T.f32], has_else=True)
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        lds_partial = SmemPtr(lds_i32.base_memref, 0, T.i32, shape=(2048,))
        lds_idx = _mlir_arith.index_cast(ir.IndexType.get(), split_idx)
        partial_map_val = lds_partial.load([lds_idx])
        lse_offset = (partial_map_val + local_seq) * num_heads_q + head_idx
        loaded = buffer_ops.buffer_load(
            partial_lse_rsrc,
            lse_offset,
            vec_width=1,
            dtype=T.f32,
        )
        scf.YieldOp([_raw(loaded)])
    with ir.InsertionPoint(if_op.regions[1].blocks[0]):
        scf.YieldOp([neg_inf])
    return if_op.results[0]


def _finalize_lse(
    max_lse,
    sum_lse,
    output_lse,
    final_lse_rsrc,
    seq_idx,
    num_heads_q,
    head_idx,
    lane_idx,
):
    """Compute global_lse from warp-reduced max and sum. Optionally store final LSE."""
    zero = _mlir_arith.constant(T.f32, 0.0)
    inf_val = _mlir_arith.constant(T.f32, float("inf"))
    is_zero = _mlir_arith.cmpf(_mlir_arith.CmpFPredicate.OEQ, sum_lse, zero)
    is_nan = _mlir_arith.cmpf(_mlir_arith.CmpFPredicate.UNE, sum_lse, sum_lse)
    is_invalid = _mlir_arith.ori(is_zero, is_nan)
    log_sum = mlir_math.log(sum_lse)
    global_lse_normal = _mlir_arith.addf(log_sum, max_lse)
    global_lse = _mlir_arith.select(is_invalid, inf_val, global_lse_normal)

    if output_lse:
        is_lane0 = _mlir_arith.cmpi(
            _mlir_arith.CmpIPredicate.eq, lane_idx, _mlir_arith.constant(T.i32, 0)
        )
        if_lane0 = scf.IfOp(is_lane0, [], has_else=False)
        with ir.InsertionPoint(if_lane0.regions[0].blocks[0]):
            lse_out_offset = seq_idx * num_heads_q + head_idx
            buffer_ops.buffer_store(global_lse, final_lse_rsrc, lse_out_offset)
            scf.YieldOp([])

    return global_lse


def _write_lse_scale_to_lds(
    local_lse, global_lse, split_idx, max_splits, lds_i32, in_bounds
):
    """Write exp(local_lse - global_lse) to LDS lse_scale region.

    LDS layout: partial_map[0..max_splits-1] then lse_scale[0..max_splits-1]
    in i32 units.  lse_scale[i] is at lds_i32[max_splits + i].
    """
    lse_scale = mlir_math.exp(_mlir_arith.subf(local_lse, global_lse))
    lse_scale_i32 = _mlir_arith.bitcast(T.i32, lse_scale)
    if_op = scf.IfOp(in_bounds, [], has_else=False)
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        lds_w = SmemPtr(lds_i32.base_memref, 0, T.i32, shape=(2048,))
        store_idx = _mlir_arith.addi(_raw(max_splits), split_idx)
        store_idx_index = _mlir_arith.index_cast(ir.IndexType.get(), store_idx)
        lds_w.store(lse_scale_i32, [store_idx_index])
        scf.YieldOp([])


class _LocalLse64:
    """Per-lane LSE storage for <= 64 splits: 1 value in a register."""

    num_iters = 1

    def __init__(self, max_splits, lane_idx, lds_i32):
        self._val = None

    def store(self, local_idx, val):
        self._val = val

    def load(self, local_idx):
        return self._val


class _LocalLse256:
    """Per-lane LSE storage for <= 256 splits: 4 values in registers."""

    num_iters = 4

    def __init__(self, max_splits, lane_idx, lds_i32):
        self._vals = [None] * 4

    def store(self, local_idx, val):
        self._vals[local_idx] = val

    def load(self, local_idx):
        return self._vals[local_idx]


class _LocalLseLds:
    """Per-lane LSE storage for > 256 splits: values in LDS.

    LDS layout: partial_map[max_splits] | lse_scale[max_splits] | local_lse[...]
    local_lse[split_idx] stored at lds_i32[max_splits*2 + split_idx] as bitcast i32.
    """

    def __init__(self, max_splits, lane_idx, lds_i32):
        self._lane_idx = lane_idx
        self._lds_i32 = lds_i32
        self._max_splits_2 = _mlir_arith.muli(
            _raw(max_splits), _mlir_arith.constant(T.i32, 2)
        )
        # num_iters = ceil(max_splits / WARP_SIZE)
        warp_m1 = _mlir_arith.constant(T.i32, WARP_SIZE - 1)
        log2_warp = _mlir_arith.constant(T.i32, 6)  # log2(64) = 6
        self.num_iters = _mlir_arith.shrui(
            _mlir_arith.addi(_raw(max_splits), warp_m1), log2_warp
        )

    def _split_idx(self, local_idx):
        warp_sz = _mlir_arith.constant(T.i32, WARP_SIZE)
        return _mlir_arith.addi(_mlir_arith.muli(local_idx, warp_sz), self._lane_idx)

    def store(self, local_idx, val):
        split_idx = self._split_idx(local_idx)
        lds_idx = _mlir_arith.addi(self._max_splits_2, split_idx)
        lds_idx_ix = _mlir_arith.index_cast(ir.IndexType.get(), lds_idx)
        val_i32 = _mlir_arith.bitcast(T.i32, val)
        self._lds_i32.store(val_i32, [lds_idx_ix])

    def load(self, local_idx):
        split_idx = self._split_idx(local_idx)
        lds_idx = _mlir_arith.addi(self._max_splits_2, split_idx)
        lds_idx_ix = _mlir_arith.index_cast(ir.IndexType.get(), lds_idx)
        val_i32 = self._lds_i32.load([lds_idx_ix])
        return _mlir_arith.bitcast(T.f32, _raw(val_i32))


def _reduce_lse_massive(
    local_lse_storage,
    lane_idx,
    partial_lse_rsrc,
    head_idx,
    num_heads_q,
    reduce_tile_start,
    reduce_tile_end,
    max_splits,
    lds_i32,
    output_lse,
    final_lse_rsrc,
    seq_idx,
    local_seq,
):
    """Unified LSE reduction using local_lse_storage for per-lane value management."""
    neg_inf = _mlir_arith.constant(T.f32, float("-inf"))
    zero_f32 = _mlir_arith.constant(T.f32, 0.0)
    num_splits = _mlir_arith.subi(_raw(reduce_tile_end), _raw(reduce_tile_start))
    num_iters = local_lse_storage.num_iters

    if isinstance(num_iters, int):
        # Static path: Python-unrolled
        max_lse = neg_inf
        for i in range(num_iters):
            local_idx_i = _mlir_arith.constant(T.i32, i)
            lse_i = _load_lse_for_lane(
                local_idx_i,
                lane_idx,
                reduce_tile_start,
                reduce_tile_end,
                lds_i32,
                partial_lse_rsrc,
                local_seq,
                num_heads_q,
                head_idx,
            )
            local_lse_storage.store(i, lse_i)
            max_lse = _mlir_arith.maximumf(max_lse, lse_i)

        max_lse = _warp_reduce_max(max_lse)

        sum_lse = zero_f32
        for i in range(num_iters):
            lse_i = local_lse_storage.load(i)
            exp_i = mlir_math.exp(_mlir_arith.subf(lse_i, max_lse))
            sum_lse = _mlir_arith.addf(sum_lse, exp_i)
        sum_lse = _warp_reduce_add(sum_lse)

        global_lse = _finalize_lse(
            max_lse,
            sum_lse,
            output_lse,
            final_lse_rsrc,
            seq_idx,
            num_heads_q,
            head_idx,
            lane_idx,
        )

        warp_sz = _mlir_arith.constant(T.i32, WARP_SIZE)
        for i in range(num_iters):
            split_idx = _mlir_arith.addi(
                _mlir_arith.muli(_mlir_arith.constant(T.i32, i), warp_sz), lane_idx
            )
            in_bounds = _mlir_arith.cmpi(
                _mlir_arith.CmpIPredicate.slt,
                split_idx,
                num_splits,
            )
            lse_i = local_lse_storage.load(i)
            _write_lse_scale_to_lds(
                lse_i,
                global_lse,
                split_idx,
                max_splits,
                lds_i32,
                in_bounds,
            )
    else:
        # Dynamic path: scf.ForOp
        n_idx = _mlir_arith.index_cast(ir.IndexType.get(), num_iters)
        zero_idx = _to_mlir(0, index=True)
        one_idx = _to_mlir(1, index=True)
        warp_sz = _mlir_arith.constant(T.i32, WARP_SIZE)

        # Phase 1: Load LSEs, store to storage, compute max
        for_op = scf.ForOp(zero_idx, n_idx, one_idx, iter_args=[neg_inf])
        with ir.InsertionPoint(for_op.body):
            local_idx = _mlir_arith.index_cast(T.i32, for_op.induction_variable)
            max_carry = for_op.inner_iter_args[0]
            lse_val = _load_lse_for_lane(
                local_idx,
                lane_idx,
                reduce_tile_start,
                reduce_tile_end,
                lds_i32,
                partial_lse_rsrc,
                local_seq,
                num_heads_q,
                head_idx,
            )
            local_lse_storage.store(local_idx, lse_val)
            new_max = _mlir_arith.maximumf(max_carry, lse_val)
            scf.YieldOp([new_max])

        max_lse = _warp_reduce_max(for_op.results[0])

        # Phase 2: Compute sum of exp(lse - max)
        for_op2 = scf.ForOp(zero_idx, n_idx, one_idx, iter_args=[zero_f32])
        with ir.InsertionPoint(for_op2.body):
            local_idx2 = _mlir_arith.index_cast(T.i32, for_op2.induction_variable)
            sum_carry = for_op2.inner_iter_args[0]
            lse_val2 = local_lse_storage.load(local_idx2)
            exp_val = mlir_math.exp(_mlir_arith.subf(lse_val2, max_lse))
            new_sum = _mlir_arith.addf(sum_carry, exp_val)
            scf.YieldOp([new_sum])

        sum_lse = _warp_reduce_add(for_op2.results[0])

        global_lse = _finalize_lse(
            max_lse,
            sum_lse,
            output_lse,
            final_lse_rsrc,
            seq_idx,
            num_heads_q,
            head_idx,
            lane_idx,
        )

        # Phase 3: Write lse_scale to LDS
        for_op3 = scf.ForOp(zero_idx, n_idx, one_idx)
        with ir.InsertionPoint(for_op3.body):
            local_idx3 = _mlir_arith.index_cast(T.i32, for_op3.induction_variable)
            split_idx3 = _mlir_arith.addi(
                _mlir_arith.muli(local_idx3, warp_sz), lane_idx
            )
            in_bounds3 = _mlir_arith.cmpi(
                _mlir_arith.CmpIPredicate.slt,
                split_idx3,
                num_splits,
            )
            lse_val3 = local_lse_storage.load(local_idx3)
            _write_lse_scale_to_lds(
                lse_val3,
                global_lse,
                split_idx3,
                max_splits,
                lds_i32,
                in_bounds3,
            )
            scf.YieldOp([])


def _reduce_output_massive(
    partial_output_rsrc,
    final_output_rsrc,
    head_idx,
    size_dv,
    num_heads_q,
    reduce_tile_start,
    reduce_tile_end,
    seq_idx,
    local_seq,
    stride_s_o,
    stride_h_o,
    max_splits,
    lds_i32,
    out_elem_bytes,
):
    """Accumulate partial outputs weighted by pre-computed lse_scale from LDS."""
    VEC_WIDTH = size_dv // NUM_THREADS
    use_vec = VEC_WIDTH > 1

    thread_id = gpu.thread_id("x")
    thread_id_i32 = arith.index_cast(T.i32, thread_id)

    if use_vec:
        vec_type = ir.VectorType.get([VEC_WIDTH], T.f32)

    def _scale_vec(scale, data):
        if use_vec:
            sv = _mlir_vector.broadcast(vec_type, scale)
            return _mlir_arith.mulf(sv, data)
        return _mlir_arith.mulf(scale, data)

    # Initialize reg_out = 0
    if use_vec:
        init_out = _mlir_vector.broadcast(vec_type, _raw(arith.constant(0.0)))
    else:
        init_out = _raw(arith.constant(0.0))

    # Loop over all splits
    rts_idx = _to_mlir(arith.index_cast(T.index, _raw(reduce_tile_start)))
    rte_idx = _to_mlir(arith.index_cast(T.index, _raw(reduce_tile_end)))
    one_idx = _to_mlir(arith.constant(1, index=True))

    # LDS pointers are loop-invariant -- hoist outside the loop
    lds_map = SmemPtr(lds_i32.base_memref, 0, T.i32, shape=(2048,))
    lds_scale = SmemPtr(lds_i32.base_memref, 0, T.i32, shape=(2048,))
    max_splits_idx = _mlir_arith.index_cast(ir.IndexType.get(), _raw(max_splits))

    for_op = scf.ForOp(rts_idx, rte_idx, one_idx, iter_args=[init_out])
    with ir.InsertionPoint(for_op.body):
        split_iv = for_op.induction_variable
        reg_out_v = for_op.inner_iter_args[0]

        # Read partial_map_val from LDS
        lds_idx = _mlir_arith.subi(split_iv, rts_idx)
        partial_map_val = lds_map.load([lds_idx])

        # Read lse_scale from LDS (at offset max_splits + lds_idx, bitcast to f32)
        scale_lds_idx = _mlir_arith.addi(max_splits_idx, lds_idx)
        scale_i32 = lds_scale.load([scale_lds_idx])
        lse_scale = _mlir_arith.bitcast(T.f32, _raw(scale_i32))

        # Compute output offset and load partial output
        split_lse_base = (partial_map_val + local_seq) * num_heads_q
        split_out_base = split_lse_base * size_dv
        split_out_offset = (
            split_out_base + head_idx * size_dv + thread_id_i32 * VEC_WIDTH
        )

        oaccu = buffer_ops.buffer_load(
            partial_output_rsrc,
            split_out_offset,
            vec_width=VEC_WIDTH,
            dtype=T.f32,
        )

        # reg_out += lse_scale * oaccu
        scaled = _scale_vec(lse_scale, _raw(oaccu))
        new_reg_out = _mlir_arith.addf(reg_out_v, scaled)
        scf.YieldOp([new_reg_out])

    final_reg_out = for_op.results[0]

    # Cast and store to final_output
    final_out_offset = (
        seq_idx * stride_s_o + head_idx * stride_h_o + thread_id_i32 * VEC_WIDTH
    )
    if out_elem_bytes == 2:
        if use_vec:
            out_type = ir.VectorType.get([VEC_WIDTH], ir.BF16Type.get())
        else:
            out_type = ir.BF16Type.get()
        store_val = _mlir_arith.truncf(out_type, final_reg_out)
    else:
        store_val = final_reg_out
    buffer_ops.buffer_store(store_val, final_output_rsrc, final_out_offset)


def attn_reduce_massive(
    reduce_final_map_rsrc: ir.Value,
    reduce_partial_map_rsrc: ir.Value,
    final_lse_rsrc: ir.Value,
    final_output_rsrc: ir.Value,
    partial_lse_rsrc: ir.Value,
    partial_output_rsrc: ir.Value,
    stride_s_o,
    stride_h_o,
    max_splits,
    output_lse: int,
    use_reduce_final_map: int,
    num_heads_q: int,
    size_dv: int,
    out_elem_bytes: int,
    num_wg_per_bh,
    head_idx,
    block_idx,
    tile_idx,
    reduce_tile_start,
    reduce_tile_end,
    lds_allocator,
    local_lse_cls,
):
    """Two-phase massive attention reduce for num_splits >= MASSIVE_THRESHOLD.

    Phase 1 (warp 0): reduce LSE values, compute per-split lse_scale -> LDS.
    Phase 2 (all threads): accumulate partial outputs weighted by lse_scale.

    local_lse_cls: one of _LocalLse64, _LocalLse256, _LocalLseLds -- controls
    how per-lane LSE values are stored during phase 1.
    """
    lds_buffer = lds_allocator.get_base()
    lds_i32 = SmemPtr(lds_buffer, 0, T.i32, shape=(2048,))
    lds_i32.get()  # emit memref.view at outer scope so the cache dominates everywhere

    thread_id = gpu.thread_id("x")
    num_splits = reduce_tile_end - reduce_tile_start
    num_splits_idx = arith.index_cast(T.index, _raw(num_splits))
    reduce_tile_start_idx = arith.index_cast(T.index, _raw(reduce_tile_start))

    # -- Step 1: Load reduce_partial_map into LDS --
    for effective_idx in _scf_for(thread_id, num_splits_idx, NUM_THREADS):
        in_bounds = arith.cmpi(arith.CmpIPredicate.ult, effective_idx, num_splits_idx)
        if_op = scf.IfOp(_raw(in_bounds), [], has_else=False)
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            val = buffer_ops.buffer_load(
                reduce_partial_map_rsrc,
                arith.index_cast(T.i32, reduce_tile_start_idx + effective_idx),
                vec_width=1,
                dtype=T.i32,
            )
            lds_i32.store(val, [effective_idx])
            scf.YieldOp([])

    gpu.barrier()

    # -- Step 2: Read partial_map[0], [1] from LDS --
    partial_map_0 = lds_i32.load([arith.constant(0, index=True)])
    partial_map_1 = lds_i32.load([arith.constant(1, index=True)])

    # -- Step 3: Compute q_start, q_end --
    if use_reduce_final_map:
        final_map_pair = buffer_ops.buffer_load(
            reduce_final_map_rsrc,
            tile_idx * 2,
            vec_width=2,
            dtype=T.i32,
        )
        q_start = vector.extract(final_map_pair, [0])
        q_end = vector.extract(final_map_pair, [1])
    else:
        qo_len = partial_map_1 - partial_map_0
        q_start = tile_idx * qo_len
        q_end = (tile_idx + 1) * qo_len

    # -- Step 4: Outer seq loop --
    q_start_idx = _to_mlir(arith.index_cast(T.index, _raw(q_start + block_idx)))
    q_end_idx = _to_mlir(arith.index_cast(T.index, _raw(q_end)))
    # num_wg_per_bh is a Python int (Constexpr); pass directly to _scf_for
    # which converts to index internally.
    num_wg_idx = num_wg_per_bh

    # Compute lane_idx once at outer scope (used by _reduce_lse_massive)
    lane_idx = _mlir_arith.andi(
        _raw(arith.index_cast(T.i32, thread_id)),
        _mlir_arith.constant(T.i32, WARP_SIZE - 1),
    )

    # Construct local LSE storage at outer scope (before the seq loop).
    # For _LocalLseLds, this emits the num_iters computation which must dominate
    # all uses inside the loop body.
    local_lse_storage = local_lse_cls(max_splits, lane_idx, lds_i32)

    for seq_iv in _scf_for(q_start_idx, q_end_idx, num_wg_idx):
        seq_idx = arith.index_cast(T.i32, seq_iv)
        local_seq = seq_idx - q_start

        # -- Phase 1: LSE reduction (warp 0 only) --
        warp_id = arith.index_cast(T.i32, gpu.thread_id("x")) // WARP_SIZE
        is_warp0 = _mlir_arith.cmpi(
            _mlir_arith.CmpIPredicate.eq,
            _raw(warp_id),
            _mlir_arith.constant(T.i32, 0),
        )
        if_warp0 = scf.IfOp(is_warp0, [], has_else=False)
        with ir.InsertionPoint(if_warp0.regions[0].blocks[0]):
            _reduce_lse_massive(
                local_lse_storage,
                lane_idx,
                partial_lse_rsrc,
                head_idx,
                num_heads_q,
                reduce_tile_start,
                reduce_tile_end,
                max_splits,
                lds_i32,
                output_lse,
                final_lse_rsrc,
                seq_idx,
                local_seq,
            )
            scf.YieldOp([])

        # -- Barrier between phase 1 and phase 2 --
        gpu.barrier()

        # -- Phase 2: Output accumulation (all threads) --
        _reduce_output_massive(
            partial_output_rsrc,
            final_output_rsrc,
            head_idx,
            size_dv,
            num_heads_q,
            reduce_tile_start,
            reduce_tile_end,
            seq_idx,
            local_seq,
            stride_s_o,
            stride_h_o,
            max_splits,
            lds_i32,
            out_elem_bytes,
        )


def attn_reduce_simple(
    reduce_final_map_rsrc: ir.Value,
    reduce_partial_map_rsrc: ir.Value,
    final_lse_rsrc: ir.Value,
    final_output_rsrc: ir.Value,
    partial_lse_rsrc: ir.Value,
    partial_output_rsrc: ir.Value,
    stride_s_o,
    stride_h_o,
    max_splits,
    output_lse: int,
    use_reduce_final_map: int,
    num_heads_q: int,
    size_dv: int,
    out_elem_bytes: int,
    num_wg_per_bh,
    head_idx,
    block_idx,
    tile_idx,
    reduce_tile_start,
    reduce_tile_end,
    lds_allocator,
):
    # Derived constants (Python-level, becomes Constexpr)
    VEC_WIDTH = size_dv // NUM_THREADS

    # get the LDS memref inside the kernel body
    lds_buffer = lds_allocator.get_base()

    thread_id = gpu.thread_id("x")

    num_splits = reduce_tile_end - reduce_tile_start
    num_splits_idx = arith.index_cast(T.index, _raw(num_splits))
    reduce_tile_start_idx = arith.index_cast(T.index, _raw(reduce_tile_start))

    # -- Step 1: Load reduce_partial_map into LDS --
    lds_i32 = SmemPtr(lds_buffer, 0, T.i32, shape=(2048,))
    for effective_idx in _scf_for(thread_id, num_splits_idx, NUM_THREADS):
        in_bounds = arith.cmpi(arith.CmpIPredicate.ult, effective_idx, num_splits_idx)
        if_op = scf.IfOp(_raw(in_bounds), [], has_else=False)
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            val = buffer_ops.buffer_load(
                reduce_partial_map_rsrc,
                arith.index_cast(T.i32, reduce_tile_start_idx + effective_idx),
                vec_width=1,
                dtype=T.i32,
            )
            lds_i32.store(val, [effective_idx])
            scf.YieldOp([])

    # -- Step 2: barrier --
    gpu.barrier()

    # -- Step 3: Read partial_map[0], [1] from LDS --
    # Fresh SmemPtr to avoid SSA dominance issues with the view cached by Step 1.
    lds_i32_r = SmemPtr(lds_buffer, 0, T.i32, shape=(2048,))
    partial_map_0 = lds_i32_r.load([arith.constant(0, index=True)])
    partial_map_1 = lds_i32_r.load([arith.constant(1, index=True)])

    # -- Step 4: Compute q_start, q_end --
    if use_reduce_final_map:
        # Load reduce_final_map[tile_idx] as a pair of i32 (q_start, q_end)
        final_map_pair = buffer_ops.buffer_load(
            reduce_final_map_rsrc,
            tile_idx * 2,
            vec_width=2,
            dtype=T.i32,
        )
        q_start = vector.extract(final_map_pair, [0])
        q_end = vector.extract(final_map_pair, [1])
    else:
        qo_len = partial_map_1 - partial_map_0
        q_start = tile_idx * qo_len
        q_end = (tile_idx + 1) * qo_len

    # -- Step 5: Outer seq loop --
    thread_id_i32 = arith.index_cast(T.i32, thread_id)
    q_start_idx = _to_mlir(arith.index_cast(T.index, _raw(q_start + block_idx)))
    q_end_idx = _to_mlir(arith.index_cast(T.index, _raw(q_end)))
    # num_wg_per_bh is a Python int (Constexpr); pass directly to _scf_for
    # which converts to index internally.
    num_wg_idx = num_wg_per_bh

    use_vec = VEC_WIDTH > 1
    if use_vec:
        vec_type = ir.VectorType.get([VEC_WIDTH], T.f32)
    one_idx = _to_mlir(arith.constant(1, index=True))

    def _scale_accum(scale, data):
        """Multiply scalar scale by data (scalar or vector)."""
        if use_vec:
            sv = _mlir_vector.broadcast(vec_type, scale)
            return _mlir_arith.mulf(sv, data)
        return _mlir_arith.mulf(scale, data)

    for seq_iv in _scf_for(q_start_idx, q_end_idx, num_wg_idx):
        seq_idx = arith.index_cast(T.i32, seq_iv)
        local_seq = seq_idx - q_start

        # -- Step 5a: Load first split's output tile and LSE --
        # partial_lse offset = (partial_map_0 + local_seq) * num_heads_q + head_idx
        lse_base = (partial_map_0 + local_seq) * num_heads_q
        lse_offset = lse_base + head_idx

        # partial_output offset =
        #   (partial_map_0 + local_seq) * num_heads_q * size_dv
        #   + head_idx * size_dv + thread_id * VEC_WIDTH
        out_base = lse_base * size_dv
        out_offset = out_base + head_idx * size_dv + thread_id_i32 * VEC_WIDTH

        reg_out = buffer_ops.buffer_load(
            partial_output_rsrc,
            out_offset,
            vec_width=VEC_WIDTH,
            dtype=T.f32,
        )
        max_lse = buffer_ops.buffer_load(
            partial_lse_rsrc,
            lse_offset,
            vec_width=1,
            dtype=T.f32,
        )
        sum_e_lse = arith.constant(1.0)

        # -- Step 5b: Inner split loop (carried state: reg_out, max_lse, sum_e_lse) --
        rts_idx = _to_mlir(arith.index_cast(T.index, _raw(reduce_tile_start)))
        rte_idx = _to_mlir(arith.index_cast(T.index, _raw(reduce_tile_end)))
        loop_start = _mlir_arith.addi(rts_idx, one_idx)

        for_op = scf.ForOp(
            loop_start,
            rte_idx,
            one_idx,
            iter_args=[_raw(reg_out), _raw(max_lse), _raw(sum_e_lse)],
        )
        with ir.InsertionPoint(for_op.body):
            split_iv = for_op.induction_variable
            reg_out_v = for_op.inner_iter_args[0]
            max_lse_v = for_op.inner_iter_args[1]
            sum_v = for_op.inner_iter_args[2]

            # Read partial_map_val from LDS (fresh SmemPtr for this scope)
            lds_i32_inner = SmemPtr(lds_buffer, 0, T.i32, shape=(2048,))
            lds_idx = _mlir_arith.subi(split_iv, rts_idx)
            partial_map_val = lds_i32_inner.load([lds_idx])

            # Compute offsets for this split
            split_lse_base = (partial_map_val + local_seq) * num_heads_q
            split_lse_offset = split_lse_base + head_idx
            split_out_base = split_lse_base * size_dv
            split_out_offset = (
                split_out_base + head_idx * size_dv + thread_id_i32 * VEC_WIDTH
            )

            # Load split's output tile and LSE
            oaccu = buffer_ops.buffer_load(
                partial_output_rsrc,
                split_out_offset,
                vec_width=VEC_WIDTH,
                dtype=T.f32,
            )
            new_lse = buffer_ops.buffer_load(
                partial_lse_rsrc,
                split_lse_offset,
                vec_width=1,
                dtype=T.f32,
            )

            # Online softmax: rescale and accumulate
            new_max_lse = _mlir_arith.maximumf(_raw(max_lse_v), _raw(new_lse))
            old_scale = mlir_math.exp(_mlir_arith.subf(_raw(max_lse_v), new_max_lse))
            new_scale = mlir_math.exp(_mlir_arith.subf(_raw(new_lse), new_max_lse))

            # Scale and accumulate (handles both scalar and vector reg_out)
            scaled_old = _scale_accum(old_scale, reg_out_v)
            scaled_new = _scale_accum(new_scale, _raw(oaccu))
            new_reg_out = _mlir_arith.addf(scaled_old, scaled_new)

            new_sum = _mlir_arith.addf(
                _mlir_arith.mulf(_raw(sum_v), old_scale), new_scale
            )

            scf.YieldOp([new_reg_out, new_max_lse, new_sum])

        final_reg_out = for_op.results[0]
        final_max = for_op.results[1]
        final_sum = for_op.results[2]

        # -- Step 5c: Normalize: reg_out /= sum_e_lse --
        inv_sum = _mlir_arith.divf(_raw(arith.constant(1.0)), final_sum)
        normed = _scale_accum(inv_sum, final_reg_out)

        # -- Step 5d: Cast and store to final_output --
        # offset = seq_idx * stride_s_o + head_idx * stride_h_o + thread_id * VEC_WIDTH
        final_out_offset = (
            seq_idx * stride_s_o + head_idx * stride_h_o + thread_id_i32 * VEC_WIDTH
        )
        if out_elem_bytes == 2:
            # f32 -> bf16 truncation
            if use_vec:
                out_type = ir.VectorType.get([VEC_WIDTH], ir.BF16Type.get())
            else:
                out_type = ir.BF16Type.get()
            store_val = _mlir_arith.truncf(out_type, normed)
        else:
            store_val = normed
        buffer_ops.buffer_store(store_val, final_output_rsrc, final_out_offset)

        # -- Step 5e: Output LSE --
        if output_lse:
            log_sum = mlir_math.log(final_sum)
            final_lse_val = _mlir_arith.addf(log_sum, final_max)

            # Handle NaN/zero: if sum==0 or sum!=sum -> infinity
            zero = _raw(arith.constant(0.0))
            is_zero = _mlir_arith.cmpf(_mlir_arith.CmpFPredicate.OEQ, final_sum, zero)
            is_nan = _mlir_arith.cmpf(
                _mlir_arith.CmpFPredicate.UNE, final_sum, final_sum
            )
            is_invalid = _mlir_arith.ori(is_zero, is_nan)
            inf_val = _raw(arith.constant(float("inf")))
            final_lse_val = _mlir_arith.select(is_invalid, inf_val, final_lse_val)

            # Store: offset = seq_idx * num_heads_q + head_idx
            lse_out_offset = seq_idx * num_heads_q + head_idx
            buffer_ops.buffer_store(
                final_lse_val,
                final_lse_rsrc,
                lse_out_offset,
            )


@flyc.kernel
def kn_attn_reduce(
    reduce_indptr: fx.Tensor,
    reduce_final_map: fx.Tensor,
    reduce_partial_map: fx.Tensor,
    final_lse: fx.Tensor,
    final_output: fx.Tensor,
    partial_lse: fx.Tensor,
    partial_output: fx.Tensor,
    stride_s_o: fx.Int32,
    stride_h_o: fx.Int32,
    max_splits: fx.Int32,
    output_lse: fx.Constexpr[int],
    use_reduce_final_map: fx.Constexpr[int],
    num_heads_q: fx.Constexpr[int],
    size_dv: fx.Constexpr[int],
    out_elem_bytes: fx.Constexpr[int],
    num_wg_per_bh: fx.Constexpr[int],
):
    reduce_indptr_rsrc = buffer_ops.create_buffer_resource(reduce_indptr)
    reduce_final_map_rsrc = buffer_ops.create_buffer_resource(reduce_final_map)
    reduce_partial_map_rsrc = buffer_ops.create_buffer_resource(reduce_partial_map)
    final_lse_rsrc = buffer_ops.create_buffer_resource(final_lse)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
    partial_lse_rsrc = buffer_ops.create_buffer_resource(partial_lse)
    partial_output_rsrc = buffer_ops.create_buffer_resource(partial_output)

    head_idx = arith.index_cast(T.i32, gpu.block_id("x"))
    block_idx = arith.index_cast(T.i32, gpu.block_id("y"))
    tile_idx = arith.index_cast(T.i32, gpu.block_id("z"))

    reduce_tile_range = buffer_ops.buffer_load(
        reduce_indptr_rsrc, tile_idx, vec_width=2, dtype=T.i32
    )
    reduce_tile_start = vector.extract(reduce_tile_range, [0])
    reduce_tile_end = vector.extract(reduce_tile_range, [1])
    num_splits = reduce_tile_end - reduce_tile_start

    # config lds buffer
    from flydsl.compiler.kernel_function import CompilationContext

    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)
    lds_allocator.ptr = 65536 // 8  # reserve LDS bytes

    # finalize must be emitted at gpu.module level, not inside gpu.func
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        lds_allocator.finalize()

    _reduce_args = (
        reduce_final_map_rsrc,
        reduce_partial_map_rsrc,
        final_lse_rsrc,
        final_output_rsrc,
        partial_lse_rsrc,
        partial_output_rsrc,
        stride_s_o,
        stride_h_o,
        max_splits,
        output_lse,
        use_reduce_final_map,
        num_heads_q,
        size_dv,
        out_elem_bytes,
        num_wg_per_bh,
        head_idx,
        block_idx,
        tile_idx,
        reduce_tile_start,
        reduce_tile_end,
        lds_allocator,
    )

    if num_splits >= MASSIVE_THRESHOLD:
        if num_splits <= 64:
            attn_reduce_massive(*_reduce_args, local_lse_cls=_LocalLse64)
        elif num_splits <= 256:
            attn_reduce_massive(*_reduce_args, local_lse_cls=_LocalLse256)
        else:
            attn_reduce_massive(*_reduce_args, local_lse_cls=_LocalLseLds)
    elif num_splits > 1:
        attn_reduce_simple(*_reduce_args)


def _ps_tail_loop(work_idx, total_wg_cnt, main_loop_fn):
    """Persistent-scheduling tail loop (module-level to avoid AST rewriter).

    Iterates work_idx from (work_idx + grid_dim) to total_wg_cnt stepping by
    grid_dim.  Carries an i1 flag so that once main_loop_fn returns False
    (sentinel tile), all remaining iterations are skipped.
    """
    work_idx_idx = _to_mlir(arith.index_cast(T.index, _raw(work_idx)))
    total_wg_idx = _to_mlir(arith.index_cast(T.index, _raw(total_wg_cnt)))
    stride = _to_mlir(arith.index_cast(T.index, gpu.grid_dim.x))
    # first_iter = work_idx + grid_dim  (skip the iteration already done)
    first_iter = _mlir_arith.addi(work_idx_idx, stride)

    i1_ty = ir.IntegerType.get_signless(1)
    true_val = _mlir_arith.constant(i1_ty, 1)

    for_op = scf.ForOp(first_iter, total_wg_idx, stride, iter_args=[true_val])
    with ir.InsertionPoint(for_op.body):
        # barrier must be unconditional -- GPU hangs if it's inside scf.IfOp
        gpu.barrier()
        carry = for_op.inner_iter_args[0]
        if_op = scf.IfOp(carry, [i1_ty], has_else=True)
        # then-branch: do real work
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            wi = arith.index_cast(T.i32, for_op.induction_variable)
            flag = main_loop_fn(wi)
            scf.YieldOp([_raw(flag)])
        # else-branch: sentinel was hit, stay False
        with ir.InsertionPoint(if_op.regions[1].blocks[0]):
            scf.YieldOp([_mlir_arith.constant(i1_ty, 0)])
        scf.YieldOp([if_op.results[0]])


@flyc.kernel
def kn_attn_reduce_ps(
    reduce_indptr: fx.Tensor,
    reduce_final_map: fx.Tensor,
    reduce_partial_map: fx.Tensor,
    final_lse: fx.Tensor,
    final_output: fx.Tensor,
    partial_lse: fx.Tensor,
    partial_output: fx.Tensor,
    stride_s_o: fx.Int32,
    stride_h_o: fx.Int32,
    num_reduce_tile: fx.Int32,
    max_splits: fx.Int32,
    output_lse: fx.Constexpr[int],
    use_reduce_final_map: fx.Constexpr[int],
    num_heads_q: fx.Constexpr[int],
    size_dv: fx.Constexpr[int],
    out_elem_bytes: fx.Constexpr[int],
    num_wg_per_bh: fx.Constexpr[int],
):
    reduce_indptr_rsrc = buffer_ops.create_buffer_resource(reduce_indptr)
    reduce_final_map_rsrc = buffer_ops.create_buffer_resource(reduce_final_map)
    reduce_partial_map_rsrc = buffer_ops.create_buffer_resource(reduce_partial_map)
    final_lse_rsrc = buffer_ops.create_buffer_resource(final_lse)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
    partial_lse_rsrc = buffer_ops.create_buffer_resource(partial_lse)
    partial_output_rsrc = buffer_ops.create_buffer_resource(partial_output)

    total_wg_cnt = num_heads_q * num_wg_per_bh * num_reduce_tile
    last_reduce_tile = buffer_ops.buffer_load(
        reduce_indptr_rsrc, num_reduce_tile, vec_width=1, dtype=T.i32
    )

    # config lds buffer
    from flydsl.compiler.kernel_function import CompilationContext

    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)
    lds_allocator.ptr = 65536 // 8  # reserve LDS bytes

    # finalize must be emitted at gpu.module level, not inside gpu.func
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        lds_allocator.finalize()

    def main_loop(work_idx):
        head_idx = work_idx % num_heads_q
        block_idx = (work_idx // num_heads_q) % num_wg_per_bh
        tile_idx = (work_idx // num_heads_q) // num_wg_per_bh
        reduce_tile_range = buffer_ops.buffer_load(
            reduce_indptr_rsrc, tile_idx, vec_width=2, dtype=T.i32
        )
        reduce_tile_start = vector.extract(reduce_tile_range, [0])
        reduce_tile_end = vector.extract(reduce_tile_range, [1])

        continue_reduce = reduce_tile_start != last_reduce_tile

        if_op = scf.IfOp(continue_reduce, [], has_else=False)
        with ir.InsertionPoint(if_op.regions[0].blocks[0]):
            num_splits = reduce_tile_end - reduce_tile_start
            _reduce_args = (
                reduce_final_map_rsrc,
                reduce_partial_map_rsrc,
                final_lse_rsrc,
                final_output_rsrc,
                partial_lse_rsrc,
                partial_output_rsrc,
                stride_s_o,
                stride_h_o,
                max_splits,
                output_lse,
                use_reduce_final_map,
                num_heads_q,
                size_dv,
                out_elem_bytes,
                num_wg_per_bh,
                head_idx,
                block_idx,
                tile_idx,
                reduce_tile_start,
                reduce_tile_end,
                lds_allocator,
            )
            # Manual scf.IfOp: AST rewriter doesn't transform nested functions
            is_massive = _mlir_arith.cmpi(
                _mlir_arith.CmpIPredicate.sge,
                _raw(num_splits),
                _mlir_arith.constant(T.i32, MASSIVE_THRESHOLD),
            )
            if_massive = scf.IfOp(is_massive, [], has_else=True)
            with ir.InsertionPoint(if_massive.regions[0].blocks[0]):
                # massive path: pick local_lse_cls by num_splits
                is_le64 = _mlir_arith.cmpi(
                    _mlir_arith.CmpIPredicate.sle,
                    _raw(num_splits),
                    _mlir_arith.constant(T.i32, 64),
                )
                if_le64 = scf.IfOp(is_le64, [], has_else=True)
                with ir.InsertionPoint(if_le64.regions[0].blocks[0]):
                    attn_reduce_massive(*_reduce_args, local_lse_cls=_LocalLse64)
                    scf.YieldOp([])
                with ir.InsertionPoint(if_le64.regions[1].blocks[0]):
                    is_le256 = _mlir_arith.cmpi(
                        _mlir_arith.CmpIPredicate.sle,
                        _raw(num_splits),
                        _mlir_arith.constant(T.i32, 256),
                    )
                    if_le256 = scf.IfOp(is_le256, [], has_else=True)
                    with ir.InsertionPoint(if_le256.regions[0].blocks[0]):
                        attn_reduce_massive(*_reduce_args, local_lse_cls=_LocalLse256)
                        scf.YieldOp([])
                    with ir.InsertionPoint(if_le256.regions[1].blocks[0]):
                        attn_reduce_massive(*_reduce_args, local_lse_cls=_LocalLseLds)
                        scf.YieldOp([])
                    scf.YieldOp([])
                scf.YieldOp([])
            with ir.InsertionPoint(if_massive.regions[1].blocks[0]):
                # simple path: num_splits > 1
                is_gt1 = _mlir_arith.cmpi(
                    _mlir_arith.CmpIPredicate.sgt,
                    _raw(num_splits),
                    _mlir_arith.constant(T.i32, 1),
                )
                if_gt1 = scf.IfOp(is_gt1, [], has_else=False)
                with ir.InsertionPoint(if_gt1.regions[0].blocks[0]):
                    attn_reduce_simple(*_reduce_args)
                    scf.YieldOp([])
                scf.YieldOp([])
            scf.YieldOp([])

        return continue_reduce

    work_idx = arith.index_cast(T.i32, gpu.block_id("x"))
    if work_idx < total_wg_cnt:
        continue_flag = main_loop(work_idx)
        if continue_flag:
            # -- while loop replaced with scf.ForOp (FlyDSL while-loop issue) --
            # work_idx += gpu.grid_dim.x
            # while work_idx < total_wg_cnt:
            #     gpu.barrier()
            #     continue_flag = main_loop(work_idx)
            #     if not continue_flag:
            #         break
            #     work_idx += gpu.grid_dim.x
            _ps_tail_loop(work_idx, total_wg_cnt, main_loop)


@flyc.jit
def launch_attn_reduce(
    reduce_indptr: fx.Tensor,
    reduce_final_map: fx.Tensor,
    reduce_partial_map: fx.Tensor,
    final_lse: fx.Tensor,
    final_output: fx.Tensor,
    partial_lse: fx.Tensor,
    partial_output: fx.Tensor,
    stride_s_o: fx.Int32,
    stride_h_o: fx.Int32,
    num_reduce_tile: fx.Int32,
    lds_size: fx.Int32,
    max_splits: fx.Int32,
    num_wg_per_bh: fx.Constexpr[int],
    output_lse: fx.Constexpr[int],
    use_reduce_final_map: fx.Constexpr[int],
    num_heads_q: fx.Constexpr[int],
    size_dv: fx.Constexpr[int],
    out_elem_bytes: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    kn_attn_reduce(
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_lse,
        final_output,
        partial_lse,
        partial_output,
        stride_s_o,
        stride_h_o,
        max_splits,
        output_lse,
        use_reduce_final_map,
        num_heads_q,
        size_dv,
        out_elem_bytes,
        num_wg_per_bh,
    ).launch(
        grid=(num_heads_q, num_wg_per_bh, num_reduce_tile),
        block=(NUM_THREADS, 1, 1),
        # smem=lds_size,
        stream=stream,
    )


@flyc.jit
def launch_attn_reduce_ps(
    reduce_indptr: fx.Tensor,
    reduce_final_map: fx.Tensor,
    reduce_partial_map: fx.Tensor,
    final_lse: fx.Tensor,
    final_output: fx.Tensor,
    partial_lse: fx.Tensor,
    partial_output: fx.Tensor,
    stride_s_o: fx.Int32,
    stride_h_o: fx.Int32,
    num_reduce_tile: fx.Int32,
    ps_grid_size: fx.Int32,
    lds_size: fx.Int32,
    max_splits: fx.Int32,
    num_wg_per_bh: fx.Constexpr[int],
    output_lse: fx.Constexpr[int],
    use_reduce_final_map: fx.Constexpr[int],
    num_heads_q: fx.Constexpr[int],
    size_dv: fx.Constexpr[int],
    out_elem_bytes: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    kn_attn_reduce_ps(
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        final_lse,
        final_output,
        partial_lse,
        partial_output,
        stride_s_o,
        stride_h_o,
        num_reduce_tile,
        max_splits,
        output_lse,
        use_reduce_final_map,
        num_heads_q,
        size_dv,
        out_elem_bytes,
        num_wg_per_bh,
    ).launch(
        grid=(ps_grid_size, 1, 1),
        block=(NUM_THREADS, 1, 1),
        # smem=lds_size,
        stream=stream,
    )
