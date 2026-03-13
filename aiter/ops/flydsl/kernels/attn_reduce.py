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
from flydsl._mlir.dialects import math as mlir_math
from flydsl._mlir.dialects import vector as _mlir_vector
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.expr.arith import _to_raw as _raw

WARP_SIZE = 64  # warp_size = 64 on AMD MI3xx GPU
NUM_WARPS = 2
NUM_THREADS = NUM_WARPS * WARP_SIZE
OCCUPANCY = 8
MASSIVE_THRESHOLD = 4


def _to_index(val):
    """Convert a Python int, ArithValue, or ir.Value to a raw MLIR index Value."""
    if isinstance(val, int):
        return _raw(arith.constant(val, index=True))
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
    for_op = scf.ForOp(_to_index(start), _to_index(stop), _to_index(step))
    with ir.InsertionPoint(for_op.body):
        yield for_op.induction_variable
        scf.YieldOp([])


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
):
    # Derived constants (Python-level, becomes Constexpr)
    VEC_WIDTH = size_dv // NUM_THREADS

    # config lds buffer
    from flydsl.compiler.kernel_function import CompilationContext

    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)
    lds_allocator.ptr = 65536 // 8  # reserve LDS bytes

    # finalize must be emitted at gpu.module level, not inside gpu.func
    ctx = CompilationContext.get_current()
    with ir.InsertionPoint(ctx.gpu_module_body):
        lds_allocator.finalize()

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
    q_start_idx = _to_index(arith.index_cast(T.index, _raw(q_start + block_idx)))
    q_end_idx = _to_index(arith.index_cast(T.index, _raw(q_end)))
    num_wg_idx = _to_index(num_wg_per_bh)

    f32_ty = ir.F32Type.get()
    use_vec = VEC_WIDTH > 1
    if use_vec:
        vec_type = ir.VectorType.get([VEC_WIDTH], f32_ty)
    one_idx = _to_index(arith.constant(1, index=True))

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
        rts_idx = _to_index(arith.index_cast(T.index, _raw(reduce_tile_start)))
        rte_idx = _to_index(arith.index_cast(T.index, _raw(reduce_tile_end)))
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

    # if num_splits >= MASSIVE_THRESHOLD:
    if False:
        pass
    elif num_splits > 1:
        attn_reduce_simple(
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
        )


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

    def main_loop(work_idx):
        head_idx = work_idx % num_heads_q
        block_idx = (work_idx // num_heads_q) % num_wg_per_bh
        tile_idx = (work_idx // num_heads_q) // num_wg_per_bh
        reduce_tile_range = buffer_ops.buffer_load(
            reduce_indptr_rsrc, tile_idx, vec_width=2, dtype=T.i32
        )
        reduce_tile_start = vector.extract(reduce_tile_range, [0])
        reduce_tile_end = vector.extract(reduce_tile_range, [1])

        ret = reduce_tile_start != last_reduce_tile
        if ret:
            num_splits = reduce_tile_end - reduce_tile_start
            # if num_splits >= MASSIVE_THRESHOLD:
            if False:
                pass
            elif num_splits > 1:
                attn_reduce_simple(
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
                )
        return ret

    work_idx = arith.index_cast(T.i32, gpu.block_id("x"))
    if work_idx < total_wg_cnt:
        continue_flag = main_loop(work_idx)
        if continue_flag:
            work_idx += gpu.grid_dim.x
            while work_idx < total_wg_cnt:
                gpu.barrier()
                continue_flag = main_loop(work_idx)
                if not continue_flag:
                    break
                work_idx += gpu.grid_dim.x


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
