# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL attention reduce kernel: kernel definitions and launchers."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx

from flydsl.expr import gpu, buffer_ops, vector, rocdl, arith
from flydsl.expr.primitive import printf
from flydsl.expr.typing import T

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf, memref as memref_dialect
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
from flydsl.expr.arith import _to_raw as _raw

NUM_WARPS = 2
NUM_THREADS = NUM_WARPS * 64  # warp_size = 64 on AMD
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
    stride_s_o: fx.Int32,
    stride_h_o: fx.Int32,
    max_splits: fx.Int32,
    output_lse: fx.Constexpr[int],
    use_reduce_final_map: fx.Constexpr[int],
    head_idx: fx.Int32,
    block_idx: fx.Int32,
    tile_idx: fx.Int32,
    reduce_tile_start: fx.Int32,
    reduce_tile_end: fx.Int32,
):
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

    # Extract LDS base address as !llvm.ptr<3>
    lds_addr = memref_dialect.extract_aligned_pointer_as_index(lds_buffer)
    lds_ptr_type = ir.Type.parse("!llvm.ptr<3>")

    num_splits = reduce_tile_end - reduce_tile_start
    num_splits_idx = arith.index_cast(T.index, _raw(num_splits))

    for i in _scf_for(0, num_splits_idx, NUM_THREADS):
        size_i32 = arith.constant(4, type=T.i32)
        soffset = arith.constant(0, type=T.i32)
        offset_imm = arith.constant(0, type=T.i32)
        aux = arith.constant(1, type=T.i32)

        global_offset = arith.index_cast(T.i32, (i * NUM_THREADS + thread_id) * 4)
        lds_ptr_i64 = rocdl.readfirstlane(
            T.i64, arith.index_cast(T.i64, lds_addr + i * NUM_THREADS * 4)
        )
        lds_ptr = llvm.inttoptr(lds_ptr_type, lds_ptr_i64)
        rocdl.raw_ptr_buffer_load_lds(
            reduce_partial_map_rsrc,
            lds_ptr,
            _raw(size_i32),
            _raw(global_offset),
            _raw(soffset),
            _raw(offset_imm),
            _raw(aux),
        )
    gpu.barrier()

    # Debug: print all LDS elements (head_idx==3, thread 0 only)
    cond_t0 = arith.cmpi(
        arith.CmpIPredicate.eq, thread_id, arith.constant(0, index=True)
    )
    # cond_h3 = arith.cmpi(arith.CmpIPredicate.eq, head_idx, arith.constant(3, index=True))
    # cond = arith.andi(_raw(cond_t0), _raw(cond_h3))
    if_op = scf.IfOp(cond_t0, [], has_else=False)
    with ir.InsertionPoint(if_op.regions[0].blocks[0]):
        lds_i32 = SmemPtr(lds_buffer, 0, T.i32, shape=(2048,))
        for idx in _scf_for(0, num_splits_idx, 1):
            val = lds_i32.load([idx])
            printf(
                "hid={}, lds[{}/{}] = {}",
                head_idx,
                arith.index_cast(T.i32, idx),
                num_splits_idx,
                val,
            )
        scf.YieldOp([])


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
):
    head_idx = gpu.block_id("x")
    block_idx = gpu.block_id("y")
    tile_idx = gpu.block_id("z")

    reduce_indptr_rsrc = buffer_ops.create_buffer_resource(reduce_indptr)
    reduce_final_map_rsrc = buffer_ops.create_buffer_resource(reduce_final_map)
    reduce_partial_map_rsrc = buffer_ops.create_buffer_resource(reduce_partial_map)
    final_lse_rsrc = buffer_ops.create_buffer_resource(final_lse)
    final_output_rsrc = buffer_ops.create_buffer_resource(final_output)
    partial_lse_rsrc = buffer_ops.create_buffer_resource(partial_lse)
    partial_output_rsrc = buffer_ops.create_buffer_resource(partial_output)

    reduce_tile_range = buffer_ops.buffer_load(
        reduce_indptr_rsrc, tile_idx, vec_width=2, dtype=T.i32
    )
    reduce_tile_start = vector.extract(reduce_tile_range, [0])
    reduce_tile_end = vector.extract(reduce_tile_range, [1])
    num_splits = reduce_tile_end - reduce_tile_start

    # if num_splits >= MASSIVE_THRESHOLD:
    if False:
        pass
    else:
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
):
    # Empty kernel body -- persistent scheduling variant, to be implemented later.
    pass


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
    num_heads: fx.Int32,
    num_wg_per_bh: fx.Int32,
    lds_size: fx.Int32,
    max_splits: fx.Int32,
    output_lse: fx.Constexpr[int],
    use_reduce_final_map: fx.Constexpr[int],
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
    ).launch(
        grid=(num_heads, num_wg_per_bh, num_reduce_tile),
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
    output_lse: fx.Constexpr[int],
    use_reduce_final_map: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):
    arch = get_hip_arch()
    lds_allocator = SmemAllocator(None, arch=arch)

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
    ).launch(
        grid=(ps_grid_size, 1, 1),
        block=(NUM_THREADS, 1, 1),
        # smem=lds_size,
        stream=stream,
    )
