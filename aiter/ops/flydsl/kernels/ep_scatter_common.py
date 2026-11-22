# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""TDM EP-scatter emission and persistent GEMM overlap callbacks.

The GEMM owns LDS allocation and input/compute operations. These factories
bind scatter geometry and pending writes at trace time; their callbacks emit
inline code, with no runtime plugin dispatch.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

from .gemm_common_gfx1250 import workgroup_barrier
from .mega_moe_gfx1250.tdm_gather_shim import (
    make_tensor_gather_descriptor,
    tensor_store_gather,
)


def make_scatter_store(
    lsa0,
    lsa1,
    input_offset,
    slot_stride_bytes,
    world_size,
    c_lds_ptr,
    tile_m,
    row_elems,
    lds_row_elems,
    output_type,
    elem_bytes,
    group_rows,
    overlap_scatter,
):
    """Bind the symmetric combine arena and return (emitter, rank rows, OOB).

    A peer/slot pair folds into one row index, peer * rank_rows + slot.
    Quantized rows include their scales and are addressed in bytes. Ordinary
    and overlapped scatter share descriptors; the latter's callbacks replace
    the shim's implicit wait with explicit burst drains.
    """
    rank_rows = fx.Int32((lsa1 - lsa0) // fx.Int64(slot_stride_bytes))
    # Hardware drops padding rows at this exclusive upper bound.
    oob = rank_rows * fx.Int32(world_size)
    stride_elems = slot_stride_bytes // elem_bytes
    ptr_type = fx.PointerType.get(
        T.i8 if elem_bytes == 1 else T.i16,
        address_space=fx.AddressSpace.Global,
        alignment=16,
    )
    combine_ptr = fx.inttoptr(ptr_type, lsa0 + fx.Int64(input_offset))
    combine_view = fx.Tensor(
        fx.make_view(combine_ptr, fx.make_layout((oob, row_elems), (stride_elems, 1)))
    )
    lds_c = fx.Tensor(
        fx.make_view(
            fx.recast_iter(output_type, c_lds_ptr),
            fx.make_layout((tile_m, lds_row_elems), (lds_row_elems, 1)),
        )
    )

    def issue_scatter_group(group, index_lane, vector_pass, column_offset):
        base_row = group * fx.Int32(group_rows)
        row_indices = []
        for i in range_constexpr(group_rows):
            src_lane = fx.Int32(vector_pass * group_rows + i)
            row_indices.append(
                fx.Int32(
                    rocdl.readlane(
                        T.i32,
                        _raw(index_lane),
                        _raw(src_lane),
                    )
                )
            )
        # Geometry is explicit because traced layout leaves are not
        # Python integers suitable for the descriptor bitfields.
        desc = make_tensor_gather_descriptor(
            combine_view,
            lds_c,
            row_indices,
            row_width=lds_row_elems,
            tensor_dim0=row_elems,
            tensor_dim1=oob.ir_value(),
            stride=stride_elems,
            elem_bytes=elem_bytes,
            index_size=32,
            lds_byte_offset=fx.index_cast(
                T.index,
                base_row * fx.Int32(lds_row_elems * elem_bytes),
            ),
            global_byte_offset=column_offset,
        )
        if const_expr(overlap_scatter):
            # tensor_store_gather emits redundant tensorcnt(0xa).
            # Store-only waves issue at most eight stores between
            # explicit drains; inline asm keeps that burst wait-free.
            llvm_dialect.inline_asm(
                None,
                [desc.dgroup0, desc.dgroup1, desc.dgroup2, desc.dgroup3],
                "tensor_store_from_lds $0, $1, $2, $3\n" "s_nop 0",
                "s,s,s,s,~{memory}",
                has_side_effects=True,
            )
        else:
            tensor_store_gather(desc)

    return issue_scatter_group, rank_rows, oob


def make_overlap_callbacks(
    issue_scatter_group,
    wave,
    scatter_wave,
    pending,
    last_task,
    issue,
    init_accumulators,
    load_wait,
    tdm_per,
    load_first,
):
    """Bind the two-store-wave, 8+8 gather schedule to GEMM operations.

    Returns prologue, epilogue and skipped-task callbacks. ``pending`` holds
    the previous task's two tail index vectors, column offset and valid flag.
    The lo/hi vectors cover C rows 128:192 and 192:256 respectively. Each lane
    holds one flattened peer/slot index (or OOB); each store wave contributes
    32 indices per vector. The output data stays in LDS until the tail retires.
    The epilogue returns the same four values for the next task; it never
    mutates captured state. ``load_first`` loads the first register tile and
    preserves GEMM's scheduling fence.
    """
    pending_tail_index_lo, pending_tail_index_hi, pending_col, pending_valid = pending

    def issue_scatter_burst(first, idx0, idx1, column_offset):
        for i in range_constexpr(8):
            issue_scatter_group(
                scatter_wave + fx.Int32((first + i) * 2),
                idx0 if i < 4 else idx1,
                i % 4,
                column_offset,
            )

    @flyc.jit
    def overlap_prologue(interleaved_lds_load):
        # Stages 0/1 are disjoint from old C. One rendezvous covers both
        # the first input's readiness and retirement of C's first half.
        for i in range_constexpr(2):
            issue(i, i)
        init_accumulators()
        load_wait(tdm_per)
        if (wave >= 2) & (pending_valid != 0):
            tdm_ops.tensor_wait(0)
        workgroup_barrier()
        load_first(interleaved_lds_load)
        if (wave >= 2) & (pending_valid != 0):
            issue_scatter_burst(
                8, pending_tail_index_lo, pending_tail_index_hi, pending_col
            )
        issue(2, 2)

    @flyc.jit
    def overlap_epilogue(index_lanes, column_offset):
        next_tail_index_lo = fx.Int32(0)
        next_tail_index_hi = fx.Int32(0)
        next_col = fx.Int32(0)
        next_valid = fx.Int32(0)
        if wave >= 2:
            issue_scatter_burst(0, index_lanes[0], index_lanes[1], column_offset)
            next_tail_index_lo = index_lanes[2]
            next_tail_index_hi = index_lanes[3]
            next_col = fx.Int32(column_offset)
            next_valid = fx.Int32(1)
            if last_task != 0:
                # No following prologue will issue the last task's tail.
                tdm_ops.tensor_wait(0)
                issue_scatter_burst(8, index_lanes[2], index_lanes[3], column_offset)
                next_valid = fx.Int32(0)
            rocdl.s_nop(0)
        return next_tail_index_lo, next_tail_index_hi, next_col, next_valid

    @flyc.jit
    def skip_task():
        # A partial M-cluster tail has no prologue to issue the deferred burst.
        if (wave >= 2) & (pending_valid != 0):
            tdm_ops.tensor_wait(0)
            issue_scatter_burst(
                8, pending_tail_index_lo, pending_tail_index_hi, pending_col
            )

    return overlap_prologue, overlap_epilogue, skip_task


@flyc.jit
def retire_scatter():
    """Retire the previous C's stores before any wave overwrites its LDS."""
    tdm_ops.tensor_wait(0)
    workgroup_barrier()
