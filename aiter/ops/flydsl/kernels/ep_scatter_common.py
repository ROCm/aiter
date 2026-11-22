# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors

"""TDM EP-scatter emission and persistent GEMM overlap callbacks.

The GEMM owns LDS allocation and input/compute operations. These factories
bind scatter geometry and destination indices at trace time. Their callbacks
emit inline code, with no runtime plugin dispatch.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl, tdm_ops
from flydsl.expr.arith import _to_raw as _raw
from flydsl.expr.typing import T

from .gemm_common_gfx1250 import workgroup_barrier
from .mega_moe_gfx1250.tdm_gather_shim import (
    make_tensor_gather_descriptor,
    tensor_store_gather,
)

# Per wave: (m: overlaps next prologue, n: overlaps next mainloop).
# Both the GEMM's wave ownership and the callbacks use this table.
# The m indices are consumed in the current epilogue. The n indices cross the
# task boundary as previous_task_scatter_dst_index_low/high. Each wave-wide
# index vector supplies GATHERS_PER_INDEX_VECTOR gathers (four for wave32
# / eight-row gather).
SCATTER_SCHEDULE = (
    (0, 0),  # wave 0: load
    (0, 0),  # wave 1: load
    (8, 8),  # wave 2: store
    (8, 8),  # wave 3: store
)
WAVE_SIZE = 32
ROWS_PER_GATHER = 8
GATHERS_PER_INDEX_VECTOR = WAVE_SIZE // ROWS_PER_GATHER
STORE_WAVES = tuple(w for w, counts in enumerate(SCATTER_SCHEDULE) if any(counts))
LOAD_WAVES = tuple(w for w, counts in enumerate(SCATTER_SCHEDULE) if not any(counts))
STORE_WAVE_COUNT = len(STORE_WAVES)
FIRST_STORE_WAVE = STORE_WAVES[0]
PROLOGUE_GATHERS, MAINLOOP_GATHERS = SCATTER_SCHEDULE[FIRST_STORE_WAVE]
TAIL_FIRST_VECTOR = PROLOGUE_GATHERS // GATHERS_PER_INDEX_VECTOR

# The current row mapping interleaves equally sized bursts across a contiguous
# suffix of store waves. The persistent task ABI carries two tail index vectors.
assert len(LOAD_WAVES) == 2
assert STORE_WAVES == tuple(range(FIRST_STORE_WAVE, len(SCATTER_SCHEDULE)))
assert all(
    counts == (PROLOGUE_GATHERS, MAINLOOP_GATHERS)
    for wave, counts in enumerate(SCATTER_SCHEDULE)
    if wave in STORE_WAVES
)
assert PROLOGUE_GATHERS > 0 and PROLOGUE_GATHERS % GATHERS_PER_INDEX_VECTOR == 0
assert 0 <= MAINLOOP_GATHERS <= 2 * GATHERS_PER_INDEX_VECTOR


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
):
    """Bind the symmetric combine arena and return (emitter, rank rows, OOB).

    A peer/slot pair folds into one row index, peer * rank_rows + slot.
    Quantized rows include their scales and are addressed in bytes. Ordinary
    and overlapped scatter use the same descriptor and store intrinsic.
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
        # This intrinsic emits the gather scatter itself; gather mode and the
        # destination rows are encoded in desc, not in the instruction name.
        # Keep the compiler-inserted tensorcnt(0xa) hardware workaround: it
        # limits outstanding tensor operations to 10, rather than draining all
        # stores. In the overlap path, waves 2/3 issue no tensor loads and at
        # most eight stores between explicit drains, so this guard does not
        # require the burst to finish. The callbacks still use tensor_wait(0)
        # at LDS reuse boundaries; the compiler guard cannot replace those
        # completion waits. Ordinary and overlapped scatter share this emitter.
        tensor_store_gather(desc)

    return issue_scatter_group, rank_rows, oob


def make_overlap_callbacks(
    issue_scatter_group,
    wave,
    scatter_wave,
    scatter_dst_state,
    last_task,
    issue,
    init_accumulators,
    load_wait,
    tdm_per,
    load_first,
):
    """Bind SCATTER_SCHEDULE to GEMM operations.

    SCATTER_SCHEDULE is the source of the per-wave burst counts. Each gather
    consumes ROWS_PER_GATHER lane indices; a full index vector spans WAVE_SIZE
    lanes, so it supplies GATHERS_PER_INDEX_VECTOR gathers.

    Task t's epilogue issues m. Task t+1's prologue waits for m after issuing
    input stages 0/1, then uses one barrier before reusing C's head. It issues
    n before input stage 2; n must retire before task t+1 overwrites C in its
    epilogue. The final task issues both bursts without a following prologue.

    Returns prologue, epilogue and skipped-task callbacks. ``scatter_dst_state``
    holds the previous task's two destination index vectors, column offset and
    valid flag.
    Both index vectors belong to the table's n column: low supplies its first
    GATHERS_PER_INDEX_VECTOR gathers, high the next group. The suffixes refer
    to C row ranges, not address bits. With the default schedule, low covers
    C rows 128:192 and high covers 192:256 across the store waves. Each lane
    holds one flattened peer/slot index (or OOB), not a pointer. These indices
    must survive the task boundary because the next task may target different
    rows. The C data stays in LDS.
    The epilogue returns the same four values for the next task; it never
    mutates captured state. ``load_first`` loads the first register tile and
    preserves GEMM's scheduling fence.
    """
    (
        previous_task_scatter_dst_index_low,
        previous_task_scatter_dst_index_high,
        scatter_column_offset,
        scatter_valid,
    ) = scatter_dst_state

    def issue_scatter_burst(first, count, index_lanes, column_offset):
        for i in range_constexpr(count):
            issue_scatter_group(
                scatter_wave + fx.Int32((first + i) * STORE_WAVE_COUNT),
                index_lanes[i // GATHERS_PER_INDEX_VECTOR],
                i % GATHERS_PER_INDEX_VECTOR,
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
        if (wave >= FIRST_STORE_WAVE) & (scatter_valid != 0):
            tdm_ops.tensor_wait(0)
        workgroup_barrier()
        load_first(interleaved_lds_load)
        if (wave >= FIRST_STORE_WAVE) & (scatter_valid != 0):
            issue_scatter_burst(
                PROLOGUE_GATHERS,
                MAINLOOP_GATHERS,
                (
                    previous_task_scatter_dst_index_low,
                    previous_task_scatter_dst_index_high,
                ),
                scatter_column_offset,
            )
        issue(2, 2)

    @flyc.jit
    def overlap_epilogue(index_lanes, column_offset):
        next_scatter_dst_index_low = fx.Int32(0)
        next_scatter_dst_index_high = fx.Int32(0)
        next_scatter_column_offset = fx.Int32(0)
        next_scatter_valid = fx.Int32(0)
        if wave >= FIRST_STORE_WAVE:
            issue_scatter_burst(0, PROLOGUE_GATHERS, index_lanes, column_offset)
            if const_expr(MAINLOOP_GATHERS > 0):
                next_scatter_dst_index_low = index_lanes[TAIL_FIRST_VECTOR]
            if const_expr(MAINLOOP_GATHERS > GATHERS_PER_INDEX_VECTOR):
                next_scatter_dst_index_high = index_lanes[TAIL_FIRST_VECTOR + 1]
            next_scatter_column_offset = fx.Int32(column_offset)
            next_scatter_valid = fx.Int32(1)
            if last_task != 0:
                # No following prologue will issue the last task's tail.
                tdm_ops.tensor_wait(0)
                issue_scatter_burst(
                    PROLOGUE_GATHERS,
                    MAINLOOP_GATHERS,
                    (next_scatter_dst_index_low, next_scatter_dst_index_high),
                    column_offset,
                )
                next_scatter_valid = fx.Int32(0)
            rocdl.s_nop(0)
        return (
            next_scatter_dst_index_low,
            next_scatter_dst_index_high,
            next_scatter_column_offset,
            next_scatter_valid,
        )

    @flyc.jit
    def skip_task():
        # A partial M-cluster tail has no prologue to issue the deferred burst.
        if (wave >= FIRST_STORE_WAVE) & (scatter_valid != 0):
            tdm_ops.tensor_wait(0)
            issue_scatter_burst(
                PROLOGUE_GATHERS,
                MAINLOOP_GATHERS,
                (
                    previous_task_scatter_dst_index_low,
                    previous_task_scatter_dst_index_high,
                ),
                scatter_column_offset,
            )

    return overlap_prologue, overlap_epilogue, skip_task


@flyc.jit
def retire_scatter():
    """Retire the previous C's stores before any wave overwrites its LDS."""
    tdm_ops.tensor_wait(0)
    workgroup_barrier()
