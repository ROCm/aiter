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

# Waves 0/1 load inputs; waves 2/3 each scatter 16 eight-row groups.
# The default issues eight groups in the epilogue and defers eight to the
# next task's mainloop. Tuning may defer nine or ten groups instead.
WAVE_SIZE = 32
ROWS_PER_GATHER = 8
GATHERS_PER_INDEX_VECTOR = WAVE_SIZE // ROWS_PER_GATHER
LOAD_WAVES = (0, 1)
STORE_WAVES = (2, 3)
STORE_WAVE_COUNT = len(STORE_WAVES)
TDM_STORE_WAVE_START = 2


def scatter_schedule(prologue_gathers=8):
    """Return the supported per-wave (prologue, mainloop) gather counts."""
    if prologue_gathers not in (6, 7, 8):
        raise ValueError("scatter prologue gathers must be 6, 7, or 8")
    return (
        (0, 0),
        (0, 0),
        (prologue_gathers, 16 - prologue_gathers),
        (prologue_gathers, 16 - prologue_gathers),
    )


def overlap_output_offset(input_end, stage2_start, row_bytes, prologue_gathers):
    """Keep pending C rows beyond all three input stages, aligned to 512B."""
    scatter_schedule(prologue_gathers)
    retired_bytes = prologue_gathers * STORE_WAVE_COUNT * ROWS_PER_GATHER * row_bytes
    offset = max(stage2_start, input_end - retired_bytes)
    return (offset + 511) // 512 * 512


def can_overlap_scatter(
    *,
    enable_ep_scatter,
    ep_world_size,
    ep_quant_bits,
    a_is_fp4,
    a_preshuffled,
    tile_m,
    tile_n,
    tile_k,
    m_warp,
    n_warp,
    next_stage_prefetch,
    tdm_as_in_prologue,
    k_waves,
    k_tiles,
):
    """Check layout eligibility before choosing workers and LDS buffers.

    The host launcher and kernel builder share this gate so the host only
    defaults scheduling parameters when the kernel can use the overlap ABI.
    """
    return bool(
        enable_ep_scatter
        and ep_world_size > 1
        and not ep_quant_bits
        and a_is_fp4
        and a_preshuffled
        and tile_m == tile_n == tile_k == 256
        and m_warp == n_warp == 2
        and next_stage_prefetch
        and not tdm_as_in_prologue
        and k_waves == 1
        and k_tiles >= 3
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
        # most ten stores between explicit drains, so this guard does not
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
    tdm_wait_per_wave,
    tdm_per,
    load_first,
    prologue_gathers=8,
):
    """Overlap each store wave's two bursts with the following GEMM task.

    The state carries up to three destination vectors, an offset and validity.
    Six/seven-gather first bursts end inside a vector; their tails retain that
    vector and start at its original lane offset. The default eight-gather
    first burst needs only two vectors; its third state value stays zero.
    Stages 0/1 are disjoint from C. Stage 2 may overwrite only the retired
    prefix. The next epilogue waits for the deferred tail before reusing C.
    """
    scatter_schedule(prologue_gathers)
    mainloop_gathers = 16 - prologue_gathers
    tail_first_vector = prologue_gathers // GATHERS_PER_INDEX_VECTOR
    tail_vectors = 4 - tail_first_vector
    (
        previous_task_scatter_dst_index_low,
        previous_task_scatter_dst_index_high,
        previous_task_scatter_dst_index_extra,
        scatter_column_offset,
        scatter_valid,
    ) = scatter_dst_state

    def issue_scatter_burst(first, count, index_lanes, column_offset):
        for i in range_constexpr(count):
            issue_scatter_group(
                scatter_wave + fx.Int32((first + i) * STORE_WAVE_COUNT),
                index_lanes[
                    (first % GATHERS_PER_INDEX_VECTOR + i) // GATHERS_PER_INDEX_VECTOR
                ],
                (first + i) % GATHERS_PER_INDEX_VECTOR,
                column_offset,
            )

    @flyc.jit
    def overlap_prologue(interleaved_lds_load):
        # Stages 0/1 are disjoint from old C. One rendezvous covers both
        # the first input's readiness and retirement of C's first burst.
        for i in range_constexpr(2):
            issue(i, i)
        init_accumulators()
        tdm_wait_per_wave(tdm_per)
        if (wave >= TDM_STORE_WAVE_START) & (scatter_valid != 0):
            tdm_ops.tensor_wait(0)
        workgroup_barrier()
        load_first(interleaved_lds_load)
        if (wave >= TDM_STORE_WAVE_START) & (scatter_valid != 0):
            issue_scatter_burst(
                prologue_gathers,
                mainloop_gathers,
                (
                    previous_task_scatter_dst_index_low,
                    previous_task_scatter_dst_index_high,
                    previous_task_scatter_dst_index_extra,
                ),
                scatter_column_offset,
            )
        issue(2, 2)

    @flyc.jit
    def overlap_epilogue(index_lanes, column_offset):
        next_scatter_dst_index_low = fx.Int32(0)
        next_scatter_dst_index_high = fx.Int32(0)
        next_scatter_dst_index_extra = fx.Int32(0)
        next_scatter_column_offset = fx.Int32(0)
        next_scatter_valid = fx.Int32(0)
        if wave >= TDM_STORE_WAVE_START:
            issue_scatter_burst(0, prologue_gathers, index_lanes, column_offset)
            if const_expr(mainloop_gathers > 0):
                next_scatter_dst_index_low = index_lanes[tail_first_vector]
            if const_expr(mainloop_gathers > GATHERS_PER_INDEX_VECTOR):
                next_scatter_dst_index_high = index_lanes[tail_first_vector + 1]
            if const_expr(tail_vectors > 2):
                next_scatter_dst_index_extra = index_lanes[tail_first_vector + 2]
            next_scatter_column_offset = fx.Int32(column_offset)
            next_scatter_valid = fx.Int32(1)
            if last_task != 0:
                # No following prologue will issue the last task's tail.
                tdm_ops.tensor_wait(0)
                issue_scatter_burst(
                    prologue_gathers,
                    mainloop_gathers,
                    (
                        next_scatter_dst_index_low,
                        next_scatter_dst_index_high,
                        next_scatter_dst_index_extra,
                    ),
                    column_offset,
                )
                next_scatter_valid = fx.Int32(0)
            rocdl.s_nop(0)
        return (
            next_scatter_dst_index_low,
            next_scatter_dst_index_high,
            next_scatter_dst_index_extra,
            next_scatter_column_offset,
            next_scatter_valid,
        )

    @flyc.jit
    def skip_task():
        # A partial M-cluster tail has no prologue to issue the deferred burst.
        if (wave >= TDM_STORE_WAVE_START) & (scatter_valid != 0):
            tdm_ops.tensor_wait(0)
            issue_scatter_burst(
                prologue_gathers,
                mainloop_gathers,
                (
                    previous_task_scatter_dst_index_low,
                    previous_task_scatter_dst_index_high,
                    previous_task_scatter_dst_index_extra,
                ),
                scatter_column_offset,
            )

    return overlap_prologue, overlap_epilogue, skip_task


@flyc.jit
def retire_scatter():
    """Retire the previous C's stores before any wave overwrites its LDS."""
    tdm_ops.tensor_wait(0)
    workgroup_barrier()
