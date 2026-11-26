# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Isolated gfx1250 MegaMoE payload transfers, without routing or GEMMs."""

from __future__ import annotations

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.cco.device.flydsl as cco
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, rocdl
from flydsl.expr.rocdl import tdm_ops

from aiter.ops.flydsl.kernels.mega_moe_gfx1250.s1f_stage1 import (
    SCOPE_SYS,
    _now,
    _store_i64_sys,
)
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.tdm_gather_shim import (
    TDMGatherDescriptor,
    make_tensor_gather_descriptor,
    tensor_store_gather,
)
from aiter.ops.flydsl.kernels.mega_moe_gfx1250.tdm_prims import (
    tdm_group0,
    tdm_group1,
    tdm_store,
)

ROWS = 8
SCATTER_COLUMNS = 256
SCATTER_LDS_COLUMNS = 264
SCATTER_FIRST_SLOT = 3
SCATTER_SLOT_STEP = 17
SCATTER_COLUMN_OFFSET = 256
DISPATCH_FIRST_ROW = 16
BLOCK_THREADS = 128


def _timestamp():
    # FlyDSL has no compiler-only memory barrier or typed steady-counter API.
    # Preserve the issue/wait boundaries around the existing _now() intrinsic;
    # the input dependency also requires its returned SGPR pair to be available.
    # MI400 Shader Programming Guide 3.4.13.1 documents REALTIME, not a fixed Hz.
    llvm.InlineAsmOp(
        res=None,
        operands_=[],
        asm_string="",
        constraints="~{memory}",
        has_side_effects=True,
        is_align_stack=False,
    )
    value = _now()
    llvm.InlineAsmOp(
        res=None,
        operands_=[value.ir_value()],
        asm_string="",
        constraints="s,~{memory}",
        has_side_effects=True,
        is_align_stack=False,
    )
    return value


@flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
def transfer_kernel(
    source: fx.Pointer,
    rowmap: fx.Pointer,
    times: fx.Pointer,
    arena_handle: fx.Int64,
    clock_ticks: fx.Int64,
    op: fx.Constexpr[str],
    wire_row: fx.Constexpr[int],
    slot_bytes: fx.Constexpr[int],
    world: fx.Constexpr[int],
    peers: fx.Constexpr[tuple],
    scatter_offset: fx.Constexpr[int],
    wire_offset: fx.Constexpr[int],
    rowmap_offset: fx.Constexpr[int],
    samples: fx.Constexpr[int],
    instrument: fx.Constexpr[bool],
    empty: fx.Constexpr[bool],
):
    tid = fx.thread_idx.x
    lane = tid % 32
    wave = fx.Int32(rocdl.readfirstlane(fx.Int32.ir_type, (tid // 32).ir_value()))

    if const_expr(op == "clock"):
        if wave == 0:
            start = _timestamp()
            stop = _timestamp()
            while stop - start < clock_ticks:
                stop = _timestamp()
            if lane == 0:
                times[0] = start
                times[1] = stop
    else:
        source_dwords = (
            ROWS * wire_row // 4
            if op == "dispatch"
            else ROWS * SCATTER_LDS_COLUMNS * 2 // 4
        )
        owner_wave = 2 if op == "dispatch" else 0
        storage = fx.SharedAllocator()
        source_lds = storage.allocate(fx.Array[fx.Int32, source_dwords, 16]).peek()
        source_lds_ptr = source_lds.ptr

        # The wrapper proves 16-byte alignment; each copy index is a 16 B unit.
        source_ptr = fx.inttoptr(
            fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Global, 16),
            fx.Int64(fx.ptrtoint(source)),
        )
        source_view = fx.make_view(source_ptr, fx.make_layout(source_dwords, 1))
        lds_view = source_lds.view(fx.make_layout(source_dwords, 1))
        source_chunks = fx.logical_divide(source_view, fx.make_layout(4, 1))
        lds_chunks = fx.logical_divide(lds_view, fx.make_layout(4, 1))
        copy = fx.make_copy_atom(fx.UniversalCopy(128), fx.Int32)
        fragment = fx.make_rmem_tensor(4, fx.Int32)
        for chunk in range(tid, fx.Int32(source_dwords // 4), fx.Int32(BLOCK_THREADS)):
            fx.copy(copy, fx.slice(source_chunks, (None, chunk)), fragment)
            fx.copy(copy, fragment, fx.slice(lds_chunks, (None, chunk)))

        if const_expr(instrument):
            record_ptr = (
                storage.allocate(fx.Array[fx.Int64, samples * 2, 16]).peek().ptr
            )

        win = cco.CachedWindow(arena_handle)
        win_base = win.base
        win_stride = win.stride
        fx.gpu.barrier()

        # Tensor instructions ignore EXEC (CDNA5 ISA 10.11.1). Select a whole
        # wave; a lane==0 predicate is not a way to select their issuing wave.
        if wave == owner_wave:
            zero4 = fx.Vector.filled(4, 0, fx.Int32)
            map_value = fx.Int64(0)
            map_address = fx.Int64(0)
            if const_expr(op == "dispatch"):
                dest = win_base + fx.Int64(peers[0]) * win_stride
                wire_address = dest + wire_offset + DISPATCH_FIRST_ROW * wire_row
                dg0 = fx.Vector(
                    tdm_group0(fx.Int32(fx.ptrtoint(source_lds_ptr)), wire_address)
                )
                dg1 = fx.Vector(tdm_group1(wire_row, ROWS, 1))
                dg2 = zero4
                dg3 = zero4
                map_address = (
                    dest + rowmap_offset + fx.Int64(DISPATCH_FIRST_ROW + lane) * 8
                )
                if lane < ROWS:
                    map_value = rowmap[lane]
            else:
                rows_per_rank = fx.Int32(win_stride // fx.Int64(slot_bytes))
                row_bound = rows_per_rank * world
                row_indices = [
                    fx.Int32(peers[i]) * rows_per_rank
                    + SCATTER_FIRST_SLOT
                    + i * SCATTER_SLOT_STEP
                    for i in range(ROWS)
                ]
                global_ptr = fx.inttoptr(
                    fx.PointerType.get(fx.Int16.ir_type, fx.AddressSpace.Global, 16),
                    win_base + scatter_offset,
                )
                global_view = fx.make_view(
                    global_ptr,
                    fx.make_layout((row_bound, SCATTER_COLUMNS), (slot_bytes // 2, 1)),
                )
                scatter_lds = fx.make_view(
                    fx.recast_iter(fx.Int16, source_lds_ptr),
                    fx.make_layout(
                        (ROWS, SCATTER_LDS_COLUMNS),
                        (SCATTER_LDS_COLUMNS, 1),
                    ),
                )
                desc = make_tensor_gather_descriptor(
                    global_view,
                    scatter_lds,
                    row_indices,
                    row_width=SCATTER_LDS_COLUMNS,
                    tensor_dim0=SCATTER_COLUMNS,
                    tensor_dim1=row_bound.ir_value(),
                    stride=slot_bytes // 2,
                    elem_bytes=2,
                    index_size=32,
                    global_byte_offset=SCATTER_COLUMN_OFFSET * 2,
                )
                # Carry individual IR values, not a Python descriptor object,
                # across the runtime sampling loop.
                dg0 = fx.Vector(desc.dgroup0)
                dg1 = fx.Vector(desc.dgroup1)
                dg2 = fx.Vector(desc.dgroup2)
                dg3 = fx.Vector(desc.dgroup3)

            rocdl.s_wait_loadcnt(0)
            for sample in range(fx.Int32(samples)):
                tdm_ops.tensor_wait(0)
                rocdl.s_wait_storecnt(0)
                # Previous timestamps are LDS stores, never rowmap VMEM stores.
                rocdl.s_wait_dscnt(0)
                if const_expr(instrument):
                    start = _timestamp()
                if const_expr(not empty):
                    if const_expr(op == "dispatch"):
                        tdm_store(dg0.ir_value(), dg1.ir_value(), SCOPE_SYS)
                        if lane < ROWS:
                            _store_i64_sys(map_address, map_value)
                    else:
                        tensor_store_gather(
                            TDMGatherDescriptor(
                                dg0.ir_value(),
                                dg1.ir_value(),
                                dg2.ir_value(),
                                dg3.ir_value(),
                            ),
                            cache_policy=SCOPE_SYS,
                        )
                tdm_ops.tensor_wait(0)
                if const_expr(op == "dispatch"):
                    # Tensor and VMEM have independent completion counters.
                    rocdl.s_wait_storecnt(0)
                if const_expr(instrument):
                    stop = _timestamp()
                    if lane == 0:
                        # add_offset counts int64 elements, not bytes.
                        fx.generic_store(
                            fx.add_offset(record_ptr, sample * 2),
                            start,
                            volatile=True,
                        )
                        fx.generic_store(
                            fx.add_offset(record_ptr, sample * 2 + 1),
                            stop,
                            volatile=True,
                        )

        # The source LDS is reused only after the issuing wave's wait returned.
        fx.gpu.barrier()
        if const_expr(instrument):
            record_view = fx.make_view(record_ptr, fx.make_layout(samples * 2, 1))
            output_view = fx.make_view(times, fx.make_layout(samples * 2, 1))
            records = fx.logical_divide(record_view, fx.make_layout(1, 1))
            outputs = fx.logical_divide(output_view, fx.make_layout(1, 1))
            record_copy = fx.make_copy_atom(fx.UniversalCopy(64), fx.Int64)
            record_fragment = fx.make_rmem_tensor(1, fx.Int64)
            for index in range(tid, fx.Int32(samples * 2), fx.Int32(BLOCK_THREADS)):
                fx.copy(
                    record_copy,
                    fx.slice(records, (None, index)),
                    record_fragment,
                )
                fx.copy(
                    record_copy,
                    record_fragment,
                    fx.slice(outputs, (None, index)),
                )
