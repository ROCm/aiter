# SPDX-License-Identifier: MIT

"""Small FlyDSL diagnostics for cooperative TopK primitives.

These kernels intentionally avoid TopK math. They only answer whether a
multi-CTA-per-row launch can cover row chunks exactly once and whether row-local
global-memory counters can publish a value between cooperating CTAs.
"""

from __future__ import annotations

import functools

import torch
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from .tensor_shim import GTensor, _run_compiled

BLOCK_THREADS = 1024
LOAD_VEC = 4
STATIC_GROUPS = 32

BARRIER_META_SLOTS = 4
META_ARRIVAL_COUNT = 0
META_PHASE_DONE = 1
META_PUBLISHED = 2
META_INIT_EPOCH = 3

COVERAGE_MODES = ("strided", "contiguous", "gated", "static")


def _check_i32_cuda(name: str, tensor: torch.Tensor) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA/ROCm tensor")
    if tensor.dtype is not torch.int32:
        raise TypeError(f"{name} must be torch.int32, got {tensor.dtype}")


def barrier_workspace_slots(num_rows: int) -> int:
    return int(num_rows) * BARRIER_META_SLOTS


@functools.lru_cache(maxsize=32)
def create_coop_coverage_diagnostic_kernel(partitions_per_row: int, mode: str):
    """Return a launcher that increments one counter per covered vector block."""

    if partitions_per_row not in (4, 8, 16):
        raise ValueError(
            "partitions_per_row must be one of 4, 8, or 16, "
            f"got {partitions_per_row}"
        )
    if mode not in COVERAGE_MODES:
        raise ValueError(f"mode must be one of {COVERAGE_MODES}, got {mode!r}")

    kernel_name = f"topk_coop_coverage_diag_p{partitions_per_row}_{mode}"

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def coop_coverage_diag_kernel(
        row_lengths: fx.Tensor,
        counts: fx.Tensor,
        owner_sums: fx.Tensor,
        num_rows: fx.Int32,
        max_vec_blocks: fx.Int32,
    ):
        global_block = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        tid_idx = arith.index_cast(T.index, fx.thread_idx.x)

        c_zero = arith.constant(0, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_vec = arith.constant(LOAD_VEC, type=T.i32)
        c_vec_shift = arith.constant(2, type=T.i32)
        c_parts = arith.constant(partitions_per_row, type=T.i32)
        c_block_i32 = arith.constant(BLOCK_THREADS, type=T.i32)
        c_block_idx = fx.Index(BLOCK_THREADS)
        c_part_stride_idx = fx.Index(BLOCK_THREADS * partitions_per_row)

        row = global_block // ArithValue(c_parts)
        part = global_block - row * ArithValue(c_parts)
        part_idx = arith.index_cast(T.index, part)

        lengths_t = GTensor(row_lengths, dtype=T.i32, shape=(-1,))
        counts_base_idx = buffer_ops.extract_base_index(counts, address_space=1)
        owners_base_idx = buffer_ops.extract_base_index(owner_sums, address_space=1)

        def global_i32_ptr(base_idx, elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(base_idx) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def global_atomic_add_i32(base_idx, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                global_i32_ptr(base_idx, elem_i32),
                value,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result

        def record_vblk(vblk_i32):
            linear = row * ArithValue(max_vec_blocks) + ArithValue(vblk_i32)
            owner_value = ArithValue(part) + ArithValue(c_one)
            global_atomic_add_i32(counts_base_idx, linear, c_one)
            global_atomic_add_i32(owners_base_idx, linear, owner_value)

        row_len = ArithValue(lengths_t[row])
        row_len = (row_len > ArithValue(c_zero)).select(row_len, c_zero)
        vec_blocks_i32 = (
            ArithValue(row_len) + ArithValue(c_vec) - ArithValue(c_one)
        ).shrui(c_vec_shift)
        vec_blocks_idx = arith.index_cast(T.index, vec_blocks_i32)

        if const_expr(mode == "strided"):
            start = ArithValue(tid_idx) + ArithValue(part_idx) * ArithValue(c_block_idx)
            for vblk in range(
                ArithValue(start),
                ArithValue(vec_blocks_idx),
                ArithValue(c_part_stride_idx),
            ):
                record_vblk(arith.index_cast(T.i32, vblk))

        elif const_expr(mode == "contiguous"):
            part_len_i32 = (
                ArithValue(vec_blocks_i32) + ArithValue(c_parts) - ArithValue(c_one)
            ) // ArithValue(c_parts)
            part_vblk_start_i32 = ArithValue(part) * ArithValue(part_len_i32)
            part_remaining_i32 = (
                ArithValue(vec_blocks_i32) - ArithValue(part_vblk_start_i32)
            )
            part_remaining_i32 = (
                ArithValue(part_remaining_i32) > ArithValue(c_zero)
            ).select(part_remaining_i32, c_zero)
            part_blocks_i32 = (
                ArithValue(part_remaining_i32) < ArithValue(part_len_i32)
            ).select(part_remaining_i32, part_len_i32)
            part_blocks_idx = arith.index_cast(T.index, part_blocks_i32)

            for local_vblk in range(
                ArithValue(tid_idx),
                ArithValue(part_blocks_idx),
                ArithValue(c_block_idx),
            ):
                vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                    arith.index_cast(T.i32, local_vblk)
                )
                record_vblk(vblk)

        elif const_expr(mode == "gated"):
            for vblk in range(
                ArithValue(tid_idx),
                ArithValue(vec_blocks_idx),
                ArithValue(c_block_idx),
            ):
                vblk_i32 = arith.index_cast(T.i32, vblk)
                chunk = ArithValue(vblk_i32) // ArithValue(c_block_i32)
                owner = ArithValue(chunk) - (
                    (ArithValue(chunk) // ArithValue(c_parts)) * ArithValue(c_parts)
                )
                if ArithValue(owner) == ArithValue(part):
                    record_vblk(vblk_i32)

        else:
            for group in range_constexpr(STATIC_GROUPS):
                group_base = arith.constant(
                    group * partitions_per_row * BLOCK_THREADS,
                    type=T.i32,
                )
                vblk_i32 = (
                    ArithValue(tid)
                    + ArithValue(group_base)
                    + ArithValue(part) * ArithValue(c_block_i32)
                )
                if ArithValue(vblk_i32) < ArithValue(vec_blocks_i32):
                    record_vblk(vblk_i32)

    @flyc.jit
    def launch_coop_coverage_diag(
        row_lengths: fx.Tensor,
        counts: fx.Tensor,
        owner_sums: fx.Tensor,
        num_rows: fx.Int32,
        max_vec_blocks: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, num_rows * partitions_per_row)
        coop_coverage_diag_kernel(
            row_lengths,
            counts,
            owner_sums,
            num_rows,
            max_vec_blocks,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_coop_coverage_diag


@functools.lru_cache(maxsize=8)
def create_coop_barrier_diagnostic_kernel(partitions_per_row: int):
    """Return a launcher that tests row-local publish/observe synchronization."""

    if partitions_per_row not in (4, 8, 16):
        raise ValueError(
            "partitions_per_row must be one of 4, 8, or 16, "
            f"got {partitions_per_row}"
        )

    kernel_name = f"topk_coop_barrier_diag_p{partitions_per_row}"

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def coop_barrier_diag_kernel(
        values: fx.Tensor,
        observed: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        epoch: fx.Int32,
    ):
        global_block = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)

        c_zero = arith.constant(0, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_parts = arith.constant(partitions_per_row, type=T.i32)
        c_meta_slots = arith.constant(BARRIER_META_SLOTS, type=T.i32)

        row = global_block // ArithValue(c_parts)
        part = global_block - row * ArithValue(c_parts)
        row_meta_base = row * ArithValue(c_meta_slots)
        row_part_base = row * ArithValue(c_parts)

        values_t = GTensor(values, dtype=T.i32, shape=(-1,))
        observed_t = GTensor(observed, dtype=T.i32, shape=(-1,))
        workspace_t = GTensor(workspace, dtype=T.i32, shape=(-1,))
        values_rsrc = values_t.rsrc
        observed_rsrc = observed_t.rsrc
        workspace_rsrc = workspace_t.rsrc
        workspace_base_idx = buffer_ops.extract_base_index(workspace, address_space=1)

        def global_i32_ptr(base_idx, elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(base_idx) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def global_atomic_add_i32(
            base_idx, elem_i32, value, ordering=llvm.AtomicOrdering.monotonic
        ):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                global_i32_ptr(base_idx, elem_i32),
                value,
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_xchg_i32(base_idx, elem_i32, value, ordering):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.xchg,
                global_i32_ptr(base_idx, elem_i32),
                arith.unwrap(value),
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_load_i32_acquire(base_idx, elem_i32):
            return global_atomic_add_i32(
                base_idx,
                elem_i32,
                c_zero,
                llvm.AtomicOrdering.acquire,
            )

        def values_load(elem_i32):
            return buffer_ops.buffer_load(values_rsrc, elem_i32, vec_width=1, dtype=T.i32)

        def values_store(elem_i32, value):
            buffer_ops.buffer_store(value, values_rsrc, elem_i32)

        def observed_store(elem_i32, value):
            buffer_ops.buffer_store(value, observed_rsrc, elem_i32)

        def ws_load(elem_i32):
            return buffer_ops.buffer_load(
                workspace_rsrc, elem_i32, vec_width=1, dtype=T.i32
            )

        def ws_store(elem_i32, value):
            buffer_ops.buffer_store(value, workspace_rsrc, elem_i32)

        def meta_slot(slot):
            return row_meta_base + ArithValue(arith.constant(slot, type=T.i32))

        def spin_until_slot_eq(elem_i32, target):
            init_cur = c_zero
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.ne,
                    cur,
                    arith.unwrap(target),
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                rocdl.s_sleep(1)
                data = global_atomic_load_i32_acquire(workspace_base_idx, elem_i32)
                scf.YieldOp([data])

        def spin_until_slot_ge(elem_i32, target):
            init_cur = c_zero
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.slt,
                    cur,
                    arith.unwrap(target),
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                rocdl.s_sleep(1)
                data = global_atomic_load_i32_acquire(workspace_base_idx, elem_i32)
                scf.YieldOp([data])

        def row_barrier(token):
            token_value = arith.constant(token, type=T.i32)
            target_arrivals = ArithValue(token_value) * ArithValue(c_parts)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tid == ArithValue(c_zero):
                prev = global_atomic_add_i32(
                    workspace_base_idx,
                    meta_slot(META_ARRIVAL_COUNT),
                    c_one,
                    llvm.AtomicOrdering.acq_rel,
                )
                last = (ArithValue(prev) + ArithValue(c_one)) == target_arrivals
                if last:
                    global_atomic_xchg_i32(
                        workspace_base_idx,
                        meta_slot(META_PHASE_DONE),
                        token_value,
                        llvm.AtomicOrdering.release,
                    )
                else:
                    spin_until_slot_ge(meta_slot(META_PHASE_DONE), token_value)
            gpu.barrier()

        if (part == ArithValue(c_zero)) & (tid == ArithValue(c_zero)):
            ws_store(meta_slot(META_ARRIVAL_COUNT), c_zero)
            ws_store(meta_slot(META_PHASE_DONE), c_zero)
            ws_store(meta_slot(META_PUBLISHED), c_zero)
            global_atomic_xchg_i32(
                workspace_base_idx,
                meta_slot(META_INIT_EPOCH),
                epoch,
                llvm.AtomicOrdering.release,
            )

        if tid == ArithValue(c_zero):
            spin_until_slot_eq(meta_slot(META_INIT_EPOCH), epoch)
        gpu.barrier()

        if tid == ArithValue(c_zero):
            values_store(
                row_part_base + ArithValue(part),
                ArithValue(part) + ArithValue(c_one),
            )

        row_barrier(1)

        if (part == ArithValue(c_zero)) & (tid == ArithValue(c_zero)):
            total = c_zero
            for p in range_constexpr(partitions_per_row):
                part_off = arith.constant(p, type=T.i32)
                total = ArithValue(total) + ArithValue(
                    values_load(row_part_base + ArithValue(part_off))
                )
            ws_store(meta_slot(META_PUBLISHED), total)

        row_barrier(2)

        if tid == ArithValue(c_zero):
            published = ws_load(meta_slot(META_PUBLISHED))
            observed_store(row_part_base + ArithValue(part), published)

    @flyc.jit
    def launch_coop_barrier_diag(
        values: fx.Tensor,
        observed: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        epoch: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, num_rows * partitions_per_row)
        coop_barrier_diag_kernel(
            values,
            observed,
            workspace,
            num_rows,
            epoch,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_coop_barrier_diag


@functools.lru_cache(maxsize=8)
def create_coop_arrival_diagnostic_kernel(partitions_per_row: int):
    """Return a no-spin launcher that tests launch coverage plus atomic arrival."""

    if partitions_per_row not in (4, 8, 16):
        raise ValueError(
            "partitions_per_row must be one of 4, 8, or 16, "
            f"got {partitions_per_row}"
        )

    kernel_name = f"topk_coop_arrival_diag_p{partitions_per_row}"

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def coop_arrival_diag_kernel(
        values: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
    ):
        global_block = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)

        c_zero = arith.constant(0, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_parts = arith.constant(partitions_per_row, type=T.i32)
        c_meta_slots = arith.constant(BARRIER_META_SLOTS, type=T.i32)

        row = global_block // ArithValue(c_parts)
        part = global_block - row * ArithValue(c_parts)
        row_meta_base = row * ArithValue(c_meta_slots)
        row_part_base = row * ArithValue(c_parts)

        values_t = GTensor(values, dtype=T.i32, shape=(-1,))
        workspace_t = GTensor(workspace, dtype=T.i32, shape=(-1,))
        values_rsrc = values_t.rsrc
        workspace_rsrc = workspace_t.rsrc
        workspace_base_idx = buffer_ops.extract_base_index(workspace, address_space=1)

        def global_i32_ptr(base_idx, elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(base_idx) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def global_atomic_add_i32(base_idx, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                global_i32_ptr(base_idx, elem_i32),
                value,
                llvm.AtomicOrdering.monotonic,
                syncscope="agent",
                alignment=4,
            ).result

        def values_store(elem_i32, value):
            buffer_ops.buffer_store(value, values_rsrc, elem_i32)

        def ws_store(elem_i32, value):
            buffer_ops.buffer_store(value, workspace_rsrc, elem_i32)

        def meta_slot(slot):
            return row_meta_base + ArithValue(arith.constant(slot, type=T.i32))

        if tid == ArithValue(c_zero):
            values_store(
                row_part_base + ArithValue(part),
                ArithValue(part) + ArithValue(c_one),
            )
            prev = global_atomic_add_i32(
                workspace_base_idx,
                meta_slot(META_ARRIVAL_COUNT),
                c_one,
            )
            last = ArithValue(prev) == (ArithValue(c_parts) - ArithValue(c_one))
            if last:
                ws_store(meta_slot(META_PHASE_DONE), c_one)

    @flyc.jit
    def launch_coop_arrival_diag(
        values: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, num_rows * partitions_per_row)
        coop_arrival_diag_kernel(
            values,
            workspace,
            num_rows,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_coop_arrival_diag


def run_coop_coverage_diagnostic(
    row_lengths: torch.Tensor,
    counts: torch.Tensor,
    owner_sums: torch.Tensor,
    *,
    partitions_per_row: int,
    max_vec_blocks: int,
    mode: str,
    stream: torch.cuda.Stream | None = None,
) -> None:
    _check_i32_cuda("row_lengths", row_lengths)
    _check_i32_cuda("counts", counts)
    _check_i32_cuda("owner_sums", owner_sums)
    if row_lengths.ndim != 1:
        raise ValueError("row_lengths must be a 1D tensor")
    if counts.numel() != owner_sums.numel():
        raise ValueError("counts and owner_sums must have the same number of elements")
    if stream is None:
        stream = torch.cuda.current_stream(row_lengths.device)

    exe = create_coop_coverage_diagnostic_kernel(partitions_per_row, mode)
    _run_compiled(
        exe,
        row_lengths,
        counts,
        owner_sums,
        int(row_lengths.numel()),
        int(max_vec_blocks),
        stream,
    )


def run_coop_barrier_diagnostic(
    values: torch.Tensor,
    observed: torch.Tensor,
    workspace: torch.Tensor,
    *,
    partitions_per_row: int,
    epoch: int = 1,
    stream: torch.cuda.Stream | None = None,
) -> None:
    _check_i32_cuda("values", values)
    _check_i32_cuda("observed", observed)
    _check_i32_cuda("workspace", workspace)
    if values.numel() != observed.numel():
        raise ValueError("values and observed must have the same number of elements")
    if stream is None:
        stream = torch.cuda.current_stream(values.device)

    num_rows = values.numel() // partitions_per_row
    exe = create_coop_barrier_diagnostic_kernel(partitions_per_row)
    _run_compiled(
        exe,
        values,
        observed,
        workspace,
        int(num_rows),
        int(epoch),
        stream,
    )


def run_coop_arrival_diagnostic(
    values: torch.Tensor,
    workspace: torch.Tensor,
    *,
    partitions_per_row: int,
    stream: torch.cuda.Stream | None = None,
) -> None:
    _check_i32_cuda("values", values)
    _check_i32_cuda("workspace", workspace)
    if stream is None:
        stream = torch.cuda.current_stream(values.device)

    num_rows = values.numel() // partitions_per_row
    exe = create_coop_arrival_diagnostic_kernel(partitions_per_row)
    _run_compiled(
        exe,
        values,
        workspace,
        int(num_rows),
        stream,
    )
