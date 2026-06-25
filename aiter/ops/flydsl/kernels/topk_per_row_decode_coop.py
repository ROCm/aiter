# SPDX-License-Identifier: MIT

"""Cooperative long-row FlyDSL TopK-per-row decode kernels.

This module implements the first single-launch multi-CTA radix-select prototype
for long decode rows. It intentionally covers the production-relevant unordered
``K=512`` path first: ``PARTITIONS_PER_ROW`` CTAs cooperatively scan one row,
use a row-local global-memory barrier between radix passes, and directly append
the final TopK set to the output.
"""

import functools

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from .tensor_shim import GTensor

BLOCK_THREADS = 1024
WARP_SIZE = 64
RED_SLOTS = (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE
LOAD_VEC = 4
RADIX_BITS_K512 = 11
RADIX_BINS_K512 = 1 << RADIX_BITS_K512
COOP_META_SLOTS = 16
COOP_NUM_PASSES = 3
LOCAL_TOPK_MERGE_CANDIDATES_PER_PARTITION = 512
# Compile-time long-row bucket. With LOAD_VEC=4 this covers:
#   P=4  -> 262k columns
#   P=8  -> 524k columns
#   P=16 -> 1M columns
COOP_MAX_VEC_GROUPS = 16

META_ARRIVAL_COUNT = 0
META_PHASE_DONE = 1
META_OUT_FRONT = 2
META_OUT_BACK = 3
META_INIT_EPOCH = 4
META_FIRST_ABOVE = 5
META_FIRST_THRESHOLD = 6
META_SECOND_ABOVE = 7
META_SECOND_THRESHOLD = 8
META_THIRD_ABOVE = 9
META_THIRD_THRESHOLD = 10


def cooperative_workspace_slots(
    num_rows: int,
    partitions_per_row: int = 8,
    *,
    atomic_histogram: bool = False,
    local_topk_merge: bool = False,
) -> int:
    """Return the int32 workspace length for the cooperative K=512 kernel."""

    if num_rows < 0:
        raise ValueError(f"num_rows must be non-negative, got {num_rows}")
    if partitions_per_row <= 0:
        raise ValueError(
            f"partitions_per_row must be positive, got {partitions_per_row}"
        )
    if local_topk_merge:
        row_slots = (
            COOP_META_SLOTS
            + 2 * partitions_per_row * LOCAL_TOPK_MERGE_CANDIDATES_PER_PARTITION
        )
    else:
        hist_partitions = 1 if atomic_histogram else partitions_per_row
        row_slots = (
            COOP_META_SLOTS
            + COOP_NUM_PASSES * hist_partitions * RADIX_BINS_K512
        )
    return int(num_rows) * row_slots


@functools.lru_cache(maxsize=16)
def create_topk_per_row_decode_cooperative_k512_kernel(
    partitions_per_row: int = 8,
    *,
    atomic_histogram: bool = False,
):
    """Return a cached cooperative K=512 unordered radix-select launcher."""

    if partitions_per_row not in (4, 8, 16):
        raise ValueError(
            "cooperative TopK only supports partitions_per_row in (4, 8, 16), "
            f"got {partitions_per_row}"
        )

    hist_partitions = 1 if atomic_histogram else partitions_per_row
    row_workspace_slots = (
        COOP_META_SLOTS
        + COOP_NUM_PASSES * hist_partitions * RADIX_BINS_K512
    )
    hist_pass_stride = hist_partitions * RADIX_BINS_K512
    kernel_kind = "atomic_hist" if atomic_histogram else "partial_hist"
    kernel_name = (
        "topk_per_row_decode_radix_coop_k512_"
        f"p{partitions_per_row}_{kernel_kind}_v19"
    )

    @fx.struct
    class SharedStorage:
        s_hist: fx.Array[fx.Int32, RADIX_BINS_K512, 16]
        s_hist2: fx.Array[fx.Int32, RADIX_BINS_K512, 16]
        s_meta: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def topk_per_row_decode_radix_coop_k512_kernel(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        epoch: fx.Int32,
    ):
        del stride1  # The Python wrapper admits only stride1 == 1 for this path.

        global_block = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        tid_idx = arith.index_cast(T.index, fx.thread_idx.x)
        lane = fx.thread_idx.x % WARP_SIZE
        wave = fx.thread_idx.x // WARP_SIZE

        c_zero = arith.constant(0, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_two = arith.constant(2, type=T.i32)
        c_four = arith.constant(4, type=T.i32)
        c_sixteen = arith.constant(RED_SLOTS, type=T.i32)
        c_last_wave = arith.constant(RED_SLOTS - 1, type=T.i32)
        c_last_lane = arith.constant(WARP_SIZE - 1, type=T.i32)
        c_vec = arith.constant(LOAD_VEC, type=T.i32)
        c_parts = arith.constant(partitions_per_row, type=T.i32)
        c_top_k = arith.constant(512, type=T.i32)
        c_block_i32 = arith.constant(BLOCK_THREADS, type=T.i32)
        c_row_stride = arith.constant(row_workspace_slots, type=T.i32)
        c_hist_pass_stride = arith.constant(hist_pass_stride, type=T.i32)
        c_meta_slots = arith.constant(COOP_META_SLOTS, type=T.i32)
        c_bins_i32 = arith.constant(RADIX_BINS_K512, type=T.i32)
        c_bins_idx = fx.Index(RADIX_BINS_K512)
        c_block_idx = fx.Index(BLOCK_THREADS)
        c_part_stride_idx = fx.Index(BLOCK_THREADS * partitions_per_row)
        c_shift = arith.constant(32 - RADIX_BITS_K512, type=T.i32)
        c_mid_shift = arith.constant(10, type=T.i32)
        c_bin_mask = arith.constant(RADIX_BINS_K512 - 1, type=T.i32)
        c_low_mask = arith.constant((1 << 10) - 1, type=T.i32)
        c_sign_bit = arith.constant(-2147483648, type=T.i32)
        c_zero_f32 = fx.Float32(0.0)
        c_neg_one = arith.constant(-1, type=T.i32)

        row = global_block // ArithValue(c_parts)
        part = global_block - row * ArithValue(c_parts)
        part_idx = arith.index_cast(T.index, part)
        row_ws_base = row * ArithValue(c_row_stride)
        row_hist_base = row_ws_base + ArithValue(c_meta_slots)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_hist = lds.s_hist.view(fx.make_layout(RADIX_BINS_K512, 1))
        s_hist2 = lds.s_hist2.view(fx.make_layout(RADIX_BINS_K512, 1))
        s_meta = lds.s_meta.view(fx.make_layout(8, 1))

        logits_t = GTensor(logits, dtype=T.f32, shape=(-1,))
        logits_rsrc = logits_t.rsrc
        seq_lens_t = GTensor(seq_lens, dtype=T.i32, shape=(-1,))
        indices_rsrc = GTensor(indices, dtype=T.i32, shape=(-1,)).rsrc
        workspace_t = GTensor(workspace, dtype=T.i32, shape=(-1,))
        workspace_rsrc = workspace_t.rsrc
        workspace_base_idx = buffer_ops.extract_base_index(workspace, address_space=1)

        hist_base = fx.ptrtoint(lds.s_hist.ptr)
        meta_base = fx.ptrtoint(lds.s_meta.ptr)

        def lds_i32_ptr(base, elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(base) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def lds_atomic_add_i32(base, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                lds_i32_ptr(base, elem_i32),
                value,
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result

        def global_i32_ptr(elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(workspace_base_idx) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def global_load_i32(elem_i32):
            ptr = global_i32_ptr(elem_i32)
            data = llvm.InlineAsmOp(
                T.i32,
                [ptr],
                "global_load_dword $0, $1, off sc0 sc1",
                "=v,v",
                has_side_effects=True,
            ).result
            rocdl.s_waitcnt(0)
            return data

        def global_store_i32(elem_i32, value):
            ptr = global_i32_ptr(elem_i32)
            llvm.InlineAsmOp(
                None,
                [ptr, arith.unwrap(value)],
                "global_store_dword $0, $1, off sc0 sc1",
                "v,v",
                has_side_effects=True,
            )

        def global_atomic_add_i32(
            elem_i32, value, ordering=llvm.AtomicOrdering.monotonic
        ):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                global_i32_ptr(elem_i32),
                value,
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_xchg_i32(elem_i32, value, ordering):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.xchg,
                global_i32_ptr(elem_i32),
                arith.unwrap(value),
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_load_i32_acquire(elem_i32):
            return global_atomic_add_i32(
                elem_i32,
                c_zero,
                llvm.AtomicOrdering.acquire,
            )

        def ws_load(elem_i32):
            return global_load_i32(elem_i32)

        def ws_store(elem_i32, value):
            global_store_i32(elem_i32, value)

        def spin_until_slot_eq(elem_i32, target):
            init_cur = c_zero
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.ne, cur, arith.unwrap(target)
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                rocdl.s_sleep(1)
                data = global_atomic_load_i32_acquire(elem_i32)
                scf.YieldOp([data])

        def spin_until_slot_ge(elem_i32, target):
            init_cur = c_zero
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.slt, cur, arith.unwrap(target)
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                rocdl.s_sleep(1)
                data = global_atomic_load_i32_acquire(elem_i32)
                scf.YieldOp([data])

        def meta_slot(slot):
            return row_ws_base + ArithValue(arith.constant(slot, type=T.i32))

        def pass_hist_base(pass_id):
            return row_hist_base + ArithValue(
                arith.constant(pass_id * hist_pass_stride, type=T.i32)
            )

        def row_barrier(token):
            token_value = arith.constant(token, type=T.i32)
            target_arrivals = ArithValue(token_value) * ArithValue(c_parts)
            # Drain prior global writes from all lanes in this CTA before the
            # row-local arrival counter can release peer CTAs.
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tid == ArithValue(c_zero):
                prev = global_atomic_add_i32(
                    meta_slot(META_ARRIVAL_COUNT),
                    c_one,
                    llvm.AtomicOrdering.acq_rel,
                )
                last = (ArithValue(prev) + ArithValue(c_one)) == target_arrivals
                if last:
                    global_atomic_xchg_i32(
                        meta_slot(META_PHASE_DONE),
                        token_value,
                        llvm.AtomicOrdering.release,
                    )
                else:
                    spin_until_slot_ge(meta_slot(META_PHASE_DONE), token_value)
            gpu.barrier()

        def ordered_key(val):
            key_val = (ArithValue(val) == ArithValue(c_zero_f32)).select(c_zero_f32, val)
            bits = ArithValue(key_val).bitcast(T.i32)
            sign = ArithValue(bits).shrui(arith.constant(31, type=T.i32))
            neg_key = ~ArithValue(bits)
            pos_key = ArithValue(bits) ^ ArithValue(c_sign_bit)
            return (ArithValue(sign) != ArithValue(c_zero)).select(neg_key, pos_key)

        def radix_bucket(val, shift, mask):
            return ArithValue(ArithValue(ordered_key(val)).shrui(shift)) & ArithValue(mask)

        def ordered_bucket(val):
            return ArithValue(ordered_key(val)).shrui(c_shift)

        def partition_owns_vblk(vblk):
            vblk_i32 = arith.index_cast(T.i32, vblk)
            chunk = ArithValue(vblk_i32) // ArithValue(c_block_i32)
            owner = ArithValue(chunk) - (
                (ArithValue(chunk) // ArithValue(c_parts)) * ArithValue(c_parts)
            )
            return ArithValue(owner) == ArithValue(part)

        def partition_group_vblk(group):
            group_base = arith.constant(
                group * partitions_per_row * BLOCK_THREADS, type=T.i32
            )
            return (
                ArithValue(tid)
                + ArithValue(group_base)
                + ArithValue(part) * ArithValue(c_block_i32)
            )

        def load_row_vec(col_base_i32):
            return buffer_ops.buffer_load(
                logits_rsrc,
                row_base + ArithValue(col_base_i32),
                vec_width=LOAD_VEC,
                dtype=T.f32,
            )

        def clear_local_histogram():
            for hist_idx in range(
                ArithValue(tid_idx), ArithValue(c_bins_idx), ArithValue(c_block_idx)
            ):
                hist_i32 = arith.index_cast(T.i32, hist_idx)
                fx.memref_store(c_zero, s_hist, hist_i32)
            gpu.barrier()

        def wave_inclusive_scan_i32(value):
            cur = ArithValue(value)
            for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
                d = arith.constant(1 << sh, type=T.i32)
                src_lane = ArithValue(lane) - ArithValue(d)
                byte_addr = ArithValue(src_lane) * ArithValue(c_four)
                peer = rocdl.ds_bpermute(
                    T.i32, arith.unwrap(byte_addr), arith.unwrap(cur)
                )
                take = ArithValue(lane) >= ArithValue(d)
                cur = take.select(ArithValue(cur) + ArithValue(peer), cur)
            return cur

        def choose_threshold_parallel(target_k, above_slot, threshold_slot):
            two_tid = ArithValue(tid) * ArithValue(c_two)
            c0 = fx.memref_load(s_hist, two_tid)
            c1 = fx.memref_load(s_hist, ArithValue(two_tid) + ArithValue(c_one))
            local_total = ArithValue(c0) + ArithValue(c1)

            wave_incl = wave_inclusive_scan_i32(local_total)
            wave_excl_thread = ArithValue(wave_incl) - ArithValue(local_total)

            if ArithValue(lane) == ArithValue(c_last_lane):
                fx.memref_store(wave_incl, s_hist2, wave)
            gpu.barrier()

            if wave == ArithValue(c_zero):
                in16 = ArithValue(lane) < ArithValue(c_sixteen)
                lane_safe = in16.select(lane, c_zero)
                wtot = in16.select(fx.memref_load(s_hist2, lane_safe), c_zero)
                wincl = wave_inclusive_scan_i32(wtot)
                wexcl = ArithValue(wincl) - ArithValue(wtot)
                if in16:
                    fx.memref_store(
                        wexcl, s_hist2, ArithValue(lane) + ArithValue(c_sixteen)
                    )
            gpu.barrier()

            wave_off = fx.memref_load(s_hist2, ArithValue(wave) + ArithValue(c_sixteen))
            last_off = fx.memref_load(
                s_hist2, ArithValue(c_last_wave) + ArithValue(c_sixteen)
            )
            last_tot = fx.memref_load(s_hist2, c_last_wave)
            total = ArithValue(last_off) + ArithValue(last_tot)
            kprime = ArithValue(total) - ArithValue(target_k)

            excl0 = ArithValue(wave_off) + ArithValue(wave_excl_thread)
            incl0 = ArithValue(excl0) + ArithValue(c0)
            incl1 = ArithValue(incl0) + ArithValue(c1)

            def emit_find(b, excl, incl):
                crosses = (ArithValue(excl) <= ArithValue(kprime)) & (
                    ArithValue(incl) > ArithValue(kprime)
                )
                if crosses:
                    fx.memref_store(b, s_meta, threshold_slot)
                    fx.memref_store(
                        ArithValue(total) - ArithValue(incl), s_meta, above_slot
                    )

            emit_find(two_tid, excl0, incl0)
            emit_find(ArithValue(two_tid) + ArithValue(c_one), incl0, incl1)
            gpu.barrier()

        def flush_local_histogram(pass_id):
            base = pass_hist_base(pass_id)
            if const_expr(atomic_histogram):
                for hist_idx in range(
                    ArithValue(tid_idx),
                    ArithValue(c_bins_idx),
                    ArithValue(c_block_idx),
                ):
                    hist_i32 = arith.index_cast(T.i32, hist_idx)
                    count = fx.memref_load(s_hist, hist_i32)
                    if ArithValue(count) != ArithValue(c_zero):
                        global_atomic_add_i32(base + ArithValue(hist_i32), count)
            else:
                # Publish each partition's private histogram without contended bins.
                part_base = base + part * ArithValue(c_bins_i32)
                for hist_idx in range(
                    ArithValue(tid_idx),
                    ArithValue(c_bins_idx),
                    ArithValue(c_block_idx),
                ):
                    hist_i32 = arith.index_cast(T.i32, hist_idx)
                    count = fx.memref_load(s_hist, hist_i32)
                    ws_store(part_base + ArithValue(hist_i32), count)

        def clear_global_histogram(pass_id):
            if const_expr(atomic_histogram):
                base = pass_hist_base(pass_id)
                start = ArithValue(tid_idx) + ArithValue(part_idx) * ArithValue(
                    c_block_idx
                )
                for hist_idx in range(
                    ArithValue(start),
                    ArithValue(c_bins_idx),
                    ArithValue(c_part_stride_idx),
                ):
                    hist_i32 = arith.index_cast(T.i32, hist_idx)
                    ws_store(base + ArithValue(hist_i32), c_zero)

        def load_global_histogram(pass_id):
            base = pass_hist_base(pass_id)
            for hist_idx in range(
                ArithValue(tid_idx),
                ArithValue(c_bins_idx),
                ArithValue(c_block_idx),
            ):
                hist_i32 = arith.index_cast(T.i32, hist_idx)
                total = c_zero
                if const_expr(atomic_histogram):
                    total = ws_load(base + ArithValue(hist_i32))
                else:
                    for p in range_constexpr(partitions_per_row):
                        p_off = arith.constant(p * RADIX_BINS_K512, type=T.i32)
                        total = ArithValue(total) + ArithValue(
                            global_atomic_load_i32_acquire(
                                base + ArithValue(p_off) + ArithValue(hist_i32)
                            )
                        )
                fx.memref_store(total, s_hist, hist_i32)
            gpu.barrier()

        def choose_global_histogram_locally(pass_id, target_k, above_slot, threshold_slot):
            load_global_histogram(pass_id)
            choose_threshold_parallel(target_k, above_slot, threshold_slot)
            if (part == ArithValue(c_zero)) & (tid == ArithValue(c_zero)):
                above = fx.memref_load(s_meta, above_slot)
                threshold = fx.memref_load(s_meta, threshold_slot)
                if const_expr(pass_id == 0):
                    global_store_i32(meta_slot(META_FIRST_ABOVE), above)
                    global_store_i32(meta_slot(META_FIRST_THRESHOLD), threshold)
                elif const_expr(pass_id == 1):
                    global_store_i32(meta_slot(META_SECOND_ABOVE), above)
                    global_store_i32(meta_slot(META_SECOND_THRESHOLD), threshold)
                else:
                    global_store_i32(meta_slot(META_THIRD_ABOVE), above)
                    global_store_i32(meta_slot(META_THIRD_THRESHOLD), threshold)

        # Per-row decode metadata.
        seq_row = row // ArithValue(next_n)
        slot = row - seq_row * ArithValue(next_n)
        seq_len = ArithValue(seq_lens_t[seq_row])
        row_len = seq_len - ArithValue(next_n) + slot + ArithValue(c_one)
        row_len = (row_len > ArithValue(c_zero)).select(row_len, c_zero)
        row_base = row * ArithValue(stride0)
        row_out = row * ArithValue(c_top_k)

        # Part 0 initializes row-local global state. The epoch makes the cached
        # workspace safe across repeated launches without a host memset.
        if (part == ArithValue(c_zero)) & (tid == ArithValue(c_zero)):
            global_store_i32(meta_slot(META_ARRIVAL_COUNT), c_zero)
            global_store_i32(meta_slot(META_PHASE_DONE), c_zero)
            global_store_i32(meta_slot(META_OUT_FRONT), c_zero)
            global_store_i32(meta_slot(META_OUT_BACK), c_zero)
            global_store_i32(meta_slot(META_FIRST_ABOVE), c_zero)
            global_store_i32(meta_slot(META_FIRST_THRESHOLD), c_zero)
            global_store_i32(meta_slot(META_SECOND_ABOVE), c_zero)
            global_store_i32(meta_slot(META_SECOND_THRESHOLD), c_zero)
            global_store_i32(meta_slot(META_THIRD_ABOVE), c_zero)
            global_store_i32(meta_slot(META_THIRD_THRESHOLD), c_zero)
            global_atomic_xchg_i32(
                meta_slot(META_INIT_EPOCH),
                epoch,
                llvm.AtomicOrdering.release,
            )

        if tid == ArithValue(c_zero):
            spin_until_slot_eq(meta_slot(META_INIT_EPOCH), epoch)
        gpu.barrier()

        direct_fill = row_len <= ArithValue(c_top_k)
        if (part == ArithValue(c_zero)) & direct_fill:
            for out_col in range(
                ArithValue(tid_idx), fx.Index(512), ArithValue(c_block_idx)
            ):
                out_col_i32 = arith.index_cast(T.i32, out_col)
                valid = ArithValue(out_col_i32) < row_len
                out_val = valid.select(out_col_i32, c_neg_one)
                buffer_ops.buffer_store(
                    out_val, indices_rsrc, row_out + ArithValue(out_col_i32)
                )

        long_active = row_len > ArithValue(c_top_k)
        active_len_i32 = long_active.select(row_len, c_zero)
        vec_blocks_i32 = ArithValue(
            ArithValue(active_len_i32) + ArithValue(c_vec) - ArithValue(c_one)
        ).shrui(arith.constant(2, type=T.i32))
        vec_blocks_idx = arith.index_cast(T.index, vec_blocks_i32)
        part_len_i32 = (
            ArithValue(vec_blocks_i32) + ArithValue(c_parts) - ArithValue(c_one)
        ) // ArithValue(c_parts)
        part_vblk_start_i32 = ArithValue(part) * ArithValue(part_len_i32)
        part_remaining_i32 = ArithValue(vec_blocks_i32) - ArithValue(part_vblk_start_i32)
        part_remaining_i32 = (
            ArithValue(part_remaining_i32) > ArithValue(c_zero)
        ).select(part_remaining_i32, c_zero)
        part_blocks_i32 = (
            ArithValue(part_remaining_i32) < ArithValue(part_len_i32)
        ).select(part_remaining_i32, part_len_i32)
        part_blocks_idx = arith.index_cast(T.index, part_blocks_i32)

        # Pass 1: high 11 bits across the whole row. Each CTA scans a
        # disjoint contiguous slice; the default partial-histogram path stores
        # one 2048-bin histogram per partition, avoiding global atomic hot bins.
        if const_expr(atomic_histogram):
            clear_global_histogram(0)
            row_barrier(1)

        clear_local_histogram()
        for local_vblk, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(part_blocks_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                arith.index_cast(T.i32, local_vblk)
            )
            col_base = ArithValue(vblk) * ArithValue(c_vec)
            vec = load_row_vec(col_base)
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(arith.constant(j, type=T.i32))
                in_range = ArithValue(col_i32) < row_len
                if in_range:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    bucket = ordered_bucket(val)
                    lds_atomic_add_i32(hist_base, bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()
        flush_local_histogram(0)
        row_barrier(2 if atomic_histogram else 1)
        choose_global_histogram_locally(0, c_top_k, 0, 1)

        first_above = fx.memref_load(s_meta, c_zero)
        first_threshold = fx.memref_load(s_meta, c_one)
        need_after_first = ArithValue(c_top_k) - ArithValue(first_above)

        # Pass 2: mid 11 bits inside the high-bit boundary bucket.
        if const_expr(atomic_histogram):
            clear_global_histogram(1)
            row_barrier(3)

        clear_local_histogram()
        for local_vblk, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(part_blocks_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                arith.index_cast(T.i32, local_vblk)
            )
            col_base = ArithValue(vblk) * ArithValue(c_vec)
            vec = load_row_vec(col_base)
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(arith.constant(j, type=T.i32))
                in_range = ArithValue(col_i32) < row_len
                if in_range:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    if ArithValue(ordered_bucket(val)) == ArithValue(first_threshold):
                        bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                        lds_atomic_add_i32(hist_base, bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()
        flush_local_histogram(1)
        row_barrier(4 if atomic_histogram else 2)
        choose_global_histogram_locally(
            1,
            need_after_first,
            2,
            3,
        )

        second_above = fx.memref_load(s_meta, c_two)
        second_threshold = fx.memref_load(s_meta, arith.constant(3, type=T.i32))
        need_after_second = ArithValue(need_after_first) - ArithValue(second_above)

        # Pass 3: low 10 bits inside the high+mid boundary.
        if const_expr(atomic_histogram):
            clear_global_histogram(2)
            row_barrier(5)

        clear_local_histogram()
        for local_vblk, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(part_blocks_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                arith.index_cast(T.i32, local_vblk)
            )
            col_base = ArithValue(vblk) * ArithValue(c_vec)
            vec = load_row_vec(col_base)
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(arith.constant(j, type=T.i32))
                in_range = ArithValue(col_i32) < row_len
                if in_range:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    high_bucket = ordered_bucket(val)
                    mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                    matches_refine = (
                        ArithValue(high_bucket) == ArithValue(first_threshold)
                    ) & (ArithValue(mid_bucket) == ArithValue(second_threshold))
                    if matches_refine:
                        bucket = radix_bucket(val, c_zero, c_low_mask)
                        lds_atomic_add_i32(hist_base, bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()
        flush_local_histogram(2)
        row_barrier(6 if atomic_histogram else 3)
        choose_global_histogram_locally(
            2,
            need_after_second,
            4,
            5,
        )

        third_above = fx.memref_load(s_meta, c_four)
        third_threshold = fx.memref_load(s_meta, arith.constant(5, type=T.i32))
        num_needed = ArithValue(need_after_second) - ArithValue(third_above)

        # Final phase: all partitions append results from their own row slice.
        for local_vblk, write_state in range(
            ArithValue(tid_idx),
            ArithValue(part_blocks_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                arith.index_cast(T.i32, local_vblk)
            )
            col_base = ArithValue(vblk) * ArithValue(c_vec)
            vec = load_row_vec(col_base)
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(arith.constant(j, type=T.i32))
                in_range = ArithValue(col_i32) < row_len
                if in_range:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    high_bucket = ordered_bucket(val)
                    mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                    low_bucket = radix_bucket(val, c_zero, c_low_mask)
                    above_first = ArithValue(high_bucket) > ArithValue(first_threshold)
                    at_first = ArithValue(high_bucket) == ArithValue(first_threshold)
                    above_second = ArithValue(mid_bucket) > ArithValue(second_threshold)
                    at_second = ArithValue(mid_bucket) == ArithValue(second_threshold)
                    above_low = ArithValue(low_bucket) > ArithValue(third_threshold)
                    at_low = ArithValue(low_bucket) == ArithValue(third_threshold)
                    strictly_above = above_first | (
                        at_first & (above_second | (at_second & above_low))
                    )
                    at_boundary = at_first & at_second & at_low
                    if strictly_above:
                        pos = global_atomic_add_i32(meta_slot(META_OUT_FRONT), c_one)
                        if ArithValue(pos) < ArithValue(c_top_k):
                            buffer_ops.buffer_store(
                                col_i32, indices_rsrc, row_out + ArithValue(pos)
                            )
                    if at_boundary:
                        back = global_atomic_add_i32(meta_slot(META_OUT_BACK), c_one)
                        if ArithValue(back) < ArithValue(num_needed):
                            out_pos = (
                                ArithValue(c_top_k) - ArithValue(c_one) - ArithValue(back)
                            )
                            buffer_ops.buffer_store(
                                col_i32, indices_rsrc, row_out + ArithValue(out_pos)
                            )
            write_results = yield [write_state[0]]

    @flyc.jit
    def launch_topk_per_row_decode_radix_coop_k512(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        epoch: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, num_rows * partitions_per_row)
        topk_per_row_decode_radix_coop_k512_kernel(
            logits,
            next_n,
            seq_lens,
            indices,
            workspace,
            num_rows,
            stride0,
            stride1,
            epoch,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_topk_per_row_decode_radix_coop_k512


@functools.lru_cache(maxsize=16)
def create_topk_per_row_decode_cooperative_local_topk_merge_k512_kernel(
    partitions_per_row: int = 8,
):
    """Return a single-launch partition-local TopK + merge K=512 launcher."""

    if partitions_per_row not in (4, 8, 16, 32):
        raise ValueError(
            "local TopK merge only supports partitions_per_row in "
            f"(4, 8, 16, 32), got {partitions_per_row}"
        )

    row_workspace_slots = (
        COOP_META_SLOTS
        + 2 * partitions_per_row * LOCAL_TOPK_MERGE_CANDIDATES_PER_PARTITION
    )
    candidate_slots = partitions_per_row * LOCAL_TOPK_MERGE_CANDIDATES_PER_PARTITION
    kernel_name = (
        "topk_per_row_decode_radix_coop_local_topk_merge_k512_"
        f"p{partitions_per_row}_v1"
    )

    @fx.struct
    class SharedStorage:
        s_hist: fx.Array[fx.Int32, RADIX_BINS_K512, 16]
        s_hist2: fx.Array[fx.Int32, RADIX_BINS_K512, 16]
        s_meta: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def topk_per_row_decode_radix_coop_local_topk_merge_k512_kernel(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        epoch: fx.Int32,
    ):
        del stride1

        global_block = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        tid_idx = arith.index_cast(T.index, fx.thread_idx.x)
        lane = fx.thread_idx.x % WARP_SIZE
        wave = fx.thread_idx.x // WARP_SIZE

        c_zero = arith.constant(0, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_two = arith.constant(2, type=T.i32)
        c_four = arith.constant(4, type=T.i32)
        c_six = arith.constant(6, type=T.i32)
        c_seven = arith.constant(7, type=T.i32)
        c_sixteen = arith.constant(RED_SLOTS, type=T.i32)
        c_last_wave = arith.constant(RED_SLOTS - 1, type=T.i32)
        c_last_lane = arith.constant(WARP_SIZE - 1, type=T.i32)
        c_vec = arith.constant(LOAD_VEC, type=T.i32)
        c_parts = arith.constant(partitions_per_row, type=T.i32)
        c_top_k = arith.constant(512, type=T.i32)
        c_top_k_idx = fx.Index(512)
        c_row_stride = arith.constant(row_workspace_slots, type=T.i32)
        c_meta_slots = arith.constant(COOP_META_SLOTS, type=T.i32)
        c_candidates_i32 = arith.constant(candidate_slots, type=T.i32)
        c_candidates_idx = fx.Index(candidate_slots)
        c_block_i32 = arith.constant(BLOCK_THREADS, type=T.i32)
        c_block_idx = fx.Index(BLOCK_THREADS)
        c_bins_idx = fx.Index(RADIX_BINS_K512)
        c_shift = arith.constant(32 - RADIX_BITS_K512, type=T.i32)
        c_mid_shift = arith.constant(10, type=T.i32)
        c_bin_mask = arith.constant(RADIX_BINS_K512 - 1, type=T.i32)
        c_low_mask = arith.constant((1 << 10) - 1, type=T.i32)
        c_sign_bit = arith.constant(-2147483648, type=T.i32)
        c_zero_f32 = fx.Float32(0.0)
        c_neg_one = arith.constant(-1, type=T.i32)

        row = global_block // ArithValue(c_parts)
        part = global_block - row * ArithValue(c_parts)
        row_ws_base = row * ArithValue(c_row_stride)
        row_candidate_base = row_ws_base + ArithValue(c_meta_slots)
        row_candidate_key_base = row_candidate_base + ArithValue(c_candidates_i32)
        part_candidate_base = (
            row_candidate_base
            + ArithValue(part)
            * ArithValue(arith.constant(LOCAL_TOPK_MERGE_CANDIDATES_PER_PARTITION, type=T.i32))
        )
        part_candidate_key_base = (
            row_candidate_key_base
            + ArithValue(part)
            * ArithValue(arith.constant(LOCAL_TOPK_MERGE_CANDIDATES_PER_PARTITION, type=T.i32))
        )

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_hist = lds.s_hist.view(fx.make_layout(RADIX_BINS_K512, 1))
        s_hist2 = lds.s_hist2.view(fx.make_layout(RADIX_BINS_K512, 1))
        s_meta = lds.s_meta.view(fx.make_layout(8, 1))

        logits_t = GTensor(logits, dtype=T.f32, shape=(-1,))
        logits_rsrc = logits_t.rsrc
        seq_lens_t = GTensor(seq_lens, dtype=T.i32, shape=(-1,))
        indices_rsrc = GTensor(indices, dtype=T.i32, shape=(-1,)).rsrc
        workspace_base_idx = buffer_ops.extract_base_index(workspace, address_space=1)

        hist_base = fx.ptrtoint(lds.s_hist.ptr)
        meta_base = fx.ptrtoint(lds.s_meta.ptr)

        def lds_i32_ptr(base, elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(base) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def lds_atomic_add_i32(base, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                lds_i32_ptr(base, elem_i32),
                value,
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result

        def global_i32_ptr(elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(workspace_base_idx) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def global_load_i32(elem_i32):
            ptr = global_i32_ptr(elem_i32)
            data = llvm.InlineAsmOp(
                T.i32,
                [ptr],
                "global_load_dword $0, $1, off sc1",
                "=v,v",
                has_side_effects=True,
            ).result
            rocdl.s_waitcnt(0)
            return data

        def global_store_i32(elem_i32, value):
            ptr = global_i32_ptr(elem_i32)
            llvm.InlineAsmOp(
                None,
                [ptr, arith.unwrap(value)],
                "global_store_dword $0, $1, off sc0 sc1",
                "v,v",
                has_side_effects=True,
            )

        def global_atomic_add_i32(
            elem_i32, value, ordering=llvm.AtomicOrdering.monotonic
        ):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                global_i32_ptr(elem_i32),
                value,
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_xchg_i32(elem_i32, value, ordering):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.xchg,
                global_i32_ptr(elem_i32),
                arith.unwrap(value),
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_load_i32_acquire(elem_i32):
            return global_atomic_add_i32(
                elem_i32,
                c_zero,
                llvm.AtomicOrdering.acquire,
            )

        def spin_until_slot_eq(elem_i32, target):
            init_cur = c_zero
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.ne, cur, arith.unwrap(target)
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                rocdl.s_sleep(1)
                data = global_atomic_load_i32_acquire(elem_i32)
                scf.YieldOp([data])

        def spin_until_slot_ge(elem_i32, target):
            init_cur = c_zero
            w = scf.WhileOp([T.i32], [init_cur])
            before = ir.Block.create_at_start(w.before, [T.i32])
            after = ir.Block.create_at_start(w.after, [T.i32])
            with ir.InsertionPoint(before):
                cur = before.arguments[0]
                need_wait = arith.CmpIOp(
                    arith.CmpIPredicate.slt, cur, arith.unwrap(target)
                ).result
                scf.ConditionOp(need_wait, [cur])
            with ir.InsertionPoint(after):
                rocdl.s_sleep(1)
                data = global_atomic_load_i32_acquire(elem_i32)
                scf.YieldOp([data])

        def meta_slot(slot):
            return row_ws_base + ArithValue(arith.constant(slot, type=T.i32))

        def row_barrier(token):
            token_value = arith.constant(token, type=T.i32)
            target_arrivals = ArithValue(token_value) * ArithValue(c_parts)
            rocdl.s_waitcnt(0)
            gpu.barrier()
            if tid == ArithValue(c_zero):
                prev = global_atomic_add_i32(
                    meta_slot(META_ARRIVAL_COUNT),
                    c_one,
                    llvm.AtomicOrdering.acq_rel,
                )
                last = (ArithValue(prev) + ArithValue(c_one)) == target_arrivals
                if last:
                    global_atomic_xchg_i32(
                        meta_slot(META_PHASE_DONE),
                        token_value,
                        llvm.AtomicOrdering.release,
                    )
                else:
                    spin_until_slot_ge(meta_slot(META_PHASE_DONE), token_value)
            gpu.barrier()

        def ws_load(elem_i32):
            return global_load_i32(elem_i32)

        def ws_store(elem_i32, value):
            global_store_i32(elem_i32, value)

        def ordered_key(val):
            key_val = (ArithValue(val) == ArithValue(c_zero_f32)).select(c_zero_f32, val)
            bits = ArithValue(key_val).bitcast(T.i32)
            sign = ArithValue(bits).shrui(arith.constant(31, type=T.i32))
            neg_key = ~ArithValue(bits)
            pos_key = ArithValue(bits) ^ ArithValue(c_sign_bit)
            return (ArithValue(sign) != ArithValue(c_zero)).select(neg_key, pos_key)

        def radix_bucket(val, shift, mask):
            return ArithValue(ArithValue(ordered_key(val)).shrui(shift)) & ArithValue(mask)

        def ordered_bucket(val):
            return ArithValue(ordered_key(val)).shrui(c_shift)

        def load_row_vec(col_base_i32):
            return buffer_ops.buffer_load(
                logits_rsrc,
                row_base + ArithValue(col_base_i32),
                vec_width=LOAD_VEC,
                dtype=T.f32,
            )

        def load_row_scalar(col_i32):
            return buffer_ops.buffer_load(
                logits_rsrc,
                row_base + ArithValue(col_i32),
                vec_width=1,
                dtype=T.f32,
            )

        def clear_histogram():
            for hist_idx in range(
                ArithValue(tid_idx), ArithValue(c_bins_idx), ArithValue(c_block_idx)
            ):
                hist_i32 = arith.index_cast(T.i32, hist_idx)
                fx.memref_store(c_zero, s_hist, hist_i32)
            gpu.barrier()

        def clear_meta():
            if tid == ArithValue(c_zero):
                fx.memref_store(c_zero, s_meta, 0)
                fx.memref_store(c_zero, s_meta, 1)
                fx.memref_store(c_zero, s_meta, 2)
                fx.memref_store(c_zero, s_meta, 3)
                fx.memref_store(c_zero, s_meta, 4)
                fx.memref_store(c_zero, s_meta, 5)
                fx.memref_store(c_zero, s_meta, 6)
                fx.memref_store(c_zero, s_meta, 7)
            gpu.barrier()

        def wave_inclusive_scan_i32(value):
            cur = ArithValue(value)
            for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
                d = arith.constant(1 << sh, type=T.i32)
                src_lane = ArithValue(lane) - ArithValue(d)
                byte_addr = ArithValue(src_lane) * ArithValue(c_four)
                peer = rocdl.ds_bpermute(
                    T.i32, arith.unwrap(byte_addr), arith.unwrap(cur)
                )
                take = ArithValue(lane) >= ArithValue(d)
                cur = take.select(ArithValue(cur) + ArithValue(peer), cur)
            return cur

        def choose_threshold_parallel(target_k, above_slot, threshold_slot):
            two_tid = ArithValue(tid) * ArithValue(c_two)
            c0 = fx.memref_load(s_hist, two_tid)
            c1 = fx.memref_load(s_hist, ArithValue(two_tid) + ArithValue(c_one))
            local_total = ArithValue(c0) + ArithValue(c1)

            wave_incl = wave_inclusive_scan_i32(local_total)
            wave_excl_thread = ArithValue(wave_incl) - ArithValue(local_total)

            if ArithValue(lane) == ArithValue(c_last_lane):
                fx.memref_store(wave_incl, s_hist2, wave)
            gpu.barrier()

            if wave == ArithValue(c_zero):
                in16 = ArithValue(lane) < ArithValue(c_sixteen)
                lane_safe = in16.select(lane, c_zero)
                wtot = in16.select(fx.memref_load(s_hist2, lane_safe), c_zero)
                wincl = wave_inclusive_scan_i32(wtot)
                wexcl = ArithValue(wincl) - ArithValue(wtot)
                if in16:
                    fx.memref_store(
                        wexcl, s_hist2, ArithValue(lane) + ArithValue(c_sixteen)
                    )
            gpu.barrier()

            wave_off = fx.memref_load(s_hist2, ArithValue(wave) + ArithValue(c_sixteen))
            last_off = fx.memref_load(
                s_hist2, ArithValue(c_last_wave) + ArithValue(c_sixteen)
            )
            last_tot = fx.memref_load(s_hist2, c_last_wave)
            total = ArithValue(last_off) + ArithValue(last_tot)
            kprime = ArithValue(total) - ArithValue(target_k)

            excl0 = ArithValue(wave_off) + ArithValue(wave_excl_thread)
            incl0 = ArithValue(excl0) + ArithValue(c0)
            incl1 = ArithValue(incl0) + ArithValue(c1)

            def emit_find(b, excl, incl):
                crosses = (ArithValue(excl) <= ArithValue(kprime)) & (
                    ArithValue(incl) > ArithValue(kprime)
                )
                if crosses:
                    fx.memref_store(b, s_meta, threshold_slot)
                    fx.memref_store(
                        ArithValue(total) - ArithValue(incl), s_meta, above_slot
                    )

            emit_find(two_tid, excl0, incl0)
            emit_find(ArithValue(two_tid) + ArithValue(c_one), incl0, incl1)
            gpu.barrier()

        seq_row = row // ArithValue(next_n)
        slot = row - seq_row * ArithValue(next_n)
        seq_len = ArithValue(seq_lens_t[seq_row])
        row_len = seq_len - ArithValue(next_n) + slot + ArithValue(c_one)
        row_len = (row_len > ArithValue(c_zero)).select(row_len, c_zero)
        row_base = row * ArithValue(stride0)
        row_out = row * ArithValue(c_top_k)

        if (part == ArithValue(c_zero)) & (tid == ArithValue(c_zero)):
            global_store_i32(meta_slot(META_ARRIVAL_COUNT), c_zero)
            global_store_i32(meta_slot(META_PHASE_DONE), c_zero)
            global_atomic_xchg_i32(
                meta_slot(META_INIT_EPOCH),
                epoch,
                llvm.AtomicOrdering.release,
            )

        if tid == ArithValue(c_zero):
            spin_until_slot_eq(meta_slot(META_INIT_EPOCH), epoch)
        gpu.barrier()

        direct_fill = row_len <= ArithValue(c_top_k)
        if (part == ArithValue(c_zero)) & direct_fill:
            for out_col in range(
                ArithValue(tid_idx), c_top_k_idx, ArithValue(c_block_idx)
            ):
                out_col_i32 = arith.index_cast(T.i32, out_col)
                valid = ArithValue(out_col_i32) < row_len
                out_val = valid.select(out_col_i32, c_neg_one)
                buffer_ops.buffer_store(
                    out_val, indices_rsrc, row_out + ArithValue(out_col_i32)
                )

        long_active = row_len > ArithValue(c_top_k)
        if long_active:
            active_len_i32 = row_len
            vec_blocks_i32 = ArithValue(
                ArithValue(active_len_i32) + ArithValue(c_vec) - ArithValue(c_one)
            ).shrui(arith.constant(2, type=T.i32))
            part_len_i32 = (
                ArithValue(vec_blocks_i32) + ArithValue(c_parts) - ArithValue(c_one)
            ) // ArithValue(c_parts)
            part_vblk_start_i32 = ArithValue(part) * ArithValue(part_len_i32)
            part_start_col_i32 = ArithValue(part_vblk_start_i32) * ArithValue(c_vec)
            part_remaining_i32 = ArithValue(vec_blocks_i32) - ArithValue(
                part_vblk_start_i32
            )
            part_remaining_i32 = (
                ArithValue(part_remaining_i32) > ArithValue(c_zero)
            ).select(part_remaining_i32, c_zero)
            part_blocks_i32 = (
                ArithValue(part_remaining_i32) < ArithValue(part_len_i32)
            ).select(part_remaining_i32, part_len_i32)
            part_blocks_idx = arith.index_cast(T.index, part_blocks_i32)
            part_elem_capacity = ArithValue(part_blocks_i32) * ArithValue(c_vec)
            part_row_remaining = row_len - ArithValue(part_start_col_i32)
            part_row_remaining = (
                ArithValue(part_row_remaining) > ArithValue(c_zero)
            ).select(part_row_remaining, c_zero)
            part_elem_len = (
                ArithValue(part_row_remaining) < ArithValue(part_elem_capacity)
            ).select(part_row_remaining, part_elem_capacity)

            clear_meta()

            local_direct = ArithValue(part_elem_len) <= ArithValue(c_top_k)
            if local_direct:
                for cand in range(
                    ArithValue(tid_idx), c_top_k_idx, ArithValue(c_block_idx)
                ):
                    cand_i32 = arith.index_cast(T.i32, cand)
                    valid = ArithValue(cand_i32) < ArithValue(part_elem_len)
                    out_idx = valid.select(
                        ArithValue(part_start_col_i32) + ArithValue(cand_i32),
                        c_neg_one,
                    )
                    ws_store(part_candidate_base + ArithValue(cand_i32), out_idx)
                    if valid:
                        val = load_row_scalar(out_idx)
                        ws_store(
                            part_candidate_key_base + ArithValue(cand_i32),
                            ordered_key(val),
                        )

            if ~local_direct:
                # Partition-local pass 1.
                clear_histogram()
                for local_vblk, pass_state in range(
                    ArithValue(tid_idx),
                    ArithValue(part_blocks_idx),
                    ArithValue(c_block_idx),
                    init=[c_zero],
                ):
                    vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                        arith.index_cast(T.i32, local_vblk)
                    )
                    col_base = ArithValue(vblk) * ArithValue(c_vec)
                    vec = load_row_vec(col_base)
                    for j in range_constexpr(LOAD_VEC):
                        col_i32 = ArithValue(col_base) + ArithValue(
                            arith.constant(j, type=T.i32)
                        )
                        in_range = ArithValue(col_i32) < row_len
                        if in_range:
                            val = vector.extract(
                                vec, static_position=[j], dynamic_position=[]
                            )
                            bucket = ordered_bucket(val)
                            lds_atomic_add_i32(hist_base, bucket, c_one)
                    pass_results = yield [pass_state[0]]
                gpu.barrier()
                choose_threshold_parallel(c_top_k, 0, 1)

                first_above = fx.memref_load(s_meta, c_zero)
                first_threshold = fx.memref_load(s_meta, c_one)
                need_after_first = ArithValue(c_top_k) - ArithValue(first_above)

                # Partition-local pass 2.
                clear_histogram()
                for local_vblk, pass_state in range(
                    ArithValue(tid_idx),
                    ArithValue(part_blocks_idx),
                    ArithValue(c_block_idx),
                    init=[c_zero],
                ):
                    vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                        arith.index_cast(T.i32, local_vblk)
                    )
                    col_base = ArithValue(vblk) * ArithValue(c_vec)
                    vec = load_row_vec(col_base)
                    for j in range_constexpr(LOAD_VEC):
                        col_i32 = ArithValue(col_base) + ArithValue(
                            arith.constant(j, type=T.i32)
                        )
                        in_range = ArithValue(col_i32) < row_len
                        if in_range:
                            val = vector.extract(
                                vec, static_position=[j], dynamic_position=[]
                            )
                            if ArithValue(ordered_bucket(val)) == ArithValue(
                                first_threshold
                            ):
                                mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                                lds_atomic_add_i32(hist_base, mid_bucket, c_one)
                    pass_results = yield [pass_state[0]]
                gpu.barrier()
                choose_threshold_parallel(need_after_first, 2, 3)

                second_above = fx.memref_load(s_meta, c_two)
                second_threshold = fx.memref_load(s_meta, arith.constant(3, type=T.i32))
                need_after_second = ArithValue(need_after_first) - ArithValue(
                    second_above
                )

                # Partition-local pass 3.
                clear_histogram()
                for local_vblk, pass_state in range(
                    ArithValue(tid_idx),
                    ArithValue(part_blocks_idx),
                    ArithValue(c_block_idx),
                    init=[c_zero],
                ):
                    vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                        arith.index_cast(T.i32, local_vblk)
                    )
                    col_base = ArithValue(vblk) * ArithValue(c_vec)
                    vec = load_row_vec(col_base)
                    for j in range_constexpr(LOAD_VEC):
                        col_i32 = ArithValue(col_base) + ArithValue(
                            arith.constant(j, type=T.i32)
                        )
                        in_range = ArithValue(col_i32) < row_len
                        if in_range:
                            val = vector.extract(
                                vec, static_position=[j], dynamic_position=[]
                            )
                            high_bucket = ordered_bucket(val)
                            mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                            matches_refine = (
                                ArithValue(high_bucket) == ArithValue(first_threshold)
                            ) & (
                                ArithValue(mid_bucket) == ArithValue(second_threshold)
                            )
                            if matches_refine:
                                low_bucket = radix_bucket(val, c_zero, c_low_mask)
                                lds_atomic_add_i32(hist_base, low_bucket, c_one)
                    pass_results = yield [pass_state[0]]
                gpu.barrier()
                choose_threshold_parallel(need_after_second, 4, 5)

                third_above = fx.memref_load(s_meta, c_four)
                third_threshold = fx.memref_load(s_meta, arith.constant(5, type=T.i32))
                num_needed = ArithValue(need_after_second) - ArithValue(third_above)

                if tid == ArithValue(c_zero):
                    fx.memref_store(c_zero, s_meta, c_six)
                    fx.memref_store(c_zero, s_meta, c_seven)
                gpu.barrier()

                for local_vblk, write_state in range(
                    ArithValue(tid_idx),
                    ArithValue(part_blocks_idx),
                    ArithValue(c_block_idx),
                    init=[c_zero],
                ):
                    vblk = ArithValue(part_vblk_start_i32) + ArithValue(
                        arith.index_cast(T.i32, local_vblk)
                    )
                    col_base = ArithValue(vblk) * ArithValue(c_vec)
                    vec = load_row_vec(col_base)
                    for j in range_constexpr(LOAD_VEC):
                        col_i32 = ArithValue(col_base) + ArithValue(
                            arith.constant(j, type=T.i32)
                        )
                        in_range = ArithValue(col_i32) < row_len
                        if in_range:
                            val = vector.extract(
                                vec, static_position=[j], dynamic_position=[]
                            )
                            high_bucket = ordered_bucket(val)
                            mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                            low_bucket = radix_bucket(val, c_zero, c_low_mask)
                            above_first = ArithValue(high_bucket) > ArithValue(
                                first_threshold
                            )
                            at_first = ArithValue(high_bucket) == ArithValue(
                                first_threshold
                            )
                            above_second = ArithValue(mid_bucket) > ArithValue(
                                second_threshold
                            )
                            at_second = ArithValue(mid_bucket) == ArithValue(
                                second_threshold
                            )
                            above_low = ArithValue(low_bucket) > ArithValue(
                                third_threshold
                            )
                            at_low = ArithValue(low_bucket) == ArithValue(
                                third_threshold
                            )
                            strictly_above = above_first | (
                                at_first & (above_second | (at_second & above_low))
                            )
                            at_boundary = at_first & at_second & at_low
                            if strictly_above:
                                pos = lds_atomic_add_i32(meta_base, c_six, c_one)
                                if ArithValue(pos) < ArithValue(c_top_k):
                                    ws_store(
                                        part_candidate_base + ArithValue(pos),
                                        col_i32,
                                    )
                                    ws_store(
                                        part_candidate_key_base + ArithValue(pos),
                                        ordered_key(val),
                                    )
                            if at_boundary:
                                back = lds_atomic_add_i32(meta_base, c_seven, c_one)
                                if ArithValue(back) < ArithValue(num_needed):
                                    cand_pos = (
                                        ArithValue(c_top_k)
                                        - ArithValue(c_one)
                                        - ArithValue(back)
                                    )
                                    ws_store(
                                        part_candidate_base + ArithValue(cand_pos),
                                        col_i32,
                                    )
                                    ws_store(
                                        part_candidate_key_base + ArithValue(cand_pos),
                                        ordered_key(val),
                                    )
                    write_results = yield [write_state[0]]

            row_barrier(1)

            if part == ArithValue(c_zero):
                clear_meta()

                # Merge pass 1 over P*K partition candidates.
                clear_histogram()
                for cand, pass_state in range(
                    ArithValue(tid_idx),
                    c_candidates_idx,
                    ArithValue(c_block_idx),
                    init=[c_zero],
                ):
                    cand_i32 = arith.index_cast(T.i32, cand)
                    col_i32 = ws_load(row_candidate_base + ArithValue(cand_i32))
                    valid = ArithValue(col_i32) >= ArithValue(c_zero)
                    if valid:
                        key_i32 = ws_load(
                            row_candidate_key_base + ArithValue(cand_i32)
                        )
                        bucket = ArithValue(key_i32).shrui(c_shift)
                        lds_atomic_add_i32(hist_base, bucket, c_one)
                    pass_results = yield [pass_state[0]]
                gpu.barrier()
                choose_threshold_parallel(c_top_k, 0, 1)

                first_above_m = fx.memref_load(s_meta, c_zero)
                first_threshold_m = fx.memref_load(s_meta, c_one)
                need_after_first_m = ArithValue(c_top_k) - ArithValue(first_above_m)

                # Merge pass 2.
                clear_histogram()
                for cand, pass_state in range(
                    ArithValue(tid_idx),
                    c_candidates_idx,
                    ArithValue(c_block_idx),
                    init=[c_zero],
                ):
                    cand_i32 = arith.index_cast(T.i32, cand)
                    col_i32 = ws_load(row_candidate_base + ArithValue(cand_i32))
                    valid = ArithValue(col_i32) >= ArithValue(c_zero)
                    if valid:
                        key_i32 = ws_load(
                            row_candidate_key_base + ArithValue(cand_i32)
                        )
                        high_bucket = ArithValue(key_i32).shrui(c_shift)
                        if ArithValue(high_bucket) == ArithValue(
                            first_threshold_m
                        ):
                            mid_bucket = (
                                ArithValue(key_i32).shrui(c_mid_shift)
                            ) & ArithValue(c_bin_mask)
                            lds_atomic_add_i32(hist_base, mid_bucket, c_one)
                    pass_results = yield [pass_state[0]]
                gpu.barrier()
                choose_threshold_parallel(need_after_first_m, 2, 3)

                second_above_m = fx.memref_load(s_meta, c_two)
                second_threshold_m = fx.memref_load(s_meta, arith.constant(3, type=T.i32))
                need_after_second_m = ArithValue(need_after_first_m) - ArithValue(
                    second_above_m
                )

                # Merge pass 3.
                clear_histogram()
                for cand, pass_state in range(
                    ArithValue(tid_idx),
                    c_candidates_idx,
                    ArithValue(c_block_idx),
                    init=[c_zero],
                ):
                    cand_i32 = arith.index_cast(T.i32, cand)
                    col_i32 = ws_load(row_candidate_base + ArithValue(cand_i32))
                    valid = ArithValue(col_i32) >= ArithValue(c_zero)
                    if valid:
                        key_i32 = ws_load(
                            row_candidate_key_base + ArithValue(cand_i32)
                        )
                        high_bucket = ArithValue(key_i32).shrui(c_shift)
                        mid_bucket = (ArithValue(key_i32).shrui(c_mid_shift)) & ArithValue(
                            c_bin_mask
                        )
                        matches_refine = (
                            ArithValue(high_bucket) == ArithValue(first_threshold_m)
                        ) & (
                            ArithValue(mid_bucket) == ArithValue(second_threshold_m)
                        )
                        if matches_refine:
                            low_bucket = ArithValue(key_i32) & ArithValue(c_low_mask)
                            lds_atomic_add_i32(hist_base, low_bucket, c_one)
                    pass_results = yield [pass_state[0]]
                gpu.barrier()
                choose_threshold_parallel(need_after_second_m, 4, 5)

                third_above_m = fx.memref_load(s_meta, c_four)
                third_threshold_m = fx.memref_load(s_meta, arith.constant(5, type=T.i32))
                num_needed_m = ArithValue(need_after_second_m) - ArithValue(
                    third_above_m
                )

                if tid == ArithValue(c_zero):
                    fx.memref_store(c_zero, s_meta, c_six)
                    fx.memref_store(c_zero, s_meta, c_seven)
                gpu.barrier()

                for cand, write_state in range(
                    ArithValue(tid_idx),
                    c_candidates_idx,
                    ArithValue(c_block_idx),
                    init=[c_zero],
                ):
                    cand_i32 = arith.index_cast(T.i32, cand)
                    col_i32 = ws_load(row_candidate_base + ArithValue(cand_i32))
                    valid = ArithValue(col_i32) >= ArithValue(c_zero)
                    if valid:
                        key_i32 = ws_load(
                            row_candidate_key_base + ArithValue(cand_i32)
                        )
                        high_bucket = ArithValue(key_i32).shrui(c_shift)
                        mid_bucket = (ArithValue(key_i32).shrui(c_mid_shift)) & ArithValue(
                            c_bin_mask
                        )
                        low_bucket = ArithValue(key_i32) & ArithValue(c_low_mask)
                        above_first = ArithValue(high_bucket) > ArithValue(
                            first_threshold_m
                        )
                        at_first = ArithValue(high_bucket) == ArithValue(
                            first_threshold_m
                        )
                        above_second = ArithValue(mid_bucket) > ArithValue(
                            second_threshold_m
                        )
                        at_second = ArithValue(mid_bucket) == ArithValue(
                            second_threshold_m
                        )
                        above_low = ArithValue(low_bucket) > ArithValue(
                            third_threshold_m
                        )
                        at_low = ArithValue(low_bucket) == ArithValue(
                            third_threshold_m
                        )
                        strictly_above = above_first | (
                            at_first & (above_second | (at_second & above_low))
                        )
                        at_boundary = at_first & at_second & at_low
                        if strictly_above:
                            pos = lds_atomic_add_i32(meta_base, c_six, c_one)
                            if ArithValue(pos) < ArithValue(c_top_k):
                                buffer_ops.buffer_store(
                                    col_i32,
                                    indices_rsrc,
                                    row_out + ArithValue(pos),
                                )
                        if at_boundary:
                            back = lds_atomic_add_i32(meta_base, c_seven, c_one)
                            if ArithValue(back) < ArithValue(num_needed_m):
                                out_pos = (
                                    ArithValue(c_top_k)
                                    - ArithValue(c_one)
                                    - ArithValue(back)
                                )
                                buffer_ops.buffer_store(
                                    col_i32,
                                    indices_rsrc,
                                    row_out + ArithValue(out_pos),
                                )
                    write_results = yield [write_state[0]]

    @flyc.jit
    def launch_topk_per_row_decode_radix_coop_local_topk_merge_k512(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        epoch: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, num_rows * partitions_per_row)
        topk_per_row_decode_radix_coop_local_topk_merge_k512_kernel(
            logits,
            next_n,
            seq_lens,
            indices,
            workspace,
            num_rows,
            stride0,
            stride1,
            epoch,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_topk_per_row_decode_radix_coop_local_topk_merge_k512
