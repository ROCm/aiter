# SPDX-License-Identifier: MIT

"""AITER-style persistent multi-block radix TopK-per-row decode kernel.

This module intentionally keeps the implementation separate from the older
cooperative experiments.  It ports the production AITER HIP multi-block shape:
one persistent launch, ``grid=(blocks_per_row, num_rows)``, per-block LDS
histograms, pass-private global histograms, and a row-local acquire/release
barrier between radix passes.  The output is the unordered K=512 selected set.
"""

import functools

from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from .layout_utils import crd2idx
from .tensor_shim import GTensor

BLOCK_THREADS = 1024
WARP_SIZE = 64
RED_SLOTS = (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE
LOAD_VEC = 4
TOP_K = 512

# Match AITER Counter's 128B spacing for hot inter-CTA fields while keeping the
# workspace as an int32 tensor.
COUNTER_SLOTS = 192
COUNTER_ARRIVALS = 64
COUNTER_OUT_FRONT = 96
COUNTER_OUT_BACK = 128
COUNTER_PASS_DONE = 160

SMEM_META_K = 0
SMEM_META_LEN = 1
SMEM_META_THRESHOLD = 2
SMEM_META_ABOVE = 3


def _num_passes(bits_per_pass: int) -> int:
    return (32 + bits_per_pass - 1) // bits_per_pass


def aiter_persistent_workspace_slots(
    num_rows: int,
    blocks_per_row: int,
    *,
    bits_per_pass: int,
) -> int:
    """Return int32 workspace slots for the AITER-persistent K=512 path."""

    del blocks_per_row  # Global histograms are atomically merged per row.
    if num_rows < 0:
        raise ValueError(f"num_rows must be non-negative, got {num_rows}")
    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    row_slots = COUNTER_SLOTS + _num_passes(bits_per_pass) * (1 << bits_per_pass)
    return int(num_rows) * row_slots


@functools.lru_cache(maxsize=16)
def create_topk_per_row_decode_aiter_persistent_k512_kernel(
    blocks_per_row: int = 8,
    *,
    bits_per_pass: int = 10,
    scan_stages: int = 2,
    tiered: bool = False,
    tiered_short_max: int = 16384,
    tiered_mid_cap: int = 16,
    tiered_mid_max: int = 65536,
    tiered_long_cap: int = 32,
):
    """Return a cached AITER-style persistent K=512 radix-select launcher."""

    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    if tiered_short_max < 0:
        raise ValueError(f"tiered_short_max must be non-negative, got {tiered_short_max}")
    if tiered_mid_cap < 2:
        raise ValueError(f"tiered_mid_cap must be >= 2, got {tiered_mid_cap}")
    if tiered_long_cap < 2:
        raise ValueError(f"tiered_long_cap must be >= 2, got {tiered_long_cap}")
    if tiered_mid_max < tiered_short_max:
        raise ValueError(
            "tiered_mid_max must be >= tiered_short_max, got "
            f"{tiered_mid_max} < {tiered_short_max}"
        )
    if scan_stages not in (1, 2, 4, 8):
        raise ValueError(f"scan_stages must be one of (1, 2, 4, 8), got {scan_stages}")
    block_threads = BLOCK_THREADS
    if not 2 <= blocks_per_row <= 32:
        raise ValueError(
            "AITER persistent TopK only supports blocks_per_row in [2, 32], "
            f"got {blocks_per_row}"
        )

    # The short-row tier runs a faithful clone of the standalone one-CTA
    # unordered radix-select path on part 0 only. It uses the 11/11/10 ordered
    # key scheme, which needs a 2048-bin LDS histogram, so it is only available
    # when bits_per_pass == 11 (num_buckets == 2048). When unavailable (the
    # non-default bpp10 device), the tiered short tier falls back to the
    # single-part persistent path.
    short_clone = tiered and bits_per_pass == 11
    red_slots = (block_threads + WARP_SIZE - 1) // WARP_SIZE
    threads_per_row = block_threads * blocks_per_row
    num_passes = _num_passes(bits_per_pass)
    num_buckets = 1 << bits_per_pass
    row_workspace_slots = COUNTER_SLOTS + num_passes * num_buckets
    kernel_name = (
        "topk_per_row_decode_aiter_persistent_k512_"
        f"bpp{bits_per_pass}_g{blocks_per_row}_v2"
        f"_stage{scan_stages}"
        f"{'_tiered' if tiered else ''}"
        f"{f'_s{tiered_short_max}_mc{tiered_mid_cap}' if tiered else ''}"
        f"{f'_mm{tiered_mid_max}_lc{tiered_long_cap}' if tiered else ''}"
        f"{'_oneClone' if short_clone else ''}"
    )

    @fx.struct
    class SharedStorage:
        s_hist: fx.Array[fx.Int32, num_buckets, 16]
        s_scan: fx.Array[fx.Int32, red_slots * 2, 16]
        s_meta: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def topk_per_row_decode_aiter_persistent_k512_kernel(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
    ):
        del stride1  # The Python wrapper admits only stride1 == 1.

        block_x = gpu.block_id("x")
        block_y = gpu.block_id("y")
        thread_x = gpu.thread_id("x")
        part = ArithValue(arith.index_cast(T.i32, block_x))
        row = ArithValue(arith.index_cast(T.i32, block_y))
        tid = ArithValue(arith.index_cast(T.i32, thread_x))
        tid_idx = arith.index_cast(T.index, thread_x)
        row_idx = arith.index_cast(T.index, block_y)
        lane = tid % ArithValue(fx.Int32(WARP_SIZE))
        wave = tid // ArithValue(fx.Int32(WARP_SIZE))

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_two = fx.Int32(2)
        c_four = fx.Int32(4)
        c_sixteen = fx.Int32(red_slots)
        c_last_wave = fx.Int32(red_slots - 1)
        c_last_lane = fx.Int32(WARP_SIZE - 1)
        c_vec = fx.Int32(LOAD_VEC)
        c_top_k = fx.Int32(TOP_K)
        c_block_i32 = fx.Int32(block_threads)
        c_block_idx = fx.Index(block_threads)
        c_bins_i32 = fx.Int32(num_buckets)
        c_bins_idx = fx.Index(num_buckets)
        c_parts = fx.Int32(blocks_per_row)
        c_parts_stride_idx = fx.Index(threads_per_row)
        c_parts_stride4_idx = fx.Index(threads_per_row * 4)
        c_sign_bit = fx.Int32(-2147483648)
        c_neg_one = fx.Int32(-1)
        c_zero_f32 = fx.Float32(0.0)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_hist = lds.s_hist.view(fx.make_layout(num_buckets, 1))
        s_scan = lds.s_scan.view(fx.make_layout(red_slots * 2, 1))
        s_meta = lds.s_meta.view(fx.make_layout(8, 1))

        logits_rsrc = GTensor(logits, dtype=T.f32, shape=(-1,)).rsrc
        seq_lens_t = GTensor(seq_lens, dtype=T.i32, shape=(-1,))
        indices_rsrc = GTensor(indices, dtype=T.i32, shape=(-1,)).rsrc
        workspace_t = GTensor(workspace, dtype=T.i32, shape=(-1,))
        workspace_rsrc = workspace_t.rsrc
        workspace_base_idx = buffer_ops.extract_base_index(workspace, address_space=1)

        hist_base_ptr = fx.ptrtoint(lds.s_hist.ptr)
        meta_base_ptr = fx.ptrtoint(lds.s_meta.ptr)

        # Decode row metadata before helper definitions so tiered dispatch can
        # feed both scan strides and inter-CTA barrier participant counts.
        seq_row = row // ArithValue(next_n)
        slot = row - seq_row * ArithValue(next_n)
        seq_len = ArithValue(seq_lens_t[seq_row])
        row_len = seq_len - ArithValue(next_n) + slot + ArithValue(c_one)
        row_len = (row_len > ArithValue(c_zero)).select(row_len, c_zero)
        row_base = row * ArithValue(stride0)
        row_out = row * ArithValue(c_top_k)

        active_parts = c_parts
        if const_expr(tiered):
            # Per-bucket active-part caps over the fixed launch grid (excess
            # blocks return immediately). Short rows run the single-CTA clone;
            # the mid and long cooperative buckets cap how many of the grid's
            # blocks participate, trading CU coverage against barrier/atomic
            # contention. Caps are clamped to the grid (``c_parts``).
            c_short_row_len = fx.Int32(tiered_short_max)
            c_mid_row_len = fx.Int32(tiered_mid_max)
            c_mid_parts_cap = fx.Int32(tiered_mid_cap)
            c_long_parts_cap = fx.Int32(tiered_long_cap)
            mid_parts = (ArithValue(c_parts) < ArithValue(c_mid_parts_cap)).select(
                c_parts, c_mid_parts_cap
            )
            long_parts = (ArithValue(c_parts) < ArithValue(c_long_parts_cap)).select(
                c_parts, c_long_parts_cap
            )
            active_parts = (row_len <= ArithValue(c_short_row_len)).select(
                c_one,
                (row_len <= ArithValue(c_mid_row_len)).select(mid_parts, long_parts),
            )
        active_threads = ArithValue(active_parts) * ArithValue(c_block_i32)
        active_stride_idx = arith.index_cast(T.index, active_threads)
        active_stride2_idx = arith.index_cast(
            T.index, ArithValue(active_threads) * ArithValue(c_two)
        )
        active_stride3_idx = arith.index_cast(
            T.index, ArithValue(active_threads) * ArithValue(fx.Int32(3))
        )
        active_stride4_idx = arith.index_cast(
            T.index, ArithValue(active_threads) * ArithValue(c_four)
        )
        active_stride7_idx = arith.index_cast(
            T.index, ArithValue(active_threads) * ArithValue(fx.Int32(7))
        )
        active_stride8_idx = arith.index_cast(
            T.index, ArithValue(active_threads) * ArithValue(fx.Int32(8))
        )
        single_part_active = ArithValue(active_parts) == ArithValue(c_one)

        ws_layout = fx.make_layout(
            (1, row_workspace_slots),
            (row_workspace_slots, 1),
        )
        hist_layout = fx.make_layout(
            (1, num_passes, num_buckets),
            (row_workspace_slots, num_buckets, 1),
        )

        def to_i32_index(idx):
            return arith.index_cast(T.i32, idx)

        def workspace_slot(slot_i32):
            slot_idx = arith.index_cast(T.index, slot_i32)
            return to_i32_index(crd2idx((row_idx, slot_idx), ws_layout))

        def histogram_slot(pass_id: int, bin_i32):
            bin_idx = arith.index_cast(T.index, bin_i32)
            pass_offset = crd2idx((row_idx, fx.Index(pass_id), bin_idx), hist_layout)
            return to_i32_index(fx.Index(COUNTER_SLOTS) + pass_offset)

        def counter_slot(slot: int):
            return workspace_slot(fx.Int32(slot))

        def counter_slot_i32(slot_i32):
            return workspace_slot(slot_i32)

        def global_i32_ptr(elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(workspace_base_idx) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=1)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def lds_i32_ptr(base, elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(base) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def ws_load(elem_i32):
            return buffer_ops.buffer_load(
                workspace_rsrc, elem_i32, vec_width=1, dtype=T.i32
            )

        def ws_store(elem_i32, value):
            buffer_ops.buffer_store(value, workspace_rsrc, elem_i32)

        def global_atomic_add_i32(
            elem_i32, value, ordering=llvm.AtomicOrdering.monotonic
        ):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                global_i32_ptr(elem_i32),
                arith.unwrap(value),
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
            # True acquire load of the slot rather than an atomicAdd(slot, 0)
            # read-modify-write. The RMW form carried wave-reduce scaffolding
            # (mbcnt / readfirstlane / exec-mask toggling) on every spin
            # iteration; a plain acquire load (matching the AITER HIP
            # __atomic_load_n(..., __ATOMIC_ACQUIRE) spin) reads the L2-coherent
            # value without it. Kept volatile + agent-scoped acquire so it is not
            # hoisted/cached and still establishes the cross-CTA happens-before
            # with the release publish below.
            return llvm.LoadOp(
                T.i32,
                global_i32_ptr(elem_i32),
                alignment=4,
                volatile_=True,
                ordering=llvm.AtomicOrdering.acquire,
                syncscope="agent",
            ).result

        def lds_atomic_add_i32(base, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                lds_i32_ptr(base, elem_i32),
                arith.unwrap(value),
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result

        def spin_until_slot_ge(elem_i32, target):
            init_cur = c_zero
            w = scf.WhileOp([T.i32], [arith.unwrap(init_cur)])
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

        def row_barrier(token: int):
            token_value = fx.Int32(token)
            target_arrivals = ArithValue(token_value) * ArithValue(active_parts)
            gpu.barrier()
            if tid == ArithValue(c_zero):
                prev = global_atomic_add_i32(
                    counter_slot(COUNTER_ARRIVALS),
                    c_one,
                    llvm.AtomicOrdering.acq_rel,
                )
                last = (ArithValue(prev) + ArithValue(c_one)) == target_arrivals
                if last:
                    global_atomic_xchg_i32(
                        counter_slot(COUNTER_PASS_DONE),
                        token_value,
                        llvm.AtomicOrdering.release,
                    )
                else:
                    spin_until_slot_ge(counter_slot(COUNTER_PASS_DONE), token_value)
            gpu.barrier()

        def aiter_twiddle_key(val):
            # AITER's float twiddle maps larger fp32 values to smaller unsigned
            # keys. Normalize signed zero to keep tie handling value-equivalent.
            key_val = (ArithValue(val) == ArithValue(c_zero_f32)).select(c_zero_f32, val)
            bits = ArithValue(key_val).bitcast(T.i32)
            sign = ArithValue(bits).shrui(fx.Int32(31))
            positive_mask = ArithValue(bits) ^ ArithValue(fx.Int32(0x7FFFFFFF))
            return (ArithValue(sign) == ArithValue(c_zero)).select(positive_mask, bits)

        def bucket_for_key(key, start_bit: int):
            return (ArithValue(key).shrui(fx.Int32(start_bit))) & ArithValue(
                fx.Int32(num_buckets - 1)
            )

        def prefix_for_key(key, previous_start_bit: int):
            return arith.shli(
                ArithValue(key).shrui(fx.Int32(previous_start_bit)),
                fx.Int32(previous_start_bit),
            )

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
                d = fx.Int32(1 << sh)
                src_lane = ArithValue(lane) - ArithValue(d)
                byte_addr = ArithValue(src_lane) * ArithValue(c_four)
                peer = rocdl.ds_bpermute(
                    T.i32, arith.unwrap(byte_addr), arith.unwrap(cur)
                )
                take = ArithValue(lane) >= ArithValue(d)
                cur = take.select(ArithValue(cur) + ArithValue(peer), cur)
            return cur

        def choose_bucket_prefix(target_k):
            bins_per_scan_thread = 4 if block_threads == 512 and bits_per_pass == 11 else 2
            first_bin = ArithValue(tid) * ArithValue(fx.Int32(bins_per_scan_thread))
            bin0_valid = ArithValue(first_bin) < ArithValue(c_bins_i32)
            bin1 = ArithValue(first_bin) + ArithValue(c_one)
            bin1_valid = ArithValue(bin1) < ArithValue(c_bins_i32)
            safe0 = bin0_valid.select(first_bin, c_zero)
            safe1 = bin1_valid.select(bin1, c_zero)
            c0_raw = fx.memref_load(s_hist, safe0)
            c1_raw = fx.memref_load(s_hist, safe1)
            c0 = bin0_valid.select(c0_raw, c_zero)
            c1 = bin1_valid.select(c1_raw, c_zero)
            if const_expr(bins_per_scan_thread == 4):
                bin2 = ArithValue(first_bin) + ArithValue(c_two)
                bin3 = ArithValue(first_bin) + ArithValue(fx.Int32(3))
                bin2_valid = ArithValue(bin2) < ArithValue(c_bins_i32)
                bin3_valid = ArithValue(bin3) < ArithValue(c_bins_i32)
                safe2 = bin2_valid.select(bin2, c_zero)
                safe3 = bin3_valid.select(bin3, c_zero)
                c2_raw = fx.memref_load(s_hist, safe2)
                c3_raw = fx.memref_load(s_hist, safe3)
                c2 = bin2_valid.select(c2_raw, c_zero)
                c3 = bin3_valid.select(c3_raw, c_zero)
                local_total = (
                    ArithValue(c0) + ArithValue(c1) + ArithValue(c2) + ArithValue(c3)
                )
            else:
                c2 = c_zero
                c3 = c_zero
                local_total = ArithValue(c0) + ArithValue(c1)

            wave_incl = wave_inclusive_scan_i32(local_total)
            wave_excl_thread = ArithValue(wave_incl) - ArithValue(local_total)

            if ArithValue(lane) == ArithValue(c_last_lane):
                fx.memref_store(wave_incl, s_scan, wave)
            gpu.barrier()

            if wave == ArithValue(c_zero):
                in16 = ArithValue(lane) < ArithValue(c_sixteen)
                lane_safe = in16.select(lane, c_zero)
                wtot = in16.select(fx.memref_load(s_scan, lane_safe), c_zero)
                wincl = wave_inclusive_scan_i32(wtot)
                wexcl = ArithValue(wincl) - ArithValue(wtot)
                if in16:
                    fx.memref_store(
                        wexcl, s_scan, ArithValue(lane) + ArithValue(c_sixteen)
                    )
            gpu.barrier()

            wave_off = fx.memref_load(s_scan, ArithValue(wave) + ArithValue(c_sixteen))
            excl0 = ArithValue(wave_off) + ArithValue(wave_excl_thread)
            incl0 = ArithValue(excl0) + ArithValue(c0)
            incl1 = ArithValue(incl0) + ArithValue(c1)

            def emit_find(bucket, excl, incl, count):
                crosses = (ArithValue(excl) < ArithValue(target_k)) & (
                    ArithValue(incl) >= ArithValue(target_k)
                )
                if crosses:
                    fx.memref_store(
                        ArithValue(target_k) - ArithValue(excl),
                        s_meta,
                        fx.Int32(SMEM_META_K),
                    )
                    fx.memref_store(count, s_meta, fx.Int32(SMEM_META_LEN))
                    fx.memref_store(bucket, s_meta, fx.Int32(SMEM_META_THRESHOLD))
                    fx.memref_store(excl, s_meta, fx.Int32(SMEM_META_ABOVE))

            emit_find(first_bin, excl0, incl0, c0)
            emit_find(bin1, incl0, incl1, c1)
            if const_expr(bins_per_scan_thread == 4):
                bin2 = ArithValue(first_bin) + ArithValue(c_two)
                bin3 = ArithValue(first_bin) + ArithValue(fx.Int32(3))
                incl2 = ArithValue(incl1) + ArithValue(c2)
                incl3 = ArithValue(incl2) + ArithValue(c3)
                emit_find(bin2, incl1, incl2, c2)
                emit_find(bin3, incl2, incl3, c3)
            gpu.barrier()

        def flush_local_histogram(pass_id: int):
            for hist_idx in range(
                ArithValue(tid_idx), ArithValue(c_bins_idx), ArithValue(c_block_idx)
            ):
                hist_i32 = arith.index_cast(T.i32, hist_idx)
                count = fx.memref_load(s_hist, hist_i32)
                if ArithValue(count) != ArithValue(c_zero):
                    global_atomic_add_i32(histogram_slot(pass_id, hist_i32), count)

        def load_global_histogram(pass_id: int):
            for hist_idx in range(
                ArithValue(tid_idx), ArithValue(c_bins_idx), ArithValue(c_block_idx)
            ):
                hist_i32 = arith.index_cast(T.i32, hist_idx)
                total = ws_load(histogram_slot(pass_id, hist_i32))
                fx.memref_store(total, s_hist, hist_i32)
            gpu.barrier()

        def process_loaded_scan_vec(
            col_base, vec, pass_id: int, start_bit: int, previous_start_bit: int, current_bits
        ):
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                in_range = ArithValue(col_i32) < row_len
                if in_range:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    key = aiter_twiddle_key(val)
                    matches_prefix = True
                    if const_expr(pass_id != 0):
                        matches_prefix = ArithValue(
                            prefix_for_key(key, previous_start_bit)
                        ) == ArithValue(current_bits)
                    if matches_prefix:
                        bucket = bucket_for_key(key, start_bit)
                        lds_atomic_add_i32(hist_base_ptr, bucket, c_one)

        def scan_vec_block(vblk, pass_id: int, start_bit: int, previous_start_bit: int, current_bits):
            vblk_i32 = arith.index_cast(T.i32, vblk)
            col_base = ArithValue(vblk_i32) * ArithValue(c_vec)
            vec = load_row_vec(col_base)
            process_loaded_scan_vec(
                col_base, vec, pass_id, start_bit, previous_start_bit, current_bits
            )

        def staged_scan_vec_blocks(
            vblk, pass_id: int, start_bit: int, previous_start_bit: int, current_bits
        ):
            if const_expr(scan_stages == 1):
                scan_vec_block(vblk, pass_id, start_bit, previous_start_bit, current_bits)
            elif const_expr(scan_stages == 2):
                vblk0 = vblk
                vblk1 = ArithValue(vblk) + ArithValue(active_stride_idx)
                col0 = ArithValue(arith.index_cast(T.i32, vblk0)) * ArithValue(c_vec)
                col1 = ArithValue(arith.index_cast(T.i32, vblk1)) * ArithValue(c_vec)
                vec0 = load_row_vec(col0)
                vec1 = load_row_vec(col1)
                process_loaded_scan_vec(
                    col0, vec0, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col1, vec1, pass_id, start_bit, previous_start_bit, current_bits
                )
            elif const_expr(scan_stages == 4):
                vblk0 = vblk
                vblk1 = ArithValue(vblk) + ArithValue(active_stride_idx)
                vblk2 = ArithValue(vblk) + ArithValue(active_stride2_idx)
                vblk3 = ArithValue(vblk) + ArithValue(active_stride3_idx)
                col0 = ArithValue(arith.index_cast(T.i32, vblk0)) * ArithValue(c_vec)
                col1 = ArithValue(arith.index_cast(T.i32, vblk1)) * ArithValue(c_vec)
                col2 = ArithValue(arith.index_cast(T.i32, vblk2)) * ArithValue(c_vec)
                col3 = ArithValue(arith.index_cast(T.i32, vblk3)) * ArithValue(c_vec)
                vec0 = load_row_vec(col0)
                vec1 = load_row_vec(col1)
                vec2 = load_row_vec(col2)
                vec3 = load_row_vec(col3)
                process_loaded_scan_vec(
                    col0, vec0, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col1, vec1, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col2, vec2, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col3, vec3, pass_id, start_bit, previous_start_bit, current_bits
                )
            else:
                vblk0 = vblk
                vblk1 = ArithValue(vblk) + ArithValue(active_stride_idx)
                vblk2 = ArithValue(vblk) + ArithValue(active_stride2_idx)
                vblk3 = ArithValue(vblk) + ArithValue(active_stride3_idx)
                vblk4 = ArithValue(vblk) + ArithValue(active_stride4_idx)
                vblk5 = ArithValue(vblk) + ArithValue(active_stride4_idx) + ArithValue(active_stride_idx)
                vblk6 = ArithValue(vblk) + ArithValue(active_stride4_idx) + ArithValue(active_stride2_idx)
                vblk7 = ArithValue(vblk) + ArithValue(active_stride7_idx)
                col0 = ArithValue(arith.index_cast(T.i32, vblk0)) * ArithValue(c_vec)
                col1 = ArithValue(arith.index_cast(T.i32, vblk1)) * ArithValue(c_vec)
                col2 = ArithValue(arith.index_cast(T.i32, vblk2)) * ArithValue(c_vec)
                col3 = ArithValue(arith.index_cast(T.i32, vblk3)) * ArithValue(c_vec)
                col4 = ArithValue(arith.index_cast(T.i32, vblk4)) * ArithValue(c_vec)
                col5 = ArithValue(arith.index_cast(T.i32, vblk5)) * ArithValue(c_vec)
                col6 = ArithValue(arith.index_cast(T.i32, vblk6)) * ArithValue(c_vec)
                col7 = ArithValue(arith.index_cast(T.i32, vblk7)) * ArithValue(c_vec)
                vec0 = load_row_vec(col0)
                vec1 = load_row_vec(col1)
                vec2 = load_row_vec(col2)
                vec3 = load_row_vec(col3)
                vec4 = load_row_vec(col4)
                vec5 = load_row_vec(col5)
                vec6 = load_row_vec(col6)
                vec7 = load_row_vec(col7)
                process_loaded_scan_vec(
                    col0, vec0, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col1, vec1, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col2, vec2, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col3, vec3, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col4, vec4, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col5, vec5, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col6, vec6, pass_id, start_bit, previous_start_bit, current_bits
                )
                process_loaded_scan_vec(
                    col7, vec7, pass_id, start_bit, previous_start_bit, current_bits
                )

        def process_loaded_write_vec(col_base, vec, local_k, kth_bits):
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                in_range = ArithValue(col_i32) < row_len
                if in_range:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    key = aiter_twiddle_key(val)
                    key_before_kth = arith.cmpi(
                        arith.CmpIPredicate.ult, key, kth_bits
                    )
                    if key_before_kth:
                        pos = global_atomic_add_i32(
                            counter_slot(COUNTER_OUT_FRONT), c_one
                        )
                        if ArithValue(pos) < ArithValue(c_top_k):
                            buffer_ops.buffer_store(
                                col_i32, indices_rsrc, row_out + ArithValue(pos)
                            )
                    if ArithValue(key) == ArithValue(kth_bits):
                        back = global_atomic_add_i32(
                            counter_slot(COUNTER_OUT_BACK), c_one
                        )
                        if ArithValue(back) < ArithValue(local_k):
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

        def write_vec_block(vblk, local_k, kth_bits):
            vblk_i32 = arith.index_cast(T.i32, vblk)
            col_base = ArithValue(vblk_i32) * ArithValue(c_vec)
            vec = load_row_vec(col_base)
            process_loaded_write_vec(col_base, vec, local_k, kth_bits)

        def scan_pass(pass_id: int, current_k, current_bits, barrier_token: int):
            start_bit = max(32 - (pass_id + 1) * bits_per_pass, 0)
            previous_start_bit = max(32 - pass_id * bits_per_pass, 0)

            clear_local_histogram()
            if const_expr(scan_stages == 8):
                unroll_limit_idx = ArithValue(vec_blocks_idx) - ArithValue(
                    active_stride7_idx
                )
                staged_stride_idx = active_stride8_idx
            elif const_expr(scan_stages == 4):
                unroll_limit_idx = ArithValue(vec_blocks_idx) - ArithValue(
                    active_stride3_idx
                )
                staged_stride_idx = active_stride4_idx
            elif const_expr(scan_stages == 2):
                unroll_limit_idx = ArithValue(vec_blocks_idx) - ArithValue(
                    active_stride_idx
                )
                staged_stride_idx = active_stride2_idx
            else:
                unroll_limit_idx = vec_blocks_idx
                staged_stride_idx = active_stride_idx
            for vblk, pass_state in range(
                ArithValue(global_vec_tid_idx),
                unroll_limit_idx,
                ArithValue(staged_stride_idx),
                init=[global_vec_tid_idx],
            ):
                staged_scan_vec_blocks(
                    vblk,
                    pass_id,
                    start_bit,
                    previous_start_bit,
                    current_bits,
                )
                next_tail = ArithValue(vblk) + ArithValue(staged_stride_idx)
                pass_results = yield [next_tail]
            for vblk, pass_state in range(
                pass_results,
                ArithValue(vec_blocks_idx),
                ArithValue(active_stride_idx),
                init=[c_zero],
            ):
                scan_vec_block(vblk, pass_id, start_bit, previous_start_bit, current_bits)
                pass_results = yield [pass_state[0]]
            gpu.barrier()

            if single_part_active:
                choose_bucket_prefix(current_k)
            if ~single_part_active:
                flush_local_histogram(pass_id)
                row_barrier(barrier_token)
                load_global_histogram(pass_id)
                choose_bucket_prefix(current_k)

            chosen_bucket = fx.memref_load(s_meta, fx.Int32(SMEM_META_THRESHOLD))
            next_k = fx.memref_load(s_meta, fx.Int32(SMEM_META_K))
            next_len = fx.memref_load(s_meta, fx.Int32(SMEM_META_LEN))
            next_bits = ArithValue(current_bits) | ArithValue(
                arith.shli(chosen_bucket, fx.Int32(start_bit))
            )
            if const_expr(pass_id == num_passes - 1):
                unroll_limit_idx = ArithValue(vec_blocks_idx) - ArithValue(
                    active_stride3_idx
                )
                for vblk, write_state in range(
                    ArithValue(global_vec_tid_idx),
                    unroll_limit_idx,
                    ArithValue(active_stride4_idx),
                    init=[global_vec_tid_idx],
                ):
                    for unroll_id in range_constexpr(4):
                        write_vec_block(
                            ArithValue(vblk)
                            + ArithValue(
                                arith.index_cast(
                                    T.index,
                                    ArithValue(active_threads)
                                    * ArithValue(fx.Int32(unroll_id)),
                                )
                            ),
                            next_k,
                            next_bits,
                        )
                    next_tail = ArithValue(vblk) + ArithValue(active_stride4_idx)
                    write_results = yield [next_tail]
                for vblk, write_state in range(
                    write_results,
                    ArithValue(vec_blocks_idx),
                    ArithValue(active_stride_idx),
                    init=[c_zero],
                ):
                    write_vec_block(vblk, next_k, next_bits)
                    write_results = yield [write_state[0]]
            return next_k, next_len, next_bits

        global_vec_tid = part * ArithValue(c_block_i32) + tid
        global_vec_tid_idx = arith.index_cast(T.index, global_vec_tid)
        vec_blocks_i32 = ArithValue(
            row_len + ArithValue(c_vec) - ArithValue(c_one)
        ).shrui(fx.Int32(2))
        vec_blocks_idx = arith.index_cast(T.index, vec_blocks_i32)

        def one_cta_short_clone():
            # Faithful clone of the standalone one-CTA unordered radix-select
            # path (``topk_per_row_decode_radix_unordered_k512``). Runs entirely
            # within a single CTA (part 0): LDS-only histograms, a hierarchical
            # block scan to locate each radix threshold, and a direct LDS-counter
            # atomic-append write. Three order-preserving passes peel 11/11/10
            # bits of a twiddled fp32 key, so it requires a 2048-bin histogram
            # (``num_buckets == 2048``, i.e. bits_per_pass == 11). It reuses the
            # persistent kernel's existing LDS (``s_hist`` 2048 ints, ``s_scan``
            # 32 ints, ``s_meta`` 8 ints) with no extra shared memory.
            c_shift = fx.Int32(32 - 11)
            c_mid_shift = fx.Int32(10)
            c_bin_mask = fx.Int32((1 << 11) - 1)
            c_low_mask = fx.Int32((1 << 10) - 1)
            c_six = fx.Int32(6)
            c_seven = fx.Int32(7)
            c_thirtyone = fx.Int32(31)

            def ordered_key(val):
                key_val = (ArithValue(val) == ArithValue(c_zero_f32)).select(
                    c_zero_f32, val
                )
                bits = ArithValue(key_val).bitcast(T.i32)
                sign = ArithValue(bits).shrui(c_thirtyone)
                neg_key = ~ArithValue(bits)
                pos_key = ArithValue(bits) ^ ArithValue(c_sign_bit)
                return (ArithValue(sign) != ArithValue(c_zero)).select(neg_key, pos_key)

            def ordered_bucket(val):
                return ArithValue(ordered_key(val)).shrui(c_shift)

            def radix_bucket(val, shift, mask):
                return ArithValue(ArithValue(ordered_key(val)).shrui(shift)) & ArithValue(
                    mask
                )

            def clear_hist():
                for h in range(
                    ArithValue(tid_idx), ArithValue(c_bins_idx), ArithValue(c_block_idx)
                ):
                    fx.memref_store(c_zero, s_hist, arith.index_cast(T.i32, h))
                gpu.barrier()

            def choose_threshold(target_k, above_slot, threshold_slot):
                # Hierarchical inclusive block scan over the 2048-bin histogram;
                # each thread owns the contiguous bin pair (2*tid, 2*tid+1). The
                # kth-largest boundary is the first bucket whose inclusive prefix
                # passes ``K' = total - target_k`` (excl <= K' < incl).
                two_tid = ArithValue(tid) * ArithValue(c_two)
                c0 = fx.memref_load(s_hist, two_tid)
                c1 = fx.memref_load(s_hist, ArithValue(two_tid) + ArithValue(c_one))
                local_total = ArithValue(c0) + ArithValue(c1)

                wave_incl = wave_inclusive_scan_i32(local_total)
                wave_excl_thread = ArithValue(wave_incl) - ArithValue(local_total)

                if ArithValue(lane) == ArithValue(c_last_lane):
                    fx.memref_store(wave_incl, s_scan, wave)
                gpu.barrier()

                if wave == ArithValue(c_zero):
                    in16 = ArithValue(lane) < ArithValue(c_sixteen)
                    lane_safe = in16.select(lane, c_zero)
                    wtot = in16.select(fx.memref_load(s_scan, lane_safe), c_zero)
                    wincl = wave_inclusive_scan_i32(wtot)
                    wexcl = ArithValue(wincl) - ArithValue(wtot)
                    if in16:
                        fx.memref_store(
                            wexcl, s_scan, ArithValue(lane) + ArithValue(c_sixteen)
                        )
                gpu.barrier()

                wave_off = fx.memref_load(s_scan, ArithValue(wave) + ArithValue(c_sixteen))
                last_off = fx.memref_load(
                    s_scan, ArithValue(c_last_wave) + ArithValue(c_sixteen)
                )
                last_tot = fx.memref_load(s_scan, c_last_wave)
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

            if tid == ArithValue(c_zero):
                for meta_slot in range_constexpr(8):
                    fx.memref_store(c_zero, s_meta, fx.Int32(meta_slot))
            gpu.barrier()

            # Per-chunk bodies for the three radix passes plus the final scatter.
            # Each takes the chunk's first column index and its already-loaded
            # vec4 and feeds a fresh HBM load through the exact same logic.
            def hist_pass1_chunk(col_base, vec):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                    if ArithValue(col_i32) < row_len:
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        lds_atomic_add_i32(hist_base_ptr, ordered_bucket(val), c_one)

            def hist_pass2_chunk(col_base, vec, first_threshold):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                    if ArithValue(col_i32) < row_len:
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        if ArithValue(ordered_bucket(val)) == ArithValue(first_threshold):
                            lds_atomic_add_i32(
                                hist_base_ptr,
                                radix_bucket(val, c_mid_shift, c_bin_mask),
                                c_one,
                            )

            def hist_pass3_chunk(col_base, vec, first_threshold, second_threshold):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                    if ArithValue(col_i32) < row_len:
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        high_bucket = ordered_bucket(val)
                        mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                        matches_refine = (
                            ArithValue(high_bucket) == ArithValue(first_threshold)
                        ) & (ArithValue(mid_bucket) == ArithValue(second_threshold))
                        if matches_refine:
                            lds_atomic_add_i32(
                                hist_base_ptr,
                                radix_bucket(val, c_zero, c_low_mask),
                                c_one,
                            )

            def final_scatter_chunk(
                col_base, vec, first_threshold, second_threshold, third_threshold, num_needed
            ):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                    if ArithValue(col_i32) < row_len:
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
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
                            pos = lds_atomic_add_i32(meta_base_ptr, c_six, c_one)
                            buffer_ops.buffer_store(
                                col_i32, indices_rsrc, row_out + ArithValue(pos)
                            )
                        if at_boundary:
                            back = lds_atomic_add_i32(meta_base_ptr, c_seven, c_one)
                            if ArithValue(back) < ArithValue(num_needed):
                                out_pos = (
                                    ArithValue(c_top_k)
                                    - ArithValue(c_one)
                                    - ArithValue(back)
                                )
                                buffer_ops.buffer_store(
                                    col_i32, indices_rsrc, row_out + ArithValue(out_pos)
                                )

            # Reread driver: stream the whole valid row from HBM once per pass.
            # Each radix pass and the final scatter funnel through the same
            # per-chunk body. A register-resident variant (keeping the row live
            # across passes to skip the rereads) was implemented and measured: it
            # won ~5-10% only at the very top of the short tier (14K-16K), but
            # regressed every row below ~12K and raised the universal kernel's
            # VGPR high-water mark (40 -> 56), taxing long-tier occupancy. It was
            # rejected and removed; the reread clone already beats every
            # reference through L=16384.
            def reread_pass(chunk_fn):
                for vblk in range(
                    ArithValue(tid_idx),
                    ArithValue(vec_blocks_idx),
                    ArithValue(c_block_idx),
                ):
                    col_base = ArithValue(arith.index_cast(T.i32, vblk)) * ArithValue(c_vec)
                    chunk_fn(col_base, load_row_vec(col_base))

            # Pass 1: high 11 bits over the whole valid row.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass1_chunk(cb, v))
            gpu.barrier()
            choose_threshold(c_top_k, fx.Int32(0), fx.Int32(1))
            first_threshold = fx.memref_load(s_meta, fx.Int32(1))

            # Pass 2: mid 11 bits within the high boundary bucket.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass2_chunk(cb, v, first_threshold))
            gpu.barrier()
            first_above = fx.memref_load(s_meta, fx.Int32(0))
            need_after_first = ArithValue(c_top_k) - ArithValue(first_above)
            choose_threshold(need_after_first, fx.Int32(2), fx.Int32(3))
            second_threshold = fx.memref_load(s_meta, fx.Int32(3))

            # Pass 3: low 10 bits within the high+mid boundary.
            clear_hist()
            reread_pass(
                lambda cb, v: hist_pass3_chunk(
                    cb, v, first_threshold, second_threshold
                )
            )
            gpu.barrier()
            second_above = fx.memref_load(s_meta, fx.Int32(2))
            need_after_second = ArithValue(need_after_first) - ArithValue(second_above)
            choose_threshold(need_after_second, fx.Int32(4), fx.Int32(5))
            third_threshold = fx.memref_load(s_meta, fx.Int32(5))
            third_above = fx.memref_load(s_meta, fx.Int32(4))
            num_needed = ArithValue(need_after_second) - ArithValue(third_above)

            # Final phase: direct atomic-append write (LDS counters only).
            reread_pass(
                lambda cb, v: final_scatter_chunk(
                    cb,
                    v,
                    first_threshold,
                    second_threshold,
                    third_threshold,
                    num_needed,
                )
            )

        direct_fill = row_len <= ArithValue(c_top_k)
        direct_fill_active = (part == ArithValue(c_zero)) & direct_fill
        direct_fill_iters = direct_fill_active.select(fx.Index(TOP_K), fx.Index(0))
        for out_col in range(
            ArithValue(tid_idx), ArithValue(direct_fill_iters), ArithValue(c_block_idx)
        ):
            out_col_i32 = arith.index_cast(T.i32, out_col)
            valid = ArithValue(out_col_i32) < row_len
            out_val = valid.select(out_col_i32, c_neg_one)
            buffer_ops.buffer_store(
                out_val, indices_rsrc, row_out + ArithValue(out_col_i32)
            )

        if const_expr(short_clone):
            # Short tier: only part 0 runs the one-CTA clone; every part > 0
            # returns immediately without touching a row barrier or any
            # workspace counter. The persistent multi-block path handles the
            # mid/long tiers (active_parts > 1).
            short_active = (
                single_part_active
                & (part == ArithValue(c_zero))
                & (row_len > ArithValue(c_top_k))
            )
            if short_active:
                one_cta_short_clone()
            cooperative_active = (
                (row_len > ArithValue(c_top_k))
                & (part < ArithValue(active_parts))
                & (~single_part_active)
            )
        else:
            cooperative_active = (row_len > ArithValue(c_top_k)) & (
                part < ArithValue(active_parts)
            )
        if cooperative_active:
            local_k = c_top_k
            local_len = row_len
            kth_bits = c_zero

            for pass_id in range_constexpr(num_passes):
                local_k, local_len, kth_bits = scan_pass(
                    pass_id, local_k, kth_bits, pass_id + 1
                )

    @flyc.jit
    def launch_topk_per_row_decode_aiter_persistent_k512(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_y = arith.index_cast(T.index, num_rows)
        topk_per_row_decode_aiter_persistent_k512_kernel(
            logits,
            next_n,
            seq_lens,
            indices,
            workspace,
            num_rows,
            stride0,
            stride1,
        ).launch(
            grid=(blocks_per_row, grid_y, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_topk_per_row_decode_aiter_persistent_k512
