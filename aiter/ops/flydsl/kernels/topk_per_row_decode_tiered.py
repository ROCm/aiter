"""FlyDSL decode TopK-per-row kernel (tiered persistent multi-block radix-select)

Computes an unordered Top-K index set per decode row, fusing a single-workgroup and
a multi-block radix-select into one persistent launch grid=(blocks_per_row, num_rows)
that picks a per-row strategy by valid length. Which is required when decode batch's
sequences differ in length. Each row derives how many of its blocks_per_row
workgroups cooperate (active_parts); the rest return immediately.

Inputs/outputs:
  - logits: fp32, logical shape (num_rows, L), strides (stride0, stride1) with
    stride1 == 1 (contiguous within a row).
  - seq_lens: int32 causal lengths per sequence; row r scores sequence r // next_n at
    decode slot r % next_n, valid length seq_len - next_n + slot + 1.
  - indices: flattened int32 output with shape (num_rows, top_k); each row writes its
    unordered Top-K index set. A row with fewer than top_k valid entries is
    identity-filled and padded with -1.
  - workspace: row-major int32 scratch sized by topk_workspace_slots(num_rows,
    bits_per_pass). The multi-block tiers merge per-block LDS histograms into its
    pass-private global histograms over an inter-workgroup acquire/release barrier and
    coordinate through its counters; the single-workgroup tier never touches it.

Paths (per row, by valid length row_len):
  - short (row_len <= short_max): active_parts = 1; part 0 runs the whole radix-select
    in one workgroup — LDS-only histograms, no inter-workgroup barrier, no workspace
    round-trip.
  - mid (short_max < row_len <= mid_max): active_parts = min(blocks_per_row, mid_cap).
  - long (row_len > mid_max): active_parts = min(blocks_per_row, long_cap).

Constraints:
  - logits are fp32; the order-preserving radix key twiddle is fp32-specific.
  - bits_per_pass is 10 or 11; the short tier requires 11 bits (2048-bin LDS histogram).
  - BLOCK_THREADS is fixed at 1024 (wave64); the histogram/scan layout and the
    occupancy deadlock guard rely on it.
  - workspace must be zeroed before any launch that enters a multi-block tier; its
    counters and histograms accumulate from zero (needs_workspace_zero reports when).
  - The row barrier spins (s_sleep), so a row's blocks_per_row workgroups must be
    co-resident. This is a regular launch, not hipLaunchCooperativeKernel: it is safe
    only because HIP flattens the grid x-fastest, so a row's parts launch contiguously
    and drain in order — scheduler launch order, not a cooperative guarantee. Keep
    num_rows * blocks_per_row co-resident; the wrapper's deadlock guard enforces this,
    forcing larger batches onto the barrier-free short tier.
"""

from functools import cache
from typing import Any

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm, scf
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

# HW max block size; also assumed by the bucket scan (2 bins/thread -> 2048 bins)
# and the occupancy=2 deadlock guard. Changing it breaks both.
BLOCK_THREADS = 1024
WARP_SIZE = 64
LOAD_VEC = 4
# Default histogram-scan staging (one of 1/2/4/8)
SCAN_STAGES = 2

# 128B-spaced inter-workgroup counter groups (32 int32 == 128B each), kept in the int32 workspace.
COUNTER_STRIDE = 32
COUNTER_SLOTS = 6 * COUNTER_STRIDE
COUNTER_ARRIVALS = 2 * COUNTER_STRIDE
COUNTER_OUT_FRONT = 3 * COUNTER_STRIDE
COUNTER_OUT_BACK = 4 * COUNTER_STRIDE
COUNTER_PASS_DONE = 5 * COUNTER_STRIDE

SMEM_META_K = 0
SMEM_META_LEN = 1
SMEM_META_THRESHOLD = 2
SMEM_META_ABOVE = 3

# Short-tier one-workgroup metadata reuses the same 8-int LDS block after zeroing.
SMEM_META_SHORT_FIRST_ABOVE = 0
SMEM_META_SHORT_FIRST_THRESHOLD = 1
SMEM_META_SHORT_SECOND_ABOVE = 2
SMEM_META_SHORT_SECOND_THRESHOLD = 3
SMEM_META_SHORT_THIRD_ABOVE = 4
SMEM_META_SHORT_THIRD_THRESHOLD = 5
SMEM_META_SHORT_FRONT_COUNT = 6
SMEM_META_SHORT_BACK_COUNT = 7


def _num_passes(bits_per_pass: int) -> int:
    return (32 + bits_per_pass - 1) // bits_per_pass


def topk_workspace_slots(
    num_rows: int,
    bits_per_pass: int = 11,
) -> int:
    """Return int32 workspace slots for the tiered path (row-major, per row)."""
    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    row_slots = COUNTER_SLOTS + _num_passes(bits_per_pass) * (1 << bits_per_pass)
    return int(num_rows) * row_slots


def needs_workspace_zero(max_row_len: int, top_k: int, short_max: int) -> bool:
    """Return whether any row can enter the persistent multi-block path."""
    return max_row_len > max(short_max, top_k)


@cache
def create_topk_per_row_decode_tiered_kernel(
    blocks_per_row: int = 8,
    *,
    top_k: int,
    bits_per_pass: int = 11,
    scan_stages: int = SCAN_STAGES,
    tiered: bool = True,
    tiered_short_max: int = 16384,
    tiered_mid_cap: int = 16,
    tiered_mid_max: int = 65536,
    tiered_long_cap: int = 32,
    mask_non_finite: bool = True,
) -> Any:
    short_max = tiered_short_max
    mid_cap = tiered_mid_cap
    mid_max = tiered_mid_max
    long_cap = tiered_long_cap
    """Return a cached tiered persistent decode radix-select launcher."""

    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    if scan_stages not in (1, 2, 4, 8):
        raise ValueError(f"scan_stages must be one of (1, 2, 4, 8), got {scan_stages}")
    if not 2 <= blocks_per_row <= 32:
        raise ValueError(f"blocks_per_row must be in [2, 32], got {blocks_per_row}")
    if mid_cap < 2 or long_cap < 2:
        raise ValueError(f"mid_cap/long_cap must be >= 2, got {mid_cap}/{long_cap}")
    if mid_max < short_max:
        raise ValueError(f"mid_max must be >= short_max, got {mid_max} < {short_max}")

    # The short tier runs the standalone one-workgroup radix-select (11/11/10 ordered key, 2048-bin LDS
    # histogram), so it is only available when bits_per_pass == 11.
    short_tier = tiered and bits_per_pass == 11
    block_threads = BLOCK_THREADS
    red_slots = (block_threads + WARP_SIZE - 1) // WARP_SIZE
    num_passes = _num_passes(bits_per_pass)
    num_buckets = 1 << bits_per_pass
    row_workspace_slots = COUNTER_SLOTS + num_passes * num_buckets
    kernel_name = (
        f"topk_per_row_decode_persistent_k{top_k}_"
        f"bpp{bits_per_pass}_g{blocks_per_row}_v2"
        f"_stage{scan_stages}"
        f"{'_tiered' if tiered else ''}"
        f"{f'_s{tiered_short_max}_mc{tiered_mid_cap}' if tiered else ''}"
        f"{f'_mm{tiered_mid_max}_lc{tiered_long_cap}' if tiered else ''}"
        f"{'_1wg' if short_tier else ''}"
        f"{'_mf' if mask_non_finite else ''}"
    )

    @fx.struct
    class SharedStorage:
        s_hist: fx.Array[fx.Int32, num_buckets, 16]
        s_scan: fx.Array[fx.Int32, red_slots * 2, 16]
        s_meta: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def topk_per_row_decode_tiered_kernel(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        stride0: fx.Int32,
    ) -> None:
        block_x = gpu.block_id("x")
        block_y = gpu.block_id("y")
        thread_x = gpu.thread_id("x")
        part = ArithValue(arith.index_cast(T.i32, block_x))
        row = ArithValue(arith.index_cast(T.i32, block_y))
        tid = ArithValue(arith.index_cast(T.i32, thread_x))
        tid_idx = arith.index_cast(T.index, thread_x)
        lane = tid % ArithValue(fx.Int32(WARP_SIZE))
        wave = tid // ArithValue(fx.Int32(WARP_SIZE))

        c_zero = fx.Int32(0)
        c_one = fx.Int32(1)
        c_two = fx.Int32(2)
        c_four = fx.Int32(4)
        c_red_slots = fx.Int32(red_slots)
        c_last_wave = fx.Int32(red_slots - 1)
        c_last_lane = fx.Int32(WARP_SIZE - 1)
        c_vec = fx.Int32(LOAD_VEC)
        c_top_k = fx.Int32(top_k)
        c_block_i32 = fx.Int32(block_threads)
        c_block_idx = fx.Index(block_threads)
        c_bins_i32 = fx.Int32(num_buckets)
        c_bins_idx = fx.Index(num_buckets)
        c_parts = fx.Int32(blocks_per_row)
        c_sign_bit = fx.Int32(-2147483648)
        c_exp_mask = fx.Int32(0x7F800000)  # fp32 exponent bits (all-ones => inf/NaN)
        c_neg_fltmax = fx.Float32(-3.4028234663852886e38)  # torch.finfo(f32).min
        c_neg_one = fx.Int32(-1)
        c_zero_f32 = fx.Float32(0.0)
        c_row_ws = fx.Int32(row_workspace_slots)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_hist = lds.s_hist.view(fx.make_layout(num_buckets, 1))
        s_scan = lds.s_scan.view(fx.make_layout(red_slots * 2, 1))
        s_meta = lds.s_meta.view(fx.make_layout(8, 1))

        logits_rsrc = buffer_ops.create_buffer_resource(logits, max_size=True)
        seq_lens_rsrc = buffer_ops.create_buffer_resource(seq_lens, max_size=True)
        indices_rsrc = buffer_ops.create_buffer_resource(indices, max_size=True)
        workspace_rsrc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        workspace_base_idx = buffer_ops.extract_base_index(workspace, address_space=1)

        hist_base_ptr = fx.ptrtoint(lds.s_hist.ptr)
        meta_base_ptr = fx.ptrtoint(lds.s_meta.ptr)

        # Decode row geometry.
        seq_row = ArithValue(row) // ArithValue(next_n)
        slot = ArithValue(row) - ArithValue(seq_row) * ArithValue(next_n)
        seq_len = ArithValue(
            buffer_ops.buffer_load(seq_lens_rsrc, seq_row, vec_width=1, dtype=T.i32)
        )
        row_len = (
            ArithValue(seq_len)
            - ArithValue(next_n)
            + ArithValue(slot)
            + ArithValue(c_one)
        )
        row_len = (ArithValue(row_len) > ArithValue(c_zero)).select(row_len, c_zero)
        row_base = ArithValue(row) * ArithValue(stride0)
        row_out = ArithValue(row) * ArithValue(c_top_k)
        row_ws_base = ArithValue(row) * ArithValue(c_row_ws)

        # Per-row active-part cap over the fixed grid (excess blocks return immediately).
        active_parts = c_parts
        if const_expr(tiered):
            # Per-bucket active-part caps over the fixed launch grid (excess
            # blocks return immediately). Short rows run the single-workgroup short tier;
            # the mid and long persistent buckets cap how many of the grid's
            # blocks participate, trading CU coverage against barrier/atomic
            # contention. Caps are clamped to the grid (``c_parts``).
            c_short = fx.Int32(short_max)
            c_mid = fx.Int32(mid_max)
            c_mid_cap = fx.Int32(mid_cap)
            c_long_cap = fx.Int32(long_cap)
            mid_parts = (ArithValue(c_parts) < ArithValue(c_mid_cap)).select(
                c_parts, c_mid_cap
            )
            long_parts = (ArithValue(c_parts) < ArithValue(c_long_cap)).select(
                c_parts, c_long_cap
            )
            active_parts = (ArithValue(row_len) <= ArithValue(c_short)).select(
                c_one,
                (ArithValue(row_len) <= ArithValue(c_mid)).select(
                    mid_parts, long_parts
                ),
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

        def counter_slot(slot_const: int):
            return ArithValue(row_ws_base) + ArithValue(fx.Int32(slot_const))

        def histogram_slot(pass_id: int, bin_i32):
            return (
                ArithValue(row_ws_base)
                + ArithValue(fx.Int32(COUNTER_SLOTS + pass_id * num_buckets))
                + ArithValue(bin_i32)
            )

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
            # Volatile agent-scoped acquire load for the row-barrier spin. The
            # matching release publish below makes histogram updates visible to
            # peer workgroups without issuing a read-modify-write on the polled slot.
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
            w = scf.WhileOp([T.i32], [arith.unwrap(c_zero)])
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
            # Intentional no-drain acquire/release protocol: workgroup barriers
            # bracket local LDS work, the last workgroup release-publishes pass_done,
            # and peers spin with acquire loads. A full waitcnt drain here is
            # performance/correctness sensitive.
            token_value = fx.Int32(token)
            target_arrivals = ArithValue(token_value) * ArithValue(active_parts)
            gpu.barrier()
            if tid == ArithValue(c_zero):
                prev = global_atomic_add_i32(
                    counter_slot(COUNTER_ARRIVALS), c_one, llvm.AtomicOrdering.acq_rel
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

        def mask_nonfinite(val):
            if const_expr(not mask_non_finite):
                return val
            bits = ArithValue(val).bitcast(T.i32)
            is_nonfinite = ArithValue(
                ArithValue(bits) & ArithValue(c_exp_mask)
            ) == ArithValue(c_exp_mask)
            return is_nonfinite.select(c_neg_fltmax, val)

        def radix_twiddle_key(val):
            # Map larger fp32 values to smaller unsigned keys so ascending
            # bucket scans select descending values. Normalize signed zero to
            # keep tie handling value-equivalent.
            val = mask_nonfinite(val)
            key_val = (ArithValue(val) == ArithValue(c_zero_f32)).select(
                c_zero_f32, val
            )
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

        def clear_local_histogram():
            for hist_idx in range(
                ArithValue(tid_idx), ArithValue(c_bins_idx), ArithValue(c_block_idx)
            ):
                fx.memref_store(c_zero, s_hist, arith.index_cast(T.i32, hist_idx))
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
            # Multi-block ascending block scan over the LDS histogram; each thread owns a bin pair.
            first_bin = ArithValue(tid) * ArithValue(c_two)
            bin0_valid = ArithValue(first_bin) < ArithValue(c_bins_i32)
            bin1 = ArithValue(first_bin) + ArithValue(c_one)
            bin1_valid = ArithValue(bin1) < ArithValue(c_bins_i32)
            safe0 = bin0_valid.select(first_bin, c_zero)
            safe1 = bin1_valid.select(bin1, c_zero)
            c0 = bin0_valid.select(fx.memref_load(s_hist, safe0), c_zero)
            c1 = bin1_valid.select(fx.memref_load(s_hist, safe1), c_zero)
            local_total = ArithValue(c0) + ArithValue(c1)

            wave_incl = wave_inclusive_scan_i32(local_total)
            wave_excl_thread = ArithValue(wave_incl) - ArithValue(local_total)

            if ArithValue(lane) == ArithValue(c_last_lane):
                fx.memref_store(wave_incl, s_scan, wave)
            gpu.barrier()

            if wave == ArithValue(c_zero):
                in16 = ArithValue(lane) < ArithValue(c_red_slots)
                lane_safe = in16.select(lane, c_zero)
                wtot = in16.select(fx.memref_load(s_scan, lane_safe), c_zero)
                wincl = wave_inclusive_scan_i32(wtot)
                wexcl = ArithValue(wincl) - ArithValue(wtot)
                if in16:
                    fx.memref_store(
                        wexcl, s_scan, ArithValue(lane) + ArithValue(c_red_slots)
                    )
            gpu.barrier()

            wave_off = fx.memref_load(
                s_scan, ArithValue(wave) + ArithValue(c_red_slots)
            )
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
            # Vectorized reload
            n_vec = num_buckets // LOAD_VEC
            c_nvec_idx = fx.Index(n_vec)
            for grp in range(
                ArithValue(tid_idx), ArithValue(c_nvec_idx), ArithValue(c_block_idx)
            ):
                base_bin = ArithValue(arith.index_cast(T.i32, grp)) * ArithValue(c_vec)
                vec = buffer_ops.buffer_load(
                    workspace_rsrc,
                    histogram_slot(pass_id, base_bin),
                    vec_width=LOAD_VEC,
                    dtype=T.i32,
                )
                for j in range_constexpr(LOAD_VEC):
                    total = vector.extract(
                        vec, static_position=[j], dynamic_position=[]
                    )
                    fx.memref_store(
                        total, s_hist, ArithValue(base_bin) + ArithValue(fx.Int32(j))
                    )
            gpu.barrier()

        def process_loaded_scan_vec(
            col_base,
            vec,
            pass_id: int,
            start_bit: int,
            previous_start_bit: int,
            current_bits,
        ):
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                if ArithValue(col_i32) < ArithValue(row_len):
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    key = radix_twiddle_key(val)
                    matches_prefix = True
                    if const_expr(pass_id != 0):
                        matches_prefix = ArithValue(
                            prefix_for_key(key, previous_start_bit)
                        ) == ArithValue(current_bits)
                    if matches_prefix:
                        lds_atomic_add_i32(
                            hist_base_ptr, bucket_for_key(key, start_bit), c_one
                        )

        def scan_vec_block(
            vblk, pass_id: int, start_bit: int, previous_start_bit: int, current_bits
        ):
            col_base = ArithValue(arith.index_cast(T.i32, vblk)) * ArithValue(c_vec)
            process_loaded_scan_vec(
                col_base,
                load_row_vec(col_base),
                pass_id,
                start_bit,
                previous_start_bit,
                current_bits,
            )

        def staged_scan_vec_blocks(
            vblk,
            pass_id: int,
            start_bit: int,
            previous_start_bit: int,
            current_bits,
        ):
            if const_expr(scan_stages == 1):
                strides = [fx.Index(0)]
            elif const_expr(scan_stages == 2):
                strides = [fx.Index(0), active_stride_idx]
            elif const_expr(scan_stages == 4):
                strides = [
                    fx.Index(0),
                    active_stride_idx,
                    active_stride2_idx,
                    active_stride3_idx,
                ]
            else:
                strides = [
                    fx.Index(0),
                    active_stride_idx,
                    active_stride2_idx,
                    active_stride3_idx,
                    active_stride4_idx,
                    ArithValue(active_stride4_idx) + ArithValue(active_stride_idx),
                    ArithValue(active_stride4_idx) + ArithValue(active_stride2_idx),
                    active_stride7_idx,
                ]
            cols_v = [
                ArithValue(arith.index_cast(T.i32, ArithValue(vblk) + ArithValue(s)))
                * ArithValue(c_vec)
                for s in strides
            ]
            vecs = [load_row_vec(cb) for cb in cols_v]
            for cb, vc in zip(cols_v, vecs):
                process_loaded_scan_vec(
                    cb, vc, pass_id, start_bit, previous_start_bit, current_bits
                )

        def process_loaded_write_vec(col_base, vec, local_k, kth_bits):
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                if ArithValue(col_i32) < ArithValue(row_len):
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    key = radix_twiddle_key(val)
                    if arith.cmpi(arith.CmpIPredicate.ult, key, kth_bits):
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
                                col_i32, indices_rsrc, row_out + ArithValue(out_pos)
                            )

        def write_vec_block(vblk, local_k, kth_bits):
            col_base = ArithValue(arith.index_cast(T.i32, vblk)) * ArithValue(c_vec)
            process_loaded_write_vec(
                col_base, load_row_vec(col_base), local_k, kth_bits
            )

        global_vec_tid = part * ArithValue(c_block_i32) + tid
        global_vec_tid_idx = arith.index_cast(T.index, global_vec_tid)
        vec_blocks_i32 = ArithValue(
            ArithValue(row_len) + ArithValue(c_vec) - ArithValue(c_one)
        ).shrui(fx.Int32(2))
        vec_blocks_idx = arith.index_cast(T.index, vec_blocks_i32)

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
                    vblk, pass_id, start_bit, previous_start_bit, current_bits
                )
                pass_results = yield [ArithValue(vblk) + ArithValue(staged_stride_idx)]
            for vblk, pass_state in range(
                pass_results,
                ArithValue(vec_blocks_idx),
                ArithValue(active_stride_idx),
                init=[c_zero],
            ):
                scan_vec_block(
                    vblk, pass_id, start_bit, previous_start_bit, current_bits
                )
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
                    write_results = yield [
                        ArithValue(vblk) + ArithValue(active_stride4_idx)
                    ]
                for vblk, write_state in range(
                    write_results,
                    ArithValue(vec_blocks_idx),
                    ArithValue(active_stride_idx),
                    init=[c_zero],
                ):
                    write_vec_block(vblk, next_k, next_bits)
                    write_results = yield [write_state[0]]
            return next_k, next_len, next_bits

        def one_workgroup_short_tier():
            # Faithful copy of the standalone one-workgroup unordered radix-select.
            # Runs entirely within a single workgroup (part 0):
            # LDS-only histograms, a hierarchical block scan to locate each
            # radix threshold, and a direct LDS-counter atomic-append write.
            # Unlike the persistent multi-block path, this short tier uses the
            # standalone ascending ordered key and total-k threshold convention.
            # Three order-preserving passes peel 11/11/10 bits of that key, so
            # it requires a 2048-bin histogram
            # (``num_buckets == 2048``, i.e. bits_per_pass == 11). It reuses the
            # persistent kernel's existing LDS (``s_hist`` 2048 ints,
            # ``s_scan`` 32 ints, ``s_meta`` 8 ints) with no extra shared memory.
            c_shift = fx.Int32(32 - 11)
            c_mid_shift = fx.Int32(10)
            c_bin_mask = fx.Int32((1 << 11) - 1)
            c_low_mask = fx.Int32((1 << 10) - 1)
            c_thirtyone = fx.Int32(31)

            def ordered_key(val):
                val = mask_nonfinite(val)
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
                return ArithValue(
                    ArithValue(ordered_key(val)).shrui(shift)
                ) & ArithValue(mask)

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
                    in16 = ArithValue(lane) < ArithValue(c_red_slots)
                    lane_safe = in16.select(lane, c_zero)
                    wtot = in16.select(fx.memref_load(s_scan, lane_safe), c_zero)
                    wincl = wave_inclusive_scan_i32(wtot)
                    wexcl = ArithValue(wincl) - ArithValue(wtot)
                    if in16:
                        fx.memref_store(
                            wexcl, s_scan, ArithValue(lane) + ArithValue(c_red_slots)
                        )
                gpu.barrier()

                wave_off = fx.memref_load(
                    s_scan, ArithValue(wave) + ArithValue(c_red_slots)
                )
                last_off = fx.memref_load(
                    s_scan, ArithValue(c_last_wave) + ArithValue(c_red_slots)
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
                    if ArithValue(col_i32) < ArithValue(row_len):
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        lds_atomic_add_i32(hist_base_ptr, ordered_bucket(val), c_one)

            def hist_pass2_chunk(col_base, vec, first_threshold):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                    if ArithValue(col_i32) < ArithValue(row_len):
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        if ArithValue(ordered_bucket(val)) == ArithValue(
                            first_threshold
                        ):
                            lds_atomic_add_i32(
                                hist_base_ptr,
                                radix_bucket(val, c_mid_shift, c_bin_mask),
                                c_one,
                            )

            def hist_pass3_chunk(col_base, vec, first_threshold, second_threshold):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                    if ArithValue(col_i32) < ArithValue(row_len):
                        val = vector.extract(
                            vec, static_position=[j], dynamic_position=[]
                        )
                        high_bucket = ordered_bucket(val)
                        mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                        if (ArithValue(high_bucket) == ArithValue(first_threshold)) & (
                            ArithValue(mid_bucket) == ArithValue(second_threshold)
                        ):
                            lds_atomic_add_i32(
                                hist_base_ptr,
                                radix_bucket(val, c_zero, c_low_mask),
                                c_one,
                            )

            def final_scatter_chunk(
                col_base,
                vec,
                first_threshold,
                second_threshold,
                third_threshold,
                num_needed,
            ):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = ArithValue(col_base) + ArithValue(fx.Int32(j))
                    if ArithValue(col_i32) < ArithValue(row_len):
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
                        above_low = ArithValue(low_bucket) > ArithValue(third_threshold)
                        at_low = ArithValue(low_bucket) == ArithValue(third_threshold)
                        strictly_above = above_first | (
                            at_first & (above_second | (at_second & above_low))
                        )
                        at_boundary = at_first & at_second & at_low
                        if strictly_above:
                            pos = lds_atomic_add_i32(
                                meta_base_ptr,
                                fx.Int32(SMEM_META_SHORT_FRONT_COUNT),
                                c_one,
                            )
                            buffer_ops.buffer_store(
                                col_i32, indices_rsrc, row_out + ArithValue(pos)
                            )
                        if at_boundary:
                            back = lds_atomic_add_i32(
                                meta_base_ptr,
                                fx.Int32(SMEM_META_SHORT_BACK_COUNT),
                                c_one,
                            )
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
            # This keeps the short tier's VGPR footprint low while sharing the
            # same per-chunk logic across radix passes and final scatter.
            def reread_pass(chunk_fn):
                for vblk in range(
                    ArithValue(tid_idx),
                    ArithValue(vec_blocks_idx),
                    ArithValue(c_block_idx),
                ):
                    col_base = ArithValue(arith.index_cast(T.i32, vblk)) * ArithValue(
                        c_vec
                    )
                    chunk_fn(col_base, load_row_vec(col_base))

            # Pass 1: high 11 bits over the whole valid row.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass1_chunk(cb, v))
            gpu.barrier()
            choose_threshold(
                c_top_k,
                fx.Int32(SMEM_META_SHORT_FIRST_ABOVE),
                fx.Int32(SMEM_META_SHORT_FIRST_THRESHOLD),
            )
            first_threshold = fx.memref_load(
                s_meta, fx.Int32(SMEM_META_SHORT_FIRST_THRESHOLD)
            )

            # Pass 2: mid 11 bits within the high boundary bucket.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass2_chunk(cb, v, first_threshold))
            gpu.barrier()
            first_above = fx.memref_load(s_meta, fx.Int32(SMEM_META_SHORT_FIRST_ABOVE))
            need_after_first = ArithValue(c_top_k) - ArithValue(first_above)
            choose_threshold(
                need_after_first,
                fx.Int32(SMEM_META_SHORT_SECOND_ABOVE),
                fx.Int32(SMEM_META_SHORT_SECOND_THRESHOLD),
            )
            second_threshold = fx.memref_load(
                s_meta, fx.Int32(SMEM_META_SHORT_SECOND_THRESHOLD)
            )

            # Pass 3: low 10 bits within the high+mid boundary.
            clear_hist()
            reread_pass(
                lambda cb, v: hist_pass3_chunk(cb, v, first_threshold, second_threshold)
            )
            gpu.barrier()
            second_above = fx.memref_load(
                s_meta, fx.Int32(SMEM_META_SHORT_SECOND_ABOVE)
            )
            need_after_second = ArithValue(need_after_first) - ArithValue(second_above)
            choose_threshold(
                need_after_second,
                fx.Int32(SMEM_META_SHORT_THIRD_ABOVE),
                fx.Int32(SMEM_META_SHORT_THIRD_THRESHOLD),
            )
            third_threshold = fx.memref_load(
                s_meta, fx.Int32(SMEM_META_SHORT_THIRD_THRESHOLD)
            )
            third_above = fx.memref_load(s_meta, fx.Int32(SMEM_META_SHORT_THIRD_ABOVE))
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

        # Direct-fill: rows with row_len <= top_k (part 0 only) emit identity indices + -1.
        direct_fill = ArithValue(row_len) <= ArithValue(c_top_k)
        direct_fill_active = (part == ArithValue(c_zero)) & direct_fill
        direct_fill_iters = direct_fill_active.select(fx.Index(top_k), fx.Index(0))
        for out_col in range(
            ArithValue(tid_idx), ArithValue(direct_fill_iters), ArithValue(c_block_idx)
        ):
            out_col_i32 = arith.index_cast(T.i32, out_col)
            valid = ArithValue(out_col_i32) < ArithValue(row_len)
            out_val = valid.select(out_col_i32, c_neg_one)
            buffer_ops.buffer_store(
                out_val, indices_rsrc, row_out + ArithValue(out_col_i32)
            )

        if const_expr(short_tier):
            short_active = (
                single_part_active
                & (part == ArithValue(c_zero))
                & (ArithValue(row_len) > ArithValue(c_top_k))
            )
            if short_active:
                one_workgroup_short_tier()
            persistent_active = (
                (ArithValue(row_len) > ArithValue(c_top_k))
                & (part < ArithValue(active_parts))
                & (~single_part_active)
            )
        else:
            persistent_active = (ArithValue(row_len) > ArithValue(c_top_k)) & (
                part < ArithValue(active_parts)
            )

        if persistent_active:
            local_k = c_top_k
            kth_bits = c_zero
            for pass_id in range_constexpr(num_passes):
                local_k, _local_len, kth_bits = scan_pass(
                    pass_id, local_k, kth_bits, pass_id + 1
                )

    @flyc.jit
    def launcher(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid_y = arith.index_cast(T.index, num_rows)
        topk_per_row_decode_tiered_kernel(
            logits, next_n, seq_lens, indices, workspace, stride0
        ).launch(
            grid=(blocks_per_row, grid_y, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launcher
