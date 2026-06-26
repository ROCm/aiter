# SPDX-License-Identifier: MIT

"""FlyDSL decode TopK-per-row kernel builders.

This module contains the single-CTA unordered radix-select implementation used
for non-tiered TopK sizes. K=512 is served by the tiered persistent kernel, so
the non-tiered ordered compact/bitonic paths are intentionally absent.

The selected indices are emitted as an unordered set equivalent to ``torch.topk``
by value. Consumers must not rely on slot order.
"""

import functools

from flydsl._mlir.dialects import llvm
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
from flydsl.expr.arith import ArithValue
from flydsl.expr.typing import T

from .tensor_shim import GTensor

BLOCK_THREADS = 1024
WARP_SIZE = 64
RED_SLOTS = (BLOCK_THREADS + WARP_SIZE - 1) // WARP_SIZE
# Width of the vectorized (stride-1) global load used by the radix passes.
# Reading 4 contiguous fp32 logits per buffer instruction quarters the VMEM
# instruction count of every full-row pass, which is the dominant cost of the
# one-CTA-per-row decode kernel (cost grows ~linearly with row length).
LOAD_VEC = 4
RADIX_BITS = 11
RADIX_BINS = 1 << RADIX_BITS


@functools.lru_cache(maxsize=32)
def create_topk_per_row_decode_kernel(top_k: int):
    """Return a cached unordered FlyDSL launcher specialized for ``top_k``.

    Launcher signature matches the existing decode interface:
    ``(logits, next_n, seqLens, indices, numRows, stride0, stride1, stream=...)``.
    The output is an unordered Top-K set. K=512 is intentionally handled by the
    tiered persistent kernel, which embeds the one-CTA clone for short rows.
    """

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")
    if top_k == 512:
        raise NotImplementedError(
            "K=512 FlyDSL decode TopK uses the tiered persistent kernel; "
            "the non-tiered K=512 path was pruned"
        )

    return _create_topk_per_row_decode_radix_unordered(top_k)


def _create_topk_per_row_decode_radix_unordered(top_k: int):
    """AITER-style one-block radix-select with unordered (set) output.

    Three order-preserving radix passes (11/11/10 bits over a twiddled fp32 key)
    narrow the row to the exact kth-largest 32-bit boundary. The final phase
    then writes results directly:

    * keys strictly above the boundary are guaranteed Top-K members and are
      atomic-appended to the front of the output;
    * keys exactly at the boundary fill the remaining ``num_needed`` slots from
      the back.

    There is no candidate buffer and no sort. Output slot order is
    arbitrary but the index *set* matches ``torch.topk`` (ties resolved by
    value). LDS footprint is two 2048-int histograms (16 KiB) regardless of
    ``top_k``, well within gfx942's 64 KiB.
    """

    @fx.struct
    class SharedStorage:
        s_hist: fx.Array[fx.Int32, RADIX_BINS, 16]
        s_hist2: fx.Array[fx.Int32, RADIX_BINS, 16]
        s_meta: fx.Array[fx.Int32, 8, 16]

    kernel_name = f"topk_per_row_decode_radix_unordered_k{top_k}"

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def topk_per_row_decode_radix_unordered_kernel(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
    ):
        row = ArithValue(fx.block_idx.x)
        tid = ArithValue(fx.thread_idx.x)
        tid_idx = arith.index_cast(T.index, fx.thread_idx.x)
        lane = fx.thread_idx.x % WARP_SIZE
        wave = fx.thread_idx.x // WARP_SIZE

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_hist = lds.s_hist.view(fx.make_layout(RADIX_BINS, 1))
        s_hist2 = lds.s_hist2.view(fx.make_layout(RADIX_BINS, 1))
        s_meta = lds.s_meta.view(fx.make_layout(8, 1))

        logits_t = GTensor(logits, dtype=T.f32, shape=(-1,))
        logits_rsrc = logits_t.rsrc
        seq_lens_t = GTensor(seq_lens, dtype=T.i32, shape=(-1,))
        indices_rsrc = GTensor(indices, dtype=T.i32, shape=(-1,)).rsrc

        c_zero = arith.constant(0, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_two = arith.constant(2, type=T.i32)
        c_four = arith.constant(4, type=T.i32)
        c_sixteen = arith.constant(RED_SLOTS, type=T.i32)
        c_last_wave = arith.constant(RED_SLOTS - 1, type=T.i32)
        c_last_lane = arith.constant(WARP_SIZE - 1, type=T.i32)
        c_warp_i32 = arith.constant(WARP_SIZE, type=T.i32)
        c_six = arith.constant(6, type=T.i32)
        c_seven = arith.constant(7, type=T.i32)
        c_neg_one = arith.constant(-1, type=T.i32)
        c_vec = arith.constant(LOAD_VEC, type=T.i32)
        c_block_idx = fx.Index(BLOCK_THREADS)
        c_block_i32 = arith.constant(BLOCK_THREADS, type=T.i32)
        c_bins_i32 = arith.constant(RADIX_BINS, type=T.i32)
        c_top_k = arith.constant(top_k, type=T.i32)
        c_top_k_idx = fx.Index(top_k)
        c_bins_idx = fx.Index(RADIX_BINS)
        c_shift = arith.constant(32 - RADIX_BITS, type=T.i32)
        c_mid_shift = arith.constant(10, type=T.i32)
        c_bin_mask = arith.constant(RADIX_BINS - 1, type=T.i32)
        c_low_mask = arith.constant((1 << 10) - 1, type=T.i32)
        c_sign_bit = arith.constant(-2147483648, type=T.i32)
        c_zero_f32 = fx.Float32(0.0)
        c_neg_inf = fx.Float32(float("-inf"))

        hist_base = fx.ptrtoint(lds.s_hist.ptr)
        meta_base = fx.ptrtoint(lds.s_meta.ptr)

        def lds_i32_ptr(base, elem_i32):
            elem_idx = arith.index_cast(T.index, elem_i32)
            addr = fx.Index(base) + fx.Index(elem_idx) * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=3)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def atomic_add_i32(base, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                lds_i32_ptr(base, elem_i32),
                value,
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result

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
            # One coalesced ``LOAD_VEC``-wide fp32 buffer load starting at the
            # element offset ``row_base + col_base``. ``stride1 == 1`` is enforced
            # by the host wrapper, so the four logits are contiguous. Returns the
            # raw vector; callers extract lanes and mask the ``col >= row_len``
            # tail to ``-inf`` so out-of-range/padding logits never participate.
            return buffer_ops.buffer_load(
                logits_rsrc,
                row_base + ArithValue(col_base_i32),
                vec_width=LOAD_VEC,
                dtype=T.f32,
            )

        def clear_histogram():
            for hist_idx in range(
                ArithValue(tid_idx), ArithValue(c_bins_idx), ArithValue(c_block_idx)
            ):
                hist_i32 = arith.index_cast(T.i32, hist_idx)
                fx.memref_store(c_zero, s_hist, hist_i32)
            gpu.barrier()

        def wave_inclusive_scan_i32(value):
            # Kogge-Stone inclusive prefix sum across the 64 lanes of one wave,
            # using ``ds_bpermute`` to read lane ``l-d`` (no barriers, no LDS
            # round-trips). Lanes with ``l < d`` keep their own partial. After
            # ``log2(WARP_SIZE)`` steps every lane holds the inclusive sum of all
            # lanes ``<=`` it; lane 63 holds the wave total.
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
            # AITER/hipcub-style hierarchical inclusive block scan over the
            # 2048-bin histogram, replacing the 11-step Hillis-Steele suffix
            # ping-pong (11 barriers + 11 LDS read/write rounds) with a
            # wave-scan + cross-wave-totals scheme (3 barriers, the per-lane
            # prefixes flowing through registers via ``ds_bpermute``).
            #
            # Each thread owns the contiguous bin pair ``(2*tid, 2*tid+1)`` so
            # waves cover contiguous bin ranges and the block scan order matches
            # ascending bin index. We compute the *forward* inclusive prefix
            # ``incl[b] = sum_{i <= b} count[i]`` (and ``excl[b] = incl[b] -
            # count[b]``); the kth-largest boundary is the first bucket whose
            # inclusive prefix passes ``K' = total - target_k``:
            #   ``excl[b] <= K'`` and ``incl[b] > K'``.
            # This is algebraically identical to a suffix-count formulation
            # (``suffix[b] = total - excl[b]``) but lets us reuse a standard
            # ascending block scan. ``above = total - incl[b]`` is the count
            # strictly above the boundary bucket, exactly as before.
            two_tid = ArithValue(tid) * ArithValue(c_two)
            c0 = fx.memref_load(s_hist, two_tid)
            c1 = fx.memref_load(s_hist, ArithValue(two_tid) + ArithValue(c_one))
            local_total = ArithValue(c0) + ArithValue(c1)

            wave_incl = wave_inclusive_scan_i32(local_total)
            wave_excl_thread = ArithValue(wave_incl) - ArithValue(local_total)

            # Lane 63 of each wave publishes that wave's 128-bin total.
            if ArithValue(lane) == ArithValue(c_last_lane):
                fx.memref_store(wave_incl, s_hist2, wave)
            gpu.barrier()

            # Wave 0 scans the 16 wave totals into exclusive wave offsets.
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

        direct_fill = row_len <= ArithValue(c_top_k)
        direct_fill_iters = direct_fill.select(c_top_k_idx, fx.Index(0))
        for out_col in range(
            ArithValue(tid_idx), ArithValue(direct_fill_iters), ArithValue(c_block_idx)
        ):
            out_col_i32 = arith.index_cast(T.i32, out_col)
            valid = ArithValue(out_col_i32) < row_len
            out_val = valid.select(out_col_i32, c_neg_one)
            buffer_ops.buffer_store(out_val, indices_rsrc, row_out + ArithValue(out_col_i32))

        long_active = row_len > ArithValue(c_top_k)
        # Each active thread strides over LOAD_VEC-wide blocks; the block count
        # is ceil(row_len / LOAD_VEC). The vectorized body masks the final
        # partial block (cols >= row_len) so the result is identical to the
        # scalar per-column scan.
        active_len_i32 = long_active.select(row_len, c_zero)
        vec_blocks_i32 = ArithValue(
            ArithValue(active_len_i32) + ArithValue(c_vec) - ArithValue(c_one)
        ).shrui(arith.constant(2, type=T.i32))
        vec_blocks_idx = arith.index_cast(T.index, vec_blocks_i32)

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

        # Pass 1: histogram of the high 11 bits over the whole row.
        clear_histogram()
        for vblk, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(vec_blocks_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            col_base = ArithValue(arith.index_cast(T.i32, vblk)) * ArithValue(c_vec)
            vec = load_row_vec(col_base)
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(arith.constant(j, type=T.i32))
                in_range = ArithValue(col_i32) < row_len
                if in_range:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    bucket = ordered_bucket(val)
                    atomic_add_i32(hist_base, bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()

        choose_threshold_parallel(c_top_k, 0, 1)

        # Pass 2: histogram of the mid 11 bits within the high boundary bucket.
        first_threshold = fx.memref_load(s_meta, 1)
        clear_histogram()
        for vblk, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(vec_blocks_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            col_base = ArithValue(arith.index_cast(T.i32, vblk)) * ArithValue(c_vec)
            vec = load_row_vec(col_base)
            for j in range_constexpr(LOAD_VEC):
                col_i32 = ArithValue(col_base) + ArithValue(arith.constant(j, type=T.i32))
                in_range = ArithValue(col_i32) < row_len
                if in_range:
                    val = vector.extract(vec, static_position=[j], dynamic_position=[])
                    bucket = ordered_bucket(val)
                    if ArithValue(bucket) == ArithValue(first_threshold):
                        mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                        atomic_add_i32(hist_base, mid_bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()

        first_above = fx.memref_load(s_meta, 0)
        need_after_first = ArithValue(c_top_k) - ArithValue(first_above)
        choose_threshold_parallel(need_after_first, 2, 3)

        # Pass 3: histogram of the low 10 bits within the high+mid boundary.
        second_threshold = fx.memref_load(s_meta, 3)
        clear_histogram()
        for vblk, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(vec_blocks_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            col_base = ArithValue(arith.index_cast(T.i32, vblk)) * ArithValue(c_vec)
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
                        low_bucket = radix_bucket(val, c_zero, c_low_mask)
                        atomic_add_i32(hist_base, low_bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()

        second_above = fx.memref_load(s_meta, 2)
        need_after_second = ArithValue(need_after_first) - ArithValue(second_above)
        choose_threshold_parallel(need_after_second, 4, 5)

        # Final phase: direct atomic-append write (no compaction, no sort).
        third_threshold = fx.memref_load(s_meta, 5)
        third_above = fx.memref_load(s_meta, 4)
        num_needed = ArithValue(need_after_second) - ArithValue(third_above)

        for vblk, write_state in range(
            ArithValue(tid_idx),
            ArithValue(vec_blocks_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            col_base = ArithValue(arith.index_cast(T.i32, vblk)) * ArithValue(c_vec)
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
                        pos = atomic_add_i32(meta_base, c_six, c_one)
                        buffer_ops.buffer_store(
                            col_i32, indices_rsrc, row_out + ArithValue(pos)
                        )
                    if at_boundary:
                        back = atomic_add_i32(meta_base, c_seven, c_one)
                        if ArithValue(back) < ArithValue(num_needed):
                            out_pos = (
                                ArithValue(c_top_k)
                                - ArithValue(c_one)
                                - ArithValue(back)
                            )
                            buffer_ops.buffer_store(
                                col_i32, indices_rsrc, row_out + ArithValue(out_pos)
                            )
            write_results = yield [write_state[0]]

    @flyc.jit
    def launch_topk_per_row_decode_radix_unordered(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        grid_x = arith.index_cast(T.index, num_rows)
        topk_per_row_decode_radix_unordered_kernel(
            logits,
            next_n,
            seq_lens,
            indices,
            num_rows,
            stride0,
            stride1,
        ).launch(
            grid=(grid_x, 1, 1),
            block=(BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_topk_per_row_decode_radix_unordered
