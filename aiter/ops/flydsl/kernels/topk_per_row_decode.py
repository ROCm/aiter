# SPDX-License-Identifier: MIT

"""FlyDSL decode TopK-per-row kernel builders.

One CTA per row, compile-time ``TOP_K``, decode row-length semantics, and
direct-fill for rows whose valid length is already at most ``TOP_K``. Long rows
use one of two compile-time-specialized strategies:

* ``k == 512``: an AITER/vLLM-style **radix-select** path. Three order-preserving
  radix passes (11/11/10 bits over a twiddled fp32 key) narrow the row to the
  exact kth-largest boundary using a block-parallel suffix scan to choose each
  threshold bucket. Survivors (keys at or above the boundary) are compacted into
  LDS and bitonic-sorted by ``(value desc, index asc)`` so the emitted Top-K is
  fully ordered with deterministic tie-breaking. A correctness fallback covers
  the rare case where boundary ties overflow the candidate buffer.
* other ``k`` (e.g. ``2048``): a block-parallel bitonic sort of the row in LDS.
  Radix narrowing buys little here because ``k`` is a large fraction of the row
  length used in practice, so the candidate set would barely shrink.

Output ordering matches ``torch.topk`` (value descending, lower index wins ties).
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
RADIX_BITS_K512 = 11
RADIX_BINS_K512 = 1 << RADIX_BITS_K512
CANDIDATE_CAP_K512 = 1024
BITONIC_STAGES_K512 = tuple(
    (size, j)
    for size in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    for j in (size // (2**p) for p in range(1, int.bit_length(size)))
)
BITONIC_CAP_K2048 = 4096
BITONIC_STAGES_K2048 = tuple(
    (size, j)
    for size in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
    for j in (size // (2**p) for p in range(1, int.bit_length(size)))
)


@functools.lru_cache(maxsize=32)
def create_topk_per_row_decode_kernel(top_k: int, ordered: bool = True):
    """Return a cached FlyDSL launcher specialized for ``top_k``.

    Launcher signature matches the existing decode interface:
    ``(logits, next_n, seqLens, indices, numRows, stride0, stride1, stream=...)``.
    ``stride0``/``stride1`` are threaded through now so the later radix path can
    use the same compiled signature; the direct-fill phase only needs row
    lengths and output indices.

    ``ordered=True`` (default) preserves the historical ``torch.topk``-identical
    output (value descending, lower index wins ties). ``ordered=False`` selects
    an AITER-style radix-**select** path that emits the Top-K as a *set* in
    arbitrary slot order: it shares the three-pass radix narrowing of the
    ordered path but replaces the candidate compaction + bitonic sort (the
    dominant cost) with a direct atomic-append write, mirroring AITER HIP's
    ``last_filter``. The selected indices are guaranteed set-equivalent to
    ``torch.topk`` (ties resolved by value, not by index), which is what the
    sparse-indexer consumer and the benchmark harness validate.
    """

    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    if not ordered:
        return _create_topk_per_row_decode_radix_unordered(top_k)

    if top_k == 512:
        return _create_topk_per_row_decode_radix_compact_k512()

    return _create_topk_per_row_decode_parallel_select(top_k)


def _create_topk_per_row_decode_radix_compact_k512():
    @fx.struct
    class SharedStorage:
        s_vals: fx.Array[fx.Float32, RED_SLOTS, 16]
        s_idxs: fx.Array[fx.Int32, RED_SLOTS, 16]
        s_hist: fx.Array[fx.Int32, RADIX_BINS_K512, 16]
        s_hist2: fx.Array[fx.Int32, RADIX_BINS_K512, 16]
        s_cand_vals: fx.Array[fx.Float32, CANDIDATE_CAP_K512, 16]
        s_cand_idxs: fx.Array[fx.Int32, CANDIDATE_CAP_K512, 16]
        s_meta: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(name="topk_per_row_decode_radix_refine_k512", known_block_size=[BLOCK_THREADS, 1, 1])
    def topk_per_row_decode_radix_refine_k512_kernel(
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
        s_vals = lds.s_vals.view(fx.make_layout(RED_SLOTS, 1))
        s_idxs = lds.s_idxs.view(fx.make_layout(RED_SLOTS, 1))
        s_hist = lds.s_hist.view(fx.make_layout(RADIX_BINS_K512, 1))
        s_hist2 = lds.s_hist2.view(fx.make_layout(RADIX_BINS_K512, 1))
        s_cand_vals = lds.s_cand_vals.view(fx.make_layout(CANDIDATE_CAP_K512, 1))
        s_cand_idxs = lds.s_cand_idxs.view(fx.make_layout(CANDIDATE_CAP_K512, 1))
        s_meta = lds.s_meta.view(fx.make_layout(8, 1))

        logits_t = GTensor(logits, dtype=T.f32, shape=(-1,))
        seq_lens_t = GTensor(seq_lens, dtype=T.i32, shape=(-1,))
        indices_rsrc = GTensor(indices, dtype=T.i32, shape=(-1,)).rsrc

        c_zero = arith.constant(0, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_two = arith.constant(2, type=T.i32)
        c_three = arith.constant(3, type=T.i32)
        c_four = arith.constant(4, type=T.i32)
        c_five = arith.constant(5, type=T.i32)
        c_six = arith.constant(6, type=T.i32)
        c_seven = arith.constant(7, type=T.i32)
        c_neg_one = arith.constant(-1, type=T.i32)
        c_block_idx = fx.Index(BLOCK_THREADS)
        c_block_i32 = arith.constant(BLOCK_THREADS, type=T.i32)
        c_bins_i32 = arith.constant(RADIX_BINS_K512, type=T.i32)
        c_top_k = arith.constant(512, type=T.i32)
        c_top_k_idx = fx.Index(512)
        c_two_idx = fx.Index(2)
        c_four_idx = fx.Index(4)
        c_bins_idx = fx.Index(RADIX_BINS_K512)
        c_candidate_cap = arith.constant(CANDIDATE_CAP_K512, type=T.i32)
        c_candidate_cap_idx = fx.Index(CANDIDATE_CAP_K512)
        c_shift = arith.constant(32 - RADIX_BITS_K512, type=T.i32)
        c_mid_shift = arith.constant(10, type=T.i32)
        c_bin_mask = arith.constant(RADIX_BINS_K512 - 1, type=T.i32)
        c_low_mask = arith.constant((1 << 10) - 1, type=T.i32)
        c_sign_bit = arith.constant(-2147483648, type=T.i32)
        c_bin_last_idx = fx.Index(RADIX_BINS_K512 - 1)
        c_neg_one_idx = fx.Index(-1)
        c_pos_inf = fx.Float32(float("inf"))
        c_neg_inf = fx.Float32(float("-inf"))
        c_zero_f32 = fx.Float32(0.0)

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

        def atomic_max_i32(base, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.max,
                lds_i32_ptr(base, elem_i32),
                value,
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result

        def better_pair(lhs_val, lhs_idx, rhs_val, rhs_idx):
            same_val = ArithValue(lhs_val) == ArithValue(rhs_val)
            return (lhs_val > rhs_val) | (
                same_val & (ArithValue(lhs_idx) < ArithValue(rhs_idx))
            )

        def ordered_key(val):
            # Normalize signed zeros so the radix prefilter cannot split values
            # that the final comparator treats as ties.
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

        def wave_reduce_best(best_val, best_idx):
            val = best_val
            idx = best_idx
            width = fx.Int32(WARP_SIZE)
            for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
                off = fx.Int32(1 << sh)
                peer_val = val.shuffle_xor(off, width)
                peer_idx = idx.shuffle_xor(off, width)
                take_peer = better_pair(peer_val, peer_idx, val, idx)
                val = take_peer.select(peer_val, val)
                idx = take_peer.select(peer_idx, idx)
            return val, idx

        def block_reduce_best(best_val, best_idx):
            wave_val, wave_idx = wave_reduce_best(best_val, best_idx)
            if lane == 0:
                fx.memref_store(wave_val, s_vals, wave)
                fx.memref_store(wave_idx, s_idxs, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                partial_val = fx.memref_load(s_vals, lane_safe)
                partial_idx = fx.memref_load(s_idxs, lane_safe)
                red_val = in_range.select(partial_val, c_neg_inf)
                red_idx = in_range.select(partial_idx, row_len)
                red_val, red_idx = wave_reduce_best(red_val, red_idx)
                if lane == 0:
                    fx.memref_store(red_val, s_vals, 0)
                    fx.memref_store(red_idx, s_idxs, 0)
            gpu.barrier()

            return fx.memref_load(s_vals, 0), fx.memref_load(s_idxs, 0)

        def repeated_select_from_global(active):
            active_iters = active.select(c_top_k_idx, fx.Index(0))
            init_prev = [c_pos_inf, c_neg_one]
            for out_col, prev_state in range(
                ArithValue(fx.Index(0)),
                ArithValue(active_iters),
                ArithValue(c_four_idx),
                init=init_prev,
            ):
                out_col_i32 = arith.index_cast(T.i32, out_col)
                prev_val = prev_state[0]
                prev_idx = ArithValue(prev_state[1])

                init_best = [
                    c_neg_inf,
                    row_len,
                    c_neg_inf,
                    row_len,
                    c_neg_inf,
                    row_len,
                    c_neg_inf,
                    row_len,
                ]
                for col, best_state in range(
                    ArithValue(tid_idx),
                    ArithValue(row_len_idx),
                    ArithValue(c_block_idx),
                    init=init_best,
                ):
                    col_i32 = arith.index_cast(T.i32, col)
                    best1_val = best_state[0]
                    best1_idx = ArithValue(best_state[1])
                    best2_val = best_state[2]
                    best2_idx = ArithValue(best_state[3])
                    best3_val = best_state[4]
                    best3_idx = ArithValue(best_state[5])
                    best4_val = best_state[6]
                    best4_idx = ArithValue(best_state[7])
                    val = logits_t.load(row_base + ArithValue(col_i32) * ArithValue(stride1))

                    same_as_prev = ArithValue(val) == ArithValue(prev_val)
                    worse_than_prev = (val < prev_val) | (
                        same_as_prev & (ArithValue(col_i32) > prev_idx)
                    )
                    take_best1 = worse_than_prev & better_pair(
                        val, col_i32, best1_val, best1_idx
                    )
                    take_best2 = (~take_best1) & worse_than_prev & better_pair(
                        val, col_i32, best2_val, best2_idx
                    )

                    next_best1_val = take_best1.select(val, best1_val)
                    next_best1_idx = take_best1.select(col_i32, best1_idx)
                    promoted_best2_val = take_best1.select(best1_val, best2_val)
                    promoted_best2_idx = take_best1.select(best1_idx, best2_idx)
                    promoted_best3_val = take_best1.select(best2_val, best3_val)
                    promoted_best3_idx = take_best1.select(best2_idx, best3_idx)
                    promoted_best4_val = take_best1.select(best3_val, best4_val)
                    promoted_best4_idx = take_best1.select(best3_idx, best4_idx)
                    next_best2_val = take_best2.select(val, promoted_best2_val)
                    next_best2_idx = take_best2.select(col_i32, promoted_best2_idx)
                    promoted_best3_val = take_best2.select(promoted_best2_val, promoted_best3_val)
                    promoted_best3_idx = take_best2.select(promoted_best2_idx, promoted_best3_idx)
                    promoted_best4_val = take_best2.select(promoted_best3_val, promoted_best4_val)
                    promoted_best4_idx = take_best2.select(promoted_best3_idx, promoted_best4_idx)
                    take_best3 = (~take_best1) & (~take_best2) & worse_than_prev & better_pair(
                        val, col_i32, promoted_best3_val, promoted_best3_idx
                    )
                    next_best3_val = take_best3.select(val, promoted_best3_val)
                    next_best3_idx = take_best3.select(col_i32, promoted_best3_idx)
                    promoted_best4_val = take_best3.select(promoted_best3_val, promoted_best4_val)
                    promoted_best4_idx = take_best3.select(promoted_best3_idx, promoted_best4_idx)
                    take_best4 = (
                        (~take_best1)
                        & (~take_best2)
                        & (~take_best3)
                        & worse_than_prev
                        & better_pair(val, col_i32, promoted_best4_val, promoted_best4_idx)
                    )
                    next_best4_val = take_best4.select(val, promoted_best4_val)
                    next_best4_idx = take_best4.select(col_i32, promoted_best4_idx)
                    scan_results = yield [
                        next_best1_val,
                        next_best1_idx,
                        next_best2_val,
                        next_best2_idx,
                        next_best3_val,
                        next_best3_idx,
                        next_best4_val,
                        next_best4_idx,
                    ]

                selected_val, selected_idx = block_reduce_best(scan_results[0], scan_results[1])
                first_available = ArithValue(scan_results[1]) != ArithValue(selected_idx)
                second_candidate_val = first_available.select(scan_results[0], scan_results[2])
                second_candidate_idx = first_available.select(scan_results[1], scan_results[3])
                selected2_val, selected2_idx = block_reduce_best(
                    second_candidate_val, second_candidate_idx
                )
                first_available = (ArithValue(scan_results[1]) != ArithValue(selected_idx)) & (
                    ArithValue(scan_results[1]) != ArithValue(selected2_idx)
                )
                second_available = (ArithValue(scan_results[3]) != ArithValue(selected_idx)) & (
                    ArithValue(scan_results[3]) != ArithValue(selected2_idx)
                )
                third_candidate_val = first_available.select(
                    scan_results[0], second_available.select(scan_results[2], scan_results[4])
                )
                third_candidate_idx = first_available.select(
                    scan_results[1], second_available.select(scan_results[3], scan_results[5])
                )
                selected3_val, selected3_idx = block_reduce_best(
                    third_candidate_val, third_candidate_idx
                )
                first_available = (
                    (ArithValue(scan_results[1]) != ArithValue(selected_idx))
                    & (ArithValue(scan_results[1]) != ArithValue(selected2_idx))
                    & (ArithValue(scan_results[1]) != ArithValue(selected3_idx))
                )
                second_available = (
                    (ArithValue(scan_results[3]) != ArithValue(selected_idx))
                    & (ArithValue(scan_results[3]) != ArithValue(selected2_idx))
                    & (ArithValue(scan_results[3]) != ArithValue(selected3_idx))
                )
                third_available = (
                    (ArithValue(scan_results[5]) != ArithValue(selected_idx))
                    & (ArithValue(scan_results[5]) != ArithValue(selected2_idx))
                    & (ArithValue(scan_results[5]) != ArithValue(selected3_idx))
                )
                fourth_candidate_val = first_available.select(
                    scan_results[0],
                    second_available.select(
                        scan_results[2], third_available.select(scan_results[4], scan_results[6])
                    ),
                )
                fourth_candidate_idx = first_available.select(
                    scan_results[1],
                    second_available.select(
                        scan_results[3], third_available.select(scan_results[5], scan_results[7])
                    ),
                )
                selected4_val, selected4_idx = block_reduce_best(
                    fourth_candidate_val, fourth_candidate_idx
                )
                if tid == ArithValue(c_zero):
                    buffer_ops.buffer_store(
                        selected_idx, indices_rsrc, row_out + ArithValue(out_col_i32)
                    )
                    buffer_ops.buffer_store(
                        selected2_idx,
                        indices_rsrc,
                        row_out + ArithValue(out_col_i32) + ArithValue(c_one),
                    )
                    buffer_ops.buffer_store(
                        selected3_idx,
                        indices_rsrc,
                        row_out + ArithValue(out_col_i32) + ArithValue(c_two),
                    )
                    buffer_ops.buffer_store(
                        selected4_idx,
                        indices_rsrc,
                        row_out + ArithValue(out_col_i32) + ArithValue(c_three),
                    )
                order_results = yield [selected4_val, selected4_idx]

        def clear_histogram():
            for hist_idx in range(
                ArithValue(tid_idx), ArithValue(c_bins_idx), ArithValue(c_block_idx)
            ):
                hist_i32 = arith.index_cast(T.i32, hist_idx)
                fx.memref_store(c_zero, s_hist, hist_i32)
            gpu.barrier()

        def choose_threshold_parallel(target_k, above_slot, threshold_slot):
            # Locate the radix threshold bucket with the whole block instead of
            # a single-threaded descending scan. A Hillis-Steele inclusive
            # *suffix* scan rewrites the histogram so that
            # ``hist[b] == sum_{i >= b} count_orig[i]``. With ascending ordered
            # keys, larger logits land in higher bins, so the suffix sum is the
            # top-down cumulative count used to find the kth-largest boundary.
            # Two LDS buffers ping-pong so each step needs a single barrier; the
            # histogram is rebuilt (cleared) before the next pass, so consuming
            # both scratch buffers here is safe.
            b0 = ArithValue(tid)
            b1 = ArithValue(tid) + ArithValue(c_block_i32)
            cur_buf = s_hist
            nxt_buf = s_hist2
            for sh in range_constexpr(int.bit_length(RADIX_BINS_K512) - 1):
                d = arith.constant(1 << sh, type=T.i32)
                src0 = ArithValue(b0) + ArithValue(d)
                src1 = ArithValue(b1) + ArithValue(d)
                in0 = ArithValue(src0) < ArithValue(c_bins_i32)
                in1 = ArithValue(src1) < ArithValue(c_bins_i32)
                add0 = in0.select(
                    fx.memref_load(cur_buf, in0.select(src0, c_zero)), c_zero
                )
                add1 = in1.select(
                    fx.memref_load(cur_buf, in1.select(src1, c_zero)), c_zero
                )
                fx.memref_store(
                    ArithValue(fx.memref_load(cur_buf, b0)) + ArithValue(add0),
                    nxt_buf,
                    b0,
                )
                fx.memref_store(
                    ArithValue(fx.memref_load(cur_buf, b1)) + ArithValue(add1),
                    nxt_buf,
                    b1,
                )
                gpu.barrier()
                cur_buf, nxt_buf = nxt_buf, cur_buf

            # ``cur_buf`` now holds the suffix sums.
            # b* = largest bucket whose suffix count still covers target_k; the
            # suffix of the next-higher bucket is the count strictly above it.
            def emit_find(b):
                suffix = fx.memref_load(cur_buf, b)
                nb = ArithValue(b) + ArithValue(c_one)
                in_nb = ArithValue(nb) < ArithValue(c_bins_i32)
                above = in_nb.select(
                    fx.memref_load(cur_buf, in_nb.select(nb, c_zero)), c_zero
                )
                crosses = (ArithValue(suffix) >= ArithValue(target_k)) & (
                    ArithValue(above) < ArithValue(target_k)
                )
                if crosses:
                    fx.memref_store(b, s_meta, threshold_slot)
                    fx.memref_store(above, s_meta, above_slot)

            emit_find(b0)
            emit_find(b1)
            gpu.barrier()

        seq_row = row // ArithValue(next_n)
        slot = row - seq_row * ArithValue(next_n)
        seq_len = ArithValue(seq_lens_t[seq_row])
        row_len = seq_len - ArithValue(next_n) + slot + ArithValue(c_one)
        row_len = (row_len > ArithValue(c_zero)).select(row_len, c_zero)
        row_len_idx = arith.index_cast(T.index, row_len)
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
        active_len_idx = row_len_idx
        for cand_idx in range(
            ArithValue(tid_idx), ArithValue(c_candidate_cap_idx), ArithValue(c_block_idx)
        ):
            cand_i32 = arith.index_cast(T.i32, cand_idx)
            fx.memref_store(c_neg_inf, s_cand_vals, cand_i32)
            fx.memref_store(row_len, s_cand_idxs, cand_i32)
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

        clear_histogram()
        for col, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(active_len_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            col_i32 = arith.index_cast(T.i32, col)
            val = logits_t.load(row_base + ArithValue(col_i32) * ArithValue(stride1))
            bucket = ordered_bucket(val)
            atomic_add_i32(hist_base, bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()

        choose_threshold_parallel(c_top_k, 0, 1)

        first_threshold = fx.memref_load(s_meta, 1)
        clear_histogram()
        for col, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(active_len_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            col_i32 = arith.index_cast(T.i32, col)
            val = logits_t.load(row_base + ArithValue(col_i32) * ArithValue(stride1))
            bucket = ordered_bucket(val)
            if ArithValue(bucket) == ArithValue(first_threshold):
                mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
                atomic_add_i32(hist_base, mid_bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()

        first_above = fx.memref_load(s_meta, 0)
        need_after_first = ArithValue(c_top_k) - ArithValue(first_above)
        choose_threshold_parallel(need_after_first, 2, 3)

        second_threshold = fx.memref_load(s_meta, 3)
        clear_histogram()
        for col, pass_state in range(
            ArithValue(tid_idx),
            ArithValue(active_len_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            col_i32 = arith.index_cast(T.i32, col)
            val = logits_t.load(row_base + ArithValue(col_i32) * ArithValue(stride1))
            high_bucket = ordered_bucket(val)
            mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
            matches_refine = (ArithValue(high_bucket) == ArithValue(first_threshold)) & (
                ArithValue(mid_bucket) == ArithValue(second_threshold)
            )
            if matches_refine:
                low_bucket = radix_bucket(val, c_zero, c_low_mask)
                atomic_add_i32(hist_base, low_bucket, c_one)
            pass_results = yield [pass_state[0]]
        gpu.barrier()

        second_above = fx.memref_load(s_meta, 2)
        need_after_second = (
            ArithValue(c_top_k) - ArithValue(first_above) - ArithValue(second_above)
        )
        choose_threshold_parallel(need_after_second, 4, 5)

        third_threshold = fx.memref_load(s_meta, 5)
        for col, compact_state in range(
            ArithValue(tid_idx),
            ArithValue(active_len_idx),
            ArithValue(c_block_idx),
            init=[c_zero],
        ):
            col_i32 = arith.index_cast(T.i32, col)
            val = logits_t.load(row_base + ArithValue(col_i32) * ArithValue(stride1))
            high_bucket = ordered_bucket(val)
            mid_bucket = radix_bucket(val, c_mid_shift, c_bin_mask)
            low_bucket = radix_bucket(val, c_zero, c_low_mask)
            above_first = ArithValue(high_bucket) > ArithValue(first_threshold)
            at_first = ArithValue(high_bucket) == ArithValue(first_threshold)
            above_second = ArithValue(mid_bucket) > ArithValue(second_threshold)
            at_second = ArithValue(mid_bucket) == ArithValue(second_threshold)
            in_low_tail = ArithValue(low_bucket) >= ArithValue(third_threshold)
            keep = above_first | (at_first & (above_second | (at_second & in_low_tail)))
            if keep:
                slot = atomic_add_i32(meta_base, c_six, c_one)
                in_cap = ArithValue(slot) < ArithValue(c_candidate_cap)
                if in_cap:
                    fx.memref_store(val, s_cand_vals, slot)
                    fx.memref_store(col_i32, s_cand_idxs, slot)
                if ~in_cap:
                    atomic_max_i32(meta_base, c_seven, c_one)
            compact_results = yield [compact_state[0]]
        gpu.barrier()

        candidate_count = fx.memref_load(s_meta, 6)
        overflow = fx.memref_load(s_meta, 7)
        compact_ok = long_active & (ArithValue(overflow) == ArithValue(c_zero)) & (
            ArithValue(candidate_count) >= ArithValue(c_top_k)
        )
        sort_iters = compact_ok.select(c_candidate_cap_idx, fx.Index(0))
        for stage in range_constexpr(len(BITONIC_STAGES_K512)):
            size, j = BITONIC_STAGES_K512[stage]
            size_i32 = arith.constant(size, type=T.i32)
            j_i32 = arith.constant(j, type=T.i32)
            for cand in range(
                ArithValue(tid_idx),
                ArithValue(sort_iters),
                ArithValue(c_block_idx),
            ):
                cand_i32 = arith.index_cast(T.i32, cand)
                partner_i32 = ArithValue(cand_i32) ^ ArithValue(j_i32)
                owns_pair = ArithValue(partner_i32) > ArithValue(cand_i32)
                if owns_pair:
                    left_val = fx.memref_load(s_cand_vals, cand_i32)
                    left_idx = fx.memref_load(s_cand_idxs, cand_i32)
                    right_val = fx.memref_load(s_cand_vals, partner_i32)
                    right_idx = fx.memref_load(s_cand_idxs, partner_i32)
                    descending = (
                        (ArithValue(cand_i32) & ArithValue(size_i32)) == ArithValue(c_zero)
                    )
                    swap_desc = better_pair(right_val, right_idx, left_val, left_idx)
                    swap_asc = better_pair(left_val, left_idx, right_val, right_idx)
                    swap = descending.select(swap_desc, swap_asc)
                    new_left_val = swap.select(right_val, left_val)
                    new_left_idx = swap.select(right_idx, left_idx)
                    new_right_val = swap.select(left_val, right_val)
                    new_right_idx = swap.select(left_idx, right_idx)
                    fx.memref_store(new_left_val, s_cand_vals, cand_i32)
                    fx.memref_store(new_left_idx, s_cand_idxs, cand_i32)
                    fx.memref_store(new_right_val, s_cand_vals, partner_i32)
                    fx.memref_store(new_right_idx, s_cand_idxs, partner_i32)
            gpu.barrier()

        out_iters = compact_ok.select(c_top_k_idx, fx.Index(0))
        for out_col in range(
            ArithValue(tid_idx), ArithValue(out_iters), ArithValue(c_block_idx)
        ):
            out_col_i32 = arith.index_cast(T.i32, out_col)
            out_idx = fx.memref_load(s_cand_idxs, out_col_i32)
            buffer_ops.buffer_store(out_idx, indices_rsrc, row_out + ArithValue(out_col_i32))

        repeated_select_from_global(long_active & (~compact_ok))

    @flyc.jit
    def launch_topk_per_row_decode_radix_compact_k512(
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
        topk_per_row_decode_radix_refine_k512_kernel(
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

    return launch_topk_per_row_decode_radix_compact_k512


def _create_topk_per_row_decode_radix_unordered(top_k: int):
    """AITER-style one-block radix-select with unordered (set) output.

    Three order-preserving radix passes (11/11/10 bits over a twiddled fp32 key)
    narrow the row to the exact kth-largest 32-bit boundary, identical to the
    ordered ``k=512`` path. The final phase then writes results directly:

    * keys strictly above the boundary are guaranteed Top-K members and are
      atomic-appended to the front of the output;
    * keys exactly at the boundary fill the remaining ``num_needed`` slots from
      the back.

    There is no candidate buffer and no sort, so the per-pass barrier count and
    LDS footprint are far smaller than the ordered path. Output slot order is
    arbitrary but the index *set* matches ``torch.topk`` (ties resolved by
    value). LDS footprint is two 2048-int histograms (16 KiB) regardless of
    ``top_k``, well within gfx942's 64 KiB.
    """

    @fx.struct
    class SharedStorage:
        s_hist: fx.Array[fx.Int32, RADIX_BINS_K512, 16]
        s_hist2: fx.Array[fx.Int32, RADIX_BINS_K512, 16]
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
        s_hist = lds.s_hist.view(fx.make_layout(RADIX_BINS_K512, 1))
        s_hist2 = lds.s_hist2.view(fx.make_layout(RADIX_BINS_K512, 1))
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
        c_bins_i32 = arith.constant(RADIX_BINS_K512, type=T.i32)
        c_top_k = arith.constant(top_k, type=T.i32)
        c_top_k_idx = fx.Index(top_k)
        c_bins_idx = fx.Index(RADIX_BINS_K512)
        c_shift = arith.constant(32 - RADIX_BITS_K512, type=T.i32)
        c_mid_shift = arith.constant(10, type=T.i32)
        c_bin_mask = arith.constant(RADIX_BINS_K512 - 1, type=T.i32)
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
            # This is algebraically identical to the old suffix test
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


def _create_topk_per_row_decode_parallel_select(top_k: int):
    @fx.struct
    class SharedStorage:
        s_vals: fx.Array[fx.Float32, RED_SLOTS, 16]
        s_idxs: fx.Array[fx.Int32, RED_SLOTS, 16]
        s_sort_vals: fx.Array[fx.Float32, BITONIC_CAP_K2048, 16]
        s_sort_idxs: fx.Array[fx.Int32, BITONIC_CAP_K2048, 16]

    kernel_name = f"topk_per_row_decode_parallel_select_k{top_k}"

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def topk_per_row_decode_parallel_select_kernel(
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
        s_vals = lds.s_vals.view(fx.make_layout(RED_SLOTS, 1))
        s_idxs = lds.s_idxs.view(fx.make_layout(RED_SLOTS, 1))
        s_sort_vals = lds.s_sort_vals.view(fx.make_layout(BITONIC_CAP_K2048, 1))
        s_sort_idxs = lds.s_sort_idxs.view(fx.make_layout(BITONIC_CAP_K2048, 1))

        logits_t = GTensor(logits, dtype=T.f32, shape=(-1,))
        seq_lens_t = GTensor(seq_lens, dtype=T.i32, shape=(-1,))
        indices_rsrc = GTensor(indices, dtype=T.i32, shape=(-1,)).rsrc
        logits_rsrc = logits_t.rsrc

        c_zero = arith.constant(0, type=T.i32)
        c_one = arith.constant(1, type=T.i32)
        c_two = arith.constant(2, type=T.i32)
        c_three = arith.constant(3, type=T.i32)
        c_neg_one = arith.constant(-1, type=T.i32)
        c_block = arith.constant(BLOCK_THREADS, type=T.i32)
        c_top_k = arith.constant(top_k, type=T.i32)
        c_zero_idx = fx.Index(0)
        c_one_idx = fx.Index(1)
        c_two_idx = fx.Index(2)
        c_four_idx = fx.Index(4)
        c_block_idx = fx.Index(BLOCK_THREADS)
        c_top_k_idx = fx.Index(top_k)
        c_bitonic_cap_idx = fx.Index(BITONIC_CAP_K2048)
        c_bitonic_cap = arith.constant(BITONIC_CAP_K2048, type=T.i32)
        c_pos_inf = fx.Float32(float("inf"))
        c_neg_inf = fx.Float32(float("-inf"))

        def better_pair(lhs_val, lhs_idx, rhs_val, rhs_idx):
            same_val = ArithValue(lhs_val) == ArithValue(rhs_val)
            return (lhs_val > rhs_val) | (
                same_val & (ArithValue(lhs_idx) < ArithValue(rhs_idx))
            )

        def wave_reduce_best(best_val, best_idx):
            val = best_val
            idx = best_idx
            width = fx.Int32(WARP_SIZE)
            for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
                off = fx.Int32(1 << sh)
                peer_val = val.shuffle_xor(off, width)
                peer_idx = idx.shuffle_xor(off, width)
                take_peer = better_pair(peer_val, peer_idx, val, idx)
                val = take_peer.select(peer_val, val)
                idx = take_peer.select(peer_idx, idx)
            return val, idx

        def block_reduce_best(best_val, best_idx):
            if const_expr(RED_SLOTS == 1):
                return wave_reduce_best(best_val, best_idx)

            wave_val, wave_idx = wave_reduce_best(best_val, best_idx)
            if lane == 0:
                fx.memref_store(wave_val, s_vals, wave)
                fx.memref_store(wave_idx, s_idxs, wave)
            gpu.barrier()

            if wave == 0:
                in_range = lane < RED_SLOTS
                lane_safe = in_range.select(lane, 0)
                partial_val = fx.memref_load(s_vals, lane_safe)
                partial_idx = fx.memref_load(s_idxs, lane_safe)
                red_val = in_range.select(partial_val, c_neg_inf)
                red_idx = in_range.select(partial_idx, row_len)
                red_val, red_idx = wave_reduce_best(red_val, red_idx)
                if lane == 0:
                    fx.memref_store(red_val, s_vals, 0)
                    fx.memref_store(red_idx, s_idxs, 0)
            gpu.barrier()

            return fx.memref_load(s_vals, 0), fx.memref_load(s_idxs, 0)

        seq_row = row // ArithValue(next_n)
        slot = row - seq_row * ArithValue(next_n)
        seq_len = ArithValue(seq_lens_t[seq_row])
        row_len = seq_len - ArithValue(next_n) + slot + ArithValue(c_one)
        row_len = (row_len > ArithValue(c_zero)).select(row_len, c_zero)
        row_len_idx = arith.index_cast(T.index, row_len)

        # Existing decode kernels write a contiguous (numRows, K) index matrix
        # and do not accept output strides. The launch grid is exactly numRows.
        row_out = row * ArithValue(c_top_k)
        direct_fill = row_len <= ArithValue(c_top_k)
        direct_fill_iters = direct_fill.select(c_top_k_idx, c_zero_idx)
        for out_col in range(
            ArithValue(tid_idx), ArithValue(direct_fill_iters), ArithValue(c_block_idx)
        ):
            out_col_i32 = arith.index_cast(T.i32, out_col)
            valid = ArithValue(out_col_i32) < row_len
            out_val = valid.select(out_col_i32, c_neg_one)
            buffer_ops.buffer_store(out_val, indices_rsrc, row_out + ArithValue(out_col_i32))

        # Correctness-first long-row path. All threads scan disjoint columns for
        # each output slot, then reduce the per-thread candidates across the CTA.
        long_active = row_len > ArithValue(c_top_k)
        row_base = row * ArithValue(stride0)
        if const_expr(top_k == 2048):
            bitonic_active = long_active & (row_len <= ArithValue(c_bitonic_cap))
            load_iters = bitonic_active.select(c_bitonic_cap_idx, c_zero_idx)
            for col in range(
                ArithValue(tid_idx), ArithValue(load_iters), ArithValue(c_block_idx)
            ):
                col_i32 = arith.index_cast(T.i32, col)
                valid = ArithValue(col_i32) < ArithValue(row_len)
                if valid:
                    val = buffer_ops.buffer_load(
                        logits_rsrc,
                        row_base + ArithValue(col_i32) * ArithValue(stride1),
                        vec_width=1,
                        dtype=T.f32,
                    )
                    fx.memref_store(val, s_sort_vals, col_i32)
                    fx.memref_store(col_i32, s_sort_idxs, col_i32)
                if ~valid:
                    fx.memref_store(c_neg_inf, s_sort_vals, col_i32)
                    fx.memref_store(row_len, s_sort_idxs, col_i32)
            gpu.barrier()

            for stage in range_constexpr(len(BITONIC_STAGES_K2048)):
                size, j = BITONIC_STAGES_K2048[stage]
                size_i32 = arith.constant(size, type=T.i32)
                j_i32 = arith.constant(j, type=T.i32)
                for col in range(
                    ArithValue(tid_idx),
                    ArithValue(load_iters),
                    ArithValue(c_block_idx),
                ):
                    col_i32 = arith.index_cast(T.i32, col)
                    partner_i32 = ArithValue(col_i32) ^ ArithValue(j_i32)
                    owns_pair = ArithValue(partner_i32) > ArithValue(col_i32)
                    if owns_pair:
                        left_val = fx.memref_load(s_sort_vals, col_i32)
                        left_idx = fx.memref_load(s_sort_idxs, col_i32)
                        right_val = fx.memref_load(s_sort_vals, partner_i32)
                        right_idx = fx.memref_load(s_sort_idxs, partner_i32)
                        descending = (
                            (ArithValue(col_i32) & ArithValue(size_i32))
                            == ArithValue(c_zero)
                        )
                        swap_desc = better_pair(right_val, right_idx, left_val, left_idx)
                        swap_asc = better_pair(left_val, left_idx, right_val, right_idx)
                        swap = descending.select(swap_desc, swap_asc)
                        new_left_val = swap.select(right_val, left_val)
                        new_left_idx = swap.select(right_idx, left_idx)
                        new_right_val = swap.select(left_val, right_val)
                        new_right_idx = swap.select(left_idx, right_idx)
                        fx.memref_store(new_left_val, s_sort_vals, col_i32)
                        fx.memref_store(new_left_idx, s_sort_idxs, col_i32)
                        fx.memref_store(new_right_val, s_sort_vals, partner_i32)
                        fx.memref_store(new_right_idx, s_sort_idxs, partner_i32)
                gpu.barrier()

            out_iters = bitonic_active.select(c_top_k_idx, c_zero_idx)
            for out_col in range(
                ArithValue(tid_idx), ArithValue(out_iters), ArithValue(c_block_idx)
            ):
                out_col_i32 = arith.index_cast(T.i32, out_col)
                out_idx = fx.memref_load(s_sort_idxs, out_col_i32)
                buffer_ops.buffer_store(
                    out_idx, indices_rsrc, row_out + ArithValue(out_col_i32)
                )
            select_active = long_active & (~bitonic_active)
        else:
            select_active = long_active

        long_iters = select_active.select(c_top_k_idx, c_zero_idx)
        init_prev = [c_pos_inf, c_neg_one]
        for out_col, prev_state in range(
            ArithValue(c_zero_idx),
            ArithValue(long_iters),
            ArithValue(c_four_idx),
            init=init_prev,
        ):
            out_col_i32 = arith.index_cast(T.i32, out_col)
            prev_val = prev_state[0]
            prev_idx = ArithValue(prev_state[1])

            init_best = [
                c_neg_inf,
                row_len,
                c_neg_inf,
                row_len,
                c_neg_inf,
                row_len,
                c_neg_inf,
                row_len,
            ]
            for col, best_state in range(
                ArithValue(tid_idx),
                ArithValue(row_len_idx),
                ArithValue(c_block_idx),
                init=init_best,
            ):
                col_i32 = arith.index_cast(T.i32, col)
                best1_val = best_state[0]
                best1_idx = ArithValue(best_state[1])
                best2_val = best_state[2]
                best2_idx = ArithValue(best_state[3])
                best3_val = best_state[4]
                best3_idx = ArithValue(best_state[5])
                best4_val = best_state[6]
                best4_idx = ArithValue(best_state[7])
                val = logits_t.load(row_base + ArithValue(col_i32) * ArithValue(stride1))

                same_as_prev = ArithValue(val) == ArithValue(prev_val)
                worse_than_prev = (val < prev_val) | (
                    same_as_prev & (ArithValue(col_i32) > prev_idx)
                )
                take_best1 = worse_than_prev & better_pair(
                    val, col_i32, best1_val, best1_idx
                )
                take_best2 = (~take_best1) & worse_than_prev & better_pair(
                    val, col_i32, best2_val, best2_idx
                )

                next_best1_val = take_best1.select(val, best1_val)
                next_best1_idx = take_best1.select(col_i32, best1_idx)
                promoted_best2_val = take_best1.select(best1_val, best2_val)
                promoted_best2_idx = take_best1.select(best1_idx, best2_idx)
                promoted_best3_val = take_best1.select(best2_val, best3_val)
                promoted_best3_idx = take_best1.select(best2_idx, best3_idx)
                promoted_best4_val = take_best1.select(best3_val, best4_val)
                promoted_best4_idx = take_best1.select(best3_idx, best4_idx)
                next_best2_val = take_best2.select(val, promoted_best2_val)
                next_best2_idx = take_best2.select(col_i32, promoted_best2_idx)
                promoted_best3_val = take_best2.select(promoted_best2_val, promoted_best3_val)
                promoted_best3_idx = take_best2.select(promoted_best2_idx, promoted_best3_idx)
                promoted_best4_val = take_best2.select(promoted_best3_val, promoted_best4_val)
                promoted_best4_idx = take_best2.select(promoted_best3_idx, promoted_best4_idx)
                take_best3 = (~take_best1) & (~take_best2) & worse_than_prev & better_pair(
                    val, col_i32, promoted_best3_val, promoted_best3_idx
                )
                next_best3_val = take_best3.select(val, promoted_best3_val)
                next_best3_idx = take_best3.select(col_i32, promoted_best3_idx)
                promoted_best4_val = take_best3.select(promoted_best3_val, promoted_best4_val)
                promoted_best4_idx = take_best3.select(promoted_best3_idx, promoted_best4_idx)
                take_best4 = (
                    (~take_best1)
                    & (~take_best2)
                    & (~take_best3)
                    & worse_than_prev
                    & better_pair(val, col_i32, promoted_best4_val, promoted_best4_idx)
                )
                next_best4_val = take_best4.select(val, promoted_best4_val)
                next_best4_idx = take_best4.select(col_i32, promoted_best4_idx)
                scan_results = yield [
                    next_best1_val,
                    next_best1_idx,
                    next_best2_val,
                    next_best2_idx,
                    next_best3_val,
                    next_best3_idx,
                    next_best4_val,
                    next_best4_idx,
                ]

            selected_val, selected_idx = block_reduce_best(scan_results[0], scan_results[1])
            first_available = ArithValue(scan_results[1]) != ArithValue(selected_idx)
            second_candidate_val = first_available.select(scan_results[0], scan_results[2])
            second_candidate_idx = first_available.select(scan_results[1], scan_results[3])
            selected2_val, selected2_idx = block_reduce_best(
                second_candidate_val, second_candidate_idx
            )
            first_available = (ArithValue(scan_results[1]) != ArithValue(selected_idx)) & (
                ArithValue(scan_results[1]) != ArithValue(selected2_idx)
            )
            second_available = (ArithValue(scan_results[3]) != ArithValue(selected_idx)) & (
                ArithValue(scan_results[3]) != ArithValue(selected2_idx)
            )
            third_candidate_val = first_available.select(
                scan_results[0], second_available.select(scan_results[2], scan_results[4])
            )
            third_candidate_idx = first_available.select(
                scan_results[1], second_available.select(scan_results[3], scan_results[5])
            )
            selected3_val, selected3_idx = block_reduce_best(
                third_candidate_val, third_candidate_idx
            )
            first_available = (
                (ArithValue(scan_results[1]) != ArithValue(selected_idx))
                & (ArithValue(scan_results[1]) != ArithValue(selected2_idx))
                & (ArithValue(scan_results[1]) != ArithValue(selected3_idx))
            )
            second_available = (
                (ArithValue(scan_results[3]) != ArithValue(selected_idx))
                & (ArithValue(scan_results[3]) != ArithValue(selected2_idx))
                & (ArithValue(scan_results[3]) != ArithValue(selected3_idx))
            )
            third_available = (
                (ArithValue(scan_results[5]) != ArithValue(selected_idx))
                & (ArithValue(scan_results[5]) != ArithValue(selected2_idx))
                & (ArithValue(scan_results[5]) != ArithValue(selected3_idx))
            )
            fourth_candidate_val = first_available.select(
                scan_results[0],
                second_available.select(
                    scan_results[2], third_available.select(scan_results[4], scan_results[6])
                ),
            )
            fourth_candidate_idx = first_available.select(
                scan_results[1],
                second_available.select(
                    scan_results[3], third_available.select(scan_results[5], scan_results[7])
                ),
            )
            selected4_val, selected4_idx = block_reduce_best(
                fourth_candidate_val, fourth_candidate_idx
            )
            if tid == ArithValue(c_zero):
                buffer_ops.buffer_store(
                    selected_idx, indices_rsrc, row_out + ArithValue(out_col_i32)
                )
                buffer_ops.buffer_store(
                    selected2_idx,
                    indices_rsrc,
                    row_out + ArithValue(out_col_i32) + ArithValue(c_one),
                )
                buffer_ops.buffer_store(
                    selected3_idx,
                    indices_rsrc,
                    row_out + ArithValue(out_col_i32) + ArithValue(c_two),
                )
                buffer_ops.buffer_store(
                    selected4_idx,
                    indices_rsrc,
                    row_out + ArithValue(out_col_i32) + ArithValue(c_three),
                )
            order_results = yield [selected4_val, selected4_idx]

    @flyc.jit
    def launch_topk_per_row_decode_parallel_select(
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
        topk_per_row_decode_parallel_select_kernel(
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

    return launch_topk_per_row_decode_parallel_select
