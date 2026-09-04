# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL DCP decode TopK merge: global threshold select + owned-slot emit."""

# mypy: allow-untyped-defs

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import (
    Array,
    Int32,
    arith,
    const_expr,
    gpu,
    range_constexpr,
)
from flydsl.expr import (
    rocdl as fly_rocdl,
)
from flydsl.expr.typing import T

from .kernels_common import atomic_add_i32, uint32_to_int32

_RADIX_BITS = 8
_RADIX_MASK = (1 << _RADIX_BITS) - 1
_RADIX_SIGN_BIT = 1 << (_RADIX_BITS - 1)
_NUM_RADIX_PASSES = 32 // _RADIX_BITS
# Sentinel for out-of-range candidate slots: sorts below every real ord value.
_INT32_MIN = -(1 << 31)
# Pack (above, equal) counts into one i32 for a single block scan. Both are
# bounded by k_loc per row, well under 2^15, so 16 bits each is safe.
_PACK_SHIFT = 16
_PACK_MASK = (1 << _PACK_SHIFT) - 1
_N_HIST_BINS = 1 << _RADIX_BITS
_BLOCK_THREADS = 256
_WAVE_SIZE = 64
_NUM_WAVES = _BLOCK_THREADS // _WAVE_SIZE
_DPP_ROW_SHR_1 = 0x111
_DPP_ROW_SHR_2 = 0x112
_DPP_ROW_SHR_4 = 0x114
_DPP_ROW_SHR_8 = 0x118
_DPP_ROW_MASK = 0xF
_DPP_BANK_MASK = 0xF


def _f32_to_ord(val):
    """Order-preserving float32 -> int32 map, with NaN forced to the bottom.

    NaN maps to INT32_MIN, the same ordinal -inf gets, so a NaN score loses the
    global threshold comparison instead of winning it. The reference model
    treats every non-finite value as -inf (torch.isfinite), and a disagreement
    here is not a crash: a NaN that sorts to the top steals a real candidate's
    slot and silently changes which rank owns that KV.
    """
    bits = val.bitcast(Int32)
    ords = bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
    abs_bits = bits & fx.Int32(0x7FFFFFFF)
    is_nan = arith.cmpi(arith.CmpIPredicate.ugt, abs_bits, fx.Int32(0x7F800000))
    return arith.select(is_nan, fx.Int32(uint32_to_int32(0x80000000)), ords)


def _make_storage():
    @fx.struct
    class Storage:
        bins: Array[Int32, _N_HIST_BINS, 16]
        scan: Array[Int32, _NUM_WAVES + 1, 16]
        # 5 slots: 0=prefix 1=mask 2=remaining_k 3=emitted-count 4=equal-seen
        acc: Array[Int32, 5, 4]

    return Storage


def build_dcp_topk_merge_module(n_cand, k_loc, topk_tokens, page_size, world_size):
    steps = (n_cand + _BLOCK_THREADS - 1) // _BLOCK_THREADS
    own_steps = (k_loc + _BLOCK_THREADS - 1) // _BLOCK_THREADS

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def select_kernel(
        gathered_scores: fx.Tensor,
        local_idx: fx.Tensor,
        block_table: fx.Tensor,
        out_kv_indices: fx.Tensor,
        out_kv_indptr: fx.Tensor,
        staging: fx.Tensor,
        owned_counts: fx.Tensor,
        rank: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        row_sc = fx.slice(gathered_scores, (row, None))

        # Stage the whole row's candidates in registers, converted to ord space
        # once. The four radix passes and the prior-equal sweep all re-scan
        # these same values; reading them from global each time cost 4 x n_cand
        # loads whose latency dominated the kernel (the data is only 64 KB per
        # row -- this was never bandwidth-bound).
        #
        # The emit loop is deliberately NOT on this path: it walks only this
        # rank's own plane (k_loc, i.e. 1/W of the row) in a different index
        # order, so it re-reads those from global rather than holding the whole
        # row's staging alive across it.
        ords_reg = fx.make_rmem_tensor(steps, Int32)
        for step in range_constexpr(steps):
            c = step * _BLOCK_THREADS + tid
            in_cand = c < fx.Int32(n_cand)
            ords_reg[step] = in_cand.select(
                _f32_to_ord(row_sc[in_cand.select(c, fx.Int32(0))]),
                fx.Int32(_INT32_MIN),
            )

        storage = fx.SharedAllocator().allocate(_make_storage())
        s_hist = storage.bins.peek().view(fx.make_layout(_N_HIST_BINS, 1))
        s_scan = storage.scan.peek().view(fx.make_layout(_NUM_WAVES + 1, 1))
        s_acc = storage.acc.peek().view(fx.make_layout(5, 1))

        def unwrap_val(val):
            return val.ir_value() if hasattr(val, "ir_value") else arith.unwrap(val)

        def warp_inclusive_prefix_i32(val, lane):
            val_raw = unwrap_val(val)
            zero_raw = unwrap_val(0)
            for dpp_op, threshold in (
                (_DPP_ROW_SHR_1, 1),
                (_DPP_ROW_SHR_2, 2),
                (_DPP_ROW_SHR_4, 4),
                (_DPP_ROW_SHR_8, 8),
            ):
                remote = fly_rocdl.update_dpp(
                    T.i32,
                    zero_raw,
                    val_raw,
                    dpp_op,
                    _DPP_ROW_MASK,
                    _DPP_BANK_MASK,
                    True,
                )
                val = (lane >= fx.Int32(threshold)).select(val + fx.Int32(remote), val)
                val_raw = unwrap_val(val)
            src16 = (lane & fx.Int32(0x30)) - 1
            r16 = fly_rocdl.ds_bpermute(T.i32, src16 * fx.Int32(4), val)
            val = (lane >= fx.Int32(16)).select(val + fx.Int32(r16), val)
            src32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
            r32 = fly_rocdl.ds_bpermute(T.i32, src32 * fx.Int32(4), val)
            return (lane >= fx.Int32(32)).select(val + fx.Int32(r32), val)

        def block_exclusive_prefix_i32(val, scan):
            lane = tid % fx.Int32(_WAVE_SIZE)
            warp = tid // fx.Int32(_WAVE_SIZE)
            inclusive = warp_inclusive_prefix_i32(val, lane)
            exclusive = inclusive - val
            if lane == fx.Int32(_WAVE_SIZE - 1):
                scan[warp] = inclusive
            gpu.barrier()
            if warp == 0:
                wv = fx.Int32(0)
                if lane < fx.Int32(_NUM_WAVES):
                    wv = scan[lane]
                wi = warp_inclusive_prefix_i32(wv, lane)
                if lane < fx.Int32(_NUM_WAVES):
                    scan[lane] = wi - wv
                if lane == fx.Int32(_NUM_WAVES - 1):
                    scan[_NUM_WAVES] = wi
            gpu.barrier()
            res = scan[warp] + exclusive
            tot = scan[_NUM_WAVES]
            # Trailing barrier: without it a fast wave entering the *next* call
            # would execute `scan[warp] = inclusive` above while a slow wave is
            # still reading this call's results, corrupting the totals. All call
            # sites are in uniform control flow (constexpr loop bodies), so every
            # thread reaches both barriers.
            gpu.barrier()
            return res, tot

        if tid == 0:
            s_acc[0] = 0
            s_acc[1] = 0
            # Clamp to the candidate count: with topk_tokens > n_cand the radix
            # select can never satisfy `above + cnt >= rem`, would leave the
            # threshold at 0 (the ord of +0.0) and silently emit a wrong set.
            # Clamping degenerates the selection to "take everything", which
            # matches the documented expected total min(topk, world * k_loc).
            s_acc[2] = fx.Int32(min(topk_tokens, n_cand))
        gpu.barrier()

        # 4 radix passes, entirely inside this block.
        for byte_pos in range_constexpr(_NUM_RADIX_PASSES):
            shift = (_NUM_RADIX_PASSES - 1 - byte_pos) * _RADIX_BITS
            xor_val = _RADIX_SIGN_BIT if byte_pos == 0 else 0
            for hi in range_constexpr(
                (_N_HIST_BINS + _BLOCK_THREADS - 1) // _BLOCK_THREADS
            ):
                b = tid + hi * _BLOCK_THREADS
                if b < _N_HIST_BINS:
                    s_hist[b] = 0
            gpu.barrier()

            prefix = s_acc[0]
            dmask = s_acc[1]
            for step in range_constexpr(steps):
                c = step * _BLOCK_THREADS + tid
                if c < fx.Int32(n_cand):
                    ords = ords_reg[step]
                    keep = fx.Int32(1) if const_expr(byte_pos == 0) else fx.Int32(0)
                    if const_expr(byte_pos == 0):
                        pass
                    else:
                        keep = ((ords & dmask) == prefix).select(
                            fx.Int32(1), fx.Int32(0)
                        )
                    if keep != 0:
                        bv = (
                            (ords >> fx.Int32(shift)) & fx.Int32(_RADIX_MASK)
                        ) ^ fx.Int32(xor_val)
                        atomic_add_i32(s_hist, 1, bv, "workgroup")
            gpu.barrier()

            sel = fx.Int32(_N_HIST_BINS - 1) - tid
            cnt = s_hist[sel]
            above, _tot = block_exclusive_prefix_i32(cnt, s_scan)
            rem = s_acc[2]
            # Every thread reads `rem` above, but only the thread whose bucket
            # holds the k-th element writes it below. Without this barrier a
            # fast writer can update s_acc[2] while a slow thread is still
            # reading it; the slow thread then tests the hit condition against
            # the NEW remaining_k, can spuriously satisfy it, and writes a
            # second wrong bucket into s_acc[0] -- a corrupted threshold.
            gpu.barrier()
            if above < rem and above + cnt >= rem:
                actual = sel ^ fx.Int32(xor_val)
                s_acc[0] = prefix | (actual << fx.Int32(shift))
                s_acc[1] = dmask | fx.Int32(uint32_to_int32(_RADIX_MASK << shift))
                s_acc[2] = rem - above
            gpu.barrier()

        # --- own plane only: classify, prefix-sum, map to slot, compact ---
        threshold = s_acc[0]
        remaining_k = s_acc[2]
        base = rank * fx.Int32(k_loc)
        row_local = fx.slice(local_idx, (row, None))
        row_bt = fx.slice(block_table, (row, None))
        row_stage = fx.slice(staging, (row, None))

        if tid == 0:
            s_acc[3] = 0  # emitted-count / write cursor
            s_acc[4] = 0  # equal-to-threshold candidates seen in planes < rank
        gpu.barrier()

        # Count equal-threshold candidates from all planes before this rank
        # (flat positions 0 .. rank*k_loc - 1).  This is the global offset that
        # the admission cap must account for: among the remaining_k equal slots
        # allowed globally, this many are already "claimed" by earlier ranks.
        # Only the TOTAL is needed, never a per-thread offset, so accumulate in
        # registers across the sweep and reduce once at the end. Scanning inside
        # the loop instead cost `steps` block scans (2 barriers each) to produce
        # one number -- 44% of this kernel's runtime at W=8, k_loc=2048.
        prior_eq_local = fx.Int32(0)
        for step in range_constexpr(steps):
            c = step * _BLOCK_THREADS + tid
            in_cand = c < fx.Int32(n_cand)
            before_base = in_cand.select(
                (c < base).select(fx.Int32(1), fx.Int32(0)), fx.Int32(0)
            )
            ords_all = ords_reg[step]
            prior_eq_local = prior_eq_local + (before_base != 0).select(
                (ords_all == threshold).select(fx.Int32(1), fx.Int32(0)), fx.Int32(0)
            )
        _, prior_eq_total = block_exclusive_prefix_i32(prior_eq_local, s_scan)
        if tid == 0:
            s_acc[4] = prior_eq_total
        gpu.barrier()

        # Candidates strictly above the threshold are unconditionally in.
        # Candidates equal to it are admitted in flat-position order, capped by
        # how many equal-valued slots the global selection left for us.
        # s_acc[3] = total emitted so far (write cursor)
        # s_acc[4] = total equal-to-threshold candidates seen so far (cap counter)
        for step in range_constexpr(own_steps):
            t = step * _BLOCK_THREADS + tid
            in_range = t < fx.Int32(k_loc)
            safe_t = in_range.select(t, fx.Int32(0))
            ords = _f32_to_ord(row_sc[base + safe_t])
            j = row_local[safe_t]
            live = in_range.select(
                (j >= 0).select(fx.Int32(1), fx.Int32(0)), fx.Int32(0)
            )
            above = (live != 0).select(
                (ords > threshold).select(fx.Int32(1), fx.Int32(0)), fx.Int32(0)
            )
            equal = (live != 0).select(
                (ords == threshold).select(fx.Int32(1), fx.Int32(0)), fx.Int32(0)
            )
            # ONE packed scan instead of two sequential ones. The naive form
            # scans `equal`, derives `keep` from its prefix, then scans `keep` --
            # a serial pair of barrier chains per step, which cost ~1.2 us/step.
            #
            # It fuses because `above` does not depend on the admission decision:
            # every above-threshold candidate is kept. So scan (above, equal)
            # together, then reconstruct the write offset arithmetically:
            #   dst_off = ab_before + (admitted equals before me)
            # and the admitted-equals prefix is just the equal prefix clamped to
            # whatever room `remaining_k` left.
            packed = (above << fx.Int32(_PACK_SHIFT)) + equal
            pack_before, pack_total = block_exclusive_prefix_i32(packed, s_scan)
            ab_before = pack_before >> fx.Int32(_PACK_SHIFT)
            eq_before = pack_before & fx.Int32(_PACK_MASK)
            ab_total = pack_total >> fx.Int32(_PACK_SHIFT)
            eq_total = pack_total & fx.Int32(_PACK_MASK)

            eq_run = s_acc[4]
            # Room left for equal-valued candidates, globally.
            room = remaining_k - eq_run
            room = (room < 0).select(fx.Int32(0), room)
            admit_eq = (equal != 0).select(
                (eq_before < room).select(fx.Int32(1), fx.Int32(0)), fx.Int32(0)
            )
            keep = above + admit_eq
            # Equals admitted strictly before this thread this step.
            eq_adm_before = (eq_before < room).select(eq_before, room)
            dst_off = ab_before + eq_adm_before
            # Total admitted this step = all aboves + the equals that fit.
            eq_adm_total = (eq_total < room).select(eq_total, room)
            keep_total = ab_total + eq_adm_total
            if keep != 0:
                slot = row_bt[j // fx.Int32(page_size)] * fx.Int32(page_size) + (
                    j % fx.Int32(page_size)
                )
                row_stage[s_acc[3] + dst_off] = slot
            gpu.barrier()
            if tid == 0:
                s_acc[3] = s_acc[3] + keep_total  # advance write cursor by # emitted
                s_acc[4] = s_acc[4] + eq_total  # advance equal-seen count
            gpu.barrier()

        if tid == 0:
            owned_counts[row] = s_acc[3]

    @flyc.kernel(known_block_size=[_BLOCK_THREADS, 1, 1])
    def pack_kernel(
        staging: fx.Tensor,
        owned_counts: fx.Tensor,
        out_kv_indices: fx.Tensor,
        out_kv_indptr: fx.Tensor,
        rows_n: fx.Int32,
    ):
        """One block per row: scan the counts, then copy this row's slots.

        The scan used to be its own grid=(1,) kernel walking the rows on a single
        thread. That is O(rows) serial work between two parallel kernels, and it
        showed: 37% of the op's runtime at rows=128 (15.7 of 42.3 us) to prefix-
        sum 128 integers.

        Instead every block recomputes the prefix sum it needs, in parallel. The
        redundancy is real -- rows blocks each scan rows counts -- but a block
        scan over <=256 values is a handful of barriers, and it removes both the
        serial loop and a kernel launch. Only this row's exclusive prefix and the
        running total are needed, so the scan is over a predicate, not a
        materialised array.

        Block 0 also writes out_kv_indptr, which nothing here reads back: each
        block derives its own `dst` locally, so no block waits on another. That
        keeps the kernel free of any cross-block dependency -- the reason this
        could not simply be merged into select_kernel, where `dst` depends on
        every OTHER row's count.
        """
        row = fx.block_idx.x
        tid = fx.thread_idx.x

        storage = fx.SharedAllocator().allocate(_make_storage())
        s_scan = storage.scan.peek().view(fx.make_layout(_NUM_WAVES + 1, 1))
        s_dst = storage.acc.peek().view(fx.make_layout(5, 1))

        def unwrap_val(val):
            return val.ir_value() if hasattr(val, "ir_value") else arith.unwrap(val)

        def warp_inclusive_prefix_i32(val, lane):
            val_raw = unwrap_val(val)
            zero_raw = unwrap_val(0)
            for dpp_op, threshold in (
                (_DPP_ROW_SHR_1, 1),
                (_DPP_ROW_SHR_2, 2),
                (_DPP_ROW_SHR_4, 4),
                (_DPP_ROW_SHR_8, 8),
            ):
                remote = fly_rocdl.update_dpp(
                    T.i32,
                    zero_raw,
                    val_raw,
                    dpp_op,
                    _DPP_ROW_MASK,
                    _DPP_BANK_MASK,
                    True,
                )
                val = (lane >= fx.Int32(threshold)).select(val + fx.Int32(remote), val)
                val_raw = unwrap_val(val)
            src16 = (lane & fx.Int32(0x30)) - 1
            r16 = fly_rocdl.ds_bpermute(T.i32, src16 * fx.Int32(4), val)
            val = (lane >= fx.Int32(16)).select(val + fx.Int32(r16), val)
            src32 = (lane & fx.Int32(0x30)) - fx.Int32(17)
            r32 = fly_rocdl.ds_bpermute(T.i32, src32 * fx.Int32(4), val)
            return (lane >= fx.Int32(32)).select(val + fx.Int32(r32), val)

        def block_exclusive_prefix_i32(val, scan):
            lane = tid % fx.Int32(_WAVE_SIZE)
            warp = tid // fx.Int32(_WAVE_SIZE)
            inclusive = warp_inclusive_prefix_i32(val, lane)
            exclusive = inclusive - val
            if lane == fx.Int32(_WAVE_SIZE - 1):
                scan[warp] = inclusive
            gpu.barrier()
            if warp == 0:
                wv = fx.Int32(0)
                if lane < fx.Int32(_NUM_WAVES):
                    wv = scan[lane]
                wi = warp_inclusive_prefix_i32(wv, lane)
                if lane < fx.Int32(_NUM_WAVES):
                    scan[lane] = wi - wv
                if lane == fx.Int32(_NUM_WAVES - 1):
                    scan[_NUM_WAVES] = wi
            gpu.barrier()
            res = scan[warp] + exclusive
            tot = scan[_NUM_WAVES]
            gpu.barrier()
            return res, tot

        # Exclusive prefix of the counts, chunked so rows > _BLOCK_THREADS still
        # works -- the caller does not bound rows. Production runs <= 256, a
        # single chunk, but 257 must not silently truncate.
        if tid == 0:
            s_dst[0] = 0
        gpu.barrier()
        base = fx.Int32(0)
        running = fx.Int32(0)
        step_blk = fx.Int32(_BLOCK_THREADS)
        for _ in range(fx.Int32(0), (rows_n + step_blk - 1) // step_blk, fx.Int32(1)):
            r = base + tid
            cnt = (r < rows_n).select(
                owned_counts[(r < rows_n).select(r, fx.Int32(0))], fx.Int32(0)
            )
            excl, chunk_total = block_exclusive_prefix_i32(cnt, s_scan)
            # Exactly one thread holds this block's row, and `excl` is a private
            # register -- publish it through LDS so the whole block can address
            # the output. Without this only thread `row` has a correct `dst` and
            # every other thread writes at offset 0.
            if r == row:
                s_dst[0] = running + excl
            # Block 0 publishes the indptr; every other block only needs `dst`.
            if row == 0 and r < rows_n:
                out_kv_indptr[r + fx.Int32(1)] = running + excl + cnt
            running = running + chunk_total
            base = base + step_blk
        if row == 0 and tid == 0:
            out_kv_indptr[0] = 0
        gpu.barrier()
        dst = s_dst[0]

        n = owned_counts[row]
        row_stage = fx.slice(staging, (row, None))
        for step in range_constexpr(own_steps):
            t = step * _BLOCK_THREADS + tid
            if t < n:
                out_kv_indices[dst + t] = row_stage[t]

    # Fixed launcher signature: 7 tensors
    @flyc.jit
    def launch(
        gathered_scores: fx.Tensor,
        local_idx: fx.Tensor,
        block_table: fx.Tensor,
        out_kv_indices: fx.Tensor,
        out_kv_indptr: fx.Tensor,
        staging: fx.Tensor,
        owned_counts: fx.Tensor,
        rows_m: fx.Int32,
        rank_i32: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008  framework idiom
    ):
        sel = select_kernel(
            gathered_scores,
            local_idx,
            block_table,
            out_kv_indices,
            out_kv_indptr,
            staging,
            owned_counts,
            rank_i32,
        )
        sel.launch(
            grid=(rows_m, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )
        pk = pack_kernel(staging, owned_counts, out_kv_indices, out_kv_indptr, rows_m)
        pk.launch(grid=(rows_m, 1, 1), block=(_BLOCK_THREADS, 1, 1), stream=stream)

    return launch
