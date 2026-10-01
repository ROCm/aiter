# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-workgroup FlyDSL decode TopK."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ..kernels_common import atomic_add_i32, kernel_signature
from .topk_per_row_decode import (
    _f32_to_ord,
    _load_f32x4,
    _row_length,
    _warp_inclusive_prefix_i32,
)

_BLOCK_THREADS = 1024
# Two contiguous float4s per thread, so an 8192-wide row is one scatter scan
# instead of two. Column order stays thread-major: thread t walks [2t, 2t+2).
_SCATTER_PAIR = 2
_VEC = 4
_RADIX_BITS = 11
_NUM_BUCKETS = 1 << _RADIX_BITS
_MID_SHIFT = 10
_LOW_MASK = (1 << _MID_SHIFT) - 1
# A long row rereads the score buffer for every radix pass. Once the first
# threshold is known, the entries that can still win fit in a few thousand
# slots on real indexer scores. Keep those, in column order, and run the
# remaining passes there. Wider ties fall back to the full-row passes.
_COMPACT_LIMIT = 4096
_COMPACT_MIN_ROW = 20_000

_FIRST_ABOVE = 0
_FIRST_THRESHOLD = 1
_SECOND_ABOVE = 2
_SECOND_THRESHOLD = 3
_THIRD_ABOVE = 4
_THIRD_THRESHOLD = 5
_RUNNING_ABOVE = 6
_RUNNING_EQUAL = 7


@cache
def build_topk_per_row_decode_one_workgroup_module(
    k: int,
    wave_size: int,
    write_values: bool = False,
    compact: bool = False,
    pair_scatter: bool = False,
):
    if wave_size not in (32, 64):
        raise ValueError("wave size must be 32 or 64")
    num_waves = _BLOCK_THREADS // wave_size
    output_steps = (k + _BLOCK_THREADS - 1) // _BLOCK_THREADS
    name_params = {
        "k": k,
        "wave": wave_size,
        "wv": write_values,
        "compact": compact,
    }
    if pair_scatter:
        name_params["pair"] = _SCATTER_PAIR

    if compact:

        @fx.struct
        class SharedStorage:
            histogram: fx.Array[fx.Int32, _NUM_BUCKETS, 16]
            scan: fx.Array[fx.Int32, num_waves * 2, 16]
            metadata: fx.Array[fx.Int32, 8, 16]
            cand_bits: fx.Array[fx.Int32, _COMPACT_LIMIT, 16]
            cand_cols: fx.Array[fx.Int32, _COMPACT_LIMIT, 16]

    else:

        @fx.struct
        class SharedStorage:
            histogram: fx.Array[fx.Int32, _NUM_BUCKETS, 16]
            scan: fx.Array[fx.Int32, num_waves * 2, 16]
            metadata: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(
        name="topk_per_row_decode_1wg_" + kernel_signature(**name_params),
        known_block_size=[_BLOCK_THREADS, 1, 1],
    )
    def topk_per_row_decode_one_workgroup_kernel(
        input: fx.Tensor,
        row_ends: fx.Tensor,
        indices: fx.Tensor,
        values: fx.Tensor,
        width: fx.Int32,
        next_n: fx.Int32,
        stride0: fx.Int32,
        write_values: fx.Constexpr[bool],
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        lane = tid % wave_size
        wave = tid // wave_size

        zero = fx.Int32(0)
        one = fx.Int32(1)
        two = fx.Int32(2)
        vec_width = fx.Int32(_VEC)
        block_threads = fx.Int32(_BLOCK_THREADS)
        top_k = fx.Int32(k)
        sign_bit = fx.Int32(-2147483648)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        histogram = storage.histogram.peek().view(fx.make_layout(_NUM_BUCKETS, 1))
        scan = storage.scan.peek().view(fx.make_layout(num_waves * 2, 1))
        metadata = storage.metadata.peek().view(fx.make_layout(8, 1))
        if const_expr(compact):
            cand_bits = storage.cand_bits.peek().view(fx.make_layout(_COMPACT_LIMIT, 1))
            cand_cols = storage.cand_cols.peek().view(fx.make_layout(_COMPACT_LIMIT, 1))

        # Slice the row first, then build the descriptor over it. Built over
        # the whole tensor and sliced afterwards, the row offset has to fit the
        # descriptor's 32-bit byte count and its 32-bit voffset, so anything at
        # or past 4 GiB is unaddressable -- measured on the small-k selector,
        # which had the same shape: at exactly 4 GiB every row came back wrong
        # with nothing raised. A row is 4 MiB at the widest width here.
        input_resource = fx.logical_divide(
            fx.rocdl.make_buffer_tensor(fx.slice(input, (row, None)), max_size=False),
            fx.make_layout(_VEC, 1),
        )
        row_len = _row_length(row, row_ends, width, next_n)
        row_indices = fx.slice(indices, (row, None))
        row_values = fx.slice(values, (row, None))
        row_vectors = (row_len + vec_width - one) // vec_width

        def ordered_key(value):
            return _f32_to_ord(value) ^ sign_bit

        def high_bucket(value):
            return ordered_key(value).shrui(fx.Int32(21))

        def radix_bucket(value, shift, mask):
            return ordered_key(value).shrui(shift) & mask

        def clear_histogram(histogram):
            histogram[tid] = zero
            histogram[tid + _BLOCK_THREADS] = zero
            gpu.barrier()

        def block_exclusive_scan_pair(first, second, scan, metadata):
            first_inclusive = _warp_inclusive_prefix_i32(first, lane, wave_size)
            second_inclusive = _warp_inclusive_prefix_i32(second, lane, wave_size)
            first_exclusive = first_inclusive - first
            second_exclusive = second_inclusive - second
            if lane == wave_size - 1:
                scan[wave] = first_inclusive
                scan[wave + num_waves] = second_inclusive
            gpu.barrier()

            if wave == 0:
                active = lane < num_waves
                safe_lane = active.select(lane, zero)
                first_wave = active.select(scan[safe_lane], zero)
                second_wave = active.select(scan[safe_lane + num_waves], zero)
                first_wave_inclusive = _warp_inclusive_prefix_i32(
                    first_wave, lane, wave_size
                )
                second_wave_inclusive = _warp_inclusive_prefix_i32(
                    second_wave, lane, wave_size
                )
                if active:
                    scan[lane] = first_wave_inclusive - first_wave
                    scan[lane + num_waves] = second_wave_inclusive - second_wave
                if lane == num_waves - 1:
                    metadata[_THIRD_ABOVE] = first_wave_inclusive
                    metadata[_THIRD_THRESHOLD] = second_wave_inclusive
            gpu.barrier()
            return (
                scan[wave] + first_exclusive,
                scan[wave + num_waves] + second_exclusive,
                metadata[_THIRD_ABOVE],
                metadata[_THIRD_THRESHOLD],
            )

        def choose_threshold(
            target_k,
            above_slot,
            threshold_slot,
            histogram,
            scan,
            metadata,
        ):
            first_bin = tid * two
            count0 = histogram[first_bin]
            count1 = histogram[first_bin + one]
            local_total = count0 + count1
            wave_inclusive = _warp_inclusive_prefix_i32(local_total, lane, wave_size)
            wave_exclusive = wave_inclusive - local_total

            if lane == wave_size - 1:
                scan[wave] = wave_inclusive
            gpu.barrier()

            if wave == 0:
                active = lane < num_waves
                safe_lane = active.select(lane, zero)
                wave_total = active.select(scan[safe_lane], zero)
                wave_prefix = (
                    _warp_inclusive_prefix_i32(wave_total, lane, wave_size) - wave_total
                )
                if active:
                    scan[lane + num_waves] = wave_prefix
            gpu.barrier()

            wave_offset = scan[wave + num_waves]
            total = scan[num_waves - 1] + scan[num_waves * 2 - 1]
            target_prefix = total - target_k
            exclusive0 = wave_offset + wave_exclusive
            inclusive0 = exclusive0 + count0
            inclusive1 = inclusive0 + count1

            def emit(bucket, exclusive, inclusive, metadata):
                if (exclusive <= target_prefix) & (inclusive > target_prefix):
                    metadata[threshold_slot] = bucket
                    metadata[above_slot] = total - inclusive

            emit(first_bin, exclusive0, inclusive0, metadata)
            emit(first_bin + one, inclusive0, inclusive1, metadata)
            gpu.barrier()

        def reread_row(chunk):
            for vector_idx in range(tid, row_vectors, block_threads):
                col_base = vector_idx * vec_width
                chunk(col_base, _load_f32x4(input_resource, vector_idx))

        def histogram_pass1(col_base, values):
            for lane_idx in range_constexpr(_VEC):
                col = col_base + lane_idx
                if col < row_len:
                    atomic_add_i32(
                        histogram,
                        one,
                        high_bucket(values[lane_idx]),
                        "workgroup",
                    )

        def histogram_pass2(col_base, values, first_threshold):
            for lane_idx in range_constexpr(_VEC):
                col = col_base + lane_idx
                if col < row_len:
                    value = values[lane_idx]
                    if high_bucket(value) == first_threshold:
                        atomic_add_i32(
                            histogram,
                            one,
                            radix_bucket(
                                value,
                                fx.Int32(_MID_SHIFT),
                                fx.Int32(_NUM_BUCKETS - 1),
                            ),
                            "workgroup",
                        )

        def histogram_pass3(
            col_base,
            values,
            first_threshold,
            second_threshold,
        ):
            for lane_idx in range_constexpr(_VEC):
                col = col_base + lane_idx
                if col < row_len:
                    value = values[lane_idx]
                    if (high_bucket(value) == first_threshold) & (
                        radix_bucket(
                            value,
                            fx.Int32(_MID_SHIFT),
                            fx.Int32(_NUM_BUCKETS - 1),
                        )
                        == second_threshold
                    ):
                        atomic_add_i32(
                            histogram,
                            one,
                            radix_bucket(value, zero, fx.Int32(_LOW_MASK)),
                            "workgroup",
                        )

        def classify(
            value,
            first_threshold,
            second_threshold,
            third_threshold,
        ):
            first = high_bucket(value)
            second = radix_bucket(
                value,
                fx.Int32(_MID_SHIFT),
                fx.Int32(_NUM_BUCKETS - 1),
            )
            third = radix_bucket(value, zero, fx.Int32(_LOW_MASK))
            above = (first > first_threshold) | (
                (first == first_threshold)
                & (
                    (second > second_threshold)
                    | ((second == second_threshold) & (third > third_threshold))
                )
            )
            equal = (
                (first == first_threshold)
                & (second == second_threshold)
                & (third == third_threshold)
            )
            return above, equal

        def stable_scatter(
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            row_indices,
            row_values,
            scan,
            metadata,
        ):
            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()

            if const_expr(pair_scatter):
                span = fx.Int32(_BLOCK_THREADS * _SCATTER_PAIR)
                num_steps = (row_vectors + span - one) // span
                classes = fx.make_rmem_tensor(_SCATTER_PAIR * _VEC, fx.Int32)
                stored = fx.make_rmem_tensor(_SCATTER_PAIR * _VEC, fx.Float32)
                for step in range(zero, num_steps, one):
                    vector_base = step * span + tid * fx.Int32(_SCATTER_PAIR)
                    local_above = zero
                    local_equal = zero
                    for vec_offset in range_constexpr(_SCATTER_PAIR):
                        vector_idx = vector_base + vec_offset
                        active_vector = vector_idx < row_vectors
                        safe_vector_idx = active_vector.select(vector_idx, zero)
                        col_base = safe_vector_idx * vec_width
                        rvals = _load_f32x4(input_resource, safe_vector_idx)
                        for lane_idx in range_constexpr(_VEC):
                            col = col_base + lane_idx
                            above, equal = classify(
                                rvals[lane_idx],
                                first_threshold,
                                second_threshold,
                                third_threshold,
                            )
                            active = active_vector & (col < row_len)
                            above_i32 = (active & above).select(one, zero)
                            equal_i32 = (active & equal).select(one, zero)
                            slot = vec_offset * _VEC + lane_idx
                            classes[slot] = above_i32 * two + equal_i32
                            stored[slot] = rvals[lane_idx]
                            local_above = local_above + above_i32
                            local_equal = local_equal + equal_i32

                    (
                        above_prefix,
                        equal_prefix,
                        block_above,
                        block_equal,
                    ) = block_exclusive_scan_pair(
                        local_above, local_equal, scan, metadata
                    )
                    my_above = metadata[_RUNNING_ABOVE] + above_prefix
                    my_equal = metadata[_RUNNING_EQUAL] + equal_prefix
                    for vec_offset in range_constexpr(_SCATTER_PAIR):
                        vector_idx = vector_base + vec_offset
                        safe_vector_idx = (vector_idx < row_vectors).select(
                            vector_idx, zero
                        )
                        col_base = safe_vector_idx * vec_width
                        for lane_idx in range_constexpr(_VEC):
                            slot = vec_offset * _VEC + lane_idx
                            cls = classes[slot]
                            col = col_base + lane_idx
                            accepted_equal = (my_equal < num_needed).select(
                                my_equal, num_needed
                            )
                            out_pos = my_above + accepted_equal
                            if cls == two:
                                row_indices[out_pos] = col
                                if const_expr(write_values):
                                    row_values[out_pos] = stored[slot]
                                my_above = my_above + one
                            elif cls == one:
                                if my_equal < num_needed:
                                    row_indices[out_pos] = col
                                    if const_expr(write_values):
                                        row_values[out_pos] = stored[slot]
                                my_equal = my_equal + one
                    if tid == 0:
                        metadata[_RUNNING_ABOVE] = (
                            metadata[_RUNNING_ABOVE] + block_above
                        )
                        metadata[_RUNNING_EQUAL] = (
                            metadata[_RUNNING_EQUAL] + block_equal
                        )
                    gpu.barrier()
            else:
                num_steps = (row_vectors + block_threads - one) // block_threads
                for step in range(zero, num_steps, one):
                    vector_idx = step * block_threads + tid
                    active_vector = vector_idx < row_vectors
                    safe_vector_idx = active_vector.select(vector_idx, zero)
                    col_base = safe_vector_idx * vec_width
                    rvals = _load_f32x4(input_resource, safe_vector_idx)
                    classes = fx.make_rmem_tensor(_VEC, fx.Int32)
                    local_above = zero
                    local_equal = zero
                    for lane_idx in range_constexpr(_VEC):
                        col = col_base + lane_idx
                        above, equal = classify(
                            rvals[lane_idx],
                            first_threshold,
                            second_threshold,
                            third_threshold,
                        )
                        active = active_vector & (col < row_len)
                        above_i32 = (active & above).select(one, zero)
                        equal_i32 = (active & equal).select(one, zero)
                        classes[lane_idx] = above_i32 * two + equal_i32
                        local_above = local_above + above_i32
                        local_equal = local_equal + equal_i32

                    (
                        above_prefix,
                        equal_prefix,
                        block_above,
                        block_equal,
                    ) = block_exclusive_scan_pair(
                        local_above, local_equal, scan, metadata
                    )
                    my_above = metadata[_RUNNING_ABOVE] + above_prefix
                    my_equal = metadata[_RUNNING_EQUAL] + equal_prefix
                    for lane_idx in range_constexpr(_VEC):
                        cls = classes[lane_idx]
                        col = col_base + lane_idx
                        accepted_equal = (my_equal < num_needed).select(
                            my_equal, num_needed
                        )
                        out_pos = my_above + accepted_equal
                        if cls == two:
                            row_indices[out_pos] = col
                            if const_expr(write_values):
                                row_values[out_pos] = rvals[lane_idx]
                            my_above = my_above + one
                        elif cls == one:
                            if my_equal < num_needed:
                                row_indices[out_pos] = col
                                if const_expr(write_values):
                                    row_values[out_pos] = rvals[lane_idx]
                            my_equal = my_equal + one
                    if tid == 0:
                        metadata[_RUNNING_ABOVE] = (
                            metadata[_RUNNING_ABOVE] + block_above
                        )
                        metadata[_RUNNING_EQUAL] = (
                            metadata[_RUNNING_EQUAL] + block_equal
                        )
                    gpu.barrier()

        def finish_global(
            first_threshold,
            first_above,
            histogram,
            scan,
            metadata,
            row_indices,
            row_values,
        ):
            clear_histogram(histogram)
            reread_row(
                lambda col, values: histogram_pass2(col, values, first_threshold)
            )
            gpu.barrier()
            need_after_first = top_k - first_above
            choose_threshold(
                need_after_first,
                _SECOND_ABOVE,
                _SECOND_THRESHOLD,
                histogram,
                scan,
                metadata,
            )
            second_threshold = metadata[_SECOND_THRESHOLD]

            clear_histogram(histogram)
            reread_row(
                lambda col, values: histogram_pass3(
                    col,
                    values,
                    first_threshold,
                    second_threshold,
                )
            )
            gpu.barrier()
            need_after_second = need_after_first - metadata[_SECOND_ABOVE]
            choose_threshold(
                need_after_second,
                _THIRD_ABOVE,
                _THIRD_THRESHOLD,
                histogram,
                scan,
                metadata,
            )
            third_threshold = metadata[_THIRD_THRESHOLD]
            num_needed = need_after_second - metadata[_THIRD_ABOVE]
            stable_scatter(
                first_threshold,
                second_threshold,
                third_threshold,
                num_needed,
                row_indices,
                row_values,
                scan,
                metadata,
            )

        def finish_compact(
            first_threshold,
            first_above,
            candidate_count,
            cand_bits,
            cand_cols,
            histogram,
            scan,
            metadata,
            row_indices,
            row_values,
        ):
            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
            gpu.barrier()
            num_steps = (row_vectors + block_threads - one) // block_threads
            for step in range(zero, num_steps, one):
                vector_idx = step * block_threads + tid
                active_vector = vector_idx < row_vectors
                safe_vector_idx = active_vector.select(vector_idx, zero)
                col_base = safe_vector_idx * vec_width
                rvals = _load_f32x4(input_resource, safe_vector_idx)
                flags = fx.make_rmem_tensor(_VEC, fx.Int32)
                for lane_idx in range_constexpr(_VEC):
                    col = col_base + lane_idx
                    bucket = high_bucket(rvals[lane_idx])
                    keep = (
                        active_vector
                        & (col < row_len)
                        & ((bucket > first_threshold) | (bucket == first_threshold))
                    )
                    flags[lane_idx] = keep.select(one, zero)
                local = flags[0] + flags[1] + flags[2] + flags[3]
                (
                    prefix,
                    _equal_prefix,
                    block_total,
                    _block_equal,
                ) = block_exclusive_scan_pair(local, zero, scan, metadata)
                cursor = metadata[_RUNNING_ABOVE] + prefix
                for lane_idx in range_constexpr(_VEC):
                    if flags[lane_idx] == one:
                        cand_bits[cursor] = rvals[lane_idx].bitcast(fx.Int32)
                        cand_cols[cursor] = col_base + lane_idx
                        cursor = cursor + one
                if tid == 0:
                    metadata[_RUNNING_ABOVE] = metadata[_RUNNING_ABOVE] + block_total
                gpu.barrier()

            clear_histogram(histogram)
            for index in range(tid, candidate_count, block_threads):
                value = cand_bits[index].bitcast(fx.Float32)
                if high_bucket(value) == first_threshold:
                    atomic_add_i32(
                        histogram,
                        one,
                        radix_bucket(
                            value,
                            fx.Int32(_MID_SHIFT),
                            fx.Int32(_NUM_BUCKETS - 1),
                        ),
                        "workgroup",
                    )
            gpu.barrier()
            need_after_first = top_k - first_above
            choose_threshold(
                need_after_first,
                _SECOND_ABOVE,
                _SECOND_THRESHOLD,
                histogram,
                scan,
                metadata,
            )
            second_threshold = metadata[_SECOND_THRESHOLD]

            clear_histogram(histogram)
            for index in range(tid, candidate_count, block_threads):
                value = cand_bits[index].bitcast(fx.Float32)
                mid = radix_bucket(
                    value,
                    fx.Int32(_MID_SHIFT),
                    fx.Int32(_NUM_BUCKETS - 1),
                )
                if (high_bucket(value) == first_threshold) & (mid == second_threshold):
                    atomic_add_i32(
                        histogram,
                        one,
                        radix_bucket(value, zero, fx.Int32(_LOW_MASK)),
                        "workgroup",
                    )
            gpu.barrier()
            need_after_second = need_after_first - metadata[_SECOND_ABOVE]
            choose_threshold(
                need_after_second,
                _THIRD_ABOVE,
                _THIRD_THRESHOLD,
                histogram,
                scan,
                metadata,
            )
            third_threshold = metadata[_THIRD_THRESHOLD]
            num_needed = need_after_second - metadata[_THIRD_ABOVE]

            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()
            scatter_steps = (candidate_count + block_threads - one) // block_threads
            for step in range(zero, scatter_steps, one):
                index = step * block_threads + tid
                active = index < candidate_count
                safe = active.select(index, zero)
                value = cand_bits[safe].bitcast(fx.Float32)
                col = cand_cols[safe]
                above, equal = classify(
                    value,
                    first_threshold,
                    second_threshold,
                    third_threshold,
                )
                above_i32 = (active & above).select(one, zero)
                equal_i32 = (active & equal).select(one, zero)
                (
                    above_prefix,
                    equal_prefix,
                    block_above,
                    block_equal,
                ) = block_exclusive_scan_pair(above_i32, equal_i32, scan, metadata)
                my_above = metadata[_RUNNING_ABOVE] + above_prefix
                my_equal = metadata[_RUNNING_EQUAL] + equal_prefix
                accepted_equal = (my_equal < num_needed).select(my_equal, num_needed)
                out_pos = my_above + accepted_equal
                cls = above_i32 * two + equal_i32
                if cls == two:
                    row_indices[out_pos] = col
                    if const_expr(write_values):
                        row_values[out_pos] = value
                elif cls == one:
                    if my_equal < num_needed:
                        row_indices[out_pos] = col
                        if const_expr(write_values):
                            row_values[out_pos] = value
                if tid == 0:
                    metadata[_RUNNING_ABOVE] = metadata[_RUNNING_ABOVE] + block_above
                    metadata[_RUNNING_EQUAL] = metadata[_RUNNING_EQUAL] + block_equal
                gpu.barrier()

        if row_len <= top_k:
            for output_step in range_constexpr(output_steps):
                out_pos = output_step * _BLOCK_THREADS + tid
                if out_pos < k:
                    valid = out_pos < row_len
                    row_indices[out_pos] = valid.select(out_pos, fx.Int32(-1))
                    if const_expr(write_values):
                        row_values[out_pos] = valid.select(
                            input[row, out_pos],
                            fx.Float32(float("-inf")),
                        )

        if row_len > top_k:
            if tid < 8:
                metadata[tid] = zero
            gpu.barrier()

            clear_histogram(histogram)
            reread_row(histogram_pass1)
            gpu.barrier()
            choose_threshold(
                top_k,
                _FIRST_ABOVE,
                _FIRST_THRESHOLD,
                histogram,
                scan,
                metadata,
            )
            first_threshold = metadata[_FIRST_THRESHOLD]
            first_above = metadata[_FIRST_ABOVE]
            if const_expr(compact):
                candidate_count = first_above + histogram[first_threshold]
                use_compact = (
                    (row_len > fx.Int32(_COMPACT_MIN_ROW))
                    & (candidate_count > zero)
                    & (candidate_count <= fx.Int32(_COMPACT_LIMIT))
                )
                if use_compact:
                    finish_compact(
                        first_threshold,
                        first_above,
                        candidate_count,
                        cand_bits,
                        cand_cols,
                        histogram,
                        scan,
                        metadata,
                        row_indices,
                        row_values,
                    )
                else:
                    finish_global(
                        first_threshold,
                        first_above,
                        histogram,
                        scan,
                        metadata,
                        row_indices,
                        row_values,
                    )
            else:
                finish_global(
                    first_threshold,
                    first_above,
                    histogram,
                    scan,
                    metadata,
                    row_indices,
                    row_values,
                )

    @flyc.jit
    def launch_topk_per_row_decode_one_workgroup(
        input: fx.Tensor,
        row_ends: fx.Tensor,
        indices: fx.Tensor,
        values: fx.Tensor,
        width: fx.Int32,
        next_n: fx.Int32,
        stride0: fx.Int32,
        rows_m: fx.Int32,
        stream: fx.Stream,
    ):
        topk_per_row_decode_one_workgroup_kernel(
            input,
            row_ends,
            indices,
            values,
            width,
            next_n,
            stride0,
            write_values,
        ).launch(
            grid=(rows_m, 1, 1),
            block=(_BLOCK_THREADS, 1, 1),
            stream=stream,
        )

    return launch_topk_per_row_decode_one_workgroup
