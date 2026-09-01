# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""One-workgroup FlyDSL decode TopK."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from .kernels_common import atomic_add_i32
from .topk_per_row_decode import (
    _f32_to_ord,
    _load_f32x4,
    _row_length,
    _warp_inclusive_prefix_i32,
)

_BLOCK_THREADS = 1024
_VEC = 4
_RADIX_BITS = 11
_NUM_BUCKETS = 1 << _RADIX_BITS
_MID_SHIFT = 10
_LOW_MASK = (1 << _MID_SHIFT) - 1

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
):
    if wave_size not in (32, 64):
        raise ValueError("wave size must be 32 or 64")
    num_waves = _BLOCK_THREADS // wave_size
    output_steps = (k + _BLOCK_THREADS - 1) // _BLOCK_THREADS

    @fx.struct
    class SharedStorage:
        histogram: fx.Array[fx.Int32, _NUM_BUCKETS, 16]
        scan: fx.Array[fx.Int32, num_waves * 2, 16]
        metadata: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(
        name=f"topk_per_row_decode_1wg_k{k}",
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

        input_buffer = fx.rocdl.make_buffer_tensor(input, max_size=False)
        input_resource = fx.logical_divide(
            fx.slice(input_buffer, (row, None)), fx.make_layout(_VEC, 1)
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
                ) = block_exclusive_scan_pair(local_above, local_equal, scan, metadata)
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

            clear_histogram(histogram)
            reread_row(
                lambda col, values: histogram_pass2(col, values, first_threshold)
            )
            gpu.barrier()
            need_after_first = top_k - metadata[_FIRST_ABOVE]
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
