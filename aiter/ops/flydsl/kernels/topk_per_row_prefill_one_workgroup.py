# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 prefill TopK with one workgroup per input row.

Short rows cache ordered keys in LDS. Long rows compact the selected radix
bucket, while stable modes add deterministic index ordering and tie-breaking.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr

from .kernels_common import atomic_add_i32
from .topk_per_row_decode import _load_f32x4, _warp_inclusive_prefix_i32

_WAVE_SIZE = 32
_VEC = 4
_LOAD_UNROLL = 4
_KEY_BITS = 32
_HIGH_BITS = 12
_MIDDLE_BITS = 10
_LOW_BITS = _KEY_BITS - _HIGH_BITS - _MIDDLE_BITS
_HIGH_BUCKETS = 1 << _HIGH_BITS
_LATER_BUCKETS = 1 << max(_MIDDLE_BITS, _LOW_BITS)
_HIGH_SHIFT = _MIDDLE_BITS + _LOW_BITS
_MIDDLE_SHIFT = _LOW_BITS
_MIDDLE_MASK = (1 << _MIDDLE_BITS) - 1
_LOW_MASK = (1 << _LOW_BITS) - 1
_SHORT_HIGH_BITS = 11
_SHORT_MIDDLE_BITS = 10
_SHORT_LOW_BITS = _KEY_BITS - _SHORT_HIGH_BITS - _SHORT_MIDDLE_BITS
_SHORT_HIGH_BUCKETS = 1 << _SHORT_HIGH_BITS
_SHORT_HIGH_SHIFT = _SHORT_MIDDLE_BITS + _SHORT_LOW_BITS
_SHORT_MIDDLE_SHIFT = _SHORT_LOW_BITS
_SHORT_MIDDLE_MASK = (1 << _SHORT_MIDDLE_BITS) - 1
_SHORT_LOW_MASK = (1 << _SHORT_LOW_BITS) - 1
_MAX_ROW_ELEMENTS = ((1 << 32) - 1) // 4
_COMPACT_CAPACITY = 4096
_STABLE_FAST_MIN_ROW_LEN = 1 << 15

_FIRST_ABOVE = 0
_FIRST_THRESHOLD = 1
_SECOND_ABOVE = 2
_SECOND_THRESHOLD = 3
_THIRD_ABOVE = 4
_THIRD_THRESHOLD = 5
_RUNNING_ABOVE = 6
_RUNNING_EQUAL = 7
_CANDIDATE_COUNT = 8
_SELECTED_BUCKET_COUNT = 9
_METADATA_SIZE = _SELECTED_BUCKET_COUNT + 1
_PACKED_COUNT_BITS = 16
_PACKED_COUNT_MASK = (1 << _PACKED_COUNT_BITS) - 1


def _build_bitonic_schedule(capacity: int) -> tuple[tuple[int, ...], ...]:
    sizes = []
    strides = []
    size = 2
    while size <= capacity:
        stride = size // 2
        while stride:
            sizes.append(size)
            strides.append(stride)
            stride //= 2
        size *= 2
    return tuple(sizes), tuple(strides)


@cache
def build_topk_per_row_prefill_one_workgroup_module(
    k: int,
    block_threads: int = 1024,
    write_values: bool = False,
    stable: bool = False,
    device_index: int = 0,
    backend: str = "rocm",
):
    del device_index, backend
    if k <= 0:
        raise ValueError("k must be positive")
    if block_threads not in (256, 1024):
        raise ValueError("block_threads must be 256 or 1024")
    if block_threads * _VEC > _PACKED_COUNT_MASK:
        raise ValueError("one scan tile exceeds the packed count range")

    num_waves = block_threads // _WAVE_SIZE
    high_bins_per_thread = _HIGH_BUCKETS // block_threads
    later_bins_per_thread = _LATER_BUCKETS // block_threads
    short_bins_per_thread = _SHORT_HIGH_BUCKETS // block_threads
    full_key_vector_steps = (
        (_COMPACT_CAPACITY // _VEC) + block_threads - 1
    ) // block_threads
    stable_sort_enabled = stable and block_threads == 1024 and k <= 2048
    stable_sort_capacity = (
        1 << (k - 1).bit_length() if stable_sort_enabled else 1
    )
    stable_stage_capacity = (
        k if stable_sort_enabled else 1
    )
    stable_sort_items_per_thread = (
        stable_sort_capacity + block_threads - 1
    ) // block_threads
    stable_sort_sizes, stable_sort_strides = _build_bitonic_schedule(
        stable_sort_capacity
    )
    output_vector_count = k // _VEC
    output_vector_steps = (
        output_vector_count + block_threads - 1
    ) // block_threads
    output_vector_elems = max(_VEC, output_vector_count * _VEC)

    # LDS layouts

    @fx.struct
    class LongPass1Storage:
        histogram0: fx.Array[fx.Int32, _HIGH_BUCKETS, 16]
        histogram1: fx.Array[fx.Int32, _HIGH_BUCKETS, 16]

    @fx.struct
    class LongLaterStorage:
        histogram: fx.Array[fx.Int32, _LATER_BUCKETS, 16]
        candidate_keys: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]
        candidate_indices: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]
        stable_keys: fx.Array[fx.Int32, stable_stage_capacity, 16]
        stable_indices: fx.Array[fx.Int32, stable_stage_capacity, 16]

    @fx.struct
    class ShortPass1Storage:
        histogram0: fx.Array[fx.Int32, _SHORT_HIGH_BUCKETS, 16]
        histogram1: fx.Array[fx.Int32, _SHORT_HIGH_BUCKETS, 16]
        full_keys: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]

    @fx.union
    class ArenaStorage:
        long_pass1: LongPass1Storage
        long_later: LongLaterStorage
        short_pass1: ShortPass1Storage

    @fx.struct
    class SharedStorage:
        arena: ArenaStorage
        scan: fx.Array[fx.Int32, num_waves * 2, 16]
        metadata: fx.Array[fx.Int32, _METADATA_SIZE, 16]

    @flyc.kernel(
        name=(
            f"topk_per_row_prefill_1wg_gfx1250_k{k}_b{block_threads}"
            f"_v{int(write_values)}_s{int(stable)}"
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def topk_per_row_prefill_one_workgroup_kernel(
        input: fx.Tensor,
        row_starts: fx.Tensor,
        row_ends: fx.Tensor,
        indices: fx.Tensor,
        value_output: fx.Tensor,
    ):
        row = fx.Int32(fx.block_idx.x)
        tid = fx.thread_idx.x
        lane = tid % _WAVE_SIZE
        wave = tid // _WAVE_SIZE

        zero = fx.Int32(0)
        one = fx.Int32(1)
        above_class = fx.Int32(2)
        vec_width = fx.Int32(_VEC)
        block_size = fx.Int32(block_threads)
        top_k = fx.Int32(k)
        sign_bit = fx.Int32(-2147483648)

        # Row views
        row_start = row_starts[row]
        row_len = row_ends[row] - row_start
        physical_row = fx.slice(input, (row, None))
        input_row_iter = fx.add_offset(
            fx.get_iter(physical_row), row_start
        )
        input_row_addr = fx.Int64(ptrtoint(input_row_iter))
        input_row = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                input_row_iter,
                fx.make_layout(_MAX_ROW_ELEMENTS, 1),
            ),
            num_records_bytes=fx.Int64(row_len) * fx.Int64(4),
        )

        misalign = input_row_addr & fx.Int64(15)
        skip_cnt = fx.Int32(
            (misalign == fx.Int64(0)).select(
                fx.Int64(0),
                (fx.Int64(16) - misalign) / fx.Int64(4),
            )
        )
        skip_cnt = (skip_cnt > row_len).select(row_len, skip_cnt)
        aligned_len = row_len - skip_cnt
        len_cast = aligned_len // vec_width
        aligned_iter = fx.add_offset(input_row_iter, skip_cnt)
        aligned_row = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                aligned_iter,
                fx.make_layout(_MAX_ROW_ELEMENTS, 1),
            ),
            num_records_bytes=fx.Int64(aligned_len) * fx.Int64(4),
        )
        input_resource = fx.logical_divide(
            aligned_row, fx.make_layout(_VEC, 1)
        )

        row_indices = fx.slice(indices, (row, None))
        row_values = fx.slice(value_output, (row, None))
        row_index_tiles = fx.logical_divide(
            fx.make_view(
                fx.get_iter(row_indices),
                fx.make_layout(output_vector_elems, 1),
            ),
            fx.make_layout(_VEC, 1),
        )
        index_store_atom = fx.make_copy_atom(
            fx.UniversalCopy128b(), fx.Int32
        )
        index_fragment_layout = fx.make_layout(_VEC, 1)
        row_value_tiles = fx.logical_divide(
            fx.make_view(
                fx.get_iter(row_values),
                fx.make_layout(output_vector_elems, 1),
            ),
            fx.make_layout(_VEC, 1),
        )
        value_store_atom = fx.make_copy_atom(
            fx.UniversalCopy128b(), fx.Float32
        )
        value_fragment_layout = fx.make_layout(_VEC, 1)

        storage = fx.SharedAllocator().allocate(SharedStorage)
        pass1_histogram0 = storage.arena.long_pass1.histogram0.peek().view(
            fx.make_layout(_HIGH_BUCKETS, 1)
        )
        pass1_histogram1 = storage.arena.long_pass1.histogram1.peek().view(
            fx.make_layout(_HIGH_BUCKETS, 1)
        )
        histogram = storage.arena.long_later.histogram.peek().view(
            fx.make_layout(_LATER_BUCKETS, 1)
        )
        short_pass1_histogram0 = (
            storage.arena.short_pass1.histogram0.peek().view(
                fx.make_layout(_SHORT_HIGH_BUCKETS, 1)
            )
        )
        short_pass1_histogram1 = (
            storage.arena.short_pass1.histogram1.peek().view(
                fx.make_layout(_SHORT_HIGH_BUCKETS, 1)
            )
        )
        short_histogram = short_pass1_histogram0
        scan = storage.scan.peek().view(fx.make_layout(num_waves * 2, 1))
        metadata = storage.metadata.peek().view(
            fx.make_layout(_METADATA_SIZE, 1)
        )
        candidate_keys = storage.arena.long_later.candidate_keys.peek().view(
            fx.make_layout(_COMPACT_CAPACITY, 1)
        )
        candidate_indices = (
            storage.arena.long_later.candidate_indices.peek().view(
                fx.make_layout(_COMPACT_CAPACITY, 1)
            )
        )
        stable_keys = storage.arena.long_later.stable_keys.peek().view(
            fx.make_layout(stable_stage_capacity, 1)
        )
        stable_indices = (
            storage.arena.long_later.stable_indices.peek().view(
                fx.make_layout(stable_stage_capacity, 1)
            )
        )
        full_keys = storage.arena.short_pass1.full_keys.peek().view(
            fx.make_layout(_COMPACT_CAPACITY, 1)
        )
        full_key_tiles = fx.logical_divide(
            full_keys, fx.make_layout(_VEC, 1)
        )
        full_key_load_atom = fx.make_copy_atom(
            fx.UniversalCopy128b(), fx.Int32
        )
        full_key_fragment_layout = fx.make_layout(_VEC, 1)

        # Key encoding and classification

        def ordered_key(value):
            bits = value.bitcast(fx.Int32)
            return (
                bits
                ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF))
                ^ sign_bit
            )

        def high_bucket(key):
            return key.shrui(fx.Int32(_HIGH_SHIFT))

        def middle_bucket(key):
            return (
                key.shrui(fx.Int32(_MIDDLE_SHIFT))
                & fx.Int32(_MIDDLE_MASK)
            )

        def low_bucket(key):
            return key & fx.Int32(_LOW_MASK)

        def short_high_bucket(key):
            return key.shrui(fx.Int32(_SHORT_HIGH_SHIFT))

        def short_middle_bucket(key):
            return (
                key.shrui(fx.Int32(_SHORT_MIDDLE_SHIFT))
                & fx.Int32(_SHORT_MIDDLE_MASK)
            )

        def short_low_bucket(key):
            return key & fx.Int32(_SHORT_LOW_MASK)

        def classify_levels(
            first,
            second,
            third,
            first_threshold,
            second_threshold,
            third_threshold,
            levels,
        ):
            if const_expr(levels == 1):
                above = first > first_threshold
                equal = first == first_threshold
            elif const_expr(levels == 2):
                above = (first > first_threshold) | (
                    (first == first_threshold)
                    & (second > second_threshold)
                )
                equal = (
                    (first == first_threshold)
                    & (second == second_threshold)
                )
            else:
                above = (first > first_threshold) | (
                    (first == first_threshold)
                    & (
                        (second > second_threshold)
                        | (
                            (second == second_threshold)
                            & (third > third_threshold)
                        )
                    )
                )
                equal = (
                    (first == first_threshold)
                    & (second == second_threshold)
                    & (third == third_threshold)
                )
            return above, equal

        def classify(
            key,
            first_threshold,
            second_threshold,
            third_threshold,
        ):
            return classify_levels(
                high_bucket(key),
                middle_bucket(key),
                low_bucket(key),
                first_threshold,
                second_threshold,
                third_threshold,
                3,
            )

        def ordered_value(key):
            bits = (key < zero).select(
                key ^ sign_bit, key ^ fx.Int32(-1)
            )
            return bits.bitcast(fx.Float32)

        def threshold_key(first, second, third):
            return (
                first * fx.Int32(1 << _HIGH_SHIFT)
                + second * fx.Int32(1 << _MIDDLE_SHIFT)
                + third
            )

        def short_threshold_key(first, second, third):
            return (
                first * fx.Int32(1 << _SHORT_HIGH_SHIFT)
                + second * fx.Int32(1 << _SHORT_MIDDLE_SHIFT)
                + third
            )

        def store_key_result(pos, col, key, row_indices, row_values):
            row_indices[pos] = row_start + col
            if const_expr(write_values):
                row_values[pos] = ordered_value(key)

        def store_loaded_result(
            pos, col, value, row_indices, row_values
        ):
            row_indices[pos] = row_start + col
            if const_expr(write_values):
                row_values[pos] = value

        def scatter_unstable_key(
            col,
            key,
            above,
            equal,
            num_needed,
            row_indices,
            row_values,
            metadata,
        ):
            if above:
                out_pos = atomic_add_i32(
                    metadata,
                    one,
                    _RUNNING_ABOVE,
                    "workgroup",
                )
                if out_pos < top_k:
                    store_key_result(
                        out_pos,
                        col,
                        key,
                        row_indices,
                        row_values,
                    )
            elif equal:
                back_pos = atomic_add_i32(
                    metadata,
                    one,
                    _RUNNING_EQUAL,
                    "workgroup",
                )
                if back_pos < num_needed:
                    store_key_result(
                        top_k - one - back_pos,
                        col,
                        key,
                        row_indices,
                        row_values,
                    )

        def scatter_stable_equal_row(key, row_indices, row_values):
            for pos in range(tid, top_k, block_size):
                row_indices[pos] = row_start + pos
                if const_expr(write_values):
                    row_values[pos] = ordered_value(key)

        # Histogram and workgroup scan primitives

        def clear_histogram(histogram, bins_per_thread):
            for item in range_constexpr(bins_per_thread):
                histogram[tid + item * block_threads] = zero
            gpu.barrier()

        def clear_pass1_histograms(
            histogram0, histogram1, bins_per_thread
        ):
            for item in range_constexpr(bins_per_thread):
                pos = tid + item * block_threads
                histogram0[pos] = zero
                histogram1[pos] = zero
            gpu.barrier()

        def merge_pass1_histograms(
            histogram0, histogram1, bins_per_thread
        ):
            for item in range_constexpr(bins_per_thread):
                pos = tid + item * block_threads
                histogram0[pos] = histogram0[pos] + histogram1[pos]
            gpu.barrier()

        def block_exclusive_scan_pair(first, second, scan, metadata):
            packed = (
                first * fx.Int32(1 << _PACKED_COUNT_BITS) + second
            )
            packed_inclusive = _warp_inclusive_prefix_i32(
                packed, lane, _WAVE_SIZE
            )
            packed_exclusive = packed_inclusive - packed
            if lane == _WAVE_SIZE - 1:
                scan[wave] = packed_inclusive
            gpu.barrier()

            if wave == 0:
                active = lane < num_waves
                safe_lane = active.select(lane, zero)
                wave_total = active.select(scan[safe_lane], zero)
                wave_inclusive = _warp_inclusive_prefix_i32(
                    wave_total, lane, _WAVE_SIZE
                )
                if active:
                    scan[lane + num_waves] = (
                        wave_inclusive - wave_total
                    )
                if lane == num_waves - 1:
                    # Threshold values are already live in registers here.
                    # Reuse this metadata slot for the packed block total.
                    metadata[_THIRD_ABOVE] = wave_inclusive
            gpu.barrier()
            packed_prefix = (
                scan[wave + num_waves] + packed_exclusive
            )
            packed_total = metadata[_THIRD_ABOVE]
            return (
                packed_prefix.shrui(fx.Int32(_PACKED_COUNT_BITS)),
                packed_prefix & fx.Int32(_PACKED_COUNT_MASK),
                packed_total.shrui(fx.Int32(_PACKED_COUNT_BITS)),
                packed_total & fx.Int32(_PACKED_COUNT_MASK),
            )

        def choose_threshold(
            target_k,
            above_slot,
            threshold_slot,
            count_slot,
            histogram,
            scan,
            metadata,
            bins_per_thread,
        ):
            first_bin = tid * fx.Int32(bins_per_thread)
            counts = fx.make_rmem_tensor(
                bins_per_thread, fx.Int32
            )
            local_total = zero
            for item in range_constexpr(bins_per_thread):
                count = histogram[first_bin + item]
                counts[item] = count
                local_total = local_total + count
            wave_inclusive = _warp_inclusive_prefix_i32(
                local_total, lane, _WAVE_SIZE
            )
            wave_exclusive = wave_inclusive - local_total

            if lane == _WAVE_SIZE - 1:
                scan[wave] = wave_inclusive
            gpu.barrier()

            if wave == 0:
                active = lane < num_waves
                safe_lane = active.select(lane, zero)
                wave_total = active.select(scan[safe_lane], zero)
                wave_prefix = (
                    _warp_inclusive_prefix_i32(
                        wave_total, lane, _WAVE_SIZE
                    )
                    - wave_total
                )
                if active:
                    scan[lane + num_waves] = wave_prefix
            gpu.barrier()

            wave_offset = scan[wave + num_waves]
            total = scan[num_waves - 1] + scan[num_waves * 2 - 1]
            target_prefix = total - target_k
            exclusive = wave_offset + wave_exclusive
            for item in range_constexpr(bins_per_thread):
                inclusive = exclusive + counts[item]
                if (exclusive <= target_prefix) & (
                    inclusive > target_prefix
                ):
                    metadata[threshold_slot] = first_bin + item
                    metadata[above_slot] = total - inclusive
                    metadata[count_slot] = inclusive - exclusive
                exclusive = inclusive
            gpu.barrier()

        # Global and LDS row iterators

        def scan_row(on_vec, on_scalar):
            unroll_stride = block_size * fx.Int32(_LOAD_UNROLL)
            unroll_end = (
                len_cast > block_size * fx.Int32(_LOAD_UNROLL - 1)
            ).select(
                len_cast - block_size * fx.Int32(_LOAD_UNROLL - 1),
                zero,
            )
            for vector_idx in range(tid, unroll_end, unroll_stride):
                vector_idx1 = vector_idx + block_size
                vector_idx2 = vector_idx1 + block_size
                vector_idx3 = vector_idx2 + block_size
                values0 = _load_f32x4(input_resource, vector_idx)
                values1 = _load_f32x4(input_resource, vector_idx1)
                on_vec(skip_cnt + vector_idx * vec_width, values0)
                values2 = _load_f32x4(input_resource, vector_idx2)
                values3 = _load_f32x4(input_resource, vector_idx3)
                on_vec(skip_cnt + vector_idx1 * vec_width, values1)
                on_vec(skip_cnt + vector_idx2 * vec_width, values2)
                on_vec(skip_cnt + vector_idx3 * vec_width, values3)

            n_unroll = (tid < unroll_end).select(
                (unroll_end - one - tid) // unroll_stride + one,
                zero,
            )
            cleanup_start = tid + n_unroll * unroll_stride
            for vector_idx in range(
                cleanup_start, len_cast, block_size
            ):
                on_vec(
                    skip_cnt + vector_idx * vec_width,
                    _load_f32x4(input_resource, vector_idx),
                )

            if tid < skip_cnt:
                on_scalar(tid, input_row[tid])
            remain_col = skip_cnt + len_cast * vec_width + tid
            if remain_col < row_len:
                on_scalar(remain_col, input_row[remain_col])

        def scan_row_reverse(on_vec, on_scalar):
            remain_col = skip_cnt + len_cast * vec_width + tid
            if remain_col < row_len:
                on_scalar(remain_col, input_row[remain_col])

            unroll_stride = block_size * fx.Int32(_LOAD_UNROLL)
            unroll_end = (
                len_cast > block_size * fx.Int32(_LOAD_UNROLL - 1)
            ).select(
                len_cast - block_size * fx.Int32(_LOAD_UNROLL - 1),
                zero,
            )
            for offset in range(tid, unroll_end, unroll_stride):
                vector_idx0 = len_cast - one - offset
                vector_idx1 = vector_idx0 - block_size
                vector_idx2 = vector_idx1 - block_size
                vector_idx3 = vector_idx2 - block_size
                values0 = _load_f32x4(input_resource, vector_idx0)
                values1 = _load_f32x4(input_resource, vector_idx1)
                values2 = _load_f32x4(input_resource, vector_idx2)
                values3 = _load_f32x4(input_resource, vector_idx3)
                on_vec(skip_cnt + vector_idx0 * vec_width, values0)
                on_vec(skip_cnt + vector_idx1 * vec_width, values1)
                on_vec(skip_cnt + vector_idx2 * vec_width, values2)
                on_vec(skip_cnt + vector_idx3 * vec_width, values3)

            n_unroll = (tid < unroll_end).select(
                (unroll_end - one - tid) // unroll_stride + one,
                zero,
            )
            cleanup_offset = tid + n_unroll * unroll_stride
            for offset in range(cleanup_offset, len_cast, block_size):
                vector_idx = len_cast - one - offset
                on_vec(
                    skip_cnt + vector_idx * vec_width,
                    _load_f32x4(input_resource, vector_idx),
                )

            if tid < skip_cnt:
                on_scalar(tid, input_row[tid])

        def scan_full_keys(body):
            row_vectors = (row_len + fx.Int32(_VEC - 1)) // fx.Int32(
                _VEC
            )
            for step in range_constexpr(full_key_vector_steps):
                vector_idx = step * block_threads + tid
                active = vector_idx < row_vectors
                safe_idx = active.select(vector_idx, zero)
                fragment = fx.make_rmem_tensor(
                    full_key_fragment_layout, fx.Int32
                )
                fx.copy_atom_call(
                    full_key_load_atom,
                    fx.slice(full_key_tiles, (None, safe_idx)),
                    fragment,
                )
                values = fragment.load()
                col_base = vector_idx * vec_width
                for item in range_constexpr(_VEC):
                    col = col_base + item
                    if active & (col < row_len):
                        body(col, values[item])

        # Radix histogram passes

        def add_high_bucket(key, histogram0, histogram1):
            bucket = high_bucket(key)
            if (wave & one) == zero:
                atomic_add_i32(
                    histogram0,
                    one,
                    bucket,
                    "workgroup",
                )
            else:
                atomic_add_i32(
                    histogram1,
                    one,
                    bucket,
                    "workgroup",
                )

        def add_short_high_bucket(key, histogram0, histogram1):
            bucket = short_high_bucket(key)
            if (wave & one) == zero:
                atomic_add_i32(
                    histogram0,
                    one,
                    bucket,
                    "workgroup",
                )
            else:
                atomic_add_i32(
                    histogram1,
                    one,
                    bucket,
                    "workgroup",
                )

        def pass1_one(col, value, histogram0, histogram1):
            if col < row_len:
                add_high_bucket(
                    ordered_key(value), histogram0, histogram1
                )

        def pass1_vec(col, values, histogram0, histogram1):
            for item in range_constexpr(_VEC):
                pass1_one(
                    col + item,
                    values[item],
                    histogram0,
                    histogram1,
                )

        def pass1_cache_one(
            col,
            value,
            histogram0,
            histogram1,
            full_keys,
        ):
            if col < row_len:
                key = ordered_key(value)
                full_keys[col] = key
                add_short_high_bucket(key, histogram0, histogram1)

        def pass1_cache_vec(
            col,
            values,
            histogram0,
            histogram1,
            full_keys,
        ):
            for item in range_constexpr(_VEC):
                pass1_cache_one(
                    col + item,
                    values[item],
                    histogram0,
                    histogram1,
                    full_keys,
                )

        def pass2_key(
            col,
            key,
            first_threshold,
            histogram,
            metadata,
            candidate_keys,
            candidate_indices,
            stable_keys,
            stable_indices,
            row_indices,
            row_values,
        ):
            first = high_bucket(key)
            if first > first_threshold:
                out_pos = atomic_add_i32(
                    metadata,
                    one,
                    _RUNNING_ABOVE,
                    "workgroup",
                )
                if out_pos < top_k:
                    if const_expr(stable):
                        stable_indices[out_pos] = col
                        if const_expr(write_values):
                            stable_keys[out_pos] = ordered_value(
                                key
                            ).bitcast(fx.Int32)
                    else:
                        store_key_result(
                            out_pos,
                            col,
                            key,
                            row_indices,
                            row_values,
                        )
            elif first == first_threshold:
                atomic_add_i32(
                    histogram,
                    one,
                    middle_bucket(key),
                    "workgroup",
                )
                candidate_pos = atomic_add_i32(
                    metadata,
                    one,
                    _CANDIDATE_COUNT,
                    "workgroup",
                )
                if candidate_pos < fx.Int32(_COMPACT_CAPACITY):
                    candidate_keys[candidate_pos] = key
                    candidate_indices[candidate_pos] = col

        def pass2_cached_key(
            key,
            first_threshold,
            histogram,
        ):
            first = short_high_bucket(key)
            if first == first_threshold:
                atomic_add_i32(
                    histogram,
                    one,
                    short_middle_bucket(key),
                    "workgroup",
                )

        def pass2_one(
            col,
            value,
            first_threshold,
            histogram,
            metadata,
            candidate_keys,
            candidate_indices,
            stable_keys,
            stable_indices,
            row_indices,
            row_values,
        ):
            if col < row_len:
                pass2_key(
                    col,
                    ordered_key(value),
                    first_threshold,
                    histogram,
                    metadata,
                    candidate_keys,
                    candidate_indices,
                    stable_keys,
                    stable_indices,
                    row_indices,
                    row_values,
                )

        def pass2_vec(
            col,
            values,
            first_threshold,
            histogram,
            metadata,
            candidate_keys,
            candidate_indices,
            stable_keys,
            stable_indices,
            row_indices,
            row_values,
        ):
            for item in range_constexpr(_VEC):
                pass2_one(
                    col + item,
                    values[item],
                    first_threshold,
                    histogram,
                    metadata,
                    candidate_keys,
                    candidate_indices,
                    stable_keys,
                    stable_indices,
                    row_indices,
                    row_values,
                )

        def pass3_one(
            col,
            value,
            first_threshold,
            second_threshold,
        ):
            if col < row_len:
                key = ordered_key(value)
                if (high_bucket(key) == first_threshold) & (
                    middle_bucket(key) == second_threshold
                ):
                    atomic_add_i32(
                        histogram,
                        one,
                        low_bucket(key),
                        "workgroup",
                    )

        def pass3_vec(
            col,
            values,
            first_threshold,
            second_threshold,
        ):
            for item in range_constexpr(_VEC):
                pass3_one(
                    col + item,
                    values[item],
                    first_threshold,
                    second_threshold,
                )

        def pass3_cached_key(
            key,
            first_threshold,
            second_threshold,
            histogram,
        ):
            if (
                short_high_bucket(key) == first_threshold
            ) & (
                short_middle_bucket(key) == second_threshold
            ):
                atomic_add_i32(
                    histogram,
                    one,
                    short_low_bucket(key),
                    "workgroup",
                )

        def compact_pass3(
            candidate_count,
            second_threshold,
            histogram,
            candidate_keys,
        ):
            for pos in range(tid, candidate_count, block_size):
                key = candidate_keys[pos]
                if middle_bucket(key) == second_threshold:
                    atomic_add_i32(
                        histogram,
                        one,
                        low_bucket(key),
                        "workgroup",
                    )

        # Non-stable emitters

        def scatter_cached_unstable(
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            levels,
            row_indices,
            row_values,
            metadata,
        ):
            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()

            def emit(col, key, row_indices, row_values, metadata):
                above, equal = classify_levels(
                    short_high_bucket(key),
                    short_middle_bucket(key),
                    short_low_bucket(key),
                    first_threshold,
                    second_threshold,
                    third_threshold,
                    levels,
                )
                scatter_unstable_key(
                    col,
                    key,
                    above,
                    equal,
                    num_needed,
                    row_indices,
                    row_values,
                    metadata,
                )

            scan_full_keys(
                lambda col, key: emit(
                    col,
                    key,
                    row_indices,
                    row_values,
                    metadata,
                )
            )

        def scatter_full_unstable(
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            row_indices,
            row_values,
            metadata,
        ):
            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()

            def scatter_one(
                col,
                value,
                row_indices,
                row_values,
                metadata,
            ):
                if col < row_len:
                    above, equal = classify(
                        ordered_key(value),
                        first_threshold,
                        second_threshold,
                        third_threshold,
                    )
                    if above:
                        pos = atomic_add_i32(
                            metadata,
                            one,
                            _RUNNING_ABOVE,
                            "workgroup",
                        )
                        if pos < top_k:
                            store_loaded_result(
                                pos,
                                col,
                                value,
                                row_indices,
                                row_values,
                            )
                    elif equal:
                        back_pos = atomic_add_i32(
                            metadata,
                            one,
                            _RUNNING_EQUAL,
                            "workgroup",
                        )
                        if back_pos < num_needed:
                            store_loaded_result(
                                top_k - one - back_pos,
                                col,
                                value,
                                row_indices,
                                row_values,
                            )

            def scatter_vec(col, values):
                for item in range_constexpr(_VEC):
                    scatter_one(
                        col + item,
                        values[item],
                        row_indices,
                        row_values,
                        metadata,
                    )

            scan_row(
                scatter_vec,
                lambda col, value: scatter_one(
                    col,
                    value,
                    row_indices,
                    row_values,
                    metadata,
                ),
            )

        # Stable emitters

        def sort_and_store_stable(
            row_indices,
            row_values,
            candidate_keys,
            candidate_indices,
        ):
            for item in range_constexpr(stable_sort_items_per_thread):
                pos = tid + item * block_threads
                if (pos >= top_k) & (
                    pos < fx.Int32(stable_sort_capacity)
                ):
                    candidate_indices[pos] = fx.Int32(2147483647)
                    if const_expr(write_values):
                        candidate_keys[pos] = zero
            gpu.barrier()

            for stage in range_constexpr(len(stable_sort_sizes)):
                size = stable_sort_sizes[stage]
                stride = stable_sort_strides[stage]
                for item in range_constexpr(
                    stable_sort_items_per_thread
                ):
                    pos = tid + item * block_threads
                    partner = pos ^ fx.Int32(stride)
                    if (
                        pos < fx.Int32(stable_sort_capacity)
                    ) & (partner > pos):
                        left = candidate_indices[pos]
                        right = candidate_indices[partner]
                        ascending = (pos & fx.Int32(size)) == zero
                        swap = ascending.select(
                            left > right, left < right
                        )
                        candidate_indices[pos] = swap.select(
                            right, left
                        )
                        candidate_indices[partner] = swap.select(
                            left, right
                        )
                        if const_expr(write_values):
                            left_value = candidate_keys[pos]
                            right_value = candidate_keys[partner]
                            candidate_keys[pos] = swap.select(
                                right_value, left_value
                            )
                            candidate_keys[partner] = swap.select(
                                left_value, right_value
                            )
                gpu.barrier()

            for item in range_constexpr(stable_sort_items_per_thread):
                pos = tid + item * block_threads
                if pos < top_k:
                    col = candidate_indices[pos]
                    row_indices[pos] = row_start + col
                    if const_expr(write_values):
                        row_values[pos] = candidate_keys[pos].bitcast(
                            fx.Float32
                        )

        def scatter_stable_full_scan(
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            row_indices,
            row_values,
            candidate_keys,
            candidate_indices,
            metadata,
        ):
            definite_expected = top_k - num_needed
            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()

            def collect_one(
                col,
                value,
                candidate_keys,
                candidate_indices,
                metadata,
            ):
                above, equal = classify(
                    ordered_key(value),
                    first_threshold,
                    second_threshold,
                    third_threshold,
                )
                if above:
                    pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_ABOVE,
                        "workgroup",
                    )
                    if pos < definite_expected:
                        candidate_indices[pos] = col
                        if const_expr(write_values):
                            candidate_keys[pos] = value.bitcast(fx.Int32)
                elif equal:
                    tie_pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_EQUAL,
                        "workgroup",
                    )
                    if tie_pos < num_needed:
                        pos = definite_expected + tie_pos
                        candidate_indices[pos] = col
                        if const_expr(write_values):
                            candidate_keys[pos] = value.bitcast(fx.Int32)

            def collect_vec(
                col,
                values,
                candidate_keys,
                candidate_indices,
                metadata,
            ):
                for item in range_constexpr(_VEC):
                    collect_one(
                        col + item,
                        values[item],
                        candidate_keys,
                        candidate_indices,
                        metadata,
                    )

            scan_row(
                lambda col, values: collect_vec(
                    col,
                    values,
                    candidate_keys,
                    candidate_indices,
                    metadata,
                ),
                lambda col, value: collect_one(
                    col,
                    value,
                    candidate_keys,
                    candidate_indices,
                    metadata,
                ),
            )
            gpu.barrier()
            sort_and_store_stable(
                row_indices,
                row_values,
                candidate_keys,
                candidate_indices,
            )

        def scatter_stable_candidates(
            candidate_count,
            first_threshold,
            second_threshold,
            third_threshold,
            levels,
            row_indices,
            row_values,
            candidate_keys,
            candidate_indices,
            stable_keys,
            stable_indices,
            metadata,
        ):
            for candidate_pos in range(
                tid, candidate_count, block_size
            ):
                key = candidate_keys[candidate_pos]
                above, equal = classify_levels(
                    high_bucket(key),
                    middle_bucket(key),
                    low_bucket(key),
                    first_threshold,
                    second_threshold,
                    third_threshold,
                    levels,
                )
                if above | equal:
                    out_pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_ABOVE,
                        "workgroup",
                    )
                    if out_pos < top_k:
                        stable_indices[out_pos] = (
                            candidate_indices[candidate_pos]
                        )
                        if const_expr(write_values):
                            stable_keys[out_pos] = ordered_value(
                                key
                            ).bitcast(fx.Int32)
            gpu.barrier()

            for item in range_constexpr(stable_sort_items_per_thread):
                pos = tid + item * block_threads
                if pos < top_k:
                    candidate_indices[pos] = stable_indices[pos]
                    if const_expr(write_values):
                        candidate_keys[pos] = stable_keys[pos]
            gpu.barrier()
            sort_and_store_stable(
                row_indices,
                row_values,
                candidate_keys,
                candidate_indices,
            )

        def scatter_stable_cached(
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            levels,
            row_indices,
            row_values,
            scan,
            metadata,
        ):
            above_base = zero
            equal_base = zero
            for step in range_constexpr(full_key_vector_steps):
                vector_idx = step * block_threads + tid
                active_vector = (
                    vector_idx < (row_len + fx.Int32(_VEC - 1)) // vec_width
                )
                safe_vector_idx = active_vector.select(
                    vector_idx, zero
                )
                fragment = fx.make_rmem_tensor(
                    full_key_fragment_layout, fx.Int32
                )
                fx.copy_atom_call(
                    full_key_load_atom,
                    fx.slice(
                        full_key_tiles,
                        (None, safe_vector_idx),
                    ),
                    fragment,
                )
                keys = fragment.load()
                classes = fx.make_rmem_tensor(_VEC, fx.Int32)
                local_above = zero
                local_equal = zero
                for item in range_constexpr(_VEC):
                    col = vector_idx * vec_width + item
                    above, equal = classify_levels(
                        short_high_bucket(keys[item]),
                        short_middle_bucket(keys[item]),
                        short_low_bucket(keys[item]),
                        first_threshold,
                        second_threshold,
                        third_threshold,
                        levels,
                    )
                    active = active_vector & (col < row_len)
                    above_i32 = (active & above).select(one, zero)
                    equal_i32 = (active & equal).select(one, zero)
                    classes[item] = above_i32 * above_class + equal_i32
                    local_above = local_above + above_i32
                    local_equal = local_equal + equal_i32

                (
                    above_prefix,
                    equal_prefix,
                    block_above,
                    block_equal,
                ) = block_exclusive_scan_pair(
                    local_above,
                    local_equal,
                    scan,
                    metadata,
                )
                my_above = above_base + above_prefix
                my_equal = equal_base + equal_prefix
                for item in range_constexpr(_VEC):
                    col = vector_idx * vec_width + item
                    accepted_equal = (my_equal < num_needed).select(
                        my_equal, num_needed
                    )
                    out_pos = my_above + accepted_equal
                    if classes[item] == above_class:
                        store_key_result(
                            out_pos,
                            col,
                            keys[item],
                            row_indices,
                            row_values,
                        )
                        my_above = my_above + one
                    elif classes[item] == one:
                        if my_equal < num_needed:
                            store_key_result(
                                out_pos,
                                col,
                                keys[item],
                                row_indices,
                                row_values,
                            )
                        my_equal = my_equal + one
                above_base = above_base + block_above
                next_equal_base = equal_base + block_equal
                equal_base = (
                    next_equal_base < num_needed
                ).select(next_equal_base, num_needed)
                gpu.barrier()

        def scatter_stable_ordered(
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

            if tid == 0:
                running_above = metadata[_RUNNING_ABOVE]
                running_equal = metadata[_RUNNING_EQUAL]
                for item in range_constexpr(_VEC - 1):
                    if item < skip_cnt:
                        value = input_row[item]
                        above, equal = classify(
                            ordered_key(value),
                            first_threshold,
                            second_threshold,
                            third_threshold,
                        )
                        accepted_equal = (
                            running_equal < num_needed
                        ).select(running_equal, num_needed)
                        out_pos = running_above + accepted_equal
                        if above:
                            store_loaded_result(
                                out_pos,
                                fx.Int32(item),
                                value,
                                row_indices,
                                row_values,
                            )
                            running_above = running_above + one
                        elif equal:
                            if running_equal < num_needed:
                                store_loaded_result(
                                    out_pos,
                                    fx.Int32(item),
                                    value,
                                    row_indices,
                                    row_values,
                                )
                            running_equal = running_equal + one
                metadata[_RUNNING_ABOVE] = running_above
                metadata[_RUNNING_EQUAL] = running_equal
            gpu.barrier()
            above_base = metadata[_RUNNING_ABOVE]
            initial_equal = metadata[_RUNNING_EQUAL]
            equal_base = (initial_equal < num_needed).select(
                initial_equal, num_needed
            )

            num_steps = (
                len_cast + block_size - one
            ) // block_size
            for step in range(zero, num_steps, one):
                vector_idx = step * block_size + tid
                active_vector = vector_idx < len_cast
                safe_vector_idx = active_vector.select(vector_idx, zero)
                col_base = skip_cnt + safe_vector_idx * vec_width
                values = _load_f32x4(
                    input_resource, safe_vector_idx
                )
                classes = fx.make_rmem_tensor(_VEC, fx.Int32)
                local_above = zero
                local_equal = zero
                for item in range_constexpr(_VEC):
                    col = col_base + item
                    above, equal = classify(
                        ordered_key(values[item]),
                        first_threshold,
                        second_threshold,
                        third_threshold,
                    )
                    active = active_vector & (col < row_len)
                    above_i32 = (active & above).select(one, zero)
                    equal_i32 = (active & equal).select(one, zero)
                    classes[item] = above_i32 * above_class + equal_i32
                    local_above = local_above + above_i32
                    local_equal = local_equal + equal_i32

                (
                    above_prefix,
                    equal_prefix,
                    block_above,
                    block_equal,
                ) = block_exclusive_scan_pair(
                    local_above,
                    local_equal,
                    scan,
                    metadata,
                )
                my_above = above_base + above_prefix
                my_equal = equal_base + equal_prefix
                for item in range_constexpr(_VEC):
                    col = col_base + item
                    accepted_equal = (my_equal < num_needed).select(
                        my_equal, num_needed
                    )
                    out_pos = my_above + accepted_equal
                    if classes[item] == above_class:
                        store_loaded_result(
                            out_pos,
                            col,
                            values[item],
                            row_indices,
                            row_values,
                        )
                        my_above = my_above + one
                    elif classes[item] == one:
                        if my_equal < num_needed:
                            store_loaded_result(
                                out_pos,
                                col,
                                values[item],
                                row_indices,
                                row_values,
                            )
                        my_equal = my_equal + one
                above_base = above_base + block_above
                next_equal_base = equal_base + block_equal
                equal_base = (
                    next_equal_base < num_needed
                ).select(next_equal_base, num_needed)
                gpu.barrier()

            if tid == 0:
                running_above = above_base
                running_equal = equal_base
                remain_base = skip_cnt + len_cast * vec_width
                for item in range_constexpr(_VEC - 1):
                    col = remain_base + item
                    if col < row_len:
                        value = input_row[col]
                        above, equal = classify(
                            ordered_key(value),
                            first_threshold,
                            second_threshold,
                            third_threshold,
                        )
                        accepted_equal = (
                            running_equal < num_needed
                        ).select(running_equal, num_needed)
                        out_pos = running_above + accepted_equal
                        if above:
                            store_loaded_result(
                                out_pos,
                                col,
                                value,
                                row_indices,
                                row_values,
                            )
                            running_above = running_above + one
                        elif equal:
                            if running_equal < num_needed:
                                store_loaded_result(
                                    out_pos,
                                    col,
                                    value,
                                    row_indices,
                                    row_values,
                                )
                            running_equal = running_equal + one
            gpu.barrier()

        # Compact candidate emitters

        def scatter_candidates_after_first(
            candidate_count,
            row_indices,
            row_values,
            candidate_keys,
            candidate_indices,
        ):
            for pos in range(tid, candidate_count, block_size):
                store_key_result(
                    top_k - one - pos,
                    candidate_indices[pos],
                    candidate_keys[pos],
                    row_indices,
                    row_values,
                )

        def scatter_candidates_after_second(
            candidate_count,
            second_threshold,
            num_needed,
            row_indices,
            row_values,
            metadata,
            candidate_keys,
            candidate_indices,
        ):
            if tid == 0:
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()

            for pos in range(tid, candidate_count, block_size):
                col = candidate_indices[pos]
                key = candidate_keys[pos]
                second = middle_bucket(key)
                scatter_unstable_key(
                    col,
                    key,
                    second > second_threshold,
                    second == second_threshold,
                    num_needed,
                    row_indices,
                    row_values,
                    metadata,
                )

        def scatter_candidates_after_third(
            candidate_count,
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            row_indices,
            row_values,
            metadata,
            candidate_keys,
            candidate_indices,
        ):
            if tid == 0:
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()

            for pos in range(tid, candidate_count, block_size):
                col = candidate_indices[pos]
                key = candidate_keys[pos]
                above, equal = classify(
                    key,
                    first_threshold,
                    second_threshold,
                    third_threshold,
                )
                scatter_unstable_key(
                    col,
                    key,
                    above,
                    equal,
                    num_needed,
                    row_indices,
                    row_values,
                    metadata,
                )

        # Path finalization

        def scatter_cached_selection(
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            levels,
            row_indices,
            row_values,
            scan,
            metadata,
        ):
            if const_expr(stable):
                if const_expr(levels == 3):
                    if metadata[_SELECTED_BUCKET_COUNT] == row_len:
                        scatter_stable_equal_row(
                            short_threshold_key(
                                first_threshold,
                                second_threshold,
                                third_threshold,
                            ),
                            row_indices,
                            row_values,
                        )
                    else:
                        scatter_stable_cached(
                            first_threshold,
                            second_threshold,
                            third_threshold,
                            num_needed,
                            levels,
                            row_indices,
                            row_values,
                            scan,
                            metadata,
                        )
                else:
                    scatter_stable_cached(
                        first_threshold,
                        second_threshold,
                        third_threshold,
                        num_needed,
                        levels,
                        row_indices,
                        row_values,
                        scan,
                        metadata,
                    )
            else:
                scatter_cached_unstable(
                    first_threshold,
                    second_threshold,
                    third_threshold,
                    num_needed,
                    levels,
                    row_indices,
                    row_values,
                    metadata,
                )

        def finish_cached_selection(
            first_threshold,
            need_after_first,
            histogram,
            scan,
            metadata,
            row_indices,
            row_values,
        ):
            if metadata[_SELECTED_BUCKET_COUNT] == need_after_first:
                scatter_cached_selection(
                    first_threshold,
                    zero,
                    zero,
                    need_after_first,
                    1,
                    row_indices,
                    row_values,
                    scan,
                    metadata,
                )
            else:
                choose_threshold(
                    need_after_first,
                    _SECOND_ABOVE,
                    _SECOND_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    histogram,
                    scan,
                    metadata,
                    short_bins_per_thread,
                )
                second_threshold = metadata[_SECOND_THRESHOLD]
                need_after_second = (
                    need_after_first - metadata[_SECOND_ABOVE]
                )
                if (
                    metadata[_SELECTED_BUCKET_COUNT]
                    == need_after_second
                ):
                    scatter_cached_selection(
                        first_threshold,
                        second_threshold,
                        zero,
                        need_after_second,
                        2,
                        row_indices,
                        row_values,
                        scan,
                        metadata,
                    )
                else:
                    clear_histogram(histogram, short_bins_per_thread)
                    scan_full_keys(
                        lambda _, key: pass3_cached_key(
                            key,
                            first_threshold,
                            second_threshold,
                            histogram,
                        )
                    )
                    gpu.barrier()
                    choose_threshold(
                        need_after_second,
                        _THIRD_ABOVE,
                        _THIRD_THRESHOLD,
                        _SELECTED_BUCKET_COUNT,
                        histogram,
                        scan,
                        metadata,
                        short_bins_per_thread,
                    )
                    scatter_cached_selection(
                        first_threshold,
                        second_threshold,
                        metadata[_THIRD_THRESHOLD],
                        need_after_second - metadata[_THIRD_ABOVE],
                        3,
                        row_indices,
                        row_values,
                        scan,
                        metadata,
                    )

        def finish_candidate_unstable(
            first_threshold,
            need_after_first,
            candidate_count,
            histogram,
            scan,
            metadata,
            candidate_keys,
            candidate_indices,
            row_indices,
            row_values,
        ):
            can_finish_after_first = (
                candidate_count <= fx.Int32(_COMPACT_CAPACITY)
            ) & (
                metadata[_SELECTED_BUCKET_COUNT] == need_after_first
            )
            if can_finish_after_first:
                scatter_candidates_after_first(
                    candidate_count,
                    row_indices,
                    row_values,
                    candidate_keys,
                    candidate_indices,
                )
            else:
                choose_threshold(
                    need_after_first,
                    _SECOND_ABOVE,
                    _SECOND_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    histogram,
                    scan,
                    metadata,
                    later_bins_per_thread,
                )
                second_threshold = metadata[_SECOND_THRESHOLD]
                need_after_second = (
                    need_after_first - metadata[_SECOND_ABOVE]
                )
                can_finish_after_second = (
                    candidate_count <= fx.Int32(_COMPACT_CAPACITY)
                ) & (
                    metadata[_SELECTED_BUCKET_COUNT]
                    == need_after_second
                )
                if can_finish_after_second:
                    scatter_candidates_after_second(
                        candidate_count,
                        second_threshold,
                        need_after_second,
                        row_indices,
                        row_values,
                        metadata,
                        candidate_keys,
                        candidate_indices,
                    )
                else:
                    clear_histogram(histogram, later_bins_per_thread)
                    if candidate_count <= fx.Int32(
                        _COMPACT_CAPACITY
                    ):
                        compact_pass3(
                            candidate_count,
                            second_threshold,
                            histogram,
                            candidate_keys,
                        )
                    else:
                        scan_row(
                            lambda col, values: pass3_vec(
                                col,
                                values,
                                first_threshold,
                                second_threshold,
                            ),
                            lambda col, value: pass3_one(
                                col,
                                value,
                                first_threshold,
                                second_threshold,
                            ),
                        )
                    gpu.barrier()
                    choose_threshold(
                        need_after_second,
                        _THIRD_ABOVE,
                        _THIRD_THRESHOLD,
                        _SELECTED_BUCKET_COUNT,
                        histogram,
                        scan,
                        metadata,
                        later_bins_per_thread,
                    )
                    third_threshold = metadata[_THIRD_THRESHOLD]
                    num_needed = (
                        need_after_second - metadata[_THIRD_ABOVE]
                    )
                    if candidate_count <= fx.Int32(
                        _COMPACT_CAPACITY
                    ):
                        scatter_candidates_after_third(
                            candidate_count,
                            first_threshold,
                            second_threshold,
                            third_threshold,
                            num_needed,
                            row_indices,
                            row_values,
                            metadata,
                            candidate_keys,
                            candidate_indices,
                        )
                    else:
                        scatter_full_unstable(
                            first_threshold,
                            second_threshold,
                            third_threshold,
                            num_needed,
                            row_indices,
                            row_values,
                            metadata,
                        )

        def finish_candidate_stable(
            first_threshold,
            need_after_first,
            candidate_count,
            histogram,
            scan,
            metadata,
            candidate_keys,
            candidate_indices,
            stable_keys,
            stable_indices,
            row_indices,
            row_values,
        ):
            choose_threshold(
                need_after_first,
                _SECOND_ABOVE,
                _SECOND_THRESHOLD,
                _SELECTED_BUCKET_COUNT,
                histogram,
                scan,
                metadata,
                later_bins_per_thread,
            )
            second_threshold = metadata[_SECOND_THRESHOLD]
            need_after_second = (
                need_after_first - metadata[_SECOND_ABOVE]
            )
            clear_histogram(histogram, later_bins_per_thread)
            if candidate_count <= fx.Int32(_COMPACT_CAPACITY):
                compact_pass3(
                    candidate_count,
                    second_threshold,
                    histogram,
                    candidate_keys,
                )
            else:
                scan_row(
                    lambda col, values: pass3_vec(
                        col,
                        values,
                        first_threshold,
                        second_threshold,
                    ),
                    lambda col, value: pass3_one(
                        col,
                        value,
                        first_threshold,
                        second_threshold,
                    ),
                )
            gpu.barrier()
            choose_threshold(
                need_after_second,
                _THIRD_ABOVE,
                _THIRD_THRESHOLD,
                _SELECTED_BUCKET_COUNT,
                histogram,
                scan,
                metadata,
                later_bins_per_thread,
            )
            third_threshold = metadata[_THIRD_THRESHOLD]
            num_needed = (
                need_after_second - metadata[_THIRD_ABOVE]
            )
            if metadata[_SELECTED_BUCKET_COUNT] == row_len:
                scatter_stable_equal_row(
                    threshold_key(
                        first_threshold,
                        second_threshold,
                        third_threshold,
                    ),
                    row_indices,
                    row_values,
                )
            else:
                if const_expr(stable_sort_enabled):
                    can_use_fast = (
                        row_len >= fx.Int32(_STABLE_FAST_MIN_ROW_LEN)
                    ) & (
                        metadata[_SELECTED_BUCKET_COUNT] == num_needed
                    )
                    can_use_compact_fast = can_use_fast & (
                        candidate_count
                        <= fx.Int32(_COMPACT_CAPACITY)
                    )
                    if can_use_compact_fast:
                        scatter_stable_candidates(
                            candidate_count,
                            first_threshold,
                            second_threshold,
                            third_threshold,
                            3,
                            row_indices,
                            row_values,
                            candidate_keys,
                            candidate_indices,
                            stable_keys,
                            stable_indices,
                            metadata,
                        )
                    else:
                        if can_use_fast:
                            scatter_stable_full_scan(
                                first_threshold,
                                second_threshold,
                                third_threshold,
                                num_needed,
                                row_indices,
                                row_values,
                                candidate_keys,
                                candidate_indices,
                                metadata,
                            )
                        else:
                            scatter_stable_ordered(
                                first_threshold,
                                second_threshold,
                                third_threshold,
                                num_needed,
                                row_indices,
                                row_values,
                                scan,
                                metadata,
                            )
                else:
                    scatter_stable_ordered(
                        first_threshold,
                        second_threshold,
                        third_threshold,
                        num_needed,
                        row_indices,
                        row_values,
                        scan,
                        metadata,
                    )

        # Kernel control flow

        if row_len <= top_k:
            for step in range_constexpr(output_vector_steps):
                vector_idx = step * block_threads + tid
                if vector_idx < output_vector_count:
                    col = vector_idx * vec_width
                    index_values = [
                        (col + item < row_len).select(
                            row_start + col + item,
                            fx.Int32(-1),
                        )
                        for item in range_constexpr(_VEC)
                    ]
                    fragment = fx.make_rmem_tensor(
                        index_fragment_layout, fx.Int32
                    )
                    fragment.store(
                        fx.Vector.from_elements(
                            index_values, dtype=fx.Int32
                        )
                    )
                    fx.copy_atom_call(
                        index_store_atom,
                        fragment,
                        fx.slice(
                            row_index_tiles,
                            (None, vector_idx),
                        ),
                    )
                    if const_expr(write_values):
                        output_values = []
                        for item in range_constexpr(_VEC):
                            local_col = col + item
                            valid = local_col < row_len
                            safe_col = valid.select(local_col, zero)
                            output_values.append(
                                valid.select(
                                    input_row[safe_col],
                                    fx.Float32(float("-inf")),
                                )
                            )
                        value_fragment = fx.make_rmem_tensor(
                            value_fragment_layout, fx.Float32
                        )
                        value_fragment.store(
                            fx.Vector.from_elements(
                                output_values, dtype=fx.Float32
                            )
                        )
                        fx.copy_atom_call(
                            value_store_atom,
                            value_fragment,
                            fx.slice(
                                row_value_tiles,
                                (None, vector_idx),
                            ),
                        )
            tail = output_vector_count * _VEC + tid
            if tail < k:
                valid = tail < row_len
                safe_tail = valid.select(tail, zero)
                row_indices[tail] = valid.select(
                    row_start + tail,
                    fx.Int32(-1),
                )
                if const_expr(write_values):
                    row_values[tail] = valid.select(
                        input_row[safe_tail],
                        fx.Float32(float("-inf")),
                    )

        if row_len > top_k:
            if tid < _METADATA_SIZE:
                metadata[tid] = zero
            gpu.barrier()

            cache_full_row = row_len <= fx.Int32(_COMPACT_CAPACITY)
            if cache_full_row:
                clear_pass1_histograms(
                    short_pass1_histogram0,
                    short_pass1_histogram1,
                    short_bins_per_thread,
                )
                scan_row(
                    lambda col, values: pass1_cache_vec(
                        col,
                        values,
                        short_pass1_histogram0,
                        short_pass1_histogram1,
                        full_keys,
                    ),
                    lambda col, value: pass1_cache_one(
                        col,
                        value,
                        short_pass1_histogram0,
                        short_pass1_histogram1,
                        full_keys,
                    ),
                )
                gpu.barrier()
                merge_pass1_histograms(
                    short_pass1_histogram0,
                    short_pass1_histogram1,
                    short_bins_per_thread,
                )
                choose_threshold(
                    top_k,
                    _FIRST_ABOVE,
                    _FIRST_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    short_pass1_histogram0,
                    scan,
                    metadata,
                    short_bins_per_thread,
                )
            else:
                clear_pass1_histograms(
                    pass1_histogram0,
                    pass1_histogram1,
                    high_bins_per_thread,
                )
                scan_row(
                    lambda col, values: pass1_vec(
                        col,
                        values,
                        pass1_histogram0,
                        pass1_histogram1,
                    ),
                    lambda col, value: pass1_one(
                        col,
                        value,
                        pass1_histogram0,
                        pass1_histogram1,
                    ),
                )
                gpu.barrier()
                merge_pass1_histograms(
                    pass1_histogram0,
                    pass1_histogram1,
                    high_bins_per_thread,
                )
                choose_threshold(
                    top_k,
                    _FIRST_ABOVE,
                    _FIRST_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    pass1_histogram0,
                    scan,
                    metadata,
                    high_bins_per_thread,
                )
            first_threshold = metadata[_FIRST_THRESHOLD]

            if cache_full_row:
                clear_histogram(
                    short_histogram, short_bins_per_thread
                )
            else:
                clear_histogram(histogram, later_bins_per_thread)
            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
                metadata[_CANDIDATE_COUNT] = zero
            gpu.barrier()
            if cache_full_row:
                scan_full_keys(
                    lambda _, key: pass2_cached_key(
                        key,
                        first_threshold,
                        short_histogram,
                    )
                )
            else:
                scan_row_reverse(
                    lambda col, values: pass2_vec(
                        col,
                        values,
                        first_threshold,
                        histogram,
                        metadata,
                        candidate_keys,
                        candidate_indices,
                        stable_keys,
                        stable_indices,
                        row_indices,
                        row_values,
                    ),
                    lambda col, value: pass2_one(
                        col,
                        value,
                        first_threshold,
                        histogram,
                        metadata,
                        candidate_keys,
                        candidate_indices,
                        stable_keys,
                        stable_indices,
                        row_indices,
                        row_values,
                    ),
                )
            gpu.barrier()
            need_after_first = top_k - metadata[_FIRST_ABOVE]
            if cache_full_row:
                finish_cached_selection(
                    first_threshold,
                    need_after_first,
                    short_histogram,
                    scan,
                    metadata,
                    row_indices,
                    row_values,
                )
            else:
                if const_expr(stable):
                    finish_candidate_stable(
                        first_threshold,
                        need_after_first,
                        metadata[_CANDIDATE_COUNT],
                        histogram,
                        scan,
                        metadata,
                        candidate_keys,
                        candidate_indices,
                        stable_keys,
                        stable_indices,
                        row_indices,
                        row_values,
                    )
                else:
                    finish_candidate_unstable(
                        first_threshold,
                        need_after_first,
                        metadata[_CANDIDATE_COUNT],
                        histogram,
                        scan,
                        metadata,
                        candidate_keys,
                        candidate_indices,
                        row_indices,
                        row_values,
                    )

    @flyc.jit
    def launch_topk_per_row_prefill_one_workgroup(
        input: fx.Tensor,
        row_starts: fx.Tensor,
        row_ends: fx.Tensor,
        indices: fx.Tensor,
        values: fx.Tensor,
        rows_m: fx.Int32,
        stream: fx.Stream,
    ):
        topk_per_row_prefill_one_workgroup_kernel(
            input,
            row_starts,
            row_ends,
            indices,
            values,
        ).launch(
            grid=(rows_m, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_topk_per_row_prefill_one_workgroup
