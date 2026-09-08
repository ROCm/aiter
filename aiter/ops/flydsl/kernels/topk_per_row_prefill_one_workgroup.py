# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 prefill TopK with one workgroup per input row.

Short rows cache ordered keys in LDS. Long rows compact the selected radix
bucket, while stable modes add deterministic index ordering and tie-breaking.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from .kernels_common import atomic_add_i32
from .topk_per_row_decode import _load_f32x4, _warp_inclusive_prefix_i32

_WAVE_SIZE = 32
_VEC = 4
_LOAD_UNROLL = 4
_PASS1_HISTOGRAM_STAGES = 2
_KEY_BITS = 32
_LONG_RADIX_BITS = (12, 10, 10)
_SHORT_RADIX_BITS = (11, 10, 11)
_LONG_RADIX_SHIFTS = (_LONG_RADIX_BITS[1] + _LONG_RADIX_BITS[2], _LONG_RADIX_BITS[2], 0)
_SHORT_RADIX_SHIFTS = (_SHORT_RADIX_BITS[1] + _SHORT_RADIX_BITS[2], _SHORT_RADIX_BITS[2], 0)
_LONG_RADIX_MASKS = tuple((1 << bits) - 1 for bits in _LONG_RADIX_BITS)
_SHORT_RADIX_MASKS = tuple((1 << bits) - 1 for bits in _SHORT_RADIX_BITS)
_HIGH_BUCKETS = 1 << _LONG_RADIX_BITS[0]
_LATER_BUCKETS = 1 << max(_LONG_RADIX_BITS[1:])
_SHORT_HIGH_BUCKETS = 1 << _SHORT_RADIX_BITS[0]
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
):
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
    full_key_vector_steps = ((_COMPACT_CAPACITY // _VEC) + block_threads - 1) // block_threads
    stable_sort_enabled = stable and block_threads == 1024 and k <= 2048
    stable_sort_capacity = 1 << (k - 1).bit_length() if stable_sort_enabled else 1
    stable_stage_capacity = k if stable_sort_enabled else 1
    stable_sort_items_per_thread = (stable_sort_capacity + block_threads - 1) // block_threads
    stable_sort_sizes, stable_sort_strides = _build_bitonic_schedule(stable_sort_capacity)
    output_vector_count = k // _VEC
    output_vector_steps = (output_vector_count + block_threads - 1) // block_threads
    output_vector_elems = max(_VEC, output_vector_count * _VEC)

    # LDS layouts

    @fx.struct
    class LongPass1Storage:
        histograms: fx.Array[fx.Int32, _PASS1_HISTOGRAM_STAGES * _HIGH_BUCKETS, 16]

    @fx.struct
    class LongLaterStorage:
        histogram: fx.Array[fx.Int32, _LATER_BUCKETS, 16]
        candidate_keys: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]
        candidate_indices: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]
        stable_keys: fx.Array[fx.Int32, stable_stage_capacity, 16]
        stable_indices: fx.Array[fx.Int32, stable_stage_capacity, 16]

    @fx.struct
    class ShortPass1Storage:
        histograms: fx.Array[fx.Int32, _PASS1_HISTOGRAM_STAGES * _SHORT_HIGH_BUCKETS, 16]
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
        vec_width = fx.Int32(_VEC)
        block_size = fx.Int32(block_threads)
        top_k = fx.Int32(k)
        sign_bit = fx.Int32(-2147483648)

        # Row bounds
        row_start = row_starts[row]
        row_len = row_ends[row] - row_start
        full_vector_count = row_len // vec_width

        # Input and output views
        physical_row = fx.slice(input, (row, None))
        input_row_iter = fx.add_offset(fx.get_iter(physical_row), row_start)
        input_row = fx.rocdl.make_buffer_tensor(
            fx.make_view(input_row_iter, fx.make_layout(_MAX_ROW_ELEMENTS, 1)),
            num_records_bytes=fx.Int64(row_len) * fx.Int64(4),
        )
        input_vector_tiles = fx.logical_divide(input_row, fx.make_layout(_VEC, 1))

        row_indices = fx.slice(indices, (row, None))
        row_values = fx.slice(value_output, (row, None))
        row_index_tiles = fx.logical_divide(
            fx.make_view(fx.get_iter(row_indices), fx.make_layout(output_vector_elems, 1)),
            fx.make_layout(_VEC, 1),
        )
        row_value_tiles = fx.logical_divide(
            fx.make_view(fx.get_iter(row_values), fx.make_layout(output_vector_elems, 1)),
            fx.make_layout(_VEC, 1),
        )

        # Copy primitives
        index_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        index_fragment_layout = fx.make_layout(_VEC, 1)
        value_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        value_fragment_layout = fx.make_layout(_VEC, 1)
        full_key_load_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        full_key_fragment_layout = fx.make_layout(_VEC, 1)

        # LDS views
        storage = fx.SharedAllocator().allocate(SharedStorage)

        # Long-row arena
        long_histogram_matrix = storage.arena.long_pass1.histograms.peek().view(
            fx.make_layout((_PASS1_HISTOGRAM_STAGES, _HIGH_BUCKETS), (_HIGH_BUCKETS, 1))
        )
        long_histograms = (
            fx.slice(long_histogram_matrix, (0, None)),
            fx.slice(long_histogram_matrix, (1, None)),
        )
        histogram = storage.arena.long_later.histogram.peek().view(
            fx.make_layout(_LATER_BUCKETS, 1)
        )
        candidate_keys = storage.arena.long_later.candidate_keys.peek().view(
            fx.make_layout(_COMPACT_CAPACITY, 1)
        )
        candidate_indices = storage.arena.long_later.candidate_indices.peek().view(
            fx.make_layout(_COMPACT_CAPACITY, 1)
        )
        stable_keys = storage.arena.long_later.stable_keys.peek().view(
            fx.make_layout(stable_stage_capacity, 1)
        )
        stable_indices = storage.arena.long_later.stable_indices.peek().view(
            fx.make_layout(stable_stage_capacity, 1)
        )

        # Short-row arena
        short_histogram_matrix = storage.arena.short_pass1.histograms.peek().view(
            fx.make_layout((_PASS1_HISTOGRAM_STAGES, _SHORT_HIGH_BUCKETS), (_SHORT_HIGH_BUCKETS, 1))
        )
        short_histograms = (
            fx.slice(short_histogram_matrix, (0, None)),
            fx.slice(short_histogram_matrix, (1, None)),
        )
        full_keys = storage.arena.short_pass1.full_keys.peek().view(
            fx.make_layout(_COMPACT_CAPACITY, 1)
        )
        full_key_tiles = fx.logical_divide(full_keys, fx.make_layout(_VEC, 1))

        # Shared scratch
        scan = storage.scan.peek().view(fx.make_layout(num_waves * 2, 1))
        metadata = storage.metadata.peek().view(fx.make_layout(_METADATA_SIZE, 1))

        # Key encoding and classification
        def ordered_key(value):
            bits = value.bitcast(fx.Int32)
            return bits ^ ((bits >> fx.Int32(31)) & fx.Int32(0x7FFFFFFF)) ^ sign_bit

        def radix_bucket(key, shift, mask):
            return (key >> fx.Int32(shift)) & fx.Int32(mask)

        def classify_prefix(key, radix_shifts, prefix_threshold, levels):
            prefix_shift = radix_shifts[levels - 1]
            prefix_mask = -1 if prefix_shift == 0 else (1 << (_KEY_BITS - prefix_shift)) - 1
            prefix = radix_bucket(key, prefix_shift, prefix_mask)
            comparable_prefix = prefix ^ sign_bit if levels == 3 else prefix
            comparable_threshold = prefix_threshold ^ sign_bit if levels == 3 else prefix_threshold
            return (comparable_prefix > comparable_threshold, prefix == prefix_threshold)

        def ordered_value(key):
            bits = (key < zero).select(key ^ sign_bit, key ^ fx.Int32(-1))
            return bits.bitcast(fx.Float32)

        def scatter_unstable_key(
            col, key, above, equal, num_needed, row_indices, row_values, metadata
        ):
            if above:
                out_pos = atomic_add_i32(metadata, one, _RUNNING_ABOVE, "workgroup")
                if out_pos < top_k:
                    row_indices[out_pos] = row_start + col
                    if const_expr(write_values):
                        row_values[out_pos] = ordered_value(key)
            elif equal:
                back_pos = atomic_add_i32(metadata, one, _RUNNING_EQUAL, "workgroup")
                if back_pos < num_needed:
                    out_pos = top_k - one - back_pos
                    row_indices[out_pos] = row_start + col
                    if const_expr(write_values):
                        row_values[out_pos] = ordered_value(key)

        def scatter_equal_row(key, row_indices, row_values):
            for pos in range(tid, top_k, block_size):
                row_indices[pos] = row_start + pos
                if const_expr(write_values):
                    row_values[pos] = ordered_value(key)

        def reset_scatter_counters(metadata, reset_above=True):
            if tid == 0:
                if const_expr(reset_above):
                    metadata[_RUNNING_ABOVE] = zero
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()

        # Histogram and workgroup scan primitives
        def clear_histograms(histograms, bins_per_thread):
            for item in range_constexpr(bins_per_thread):
                pos = tid + item * block_threads
                for stage in range_constexpr(len(histograms)):
                    histograms[stage][pos] = zero
            gpu.barrier()

        def merge_histograms(histograms, bins_per_thread):
            for item in range_constexpr(bins_per_thread):
                pos = tid + item * block_threads
                merged = histograms[0][pos]
                for stage in range_constexpr(1, len(histograms)):
                    merged = merged + histograms[stage][pos]
                histograms[0][pos] = merged
            gpu.barrier()

        def block_excl_prefix_i32(packed_local, scan):
            packed_inclusive = _warp_inclusive_prefix_i32(packed_local, lane, _WAVE_SIZE)
            packed_exclusive = packed_inclusive - packed_local
            if lane == _WAVE_SIZE - 1:
                scan[wave] = packed_inclusive
            gpu.barrier()

            if wave == 0:
                wave_val = zero
                if lane < num_waves:
                    wave_val = scan[lane]
                wave_inclusive = _warp_inclusive_prefix_i32(wave_val, lane, _WAVE_SIZE)
                wave_exclusive = wave_inclusive - wave_val
                if lane < num_waves:
                    scan[lane] = wave_exclusive
                if lane == num_waves - 1:
                    scan[num_waves] = wave_inclusive
            gpu.barrier()

            packed_prefix = scan[wave] + packed_exclusive
            packed_total = scan[num_waves]
            gpu.barrier()
            return packed_prefix, packed_total

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
            counts = fx.make_rmem_tensor(bins_per_thread, fx.Int32)
            local_total = zero
            for item in range_constexpr(bins_per_thread):
                count = histogram[first_bin + item]
                counts[item] = count
                local_total = local_total + count
            wave_inclusive = _warp_inclusive_prefix_i32(local_total, lane, _WAVE_SIZE)
            wave_exclusive = wave_inclusive - local_total

            if lane == _WAVE_SIZE - 1:
                scan[wave] = wave_inclusive
            gpu.barrier()

            if wave == 0:
                active = lane < num_waves
                safe_lane = active.select(lane, zero)
                wave_total = active.select(scan[safe_lane], zero)
                wave_prefix = _warp_inclusive_prefix_i32(wave_total, lane, _WAVE_SIZE) - wave_total
                if active:
                    scan[lane + num_waves] = wave_prefix
            gpu.barrier()

            wave_offset = scan[wave + num_waves]
            total = scan[num_waves - 1] + scan[num_waves * 2 - 1]
            target_prefix = total - target_k
            exclusive = wave_offset + wave_exclusive
            for item in range_constexpr(bins_per_thread):
                inclusive = exclusive + counts[item]
                if (exclusive <= target_prefix) & (inclusive > target_prefix):
                    metadata[threshold_slot] = first_bin + item  # Kth bucket
                    metadata[above_slot] = total - inclusive  # Higher-bucket count
                    metadata[count_slot] = inclusive - exclusive  # Bucket count
                exclusive = inclusive
            gpu.barrier()

        # Global and LDS row iterators
        def scan_gm_row(visit_one, reverse=False):
            def visit_vector(col, values):
                for item in range_constexpr(_VEC):
                    visit_one(col + item, values[item])

            unroll_stride = block_size * fx.Int32(_LOAD_UNROLL)
            unroll_end = (full_vector_count > block_size * fx.Int32(_LOAD_UNROLL - 1)).select(
                full_vector_count - block_size * fx.Int32(_LOAD_UNROLL - 1), zero
            )
            n_unroll = (tid < unroll_end).select(
                (unroll_end - one - tid) // unroll_stride + one, zero
            )
            remain_col = full_vector_count * vec_width + tid
            if const_expr(reverse):
                vector_origin = full_vector_count - one
                vector_direction = -one
                if remain_col < row_len:
                    visit_one(remain_col, input_row[remain_col])
            else:
                vector_origin = zero
                vector_direction = one

            for offset in range(tid, unroll_end, unroll_stride):
                vector_idx0 = vector_origin + vector_direction * offset
                vector_stride = vector_direction * block_size
                vector_idx1 = vector_idx0 + vector_stride
                vector_idx2 = vector_idx1 + vector_stride
                vector_idx3 = vector_idx2 + vector_stride
                values0 = _load_f32x4(input_vector_tiles, vector_idx0)
                values1 = _load_f32x4(input_vector_tiles, vector_idx1)
                visit_vector(vector_idx0 * vec_width, values0)
                values2 = _load_f32x4(input_vector_tiles, vector_idx2)
                values3 = _load_f32x4(input_vector_tiles, vector_idx3)
                visit_vector(vector_idx1 * vec_width, values1)
                visit_vector(vector_idx2 * vec_width, values2)
                visit_vector(vector_idx3 * vec_width, values3)

            cleanup_offset = tid + n_unroll * unroll_stride
            for offset in range(cleanup_offset, full_vector_count, block_size):
                vector_idx = vector_origin + vector_direction * offset
                visit_vector(vector_idx * vec_width, _load_f32x4(input_vector_tiles, vector_idx))

            if const_expr(not reverse):
                if remain_col < row_len:
                    visit_one(remain_col, input_row[remain_col])

        def scan_lds_keys(body):
            row_vectors = (row_len + fx.Int32(_VEC - 1)) // fx.Int32(_VEC)
            for step in range_constexpr(full_key_vector_steps):
                vector_idx = step * block_threads + tid
                active = vector_idx < row_vectors
                safe_idx = active.select(vector_idx, zero)
                fragment = fx.make_rmem_tensor(full_key_fragment_layout, fx.Int32)
                fx.copy_atom_call(
                    full_key_load_atom, fx.slice(full_key_tiles, (None, safe_idx)), fragment
                )
                values = fragment.load()
                col_base = vector_idx * vec_width
                for item in range_constexpr(_VEC):
                    col = col_base + item
                    if active & (col < row_len):
                        body(col, values[item])

        # Radix histogram passes
        def accumulate_histogram_bucket(
            col,
            key,
            shift,
            mask,
            histograms,
            match_prefix=False,
            prefix_threshold=None,
            cache_key_index=False,
            use_ping_pong=False,
            full_keys=None,
            candidate_keys=None,
            candidate_indices=None,
            stable_keys=None,
            stable_indices=None,
            row_indices=None,
            row_values=None,
        ):
            active = col < row_len
            if full_keys is not None:
                if active:
                    full_keys[col] = key
            if match_prefix:
                prefix_shift = shift + mask.bit_length()
                prefix_mask = (1 << (_KEY_BITS - prefix_shift)) - 1
                prefix = radix_bucket(key, prefix_shift, prefix_mask)
                if cache_key_index:
                    if active & (prefix > prefix_threshold):
                        out_pos = atomic_add_i32(metadata, one, _RUNNING_ABOVE, "workgroup")
                        if out_pos < top_k:
                            if const_expr(stable):
                                stable_indices[out_pos] = col
                                if const_expr(write_values):
                                    stable_keys[out_pos] = ordered_value(key).bitcast(fx.Int32)
                            else:
                                row_indices[out_pos] = row_start + col
                                if const_expr(write_values):
                                    row_values[out_pos] = ordered_value(key)
                active = active & (prefix == prefix_threshold)
            if active:
                bucket = radix_bucket(key, shift, mask)
                if not use_ping_pong or (wave & one) == zero:
                    atomic_add_i32(histograms[0], one, bucket, "workgroup")
                else:
                    atomic_add_i32(histograms[1], one, bucket, "workgroup")
            if cache_key_index:
                if active:
                    candidate_pos = atomic_add_i32(metadata, one, _CANDIDATE_COUNT, "workgroup")
                    if candidate_pos < fx.Int32(_COMPACT_CAPACITY):
                        candidate_keys[candidate_pos] = key
                        candidate_indices[candidate_pos] = col

        # Non-stable emitters
        def scatter_unstable_keys(
            source,
            radix_shifts,
            prefix_threshold,
            num_needed,
            levels,
            row_indices,
            row_values,
            metadata,
            reset_above=True,
            candidate_count=None,
            candidate_keys=None,
            candidate_indices=None,
        ):
            reset_scatter_counters(metadata, reset_above=reset_above)

            def emit(col, key):
                above, equal = classify_prefix(key, radix_shifts, prefix_threshold, levels)
                scatter_unstable_key(
                    col, key, above, equal, num_needed, row_indices, row_values, metadata
                )

            if source == "lds":
                scan_lds_keys(emit)
            elif source == "gm":
                scan_gm_row(lambda col, value: emit(col, ordered_key(value)))
            else:
                for pos in range(tid, candidate_count, block_size):
                    emit(candidate_indices[pos], candidate_keys[pos])

        # Stable emitters
        def sort_and_store_stable(row_indices, row_values, candidate_keys, candidate_indices):
            for item in range_constexpr(stable_sort_items_per_thread):
                pos = tid + item * block_threads
                if (pos >= top_k) & (pos < fx.Int32(stable_sort_capacity)):
                    candidate_indices[pos] = fx.Int32(2147483647)
                    if const_expr(write_values):
                        candidate_keys[pos] = zero
            gpu.barrier()

            for stage in range_constexpr(len(stable_sort_sizes)):
                size = stable_sort_sizes[stage]
                stride = stable_sort_strides[stage]
                for item in range_constexpr(stable_sort_items_per_thread):
                    pos = tid + item * block_threads
                    partner = pos ^ fx.Int32(stride)
                    if (pos < fx.Int32(stable_sort_capacity)) & (partner > pos):
                        left = candidate_indices[pos]
                        right = candidate_indices[partner]
                        ascending = (pos & fx.Int32(size)) == zero
                        swap = ascending.select(left > right, left < right)
                        candidate_indices[pos] = swap.select(right, left)
                        candidate_indices[partner] = swap.select(left, right)
                        if const_expr(write_values):
                            left_value = candidate_keys[pos]
                            right_value = candidate_keys[partner]
                            candidate_keys[pos] = swap.select(right_value, left_value)
                            candidate_keys[partner] = swap.select(left_value, right_value)
                gpu.barrier()

            for item in range_constexpr(stable_sort_items_per_thread):
                pos = tid + item * block_threads
                if pos < top_k:
                    col = candidate_indices[pos]
                    row_indices[pos] = row_start + col
                    if const_expr(write_values):
                        row_values[pos] = candidate_keys[pos].bitcast(fx.Float32)

        def scatter_stable_sorted_keys(
            source,
            prefix_threshold,
            num_needed,
            levels,
            row_indices,
            row_values,
            candidate_keys,
            candidate_indices,
            stable_keys,
            stable_indices,
            metadata,
            candidate_count=None,
        ):
            definite_expected = top_k - num_needed
            if source == "gm":
                reset_scatter_counters(metadata)

            def collect(
                col, key, candidate_keys, candidate_indices, stable_keys, stable_indices, metadata
            ):
                above, equal = classify_prefix(key, _LONG_RADIX_SHIFTS, prefix_threshold, levels)
                if source == "gm":
                    if above:
                        pos = atomic_add_i32(metadata, one, _RUNNING_ABOVE, "workgroup")
                        if pos < definite_expected:
                            candidate_indices[pos] = col
                            if const_expr(write_values):
                                candidate_keys[pos] = ordered_value(key).bitcast(fx.Int32)
                    elif equal:
                        tie_pos = atomic_add_i32(metadata, one, _RUNNING_EQUAL, "workgroup")
                        if tie_pos < num_needed:
                            pos = definite_expected + tie_pos
                            candidate_indices[pos] = col
                            if const_expr(write_values):
                                candidate_keys[pos] = ordered_value(key).bitcast(fx.Int32)
                elif above | equal:
                    out_pos = atomic_add_i32(metadata, one, _RUNNING_ABOVE, "workgroup")
                    if out_pos < top_k:
                        stable_indices[out_pos] = col
                        if const_expr(write_values):
                            stable_keys[out_pos] = ordered_value(key).bitcast(fx.Int32)

            if source == "gm":
                scan_gm_row(
                    lambda col, value: collect(
                        col,
                        ordered_key(value),
                        candidate_keys,
                        candidate_indices,
                        stable_keys,
                        stable_indices,
                        metadata,
                    )
                )
            else:
                for pos in range(tid, candidate_count, block_size):
                    collect(
                        candidate_indices[pos],
                        candidate_keys[pos],
                        candidate_keys,
                        candidate_indices,
                        stable_keys,
                        stable_indices,
                        metadata,
                    )
            gpu.barrier()

            if source == "candidates":
                for item in range_constexpr(stable_sort_items_per_thread):
                    pos = tid + item * block_threads
                    if pos < top_k:
                        candidate_indices[pos] = stable_indices[pos]
                        if const_expr(write_values):
                            candidate_keys[pos] = stable_keys[pos]
                gpu.barrier()

            sort_and_store_stable(row_indices, row_values, candidate_keys, candidate_indices)

        def scatter_stable_ordered_keys(
            source,
            radix_shifts,
            prefix_threshold,
            num_needed,
            levels,
            row_indices,
            row_values,
            scan,
            metadata,
        ):
            def scatter_step(elems, above_base, equal_base, row_indices, row_values):
                num_elems = len(elems)
                # Class encoding: 2 = above, 1 = equal, 0 = below.
                classes = fx.make_rmem_tensor(num_elems, fx.Int32)
                packed_local = 0
                for item in range_constexpr(num_elems):
                    valid, key, _ = elems[item]
                    classes[item] = 0
                    if valid:
                        above, equal = classify_prefix(key, radix_shifts, prefix_threshold, levels)
                        if above:
                            classes[item] = 2
                            packed_local = packed_local + fx.Int32(1 << 16)
                        elif equal:
                            classes[item] = 1
                            packed_local = packed_local + 1

                packed_prefix, packed_step_total = block_excl_prefix_i32(packed_local, scan)
                my_above = above_base + (packed_prefix >> 16)
                my_eq = equal_base + (packed_prefix & fx.Int32(0xFFFF))
                for item in range_constexpr(num_elems):
                    valid, key, col = elems[item]
                    cls = classes[item]
                    if valid:
                        accepted_equal = (my_eq < num_needed).select(my_eq, num_needed)
                        out_pos = my_above + accepted_equal
                        if cls == 2:
                            row_indices[out_pos] = row_start + col
                            if const_expr(write_values):
                                row_values[out_pos] = ordered_value(key)
                            my_above = my_above + 1
                        elif cls == 1:
                            if my_eq < num_needed:
                                row_indices[out_pos] = row_start + col
                                if const_expr(write_values):
                                    row_values[out_pos] = ordered_value(key)
                            my_eq = my_eq + 1
                above_base = above_base + (packed_step_total >> 16)
                next_eq_base = equal_base + (packed_step_total & fx.Int32(0xFFFF))
                equal_base = (next_eq_base < num_needed).select(next_eq_base, num_needed)
                gpu.barrier()
                return above_base, equal_base

            above_base = 0
            equal_base = 0
            if source == "lds":
                row_vectors = (row_len + fx.Int32(_VEC - 1)) // vec_width
                for step in range_constexpr(full_key_vector_steps):
                    vector_idx = step * block_threads + tid
                    active_vector = vector_idx < row_vectors
                    safe_vector_idx = active_vector.select(vector_idx, 0)
                    fragment = fx.make_rmem_tensor(full_key_fragment_layout, fx.Int32)
                    fx.copy_atom_call(
                        full_key_load_atom,
                        fx.slice(full_key_tiles, (None, safe_vector_idx)),
                        fragment,
                    )
                    keys = fragment.load()
                    elems = []
                    for item in range_constexpr(_VEC):
                        col = vector_idx * vec_width + item
                        elems.append((active_vector & (col < row_len), keys[item], col))
                    above_base, equal_base = scatter_step(
                        elems, above_base, equal_base, row_indices, row_values
                    )
            else:
                num_steps = (full_vector_count + block_size - 1) // block_size
                for step in range(0, num_steps, 1):
                    vector_idx = step * block_size + tid
                    active_vector = vector_idx < full_vector_count
                    safe_vector_idx = active_vector.select(vector_idx, 0)
                    values = _load_f32x4(input_vector_tiles, safe_vector_idx)
                    keys = fx.make_rmem_tensor(_VEC, fx.Int32)
                    for item in range_constexpr(_VEC):
                        keys[item] = ordered_key(values[item])
                    elems = []
                    for item in range_constexpr(_VEC):
                        col = vector_idx * vec_width + item
                        elems.append((active_vector & (col < row_len), keys[item], col))
                    above_base, equal_base = scatter_step(
                        elems, above_base, equal_base, row_indices, row_values
                    )

                remain_base = full_vector_count * vec_width
                if remain_base < row_len:
                    col = remain_base + tid
                    valid = col < row_len
                    safe_col = valid.select(col, 0)
                    elems = [(valid, ordered_key(input_row[safe_col]), col)]
                    above_base, equal_base = scatter_step(
                        elems, above_base, equal_base, row_indices, row_values
                    )

        def scatter_streaming_stable(
            candidate_count,
            prefix_threshold,
            num_needed,
            row_indices,
            row_values,
            scan,
            metadata,
            candidate_keys,
            candidate_indices,
            stable_keys,
            stable_indices,
        ):
            if const_expr(stable_sort_enabled):
                can_use_fast = (row_len >= fx.Int32(_STABLE_FAST_MIN_ROW_LEN)) & (
                    metadata[_SELECTED_BUCKET_COUNT] == num_needed
                )
                can_use_compact_fast = can_use_fast & (
                    candidate_count <= fx.Int32(_COMPACT_CAPACITY)
                )
                if can_use_compact_fast:
                    scatter_stable_sorted_keys(
                        "candidates",
                        prefix_threshold,
                        num_needed,
                        3,
                        row_indices,
                        row_values,
                        candidate_keys,
                        candidate_indices,
                        stable_keys,
                        stable_indices,
                        metadata,
                        candidate_count=candidate_count,
                    )
                else:
                    if can_use_fast:
                        scatter_stable_sorted_keys(
                            "gm",
                            prefix_threshold,
                            num_needed,
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
                        scatter_stable_ordered_keys(
                            "gm",
                            _LONG_RADIX_SHIFTS,
                            prefix_threshold,
                            num_needed,
                            3,  # level
                            row_indices,
                            row_values,
                            scan,
                            metadata,
                        )
            else:
                scatter_stable_ordered_keys(
                    "gm",
                    _LONG_RADIX_SHIFTS,
                    prefix_threshold,
                    num_needed,
                    3,  # level
                    row_indices,
                    row_values,
                    scan,
                    metadata,
                )

        def run_cached_path(histograms, full_keys, row_indices, row_values, scan, metadata):
            def scatter_selection(prefix_threshold, num_needed, levels):
                if const_expr(stable):
                    scatter_stable_ordered_keys(
                        "lds",
                        _SHORT_RADIX_SHIFTS,
                        prefix_threshold,
                        num_needed,
                        levels,
                        row_indices,
                        row_values,
                        scan,
                        metadata,
                    )
                else:
                    scatter_unstable_keys(
                        "lds",
                        _SHORT_RADIX_SHIFTS,
                        prefix_threshold,
                        num_needed,
                        levels,
                        row_indices,
                        row_values,
                        metadata,
                    )

            # A return inside a dynamic FlyDSL if only exits its generated
            # scf.if helper. Nest the remaining stages to make the exit real.
            # First radix pass.
            clear_histograms(histograms, short_bins_per_thread)
            scan_gm_row(
                lambda col, value: accumulate_histogram_bucket(
                    col,
                    ordered_key(value),
                    _SHORT_RADIX_SHIFTS[0],
                    _SHORT_RADIX_MASKS[0],
                    histograms,
                    use_ping_pong=True,
                    full_keys=full_keys,
                )
            )
            gpu.barrier()
            merge_histograms(histograms, short_bins_per_thread)
            choose_threshold(
                top_k,
                _FIRST_ABOVE,
                _FIRST_THRESHOLD,
                _SELECTED_BUCKET_COUNT,
                histograms[0],
                scan,
                metadata,
                short_bins_per_thread,
            )

            remaining_k = top_k - metadata[_FIRST_ABOVE]
            prefix_threshold = metadata[_FIRST_THRESHOLD]
            if metadata[_SELECTED_BUCKET_COUNT] == remaining_k:
                # Fast exit after the first radix pass.
                scatter_selection(prefix_threshold, remaining_k, 1)
            else:
                # Second radix pass.
                clear_histograms((histograms[0],), short_bins_per_thread)
                scan_lds_keys(
                    lambda col, key: accumulate_histogram_bucket(
                        col,
                        key,
                        _SHORT_RADIX_SHIFTS[1],
                        _SHORT_RADIX_MASKS[1],
                        (histograms[0],),
                        match_prefix=True,
                        prefix_threshold=prefix_threshold,
                    )
                )
                gpu.barrier()
                choose_threshold(
                    remaining_k,
                    _SECOND_ABOVE,
                    _SECOND_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    histograms[0],
                    scan,
                    metadata,
                    short_bins_per_thread,
                )

                remaining_k = remaining_k - metadata[_SECOND_ABOVE]
                prefix_threshold = (
                    prefix_threshold << fx.Int32(_SHORT_RADIX_BITS[1])
                ) | metadata[_SECOND_THRESHOLD]
                if metadata[_SELECTED_BUCKET_COUNT] == remaining_k:
                    # Fast exit after the second radix pass.
                    scatter_selection(prefix_threshold, remaining_k, 2)
                else:
                    # Third radix pass.
                    clear_histograms((histograms[0],), short_bins_per_thread)
                    scan_lds_keys(
                        lambda col, key: accumulate_histogram_bucket(
                            col,
                            key,
                            _SHORT_RADIX_SHIFTS[2],
                            _SHORT_RADIX_MASKS[2],
                            (histograms[0],),
                            match_prefix=True,
                            prefix_threshold=prefix_threshold,
                        )
                    )
                    gpu.barrier()
                    choose_threshold(
                        remaining_k,
                        _THIRD_ABOVE,
                        _THIRD_THRESHOLD,
                        _SELECTED_BUCKET_COUNT,
                        histograms[0],
                        scan,
                        metadata,
                        short_bins_per_thread,
                    )

                    remaining_k = remaining_k - metadata[_THIRD_ABOVE]
                    prefix_threshold = (
                        prefix_threshold << fx.Int32(_SHORT_RADIX_BITS[2])
                    ) | metadata[_THIRD_THRESHOLD]
                    scatter_selection(prefix_threshold, remaining_k, 3)

        def run_streaming_path(
            histograms,
            histogram,
            candidate_keys,
            candidate_indices,
            stable_keys,
            stable_indices,
            row_indices,
            row_values,
            scan,
            metadata,
        ):
            # Level 1: select the high radix bucket.
            clear_histograms(histograms, high_bins_per_thread)
            scan_gm_row(
                lambda col, value: accumulate_histogram_bucket(
                    col,
                    ordered_key(value),
                    _LONG_RADIX_SHIFTS[0],
                    _LONG_RADIX_MASKS[0],
                    histograms,
                    use_ping_pong=True,
                )
            )
            gpu.barrier()
            merge_histograms(histograms, high_bins_per_thread)
            choose_threshold(
                top_k,
                _FIRST_ABOVE,
                _FIRST_THRESHOLD,
                _SELECTED_BUCKET_COUNT,
                histograms[0],
                scan,
                metadata,
                high_bins_per_thread,
            )

            remaining_k = top_k - metadata[_FIRST_ABOVE]
            prefix_threshold = metadata[_FIRST_THRESHOLD]
            candidate_count = zero
            unstable_mode = fx.Int32(int(not stable)) == one
            can_finish = unstable_mode & (
                metadata[_SELECTED_BUCKET_COUNT] == remaining_k
            )
            if can_finish:
                # Fast exit after the first radix pass.
                scatter_unstable_keys(
                    "gm",
                    _LONG_RADIX_SHIFTS,
                    prefix_threshold,
                    remaining_k,
                    1,
                    row_indices,
                    row_values,
                    metadata,
                )
            else:
                # Level 2: compact the selected high bucket and select its middle bucket.
                clear_histograms((histogram,), later_bins_per_thread)
                scan_gm_row(
                    lambda col, value: accumulate_histogram_bucket(
                        col,
                        ordered_key(value),
                        _LONG_RADIX_SHIFTS[1],
                        _LONG_RADIX_MASKS[1],
                        (histogram,),
                        match_prefix=True,
                        prefix_threshold=prefix_threshold,
                        cache_key_index=True,
                        candidate_keys=candidate_keys,
                        candidate_indices=candidate_indices,
                        stable_keys=stable_keys,
                        stable_indices=stable_indices,
                        row_indices=row_indices,
                        row_values=row_values,
                    ),
                    reverse=True,
                )
                gpu.barrier()
                candidate_count = metadata[_CANDIDATE_COUNT]
                choose_threshold(
                    remaining_k,
                    _SECOND_ABOVE,
                    _SECOND_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    histogram,
                    scan,
                    metadata,
                    later_bins_per_thread,
                )

                remaining_k = remaining_k - metadata[_SECOND_ABOVE]
                prefix_threshold = (
                    prefix_threshold << fx.Int32(_LONG_RADIX_BITS[1])
                ) | metadata[_SECOND_THRESHOLD]
                can_finish = (
                    unstable_mode
                    & (metadata[_SELECTED_BUCKET_COUNT] == remaining_k)
                    & (candidate_count <= fx.Int32(_COMPACT_CAPACITY))
                )
                if can_finish:
                    # Fast exit after the second radix pass.
                    scatter_unstable_keys(
                        "candidates",
                        _LONG_RADIX_SHIFTS,
                        prefix_threshold,
                        remaining_k,
                        2,
                        row_indices,
                        row_values,
                        metadata,
                        reset_above=False,
                        candidate_count=candidate_count,
                        candidate_keys=candidate_keys,
                        candidate_indices=candidate_indices,
                    )
                else:
                    # Level 3: select the low bucket and emit the final result.
                    clear_histograms((histogram,), later_bins_per_thread)
                    if candidate_count <= fx.Int32(_COMPACT_CAPACITY):
                        for pos in range(tid, candidate_count, block_size):
                            accumulate_histogram_bucket(
                                pos,
                                candidate_keys[pos],
                                _LONG_RADIX_SHIFTS[2],
                                _LONG_RADIX_MASKS[2],
                                (histogram,),
                                match_prefix=True,
                                prefix_threshold=prefix_threshold,
                            )
                    else:
                        scan_gm_row(
                            lambda col, value: accumulate_histogram_bucket(
                                col,
                                ordered_key(value),
                                _LONG_RADIX_SHIFTS[2],
                                _LONG_RADIX_MASKS[2],
                                (histogram,),
                                match_prefix=True,
                                prefix_threshold=prefix_threshold,
                            )
                        )
                    gpu.barrier()
                    choose_threshold(
                        remaining_k,
                        _THIRD_ABOVE,
                        _THIRD_THRESHOLD,
                        _SELECTED_BUCKET_COUNT,
                        histogram,
                        scan,
                        metadata,
                        later_bins_per_thread,
                    )

                    # All radix passes are complete; write the result.
                    remaining_k = remaining_k - metadata[_THIRD_ABOVE]
                    prefix_threshold = (
                        prefix_threshold << fx.Int32(_LONG_RADIX_BITS[2])
                    ) | metadata[_THIRD_THRESHOLD]
                    if const_expr(stable):
                        scatter_streaming_stable(
                            candidate_count,
                            prefix_threshold,
                            remaining_k,
                            row_indices,
                            row_values,
                            scan,
                            metadata,
                            candidate_keys,
                            candidate_indices,
                            stable_keys,
                            stable_indices,
                        )
                    else:
                        if candidate_count <= fx.Int32(_COMPACT_CAPACITY):
                            scatter_unstable_keys(
                                "candidates",
                                _LONG_RADIX_SHIFTS,
                                prefix_threshold,
                                remaining_k,
                                3,
                                row_indices,
                                row_values,
                                metadata,
                                reset_above=False,
                                candidate_count=candidate_count,
                                candidate_keys=candidate_keys,
                                candidate_indices=candidate_indices,
                            )
                        else:
                            scatter_unstable_keys(
                                "gm",
                                _LONG_RADIX_SHIFTS,
                                prefix_threshold,
                                remaining_k,
                                3,
                                row_indices,
                                row_values,
                                metadata,
                            )

        def write_direct_output(row_indices, row_values, row_index_tiles, row_value_tiles):
            for step in range_constexpr(output_vector_steps):
                vector_idx = step * block_threads + tid
                if vector_idx < output_vector_count:
                    col = vector_idx * vec_width
                    index_values = [
                        (col + item < row_len).select(row_start + col + item, fx.Int32(-1))
                        for item in range_constexpr(_VEC)
                    ]
                    fragment = fx.make_rmem_tensor(index_fragment_layout, fx.Int32)
                    fragment.store(fx.Vector.from_elements(index_values, dtype=fx.Int32))
                    fx.copy_atom_call(
                        index_store_atom, fragment, fx.slice(row_index_tiles, (None, vector_idx))
                    )

                    if const_expr(write_values):
                        output_values = []
                        for item in range_constexpr(_VEC):
                            local_col = col + item
                            valid = local_col < row_len
                            safe_col = valid.select(local_col, zero)
                            output_values.append(
                                valid.select(input_row[safe_col], fx.Float32(float("-inf")))
                            )
                        value_fragment = fx.make_rmem_tensor(value_fragment_layout, fx.Float32)
                        value_fragment.store(
                            fx.Vector.from_elements(output_values, dtype=fx.Float32)
                        )
                        fx.copy_atom_call(
                            value_store_atom,
                            value_fragment,
                            fx.slice(row_value_tiles, (None, vector_idx)),
                        )

            tail = output_vector_count * _VEC + tid
            if tail < k:
                valid = tail < row_len
                safe_tail = valid.select(tail, zero)
                row_indices[tail] = valid.select(row_start + tail, fx.Int32(-1))
                if const_expr(write_values):
                    row_values[tail] = valid.select(input_row[safe_tail], fx.Float32(float("-inf")))

        # Kernel control flow
        if row_len <= top_k:
            write_direct_output(row_indices, row_values, row_index_tiles, row_value_tiles)

        if row_len > top_k:
            if tid < _METADATA_SIZE:
                metadata[tid] = zero
            gpu.barrier()

            if row_len <= fx.Int32(_COMPACT_CAPACITY):
                run_cached_path(
                    short_histograms, full_keys, row_indices, row_values, scan, metadata
                )
            else:
                run_streaming_path(
                    long_histograms,
                    histogram,
                    candidate_keys,
                    candidate_indices,
                    stable_keys,
                    stable_indices,
                    row_indices,
                    row_values,
                    scan,
                    metadata,
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
            input, row_starts, row_ends, indices, values
        ).launch(grid=(rows_m, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch_topk_per_row_prefill_one_workgroup
