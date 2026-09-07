# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, ptrtoint, range_constexpr
from flydsl.expr.typing import T

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
_SHORT_LATER_BUCKETS = 1 << max(
    _SHORT_MIDDLE_BITS, _SHORT_LOW_BITS
)
_SHORT_HIGH_SHIFT = _SHORT_MIDDLE_BITS + _SHORT_LOW_BITS
_SHORT_MIDDLE_SHIFT = _SHORT_LOW_BITS
_SHORT_MIDDLE_MASK = (1 << _SHORT_MIDDLE_BITS) - 1
_SHORT_LOW_MASK = (1 << _SHORT_LOW_BITS) - 1
_MAX_ROW_ELEMENTS = ((1 << 32) - 1) // 4
_COMPACT_CAPACITY = 4096

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


@cache
def build_topk_per_row_prefill_one_workgroup_module(
    k: int,
    block_threads: int = 1024,
    device_index: int = 0,
    backend: str = "rocm",
):
    del device_index, backend
    if k <= 0:
        raise ValueError("k must be positive")
    if block_threads not in (256, 1024):
        raise ValueError("block_threads must be 256 or 1024")

    num_waves = block_threads // _WAVE_SIZE
    high_bins_per_thread = _HIGH_BUCKETS // block_threads
    later_bins_per_thread = _LATER_BUCKETS // block_threads
    short_bins_per_thread = _SHORT_HIGH_BUCKETS // block_threads
    full_key_vector_steps = (
        (_COMPACT_CAPACITY // _VEC) + block_threads - 1
    ) // block_threads
    output_vector_count = k // _VEC
    output_vector_steps = (
        output_vector_count + block_threads - 1
    ) // block_threads
    output_vector_elems = max(_VEC, output_vector_count * _VEC)

    @fx.struct
    class LongPass1Storage:
        histogram0: fx.Array[fx.Int32, _HIGH_BUCKETS, 16]
        histogram1: fx.Array[fx.Int32, _HIGH_BUCKETS, 16]

    @fx.struct
    class LongLaterStorage:
        histogram: fx.Array[fx.Int32, _LATER_BUCKETS, 16]
        candidate_keys: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]
        candidate_indices: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]

    @fx.struct
    class ShortPass1Storage:
        histogram0: fx.Array[fx.Int32, _SHORT_HIGH_BUCKETS, 16]
        histogram1: fx.Array[fx.Int32, _SHORT_HIGH_BUCKETS, 16]
        full_keys: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]

    @fx.struct
    class ShortLaterStorage:
        histogram: fx.Array[fx.Int32, _SHORT_LATER_BUCKETS, 16]
        padding: fx.Array[fx.Int32, _SHORT_LATER_BUCKETS, 16]
        full_keys: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]

    @fx.union
    class ArenaStorage:
        long_pass1: LongPass1Storage
        long_later: LongLaterStorage
        short_pass1: ShortPass1Storage
        short_later: ShortLaterStorage

    @fx.struct
    class SharedStorage:
        arena: ArenaStorage
        scan: fx.Array[fx.Int32, num_waves * 2, 16]
        metadata: fx.Array[fx.Int32, 10, 16]

    @flyc.kernel(
        name=f"topk_per_row_prefill_1wg_gfx1250_k{k}_b{block_threads}",
        known_block_size=[block_threads, 1, 1],
    )
    def topk_per_row_prefill_one_workgroup_kernel(
        input_ptr: fx.Pointer,
        row_starts_ptr: fx.Pointer,
        row_ends_ptr: fx.Pointer,
        indices_ptr: fx.Pointer,
        stride0: fx.Int32,
    ):
        row = fx.Int32(fx.block_idx.x)
        tid = fx.thread_idx.x
        lane = tid % _WAVE_SIZE
        wave = tid // _WAVE_SIZE

        zero = fx.Int32(0)
        one = fx.Int32(1)
        two = fx.Int32(2)
        vec_width = fx.Int32(_VEC)
        block_size = fx.Int32(block_threads)
        top_k = fx.Int32(k)
        sign_bit = fx.Int32(-2147483648)

        i32_ptr_type = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        f32_ptr_type = fx.PointerType.get(
            T.f32, address_space=fx.AddressSpace.Global, alignment=4
        )

        row_starts = fx.inttoptr(
            i32_ptr_type, fx.Int64(ptrtoint(row_starts_ptr))
        )
        row_ends = fx.inttoptr(
            i32_ptr_type, fx.Int64(ptrtoint(row_ends_ptr))
        )
        row_start = row_starts[row]
        row_len = row_ends[row] - row_start

        input_row_addr = (
            fx.Int64(ptrtoint(input_ptr))
            + fx.Int64(row) * fx.Int64(stride0) * fx.Int64(4)
            + fx.Int64(row_start) * fx.Int64(4)
        )
        input_row = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.inttoptr(f32_ptr_type, input_row_addr),
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
        aligned_addr = input_row_addr + fx.Int64(skip_cnt) * fx.Int64(4)
        aligned_row = fx.rocdl.make_buffer_tensor(
            fx.make_view(
                fx.inttoptr(f32_ptr_type, aligned_addr),
                fx.make_layout(_MAX_ROW_ELEMENTS, 1),
            ),
            num_records_bytes=fx.Int64(aligned_len) * fx.Int64(4),
        )
        input_resource = fx.logical_divide(
            aligned_row, fx.make_layout(_VEC, 1)
        )

        row_indices = fx.inttoptr(
            i32_ptr_type,
            fx.Int64(ptrtoint(indices_ptr))
            + fx.Int64(row) * fx.Int64(k * 4),
        )
        row_index_tiles = fx.logical_divide(
            fx.make_view(
                row_indices,
                fx.make_layout(output_vector_elems, 1),
            ),
            fx.make_layout(_VEC, 1),
        )
        index_store_atom = fx.make_copy_atom(
            fx.UniversalCopy128b(), fx.Int32
        )
        index_fragment_layout = fx.make_layout(_VEC, 1)

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
        short_histogram = (
            storage.arena.short_later.histogram.peek().view(
                fx.make_layout(_SHORT_LATER_BUCKETS, 1)
            )
        )
        scan = storage.scan.peek().view(fx.make_layout(num_waves * 2, 1))
        metadata = storage.metadata.peek().view(fx.make_layout(10, 1))
        candidate_keys = storage.arena.long_later.candidate_keys.peek().view(
            fx.make_layout(_COMPACT_CAPACITY, 1)
        )
        candidate_indices = (
            storage.arena.long_later.candidate_indices.peek().view(
                fx.make_layout(_COMPACT_CAPACITY, 1)
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

        def choose_high_threshold(
            target_k,
            above_slot,
            threshold_slot,
            count_slot,
            histogram,
            scan,
            metadata,
        ):
            first_bin = tid * fx.Int32(high_bins_per_thread)
            count0 = histogram[first_bin]
            count1 = histogram[first_bin + one]
            count2 = histogram[first_bin + two]
            count3 = histogram[first_bin + fx.Int32(3)]
            local_total = count0 + count1 + count2 + count3
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
            exclusive0 = wave_offset + wave_exclusive
            inclusive0 = exclusive0 + count0
            inclusive1 = inclusive0 + count1
            inclusive2 = inclusive1 + count2
            inclusive3 = inclusive2 + count3

            def emit(bucket, exclusive, inclusive, metadata):
                if (exclusive <= target_prefix) & (
                    inclusive > target_prefix
                ):
                    metadata[threshold_slot] = bucket
                    metadata[above_slot] = total - inclusive
                    metadata[count_slot] = inclusive - exclusive

            emit(first_bin, exclusive0, inclusive0, metadata)
            emit(first_bin + one, inclusive0, inclusive1, metadata)
            emit(first_bin + two, inclusive1, inclusive2, metadata)
            emit(
                first_bin + fx.Int32(3),
                inclusive2,
                inclusive3,
                metadata,
            )
            gpu.barrier()

        def choose_later_threshold(
            target_k,
            above_slot,
            threshold_slot,
            count_slot,
            histogram,
            scan,
            metadata,
        ):
            count = histogram[tid]
            wave_inclusive = _warp_inclusive_prefix_i32(
                count, lane, _WAVE_SIZE
            )
            wave_exclusive = wave_inclusive - count

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

            exclusive = scan[wave + num_waves] + wave_exclusive
            inclusive = exclusive + count
            total = scan[num_waves - 1] + scan[num_waves * 2 - 1]
            target_prefix = total - target_k
            if (exclusive <= target_prefix) & (
                inclusive > target_prefix
            ):
                metadata[threshold_slot] = tid
                metadata[above_slot] = total - inclusive
                metadata[count_slot] = count
            gpu.barrier()

        def choose_short_threshold(
            target_k,
            above_slot,
            threshold_slot,
            count_slot,
            histogram,
            scan,
            metadata,
        ):
            first_bin = tid * fx.Int32(short_bins_per_thread)
            counts = fx.make_rmem_tensor(
                short_bins_per_thread, fx.Int32
            )
            local_total = zero
            for item in range_constexpr(short_bins_per_thread):
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
            for item in range_constexpr(short_bins_per_thread):
                inclusive = exclusive + counts[item]
                if (exclusive <= target_prefix) & (
                    inclusive > target_prefix
                ):
                    metadata[threshold_slot] = first_bin + item
                    metadata[above_slot] = total - inclusive
                    metadata[count_slot] = inclusive - exclusive
                exclusive = inclusive
            gpu.barrier()

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
            row_indices,
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
                    row_indices[out_pos] = row_start + col
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
            col,
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
            row_indices,
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
                    row_indices,
                )

        def pass2_vec(
            col,
            values,
            first_threshold,
            histogram,
            metadata,
            candidate_keys,
            candidate_indices,
            row_indices,
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
                    row_indices,
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

        def classify(
            key,
            first_threshold,
            second_threshold,
            third_threshold,
        ):
            first = high_bucket(key)
            second = middle_bucket(key)
            third = low_bucket(key)
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

        def cached_scatter(
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            levels,
            row_indices,
            metadata,
        ):
            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
                metadata[_RUNNING_EQUAL] = zero
            gpu.barrier()

            def emit(col, key, row_indices, metadata):
                first = short_high_bucket(key)
                second = short_middle_bucket(key)
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
                    third = short_low_bucket(key)
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

                if above:
                    pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_ABOVE,
                        "workgroup",
                    )
                    if pos < top_k:
                        row_indices[pos] = row_start + col
                elif equal:
                    back_pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_EQUAL,
                        "workgroup",
                    )
                    if back_pos < num_needed:
                        row_indices[top_k - one - back_pos] = (
                            row_start + col
                        )

            scan_full_keys(
                lambda col, key: emit(
                    col,
                    key,
                    row_indices,
                    metadata,
                )
            )

        def full_scatter(
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            row_indices,
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
                            row_indices[pos] = row_start + col
                    elif equal:
                        back_pos = atomic_add_i32(
                            metadata,
                            one,
                            _RUNNING_EQUAL,
                            "workgroup",
                        )
                        if back_pos < num_needed:
                            row_indices[top_k - one - back_pos] = (
                                row_start + col
                            )

            def scatter_vec(col, values):
                for item in range_constexpr(_VEC):
                    scatter_one(
                        col + item,
                        values[item],
                        row_indices,
                        metadata,
                    )

            scan_row(
                scatter_vec,
                lambda col, value: scatter_one(
                    col,
                    value,
                    row_indices,
                    metadata,
                ),
            )

        def compact_scatter_high(
            candidate_count,
            row_indices,
            candidate_indices,
        ):
            for pos in range(tid, candidate_count, block_size):
                row_indices[top_k - one - pos] = (
                    row_start + candidate_indices[pos]
                )

        def compact_scatter_middle(
            candidate_count,
            second_threshold,
            num_needed,
            row_indices,
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
                if second > second_threshold:
                    out_pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_ABOVE,
                        "workgroup",
                    )
                    if out_pos < top_k:
                        row_indices[out_pos] = row_start + col
                elif second == second_threshold:
                    back_pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_EQUAL,
                        "workgroup",
                    )
                    if back_pos < num_needed:
                        row_indices[top_k - one - back_pos] = (
                            row_start + col
                        )

        def compact_scatter(
            candidate_count,
            first_threshold,
            second_threshold,
            third_threshold,
            num_needed,
            row_indices,
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
                if above:
                    out_pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_ABOVE,
                        "workgroup",
                    )
                    if out_pos < top_k:
                        row_indices[out_pos] = row_start + col
                elif equal:
                    back_pos = atomic_add_i32(
                        metadata,
                        one,
                        _RUNNING_EQUAL,
                        "workgroup",
                    )
                    if back_pos < num_needed:
                        row_indices[top_k - one - back_pos] = (
                            row_start + col
                        )

        def finish_cached_path(
            first_threshold,
            need_after_first,
            histogram,
            scan,
            metadata,
            row_indices,
        ):
            if metadata[_SELECTED_BUCKET_COUNT] == need_after_first:
                cached_scatter(
                    first_threshold,
                    zero,
                    zero,
                    need_after_first,
                    1,
                    row_indices,
                    metadata,
                )
            else:
                choose_short_threshold(
                    need_after_first,
                    _SECOND_ABOVE,
                    _SECOND_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    histogram,
                    scan,
                    metadata,
                )
                second_threshold = metadata[_SECOND_THRESHOLD]
                need_after_second = (
                    need_after_first - metadata[_SECOND_ABOVE]
                )
                if (
                    metadata[_SELECTED_BUCKET_COUNT]
                    == need_after_second
                ):
                    cached_scatter(
                        first_threshold,
                        second_threshold,
                        zero,
                        need_after_second,
                        2,
                        row_indices,
                        metadata,
                    )
                else:
                    clear_histogram(histogram, short_bins_per_thread)

                    def low_histogram(col, key):
                        if (
                            short_high_bucket(key) == first_threshold
                        ) & (
                            short_middle_bucket(key)
                            == second_threshold
                        ):
                            atomic_add_i32(
                                histogram,
                                one,
                                short_low_bucket(key),
                                "workgroup",
                            )

                    scan_full_keys(low_histogram)
                    gpu.barrier()
                    choose_short_threshold(
                        need_after_second,
                        _THIRD_ABOVE,
                        _THIRD_THRESHOLD,
                        _SELECTED_BUCKET_COUNT,
                        histogram,
                        scan,
                        metadata,
                    )
                    cached_scatter(
                        first_threshold,
                        second_threshold,
                        metadata[_THIRD_THRESHOLD],
                        need_after_second - metadata[_THIRD_ABOVE],
                        3,
                        row_indices,
                        metadata,
                    )

        def finish_candidate_path(
            first_threshold,
            need_after_first,
            candidate_count,
            histogram,
            scan,
            metadata,
            candidate_keys,
            candidate_indices,
            row_indices,
        ):
            can_finish_after_first = (
                candidate_count <= fx.Int32(_COMPACT_CAPACITY)
            ) & (
                metadata[_SELECTED_BUCKET_COUNT] == need_after_first
            )
            if can_finish_after_first:
                compact_scatter_high(
                    candidate_count,
                    row_indices,
                    candidate_indices,
                )
            else:
                choose_later_threshold(
                    need_after_first,
                    _SECOND_ABOVE,
                    _SECOND_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    histogram,
                    scan,
                    metadata,
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
                    compact_scatter_middle(
                        candidate_count,
                        second_threshold,
                        need_after_second,
                        row_indices,
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
                    choose_later_threshold(
                        need_after_second,
                        _THIRD_ABOVE,
                        _THIRD_THRESHOLD,
                        _SELECTED_BUCKET_COUNT,
                        histogram,
                        scan,
                        metadata,
                    )
                    third_threshold = metadata[_THIRD_THRESHOLD]
                    num_needed = (
                        need_after_second - metadata[_THIRD_ABOVE]
                    )
                    if candidate_count <= fx.Int32(
                        _COMPACT_CAPACITY
                    ):
                        compact_scatter(
                            candidate_count,
                            first_threshold,
                            second_threshold,
                            third_threshold,
                            num_needed,
                            row_indices,
                            metadata,
                            candidate_keys,
                            candidate_indices,
                        )
                    else:
                        full_scatter(
                            first_threshold,
                            second_threshold,
                            third_threshold,
                            num_needed,
                            row_indices,
                            metadata,
                        )

        if row_len <= top_k:
            for step in range_constexpr(output_vector_steps):
                vector_idx = step * block_threads + tid
                if vector_idx < output_vector_count:
                    col = vector_idx * vec_width
                    values = [
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
                        fx.Vector.from_elements(values, dtype=fx.Int32)
                    )
                    fx.copy_atom_call(
                        index_store_atom,
                        fragment,
                        fx.slice(
                            row_index_tiles,
                            (None, vector_idx),
                        ),
                    )
            tail = output_vector_count * _VEC + tid
            if tail < k:
                row_indices[tail] = (tail < row_len).select(
                    row_start + tail,
                    fx.Int32(-1),
                )

        if row_len > top_k:
            if tid < 10:
                metadata[tid] = zero
            gpu.barrier()

            cache_all = row_len <= fx.Int32(_COMPACT_CAPACITY)
            if cache_all:
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
                choose_short_threshold(
                    top_k,
                    _FIRST_ABOVE,
                    _FIRST_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    short_pass1_histogram0,
                    scan,
                    metadata,
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
                choose_high_threshold(
                    top_k,
                    _FIRST_ABOVE,
                    _FIRST_THRESHOLD,
                    _SELECTED_BUCKET_COUNT,
                    pass1_histogram0,
                    scan,
                    metadata,
                )
            first_threshold = metadata[_FIRST_THRESHOLD]

            if cache_all:
                clear_histogram(
                    short_histogram, short_bins_per_thread
                )
            else:
                clear_histogram(histogram, later_bins_per_thread)
            if tid == 0:
                metadata[_RUNNING_ABOVE] = zero
                metadata[_CANDIDATE_COUNT] = zero
            gpu.barrier()
            if cache_all:
                scan_full_keys(
                    lambda col, key: pass2_cached_key(
                        col,
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
                        row_indices,
                    ),
                    lambda col, value: pass2_one(
                        col,
                        value,
                        first_threshold,
                        histogram,
                        metadata,
                        candidate_keys,
                        candidate_indices,
                        row_indices,
                    ),
                )
            gpu.barrier()
            need_after_first = top_k - metadata[_FIRST_ABOVE]
            if cache_all:
                finish_cached_path(
                    first_threshold,
                    need_after_first,
                    short_histogram,
                    scan,
                    metadata,
                    row_indices,
                )
            else:
                finish_candidate_path(
                    first_threshold,
                    need_after_first,
                    metadata[_CANDIDATE_COUNT],
                    histogram,
                    scan,
                    metadata,
                    candidate_keys,
                    candidate_indices,
                    row_indices,
                )

    @flyc.jit
    def launch_topk_per_row_prefill_one_workgroup(
        input_ptr: fx.Pointer,
        row_starts_ptr: fx.Pointer,
        row_ends_ptr: fx.Pointer,
        indices_ptr: fx.Pointer,
        stride0: fx.Int32,
        rows_m: fx.Int32,
        stream: fx.Stream,
    ):
        topk_per_row_prefill_one_workgroup_kernel(
            input_ptr,
            row_starts_ptr,
            row_ends_ptr,
            indices_ptr,
            stride0,
        ).launch(
            grid=(rows_m, 1, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launch_topk_per_row_prefill_one_workgroup
