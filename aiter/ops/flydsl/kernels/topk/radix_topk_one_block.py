# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wave32/wave64 radix TopK with one block per input row.

Short rows cache ordered keys in LDS. Long rows compact the selected radix
bucket; stable modes preserve index ordering and tie-breaking.
Rows with length <= k copy/pad in the same kernel without running radix.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from ..dpp_utils import update_dpp_i32
from ..kernels_common import atomic_add_i32, atomic_or_i32
from .topk_common import (
    _f32_to_ord,
    _load_f32x4,
    _row_length,
    _warp_inclusive_prefix_i32,
)

_VEC = 4
_LOAD_UNROLL = 4
_PASS1_HISTOGRAM_REPLICAS = 2
_KEY_BITS = 32
_LONG_RADIX_DEFAULT = ((12, 10, 10), _PASS1_HISTOGRAM_REPLICAS)
_LONG_RADIX_BY_ARCH = {
    "gfx950": ((14, 10, 8), 1),
    "gfx1250": ((13, 13, 6), 2),
}
_SHORT_RADIX_BITS = (11, 10, 11)
_SHORT_RADIX_SHIFTS = (
    _SHORT_RADIX_BITS[1] + _SHORT_RADIX_BITS[2],
    _SHORT_RADIX_BITS[2],
    0,
)
_SHORT_RADIX_MASKS = tuple((1 << bits) - 1 for bits in _SHORT_RADIX_BITS)
_SHORT_HIGH_BUCKETS = 1 << _SHORT_RADIX_BITS[0]
_MAX_ROW_ELEMENTS = ((1 << 32) - 1) // 4
_COMPACT_CAPACITY = 4096
# Keep two 1024-thread blocks resident by using half the per-CU LDS minus 2 KiB
# for padding: gfx950 has 160 KiB per CU and gfx1250 has 320 KiB. Other arches
# keep the original layout when this is 0.
_ONE_BLOCK_LDS_BUDGET_BYTES = {
    "gfx950": 78 * 1024,
    "gfx1250": 158 * 1024,
}
# A 512-thread block on these arches: two of them resident per CU, each with
# half the budget above.
_HALF_BLOCK_LDS_BUDGET_BYTES = {
    "gfx1250": 78 * 1024,
}
# Stable wave32 rows up to this many keys are read once into registers. A
# provable lower bound on the k-th largest key then leaves a few percent of
# the row as LDS candidates for the radix levels and a bitmap emission, with
# no global re-read and no per-tile block scan. Rows whose candidates overflow
# fall back to the streaming path.
_BOUNDED_ROW_CAPACITY_BY_ARCH = {
    "gfx1250": 64 * 1024,
}
# Vector steps whose keys stay in registers; longer rows reload the rest
# from L2 after the bound is known (1024-thread blocks get 128 VGPRs a wave).
_BOUNDED_REGISTER_STEPS = 8
_BOUNDED_CANDIDATE_CAPACITY = 6144
_BOUND_BIN_BITS = 10
_STABLE_INDEX_SORT_MIN_ROW_LEN = 1 << 15
_STABLE_COMPACT_SORT_MIN_ROW_LEN = 48 * 1024
# Arches whose streaming rows always take the index sort: there it beats the
# ordered re-scan of the row at every length past the cached capacity.
_STABLE_SORT_ALL_STREAMING_ROWS = ("gfx1250",)

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
def build_radix_topk_one_block_body(
    k: int,
    block_threads: int = 1024,
    write_values: bool = False,
    stable: bool = False,
    short_rows: bool = False,
    *,
    wave_size: int,
    arch: str = "",
):
    """Build the one-block selector shared by the row kernels.

    Returns (SharedStorage, run_one_block_body). The caller allocates the
    storage once per CTA and may run the body more than once over it.
    short_rows requires every effective row length <= 4096.
    wave_size must match the target architecture and is part of the cache key.
    arch selects the first radix level from _LONG_RADIX_BY_ARCH; the arches
    listed in _ONE_BLOCK_LDS_BUDGET_BYTES also grow the candidate buffer into
    whatever that budget leaves over. The rest keep the original layout.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if block_threads not in (256, 512, 1024):
        raise ValueError("block_threads must be 256, 512 or 1024")
    if wave_size not in (32, 64):
        raise ValueError("wave_size must be 32 or 64")
    if block_threads * _VEC > _PACKED_COUNT_MASK:
        raise ValueError("one scan tile exceeds the packed count range")

    num_waves = block_threads // wave_size
    long_radix_bits, pass1_replicas = _LONG_RADIX_BY_ARCH.get(arch, _LONG_RADIX_DEFAULT)
    if sum(long_radix_bits) != _KEY_BITS:
        # Level 1 masks off whatever the three levels do not cover, and the later
        # levels never see it.
        raise ValueError(f"radix levels {long_radix_bits} must cover {_KEY_BITS} bits")
    high_buckets = 1 << long_radix_bits[0]
    long_shifts = (long_radix_bits[1] + long_radix_bits[2], long_radix_bits[2], 0)
    long_masks = tuple((1 << bits) - 1 for bits in long_radix_bits)
    high_bins_per_thread = high_buckets // block_threads
    later_buckets = 1 << max(long_radix_bits[1:])
    later_bins_per_thread = later_buckets // block_threads
    short_bins_per_thread = _SHORT_HIGH_BUCKETS // block_threads
    full_key_vector_steps = (
        (_COMPACT_CAPACITY // _VEC) + block_threads - 1
    ) // block_threads
    half_block = block_threads == 512 and arch in _HALF_BLOCK_LDS_BUDGET_BYTES
    # The bounded path bounds the k-th largest key with the block's per-thread
    # top-2 keys and reduces across waves through one lane per wave. Only
    # 1024-thread blocks keep both the register and the reload keys of a row
    # within one 32-bit hit mask each.
    bounded_rows = (
        stable
        and not write_values
        and not short_rows
        and wave_size == 32
        and block_threads == 1024
        and num_waves <= wave_size
        and k <= 2 * block_threads
        and arch in _BOUNDED_ROW_CAPACITY_BY_ARCH
    )
    bounded_row_capacity = (
        _BOUNDED_ROW_CAPACITY_BY_ARCH[arch] if bounded_rows else _COMPACT_CAPACITY
    )
    bounded_key_vector_steps = (
        (bounded_row_capacity // _VEC) + block_threads - 1
    ) // block_threads
    register_key_steps = min(_BOUNDED_REGISTER_STEPS, bounded_key_vector_steps)
    reload_key_steps = bounded_key_vector_steps - register_key_steps
    register_key_capacity = register_key_steps * block_threads * _VEC
    if bounded_rows and max(register_key_steps, reload_key_steps) * _VEC > 32:
        raise ValueError("register and reload keys must each fit one 32-bit hit mask")
    # Consecutive bitmap words per thread restore column order for the emission.
    bitmap_words = max(bounded_row_capacity // 32, 1)
    words_per_thread = -(-bitmap_words // block_threads)
    bounded_candidate_capacity = _BOUNDED_CANDIDATE_CAPACITY if bounded_rows else 16
    wave_reduce_offsets = tuple(
        wave_size >> s for s in range(1, wave_size.bit_length())
    )
    stable_sort_enabled = (
        stable
        and not short_rows
        and (block_threads == 1024 or half_block)
        and k <= 2048
    )
    if arch in _STABLE_SORT_ALL_STREAMING_ROWS:
        # Rows up to the cached capacity never reach the streaming path.
        stable_sort_min_row_len = stable_compact_sort_min_row_len = 0
    else:
        # Keep the crossover for a full-row gather followed by index sorting.
        stable_sort_min_row_len = _STABLE_INDEX_SORT_MIN_ROW_LEN * (
            2 if k > 1024 else 1
        )
        # Cached output uses a fixed 48K crossover on every architecture.
        # Preserve existing configurations whose crossover is already lower.
        stable_compact_sort_min_row_len = min(
            stable_sort_min_row_len, _STABLE_COMPACT_SORT_MIN_ROW_LEN
        )
    stable_sort_capacity = 1 << (k - 1).bit_length() if stable_sort_enabled else 1
    stable_sort_items_per_thread = (
        stable_sort_capacity + block_threads - 1
    ) // block_threads
    stable_sort_sizes, stable_sort_strides = _build_bitonic_schedule(
        stable_sort_capacity
    )
    output_vector_count = k // _VEC
    output_vector_steps = (output_vector_count + block_threads - 1) // block_threads
    output_vector_elems = max(_VEC, output_vector_count * _VEC)

    # Once the selected bucket overflows the candidate buffer, level 3 and the
    # final emit re-scan the row from global instead of reading LDS, so the
    # buffer gets every byte the budget leaves over. Stable index sorting
    # reserves a power-of-two staging array (including its padding slots).
    histogram_bytes = later_buckets * 4
    stable_stage_bytes = stable_sort_capacity * 4
    scratch_bytes = (3 * num_waves + _METADATA_SIZE) * 4
    # Writer and readers of the candidate columns are all gated on this.
    candidate_columns_used = not stable or stable_sort_enabled
    candidate_entry_bytes = 4 + (4 if candidate_columns_used else 0)
    lds_budget_bytes = (
        _HALF_BLOCK_LDS_BUDGET_BYTES[arch]
        if half_block
        else _ONE_BLOCK_LDS_BUDGET_BYTES.get(arch, 0)
    )

    spare_bytes = (
        lds_budget_bytes - histogram_bytes - stable_stage_bytes - scratch_bytes
    )
    candidate_capacity = max(_COMPACT_CAPACITY, spare_bytes // candidate_entry_bytes)
    candidate_index_capacity = candidate_capacity if candidate_columns_used else 1
    # Compare bitmap word work with the compare/exchanges in the sorting
    # network. Tiny k should retain its short register sorting network.
    bitmap_capacity = min(
        candidate_capacity, stable_sort_capacity // 2 * len(stable_sort_sizes)
    )

    # Block-wide prefix sums come from the cooperative library; its shared
    # storage is one slot per wave.
    block_scan = fx.coop.BlockScan[fx.Int32, block_threads]

    # LDS layouts
    @fx.struct
    class LongPass1Storage:
        histograms: fx.Array[fx.Int32, pass1_replicas * high_buckets, 16]

    @fx.struct
    class LongLaterStorage:
        histogram: fx.Array[fx.Int32, later_buckets, 16]
        candidate_ordered_keys: fx.Array[fx.Int32, candidate_capacity, 16]
        candidate_local_indices: fx.Array[fx.Int32, candidate_index_capacity, 16]
        stable_data: fx.Array[fx.Int32, stable_sort_capacity, 16]

    @fx.struct
    class ShortPass1Storage:
        histograms: fx.Array[
            fx.Int32, _PASS1_HISTOGRAM_REPLICAS * _SHORT_HIGH_BUCKETS, 16
        ]
        full_keys: fx.Array[fx.Int32, _COMPACT_CAPACITY, 16]

    @fx.struct
    class BoundedStorage:
        histogram: fx.Array[fx.Int32, _SHORT_HIGH_BUCKETS, 16]
        above_bits: fx.Array[fx.Int32, bitmap_words, 16]
        equal_bits: fx.Array[fx.Int32, bitmap_words, 16]
        word_prefix: fx.Array[fx.Int32, bitmap_words, 16]
        candidate_keys: fx.Array[fx.Int32, bounded_candidate_capacity, 16]
        candidate_cols: fx.Array[fx.Int32, bounded_candidate_capacity, 16]

    if bounded_rows:

        @fx.union
        class ArenaStorage:
            long_pass1: LongPass1Storage
            long_later: LongLaterStorage
            short_pass1: ShortPass1Storage
            bounded: BoundedStorage

    else:

        @fx.union
        class ArenaStorage:
            long_pass1: LongPass1Storage
            long_later: LongLaterStorage
            short_pass1: ShortPass1Storage

    shared_arena_type = ShortPass1Storage if short_rows else ArenaStorage

    @fx.struct
    class SharedStorage:
        arena: shared_arena_type
        scan: block_scan.SharedStorage
        metadata: fx.Array[fx.Int32, _METADATA_SIZE, 16]
        wave_scratch: fx.Array[fx.Int32, num_waves, 16]
        wave_max: fx.Array[fx.Int32, num_waves, 16]

    @flyc.jit
    def run_one_block_body(
        storage,
        input_row_iter,
        row_len,
        row_indices,
        row_values,
        index_base,
        row_labels=None,
        write_row_values=True,
    ):
        """Select the top k of one row segment into row_indices/row_values.

        input_row_iter points at the segment's first element; indices are
        reported as index_base + column. The hooks let one part of a split row
        reuse the body: row_labels (a buffer view) replaces the reported
        indices and write_row_values gates value writes. The copy for rows no
        longer than k writes identity indices and is
        intentionally disabled for labelled merge rows, which must be longer
        than k.
        """
        tid = fx.thread_idx.x
        lane = tid % wave_size

        zero = fx.Int32(0)
        one = fx.Int32(1)
        vec_width = fx.Int32(_VEC)
        block_size = fx.Int32(block_threads)
        top_k = fx.Int32(k)
        sign_bit = fx.Int32(-2147483648)
        full_vector_count = row_len // vec_width

        # Input and output views
        input_row = fx.rocdl.make_buffer_tensor(
            fx.make_view(input_row_iter, fx.make_layout(_MAX_ROW_ELEMENTS, 1)),
            num_records_bytes=fx.Int64(row_len) * 4,
        )
        input_vector_tiles = fx.logical_divide(input_row, fx.make_layout(_VEC, 1))
        row_index_tiles = fx.logical_divide(
            fx.make_view(
                fx.get_iter(row_indices), fx.make_layout(output_vector_elems, 1)
            ),
            fx.make_layout(_VEC, 1),
        )
        row_value_tiles = fx.logical_divide(
            fx.make_view(
                fx.get_iter(row_values), fx.make_layout(output_vector_elems, 1)
            ),
            fx.make_layout(_VEC, 1),
        )

        def reported_index(col):
            if const_expr(row_labels is None):
                return index_base + col
            return row_labels[col]

        # Copy primitives
        index_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        index_fragment_layout = fx.make_layout(_VEC, 1)
        value_store_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)
        value_fragment_layout = fx.make_layout(_VEC, 1)
        full_key_load_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
        full_key_fragment_layout = fx.make_layout(_VEC, 1)

        # LDS views

        if const_expr(short_rows):
            short_storage = storage.arena
        else:
            long_histogram_matrix = storage.arena.long_pass1.histograms.peek().view(
                fx.make_layout((pass1_replicas, high_buckets), (high_buckets, 1))
            )
            long_histograms = tuple(
                fx.slice(long_histogram_matrix, (r, None))
                for r in range(pass1_replicas)
            )
            long_histograms_flat = storage.arena.long_pass1.histograms.peek().view(
                fx.make_layout(pass1_replicas * high_buckets, 1)
            )
            histogram = storage.arena.long_later.histogram.peek().view(
                fx.make_layout(later_buckets, 1)
            )
            candidate_ordered_keys = (
                storage.arena.long_later.candidate_ordered_keys.peek().view(
                    fx.make_layout(candidate_capacity, 1)
                )
            )
            candidate_local_indices = (
                storage.arena.long_later.candidate_local_indices.peek().view(
                    fx.make_layout(candidate_index_capacity, 1)
                )
            )
            staged_local_indices = storage.arena.long_later.stable_data.peek().view(
                fx.make_layout(stable_sort_capacity, 1)
            )
            short_storage = storage.arena.short_pass1
            if const_expr(bounded_rows):
                bounded_storage = storage.arena.bounded
                bounded_histogram = bounded_storage.histogram.peek().view(
                    fx.make_layout(_SHORT_HIGH_BUCKETS, 1)
                )
                candidate_keys = bounded_storage.candidate_keys.peek().view(
                    fx.make_layout(bounded_candidate_capacity, 1)
                )
                above_bits = bounded_storage.above_bits.peek().view(
                    fx.make_layout(bitmap_words, 1)
                )
                equal_bits = bounded_storage.equal_bits.peek().view(
                    fx.make_layout(bitmap_words, 1)
                )
                word_prefix = bounded_storage.word_prefix.peek().view(
                    fx.make_layout(bitmap_words, 1)
                )
                candidate_cols = bounded_storage.candidate_cols.peek().view(
                    fx.make_layout(bounded_candidate_capacity, 1)
                )

        # Short-row arena
        short_histogram_matrix = short_storage.histograms.peek().view(
            fx.make_layout(
                (_PASS1_HISTOGRAM_REPLICAS, _SHORT_HIGH_BUCKETS),
                (_SHORT_HIGH_BUCKETS, 1),
            )
        )
        short_histograms = tuple(
            fx.slice(short_histogram_matrix, (replica, None))
            for replica in range(_PASS1_HISTOGRAM_REPLICAS)
        )
        short_histograms_flat = short_storage.histograms.peek().view(
            fx.make_layout(_PASS1_HISTOGRAM_REPLICAS * _SHORT_HIGH_BUCKETS, 1)
        )
        full_keys = short_storage.full_keys.peek().view(
            fx.make_layout(_COMPACT_CAPACITY, 1)
        )
        full_key_tiles = fx.logical_divide(full_keys, fx.make_layout(_VEC, 1))

        # Shared scratch
        scan = storage.scan.peek()
        metadata = storage.metadata.peek().view(fx.make_layout(_METADATA_SIZE, 1))
        wave_scratch = storage.wave_scratch.peek().view(fx.make_layout(num_waves, 1))
        wave_max = storage.wave_max.peek().view(fx.make_layout(num_waves, 1))
        wave_id = tid // fx.Int32(wave_size)

        # Key encoding and classification
        def ordered_key(value):
            return _f32_to_ord(value) ^ sign_bit

        def radix_bucket(key, shift, mask):
            return (key >> fx.Int32(shift)) & fx.Int32(mask)

        def classify_prefix(key, radix_shifts, prefix_threshold, levels, key_mask=None):
            if key_mask is not None:
                # Only classification is coarsened; emitters retain the original
                # key for values, including when a whole prefix is selected.
                key = key & key_mask
            prefix_shift = radix_shifts[levels - 1]
            prefix_mask = (
                -1 if prefix_shift == 0 else (1 << (_KEY_BITS - prefix_shift)) - 1
            )
            prefix = radix_bucket(key, prefix_shift, prefix_mask)
            comparable_prefix = prefix ^ sign_bit if levels == 3 else prefix
            comparable_threshold = (
                prefix_threshold ^ sign_bit if levels == 3 else prefix_threshold
            )
            return (
                comparable_prefix > comparable_threshold,
                prefix == prefix_threshold,
            )

        def ordered_value(key):
            bits = (key < zero).select(key ^ sign_bit, key ^ fx.Int32(-1))
            return fx.Int32(bits).bitcast(fx.Float32)

        def write_selected_value(view, pos, col, key):
            if const_expr(write_values) and write_row_values:
                # All NaNs share key -1, so reload only NaNs to preserve their
                # sign/payload. Every normal value is rebuilt from its VGPR key.
                if key == fx.Int32(-1):
                    view[pos] = input_row[col]
                else:
                    view[pos] = ordered_value(key)

        # FlyDSL tracks indexed stores as SSA writes: store helpers take writable
        # views explicitly; orchestration helpers capture the fixed kernel views.
        def scatter_unstable_key(
            col, key, above, equal, num_needed, row_indices, row_values
        ):
            if above:
                out_pos = atomic_add_i32(metadata, one, _RUNNING_ABOVE, "workgroup")
                if out_pos < top_k:
                    row_indices[out_pos] = reported_index(col)
                    write_selected_value(row_values, out_pos, col, key)
            elif equal:
                back_pos = atomic_add_i32(metadata, one, _RUNNING_EQUAL, "workgroup")
                if back_pos < num_needed:
                    out_pos = top_k - one - back_pos
                    row_indices[out_pos] = reported_index(col)
                    write_selected_value(row_values, out_pos, col, key)

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
                for replica in range_constexpr(len(histograms)):
                    histograms[replica][pos] = zero
            gpu.barrier()

        def merge_histograms(histograms, bins_per_thread):
            # One replica is already merged; folding it into itself is not free.
            if const_expr(len(histograms) > 1):
                for item in range_constexpr(bins_per_thread):
                    pos = tid + item * block_threads
                    merged = histograms[0][pos]
                    for replica in range_constexpr(1, len(histograms)):
                        merged = merged + histograms[replica][pos]
                    histograms[0][pos] = merged
                gpu.barrier()

        def choose_threshold(
            target_k,
            above_slot,
            threshold_slot,
            histogram,
            bins_per_thread,
            metadata,
            replicas=(),
        ):
            first_bin = tid * fx.Int32(bins_per_thread)
            counts = fx.make_rmem_tensor(bins_per_thread, fx.Int32)
            local_total = zero
            for item in range_constexpr(bins_per_thread):
                count = histogram[first_bin + item]
                for replica in range_constexpr(len(replicas)):
                    count = count + replicas[replica][first_bin + item]
                counts[item] = count
                local_total = local_total + count
            exclusive, total = block_scan.exclusive_with_aggregate(
                local_total, fx.ReductionOp.ADD, storage=scan
            )
            target_prefix = total - target_k
            for item in range_constexpr(bins_per_thread):
                inclusive = exclusive + counts[item]
                if (exclusive <= target_prefix) & (inclusive > target_prefix):
                    metadata[threshold_slot] = first_bin + item  # Kth bucket
                    metadata[above_slot] = total - inclusive  # Higher-bucket count
                    metadata[_SELECTED_BUCKET_COUNT] = (
                        inclusive - exclusive
                    )  # Bucket count
                exclusive = inclusive
            gpu.barrier()

        # Global and LDS row iterators
        def scan_gm_row(visit_one, reverse=False, visit_four=None):
            """Visit per-thread work; callbacks must not use block collectives."""

            def visit_vector(col, values):
                if visit_four is not None:
                    visit_four(col, values)
                else:
                    for item in range_constexpr(_VEC):
                        visit_one(col + item, values[item])

            unroll_stride = block_size * fx.Int32(_LOAD_UNROLL)
            unroll_end = (
                full_vector_count > block_size * fx.Int32(_LOAD_UNROLL - 1)
            ).select(full_vector_count - block_size * fx.Int32(_LOAD_UNROLL - 1), zero)
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
                visit_vector(
                    vector_idx * vec_width,
                    _load_f32x4(input_vector_tiles, vector_idx),
                )

            if const_expr(not reverse):  # noqa: SIM102 - constexpr guard
                if remain_col < row_len:
                    visit_one(remain_col, input_row[remain_col])

        def scan_lds_keys(body):
            """Visit valid cached keys; callbacks may run on only part of the block."""
            row_vectors = (row_len + fx.Int32(_VEC - 1)) // fx.Int32(_VEC)
            for step in range_constexpr(full_key_vector_steps):
                vector_idx = step * block_threads + tid
                active = vector_idx < row_vectors
                safe_idx = active.select(vector_idx, zero)
                fragment = fx.make_rmem_tensor(full_key_fragment_layout, fx.Int32)
                fx.copy(
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
        def accumulate_first_histogram(
            col, key, shift, mask, flat_histograms, bins, replicas, cached_keys=None
        ):
            active = col < row_len
            if cached_keys is not None:  # noqa: SIM102 - constexpr guard
                if active:
                    cached_keys[col] = key
            if active:
                bucket = radix_bucket(key, shift, mask)
                # Lane picks the replica, so a bin holding most of the row does
                # not serialise a whole wave on one address. Indexing by wave
                # only separates waves that already ran concurrently.
                replica = lane % fx.Int32(replicas)
                atomic_add_i32(
                    flat_histograms, one, replica * fx.Int32(bins) + bucket, "workgroup"
                )

        def preceding_prefix(key, shift, mask):
            prefix_shift = shift + mask.bit_length()
            prefix_mask = (1 << (_KEY_BITS - prefix_shift)) - 1
            return radix_bucket(key, prefix_shift, prefix_mask)

        def accumulate_prefix_histogram(key, shift, mask, histogram, prefix_threshold):
            # Callers visit only columns inside the row.
            if preceding_prefix(key, shift, mask) == prefix_threshold:
                bucket = radix_bucket(key, shift, mask)
                atomic_add_i32(histogram, one, bucket, "workgroup")

        # Non-stable emitters
        def scatter_unstable_keys(
            source,
            radix_shifts,
            prefix_threshold,
            num_needed,
            levels,
            reset_above=True,
            candidate_count=None,
            candidate_ordered_keys=None,
            candidate_local_indices=None,
        ):
            reset_scatter_counters(metadata, reset_above=reset_above)

            def emit(col, key):
                above, equal = classify_prefix(
                    key, radix_shifts, prefix_threshold, levels
                )
                scatter_unstable_key(
                    col, key, above, equal, num_needed, row_indices, row_values
                )

            if source == "cached_row":
                scan_lds_keys(emit)
            elif source == "global_row":
                scan_gm_row(lambda col, value: emit(col, ordered_key(value)))
            else:
                for pos in range(tid, candidate_count, block_size):
                    emit(candidate_local_indices[pos], candidate_ordered_keys[pos])

        # Stable emitters
        def bitonic_sort_and_store_stable(
            selected_local_indices, row_indices, row_values
        ):
            local_indices = fx.make_rmem_tensor(stable_sort_items_per_thread, fx.Int32)
            for item in range_constexpr(stable_sort_items_per_thread):
                pos = tid + item * block_threads
                local_indices[item] = fx.Int32(2147483647)
                if pos < top_k:
                    local_indices[item] = selected_local_indices[pos]
            gpu.barrier()

            for stage in range_constexpr(len(stable_sort_sizes)):
                size = stable_sort_sizes[stage]
                stride = stable_sort_strides[stage]
                # Keep wave-local compare/exchanges in registers. Only a
                # cross-wave partner needs LDS and workgroup synchronization.
                if const_expr(stride >= wave_size):
                    for item in range_constexpr(stable_sort_items_per_thread):
                        pos = tid + item * block_threads
                        if pos < fx.Int32(stable_sort_capacity):
                            selected_local_indices[pos] = local_indices[item]
                    gpu.barrier()

                for item in range_constexpr(stable_sort_items_per_thread):
                    pos = tid + item * block_threads
                    left = local_indices[item]
                    right = fx.Int32(2147483647)
                    if const_expr(stride < wave_size):
                        right = left.shuffle_xor(fx.Int32(stride), fx.Int32(wave_size))
                    else:
                        if pos < fx.Int32(stable_sort_capacity):
                            partner = pos ^ fx.Int32(stride)
                            right = selected_local_indices[partner]
                    ascending = (pos & fx.Int32(size)) == zero
                    lower_half = (pos & fx.Int32(stride)) == zero
                    take_min = ascending == lower_half
                    swap = take_min.select(left > right, left < right)
                    local_indices[item] = swap.select(right, left)

                if const_expr(stride >= wave_size):
                    # Finish every LDS read before another wave can overwrite it.
                    gpu.barrier()

            for item in range_constexpr(stable_sort_items_per_thread):
                pos = tid + item * block_threads
                if pos < top_k:
                    row_indices[pos] = reported_index(local_indices[item])
                    if const_expr(write_values) and write_row_values:
                        row_values[pos] = input_row[local_indices[item]]

        def sort_and_store_stable(
            selected_local_indices, bitmap, row_indices, row_values
        ):
            # Candidate keys are dead after collect's barrier, so their LDS
            # can hold the bitmap without increasing the shared allocation.
            if row_len <= fx.Int32(bitmap_capacity * 32):
                num_words = (row_len + fx.Int32(31)) // fx.Int32(32)
                for pos in range(tid, num_words, block_size):
                    bitmap[pos] = zero
                gpu.barrier()
                for item in range_constexpr(stable_sort_items_per_thread):
                    pos = tid + item * block_threads
                    if pos < top_k:
                        col = selected_local_indices[pos]
                        bit = one << (col % fx.Int32(32))
                        atomic_or_i32(bitmap, bit, col // fx.Int32(32), "workgroup")
                gpu.barrier()

                tile_output_base = zero
                num_tiles = (num_words + block_size - one) // block_size
                for tile in range(zero, num_tiles, one):
                    word_idx = tile * block_size + tid
                    word = zero
                    if word_idx < num_words:
                        word = bitmap[word_idx]
                    count = fx.ctpop(word)
                    prefix, total = block_scan.exclusive_with_aggregate(
                        count, fx.ReductionOp.ADD, storage=scan
                    )
                    out_pos = tile_output_base + prefix
                    while word != zero:
                        bit_idx = fx.math.cttz(word)
                        col = word_idx * fx.Int32(32) + bit_idx
                        row_indices[out_pos] = reported_index(col)
                        if const_expr(write_values) and write_row_values:
                            row_values[out_pos] = input_row[col]
                        out_pos = out_pos + one
                        word = word & (word - one)
                    tile_output_base = tile_output_base + total
                    # All scan reads finish before the next tile overwrites it.
                    gpu.barrier()
            else:
                bitonic_sort_and_store_stable(
                    selected_local_indices, row_indices, row_values
                )

        def scatter_stable_sorted_keys(
            source,
            prefix_threshold,
            num_needed,
            levels,
            candidate_ordered_keys,
            candidate_local_indices,
            staged_local_indices,
            candidate_count=None,
            key_mask=None,
        ):
            # Consume the compact stage directly; global collection uses the
            # candidate index array. Both have room for bitonic padding.
            selected_local_indices = (
                staged_local_indices
                if source == "compacted_candidates"
                else candidate_local_indices
            )
            definite_expected = top_k - num_needed
            if source == "global_row":
                reset_scatter_counters(metadata)

            def collect(col, key, output_local_indices):
                above, equal = classify_prefix(
                    key, long_shifts, prefix_threshold, levels, key_mask
                )
                if source == "global_row":
                    if above:
                        pos = atomic_add_i32(metadata, one, _RUNNING_ABOVE, "workgroup")
                        if pos < definite_expected:
                            output_local_indices[pos] = col
                    elif equal:
                        tie_pos = atomic_add_i32(
                            metadata, one, _RUNNING_EQUAL, "workgroup"
                        )
                        if tie_pos < num_needed:
                            pos = definite_expected + tie_pos
                            output_local_indices[pos] = col
                elif above | equal:
                    out_pos = atomic_add_i32(metadata, one, _RUNNING_ABOVE, "workgroup")
                    if out_pos < top_k:
                        output_local_indices[out_pos] = col

            if source == "global_row":
                scan_gm_row(
                    lambda col, value: collect(
                        col, ordered_key(value), selected_local_indices
                    )
                )
            else:
                for pos in range(tid, candidate_count, block_size):
                    collect(
                        candidate_local_indices[pos],
                        candidate_ordered_keys[pos],
                        staged_local_indices,
                    )
            gpu.barrier()

            sort_and_store_stable(
                selected_local_indices, candidate_ordered_keys, row_indices, row_values
            )

        def scatter_stable_ordered_keys(
            source, radix_shifts, prefix_threshold, num_needed, levels, key_mask=None
        ):
            """Scan uniform block tiles: every thread must enter each scatter_step."""

            def scatter_step(elems, above_base, equal_base, row_indices, row_values):
                num_elems = len(elems)
                # Class encoding: 2 = above, 1 = equal, 0 = below.
                classes = fx.make_rmem_tensor(num_elems, fx.Int32)
                packed_local = 0
                for item in range_constexpr(num_elems):
                    valid, key, _ = elems[item]
                    classes[item] = 0
                    if valid:
                        above, equal = classify_prefix(
                            key, radix_shifts, prefix_threshold, levels, key_mask
                        )
                        if above:
                            classes[item] = 2
                            packed_local = packed_local + fx.Int32(
                                1 << _PACKED_COUNT_BITS
                            )
                        elif equal:
                            classes[item] = 1
                            packed_local = packed_local + 1

                packed_prefix, packed_step_total = block_scan.exclusive_with_aggregate(
                    packed_local, fx.ReductionOp.ADD, storage=scan
                )
                my_above = above_base + (packed_prefix >> _PACKED_COUNT_BITS)
                my_eq = equal_base + (packed_prefix & fx.Int32(_PACKED_COUNT_MASK))
                for item in range_constexpr(num_elems):
                    valid, key, col = elems[item]
                    cls = classes[item]
                    if valid:
                        accepted_equal = (my_eq < num_needed).select(my_eq, num_needed)
                        out_pos = my_above + accepted_equal
                        if cls == 2:
                            row_indices[out_pos] = reported_index(col)
                            write_selected_value(row_values, out_pos, col, key)
                            my_above = my_above + 1
                        elif cls == 1:
                            if my_eq < num_needed:
                                row_indices[out_pos] = reported_index(col)
                                write_selected_value(row_values, out_pos, col, key)
                            my_eq = my_eq + 1
                above_base = above_base + (packed_step_total >> _PACKED_COUNT_BITS)
                next_eq_base = equal_base + (
                    packed_step_total & fx.Int32(_PACKED_COUNT_MASK)
                )
                equal_base = (next_eq_base < num_needed).select(
                    next_eq_base, num_needed
                )
                gpu.barrier()
                return above_base, equal_base

            above_base = 0
            equal_base = 0
            if source == "cached_row":
                row_vectors = (row_len + fx.Int32(_VEC - 1)) // vec_width
                for step in range_constexpr(full_key_vector_steps):
                    vector_idx = step * block_threads + tid
                    active_vector = vector_idx < row_vectors
                    safe_vector_idx = active_vector.select(vector_idx, 0)
                    fragment = fx.make_rmem_tensor(full_key_fragment_layout, fx.Int32)
                    fx.copy(
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
            candidate_ordered_keys,
            candidate_local_indices,
            staged_local_indices,
            key_mask,
        ):
            def scatter_ordered():
                scatter_stable_ordered_keys(
                    "global_row", long_shifts, prefix_threshold, num_needed, 3, key_mask
                )

            if const_expr(stable_sort_enabled):
                can_use_index_sort = (row_len >= fx.Int32(stable_sort_min_row_len)) & (
                    metadata[_SELECTED_BUCKET_COUNT] == num_needed
                )
                can_sort_compacted_indices = (
                    (row_len >= fx.Int32(stable_compact_sort_min_row_len))
                    & (metadata[_SELECTED_BUCKET_COUNT] == num_needed)
                    & (candidate_count <= fx.Int32(candidate_capacity))
                )
                if can_sort_compacted_indices:
                    scatter_stable_sorted_keys(
                        "compacted_candidates",
                        prefix_threshold,
                        num_needed,
                        3,
                        candidate_ordered_keys,
                        candidate_local_indices,
                        staged_local_indices,
                        candidate_count=candidate_count,
                        key_mask=key_mask,
                    )
                else:
                    if can_use_index_sort:
                        scatter_stable_sorted_keys(
                            "global_row",
                            prefix_threshold,
                            num_needed,
                            3,
                            candidate_ordered_keys,
                            candidate_local_indices,
                            staged_local_indices,
                            key_mask=key_mask,
                        )
                    else:
                        scatter_ordered()
            else:
                scatter_ordered()

        def run_cached_path(histograms):
            def scatter_selection(prefix_threshold, num_needed, levels):
                if const_expr(stable):
                    scatter_stable_ordered_keys(
                        "cached_row",
                        _SHORT_RADIX_SHIFTS,
                        prefix_threshold,
                        num_needed,
                        levels,
                    )
                else:
                    scatter_unstable_keys(
                        "cached_row",
                        _SHORT_RADIX_SHIFTS,
                        prefix_threshold,
                        num_needed,
                        levels,
                    )

            # A return inside a dynamic FlyDSL if only exits its generated
            # scf.if helper. Nest the remaining stages to make the exit real.
            # First radix pass.
            clear_histograms(histograms, short_bins_per_thread)
            scan_gm_row(
                lambda col, value: accumulate_first_histogram(
                    col,
                    ordered_key(value),
                    _SHORT_RADIX_SHIFTS[0],
                    _SHORT_RADIX_MASKS[0],
                    short_histograms_flat,
                    _SHORT_HIGH_BUCKETS,
                    _PASS1_HISTOGRAM_REPLICAS,
                    cached_keys=full_keys,
                )
            )
            gpu.barrier()
            # Fusing the merge increases scan register pressure at 256 threads.
            if const_expr(block_threads == 256):
                merge_histograms(histograms, short_bins_per_thread)
            choose_threshold(
                top_k,
                _FIRST_ABOVE,
                _FIRST_THRESHOLD,
                histograms[0],
                short_bins_per_thread,
                metadata,
                replicas=histograms[1:] if block_threads != 256 else (),
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
                    lambda _col, key: accumulate_prefix_histogram(
                        key,
                        _SHORT_RADIX_SHIFTS[1],
                        _SHORT_RADIX_MASKS[1],
                        histograms[0],
                        prefix_threshold,
                    )
                )
                gpu.barrier()
                choose_threshold(
                    remaining_k,
                    _SECOND_ABOVE,
                    _SECOND_THRESHOLD,
                    histograms[0],
                    short_bins_per_thread,
                    metadata,
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
                        lambda _col, key: accumulate_prefix_histogram(
                            key,
                            _SHORT_RADIX_SHIFTS[2],
                            _SHORT_RADIX_MASKS[2],
                            histograms[0],
                            prefix_threshold,
                        )
                    )
                    gpu.barrier()
                    choose_threshold(
                        remaining_k,
                        _THIRD_ABOVE,
                        _THIRD_THRESHOLD,
                        histograms[0],
                        short_bins_per_thread,
                        metadata,
                    )

                    remaining_k = remaining_k - metadata[_THIRD_ABOVE]
                    prefix_threshold = (
                        prefix_threshold << fx.Int32(_SHORT_RADIX_BITS[2])
                    ) | metadata[_THIRD_THRESHOLD]
                    scatter_selection(prefix_threshold, remaining_k, 3)

        def run_streaming_path(
            histograms,
            histogram,
            candidate_ordered_keys,
            candidate_local_indices,
            staged_local_indices,
        ):
            def accumulate_first_vector(col, values):
                keys = tuple(ordered_key(values[item]) for item in range(_VEC))
                differing_bits = (
                    (keys[0] ^ keys[1]) | (keys[0] ^ keys[2]) | (keys[0] ^ keys[3])
                )
                # Merge only complete vectors; scan_gm_row handles the tail
                # with the scalar callback. All four keys share this digit.
                if (differing_bits >> fx.Int32(long_shifts[0])) == zero:
                    bucket = radix_bucket(keys[0], long_shifts[0], long_masks[0])
                    replica = lane % fx.Int32(pass1_replicas)
                    # This branch may contain only part of a wave. Ballot
                    # counts that active cohort, never inactive tail lanes.
                    first_bucket = fx.Int32(fx.rocdl.readfirstlane(T.i32, bucket))
                    active_mask = fx.Uint64(fx.rocdl.ballot(T.i64, fx.Boolean(True)))
                    matching = fx.Uint64(fx.rocdl.ballot(T.i64, bucket == first_bucket))
                    all_same = matching == active_mask
                    leader = fx.Int32(fx.rocdl.readfirstlane(T.i32, lane))
                    count = fx.Int32(fx.ctpop(active_mask))  # ballot mask is 64-bit
                    if (matching != active_mask) | (lane == leader):
                        atomic_add_i32(
                            long_histograms_flat,
                            all_same.select(count * fx.Int32(_VEC), fx.Int32(_VEC)),
                            replica * fx.Int32(high_buckets) + bucket,
                            "workgroup",
                        )
                else:
                    for item in range_constexpr(_VEC):
                        accumulate_first_histogram(
                            col + item,
                            keys[item],
                            long_shifts[0],
                            long_masks[0],
                            long_histograms_flat,
                            high_buckets,
                            pass1_replicas,
                        )

            def accumulate_compacted_histogram(
                col,
                key,
                prefix_threshold,
                compact_candidates,
                candidate_ordered_keys,
                candidate_local_indices,
                staged_local_indices,
                row_indices,
                row_values,
            ):
                """Collect higher keys, then histogram and compact the matching bucket."""
                shift = long_shifts[1]
                mask = long_masks[1]
                active = col < row_len
                prefix = preceding_prefix(key, shift, mask)
                collect_above = fx.Int32(int(not stable)) == one
                if const_expr(stable_sort_enabled):
                    # Retain first-level winners for the cached output path.
                    collect_above = collect_above | (
                        (row_len >= fx.Int32(stable_compact_sort_min_row_len))
                        & compact_candidates
                    )
                # Ordered streaming emission does not consume this counter or
                # the staging buffers. Avoid serializing those unused writes.
                if active & (prefix > prefix_threshold) & collect_above:
                    out_pos = atomic_add_i32(metadata, one, _RUNNING_ABOVE, "workgroup")
                    if out_pos < top_k:
                        if const_expr(stable):
                            if const_expr(stable_sort_enabled):
                                staged_local_indices[out_pos] = col
                        else:
                            row_indices[out_pos] = reported_index(col)
                            write_selected_value(row_values, out_pos, col, key)
                active = active & (prefix == prefix_threshold)
                if active:
                    bucket = radix_bucket(key, shift, mask)
                    atomic_add_i32(histogram, one, bucket, "workgroup")
                if active & compact_candidates:
                    candidate_pos = atomic_add_i32(
                        metadata, one, _CANDIDATE_COUNT, "workgroup"
                    )
                    if candidate_pos < fx.Int32(candidate_capacity):
                        candidate_ordered_keys[candidate_pos] = key
                        if const_expr(not stable or stable_sort_enabled):
                            candidate_local_indices[candidate_pos] = col

            # Level 1: select the high radix bucket.
            clear_histograms(histograms, high_bins_per_thread)
            scan_gm_row(
                lambda col, value: accumulate_first_histogram(
                    col,
                    ordered_key(value),
                    long_shifts[0],
                    long_masks[0],
                    long_histograms_flat,
                    high_buckets,
                    pass1_replicas,
                ),
                visit_four=accumulate_first_vector,
            )
            gpu.barrier()
            # Keep the smaller live range for throughput-oriented variants.
            if const_expr(not stable or k <= 1024):
                merge_histograms(histograms, high_bins_per_thread)
            choose_threshold(
                top_k,
                _FIRST_ABOVE,
                _FIRST_THRESHOLD,
                histograms[0],
                high_bins_per_thread,
                metadata,
                replicas=histograms[1:] if stable and k > 1024 else (),
            )

            remaining_k = top_k - metadata[_FIRST_ABOVE]
            prefix_threshold = metadata[_FIRST_THRESHOLD]
            # Stable emission can skip compaction when its bucket exceeds LDS
            # capacity. Keep the counter-based count for non-stable emission:
            # carrying the first-pass count across the scan hurt its throughput
            # (and also hurt stable variants with smaller k).
            if const_expr(stable and k > 1024):
                candidate_count = metadata[_SELECTED_BUCKET_COUNT]
                compact_candidates = candidate_count <= fx.Int32(candidate_capacity)
            else:
                candidate_count = zero
                compact_candidates = one > zero
            unstable_mode = fx.Int32(int(not stable)) == one
            can_finish = unstable_mode & (
                metadata[_SELECTED_BUCKET_COUNT] == remaining_k
            )
            if can_finish:
                # Fast exit after the first radix pass.
                scatter_unstable_keys(
                    "global_row", long_shifts, prefix_threshold, remaining_k, 1
                )
            else:
                # Level 2 reuses the first-pass histogram arena after choose_threshold
                # has published its metadata and synchronized the block.
                clear_histograms((histogram,), later_bins_per_thread)
                scan_gm_row(
                    lambda col, value: accumulate_compacted_histogram(
                        col,
                        ordered_key(value),
                        prefix_threshold,
                        compact_candidates,
                        candidate_ordered_keys,
                        candidate_local_indices,
                        staged_local_indices,
                        row_indices,
                        row_values,
                    ),
                    reverse=True,
                )
                gpu.barrier()
                if const_expr(not stable or k <= 1024):
                    candidate_count = metadata[_CANDIDATE_COUNT]
                choose_threshold(
                    remaining_k,
                    _SECOND_ABOVE,
                    _SECOND_THRESHOLD,
                    histogram,
                    later_bins_per_thread,
                    metadata,
                )

                remaining_k = remaining_k - metadata[_SECOND_ABOVE]
                prefix_threshold = (
                    prefix_threshold << fx.Int32(long_radix_bits[1])
                ) | metadata[_SECOND_THRESHOLD]
                final_key_mask = fx.Int32(-1)
                can_finish = metadata[_SELECTED_BUCKET_COUNT] == remaining_k
                if const_expr(not stable):
                    can_finish = can_finish & (
                        candidate_count <= fx.Int32(candidate_capacity)
                    )
                if can_finish:
                    if const_expr(stable):
                        # Compare the selected prefix in the common final emitter.
                        # Keeping one emitter avoids duplicating its scans/sort and
                        # extending scalar live ranges across separate exit paths.
                        prefix_threshold = prefix_threshold << fx.Int32(
                            long_radix_bits[2]
                        )
                        final_key_mask = fx.Int32(-1 << long_radix_bits[2])
                    else:
                        scatter_unstable_keys(
                            "compacted_candidates",
                            long_shifts,
                            prefix_threshold,
                            remaining_k,
                            2,
                            reset_above=False,
                            candidate_count=candidate_count,
                            candidate_ordered_keys=candidate_ordered_keys,
                            candidate_local_indices=candidate_local_indices,
                        )
                else:
                    # Level 3: select the low bucket and emit the final result.
                    clear_histograms((histogram,), later_bins_per_thread)
                    if candidate_count <= fx.Int32(candidate_capacity):
                        for pos in range(tid, candidate_count, block_size):
                            accumulate_prefix_histogram(
                                candidate_ordered_keys[pos],
                                long_shifts[2],
                                long_masks[2],
                                histogram,
                                prefix_threshold,
                            )
                    else:
                        scan_gm_row(
                            lambda _col, value: accumulate_prefix_histogram(
                                ordered_key(value),
                                long_shifts[2],
                                long_masks[2],
                                histogram,
                                prefix_threshold,
                            )
                        )
                    gpu.barrier()
                    choose_threshold(
                        remaining_k,
                        _THIRD_ABOVE,
                        _THIRD_THRESHOLD,
                        histogram,
                        later_bins_per_thread,
                        metadata,
                    )

                    # All radix passes are complete; write the result.
                    remaining_k = remaining_k - metadata[_THIRD_ABOVE]
                    prefix_threshold = (
                        prefix_threshold << fx.Int32(long_radix_bits[2])
                    ) | metadata[_THIRD_THRESHOLD]
                    if const_expr(not stable):
                        if candidate_count <= fx.Int32(candidate_capacity):
                            scatter_unstable_keys(
                                "compacted_candidates",
                                long_shifts,
                                prefix_threshold,
                                remaining_k,
                                3,
                                reset_above=False,
                                candidate_count=candidate_count,
                                candidate_ordered_keys=candidate_ordered_keys,
                                candidate_local_indices=candidate_local_indices,
                            )
                        else:
                            scatter_unstable_keys(
                                "global_row",
                                long_shifts,
                                prefix_threshold,
                                remaining_k,
                                3,
                            )

                if const_expr(stable):
                    scatter_streaming_stable(
                        candidate_count,
                        prefix_threshold,
                        remaining_k,
                        candidate_ordered_keys,
                        candidate_local_indices,
                        staged_local_indices,
                        final_key_mask,
                    )

        # Bounded rows (wave32, stable): see _BOUNDED_ROW_CAPACITY_BY_ARCH.
        def ballot_mask(pred):
            return fx.Int32(fx.rocdl.ballot(T.i32, pred))

        def popcount(mask):
            return fx.Int32(fx.ctpop(mask))

        def lane_xor(value, offset):
            """Value of lane ^ offset: DPP inside a 16-lane row, LDS swizzle across rows."""
            if const_expr(offset == 8):
                out = update_dpp_i32(value, value, 280, 0xF, 0xC, False)
                return fx.Int32(update_dpp_i32(out, value, 264, 0xF, 0x3, False))
            if const_expr(offset == 4):
                out = update_dpp_i32(value, value, 276, 0xF, 0xA, False)
                return fx.Int32(update_dpp_i32(out, value, 260, 0xF, 0x5, False))
            if const_expr(offset == 2):
                return fx.Int32(update_dpp_i32(value, value, 78, 0xF, 0xF, False))
            if const_expr(offset == 1):
                return fx.Int32(update_dpp_i32(value, value, 177, 0xF, 0xF, False))
            return value.shuffle_xor(fx.Int32(offset), fx.Int32(wave_size))

        def store_if(view, pos, value, pred):
            if pred:
                view[pos] = value

        def emit_selected(row_indices, row_values, out_pos, col, pred):
            if pred:
                row_indices[out_pos] = reported_index(col)

        def set_bit_if(bitmap, col, pred):
            if pred:
                atomic_or_i32(
                    bitmap,
                    one << (col % fx.Int32(32)),
                    col // fx.Int32(32),
                    "workgroup",
                )

        def reserve_slots(count):
            """Lane 0 reserves `count` candidate slots for the wave; all lanes get the base."""
            base = zero
            if lane == zero:
                base = atomic_add_i32(metadata, count, _CANDIDATE_COUNT, "workgroup")
            return fx.Int32(fx.rocdl.readfirstlane(T.i32, base))

        def block_exclusive_sum(value):
            """Exclusive block prefix sum and total.

            DPP scans inside each wave; after one barrier every lane loads one
            wave total and a second DPP scan yields both results, instead of the
            generic block scan's per-thread walk over every wave slot.
            """
            inclusive = _warp_inclusive_prefix_i32(value, lane, wave_size)
            store_if(wave_scratch, wave_id, inclusive, lane == fx.Int32(wave_size - 1))
            gpu.barrier()
            valid = lane < fx.Int32(num_waves)
            totals = valid.select(wave_scratch[valid.select(lane, zero)], zero)
            wave_inclusive = _warp_inclusive_prefix_i32(totals, lane, wave_size)
            total = fx.Int32(fx.rocdl.readlane(T.i32, wave_inclusive, wave_size - 1))
            uniform_wave = fx.Int32(fx.rocdl.readfirstlane(T.i32, wave_id))
            previous_lane = (uniform_wave > zero).select(uniform_wave - one, zero)
            before = fx.Int32(fx.rocdl.readlane(T.i32, wave_inclusive, previous_lane))
            before = (uniform_wave > zero).select(before, zero)
            return before + inclusive - value, total

        def publish_threshold(
            view, pred, threshold_slot, bucket, above_slot, above, count
        ):
            if pred:
                view[threshold_slot] = bucket
                view[above_slot] = above
                view[_SELECTED_BUCKET_COUNT] = count

        def choose_threshold_and_clear(target_k, above_slot, threshold_slot, hist):
            """choose_threshold on one histogram; each thread zeroes the bins it read."""
            first_bin = tid * fx.Int32(short_bins_per_thread)
            counts = []
            local_total = zero
            for item in range_constexpr(short_bins_per_thread):
                count = hist[first_bin + item]
                counts.append(count)
                local_total = local_total + count
            for item in range_constexpr(short_bins_per_thread):
                hist[first_bin + item] = zero
            exclusive, total = block_exclusive_sum(local_total)
            target_prefix = total - target_k
            for item in range_constexpr(short_bins_per_thread):
                inclusive = exclusive + counts[item]
                publish_threshold(
                    metadata,
                    (exclusive <= target_prefix) & (inclusive > target_prefix),
                    threshold_slot,
                    first_bin + item,
                    above_slot,
                    total - inclusive,
                    inclusive - exclusive,
                )
                exclusive = inclusive
            gpu.barrier()

        def load_key_step(step):
            vector_idx = step * block_threads + tid
            active = vector_idx < full_vector_count
            values = _load_f32x4(input_vector_tiles, active.select(vector_idx, zero))
            return (
                vector_idx,
                active,
                [_f32_to_ord(values[i]) for i in range_constexpr(_VEC)],
            )

        def hit_mask(steps, first_step, bound):
            """One bit per vector item of `steps` whose key is at or above `bound`."""
            bits = zero
            for index in range_constexpr(len(steps)):
                _, active, ords = steps[index]
                for item in range_constexpr(_VEC):
                    bit = index * _VEC + item
                    bits = bits | (active & (ords[item] >= bound)).select(
                        fx.Int32((1 << bit) if bit < 31 else -(1 << 31)), zero
                    )
            return bits

        def store_hit_columns(hit_bits, first_step, rank):
            vector_base = first_step * block_threads * _VEC + tid * vec_width
            while hit_bits != zero:
                bit = fx.math.cttz(hit_bits)
                col = (
                    (bit >> fx.Int32(2)) * fx.Int32(block_threads * _VEC)
                    + vector_base
                    + (bit & fx.Int32(_VEC - 1))
                )
                store_if(
                    candidate_cols,
                    rank,
                    col,
                    rank < fx.Int32(bounded_candidate_capacity),
                )
                rank = rank + one
                hit_bits = hit_bits & (hit_bits - one)
            return rank

        def load_row_keys():
            """Ordered keys of this thread's vectors and tail, plus its two largest."""
            first = fx.Int32(-2147483648)
            second = fx.Int32(-2147483648)

            def keep_top2(first, second, o, valid):
                o = valid.select(o, fx.Int32(-2147483648))
                second = (o > second).select((o > first).select(first, o), second)
                first = (o > first).select(o, first)
                return first, second

            vectors = []
            for step in range_constexpr(register_key_steps):
                vector_idx, active, ords = load_key_step(step)
                for o in ords:
                    first, second = keep_top2(first, second, o, active)
                vectors.append((vector_idx, active, ords))
            if const_expr(reload_key_steps > 0):  # noqa: SIM102
                if row_len > fx.Int32(register_key_capacity):
                    for step in range_constexpr(
                        register_key_steps, bounded_key_vector_steps
                    ):
                        _, active, ords = load_key_step(step)
                        for o in ords:
                            first, second = keep_top2(first, second, o, active)
            tail_col = full_vector_count * vec_width + tid
            tail_valid = tail_col < row_len
            tail_ord = _f32_to_ord(input_row[tail_valid.select(tail_col, zero)])
            first, second = keep_top2(first, second, tail_ord, tail_valid)
            return vectors, (tail_col, tail_valid, tail_ord), first, second

        def kth_largest_lower_bound(first, second):
            """Ordinal at or below the row's k-th largest, and the row maximum.

            The block's 2 * block_threads per-thread top-2 keys are distinct
            elements. Histogramming them over [min, max] and taking the lower
            edge of the bin where the count from the top reaches k leaves at
            least k elements at or above the edge. Rows on this path give every
            thread at least two elements.
            """
            low = second
            high = first
            for offset in range_constexpr(len(wave_reduce_offsets)):
                other = lane_xor(low, wave_reduce_offsets[offset])
                low = (other < low).select(other, low)
                other = lane_xor(high, wave_reduce_offsets[offset])
                high = (other > high).select(other, high)
            store_if(wave_scratch, wave_id, low, lane == zero)
            store_if(wave_max, wave_id, high, lane == zero)
            gpu.barrier()
            slot = (lane < fx.Int32(num_waves)).select(lane, zero)
            low = wave_scratch[slot]
            high = wave_max[slot]
            for offset in range_constexpr(len(wave_reduce_offsets)):
                other = lane_xor(low, wave_reduce_offsets[offset])
                low = (other < low).select(other, low)
                other = lane_xor(high, wave_reduce_offsets[offset])
                high = (other > high).select(other, high)
            bits = fx.Int32(32) - fx.Int32(fx.math.ctlz(high - low))
            bin_bits = fx.Int32(_BOUND_BIN_BITS)
            shift = (bits > bin_bits).select(bits - bin_bits, zero)
            bin_mask = fx.Int32((1 << _BOUND_BIN_BITS) - 1)
            for o in (first, second):
                atomic_add_i32(
                    bounded_histogram,
                    one,
                    ((o - low) >> shift) & bin_mask,
                    "workgroup",
                )
            gpu.barrier()
            choose_threshold_and_clear(
                top_k, _FIRST_ABOVE, _FIRST_THRESHOLD, bounded_histogram
            )
            return low + (metadata[_FIRST_THRESHOLD] << shift), high

        def compact_candidates(vectors, tail, bound):
            """Append the column of every key at or above `bound`.

            Hits become one bit per vector item, so the sparse stores loop over
            set bits only; a DPP prefix and one atomic per wave place them.
            """
            hit_bits = hit_mask(vectors, 0, bound)
            reload_bits = zero
            if const_expr(reload_key_steps > 0):  # noqa: SIM102
                if row_len > fx.Int32(register_key_capacity):
                    reload_bits = hit_mask(
                        [
                            load_key_step(step)
                            for step in range_constexpr(
                                register_key_steps, bounded_key_vector_steps
                            )
                        ],
                        register_key_steps,
                        bound,
                    )
            tail_col, tail_valid, tail_ord = tail
            tail_hit = tail_valid & (tail_ord >= bound)
            count = (
                popcount(hit_bits) + popcount(reload_bits) + tail_hit.select(one, zero)
            )
            inclusive = _warp_inclusive_prefix_i32(count, lane, wave_size)
            total = fx.Int32(fx.rocdl.readlane(T.i32, inclusive, wave_size - 1))
            rank = reserve_slots(total) + inclusive - count
            store_if(
                candidate_cols,
                rank,
                tail_col,
                tail_hit & (rank < fx.Int32(bounded_candidate_capacity)),
            )
            rank = rank + tail_hit.select(one, zero)
            rank = store_hit_columns(hit_bits, 0, rank)
            if const_expr(reload_key_steps > 0):
                store_hit_columns(reload_bits, register_key_steps, rank)

        def fetch_candidates(count, bound, shift, width):
            """Reload candidate keys as d = ordinal - bound and count the level-1 digit."""
            for pos in range(tid, count, block_size):
                d = _f32_to_ord(input_row[candidate_cols[pos]]) - bound
                candidate_keys[pos] = d
                bucket = (d >> shift) & ((one << width) - one)
                atomic_add_i32(bounded_histogram, one, bucket, "workgroup")
            gpu.barrier()

        def visit_candidates(count, body):
            for pos in range(tid, count, block_size):
                body(candidate_cols[pos], candidate_keys[pos])

        def emit_selected_candidates(count, shift, width, prefix_threshold, num_needed):
            """Ordered emission: per-word prefix counts, then each candidate places itself.

            One block scan over the bitmap words publishes how many above and
            equal keys precede every word; a candidate adds the lower bits of
            its own word, so a densely selected word costs no serial bit loop.
            """
            num_words = (row_len + fx.Int32(31)) // fx.Int32(32)
            words = []
            local = zero
            for item in range_constexpr(words_per_thread):
                word = tid * fx.Int32(words_per_thread) + fx.Int32(item)
                valid = word < num_words
                safe_word = valid.select(word, zero)
                packed = (
                    popcount(valid.select(above_bits[safe_word], zero))
                    << fx.Int32(_PACKED_COUNT_BITS)
                ) + popcount(valid.select(equal_bits[safe_word], zero))
                words.append((word, valid, packed))
                local = local + packed
            prefix, _ = block_exclusive_sum(local)
            for word, valid, packed in words:
                store_if(word_prefix, word, prefix, valid)
                prefix = prefix + packed
            gpu.barrier()

            def place(col, d):
                p = digit_prefix(d, shift, width)
                above = unsigned_gt(p, prefix_threshold)
                equal = p == prefix_threshold
                word = col >> fx.Int32(5)
                lower = (one << (col & fx.Int32(31))) - one
                base = word_prefix[word]
                my_above = (base >> fx.Int32(_PACKED_COUNT_BITS)) + popcount(
                    above_bits[word] & lower
                )
                my_equal = (base & fx.Int32(_PACKED_COUNT_MASK)) + popcount(
                    equal_bits[word] & lower
                )
                accepted_equal = (my_equal < num_needed).select(my_equal, num_needed)
                emit_selected(
                    row_indices,
                    row_values,
                    my_above + accepted_equal,
                    col,
                    above | (equal & (my_equal < num_needed)),
                )

            visit_candidates(count, place)

        def unsigned_gt(a, b):
            return (a ^ sign_bit) > (b ^ sign_bit)

        def digit_prefix(d, shift, width):
            """Bits of d at and above `shift`; `width` = bit length minus shift."""
            masked = (d >> shift) & ((one << width) - one)
            return (shift == zero).select(d, masked)

        def finish_candidates(count, shift, width, prefix_threshold, num_needed):
            def mark(col, d):
                p = digit_prefix(d, shift, width)
                set_bit_if(above_bits, col, unsigned_gt(p, prefix_threshold))
                set_bit_if(equal_bits, col, p == prefix_threshold)

            visit_candidates(count, mark)
            gpu.barrier()
            emit_selected_candidates(count, shift, width, prefix_threshold, num_needed)

        def histogram_digits(count, shift, digit_bits, match_shift, match_width, match):
            def add(_col, d):
                selected = digit_prefix(d, match_shift, match_width) == match
                if selected:
                    bucket = (d >> shift) & ((one << digit_bits) - one)
                    atomic_add_i32(bounded_histogram, one, bucket, "workgroup")

            visit_candidates(count, add)
            gpu.barrier()

        def select_from_candidates(count, bound, row_range):
            bits = fx.Int32(32) - fx.Int32(fx.math.ctlz(row_range))
            radix = fx.Int32(_SHORT_RADIX_BITS[0])
            shift1 = (bits > radix).select(bits - radix, zero)
            shift2 = (shift1 > radix).select(shift1 - radix, zero)
            width2 = shift1 - shift2
            # Level 1 histograms the top digit of every candidate.
            fetch_candidates(count, bound, shift1, bits - shift1)
            choose_threshold_and_clear(
                top_k, _FIRST_ABOVE, _FIRST_THRESHOLD, bounded_histogram
            )
            remaining_k = top_k - metadata[_FIRST_ABOVE]
            prefix_threshold = metadata[_FIRST_THRESHOLD]
            if (metadata[_SELECTED_BUCKET_COUNT] == remaining_k) | (shift1 == zero):
                finish_candidates(
                    count, shift1, bits - shift1, prefix_threshold, remaining_k
                )
            else:
                histogram_digits(
                    count, shift2, width2, shift1, bits - shift1, prefix_threshold
                )
                choose_threshold_and_clear(
                    remaining_k, _SECOND_ABOVE, _SECOND_THRESHOLD, bounded_histogram
                )
                remaining_k = remaining_k - metadata[_SECOND_ABOVE]
                prefix_threshold = (prefix_threshold << width2) | metadata[
                    _SECOND_THRESHOLD
                ]
                if (metadata[_SELECTED_BUCKET_COUNT] == remaining_k) | (shift2 == zero):
                    finish_candidates(
                        count, shift2, bits - shift2, prefix_threshold, remaining_k
                    )
                else:
                    histogram_digits(
                        count, zero, shift2, shift2, bits - shift2, prefix_threshold
                    )
                    choose_threshold_and_clear(
                        remaining_k, _THIRD_ABOVE, _THIRD_THRESHOLD, bounded_histogram
                    )
                    remaining_k = remaining_k - metadata[_THIRD_ABOVE]
                    prefix_threshold = (prefix_threshold << shift2) | metadata[
                        _THIRD_THRESHOLD
                    ]
                    finish_candidates(count, zero, bits, prefix_threshold, remaining_k)

        def reset_metadata(metadata):
            gpu.barrier()
            if tid < _METADATA_SIZE:
                metadata[tid] = zero
            gpu.barrier()

        def run_bounded_path():
            clear_histograms((bounded_histogram,), short_bins_per_thread)
            for item in range_constexpr(-(-bitmap_words // block_threads)):
                pos = tid + item * block_threads
                store_if(above_bits, pos, zero, pos < fx.Int32(bitmap_words))
                store_if(equal_bits, pos, zero, pos < fx.Int32(bitmap_words))
            vectors, tail, first, second = load_row_keys()
            bound, row_max = kth_largest_lower_bound(first, second)
            compact_candidates(vectors, tail, bound)
            gpu.barrier()
            count = metadata[_CANDIDATE_COUNT]
            if count <= fx.Int32(bounded_candidate_capacity):
                select_from_candidates(count, bound, row_max - bound)
            else:
                reset_metadata(metadata)
                run_streaming_path(
                    long_histograms,
                    histogram,
                    candidate_ordered_keys,
                    candidate_local_indices,
                    staged_local_indices,
                )

        def write_direct_scalar(row_indices, row_values):
            # One column per thread: only reached when k <= block_threads.
            col = tid
            if col < k:
                row_indices[col] = (col < row_len).select(
                    index_base + col, fx.Int32(-1)
                )
                if const_expr(write_values) and write_row_values:
                    value = fx.Float32(float("-inf"))
                    if col < row_len:
                        value = input_row[col]
                    row_values[col] = value

        def write_direct_vector(row_indices, row_values):
            for step in range_constexpr(output_vector_steps):
                vector_idx = step * block_threads + tid
                if vector_idx < output_vector_count:
                    col = vector_idx * vec_width
                    fragment = fx.make_rmem_tensor(index_fragment_layout, fx.Int32)
                    if vector_idx < full_vector_count:
                        index_values = [
                            index_base + col + item for item in range_constexpr(_VEC)
                        ]
                        fragment.store(
                            fx.Vector.from_elements(index_values, dtype=fx.Int32)
                        )
                    else:
                        if col < row_len:
                            index_values = [
                                (col + item < row_len).select(
                                    index_base + col + item, fx.Int32(-1)
                                )
                                for item in range_constexpr(_VEC)
                            ]
                            fragment.store(
                                fx.Vector.from_elements(index_values, dtype=fx.Int32)
                            )
                        else:
                            fragment.store(fx.full(_VEC, -1, fx.Int32))
                    fx.copy(
                        index_store_atom,
                        fragment,
                        fx.slice(row_index_tiles, (None, vector_idx)),
                    )

                    if const_expr(write_values) and write_row_values:
                        value_fragment = fx.make_rmem_tensor(
                            value_fragment_layout, fx.Float32
                        )
                        if vector_idx < full_vector_count:
                            value_fragment.store(
                                _load_f32x4(input_vector_tiles, vector_idx)
                            )
                        else:
                            if col < row_len:
                                output_values = []
                                for item in range_constexpr(_VEC):
                                    local_col = col + item
                                    valid = local_col < row_len
                                    output_value = fx.Float32(float("-inf"))
                                    if valid:
                                        output_value = input_row[local_col]
                                    output_values.append(output_value)
                                value_fragment.store(
                                    fx.Vector.from_elements(
                                        output_values, dtype=fx.Float32
                                    )
                                )
                            else:
                                value_fragment.store(
                                    fx.full(_VEC, float("-inf"), fx.Float32)
                                )
                        fx.copy(
                            value_store_atom,
                            value_fragment,
                            fx.slice(row_value_tiles, (None, vector_idx)),
                        )

            tail = output_vector_count * _VEC + tid
            if tail < k:
                valid = tail < row_len
                safe_tail = valid.select(tail, zero)
                row_indices[tail] = valid.select(index_base + tail, fx.Int32(-1))
                if const_expr(write_values) and write_row_values:
                    row_values[tail] = valid.select(
                        input_row[safe_tail], fx.Float32(float("-inf"))
                    )

        def write_direct_output(row_indices, row_values):
            if const_expr(k <= block_threads):
                write_direct_scalar(row_indices, row_values)
            else:
                write_direct_vector(row_indices, row_values)

        # Body control flow
        if const_expr(row_labels is None) and row_len <= top_k:
            write_direct_output(row_indices, row_values)

        if row_len > top_k:
            if tid < _METADATA_SIZE:
                metadata[tid] = zero
            gpu.barrier()

            if const_expr(short_rows):
                run_cached_path(short_histograms)
            else:
                if row_len <= fx.Int32(_COMPACT_CAPACITY):
                    run_cached_path(short_histograms)
                elif const_expr(bounded_rows) and row_len <= fx.Int32(
                    bounded_row_capacity
                ):
                    run_bounded_path()
                else:
                    run_streaming_path(
                        long_histograms,
                        histogram,
                        candidate_ordered_keys,
                        candidate_local_indices,
                        staged_local_indices,
                    )

    return SharedStorage, run_one_block_body


@cache
def build_radix_topk_one_block_module(
    k: int,
    block_threads: int = 1024,
    write_values: bool = False,
    stable: bool = False,
    short_rows: bool = False,
    is_decode: bool = False,
    *,
    wave_size: int,
    arch: str = "",
    packed_rows: bool = False,
    full_rows: bool = False,
):
    """Build a prefill/decode kernel specialized for the row-length bounds.

    packed_rows (decode only): row r is read from element row_starts[r] of
    the input's first row, while reported indices stay relative to that start.

    full_rows (decode only): every row is `width` long and row_ends is not read.

    See build_radix_topk_one_block_body for the selector parameters.
    """
    if full_rows and not is_decode:
        raise ValueError("full_rows is a decode-mode build; prefill reads both ends")
    shared_storage, run_one_block_body = build_radix_topk_one_block_body(
        k,
        block_threads,
        write_values,
        stable,
        short_rows,
        wave_size=wave_size,
        arch=arch,
    )
    row_variant = "short" if short_rows else "mixed"

    @flyc.kernel(
        name=(
            f"radix_topk_one_block_{'decode' if is_decode else 'prefill'}"
            f"_{row_variant}_k{k}_b{block_threads}_w{wave_size}"
            f"_v{int(write_values)}_s{int(stable)}"
            f"{'_packed' if packed_rows else ''}"
            f"{'_full' if full_rows else ''}"
        ),
        known_block_size=[block_threads, 1, 1],
    )
    def radix_topk_one_block_kernel(
        input: fx.Tensor,
        row_starts: fx.Tensor,
        row_ends: fx.Tensor,
        indices: fx.Tensor,
        value_output: fx.Tensor,
        width: fx.Int32,
        next_n: fx.Int32,
    ):
        row = fx.Int32(fx.block_idx.x)
        if const_expr(is_decode):
            row_start = fx.Int32(0)
            row_end = _row_length(row, row_ends, width, next_n, full_rows)
        else:
            row_start = row_starts[row]
            row_end = row_ends[row]
        read_row = fx.Int32(0) if packed_rows else row
        read_start = row_starts[row] if packed_rows else row_start
        physical_row = fx.slice(input, (read_row, None))
        run_one_block_body(
            fx.SharedAllocator().allocate(shared_storage),
            fx.add_offset(fx.get_iter(physical_row), read_start),
            row_end - row_start,
            fx.slice(indices, (row, None)),
            fx.slice(value_output, (row, None)),
            row_start,
        )

    @flyc.jit
    def launch_radix_topk_one_block(
        input: fx.Tensor,
        row_starts: fx.Tensor,
        row_ends: fx.Tensor,
        indices: fx.Tensor,
        values: fx.Tensor,
        width: fx.Int32,
        next_n: fx.Int32,
        rows_m: fx.Int32,
        stream: fx.Stream,
    ):
        radix_topk_one_block_kernel(
            input, row_starts, row_ends, indices, values, width, next_n
        ).launch(grid=(rows_m, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch_radix_topk_one_block
