# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Wave32/wave64 radix TopK with one block per input row.

Short rows cache ordered keys in LDS. Long rows compact the selected radix
bucket; stable modes preserve index ordering and tie-breaking.
Rows with length <= k copy/pad in the same kernel without running radix.
On gfx1250, long rows first select from the keys above a bound sampled from
the row, read once, and persistent blocks load each next row into an LDS row
cache while the current one is selected.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.typing import T

from aiter.jit.utils.chip_info import get_cu_num

from ..dpp_utils import update_dpp_i32
from ..kernels_common import atomic_add_i32, atomic_or_i32
from ..tdm_ops_gfx1250 import (
    TDMDescriptor2D,
    make_tensor_descriptor_2d,
    tensor_load_2d,
    tensor_wait,
)
from ..tensor_shim import buf_scalar_load
from .topk_common import _f32_to_ord, _load_f32x4, _row_length

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
# A 512-thread block without the row cache on these arches: two of them
# resident per CU, each with half the budget above. With the cache, a
# 512-thread block takes the budget above minus the cache instead.
_HALF_BLOCK_LDS_BUDGET_BYTES = {
    "gfx1250": 78 * 1024,
}
_STABLE_INDEX_SORT_MIN_ROW_LEN = 1 << 15
_STABLE_COMPACT_SORT_MIN_ROW_LEN = 48 * 1024
# Arches whose streaming rows always take the index sort: there it beats the
# ordered re-scan of the row at every length past the cached capacity.
_STABLE_SORT_ALL_STREAMING_ROWS = ("gfx1250",)
# Arches whose long rows first try a sampled bound. A sampled row takes one
# strided sample key per _SAMPLED_BOUND_KEYS_PER_SAMPLE keys, from
# _SAMPLED_BOUND_MIN_SAMPLES to _SAMPLED_BOUND_MAX_SAMPLES a thread, so the
# margin's share of the candidates stays put as rows grow. The bound is the
# level-1 bucket of the sample key whose rank is the kth key's expected sample
# count plus _SAMPLED_BOUND_SIGMAS of its standard deviations, plus the margin.
# Every key in or above that bucket is a candidate, compacted in one read of the
# row. Fewer than k candidates, or more than the buffer holds, fall back to the
# two-pass selection, so the sample decides only the speed. Rows shorter than
# _SAMPLED_BOUND_MIN_ROWS_PER_K * k keep the two-pass selection.
_SAMPLED_BOUND_ARCHES = ("gfx1250",)
_SAMPLED_BOUND_MIN_SAMPLES = 1
_SAMPLED_BOUND_MAX_SAMPLES = 8
_SAMPLED_BOUND_KEYS_PER_SAMPLE = 64
_SAMPLED_BOUND_SIGMAS = 4.0
_SAMPLED_BOUND_MARGIN = 4
_SAMPLED_BOUND_MIN_ROWS_PER_K = 4
_SAMPLED_SCAN_UNROLL = 4
# The level-1 buckets from a sampled row's bound that each lane counts in its
# own bytes, four a word, as it builds the level-1 histogram.
_SAMPLED_HOT_BUCKETS = 8
# Sampled rows read their first columns from an LDS row cache, its size by
# block size. Each block walks rows grid-strided, and the tensor data mover
# fills the cache with the block's next row while the current one is selected,
# so a cached row's scan waits on LDS instead of global memory; a longer row
# reads its head from the cache and the rest from global. With the cache a
# 1024-thread block takes all of a gfx1250 CU's LDS, as its VGPRs already take
# the CU; two 512-thread blocks share a CU, the cache out of each one's half,
# so one block's barriers and LDS waits overlap the other's work.
_ROW_CACHE_ELEMENTS = {1024: 40 << 10, 512: 16 << 10}
_LONG_ROW_512_THREAD_MAX_K = 512

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
_DPP_ROW_SHR = 0x110
# Bitmap words a thread reads per tile of the stable emit, a block apart: one
# tile covers a 64K-column row at 1024 threads. Their bit counts share one
# block scan in 16-bit fields, which a block's word of 32 columns a thread
# keeps within. The stable collect reads as many candidates a thread at once.
_BITMAP_WORDS_PER_THREAD = 2


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


def _stable_sort_enabled(k, block_threads, stable, short_rows, arch):
    # The sampled bound's arches also sort at 512 threads, two blocks to a CU.
    min_threads = 512 if arch in _SAMPLED_BOUND_ARCHES else 1024
    return stable and not short_rows and block_threads >= min_threads and k <= 2048


def _sampled_bound_enabled(k, block_threads, stable, short_rows, arch):
    # The bound builds its level-1 histogram in the later levels' one, and a
    # stable row leaves through the candidate index sort.
    long_radix_bits, _ = _LONG_RADIX_BY_ARCH.get(arch, _LONG_RADIX_DEFAULT)
    return (
        arch in _SAMPLED_BOUND_ARCHES
        and not short_rows
        and block_threads >= 512
        and max(long_radix_bits[1:]) >= long_radix_bits[0]
        and (
            _stable_sort_enabled(k, block_threads, stable, short_rows, arch)
            or not stable
        )
    )


def _row_cache_enabled(k, block_threads, stable, short_rows, arch):
    return block_threads in _ROW_CACHE_ELEMENTS and _sampled_bound_enabled(
        k, block_threads, stable, short_rows, arch
    )


def _row_cache_blocks_per_cu(block_threads):
    return 1024 // block_threads


def _sample_geometry(tid, row_len, block_threads):
    """A sampled row's samples a thread, its sample count, and a thread's sample
    columns, strided through the row."""
    thread_sample_keys = _SAMPLED_BOUND_KEYS_PER_SAMPLE * block_threads
    samples_per_thread = (row_len + fx.Int32(thread_sample_keys - 1)) // fx.Int32(
        thread_sample_keys
    )
    samples_per_thread = (
        samples_per_thread < fx.Int32(_SAMPLED_BOUND_MIN_SAMPLES)
    ).select(fx.Int32(_SAMPLED_BOUND_MIN_SAMPLES), samples_per_thread)
    samples_per_thread = (
        samples_per_thread > fx.Int32(_SAMPLED_BOUND_MAX_SAMPLES)
    ).select(fx.Int32(_SAMPLED_BOUND_MAX_SAMPLES), samples_per_thread)
    sample_count = samples_per_thread * fx.Int32(block_threads)
    sample_stride = row_len // sample_count
    positions = []
    for item in range(_SAMPLED_BOUND_MAX_SAMPLES):
        pos = (tid + fx.Int32(item * block_threads)) * sample_stride
        if item >= _SAMPLED_BOUND_MIN_SAMPLES:
            # A sample the row does not take reads past its end: from a buffer,
            # that loads nothing.
            pos = (fx.Int32(item) < samples_per_thread).select(pos, row_len)
        positions.append(pos)
    return samples_per_thread, sample_count, positions


def _long_row_block_threads(k, stable, arch, num_rows):
    # Where the row cache serves 512-thread blocks, two of them share a CU and
    # overlap each other's barriers and LDS waits. Past _LONG_ROW_512_THREAD_MAX_K
    # a sampled bound's candidates overrun the buffer such a block leaves beside
    # its cache, and long rows fall back. With no more rows than CUs, a row has
    # a CU to itself, and 1024 threads take all of its waves.
    if (
        k <= _LONG_ROW_512_THREAD_MAX_K
        and _row_cache_enabled(k, 512, stable, False, arch)
        and num_rows > get_cu_num()
    ):
        return 512
    return 1024


class _ScalarReads:
    """tensor[i] as a scalar load, for a wave-uniform i."""

    def __init__(self, tensor):
        self.tensor = fx.rocdl.make_buffer_tensor(tensor)

    def __getitem__(self, index):
        return buf_scalar_load(self.tensor, index)


def _with_unit_dim1_stride(desc):
    # A one-row tile with a zero tensor_dim1_stride (GROUP1 s6[31:16]) hangs
    # the data mover, and the 2-D builder leaves it zero.
    group1 = fx.Vector(desc.dgroup1)
    words = [fx.Int32(group1[i]) for i in range(8)]
    words[6] = fx.Int32(1 << 16)
    return TDMDescriptor2D(
        dgroup0=desc.dgroup0,
        dgroup1=fx.Vector.from_elements(words, fx.Int32).ir_value(),
    )


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
    row_cache: bool = False,
):
    """Build the one-block selector shared by the row kernels.

    Returns (SharedStorage, run_one_block_body). The caller allocates the
    storage once per CTA and may run the body more than once over it.
    short_rows requires every effective row length <= 4096.
    wave_size must match the target architecture and is part of the cache key.
    arch selects the first radix level from _LONG_RADIX_BY_ARCH; the arches
    listed in _ONE_BLOCK_LDS_BUDGET_BYTES also grow the candidate buffer into
    whatever that budget leaves over. The rest keep the original layout.
    row_cache adds the LDS row cache where _row_cache_enabled, for a caller
    that passes run_one_block_body each next row.
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
    # A 512-thread block without the row cache takes the half-block budget; with
    # it, the cache comes out of the one-block budget below.
    half_block = block_threads == 512 and arch in _HALF_BLOCK_LDS_BUDGET_BYTES
    stable_sort_enabled = _stable_sort_enabled(
        k, block_threads, stable, short_rows, arch
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
    scratch_bytes = (num_waves + _METADATA_SIZE) * 4
    # Writer and readers of the candidate columns are all gated on this.
    candidate_columns_used = not stable or stable_sort_enabled
    candidate_entry_bytes = 4 + (4 if candidate_columns_used else 0)
    sampled_bound_enabled = _sampled_bound_enabled(
        k, block_threads, stable, short_rows, arch
    )
    row_cache_enabled = row_cache and _row_cache_enabled(
        k, block_threads, stable, short_rows, arch
    )
    row_cache_elements = _ROW_CACHE_ELEMENTS.get(block_threads, 0)
    row_cache_slice = row_cache_elements // num_waves
    lds_budget_bytes = (
        _HALF_BLOCK_LDS_BUDGET_BYTES[arch]
        if half_block and not row_cache_enabled
        else _ONE_BLOCK_LDS_BUDGET_BYTES.get(arch, 0)
    )
    if row_cache_enabled and _row_cache_blocks_per_cu(block_threads) > 1:
        lds_budget_bytes -= row_cache_elements * 4

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
    # Exactly the rows the sampled path takes.
    row_cache_min_row_len = max(
        _SAMPLED_BOUND_MIN_ROWS_PER_K * k, _COMPACT_CAPACITY + 1, k + 1
    )

    # Block-wide prefix sums keep one slot per wave. The cooperative library's
    # wave32 scan shuffles through LDS and every thread then folds all the
    # slots, so the sampled path's scans use DPP and fold the slots in one wave
    # scan.
    block_scan = fx.coop.BlockScan[fx.Int32, block_threads]
    dpp_scan = sampled_bound_enabled
    scan_storage = (
        fx.Array[fx.Int32, num_waves, 16] if dpp_scan else block_scan.SharedStorage
    )
    if dpp_scan and wave_size != 32:
        raise ValueError("the DPP scans take wave32")
    if sampled_bound_enabled and _SAMPLED_HOT_BUCKETS % 4:
        raise ValueError("lane counts take the hot buckets four to a word")
    # A lane counts every block_threads-th candidate, and a byte holds each count.
    if sampled_bound_enabled and candidate_capacity > 255 * block_threads:
        raise ValueError("a lane's hot bucket counts overflow their bytes")

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

    @fx.union
    class ArenaStorage:
        long_pass1: LongPass1Storage
        long_later: LongLaterStorage
        short_pass1: ShortPass1Storage

    shared_arena_type = ShortPass1Storage if short_rows else ArenaStorage

    if row_cache_enabled:

        @fx.struct
        class SharedStorage:
            arena: shared_arena_type
            scan: scan_storage
            metadata: fx.Array[fx.Int32, _METADATA_SIZE, 16]
            row_cache: fx.Array[fx.Float32, row_cache_elements, 16]

    else:

        @fx.struct
        class SharedStorage:
            arena: shared_arena_type
            scan: scan_storage
            metadata: fx.Array[fx.Int32, _METADATA_SIZE, 16]

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
        next_row_iter=None,
        next_row_len=None,
        fill_row=None,
    ):
        """Select the top k of one row segment into row_indices/row_values.

        input_row_iter points at the segment's first element; indices are
        reported as index_base + column. The hooks let one part of a split row
        reuse the body: row_labels (a buffer view) replaces the reported
        indices and write_row_values gates value writes. The copy for rows no
        longer than k writes identity indices and is
        intentionally disabled for labelled merge rows, which must be longer
        than k.

        A caller walking rows over one storage passes the block's next row
        (next_row_len 0 after the last) and fill_row, false exactly when the
        previous call was passed this row as its next. With a row cache, the
        body then reads the row from the cache and loads the next row into it.
        """
        tid = fx.thread_idx.x
        use_row_cache = row_cache_enabled and next_row_len is not None
        if const_expr(use_row_cache):
            # An empty asm ties tid to the row, as no FlyDSL op hides a value
            # from LLVM: hoisted out of the caller's row loop, values derived
            # from tid stay live across all of it and spill.
            tid = fx.Int32(
                llvm.inline_asm(
                    T.i32, [tid.ir_value(), row_len.ir_value()], "", "=v,0,v"
                )
            ) & fx.Int32(block_threads - 1)
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
        scan_slots = scan.view(fx.make_layout(num_waves, 1)) if dpp_scan else None
        metadata = storage.metadata.peek().view(fx.make_layout(_METADATA_SIZE, 1))

        # Row cache
        if const_expr(use_row_cache):
            row_cache = storage.row_cache.peek().view(
                fx.make_layout(row_cache_elements, 1)
            )
            row_cache_tiles = fx.logical_divide(row_cache, fx.make_layout(_VEC, 1))
            row_cache_load_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Float32)

        def fill_row_cache(row_iter, length):
            """Start the tensor data mover on a row's first row_cache_elements
            columns, a slice per wave; columns past the row are not read and land
            as zeros. Each wave's tensor_wait and then a barrier publish them."""
            first = fx.Int32(fx.rocdl.wave_id()) * fx.Int32(row_cache_slice)
            valid = length - first
            if valid > zero:
                slice_layout = fx.make_layout(
                    (1, row_cache_slice), (row_cache_slice, 1)
                )
                desc = make_tensor_descriptor_2d(
                    global_ptr=fx.Tensor(
                        fx.make_view(fx.add_offset(row_iter, first), slice_layout)
                    ),
                    lds_memref=fx.Tensor(
                        fx.make_view(
                            fx.add_offset(fx.get_iter(row_cache), first), slice_layout
                        )
                    ),
                    global_offset=(0, 0),
                    tensor_shape=(1, row_cache_slice),
                    strides=(row_cache_slice, 1),
                    tile_shape=(1, row_cache_slice),
                    elem_bytes=4,
                    oob_inner_bound=(valid < fx.Int32(row_cache_slice)).select(
                        valid, fx.Int32(row_cache_slice)
                    ),
                )
                tensor_load_2d(_with_unit_dim1_stride(desc))

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

        def reset_metadata(metadata):
            if tid < _METADATA_SIZE:
                metadata[tid] = zero
            gpu.barrier()

        # Histogram and workgroup scan primitives
        def wave_inclusive_sum(value):
            # DPP shifts stay inside 16-lane rows; lane 15's sum then crosses
            # into the upper row.
            for shift in range_constexpr(4):
                value = value + fx.Int32(
                    update_dpp_i32(zero, value, _DPP_ROW_SHR | (1 << shift))
                )
            row_total = fx.coop.warp_broadcast(value, 15)
            return value + (lane >= fx.Int32(16)).select(row_total, zero)

        def wave_total(inclusive):
            return fx.coop.warp_broadcast(inclusive, wave_size - 1)

        def block_exclusive_sum(value, slots=scan_slots):
            """Each thread's exclusive prefix of value, and the block's total.
            Reads the scan slots after its barrier: another barrier must pass
            before the next call writes them."""
            if const_expr(not dpp_scan):
                return block_scan.exclusive_with_aggregate(
                    value, fx.ReductionOp.ADD, storage=scan
                )
            value = fx.Int32(value)
            wave = fx.Int32(fx.rocdl.wave_id())
            inclusive = wave_inclusive_sum(value)
            if lane == fx.Int32(wave_size - 1):
                slots[wave] = inclusive
            gpu.barrier()
            has_slot = lane < fx.Int32(num_waves)
            slot = has_slot.select(slots[has_slot.select(lane, zero)], zero)
            slot_inclusive = wave_inclusive_sum(slot)
            # warp_broadcast from a runtime lane would shuffle through LDS.
            wave_prefix = fx.Int32(
                fx.rocdl.readlane(T.i32, slot_inclusive - slot, wave)
            )
            total = fx.coop.warp_broadcast(slot_inclusive, num_waves - 1)
            return wave_prefix + inclusive - value, total

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

        def sampled_group_flip(bucket):
            return (
                (bucket >> fx.Int32(6)) & fx.Int32(later_bins_per_thread // 4 - 1)
            ) << fx.Int32(2)

        def choose_threshold(
            target_k,
            above_slot,
            threshold_slot,
            histogram,
            bins_per_thread,
            metadata,
            replicas=(),
            clear=False,
            xor_groups=False,
        ):
            # A thread's bins are consecutive buckets. With xor_groups, bucket b
            # sits at sampled_slot(b); a thread's bins lie in one 64-bucket span,
            # so they share a flip.
            first_bin = tid * fx.Int32(bins_per_thread)
            if const_expr(xor_groups):
                group_flip = sampled_group_flip(first_bin)
            counts = fx.make_rmem_tensor(bins_per_thread, fx.Int32)
            local_total = zero
            for item in range_constexpr(bins_per_thread):
                slot = first_bin + item
                if const_expr(xor_groups):
                    slot = (
                        first_bin
                        + (fx.Int32(item & ~3) ^ group_flip)
                        + fx.Int32(item & 3)
                    )
                count = histogram[slot]
                for replica in range_constexpr(len(replicas)):
                    count = count + replicas[replica][slot]
                counts[item] = count
                local_total = local_total + count
                if const_expr(clear):
                    # The closing barrier orders this before the next pass.
                    histogram[slot] = zero
                    for replica in range_constexpr(len(replicas)):
                        replicas[replica][slot] = zero
            exclusive, total = block_exclusive_sum(local_total)
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

        def step_elements(vector_indices, vectors, vector_count):
            elements = []
            for vector_idx, values in zip(vector_indices, vectors):
                valid = vector_idx < vector_count
                for item in range_constexpr(_VEC):
                    elements.append(
                        (valid, vector_idx * vec_width + item, values[item])
                    )
            return elements

        def scan_row_cache(visit_step, context, vector_count):
            # Only after the barrier that publishes the cache. In steps like
            # scan_gm_row_in_steps: unrolled whole, every load stays live.
            step_stride = block_size * fx.Int32(_SAMPLED_SCAN_UNROLL)
            for first in range(zero, vector_count, step_stride):
                vector_indices = []
                vectors = []
                for item in range_constexpr(_SAMPLED_SCAN_UNROLL):
                    vector_idx = first + tid + fx.Int32(item * block_threads)
                    fragment = fx.make_rmem_tensor(value_fragment_layout, fx.Float32)
                    fx.copy(
                        row_cache_load_atom,
                        fx.slice(
                            row_cache_tiles,
                            (
                                None,
                                (vector_idx < vector_count).select(vector_idx, zero),
                            ),
                        ),
                        fragment,
                    )
                    vector_indices.append(vector_idx)
                    vectors.append(fragment.load())
                visit_step(
                    step_elements(vector_indices, vectors, vector_count), context
                )

        def scan_cached_row(visit_step, prologue):
            """scan_gm_row_in_steps for a row the row cache holds whole."""
            context = prologue()
            scan_row_cache(visit_step, context, full_vector_count)
            remain_col = full_vector_count * vec_width + tid
            remains = remain_col < row_len
            remain_value = row_cache[remains.select(remain_col, zero)]
            visit_step([(remains, remain_col, remain_value)], context)

        def scan_gm_row_in_steps(visit_step, prologue=None, cached_vectors=None):
            """Visit the row in steps every thread enters: visit_step gets the
            step's (valid, column, value) triples and may use wave collectives,
            not block ones.

            Each step issues all of its _SAMPLED_SCAN_UNROLL loads before visiting
            any, so a row costs ceil(vectors / unroll) load latencies per thread
            where scan_gm_row's one-vector cleanup pays one per leftover vector,
            and a step's loads are issued before the previous step is visited, so
            two steps stay in flight. A step's loads past the row reread vector 0
            and are not valid. prologue runs block-wide while the first step's
            loads are in flight, and every visit gets its result. When the row
            cache holds the first cached_vectors vectors, the steps start past
            them and the cache is visited after the first step, with the second
            step in flight.
            """
            step_stride = block_size * fx.Int32(_SAMPLED_SCAN_UNROLL)
            first_vector = zero if cached_vectors is None else cached_vectors

            def load_vectors(first):
                # Past the row, these reread vector 0 and are not valid.
                vectors = []
                for item in range_constexpr(_SAMPLED_SCAN_UNROLL):
                    vector_idx = first + tid + fx.Int32(item * block_threads)
                    safe_idx = (vector_idx < full_vector_count).select(vector_idx, zero)
                    vectors.append(_load_f32x4(input_vector_tiles, safe_idx))
                return vectors

            def loaded_step(first, vectors):
                vector_indices = [
                    first + tid + fx.Int32(item * block_threads)
                    for item in range_constexpr(_SAMPLED_SCAN_UNROLL)
                ]
                return step_elements(vector_indices, vectors, full_vector_count)

            first_elements = loaded_step(first_vector, load_vectors(first_vector))
            remain_col = full_vector_count * vec_width + tid
            remains = remain_col < row_len
            remain_value = input_row[remains.select(remain_col, zero)]
            context = None
            if const_expr(prologue is not None):
                context = prologue()
            next_first = first_vector + step_stride
            vectors = load_vectors(next_first)
            visit_step(first_elements, context)
            if const_expr(cached_vectors is not None):
                scan_row_cache(visit_step, context, cached_vectors)
            for first in range(next_first, full_vector_count, step_stride):
                next_vectors = load_vectors(first + step_stride)
                visit_step(loaded_step(first, vectors), context)
                vectors = next_vectors
            visit_step([(remains, remain_col, remain_value)], context)

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

        def accumulate_prefix_histogram(
            key, shift, mask, histogram, prefix_threshold, slot_of=lambda b: b
        ):
            # Callers visit only columns inside the row.
            if preceding_prefix(key, shift, mask) == prefix_threshold:
                bucket = radix_bucket(key, shift, mask)
                atomic_add_i32(histogram, one, slot_of(bucket), "workgroup")

        def sampled_slot(bucket):
            # A thread's four-bucket groups are reordered by its place among the
            # 64-bucket spans: a wave's loads of one group cover all banks, while
            # adjacent buckets, where the atomics crowd, keep adjacent words.
            return bucket ^ sampled_group_flip(bucket)

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
                    prefix, total = block_exclusive_sum(count)
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

        def store_selected_candidates(
            prefix_threshold,
            num_needed,
            candidate_count,
            bitmap,
            candidate_ordered_keys,
            candidate_local_indices,
            staged_local_indices,
            row_indices,
            row_values,
        ):
            """Stable emit of a row that selects every candidate at or above
            prefix_threshold. The cleared histogram holds a bitmap of their
            columns, set straight from the candidates and left cleared."""
            if row_len <= fx.Int32(later_buckets * 32):
                # A candidate's two reads leave ahead of its compare.
                collect_stride = fx.Int32(_BITMAP_WORDS_PER_THREAD * block_threads)
                for first in range(zero, candidate_count, collect_stride):
                    candidates = []
                    for item in range_constexpr(_BITMAP_WORDS_PER_THREAD):
                        pos = first + tid + fx.Int32(item * block_threads)
                        valid = pos < candidate_count
                        pos = valid.select(pos, zero)
                        candidates.append(
                            (
                                valid,
                                candidate_ordered_keys[pos],
                                candidate_local_indices[pos],
                            )
                        )
                    for valid, key, col in candidates:
                        above, equal = classify_prefix(
                            key, long_shifts, prefix_threshold, 3
                        )
                        if valid & (above | equal):
                            atomic_or_i32(
                                bitmap,
                                one << (col & fx.Int32(31)),
                                col >> fx.Int32(5),
                                "workgroup",
                            )
                gpu.barrier()

                num_words = (row_len + fx.Int32(31)) >> fx.Int32(5)
                output_base = zero
                for tile in range(zero, num_words, collect_stride):
                    if tile > zero:
                        # The previous tile's scan reads finish before this
                        # one writes the slots.
                        gpu.barrier()
                    words = []
                    packed_count = zero
                    for item in range_constexpr(_BITMAP_WORDS_PER_THREAD):
                        word_idx = tile + tid + fx.Int32(item * block_threads)
                        word = zero
                        if word_idx < num_words:
                            word = bitmap[word_idx]
                            bitmap[word_idx] = zero
                        words.append((word_idx, word))
                        packed_count = packed_count + (
                            fx.ctpop(word) << fx.Int32(item * 16)
                        )
                    packed_prefix, packed_total = block_exclusive_sum(packed_count)
                    slice_base = output_base
                    for item, (word_idx, word) in enumerate(words):
                        out_pos = slice_base + (
                            (packed_prefix >> fx.Int32(item * 16)) & fx.Int32(0xFFFF)
                        )
                        slice_base = slice_base + (
                            (packed_total >> fx.Int32(item * 16)) & fx.Int32(0xFFFF)
                        )
                        while word != zero:
                            col = (word_idx << fx.Int32(5)) + fx.math.cttz(word)
                            row_indices[out_pos] = reported_index(col)
                            if const_expr(write_values) and write_row_values:
                                row_values[out_pos] = input_row[col]
                            out_pos = out_pos + one
                            word = word & (word - one)
                    output_base = slice_base
            else:
                scatter_stable_sorted_keys(
                    "compacted_candidates",
                    prefix_threshold,
                    num_needed,
                    3,
                    candidate_ordered_keys,
                    candidate_local_indices,
                    staged_local_indices,
                    candidate_count=candidate_count,
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

                packed_prefix, packed_step_total = block_exclusive_sum(packed_local)
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

        def select_from_candidates(
            candidate_count,
            histogram,
            candidate_ordered_keys,
            candidate_local_indices,
            staged_local_indices,
        ):
            """Finish a row whose candidates hold all of its top k, their level-1
            histogram already built."""

            def refine(level, remaining_k, prefix_threshold):
                # The previous choose_threshold left the histogram cleared, and
                # so does this one: the stable emit sets its bitmap there.
                above_slot = _SECOND_ABOVE if level == 1 else _THIRD_ABOVE
                threshold_slot = _SECOND_THRESHOLD if level == 1 else _THIRD_THRESHOLD
                for pos in range(tid, candidate_count, block_size):
                    accumulate_prefix_histogram(
                        candidate_ordered_keys[pos],
                        long_shifts[level],
                        long_masks[level],
                        histogram,
                        prefix_threshold,
                        slot_of=sampled_slot,
                    )
                gpu.barrier()
                choose_threshold(
                    remaining_k,
                    above_slot,
                    threshold_slot,
                    histogram,
                    later_bins_per_thread,
                    metadata,
                    clear=True,
                    xor_groups=True,
                )
                remaining_k = remaining_k - metadata[above_slot]
                prefix_threshold = (
                    prefix_threshold << fx.Int32(long_radix_bits[level])
                ) | metadata[threshold_slot]
                return remaining_k, prefix_threshold

            def count_candidate(key, bound_bucket, hot, histogram):
                # The hot buckets are counted in a lane's bytes, four a word: their
                # atomics would queue on a concentrated row's few words.
                bucket = radix_bucket(key, long_shifts[0], long_masks[0])
                offset = bucket - bound_bucket
                counted = []
                in_word = one << ((offset & fx.Int32(3)) << fx.Int32(3))
                for word in range_constexpr(len(hot)):
                    in_bucket = (offset >> fx.Int32(2)) == fx.Int32(word)
                    counted.append(hot[word] + in_bucket.select(in_word, zero))
                if offset >= fx.Int32(_SAMPLED_HOT_BUCKETS):
                    atomic_add_i32(histogram, one, sampled_slot(bucket), "workgroup")
                return counted

            def add_lane_counts(bound_bucket, hot, histogram):
                # A byte's wave sum can pass 255: a word's even and odd bytes are
                # summed apart, as two 16-bit counts each.
                totals = []
                for word in range_constexpr(len(hot)):
                    for half in range_constexpr(2):
                        pair = (hot[word] >> fx.Int32(8 * half)) & fx.Int32(0x00FF00FF)
                        total = wave_total(wave_inclusive_sum(pair))
                        for byte in range_constexpr(2):
                            totals.append(
                                (
                                    4 * word + half + 2 * byte,
                                    (total >> fx.Int32(16 * byte)) & fx.Int32(0xFFFF),
                                )
                            )
                if lane == zero:
                    for item, count in totals:
                        if count > zero:
                            atomic_add_i32(
                                histogram,
                                count,
                                sampled_slot(bound_bucket + fx.Int32(item)),
                                "workgroup",
                            )

            def build_level_1_histogram(
                bound_bucket, candidate_ordered_keys, histogram
            ):
                # From the compacted candidates after the scan: during it, an atomic
                # a candidate queues on the few words a concentrated row fills.
                hot = [zero for _ in range_constexpr(_SAMPLED_HOT_BUCKETS // 4)]
                for pos in range(tid, candidate_count, block_size):
                    hot = count_candidate(
                        candidate_ordered_keys[pos], bound_bucket, hot, histogram
                    )
                add_lane_counts(bound_bucket, hot, histogram)
                gpu.barrier()

            # The bound's bucket, still in _FIRST_THRESHOLD, is the candidates'
            # lowest.
            build_level_1_histogram(
                metadata[_FIRST_THRESHOLD], candidate_ordered_keys, histogram
            )
            choose_threshold(
                top_k,
                _FIRST_ABOVE,
                _FIRST_THRESHOLD,
                histogram,
                later_bins_per_thread,
                metadata,
                clear=True,
                xor_groups=True,
            )
            remaining_k, prefix_threshold = refine(
                1, top_k - metadata[_FIRST_ABOVE], metadata[_FIRST_THRESHOLD]
            )
            if metadata[_SELECTED_BUCKET_COUNT] == remaining_k:
                # The kth key's level-2 bucket is selected whole, so its lowest
                # possible key bounds the selection: both emitters below then
                # take every candidate at or above it.
                prefix_threshold = prefix_threshold << fx.Int32(long_radix_bits[2])
            else:
                remaining_k, prefix_threshold = refine(2, remaining_k, prefix_threshold)

            if const_expr(stable):
                # Ties split by the kth key need the lowest indices: an ordered
                # read of the row finds them.
                if metadata[_SELECTED_BUCKET_COUNT] == remaining_k:
                    store_selected_candidates(
                        prefix_threshold,
                        remaining_k,
                        candidate_count,
                        histogram,
                        candidate_ordered_keys,
                        candidate_local_indices,
                        staged_local_indices,
                        row_indices,
                        row_values,
                    )
                else:
                    scatter_stable_ordered_keys(
                        "global_row", long_shifts, prefix_threshold, remaining_k, 3
                    )
            else:
                scatter_unstable_keys(
                    "compacted_candidates",
                    long_shifts,
                    prefix_threshold,
                    remaining_k,
                    3,
                    candidate_count=candidate_count,
                    candidate_ordered_keys=candidate_ordered_keys,
                    candidate_local_indices=candidate_local_indices,
                )

        def run_sampled_path(
            histogram,
            candidate_ordered_keys,
            candidate_local_indices,
            staged_local_indices,
        ):
            """Select from the keys at or above a sampled level-1 bound, reading the
            row once. Returns whether the row still needs the two-pass selection;
            the metadata is then reset for it."""

            samples_per_thread, sample_count, sample_positions = _sample_geometry(
                tid, row_len, block_threads
            )
            cached_sample_items = len(sample_positions)
            # A cached row is short enough for _SAMPLED_BOUND_MIN_SAMPLES.
            thread_sample_keys = _SAMPLED_BOUND_KEYS_PER_SAMPLE * block_threads
            if const_expr(
                thread_sample_keys * _SAMPLED_BOUND_MIN_SAMPLES >= row_cache_elements
            ):
                cached_sample_items = _SAMPLED_BOUND_MIN_SAMPLES

            def count_sample(sample_key):
                atomic_add_i32(
                    histogram,
                    one,
                    sampled_slot(
                        radix_bucket(sample_key, long_shifts[0], long_masks[0])
                    ),
                    "workgroup",
                )

            def sample_bound(samples):
                # The kth largest key is below the sample's target-th largest
                # unless that many sample keys exceed it. The sample keys above the
                # kth key count near a Poisson draw, so the target is their
                # expected count plus _SAMPLED_BOUND_SIGMAS standard deviations
                # and the margin, and a miss only leaves fewer than k candidates.
                # A sample read from global is loaded ahead of the scan's first
                # step: the wait for it leaves the step in flight.
                sample_keys = [ordered_key(sample) for sample in samples]
                clear_histograms((histogram,), later_bins_per_thread)
                for item in range_constexpr(len(sample_keys)):
                    if const_expr(item < _SAMPLED_BOUND_MIN_SAMPLES):
                        count_sample(sample_keys[item])
                    else:
                        if fx.Int32(item) < samples_per_thread:
                            count_sample(sample_keys[item])
                gpu.barrier()
                expected = (fx.Int32(k) * sample_count).to(fx.Float32) / row_len.to(
                    fx.Float32
                )
                sample_target = (
                    expected + fx.Float32(_SAMPLED_BOUND_SIGMAS) * fx.sqrt(expected)
                ).to(fx.Int32) + fx.Int32(1 + _SAMPLED_BOUND_MARGIN)
                sample_target = (sample_target > sample_count).select(
                    sample_count, sample_target
                )
                choose_threshold(
                    sample_target,
                    _FIRST_ABOVE,
                    _FIRST_THRESHOLD,
                    histogram,
                    later_bins_per_thread,
                    metadata,
                    clear=True,
                    xor_groups=True,
                )
                return metadata[_FIRST_THRESHOLD]

            def collect_above_bound(
                elements, bound, candidate_ordered_keys, candidate_local_indices
            ):
                # A wave reserves the step's candidates with one returning
                # atomic: one per key would wait on LDS for nearly every key.
                keys = []
                taken = []
                count = zero
                for valid, _, value in elements:
                    key = ordered_key(value)
                    take = valid & (
                        radix_bucket(key, long_shifts[0], long_masks[0]) >= bound
                    )
                    count = count + take.select(one, zero)
                    keys.append(key)
                    taken.append(take)
                inclusive = wave_inclusive_sum(count)
                wave_count = wave_total(inclusive)
                wave_first = zero
                if (lane == fx.Int32(wave_size - 1)) & (wave_count > zero):
                    wave_first = atomic_add_i32(
                        metadata, wave_count, _CANDIDATE_COUNT, "workgroup"
                    )
                candidate_pos = wave_total(wave_first) + inclusive - count
                for (_, col, _), key, take in zip(elements, keys, taken):
                    if take:  # noqa: SIM102 - preserve FlyDSL branches
                        if candidate_pos < fx.Int32(candidate_capacity):
                            candidate_ordered_keys[candidate_pos] = key
                            candidate_local_indices[candidate_pos] = col
                    candidate_pos = candidate_pos + take.select(one, zero)

            def visit(elements, bound_bucket):
                # A runtime if stores only to buffers local to its function.
                collect_above_bound(
                    elements,
                    bound_bucket,
                    candidate_ordered_keys,
                    candidate_local_indices,
                )

            def load_spilled_sample(pos):
                # Both loads issue: a cached key's global one reads past the row's
                # end, which loads nothing.
                cached = pos < fx.Int32(row_cache_elements)
                return cached.select(
                    row_cache[cached.select(pos, zero)],
                    input_row[cached.select(row_len, pos)],
                )

            def scan_global_row():
                samples = [input_row[pos] for pos in sample_positions]
                scan_gm_row_in_steps(visit, prologue=lambda: sample_bound(samples))

            if const_expr(use_row_cache):
                if row_len > fx.Int32(row_cache_elements):
                    # The head is in the cache.
                    samples = [load_spilled_sample(pos) for pos in sample_positions]
                    scan_gm_row_in_steps(
                        visit,
                        prologue=lambda: sample_bound(samples),
                        cached_vectors=fx.Int32(row_cache_elements // _VEC),
                    )
                else:
                    scan_cached_row(
                        visit,
                        lambda: sample_bound(
                            [
                                row_cache[pos]
                                for pos in sample_positions[:cached_sample_items]
                            ]
                        ),
                    )
            else:
                scan_global_row()
            gpu.barrier()
            if const_expr(use_row_cache):  # noqa: SIM102 - constexpr guard
                # Every wave is past the cache: the next row lands in it while
                # this one is selected. A row that skipped the cache started
                # that on entry.
                if row_cached & next_cached:
                    fill_row_cache(next_row_iter, next_row_len)
            candidate_count = metadata[_CANDIDATE_COUNT]
            fallback = (candidate_count < top_k) | (
                candidate_count > fx.Int32(candidate_capacity)
            )
            if fallback:
                reset_metadata(metadata)
            else:
                select_from_candidates(
                    candidate_count,
                    histogram,
                    candidate_ordered_keys,
                    candidate_local_indices,
                    staged_local_indices,
                )
            return fallback

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
        if const_expr(use_row_cache):
            # A row the sampled path takes fills the cache, unless the previous
            # row started that; any other row lends the cache to the next one.

            def takes_cache(length):
                return length >= fx.Int32(row_cache_min_row_len)

            row_cached = takes_cache(row_len)
            next_cached = takes_cache(next_row_len)
            if row_cached:
                if fill_row:
                    fill_row_cache(input_row_iter, row_len)
            else:
                if next_cached:
                    fill_row_cache(next_row_iter, next_row_len)

        if const_expr(row_labels is None) and row_len <= top_k:
            write_direct_output(row_indices, row_values)

        if row_len > top_k:
            if tid < _METADATA_SIZE:
                metadata[tid] = zero
            if const_expr(use_row_cache):  # noqa: SIM102 - constexpr guard
                if row_cached:
                    tensor_wait(0)
            gpu.barrier()

            if const_expr(short_rows):
                run_cached_path(short_histograms)
            else:
                if row_len <= fx.Int32(_COMPACT_CAPACITY):
                    run_cached_path(short_histograms)
                else:
                    if const_expr(sampled_bound_enabled):
                        # One streaming copy serves short rows and fallbacks.
                        use_streaming = one > zero
                        if row_len >= fx.Int32(_SAMPLED_BOUND_MIN_ROWS_PER_K * k):
                            use_streaming = run_sampled_path(
                                histogram,
                                candidate_ordered_keys,
                                candidate_local_indices,
                                staged_local_indices,
                            )
                        if use_streaming:
                            run_streaming_path(
                                long_histograms,
                                histogram,
                                candidate_ordered_keys,
                                candidate_local_indices,
                                staged_local_indices,
                            )
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
    # A block walks rows so it can load its next row into the row cache, as many
    # blocks per CU as the cache leaves LDS for.
    persistent = _row_cache_enabled(k, block_threads, stable, short_rows, arch)
    persistent_blocks = (
        get_cu_num() * _row_cache_blocks_per_cu(block_threads) if persistent else 0
    )
    shared_storage, run_one_block_body = build_radix_topk_one_block_body(
        k,
        block_threads,
        write_values,
        stable,
        short_rows,
        wave_size=wave_size,
        arch=arch,
        row_cache=persistent,
    )
    row_variant = "short" if short_rows else "mixed"
    kernel_name = (
        f"radix_topk_one_block_{'decode' if is_decode else 'prefill'}"
        f"_{row_variant}_k{k}_b{block_threads}_w{wave_size}"
        f"_v{int(write_values)}_s{int(stable)}"
        f"{'_packed' if packed_rows else ''}"
        f"{'_full' if full_rows else ''}"
    )

    if persistent:

        @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
        def radix_topk_one_block_kernel(
            input: fx.Tensor,
            row_starts: fx.Tensor,
            row_ends: fx.Tensor,
            indices: fx.Tensor,
            value_output: fx.Tensor,
            width: fx.Int32,
            next_n: fx.Int32,
            rows_m: fx.Int32,
            grid_rows: fx.Int32,
        ):
            def row_bounds(row, starts, ends):
                # The row's length, the column of its start, and where its first
                # element sits in the row it is read from.
                if const_expr(is_decode):
                    row_start = fx.Int32(0)
                    row_end = _row_length(row, ends, width, next_n, full_rows)
                else:
                    row_start = starts[row]
                    row_end = ends[row]
                read_start = starts[row] if packed_rows else row_start
                return row_end - row_start, row_start, read_start

            def row_first(row, read_start):
                read_row = fx.Int32(0) if packed_rows else row
                physical_row = fx.slice(input, (read_row, None))
                return fx.add_offset(fx.get_iter(physical_row), read_start)

            storage = fx.SharedAllocator().allocate(shared_storage)
            first_row = fx.Int32(fx.block_idx.x)
            # The bounds are read as scalars a row ahead of their use: past the
            # first stores the compiler no longer reads them with scalar loads,
            # and the vector round trips would open every row.
            starts, ends = _ScalarReads(row_starts), _ScalarReads(row_ends)
            next_row = first_row + grid_rows
            has_next = next_row < rows_m
            row_len, row_start, read_start = row_bounds(first_row, starts, ends)
            next_len, next_start, next_read = row_bounds(
                has_next.select(next_row, first_row), starts, ends
            )
            next_len = has_next.select(next_len, fx.Int32(0))
            for row in range(first_row, rows_m, grid_rows):
                next_row = row + grid_rows
                after_row = next_row + grid_rows
                has_after = after_row < rows_m
                after_len, after_start, after_read = row_bounds(
                    has_after.select(after_row, row), starts, ends
                )
                run_one_block_body(
                    storage,
                    row_first(row, read_start),
                    row_len,
                    fx.slice(indices, (row, None)),
                    fx.slice(value_output, (row, None)),
                    row_start,
                    next_row_iter=row_first(
                        (next_row < rows_m).select(next_row, row), next_read
                    ),
                    next_row_len=next_len,
                    fill_row=row == first_row,
                )
                # The next row resets LDS the slowest wave may still be reading.
                gpu.barrier()
                row_len, row_start, read_start = next_len, next_start, next_read
                next_len = has_after.select(after_len, fx.Int32(0))
                next_start, next_read = after_start, after_read

    else:

        @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
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
        if const_expr(persistent):
            grid_rows = (rows_m < fx.Int32(persistent_blocks)).select(
                rows_m, fx.Int32(persistent_blocks)
            )
            radix_topk_one_block_kernel(
                input,
                row_starts,
                row_ends,
                indices,
                values,
                width,
                next_n,
                rows_m,
                grid_rows,
            ).launch(grid=(grid_rows, 1, 1), block=(block_threads, 1, 1), stream=stream)
        else:
            radix_topk_one_block_kernel(
                input, row_starts, row_ends, indices, values, width, next_n
            ).launch(grid=(rows_m, 1, 1), block=(block_threads, 1, 1), stream=stream)

    return launch_radix_topk_one_block
