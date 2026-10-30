# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Exact sampled decode selection with an in-CTA radix fallback."""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr

from ..kernels_common import atomic_add_i32, atomic_max_i32
from .radix_topk_one_block import build_radix_topk_one_block_body
from .topk_common import _f32_to_ord, _load_f32x4, _row_length

_THREADS = 1024
_CAPACITY = 9000
_SAMPLE_LIMIT = 4096


@cache
def build_sampled_topk_module(k=2048, report_counts=False):
    """Build gfx950 decode selection.

    Optional counts are diagnostics, not workspace: every row overwrites its
    count (-1 for a bypassed short row). No state survives between launches.
    """
    shared_storage, fallback = build_radix_topk_one_block_body(
        k, wave_size=64, arch="gfx950"
    )
    block_scan = fx.coop.BlockScan[fx.Int32, _THREADS]
    warp_scan = fx.coop.WarpScan[fx.Int32, 64]

    @flyc.jit
    def kth_key(storage, keys, count, rank, lower, upper):
        tid = fx.thread_idx.x
        scan_storage = storage.scan.peek()
        histogram = storage.arena.long_later.histogram.peek().view(
            fx.make_layout(1024, 1)
        )
        metadata = storage.metadata.peek().view(fx.make_layout(10, 1))
        # Bounds certify a shared prefix; begin with eight differing bits
        # instead of serializing the bag on its common exponent byte.
        differing = fx.Uint32(lower ^ upper)
        highest = fx.Int32(0)
        for bits in range_constexpr(16, 0, -1):
            if const_expr(bits in (16, 8, 4, 2, 1)):
                advance = differing >= fx.Uint32(1 << bits)
                highest = highest + advance.select(fx.Int32(bits), fx.Int32(0))
                differing = advance.select(differing >> fx.Uint32(bits), differing)
        start = fx.max(highest - 7, fx.Int32(0))
        initial_mask = (start < 24).select(
            fx.Int32(fx.Uint32(0xFFFFFFFF) << fx.Uint32(start + 8)), fx.Int32(0)
        )
        if tid == 0:
            metadata[1] = lower & initial_mask
            metadata[2] = rank
        gpu.barrier()
        for pass_id in range_constexpr(4):
            shift = start - pass_id * 8
            if shift > -8:
                next_bit = shift + 8
                digit_mask = (fx.Int32(1) << fx.min(next_bit, fx.Int32(8))) - 1
                shift = fx.max(shift, fx.Int32(0))
                histogram[tid] = fx.Int32(0)
                gpu.barrier()
                prefix = metadata[1]
                remaining = metadata[2]
                mask = (next_bit < 32).select(
                    fx.Int32(fx.Uint32(0xFFFFFFFF) << fx.Uint32(next_bit)), fx.Int32(0)
                )
                for i in range(tid, count, fx.Int32(_THREADS)):
                    key = keys[i]
                    if (key & mask) == prefix:
                        digit = (key >> shift) & digit_mask
                        # Interleave replicas across banks without increasing LDS.
                        atomic_add_i32(
                            histogram, fx.Int32(1), digit * 4 + tid % 4, "workgroup"
                        )
                gpu.barrier()
                local_count = fx.Int32(0)
                if tid < 256:
                    for shard in range_constexpr(4):
                        local_count = local_count + histogram[tid * 4 + shard]
                exclusive, total = block_scan.exclusive_with_aggregate(
                    local_count, fx.ReductionOp.ADD, storage=scan_storage
                )
                target = total - remaining
                inclusive = exclusive + local_count
                if (exclusive <= target) & (inclusive > target):
                    metadata[1] = prefix | (tid << shift)
                    metadata[2] = remaining - (total - inclusive)
                gpu.barrier()
        return metadata[1]

    @flyc.jit
    def reserve(metadata, amount, slot):
        _, prefix, total = warp_scan.scan_with_aggregate(amount, fx.ReductionOp.ADD)
        base = fx.Int32(0)
        if (fx.thread_idx.x % 64 == 0) & (total > 0):
            base = atomic_add_i32(metadata, total, slot, "workgroup")
        base = fx.coop.warp_broadcast(base, 0, width=64)
        return base + prefix

    @flyc.kernel(
        name=f"sampled_topk_gfx950_k{k}_counts{int(report_counts)}",
        known_block_size=[_THREADS, 1, 1],
    )
    def kernel(
        logits: fx.Tensor,
        row_ends: fx.Tensor,
        indices: fx.Tensor,
        counts: fx.Tensor,
        width: fx.Int32,
        next_n: fx.Int32,
    ):
        row = fx.block_idx.x
        tid = fx.thread_idx.x
        length = _row_length(row, row_ends, width, next_n)
        physical_row = fx.slice(logits, (row, None))
        output = fx.slice(indices, (row, None))
        storage = fx.SharedAllocator().allocate(shared_storage)
        keys = storage.arena.long_later.candidate_ordered_keys.peek().view(
            fx.make_layout(_CAPACITY, 1)
        )
        columns = storage.arena.long_later.candidate_local_indices.peek().view(
            fx.make_layout(_CAPACITY, 1)
        )
        metadata = storage.metadata.peek().view(fx.make_layout(10, 1))
        sign = fx.Int32(-2147483648)
        accepted = fx.Int32(0)
        if const_expr(report_counts):  # noqa: SIM102 -- constexpr and runtime branches
            if tid == 0:
                counts[row] = fx.Int32(-1)
        if length > 8192:
            source = fx.rocdl.make_buffer_tensor(
                fx.make_view(fx.get_iter(physical_row), fx.make_layout(1073741823, 1)),
                num_records_bytes=fx.Int64(length) * 4,
            )
            tiles = fx.logical_divide(source, fx.make_layout(4, 1))
            stratum = fx.max(
                fx.Int32(2048), ((length + _SAMPLE_LIMIT - 1) // _SAMPLE_LIMIT) * 32
            )
            chunks = (length + stratum - 1) // stratum
            tail = length - (chunks - 1) * stratum
            sample_count = (chunks - 1) * 32 + fx.min(tail, fx.Int32(32))
            # Eight lanes share a 128-byte chunk, so sparse sampling fetches
            # sparse cache lines rather than a separate line for every value.
            chunk = tid // 8
            begin = chunk * stratum
            extent = fx.min(stratum, length - begin)
            hashed = fx.Uint32(chunk + row * 4099 + 17)
            hashed = (hashed ^ (hashed >> fx.Uint32(16))) * fx.Uint32(0x7FEB352D)
            hashed = (hashed ^ (hashed >> fx.Uint32(15))) * fx.Uint32(0x846CA68B)
            hashed = hashed ^ (hashed >> fx.Uint32(16))
            slots = fx.max(extent // 32, fx.Int32(1))
            col = begin + fx.Int32(hashed % fx.Uint32(slots)) * 32 + (tid % 8) * 4
            values = _load_f32x4(tiles, (chunk < chunks).select(col // 4, length))
            for j in range_constexpr(4):
                sample_index = tid * 4 + j
                if sample_index < sample_count:
                    keys[sample_index] = _f32_to_ord(values[j]) ^ sign
            gpu.barrier()
            rank = fx.max(fx.Int32(1), sample_count * 4096 // length)
            threshold = kth_key(
                storage, keys, sample_count, rank, fx.Int32(0), fx.Int32(-1)
            )
            if tid == 0:
                metadata[0] = fx.Int32(0)
                metadata[5] = sign
            gpu.barrier()
            row_max = sign
            for tile in range(fx.Int32(0), (length + 16383) // 16384):
                prefetched = []
                for stage in range_constexpr(4):
                    prefetched.append(
                        _load_f32x4(tiles, (tile * 4 + stage) * _THREADS + tid)
                    )
                for stage in range_constexpr(4):
                    col = ((tile * 4 + stage) * _THREADS + tid) * 4
                    values = prefetched[stage]
                    local_keys = fx.make_rmem_tensor(4, fx.Int32)
                    flags = fx.make_rmem_tensor(4, fx.Int32)
                    amount = fx.Int32(0)
                    for j in range_constexpr(4):
                        key = _f32_to_ord(values[j]) ^ sign
                        keep = (col + j < length) & ((key ^ sign) >= (threshold ^ sign))
                        row_max = fx.max(row_max, keep.select(key ^ sign, sign))
                        local_keys[j] = key
                        flags[j] = keep.select(fx.Int32(1), fx.Int32(0))
                        amount = amount + flags[j]
                    pos = reserve(metadata, amount, 0)
                    for j in range_constexpr(4):
                        if flags[j] != 0:
                            if pos < _CAPACITY:
                                keys[pos] = local_keys[j]
                                columns[pos] = col + j
                            pos = pos + 1
            for offset in range_constexpr(6):
                row_max = fx.max(row_max, gpu.shuffle_xor(row_max, 1 << offset, 64))
            if tid % 64 == 0:
                atomic_max_i32(metadata, row_max, fx.Int32(5), "workgroup")
            gpu.barrier()
            count = metadata[0]
            if const_expr(report_counts):  # noqa: SIM102
                if tid == 0:
                    counts[row] = count
            if (count >= k) & (count <= _CAPACITY):
                accepted = fx.Int32(1)
                threshold = kth_key(
                    storage, keys, count, fx.Int32(k), threshold, metadata[5] ^ sign
                )
                needed = metadata[2]
                if tid == 0:
                    metadata[3] = fx.Int32(0)
                    metadata[4] = fx.Int32(0)
                gpu.barrier()
                for tile in range(fx.Int32(0), (count + _THREADS - 1) // _THREADS):
                    i = tile * _THREADS + tid
                    key = fx.Int32(0)
                    column = fx.Int32(0)
                    if i < count:
                        key = keys[i]
                        column = columns[i]
                    above = (i < count) & ((key ^ sign) > (threshold ^ sign))
                    equal = (i < count) & (key == threshold)
                    above_pos = reserve(
                        metadata, above.select(fx.Int32(1), fx.Int32(0)), 3
                    )
                    equal_pos = reserve(
                        metadata, equal.select(fx.Int32(1), fx.Int32(0)), 4
                    )
                    if above:
                        output[above_pos] = column
                    if equal & (equal_pos < needed):
                        output[fx.Int32(k - 1) - equal_pos] = column
                gpu.barrier()
        if accepted == 0:
            # The body initializes its histogram and metadata, discarding the
            # speculative bag. The allocation and global row contract are identical.
            fallback(
                storage,
                fx.get_iter(physical_row),
                length,
                output,
                physical_row,
                fx.Int32(0),
            )

    @flyc.jit
    def launch(
        logits: fx.Tensor,
        row_ends: fx.Tensor,
        indices: fx.Tensor,
        counts: fx.Tensor,
        width: fx.Int32,
        next_n: fx.Int32,
        rows: fx.Int32,
        stream: fx.Stream,
    ):
        kernel(logits, row_ends, indices, counts, width, next_n).launch(
            grid=(rows, 1, 1), block=(_THREADS, 1, 1), stream=stream
        )

    # Two 16-wave CTAs need eight waves/SIMD; without this hint the scalar
    # allocator retains too many branch masks and limits residency to one CTA.
    launch.compile_hints = {"waves_per_eu": 8}
    return launch
