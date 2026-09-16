# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Experimental paged FP8 MQA scorer with an exact LDS local-TopK reservoir."""

from functools import lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import arith, const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from ..candidate_topk_common import (
    NUM_HIST_BINS,
    NUM_RADIX_PASSES,
    RADIX_SIGN_BIT,
    WAVE_SIZE,
    f32_to_ordered_i32,
    make_block_exclusive_prefix_i32,
    make_streaming_topk_storage,
    prefix_radix_mask,
    radix_bucket,
    radix_pass_bits,
)
from ..kernels_common import atomic_add_i32
from ..tensor_shim import GTensor, _run_compiled, _to_raw

HEADS = 32
HEAD_DIM = 128
INDEX_DIM = HEAD_DIM + 4
PROFILE_WAVES = 8
BLOCK_THREADS = PROFILE_WAVES * WAVE_SIZE
BLOCK_N = PROFILE_WAVES * 32
WAVES = BLOCK_THREADS // WAVE_SIZE
MFMA_M = 16
MFMA_N = 16
MFMA_K = 128
M_TILES = HEADS // MFMA_M
N_TILES_PER_WAVE = (BLOCK_N // MFMA_N) // WAVES
DREG = 4
# Keep the 4-wave compact cadence (26 score tiles) when BLOCK_N grows.
TILES_PER_COMPACT = 26
INCOMING_CAPACITY = TILES_PER_COMPACT * BLOCK_N
NUM_XCD = 8
WORKGROUPS_PER_CU = 16 // PROFILE_WAVES
SUPPORTED_K = (128, 512, 1024, 2048)


def fp8_paged_mqa_local_topk_kernel_name(
    *,
    topk: int,
    arch: str,
    preshuffled: bool,
    packed: bool,
    prepare_merge: bool,
) -> str:
    layout_name = "preshuffled" if preshuffled else "rowmajor"
    packed_tag = "_packed" if packed else ""
    return (
        f"fp8_paged_mqa_local_topk_h32d128_k{topk}_{layout_name}{packed_tag}_"
        f"w{WAVES}_bn{BLOCK_N}_inc{INCOMING_CAPACITY}_"
        f"pm{int(prepare_merge)}_ptr64_scan3_n1_{arch}"
    )


_RETAINED = 0
_SCORE_PREFIX = 1
_SCORE_MASK = 2
_REMAINING = 3
_STATE_SIZE = 4
_MERGE_PREFIX = 0
_MERGE_MASK = 1
_MERGE_REMAINING = 2
_MERGE_WRITE_COUNTER = 3
_MERGE_EQ_COUNTER = 4
_MERGE_DIRECT = 5
_PACK_SHIFT = 16
_PACK_MASK = (1 << _PACK_SHIFT) - 1
_NEUTRAL_E8M0 = 0x7F7F7F7F


def _udiv(a, b):
    return fx.Int32(fx.Uint32(a) // fx.Uint32(b))


def _umod(a, b):
    return fx.Int32(fx.Uint32(a) % fx.Uint32(b))


def _imin(a, b):
    a, b = fx.Int32(a), fx.Int32(b)
    return (a <= b).select(a, b)


def _concat_i32x4(lo, hi):
    lo, hi = fx.Vector(lo), fx.Vector(hi)
    return fx.Vector.from_elements(
        [lo[i].ir_value() for i in range_constexpr(4)]
        + [hi[i].ir_value() for i in range_constexpr(4)],
        fx.Int32,
    )


def _load_fp8x32(i32_view, byte_base, lane_div_16):
    """Load one lane's contiguous K32 fragment for 16x16x128 scaled MFMA."""
    off = (byte_base + lane_div_16 * 32) // fx.Int32(4)
    lo = i32_view.vec_load((off,), vec_size=4)
    hi = i32_view.vec_load((off + 4,), vec_size=4)
    return _concat_i32x4(lo, hi)


def _ptr_load_i32x4(i32_ptr, offset_i64):
    return fx.Vector(
        fx.ptr_load(
            i32_ptr + fx.Int64(offset_i64),
            result_type=T.i32x4,
        )
    )


def _ptr_load_fp8x32(i32_ptr, byte_base_i64, lane_div_16):
    """Load one lane's contiguous K32 fragment through a 64-bit global pointer."""
    off = (byte_base_i64 >> fx.Int64(2)) + fx.Int64(lane_div_16) * fx.Int64(8)
    lo = _ptr_load_i32x4(i32_ptr, off)
    hi = _ptr_load_i32x4(i32_ptr, off + fx.Int64(4))
    return _concat_i32x4(lo, hi)


def _ptr_load_preshuffled_fp8x32(
    i32_ptr,
    page_i32,
    token_in_page,
    lane_div_16,
):
    """Load one preshuffled K32 fragment with an i64 page offset."""
    token_block = fx.Int64(token_in_page // fx.Int32(16))
    token_lane = fx.Int64(token_in_page % fx.Int32(16))
    dim_block = fx.Int64(lane_div_16) * fx.Int64(2)
    off = (
        page_i32
        + token_block * fx.Int64(16 * HEAD_DIM // 4)
        + dim_block * fx.Int64(16 * 16 // 4)
        + token_lane * fx.Int64(4)
    )
    lo = _ptr_load_i32x4(i32_ptr, off)
    hi = _ptr_load_i32x4(i32_ptr, off + fx.Int64(16 * 16 // 4))
    return _concat_i32x4(lo, hi)


@lru_cache(maxsize=32)
def compile_fp8_paged_mqa_local_topk(
    *,
    topk: int,
    arch: str,
    preshuffled: bool,
    page_size: int,
    prepare_merge: bool = False,
    packed: bool = False,
):
    """Compile an H32D128 Stage-A specialization."""
    if topk not in SUPPORTED_K:
        raise ValueError(f"topk must be one of {SUPPORTED_K}, got {topk}")
    if arch != "gfx950":
        raise RuntimeError(
            "the initial native-e4m3fn Stage-A specialization supports gfx950 only"
        )
    pool_capacity = topk + INCOMING_CAPACITY
    pool_steps = (pool_capacity + BLOCK_THREADS - 1) // BLOCK_THREADS
    output_steps = (topk + BLOCK_THREADS - 1) // BLOCK_THREADS
    storage_type = make_streaming_topk_storage(
        pool_capacity,
        _STATE_SIZE,
        fx.Uint16,
        num_waves=WAVES,
    )
    block_exclusive_prefix_i32 = make_block_exclusive_prefix_i32(WAVES)
    kernel_name = fp8_paged_mqa_local_topk_kernel_name(
        topk=topk,
        arch=arch,
        preshuffled=preshuffled,
        packed=packed,
        prepare_merge=prepare_merge,
    )
    page_index_dim = INDEX_DIM if packed else HEAD_DIM
    block_i32 = page_size * page_index_dim // 4
    scale_i32 = page_size * HEAD_DIM // 4

    # Packed pages are 8448 B = 8192+256; split pages are 8192 B. Both are
    # page_size*128 of K, so in-page K addressing is identical once the origin
    # is an i32 page base. page_size 64 is (p<<11)+(p<<6) vs p<<11 — not a
    # gather stride on the K payload.
    if page_size == 64 and packed:

        def _page_i32(physical_page):
            page = fx.Int64(physical_page)
            return (page << fx.Int64(11)) + (page << fx.Int64(6))

    elif page_size == 64:

        def _page_i32(physical_page):
            return fx.Int64(physical_page) << fx.Int64(11)

    else:

        def _page_i32(physical_page):
            return fx.Int64(physical_page) * fx.Int64(block_i32)

    if preshuffled:

        def _load_k(kv_i32_ptr, page_i32, token_in_page, lane):
            return _ptr_load_preshuffled_fp8x32(
                kv_i32_ptr,
                page_i32,
                token_in_page,
                lane,
            )

    else:

        def _load_k(kv_i32_ptr, page_i32, token_in_page, lane):
            byte_base = (page_i32 << fx.Int64(2)) + fx.Int64(token_in_page) * fx.Int64(
                HEAD_DIM
            )
            return _ptr_load_fp8x32(kv_i32_ptr, byte_base, lane)

    if packed:

        def _load_scale(scales_ptr, page_i32, token_in_page):
            return fx.Float32(
                fx.ptr_load(
                    scales_ptr
                    + page_i32
                    + fx.Int64(scale_i32)
                    + fx.Int64(token_in_page)
                )
            )

    else:

        def _load_scale(scales_ptr, page_i32, token_in_page):
            return fx.Float32(
                fx.ptr_load(
                    scales_ptr + (page_i32 >> fx.Int64(5)) + fx.Int64(token_in_page)
                )
            )

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        q_fp8: fx.Tensor,
        kv_cache: fx.Tensor,
        k_scales: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        block_tables: fx.Tensor,
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        merge_histogram: fx.Tensor,
        merge_state: fx.Tensor,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
    ):
        page_size_i32 = fx.Int32(page_size)
        tid = fx.Int32(gpu.thread_idx.x)
        split = fx.Int32(gpu.block_idx.x)
        row = fx.Int32(gpu.block_idx.y)
        wave = _udiv(tid, WAVE_SIZE)
        lane = _umod(tid, WAVE_SIZE)
        lane_div_16 = _udiv(lane, MFMA_N)
        lane_mod_16 = _umod(lane, MFMA_N)

        q_i32 = GTensor(q_fp8, dtype=T.i32, shape=(-1,))
        kv_i32_ptr = fx.recast_iter(
            fx.PointerType.get(T.i32, fx.AddressSpace.Global, 4),
            fx.get_iter(kv_cache),
        )
        scales_ptr = fx.recast_iter(
            fx.PointerType.get(T.f32, fx.AddressSpace.Global, 4),
            fx.get_iter(k_scales),
        )
        weight_t = GTensor(weights, dtype=T.f32, shape=(-1, HEADS))
        lengths = GTensor(context_lens, dtype=T.i32, shape=(-1,))
        tables = GTensor(block_tables, dtype=T.i32, shape=(-1,))

        storage = fx.SharedAllocator().allocate(storage_type)
        pool_values = storage.pool_values.peek().view(fx.make_layout(pool_capacity, 1))
        pool_indices = storage.pool_indices.peek().view(
            fx.make_layout(pool_capacity, 1)
        )
        histogram = storage.histogram.peek().view(fx.make_layout(NUM_HIST_BINS, 1))
        scan = storage.scan.peek().view(fx.make_layout(WAVES + 1, 1))
        state = storage.state.peek().view(fx.make_layout(_STATE_SIZE, 1))

        table_span = max_pages * page_size_i32
        valid_len = _imin(fx.Int32(lengths[row]), table_span)
        valid_len = (valid_len < 0).select(fx.Int32(0), valid_len)
        split_begin = _udiv(valid_len * split, num_splits)
        split_end = _udiv(
            valid_len * (split + fx.Int32(1)),
            num_splits,
        )
        request = row

        if tid == 0:
            state[_RETAINED] = 0
        gpu.barrier()

        q_tiles = [None] * M_TILES
        weight_frags = [[None] * DREG for _ in range_constexpr(M_TILES)]
        q_row_bytes = row * fx.Int32(HEADS * HEAD_DIM)
        for mi in range_constexpr(M_TILES):
            q_head = fx.Int32(mi * MFMA_M) + lane_mod_16
            q_tiles[mi] = _load_fp8x32(
                q_i32,
                q_row_bytes + q_head * fx.Int32(HEAD_DIM),
                lane_div_16,
            )
            for ii in range_constexpr(DREG):
                head = mi * MFMA_M + lane_div_16 * DREG + ii
                weight_frags[mi][ii] = fx.Float32(weight_t[row, head])

        def _compact(
            pool_count,
            state,
            histogram,
            scan,
            pool_values,
            pool_indices,
        ):
            if tid == 0:
                state[_SCORE_PREFIX] = 0
                state[_SCORE_MASK] = 0
                state[_REMAINING] = fx.Int32(topk)
            gpu.barrier()

            for radix_pass in range_constexpr(NUM_RADIX_PASSES):
                pass_bits = radix_pass_bits(radix_pass)
                num_bins = 1 << pass_bits
                bins_per_thread = num_bins // BLOCK_THREADS
                for hist_step in range_constexpr(
                    (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
                ):
                    histogram[hist_step * BLOCK_THREADS + tid] = 0
                gpu.barrier()
                score_prefix = state[_SCORE_PREFIX]
                score_mask = state[_SCORE_MASK]
                xor_value = RADIX_SIGN_BIT if radix_pass == 0 else 0

                for step in range_constexpr(pool_steps):
                    pool_pos = fx.Int32(step * BLOCK_THREADS) + tid
                    live = pool_pos < pool_count
                    safe_pos = live.select(pool_pos, fx.Int32(0))
                    score_ord = f32_to_ordered_i32(pool_values[safe_pos])
                    prefix_match = live & ((score_ord & score_mask) == score_prefix)
                    if prefix_match:
                        bucket = radix_bucket(score_ord, radix_pass) ^ fx.Int32(
                            xor_value
                        )
                        atomic_add_i32(histogram, 1, bucket, "workgroup")
                gpu.barrier()

                selected_high = fx.Int32(num_bins - 1) - tid * fx.Int32(bins_per_thread)
                bin_counts = [
                    histogram[selected_high - fx.Int32(bin_item)]
                    for bin_item in range_constexpr(bins_per_thread)
                ]
                group_count = fx.Int32(0)
                for bin_item in range_constexpr(bins_per_thread):
                    group_count = group_count + bin_counts[bin_item]
                before_group, _ = block_exclusive_prefix_i32(tid, group_count, scan)
                before_bin = before_group
                remaining = state[_REMAINING]
                for bin_item in range_constexpr(bins_per_thread):
                    selected = selected_high - fx.Int32(bin_item)
                    bin_count = bin_counts[bin_item]
                    if (before_bin < remaining) & (before_bin + bin_count >= remaining):
                        actual = selected ^ fx.Int32(xor_value)
                        pass_mask, shift = prefix_radix_mask(radix_pass)
                        state[_SCORE_PREFIX] = score_prefix | (
                            actual << fx.Int32(shift)
                        )
                        state[_SCORE_MASK] = score_mask | pass_mask
                        state[_REMAINING] = remaining - before_bin
                    before_bin = before_bin + bin_count
                gpu.barrier()

            threshold = state[_SCORE_PREFIX]
            equal_needed = state[_REMAINING]
            write_cursor = fx.Int32(0)
            equal_seen = fx.Int32(0)
            for step in range_constexpr(pool_steps):
                pool_pos = fx.Int32(step * BLOCK_THREADS) + tid
                live = pool_pos < pool_count
                safe_pos = live.select(pool_pos, fx.Int32(0))
                value = pool_values[safe_pos]
                score_ord = f32_to_ordered_i32(value)
                logical_position = fx.Uint16(pool_indices[safe_pos])
                better = live & (score_ord > threshold)
                equal = live & (score_ord == threshold)
                better_i32 = better.select(fx.Int32(1), fx.Int32(0))
                equal_i32 = equal.select(fx.Int32(1), fx.Int32(0))
                packed_counts = (better_i32 << fx.Int32(_PACK_SHIFT)) + equal_i32
                packed_before, packed_total = block_exclusive_prefix_i32(
                    tid, packed_counts, scan
                )
                better_before = packed_before >> fx.Int32(_PACK_SHIFT)
                equal_before = packed_before & fx.Int32(_PACK_MASK)
                better_total = packed_total >> fx.Int32(_PACK_SHIFT)
                equal_total = packed_total & fx.Int32(_PACK_MASK)
                room = equal_needed - equal_seen
                room = (room < 0).select(fx.Int32(0), room)
                admit_equal = equal & (equal_before < room)
                keep = better | admit_equal
                admitted_equal_before = (equal_before < room).select(equal_before, room)
                destination = write_cursor + better_before + admitted_equal_before
                if keep:
                    pool_values[destination] = value
                    pool_indices[destination] = logical_position
                admitted_equal_total = (equal_total < room).select(equal_total, room)
                write_cursor = write_cursor + better_total + admitted_equal_total
                equal_seen = equal_seen + equal_total
            if tid == 0:
                state[_RETAINED] = fx.Int32(topk)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()

        neutral = arith.constant(_NEUTRAL_E8M0, type=T.i32)
        result_type = fx.Vector.make_type(DREG, fx.Float32)

        def _load_pages(col0):
            wave_tile_base = wave * fx.Int32(N_TILES_PER_WAVE)
            page_i32s = []
            page_oks = []
            for ni in range_constexpr(N_TILES_PER_WAVE):
                logical = (
                    col0
                    + (wave_tile_base + fx.Int32(ni)) * fx.Int32(MFMA_N)
                    + lane_mod_16
                )
                safe_logical = _imin(logical, split_end - fx.Int32(1))
                table_offset = request * max_pages + _udiv(safe_logical, page_size_i32)
                physical_page = fx.Int32(tables[table_offset])
                page_ok = (physical_page >= 0) & (physical_page < num_pages)
                safe_page = page_ok.select(physical_page, fx.Int32(0))
                page_i32s.append(_page_i32(safe_page))
                page_oks.append(page_ok)
            return page_i32s, page_oks

        def _load_chunk(col0, page_i32s, page_oks):
            wave_tile_base = wave * fx.Int32(N_TILES_PER_WAVE)
            logical_tiles = []
            k_packs = []
            scale_tiles = []
            chunk_oks = []
            for ni in range_constexpr(N_TILES_PER_WAVE):
                logical = (
                    col0
                    + (wave_tile_base + fx.Int32(ni)) * fx.Int32(MFMA_N)
                    + lane_mod_16
                )
                safe_logical = _imin(logical, split_end - fx.Int32(1))
                page_i32 = page_i32s[ni]
                token_in_page = _umod(safe_logical, page_size_i32)
                k_pack = _load_k(
                    kv_i32_ptr,
                    page_i32,
                    token_in_page,
                    lane_div_16,
                )
                scale = _load_scale(scales_ptr, page_i32, token_in_page)
                logical_tiles.append(logical)
                k_packs.append(k_pack)
                scale_tiles.append(scale)
                chunk_oks.append(page_oks[ni])
            return logical_tiles, k_packs, scale_tiles, chunk_oks

        @flyc.jit
        def _write_candidate(
            _pred,
            _dst,
            _score,
            _relative,
            _values=pool_values,
            _indices=pool_indices,
        ):
            if _pred:
                _values[_dst] = _score
                _indices[_dst] = fx.Uint16(_relative)

        def _score_chunk(batch_first, logical_tiles, k_packs, scale_tiles, page_oks):
            for ni in range_constexpr(N_TILES_PER_WAVE):
                logical = logical_tiles[ni]
                k_pack = k_packs[ni]
                scale = scale_tiles[ni]
                total = fx.Float32(0.0)
                for mi in range_constexpr(M_TILES):
                    q_pack = q_tiles[mi]
                    acc = fx.Vector.filled(DREG, 0.0, fx.Float32)
                    acc = rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                        result_type,
                        [
                            q_pack,
                            k_pack,
                            acc,
                            0,
                            0,
                            0,
                            neutral,
                            0,
                            neutral,
                        ],
                    )
                    frag = fx.Vector(acc)
                    for ii in range_constexpr(DREG):
                        activated = fx.Float32(frag[ii]).maximumf(fx.Float32(0.0))
                        weight = weight_frags[mi][ii]
                        total = total + activated * weight
                total = total + total.shuffle_xor(16, WAVE_SIZE)
                total = total + total.shuffle_xor(32, WAVE_SIZE)
                # `scale` is per position: it is the same for all DREG heads in a
                # lane and across both shuffle_xor lane groups, so it factors out
                # of the ReLU sum. Clamping it to >= 0 keeps that exact, since a
                # negative or NaN scale must score 0 (previously supplied by
                # maximumf on each scaled product).
                total = total * scale.maximumf(fx.Float32(0.0))
                total = page_oks[ni].select(total, fx.Float32(float("-inf")))

                is_writer = (lane_div_16 == 0) & (logical < split_end)
                destination = state[_RETAINED] + logical - batch_first

                _write_candidate(
                    is_writer,
                    destination,
                    total,
                    logical - split_begin,
                )

        # Keep the second tile's paged loads in flight during the first tile's score path.
        tiles_per_group = 2
        assert TILES_PER_COMPACT % tiles_per_group == 0
        group_n = fx.Int32(tiles_per_group * BLOCK_N)
        num_tiles = _udiv(
            split_end - split_begin + fx.Int32(BLOCK_N - 1),
            fx.Int32(BLOCK_N),
        )
        paired_tiles = _udiv(num_tiles, fx.Int32(tiles_per_group)) * fx.Int32(
            tiles_per_group
        )
        paired_end = _imin(
            split_begin + paired_tiles * fx.Int32(BLOCK_N),
            split_end,
        )
        tile_number = fx.Int32(0)
        for col0 in range(split_begin, paired_end, group_n):
            batch_tile = _umod(tile_number, TILES_PER_COMPACT)
            batch_first = col0 - batch_tile * fx.Int32(BLOCK_N)
            page_i32s, page_oks = _load_pages(col0)
            next_i32s, next_oks = _load_pages(col0 + fx.Int32(BLOCK_N))
            logical_tiles, k_packs, scale_tiles, chunk_oks = _load_chunk(
                col0, page_i32s, page_oks
            )
            next_logical_tiles, next_k_packs, next_scale_tiles, next_chunk_oks = (
                _load_chunk(
                    col0 + fx.Int32(BLOCK_N),
                    next_i32s,
                    next_oks,
                )
            )
            _score_chunk(batch_first, logical_tiles, k_packs, scale_tiles, chunk_oks)
            _score_chunk(
                batch_first,
                next_logical_tiles,
                next_k_packs,
                next_scale_tiles,
                next_chunk_oks,
            )

            end_of_batch = batch_tile + fx.Int32(tiles_per_group) >= fx.Int32(
                TILES_PER_COMPACT
            )
            final_tile = col0 + group_n >= split_end
            if end_of_batch | final_tile:
                gpu.barrier()
                batch_end = _imin(col0 + group_n, split_end)
                incoming_count = batch_end - batch_first
                pool_count = state[_RETAINED] + incoming_count
                if pool_count > fx.Int32(topk):
                    _compact(
                        pool_count,
                        state,
                        histogram,
                        scan,
                        pool_values,
                        pool_indices,
                    )
                else:
                    if tid == 0:
                        state[_RETAINED] = pool_count
                    gpu.barrier()
            tile_number = tile_number + fx.Int32(tiles_per_group)

        for col0 in range(paired_end, split_end, fx.Int32(BLOCK_N)):
            batch_tile = _umod(tile_number, TILES_PER_COMPACT)
            batch_first = col0 - batch_tile * fx.Int32(BLOCK_N)
            page_i32s, page_oks = _load_pages(col0)
            logical_tiles, k_packs, scale_tiles, chunk_oks = _load_chunk(
                col0, page_i32s, page_oks
            )
            _score_chunk(batch_first, logical_tiles, k_packs, scale_tiles, chunk_oks)

            gpu.barrier()
            incoming_count = split_end - batch_first
            pool_count = state[_RETAINED] + incoming_count
            if pool_count > fx.Int32(topk):
                _compact(
                    pool_count,
                    state,
                    histogram,
                    scan,
                    pool_values,
                    pool_indices,
                )
            else:
                if tid == 0:
                    state[_RETAINED] = pool_count
                gpu.barrier()
            tile_number = tile_number + fx.Int32(1)

        retained = state[_RETAINED]
        output_scores_row = fx.slice(
            candidate_scores,
            (row, split, None),
        )
        output_positions_row = fx.slice(
            candidate_positions,
            (row, split, None),
        )
        merge_histogram_row = fx.slice(
            merge_histogram,
            (row, 0, None),
        )
        merge_state_row = fx.slice(
            merge_state,
            (row, None),
        )
        if const_expr(prepare_merge):
            for hist_step in range_constexpr(
                (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
            ):
                histogram[hist_step * BLOCK_THREADS + tid] = 0
            if split == 0 and tid == 0:
                merge_state_row[_MERGE_PREFIX] = 0
                merge_state_row[_MERGE_MASK] = 0
                merge_state_row[_MERGE_REMAINING] = _imin(
                    valid_len,
                    fx.Int32(topk),
                )
                merge_state_row[_MERGE_WRITE_COUNTER] = 0
                merge_state_row[_MERGE_EQ_COUNTER] = 0
                merge_state_row[_MERGE_DIRECT] = 0
            gpu.barrier()

        for step in range_constexpr(output_steps):
            slot = fx.Int32(step * BLOCK_THREADS) + tid
            if slot < fx.Int32(topk):
                live = slot < retained
                safe_slot = live.select(slot, fx.Int32(0))
                candidate_score = pool_values[safe_slot]
                output_scores_row[slot] = live.select(
                    candidate_score,
                    fx.Float32(float("-inf")),
                )
                output_positions_row[slot] = live.select(
                    split_begin + fx.Int32(pool_indices[safe_slot]),
                    fx.Int32(-1),
                )
                if const_expr(prepare_merge):

                    @flyc.jit
                    def _count_candidate(
                        _pred=live,
                        _score=candidate_score,
                        _histogram=histogram,
                    ):
                        if _pred:
                            bucket = radix_bucket(
                                f32_to_ordered_i32(_score),
                                0,
                            ) ^ fx.Int32(RADIX_SIGN_BIT)
                            atomic_add_i32(
                                _histogram,
                                1,
                                bucket,
                                "workgroup",
                            )

                    _count_candidate()
        if const_expr(prepare_merge):
            gpu.barrier()
            for hist_step in range_constexpr(
                (NUM_HIST_BINS + BLOCK_THREADS - 1) // BLOCK_THREADS
            ):
                hist_bin = hist_step * BLOCK_THREADS + tid
                count = histogram[hist_bin]
                if count != 0:
                    atomic_add_i32(
                        merge_histogram_row,
                        count,
                        hist_bin,
                        "agent",
                    )
            gpu.barrier()
        if tid == 0:
            fx.ptr_store(
                retained,
                fx.add_offset(
                    fx.get_iter(candidate_counts),
                    row * num_splits + split,
                ),
            )

    @flyc.jit
    def launch(
        q_fp8: fx.Tensor,
        kv_cache: fx.Tensor,
        k_scales: fx.Tensor,
        weights: fx.Tensor,
        context_lens: fx.Tensor,
        block_tables: fx.Tensor,
        candidate_scores: fx.Tensor,
        candidate_positions: fx.Tensor,
        candidate_counts: fx.Tensor,
        merge_histogram: fx.Tensor,
        merge_state: fx.Tensor,
        rows: fx.Int32,
        num_splits: fx.Int32,
        max_pages: fx.Int32,
        num_pages: fx.Int32,
        stream: fx.Stream,
    ):
        gx = arith.index_cast(T.index, _to_raw(num_splits))
        gy = arith.index_cast(T.index, _to_raw(rows))
        kernel(
            q_fp8,
            kv_cache,
            k_scales,
            weights,
            context_lens,
            block_tables,
            candidate_scores,
            candidate_positions,
            candidate_counts,
            merge_histogram,
            merge_state,
            num_splits,
            max_pages,
            num_pages,
        ).launch(grid=(gx, gy, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)

    launch.compile_hints = {"waves_per_eu": 1, "fast_fp_math": True}
    return launch


def launch_fp8_paged_mqa_local_topk(
    q_fp8,
    kv_cache,
    k_scales,
    weights,
    context_lens,
    block_tables,
    candidate_scores,
    candidate_positions,
    candidate_counts,
    *,
    topk,
    num_splits,
    preshuffled,
    arch,
    stream,
    prepare_merge=False,
    merge_histogram=None,
    merge_state=None,
    packed=False,
):
    page_size = kv_cache.shape[1]
    if q_fp8.ndim != 4 or q_fp8.shape[1:] != (1, 32, 128):
        raise ValueError(
            f"q_fp8 must have shape [B,1,32,128], got {tuple(q_fp8.shape)}"
        )
    rows = q_fp8.shape[0]
    launcher = compile_fp8_paged_mqa_local_topk(
        topk=topk,
        arch=arch,
        preshuffled=preshuffled,
        page_size=page_size,
        prepare_merge=prepare_merge,
        packed=packed,
    )
    max_pages = block_tables.shape[1]
    num_pages = kv_cache.shape[0]
    if merge_histogram is None:
        merge_histogram = candidate_scores
    if merge_state is None:
        merge_state = candidate_counts
    _run_compiled(
        launcher,
        q_fp8,
        kv_cache,
        k_scales,
        weights,
        context_lens,
        block_tables,
        candidate_scores,
        candidate_positions,
        candidate_counts,
        merge_histogram,
        merge_state,
        rows,
        num_splits,
        max_pages,
        num_pages,
        stream,
    )
