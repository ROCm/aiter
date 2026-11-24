# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale MoE GateUp (BM256) on a 2x4 wave grid, A and B both staged through LDS: the mega
kernel's GateUp task emitter, with a swiglu_limit clamp.

Per route, SiLU(gate) * up from one FP8 GEMM, quantized to FP8 per (row, 128 columns) with FP32 scale amax / 240
at sorted row block * 256 + row of [blocks * 256, I] (scales [I/128, blocks * 256]), Down's input. B is aiter
shuffle_weight(16,16) of [E, 2I, K] (gate rows, then up rows); scales are FP32: A [K/128, rows], B
[E, 2I/128, K/128]. Sorted IDs are (slot << 24) | token; padding rows use an invalid token or slot, and blocks past
ceil(num_valid_ids[0] / 256) are never read. Per K128 block, A rows are gathered by sorted ID with eight lanes per
row (128 B per row per load) into a 32 KiB LDS tile, and the B packets gate0, gate1, up0, up1 (64 rows each) fill
another 32 KiB.
The pipeline: 2 waves along M (128 rows) x 4 along N (32 gate + 32 up columns); B is read for the whole K block at
stage 0 ("held"), A streams one 64-row packet per stage, and each packet of the next K block is written to LDS one
stage after its load. B rows are placed in LDS so that every wave's gate (and up) columns are contiguous. The B
scales of all K blocks sit in one VGPR and are read with v_readlane, so the K loop issues no scalar loads.
Row maxima meet in LDS across the waves along N, and the FP8 tile goes through LDS so that every dwordx4 store
writes 8 full rows x 128 B; the stores are written through (sc1), so other XCDs see them after a vmcnt wait.
With wq="fp4", B is a8w4's MXFP4 fd2 form instead, decoded to FP8 on its way into LDS.
"""

import functools
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..cvt import _abs_bits
from ..hw import _sched_fence
from ..hw import _u as _uniform
from .util import (
    BufferTensor,
    _load_i32,
    _rsrc_of,
    buffer_load,
    buffer_store,
    decode_fd2,
    fd2_channel_factors,
    fd2_lane_offsets,
    load_fd2,
    mfma_fp8,
    readlane_f32,
    set_priority,
)

SORT_BLOCK_M = 256
TILE_N = 128
TILE_K = 128
TOKEN_MASK = 0xFFFFFF
# Past every descriptor used here; stays out of range after adding a tile-local byte offset.
OOB_OFFSET = 0x7FFF0000
LOG2E = 1.4426950408889634
PACKET_ROWS = 64
PACKET_DWORDS = PACKET_ROWS * TILE_K // 4
# Largest finite FP8 E4M3FNUZ (gfx942).
FP8_MAX = 240.0
# Cache bits of the output stores: sc1, written through to memory.
STORE_AUX = 16


def validate_parameters(inter_dim, hidden_dim, topk, num_experts):
    if inter_dim <= 0 or inter_dim % TILE_N:
        raise ValueError("inter_dim must be a positive multiple of 128")
    if hidden_dim < TILE_K or hidden_dim % TILE_K:
        raise ValueError("hidden_dim must be a positive multiple of 128")
    if not 0 < topk < 256 or num_experts <= 0:
        raise ValueError("require 0 < topk < 256 and num_experts > 0")
    # Descriptor byte counts must stay below OOB_OFFSET (the sorted-row buffers are sized by the sort capacity,
    # which the caller checks).
    if (inter_dim + TILE_N) * hidden_dim >= OOB_OFFSET:
        raise ValueError("weight tile byte extent too large")


def _lds_dwords(row, column):
    # XOR swizzle of the 16B column keeps MFMA-pattern ds_read_b128 conflict free.
    return row * (TILE_K // 4) + ((row ^ column) & 7) * 4


@functools.cache
def gateup_task_emitter(
    inter_dim, hidden_dim, topk, num_experts, swiglu_limit=0.0, wq="fp8"
):
    """One GateUp task (256 sorted rows x 128 gate + 128 up columns) for use inside a kernel.

    Returns SimpleNamespace(SharedTiles, n_tiles, emit); emit(lds, pA, pB, pC, pScaleC, pScaleA, pScaleB, pSortedIds,
    num_tokens, num_blocks, active, coords, rows=...) runs the task when `active` (an i1 or Python bool), with
    coords() giving its (block, N tile, expert) and lds a SharedTiles allocation.
    swiglu_limit > 0: SiLU(min(gate, limit)) * clamp(up, -limit, limit), as DeepSeek-V4 and aiter's silu_clamped.
    rows: the block's valid rows (its first rows; wave uniform). The MFMAs of the 64-row A packets past them are
    skipped (a uniform branch per stage).
    """
    validate_parameters(inter_dim, hidden_dim, topk, num_experts)
    I, K = inter_dim, hidden_dim
    KB = K // TILE_K
    assert (
        KB <= 64
    ), "B scales of a side's K blocks must fit in one VGPR (hidden_dim <= 8192)"
    # fp8 B scales: both sides in one VGPR (lane side * KB + K block) when they fit, else one VGPR per side.
    split_scale_b = 2 * KB > 64
    n_tiles = I // TILE_N
    b_k_bytes = TILE_K * 16
    scale_rows_b = 2 * I // 128
    # wq "fp4": W in a8w4's fd2 form (util.load_fd2). Each loader thread reads the fd2 data
    # of the 16 B unit it would load in fp8 and decodes it (group scale folded in) right
    # before the LDS write; the per-channel exponent is applied in the epilogue.
    FP4 = wq == "fp4"
    assert wq in ("fp8", "fp4"), wq
    # 16-row M / N slices per wave; N slices are gate first, then the matching up slices.
    m_slices = 8
    n_slices = 32 // m_slices
    gate_slices = n_slices // 2
    side_rows = [0, I]

    def acc_index(m_slice, n_slice):
        return m_slice * n_slices + n_slice

    def stage_pairs(stage):
        """(m_slice, [n_slice, n_slice], side) of the four dequant pairs of a stage, in issue order."""
        return [
            (stage * 2 + i % 2, [i // 2 * 2, i // 2 * 2 + 1], i // 2) for i in range(4)
        ]

    @fx.struct
    class SharedTiles:
        # A tile in dwords [0, 4 packets), B tile after it.
        ab: fx.Array[fx.Int32, 8 * PACKET_DWORDS, 16]

    B_BASE = 4 * PACKET_DWORDS
    # Per-row |x| maxima of each wave along N, 4 dwords per row after the FP8 tile (A area).
    ROW_MAX_BASE = B_BASE
    col_waves = 4

    @flyc.jit
    def emit_task(
        lds,
        pA,
        pB,
        pC,
        pScaleC,
        pScaleA,
        pScaleB,
        pSortedIds,
        num_tokens,
        num_blocks,
        active,
        coords,
        tid=None,
        epilog_hook=None,
        kloop_hook=None,
        rows=None,
    ):
        if const_expr(tid is None):
            tid = fx.thread_idx.x
        lane = tid % 64
        wave = tid // 64
        group = _uniform(tid // 256)
        wave_m = group
        wave_n = _uniform(wave % 4)
        rsrc_of, load_i32 = _rsrc_of, _load_i32

        if active:
            sorted_rsrc = rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))

            def route_ids(block, rows):
                return [
                    load_i32(sorted_rsrc, (block * SORT_BLOCK_M + row) * 4)
                    for row in rows
                ]

            def decode(fused):
                """(token, slot, valid) of a sorted ID."""
                token = fused & TOKEN_MASK
                slot = (fused >> 24) & 255
                return token, slot, (token < num_tokens) & (slot < topk)

            # A slice j covers block rows slice_rows[j] + lane % 16, in LDS packet slice_packet[j].
            slice_packet = [j // 2 for j in range(8)]
            slice_offset = [j % 2 * 32 for j in range(8)]
            slice_rows = [
                slice_packet[j] * PACKET_ROWS + slice_offset[j] + wave_m * 16
                for j in range_constexpr(m_slices)
            ]

            copy_row = tid // 8
            copy_col = tid % 8

            def copy_rows():
                return [
                    tid // 8 + copy_packet * PACKET_ROWS
                    for copy_packet in range_constexpr(4)
                ]

            def slice_lane_rows():
                return [slice_rows[j] + tid % 16 for j in range_constexpr(m_slices)]

            def task_offsets(a_ids, sa_ids):
                """A copy and scale_a byte offsets of a block's rows."""
                a_offsets, sa_offsets = [], []
                for fused in a_ids:
                    token, _, valid = decode(fused)
                    a_offsets.append(
                        valid.select(token * K + tid % 8 * 16, fx.Int32(OOB_OFFSET))
                    )
                for fused in sa_ids:
                    token, _, valid = decode(fused)
                    sa_offsets.append(valid.select(token * 4, fx.Int32(OOB_OFFSET)))
                return a_offsets, sa_offsets

            a_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(pA, fx.make_layout(1, 1)),
                max_size=False,
                num_records_bytes=num_tokens * K,
            )
            a_packet = BufferTensor(
                fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1))
            )
            scale_a_rsrc = rsrc_of(pScaleA, num_tokens * (KB * 4))
            scale_b_rsrc = rsrc_of(pScaleB, num_experts * scale_rows_b * (KB * 4))
            fp8_rsrc = rsrc_of(pC, num_blocks * (SORT_BLOCK_M * I))
            fp8_scale_rsrc = rsrc_of(
                pScaleC, num_blocks * (SORT_BLOCK_M * (I // 128) * 4)
            )

            def task_scale_b(expert, n_tile):
                """B scale of (side, K block) in lane side * KB + block (split_scale_b: [side] in lane block)."""
                scale_lane = tid % 64
                gate_scale_row = expert * scale_rows_b + n_tile
                if const_expr(split_scale_b):
                    return [
                        fx.Float32(
                            buffer_load(
                                T.f32,
                                scale_b_rsrc,
                                (scale_lane < KB).select(
                                    (gate_scale_row + side * (I // 128)) * (KB * 4)
                                    + scale_lane * 4,
                                    fx.Int32(OOB_OFFSET),
                                ),
                            )
                        )
                        for side in range_constexpr(2)
                    ]
                scale_side = scale_lane // KB
                scale_b_offset = (gate_scale_row + scale_side * (I // 128)) * (
                    KB * 4
                ) + (scale_lane - scale_side * KB) * 4
                return fx.Float32(
                    buffer_load(
                        T.f32,
                        scale_b_rsrc,
                        (scale_lane < 2 * KB).select(
                            scale_b_offset, fx.Int32(OOB_OFFSET)
                        ),
                    )
                )

            # LDS rows [16 g, 16 g + 16) of B packet q hold this 16-row weight group of side q // 2, so
            # each wave's N slices are adjacent gate (and up) columns.
            loader = wave // 2
            weight_groups = [loader * 2 + q % 2 for q in range(4)]
            b_voffsets = [
                (side_rows[q // 2] + weight_groups[q] * 16) * K
                + wave % 2 * 1024
                + lane * 16
                for q in range_constexpr(4)
            ]
            if const_expr(FP4):
                # The packet's wave-uniform part of both offsets goes in soffset (VGPR budget).
                fd2_voffsets = fd2_lane_offsets(wave, lane)
                loader_s = _uniform(loader)
                w_soffsets = [
                    (side_rows[q // 2] + (loader_s * 2 + q % 2) * 16) * (K // 2)
                    for q in range_constexpr(4)
                ]
                # scale dword (32-row pair, 256-K block, lane) of the task's rows; its
                # byte (k % 2) * 2 + 16-row group % 2, and group % 2 is q % 2
                s_soffsets = [
                    (side_rows[q // 2] // 32 + loader_s) * K for q in range_constexpr(4)
                ]
            b_write = _lds_dwords(loader * 16 + lane % 16, wave % 2 * 4 + lane // 16)
            a_write = _lds_dwords(copy_row, copy_col)
            # Per (distinct packet-relative row, half_k) read addresses; the packet goes in the immediate offset.
            a_read = [
                [
                    _lds_dwords(r * 32 + wave_m * 16 + lane % 16, lane // 16 + h * 4)
                    for h in range(2)
                ]
                for r in range(2)
            ]
            b_read = [
                _lds_dwords(wave_n * 16 + lane % 16, lane // 16 + h * 4)
                for h in range(2)
            ]

            def lds_view(dwords, count=4):
                return fx.make_view(lds.ab.ptr + dwords, fx.make_layout(count, 1))

            def read_a_slice(j, half_k):
                return lds_view(
                    a_read[j % 2][half_k] + slice_packet[j] * PACKET_DWORDS
                ).load()

            def read_b_slice(n_slice, half_k):
                return lds_view(
                    b_read[half_k] + (B_BASE + n_slice * PACKET_DWORDS)
                ).load()

            def task_ops(n_tile, expert, a_offsets, sa_offsets, sb_lanes):
                """Global loads and B scales of one task."""
                w_dwords = 2 * I * K // (8 if FP4 else 4)
                b_tensor = fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        pB
                        + (
                            fx.Int64(expert) * w_dwords
                            + fx.Int64(n_tile) * (TILE_N * w_dwords // (2 * I))
                        ),
                        fx.make_layout(1, 1),
                    ),
                    max_size=False,
                    num_records_bytes=(I + TILE_N) * K // (2 if FP4 else 1),
                )
                b_packet = BufferTensor(
                    fx.make_view(
                        fx.get_iter(b_tensor), fx.make_layout(2 if FP4 else 4, 1)
                    )
                )
                if const_expr(FP4):
                    # scale dwords from the task's first 32-row pair on
                    sb_rsrc = rsrc_of(
                        pScaleB
                        + (fx.Int64(expert) * (2 * I) + fx.Int64(n_tile) * TILE_N)
                        // 32
                        * (K // 4),
                        (I + TILE_N) * K // 32,
                    )

                def load_a(k_block, packet):
                    return a_packet.load(
                        voffset_bytes=a_offsets[packet], soffset_bytes=k_block * TILE_K
                    )

                def load_b(k_block, packet):
                    if const_expr(not FP4):
                        return b_packet.load(
                            voffset_bytes=b_voffsets[packet],
                            soffset_bytes=k_block * b_k_bytes,
                        )
                    return load_fd2(
                        b_packet,
                        sb_rsrc,
                        fd2_voffsets,
                        (w_soffsets[packet], s_soffsets[packet]),
                        k_block,
                    )

                def load_scale_a(k_block, j):
                    value = buffer_load(
                        T.f32, scale_a_rsrc, sa_offsets[j], k_block * num_tokens * 4
                    )
                    return Vec.from_elements([fx.Float32(value)], fx.Float32)

                def scale_b(side, k_block):
                    if const_expr(FP4):  # folded into the decode and the epilogue
                        return fx.Float32(1.0)
                    if const_expr(split_scale_b):
                        return readlane_f32(sb_lanes[side], k_block)
                    return readlane_f32(sb_lanes, fx.Int32(side * KB) + k_block)

                return SimpleNamespace(
                    load_a=load_a,
                    load_b=load_b,
                    load_scale_a=load_scale_a,
                    scale_b=scale_b,
                )

            def barrier():
                _sched_fence()
                gpu.barrier()
                _sched_fence()

            # The streamed operand (A) stages all four packets of the next block, the held one (B) two slots.
            s_stage = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Int32)
            h_stage = fx.make_rmem_tensor(fx.make_layout((4, 2), (1, 4)), fx.Int32)
            # held column slice * 2 + half_k for the four held slices of the current block.
            held = fx.make_rmem_tensor(fx.make_layout((4, 8), (1, 4)), fx.Int32)
            # scale_a column (stage parity) * 2 + slice-in-stage.
            scale_a = fx.make_rmem_tensor(fx.make_layout((1, 8), (1, 1)), fx.Float32)
            accum = fx.make_rmem_tensor(fx.make_layout((4, 32), (1, 4)), fx.Float32)
            partial = fx.make_rmem_tensor(fx.make_layout((4, 4), (1, 4)), fx.Float32)
            partial_scale = fx.make_rmem_tensor(
                fx.make_layout((1, 2), (1, 1)), fx.Float32
            )

            def load_s(ops, k_block, packet):
                s_stage[None, packet].store(ops.load_a(k_block, packet))

            def load_h(ops, k_block, packet, slot):
                h_stage[None, slot].store(ops.load_b(k_block, packet))

            def write_s(packet):
                lds_view(a_write + packet * PACKET_DWORDS).store(
                    s_stage[None, packet].load()
                )

            def write_h(packet, value, k_parity=0):
                """k_parity: parity of the K block of `value` (fp4 picks its scale byte)."""
                lds_view(b_write + (B_BASE + packet * PACKET_DWORDS)).store(
                    decode_b(value, (k_parity * 2 + packet % 2) * 0x01010101)
                )

            def decode_b(value, selector):
                if const_expr(not FP4):
                    return value
                return decode_fd2(value, selector)

            def read_held():
                # The four N slices of B.
                for held_index in range_constexpr(4):
                    for held_half in range_constexpr(2):
                        value = read_b_slice(held_index, held_half)
                        held[None, held_index * 2 + held_half].store(value)

            def held_word(index, half_k, word):
                return Vec(held[None, index * 2 + half_k].load()).bitcast(fx.Int64)[
                    word
                ]

            last_pair = stage_pairs(3)[3]

            def retire_element(pair, slot, index):
                """Element `index` (of 8) of `pair`'s partials (slots 2 slot + {0, 1}, scale `slot`) into accum."""
                target = accum[None, acc_index(pair[0], pair[1][index // 4])]
                element = index % 4
                target[element] = fx.fma(
                    Vec(partial[None, slot * 2 + index // 4].load())[element],
                    Vec(partial_scale[None, slot].load())[0],
                    target[element],
                )

            def issue_pairs(ops, k_block, stage, streamed, previous):
                """The MFMAs of a stage's four pairs, each retiring the pair before it."""
                for pair_id, (m_slice, n_pair, side) in enumerate(stage_pairs(stage)):
                    # Every pair of a stage alternates partial slots: entries pair_id % 2 * 2 + {0, 1}.
                    base = pair_id % 2 * 2
                    for k_step in range_constexpr(4):
                        half_k, word = divmod(k_step, 2)
                        for inner in range_constexpr(2):
                            mfma_index = k_step * 2 + inner
                            n_slice = n_pair[inner]
                            a_value = streamed[m_slice % 2][half_k][word]
                            b_value = held_word(n_slice, half_k, word)
                            mfma_fp8(
                                partial[None, base + inner],
                                b_value,
                                a_value,
                                k_step > 0,
                            )
                            # Keep MFMA -> FMA order: the FMA's partial then has two MFMAs of XDL latency cover.
                            _sched_fence()
                            if const_expr(mfma_index == 0):
                                a_scale = Vec(
                                    scale_a[None, stage % 2 * 2 + m_slice % 2].load()
                                )[0]
                                partial_scale[None, pair_id % 2].store(
                                    Vec.from_elements(
                                        [a_scale * ops.scale_b(side, k_block)],
                                        fx.Float32,
                                    )
                                )
                            # Retire one element of the previous pair under each MFMA.
                            retire_element(previous, (pair_id + 1) % 2, mfma_index)
                            _sched_fence()
                    previous = (m_slice, n_pair, side)

            def retire(pair, scale=None, active=None):
                """`pair` (partial slots 2, 3, scale 1 unless `scale`) into accum, in no MFMA shadow; with `active`,
                only where it holds (the accumulators keep their values elsewhere)."""
                last_m, last_n = pair[0], pair[1]
                final_scale = (
                    Vec(partial_scale[None, 1].load())[0] if scale is None else scale
                )
                for inner in range_constexpr(2):
                    target = accum[None, acc_index(last_m, last_n[inner])]
                    values = Vec(partial[None, 2 + inner].load())
                    for element in range_constexpr(4):
                        kept = target[element]
                        added = fx.fma(values[element], final_scale, kept)
                        target[element] = (
                            added if active is None else active.select(added, kept)
                        )

            def retire_tail():
                """Past the valid rows no MFMA retires the last pair of the last stage with valid rows; retire it,
                then zero its slots, which the next pair retires as the skipped last pair.
                """
                if rows <= 3 * PACKET_ROWS:
                    # Every candidate gets the FMAs: branches that differ only in the target become a dynamic accum
                    # index (scratch) in LLVM. Stages 0 and 1 hold valid rows when they lie below the tail, so they
                    # keep their values by a select: a scale of 0 would still add a NaN or inf partial (of the tail's
                    # rows). Stage 2 is the tail or holds only padding rows, so a scale of 0 does there.
                    scale = Vec(partial_scale[None, 1].load())[0]
                    for stage in range_constexpr(3):
                        tail = (rows > stage * PACKET_ROWS) & (
                            rows <= (stage + 1) * PACKET_ROWS
                        )
                        if const_expr(stage < 2):
                            retire(stage_pairs(stage)[3], scale, active=tail)
                        else:
                            retire(stage_pairs(stage)[3], tail.select(scale, 0.0))
                    for inner in range_constexpr(2):
                        partial[None, 2 + inner].store(Vec.filled(4, 0.0, fx.Float32))

            def run_k(ops, parity, k_block):
                # The last block re-fetches clamped blocks so it stays inside the loop body.
                k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
                retire_tail()
                previous = last_pair
                for stage in range_constexpr(4):
                    if const_expr(stage == 0):
                        write_s(3)
                    else:
                        write_s(stage - 1)
                        # the held packets are K block k_next (unused when clamped)
                        write_h(
                            stage - 1,
                            h_stage[None, (stage - 1) % 2].load(),
                            1 - parity,
                        )
                        if const_expr(stage == 3):
                            write_h(3, h_stage[None, 1].load(), 1 - parity)
                    _sched_fence()
                    if const_expr(stage == 0):
                        read_held()
                    # Streamed A slices 2s, 2s+1 of this stage.
                    streamed = [
                        [
                            Vec(read_a_slice(stage * 2 + i, h)).bitcast(fx.Int64)
                            for h in range_constexpr(2)
                        ]
                        for i in range_constexpr(2)
                    ]
                    _sched_fence()
                    for next_i in range_constexpr(2):
                        if const_expr(stage < 3):
                            scale_a[None, (stage + 1) % 2 * 2 + next_i].store(
                                ops.load_scale_a(k_block, (stage + 1) * 2 + next_i)
                            )
                        else:
                            scale_a[None, next_i].store(
                                ops.load_scale_a(k_next, next_i)
                            )
                    load_s(ops, k_next, stage)
                    if const_expr(stage < 3):
                        load_h(ops, k_next, stage, stage % 2)
                    if const_expr(stage == 2):
                        load_h(ops, k_next, 3, 1)
                    _sched_fence()
                    rocdl.s_waitcnt(lgkmcnt=0)
                    barrier()
                    set_priority(3)
                    if const_expr(stage > 0):
                        # A packet `stage` holds block rows [64 stage, 64 stage + 64).
                        if rows > stage * PACKET_ROWS:
                            issue_pairs(ops, k_block, stage, streamed, previous)
                    else:
                        issue_pairs(ops, k_block, stage, streamed, previous)
                    previous = stage_pairs(stage)[3]
                    set_priority(0)
                    barrier()

            def k_loop(ops):
                if const_expr(KB % 2):
                    run_k(ops, 0, fx.Int32(0))
                for pair in range(KB // 2):
                    k_block = fx.Int32(pair) * 2 + KB % 2
                    run_k(ops, KB % 2, k_block)
                    run_k(ops, 1 - KB % 2, k_block + 1)

            def start_k_loop():
                accum.fill(0)
                # The first stage retires these zeros as the "previous" pair.
                for pro_index in range_constexpr(4):
                    partial[None, pro_index].store(Vec.filled(4, 0.0, fx.Float32))
                for pro_index in range_constexpr(2):
                    partial_scale[None, pro_index].store(Vec.filled(1, 0.0, fx.Float32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                # Waves 4-7 trail by one barrier so their memory phase overlaps waves 0-3 compute.
                if group == 1:
                    barrier()

            def finish_k_loop():
                # The last pair has no later MFMA shadow.
                retire_tail()
                retire(last_pair)
                if group == 0:
                    barrier()

            def silu_mul(gate_values, up_values):
                result = []
                for element in range_constexpr(4):
                    gate = gate_values[element]
                    up = up_values[element]
                    if const_expr(swiglu_limit > 0):
                        # Selects, not minnumf / maxnumf, which would turn a NaN into the limit.
                        limit = fx.Float32(swiglu_limit)
                        gate = (gate > limit).select(limit, gate)
                        up = (up > limit).select(
                            limit, (up < -limit).select(-limit, up)
                        )
                    exponent = fx.exp2(gate * -LOG2E, fastmath="afn")
                    with fx.fastmath("afn"):  # v_rcp_f32, not an IEEE division
                        inverse = fx.Float32(1.0) / (exponent + 1.0)
                    result.append(gate * inverse * up)
                return result

            def row_max(row):
                """The row's amax from the column waves' partials (``_abs_bits`` maxima)."""
                partial = Vec(lds_view(ROW_MAX_BASE + row * 4, col_waves).load())
                peak = fx.Int32(partial[0])
                for wave_col in range_constexpr(1, col_waves):
                    peak = fx.max(peak, fx.Int32(partial[wave_col]))
                return peak.bitcast(fx.Float32)

            def fp8_epilogue(block, n_tile):
                # Padding rows (zero A rows) are written too: FP8 zeros with scale 0.
                finish_k_loop()
                values = [
                    [
                        silu_mul(
                            acc_values(m_slice, gate),
                            acc_values(m_slice, gate_slices + gate),
                        )
                        for gate in range_constexpr(gate_slices)
                    ]
                    for m_slice in range_constexpr(m_slices)
                ]
                for m_slice in range_constexpr(m_slices):
                    peak = fx.Int32(0)
                    for gate_values in values[m_slice]:
                        for value in gate_values:
                            peak = fx.max(peak, _abs_bits(value))
                    # Lanes l, l ^ 16, l ^ 32, l ^ 48 hold this wave's other columns of row l % 16. shuffle_xor's
                    # ds_swizzle raises the task loop's scratch ops from 2 to 9, and an unmasked index (lane is
                    # opaque) keeps shuffle_idx's bounds check: each costs ~1% at 64 tokens per rank (wq fp4).
                    for shift in (16, 32):
                        peak = fx.max(
                            peak,
                            fx.Int32(fx.gpu.shuffle_idx(peak, (lane ^ shift) & 63, 64)),
                        )
                    if lane < 16:
                        lds_view(
                            ROW_MAX_BASE + (slice_rows[m_slice] + lane) * 4 + wave_n, 1
                        ).store(Vec.from_elements([peak], fx.Int32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                for m_slice in range_constexpr(m_slices):
                    row = slice_rows[m_slice] + lane % 16
                    amax = row_max(row)
                    # |x| <= amax keeps x * 240 / amax within the FP8 range after rounding.
                    with fx.fastmath("afn"):  # v_rcp_f32, not an IEEE division
                        reciprocal = fx.Float32(1.0) / amax
                    inverse = (amax > 0.0).select(
                        reciprocal * FP8_MAX,
                        fx.Float32(0.0),
                    )
                    for gate in range_constexpr(gate_slices):
                        q = [
                            values[m_slice][gate][element] * inverse
                            for element in range_constexpr(4)
                        ]
                        # cvt_pk_fp8_f32 is unstable (not in fx.rocdl.__all__); the stable conversion,
                        # Vector.to(fx.Float8E4M3FNUZ), does not lower on FlyDSL 0.5.2 (arith.truncf has no LLVM
                        # translation; gfx942 compile-only probe).
                        low = rocdl.cvt_pk_fp8_f32(
                            T.i32, q[0], q[1], fx.Int32(0), False
                        )
                        word = fx.Int32(
                            rocdl.cvt_pk_fp8_f32(T.i32, q[2], q[3], low, True)
                        )
                        # 4 FP8 columns 4 g .. 4 g + 3 of the 16-column chunk; lane group g = lane // 16.
                        lds_view(
                            _lds_dwords(row, wave_n * gate_slices + gate) + lane // 16,
                            1,
                        ).store(Vec.from_elements([word], fx.Int32))
                rocdl.s_waitcnt(lgkmcnt=0)
                barrier()
                for part in range_constexpr(4):
                    out_row = part * 64 + tid // 8
                    buffer_store(
                        Vec(lds_view(_lds_dwords(out_row, tid % 8)).load()).ir_value(),
                        fp8_rsrc,
                        (block * SORT_BLOCK_M + out_row) * I
                        + n_tile * TILE_N
                        + tid % 8 * 16,
                        aux=STORE_AUX,
                    )
                scale_row = tid % SORT_BLOCK_M
                scale_offset = (
                    (n_tile * num_blocks + block) * SORT_BLOCK_M + scale_row
                ) * 4
                buffer_store(
                    (row_max(scale_row) * (1.0 / FP8_MAX)).ir_value(),
                    fp8_scale_rsrc,
                    (tid < SORT_BLOCK_M).select(scale_offset, fx.Int32(OOB_OFFSET)),
                    aux=STORE_AUX,
                )

            block, n_tile, expert = coords()

            def acc_values(m_slice, n_slice):
                values = Vec(accum[None, acc_index(m_slice, n_slice)].load())
                if const_expr(not FP4):
                    return values
                return [values[e] * factors[n_slice][e] for e in range_constexpr(4)]

            a_offsets, sa_offsets = task_offsets(
                route_ids(block, copy_rows()), route_ids(block, slice_lane_rows())
            )
            ops = task_ops(
                n_tile,
                expert,
                a_offsets,
                sa_offsets,
                None if FP4 else task_scale_b(expert, n_tile),
            )
            for pro_packet in range_constexpr(4):
                load_s(ops, 0, pro_packet)
            for pair_begin in range_constexpr(0, 4, 2):
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    load_h(ops, 0, pro_packet, pro_packet % 2)
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    write_h(pro_packet, h_stage[None, pro_packet % 2].load())
                _sched_fence()
            for pro_slice in range_constexpr(2):
                scale_a[None, pro_slice].store(ops.load_scale_a(0, pro_slice))
            for pro_packet in range_constexpr(4):
                write_s(pro_packet)
            start_k_loop()
            if const_expr(kloop_hook is not None):
                kloop_hook()
            k_loop(ops)
            if const_expr(FP4):
                # after the K loop, which has no register left; N slice s is side
                # s // 2, 16-row group wave_n * 2 + s % 2
                factors = fd2_channel_factors(
                    pScaleB,
                    rows=num_experts * 2 * I,
                    K=K,
                    first_columns=[
                        expert * (2 * I)
                        + side_rows[n_slice // gate_slices]
                        + n_tile * TILE_N
                        + (wave_n * 2 + n_slice % 2) * 16
                        + lane // 16 * 4
                        for n_slice in range_constexpr(n_slices)
                    ],
                )
            if const_expr(epilog_hook is not None):
                epilog_hook()
            fp8_epilogue(block, n_tile)

    return SimpleNamespace(SharedTiles=SharedTiles, n_tiles=n_tiles, emit=emit_task)
