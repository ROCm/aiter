# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx942 FP8 block-scale MoE Down (GEMM2, BM256 x BN256) on a 2x4 wave grid, A and B staged through LDS: the
mega kernel's Down task emitter.

Per route y = routing weight * (A2 row @ W2[expert]^T): A2 is FP8 [blocks * 256, I] in sorted-row order (scales
[I/128, blocks * 256]), so a block's rows are contiguous; W2 is aiter shuffle_weight(16,16) of [E, H, I] with FP32
scales [E, H/128, I/128]. The K loop is GateUp's: 2 waves along M (128 rows) x 4 along N (64 columns), B held, A
streamed; one workgroup takes a 256-row sorted block and 256 hidden columns.
y is added into its token's row of the source rank's out [Tcap, H] (BF16, zeroed beforehand) with
global_atomic_pk_add_bf16; each wave transposes its tile through LDS so that one atomic covers two rows x 128
contiguous bytes.
With wq="fp4", W2 is a8w4's MXFP4 fd2 form instead, decoded to FP8 on its way into LDS.
"""

import functools
from types import SimpleNamespace

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec

from ..hw import _sched_fence
from ..hw import _u as _uniform
from .gateup import (
    OOB_OFFSET,
    PACKET_DWORDS,
    PACKET_ROWS,
    SORT_BLOCK_M,
    TILE_K,
    TOKEN_MASK,
    _lds_dwords,
)
from .util import (
    BufferTensor,
    _load_i32,
    _rsrc_of,
    buffer_load,
    decode_fd2,
    fd2_channel_factors,
    fd2_lane_offsets,
    load_fd2,
    mfma_fp8,
    pack_bf16x2_rne,
    readlane_f32,
    set_priority,
)

TILE_N = 256
# Atomic epilogue transpose tile per wave: 16 rows x 32 dwords; a 34-dword pitch keeps writes and reads conflict free.
TRANSPOSE_PITCH = 34


def validate_parameters(hidden_dim, inter_dim, topk, num_experts):
    if hidden_dim <= 0 or hidden_dim % TILE_N:
        raise ValueError("hidden_dim must be a positive multiple of 256")
    if inter_dim < TILE_K or inter_dim % TILE_K or inter_dim > 64 * TILE_K:
        raise ValueError("inter_dim must be a multiple of 128 in [128, 8192]")
    if not 0 < topk < 256 or num_experts <= 0:
        raise ValueError("require 0 < topk < 256 and num_experts > 0")


@functools.cache
def down_task_emitter(
    hidden_dim, inter_dim, topk, num_experts, remote_token_shift, wq="fp8", wide=False
):
    """One Down task (256 sorted rows x 256 hidden columns) for use inside a kernel.

    Returns SimpleNamespace(SharedTiles, n_tiles, emit); emit(lds, pA, pB, pOut, pScaleA, pScaleB, pSortedIds,
    pSortedWeights, num_tokens, num_blocks, active, coords, rows=...) runs the task when `active` (an i1 or Python
    bool), with coords() giving its (block, N tile, expert) and lds a SharedTiles allocation. pOut is an int64 table
    of the ranks' out bases; a sorted ID's token holds its source rank above bit remote_token_shift.
    rows: the block's valid rows (its first rows; wave uniform). The MFMAs of the 64-row A packets past them are
    skipped (a uniform branch per stage).
    wide: pA (a2) of 2 GiB or more, read through a descriptor per block (from a 64-bit base).
    """
    validate_parameters(hidden_dim, inter_dim, topk, num_experts)
    assert remote_token_shift > 0
    H, K = hidden_dim, inter_dim
    KB = K // TILE_K
    n_tiles = H // TILE_N
    b_k_bytes = TILE_K * 16
    scale_rows_b = H // 128
    # wq "fp4": W2 in a8w4's fd2 form, decoded right before the LDS write (see
    # gateup_task_emitter); the per-channel exponent joins the routing weight.
    FP4 = wq == "fp4"
    assert wq in ("fp8", "fp4"), wq
    # 16-row M / N slices per wave.
    m_slices = 8
    n_slices = 32 // m_slices
    # 64-column groups of a wave: one LDS transpose each.
    col_groups = n_slices // 4
    wave_bytes = n_slices * 32

    def acc_index(m_slice, n_slice):
        return m_slice * n_slices + n_slice

    def stage_pairs(stage):
        """(m_slice, [n_slice, n_slice]) of the four dequant pairs of a stage, in issue order."""
        return [(stage * 2 + i % 2, [i // 2 * 2, i // 2 * 2 + 1]) for i in range(4)]

    def weight_group(packet, loader):
        """16-row group of the N tile held in LDS rows [16 loader, 16 loader + 16) of B packet `packet`: waves read
        N slice s from packet s, rows [16 wave_n, +16), so a wave's N slices are its contiguous columns.
        """
        return loader * 4 + packet

    @fx.struct
    class SharedTiles:
        # A tile in dwords [0, 4 packets), B tile after it.
        ab: fx.Array[fx.Int32, 8 * PACKET_DWORDS, 16]

    B_BASE = 4 * PACKET_DWORDS

    @flyc.jit
    def emit_task(
        lds,
        pA,
        pB,
        pOut,
        pScaleA,
        pScaleB,
        pSortedIds,
        pSortedWeights,
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
        load_i32 = _load_i32

        if active:
            sorted_rsrc = _rsrc_of(pSortedIds, num_blocks * (SORT_BLOCK_M * 4))
            weight_rsrc = _rsrc_of(pSortedWeights, num_blocks * (SORT_BLOCK_M * 4))
            a2_rows = num_blocks * SORT_BLOCK_M
            block, n_tile, expert = coords()

            def route_ids(rows):
                return [
                    load_i32(sorted_rsrc, (block * SORT_BLOCK_M + row) * 4)
                    for row in rows
                ]

            def decode(fused):
                """(token, valid) of a sorted ID."""
                token = fused & TOKEN_MASK
                slot = (fused >> 24) & 255
                return token, (token < num_tokens) & (slot < topk)

            # A slice j covers block rows slice_rows[j] + lane % 16, in LDS packet slice_packet[j].
            slice_packet = [j // 2 for j in range(8)]
            slice_offset = [j % 2 * 32 for j in range(8)]
            slice_rows = [
                slice_packet[j] * PACKET_ROWS + slice_offset[j] + wave_m * 16
                for j in range_constexpr(m_slices)
            ]
            lane_rows = [slice_rows[j] + lane % 16 for j in range_constexpr(m_slices)]
            copy_row = tid // 8
            copy_col = tid % 8

            a_offsets, sa_offsets = [], []
            copy_rows = [
                copy_row + copy_packet * PACKET_ROWS
                for copy_packet in range_constexpr(4)
            ]
            for row, fused in zip(copy_rows, route_ids(copy_rows)):
                _, a_valid = decode(fused)
                a_row = row if wide else block * SORT_BLOCK_M + row
                a_offsets.append(
                    a_valid.select(a_row * K + copy_col * 16, fx.Int32(OOB_OFFSET))
                )
            for row, fused in zip(lane_rows, route_ids(lane_rows)):
                _, sa_valid = decode(fused)
                sa_row = block * SORT_BLOCK_M + row
                sa_offsets.append(sa_valid.select(sa_row * 4, fx.Int32(OOB_OFFSET)))

            if const_expr(wide):
                # a2 from the block's first row on (pA holds i32 words)
                a_tensor = fx.rocdl.make_buffer_tensor(
                    fx.make_view(
                        pA + fx.Int64(block * SORT_BLOCK_M) * (K // 4),
                        fx.make_layout(1, 1),
                    ),
                    max_size=False,
                    num_records_bytes=SORT_BLOCK_M * K,
                )
            else:
                a_tensor = fx.rocdl.make_buffer_tensor(
                    fx.make_view(pA, fx.make_layout(1, 1)),
                    max_size=False,
                    num_records_bytes=a2_rows * K,
                )
            a_packet = BufferTensor(
                fx.make_view(fx.get_iter(a_tensor), fx.make_layout(4, 1))
            )
            scale_a_rsrc = _rsrc_of(pScaleA, a2_rows * (KB * 4))
            w_dwords = H * K // (8 if FP4 else 4)
            b_tensor = fx.rocdl.make_buffer_tensor(
                fx.make_view(
                    pB
                    + (
                        fx.Int64(expert) * w_dwords
                        + fx.Int64(n_tile) * (TILE_N * w_dwords // H)
                    ),
                    fx.make_layout(1, 1),
                ),
                max_size=False,
                num_records_bytes=TILE_N * K // (2 if FP4 else 1),
            )
            b_packet = BufferTensor(
                fx.make_view(fx.get_iter(b_tensor), fx.make_layout(2 if FP4 else 4, 1))
            )
            # A wave's columns lie in one 128-column scale group: B scale of K block k in lane k.
            scale_group = n_tile * 2 + wave_n // 2
            scale_b_offset = ((expert * scale_rows_b + scale_group) * KB + lane) * 4
            if const_expr(not FP4):
                sb_lanes = fx.Float32(
                    buffer_load(
                        T.f32,
                        _rsrc_of(pScaleB, num_experts * scale_rows_b * (KB * 4)),
                        (lane < KB).select(scale_b_offset, fx.Int32(OOB_OFFSET)),
                    )
                )

            loader = wave // 2
            b_voffsets = [
                weight_group(q, loader) * 16 * K + wave % 2 * 1024 + lane * 16
                for q in range_constexpr(4)
            ]
            if const_expr(FP4):
                # The packet's wave-uniform part of both offsets goes in soffset (VGPR budget).
                fd2_voffsets = fd2_lane_offsets(wave, lane)
                loader_s = _uniform(loader)
                groups = [weight_group(q, loader_s) for q in range_constexpr(4)]
                w_soffsets = [group * 16 * (K // 2) for group in groups]
                # scale dwords from the task's first 32-row pair on
                sb_rsrc = _rsrc_of(
                    pScaleB
                    + (fx.Int64(expert) * H + fx.Int64(n_tile) * TILE_N)
                    // 32
                    * (K // 4),
                    TILE_N * K // 32,
                )
                s_soffsets = [group // 2 * K for group in groups]
                group_parities = [group % 2 for group in groups]
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

            def decode_b(value, packet, k_parity):
                """fp4: `value` of `packet`, K block parity `k_parity` -> 16 B of fp8."""
                if const_expr(not FP4):
                    return value
                return decode_fd2(
                    value, (group_parities[packet] + k_parity * 2) * 0x01010101
                )

            def load_scale_a(k_block, j):
                value = buffer_load(
                    T.f32, scale_a_rsrc, sa_offsets[j], k_block * a2_rows * 4
                )
                return Vec.from_elements([fx.Float32(value)], fx.Float32)

            def scale_b(k_block):
                if const_expr(FP4):  # folded into the decode and the epilogue
                    return fx.Float32(1.0)
                return readlane_f32(sb_lanes, k_block)

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

            def load_s(k_block, packet):
                s_stage[None, packet].store(load_a(k_block, packet))

            def load_h(k_block, packet, slot):
                h_stage[None, slot].store(load_b(k_block, packet))

            def write_s(packet):
                lds_view(a_write + packet * PACKET_DWORDS).store(
                    s_stage[None, packet].load()
                )

            def write_h(packet, value, k_parity=0):
                """k_parity: parity of the packet's K block (fp4 picks its scale byte)."""
                lds_view(b_write + (B_BASE + packet * PACKET_DWORDS)).store(
                    decode_b(value, packet, k_parity)
                )

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

            def issue_pairs(k_block, stage, streamed, previous, pair_ids):
                """The MFMAs of a stage's pairs `pair_ids`, each retiring the pair before it."""
                for pair_id in pair_ids:
                    m_slice, n_pair = stage_pairs(stage)[pair_id]
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
                                        [a_scale * scale_b(k_block)], fx.Float32
                                    )
                                )
                            # Retire one element of the previous pair under each MFMA.
                            retire_element(previous, (pair_id + 1) % 2, mfma_index)
                            _sched_fence()
                    previous = (m_slice, n_pair)

            def retire(pair, slot=1, scale=None, active=None):
                """`pair` (partial slots 2 slot + {0, 1}, scale `slot` unless `scale`) into accum, in no MFMA
                shadow; with `active`, only where it holds (the accumulators keep their values elsewhere).
                """
                last_m, last_n = pair
                final_scale = (
                    Vec(partial_scale[None, slot].load())[0] if scale is None else scale
                )
                for inner in range_constexpr(2):
                    target = accum[None, acc_index(last_m, last_n[inner])]
                    values = Vec(partial[None, slot * 2 + inner].load())
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
                            retire(stage_pairs(stage)[3], 1, scale, active=tail)
                        else:
                            retire(stage_pairs(stage)[3], 1, tail.select(scale, 0.0))
                    for inner in range_constexpr(2):
                        partial[None, 2 + inner].store(Vec.filled(4, 0.0, fx.Float32))

            def run_k(parity, k_block):
                # The last block re-fetches clamped blocks so it stays inside the loop body.
                k_next = (k_block + 1 < KB).select(k_block + 1, KB - 1)
                retire_tail()
                previous = last_pair
                for stage in range_constexpr(4):
                    # packet 3 is this K block's, the others K block k_next's (unused
                    # when clamped)
                    if const_expr(stage == 0):
                        write_s(3)
                    else:
                        write_s(stage - 1)
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
                                load_scale_a(k_block, (stage + 1) * 2 + next_i)
                            )
                        else:
                            scale_a[None, next_i].store(load_scale_a(k_next, next_i))
                    load_s(k_next, stage)
                    if const_expr(stage < 3):
                        load_h(k_next, stage, stage % 2)
                    if const_expr(stage == 2):
                        load_h(k_next, 3, 1)
                    _sched_fence()
                    rocdl.s_waitcnt(lgkmcnt=0)
                    barrier()
                    set_priority(3)
                    if const_expr(stage > 0):
                        # A packet `stage` holds block rows [64 stage, 64 stage + 64).
                        if rows > stage * PACKET_ROWS:
                            issue_pairs(k_block, stage, streamed, previous, range(4))
                    else:
                        issue_pairs(k_block, stage, streamed, previous, range(4))
                    previous = stage_pairs(stage)[3]
                    set_priority(0)
                    barrier()

            def finish_k_loop():
                # The last pair has no later MFMA shadow.
                retire_tail()
                retire(last_pair)
                # Pairs with group 1's last K-loop barrier: after it no wave reads the A/B tiles any more.
                if group == 0:
                    barrier()

            def weighted_words(m_slice, n_slice, weight):
                """BF16 of routing weight * the 4 consecutive columns this lane holds (row lane % 16)."""
                values = Vec(accum[None, acc_index(m_slice, n_slice)].load())
                if const_expr(not FP4):
                    return pack_bf16x2_rne(
                        [values[element] * weight for element in range_constexpr(4)]
                    )
                return pack_bf16x2_rne(
                    [
                        values[element] * (weight * factors[n_slice][element])
                        for element in range_constexpr(4)
                    ]
                )

            # Prologue: K block 0 into LDS.
            for pro_packet in range_constexpr(4):
                load_s(0, pro_packet)
            for pair_begin in range_constexpr(0, 4, 2):
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    load_h(0, pro_packet, pro_packet % 2)
                for pro_packet in range_constexpr(pair_begin, pair_begin + 2):
                    write_h(pro_packet, h_stage[None, pro_packet % 2].load())
                _sched_fence()
            for pro_slice in range_constexpr(2):
                scale_a[None, pro_slice].store(load_scale_a(0, pro_slice))
            for pro_packet in range_constexpr(4):
                write_s(pro_packet)
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
            if const_expr(kloop_hook is not None):
                kloop_hook()

            if const_expr(KB % 2):
                run_k(0, fx.Int32(0))
            for k_pair in range(KB // 2):
                k_block = fx.Int32(k_pair) * 2 + KB % 2
                run_k(KB % 2, k_block)
                run_k(1 - KB % 2, k_block + 1)

            if const_expr(FP4):
                # after the K loop, which has no register left, and before the
                # epilogue's asm atomics, which LLVM's vmcnt waits do not count
                factors = fd2_channel_factors(
                    pScaleB,
                    rows=num_experts * H,
                    K=K,
                    first_columns=[
                        expert * H
                        + n_tile * TILE_N
                        + (wave_n * 4 + n_slice) * 16
                        + lane // 16 * 4
                        for n_slice in range_constexpr(n_slices)
                    ],
                )
            if const_expr(epilog_hook is not None):
                epilog_hook()
            # Output rows are reloaded here (not kept through the K loop) so the loads overlap the drain.
            out_ids = route_ids(lane_rows)
            weights = [
                fx.Float32(
                    buffer_load(T.f32, weight_rsrc, (block * SORT_BLOCK_M + row) * 4)
                )
                for row in lane_rows
            ]

            def table_base(table_rsrc, rank_offset):
                return (
                    fx.Uint64(fx.Uint32(load_i32(table_rsrc, rank_offset + 4))) << 32
                ) | fx.Uint64(fx.Uint32(load_i32(table_rsrc, rank_offset)))

            token_offsets = []
            row_lo, row_hi = [], []
            for fused in out_ids:
                out_token, out_valid = decode(fused)
                # 64-bit base of the token's row in its rank's out; -1 offsets still mark invalid rows.
                table_rsrc = _rsrc_of(pOut, 64 * 8)
                rank_offset = out_valid.select(
                    (out_token >> remote_token_shift) * 8, fx.Int32(OOB_OFFSET)
                )
                base = table_base(table_rsrc, rank_offset)
                address = base + fx.Uint64(
                    fx.Uint32((out_token & ((1 << remote_token_shift) - 1)) * (H * 2))
                )
                row_lo.append(fx.Int32(fx.Uint32(address)))
                row_hi.append(fx.Int32(fx.Uint32(address >> 32)))
                token_offsets.append(out_valid.select(fx.Int32(0), fx.Int32(-1)))
            transpose = wave * (16 * TRANSPOSE_PITCH)
            write_base = transpose + lane % 16 * TRANSPOSE_PITCH + lane // 16 * 2
            read_base = transpose + lane // 32 * TRANSPOSE_PITCH + lane % 32
            column_base = n_tile * (TILE_N * 2) + wave_n * wave_bytes + lane % 32 * 4
            finish_k_loop()
            # LLVM does not count the asm atomics: retire every load first so no later vmcnt wait covers them.
            rocdl.s_waitcnt(vmcnt=0)
            for j in range_constexpr(m_slices):
                # Atomic i of a group covers rows 2i (lanes 0-31) and 2i + 1 (lanes 32-63) of slice j.
                row_offsets = [
                    fx.Int32(
                        fx.gpu.shuffle_idx(token_offsets[j], lane // 32 + 2 * i, 64)
                    )
                    for i in range_constexpr(8)
                ]
                row_addresses = []
                for i in range_constexpr(8):
                    source = lane // 32 + 2 * i
                    lo = fx.Uint32(fx.gpu.shuffle_idx(row_lo[j], source, 64))
                    hi = fx.Uint32(fx.gpu.shuffle_idx(row_hi[j], source, 64))
                    row_addresses.append(
                        ((fx.Uint64(hi) << 32) | fx.Uint64(lo))
                        + fx.Uint64(fx.Uint32(column_base))
                    )
                for c in range_constexpr(col_groups):
                    for n_local in range_constexpr(4):
                        lds_view(write_base + n_local * 8, 2).store(
                            Vec.from_elements(
                                weighted_words(j, c * 4 + n_local, weights[j]),
                                fx.Int32,
                            )
                        )
                    # One opaque copy of all 8 LDS reads: one wait, and LLVM cannot sink a read into its
                    # atomic's branch (which serializes 8 LDS round trips). Raw asm: FlyDSL 0.5.2 has no stable
                    # opaque value; without it the kernel has 125 more instructions (gfx942 compile-only probe).
                    reads = [
                        Vec(lds_view(read_base + 2 * i * TRANSPOSE_PITCH, 1).load())[
                            0
                        ].ir_value()
                        for i in range_constexpr(8)
                    ]
                    copies = llvm.inline_asm(
                        ir.Type.parse("!llvm.struct<(" + ", ".join(["i32"] * 8) + ")>"),
                        reads,
                        "",
                        ",".join(["=v"] * 8 + [str(i) for i in range(8)]),
                        has_side_effects=True,
                    )
                    for i in range_constexpr(8):
                        value = fx.Int32(llvm.extractvalue(T.i32, copies, [i]))
                        if row_offsets[i] >= 0:
                            # Raw asm: the rows are 64-bit addresses on 8 ranks, one per lane, so no wave-uniform
                            # buffer resource (fx.rocdl.BufferAtomicPkAdd) covers them, and fx.atomic_add on a
                            # <2 x bf16> pointer fails an alignment assertion in FlyDSL 0.5.2 (compile-only probe).
                            llvm.inline_asm(
                                None,
                                [
                                    (row_addresses[i] + fx.Uint64(c * 128)).ir_value(),
                                    value.ir_value(),
                                ],
                                "global_atomic_pk_add_bf16 $0, $1, off",
                                "v,v",
                                has_side_effects=True,
                            )

    return SimpleNamespace(SharedTiles=SharedTiles, n_tiles=n_tiles, emit=emit_task)
