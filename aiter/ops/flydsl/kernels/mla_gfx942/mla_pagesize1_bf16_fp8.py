# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""gfx942/gfx950 persistent page-size-one MLA: BF16 Q, native E4M3 KV.

KV is E4M3FNUZ on gfx942 and OCP E4M3FN on gfx950, the formats each arch's
cvt_pk_f32_fp8 decodes.

Rows are packed as query_position * num_heads + head. All eight waves reuse
one 32-token KV tile in LDS. During QK each wave owns a 16-row query tile;
during PV all waves process all query rows, owning 64 output columns each.

Work items and normalized split outputs follow get_mla_metadata_v1's contract.
The KV gather uses 64-bit addresses; other buffers have exact descriptor bounds.
"""

from functools import cache

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from aiter.ops.flydsl.kernels.kernels_common import LOG2E

QK_DIM = 576
V_DIM = 512
WORK_INFO_COLS = 8
WAVE_SIZE = 64
MFMA_SIZE = 16
WAVES = 8
BLOCK_THREADS = WAVES * WAVE_SIZE
KV_TILE_SIZE = 32
VECTOR_BYTES = 16
BF16_VECTOR_SIZE = VECTOR_BYTES // 2

# Coordinates are (element, lane_group, half, column_group). Each lane group
# reads 16 contiguous columns per group; one 16-byte LDS load feeds two MFMAs.
# The permuted reduction order cancels in the dot product only if Q follows the
# same layout; the Q loads spell it out as offsets.
QK_COLUMN_LAYOUT = ((8, 4, 2, 9), (1, 16, 8, 64))


def _buffer(ptr, dtype, layout):
    # The record bound is the layout's cosize: out-of-range loads return zero,
    # which the masked Q rows rely on. The 32-bit bound limits a view to 2 GiB.
    view = fx.make_view(fx.recast_iter(dtype, ptr), layout)
    return fx.Tensor(fx.rocdl.make_buffer_tensor(view, max_size=False))


def _round_to_bf16_finite(value):
    # These probabilities/normalized outputs are finite. Native .to(BFloat16)
    # adds NaN compare/or/select instructions per value on gfx942/0.5.0 and
    # regressed the cold/warm panel by ~2%. Keep the finite-only RNE conversion.
    # Add half an ULP minus one, plus the retained LSB: exact ties round to even.
    bits = value.bitcast(fx.Uint32)
    retained_lsb = (bits >> fx.Uint32(16)) & fx.Uint32(1)
    rounded = bits + fx.Uint32(0x7FFF) + retained_lsb
    return (rounded >> fx.Uint32(16)).to(fx.Uint16).bitcast(fx.BFloat16)


def _mfma(a, b, acc):
    mma = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
    operand_a = fx.make_rmem_tensor(4, fx.BFloat16)
    operand_b = fx.make_rmem_tensor(4, fx.BFloat16)
    accumulator = fx.make_rmem_tensor(4, fx.Float32)
    operand_a.store(a)
    operand_b.store(b)
    accumulator.store(acc)
    fx.gemm(mma, accumulator, operand_a, operand_b, accumulator)
    return accumulator.load()


def _flatten_state(state):
    """Loop-carried softmax/output/prefetch state as the flat list scf.for carries."""
    output = [fragment for row in state["output"] for fragment in row]
    return [state["row_max"], state["row_sum"]] + output + list(state["prefetch"])


def _unflatten_state(values, row_tiles, col_tiles):
    output_end = 2 + row_tiles * col_tiles
    return {
        "row_max": values[0],
        "row_sum": values[1],
        "output": [
            [values[2 + row * col_tiles + col] for col in range(col_tiles)]
            for row in range(row_tiles)
        ],
        "prefetch": list(values[output_end:]),
    }


def _pack_fp8_as_bf16(v):
    # Native unpack is FNUZ on gfx942 and OCP on gfx950; every finite value of
    # either E4M3 format is exactly BF16. Select the high halfwords directly:
    # a general .to(BFloat16) needlessly rounds these already-exact values.
    packed = fx.rocdl.perm_b32(
        v[1].bitcast(fx.Int32), v[0].bitcast(fx.Int32), fx.Int32(0x07060302)
    )
    return fx.Vector.from_elements([fx.Int32(packed)], dtype=fx.Int32).bitcast(
        fx.BFloat16
    )


def _defer_to_use(value):
    # No public fx operation expresses an opaque identity. A plain value lets
    # LLVM hoist K loads or epilogue addresses into the KV loop's live set.
    # The baseline's large-row paths need this boundary to avoid VGPR spills.
    from flydsl._mlir.dialects import llvm

    return fx.Int32(
        llvm.inline_asm(
            T.i32,
            [fx.Int32(value).ir_value()],
            "v_mov_b32 $0, $1",
            "=v,v",
            has_side_effects=True,
        )
    )


def _wave_uniform(value):
    # Scheduling metadata and wave ids are identical in all lanes of a wave.
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(value)))


def _mfma_fragment(v, start):
    return v.shuffle(v, [start, start + 1, start + 2, start + 3])


@cache
def _build(num_heads, max_seqlen_q, causal, write_final_lse):
    assert (
        1 <= num_heads <= 16
        and 1 <= max_seqlen_q <= 8
        and num_heads * max_seqlen_q <= 128
    )
    ROW_TILES = (num_heads * max_seqlen_q + MFMA_SIZE - 1) // MFMA_SIZE
    PV_COL_TILES = V_DIM // (WAVES * MFMA_SIZE)
    KV_ROW_STRIDE = QK_DIM + BF16_VECTOR_SIZE
    PROB_ROW_STRIDE = KV_TILE_SIZE + 8
    GATHER_BYTES = 48
    GATHER_LANES = QK_DIM // GATHER_BYTES
    GATHER_CHUNKS = GATHER_BYTES // VECTOR_BYTES
    GATHER_WAVES = KV_TILE_SIZE * GATHER_LANES // WAVE_SIZE
    QK_COLUMN_GROUPS = QK_DIM // WAVE_SIZE
    QK_STEPS = QK_COLUMN_GROUPS * 2
    # Output columns are (wave, column_tile, lane_group, element).
    OUTPUT_COLUMN_LAYOUT = ((WAVES, PV_COL_TILES, 4, 4), (V_DIM // WAVES, 16, 4, 1))
    STAT_ALPHA, STAT_MAX, STAT_SUM = range(3)

    @fx.struct
    class Shared:
        kv: fx.Array[fx.BFloat16, KV_TILE_SIZE * KV_ROW_STRIDE, 16]
        p: fx.Array[fx.BFloat16, ROW_TILES * 16 * PROB_ROW_STRIDE, 16]
        stats: fx.Array[fx.Float32, ROW_TILES * 16 * 3, 16]
        pages: fx.Array[fx.Uint64, KV_TILE_SIZE, 16]

    kernel_name = (
        f"mla_pagesize1_bf16_fp8_h{num_heads}_q{max_seqlen_q}"
        f"{'_causal' if causal else ''}{'_lse' if write_final_lse else ''}"
    )

    @flyc.kernel(name=kernel_name, known_block_size=[BLOCK_THREADS, 1, 1])
    def stage1(
        partial: fx.Pointer,
        partial_lse: fx.Pointer,
        output: fx.Pointer,
        output_lse: fx.Pointer,
        q: fx.Pointer,
        kv: fx.Pointer,
        pages: fx.Pointer,
        indptr: fx.Pointer,
        info: fx.Pointer,
        kv_scale: fx.Pointer,
        scale: fx.Float32,
        total_q: fx.Int32,
        num_pages: fx.Int32,
        num_indices: fx.Int32,
        num_partial: fx.Int32,
        num_work: fx.Int32,
        num_cus: fx.Constexpr[int],
    ):
        tid = fx.thread_idx.x
        lane = tid % WAVE_SIZE
        lane_row = lane % MFMA_SIZE
        lane_group = lane // MFMA_SIZE
        wave = _wave_uniform(tid // WAVE_SIZE)
        workgroup_id = fx.block_idx.x
        q_rows = total_q * num_heads
        p_rows = num_partial * num_heads
        q_global = _buffer(
            q, fx.BFloat16, fx.make_layout((q_rows * (QK_DIM // 8), 8), (8, 1))
        )
        page_indices = _buffer(pages, fx.Int32, fx.make_layout(num_indices, 1))
        work_indptr = _buffer(indptr, fx.Int32, fx.make_layout(num_cus + 1, 1))
        work_info = _buffer(
            info,
            fx.Int32,
            fx.make_layout((num_work, WORK_INFO_COLS), (WORK_INFO_COLS, 1)),
        )
        partial_output = _buffer(
            partial,
            fx.Float32,
            fx.make_layout(
                (p_rows, OUTPUT_COLUMN_LAYOUT[0]), (V_DIM, OUTPUT_COLUMN_LAYOUT[1])
            ),
        )
        partial_lse_output = _buffer(partial_lse, fx.Float32, fx.make_layout(p_rows, 1))
        final_output = _buffer(
            output,
            fx.BFloat16,
            fx.make_layout(
                (q_rows, OUTPUT_COLUMN_LAYOUT[0]), (V_DIM, OUTPUT_COLUMN_LAYOUT[1])
            ),
        )
        final_lse_output = _buffer(output_lse, fx.Float32, fx.make_layout(q_rows, 1))
        kv_scale_global = _buffer(kv_scale, fx.Float32, fx.make_layout(1, 1))
        kv_descale = _wave_uniform(kv_scale_global[0].bitcast(fx.Int32)).bitcast(
            fx.Float32
        )
        score_scale = scale * kv_descale * fx.Float32(LOG2E)
        shared = fx.SharedAllocator().allocate(Shared).peek()
        # One padded KV slab, viewed per access pattern: K operand rows, the
        # gather's (8 BF16, part, chunk, gather lane) stores, and V^T columns.
        keys_lds = fx.Tensor(
            shared.kv.view(
                fx.make_layout(
                    (KV_TILE_SIZE, QK_COLUMN_LAYOUT[0]),
                    (KV_ROW_STRIDE, QK_COLUMN_LAYOUT[1]),
                )
            )
        )
        gathered_kv_lds = fx.Tensor(
            shared.kv.view(
                fx.make_layout(
                    (KV_TILE_SIZE, (8, GATHER_LANES, 3, 2)),
                    (KV_ROW_STRIDE, (1, GATHER_BYTES, 16, 8)),
                )
            )
        )
        # PV is V^T @ P^T. The MFMA atom partitions V into its A operand. V is
        # gathered element by element below: a vector load of the partition
        # lowers to the same ds_read_u16s but schedules ~0.4% slower cold.
        pv_mma = fx.make_tiled_mma(
            fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16)),
            fx.make_layout((1, 1, 1), (0, 0, 0)),
        ).get_slice(lane)
        values_lds = pv_mma.partition_A(
            fx.make_view(
                shared.kv.ptr + wave * (V_DIM // WAVES),
                fx.make_layout((V_DIM // WAVES, KV_TILE_SIZE), (1, KV_ROW_STRIDE)),
            )
        )
        # QK owns a different row tile in each wave. Its eight scores per lane
        # are (element, lane_group, token_tile) = element + 4*group + 16*tile.
        # Keep this view explicit: the 0.5.0 partition_C store path hits a
        # nested-coordinate rank assertion. PV reads its B operand through the
        # same view; for this 16x16x16 atom the B lane map equals QK's C map,
        # and reusing it keeps the LLVM schedule of the base kernel (~0.4%).
        qk_probs_lds = fx.Tensor(
            shared.p.view(
                fx.make_layout(
                    (ROW_TILES * MFMA_SIZE, (4, 4, KV_TILE_SIZE // MFMA_SIZE)),
                    (PROB_ROW_STRIDE, (1, 4, MFMA_SIZE)),
                )
            )
        )
        stats = fx.Tensor(
            shared.stats.view(fx.make_layout((ROW_TILES * 16, 3), (3, 1)))
        )
        page_addresses_lds = fx.Tensor(
            shared.pages.view(fx.make_layout(KV_TILE_SIZE, 1))
        )
        kv_lds_ptr = shared.kv.ptr
        work_begin = _wave_uniform(work_indptr[workgroup_id])
        work_end = _wave_uniform(work_indptr[workgroup_id + 1])
        for work_index in range(work_begin, work_end, 1):
            work = fx.Int64(work_index)
            partial_row = _wave_uniform(work_info[(work, 1)])
            q_begin = _wave_uniform(work_info[(work, 2)])
            q_end = _wave_uniform(work_info[(work, 3)])
            kv_begin = _wave_uniform(work_info[(work, 4)])
            kv_end = _wave_uniform(work_info[(work, 5)])
            kv_tail_tokens = _wave_uniform(work_info[(work, 6)])
            qlen = q_end - q_begin
            active_rows = qlen * num_heads
            query_fragments = []
            row = wave * MFMA_SIZE + lane_row
            for column_group in range_constexpr(QK_COLUMN_GROUPS):
                for half in range_constexpr(2):
                    # Match QK_COLUMN_LAYOUT in eight-element load units.
                    # Masked rows address exactly one past the Q descriptor,
                    # so the bounded buffer load returns zero. Retain this
                    # flat load view: row-view masking changes register allocation.
                    query_offset = (
                        (fx.Int64(q_begin) * num_heads + row) * QK_DIM
                        + column_group * 64
                        + lane_group * 16
                        + half * 8
                    ) // 8
                    query_offset = (row < active_rows).select(
                        query_offset, fx.Int64(total_q) * num_heads * (QK_DIM // 8)
                    )
                    query_fragments.append(q_global[(query_offset, None)].load())
            # FlyDSL 0.5 carries nested Python containers through runtime loops.
            # Keep the algorithm's state named; no hand-packed SSA slot offsets.
            loop_state = {
                "row_max": fx.Float32(float("-inf")),
                "row_sum": fx.Float32(0.0),
                "output": [
                    [fx.full(4, 0.0, fx.Float32) for _ in range_constexpr(PV_COL_TILES)]
                    for _ in range_constexpr(ROW_TILES)
                ],
                "prefetch": [],
            }
            if ROW_TILES < WAVES:
                # Tile zero is primed once; all lanes issue clamped loads, including
                # the two waves whose gathered values are not stored to LDS.
                prefetched_kv = [
                    fx.full(4, 0, fx.Uint32) for _ in range_constexpr(GATHER_CHUNKS)
                ]
                if kv_end > kv_begin:
                    first_token = (tid % (KV_TILE_SIZE * GATHER_LANES)) // GATHER_LANES
                    first_column_group = tid % GATHER_LANES
                    first_page_index = (kv_begin + first_token < kv_end).select(
                        kv_begin + first_token, kv_end - 1
                    )
                    first_page = page_indices[first_page_index]
                    first_address = kv + fx.Int64(first_page) * QK_DIM
                    for chunk in range_constexpr(GATHER_CHUNKS):
                        prefetched_kv[chunk] = fx.ptr_load(
                            first_address
                            + first_column_group * GATHER_BYTES
                            + chunk * 16,
                            result_type=fx.Vector.make_type(16, fx.Uint8),
                        ).bitcast(fx.Uint32)
                loop_state["prefetch"] = prefetched_kv
            carried = _flatten_state(loop_state)
            for tile_index, carried_in in range(
                kv_begin, kv_end, KV_TILE_SIZE, init=carried
            ):
                tile_start = fx.Int32(tile_index)
                loop_state = _unflatten_state(carried_in, ROW_TILES, PV_COL_TILES)
                # The prefetched path publishes next-tile addresses while PV
                # consumes the current tile. At eight row tiles, load in place.
                if tid < KV_TILE_SIZE:
                    lookahead = KV_TILE_SIZE if ROW_TILES < WAVES else 0
                    idx = (tile_start + lookahead + tid < kv_end).select(
                        tile_start + lookahead + tid, kv_end - 1
                    )
                    page = page_indices[idx]
                    fx.memref_store(
                        fx.ptrtoint(kv + fx.Int64(page) * QK_DIM).to(fx.Uint64),
                        page_addresses_lds,
                        tid,
                    )
                fx.barrier()
                gather_tid = _defer_to_use(tid) if ROW_TILES >= 7 else tid
                # Twelve lanes gather each token in three aligned 16-byte
                # chunks. Six waves cover all 32 tokens; the slab is shared.
                if wave < GATHER_WAVES:
                    token = gather_tid // GATHER_LANES
                    column_group = gather_tid % GATHER_LANES
                    page_base = (
                        fx.inttoptr(kv.type, page_addresses_lds[token])
                        if ROW_TILES == WAVES
                        else kv
                    )
                    # Swap adjacent 16B BF16 writes in alternating four-lane
                    # groups. All eight 16B bank slots occur once per lane octet.
                    swap = (column_group // 4) % 2
                    for chunk in range_constexpr(GATHER_CHUNKS):
                        byte_offset = column_group * GATHER_BYTES + chunk * 16
                        # Keep 64-bit addressing for production pools >4 GiB.
                        packed = (
                            loop_state["prefetch"][chunk]
                            if ROW_TILES < WAVES
                            else fx.ptr_load(
                                page_base + byte_offset,
                                result_type=fx.Vector.make_type(16, fx.Uint8),
                            ).bitcast(fx.Uint32)
                        )
                        for piece in range_constexpr(2):
                            part = fx.Int32(piece) ^ swap
                            bf16_values = []
                            for word in range_constexpr(2):
                                bits = (swap == 0).select(
                                    packed[piece * 2 + word],
                                    packed[(1 - piece) * 2 + word],
                                )
                                for half in range_constexpr(2):
                                    v = fx.Vector(
                                        fx.rocdl.cvt_pk_f32_fp8(
                                            T.f32x2, bits.to(fx.Int32), bool(half)
                                        )
                                    )
                                    pair = _pack_fp8_as_bf16(v)
                                    bf16_values += [pair[0], pair[1]]
                            gathered_kv_lds[
                                (token, (None, column_group, chunk, part))
                            ].store(
                                fx.Vector.from_elements(bf16_values, dtype=fx.BFloat16)
                            )
                fx.barrier()
                tile = wave
                row = tile * 16 + lane_row
                row_max = loop_state["row_max"]
                row_sum = loop_state["row_sum"]
                if tile < ROW_TILES:
                    score_fragments = [
                        fx.full(4, 0.0, fx.Float32)
                        for _ in range_constexpr(KV_TILE_SIZE // 16)
                    ]
                    if const_expr(ROW_TILES <= 4):
                        # Prefetch one K pair only on the small-row path.
                        # Scheduler fences keep just two operand sets live.
                        key_values = [
                            keys_lds[
                                (token_tile * 16 + lane_row, (None, lane_group, 0, 0))
                            ].load()
                            for token_tile in range_constexpr(KV_TILE_SIZE // 16)
                        ]
                        fx.rocdl.sched_barrier(0)
                        for step in range_constexpr(QK_STEPS):
                            if const_expr(step + 1 < QK_STEPS):
                                next_step = step + 1
                                next_keys = [
                                    keys_lds[
                                        (
                                            token_tile * 16 + lane_row,
                                            (
                                                None,
                                                lane_group,
                                                next_step % 2,
                                                next_step // 2,
                                            ),
                                        )
                                    ].load()
                                    for token_tile in range_constexpr(
                                        KV_TILE_SIZE // 16
                                    )
                                ]
                            fx.rocdl.sched_barrier(0)
                            query_values = query_fragments[step]
                            for sub in range_constexpr(2):
                                for token_tile in range_constexpr(KV_TILE_SIZE // 16):
                                    score_fragments[token_tile] = _mfma(
                                        _mfma_fragment(key_values[token_tile], sub * 4),
                                        _mfma_fragment(query_values, sub * 4),
                                        score_fragments[token_tile],
                                    )
                            fx.rocdl.sched_barrier(0)
                            if const_expr(step + 1 < QK_STEPS):
                                key_values = next_keys
                    else:
                        for column_group in range_constexpr(QK_COLUMN_GROUPS):
                            for half in range_constexpr(2):
                                # Prevent full-K CSE/hoisting across row tiles:
                                # retaining the full slab alongside Q/O spills.
                                key_offset = _defer_to_use(
                                    fx.get_scalar(
                                        fx.crd2idx(
                                            (
                                                lane_row,
                                                (0, lane_group, half, column_group),
                                            ),
                                            keys_lds.layout,
                                        )
                                    )
                                )
                                key_step = fx.Tensor(
                                    fx.make_view(
                                        kv_lds_ptr + key_offset,
                                        fx.make_layout(
                                            (KV_TILE_SIZE // 16, 8),
                                            (16 * KV_ROW_STRIDE, 1),
                                        ),
                                    )
                                )
                                key_values = [
                                    key_step[(token_tile, None)].load()
                                    for token_tile in range_constexpr(
                                        KV_TILE_SIZE // 16
                                    )
                                ]
                                query_values = query_fragments[column_group * 2 + half]
                                for sub in range_constexpr(2):
                                    for token_tile in range_constexpr(
                                        KV_TILE_SIZE // 16
                                    ):
                                        score_fragments[token_tile] = _mfma(
                                            _mfma_fragment(
                                                key_values[token_tile], sub * 4
                                            ),
                                            _mfma_fragment(query_values, sub * 4),
                                            score_fragments[token_tile],
                                        )
                    visible_end = kv_end
                    if causal:
                        # Bottom-right mask: query i cannot see the last
                        # qlen-1-i tokens. A fragment before that tail hides none.
                        hide = qlen - 1 - row // num_heads - kv_tail_tokens
                        visible_end = kv_end - (hide > 0).select(hide, fx.Int32(0))
                    scores = []
                    valid = []
                    for token_tile in range_constexpr(KV_TILE_SIZE // 16):
                        for i in range_constexpr(4):
                            is_visible = (row < active_rows) & (
                                tile_start + token_tile * 16 + lane_group * 4 + i
                                < visible_end
                            )
                            valid.append(is_visible)
                            scores.append(
                                is_visible.select(
                                    score_fragments[token_tile][i] * score_scale,
                                    fx.Float32(float("-inf")),
                                )
                            )
                    tile_max = scores[0]
                    for i in range_constexpr(1, KV_TILE_SIZE // 4):
                        tile_max = tile_max.maximumf(scores[i])
                    for delta in range_constexpr(2):
                        tile_max = tile_max.maximumf(
                            fx.gpu.shuffle_xor(tile_max, 16 << delta, 64)
                        )
                    # Online softmax uses exp2 units. When the running maximum
                    # grows, alpha rescales BOTH the prior sum and PV output.
                    row_max = loop_state["row_max"].maximumf(tile_max)
                    safe_max = (row_max > fx.Float32(float("-inf"))).select(
                        row_max, fx.Float32(0.0)
                    )
                    alpha = fmath.exp2(loop_state["row_max"] - safe_max)
                    probabilities = [
                        valid[i].select(
                            fmath.exp2(scores[i] - safe_max), fx.Float32(0.0)
                        )
                        for i in range_constexpr(KV_TILE_SIZE // 4)
                    ]
                    row_sum = probabilities[0]
                    for i in range_constexpr(1, KV_TILE_SIZE // 4):
                        row_sum = row_sum + probabilities[i]
                    for delta in range_constexpr(2):
                        row_sum = row_sum + fx.gpu.shuffle_xor(row_sum, 16 << delta, 64)
                    row_sum = loop_state["row_sum"] * alpha + row_sum
                    for token_tile in range_constexpr(KV_TILE_SIZE // 16):
                        for i in range_constexpr(4):
                            fx.memref_store(
                                _round_to_bf16_finite(
                                    probabilities[token_tile * 4 + i]
                                ),
                                qk_probs_lds,
                                (row, (i, lane_group, token_tile)),
                            )
                    if lane_group == 0:
                        fx.memref_store(alpha, stats, (row, STAT_ALPHA))
                        fx.memref_store(row_max, stats, (row, STAT_MAX))
                        fx.memref_store(row_sum, stats, (row, STAT_SUM))
                else:
                    row_max = loop_state["row_max"]
                    row_sum = loop_state["row_sum"]
                next_kv_bytes = []
                if ROW_TILES < WAVES:
                    # After the complete QK/softmax phase: never between MFMAs.
                    fx.rocdl.sched_barrier(0)
                    prefetch_tid = _defer_to_use(tid)
                    prefetch_token = (
                        prefetch_tid % (KV_TILE_SIZE * GATHER_LANES)
                    ) // GATHER_LANES
                    prefetch_column_group = prefetch_tid % GATHER_LANES
                    prefetch_address = fx.inttoptr(
                        kv.type, page_addresses_lds[prefetch_token]
                    )
                    for chunk in range_constexpr(GATHER_CHUNKS):
                        next_kv_bytes.append(
                            fx.ptr_load(
                                prefetch_address
                                + prefetch_column_group * GATHER_BYTES
                                + chunk * 16,
                                result_type=fx.Vector.make_type(16, fx.Uint8),
                            ).bitcast(fx.Uint32)
                        )
                    fx.rocdl.sched_barrier(0)
                fx.barrier()
                next_output = []
                # PV: A=V^T, B=P^T. All rows, disjoint column slices per wave.
                for row_tile in range_constexpr(ROW_TILES):
                    row = row_tile * 16 + lane_row
                    alpha = stats[(row, STAT_ALPHA)]
                    row_output = []
                    for column_tile in range_constexpr(PV_COL_TILES):
                        old = loop_state["output"][row_tile][column_tile]
                        acc = old * alpha
                        for token_tile in range_constexpr(KV_TILE_SIZE // 16):
                            prob_values = qk_probs_lds[
                                (
                                    row_tile * 16 + lane_row,
                                    (None, lane_group, token_tile),
                                )
                            ].load()
                            value_values = fx.Vector.from_elements(
                                [
                                    values_lds[(i, column_tile, token_tile)]
                                    for i in range_constexpr(4)
                                ],
                                dtype=fx.BFloat16,
                            )
                            acc = _mfma(value_values, prob_values, acc)
                        row_output.append(acc)
                    next_output.append(row_output)
                carried = yield _flatten_state(
                    {
                        "row_max": row_max,
                        "row_sum": row_sum,
                        "output": next_output,
                        "prefetch": next_kv_bytes,
                    }
                )
            loop_state = _unflatten_state(carried, ROW_TILES, PV_COL_TILES)
            # Shared statistics are final after the last PV barrier. A work
            # with zero tokens never enters the loop and writes zero/-inf.
            fx.barrier()
            # Keep epilogue-only div/mod and addresses out of the KV loop's
            # live set. One opaque move per work item avoids spilling them.
            store_tid = _defer_to_use(tid)
            store_lane_row = store_tid % 16
            store_lane_group = (store_tid % 64) // 16
            store_wave = store_tid // 64
            for row_tile in range_constexpr(ROW_TILES):
                row = row_tile * 16 + store_lane_row
                if row < active_rows:
                    row_sum = fx.Float32(0.0)
                    row_max = fx.Float32(float("-inf"))
                    if kv_end > kv_begin:
                        row_sum = stats[(row, STAT_SUM)]
                        row_max = stats[(row, STAT_MAX)]
                    denom = (row_sum > 0).select(row_sum, fx.Float32(1.0))
                    factor = kv_descale / denom
                    lse = (row_sum > 0).select(
                        row_max * fx.Float32(0.6931471805599453) + fmath.log(denom),
                        fx.Float32(float("-inf")),
                    )
                    query_position = row // num_heads
                    head = row % num_heads
                    if partial_row >= 0:
                        out_row = (
                            fx.Int64(partial_row + query_position) * num_heads + head
                        )
                        for column_tile in range_constexpr(PV_COL_TILES):
                            acc = loop_state["output"][row_tile][column_tile]
                            output_columns = (
                                store_wave,
                                column_tile,
                                store_lane_group,
                                None,
                            )
                            partial_output[(out_row, output_columns)].store(
                                acc * factor
                            )
                        if (store_wave == 0) & (store_lane_group == 0):
                            fx.memref_store(lse, partial_lse_output, out_row)
                    else:
                        out_row = fx.Int64(q_begin + query_position) * num_heads + head
                        for column_tile in range_constexpr(PV_COL_TILES):
                            acc = loop_state["output"][row_tile][column_tile]
                            output_columns = (
                                store_wave,
                                column_tile,
                                store_lane_group,
                                None,
                            )
                            value = _round_to_bf16_finite(acc * factor)
                            final_output[(out_row, output_columns)].store(value)
                        # Trace-time flag around a device test: they must not be merged.
                        if write_final_lse:  # noqa: SIM102
                            if (store_wave == 0) & (store_lane_group == 0):
                                fx.memref_store(lse, final_lse_output, out_row)

    return stage1


@flyc.jit
def launch_mla_pagesize1_bf16_fp8(
    partial: fx.Pointer,
    partial_lse: fx.Pointer,
    output: fx.Pointer,
    output_lse: fx.Pointer,
    q: fx.Pointer,
    kv: fx.Pointer,
    pages: fx.Pointer,
    indptr: fx.Pointer,
    info: fx.Pointer,
    kv_scale: fx.Pointer,
    scale: fx.Float32,
    total_q: fx.Int32,
    num_pages: fx.Int32,
    num_indices: fx.Int32,
    num_partial: fx.Int32,
    num_work: fx.Int32,
    num_cus: fx.Constexpr[int],
    num_heads: fx.Constexpr[int],
    max_seqlen_q: fx.Constexpr[int],
    causal: fx.Constexpr[bool],
    write_final_lse: fx.Constexpr[bool],
    stream: fx.Stream,
):
    kernel = _build(num_heads, max_seqlen_q, causal, write_final_lse)
    kernel(
        partial,
        partial_lse,
        output,
        output_lse,
        q,
        kv,
        pages,
        indptr,
        info,
        kv_scale,
        scale,
        total_q,
        num_pages,
        num_indices,
        num_partial,
        num_work,
        num_cus,
    ).launch(grid=(num_cus, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)
