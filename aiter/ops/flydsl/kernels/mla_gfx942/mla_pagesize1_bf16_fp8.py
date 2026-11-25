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

# MLA shapes: K is the full 576-wide latent+RoPE row; V is its first 512 columns.
QK_DIM = 576
V_DIM = 512
# One get_mla_metadata_v1 work item: [batch, partial, q_start, q_end,
# kv_start, kv_end, kv_tail_tokens, pad].
WORK_INFO_COLS = 8

# Workgroup shape: eight 64-lane waves around 16x16x16 BF16 MFMAs. Each KV loop
# iteration stages KV_TILE_SIZE tokens in LDS; longer ranges loop over tiles.
WAVE_SIZE = 64
MFMA_SIZE = 16
WAVES = 8
BLOCK_THREADS = WAVES * WAVE_SIZE
KV_TILE_SIZE = 32
TOKEN_TILES = KV_TILE_SIZE // MFMA_SIZE
LN2 = 0.6931471805599453

# Widest load/store (dwordx4 / ds_read_b128) and the BF16 values it holds.
VECTOR_BYTES = 16
BF16_VECTOR_SIZE = VECTOR_BYTES // 2

# QK's K permutation within one 64-column group, as (element, lane_group,
# k_step): each lane group reads 16 contiguous columns, so one 16-byte load
# feeds two MFMAs. Q and K are partitioned by the same tiled MMA, so the
# permuted reduction order cancels in the dot product.
QK_K_PERMUTATION = ((4, 4, 4), (1, 16, 4))


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


def _defer_view_to_use(view):
    # _defer_to_use for a partitioned LDS view: its per-lane address is opaque.
    address = fx.get_iter(view)
    return fx.make_view(
        fx.inttoptr(address.type, _defer_to_use(fx.ptrtoint(address))),
        fx.get_layout(view),
    )


def _wave_uniform(value):
    # Scheduling metadata and wave ids are identical in all lanes of a wave.
    return fx.Int32(fx.rocdl.readfirstlane(T.i32, fx.Int32(value)))


@cache
def _build(num_heads, max_seqlen_q, causal, write_final_lse):
    assert (
        1 <= num_heads <= 16
        and 1 <= max_seqlen_q <= 8
        and num_heads * max_seqlen_q <= 128
    )

    # Work tiling. QK gives wave w query-row tile w (16 rows); waves without a
    # row tile prefetch the next KV tile. PV gives each wave 64 output columns
    # (four 16-column MFMA tiles) across all rows.
    ROW_TILES = (num_heads * max_seqlen_q + MFMA_SIZE - 1) // MFMA_SIZE
    PREFETCH_NEXT_TILE = ROW_TILES < WAVES

    # LDS row pitches. The extra 16 bytes shift each row by four banks.
    KV_ROW_STRIDE = QK_DIM + BF16_VECTOR_SIZE
    PROB_ROW_STRIDE = KV_TILE_SIZE + 8

    # KV gather: 12 lanes per token load its 576 FP8 bytes as three 16-byte
    # chunks, so 32 tokens occupy waves 0-5.
    GATHER_BYTES = 48
    GATHER_LANES = QK_DIM // GATHER_BYTES
    GATHER_CHUNKS = GATHER_BYTES // VECTOR_BYTES
    GATHER_WAVES = KV_TILE_SIZE * GATHER_LANES // WAVE_SIZE

    # QK walks 576 columns as 9 groups x 2 halves; each step issues two MFMAs.
    QK_COLUMN_GROUPS = QK_DIM // WAVE_SIZE
    QK_STEPS = QK_COLUMN_GROUPS * 2

    WAVE_COLUMNS = V_DIM // WAVES
    STAT_ALPHA, STAT_MAX, STAT_SUM = range(3)

    # LDS: the BF16 KV tile (read as K and as V), softmax probabilities, the
    # per-row (alpha, max, sum) handed from QK owners to PV, and page addresses.
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
        num_indices: fx.Int32,
        num_partial: fx.Int32,
        num_work: fx.Int32,
        num_cus: fx.Constexpr[int],
    ):
        # Thread coordinates. lane_row is the MFMA row/column a lane owns and
        # lane_group its 4-element slice; wave is scalar for uniform branches.
        tid = fx.thread_idx.x
        lane = tid % WAVE_SIZE
        lane_row = lane % MFMA_SIZE
        lane_group = lane // MFMA_SIZE
        wave = _wave_uniform(tid // WAVE_SIZE)
        workgroup_id = fx.block_idx.x

        # Global buffers. Q rows are token * num_heads + head.
        q_rows = total_q * num_heads
        p_rows = num_partial * num_heads
        q_global = _buffer(
            q, fx.BFloat16, fx.make_layout((q_rows, QK_DIM), (QK_DIM, 1))
        )
        page_indices = _buffer(pages, fx.Int32, fx.make_layout(num_indices, 1))

        # Planner output: work_indptr[g]..[g+1] are this workgroup's work items.
        work_indptr = _buffer(indptr, fx.Int32, fx.make_layout(num_cus + 1, 1))
        work_info = _buffer(
            info,
            fx.Int32,
            fx.make_layout((num_work, WORK_INFO_COLS), (WORK_INFO_COLS, 1)),
        )

        # Outputs: FP32 split partials + LSE for mla_reduce_v1, or the final
        # BF16 output (+ optional LSE) when a request is not split.
        partial_output = _buffer(
            partial, fx.Float32, fx.make_layout((p_rows, V_DIM), (V_DIM, 1))
        )
        partial_lse_output = _buffer(partial_lse, fx.Float32, fx.make_layout(p_rows, 1))
        final_output = _buffer(
            output, fx.BFloat16, fx.make_layout((q_rows, V_DIM), (V_DIM, 1))
        )
        final_lse_output = _buffer(output_lse, fx.Float32, fx.make_layout(q_rows, 1))

        # The FP8 KV scale is applied twice, never per element: here for K,
        # folded with log2(e) so softmax can use exp2, and in the epilogue for V.
        kv_scale_global = _buffer(kv_scale, fx.Float32, fx.make_layout(1, 1))
        kv_descale = fx.Float32(fx.rocdl.readfirstlane(T.f32, kv_scale_global[0]))
        score_scale = scale * kv_descale * fx.Float32(LOG2E)

        # Both GEMMs run one 16x16x16 BF16 MFMA atom within a single wave.
        # QK is K @ Q^T: M is tokens, N query rows, and K the permuted columns.
        # PV is V^T @ P^T: M is this wave's output columns, N query rows, and
        # K tokens.
        mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 16, fx.BFloat16))
        wave_atoms = fx.make_layout((1, 1, 1), (0, 0, 0))
        qk_mma = fx.make_tiled_mma(
            mma_atom,
            wave_atoms,
            fx.make_tile(None, None, fx.make_layout(*QK_K_PERMUTATION)),
        )
        pv_mma = fx.make_tiled_mma(mma_atom, wave_atoms)
        lds_copy_128 = fx.make_copy_atom(fx.UniversalCopy128b(), fx.BFloat16)
        lds_copy_64 = fx.make_copy_atom(fx.UniversalCopy64b(), fx.BFloat16)
        lds_copy_16 = fx.make_copy_atom(fx.UniversalCopy16b(), fx.BFloat16)
        q_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.BFloat16)
        partial_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(), fx.Float32)
        final_copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy64b(), fx.BFloat16)

        shared = fx.SharedAllocator().allocate(Shared).peek()
        # One padded KV slab, viewed per access pattern: K rows, the gather's
        # (8 BF16, gather lane, chunk, part) stores, and this wave's V^T columns.
        keys_lds = shared.kv.view(
            fx.make_layout((KV_TILE_SIZE, QK_DIM), (KV_ROW_STRIDE, 1))
        )
        gathered_kv_lds = fx.Tensor(
            shared.kv.view(
                fx.make_layout(
                    (KV_TILE_SIZE, (8, GATHER_LANES, 3, 2)),
                    (KV_ROW_STRIDE, (1, GATHER_BYTES, 16, 8)),
                )
            )
        )
        values_lds = fx.make_view(
            shared.kv.ptr + wave * WAVE_COLUMNS,
            fx.make_layout((WAVE_COLUMNS, KV_TILE_SIZE), (1, KV_ROW_STRIDE)),
        )

        # Softmax probabilities P, row-major. Each QK wave stores its C tile
        # into its row tile through the transposed (token, row) view; PV reads
        # all rows back as its B operand.
        probs_lds = shared.p.view(
            fx.make_layout((ROW_TILES * MFMA_SIZE, KV_TILE_SIZE), (PROB_ROW_STRIDE, 1))
        )
        wave_probs_lds = fx.make_view(
            shared.p.ptr + wave * MFMA_SIZE * PROB_ROW_STRIDE,
            fx.make_layout((KV_TILE_SIZE, MFMA_SIZE), (1, PROB_ROW_STRIDE)),
        )

        # Per-lane partitions. QK's A operand is ((8 BF16, half), token_tile,
        # column_group): one ds_read_b128 per (half, token_tile, column_group).
        # PV's A operand is V^T, strided by a KV row per element: one
        # ds_read_u16 each. P moves as 4 BF16 (ds_write_b64/ds_read_b64).
        query_copy = fx.make_tiled_copy_B(q_copy_atom, qk_mma).get_slice(lane)
        keys_copy = fx.make_tiled_copy_A(lds_copy_128, qk_mma).get_slice(lane)
        keys_src = keys_copy.partition_S(keys_lds)
        probs_store = fx.make_tiled_copy_C(lds_copy_64, qk_mma).get_slice(lane)
        probs_dst = probs_store.partition_D(wave_probs_lds)
        values_copy = fx.make_tiled_copy_A(lds_copy_16, pv_mma).get_slice(lane)
        values_src = values_copy.partition_S(values_lds)
        probs_load = fx.make_tiled_copy_B(lds_copy_64, pv_mma).get_slice(lane)
        probs_src = probs_load.partition_S(probs_lds)

        # Register fragments. Method calls on these outer objects stay out of
        # the runtime loop bodies: FlyDSL's implicit loop carry would try to
        # carry the objects themselves.
        def load_query(rows):
            fragment = qk_mma.make_fragment_B(rows)
            fx.copy(
                q_copy_atom, query_copy.partition_S(rows), query_copy.retile(fragment)
            )
            return fragment

        def make_key_fragment():
            fragment = qk_mma.make_fragment_A(keys_lds)
            return fragment, keys_copy.retile(fragment)

        # One QK step is a (half, column_group): a 16-byte load per token tile,
        # then two K slices of MFMAs.
        def load_keys(src, key_dst, half, column_group):
            fx.copy(lds_copy_128, src, key_dst[(None, half), None, column_group])

        def multiply_keys(scores, keys, query, half, column_group):
            for sub in range_constexpr(2):
                k = (half * 2 + sub, column_group)
                fx.gemm(
                    qk_mma, scores, keys[None, None, k], query[None, None, k], scores
                )

        def make_score_fragment():
            # Scores are (element, token_tile): token element + 4*lane_group
            # + 16*token_tile of row lane_row.
            fragment = qk_mma.make_fragment_C(wave_probs_lds)
            fragment.fill(0.0)
            return fragment

        def store_probs(scores, probabilities):
            fragment = fx.make_fragment_like(scores, fx.BFloat16)
            fragment.store(fx.Vector.from_elements(probabilities, dtype=fx.BFloat16))
            fx.copy(lds_copy_64, probs_store.retile(fragment), probs_dst)

        def load_pv_operands():
            values = pv_mma.make_fragment_A(values_lds)
            fx.copy(lds_copy_16, values_src, values_copy.retile(values))
            probs = pv_mma.make_fragment_B(probs_lds)
            fx.copy(lds_copy_64, probs_src, probs_load.retile(probs))
            return values, probs

        def make_output_fragment():
            # PV accumulators: (4 values, column_tile, row_tile) per lane.
            return fx.make_rmem_tensor(
                fx.tiled_mma_partition_shape(
                    fx.MmaOperand.C, pv_mma, (WAVE_COLUMNS, ROW_TILES * MFMA_SIZE)
                ),
                fx.Float32,
            )

        def zero_output():
            fragment = make_output_fragment()
            fragment.fill(0.0)
            return fragment.load()

        def output_columns(buffer, first_row, store_wave):
            # A work's rows are contiguous in either output buffer.
            return fx.make_view(
                fx.get_iter(buffer) + first_row * V_DIM + store_wave * WAVE_COLUMNS,
                fx.make_layout((WAVE_COLUMNS, ROW_TILES * MFMA_SIZE), (1, V_DIM)),
            )

        # Per-row (alpha, max, sum): written by the row's QK wave, read by every
        # PV wave (alpha) and by the epilogue (max, sum).
        stats = fx.Tensor(
            shared.stats.view(fx.make_layout((ROW_TILES * 16, 3), (3, 1)))
        )
        # Absolute KV row addresses, computed once per token for the gather lanes.
        page_addresses_lds = fx.Tensor(
            shared.pages.view(fx.make_layout(KV_TILE_SIZE, 1))
        )

        # Persistent loop: one workgroup per CU runs its planner-assigned work
        # items. Each item is one request's query rows over one KV range.
        work_begin = _wave_uniform(work_indptr[workgroup_id])
        work_end = _wave_uniform(work_indptr[workgroup_id + 1])
        for work_index in range(work_begin, work_end, 1):
            # Work item fields. partial_row >= 0 means this is one KV fragment
            # of a split request; -1 means it covers the whole request.
            work = fx.Int64(work_index)
            partial_row = _wave_uniform(work_info[(work, 1)])
            q_begin = _wave_uniform(work_info[(work, 2)])
            q_end = _wave_uniform(work_info[(work, 3)])
            kv_begin = _wave_uniform(work_info[(work, 4)])
            kv_end = _wave_uniform(work_info[(work, 5)])
            kv_tail_tokens = _wave_uniform(work_info[(work, 6)])
            qlen = q_end - q_begin
            active_rows = qlen * num_heads

            # Q stays in registers for the whole KV range as QK's B operand:
            # each lane loads 18 x 16 bytes of its row. Masked rows start one
            # past the Q descriptor, so their bounded buffer loads return zero.
            row = wave * MFMA_SIZE + lane_row
            query_start = (row < active_rows).select(
                (q_begin * num_heads + wave * MFMA_SIZE) * QK_DIM,
                (q_rows - lane_row) * QK_DIM,
            )
            query_fragment = load_query(
                fx.make_view(
                    fx.get_iter(q_global) + query_start,
                    fx.make_layout((MFMA_SIZE, QK_DIM), (QK_DIM, 1)),
                )
            )

            # Online-softmax state carried across KV tiles: running max and sum
            # of this lane's QK row, the unnormalized PV accumulator fragment's
            # values, and the raw FP8 bytes of the next tile when spare waves
            # prefetch. FlyDSL's implicit loop carry preserves this nested state.
            loop_state = {
                "row_max": fx.Float32(float("-inf")),
                "row_sum": fx.Float32(0.0),
                "output": zero_output(),
                "prefetch": [],
            }

            if PREFETCH_NEXT_TILE:
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
                            + chunk * VECTOR_BYTES,
                            result_type=fx.Vector.make_type(16, fx.Uint8),
                        ).bitcast(fx.Uint32)
                loop_state["prefetch"] = prefetched_kv

            # KV loop: KV_TILE_SIZE tokens per iteration over this work's range.
            # Each iteration: page addresses -> gather to LDS -> QK + softmax ->
            # prefetch next tile -> PV.
            for tile_index in range(kv_begin, kv_end, KV_TILE_SIZE):
                tile_start = fx.Int32(tile_index)

                # Page addresses. The prefetched path publishes next-tile
                # addresses while PV consumes the current tile. At eight row
                # tiles, load in place.
                if tid < KV_TILE_SIZE:
                    lookahead = KV_TILE_SIZE if PREFETCH_NEXT_TILE else 0
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

                # Gather: FP8 KV rows -> BF16 LDS slab (exact conversion; the
                # scale is applied later). Twelve lanes gather each token in
                # three aligned 16-byte chunks. Six waves cover all 32 tokens.
                gather_tid = _defer_to_use(tid) if ROW_TILES >= 7 else tid
                if wave < GATHER_WAVES:
                    token = gather_tid // GATHER_LANES
                    column_group = gather_tid % GATHER_LANES
                    page_base = (
                        fx.inttoptr(kv.type, page_addresses_lds[token])
                        if not PREFETCH_NEXT_TILE
                        else kv
                    )
                    # Swap adjacent 16B BF16 writes in alternating four-lane
                    # groups. All eight 16B bank slots occur once per lane octet.
                    swap = (column_group // 4) % 2
                    for chunk in range_constexpr(GATHER_CHUNKS):
                        byte_offset = column_group * GATHER_BYTES + chunk * VECTOR_BYTES
                        # Keep 64-bit addressing for production pools >4 GiB.
                        packed = (
                            loop_state["prefetch"][chunk]
                            if PREFETCH_NEXT_TILE
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
                                            T.f32x2, bits, bool(half)
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

                # QK: wave w computes scores for row tile w against all 32
                # tokens, then runs the online-softmax update for those rows.
                row = wave * MFMA_SIZE + lane_row
                row_max = loop_state["row_max"]
                row_sum = loop_state["row_sum"]
                if wave < ROW_TILES:
                    score_fragment = make_score_fragment()
                    key_fragment, key_dst = make_key_fragment()
                    if const_expr(ROW_TILES <= 4):
                        # Small-row path (<= 64 rows): load the next K pair
                        # while the current pair feeds the MFMAs.
                        # Scheduler fences keep just two operand sets live.
                        load_keys(keys_src[(None, 0), None, 0], key_dst, 0, 0)
                        fx.rocdl.sched_barrier(0)
                        for step in range_constexpr(QK_STEPS):
                            column_group, half = divmod(step, 2)
                            if const_expr(step + 1 < QK_STEPS):
                                next_group, next_half = divmod(step + 1, 2)
                                load_keys(
                                    keys_src[(None, next_half), None, next_group],
                                    key_dst,
                                    next_half,
                                    next_group,
                                )
                            fx.rocdl.sched_barrier(0)
                            multiply_keys(
                                score_fragment,
                                key_fragment,
                                query_fragment,
                                half,
                                column_group,
                            )
                            fx.rocdl.sched_barrier(0)
                    else:
                        # Large-row path: one K step live at a time, through
                        # opaque LDS addresses, to fit beside Q and the outputs.
                        for column_group in range_constexpr(QK_COLUMN_GROUPS):
                            for half in range_constexpr(2):
                                # Prevent full-K CSE/hoisting across row tiles:
                                # retaining the full slab alongside Q/O spills.
                                load_keys(
                                    _defer_view_to_use(
                                        keys_src[(None, half), None, column_group]
                                    ),
                                    key_dst,
                                    half,
                                    column_group,
                                )
                                multiply_keys(
                                    score_fragment,
                                    key_fragment,
                                    query_fragment,
                                    half,
                                    column_group,
                                )
                    score_values = score_fragment.load()

                    # Mask: rows past active_rows, tokens past kv_end (the last
                    # partial tile), and the causal tail.
                    visible_end = kv_end
                    if causal:
                        # Bottom-right mask: query i cannot see the last
                        # qlen-1-i tokens. A fragment before that tail hides none.
                        hide = qlen - 1 - row // num_heads - kv_tail_tokens
                        visible_end = kv_end - (hide > 0).select(hide, fx.Int32(0))
                    scores = []
                    valid = []
                    for token_tile in range_constexpr(TOKEN_TILES):
                        for i in range_constexpr(4):
                            is_visible = (row < active_rows) & (
                                tile_start + token_tile * MFMA_SIZE + lane_group * 4 + i
                                < visible_end
                            )
                            valid.append(is_visible)
                            scores.append(
                                is_visible.select(
                                    score_values[token_tile * 4 + i] * score_scale,
                                    fx.Float32(float("-inf")),
                                )
                            )

                    # Row max over this tile's 32 tokens: 8 per lane, then
                    # across the 4 lane groups sharing the row.
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

                    # Publish BF16 P and the row statistics for the PV phase,
                    # where every wave needs every row.
                    store_probs(
                        score_fragment,
                        [_round_to_bf16_finite(p) for p in probabilities],
                    )
                    if lane_group == 0:
                        fx.memref_store(alpha, stats, (row, STAT_ALPHA))
                        fx.memref_store(row_max, stats, (row, STAT_MAX))
                        fx.memref_store(row_sum, stats, (row, STAT_SUM))
                else:
                    row_max = loop_state["row_max"]
                    row_sum = loop_state["row_sum"]

                # Prefetch: when spare waves exist, all lanes load the next
                # tile's FP8 bytes into registers while PV runs on this tile.
                next_kv_bytes = []
                if PREFETCH_NEXT_TILE:
                    # After the complete QK/softmax phase: never between MFMAs.
                    fx.rocdl.sched_barrier(0)
                    prefetch_token = (
                        tid % (KV_TILE_SIZE * GATHER_LANES)
                    ) // GATHER_LANES
                    prefetch_column_group = tid % GATHER_LANES
                    prefetch_address = fx.inttoptr(
                        kv.type, page_addresses_lds[prefetch_token]
                    )
                    for chunk in range_constexpr(GATHER_CHUNKS):
                        next_kv_bytes.append(
                            fx.ptr_load(
                                prefetch_address
                                + prefetch_column_group * GATHER_BYTES
                                + chunk * VECTOR_BYTES,
                                result_type=fx.Vector.make_type(16, fx.Uint8),
                            ).bitcast(fx.Uint32)
                        )
                    fx.rocdl.sched_barrier(0)
                fx.barrier()

                # PV: A=V^T, B=P^T. All rows, disjoint column slices per wave.
                # Each row tile's accumulators are first rescaled by its alpha.
                accumulators = make_output_fragment()
                accumulators.store(loop_state["output"])
                for row_tile in range_constexpr(ROW_TILES):
                    alpha = stats[(row_tile * MFMA_SIZE + lane_row, STAT_ALPHA)]
                    rescaled = accumulators[None, None, row_tile]
                    rescaled.store(rescaled.load() * alpha)
                value_fragment, prob_fragment = load_pv_operands()
                # Tokens innermost, then column tiles, then row tiles.
                fx.gemm(
                    pv_mma,
                    accumulators,
                    value_fragment,
                    prob_fragment,
                    accumulators,
                    traversal_order=fx.GemmTraversalOrder.KMN,
                )

                loop_state = {
                    "row_max": row_max,
                    "row_sum": row_sum,
                    "output": accumulators.load(),
                    "prefetch": next_kv_bytes,
                }

            # Epilogue: normalize by the row sum (and kv_descale for V), then
            # store FP32 partial + LSE for a split request, or BF16 final output.
            # Shared statistics are final after the last PV barrier. A work
            # with zero tokens never enters the loop and writes zero/-inf.
            fx.barrier()
            # Keep epilogue-only div/mod and addresses out of the KV loop's
            # live set. One opaque move per work item avoids spilling them.
            store_tid = _defer_to_use(tid)
            store_lane = store_tid % WAVE_SIZE
            store_lane_row = store_lane % MFMA_SIZE
            store_lane_group = store_lane // MFMA_SIZE
            store_wave = store_tid // WAVE_SIZE
            partial_store = fx.make_tiled_copy_C(partial_copy_atom, pv_mma).get_slice(
                store_lane
            )
            partial_dst = partial_store.partition_D(
                output_columns(partial_output, partial_row * num_heads, store_wave)
            )
            final_store = fx.make_tiled_copy_C(final_copy_atom, pv_mma).get_slice(
                store_lane
            )
            final_dst = final_store.partition_D(
                output_columns(final_output, q_begin * num_heads, store_wave)
            )
            accumulators = make_output_fragment()
            accumulators.store(loop_state["output"])

            for row_tile in range_constexpr(ROW_TILES):
                row = row_tile * MFMA_SIZE + store_lane_row
                if row < active_rows:
                    row_sum = fx.Float32(0.0)
                    row_max = fx.Float32(float("-inf"))
                    if kv_end > kv_begin:
                        row_sum = stats[(row, STAT_SUM)]
                        row_max = stats[(row, STAT_MAX)]
                    denom = (row_sum > 0).select(row_sum, fx.Float32(1.0))
                    factor = kv_descale / denom
                    # LSE in natural log: row_max is in log2 units, so * ln(2).
                    lse = (row_sum > 0).select(
                        row_max * fx.Float32(LN2) + fmath.log(denom),
                        fx.Float32(float("-inf")),
                    )
                    query_position = row // num_heads
                    head = row % num_heads
                    row_output = accumulators[None, None, row_tile]
                    normalized = row_output.load() * factor

                    if partial_row >= 0:
                        # Split request: FP32 slot in the reduce's partial buffer.
                        out_row = (
                            fx.Int64(partial_row + query_position) * num_heads + head
                        )
                        partial_values = fx.make_fragment_like(row_output)
                        partial_values.store(normalized)
                        fx.copy(
                            partial_copy_atom,
                            partial_values,
                            partial_dst[None, None, row_tile],
                        )
                        if (store_wave == 0) & (store_lane_group == 0):
                            fx.memref_store(lse, partial_lse_output, out_row)
                    else:
                        # Whole request: write the final BF16 output directly.
                        out_row = fx.Int64(q_begin + query_position) * num_heads + head
                        final_values = fx.make_fragment_like(row_output, fx.BFloat16)
                        final_values.store(_round_to_bf16_finite(normalized))
                        fx.copy(
                            final_copy_atom,
                            final_values,
                            final_dst[None, None, row_tile],
                        )
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
        num_indices,
        num_partial,
        num_work,
        num_cus,
    ).launch(grid=(num_cus, 1, 1), block=(BLOCK_THREADS, 1, 1), stream=stream)
