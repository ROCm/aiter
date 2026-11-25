# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx1250 MiniMax-M3 lightning-indexer PREFILL block scores over the NVFP4 index cache.

Drop-in for ATOM's Triton ``_index_block_score_kernel`` on the FP4 index cache
(``--index-cache-dtype fp4``): for request b, query row r, index head h and
128-token block blk,

    score[h, r, blk] = max over the block's keys k at or before r's position of
                       dot(e4m3(clamp(q[r, h], +-448)), dequant(K[page, k])) * (sm_scale * log2 e)

stored for the blocks up to the CTA's last row's block, as Triton stores them
(the same grid: 256-row query tiles x batch*heads x block chunks). Every dot is
the one ``v_wmma_scale16_f32_16x16x128_f8f6f4`` Triton's ``tl.dot_scaled`` emits
there (A = the page's e2m1 rows with their E4M3 block scales, B = the E4M3 query
with unit E8M0 scales, K = 128 = head_dim), the max is exact and the scale is
applied after it (rounding is monotone): the scores are bitwise Triton's.

What differs is the data movement. Triton gathers the FP4 rows with byte and
short global loads and byte LDS stores (128 + 36 loads and 128 LDS byte stores a
page and wave); here the page (8 KB payload + 1 KB scales) arrives by tensor DMA
into a CTA-shared LDS ring (wave 0 the payload, wave 1 the scales, the decode
tiles' one-copy-a-wave pattern: each wave waits on its own copies, then a CTA
barrier) and every wave reads the A operands with 16-byte LDS loads. Eight waves
own two 16-row query tiles each; a page is 8 key tiles x 2 query tiles = 16
WMMAs a wave.

The ``select`` build also keeps each row's 16 best blocks of the CTA's chunk, so
the scores never leave LDS: see ``compile_index_score_prefill_nvfp4``.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp, T

from aiter.jit.utils.chip_info import get_lds_capacity_bytes

from .pa_decode.op_lds import Gfx1250Memory, _gptr
from .pa_decode.pipeline_nvfp4 import _wmma_scale16

WAVE = 32
HEAD_DIM = 128
PAGE = 128
ROW = HEAD_DIM // 2  # payload bytes a key
SROW = HEAD_DIM // 16  # scale bytes a key
PAGE_BYTES = PAGE * ROW + PAGE * SROW  # 8 KB payload + 1 KB scales
E4M3_MAX = 448.0
LOG2E_TRITON = 1.4426950409  # the literal of ATOM's kernel (fp32 multiply)
TOPK = 16  # blocks a row keeps in the ``select`` build
INT32_MIN = -(1 << 31)
# An empty place of a row's list (its place in the low bits, as in every
# entry): under every block's entry.
EMPTY = INT32_MIN << 32
# ATOM's forced blocks (`_score_tile_keys`): init, then local over it.
INIT_PIN = 1e30
LOCAL_PIN = 1e29


def sm_scale_log2e(sm_scale: float) -> float:
    """``sm_scale * 1.4426950409`` in FP32, as the Triton kernel computes it."""
    return float(np.float32(sm_scale) * np.float32(LOG2E_TRITON))


def _triton_order(score):
    """A score as the int32 that orders like ATOM's ``_pack_score_key``: -0.0 is
    0.0 and a NaN is under -inf (``INT32_MIN``)."""
    bits = score.bitcast(fx.Int32)
    bits = (bits == INT32_MIN).select(fx.Int32(0), bits)
    order = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return ((bits & 0x7FFFFFFF) > 0x7F800000).select(fx.Int32(INT32_MIN), order)


class _Lds(Gfx1250Memory):
    class _T:
        kv_dtype = fx.BFloat16

    def __init__(self, base):  # noqa: D107
        self.base = base
        self.t = self._T


_CACHE = {}


# Blocks a CTA scores when its scores go out through LDS (``chunk_max``): a
# wave stores one row's chunk of at most WAVE scores with one coalesced store.
CHUNK_MAX = 32


def compile_index_score_prefill_nvfp4(
    *, block_q=256, waves=8, ring=4, chunk_max=0, select=None, _probe=""
):
    """Compile (cached) the scorer: ``block_q`` query rows a CTA (``waves`` waves of
    ``block_q / waves`` rows, a multiple of 16), ``ring`` pages in flight.

    ``select`` = (init_blocks, local_blocks) builds the fused scorer + top-16
    instead (``chunk_max`` = ``CHUNK_MAX``, waves of whole 32 rows). The scores
    stay in LDS, and ``score_ptr`` is ``[heads, total_q, 1 + chunks * TOPK]``
    int64: a row's ``TOPK`` best blocks of chunk z from ``1 + z * TOPK``, as
    ATOM's ``_pack_score_key`` keys (score, then the larger block; a NaN last;
    the forced blocks pinned), in no order, 0 for no block.

    Every ``CHUNK_MAX`` blocks a wave takes its rows a lane each, 32 at a time.
    A row keeps its ``TOPK`` best blocks of the chunk as a list in LDS; a new
    block over the list's lowest takes that one's place and the lowest is found
    again (16 loads a lane). Only blocks at or over the row's floor are tried,
    one a row a step. The floor is the score order of some full list's lowest
    block -- 16 blocks of the row are at or over it, so none under it is one of
    the row's 16 best. A row's CTAs (one a chunk) share it through column 0 of
    ``score_ptr``, which the caller sets to ``INT32_MIN``: a CTA stores the
    floor its list reaches and reads the others' at each pass. Any value read
    is some list's floor, so which one it is changes what a chunk keeps under
    the row's 16 best, never those.
    """
    tiles_per_wave = block_q // (16 * waves)
    if block_q % (16 * waves) or tiles_per_wave < 1:
        raise ValueError("block_q must be a multiple of 16 * waves")
    if waves < 2:
        # Wave 0 copies a page's payload and wave 1 its scales.
        raise ValueError(f"waves must be at least 2, got {waves}")
    if not 2 <= ring <= 8:
        raise ValueError("ring must be in [2, 8]")
    if chunk_max not in (0, CHUNK_MAX):
        raise ValueError(f"chunk_max must be 0 or {CHUNK_MAX}")
    rows_per_wave = block_q // waves
    if select and (not chunk_max or rows_per_wave % WAVE):
        raise ValueError(
            f"select needs chunk_max = {CHUNK_MAX} and waves of whole {WAVE} rows"
        )
    key = (block_q, waves, ring, chunk_max, select, _probe)
    if key in _CACHE:
        return _CACHE[key]
    threads = waves * WAVE
    ring_bytes = ring * PAGE_BYTES
    list_base = ring_bytes + block_q * chunk_max * 4
    total = list_base + (block_q * TOPK * 8 if select else 0)
    if total > get_lds_capacity_bytes("gfx1250"):
        raise ValueError(f"{total} bytes of LDS are over the workgroup's limit")

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total // 4, 16]

    name = f"index_score_prefill_nvfp4_d128_bq{block_q}_w{waves}_ring{ring}{_probe}"
    if select:
        init_blocks, local_blocks = select
        name = name.replace("score", "topk") + f"_i{init_blocks}_l{local_blocks}"
    elif chunk_max:
        name += f"_lds{chunk_max}"

    @flyc.kernel(name=name, known_block_size=(threads, 1, 1))
    def index_score_kernel(
        q_ptr: fx.Pointer,  # [total_q, heads, 128] bf16
        payload_ptr: fx.Pointer,  # [blocks, 128, 64] u8
        scale_ptr: fx.Pointer,  # [blocks, 128, 8] u8 (E4M3)
        score_ptr: fx.Pointer,  # [heads, total_q, max_block] f32 (select: keys)
        block_table_ptr: fx.Pointer,  # [reqs, max_blocks] i32
        cu_seqlens: fx.Pointer,
        seq_lens: fx.Pointer,
        prefix_lens: fx.Pointer,
        num_heads: fx.Int32,
        scale: fx.Float32,  # sm_scale * log2(e), FP32
        chunk_blocks: fx.Int32,
        stride_q_n: fx.Int32,
        stride_q_h: fx.Int32,
        stride_p_blk: fx.Int32,  # payload page stride (bytes)
        stride_s_blk: fx.Int32,  # scale page stride (bytes)
        stride_s_h: fx.Int32,
        stride_s_n: fx.Int32,
        stride_bt_b: fx.Int32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = fx.Int32(fx.rocdl.readfirstlane(T.i32, tid // WAVE))
        lane = tid % WAVE
        lane16, half = lane % 16, lane // 16
        pid_q = fx.Int32(gpu.block_id("x"))
        pid_bh = fx.Int32(gpu.block_id("y"))
        pid_chunk = fx.Int32(gpu.block_id("z"))
        pid_b = pid_bh // num_heads
        pid_h = pid_bh % num_heads
        cu = fx.recast_iter(fx.Int32, cu_seqlens)
        seq_start = fx.Int32(cu[pid_b])
        q_len = fx.Int32(cu[pid_b + 1]) - seq_start
        seq_len = fx.Int32(fx.recast_iter(fx.Int32, seq_lens)[pid_b])
        prefix = fx.Int32(fx.recast_iter(fx.Int32, prefix_lens)[pid_b])
        hi = fx.min(seq_len, prefix + (pid_q + 1) * block_q)
        blk_end = (hi + PAGE - 1) // PAGE
        blk_lo = pid_chunk * chunk_blocks
        blk_hi = fx.min(blk_lo + chunk_blocks, blk_end)
        # CTA-uniform: Triton's early exits (past the request's rows or blocks).
        live = (pid_q * block_q < q_len) & (blk_lo < blk_end)
        storage = fx.SharedAllocator().allocate(SharedStorage).peek()
        mem = _Lds(fx.recast_iter(fx.Uint8, storage.buf.ptr))

        # Closures, not ``mem.`` calls, inside runtime branches: FlyDSL's
        # rewriter would carry ``mem`` as branch state.
        def lds_load(offset, dtype, n=1):
            return mem.load(offset, dtype, n)

        def tdm(*args):
            mem.tdm_copy(*args)

        def lds_store(offset, value, dtype):
            mem.store(offset, value, dtype)

        bt = fx.recast_iter(fx.Int32, block_table_ptr)

        def stage(blk, slot):
            """Wave 0 DMAs the page's payload, wave 1 its scales (whole rows)."""
            page = fx.Int32(bt[pid_b * stride_bt_b + blk])
            base = slot * PAGE_BYTES
            if warp == 0:
                tdm(
                    _gptr(payload_ptr, fx.BFloat16, 16),
                    fx.Int64(page) * fx.Int64(stride_p_blk) // 2,
                    base,
                    (PAGE * ROW // 2,),
                    (1,),
                    (None,),
                    8,
                    8,
                )
            if warp == 1:
                tdm(
                    _gptr(scale_ptr, fx.BFloat16, 16),
                    fx.Int64(page) * fx.Int64(stride_s_blk) // 2,
                    base + PAGE * ROW,
                    (PAGE * SROW // 2,),
                    (1,),
                    (None,),
                    8,
                    8,
                )

        # Query: e4m3(clamp(q)) as the B operand of each of this wave's tiles
        # (lane (row, half): dims 32 j + 16 half + [0, 16), j < 4).
        q_bf16 = fx.recast_iter(fx.BFloat16, q_ptr)
        row0 = pid_q * block_q + warp * (16 * tiles_per_wave)
        q_ops = []
        for rt in range_constexpr(tiles_per_wave):
            r = row0 + rt * 16 + lane16
            safe_r = (r < q_len).select(r, fx.Int32(0))
            qbase = fx.Int64(seq_start + safe_r) * stride_q_n + pid_h * stride_q_h
            words = []
            for j in range_constexpr(4):
                vals = fx.Vector(
                    fx.ptr_load(
                        fx.add_offset(q_bf16, qbase + j * 32 + half * 16),
                        result_type=fx.Vector.make_type(16, fx.BFloat16),
                    )
                ).to(fx.Float32)
                live_row = (r < q_len).select(fx.Float32(1.0), fx.Float32(0.0))
                for h in range_constexpr(2):
                    eight = fx.Vector.from_elements(
                        [
                            fx.maxnumf(
                                fx.minnumf(vals[h * 8 + e] * live_row, fx.Float32(E4M3_MAX)),
                                fx.Float32(-E4M3_MAX),
                            )
                            for e in range_constexpr(8)
                        ],
                        dtype=fx.Float32,
                    )
                    pair = mem.fp8(eight)
                    words.extend([pair[0], pair[1]])
            q_ops.append(fx.Vector.from_elements(words, dtype=fx.Int32))

        if live:
            for ahead in range_constexpr(ring - 1):
                if blk_lo + ahead < blk_hi:
                    stage(blk_lo + ahead, fx.Int32(ahead))
            score = fx.recast_iter(fx.Float32, score_ptr)
            q_start = prefix + pid_q * block_q
            if const_expr(select):
                # This wave's live rows, their lists empty.
                n_rows = fx.max(
                    fx.min(q_len - row0, fx.Int32(rows_per_wave)), fx.Int32(0)
                )
                wave_row = warp * rows_per_wave
                out_row = fx.Int64(pid_h) * stride_s_h + fx.Int64(
                    seq_start + pid_q * block_q
                ) * fx.Int64(stride_s_n)
                floors = fx.recast_iter(fx.Int32, score_ptr)
                # An entry: (score order << 32) | (block + 1) << 4 | its place.
                for i in range_constexpr(rows_per_wave * TOPK // WAVE):
                    lds_store(
                        list_base + (wave_row * TOPK + i * WAVE + lane) * 8,
                        fx.Vector.from_elements(
                            [fx.Int64(EMPTY) | fx.Int64(lane16)], dtype=fx.Int64
                        ),
                        fx.Int64,
                    )

                def lowest(rl):
                    """The lowest entry of row ``rl``'s list."""
                    low = lds_load(list_base + rl * TOPK * 8, fx.Int64, 1)[0]
                    for p in range_constexpr(1, TOPK):
                        entry = lds_load(list_base + (rl * TOPK + p) * 8, fx.Int64, 1)[0]
                        low = (entry < low).select(entry, low)
                    return low

                def low_bits(n):
                    """A mask of the n lowest bits, n in [0, 32]."""
                    return (n >= 32).select(
                        fx.Int32(-1), (fx.Int32(1) << fx.min(n, fx.Int32(31))) - 1
                    )

            init = [fx.Int32(0)]
            for blk_i, state in range(blk_lo, blk_hi, 1, init=init):
                blk = fx.Int32(blk_i)
                slot = (blk - blk_lo) % ring
                # This wave's copies of the page are the oldest in flight.
                outstanding = fx.min(blk_hi - 1 - blk, fx.Int32(ring - 2))
                for n in range_constexpr(ring - 1):
                    if outstanding == n:
                        tdm_ops.tensor_wait(n)
                gpu.barrier()
                if blk + (ring - 1) < blk_hi:
                    stage(blk + (ring - 1), (blk + (ring - 1) - blk_lo) % ring)
                base = slot * PAGE_BYTES
                # Unit E8M0 scales of the query, in VGPRs (pipeline_nvfp4: a
                # uniform i64 reaches the WMMA wrongly as a constant / SGPRs).
                unit = fx.Int64(0x7F7F7F7F7F7F7F7F) | fx.Int64((lane + blk) >> 31)
                diagonal = q_start < (blk + 1) * PAGE

                def page_max(masked):
                    """Per query tile, this lane's max over its keys of the page
                    (causally masked on the diagonal pages only)."""
                    out = [fx.Float32(float("-inf")) for _ in range_constexpr(tiles_per_wave)]
                    for kt in range_constexpr(PAGE // 16):
                        key = kt * 16 + lane16
                        at = base + key * ROW + half * 16
                        frags = [lds_load(at + j * 32, fx.Int32, 4) for j in range_constexpr(2)]
                        k_rows = fx.Vector.from_elements(
                            [frags[j][i] for j in range_constexpr(2) for i in range_constexpr(4)],
                            dtype=fx.Int32,
                        )
                        k_scale = lds_load(base + PAGE * ROW + key * SROW, fx.Int64, 1)[0]
                        for rt in range_constexpr(tiles_per_wave):
                            dots = _wmma_scale16(
                                k_rows,
                                q_ops[rt],
                                fx.Vector.filled(8, 0.0, fx.Float32),
                                k_scale,
                                unit,
                                "fp4_e2m1",
                                "fp8_e4m3",
                                "e8",
                            )
                            vals = [dots[e] for e in range_constexpr(8)]
                            if const_expr(masked):
                                r_pos = prefix + row0 + rt * 16 + lane16
                                vals = [
                                    (blk * PAGE + kt * 16 + half * 8 + e <= r_pos).select(
                                        vals[e], fx.Float32(float("-inf"))
                                    )
                                    for e in range_constexpr(8)
                                ]
                            m = fx.Vector.from_elements(vals, dtype=fx.Float32).reduce(
                                ReductionOp.MAX
                            )
                            out[rt] = fx.maxnumf(out[rt], m)
                    return out

                maxes = [fx.Float32(float("-inf")) for _ in range_constexpr(tiles_per_wave)]
                if diagonal:
                    maxes = page_max(True)
                else:
                    maxes = page_max(False)
                for rt in range_constexpr(tiles_per_wave):
                    m = fx.maxnumf(maxes[rt], maxes[rt].shuffle_xor(16, WAVE))
                    r = row0 + rt * 16 + lane16
                    # Debug probe ``_nostore``: the store's condition is false at
                    # run time (the compiler cannot tell), so only the stores go.
                    limit = q_len - 1000000000 if const_expr(_probe == "_nostore") else q_len
                    if const_expr(chunk_max):
                        # Staged in LDS: a wave stores whole row chunks below,
                        # or keeps each row's best of them.
                        at = blk - blk_lo
                        if const_expr(select):
                            at = at % chunk_max
                        if half == 0:
                            lds_store(
                                ring_bytes + ((r - pid_q * block_q) * chunk_max + at) * 4,
                                fx.Vector.from_elements([m * scale], dtype=fx.Float32),
                                fx.Float32,
                            )
                    else:
                        if (half == 0) & (r < limit):
                            score[
                                fx.Int64(pid_h) * stride_s_h
                                + fx.Int64(seq_start + r) * stride_s_n
                                + blk
                            ] = m * scale
                if const_expr(select):
                    at = (blk - blk_lo) % chunk_max
                    if (at == chunk_max - 1) | (blk == blk_hi - 1):
                        # The blocks staged since the last pass: blk - at .. blk.
                        for g in range_constexpr(rows_per_wave // WAVE):
                            rl = wave_row + g * WAVE + lane
                            row_live = g * WAVE + lane < n_rows
                            cell = (
                                out_row
                                + fx.Int64(row_live.select(rl, fx.Int32(0)))
                                * fx.Int64(stride_s_n)
                            ) * 2
                            low = lowest(rl)
                            # An empty place or a NaN is the lowest order: no floor.
                            floor = fx.max(fx.Int32(floors[cell]), fx.Int32(low >> 32))
                            # The score of that order (a NaN for no floor: under
                            # no score). A NaN score passes too, to be ordered
                            # against the list.
                            bound = (floor ^ ((floor >> 31) & 0x7FFFFFFF)).bitcast(
                                fx.Float32
                            )
                            own_block = (q_start + rl) >> (PAGE.bit_length() - 1)
                            enter = fx.Int32(0)
                            for c in range_constexpr(chunk_max):
                                under = (
                                    lds_load(
                                        ring_bytes + (rl * chunk_max + c) * 4,
                                        fx.Float32,
                                        1,
                                    )[0]
                                    < bound
                                )
                                enter = enter | under.select(
                                    fx.Int32(0), fx.Int32(1 << c)
                                )
                            # A forced block is tried whatever it scores. Bit c
                            # is block ``first + c``.
                            first = blk - at
                            if const_expr(init_blocks):
                                enter = enter | low_bits(
                                    fx.max(init_blocks - first, fx.Int32(0))
                                )
                            if const_expr(local_blocks):
                                enter = enter | ~low_bits(
                                    fx.max(own_block + 1 - local_blocks - first, fx.Int32(0))
                                )
                            # Of the blocks staged, those the row may attend.
                            reach = fx.min(own_block, blk) + 1 - first
                            enter = enter & low_bits(
                                row_live.select(fx.max(reach, fx.Int32(0)), fx.Int32(0))
                            )
                            # A block a row a step, for the most any row has.
                            steps = fx.Int32(fx.ctpop(enter))
                            for half_wave in (16, 8, 4, 2, 1):
                                steps = fx.max(
                                    steps,
                                    steps.shuffle_xor(fx.Int32(half_wave), fx.Int32(WAVE)),
                                )
                            # Raw readfirstlane: fx has no lane-read operation.
                            for _ in range(
                                fx.Int32(0),
                                fx.Int32(fx.rocdl.readfirstlane(T.i32, steps)),
                                fx.Int32(1),
                            ):
                                has = enter != 0
                                c = fx.Int32(fx.cttz(enter))
                                enter = enter & (enter - 1)
                                col = first + c
                                value = lds_load(
                                    ring_bytes + (rl * chunk_max + c) * 4, fx.Float32, 1
                                )[0]
                                if const_expr(init_blocks):
                                    value = (col < init_blocks).select(
                                        fx.Float32(INIT_PIN), value
                                    )
                                if const_expr(local_blocks):
                                    value = (col > own_block - local_blocks).select(
                                        fx.Float32(LOCAL_PIN), value
                                    )
                                entry = (fx.Int64(_triton_order(value)) << 32) | fx.Int64(
                                    (col + 1) << 4
                                )
                                if has & (entry > low):
                                    place = fx.Int32(low) & (TOPK - 1)
                                    lds_store(
                                        list_base + (rl * TOPK + place) * 8,
                                        fx.Vector.from_elements(
                                            [entry | fx.Int64(place)], dtype=fx.Int64
                                        ),
                                        fx.Int64,
                                    )
                                low = lowest(rl)
                            # (A row past the request keeps an empty list.)
                            if fx.Int32(low >> 32) > floor:
                                floors[cell] = fx.Int32(low >> 32)
                final = yield [state[0]]
            tdm_ops.tensor_wait(0)
            if const_expr(select):
                keys = fx.recast_iter(fx.Int64, score_ptr)
                for i in range(fx.Int32(0), n_rows, fx.Int32(1)):
                    rl = wave_row + i
                    entry = lds_load(list_base + (rl * TOPK + lane16) * 8, fx.Int64, 1)[0]
                    block1 = (entry & 0xFFFFFFFF) >> 4
                    ordered = ((entry >> 32) ^ INT32_MIN) & 0xFFFFFFFF
                    if lane < TOPK:
                        keys[
                            out_row
                            + fx.Int64(rl) * fx.Int64(stride_s_n)
                            + (1 + pid_chunk * TOPK + lane)
                        ] = (block1 != 0).select(
                            fx.Int64(1 << 48) | (ordered << 16) | block1, fx.Int64(0)
                        )
            elif const_expr(chunk_max):
                gpu.barrier()
                n_blk = blk_hi - blk_lo
                for i in range_constexpr(rows_per_wave):
                    rl = warp * rows_per_wave + i
                    r = pid_q * block_q + rl
                    if (r < q_len) & (lane < n_blk):
                        score[
                            fx.Int64(pid_h) * stride_s_h
                            + fx.Int64(seq_start + r) * stride_s_n
                            + blk_lo
                            + lane
                        ] = lds_load(ring_bytes + (rl * chunk_max + lane) * 4, fx.Float32, 1)[0]

    @flyc.jit
    def launch(
        q_ptr: fx.Pointer,
        payload_ptr: fx.Pointer,
        scale_ptr: fx.Pointer,
        score_ptr: fx.Pointer,
        block_table_ptr: fx.Pointer,
        cu_seqlens: fx.Pointer,
        seq_lens: fx.Pointer,
        prefix_lens: fx.Pointer,
        num_heads: fx.Int32,
        scale: fx.Float32,
        chunk_blocks: fx.Int32,
        stride_q_n: fx.Int32,
        stride_q_h: fx.Int32,
        stride_p_blk: fx.Int32,
        stride_s_blk: fx.Int32,
        stride_s_h: fx.Int32,
        stride_s_n: fx.Int32,
        stride_bt_b: fx.Int32,
        q_tiles: fx.Int32,
        batch: fx.Int32,
        chunks: fx.Int32,
        stream: fx.Stream,
    ):
        index_score_kernel(
            q_ptr,
            payload_ptr,
            scale_ptr,
            score_ptr,
            block_table_ptr,
            cu_seqlens,
            seq_lens,
            prefix_lens,
            num_heads,
            scale,
            chunk_blocks,
            stride_q_n,
            stride_q_h,
            stride_p_blk,
            stride_s_blk,
            stride_s_h,
            stride_s_n,
            stride_bt_b,
        ).launch(
            grid=(q_tiles, batch * num_heads, chunks),
            block=(threads, 1, 1),
            stream=stream,
        )

    compiled = {"launch": launch, "kernel": index_score_kernel}
    _CACHE[key] = compiled
    return compiled
