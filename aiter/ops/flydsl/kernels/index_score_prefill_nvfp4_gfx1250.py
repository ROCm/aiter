# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx1250 causal prefill block scores of index queries over an NVFP4 key cache.

The cache is two byte views, each with its own page stride: e2m1 pairs
``[pages, 128, 64]`` and one E4M3 scale per 16 values ``[pages, 128, 8]``. For
request b, query row r, index head h and 128-key block blk of b's pages,

    score[h, r, blk] = max over the block's keys k at or before r's position of
                       dot(e4m3(clamp(q[r, h], +-448)), dequant(K[page, k])) * (sm_scale * log2 e)

stored for the blocks up to the block of the last row of r's ``block_q``-row tile
(grid: query tiles x batch*heads x block chunks). Every dot is one
``v_wmma_scale16_f32_16x16x128_f8f6f4`` (A = the page's e2m1 rows with their E4M3
block scales, B = the E4M3 query with unit E8M0 scales, K = 128 = head_dim), so K
is never widened. The max is taken before the scale, which is exact for a
positive scale (rounding is monotone).

Data movement: the page (8 KB payload + 1 KB scales) arrives by tensor DMA into a
CTA-shared LDS ring of ``ring`` pages (wave 0 copies the payload, wave 1 the
scales, as in the decode tiles: each wave waits on its own copies, then a CTA
barrier), and every wave reads the A operands with 16-byte LDS loads. Each wave
owns ``block_q / (16 * waves)`` 16-row query tiles (4 for the wrapper's 256 rows
over 4 waves); a page is 8 key tiles x that many WMMAs a wave. A CTA's scores are
staged in LDS and leave as one coalesced store per row chunk.

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

from .minimax_m3_index_score import DATA_BYTES, LOG2E, NVFP4_SCALE_BYTES, PAGE
from .pa_decode.op_lds import Gfx1250Memory, _gptr
from .pa_decode.pipeline_nvfp4 import _wmma_scale16

WAVE = 32
ROW, SROW = DATA_BYTES, NVFP4_SCALE_BYTES  # payload / scale bytes a key
PAGE_BYTES = PAGE * ROW + PAGE * SROW  # 8 KB payload + 1 KB scales
E4M3_MAX = 448.0
TOPK = 16  # blocks a row keeps in the ``select`` build
INT32_MIN = -(1 << 31)
# An empty place of a row's list (its place in the low bits, as in every
# entry): under every block's entry.
EMPTY = INT32_MIN << 32
# The forced blocks of the selection (ATOM's ``_score_tile_keys``): the first
# ``init_blocks`` blocks, then the row's last ``local_blocks`` over them.
INIT_PIN = 1e30
LOCAL_PIN = 1e29


def sm_scale_log2e(sm_scale: float) -> float:
    """The kernel's scale: ``sm_scale * LOG2E``, multiplied in FP32."""
    return float(np.float32(sm_scale) * np.float32(LOG2E))


def _score_order(score):
    """A score as the int32 that orders like ATOM's ``_pack_score_key``: -0.0 is
    0.0 and a NaN is under -inf (``INT32_MIN``)."""
    bits = score.bitcast(fx.Int32)
    bits = (bits == INT32_MIN).select(fx.Int32(0), bits)
    order = bits ^ ((bits >> 31) & 0x7FFFFFFF)
    return ((bits & 0x7FFFFFFF) > 0x7F800000).select(fx.Int32(INT32_MIN), order)


class _Lds(Gfx1250Memory):
    class _T:
        kv_dtype = fx.BFloat16

    def __init__(self, base):
        self.base = base
        self.t = self._T


_CACHE = {}


# Blocks a CTA scores: its scores are staged in LDS, and a wave stores one row's
# chunk of at most WAVE scores with one coalesced store (on MI455X, one 4-byte
# store per block and row instead was up to 1.37x slower, at most 0.5 us faster).
CHUNK_MAX = 32


def compile_index_score_prefill_nvfp4(
    *, block_q, waves, ring, select=None, fold_max=False, pad_lds=False, sel32=False
):
    """Compile (cached) the scorer: ``block_q`` query rows a CTA (``waves`` waves of
    ``block_q / waves`` rows, a multiple of 16), ``ring`` pages in flight.

    ``fold_max``: each WMMA's 8 values folded one by one into the running max
    instead of a vector reduction first. The same maxnum, so the same scores (NaN
    handling included); the reduction leaves two-input maxnums whose operands
    LLVM quiets, the fold is ``v_max3_num_f32`` only (4 VALU instructions a WMMA
    instead of ~5.6 on an unmasked page).

    ``pad_lds``: the page's payload rows 80 bytes apart in LDS (16 bytes of
    padding a 64-byte row, by the tensor DMA). A wave reads the A operand with
    16-byte loads, lane l16 at row l16: 64 bytes apart, the 16 lanes of a half
    hit only 4 of the 16 four-bank groups (a 4-way bank conflict); 80 bytes
    apart they hit all 16. The rows of the staged scores and of the select
    build's lists (128 bytes, half the banks' period) get 8 bytes of padding too.
    Same scores; 2 KB more LDS a ring slot.

    ``select`` = (init_blocks, local_blocks) builds the fused scorer + top-16
    instead (waves of whole 32 rows). The scores stay in LDS, and ``score_ptr``
    is ``[heads, total_q, 1 + chunks * TOPK]`` int64: a row's ``TOPK`` best
    blocks of chunk z from ``1 + z * TOPK``, as ATOM's ``_pack_score_key`` keys
    (score, then the larger block; a NaN last; the forced blocks pinned), in no
    order, 0 for no block.

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

    ``sel32`` (select build): the selection pass with 32-bit compares only (a
    64-bit min / max / compare costs about 20 times a 32-bit VALU op on this
    SIMD). Blocks are tried in rising order, so a tried block is over every
    block of the list, and over the list's lowest entry exactly when its score
    order is not under the lowest's; the lowest entry is the lowest order, then
    the lowest low word among the places at that order. The same lists.
    """
    if waves < 2:  # wave 0 copies a page's payload, wave 1 its scales
        raise ValueError("waves must be at least 2")
    tiles_per_wave = block_q // (16 * waves)
    if block_q % (16 * waves) or tiles_per_wave < 1:
        raise ValueError("block_q must be a multiple of 16 * waves")
    if not 2 <= ring <= 8:
        raise ValueError("ring must be in [2, 8]")
    rows_per_wave = block_q // waves
    if select and rows_per_wave % WAVE:
        raise ValueError(f"select needs waves of whole {WAVE} rows")
    key = (block_q, waves, ring, select, fold_max, pad_lds, sel32)
    if key in _CACHE:
        return _CACHE[key]
    threads = waves * WAVE
    # LDS pitch of a payload row, and a ring slot: the payload, then the scales.
    prow = ROW + 16 if pad_lds else ROW
    page_bytes = PAGE * prow + PAGE * SROW
    ring_bytes = ring * page_bytes
    # Pitch of a row's staged scores and of its list.
    srow_b = CHUNK_MAX * 4 + (8 if pad_lds else 0)
    lrow_b = TOPK * 8 + (8 if pad_lds else 0)
    list_base = ring_bytes + block_q * srow_b
    total = list_base + (block_q * lrow_b if select else 0)
    if total > get_lds_capacity_bytes("gfx1250"):
        raise ValueError(f"{total} bytes of LDS are over the workgroup's limit")

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total // 4, 16]

    name = f"index_score_prefill_nvfp4_d128_bq{block_q}_w{waves}_ring{ring}"
    if fold_max:
        name += "_fold"
    if pad_lds:
        name += "_pad"
    if sel32:
        name += "_s32"
    if select:
        init_blocks, local_blocks = select
        name = name.replace("score", "topk") + f"_i{init_blocks}_l{local_blocks}"
    else:
        name += f"_lds{CHUNK_MAX}"

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
        # CTA-uniform early exit: past the request's rows or blocks.
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

        def to_fp8(values):
            return mem.fp8(values)

        bt = fx.recast_iter(fx.Int32, block_table_ptr)

        def stage(blk, slot):
            """Wave 0 DMAs the page's payload, wave 1 its scales (whole rows)."""
            page = fx.Int32(bt[pid_b * stride_bt_b + blk])
            base = slot * page_bytes
            if warp == 0:
                if const_expr(pad_lds):
                    # Rows of ROW bytes, prow bytes apart in LDS.
                    tdm(
                        _gptr(payload_ptr, fx.BFloat16, 16),
                        fx.Int64(page) * fx.Int64(stride_p_blk) // 2,
                        base,
                        (PAGE, ROW // 2),
                        (ROW // 2, 1),
                        (None, None),
                        ROW // 2,
                        prow // 2,
                    )
                else:
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
                    base + PAGE * prow,
                    (PAGE * SROW // 2,),
                    (1,),
                    (None,),
                    8,
                    8,
                )

        row0 = pid_q * block_q + warp * (16 * tiles_per_wave)

        def page_max(masked, base, unit, blk, q_ops):
            """Per query tile, this lane's max over its keys of the page
            (causally masked on the diagonal pages only)."""
            out = [fx.Float32(float("-inf")) for _ in range_constexpr(tiles_per_wave)]
            for kt in range_constexpr(PAGE // 16):
                key = kt * 16 + lane16
                at = base + key * prow + half * 16
                frags = [lds_load(at + j * 32, fx.Int32, 4) for j in range_constexpr(2)]
                k_rows = fx.Vector.from_elements(
                    [
                        frags[j][i]
                        for j in range_constexpr(2)
                        for i in range_constexpr(4)
                    ],
                    dtype=fx.Int32,
                )
                k_scale = lds_load(base + PAGE * prow + key * SROW, fx.Int64, 1)[0]
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
                    if const_expr(fold_max):
                        for e in range_constexpr(8):
                            out[rt] = fx.maxnumf(out[rt], vals[e])
                    else:
                        m = fx.Vector.from_elements(vals, dtype=fx.Float32).reduce(
                            ReductionOp.MAX
                        )
                        out[rt] = fx.maxnumf(out[rt], m)
            return out

        if live:
            # Query: e4m3(clamp(q)) as the B operand of each of this wave's tiles
            # (lane (row, half): dims 32 j + 16 half + [0, 16), j < 4). Loaded by live
            # CTAs only: a dead one's request may be empty and end at idx_q's end.
            q_bf16 = fx.recast_iter(fx.BFloat16, q_ptr)
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
                                    fx.minnumf(
                                        vals[h * 8 + e] * live_row, fx.Float32(E4M3_MAX)
                                    ),
                                    fx.Float32(-E4M3_MAX),
                                )
                                for e in range_constexpr(8)
                            ],
                            dtype=fx.Float32,
                        )
                        pair = to_fp8(eight)
                        words.extend([pair[0], pair[1]])
                q_ops.append(fx.Vector.from_elements(words, dtype=fx.Int32))
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
                        list_base + (wave_row + i * 2 + half) * lrow_b + lane16 * 8,
                        fx.Vector.from_elements(
                            [fx.Int64(EMPTY) | fx.Int64(lane16)], dtype=fx.Int64
                        ),
                        fx.Int64,
                    )

                BIG = 0x7FFFFFFF  # over every low word ((block + 1) << 4 | place)

                def lowest(rl):
                    """The lowest entry of row ``rl``'s list, as (its score order,
                    the entry: 64-bit, or with sel32 its low word)."""
                    if const_expr(sel32):
                        ws = [
                            lds_load(list_base + rl * lrow_b + p * 8, fx.Int32, 2)
                            for p in range_constexpr(TOPK)
                        ]
                        order = ws[0][1]
                        for p in range_constexpr(1, TOPK):
                            order = fx.min(order, ws[p][1])
                        word = fx.Int32(BIG)
                        for p in range_constexpr(TOPK):
                            word = fx.min(
                                word,
                                (ws[p][1] == order).select(ws[p][0], fx.Int32(BIG)),
                            )
                        return order, word
                    low = lds_load(list_base + rl * lrow_b, fx.Int64, 1)[0]
                    for p in range_constexpr(1, TOPK):
                        entry = lds_load(list_base + rl * lrow_b + p * 8, fx.Int64, 1)[
                            0
                        ]
                        low = (entry < low).select(entry, low)
                    return fx.Int32(low >> 32), low

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
                base = slot * page_bytes
                # Unit E8M0 scales of the query, in VGPRs (pipeline_nvfp4: a
                # uniform i64 reaches the WMMA wrongly as a constant / SGPRs).
                unit = fx.Int64(0x7F7F7F7F7F7F7F7F) | fx.Int64((lane + blk) >> 31)
                diagonal = q_start < (blk + 1) * PAGE

                maxes = [
                    fx.Float32(float("-inf")) for _ in range_constexpr(tiles_per_wave)
                ]
                if diagonal:
                    maxes = page_max(True, base, unit, blk, q_ops)
                else:
                    maxes = page_max(False, base, unit, blk, q_ops)
                for rt in range_constexpr(tiles_per_wave):
                    m = fx.maxnumf(maxes[rt], maxes[rt].shuffle_xor(16, WAVE))
                    r = row0 + rt * 16 + lane16
                    # Staged in LDS: a wave stores whole row chunks below, or
                    # keeps each row's best of them.
                    at = (blk - blk_lo) % CHUNK_MAX
                    if half == 0:
                        lds_store(
                            ring_bytes + (r - pid_q * block_q) * srow_b + at * 4,
                            fx.Vector.from_elements([m * scale], dtype=fx.Float32),
                            fx.Float32,
                        )
                if const_expr(select):
                    at = (blk - blk_lo) % CHUNK_MAX
                    if (at == CHUNK_MAX - 1) | (blk == blk_hi - 1):
                        # The blocks staged since the last pass: blk - at .. blk.
                        for g in range_constexpr(rows_per_wave // WAVE):
                            rl = wave_row + g * WAVE + lane
                            row_live = g * WAVE + lane < n_rows
                            cell = (
                                out_row
                                + fx.Int64(row_live.select(rl, fx.Int32(0)))
                                * fx.Int64(stride_s_n)
                            ) * 2
                            low_o, low = lowest(rl)
                            # An empty place or a NaN is the lowest order: no floor.
                            floor = fx.max(fx.Int32(floors[cell]), low_o)
                            # The score of that order (a NaN for no floor: under
                            # no score). A NaN score passes too, to be ordered
                            # against the list.
                            bound = (floor ^ ((floor >> 31) & 0x7FFFFFFF)).bitcast(
                                fx.Float32
                            )
                            own_block = (q_start + rl) >> (PAGE.bit_length() - 1)
                            enter = fx.Int32(0)
                            for c in range_constexpr(CHUNK_MAX):
                                under = (
                                    lds_load(
                                        ring_bytes + rl * srow_b + c * 4,
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
                                    fx.max(
                                        own_block + 1 - local_blocks - first,
                                        fx.Int32(0),
                                    )
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
                                    steps.shuffle_xor(
                                        fx.Int32(half_wave), fx.Int32(WAVE)
                                    ),
                                )
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
                                    ring_bytes + rl * srow_b + c * 4, fx.Float32, 1
                                )[0]
                                if const_expr(init_blocks):
                                    value = (col < init_blocks).select(
                                        fx.Float32(INIT_PIN), value
                                    )
                                if const_expr(local_blocks):
                                    value = (col > own_block - local_blocks).select(
                                        fx.Float32(LOCAL_PIN), value
                                    )
                                e_order = _score_order(value)
                                if const_expr(sel32):
                                    # A tried block is over every block of the list.
                                    over = e_order >= low_o
                                else:
                                    entry = (fx.Int64(e_order) << 32) | fx.Int64(
                                        (col + 1) << 4
                                    )
                                    over = entry > low
                                if has & over:
                                    place = fx.Int32(low) & (TOPK - 1)
                                    if const_expr(sel32):
                                        lds_store(
                                            list_base + rl * lrow_b + place * 8,
                                            fx.Vector.from_elements(
                                                [((col + 1) << 4) | place, e_order],
                                                dtype=fx.Int32,
                                            ),
                                            fx.Int32,
                                        )
                                    else:
                                        lds_store(
                                            list_base + rl * lrow_b + place * 8,
                                            fx.Vector.from_elements(
                                                [entry | fx.Int64(place)],
                                                dtype=fx.Int64,
                                            ),
                                            fx.Int64,
                                        )
                                    # Only a row that took a place has a new lowest.
                                    low_o, low = lowest(rl)
                            # (A row past the request keeps an empty list.)
                            if low_o > floor:
                                floors[cell] = low_o
                final = yield [state[0]]  # noqa: F841 -- the loop result is unused
            tdm_ops.tensor_wait(0)
            if const_expr(select):
                keys = fx.recast_iter(fx.Int64, score_ptr)
                for i in range(fx.Int32(0), n_rows, fx.Int32(1)):
                    rl = wave_row + i
                    entry = lds_load(list_base + rl * lrow_b + lane16 * 8, fx.Int64, 1)[
                        0
                    ]
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
            else:
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
                        ] = lds_load(
                            ring_bytes + rl * srow_b + lane * 4, fx.Float32, 1
                        )[
                            0
                        ]

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
