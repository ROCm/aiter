# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MiniMax-M3 decode index top-k: one wave a (query row, index head).

Reads the decode scorer's fp32 block scores ``score[h, row, blk]`` and writes
what ATOM's Triton ``_topk_index_packed_kernel`` writes in decode mode: the
row's ``topk`` best blocks by ATOM's ``_pack_score_key`` order, in that order,
-1 past ``min(topk, valid_blocks)``, and optionally the attention's sparse block
table and context lengths (``_emit_sparse_block_table_row``).

Row geometry (decode, ``query_len`` rows a request): row ``r = b * query_len +
t`` sees ``valid = (seq_len[b] - query_len + t + 128) // 128`` blocks; the first
``init_blocks`` are pinned at 1e30, the last ``local_blocks`` at 1e29 (local
second), then each block becomes the 64-bit key

    (1 << 48) | ordered(score) << 16 | (blk + 1)

with -0.0 as 0.0 and a NaN under -inf: every live key is unique and above the
padding key 0, so the selection and its order are a function of the scores.

The selection is a chunk-maximum narrowing (as aiter's ``topk_per_row_small_k``,
here one wave and no barrier):

1. the live part of the score row goes global -> LDS by 16 B async copies (a
   row of the plane is read in whole 128-column steps: the plane's rows are
   padded to them, :func:`plane_columns`);
2. lane l owns the chunk of blocks l, l + 32, ... (strided rather than its
   own copies' columns: an ascending row -- the newest blocks scoring highest
   -- then leaves ~16 survivors rather than ~60); its maximum key is the
   chunk's maximum;
3. ``cut`` = the ``topk``-th largest chunk maximum (a ballot bit search, MSB
   first). ``topk`` distinct keys (those maxima) are at least ``cut``, so every
   block of the row's top ``topk`` is at least ``cut``, and lies in one of the
   ``topk`` chunks whose maximum is at least ``cut``;
4. those chunks' blocks at or above ``cut`` are compacted into LDS (at most
   ``topk`` chunks x ``ceil(max_block / 32)`` blocks; on scores ~20 at
   ``topk`` 16);
5. a survivor's rank is the number of survivors with a larger key; a block of
   the true top ``topk`` has only top-``topk`` blocks above it, all survivors,
   so ranks below ``topk`` are exactly the answer, in key order;
6. lane s writes slot s, and for the emit gathers its block's page from the
   block table.

Rows with fewer than ``topk`` live lanes end with ``cut`` 0: every live block
survives. A row with no live block (a graph pad request) writes -1s and, with
the emit, an empty table row.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .kernels_common import create_llvm_ptr

WAVE = 32
BLOCK = 128  # tokens a block
PIN_INIT = 1e30
PIN_LOCAL = 1e29
KEY_LIVE = 1 << 48
KEY_BITS = 49  # bit 48 down to bit 0
COL_STEP = 4  # columns a lane takes a step (one 16 B copy)
STEP = WAVE * COL_STEP  # columns a step
# Async copies a wave keeps in flight at most (a row of 8192 blocks is 64
# steps; the wait keeps the ASYNCcnt counter far from its width).
ASYNC_CAP = 24


def plane_columns(max_block: int) -> int:
    """The width a score plane's rows need (``max_block`` rounded up to whole
    steps): the copy of a row reads whole steps."""
    return -(-max_block // STEP) * STEP


def survivor_cap(topk: int, columns: int) -> int:
    """Survivors a row can have: ``topk`` chunks of ``columns / 32`` blocks,
    rounded up to whole 32-wide rank tiles (the rank loop reads whole tiles)."""
    return -(-topk * (columns // WAVE) // WAVE) * WAVE


def lds_bytes(topk: int, columns: int) -> int:
    """LDS of a build: the score row (then its order), the survivors, the
    slots."""
    return 4 * columns + 8 * survivor_cap(topk, columns) + 4 * WAVE


@functools.cache
def compile_index_select_decode(
    *,
    heads: int,
    query_len: int,
    topk: int,
    columns: int,
    emit: bool,
    pages_per_block: int = 1,
    block_page_stride: int = 1,
):
    """Launcher for (index heads, query rows a request, ``topk`` <= 32,
    ``columns`` = :func:`plane_columns` of the score plane's width, the emit and
    its page geometry)."""
    if not 1 <= topk <= WAVE:
        raise ValueError(f"topk must be in [1, {WAVE}], got {topk}")
    if columns <= 0 or columns % STEP:
        raise ValueError(f"columns must be a positive multiple of {STEP}")
    K = topk
    PPB = pages_per_block
    BPS = block_page_stride
    NKV = heads
    surv_cap = survivor_cap(topk, columns)
    fill = K * PPB  # a table row: topk blocks of PPB pages
    name = f"index_select_decode_h{heads}_q{query_len}_k{K}_c{columns}" + (
        f"_emit{PPB}x{BPS}" if emit else ""
    )

    @fx.struct
    class SharedStorage:
        sc: fx.Array[fx.Int32, columns, 16]  # the score row (fp32 bits)
        surv: fx.Array[fx.Int64, surv_cap, 16]
        out: fx.Array[fx.Int32, WAVE, 16]

    @flyc.kernel(name=name, known_block_size=[WAVE, 1, 1])
    def index_select_kernel(
        arg_score: fx.Pointer,  # [heads, >= rows, plane_columns] f32
        arg_len: fx.Pointer,  # [batch] i32
        arg_bt: fx.Pointer,  # [batch, >= max_block] i32 (emit only)
        arg_idx: fx.Pointer,  # [heads, >= rows, topk] i32
        arg_sbt: fx.Pointer,  # [rows * heads, topk * PPB] i32 (emit only)
        arg_sctx: fx.Pointer,  # [rows * heads] i32 (emit only)
        i32_s_head: fx.Int32,
        i32_s_row: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_ti_head: fx.Int32,
        i32_ti_row: fx.Int32,
        i32_sbt_row: fx.Int32,
        i32_max_block: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
    ):
        lane = fx.Int32(gpu.thread_id("x"))
        row = fx.Int32(gpu.block_id("x"))
        h = fx.Int32(gpu.block_id("y"))
        zero = fx.Int32(0)
        b = row // query_len
        t = row - b * query_len

        length = fx.Int32(fx.recast_iter(fx.Int32, arg_len)[b])
        prefix = length - query_len
        causal_len = prefix + t + 1
        # (prefix + t + 128) // 128: never negative (seq_len >= 0, query_len < 128)
        valid = (prefix + t + BLOCK) >> 7
        valid = (valid < zero).select(zero, valid)
        n = (valid < i32_max_block).select(valid, i32_max_block)
        local_start = n - i32_local
        local_start = (local_start < zero).select(zero, local_start)

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        sc_base = fx.Int64(fx.ptrtoint(lds.sc.ptr))
        sc = fx.recast_iter(fx.Int32, lds.sc.ptr)
        surv = fx.recast_iter(fx.Int64, lds.surv.ptr)
        out = fx.recast_iter(fx.Int32, lds.out.ptr)

        # ---- 1. the live row into LDS, whole steps -------------------------
        steps = (n + (STEP - 1)) // STEP
        s_row = fx.Int64(fx.ptrtoint(arg_score)) + 4 * (
            fx.Int64(h) * fx.Int64(i32_s_head) + fx.Int64(row) * fx.Int64(i32_s_row)
        )
        for v in range(zero, steps):
            off = fx.Int64((fx.Int32(v) * WAVE + lane) * (4 * COL_STEP))
            # Raw ROCDL: per-lane 16 B async copies to each lane's own LDS
            # slot (fx.copy's tile atoms take whole tiles, not a row's prefix).
            rocdl.global_load_async_to_lds_b128(
                create_llvm_ptr(s_row + off, address_space=1),
                create_llvm_ptr(sc_base + off, address_space=3),
                0,
            )
            # Raw s_wait_asynccnt (here and below): the async-copy counter is
            # not modelled by fx; at most ASYNC_CAP copies in flight.
            rocdl.s_wait_asynccnt(ASYNC_CAP)
        rocdl.s_wait_asynccnt(0)

        def key64(col, ordered):
            live = col < n
            key = (
                fx.Int64(KEY_LIVE)
                | ((fx.Int64(ordered) & fx.Int64(0xFFFFFFFF)) << 16)
                | fx.Int64(col + 1)
            )
            return live.select(key, fx.Int64(0))

        def ordered_of(col):
            """The block's pinned score in the key's 32-bit order."""
            live = col < n
            s = fx.Int32(sc[col]).bitcast(fx.Float32)
            s = (live & (col < i32_init)).select(fx.Float32(PIN_INIT), s)
            s = (live & (col >= local_start)).select(fx.Float32(PIN_LOCAL), s)
            bits = s.bitcast(fx.Int32)
            bits = (bits == fx.Int32(-(1 << 31))).select(zero, bits)
            ordered = (bits < zero).select(
                bits ^ fx.Int32(-1), bits ^ fx.Int32(-(1 << 31))
            )
            is_nan = (bits & fx.Int32(0x7FFFFFFF)) > fx.Int32(0x7F800000)
            return is_nan.select(zero, ordered)

        # ---- 2. each lane's chunk maximum ----------------------------------
        lane_max = fx.Int64(0)
        for v in range(zero, steps):
            for u in range_constexpr(COL_STEP):
                col = (fx.Int32(v) * COL_STEP + u) * WAVE + lane
                ordered = ordered_of(col)
                # in place: pass 2 reads back the order, not the score (each
                # lane only its own columns)
                sc[col] = ordered
                k = key64(col, ordered)
                lane_max = (k > lane_max).select(k, lane_max)

        # ---- 3. cut: the topk-th largest chunk maximum ---------------------
        cut = fx.Int64(0)
        for bit in range_constexpr(KEY_BITS):
            probe = cut | fx.Int64(1 << (KEY_BITS - 1 - bit))
            hits = fx.Int32(rocdl.ballot(T.i32, fx.as_ir_value(lane_max >= probe)))
            cut = (fx.Int32(fx.ctpop(hits)) >= K).select(probe, cut)
        chosen = lane_max >= cut

        # ---- 4. the chosen chunks' blocks at or above the cut -> LDS -------
        count = zero
        for v in range(zero, steps):
            # the step's four orders first: one LDS round trip a step
            cols = [
                (fx.Int32(v) * COL_STEP + u) * WAVE + lane
                for u in range_constexpr(COL_STEP)
            ]
            orders = [fx.Int32(sc[c]) for c in cols]
            for u in range_constexpr(COL_STEP):
                col = cols[u]
                k = key64(col, orders[u])
                keep = chosen & (k >= cut) & (k != fx.Int64(0))
                mask = fx.Int32(rocdl.ballot(T.i32, fx.as_ir_value(keep)))
                below = fx.Int32(
                    rocdl.mbcnt_lo(T.i32, fx.as_ir_value(mask), fx.as_ir_value(zero))
                )
                if keep:
                    surv[count + below] = k
                count = count + fx.Int32(fx.ctpop(mask))
        # Raw s_wait_dscnt: lanes read what other lanes wrote (one wave, LDS in
        # order; the wait keeps the compiler from reordering across it).
        rocdl.s_wait_dscnt(0)

        # ---- 5. rank: survivors with a larger key; rank < topk is the slot --
        for i0 in range(zero, count, WAVE):
            mine_i = fx.Int32(i0) + lane
            mine = (mine_i < count).select(fx.Int64(surv[mine_i]), fx.Int64(0))
            place = zero
            for j0 in range(zero, count, WAVE):
                j_i = fx.Int32(j0) + lane
                other = (j_i < count).select(fx.Int64(surv[j_i]), fx.Int64(0))
                o_lo = fx.Int32(other)
                o_hi = fx.Int32(other >> 32)
                for u in range_constexpr(WAVE):
                    # Raw readlane: a wave-uniform broadcast of lane u's key.
                    lo = fx.Int32(
                        rocdl.readlane(
                            T.i32, fx.as_ir_value(o_lo), fx.as_ir_value(fx.Int32(u))
                        )
                    )
                    hi = fx.Int32(
                        rocdl.readlane(
                            T.i32, fx.as_ir_value(o_hi), fx.as_ir_value(fx.Int32(u))
                        )
                    )
                    peer = (fx.Int64(hi) << 32) | (fx.Int64(lo) & fx.Int64(0xFFFFFFFF))
                    place = place + (peer > mine).select(fx.Int32(1), zero)
            if (mine_i < count) & (place < K):
                out[place] = fx.Int32(mine & fx.Int64(0xFFFF)) - 1
        rocdl.s_wait_dscnt(0)

        # ---- 6. top-k ids (lane = slot), then the emit ---------------------
        n_sel = (n < K).select(n, fx.Int32(K))
        slot_live = lane < n_sel
        blk = slot_live.select(
            fx.Int32(out[(lane < K).select(lane, zero)]), fx.Int32(-1)
        )
        ti = fx.recast_iter(fx.Int32, arg_idx)
        if lane < K:
            ti[
                fx.Int64(h) * fx.Int64(i32_ti_head)
                + fx.Int64(row) * fx.Int64(i32_ti_row)
                + fx.Int64(lane)
            ] = blk

        if const_expr(emit):
            self_blk = ((causal_len > zero).select(causal_len - 1, zero)) >> 7
            ok = (blk >= zero) & (blk <= self_blk)
            if const_expr(PPB == 1):
                ok = ok & (causal_len > zero)
            is_tail = ok & (blk == self_blk)
            is_full = ok & (blk < self_blk)
            full_mask = fx.Int32(rocdl.ballot(T.i32, fx.as_ir_value(is_full)))
            n_full = fx.Int32(fx.ctpop(full_mask))
            n_valid = fx.Int32(
                fx.ctpop(fx.Int32(rocdl.ballot(T.i32, fx.as_ir_value(ok))))
            )
            has_tail = fx.Int32(rocdl.ballot(T.i32, fx.as_ir_value(is_tail))) != zero
            earlier = fx.Int32(
                rocdl.mbcnt_lo(T.i32, fx.as_ir_value(full_mask), fx.as_ir_value(zero))
            )
            slot = is_full.select(earlier, n_full)
            bt = fx.recast_iter(fx.Int32, arg_bt)
            page = zero
            if ok:
                page = fx.Int32(
                    bt[fx.Int64(b) * fx.Int64(i32_bt_stride) + fx.Int64(blk)]
                )
            base_phys = page * (BPS * NKV) + h
            erow = fx.Int64(row * NKV + h)
            sbt = fx.recast_iter(fx.Int32, arg_sbt)
            row_base = erow * fx.Int64(i32_sbt_row)
            if ok:
                for j in range_constexpr(PPB):
                    sbt[row_base + fx.Int64(slot * PPB + j)] = base_phys + j * NKV
            for w in range_constexpr(-(-fill // WAVE)):
                off = lane + w * WAVE
                if (off < fill) & (off >= n_valid * PPB):
                    sbt[row_base + fx.Int64(off)] = zero
            tail_tokens = causal_len - self_blk * BLOCK
            with_tail = n_full * BLOCK + tail_tokens
            no_tail = n_valid * BLOCK
            no_tail = (no_tail < causal_len).select(no_tail, causal_len)
            ctx = has_tail.select(with_tail, no_tail)
            if const_expr(PPB == 1):
                ctx = (ctx < zero).select(zero, ctx)
            if lane == zero:
                fx.recast_iter(fx.Int32, arg_sctx)[erow] = ctx

    @flyc.jit
    def launch(
        arg_score: fx.Pointer,
        arg_len: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_idx: fx.Pointer,
        arg_sbt: fx.Pointer,
        arg_sctx: fx.Pointer,
        i32_s_head: fx.Int32,
        i32_s_row: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_ti_head: fx.Int32,
        i32_ti_row: fx.Int32,
        i32_sbt_row: fx.Int32,
        i32_max_block: fx.Int32,
        i32_init: fx.Int32,
        i32_local: fx.Int32,
        i32_rows: fx.Int32,
        stream: fx.Stream,
    ):
        index_select_kernel(
            arg_score,
            arg_len,
            arg_bt,
            arg_idx,
            arg_sbt,
            arg_sctx,
            i32_s_head,
            i32_s_row,
            i32_bt_stride,
            i32_ti_head,
            i32_ti_row,
            i32_sbt_row,
            i32_max_block,
            i32_init,
            i32_local,
        ).launch(
            grid=(fx.Int64(i32_rows), heads, 1),
            block=(WAVE, 1, 1),
            stream=stream,
        )

    return {"launch": launch, "kernel": index_select_kernel}
