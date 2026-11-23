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
barrier) and every wave reads the A operands with 16-byte LDS loads. Each wave
owns ``block_q / (16 * waves)`` 16-row query tiles (4 for the wrapper's 256 rows
over 4 waves); a page is 8 key tiles x that many WMMAs a wave.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
from flydsl.expr import const_expr, gpu, range_constexpr
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import ReductionOp, T

from .minimax_m3_index_score import DATA_BYTES, LOG2E, NVFP4_SCALE_BYTES, PAGE
from .pa_decode.op_lds import Gfx1250Memory, _gptr
from .pa_decode.pipeline_nvfp4 import _wmma_scale16

WAVE = 32
ROW, SROW = DATA_BYTES, NVFP4_SCALE_BYTES  # payload / scale bytes a key
PAGE_BYTES = PAGE * ROW + PAGE * SROW  # 8 KB payload + 1 KB scales
E4M3_MAX = 448.0


def sm_scale_log2e(sm_scale: float) -> float:
    """``sm_scale * LOG2E`` multiplied in FP32, as ATOM's Triton kernel does."""
    return float(np.float32(sm_scale) * np.float32(LOG2E))


class _Lds(Gfx1250Memory):
    class _T:
        kv_dtype = fx.BFloat16

    def __init__(self, base):
        self.base = base
        self.t = self._T


_CACHE = {}


# Blocks a CTA scores: its scores are staged in LDS, and a wave stores one row's
# chunk of at most WAVE scores with one coalesced store (94 us against 128 us
# for a 4-byte store per block and row: 14336 new rows on 99968 cached, MI455X).
CHUNK_MAX = 32


def compile_index_score_prefill_nvfp4(*, block_q, waves, ring):
    """Compile (cached) the scorer: ``block_q`` query rows a CTA (``waves`` waves of
    ``block_q / waves`` rows, a multiple of 16), ``ring`` pages in flight."""
    tiles_per_wave = block_q // (16 * waves)
    if block_q % (16 * waves) or tiles_per_wave < 1:
        raise ValueError("block_q must be a multiple of 16 * waves")
    if not 2 <= ring <= 8:
        raise ValueError("ring must be in [2, 8]")
    key = (block_q, waves, ring)
    if key in _CACHE:
        return _CACHE[key]
    threads = waves * WAVE
    ring_bytes = ring * PAGE_BYTES
    total = ring_bytes + block_q * CHUNK_MAX * 4

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total // 4, 16]

    name = (
        f"index_score_prefill_nvfp4_d128_bq{block_q}_w{waves}_ring{ring}_lds{CHUNK_MAX}"
    )

    @flyc.kernel(name=name, known_block_size=(threads, 1, 1))
    def index_score_kernel(
        q_ptr: fx.Pointer,  # [total_q, heads, 128] bf16
        payload_ptr: fx.Pointer,  # [blocks, 128, 64] u8
        scale_ptr: fx.Pointer,  # [blocks, 128, 8] u8 (E4M3)
        score_ptr: fx.Pointer,  # [heads, total_q, max_block] f32
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
                                fx.minnumf(
                                    vals[h * 8 + e] * live_row, fx.Float32(E4M3_MAX)
                                ),
                                fx.Float32(-E4M3_MAX),
                            )
                            for e in range_constexpr(8)
                        ],
                        dtype=fx.Float32,
                    )
                    pair = mem.fp8(eight)
                    words.extend([pair[0], pair[1]])
            q_ops.append(fx.Vector.from_elements(words, dtype=fx.Int32))

        def page_max(masked, base, unit, blk):
            """Per query tile, this lane's max over its keys of the page
            (causally masked on the diagonal pages only)."""
            out = [fx.Float32(float("-inf")) for _ in range_constexpr(tiles_per_wave)]
            for kt in range_constexpr(PAGE // 16):
                key = kt * 16 + lane16
                at = base + key * ROW + half * 16
                frags = [lds_load(at + j * 32, fx.Int32, 4) for j in range_constexpr(2)]
                k_rows = fx.Vector.from_elements(
                    [
                        frags[j][i]
                        for j in range_constexpr(2)
                        for i in range_constexpr(4)
                    ],
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

        if live:
            for ahead in range_constexpr(ring - 1):
                if blk_lo + ahead < blk_hi:
                    stage(blk_lo + ahead, fx.Int32(ahead))
            score = fx.recast_iter(fx.Float32, score_ptr)
            q_start = prefix + pid_q * block_q
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

                maxes = [
                    fx.Float32(float("-inf")) for _ in range_constexpr(tiles_per_wave)
                ]
                if diagonal:
                    maxes = page_max(True, base, unit, blk)
                else:
                    maxes = page_max(False, base, unit, blk)
                for rt in range_constexpr(tiles_per_wave):
                    m = fx.maxnumf(maxes[rt], maxes[rt].shuffle_xor(16, WAVE))
                    r = row0 + rt * 16 + lane16
                    # Staged in LDS; a wave stores whole row chunks below.
                    if half == 0:
                        lds_store(
                            ring_bytes
                            + ((r - pid_q * block_q) * CHUNK_MAX + (blk - blk_lo)) * 4,
                            fx.Vector.from_elements([m * scale], dtype=fx.Float32),
                            fx.Float32,
                        )
                final = yield [state[0]]  # noqa: F841 -- the loop result is unused
            tdm_ops.tensor_wait(0)
            gpu.barrier()
            n_blk = blk_hi - blk_lo
            rows_per_wave = block_q // waves
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
                        ring_bytes + (rl * CHUNK_MAX + lane) * 4, fx.Float32, 1
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
