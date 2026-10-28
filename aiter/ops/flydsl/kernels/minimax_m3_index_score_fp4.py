# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 lightning-indexer decode block-score kernel, MXFP4 index cache.

The operator is the one `minimax_m3_index_score` documents,

    score[head, row, p] = max over t in [0,128) of dot(K[page, t, :], Q[row, head, :]) * scale

with a per-query causal cutoff; only the K cache differs. K is MXFP4: e2m1
elements, two to a byte (the LOW nibble is the even element), one e8m0 scale per
32 elements (scale byte 0 covers elements 0..31, byte 1 covers 32..63, ...). The
cache reaches the kernel as two views, each with its own page stride, so the
caller may pass two tensors or two views of one `[128 x 64 B data][128 x 4 B
scale]` (8704 B) page:

    k_data   uint8 [pages, 128, 64]   stride(1) == 64, stride(2) == 1
    k_scale  uint8 [pages, 128, 4]    stride(1) == 4,  stride(2) == 1

Tile and work map are the fp8 scorer's (one wave per page, register-local max
over the page's tokens). A lane's 16 B data load (elements 32g..32g+31 for lane
group g) is exactly its four k-steps and exactly one e8m0 group.

bf16 Q: each data dword is widened with `v_cvt_scalef32_pk_bf16_fp4` and fed to
the bf16 16x16x32 MFMA. K is lifted to Q's precision and Q is never rounded
down; e2m1 x 2^e is exact in bf16, so the scores are those of `dequantize(K)`.

e4m3 Q (`idx_q.dtype == dtypes.fp8`): no widening, one
`v_mfma_scale_f32_16x16x128_f8f6f4` per tile (A = fp4 K with its e8m0 byte,
B = Q). Products are exact in fp32.

e8m0 bytes 0 and 255 are unsupported: the hardware takes the scale as the float
whose exponent field is the byte, so 0 scales by 0.0 and 255 is NaN. A conforming
writer (`indexer_fp4_scale_e8m0`) never emits either. Garbage in a page's unwritten
tail is harmless as in the fp8 cache: every such token is masked before the max.

The builder is a COPY of `build_index_score` with the K side replaced, not a
parameterisation of it; a change to one's epilogue or work-map decode has to be
mirrored in the other.
"""

import math

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from aiter import dtypes
from aiter.ops.flydsl.kernels.minimax_m3_index_score import (
    HEAD_DIM,
    KSTEPS,
    LANE_K,
    LOG2E,
    MFMA_K,
    MFMA_M,
    MFMA_N,
    NEG_INF,
    PAGE,
    Q_LOADS,
    THREADS,
    TOK_TILES,
    WAVE,
    WAVES,
    _cp_blocks,
    _grid_chunks,
    _make_map,
    _validate_arch,
    _validate_bounds,
    _validate_map,
    _validate_rows,
    _validate_tensor,
    alloc_score,
    build_work_map,
    feature_tiles,
    geometry,
    work_map_capacity,
    work_map_size,
)
from aiter.ops.flydsl.kernels.tensor_shim import (
    _run_compiled,
    _to_raw,
    buf_copy_atom,
    ptr_arg,
    ptr_buf_tensor,
)

__all__ = [
    "alloc_score",
    "build_work_map",
    "index_score_supported_fp4",
    "score_flydsl_fp4",
    "work_map_capacity",
    "work_map_size",
]

DATA_BYTES = HEAD_DIM // 2  # packed e2m1 bytes per token
SCALE_BYTES = HEAD_DIM // 32  # e8m0 bytes per token
GROUP = 32  # elements one e8m0 scale covers
# Elements one lane holds per tile: KSTEPS * LANE_K == GROUP, which is what lets
# a lane's whole K fragment share a single scale.
assert KSTEPS * LANE_K == GROUP
DATA_UNITS = DATA_BYTES // 16  # 16 B accesses per token row

# K tiles kept in flight per wave, at most TOK_TILES (8). The fp8 scorer uses 3
# at 2 KiB a tile; a tile here is half the bytes, so it needs more of them to
# keep the same bytes in flight. Swept 2/3/4/6/8 on S4H1, S4H4 CP4 and S8H4:
# 6 is 15-35% faster than 3 where batch*context is small (the launch is latency
# bound there) and within 2% of the best from 3 to 8 once it is not.
TILE_DEPTH = 6
# Cache policy of the K data stream (2 = nt): every 128 B line is read once, so
# the L1 has nothing to keep. Not `geom["nt_k"]`, which is tuned for the fp8 layout.
K_CACHE = 2


def kernel_name(S: int, H: int, geom, q_fp8: bool = False) -> str:
    name = f"m3_index_score_S{S}_H{H}_fp4{'_qf8' if q_fp8 else ''}"
    name += f"_L{geom['pages_per_wave']}"
    name += f"_d{TILE_DEPTH}"
    if geom["waves_per_tok"] > 1:
        name += f"_tw{geom['waves_per_tok']}"
    if geom["spread"]:
        name += "_sp"
    if geom["cp_world"] > 1:
        # Rank in the name for the reason the fp8 scorer gives: ranks sharing a
        # node share the JIT cache and compile different kernels.
        name += f"_cp{geom['cp_world']}r{geom['cp_rank']}"
    return name


def e8m0_to_f32(packed, byte_pos):
    """Byte `byte_pos` of the i32 `packed` as the f32 2^(byte-127).

    The scale operand of `v_cvt_scalef32_*` is a float, and a float whose
    exponent field is the e8m0 byte and whose mantissa is zero is exactly that
    power of two. 0 and 255 are the two bytes this does not mean what e8m0
    means for; see the module docstring.
    """
    b = packed.shrui(byte_pos * fx.Int32(8)) & fx.Int32(0xFF)
    return (b << fx.Int32(23)).bitcast(fx.Float32)


def fp4_dword_to_bf16(word, scale_f32):
    """One dword of 8 packed e2m1, times `scale_f32`, as 8 bf16 in element order.

    The scale-and-widen is a single instruction per output pair; `sel` picks
    which byte of the dword is unpacked, low nibble first. The ONE place the
    nibble order is decided -- NVFP4 would change the scale argument here and
    in `e8m0_to_f32` and nothing else on the compute side.
    """
    w = _to_raw(fx.Int32(word))
    s = _to_raw(scale_f32)
    elems = []
    for sel in range_constexpr(4):
        pair = fx.Vector(fx.rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), w, s, sel))
        elems += [fx.BFloat16(pair[0]), fx.BFloat16(pair[1])]
    return elems


def build_index_score_fp4(S: int, H: int, geom, q_fp8: bool = False):
    """Compile a score kernel specialised on (S, H, geometry, Q dtype); see the
    fp8 scorer's `build_index_score` for what every knob in `geom` means.

    `q_fp8` swaps the K path for one scaled MFMA per tile with Q held as e4m3."""
    pages_per_wave = geom["pages_per_wave"]
    CP_WORLD = geom["cp_world"]
    CP_RANK = geom["cp_rank"]
    SPREAD = bool(geom["spread"])
    F = S * H
    FEAT_TILES_PER_WAVE = feature_tiles(S, H)
    WAVES_PER_TOK = geom["waves_per_tok"]
    PAGE_SLOTS = WAVES // WAVES_PER_TOK
    TOK_TILES_PER_WAVE = TOK_TILES // WAVES_PER_TOK
    CHUNK = PAGE_SLOTS * pages_per_wave
    RED_SLOTS = WAVES * FEAT_TILES_PER_WAVE * WAVE if WAVES_PER_TOK > 1 else 0
    SharedStorage = (
        fx.struct(
            type(
                "SharedStorage",
                (),
                {"__annotations__": {"red": fx.Array[fx.Float32, RED_SLOTS, 16]}},
            )
        )
        if RED_SLOTS
        else None
    )

    @flyc.kernel(
        name=kernel_name(S, H, geom, q_fp8),
        known_block_size=[THREADS, 1, 1],
    )
    def score_kernel(
        arg_q: fx.Pointer,
        arg_kd: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_score: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_work: fx.Pointer,
        i32_batch: fx.Int32,
        i32_stride_q_n: fx.Int32,
        i32_stride_q_h: fx.Int32,
        i32_stride_kd_blk: fx.Int32,
        i32_stride_ks_blk: fx.Int32,
        i32_stride_s_h: fx.Int32,
        i32_stride_s_b: fx.Int32,
        i32_stride_s_k: fx.Int32,
        i32_stride_bt_b: fx.Int32,
        f32_scale: fx.Float32,
    ):
        """One CTA scores CHUNK pages of request b.

        arg_q      [batch*S, H, 128] bf16 or e4m3 query
        arg_kd     [pages, 128, 64] uint8 packed e2m1, page stride in bytes
        arg_ks     [pages, 128, 4] uint8 e8m0, page stride in bytes
        arg_score  [H, batch*S, max_block] fp32, one per (feature, page)
        arg_bt     [batch, >= max_block] int32 block table
        arg_work   [rows, 2] int32 work map; see make_work_map
        f32_scale  score multiplier, already folded with LOG2E by the host
        """
        # ==================== PROLOGUE ====================
        n = fx.Int32(gpu.block_id("y")) * i32_batch + fx.Int32(gpu.block_id("x"))
        tid = fx.Int32(gpu.thread_id("x"))
        wave = tid // fx.Int32(WAVE)
        lane = tid % fx.Int32(WAVE)
        g = lane // 16
        u = lane % 16
        if const_expr(WAVES_PER_TOK == 1):
            tw = fx.Int32(0)
            pw = wave
        else:
            tw = wave % fx.Int32(WAVES_PER_TOK)
            pw = wave // fx.Int32(WAVES_PER_TOK)
        tw_tok = tw * fx.Int32(TOK_TILES_PER_WAVE * MFMA_M)
        q_buf = ptr_buf_tensor(arg_q, fx.Int32)
        s_buf = ptr_buf_tensor(arg_score, fx.Float32)
        bt_buf = ptr_buf_tensor(arg_bt, fx.Int32)
        wm_buf = ptr_buf_tensor(arg_work, fx.Int32)
        wm = fx.add_offset(fx.get_iter(wm_buf), n * 2)
        packed = wm.load(T.i32)
        seq_len = fx.add_offset(wm, fx.Int32(1)).load(T.i32)
        b = (packed >> 16) & 0xFFFF
        c = packed & 0xFFFF
        if const_expr(SPREAD):
            span = (seq_len >> fx.Int32(24)) & fx.Int32(0xFF)
            seq_len = seq_len & fx.Int32(0xFFFFFF)
        num_pages = (seq_len + fx.Int32(PAGE - 1)) // fx.Int32(PAGE)
        if const_expr(CP_WORLD > 1):
            num_pages = _cp_blocks(num_pages, CP_WORLD, CP_RANK, fx.Int32(0))

        def global_block(p):
            """Global block id of this rank's logical block `p`."""
            if const_expr(CP_WORLD == 1 and CP_RANK == 0):
                return p
            return p * fx.Int32(CP_WORLD) + fx.Int32(CP_RANK)

        def feature_of(i):
            return fx.Int32(MFMA_N * i) + u

        def tok_head_of(i):
            f = feature_of(i)
            tok = f // fx.Int32(H)
            return tok, f - tok * fx.Int32(H)

        # k map: lane group g holds k = 32g .. 32g+31, k-step ks the 8 of them
        # from 32g + 8ks. K reads the same range as one contiguous 16 B access,
        # Q as four 16 B accesses; both sides MUST use this one map.
        def q_row_head(i):
            """Q row and head of feature tile i's column; padding columns read row 0."""
            tok, head = tok_head_of(i)
            row = b * fx.Int32(S) + tok
            in_range = feature_of(i) < fx.Int32(F)
            return in_range.select(row, fx.Int32(0)), in_range.select(head, fx.Int32(0))

        def load_q_frag(i, j):
            """One lane's 16 B of Q for (feature tile i, Q access j), from gmem."""
            row, head = q_row_head(i)
            off = (
                row * i32_stride_q_n
                + head * i32_stride_q_h
                + g * fx.Int32(GROUP)
                + fx.Int32(j * LANE_K)
            )
            return fx.add_offset(fx.get_iter(q_buf), off >> fx.Int32(1)).load(
                T.vec(4, T.i32)
            )

        def as_bf16_frag(raw16):
            t = fx.make_rmem_tensor(fx.make_layout(LANE_K, 1), fx.BFloat16)
            t.store(raw16.bitcast(fx.BFloat16))
            return t

        def load_q8(i):
            """One lane's e4m3 B operand of the scaled MFMA for feature tile i: 32
            bytes, Q[16g .. 16g+15] then Q[64+16g .. 64+16g+15] of its column.
            That is how the hardware lays an fp8 operand out against an fp4 A (the
            tests pin it), NOT the contiguous 32g.. run the fp4 K fragment holds."""
            row, head = q_row_head(i)
            base = row * i32_stride_q_n + head * i32_stride_q_h + g * fx.Int32(16)
            halves = [
                fx.add_offset(
                    fx.get_iter(q_buf), (base + fx.Int32(64 * h)) >> fx.Int32(2)
                ).load(T.vec(4, T.i32))
                for h in range_constexpr(2)
            ]
            return fx.Vector.from_elements(
                [halves[h][k] for h in range_constexpr(2) for k in range_constexpr(4)],
                dtype=fx.Int32,
            )

        lds = (
            fx.SharedAllocator().allocate(SharedStorage).peek()
            if const_expr(SharedStorage is not None)
            else None
        )
        if const_expr(q_fp8):
            q_b = [load_q8(i) for i in range_constexpr(FEAT_TILES_PER_WAVE)]
        else:
            q_raw = [
                [load_q_frag(i, j) for j in range_constexpr(Q_LOADS)]
                for i in range_constexpr(FEAT_TILES_PER_WAVE)
            ]
            q_frag = [
                [as_bf16_frag(q_raw[i][ks]) for ks in range_constexpr(KSTEPS)]
                for i in range_constexpr(FEAT_TILES_PER_WAVE)
            ]
            mma_atom = fx.make_mma_atom(
                fx.rocdl.MFMA(MFMA_M, MFMA_N, MFMA_K, fx.BFloat16)
            )
        zero4 = fx.full(4, 0.0, fx.Float32)
        neg_inf = fx.Float32(NEG_INF)
        if const_expr(SPREAD):
            first = c
            end = c + span
            limit = (end < num_pages).select(end, num_pages)
        else:
            first = c * fx.Int32(CHUNK)
            limit = num_pages
        pages = [
            first + fx.Int32(j * PAGE_SLOTS) + pw
            for j in range_constexpr(pages_per_wave)
        ]
        zero = fx.Int32(0)
        nm1 = limit - fx.Int32(1)
        last = (nm1 < zero).select(zero, nm1)
        # Widen BEFORE multiplication: a pool can exceed the descriptor's 4 GiB.
        kd_page_bytes = fx.Int64(i32_stride_kd_blk)
        ks_page_bytes = fx.Int64(i32_stride_ks_blk)
        kd_base = fx.Int64(fx.ptrtoint(arg_kd))
        ks_base = fx.Int64(fx.ptrtoint(arg_ks))
        page_data = []
        page_scale = []
        for pp in pages:
            p_clamped = (pp < last).select(pp, last)
            column = (num_pages > zero).select(global_block(p_clamped), zero)
            page = fx.add_offset(
                fx.get_iter(bt_buf), b * i32_stride_bt_b + column
            ).load(T.i32)
            # Each descriptor spans exactly the bytes its view owns in a page,
            # so a bad block id reads inside the pool rather than the neighbour
            # view -- the page STRIDE may be larger (single-region pages).
            page_data.append(
                ptr_buf_tensor(
                    fx.inttoptr(arg_kd.type, kd_base + fx.Int64(page) * kd_page_bytes),
                    fx.Int32,
                    unit_elems=4,
                    num_records_bytes=fx.Int32(PAGE * DATA_BYTES),
                )
            )
            page_scale.append(
                ptr_buf_tensor(
                    fx.inttoptr(arg_ks.type, ks_base + fx.Int64(page) * ks_page_bytes),
                    fx.Int32,
                    num_records_bytes=fx.Int32(PAGE * SCALE_BYTES),
                )
            )
        # K data is streamed; the scale lines are shared by two adjacent tiles,
        # so they keep the default policy.
        k_atom = buf_copy_atom(16, fx.Int32, cache_modifier=K_CACHE)

        def load_k16(page_buf, unit):
            src = fx.slice(page_buf, (unit, None))
            frag = fx.make_fragment_like(src)
            fx.copy(k_atom, src, frag)
            return fx.Vector(fx.memref_load_vec(frag))

        def issue_k(page_d, page_s, tok_tile):
            """Start the loads of K[page, 16*tok_tile + u, :]: this lane's 16 B
            of data and the token's 4 scale bytes (a dword, shared by the four
            lane groups of the token)."""
            tok_row = fx.Int32(MFMA_M * tok_tile) + tw_tok + u
            data = load_k16(page_d, tok_row * fx.Int32(DATA_UNITS) + g)
            scale = fx.add_offset(fx.get_iter(page_s), tok_row).load(T.i32)
            return data, scale

        def convert_k(raw, scale_f32, ks):
            """The bf16 A-fragment of k-step `ks`: dword ks of the lane's 16 B."""
            t = fx.make_rmem_tensor(fx.make_layout(LANE_K, 1), fx.BFloat16)
            t.store(
                fx.Vector.from_elements(
                    fp4_dword_to_bf16(raw[ks], scale_f32), fx.BFloat16
                )
            )
            return t

        # ==================== MAIN LOOP ====================
        def score_tok_tile(p, tok_tile, loaded, run_max):
            data, scale_word = loaded
            if const_expr(q_fp8):
                # A = K as fp4 (4 dwords; the other 4 are unused by cbsz=4), scaled by
                # this lane group's e8m0 byte (scaleA, byte 0 of the shifted word);
                # B = Q as e4m3 with unit scale (0x7F = 2^0 in all four bytes).
                a_k = fx.Vector.from_elements(
                    [data[k] for k in range_constexpr(4)]
                    + [fx.Int32(0) for _ in range_constexpr(4)],
                    dtype=fx.Int32,
                )
                scale_a = fx.Int32(scale_word).shrui(g * fx.Int32(8))
                accs = [
                    fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.f32x4,
                            [
                                a_k,
                                q_b[i],
                                zero4,
                                4,
                                0,
                                0,
                                scale_a,
                                0,
                                fx.Int32(0x7F7F7F7F),
                            ],
                        )
                    )
                    for i in range_constexpr(FEAT_TILES_PER_WAVE)
                ]
            else:
                scale_f32 = e8m0_to_f32(fx.Int32(scale_word), g)
                accs = []
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    acc = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32)
                    acc.store(zero4)
                    accs.append(acc)
                for ks in range_constexpr(KSTEPS):
                    kf = convert_k(data, scale_f32, ks)
                    for i in range_constexpr(FEAT_TILES_PER_WAVE):
                        fx.gemm(mma_atom, accs[i], kf, q_frag[i][ks], accs[i])
            tok_base = (
                global_block(p) * fx.Int32(PAGE)
                + fx.Int32(MFMA_M * tok_tile)
                + tw_tok
                + g * 4
            )
            for i in range_constexpr(FEAT_TILES_PER_WAVE):
                tok, _ = tok_head_of(i)
                causal_len = seq_len - fx.Int32(S) + tok + fx.Int32(1)
                v = (
                    accs[i]
                    if const_expr(q_fp8)
                    else fx.Vector(fx.memref_load_vec(accs[i]))
                )
                for r in range_constexpr(4):
                    ok = (tok_base + fx.Int32(r)) < causal_len
                    run_max[i] = run_max[i].maximumf(
                        ok.select(fx.Float32(v[r]), neg_inf)
                    )

        _live = True if const_expr(WAVES_PER_TOK > 1) else (pages[0] < limit)
        if _live:
            carry = None
            for j in range_constexpr(pages_per_wave):
                p = pages[j]
                nxt = j + 1 if j + 1 < pages_per_wave else None
                run_max = [neg_inf for _ in range_constexpr(FEAT_TILES_PER_WAVE)]
                depth = min(TILE_DEPTH, TOK_TILES_PER_WAVE)
                queue = list(carry) if carry is not None else []
                for tok_tile in range_constexpr(len(queue), depth):
                    queue.append(issue_k(page_data[j], page_scale[j], tok_tile))
                carry = []
                for tok_tile in range_constexpr(TOK_TILES_PER_WAVE):
                    loaded = queue.pop(0)
                    future = tok_tile + depth
                    fx.rocdl.sched_barrier(0)
                    score_tok_tile(p, tok_tile, loaded, run_max)
                    fx.rocdl.sched_barrier(0)
                    if const_expr(future < TOK_TILES_PER_WAVE):
                        queue.append(issue_k(page_data[j], page_scale[j], future))
                    elif nxt is not None:
                        carry.append(
                            issue_k(
                                page_data[nxt],
                                page_scale[nxt],
                                future - TOK_TILES_PER_WAVE,
                            )
                        )
                # ---------- EPILOGUE: fold, scale, store one value/feature ----
                folded = []
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    m = run_max[i]
                    for sh in (16, 32):
                        m = m.maximumf(m.shuffle_xor(sh, WAVE))
                    folded.append(m)
                if const_expr(WAVES_PER_TOK > 1):
                    red = lds.red.ptr
                    if const_expr(pages_per_wave > 1):
                        gpu.barrier()
                    base = wave * fx.Int32(FEAT_TILES_PER_WAVE * WAVE) + lane
                    for i in range_constexpr(FEAT_TILES_PER_WAVE):
                        fx.ptr_store(folded[i], red + (base + fx.Int32(i * WAVE)))
                    gpu.barrier()
                    reduced = []
                    for i in range_constexpr(FEAT_TILES_PER_WAVE):
                        m = folded[i]
                        for t in range_constexpr(WAVES_PER_TOK):
                            w_other = fx.Int32(t) + pw * fx.Int32(WAVES_PER_TOK)
                            m = m.maximumf(
                                fx.ptr_load(
                                    red
                                    + (
                                        w_other * fx.Int32(FEAT_TILES_PER_WAVE * WAVE)
                                        + fx.Int32(i * WAVE)
                                    )
                                    + lane
                                )
                            )
                        reduced.append(m)
                    folded = reduced
                valid = (
                    None
                    if const_expr(pages_per_wave == 1 and WAVES_PER_TOK == 1)
                    else p < limit
                )
                for i in range_constexpr(FEAT_TILES_PER_WAVE):
                    m = folded[i] * f32_scale
                    tok, head = tok_head_of(i)
                    row = b * fx.Int32(S) + tok
                    addr = (
                        head * i32_stride_s_h
                        + row * i32_stride_s_b
                        + p * i32_stride_s_k
                    )
                    is_writer = (g == 0) & (feature_of(i) < fx.Int32(F))
                    if const_expr(WAVES_PER_TOK > 1):
                        is_writer = is_writer & (tw == 0)
                    if valid is not None:
                        is_writer = is_writer & valid
                    if is_writer:
                        fx.add_offset(fx.get_iter(s_buf), addr).store(m)

    @flyc.jit
    def launch(
        arg_q: fx.Pointer,
        arg_kd: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_score: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_work: fx.Pointer,
        i32_stride_q_n: fx.Int32,
        i32_stride_q_h: fx.Int32,
        i32_stride_kd_blk: fx.Int32,
        i32_stride_ks_blk: fx.Int32,
        i32_stride_s_h: fx.Int32,
        i32_stride_s_b: fx.Int32,
        i32_stride_s_k: fx.Int32,
        i32_stride_bt_b: fx.Int32,
        f32_scale: fx.Float32,
        i32_batch: fx.Int32,
        i32_chunks: fx.Int32,
        stream: fx.Stream,
    ):
        score_kernel(
            arg_q,
            arg_kd,
            arg_ks,
            arg_score,
            arg_bt,
            arg_work,
            i32_batch,
            i32_stride_q_n,
            i32_stride_q_h,
            i32_stride_kd_blk,
            i32_stride_ks_blk,
            i32_stride_s_h,
            i32_stride_s_b,
            i32_stride_s_k,
            i32_stride_bt_b,
            f32_scale,
        ).launch(
            grid=(fx.Int64(i32_batch), fx.Int64(i32_chunks), 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return launch


_CACHE = {}


def _get(S, H, geom, device, q_fp8=False):
    key = (S, H, tuple(sorted(geom.items())), TILE_DEPTH, q_fp8, device)
    if key not in _CACHE:
        _CACHE[key] = build_index_score_fp4(S, H, geom, q_fp8)
    return _CACHE[key]


def _validate_cache(k_data, k_scale, device):
    """Shapes, dtypes and strides of the two views. Contents (block ids inside
    the pool, e8m0 not 0/255) are a caller contract."""
    _validate_tensor(k_data, "k_data", (torch.uint8,), 3, device, 16, False)
    _validate_tensor(k_scale, "k_scale", (torch.uint8,), 3, device, 4, False)
    if k_data.shape[1:] != (PAGE, DATA_BYTES) or k_data.shape[0] > 0x7FFFFFFF:
        raise ValueError("k_data: expected [pages, 128, 64] uint8 with int32 page IDs")
    if k_scale.shape != (k_data.shape[0], PAGE, SCALE_BYTES):
        raise ValueError(
            "k_scale: expected [pages, 128, 4] uint8, same pages as k_data"
        )
    if k_data.stride(1) != DATA_BYTES or k_data.stride(2) != 1:
        raise ValueError(
            f"k_data: tokens must be packed (stride(1) {DATA_BYTES}, stride(2) 1), "
            f"got {k_data.stride()}"
        )
    if k_scale.stride(1) != SCALE_BYTES or k_scale.stride(2) != 1:
        raise ValueError(
            f"k_scale: tokens must be packed (stride(1) {SCALE_BYTES}, stride(2) 1), "
            f"got {k_scale.stride()}"
        )
    # A page stride below the view's own extent makes successive pages overlap.
    if k_data.stride(0) < PAGE * DATA_BYTES or k_data.stride(0) % 16:
        raise ValueError(
            f"k_data: page stride must be a multiple of 16 B and >= {PAGE * DATA_BYTES}"
        )
    if k_scale.stride(0) < PAGE * SCALE_BYTES or k_scale.stride(0) % 4:
        raise ValueError(
            f"k_scale: page stride must be a multiple of 4 B and >= {PAGE * SCALE_BYTES}"
        )


def _validate_metadata(
    idx_q,
    k_data,
    k_scale,
    S,
    H,
    max_block,
    block_table=None,
    seq_lens=None,
    out=None,
    work_map=None,
    **cp,
):
    """Metadata only: no device reads, synchronisation or allocations. The
    fp8 scorer's `_validate_metadata` with the cache check replaced."""
    if not isinstance(S, int) or not isinstance(H, int) or S < 1 or H < 1:
        raise ValueError("S and H must be positive integers")
    device = idx_q.device
    _validate_arch(device)
    _validate_tensor(idx_q, "idx_q", (torch.bfloat16, dtypes.fp8), 3, device, 16)
    _validate_cache(k_data, k_scale, device)
    if idx_q.shape[1:] != (H, HEAD_DIM) or idx_q.shape[0] % S:
        raise ValueError("idx_q: expected [batch*S, H, 128]")
    _validate_rows(idx_q, "idx_q")
    batch = idx_q.shape[0] // S
    geom = geometry(batch, max_block, S, H, device=device, **cp)
    rows = _validate_bounds(batch, max_block, geom)
    if block_table is not None:
        _validate_tensor(block_table, "block_table", (torch.int32,), 2, device)
        if (
            block_table.shape[0] != batch
            or block_table.shape[1] < max_block
            or block_table.stride(1) != 1
            or block_table.stride(0) < block_table.shape[1]
        ):
            raise ValueError("block_table: invalid shape or layout")
    if seq_lens is not None:
        _validate_tensor(seq_lens, "seq_lens", (torch.int32,), 1, device)
        if seq_lens.shape != (batch,) or seq_lens.stride() != (1,):
            raise ValueError("seq_lens: expected contiguous [batch]")
    if out is not None:
        _validate_tensor(out, "out", (torch.float32,), 3, device)
        if out.shape != (H, batch * S, max_block):
            raise ValueError("out: expected [H, batch*S, max_block]")
        st = out.stride()
        contiguous = st[2] == 1 and st[1] >= max_block and st[0] >= batch * S * st[1]
        feature = st[0] == 1 and st[1] >= H and st[2] >= batch * S * st[1]
        if not (contiguous or feature):
            raise ValueError("out: expected page- or feature-contiguous layout")
    elif H * batch * S * max_block * 4 > 0xFFFFFFFF:
        raise ValueError("out: address span exceeds 32-bit addressing")
    if work_map is not None:
        _validate_map(work_map, rows, device, exact=True)
    return batch, geom


def index_score_supported_fp4(
    idx_q,
    k_data,
    k_scale,
    S: int,
    H: int,
    max_block: int,
    block_table,
    *,
    seq_lens,
    sm_scale=1.0,
    q_scale=1.0,
    out=None,
    work_map=None,
    **cp,
) -> bool:
    """Metadata-only predicate sharing the entry point's validation."""
    if block_table is None or seq_lens is None:
        return False
    try:
        _validate_metadata(
            idx_q,
            k_data,
            k_scale,
            S,
            H,
            max_block,
            block_table,
            seq_lens,
            out,
            work_map,
            **cp,
        )
        scaled = sm_scale * q_scale * LOG2E
        return math.isfinite(scaled) and 2**-149 <= scaled <= (2 - 2**-23) * 2**127
    except (AttributeError, IndexError, TypeError, ValueError, ZeroDivisionError):
        return False


def score_flydsl_fp4(
    idx_q,
    k_data,
    k_scale,
    block_table,
    seq_lens,
    S,
    H,
    sm_scale,
    max_block,
    out=None,
    work_map=None,
    cp_world: int = 1,
    cp_rank: int = 0,
    q_scale: float = 1.0,
):
    """Host entry point: `score_flydsl` for an MXFP4 index cache.

    `k_data`/`k_scale` are the two views described in the module docstring;
    everything else, including the CP shard, the work map and the score layout,
    is as in `score_flydsl`.

    `idx_q` is bf16 or e4m3 (`dtypes.fp8`), dispatched on its dtype. An e4m3 Q is
    used as stored times `q_scale` (default 1.0: the plain cast the writer
    produces, i.e. an unscaled q8kv4 query); `q_scale` must be positive and is
    folded into the score multiplier, like `sm_scale`.
    """
    batch, geom = _validate_metadata(
        idx_q,
        k_data,
        k_scale,
        S,
        H,
        max_block,
        block_table,
        seq_lens,
        out,
        work_map,
        cp_world=cp_world,
        cp_rank=cp_rank,
    )
    if block_table is None or seq_lens is None:
        raise ValueError("block_table and seq_lens are required")
    scaled = sm_scale * q_scale * LOG2E
    if not math.isfinite(scaled) or not 2**-149 <= scaled <= (2 - 2**-23) * 2**127:
        raise ValueError(
            "sm_scale * q_scale * LOG2E must be finite and positive in FP32"
        )
    q_fp8 = idx_q.dtype == dtypes.fp8
    if out is None:
        out = alloc_score(batch, S, H, max_block, idx_q.device)
    launch = _get(S, H, geom, idx_q.device.index, q_fp8)
    chunks = _grid_chunks(max_block, geom)
    if work_map is None:
        work_map = _make_map(seq_lens, max_block, geom, batch * chunks)
    with torch.cuda.device(idx_q.device):
        _run_compiled(
            launch,
            ptr_arg(idx_q, fx.Float8E4M3FN if q_fp8 else fx.BFloat16),
            ptr_arg(k_data, fx.Uint8),
            ptr_arg(k_scale, fx.Uint8),
            ptr_arg(out, fx.Float32),
            ptr_arg(block_table, fx.Int32),
            ptr_arg(work_map, fx.Int32),
            idx_q.stride(0),
            idx_q.stride(1),
            k_data.stride(0),
            k_scale.stride(0),
            out.stride(0),
            out.stride(1),
            out.stride(2),
            block_table.stride(0),
            float(scaled),
            batch,
            chunks,
            torch.cuda.current_stream(idx_q.device).cuda_stream,
        )
    return out
