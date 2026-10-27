# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""MXFP4 paged-attention decode tile kernel for gfx950 (query_length 1).

K and V are MXFP4: e2m1 elements, two to a byte (the LOW nibble is the even
element), one e8m0 scale per 32 elements. A 128-wide token row is 64 data bytes
plus 4 scale bytes, and each of K and V reaches the kernel as TWO uint8 views:

    data   [pages, kv_heads, block_size, 64]   token-major, stride(2) == 64
    scale  [pages, kv_heads, block_size, 4]    token-major, stride(2) == 4

Each view carries its own page and head stride (in bytes), so a caller may hand
over two tensors or two views of one single-region page
(``[block_size x 64 B data][block_size x 4 B scale]``). Scale byte j covers
elements 32j..32j+31. The kernel cannot tell the difference.

One CTA is one planned task of ``PADecodePlan`` (256-token tiles, the same
``work_info`` and the same pmax/psum/pout partials as ``pa_decode_kernel``, so
``launch_pa_decode_ps_reduce`` merges them). Inside a task each of the four waves
walks its own 16-token tiles (tile ``w``, ``w + 4``, ...) with its own online
softmax and the waves merge once in LDS; nothing in the tile loop is a barrier.
Per tile and wave (lane ``l``: ``g = l // 16``, ``s = l % 16``):

* QK, FP8 Q: one ``v_mfma_scale_f32_16x16x128_f8f6f4`` with A = K (fp4) and
  B = Q (fp8), cbsz=4, blgp=0. Lane (g, s) holds token ``s``'s elements
  ``32g..32g+31`` (one 16 B load = one e8m0 group) and that group's scale byte in
  scaleA. The fp8 B operand is NOT contiguous per lane: lane (g, s) holds Q head
  ``s``'s elements ``16g..16g+15`` in bytes 0..15 and ``64+16g..64+16g+15`` in
  bytes 16..31 (measured on gfx950, see op_tests/test_flydsl_pa_decode_mxfp4.py).
  scaleB is 1.0, so Q is used as stored; ``query_scale`` multiplies the logits.
* QK, bf16 Q (reference path): the same K tile widened with
  ``v_cvt_scalef32_pk_bf16_fp4`` (the scale rides in the instruction) and four
  ``mfma_f32_16x16x32_bf16``. Q is never rounded.
* The QK accumulator holds ``S^T[token 4g+v][head s]``, which is exactly the B
  operand layout of ``mfma_f32_16x16x16bf16_1k``, so P (bf16) never leaves
  registers.
* V: its scale runs along head_dim while PV contracts over tokens, so it cannot
  use the scaled MFMA. Lane (g, s) widens token ``s``'s 32 elements to bf16
  (exact: e2m1 x 2^e) into a wave-private LDS tile ``[16 chunks of 8 dims][16
  tokens][16 B]`` and reads it back transposed with ``ds_read_b64_tr_b16`` as the
  PV A operand.

e8m0 bytes 0 and 255 are not supported: ``v_cvt_scalef32_*`` takes the scale as a
float whose exponent field is the byte, so 0 scales by 0.0 and 255 is NaN. A
conforming writer never emits either. Rows of the unwritten tail of a page may
hold anything: invalid tokens are masked out of the scores and their V row is
zeroed after widening, so neither can poison the PV accumulation.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.compiler.protocol import dsl_size_of
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .tensor_shim import _to_raw
from .utils import rcp_f32

HEAD_DIM = 128
WAVE = 64
NUM_WAVES = 4
BLOCK_THREADS = WAVE * NUM_WAVES
TILE_TOKENS = 16
PLAN_TILE = 256  # tokens per PADecodePlan tile
LOG2E = 1.4426950408889634
DATA_BYTES = HEAD_DIM // 2  # packed e2m1 bytes per token
SCALE_BYTES = HEAD_DIM // 32  # e8m0 bytes per token
# [16 chunks of 8 dims][16 tokens][16 B] bf16, one tile per wave.
V_TILE_BYTES = (HEAD_DIM // 8) * TILE_TOKENS * 16
PAGE_STAGE_ALIGN = BLOCK_THREADS
MAX_STAGED_PAGES = 2048
# K/V tiles kept in flight per wave (a tile is 2 x 1 KiB of data). Measured on
# gfx950 with independent blocks (2048 tokens per row, 16 q heads): depth 2 is
# 0.2-2.2 us faster than 4 for 1-512 rows and equal at 4096, depth 3 is slower
# than both (why is not measured; depth 4 uses 168 VGPRs).
DEFAULT_TILE_DEPTH = 2


def _round_up(x: int, align: int) -> int:
    return (x + align - 1) // align * align


def _e8m0_to_f32(byte_i32):
    """The f32 2^(byte-127): a float whose exponent field is the e8m0 byte."""
    return (fx.Int32(byte_i32) << fx.Int32(23)).bitcast(fx.Float32)


def _fp4_dword_to_bf16(word, scale_f32):
    """One dword of 8 packed e2m1, times ``scale_f32``, as 8 bf16 in element order.

    Copy of ``minimax_m3_index_score_fp4.fp4_dword_to_bf16`` (kept separate so the
    two kernels can land independently): the scale-and-widen is one instruction
    per output pair; ``sel`` picks the byte of the dword, low nibble first.
    """
    w = _to_raw(fx.Int32(word))
    s = _to_raw(scale_f32)
    elems = []
    for sel in range_constexpr(4):
        pair = fx.Vector(rocdl.cvt_scalef32_pk_bf16_fp4(T.vec(2, T.bf16), w, s, sel))
        elems += [fx.BFloat16(pair[0]), fx.BFloat16(pair[1])]
    return elems


@functools.cache
def build_pa_decode_mxfp4(
    *,
    group_size: int,
    block_size: int,
    stage_pages: int,
    q_fp8: bool,
    has_q_scale: bool,
    out_f16: bool,
    tile_depth: int = DEFAULT_TILE_DEPTH,
):
    """Compile-time specialization of the task kernel and its launcher.

    ``stage_pages``: block-table entries staged in LDS per task (a multiple of
    256, at most ``MAX_STAGED_PAGES``); the caller bounds the pages one task can
    own by it.
    """
    assert 1 <= group_size <= 16, group_size
    assert block_size in (16, 64, 128), block_size
    assert stage_pages % PAGE_STAGE_ALIGN == 0 and 0 < stage_pages <= MAX_STAGED_PAGES
    assert tile_depth >= 1
    G = group_size
    BS = block_size
    DEPTH = tile_depth
    out_dtype = fx.Float16 if out_f16 else fx.BFloat16

    page_off = 0
    v_off = _round_up(stage_pages * 4, 16)
    # The raw [wave][head][dim] f32 partial O has its own region: a wave that
    # finishes early must not overwrite a V tile another wave is still reading.
    o_off = v_off + NUM_WAVES * V_TILE_BYTES
    m_off = o_off + NUM_WAVES * G * HEAD_DIM * 4
    l_off = m_off + NUM_WAVES * 16 * 4
    total_bytes = _round_up(l_off + NUM_WAVES * 16 * 4, 16)

    @fx.struct
    class SharedStorage:
        buf: fx.Array[fx.Int32, total_bytes // 4, 16]

    def _gptr(base, elem_ty, align):
        ty = fx.PointerType.get(elem_ty.ir_type, fx.AddressSpace.Global, align)
        return fx.recast_iter(ty, base)

    name = f"pa_decode_mxfp4_g{G}_bs{BS}_{'qfp8' if q_fp8 else 'qbf16'}_d{DEPTH}"

    @flyc.kernel(name=name, known_block_size=(BLOCK_THREADS, 1, 1))
    def pa_decode_mxfp4_kernel(
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        kd_ptr: fx.Pointer,
        ks_ptr: fx.Pointer,
        vd_ptr: fx.Pointer,
        vs_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        q_scale_ptr: fx.Pointer,
        work_info_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        kd_page_dw: fx.Int32,  # strides: data in dwords, scale in bytes
        kd_head_dw: fx.Int32,
        ks_page: fx.Int32,
        ks_head: fx.Int32,
        vd_page_dw: fx.Int32,
        vd_head_dw: fx.Int32,
        vs_page: fx.Int32,
        vs_head: fx.Int32,
        softmax_scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        warp = tid // WAVE
        lane = tid % WAVE
        g = lane // 16
        s = lane % 16
        slot = fx.Int32(gpu.block_id("x"))
        kvh = fx.Int32(gpu.block_id("y"))
        capacity = fx.Int32(gpu.grid_dim.x)

        work = fx.recast_iter(fx.Int32, work_info_ptr)
        task = fx.ptr_load(
            fx.add_offset(work, slot * 4),
            result_type=fx.Vector.make_type(4, fx.Int32),
        )
        seq = fx.Int32(task[0])
        start = fx.Int32(task[1])
        end = fx.Int32(task[2])
        ctx = fx.Int32(task[3])
        # Padding tasks of the plan have start == end: nothing to do, no barrier
        # below is reached by only part of the CTA.
        if start < end:
            tok_begin = start * PLAN_TILE
            tok_end = fx.min(end * PLAN_TILE, ctx)
            n_tokens = tok_end - tok_begin
            n_tiles = (n_tokens + (TILE_TOKENS - 1)) // TILE_TOKENS
            n_iters = (n_tiles - warp + (NUM_WAVES - 1)) // NUM_WAVES

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            lds_base = fx.recast_iter(fx.Uint8, lds.buf.ptr)

            def _lds_ptr(byte_off, elem_ty):
                p = fx.add_offset(lds_base, fx.make_int_tuple(byte_off))
                ptr_ty = fx.PointerType.get(
                    elem_ty.ir_type, fx.AddressSpace.Shared, dsl_size_of(elem_ty)
                )
                return fx.recast_iter(ptr_ty, p)

            def _lds_load(byte_off, elem_ty):
                return fx.ptr_load(_lds_ptr(byte_off, elem_ty))

            def _lds_store(byte_off, elem_ty, value):
                fx.ptr_store(value, _lds_ptr(byte_off, elem_ty))

            # ---- Stage this task's page ids in LDS (pages past the context -> 0).
            bt = _gptr(block_tables_ptr, fx.Int32, 4)
            first_page = tok_begin // BS
            n_task_pages = (n_tokens + (BS - 1)) // BS
            num_pages = (ctx + (BS - 1)) // BS
            bt_row = fx.Int64(seq) * fx.Int64(max_blocks_per_seq)
            zero_i32 = fx.Int32(0)
            for r in range_constexpr(stage_pages // BLOCK_THREADS):
                idx = tid + r * BLOCK_THREADS
                pg = first_page + idx
                ok = (idx < n_task_pages) & (pg < num_pages)
                raw = fx.ptr_load(
                    fx.add_offset(bt, bt_row + fx.Int64(ok.select(pg, zero_i32)))
                )
                _lds_store(page_off + idx * 4, fx.Int32, ok.select(raw, zero_i32))

            # ---- Q operand, resident for the whole task (padding heads alias head 0).
            head = (s < G).select(s, zero_i32)
            q_row = fx.Int64(seq) * fx.Int64(stride_q_row) + fx.Int64(
                kvh * G + head
            ) * fx.Int64(stride_q_head)
            if const_expr(q_fp8):
                qd = _gptr(q_ptr, fx.Int32, 16)
                v4_i32 = fx.Vector.make_type(4, fx.Int32)
                q_lo = fx.ptr_load(
                    fx.add_offset(qd, (q_row + fx.Int64(16 * g)) // 4),
                    result_type=v4_i32,
                )
                q_hi = fx.ptr_load(
                    fx.add_offset(qd, (q_row + fx.Int64(64 + 16 * g)) // 4),
                    result_type=v4_i32,
                )
                q_b = fx.Vector.from_elements(
                    [q_lo[i] for i in range_constexpr(4)]
                    + [q_hi[i] for i in range_constexpr(4)],
                    dtype=fx.Int32,
                )
            else:
                qb = _gptr(q_ptr, fx.BFloat16, 16)
                v8_bf16 = fx.Vector.make_type(8, fx.BFloat16)
                q_b = [
                    fx.ptr_load(
                        fx.add_offset(qb, q_row + fx.Int64(32 * g + 8 * j)),
                        result_type=v8_bf16,
                    )
                    for j in range_constexpr(4)
                ]

            scale_log2e = softmax_scale * fx.Float32(LOG2E)
            if const_expr(has_q_scale):
                scale_log2e = scale_log2e * fx.Float32(
                    fx.ptr_load(_gptr(q_scale_ptr, fx.Float32, 4))
                )
            gpu.barrier()  # staged pages

            kd = _gptr(kd_ptr, fx.Int32, 16)
            vd = _gptr(vd_ptr, fx.Int32, 16)
            ks = _gptr(ks_ptr, fx.Uint8, 1)
            vs = _gptr(vs_ptr, fx.Uint8, 1)
            v4_i32 = fx.Vector.make_type(4, fx.Int32)

            neg_inf = fx.Float32(float("-inf"))
            zero_f = fx.Float32(0.0)
            wave_v_off = v_off + warp * V_TILE_BYTES
            # ds_read_b64_tr_b16 source for the PV A operand: lane (g, s) points at
            # token 4g + s//4, dims 16m + 4*(s%4) (+ m * 512 for the d-tile m).
            v_read_off = (
                wave_v_off + ((s % 4) // 2) * 256 + (4 * g + s // 4) * 16 + (s % 2) * 8
            )

            def _tile_of(it_i32):
                # The tile a wave owns at iteration `it`, clamped to a real one so
                # prefetches past the end stay in bounds.
                last = fx.max(n_tiles - 1, zero_i32)
                return fx.min(warp + it_i32 * NUM_WAVES, last)

            def _load_tile(it_i32):
                tile = _tile_of(it_i32)
                tok0 = tok_begin + tile * TILE_TOKENS
                page_rel = tok0 // BS - first_page
                # The page is wave-uniform: reading it into an SGPR keeps the page
                # arithmetic scalar, so a tile's four loads are a uniform base plus
                # a per-lane constant.
                pid = fx.Int64(
                    fx.Int32(
                        fx.rocdl.readfirstlane(
                            T.i32, _lds_load(page_off + page_rel * 4, fx.Int32)
                        )
                    )
                )
                t_in = tok0 % BS + s
                kd_u = fx.add_offset(
                    kd,
                    pid * fx.Int64(kd_page_dw) + fx.Int64(kvh) * fx.Int64(kd_head_dw),
                )
                vd_u = fx.add_offset(
                    vd,
                    pid * fx.Int64(vd_page_dw) + fx.Int64(kvh) * fx.Int64(vd_head_dw),
                )
                ks_u = fx.add_offset(
                    ks, pid * fx.Int64(ks_page) + fx.Int64(kvh) * fx.Int64(ks_head)
                )
                vs_u = fx.add_offset(
                    vs, pid * fx.Int64(vs_page) + fx.Int64(kvh) * fx.Int64(vs_head)
                )
                d_off = t_in * (DATA_BYTES // 4) + g * 4
                s_off = t_in * SCALE_BYTES + g
                return [
                    fx.ptr_load(fx.add_offset(kd_u, d_off), result_type=v4_i32),
                    fx.Int32(fx.ptr_load(fx.add_offset(ks_u, s_off))),
                    fx.ptr_load(fx.add_offset(vd_u, d_off), result_type=v4_i32),
                    fx.Int32(fx.ptr_load(fx.add_offset(vs_u, s_off))),
                ]

            n_raw = 4

            def _step(raw, itg, m_old, l_old, o_acc):
                """One 16-token tile of this wave: consume ``raw`` (the loads of
                iteration ``itg``) and fold it into (max, sum, O). An iteration
                past the wave's last real tile is fully masked."""
                k_raw, k_sc, v_raw, v_sc = raw
                k_sc = fx.Int32(k_sc)
                v_sc = fx.Int32(v_sc)
                # Unclamped: tokens of a phantom tile are all >= tok_end.
                tok0 = tok_begin + (warp + itg * NUM_WAVES) * TILE_TOKENS

                # ---- QK: S^T[token 4g+v][head s]
                if const_expr(q_fp8):
                    a_k = fx.Vector.from_elements(
                        [k_raw[i] for i in range_constexpr(4)]
                        + [fx.Int32(0) for _ in range_constexpr(4)],
                        dtype=fx.Int32,
                    )
                    acc = fx.Vector(
                        rocdl.mfma_scale_f32_16x16x128_f8f6f4(
                            T.f32x4,
                            [
                                a_k,
                                q_b,
                                fx.Vector.filled(4, 0.0, fx.Float32),
                                4,
                                0,
                                0,
                                k_sc,
                                0,
                                fx.Int32(0x7F7F7F7F),
                            ],
                        )
                    )
                else:
                    k_scale_f32 = _e8m0_to_f32(k_sc)
                    acc = fx.Vector.filled(4, 0.0, fx.Float32)
                    for j in range_constexpr(4):
                        a_k = fx.Vector.from_elements(
                            _fp4_dword_to_bf16(k_raw[j], k_scale_f32),
                            dtype=fx.BFloat16,
                        )
                        acc = fx.Vector(
                            rocdl.mfma_f32_16x16x32_bf16(
                                T.f32x4, [a_k, q_b[j], acc, 0, 0, 0]
                            )
                        )

                # ---- V: widen, stage transposed. An invalid token's scale is forced
                # to 0.0, so its row widens to exact zeros (and whatever the scale
                # byte held, 255 included, cannot reach PV).
                my_valid = (tok0 + s) < tok_end
                v_scale_f32 = my_valid.select(_e8m0_to_f32(v_sc), zero_f)
                for i in range_constexpr(4):
                    row = fx.Vector.from_elements(
                        _fp4_dword_to_bf16(v_raw[i], v_scale_f32), dtype=fx.BFloat16
                    )
                    _lds_store(
                        wave_v_off + (4 * g + i) * 256 + s * 16, fx.BFloat16, row
                    )

                # ---- online softmax over this tile (base 2)
                scores = [
                    ((tok0 + 4 * g + v) < tok_end).select(acc[v] * scale_log2e, neg_inf)
                    for v in range_constexpr(4)
                ]
                tile_max = fx.maxnumf(
                    fx.maxnumf(scores[0], scores[1]), fx.maxnumf(scores[2], scores[3])
                )
                tile_max = fx.maxnumf(tile_max, tile_max.shuffle_xor(16, WAVE))
                tile_max = fx.maxnumf(tile_max, tile_max.shuffle_xor(32, WAVE))
                m_new = fx.maxnumf(m_old, tile_max)
                m_safe = (m_new > neg_inf).select(m_new, zero_f)
                alpha = fx.exp2(m_old - m_safe, fastmath="fast")
                probs = [
                    fx.exp2(scores[v] - m_safe, fastmath="fast")
                    for v in range_constexpr(4)
                ]
                l_new = l_old * alpha + ((probs[0] + probs[1]) + (probs[2] + probs[3]))
                p_op = (
                    fx.Vector.from_elements(probs, dtype=fx.Float32)
                    .to(fx.BFloat16)
                    .bitcast(fx.Int16)
                )

                # ---- PV: O^T[d][head s] += V^T[d][token] P^T[token][head]
                # All eight transposed reads are issued before the first MFMA so
                # their LDS latency overlaps instead of serialising.
                v_ops = [
                    fx.Vector(
                        rocdl.ds_read_tr16_b64(
                            fx.Vector.make_type(4, fx.BFloat16),
                            fx.to_llvm_ptr(_lds_ptr(v_read_off + m * 512, fx.BFloat16)),
                        ).result
                    ).bitcast(fx.Int16)
                    for m in range_constexpr(8)
                ]
                new_o = [
                    fx.Vector(
                        rocdl.mfma_f32_16x16x16bf16_1k(
                            T.f32x4, [v_ops[m], p_op, o_acc[m] * alpha, 0, 0, 0]
                        )
                    )
                    for m in range_constexpr(8)
                ]
                return m_new, l_new, new_o

            # Software pipeline with no register rotation: the loop body consumes
            # DEPTH tiles, each from its own fixed slot, and refills that slot with
            # the tile DEPTH iterations ahead, so a tile's loads are in flight
            # across DEPTH tiles of compute and a slot is never copied (a copy
            # would force a full vmcnt wait). Past the wave's last tile the
            # address clamp re-reads a real tile and the masks drop it.
            n_groups = (n_iters + (DEPTH - 1)) // DEPTH
            init = [neg_inf, zero_f] + [
                fx.Vector.filled(4, 0.0, fx.Float32) for _ in range_constexpr(8)
            ]
            for d in range_constexpr(DEPTH):
                init += _load_tile(fx.Int32(d))

            for grp, state in range(fx.Int32(0), n_groups, fx.Int32(1), init=init):
                m_c = state[0]
                l_c = state[1]
                o_c = [fx.Vector(state[2 + m]) for m in range_constexpr(8)]
                grp_i32 = fx.Int32(grp)
                refills = []
                for d in range_constexpr(DEPTH):
                    raw = [state[10 + d * n_raw + i] for i in range_constexpr(n_raw)]
                    itg = grp_i32 * DEPTH + d
                    m_c, l_c, o_c = _step(raw, itg, m_c, l_c, o_c)
                    refills += _load_tile(itg + DEPTH)
                results = yield [m_c, l_c] + o_c + refills

            m_w = results[0]
            l_w = results[1]
            o_w = [fx.Vector(results[2 + m]) for m in range_constexpr(8)]

            # ---- Merge the four waves' states, then store the task's partials.
            # One barrier: every wave publishes (m, l) per head and its raw O, and
            # each of the G * 16 output threads (one head x 8 dims) merges its own
            # slice, so nothing is recomputed per element.
            l_w = l_w + l_w.shuffle_xor(16, WAVE)
            l_w = l_w + l_w.shuffle_xor(32, WAVE)
            if g == 0:
                _lds_store(m_off + (s * NUM_WAVES + warp) * 4, fx.Float32, m_w)
                _lds_store(l_off + (s * NUM_WAVES + warp) * 4, fx.Float32, l_w)
            if s < G:
                for m in range_constexpr(8):
                    _lds_store(
                        o_off + ((warp * G + s) * HEAD_DIM + 16 * m + 4 * g) * 4,
                        fx.Float32,
                        o_w[m],
                    )
            gpu.barrier()

            if tid < G * (HEAD_DIM // 8):
                h = tid // (HEAD_DIM // 8)
                c = tid % (HEAD_DIM // 8)
                v4f = fx.Vector.make_type(4, fx.Float32)
                ms = fx.ptr_load(
                    _lds_ptr(m_off + h * NUM_WAVES * 4, fx.Float32), result_type=v4f
                )
                ls = fx.ptr_load(
                    _lds_ptr(l_off + h * NUM_WAVES * 4, fx.Float32), result_type=v4f
                )
                h_max = fx.maxnumf(fx.maxnumf(ms[0], ms[1]), fx.maxnumf(ms[2], ms[3]))
                h_max_safe = (h_max > neg_inf).select(h_max, zero_f)
                wts = [
                    fx.exp2(ms[w] - h_max_safe, fastmath="fast")
                    for w in range_constexpr(NUM_WAVES)
                ]
                h_sum = zero_f
                for w in range_constexpr(NUM_WAVES):
                    h_sum = h_sum + ls[w] * wts[w]
                inv = (h_sum > zero_f).select(fx.Float32(rcp_f32(h_sum)), zero_f)
                o_lo = fx.Vector.filled(4, 0.0, fx.Float32)
                o_hi = fx.Vector.filled(4, 0.0, fx.Float32)
                for w in range_constexpr(NUM_WAVES):
                    base = o_off + ((w * G + h) * HEAD_DIM + c * 8) * 4
                    o_lo = (
                        o_lo
                        + fx.ptr_load(_lds_ptr(base, fx.Float32), result_type=v4f)
                        * wts[w]
                    )
                    o_hi = (
                        o_hi
                        + fx.ptr_load(_lds_ptr(base + 16, fx.Float32), result_type=v4f)
                        * wts[w]
                    )
                o_out = fx.Vector.from_elements(
                    [o_lo[i] * inv for i in range_constexpr(4)]
                    + [o_hi[i] * inv for i in range_constexpr(4)],
                    dtype=fx.Float32,
                ).to(out_dtype)
                pmax = _gptr(pmax_ptr, fx.Float32, 4)
                psum = _gptr(psum_ptr, fx.Float32, 4)
                pout = _gptr(pout_ptr, out_dtype, 16)
                partial = fx.Int64(kvh) * fx.Int64(capacity) + fx.Int64(slot)
                row = partial * G + fx.Int64(h)
                fx.ptr_store(
                    o_out, fx.add_offset(pout, row * HEAD_DIM + fx.Int64(c * 8))
                )
                if c == 0:
                    # The shared reducer expects maxima in natural-log units.
                    fx.ptr_store(
                        h_max * fx.Float32(1.0 / LOG2E), fx.add_offset(pmax, row)
                    )
                    fx.ptr_store(h_sum, fx.add_offset(psum, row))

    @flyc.jit
    def launch(
        pmax_ptr: fx.Pointer,
        psum_ptr: fx.Pointer,
        pout_ptr: fx.Pointer,
        q_ptr: fx.Pointer,
        kd_ptr: fx.Pointer,
        ks_ptr: fx.Pointer,
        vd_ptr: fx.Pointer,
        vs_ptr: fx.Pointer,
        block_tables_ptr: fx.Pointer,
        q_scale_ptr: fx.Pointer,
        work_info_ptr: fx.Pointer,
        max_blocks_per_seq: fx.Int32,
        stride_q_row: fx.Int32,
        stride_q_head: fx.Int32,
        kd_page_dw: fx.Int32,
        kd_head_dw: fx.Int32,
        ks_page: fx.Int32,
        ks_head: fx.Int32,
        vd_page_dw: fx.Int32,
        vd_head_dw: fx.Int32,
        vs_page: fx.Int32,
        vs_head: fx.Int32,
        softmax_scale: fx.Float32,
        work_capacity: fx.Int32,
        num_kv_heads: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        with CompilationContext.compile_hints({"fastmath": "contract"}):
            pa_decode_mxfp4_kernel(
                pmax_ptr,
                psum_ptr,
                pout_ptr,
                q_ptr,
                kd_ptr,
                ks_ptr,
                vd_ptr,
                vs_ptr,
                block_tables_ptr,
                q_scale_ptr,
                work_info_ptr,
                max_blocks_per_seq,
                stride_q_row,
                stride_q_head,
                kd_page_dw,
                kd_head_dw,
                ks_page,
                ks_head,
                vd_page_dw,
                vd_head_dw,
                vs_page,
                vs_head,
                softmax_scale,
            ).launch(
                grid=(work_capacity, num_kv_heads, 1),
                block=(BLOCK_THREADS, 1, 1),
                stream=stream,
            )

    return launch
