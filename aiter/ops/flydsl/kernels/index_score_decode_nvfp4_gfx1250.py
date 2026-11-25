# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""gfx1250 MiniMax-M3 lightning-indexer DECODE block scores over the NVFP4 index cache.

Drop-in for ATOM's Triton ``_decode_index_score_transposed_kernel`` on the FP4
index cache: for request b, its query row t (of ``query_len``), index head h and
128-token block blk,

    score[h, b * query_len + t, blk] =
        max over the block's keys k before the row's cutoff
        (seq_len[b] - query_len + t + 1) of
        dot(hi + lo * 2^-4, dequant(K[page, k])) * (sm_scale * log2 e)

with hi = e4m3(clamp(q, +-448)) and lo = e4m3((q - hi) * 16): the query in two
E4M3 terms. Blocks past a request's ``ceil(seq_len / 128)`` are left unwritten,
as there.

Math -- Triton's ``tl.dot_scaled`` pair, same instructions, operand roles and
accumulation order: per 16-key tile one ``v_wmma_scale16_f32_16x16x128_f8f6f4``
with A = the e2m1 key rows and their E4M3 per-16 scales (keys on M), B = the hi
term (E8M0 scale 1), then the same instruction with B = the lo term (E8M0 2^-4)
accumulating onto it; the causal mask (-inf); the max over the block's keys
(``v_max*_num_f32`` and ``ds_max_num_f32``: maximumNumber, order-free); one
multiply by sm_scale * log2(e). The query split uses Triton's conversions
(``v_cvt_scalef32_pk8_fp8_f32`` / ``v_cvt_scale_pk8_f32_fp8``).

Work. Triton scores a request in one-wave workgroups of a few blocks each and
pays the query load and split in every one of them; with 32 query rows a
request (4 index heads x 8 rows at TP1) that fixed cost is what the kernel
spends its time on. Here the grid is (requests, P), sized by the batch, 4 waves
a workgroup. Workgroup p scores the request's blocks [p c, p c + c),
c = ceil(live / P) <= 64. Wave w owns key tiles 2w, 2w+1 (32 keys) of every
block: it streams its quarter of a block (2 KiB of e2m1 rows, 256 B of scales)
global -> LDS with ``global_load_async_to_lds`` into a private ring of ``ring``
slots -- the data in flight costs no registers -- and reads the WMMA operands
back from LDS (16 B pieces XOR-swizzled: conflict free). The query is converted
once a workgroup and staged through LDS. 8 WMMAs a block and wave cover all 32
query rows; the four waves' partial maxima meet by LDS atomic max; after the
loop one barrier, and each store instruction writes one row's consecutive
blocks (coalesced).

Ported from ATOM's indexer context-parallel scorer
(``atom/model_ops/minimax_m3/indexer_cp_score_flydsl.py``), which scores one
rank's shard of the blocks the same way; this is its one-GPU form.
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as _llvm
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import T

from .index_score_prefill_nvfp4_gfx1250 import E4M3_MAX, PAGE
from .kernels_common import create_llvm_ptr
from .quant_utils import emit_cvt_scalef32_pk8_fp8_f32

WAVE = 32
WAVES = 4  # a workgroup; wave w owns key tiles 2w, 2w+1 of every block
THREADS = WAVE * WAVES
MAX_ROWS = 32  # (index head, query row) pairs a request: two 16-row WMMA tiles
MAX_BLOCKS = 64  # blocks a workgroup owns at most (two block-table VGPRs)
SLOT_PAYLOAD = 32 * 64  # a wave's quarter block: 32 key rows (16 B pieces swizzled)
SLOT = SLOT_PAYLOAD + 32 * 8  # + their 32 scale rows
QOP_LANE = 64  # LDS bytes of one lane's 16-dword B operand (16 B pieces swizzled)
E8M0_ONE = 0x7F7F7F7F7F7F7F7F  # q_hi's scale bytes (2^0)
E8M0_LO = 0x7B7B7B7B7B7B7B7B  # q_lo's scale bytes (2^-4)
COPIES = 5  # async copies of a quarter block: 4 x 512 B payload + 256 B scales


def _wmma(a, b, c, sa, sb, row_a, row_b):
    """D = A(e2m1, E4M3 scales row ``row_a``) x B(e4m3, E8M0 scales row
    ``row_b``) + C.

    Raw ROCDL: FlyDSL's ``WMMAScale(block_size=16)`` atom hard-codes E8M0 scale
    formats and scale row 0. ``pa_decode.pipeline_nvfp4._wmma_scale16`` is this
    instruction at row 0 only; it is left as it is (the paged decode's kernels
    are prebuilt from that file), and a wave's second key / query tile here
    reads its scales from row 1."""

    def attr(text):
        return ir.Attribute.parse(text)

    return fx.Vector(
        rocdl.wmma_scale16_f32_16x16x128_f8f6f4(
            fx.Vector.make_type(8, fx.Float32),
            fx.as_ir_value(a),
            fx.as_ir_value(b),
            fx.as_ir_value(c),
            fx.as_ir_value(sa),
            fx.as_ir_value(sb),
            fmtA=attr("#rocdl<matrix_format fp4_e2m1>"),
            fmtB=attr("#rocdl<matrix_format fp8_e4m3>"),
            modC=attr("#rocdl<wmma_c_modifier none>"),
            scaleAType=attr(f"#rocdl<wmma_matrix_scale row{row_a}>"),
            fmtScaleA=attr("#rocdl<wmma_matrix_scale_format e4m3>"),
            scaleBType=attr(f"#rocdl<wmma_matrix_scale row{row_b}>"),
            fmtScaleB=attr("#rocdl<wmma_matrix_scale_format e8>"),
            reuseA=False,
            reuseB=False,
        ).result
    )


def _to_fp8x8(v8):
    """8 fp32 -> 8 e4m3 in two dwords (v_cvt_scalef32_pk8_fp8_f32, scale 1)."""
    return fx.Vector(
        emit_cvt_scalef32_pk8_fp8_f32(
            fx.as_ir_value(v8),
            fx.as_ir_value(fx.Float32(1.0)),
            v2i32_ty=T.i32x2,
            rocdl=rocdl,
        )
    )


# Raw ROCDL: FlyDSL has no typed conversion of packed FP8 back to FP32; this is
# the instruction Triton's ``.to(tl.float32)`` lowers to, and the query split
# must round as Triton's does.
def _from_fp8x8(w2):
    """8 e4m3 (two dwords) -> 8 fp32 (v_cvt_scale_pk8_f32_fp8, E8M0 1.0)."""
    return fx.Vector(
        rocdl.cvt_scale_pk8_f32_fp8(
            T.vec(8, T.f32), fx.as_ir_value(w2), fx.as_ir_value(fx.Int32(0x7F)), 0
        )
    )


def _swz(row, piece):
    """LDS byte offset of 16 B piece ``piece`` (0..3) of 64 B row ``row``:
    pieces XOR-swizzled by (row / 4) % 4, so 16 B reads of rows u = 0..15 at
    one piece hit distinct banks."""
    return row * 64 + ((piece ^ ((row >> 2) & 3)) * 16)


def _lds_i32(base_u8, off, n):
    return fx.recast_iter(
        fx.PointerType.get(fx.Int32.ir_type, fx.AddressSpace.Shared, min(4 * n, 16)),
        fx.add_offset(base_u8, off),
    )


@functools.cache
def compile_index_score_decode_nvfp4(*, heads: int, query_len: int, ring: int = 4):
    """Launcher for (index heads, query rows a request); ``ring``: LDS slots
    (blocks in flight + 1) a wave."""
    ql = query_len
    rows = heads * ql
    nqt = (rows + 15) // 16  # 16-row query tiles
    if not 1 <= rows <= MAX_ROWS:
        raise ValueError(f"heads * query_len must be in [1, {MAX_ROWS}], got {rows}")
    if not 2 <= ring <= 8:
        raise ValueError("ring must be in [2, 8]")
    RING = ring
    name = f"index_score_decode_nvfp4_d128_h{heads}_q{ql}_ring{ring}"
    RING_BYTES = WAVES * RING * SLOT
    PART = MAX_BLOCKS * 32  # maxima [block][row], combined by LDS atomic max
    # B operands staged in the last slots
    assert nqt * 2 * 32 * QOP_LANE <= WAVES * SLOT
    JH = 4 // (THREADS // (32 * nqt))  # K chunks a converting thread takes

    @fx.struct
    class SharedStorage:
        ring_buf: fx.Array[fx.Uint8, RING_BYTES, 16]
        part: fx.Array[fx.Float32, PART, 16]

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def index_score_kernel(
        arg_q: fx.Pointer,  # [batch * ql, heads, 128] bf16
        arg_kd: fx.Pointer,  # [pages, 128, 64] u8 (e2m1 pairs)
        arg_ks: fx.Pointer,  # [pages, 128, 8] u8 (E4M3)
        arg_bt: fx.Pointer,  # [batch, >= blocks] i32
        arg_len: fx.Pointer,  # [batch] i32
        arg_out: fx.Pointer,  # [heads, >= batch * ql, >= blocks] f32
        i32_q_tok: fx.Int32,
        i32_q_head: fx.Int32,
        i32_kd_page: fx.Int32,
        i32_ks_page: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_out_head: fx.Int32,
        i32_out_row: fx.Int32,
        i32_blocks: fx.Int32,
        i32_programs: fx.Int32,
        f32_scale: fx.Float32,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        lane = tid % WAVE
        # Raw readfirstlane: makes the wave id wave-uniform (an SGPR), which fx
        # cannot state about a value derived from the thread id.
        wave = fx.Int32(rocdl.readfirstlane(T.i32, fx.as_ir_value(tid // WAVE)))
        g = lane // 16
        u = lane % 16
        b = fx.Int32(gpu.block_id("x"))
        program = fx.Int32(gpu.block_id("y"))
        zero = fx.Int32(0)

        # ---- the query rows this thread converts (thread t: B lane t % 32 of
        # tile (t // 32) % nqt, K chunks [JH h, JH h + JH), h = t // 32 nqt)
        q_ptr = fx.recast_iter(fx.BFloat16, arg_q)
        t_lane = tid % 32
        t_qt = (tid // 32) % nqt
        t_jh = tid // (32 * nqt)
        t_on = t_jh * JH < 4
        tu, tg = t_lane % 16, t_lane // 16
        tcol = tu + t_qt * 16
        tlive = (tcol < rows) & t_on
        tcol = tlive.select(tcol, zero)
        tbase = fx.Int64(b * ql + tcol // heads) * fx.Int64(i32_q_tok) + fx.Int64(
            tcol % heads
        ) * fx.Int64(i32_q_head)
        bt = fx.recast_iter(fx.Int32, arg_bt)
        last_col = i32_blocks - 1

        def load_windows(block_fn):
            ws = []
            for h in range_constexpr(MAX_BLOCKS // 32):
                blk = block_fn(lane + 32 * h)
                blk = (blk < last_col).select(blk, last_col)
                ws.append(fx.Int32(bt[b * i32_bt_stride + blk]))
            return ws

        length = fx.Int32(fx.recast_iter(fx.Int32, arg_len)[b])
        live_end = (length + (PAGE - 1)) // PAGE
        live_end = (live_end < zero).select(zero, live_end)
        live_end = (live_end < i32_blocks).select(live_end, i32_blocks)
        # workgroup p: blocks [start, start + nblk), c = ceil(live / P)
        chunk = (live_end + i32_programs - 1) // i32_programs
        start = program * chunk
        nblk = live_end - start
        nblk = (nblk < chunk).select(nblk, chunk)

        def block_of(j):
            return start + j

        # Blocks from `mid` on may hold keys at or past a row's cutoff.
        min_cutoff = length - ql + 1
        first_masked = ((min_cutoff - PAGE) >> 7) + 1
        mid = (first_masked < zero).select(zero, first_masked)
        if nblk > zero:
            windows = load_windows(block_of)
            # query rows (issued after the block table: loads return in order)
            q_raw = []
            for jj in range_constexpr(JH):
                jq = (t_jh * JH + jj) % 4
                q_raw.append(
                    fx.Vector(
                        fx.ptr_load(
                            fx.add_offset(q_ptr, tbase + fx.Int64(tg * 16 + jq * 32)),
                            result_type=fx.Vector.make_type(16, fx.BFloat16),
                        )
                    ).to(fx.Float32)
                )

            lds = fx.SharedAllocator().allocate(SharedStorage).peek()
            part = lds.part.ptr
            ring_u8 = fx.recast_iter(fx.Uint8, lds.ring_buf.ptr)
            # the B operands stage in each wave's last ring slot (free until
            # the main loop's first copy): operand p = 2 qt + hl in wave p's
            q_u8 = ring_u8

            def q_off(p, lane_, k):
                return (
                    (p * RING + (RING - 1)) * SLOT
                    + lane_ * QOP_LANE
                    + ((k ^ ((lane_ >> 2) & 3)) * 16)
                )

            lds_base = fx.Int64(fx.ptrtoint(lds.ring_buf.ptr))
            ring_wave = lds_base + fx.Int64(wave * (RING * SLOT))
            kd_base = fx.Int64(fx.ptrtoint(arg_kd))
            ks_base = fx.Int64(fx.ptrtoint(arg_ks))
            # copy c of a quarter: lane's 16 B chunk 32 c + lane of the 2 KiB
            # (row (32 c + lane) // 4, 16 B piece lane % 4), pieces swizzled.
            cp_src = [
                fx.Int64(wave * 2048 + c * 512 + lane * 16) for c in range_constexpr(4)
            ]
            cp_dst = [
                fx.Int64(_swz((c * 32 + lane) // 4, lane % 4))
                for c in range_constexpr(4)
            ]
            sc_src = fx.Int64(wave * 256 + lane * 8)
            sc_dst = fx.Int64(SLOT_PAYLOAD + lane * 8)

            def page_of(j):
                # Raw readlane: a wave-uniform pick from a lane-indexed VGPR
                # (the block-table window); fx has no cross-lane extract.
                lo = fx.Int32(
                    rocdl.readlane(
                        T.i32, fx.as_ir_value(windows[0]), fx.as_ir_value(j % 32)
                    )
                )
                hi = fx.Int32(
                    rocdl.readlane(
                        T.i32, fx.as_ir_value(windows[1]), fx.as_ir_value(j % 32)
                    )
                )
                return (j < 32).select(lo, hi)

            def issue(j):
                """Async copies of block j's quarter into ring slot j % RING."""
                page = fx.Int64(page_of(j))
                slot = ring_wave + fx.Int64((j % RING) * SLOT)
                pd = kd_base + page * fx.Int64(i32_kd_page)
                # Raw ROCDL: fx.copy / the tensor-DMA atom move whole tiles
                # with a wave's descriptor; these per-lane 16 B / 8 B async
                # loads place each piece at its swizzled LDS offset.
                for c in range_constexpr(4):
                    rocdl.global_load_async_to_lds_b128(
                        create_llvm_ptr(pd + cp_src[c], address_space=1),
                        create_llvm_ptr(slot + cp_dst[c], address_space=3),
                        0,
                    )
                rocdl.global_load_async_to_lds_b64(
                    create_llvm_ptr(
                        ks_base + page * fx.Int64(i32_ks_page) + sc_src,
                        address_space=1,
                    ),
                    create_llvm_ptr(slot + sc_dst, address_space=3),
                    0,
                )

            for d in range_constexpr(RING - 1):
                if fx.Int32(d) < nblk:
                    issue(fx.Int32(d))

            # ---- query -> B operands (hi / lo E4M3), once a workgroup ------
            if t_on:
                for jj in range_constexpr(JH):
                    jq = t_jh * JH + jj
                    vals = q_raw[jj]
                    hw, lw = [], []
                    for h in range_constexpr(2):
                        v8 = fx.Vector.from_elements(
                            [
                                fx.minnumf(
                                    fx.maxnumf(
                                        tlive.select(
                                            fx.Float32(vals[h * 8 + e]), fx.Float32(0.0)
                                        ),
                                        fx.Float32(-E4M3_MAX),
                                    ),
                                    fx.Float32(E4M3_MAX),
                                )
                                for e in range_constexpr(8)
                            ],
                            dtype=fx.Float32,
                        )
                        hi = _to_fp8x8(v8)
                        lo = _to_fp8x8(
                            (v8 - _from_fp8x8(hi))
                            * fx.Vector.filled(8, 16.0, fx.Float32)
                        )
                        hw += [hi[0], hi[1]]
                        lw += [lo[0], lo[1]]
                    for hl, words in ((0, hw), (1, lw)):
                        fx.ptr_store(
                            fx.Vector.from_elements(words, dtype=fx.Int32),
                            _lds_i32(q_u8, q_off(t_qt * 2 + hl, t_lane, jq), 4),
                        )
            # partial maxima start as NaN (the identity of maximumNumber)
            nan = fx.Vector.from_elements(
                [fx.Int32(0x7FC00000)], dtype=fx.Int32
            ).bitcast(fx.Float32)[0]
            for k in range_constexpr(PART // THREADS):
                fx.ptr_store(nan, part + (tid + k * THREADS))
            gpu.barrier()
            q_hi, q_lo, cutoffs = [], [], []
            for qt in range_constexpr(nqt):
                col = u + qt * 16
                col = (col < rows).select(col, zero)
                cutoffs.append(length - ql + col // heads + 1)
                ops = []
                for hl in range_constexpr(2):
                    words = []
                    for k in range_constexpr(4):
                        v = fx.Vector(
                            fx.ptr_load(
                                _lds_i32(q_u8, q_off(qt * 2 + hl, lane, k), 4),
                                result_type=fx.Vector.make_type(4, fx.Int32),
                            )
                        )
                        words += [v[e] for e in range_constexpr(4)]
                    ops.append(fx.Vector.from_elements(words, dtype=fx.Int32))
                q_hi.append(ops[0])
                q_lo.append(ops[1])
            # Raw s_wait_dscnt (here and in the main loop): the operand loads
            # must have left LDS before their slot is refilled by an async
            # copy, an ordering fx does not track.
            rocdl.s_wait_dscnt(0)
            gpu.barrier()  # every wave holds its operands: the last slots are free
            # B scales must sit in VGPRs (an inline / SGPR i64 scale reads
            # wrong): OR in a lane-varying zero the compiler cannot fold.
            lane_zero = fx.Int64((lane + length) >> 31)
            s_hi = fx.Int64(E8M0_ONE) | lane_zero
            s_lo = fx.Int64(E8M0_LO) | lane_zero

            zero8 = fx.Vector.filled(8, 0.0, fx.Float32)
            neg_inf = fx.Float32(float("-inf"))
            a_off = [
                [_swz(tt * 16 + u, g + 2 * h) for h in range_constexpr(2)]
                for tt in range_constexpr(2)
            ]

            def partial_max(j, masked):
                """[nqt] per-lane max over this wave's 32 keys of block j (both
                lane halves hold every row of their tile after the cross-half
                max)."""
                slot = wave * (RING * SLOT) + (j % RING) * SLOT
                sa = fx.Vector(
                    fx.ptr_load(
                        _lds_i32(ring_u8, slot + SLOT_PAYLOAD + lane * 8, 2),
                        result_type=fx.Vector.make_type(2, fx.Int32),
                    )
                ).bitcast(fx.Int64)[0]
                key0 = block_of(j) * PAGE + wave * 32
                ms = [None] * nqt
                for tt in range_constexpr(2):
                    lo4 = fx.Vector(
                        fx.ptr_load(
                            _lds_i32(ring_u8, slot + a_off[tt][0], 4),
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    )
                    hi4 = fx.Vector(
                        fx.ptr_load(
                            _lds_i32(ring_u8, slot + a_off[tt][1], 4),
                            result_type=fx.Vector.make_type(4, fx.Int32),
                        )
                    )
                    a = fx.Vector.from_elements(
                        [lo4[e] for e in range_constexpr(4)]
                        + [hi4[e] for e in range_constexpr(4)],
                        dtype=fx.Int32,
                    )
                    for qt in range_constexpr(nqt):
                        # A = this tile's e2m1 keys, scales in row tt; B =
                        # the E4M3 query term, E8M0 scales in row qt % 2.
                        c = zero8
                        for term, s_term in ((q_hi[qt], s_hi), (q_lo[qt], s_lo)):
                            c = _wmma(a, term, c, sa, s_term, tt, qt % 2)
                        vals = [fx.Float32(c[v]) for v in range_constexpr(8)]
                        if const_expr(masked):
                            pos0 = key0 + tt * 16 + g * 8
                            vals = [
                                (pos0 + v < cutoffs[qt]).select(vals[v], neg_inf)
                                for v in range_constexpr(8)
                            ]
                        m = vals[0]
                        for v in range_constexpr(1, 8):
                            m = fx.maxnumf(m, vals[v])
                        ms[qt] = m if tt == 0 else fx.maxnumf(ms[qt], m)
                for qt in range_constexpr(nqt):
                    ms[qt] = fx.maxnumf(ms[qt], ms[qt].shuffle_xor(16, WAVE))
                return ms

            wr_row = u + g * 16  # lane (u, g) writes row 16 g + u (tile g)

            part_base = fx.Int64(fx.ptrtoint(part))

            def put(ms, j):
                val = ms[0]
                if const_expr(nqt == 2):
                    val = (g == 0).select(ms[0], ms[1])
                # Raw LLVM atomicrmw fmax (ds_max_num_f32, maximumNumber):
                # fx.atomic_fmax goes through the integer atomics and orders
                # NaN by its bits, but the partials start as NaN and a NaN
                # score must lose to a number, as in Triton's max.
                _llvm.atomicrmw(
                    _llvm.AtomicBinOp.fmax,
                    create_llvm_ptr(
                        part_base + fx.Int64((j * 32 + wr_row) * 4), address_space=3
                    ),
                    fx.as_ir_value(val),
                    _llvm.AtomicOrdering.monotonic,
                    syncscope="workgroup",
                )

            # ---- main loop: block j + RING - 1 streams in while j is scored
            for j in range(zero, nblk):
                j = fx.Int32(j)
                ja = j + (RING - 1)
                # Raw s_wait_asynccnt: the async-copy counter is not modelled by fx.
                if ja < nblk:
                    rocdl.s_wait_dscnt(0)  # slot ja % RING's last readers are done
                    issue(ja)
                    rocdl.s_wait_asynccnt(COPIES * (RING - 1))
                else:
                    rocdl.s_wait_asynccnt(0)
                if block_of(j) < mid:
                    put(partial_max(j, False), j)
                else:
                    put(partial_max(j, True), j)
            gpu.barrier()
            # ---- combine the four waves' maxima, scale, store: one row's
            # consecutive blocks per instruction (lane = block) --------------
            out = fx.recast_iter(fx.Float32, arg_out)
            for it in range_constexpr(32 * MAX_BLOCKS // THREADS):
                idx = tid + it * THREADS
                jb = idx % MAX_BLOCKS
                row = idx // MAX_BLOCKS
                if (jb < nblk) & (row < rows):
                    v = fx.Float32(fx.ptr_load(part + (jb * 32 + row))) * f32_scale
                    out[
                        fx.Int64(row % heads) * fx.Int64(i32_out_head)
                        + fx.Int64(b * ql + row // heads) * fx.Int64(i32_out_row)
                        + fx.Int64(block_of(jb))
                    ] = v

    @flyc.jit
    def launch(
        arg_q: fx.Pointer,
        arg_kd: fx.Pointer,
        arg_ks: fx.Pointer,
        arg_bt: fx.Pointer,
        arg_len: fx.Pointer,
        arg_out: fx.Pointer,
        i32_q_tok: fx.Int32,
        i32_q_head: fx.Int32,
        i32_kd_page: fx.Int32,
        i32_ks_page: fx.Int32,
        i32_bt_stride: fx.Int32,
        i32_out_head: fx.Int32,
        i32_out_row: fx.Int32,
        i32_blocks: fx.Int32,
        i32_programs: fx.Int32,
        f32_scale: fx.Float32,
        i32_batch: fx.Int32,
        stream: fx.Stream,
    ):
        index_score_kernel(
            arg_q,
            arg_kd,
            arg_ks,
            arg_bt,
            arg_len,
            arg_out,
            i32_q_tok,
            i32_q_head,
            i32_kd_page,
            i32_ks_page,
            i32_bt_stride,
            i32_out_head,
            i32_out_row,
            i32_blocks,
            i32_programs,
            f32_scale,
        ).launch(
            grid=(fx.Int64(i32_batch), fx.Int64(i32_programs), 1),
            block=(THREADS, 1, 1),
            stream=stream,
        )

    return {"launch": launch, "kernel": index_score_kernel}
