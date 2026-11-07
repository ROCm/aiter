# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FlyDSL ``rope_quant_window``: DeepSeek V4.1's RoPE / MXFP8 / window seam.

Byte-for-byte the ATOM Triton kernel (``atom.model_ops.deepseek_v41.rope_window``):
the query half either rotates the RoPE tails in place or emits MXFP8
(FP8 e4m3fn values plus one E8M0 code per 32), and the KV half quantizes the
normed row the same way into any of a dequantized BF16 copy (``qat``), packed
values/scales rows, or the decode ring (FP8 data / E8M0 scale planes or a BF16
pool).

The rule both halves share: GPT-J rotation in fp32 (a NoPE pair is rotated by
cos 1 / sin 0, which turns some -0 into +0), the result rounded to BF16, an
amax per 32 floored at 1e-10, a power-of-two step ``ceil_pow2(amax / 448)``,
and ``fp8(y / step)``. Dividing by a power of two is exact, and the BF16 amax
bounds every ``|y / step|`` by 448, so the clamp never fires.

One launch, one grid: ``[query workgroups | KV workgroups]``. The query half
has two layouts. Rows mode (gfx950 and gfx1250) gives lane ``t`` the 8 values
at ``c * 8 * wave + 8 t`` of each slice ``c``, so every load is one contiguous
``16 * wave``-byte span. TDM mode (gfx1250, large shapes) streams rows into a
double-buffered LDS ring with the tensor-data mover, one row per wave per tile.
"""

from functools import cache, lru_cache

import flydsl.compiler as flyc
import flydsl.expr as fx
import torch
from flydsl.expr import const_expr, range_constexpr
from flydsl.expr import math as fmath
from flydsl.expr.rocdl import tdm_ops
from flydsl.expr.typing import Int32, Int64, ReductionOp, Stream, T

from aiter.ops.flydsl.kernels import buffer_ops
from aiter.ops.flydsl.kernels.gemm_common_gfx1250 import make_lds_copy_ops
from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

FP8_MAX = 448.0
AMAX_FLOOR = 1e-10

QUERY_NONE, QUERY_ROTATE, QUERY_FP8 = 0, 1, 2

# Rows-mode shapes: (waves per WG, query rows per wave), short WG below
# ROWS_SMALL_MAX query rows, where a launch is latency.
ROWS_SMALL = {"gfx1250": (8, 2), "gfx950": (4, 2)}
ROWS_LARGE = (4, 4)
ROWS_SMALL_MAX = 2048
# TDM mode: (LDS buffers, tiles per WG, rows per tile = waves per WG).
TDM_MIN_ROWS = 512 * 1024
TDM_DENSE_ROWS = 1024 * 1024
TDM_DENSE = (2, 16, 8)
TDM_SPARSE = (2, 8, 8)
# In-place rotation: (waves per WG, rows per lane, all loads before any
# store), by the first row bound above the launch's rows. A wave64 pass
# covers twice the rows, so gfx950 keeps the short shape longer.
ROTATE_SHAPES = {
    "gfx1250": ((12288, (4, 1)), (None, (8, 4))),
    "gfx950": ((131072, (4, 1)), (None, (8, 4))),
}


def _lds_store_u8(lds_i32, byte_offset, value):
    ptr_ty = fx.PointerType.get(
        elem_ty=fx.Int8.ir_type, address_space=fx.AddressSpace.Shared, alignment=1
    )
    layout = fx.make_layout(1, 1)
    atom = fx.make_copy_atom(fx.UniversalCopy(8), fx.Int8)
    rmem = fx.make_rmem_tensor(layout, fx.Int8)
    rmem.store(fx.Vector.from_elements([value.to(fx.Int8)], dtype=fx.Int8))
    fx.copy(
        atom,
        rmem,
        fx.Tensor(fx.make_view(fx.inttoptr(ptr_ty, lds_i32 + byte_offset), layout)),
    )


@lru_cache(maxsize=64)
def _build(
    *,
    H,
    D,
    RD,
    wave,
    pk8,
    query_mode,
    W,
    R,
    tdm,
    K,
    CT,
    pos64,
    qat,
    packed,
    ring,
    ring_fp8,
    batch64,
    cu64,
    slot64,
    RING_SLOTS,
    SLOT_ROWS,
    RING_STRIDE,
    RUN_ROWS,
):
    NOPE = D - RD
    CW = 8 * wave  # elements per slice
    CHUNKS = D // CW
    assert D % CW == 0 and NOPE >= (CHUNKS - 1) * CW and RD % 8 == 0
    ROPE_LO = (NOPE - (CHUNKS - 1) * CW) // 8  # first RoPE lane of the last slice
    NG = D // 32
    QUADS = wave // 4  # groups per slice
    ROWS_WG = W * R
    SHARED_CS = H > 0 and H % R == 0
    WORDS = R * NG // 4  # scale dwords per wave
    assert WORDS <= wave
    ROT_LANES = RD // 8  # lanes per row in the in-place rotation
    assert wave % ROT_LANES == 0
    ROT_ROWS_WG = W * (wave // ROT_LANES)
    # TDM layout: lane t holds the 16 contiguous values at 16 t.
    VEC = D // wave
    TPAIRS = VEC // 2
    T_ROPE_LO = NOPE // VEC
    TILE_BYTES = W * D * 2
    TGROUP = H // W if H % W == 0 else 0  # tiles per token
    HOIST = TGROUP > 0 and (CT % TGROUP == 0 or TGROUP % CT == 0)
    if tdm:
        assert query_mode == QUERY_FP8 and pk8 and wave == 32 and VEC == 16
        assert NOPE % VEC == 0 and CT >= K
    LDS_BYTES = K * TILE_BYTES if tdm else W * R * NG
    EV4, OD4 = [0, 2, 4, 6], [1, 3, 5, 7]
    ILV4 = [0, 4, 1, 5, 2, 6, 3, 7]
    EV, OD = list(range(0, VEC, 2)), list(range(1, VEC, 2))
    ILV = [(i // 2) if i % 2 == 0 else (TPAIRS + i // 2) for i in range(VEC)]

    name = f"rope_quant_window_H{H}_D{D}_RD{RD}_w{wave}_q{query_mode}"
    name += f"_t{K}x{CT}x{W}" if tdm else f"_r{W}x{R}"
    name += "_pk8" if pk8 else ""
    name += "_p64" if pos64 else ""
    name += "_qat" if qat else ""
    name += "_packed" if packed else ""
    if ring:
        name += f"_ring{'p' if ring_fp8 else 'h'}{RING_SLOTS}x{SLOT_ROWS}x{RING_STRIDE}x{RUN_ROWS}"
        name += f"_i{int(batch64)}{int(cu64)}{int(slot64)}"

    @flyc.kernel(name=name, known_block_size=[wave, W, 1])
    def kernel(
        q_in: fx.Tensor,
        num_rows: Int32,
        num_tokens: Int32,
        gx_q: Int32,
        query: fx.Pointer,
        kv: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        positions: fx.Pointer,
        qat_out: fx.Pointer,
        values: fx.Pointer,
        scales: fx.Pointer,
        query_values: fx.Pointer,
        query_scales: fx.Pointer,
        pool: fx.Pointer,
        pool_scale: fx.Pointer,
        batches: fx.Pointer,
        cu: fx.Pointer,
        slots: fx.Pointer,
        kv_stride: Int32,
        values_stride: Int32,
        scales_stride: Int32,
        pool_stride: Int32,
        pool_scale_stride: Int32,
        ring_start: Int64,
    ):
        tid = fx.thread_idx.x
        wv = fx.thread_idx.y
        g = fx.block_idx.x

        def addr(p):
            return fx.Int64(fx.ptrtoint(p))

        def rsrc(a):
            return buffer_ops.create_buffer_resource_from_addr(a)

        cos_rsrc = rsrc(addr(cos))
        sin_rsrc = rsrc(addr(sin))
        pos_rsrc = rsrc(addr(positions))
        lds_ptr = fx.SharedAllocator(static=False).allocate(LDS_BYTES)._ptr
        lds_idx = fx.index_cast(T.index, fx.ptrtoint(lds_ptr))
        lds_i32 = fx.index_cast(T.i32, lds_idx)
        quad = tid >> 2
        one4 = fx.Vector.filled([4], 1.0, fx.Float32)
        zero4 = fx.Vector.filled([4], 0.0, fx.Float32)

        def load_index(r, i, wide):
            """Element ``i`` of an int32 / int64 index tensor, as int32."""
            return fx.Int32(
                buffer_ops.buffer_load(
                    r, i * 2 if wide else i, vec_width=1, dtype=T.i32
                )
            )

        def token_of(row):
            # Rows are never negative; a shift skips signed floor division's fixup.
            if const_expr(H & (H - 1) == 0):
                return row >> (H.bit_length() - 1)
            return row // H

        def position_of(tok):
            # A padding row's -1 still has to address inside the table.
            return fx.max(load_index(pos_rsrc, tok, pos64), fx.Int32(0))

        def rotate4(x, c, s):
            ev = x.shuffle(x, EV4)
            od = x.shuffle(x, OD4)
            return (ev * c - od * s).shuffle(ev * s + od * c, ILV4)

        def slice_cs(p):
            """cos/sin for the last slice's 4 pairs; identity on NoPE lanes."""
            is_rope = tid >= ROPE_LO
            off = p * (RD // 2) + fx.max(tid - ROPE_LO, fx.Int32(0)) * 4
            c = fx.Vector(
                buffer_ops.buffer_load(cos_rsrc, off, vec_width=4, dtype=T.bf16)
            )
            s = fx.Vector(
                buffer_ops.buffer_load(sin_rsrc, off, vec_width=4, dtype=T.bf16)
            )
            return is_rope.select(c.to(fx.Float32), one4), is_rope.select(
                s.to(fx.Float32), zero4
            )

        def group_code(am, lanes):
            """E8M0 code of a group whose amax is spread over ``lanes`` lanes."""
            sh = 1
            while sh < lanes:
                am = am.maximumf(am.shuffle_xor(sh, wave))
                sh *= 2
            am = am.maximumf(fx.Float32(AMAX_FLOOR))
            bits = (am * fx.Float32(1.0 / FP8_MAX)).bitcast(fx.Int32)
            return (bits >> 23) + ((bits & 0x7FFFFF) != 0).select(
                fx.Int32(1), fx.Int32(0)
            )

        def to_fp8(y, code):
            """``fp8(y / 2^(code - 127))`` for 8 values, as 2 dwords."""
            if const_expr(pk8):
                step = (code << 23).bitcast(fx.Float32)
                if const_expr(y.dtype == fx.BFloat16):
                    pk = fx.rocdl.cvt_scalef32_pk8_fp8_bf16(
                        T.vec(2, T.i32), y.ir_value(), step.ir_value()
                    )
                else:
                    pk = fx.rocdl.cvt_scalef32_pk8_fp8_f32(
                        T.vec(2, T.i32), y.ir_value(), step.ir_value()
                    )
                pk = fx.Vector(pk)
                return [pk[0], pk[1]]
            qv = y.to(fx.Float32) * ((254 - code) << 23).bitcast(fx.Float32)
            dw = []
            for d in range_constexpr(2):
                w = fx.Int32(0).ir_value()
                w = fx.rocdl.cvt_pk_fp8_f32(
                    T.i32, qv[4 * d].ir_value(), qv[4 * d + 1].ir_value(), w, 0
                )
                w = fx.rocdl.cvt_pk_fp8_f32(
                    T.i32, qv[4 * d + 2].ir_value(), qv[4 * d + 3].ir_value(), w, 1
                )
                dw.append(w)
            return dw

        def dequant(dw, code):
            """BF16 bits of ``fp8 * 2^(code - 127)``, decoded in integers.

            Exact, the FP8 value has 4 significant bits. A normal e4m3 byte
            ``s eeee mmm`` lands on exponent ``e + code - 7`` with the mantissa
            shifted up 4; a subnormal is ``m * 2^(code - 136)``, done in fp32.
            """
            b = (
                fx.Vector.from_elements(dw, dtype=fx.Int32)
                .bitcast(fx.Int8)
                .to(fx.Int32)
                & 0xFF
            )
            mag = b & 0x7F
            normal = (mag << 4) + ((code - 7) << 7)
            sub = (mag.to(fx.Float32) * ((code - 9) << 23).bitcast(fx.Float32)).bitcast(
                fx.Int32
            ) >> 16
            bits = ((b & 0x80) << 8) | (mag >= 8).select(normal, sub)
            return bits.to(fx.Int16).bitcast(fx.Int32)

        def quant_slices(xs, c, s):
            """One row in slice layout: ``(fp8 dwords, code)`` per slice."""
            out = []
            for k in range_constexpr(CHUNKS):
                xb = xs[k]
                if const_expr(k == CHUNKS - 1):
                    # Rounding is monotonic, so the rounded row's amax is the
                    # rounded amax.
                    v = rotate4(xb.to(fx.Float32), c, s)
                    am = (
                        fmath.absf(v)
                        .reduce(ReductionOp.MAX)
                        .to(fx.BFloat16)
                        .to(fx.Float32)
                    )
                    y = (
                        v.to(fx.BFloat16)
                        if const_expr(pk8)
                        else v.to(fx.BFloat16).to(fx.Float32)
                    )
                elif const_expr(pk8):
                    # fma(swap(x), (-0, +0), x) per (even, odd) pair is the cos 1
                    # / sin 0 rotation, signed zeros and all.
                    nope_c = fx.Vector.from_elements([-0.0, 0.0] * 4, dtype=fx.BFloat16)
                    y = fmath.fma(xb.shuffle(xb, [1, 0, 3, 2, 5, 4, 7, 6]), nope_c, xb)
                    mag = xb.bitcast(fx.Int32) & 0x7FFF7FFF
                    am = (
                        fx.Int32(mag.bitcast(fx.Int16).reduce(ReductionOp.MAX)) << 16
                    ).bitcast(fx.Float32)
                else:
                    y = rotate4(xb.to(fx.Float32), one4, zero4)
                    am = fmath.absf(y).reduce(ReductionOp.MAX)
                code = group_code(am, 4)
                out.append((to_fp8(y, code), code))
            return out

        def store_codes(r, base, res):
            """A row's codes, one byte per group from its first lane."""
            if (tid & 3) == 0:
                for k in range_constexpr(CHUNKS):
                    buffer_ops.buffer_store(
                        res[k][1].to(fx.Int8),
                        r,
                        base + k * QUADS + quad,
                        offset_is_bytes=True,
                    )

        def store_values(r, base, res):
            for k in range_constexpr(CHUNKS):
                buffer_ops.buffer_store(
                    fx.Vector.from_elements(res[k][0], dtype=fx.Int32).ir_value(),
                    r,
                    base + k * CW + tid * 8,
                    offset_is_bytes=True,
                )

        def store_dequant(r, res):
            for k in range_constexpr(CHUNKS):
                buffer_ops.buffer_store(
                    dequant(*res[k]).ir_value(),
                    r,
                    (k * CW + tid * 8) * 2,
                    offset_is_bytes=True,
                )

        def emit_query_rows():
            row0 = g * ROWS_WG
            in_rsrc = rsrc(addr(query) + fx.Int64(row0) * (D * 2))
            out_rsrc = rsrc(addr(query_values) + fx.Int64(row0) * D)
            sc_rsrc = rsrc(addr(query_scales) + fx.Int64(row0) * NG)
            wave_lds = wv * (R * NG)
            last = num_rows - 1
            rels = [fx.min(row0 + wv * R + j, last) - row0 for j in range(R)]
            # Positions, then the rows, then cos / sin: loads return in order,
            # so the cos / sin loads wait on the positions alone. Left free,
            # gfx1250's scheduler waits on the positions before the rows; on
            # gfx950 the pin costs more than it saves.
            ps = [
                position_of(token_of(row0 + rels[j]))
                for j in range(1 if SHARED_CS else R)
            ]
            xs = [
                [
                    fx.Vector(
                        buffer_ops.buffer_load(
                            in_rsrc,
                            rels[j] * D + c * CW + tid * 8,
                            vec_width=8,
                            dtype=T.bf16,
                        )
                    )
                    for c in range(CHUNKS)
                ]
                for j in range(R)
            ]
            if const_expr(wave == 32):
                fx.rocdl.sched_barrier(0)
            css = [slice_cs(p) for p in ps]
            for j in range_constexpr(R):
                res = quant_slices(xs[j], *css[0 if SHARED_CS else j])
                store_values(out_rsrc, rels[j] * D, res)
                for k in range_constexpr(CHUNKS):
                    _lds_store_u8(
                        lds_i32 + wave_lds, j * NG + k * QUADS + quad, res[k][1]
                    )
            # The wave's R rows of codes, one dword a lane, in one store. LDS
            # ops of a wave retire in order, so no barrier is needed.
            word = fx.Int32(lds_load_b32(lds_idx, wave_lds + tid * 4)[0])
            rel = fx.min(row0 + wv * R + (tid >> 2), last) - row0
            if tid < WORDS:
                buffer_ops.buffer_store(
                    word, sc_rsrc, rel * NG + (tid & 3) * 4, offset_is_bytes=True
                )

        def emit_query_tdm():
            tile_base = g * CT
            wg_row0 = tile_base * W
            out_rsrc = rsrc(addr(query_values) + fx.Int64(wg_row0) * D)
            sc_rsrc = rsrc(addr(query_scales) + fx.Int64(wg_row0) * NG)
            is_rope = tid >= T_ROPE_LO
            rope_rel = fx.max(tid - T_ROPE_LO, fx.Int32(0))
            nr_m1 = num_rows - 1
            wave_lds = wv * (D * 2)
            one = fx.Vector.filled([TPAIRS], 1.0, fx.Float32)
            zero = fx.Vector.filled([TPAIRS], 0.0, fx.Float32)
            nope_c = fx.Vector.from_elements([-0.0, 0.0] * TPAIRS, dtype=fx.BFloat16)
            lds_bf16_ptr_ty = fx.PointerType.get(
                elem_ty=fx.BFloat16.ir_type,
                address_space=fx.AddressSpace.Shared,
                alignment=16,
            )
            q_full = fx.Tensor(
                fx.make_view(fx.get_iter(q_in), fx.make_layout((num_rows, D), (D, 1)))
            )
            q_tile = (1, D)
            tdm_atom, q_coord = fx.rocdl.cdna5.make_tiled_tdm_atom(
                fx.rocdl.TensorLoad(),
                q_full,
                fx.make_layout(q_tile, (D, 1)),
                q_tile,
                num_warps=1,
            )
            q_blk = fx.zipped_divide(q_coord, q_tile)[None, (None, 0)]

            def row_of(t):
                return fx.min(t * W + wv, nr_m1)

            def issue(buf, t):
                dst = fx.Tensor(
                    fx.make_view(
                        fx.inttoptr(lds_bf16_ptr_ty, lds_i32 + wave_lds + buf),
                        fx.make_layout((q_tile, 1), ((D, 1), D)),
                    )
                )
                ts, tg = fx.rocdl.cdna5.tdm_partition(
                    tdm_atom, 0, fx.make_layout(1, 1), dst, q_blk
                )
                fx.copy(tdm_atom, tg[None, row_of(t)], ts[None, 0])

            def load_cs(t):
                off = position_of(token_of(row_of(t))) * (RD // 2) + rope_rel * TPAIRS
                c = fx.Vector(
                    buffer_ops.buffer_load(
                        cos_rsrc, off, vec_width=TPAIRS, dtype=T.bf16
                    )
                )
                s = fx.Vector(
                    buffer_ops.buffer_load(
                        sin_rsrc, off, vec_width=TPAIRS, dtype=T.bf16
                    )
                )
                return is_rope.select(c.to(fx.Float32), one), is_rope.select(
                    s.to(fx.Float32), zero
                )

            def compute_store(buf, t, c, s):
                parts = [
                    fx.Vector(
                        lds_load_b128(
                            lds_idx, buf + wave_lds + tid * (VEC * 2) + h * 16
                        )
                    ).bitcast(fx.BFloat16)
                    for h in range(2)
                ]
                xb = parts[0].shuffle(parts[1], list(range(VEC)))
                y_nope = fmath.fma(
                    xb.shuffle(xb, [i ^ 1 for i in range(VEC)]), nope_c, xb
                )
                mag = xb.bitcast(fx.Int32) & 0x7FFF7FFF
                am_nope = (
                    fx.Int32(mag.bitcast(fx.Int16).reduce(ReductionOp.MAX)) << 16
                ).bitcast(fx.Float32)
                x = xb.to(fx.Float32)
                ev = x.shuffle(x, EV)
                od = x.shuffle(x, OD)
                v = (ev * c - od * s).shuffle(ev * s + od * c, ILV)
                am_rope = (
                    fmath.absf(v).reduce(ReductionOp.MAX).to(fx.BFloat16).to(fx.Float32)
                )
                y = is_rope.select(v.to(fx.BFloat16), y_nope)
                code = group_code(is_rope.select(am_rope, am_nope), 2)
                dw = []
                for h in range_constexpr(2):
                    dw += to_fp8(y.shuffle(y, list(range(8 * h, 8 * h + 8))), code)
                rel = row_of(t) - wg_row0
                buffer_ops.buffer_store(
                    fx.Vector.from_elements(dw, dtype=fx.Int32).ir_value(),
                    out_rsrc,
                    rel * D + tid * VEC,
                    offset_is_bytes=True,
                )
                # The row's 16 codes as 4 dwords held by lanes 0, 8, 16, 24.
                pair = code | (code.shuffle_xor(2, wave) << 8)
                word = pair | (pair.shuffle_xor(4, wave) << 16)
                if (tid & 7) == 0:
                    buffer_ops.buffer_store(
                        word, sc_rsrc, rel * NG + (tid >> 1), offset_is_bytes=True
                    )

            bufs = [k * TILE_BYTES for k in range(K)]
            for k in range_constexpr(K):
                issue(bufs[k], tile_base + k)
            cs = load_cs(tile_base)
            for i in range_constexpr(CT):
                cur = cs
                if const_expr(i + 1 < CT and ((not HOIST) or (i + 1) % TGROUP == 0)):
                    cs = load_cs(tile_base + i + 1)
                # Tile i's load is retired once at most min(K - 1, CT - 1 - i)
                # of the later ones are still in flight.
                tdm_ops.tensor_wait(min(K - 1, CT - 1 - i))
                fx.rocdl.s_wait_dscnt(0)
                compute_store(bufs[i % K], tile_base + i, *cur)
                if const_expr(i + K < CT):
                    issue(bufs[i % K], tile_base + i + K)

        def emit_query_rotate():
            row0 = g * (ROT_ROWS_WG * R)
            last = num_rows - 1 - row0
            # Sized to this WG's live rows: a tail lane's store falls past the
            # end and the hardware drops it, with no branch around the loads.
            q_rsrc = buffer_ops.create_buffer_resource_from_addr(
                addr(query) + fx.Int64(row0) * (D * 2),
                num_records_bytes=fx.Int64(fx.min(last + 1, ROT_ROWS_WG * R) * (D * 2)),
            )
            lane = tid % ROT_LANES
            rels = [
                i * ROT_ROWS_WG + wv * (wave // ROT_LANES) + tid // ROT_LANES
                for i in range(R)
            ]
            # Every load first, so R rows a lane are in flight at once. Loads
            # return in order: the positions go first, so the cos / sin loads
            # wait on them alone while the query rows are still in flight.
            clamped = [fx.min(rels[i], last) for i in range(R)]
            ps = [position_of(token_of(row0 + rel)) for rel in clamped]
            xs = [
                fx.Vector(
                    buffer_ops.buffer_load(
                        q_rsrc, rel * D + NOPE + lane * 8, vec_width=8, dtype=T.bf16
                    )
                )
                for rel in clamped
            ]
            fx.rocdl.sched_barrier(0)
            loaded = []
            for i in range_constexpr(R):
                off = ps[i] * (RD // 2) + lane * 4
                c = fx.Vector(
                    buffer_ops.buffer_load(cos_rsrc, off, vec_width=4, dtype=T.bf16)
                )
                s = fx.Vector(
                    buffer_ops.buffer_load(sin_rsrc, off, vec_width=4, dtype=T.bf16)
                )
                loaded.append((xs[i], c, s))
            fx.rocdl.sched_barrier(0)
            for i in range_constexpr(R):
                x, c, s = loaded[i]
                y = rotate4(x.to(fx.Float32), c.to(fx.Float32), s.to(fx.Float32))
                # The rotation is in place: a tail lane clamped onto the last
                # row stores at its own row, out of range, or that row would be
                # rotated twice.
                buffer_ops.buffer_store(
                    y.to(fx.BFloat16).bitcast(fx.Int32).ir_value(),
                    q_rsrc,
                    (rels[i] * D + NOPE + lane * 8) * 2,
                    offset_is_bytes=True,
                )

        def emit_kv():
            # One token per wave; uniform, so every row base below is scalar.
            tok_v = fx.min((g - gx_q) * W + wv, num_tokens - 1)
            tok = fx.Int32(fx.rocdl.readfirstlane(T.i32, tok_v.ir_value()))
            kv_rsrc = rsrc(addr(kv) + fx.Int64(tok) * fx.Int64(kv_stride) * 2)
            xs = [
                fx.Vector(
                    buffer_ops.buffer_load(
                        kv_rsrc, c * CW + tid * 8, vec_width=8, dtype=T.bf16
                    )
                )
                for c in range(CHUNKS)
            ]
            p = position_of(tok)
            res = quant_slices(xs, *slice_cs(p))
            if const_expr(qat):
                store_dequant(rsrc(addr(qat_out) + fx.Int64(tok) * (D * 2)), res)
            if const_expr(packed):
                store_values(
                    rsrc(addr(values) + fx.Int64(tok) * fx.Int64(values_stride)), 0, res
                )
                store_codes(
                    rsrc(addr(scales) + fx.Int64(tok) * fx.Int64(scales_stride)), 0, res
                )
            if const_expr(ring):
                # A padding row owns no request, and a row the ring has
                # scrolled past has nowhere to land.
                batch = load_index(rsrc(addr(batches)), tok, batch64)
                live = fx.max(batch, fx.Int32(0))
                end = load_index(rsrc(addr(cu)), live + 1, cu64)
                slot = load_index(rsrc(addr(slots)), live, slot64)
                if (batch >= 0) & (tok >= end - RING_SLOTS):
                    ring_pos = p % RING_SLOTS
                    row = (
                        fx.Int64(slot) * SLOT_ROWS
                        + ring_start
                        + fx.Int64(
                            (ring_pos // RING_STRIDE) * RUN_ROWS
                            + ring_pos % RING_STRIDE
                        )
                    )
                    data_rsrc = rsrc(addr(pool) + row * fx.Int64(pool_stride))
                    if const_expr(ring_fp8):
                        store_values(data_rsrc, 0, res)
                        store_codes(
                            rsrc(addr(pool_scale) + row * fx.Int64(pool_scale_stride)),
                            0,
                            res,
                        )
                    else:
                        store_dequant(data_rsrc, res)

        lds_load_b32, _ = make_lds_copy_ops(32)
        lds_load_b128, _ = make_lds_copy_ops(128)

        def emit_query():
            if const_expr(query_mode == QUERY_FP8 and tdm):
                emit_query_tdm()
            elif const_expr(query_mode == QUERY_FP8):
                emit_query_rows()
            elif const_expr(query_mode == QUERY_ROTATE):
                emit_query_rotate()

        # Without a KV half every workgroup is a query one, and the branch
        # would cost a kernarg load ahead of all the others.
        if const_expr(not (qat or packed or ring)):
            emit_query()
        else:
            if g < gx_q:
                emit_query()
            else:
                emit_kv()

    @flyc.jit
    def launch(
        q_in: fx.Tensor,
        num_rows: fx.Int32,
        num_tokens: fx.Int32,
        gx_q: fx.Int32,
        query: fx.Pointer,
        kv: fx.Pointer,
        cos: fx.Pointer,
        sin: fx.Pointer,
        positions: fx.Pointer,
        qat_out: fx.Pointer,
        values: fx.Pointer,
        scales: fx.Pointer,
        query_values: fx.Pointer,
        query_scales: fx.Pointer,
        pool: fx.Pointer,
        pool_scale: fx.Pointer,
        batches: fx.Pointer,
        cu: fx.Pointer,
        slots: fx.Pointer,
        kv_stride: fx.Int32,
        values_stride: fx.Int32,
        scales_stride: fx.Int32,
        pool_stride: fx.Int32,
        pool_scale_stride: fx.Int32,
        ring_start: fx.Int64,
        num_wg: fx.Int32,
        stream: fx.Stream,
    ):
        k = kernel(
            q_in,
            num_rows,
            num_tokens,
            gx_q,
            query,
            kv,
            cos,
            sin,
            positions,
            qat_out,
            values,
            scales,
            query_values,
            query_scales,
            pool,
            pool_scale,
            batches,
            cu,
            slots,
            kv_stride,
            values_stride,
            scales_stride,
            pool_stride,
            pool_scale_stride,
            ring_start,
        )
        k.launch(grid=(fx.Int64(num_wg), 1, 1), block=(wave, W, 1), stream=stream)

    # The leading pointers arrive in SGPRs with the dispatch, not by a scalar load.
    launch.compile_hints = {"llvm_options": {"amdgpu-kernarg-preload-count": 16}}
    if tdm:
        launch.compile_hints["waves_per_eu"] = 8
    return launch


_DUMMY = {}


def _dummy_tensor(device):
    """A placeholder for the TDM operand when the query takes another path."""
    if device not in _DUMMY:
        t = torch.zeros(1, 16, device=device, dtype=torch.bfloat16)
        _DUMMY[device] = (t, flyc.from_dlpack(t))
    return _DUMMY[device][1]


@cache
def _arch():
    from aiter.jit.utils.chip_info import get_gfx

    return get_gfx()


def rope_quant_window(
    query,
    kv,
    cos,
    sin,
    positions,
    *,
    rope_dim,
    qat=None,
    values=None,
    scales=None,
    ring=None,
    query_fp8=None,
):
    """Drop-in for ATOM's ``rope_quant_window``; same arguments, same bytes.

    ``ring`` is ``(pool, step, window)`` with ``step.batch_ids``,
    ``step.cu_seqlens_q`` and ``step.slots``, and ``window`` the pool's
    ``ring_start``, ``ring_slots``, ``slot_rows``, ``ring_stride`` and
    ``run_rows``, all in rows. ``pool`` is either ``(data, scale)``, an FP8
    ``[rows, head_dim]`` plane and an E8M0 ``[rows, head_dim / 32]`` plane, or a
    ``[rows, head_dim]`` BF16 tensor; each plane keeps its own row stride.
    """
    tokens, heads, dim = query.shape
    kv_half = ring is not None or qat is not None or values is not None
    if tokens == 0 or (heads == 0 and not kv_half):
        return
    arch = _arch()
    wave = 32 if arch == "gfx1250" else 64
    pk8 = arch == "gfx1250"
    query_values, query_scales = query_fp8 or (None, None)
    query_mode = (
        QUERY_NONE
        if heads == 0
        else QUERY_FP8 if query_fp8 is not None else QUERY_ROTATE
    )
    rows = tokens * heads
    tdm = (
        pk8
        and query_mode == QUERY_FP8
        and rows >= TDM_MIN_ROWS
        and query.is_contiguous()
    )
    if tdm:
        K, CT, W = TDM_DENSE if rows >= TDM_DENSE_ROWS else TDM_SPARSE
        R = 1
        gx_q = -(-rows // (CT * W))
    else:
        K, CT = 1, 1
        W, R = ROWS_SMALL.get(arch, (4, 2)) if rows < ROWS_SMALL_MAX else ROWS_LARGE
        if query_mode == QUERY_FP8:
            gx_q = -(-rows // (W * R))
        elif query_mode == QUERY_ROTATE:
            W, R = next(
                shape
                for bound, shape in ROTATE_SHAPES.get(arch, ROTATE_SHAPES["gfx950"])
                if bound is None or rows < bound
            )
            gx_q = -(-rows // (W * R * wave // (rope_dim // 8)))
        else:
            gx_q = 0
    gx_kv = -(-tokens // W) if kv_half else 0

    pool, step, window = ring or (None, None, None)
    ring_fp8 = isinstance(pool, tuple)
    pool, pool_scale = pool if ring_fp8 else (pool, None)
    if pool is not None:
        if ring_fp8:
            ok = (
                pool.element_size() == 1
                and pool_scale.element_size() == 1
                and pool_scale.dim() == 2
                and pool_scale.shape == (pool.shape[0], dim // 32)
                and pool_scale.stride(1) == 1
            )
        else:
            ok = pool.dtype == torch.bfloat16
        if not (
            ok
            and pool.dim() == 2
            and pool.shape[1] == dim
            and pool.stride(1) == 1
            and pool.stride(0) * pool.element_size() % 4 == 0
            and pool.data_ptr() % 4 == 0
        ):
            raise ValueError(
                "ring pool must be (fp8 [rows, D], e8m0 [rows, D / 32]) planes or a "
                "bf16 [rows, D] tensor, with dword-aligned data rows"
            )
    index = (
        (None, None, None)
        if step is None
        else (step.batch_ids, step.cu_seqlens_q, step.slots)
    )
    launcher = _build(
        H=heads,
        D=dim,
        RD=rope_dim,
        wave=wave,
        pk8=pk8,
        query_mode=query_mode,
        W=W,
        R=R,
        tdm=tdm,
        K=K,
        CT=CT,
        pos64=positions.dtype == torch.int64,
        qat=qat is not None,
        packed=values is not None,
        ring=ring is not None,
        ring_fp8=ring_fp8,
        batch64=index[0] is not None and index[0].dtype == torch.int64,
        cu64=index[1] is not None and index[1].dtype == torch.int64,
        slot64=index[2] is not None and index[2].dtype == torch.int64,
        RING_SLOTS=1 if window is None else int(window.ring_slots),
        SLOT_ROWS=0 if window is None else int(window.slot_rows),
        RING_STRIDE=1 if window is None else int(window.ring_stride),
        RUN_ROWS=0 if window is None else int(window.run_rows),
    )

    def ptr(t):
        t = query if t is None else t
        return flyc.from_c_void_p(fx.Uint8, t.data_ptr())

    _run_compiled(
        launcher,
        flyc.from_dlpack(query.view(rows, dim)) if tdm else _dummy_tensor(query.device),
        rows,
        tokens,
        gx_q,
        ptr(query),
        ptr(kv if kv_half else None),
        ptr(cos),
        ptr(sin),
        ptr(positions),
        ptr(qat),
        ptr(values),
        ptr(scales),
        ptr(query_values),
        ptr(query_scales),
        ptr(pool),
        ptr(pool_scale),
        ptr(index[0]),
        ptr(index[1]),
        ptr(index[2]),
        kv.stride(0) if kv_half else 0,
        0 if values is None else values.stride(-2),
        0 if scales is None else scales.stride(-2),
        0 if pool is None else pool.stride(0) * pool.element_size(),
        0 if pool_scale is None else pool_scale.stride(0),
        0 if window is None else int(window.ring_start),
        gx_q + gx_kv,
        Stream(torch.cuda.current_stream()),
    )
