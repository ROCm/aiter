# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL helper kernels of the MXFP6 MoE (gfx950): MX row quantization, routing
(one-block and multi-block counting sort) and the deterministic combine. The launch
wrappers are in aiter.ops.flydsl.mxfp6_moe_kernels.

Row quantizer: bf16 [R, K] -> "A" rows (mxfp6_moe_quant storage, MFMA K order) and
[R, KQ, 4] int32 scales. One thread per 32-block, one wave per (row, 64 blocks); row
bases are 64-bit (rebased descriptors), so R * K may exceed 2^31 bytes. Numerics =
mxfp6_moe_quant.quant_mx (scale 2^ceil(log2(amax/max)), RNE onto the grid, clamp),
computed with integer bit arithmetic.

Routers: counting sort of the (token, slot) entries by expert into BM-row tiles.

Combine: y[t, :] = sum_{j<K} w[t, j] * o[pos[t, j], :] in fixed j order (fp32), bf16
out.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, range_constexpr, rocdl
from flydsl.expr.typing import T

from .tensor_shim import buf_copy_load, buf_copy_store, ptr_buf_tensor

# fmt: (max, mbits, min normal exponent, max code magnitude, bits)
QFMT = {
    "e2m3": (7.5, 3, 0, 31, 6),
    "e3m2": (28.0, 2, -2, 31, 6),
    "e4m3": (448.0, 3, -6, 126, 8),
}


def _i64(p):
    return fx.Int64(fx.ptrtoint(p))


def _f2i(v):
    return fx.Float32(v).bitcast(fx.Int32)


def _i2f(v):
    return fx.Int32(v).bitcast(fx.Float32)


def mx_quant_store(vals, fmt, y_base, y_row_off, kb, s_base, s_row_off, KQ):
    """Quantize one 32-element block (vals: 32 Float32, already bf16-exact) of a row to
    MX `fmt` and store it in the "A" layout of mxfp6_moe_quant: codes at byte y_row_off
    + (block kb position) of the i64 address y_base, E8M0 byte into the [KQ, 4] int32
    scale row at byte s_row_off of s_base. Numerics = mxfp6_moe_quant.quant_mx: scale
    2^ceil(log2(amax/max)), RNE onto the grid, clamp (bit arithmetic)."""
    vmax, mbits, minexp, cmax, bits = QFMT[fmt]
    amax = fx.Float32(0.0)
    for v in vals:
        amax = fx.max(amax, fx.absf(v))
    # e = ceil(log2(amax / vmax)), clamped to [-126, 127]; amax == 0 -> scale byte 0
    tb = _f2i(amax / fx.Float32(vmax))
    ex = (tb >> fx.Int32(23)) - fx.Int32(127)
    ex = ((tb & fx.Int32(0x7FFFFF)) != fx.Int32(0)).select(ex + fx.Int32(1), ex)
    ex = fx.max(ex, fx.Int32(-126))
    ex = fx.min(ex, fx.Int32(127))
    sbyte = (amax > fx.Float32(0.0)).select(ex + fx.Int32(127), fx.Int32(0))
    inv = _i2f((fx.Int32(127) - ex) << fx.Int32(23))  # 2^-e
    codes = []
    for v in vals:
        u = fx.absf(v) * inv
        ue = (_f2i(u) >> fx.Int32(23)) - fx.Int32(127)
        ee = fx.max(ue, fx.Int32(minexp))
        inv_step = _i2f((fx.Int32(127 + mbits) - ee) << fx.Int32(23))
        qv = fx.Int32(fx.roundeven(u * inv_step))
        code = (ee - fx.Int32(minexp)) * fx.Int32(1 << mbits) + qv
        code = fx.min(code, fx.Int32(cmax))
        sign = (v < fx.Float32(0.0)).select(fx.Int32(1 << (bits - 1)), fx.Int32(0))
        codes.append(code | sign)
    if const_expr(bits == 6):
        words = []
        for wi in range_constexpr(6):
            acc = fx.Int32(0)
            for j in range_constexpr(32):
                lo = 6 * j
                if lo // 32 == wi:
                    acc = acc | (codes[j] << fx.Int32(lo % 32))
                elif (lo + 5) // 32 == wi:
                    acc = acc | fx.Int32(fx.Uint32(codes[j]) >> fx.Uint32(32 - lo % 32))
            words.append(acc)
        base = y_row_off // fx.Int32(4) + kb * fx.Int32(6)  # dwords
        y4 = ptr_buf_tensor(y_base, fx.Int32, unit_elems=4, unit_stride=1)
        y2 = ptr_buf_tensor(y_base, fx.Int32, unit_elems=2, unit_stride=1)
        buf_copy_store(
            y4, base, fx.Vector.from_elements(words[:4], fx.Int32), fx.Int32, 4
        )
        buf_copy_store(
            y2,
            base + fx.Int32(4),
            fx.Vector.from_elements(words[4:], fx.Int32),
            fx.Int32,
            2,
        )
    else:
        words = []
        for wi in range_constexpr(8):
            acc = fx.Int32(0)
            for bi in range_constexpr(4):
                acc = acc | ((codes[4 * wi + bi] & fx.Int32(0xFF)) << fx.Int32(8 * bi))
            words.append(acc)
        s = kb // fx.Int32(4)
        b = kb % fx.Int32(4)
        p0 = (
            y_row_off
            + s * fx.Int32(128)
            + (b % fx.Int32(2)) * fx.Int32(64)
            + (b // fx.Int32(2)) * fx.Int32(16)
        )
        y4 = ptr_buf_tensor(y_base, fx.Int32, unit_elems=4, unit_stride=1)
        buf_copy_store(
            y4,
            p0 // fx.Int32(4),
            fx.Vector.from_elements(words[:4], fx.Int32),
            fx.Int32,
            4,
        )
        buf_copy_store(
            y4,
            (p0 + fx.Int32(32)) // fx.Int32(4),
            fx.Vector.from_elements(words[4:], fx.Int32),
            fx.Int32,
            4,
        )
    # scale byte j of dword (q, b): step s = 4q + j
    s = kb // fx.Int32(4)
    b = kb % fx.Int32(4)
    # [R][KQ][4 b][4 j]
    boff = (
        s_row_off
        + ((s // fx.Int32(4)) * fx.Int32(4) + b) * fx.Int32(4)
        + s % fx.Int32(4)
    )
    buf_copy_store(ptr_buf_tensor(s_base, fx.Int8), boff, fx.Int8(sbyte), fx.Int8, 1)


def compile_mxfp6_moe_quant_rows(K: int, fmt: str):
    """Row quantizer of K-wide bf16 rows to MX `fmt`: launch(X, Y, S, rows, stream)
    writes the "A" rows Y and their [KQ, 4] int32 scales S (mx_quant_store)."""
    bits = QFMT[fmt][4]
    NB = K // 32
    KS = K // 128
    KQ = (KS + 3) // 4
    out_row_bytes = K * bits // 8
    WPB = 64  # threads per block
    name = f"mxfp6_moe_quant_rows_{fmt}_k{K}"

    @flyc.kernel(name=name, known_block_size=[WPB, 1, 1])
    def qk(X: fx.Pointer, Y: fx.Pointer, S: fx.Pointer):
        tid = fx.thread_idx.x
        row = fx.block_idx.y
        kb = fx.block_idx.x * fx.Int32(WPB) + tid
        if kb < fx.Int32(NB):
            # row bases folded into 64-bit addresses: per-row offsets stay small
            # whatever R * K is
            xt = ptr_buf_tensor(
                _i64(X) + fx.Int64(row) * fx.Int64(K * 2),
                fx.Int32,
                unit_elems=4,
                unit_stride=1,
            )
            y_base = _i64(Y) + fx.Int64(row) * fx.Int64(out_row_bytes)
            s_base = _i64(S) + fx.Int64(row) * fx.Int64(4 * KQ * 4)
            vals = []
            for c in range_constexpr(4):
                v = buf_copy_load(xt, kb * fx.Int32(16) + fx.Int32(4 * c), fx.Int32, 4)
                for i in range_constexpr(4):
                    wd = fx.Int32(v[i])
                    vals.append(_i2f(wd << fx.Int32(16)))
                    vals.append(_i2f(wd & fx.Int32(-65536)))
            mx_quant_store(vals, fmt, y_base, fx.Int32(0), kb, s_base, fx.Int32(0), KQ)

    @flyc.jit
    def launch(
        X: fx.Pointer, Y: fx.Pointer, S: fx.Pointer, rows: fx.Int32, stream: fx.Stream
    ):
        qk(X, Y, S).launch(
            grid=((NB + WPB - 1) // WPB, rows, 1), block=(WPB, 1, 1), stream=stream
        )

    return launch


def compile_mxfp6_moe_combine(H: int, K: int):
    """Combine of K routed rows per token: launch(O, POS, W, Y, tokens, stream) writes
    y[t, :] = sum_{j<K} w[t, j] * o[pos[t, j], :], H bf16 columns, summed in fp32 in
    fixed j order (no atomics: deterministic)."""
    if H % 8:
        raise ValueError(f"mxfp6_moe_combine: H {H} must be a multiple of 8")
    TPB = 256
    CH = TPB * 8
    name = f"mxfp6_moe_combine_h{H}_k{K}"

    @flyc.kernel(name=name, known_block_size=[TPB, 1, 1])
    def ck(O: fx.Pointer, POS: fx.Pointer, W: fx.Pointer, Y: fx.Pointer):
        tid = fx.thread_idx.x
        t = fx.block_idx.y
        c0 = fx.block_idx.x * fx.Int32(CH) + tid * fx.Int32(8)
        if c0 < fx.Int32(H):
            i32pt = fx.PointerType.get(
                T.i32, address_space=fx.AddressSpace.Global, alignment=4
            )
            f32pt = fx.PointerType.get(
                T.f32, address_space=fx.AddressSpace.Global, alignment=4
            )
            posp = fx.inttoptr(i32pt, _i64(POS))
            wp = fx.inttoptr(f32pt, _i64(W))
            acc = [fx.Float32(0.0) for _ in range(8)]
            for j in range_constexpr(K):
                # rocdl.readfirstlane: pos[t, j] is the same in every lane (t is the
                # block), and the buffer descriptor based on its row must be
                # wave-uniform. flydsl.expr has no fx-level form of it
                p = rocdl.readfirstlane(
                    T.i32, fx.Int32(posp[t * fx.Int32(K) + fx.Int32(j)]).ir_value()
                )
                wj = wp[t * fx.Int32(K) + fx.Int32(j)]
                ot = ptr_buf_tensor(
                    _i64(O) + fx.Int64(fx.Int32(p)) * fx.Int64(H * 2),
                    fx.Int32,
                    unit_elems=4,
                    unit_stride=1,
                )
                v = buf_copy_load(ot, c0 // fx.Int32(2), fx.Int32, 4)
                for i in range_constexpr(4):
                    wd = fx.Int32(v[i])
                    acc[2 * i] = acc[2 * i] + wj * _i2f(wd << fx.Int32(16))
                    acc[2 * i + 1] = acc[2 * i + 1] + wj * _i2f(wd & fx.Int32(-65536))
            out = fx.Vector.from_elements(acc, fx.Float32).to(fx.BFloat16)
            yt = ptr_buf_tensor(
                _i64(Y) + fx.Int64(t) * fx.Int64(H * 2),
                fx.BFloat16,
                unit_elems=8,
                unit_stride=1,
            )
            buf_copy_store(yt, c0, out, fx.BFloat16, 8)

    @flyc.jit
    def launch(
        O: fx.Pointer,
        POS: fx.Pointer,
        W: fx.Pointer,
        Y: fx.Pointer,
        T_: fx.Int32,
        stream: fx.Stream,
    ):
        ck(O, POS, W, Y).launch(
            grid=((H + CH - 1) // CH, T_, 1), block=(TPB, 1, 1), stream=stream
        )

    return launch


def compile_mxfp6_moe_route(E: int, K: int, BM: int):
    """Routing in one kernel (one 1024-thread block): counting sort by expert with LDS
    atomics. The slot of a token inside its expert block is arbitrary, but every output
    row depends only on its own input row, so the result is bitwise independent of that
    order (deterministic output).

    Outputs: RIDX [n_tiles*BM] (pad slots -> row 0), POS [TK], TEXP [n_tiles + 1] (tile
    -> expert, -1 for unused, TEXP[n_tiles] = number of used tiles). RANK [TK] is
    scratch."""
    NTH = 1024
    if E > NTH:
        raise ValueError(f"mxfp6_moe_route: {E} experts, at most {NTH}")
    name = f"mxfp6_moe_route_e{E}_k{K}_bm{BM}"

    @fx.struct
    class SS:
        cnt: fx.Array[fx.Int32, E, 4]
        tst: fx.Array[fx.Int32, E + 1, 4]

    @flyc.kernel(name=name, known_block_size=[NTH, 1, 1])
    def rk(
        IDS: fx.Pointer,
        RIDX: fx.Pointer,
        POS: fx.Pointer,
        TEXP: fx.Pointer,
        RANK: fx.Pointer,
        TK: fx.Int32,
        NTILES: fx.Int32,
    ):
        ss = fx.SharedAllocator().allocate(SS).peek()
        cnt = fx.recast_iter(fx.Int32, ss.cnt.ptr)
        tst = fx.recast_iter(fx.Int32, ss.tst.ptr)
        tid = fx.thread_idx.x
        i32pt = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        ids = fx.inttoptr(i32pt, _i64(IDS))
        ridx = fx.inttoptr(i32pt, _i64(RIDX))
        pos = fx.inttoptr(i32pt, _i64(POS))
        texp = fx.inttoptr(i32pt, _i64(TEXP))
        rank = fx.inttoptr(i32pt, _i64(RANK))
        if tid < fx.Int32(E):
            fx.ptr_store(fx.Int32(0), fx.add_offset(cnt, tid))
        fx.barrier()
        for i in range(tid, TK, fx.Int32(NTH)):
            e = ids[i]
            rank[i] = fx.atomic_add(
                fx.add_offset(cnt, e),
                fx.Int32(1),
                syncscope=fx.rocdl.SyncScope.Workgroup,
            )
        fx.barrier()
        if tid == fx.Int32(0):
            acc = fx.Int32(0)
            for e in range_constexpr(E):
                fx.ptr_store(acc, fx.add_offset(tst, fx.Int32(e)))
                c = fx.ptr_load(fx.add_offset(cnt, fx.Int32(e)))
                acc = acc + (c + fx.Int32(BM - 1)) // fx.Int32(BM)
            fx.ptr_store(acc, fx.add_offset(tst, fx.Int32(E)))
            texp[NTILES] = acc
        fx.barrier()
        total = fx.ptr_load(fx.add_offset(tst, fx.Int32(E)))
        if tid < fx.Int32(E):
            t0 = fx.ptr_load(fx.add_offset(tst, tid))
            t1 = fx.ptr_load(fx.add_offset(tst, tid + fx.Int32(1)))
            for t in range(t0, t1, fx.Int32(1)):
                texp[t] = tid
            c = fx.ptr_load(fx.add_offset(cnt, tid))
            for sl in range(t0 * fx.Int32(BM) + c, t1 * fx.Int32(BM), fx.Int32(1)):
                ridx[sl] = fx.Int32(0)
        for t in range(total + tid, NTILES, fx.Int32(NTH)):
            texp[t] = fx.Int32(-1)
        for i in range(tid, TK, fx.Int32(NTH)):
            e = ids[i]
            dest = fx.ptr_load(fx.add_offset(tst, e)) * fx.Int32(BM) + rank[i]
            ridx[dest] = i // fx.Int32(K)
            pos[i] = dest

    @flyc.jit
    def launch(
        IDS: fx.Pointer,
        RIDX: fx.Pointer,
        POS: fx.Pointer,
        TEXP: fx.Pointer,
        RANK: fx.Pointer,
        TK: fx.Int32,
        NTILES: fx.Int32,
        stream: fx.Stream,
    ):
        rk(IDS, RIDX, POS, TEXP, RANK, TK, NTILES).launch(
            grid=(1, 1, 1), block=(NTH, 1, 1), stream=stream
        )

    return launch


# multi-block router: entries per block, threads per block
RCH = 4096
RNT = 256


def compile_mxfp6_moe_route_mb(E: int, K: int, BM: int):
    """Multi-block router, three kernels in one launch: (1) per-block LDS histograms ->
    CNT[nb, E]; (2) one block: per-expert scan over blocks (BOFF), expert tile starts,
    tile -> expert map, pad slots, used-tile count, TST; (3) per-block scatter: dest =
    TST[e]*BM + BOFF[b, e] + LDS-atomic rank. Same outputs and contract as
    compile_mxfp6_moe_route; the slot order inside an expert is arbitrary (output bits
    do not depend on it)."""
    if E > 1024:
        raise ValueError(f"mxfp6_moe_route_mb: {E} experts, at most 1024")

    @fx.struct
    class SH:
        cnt: fx.Array[fx.Int32, E, 4]

    @flyc.kernel(name=f"mxfp6_moe_route_hist_e{E}", known_block_size=[RNT, 1, 1])
    def hist(IDS: fx.Pointer, CNT: fx.Pointer, TK: fx.Int32):
        cnt = fx.recast_iter(fx.Int32, fx.SharedAllocator().allocate(SH).peek().cnt.ptr)
        tid = fx.thread_idx.x
        b = fx.block_idx.x
        i32pt = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        ids = fx.inttoptr(i32pt, _i64(IDS))
        cg = fx.inttoptr(i32pt, _i64(CNT))
        for e in range(tid, fx.Int32(E), fx.Int32(RNT)):
            fx.ptr_store(fx.Int32(0), fx.add_offset(cnt, e))
        fx.barrier()
        i0 = b * fx.Int32(RCH)
        i1 = i0 + fx.Int32(RCH)
        i1 = (i1 < TK).select(i1, TK)
        for i in range(i0 + tid, i1, fx.Int32(RNT)):
            fx.atomic_add(
                fx.add_offset(cnt, ids[i]),
                fx.Int32(1),
                syncscope=fx.rocdl.SyncScope.Workgroup,
            )
        fx.barrier()
        for e in range(tid, fx.Int32(E), fx.Int32(RNT)):
            cg[b * fx.Int32(E) + e] = fx.ptr_load(fx.add_offset(cnt, e))

    @fx.struct
    class SS:
        tst: fx.Array[fx.Int32, E + 1, 4]
        tot: fx.Array[fx.Int32, E, 4]

    @flyc.kernel(
        name=f"mxfp6_moe_route_scan_e{E}_bm{BM}", known_block_size=[1024, 1, 1]
    )
    def scan(
        CNT: fx.Pointer,
        BOFF: fx.Pointer,
        TST: fx.Pointer,
        RIDX: fx.Pointer,
        TEXP: fx.Pointer,
        NB: fx.Int32,
        NTILES: fx.Int32,
    ):
        sp = fx.SharedAllocator().allocate(SS).peek()
        tst = fx.recast_iter(fx.Int32, sp.tst.ptr)
        tot = fx.recast_iter(fx.Int32, sp.tot.ptr)
        tid = fx.thread_idx.x
        i32pt = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        cg = fx.inttoptr(i32pt, _i64(CNT))
        bo_ = fx.inttoptr(i32pt, _i64(BOFF))
        tg = fx.inttoptr(i32pt, _i64(TST))
        ridx = fx.inttoptr(i32pt, _i64(RIDX))
        texp = fx.inttoptr(i32pt, _i64(TEXP))
        if tid < fx.Int32(E):
            run = fx.Int32(0)
            for b, st in range(fx.Int32(0), NB, fx.Int32(1), init=[run]):
                acc = st[0]
                bo_[fx.Int32(b) * fx.Int32(E) + tid] = acc
                res = yield [acc + cg[fx.Int32(b) * fx.Int32(E) + tid]]
            fx.ptr_store(
                res[0] if isinstance(res, (list, tuple)) else res,
                fx.add_offset(tot, tid),
            )
        fx.barrier()
        if tid == fx.Int32(0):
            acc = fx.Int32(0)
            for e in range_constexpr(E):
                fx.ptr_store(acc, fx.add_offset(tst, fx.Int32(e)))
                tg[fx.Int32(e)] = acc
                c = fx.ptr_load(fx.add_offset(tot, fx.Int32(e)))
                acc = acc + (c + fx.Int32(BM - 1)) // fx.Int32(BM)
            fx.ptr_store(acc, fx.add_offset(tst, fx.Int32(E)))
            tg[fx.Int32(E)] = acc
            texp[NTILES] = acc
        fx.barrier()
        total = fx.ptr_load(fx.add_offset(tst, fx.Int32(E)))
        if tid < fx.Int32(E):
            t0 = fx.ptr_load(fx.add_offset(tst, tid))
            t1 = fx.ptr_load(fx.add_offset(tst, tid + fx.Int32(1)))
            for t in range(t0, t1, fx.Int32(1)):
                texp[t] = tid
            c = fx.ptr_load(fx.add_offset(tot, tid))
            for sl in range(t0 * fx.Int32(BM) + c, t1 * fx.Int32(BM), fx.Int32(1)):
                ridx[sl] = fx.Int32(0)
        for t in range(total + tid, NTILES, fx.Int32(1024)):
            texp[t] = fx.Int32(-1)

    @flyc.kernel(
        name=f"mxfp6_moe_route_scatter_e{E}_k{K}_bm{BM}", known_block_size=[RNT, 1, 1]
    )
    def scatter(
        IDS: fx.Pointer,
        BOFF: fx.Pointer,
        TST: fx.Pointer,
        RIDX: fx.Pointer,
        POS: fx.Pointer,
        TK: fx.Int32,
    ):
        cnt = fx.recast_iter(fx.Int32, fx.SharedAllocator().allocate(SH).peek().cnt.ptr)
        tid = fx.thread_idx.x
        b = fx.block_idx.x
        i32pt = fx.PointerType.get(
            T.i32, address_space=fx.AddressSpace.Global, alignment=4
        )
        ids = fx.inttoptr(i32pt, _i64(IDS))
        bo_ = fx.inttoptr(i32pt, _i64(BOFF))
        tg = fx.inttoptr(i32pt, _i64(TST))
        ridx = fx.inttoptr(i32pt, _i64(RIDX))
        pos = fx.inttoptr(i32pt, _i64(POS))
        for e in range(tid, fx.Int32(E), fx.Int32(RNT)):
            fx.ptr_store(
                tg[e] * fx.Int32(BM) + bo_[b * fx.Int32(E) + e], fx.add_offset(cnt, e)
            )
        fx.barrier()
        i0 = b * fx.Int32(RCH)
        i1 = i0 + fx.Int32(RCH)
        i1 = (i1 < TK).select(i1, TK)
        for i in range(i0 + tid, i1, fx.Int32(RNT)):
            dest = fx.atomic_add(
                fx.add_offset(cnt, ids[i]),
                fx.Int32(1),
                syncscope=fx.rocdl.SyncScope.Workgroup,
            )
            ridx[dest] = i // fx.Int32(K)
            pos[i] = dest

    @flyc.jit
    def launch(
        IDS: fx.Pointer,
        CNT: fx.Pointer,
        BOFF: fx.Pointer,
        TST: fx.Pointer,
        RIDX: fx.Pointer,
        POS: fx.Pointer,
        TEXP: fx.Pointer,
        TK: fx.Int32,
        NB: fx.Int32,
        NTILES: fx.Int32,
        stream: fx.Stream,
    ):
        hist(IDS, CNT, TK).launch(grid=(NB, 1, 1), block=(RNT, 1, 1), stream=stream)
        scan(CNT, BOFF, TST, RIDX, TEXP, NB, NTILES).launch(
            grid=(1, 1, 1), block=(1024, 1, 1), stream=stream
        )
        scatter(IDS, BOFF, TST, RIDX, POS, TK).launch(
            grid=(NB, 1, 1), block=(RNT, 1, 1), stream=stream
        )

    return launch
