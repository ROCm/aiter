# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""The K loop of the a8w4 / a8w8 GEMM tiles, and the dispatch to its row variants.

W is streamed from HBM straight into VGPRs: fd2 mxfp4 (a8w4) is decoded to fp8 there,
fp8 (a8w8, aiter's (16, 16) preshuffle) is the MFMA operand as loaded; fp8 A goes
through LDS.
"""

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr.typing import Vector as Vec

from .cvt import _fp4x8_to_fp8x8_fast, _fp8_lut_predelta, _perm_u32
from .hw import _sched_fence
from .mem import (
    _global_i32_buffer_tiles,
    _global_i32_buffer_view,
)


def _dispatch_rows(rowv, pads, run_rows):
    """Run the smallest row variant whose next probe row is padding (nested ifs:
    a compile-time if followed by a runtime elif is unreliable in FlyDSL)."""

    def level(i):
        if const_expr(i == len(rowv) - 1):
            run_rows(rowv[-1])
            return

        @flyc.jit
        def _d():
            if pads[i]:
                run_rows(rowv[i])
            else:
                level(i + 1)

        _d()

    level(0)


def _frag(v_i32, elem, n):
    """MFMA operand rmem fragment of ``n`` ``elem`` from a vector of i32."""
    t = fx.make_rmem_tensor(fx.make_layout(n, 1), elem)
    t.store(Vec(v_i32).bitcast(elem))
    return t


@flyc.jit
def _masked_lds_store(atom, reg, dst, ok):
    if ok:
        fx.copy(atom, reg, dst)


def _make_core(
    *,
    K,
    W_ROWS,
    BM,
    PF,
    bcm,
    cols,
    lds_ptr,
    LDS_BYTES,
    lane,
    tx,
    NT,
    arg_w,
    arg_ws,
    arg_a,
    a_num_bytes,
    ARB=256,
    acm=2,
    ST_STEP=None,
    TR=False,
    WQ="fp4",
):
    """Trace-time builder of the streamed-W K loop.

    ``cols``: list of (key, n0, skey): the 16-col W blocks this wave owns (``n0`` =
    first W row over all experts, ``skey`` groups the blocks sharing a scale dword).
    Returns ``run(R, row_base16, hook) -> acc[key][mi]`` (rmem f32x4, MFMA C layout:
    lane L holds rows (L//16)*4 + v, column L%16; ``TR``: the W fragment is the
    MFMA A operand, so acc is transposed: lane L holds row L%16 of the 16-row block
    and the 4 consecutive columns (L//16)*4 + v). ``row_base16(row)``: 16 B-unit
    index of A row ``row`` (< R) of the block (indices past ``a_num_bytes`` read 0).
    ``hook(acc, kt, s)``: called before MFMA step s of every tile (group rescale).
    ``pre_a()``: after the W prefetch is issued, before the first A load; ``post_a()``:
    after the first A tile's loads are issued, before they go to LDS (e.g. finish a
    table from loads ``pre_a`` issued without waiting for A in between).
    ``ARB``: A bytes per row per K tile (128 / 256).
    ``WQ``: the W format. "fp4": fd2 mxfp4, each 32-K group decoded with its scale
    folded in (the caller applies the per-channel exponent). "fp8": fp8 in aiter's
    (16, 16) preshuffle, two 16 B halves per lane per 128-K chunk, used as loaded; the
    caller applies the 128x128 block scales (``hook`` and its epilogue).

    Each K tile's W prefetch, the next A tile's LDS store and the one after's global
    load are issued between MFMA steps (step ``W_STEP`` / ``ST_STEP``), A fragments
    (read as step pairs) and B decodes run one step ahead, and every step is a
    separate scheduling region. The MFMAs of a tile's last step are issued after the
    next tile's barrier, so the A reads of the new tile overlap MFMA work.
    """
    TK = ARB  # K per tile (1 B per fp8 A element)
    KG = TK // 128
    assert KG >= 1, "a K tile holds whole 128-K W chunks"
    KT = K // TK
    STEPS = TK // 32  # MFMA K steps per tile
    assert PF >= 1 and KT % (2 * PF) == 0 and KT // (2 * PF) >= 2, (
        f"K loop: K={K} gives {KT} K tiles of {TK}, prefetch {PF} needs a multiple of "
        f"{2 * PF} and at least {4 * PF}: K is too short for this config"
    )
    W_STEP = 0
    ST_STEP = STEPS // 2 if ST_STEP is None else ST_STEP
    assert 0 <= ST_STEP < STEPS
    SLOT = BM * ARB
    W_K0 = K // 128
    FP8W = WQ == "fp8"
    assert WQ in ("fp4", "fp8"), WQ
    # 16 B W loads per lane per 128-K chunk: 32 fp4 or 2 x 16 fp8 of one 32-K group
    WL = 2 if FP8W else 1
    W_BYTES = W_ROWS * K // (1 if FP8W else 2)
    WS_BYTES = W_ROWS * K // 32
    SC_STRIDE_N0 = (K // 256) * 64  # scale dwords per 32-col block
    CH = ARB // 16  # 16 B A chunks per row per tile
    lane_div_16 = lane // fx.Int32(16)
    lane_mod_16 = lane % fx.Int32(16)

    w_tiles = _global_i32_buffer_tiles(arg_w, W_BYTES, 4)
    w_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(bcm), fx.Int32)
    ws_tiles = _global_i32_buffer_tiles(arg_ws, WS_BYTES, 1)
    ws_atom = fx.make_copy_atom(fx.rocdl.BufferCopy32b(0), fx.Int32)
    keys = [k for k, _, _ in cols]
    # 16 B unit of lane L in a 128-K chunk of a 16-row block: fd2, L; fp8 (16, 16)
    # preshuffle [32-K group 4, half 2, row 16, 16 B]: row L % 16 of group L // 16, its
    # second half 16 units further
    lane_w = lane_div_16 * fx.Int32(32) + lane_mod_16 if FP8W else lane * WL
    w_base, npack, s_base, skey_of = {}, {}, {}, {}
    for key, n0, skey in cols:
        w_base[key] = (n0 // fx.Int32(16)) * fx.Int32(W_K0 * 64 * WL) + lane_w
        npack[key] = (n0 // fx.Int32(16)) % fx.Int32(2)
        skey_of[key] = skey
        if skey not in s_base and not FP8W:
            s_base[skey] = (n0 // fx.Int32(32)) * fx.Int32(SC_STRIDE_N0) + lane

    def new_wbuf():
        if const_expr(FP8W):
            return {
                "w": {
                    (k, g, h): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                    for k in keys
                    for g in range(KG)
                    for h in range(WL)
                },
                "s": {},
            }
        return {
            "w": {
                (k, g): fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
                for k in keys
                for g in range(KG)
            },
            "s": {
                sk: fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Int32) for sk in s_base
            },
        }

    def load_w(buf, kt):
        k128 = kt * fx.Int32(KG)
        if const_expr(FP8W):
            for (k, g, h), r in buf["w"].items():
                off = (
                    w_base[k]
                    + (k128 + fx.Int32(g)) * fx.Int32(64 * WL)
                    + fx.Int32(h * 16)
                )
                fx.copy(w_atom, fx.slice(w_tiles, (None, off)), r)
            return
        for (k, g), r in buf["w"].items():
            fx.copy(
                w_atom,
                fx.slice(
                    w_tiles, (None, w_base[k] + (k128 + fx.Int32(g)) * fx.Int32(64))
                ),
                r,
            )
        s_off = (k128 // fx.Int32(2)) * fx.Int32(64)  # 256-K scale block
        for sk, r in buf["s"].items():
            fx.copy(ws_atom, fx.slice(ws_tiles, (None, s_base[sk] + s_off)), r)

    def read_w(buf, kt):
        """-> raw[key, g] = 4 x i32 (8 fp4 each), lut[key, g]; fp8: 8 x i32 (4 fp8
        each), no lut."""
        if const_expr(FP8W):
            raw = {}
            for k in keys:
                for g in range(KG):
                    words = []
                    for h in range(WL):
                        v4 = Vec(fx.memref_load_vec(buf["w"][k, g, h]))
                        words += [fx.Int32(v4[j]) for j in range(4)]
                    raw[k, g] = words
            return raw, None
        raw, lut = {}, {}
        packed = {
            sk: fx.Int32(Vec(fx.memref_load_vec(r))[0]) for sk, r in buf["s"].items()
        }
        for (k, g), r in buf["w"].items():
            v4 = Vec(fx.memref_load_vec(r))
            raw[k, g] = [fx.Int32(v4[j]) for j in range(4)]
            p = packed[skey_of[k]]
            kpack = (kt * fx.Int32(KG) + fx.Int32(g)) % fx.Int32(
                2
            )  # 128-K chunk parity
            sel = (kpack * fx.Int32(2) + npack[k]) * fx.Int32(0x01010101)
            lut[k, g] = _fp8_lut_predelta(_perm_u32(p, p, sel))
        return raw, lut

    a_tiles = fx.logical_divide(
        _global_i32_buffer_view(arg_a, a_num_bytes), fx.make_layout(4, 1)
    )
    a_atom = fx.make_copy_atom(fx.rocdl.BufferCopy128b(acm), fx.Int32)
    lds_flat = fx.make_view(
        fx.recast_iter(fx.Int32, lds_ptr), fx.make_layout(LDS_BYTES // 4, 1)
    )
    lds16 = fx.logical_divide(lds_flat, fx.make_layout(4, 1))
    lds_w_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    lds_r2_atom = fx.make_copy_atom(fx.UniversalCopy128b(), fx.Int32)
    mma_atom = fx.make_mma_atom(fx.rocdl.MFMA(16, 16, 32, fx.Float8E4M3FNUZ))

    def swz(row, col16):
        return col16 ^ (row & fx.Int32(CH - 1))

    def run(R, row_base16, hook=None, pre_a=None, post_a=None):
        MR = R // 16
        A_LOADS = -(-(R * CH) // NT)  # 16 B A chunks per thread per tile
        a_src, a_dst, a_ok, a_row, a_c16 = [], [], [], [], []
        for i in range_constexpr(A_LOADS):
            ch = tx + fx.Int32(i * NT)
            row = ch // fx.Int32(CH)
            c16 = ch % fx.Int32(CH)
            ok = row < fx.Int32(R)
            rowc = ok.select(row, fx.Int32(0))
            a_dst.append(rowc * fx.Int32(CH) + swz(rowc, c16))
            a_ok.append(ok)
            a_row.append(rowc)
            a_c16.append(c16)

        def a_sources():  # after pre_a: row_base16 may read rows written by a producer
            for i in range_constexpr(A_LOADS):
                a_src.append(row_base16(a_row[i]) + a_c16[i])

        a_regs = [
            fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32) for _ in range(A_LOADS)
        ]

        def load_a(kt):
            k16 = kt * fx.Int32(CH)
            for i in range_constexpr(A_LOADS):
                fx.copy(a_atom, fx.slice(a_tiles, (None, a_src[i] + k16)), a_regs[i])

        def store_a(slot):
            s16 = fx.Int32(slot) * fx.Int32(SLOT // 16)
            for i in range_constexpr(A_LOADS):
                dst = fx.slice(lds16, (None, a_dst[i] + s16))
                src = a_regs[i]
                if const_expr((i + 1) * NT <= R * CH):
                    fx.copy(lds_w_atom, src, dst)
                else:
                    _masked_lds_store(lds_w_atom, src, dst, a_ok[i])

        def read_a2(mi, s, slot):
            """16 B = the A fragments of MFMA steps s, s+1 (s even): one
            ds_read_b128 (conflict-free with the 16 B XOR swizzle; two b64 reads of
            the same bytes are 2-way bank conflicted)."""
            row = lane_mod_16 + fx.Int32(mi * 16)
            byte = fx.Int32((s // 4) * 128 + (s % 4) * 8) + lane_div_16 * fx.Int32(32)
            off16 = (
                row * fx.Int32(CH)
                + swz(row, byte // fx.Int32(16))
                + fx.Int32(slot) * fx.Int32(SLOT // 16)
            )
            r = fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Int32)
            fx.copy_atom_call(lds_r2_atom, fx.slice(lds16, (None, off16)), r)
            return Vec(fx.memref_load_vec(r))

        acc = {
            k: [
                fx.make_rmem_tensor(fx.make_layout(4, 1), fx.Float32) for _ in range(MR)
            ]
            for k in keys
        }
        zero4 = Vec.filled(4, 0.0, fx.Float32)
        for k in keys:
            for mi in range_constexpr(MR):
                acc[k][mi].store(zero4)

        def decode_b(raw, lut, s):
            """B operand words (2 x i32) of MFMA step s, per key."""
            if const_expr(FP8W):
                j = s % 4
                return {
                    k: Vec.from_elements(
                        [raw[k, s // 4][2 * j], raw[k, s // 4][2 * j + 1]], fx.Int32
                    )
                    for k in keys
                }
            bws = {}
            for k in keys:  # step s = 128-K chunk s//4, dword s%4
                b8 = _fp4x8_to_fp8x8_fast(raw[k, s // 4][s % 4], lut[k, s // 4])
                bws[k] = Vec(b8.bitcast(fx.Int32))
            return bws

        ELEM = fx.Float8E4M3FNUZ
        NEL = 8

        def mfmas(a_cur, bws):
            for k in keys:
                bf = _frag(bws[k], ELEM, NEL)
                for mi in range_constexpr(MR):
                    af = _frag(a_cur[mi], ELEM, NEL)
                    if const_expr(TR):
                        fx.gemm(mma_atom, acc[k][mi], bf, af, acc[k][mi])
                    else:
                        fx.gemm(mma_atom, acc[k][mi], af, bf, acc[k][mi])

        # W prefetch buffers: wbufs[u][p] holds the tiles kt = u (mod PF) of parity
        # p = (kt // PF) % 2; the loop body covers 2*PF tiles.
        wbufs = [[new_wbuf() for _ in range(2)] for _ in range(PF)]
        for u in range_constexpr(PF):
            load_w(wbufs[u][0], fx.Int32(u))
        if const_expr(pre_a is not None):
            pre_a()  # e.g. wait for A's producer; the W prefetch is already in flight
        a_sources()
        load_a(fx.Int32(0))
        if const_expr(post_a is not None):
            post_a()
        store_a(0)
        load_a(fx.Int32(1))

        # The MFMAs of every tile's last step are issued after the next tile's barrier
        # (their operands wait in a_def / b_def), so the A reads of the new tile overlap
        # MFMA work instead of an idle pipe. Tile 0 "issues" a zero-A deferred step
        # (adds 0).
        a_def = [fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32) for _ in range(MR)]
        b_def = {k: fx.make_rmem_tensor(fx.make_layout(2, 1), fx.Int32) for k in keys}
        for mi in range_constexpr(MR):
            a_def[mi].store(Vec.filled(2, 0, fx.Int32))
        for k in keys:
            b_def[k].store(Vec.filled(2, 0, fx.Int32))

        def flush_deferred():
            mfmas(
                [Vec(fx.memref_load_vec(a_def[mi])) for mi in range_constexpr(MR)],
                {k: Vec(fx.memref_load_vec(b_def[k])) for k in keys},
            )

        def tile_step(kt, u, p, more_w, store_next, load_next2):
            slot = kt % fx.Int32(2)
            raw, lut = read_w(wbufs[u][p], kt)  # W regs only: before the barrier
            b_cur = decode_b(raw, lut, 0)
            rocdl.s_waitcnt(lgkmcnt=0)
            gpu.barrier()  # A(kt) visible; slot (kt+1)%2 free (read at kt-1)
            a_pair = [read_a2(mi, 0, slot) for mi in range_constexpr(MR)]
            flush_deferred()  # tile kt-1's last step
            _sched_fence()
            for s in range_constexpr(STEPS):
                if const_expr(s == W_STEP and more_w):
                    load_w(wbufs[u][1 - p], kt + fx.Int32(PF))
                if const_expr(s == ST_STEP and store_next):
                    store_a(fx.Int32(1) - slot)
                    if const_expr(load_next2):
                        load_a(kt + fx.Int32(2))
                a_cur = [
                    Vec.from_elements(
                        [a_pair[mi][2 * (s % 2)], a_pair[mi][2 * (s % 2) + 1]], fx.Int32
                    )
                    for mi in range_constexpr(MR)
                ]
                if const_expr(
                    s % 2 == 1 and s + 1 < STEPS
                ):  # next pair, one step ahead
                    a_pair = [read_a2(mi, s + 1, slot) for mi in range_constexpr(MR)]
                if const_expr(hook is not None):
                    hook(acc, kt, s)
                if const_expr(s + 1 < STEPS):
                    b_nxt = decode_b(raw, lut, s + 1)
                    mfmas(a_cur, b_cur)
                    _sched_fence()
                    b_cur = b_nxt
                else:
                    for mi in range_constexpr(MR):
                        a_def[mi].store(a_cur[mi])
                    for k in keys:
                        b_def[k].store(b_cur[k])

        @flyc.jit
        def _k_loop():
            for it in range(KT // (2 * PF) - 1):
                for q in range_constexpr(2 * PF):
                    kt = fx.Int32(it) * fx.Int32(2 * PF) + fx.Int32(q)
                    tile_step(kt, q % PF, q // PF, True, True, True)

        _k_loop()
        for q in range_constexpr(2 * PF):  # last 2*PF tiles
            kt = KT - 2 * PF + q
            tile_step(
                fx.Int32(kt),
                q % PF,
                q // PF,
                kt + PF < KT,
                kt + 1 < KT,
                kt + 2 < KT,
            )
        flush_deferred()
        rocdl.s_waitcnt(lgkmcnt=0)
        gpu.barrier()  # every wave is done reading A from LDS
        return acc

    return run
