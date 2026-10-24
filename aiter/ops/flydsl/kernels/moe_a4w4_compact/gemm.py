# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Grouped A4W4 GEMM for the MoE, both stages, one kernel template.

Work unit: a tile (expert, row_start, nrows) of the compact, expert-sorted row
array (exactly T*topk rows, never padded to BM) times a BN-column N block.

pipe="async":   A tile, B atoms, B scales and A scales for each 128-K step go
                global -> LDS by async DMA into a ring (no staging VGPRs).
pipe="hybrid":  A through an async LDS ring, B straight to VGPRs, D steps ahead.
pipe="hybrid2": hybrid with the next step's LDS operands read during the MFMAs.
pipe="il4":     persistent 2x2-wave 256x256 tile, accumulators in AGPRs, one wave
                per SIMD; the next tile's prologue overlaps this tile's epilogue.

stage 1: A = quantized hidden rows gathered by token; epilogue = SwiGLU-OAI +
         fp4 requant of h, written in compact row order ([R, I/2] + [R, I/32]) or,
         with HT, K-step-major (h_t [I/128][R][64 B], scales [I/128][R][4 B]).
stage 2: A = h rows; epilogue = topk weight applied, then
         "rows"  : bf16 per compact row, summed by combine.py (deterministic)
         "fused" : shared-expert tiles add the token's routed rows and write out
"""

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import T

from . import primitives


@functools.cache
def _num_cus():
    from aiter.jit.utils.chip_info import get_cu_num

    return max(8, get_cu_num() // 8 * 8)


OOB = primitives.OOB


def b_tile(stage, n, g, j):
    """Natural n16 tile of B (AITER's shuffle_weight layout) read as tile j of 64-column
    block g. Stage 1 (W13 = [gate; up]): gate tiles 2g, 2g+1 and up tiles n/32 + 2g, + 1,
    so lane c holds gate and up of intermediate columns 32g + c and 32g + 16 + c."""
    if stage == 2:
        return g * 4 + j
    return (j // 2) * (n // 32) + g * 2 + j % 2


def _swz(row):
    """XOR (in 16 B units) on the logical K chunk of an A-tile LDS row (64 B rows):
    zero bank conflicts for the 16 B/lane operand reads (measured)."""
    return ((row >> 1) & 3) * 16


def _il4_plan(n_dma, has_next, mbw, bar):
    """il4 issue plan for one K step: per MFMA slot (MFMA i = half i // 32, row block
    (i % 32) // 4, tile 4 * half + i % 4), the (kind, index) ops issued after that MFMA.
      own: this step's B scale / B tiles 4-7 (5 reads, slots 0-4)
      bar: DMA wait + barrier (slot `bar`); d: next DMAs spread over the rest of the step
           (bursts fill the TA queue and stall issue)
      nb:  next step's B scale / B tiles 0-3 (5 reads, after slot 31, their last use)
      na:  next step's A row block rb + its scale (after slot 35 + 4 rb, its last use)
    """
    slots = [[] for _ in range(32 * 2)]
    for k in range(5):
        slots[k].append(("own", k))
    if has_next:
        slots[bar].append(("bar", 0))
    first = bar + 1
    step = max(1, (63 - first) // max(n_dma, 1))
    for k in range(n_dma):
        slots[min(first + step * k, 63)].append(("d", k))
    if has_next:
        for k in range(5):
            slots[31 + k].append(("nb", k))
        for rb in range(mbw):
            slots[35 + 4 * rb].append(("na", rb))
    return slots


@functools.cache
def build_gemm(
    stage: int,
    K: int,
    N: int,
    BM: int,
    D: int = 3,
    epi: str = "rows",
    pipe: str = "async",
    NW: int = 4,
    diag: str = "",
    WM: int = 1,
    EF: bool = False,
    MV: int = 0,
    AST: bool = False,
    HT: bool = False,
    KTOP: int = 5,
    PERS: int = 0,
    alpha: float = 1.702,
    limit: float = 7.0,
):
    assert stage in (1, 2)
    assert epi in ("rows", "fused")
    assert epi != "fused" or stage == 2
    assert pipe in ("async", "hybrid", "hybrid2", "il4")
    assert NW in (1, 2, 4) and WM in (1, 2) and NW % WM == 0
    WN = NW // WM
    # il4: one wave per SIMD owning 128 output columns (two 64-column groups, 8 n16 tiles)
    CG = 2 if pipe == "il4" else 1
    TWN = 4 * CG  # n16 tiles per wave
    BN = 64 * WN * CG
    THREADS = 64 * NW
    assert K % 128 == 0 and N % BN == 0 and BM % (16 * WM) == 0
    assert WM == 1 or pipe == "il4", "WM > 1: il4 only"
    assert pipe != "il4" or (
        NW == 4 and AST and BM == 256 and WM == 2 and PERS
    ), "il4: persistent 4 waves as 2x2 (256x256 CTA), step-major A scales"
    assert pipe == "il4" or not PERS, "PERS: il4 only"
    assert (
        pipe != "il4" or stage == 1 or (HT and epi == "rows")
    ), "il4 stage 2: h_t A, rows"
    KS = K // 128
    MB = BM // 16
    MBW = MB // WM  # row blocks per wave
    NB = N // BN
    A_INSTR = BM // 16  # 1 KB (16 rows x 64 B) per wave-wide DMA
    A_IT = (A_INSTR + NW - 1) // NW
    KH = K // 2
    KG = K // 32
    INTER = N // 2
    B_GSTEP = 2 if stage == 1 else 4  # natural tile step between adjacent 64-col blocks

    prefetch = pipe == "hybrid2"
    NSTG = max(3 if pipe in ("hybrid2", "il4") else 2, min(D, KS + 1))
    DB = min(D, KS) if pipe in ("hybrid", "hybrid2") else min(3, KS)
    b_lds = pipe in ("async", "il4")
    # LDS is laid out per buffer kind across ring slots ([A x NSTG][B x NSTG][BS][AS]) so
    # every LDS read is (one hoisted base VGPR per kind) + an immediate offset < 64 KB.
    SA = BM * 64
    SB = WN * CG * 4096 if b_lds else 0
    SBS = WN * CG * 256 if b_lds else 0
    SAS = max(BM * 4, 1024) if (AST and BM >= 256) else max(BM, 64) * 4
    # il4 sc2: on even steps each wave issues one 16 B/lane scale DMA (waves 0/1: B / A
    # scales of step t, waves 2/3: of step t+1), 2 per step per CTA instead of 8 x 4 B.
    # Step t+1's scales land a step early, hence one more scale ring slot than NSTG.
    SC2 = pipe == "il4"
    RING = NSTG
    SCD = NSTG + 1 if SC2 else NSTG
    RB = RING * SA
    RBS = RB + RING * SB
    RAS = RBS + SCD * SBS
    lds_bytes = RAS + SCD * SAS
    assert lds_bytes <= 160 * 1024, lds_bytes

    name = f"moe_a4w4_compact_s{stage}_k{K}_n{N}_bm{BM}_w{NW}{'x' + str(WM) if WM > 1 else ''}_{pipe}{D}"
    if stage == 2:
        name += f"_{epi}"
    name += "_xcd"
    if EF:
        name += "_ef"
    if MV:
        name += f"_mv{MV}"
    if AST:
        name += "_ast"
    if HT:
        name += "_ht"
    if epi == "fused":
        name += f"_k{KTOP}"
    if PERS:
        name += f"_p{PERS}"
    NEG_ALPHA_LOG2E = -alpha * 1.4426950408889634
    if diag:
        name += f"_diag{diag.replace('+', '_')}"
    if (alpha, limit) != (1.702, 7.0):
        name += f"_a{alpha}_l{limit}".replace(".", "p")
    name += "_" + primitives.SRC_HASH
    DG = (
        tuple(sorted(diag.split("+"))) if diag else ()
    )  # tuple: part of the JIT cache key
    IL4_BAR = next(
        (int(t[3:]) for t in DG if t.startswith("bar") and t[3:].isdigit()), 5
    )
    SE = (
        "se" in DG
    )  # "even" e8m0 scale rule (checkpoint / runtime quant), else ceil_pow2(amax/6)
    HTA = HT and stage == 2  # A is K-step-major: [K/128][R][64 B]
    # S1S: per-wave LDS staging for the il4 stage-1 epilogue, past the ring.
    S1S = stage == 1 and pipe == "il4"
    HTW = (
        HT and stage == 1 and not S1S
    )  # stage-1 epilogue writes h_t + step-major h scales
    # Stage-2 wide stores: stage each wave's bf16 tile in LDS, then store full 16 B rows.
    S2W = (
        stage == 2
        and pipe != "il4"
        and ((epi == "rows" and "nos2w" not in DG) or epi == "fused")
    )
    # il4 stage 2 C^T: transpose 16-row blocks through a per-wave LDS buffer for wide stores.
    S2I = stage == 2 and pipe == "il4"
    assert not S2I or "s2tl" in DG, "il4 stage 2 needs s2tl"
    # s1tr: C^T stage-1 MFMAs; a lane owns 8 consecutive h values of one row and group.
    S1TR = stage == 1 and pipe == "il4" and "s1tr" in DG
    assert not S1TR or EF, "s1tr: EF epilogue"
    S2NT = "s2nt" in DG
    ROWS_W = MBW * 16
    if S2W:
        lds_bytes = max(lds_bytes, NW * ROWS_W * 128)
    RWP = max(ROWS_W, 64)
    if HTW:
        lds_bytes = max(lds_bytes, NW * RWP * 16)
    S1S_H = ROWS_W * CG * 16
    S1S_W = S1S_H + ROWS_W * CG
    S1S_BASE = lds_bytes
    S2I_W = 16 * (16 * TWN) * 2  # one row block: 16 rows x (16 * TWN) cols bf16
    if S1S:
        lds_bytes = S1S_BASE + NW * S1S_W
    # s2db: two row-block buffers per wave (block rb + 1's writes need not wait for rb's reads)
    S2NB = 2 if S2I and "s2db" in DG else 1
    if S2I:
        lds_bytes = S1S_BASE + NW * S2I_W * S2NB
    assert lds_bytes <= 160 * 1024, lds_bytes

    @fx.struct
    class SharedStorage:
        raw: fx.Array[fx.Uint8, lds_bytes, 16]

    def _kbody(
        a_ptr,
        as_ptr,
        b_ptr,
        bs_ptr,
        tiles_ptr,
        ntiles_ptr,
        rtok_ptr,
        rw_ptr,
        o_ptr,
        os_ptr,
        aux_ptr,
        n_a_rows,
        n_rows,
        n_out,
        lds_base,
    ):
        tid = fx.Int32(gpu.thread_id("x"))
        bid = fx.Int32(gpu.block_id("x"))
        lane = tid % 64
        wave = fx.Int32(rocdl.readfirstlane(T.i32, primitives.raw(tid // 64)))

        r_misc = primitives.rsrc(ntiles_ptr, 4)
        ntiles = fx.Int32(
            rocdl.readfirstlane(T.i32, primitives.bload(r_misc, 0, T.i32))
        )
        bound = ntiles * NB
        n_rows_k, n_a_rows_k, lane_k, wave_k = n_rows, n_a_rows, lane, wave

        def _decode(work):
            return work // NB, work % NB

        def _tile_body(work, mode="full", nxt=None, st=None, defer=False):
            # il4 modes: "pro" sets up a tile and issues its prologue DMAs; "main" runs a tile
            # whose prologue is in flight and issues tile `nxt`'s. `st` carries loaded tile state.
            # Per-tile opaque copies: LICM would otherwise hoist every n_rows-derived
            # per-step scale offset out of the tile loop and spill them.
            n_rows = (
                primitives.s_opaque(n_rows_k, work) if const_expr(PERS) else n_rows_k
            )
            n_a_rows = (
                primitives.s_opaque(n_a_rows_k, work)
                if const_expr(PERS)
                else n_a_rows_k
            )
            lane = primitives.v_opaque(lane_k, work) if const_expr(PERS) else lane_k
            wave = primitives.s_opaque(wave_k, work) if const_expr(PERS) else wave_k
            wm, wn = wave // WN, wave % WN
            tile, nblk = _decode(work)
            r_tiles = primitives.rsrc(tiles_ptr)
            if const_expr(st is None):
                tv = fx.Vector(primitives.bload(r_tiles, tile * 16, T.i32x4))
                e = fx.Int32(rocdl.readfirstlane(T.i32, primitives.raw(tv[0])))
                row_start = fx.Int32(rocdl.readfirstlane(T.i32, primitives.raw(tv[1])))
                nrows = fx.Int32(rocdl.readfirstlane(T.i32, primitives.raw(tv[2])))
                toks = None
            else:
                e, row_start, nrows = st[0], st[1], st[2]
                toks = st[3:]
            st_out = None
            pro_ops = []

            r_a = primitives.rsrc(a_ptr, fx.Int64(n_a_rows) * fx.Int64(KH))
            r_as = primitives.rsrc(
                as_ptr, fx.Int64(n_rows if const_expr(AST) else n_a_rows) * fx.Int64(KG)
            )
            r_tok = primitives.rsrc(rtok_ptr, fx.Int64(n_rows) * fx.Int64(4))
            b_exp = fx.Int64(N // 16) * fx.Int64(KS * 1024)
            bs_exp = fx.Int64(N // 64) * fx.Int64(KS * 256)
            r_b = primitives.rsrc(fx.Int64(b_ptr) + fx.Int64(e) * b_exp, b_exp)
            r_bs = primitives.rsrc(fx.Int64(bs_ptr) + fx.Int64(e) * bs_exp, bs_exp)

            def a_row_of(row, it=None):
                valid = row < nrows
                if const_expr(stage == 1 and toks is not None and it is not None):
                    t = toks[it]
                elif const_expr(stage == 1):
                    t = fx.Int32(primitives.bload(r_tok, (row_start + row) * 4, T.i32))
                else:
                    t = row_start + row
                # HT planes are addressed via soffset, which the buffer range check ignores:
                # invalid rows need an out-of-range voffset of their own.
                return valid.select(t, (OOB // 64) if const_expr(HTA) else n_a_rows)

            def as_off(row):
                return (row < nrows).select((row_start + row) * 4, OOB)

            l16 = lane % 16
            lg = lane // 16

            def a_soff(s):
                # h_t layout: K step s of every row lives in plane s ([KS][R][64 B])
                return n_rows * (s * 64) if const_expr(HTA) else s * 64

            a_rd = l16 * 64 + ((lg * 16) ^ _swz(l16))
            if const_expr(stage == 2 and prefetch):
                # wave-uniform tile offsets ride soffset: one lane VGPR for all 4 tiles
                b_voff = [lane * 16] * 4
                b_soff = [
                    b_tile(stage, N, nblk * WN + wn, j) * (KS * 1024)
                    for j in range_constexpr(4)
                ]
            else:
                b_voff = [
                    b_tile(stage, N, nblk * WN + wn, j) * (KS * 1024) + lane * 16
                    for j in range_constexpr(4)
                ]
                b_soff = [0] * 4
            if const_expr(stage == 1):
                # stage-1 B scales are K-step-major (layout.pack_scales)
                bs_voff = (nblk * WN + wn) * 256 + lane * 4
                BS_STEP = (N // 64) * 256
            else:
                bs_voff = (nblk * WN + wn) * (KS * 256) + lane * 4
                BS_STEP = 256
            # il4: wave w DMAs tile j = w of the CTA's 64-column blocks q = 0 .. 3
            b4_voff = b_tile(stage, N, nblk * (WN * CG), wave) * (KS * 1024) + lane * 16
            BSB = 256 if stage == 1 else KS * 256
            # One 16 B/lane DMA covers the CTA's 4 B-scale blocks (1 KB) for a step.
            BS1 = WN == 4 and stage == 1
            bs16_voff = (nblk * WN) * 256 + lane * 16
            rb0 = wm * MBW  # first row block owned by this wave
            zero = primitives.raw(fx.Vector.filled(4, 0.0, fx.Float32))
            acc = [[zero] * TWN for _ in range(MBW)]

            # Physical 16 B chunk pc of LDS row r holds logical chunk pc ^ swz(r).
            pc = lane % 4
            a_dma_voff = []
            if const_expr(PERS and toks is None and stage == 1):
                toks = [
                    fx.Int32(
                        primitives.bload(
                            r_tok,
                            (row_start + (wave + it * NW) * 16 + lane // 4) * 4,
                            T.i32,
                        )
                    )
                    for it in range_constexpr(A_IT)
                ]
                st_out = [e, row_start, nrows] + toks
            elif const_expr(PERS and st is None):
                st_out = [e, row_start, nrows]
            for it in range_constexpr(A_IT):
                row = (wave + it * NW) * 16 + lane // 4
                a_dma_voff.append(
                    a_row_of(row, it) * (64 if HTA else KH) + ((pc * 16) ^ _swz(row))
                )
            # sc2: the CTA's 1 KB of B scales / 256 rows x 4 B of A scales for one step
            bs16_voff4 = (nblk * (WN * CG) + lane // 16) * BSB + (lane % 16) * 16
            as16_voff = as_off(lane * 4)
            AS16 = AST and BM >= 256
            if const_expr(AS16):
                # K-step-major compact scales: 256 rows x 4 B = one 16 B/lane DMA per 256 rows.
                AS_W = (BM + 255) // 256
                AS_IT = (AS_W + NW - 1) // NW
                as_dma_voff = [
                    as_off((wave + it * NW) * 256 + lane * 4)
                    for it in range_constexpr(AS_IT)
                ]
            elif const_expr(AST):
                # K-step-major compact scales, 64 rows (256 B) per 4 B/lane DMA: exact for BM < 256.
                AS_W = max(BM, 64) // 64
                AS_IT = (AS_W + NW - 1) // NW
                as_dma_voff = [
                    as_off((wave + it * NW) * 64 + lane)
                    for it in range_constexpr(AS_IT)
                ]
            else:
                AS_W = max(BM, 64) // 64
                AS_IT = (AS_W + NW - 1) // NW
                as_dma_voff = [
                    a_row_of((wave + it * NW) * 64 + lane) * KG
                    for it in range_constexpr(AS_IT)
                ]

            def dma_ops(s):
                """This wave's DMA instructions for K step s, as thunks (issue order)."""
                slot = s % RING
                oA, oB = slot * SA, RB + slot * SB
                oBS, oAS = RBS + (s % SCD) * SBS, RAS + (s % SCD) * SAS
                ops = []
                for it in range_constexpr(A_IT):
                    if const_expr(A_INSTR % NW == 0):
                        ops.append(
                            lambda it=it: primitives.dma_async(
                                r_a,
                                lds_base,
                                wave * 1024 + (oA + it * NW * 1024),
                                a_dma_voff[it],
                                soff=a_soff(s),
                            )
                        )
                    else:

                        def _a(it=it):
                            if wave + it * NW < fx.Int32(A_INSTR):
                                primitives.dma_async(
                                    r_a,
                                    lds_base,
                                    wave * 1024 + (oA + it * NW * 1024),
                                    a_dma_voff[it],
                                    soff=a_soff(s),
                                )

                        ops.append(_a)
                if const_expr(pipe == "il4"):
                    # Branch-free: every wave issues the same DMA mix. B: WN*TWN 1 KB tiles
                    # dealt round-robin; scales: one sc2 DMA on even steps.
                    for q in range_constexpr((WN * TWN) // NW):
                        ops.append(
                            lambda q=q: primitives.dma_async(
                                r_b,
                                lds_base,
                                wave * 1024 + (oB + q * NW * 1024),
                                b4_voff + (q * B_GSTEP) * (KS * 1024),
                                soff=s * 1024,
                            )
                        )

                    def _sc2():
                        ts = fx.Int32(s) + (wave >> 1)
                        isb = (wave & 1) == fx.Int32(0)
                        sl = ((wave >> 1) == fx.Int32(0)).select(
                            fx.Int32(s % SCD), fx.Int32((s + 1) % SCD)
                        )
                        primitives.dma_async(
                            primitives.select_rsrc(isb, r_bs, r_as),
                            lds_base,
                            isb.select(RBS + sl * SBS, RAS + sl * SAS),
                            isb.select(bs16_voff4, as16_voff),
                            soff=isb.select(ts * BS_STEP, n_rows * (ts * 4)),
                        )

                    if const_expr(s % 2 == 0):
                        ops.append(_sc2)
                    return ops
                elif const_expr(b_lds):
                    for jj in range_constexpr(4):
                        ops.append(
                            lambda jj=jj: primitives.dma_async(
                                r_b,
                                lds_base,
                                wn * 4096 + (oB + jj * 1024),
                                b_voff[jj],
                                soff=b_soff[jj] + s * 1024,
                            )
                        )
                    if const_expr(BS1 and NW > 1):

                        def _bs1():
                            if wave == fx.Int32(1):
                                primitives.dma_async(
                                    r_bs,
                                    lds_base,
                                    oBS,
                                    bs16_voff,
                                    soff=s * BS_STEP,
                                    nbytes=16,
                                )

                        ops.append(_bs1)
                    else:
                        ops.append(
                            lambda: primitives.dma_async(
                                r_bs,
                                lds_base,
                                wn * 256 + oBS,
                                bs_voff,
                                soff=s * BS_STEP,
                                nbytes=4,
                            )
                        )
                for it in range_constexpr(AS_IT):
                    if const_expr(AS16):

                        def _as(it=it):
                            if wave + it * NW < fx.Int32(AS_W):
                                primitives.dma_async(
                                    r_as,
                                    lds_base,
                                    wave * 1024 + (oAS + it * NW * 1024),
                                    as_dma_voff[it],
                                    soff=n_rows * (s * 4),
                                    nbytes=16,
                                )

                    elif const_expr(AST):

                        def _as(it=it):
                            if wave + it * NW < fx.Int32(AS_W):
                                primitives.dma_async(
                                    r_as,
                                    lds_base,
                                    wave * 256 + (oAS + it * NW * 256),
                                    as_dma_voff[it],
                                    soff=n_rows * (s * 4),
                                    nbytes=4,
                                )

                    else:

                        def _as(it=it):
                            if wave + it * NW < fx.Int32(AS_W):
                                primitives.dma_async(
                                    r_as,
                                    lds_base,
                                    wave * 256 + (oAS + it * NW * 256),
                                    as_dma_voff[it],
                                    soff=s * 4,
                                    nbytes=4,
                                )

                    ops.append(_as)
                return ops

            def issue(s):
                for op in dma_ops(s):
                    op()
                rocdl.asyncmark()

            def issue_b(s):
                bb = [
                    primitives.bload(r_b, b_voff[j], T.i32x4, soff=b_soff[j] + s * 1024)
                    for j in range_constexpr(4)
                ]
                return bb, primitives.bload(r_bs, bs_voff, T.i32, soff=s * BS_STEP)

            breg = {}
            if const_expr(not b_lds and mode != "pro"):
                for p in range_constexpr(DB):
                    breg[p] = issue_b(p)
            if const_expr(mode == "pro" and defer):
                # (issue order) the prologue's DMAs + asyncmarks, for the caller to spread
                thunks = []
                for p in range_constexpr(min(NSTG, KS)):
                    thunks += dma_ops(p) + [rocdl.asyncmark]
                return thunks
            NPRO = min(NSTG if pipe == "il4" else NSTG - 1, KS)
            for p in range_constexpr(NPRO if mode != "main" else 0):
                issue(p)
            if const_expr(mode == "pro"):
                return st_out

            as_rd = l16 * 4 + lg + rb0 * 64 + RAS
            b_rd = lane * 16 + wn * 4096 + RB
            bs_rd = lane * 4 + wn * 256 + RBS

            def read_a(s):
                """LDS operand reads for step s."""
                slot = s % RING
                oA, oB, oBS, oAS = slot * SA, slot * SB, slot * SBS, slot * SAS
                ops = [
                    primitives.lds_load(
                        lds_base, a_rd + rb0 * 1024 + (oA + rb * 1024), T.i32x4
                    )
                    for rb in range_constexpr(MBW)
                ]
                sc = [
                    fx.Int32(
                        fx.Uint8(
                            primitives.lds_load(
                                lds_base, as_rd + (oAS + rb * 64), T.i8, align=1
                            )
                        )
                    )
                    for rb in range_constexpr(MBW)
                ]
                if const_expr(b_lds):
                    bo = [
                        primitives.lds_load(lds_base, b_rd + (oB + j * 1024), T.i32x4)
                        for j in range_constexpr(4)
                    ]
                    bsv = primitives.lds_load(lds_base, bs_rd + oBS, T.i32, align=4)
                    return ops, sc, (bo, bsv)
                return ops, sc, None

            if const_expr(pipe == "il4"):
                # One wave per SIMD, 128x128 per wave. B tiles of step s+1 load during the second
                # half of step s; the mid-step barrier frees step s's slot for step s+NSTG's DMAs.
                b_rd4 = lane * 16 + wn * (TWN * 1024) + RB
                bs_rd4 = lane * 4 + wn * (CG * 256) + RBS
                a_rd4 = a_rd + rb0 * 1024

                def offs(s):
                    slot = s % RING
                    return slot * SA, slot * SB, (s % SCD) * SBS, (s % SCD) * SAS

                def rd_bh(s, h):
                    _, oB, oBS, _ = offs(s)
                    return [
                        lambda: primitives.lds_load(
                            lds_base, bs_rd4 + (oBS + h * 256), T.i32, align=4
                        )
                    ] + [
                        lambda j=j: primitives.lds_load(
                            lds_base, b_rd4 + (oB + (4 * h + j) * 1024), T.i32x4
                        )
                        for j in range_constexpr(4)
                    ]

                def rd_a(s, rb):
                    oA, _, _, oAS = offs(s)
                    return [
                        lambda: primitives.lds_load(
                            lds_base, a_rd4 + (oA + rb * 1024), T.i32x4
                        ),
                        lambda: primitives.lds_load_u8(
                            lds_base, as_rd + (oAS + rb * 64)
                        ),
                    ]

                rocdl.wait_asyncmark(max(0, min(NSTG, KS) - 1))
                gpu.barrier()
                v = [f() for f in rd_bh(0, 0)]
                bsv = [v[0], None]
                b_ops = v[1:] + [None] * 4
                a_ops, sa = [None] * MBW, [None] * MBW
                for rb in range_constexpr(MBW):
                    a_ops[rb], sa[rb] = [f() for f in rd_a(0, rb)]
                NMF = MBW * TWN
                if const_expr(mode == "main"):
                    tvn = fx.Vector(
                        primitives.bload(r_tiles, _decode(nxt)[0] * 16, T.i32x4)
                    )
                for s in range_constexpr(KS):
                    has_next = s + 1 < KS
                    dms = dma_ops(s + NSTG) if const_expr(s + NSTG < KS) else []
                    plan = _il4_plan(len(dms), has_next, MBW, IL4_BAR)
                    own = rd_bh(s, 1)
                    nb0 = rd_bh(s + 1, 0) if const_expr(has_next) else []
                    nxt_b, nxt_bs = [None] * 4, None
                    nxt_a, nxt_sa = [None] * MBW, [None] * MBW
                    for i in range_constexpr(NMF):
                        h, rb, j = i // 32, (i % 32) // 4, i % 4
                        t = 4 * h + j
                        if const_expr(S2I or S1TR):
                            acc[rb][t] = primitives.mfma_fp4_agpr(
                                acc[rb][t] if const_expr(s > 0) else None,
                                b_ops[t],
                                a_ops[rb],
                                bsv[h],
                                sa[rb],
                                0,
                                j,
                            )
                        else:
                            acc[rb][t] = primitives.mfma_fp4_agpr(
                                acc[rb][t] if const_expr(s > 0) else None,
                                a_ops[rb],
                                b_ops[t],
                                sa[rb],
                                bsv[h],
                                j,
                            )
                        if const_expr(mode == "main" and s == KS // 2 and i == 16):
                            rocdl.sched_barrier(0)
                            e_n, rs_n, nr_n = [
                                fx.Int32(
                                    rocdl.readfirstlane(T.i32, primitives.raw(tvn[q]))
                                )
                                for q in range_constexpr(3)
                            ]
                            st_out = [e_n, rs_n, nr_n] + [
                                fx.Int32(
                                    primitives.bload(
                                        r_tok,
                                        (rs_n + (wave + it * NW) * 16 + lane // 4) * 4,
                                        T.i32,
                                    )
                                )
                                for it in range_constexpr(A_IT if stage == 1 else 0)
                            ]
                            rocdl.sched_barrier(0)
                        for k2 in range_constexpr(len(plan[i])):
                            kind, idx = plan[i][k2]
                            rocdl.sched_barrier(0)
                            if const_expr(kind == "own"):
                                val = own[idx]()
                                if const_expr(idx == 0):
                                    bsv[1] = val
                                else:
                                    b_ops[4 + idx - 1] = val
                            elif const_expr(kind == "bar"):
                                last = min(s - 1 + NSTG, KS - 1)
                                rocdl.wait_asyncmark(max(0, last - (s + 1)))
                                gpu.barrier()
                            elif const_expr(kind == "d"):
                                dms[idx]()
                                if const_expr(idx == len(dms) - 1):
                                    rocdl.asyncmark()
                            elif const_expr(kind == "nb"):
                                val = nb0[idx]()
                                if const_expr(idx == 0):
                                    nxt_bs = val
                                else:
                                    nxt_b[idx - 1] = val
                            else:
                                nxt_a[idx], nxt_sa[idx] = [
                                    f() for f in rd_a(s + 1, idx)
                                ]
                            rocdl.sched_barrier(0)
                    if const_expr(has_next):
                        bsv = [nxt_bs, None]
                        b_ops = nxt_b + [None] * 4
                        a_ops, sa = nxt_a, nxt_sa
                rocdl.sched_barrier(0)
                fenced = primitives.mfma_drain(
                    [
                        acc[rb][t]
                        for rb in range_constexpr(MBW)
                        for t in range_constexpr(TWN)
                    ]
                )
                acc = [fenced[rb * TWN : (rb + 1) * TWN] for rb in range_constexpr(MBW)]
                rocdl.sched_barrier(0)
                if const_expr(mode == "main"):
                    gpu.barrier()  # every wave is past its last ring read
                    pro_ops = _tile_body(nxt, "pro", st=st_out, defer=True)
                    rocdl.sched_barrier(0)

            if const_expr(prefetch):
                # LDS operands for step s+1 are read while step s's MFMAs run.
                rocdl.wait_asyncmark(max(0, min(NSTG - 2, KS - 1)))
                gpu.barrier()
                cur = read_a(0)
                for s in range_constexpr(KS):
                    a_ops, sa, bl = cur
                    if const_expr(b_lds):
                        b_ops, bs = bl
                    else:
                        b_ops, bs = breg[s]
                        if const_expr(s + DB < KS):
                            breg[s + DB] = issue_b(s + DB)
                    for rb in range_constexpr(MBW):
                        for j in range_constexpr(4):
                            acc[rb][j] = primitives.mfma_fp4(
                                acc[rb][j], a_ops[rb], b_ops[j], sa[rb], bs, 0, j
                            )
                    if const_expr(s + 1 < KS):
                        rocdl.wait_asyncmark(max(0, min(NSTG - 3, KS - 2 - s)))
                        gpu.barrier()
                        if const_expr(s + NSTG - 1 < KS):
                            issue(s + NSTG - 1)
                        cur = read_a(s + 1)
            for s in range_constexpr(KS if pipe in ("async", "hybrid") else 0):
                rocdl.wait_asyncmark(min(NSTG - 2, KS - 1 - s))
                gpu.barrier()
                if const_expr(s + NSTG - 1 < KS):
                    issue(s + NSTG - 1)
                if const_expr((not b_lds) and s + DB < KS):
                    breg[s + DB] = issue_b(s + DB)
                a_ops, sa, bl = read_a(s)
                if const_expr(b_lds):
                    b_ops, bs = bl
                else:
                    b_ops, bs = breg.pop(s)
                for rb in range_constexpr(MBW):
                    for j in range_constexpr(4):
                        acc[rb][j] = primitives.mfma_fp4(
                            acc[rb][j], a_ops[rb], b_ops[j], sa[rb], bs, 0, j
                        )

            if const_expr(stage == 1):
                _epi1(acc, pro_ops, wave, lane, wn, nblk, row_start, nrows, n_rows, rb0)
            elif const_expr(S2I):
                _epi2_il4(
                    acc, pro_ops, wave, lane, wn, nblk, row_start, nrows, n_rows, rb0
                )
            else:
                if const_expr(prefetch):
                    rocdl.sched_barrier(0)
                _epi2(acc, wave, lane, wn, nblk, row_start, nrows, n_rows, rb0, r_tok)
            return st_out

        def _epi1(acc, pro_ops, wave, lane, wn, nblk, row_start, nrows, n_rows, rb0):
            """Stage-1 epilogue: SwiGLU-OAI, per-32 amax, e8m0 + fp4 requant, h stores.
            Tiles (4gi, 4gi+1) / (4gi+2, 4gi+3) are gate / up of columns (c, 16 + c) of a
            32-column group; a lane-pair exchange makes each fp4 byte two adjacent columns
            (byte hb of the group's 16)."""
            l16 = lane % 16
            lg = lane // 16
            odd = (l16 & fx.Int32(1)) == fx.Int32(1)
            hb = primitives.pair_col(l16) >> fx.Int32(1)
            r_o = primitives.rsrc(o_ptr, fx.Int64(n_rows) * fx.Int64(INTER // 2))
            if const_expr(HTW):
                gpu.barrier()  # all waves done with the LDS ring
                hwb = wave * (RWP * 16)
            r_os = primitives.rsrc(os_ptr, fx.Int64(n_rows) * fx.Int64(INTER // 32))
            lim = fx.Float32(limit)

            def swiglu(gv, uv):
                gg = fx.min(fx.Float32(gv), lim)
                uu = fx.max(fx.min(fx.Float32(uv), lim), -lim)
                if const_expr(EF):
                    sig = primitives.fast_rcp(
                        fx.Float32(1.0)
                        + primitives.exp2_raw(gg * fx.Float32(NEG_ALPHA_LOG2E))
                    )
                else:
                    sig = fx.Float32(1.0) / (
                        fx.Float32(1.0) + fmath.exp(gg * fx.Float32(-alpha))
                    )
                return gg * sig * (uu + fx.Float32(1.0))

            def swiglu2(g0, g1, u0, u1):
                """swiglu of an (even, odd) pair: same operations and order, the
                elementwise mul/add as packed v_pk_*_f32. EF only."""
                gg = fx.Vector.from_elements(
                    [fx.min(fx.Float32(g0), lim), fx.min(fx.Float32(g1), lim)],
                    fx.Float32,
                )
                uu = fx.Vector.from_elements(
                    [primitives.fmed3(u0, -lim, lim), primitives.fmed3(u1, -lim, lim)],
                    fx.Float32,
                )
                t = gg * fx.Vector.filled(2, NEG_ALPHA_LOG2E, fx.Float32)
                den = fx.Vector.from_elements(
                    [primitives.exp2_raw(t[0]), primitives.exp2_raw(t[1])], fx.Float32
                ) + fx.Vector.filled(2, 1.0, fx.Float32)
                sig = fx.Vector.from_elements(
                    [primitives.fast_rcp(den[0]), primitives.fast_rcp(den[1])],
                    fx.Float32,
                )
                h = gg * sig * (uu + fx.Vector.filled(2, 1.0, fx.Float32))
                return fx.Float32(h[0]), fx.Float32(h[1])

            def e8m0(m, bounded):
                if const_expr(SE):
                    # |h| <= limit * (limit + 1): the bounded form is exact
                    return (
                        primitives.e8m0_even_small(m)
                        if const_expr(bounded)
                        else primitives.e8m0_even(m)
                    )
                bits = (m * fx.Float32(1.0 / 6.0)).bitcast(fx.Int32)
                bexp = ((bits + fx.Int32(0x7FFFFF)).shrui(fx.Int32(23))) & fx.Int32(
                    0xFF
                )
                return fx.min(bexp, fx.Int32(254))

            EPG = next(
                (int(t[3:]) for t in DG if t.startswith("epg") and t[3:].isdigit()), 1
            )
            assert MBW % EPG == 0
            for rbb in range_constexpr(MBW // EPG if S1S else 0):
                # All CG*4 (group, row) chains of a row block at once: the DPP row-max
                # levels interleave across chains instead of stalling on DPP hazards.
                per = -(-len(pro_ops) // (MBW // EPG))
                per1 = -(-len(pro_ops) // MBW)
                for op in (
                    pro_ops[rbb * per : (rbb + 1) * per] if const_expr(not S1TR) else []
                ):
                    rocdl.sched_barrier(0)
                    op()
                    rocdl.sched_barrier(0)
                sw = S1S_BASE + wave * S1S_W
                if const_expr(S1TR):
                    # lane (row l16, quarter lg) holds columns 4lg..4lg+3 of every tile: h bytes
                    # 4lg..4lg+3 of group gi of row rb*16 + l16
                    hv, ms = [], []
                    for rb in range_constexpr(rbb * EPG, (rbb + 1) * EPG):
                        for op in pro_ops[rb * per1 : (rb + 1) * per1]:
                            rocdl.sched_barrier(0)
                            op()
                            rocdl.sched_barrier(0)
                        for gi in range_constexpr(CG):
                            vg_e = fx.Vector(acc[rb][4 * gi + 0])
                            vg_o = fx.Vector(acc[rb][4 * gi + 1])
                            vu_e = fx.Vector(acc[rb][4 * gi + 2])
                            vu_o = fx.Vector(acc[rb][4 * gi + 3])
                            hs = [
                                swiglu2(vg_e[i], vg_o[i], vu_e[i], vu_o[i])
                                for i in range_constexpr(4)
                            ]
                            m = None
                            for h0, h1 in hs:
                                mm = fx.max(fmath.absf(h0), fmath.absf(h1))
                                m = mm if m is None else fx.max(m, mm)
                            hv.append((rb, gi, hs))
                            ms.append(m)
                    ms = primitives.rows4_max_nonneg_f32_multi(ms)
                    bx = {}
                    for k in range_constexpr(len(hv)):
                        rb, gi, hs = hv[k]
                        bexp = e8m0(ms[k], True)
                        qs = (bexp << fx.Int32(23)).bitcast(fx.Float32)
                        # bytes 0-1: columns 4lg .. +3, bytes 2-3: 16 + 4lg .. +3
                        pk = primitives.raw(fx.Int32(0))
                        for i in range_constexpr(4):
                            pk = rocdl.cvt_scalef32_pk_fp4_f32(
                                T.i32,
                                pk,
                                primitives.raw(hs[2 * (i % 2)][i // 2]),
                                primitives.raw(hs[2 * (i % 2) + 1][i // 2]),
                                primitives.raw(qs),
                                i,
                            )
                        pk = fx.Int32(pk)
                        rl = fx.Int32(rb * 16) + l16
                        ho = sw + rl * (CG * 16) + (gi * 16) + lg * 2
                        primitives.lds_store(pk.to(fx.Int16), lds_base, ho, align=2)
                        primitives.lds_store(
                            pk.shrui(fx.Int32(16)).to(fx.Int16),
                            lds_base,
                            ho + 8,
                            align=2,
                        )
                        bx[(rb, gi)] = bexp
                    for rb in range_constexpr(rbb * EPG, (rbb + 1) * EPG):
                        # row rb*16 + l16: its CG == 2 scale bytes (the 4 quarters write the same)
                        sd = bx[(rb, 0)] | (bx[(rb, 1)] << fx.Int32(8))
                        primitives.lds_store(
                            sd.to(fx.Int16),
                            lds_base,
                            sw + S1S_H + (fx.Int32(rb * 16) + l16) * CG,
                            align=2,
                        )
                else:
                    ch = []
                    for rb in range_constexpr(rbb * EPG, (rbb + 1) * EPG):
                        for gi in range_constexpr(CG):
                            vg_e = fx.Vector(acc[rb][4 * gi + 0])
                            vg_o = fx.Vector(acc[rb][4 * gi + 1])
                            vu_e = fx.Vector(acc[rb][4 * gi + 2])
                            vu_o = fx.Vector(acc[rb][4 * gi + 3])
                            for v in range_constexpr(4):
                                if const_expr(EF):
                                    ch.append(
                                        (rb, gi, v)
                                        + swiglu2(vg_e[v], vg_o[v], vu_e[v], vu_o[v])
                                    )
                                else:
                                    ch.append(
                                        (
                                            rb,
                                            gi,
                                            v,
                                            swiglu(vg_e[v], vu_e[v]),
                                            swiglu(vg_o[v], vu_o[v]),
                                        )
                                    )
                    ms = [
                        fx.max(fmath.absf(h0), fmath.absf(h1)) for _, _, _, h0, h1 in ch
                    ]
                    if const_expr(EF):
                        ms = primitives.row16_max_nonneg_f32_multi(ms)
                    else:
                        for k in range_constexpr(len(ms)):
                            for off in (1, 2, 4, 8):
                                ms[k] = fx.max(
                                    ms[k],
                                    ms[k].shuffle_xor(fx.Int32(off), fx.Int32(64)),
                                )
                    bx = {}
                    for k in range_constexpr(len(ch)):
                        rb, gi, v, h0, h1 = ch[k]
                        bexp = e8m0(ms[k], True)
                        qs = (bexp << fx.Int32(23)).bitcast(fx.Float32)
                        h0, h1 = primitives.pair_cols_f32(h0, h1, odd)
                        pk = rocdl.cvt_scalef32_pk_fp4_f32(
                            T.i32,
                            primitives.raw(fx.Int32(0)),
                            primitives.raw(h0),
                            primitives.raw(h1),
                            primitives.raw(qs),
                            0,
                        )
                        rl = fx.Int32(rb * 16) + lg * 4 + v
                        primitives.lds_store(
                            fx.Int32(pk).to(fx.Int8),
                            lds_base,
                            sw + rl * (CG * 16) + (gi * 16) + hb,
                            align=1,
                        )
                        bx[(rb, v, gi)] = bexp
                    for rb in range_constexpr(rbb * EPG, (rbb + 1) * EPG):
                        # One 8 B write per lane; the 16 lanes of a row group write the same bytes.
                        sd = [
                            bx[(rb, 2 * q, 0)]
                            | (bx[(rb, 2 * q, 1)] << fx.Int32(8))
                            | (bx[(rb, 2 * q + 1, 0)] << fx.Int32(16))
                            | (bx[(rb, 2 * q + 1, 1)] << fx.Int32(24))
                            for q in range_constexpr(2)
                        ]
                        primitives.lds_store(
                            fx.Vector.from_elements(sd, fx.Int32),
                            lds_base,
                            sw + S1S_H + (fx.Int32(rb * 16) + lg * 4) * CG,
                            align=8,
                        )
            for rbg in range_constexpr(MBW * CG if not S1S else 0):
                rb, gi = rbg // CG, rbg % CG
                g = (nblk * WN + wn) * CG + gi
                vg_e = fx.Vector(acc[rb][4 * gi + 0])
                vg_o = fx.Vector(acc[rb][4 * gi + 1])
                vu_e = fx.Vector(acc[rb][4 * gi + 2])
                vu_o = fx.Vector(acc[rb][4 * gi + 3])
                for v in range_constexpr(4):
                    row = rb0 * 16 + fx.Int32(rb * 16) + lg * 4 + v
                    hs = [swiglu(vg_e[v], vu_e[v]), swiglu(vg_o[v], vu_o[v])]
                    m = fx.max(fmath.absf(hs[0]), fmath.absf(hs[1]))
                    if const_expr(EF):
                        m = primitives.row16_max_nonneg_f32(m)
                    else:
                        for off in (1, 2, 4, 8):
                            m = fx.max(m, m.shuffle_xor(fx.Int32(off), fx.Int32(64)))
                    bexp = e8m0(m, False)
                    qs = (bexp << fx.Int32(23)).bitcast(fx.Float32)
                    hs = primitives.pair_cols_f32(hs[0], hs[1], odd)
                    pk = rocdl.cvt_scalef32_pk_fp4_f32(
                        T.i32,
                        primitives.raw(fx.Int32(0)),
                        primitives.raw(hs[0]),
                        primitives.raw(hs[1]),
                        primitives.raw(qs),
                        0,
                    )
                    valid = row < nrows
                    grow = row_start + row
                    soff_ok = valid & (l16 == fx.Int32(0))
                    if const_expr(HTW):
                        rl = fx.Int32(rb * 16) + lg * 4 + v
                        primitives.lds_store(
                            fx.Int32(pk).to(fx.Int8),
                            lds_base,
                            hwb + rl * 16 + hb,
                            align=1,
                        )
                        hs_off = ((g // 4) * n_rows + grow) * 4 + (g % 4)
                    else:
                        off = valid.select(
                            grow * (INTER // 2) + g * 16 + hb, fx.Int32(0x7FFFFFF0)
                        )
                        primitives.bstore(fx.Int32(pk).to(fx.Int8), r_o, off)
                        hs_off = grow * (INTER // 32) + g
                    soff = soff_ok.select(hs_off, fx.Int32(0x7FFFFFF0))
                    primitives.bstore(bexp.to(fx.Int8), r_os, soff)
            if const_expr(S1S):
                # h[row][g0*16 .. g0*16 + CG*16): CG*16 B per row, 16 B per lane
                sw = S1S_BASE + wave * S1S_W
                g0 = (nblk * WN + wn) * CG
                LPR = CG  # lanes per row
                for it in range_constexpr((ROWS_W * LPR) // 64):
                    rl = fx.Int32(it * (64 // LPR)) + lane // LPR
                    part = lane % LPR
                    v16 = primitives.lds_load(
                        lds_base, sw + rl * (CG * 16) + part * 16, T.i32x4
                    )
                    row = rb0 * 16 + rl
                    if const_expr(HT):
                        # h_t[g//4][row][64 B]; g0 is even, so both groups sit in plane g0//4
                        off = (
                            ((g0 // 4) * n_rows + row_start + row) * 64
                            + (g0 % 4) * 16
                            + part * 16
                        )
                    else:
                        off = (row_start + row) * (INTER // 2) + g0 * 16 + part * 16
                    primitives.bstore(
                        v16, r_o, (row < nrows).select(off, fx.Int32(0x7FFFFFF0))
                    )
                # scales: CG bytes per row at h_s[row][g0 .. g0 + CG) (HT: [g0//4][row][4 B])
                for it in range_constexpr((ROWS_W + 63) // 64):
                    rl = fx.Int32(it * 64) + lane
                    row = rb0 * 16 + rl
                    sv = primitives.lds_load(
                        lds_base, sw + S1S_H + rl * CG, T.i16, align=2
                    )
                    if const_expr(HT):
                        off = ((g0 // 4) * n_rows + row_start + row) * 4 + (g0 % 4)
                    else:
                        off = (row_start + row) * (INTER // 32) + g0
                    ok = (row < nrows) & (rl < fx.Int32(ROWS_W))
                    primitives.bstore(sv, r_os, ok.select(off, fx.Int32(0x7FFFFFF0)))
            if const_expr(HTW):
                # h_t[g//4][row][64 B]: this wave's 16 B column group of each row
                g = nblk * WN + wn
                for it in range_constexpr((ROWS_W + 63) // 64):
                    rl = fx.Int32(it * 64) + lane
                    v16 = primitives.lds_load(lds_base, hwb + rl * 16, T.i32x4)
                    row = rb0 * 16 + rl
                    off = ((g // 4) * n_rows + row_start + row) * 64 + (g % 4) * 16
                    ok = (row < nrows) & (rl < fx.Int32(ROWS_W))
                    primitives.bstore(v16, r_o, ok.select(off, fx.Int32(0x7FFFFFF0)))

        def _epi2_il4(
            acc, pro_ops, wave, lane, wn, nblk, row_start, nrows, n_rows, rb0
        ):
            """il4 stage-2 epilogue: C^T lane = row l16, output columns 64gi + 32p + 16q + 4lg
            .. +3 of the wave's 128 (tile 4gi + 2p + q), through the private LDS row buffer:
            16 B chunk x of the row at x ^ (row & 7) (8 consecutive rows of one write hit
            distinct banks; the 16 B row reads stay conflict-free)."""
            l16 = lane % 16
            lg = lane // 16
            r_w = primitives.rsrc(rw_ptr, fx.Int64(n_rows) * fx.Int64(4))
            r_o = primitives.rsrc(
                fx.Int64(o_ptr) + fx.Int64(row_start) * fx.Int64(N * 2),
                fx.Int64(n_rows - row_start) * fx.Int64(N * 2),
            )
            cm_o = primitives.NT if const_expr(S2NT) else 0
            sw0 = S1S_BASE + wave * (S2I_W * S2NB)
            per = -(-len(pro_ops) // MBW)
            # Every row block's weights before any store: vmcnt also counts stores, so a
            # per-block load would wait for all earlier stores to drain.
            wts = [
                fx.Float32(
                    primitives.bload(
                        r_w, (row_start + rb0 * 16 + fx.Int32(rb * 16) + l16) * 4, T.f32
                    )
                )
                for rb in range_constexpr(MBW)
            ]
            for rb in range_constexpr(MBW):
                for op in pro_ops[rb * per : (rb + 1) * per]:
                    rocdl.sched_barrier(0)
                    op()
                    rocdl.sched_barrier(0)
                sw = sw0 + (rb % S2NB) * S2I_W
                wv8 = fx.Vector.from_elements([wts[rb]] * 8, fx.Float32)
                for gi in range_constexpr(CG):
                    for p in range_constexpr(2):
                        ve_ = fx.Vector(acc[rb][4 * gi + 2 * p])
                        vo_ = fx.Vector(acc[rb][4 * gi + 2 * p + 1])
                        v8 = fx.Vector.from_elements(
                            [
                                fx.Float32(x[v])
                                for x in (ve_, vo_)
                                for v in range_constexpr(4)
                            ],
                            fx.Float32,
                        )
                        b8 = fx.Vector((v8 * wv8).to(fx.BFloat16)).bitcast(fx.Int32)
                        # columns 4lg .. +3 of tile 2p, then of tile 2p + 1 (16 columns on)
                        for q in range_constexpr(2):
                            c = (fx.Int32(gi * 8 + p * 4 + q * 2) + lg // 2) ^ (l16 & 7)
                            primitives.lds_store(
                                fx.Vector.from_elements(
                                    [fx.Int32(b8[2 * q]), fx.Int32(b8[2 * q + 1])],
                                    fx.Int32,
                                ),
                                lds_base,
                                sw + l16 * 256 + c * 16 + (lg % 2) * 8,
                                align=8,
                            )
                for it in range_constexpr(4):
                    rl = fx.Int32(it * 4) + lane // 16
                    c = lane % 16
                    v16 = primitives.lds_load(
                        lds_base, sw + rl * 256 + ((c ^ (rl & 7)) * 16), T.i32x4
                    )
                    row = rb0 * 16 + fx.Int32(rb * 16) + rl
                    off = row * (N * 2) + (nblk * BN + wn * 128) * 2 + c * 16
                    primitives.bstore(
                        v16,
                        r_o,
                        (row < nrows).select(off, fx.Int32(0x7FFFFF00)),
                        cm=cm_o,
                    )
            for op in pro_ops[MBW * per :]:
                op()

        def _epi2(acc, wave, lane, wn, nblk, row_start, nrows, n_rows, rb0, r_tok):
            """Stage-2 epilogue (async / hybrid pipes): tile pair p gives lane c output
            columns (32p + c, 32p + 16 + c); a lane-pair exchange of the packed bf16x2 makes
            them two adjacent columns 32p + pair_col(c) (one 4 B store)."""
            if const_expr(prefetch):
                lane = primitives.lane_id()
            l16 = lane % 16
            lg = lane // 16
            psel = primitives.pair_sel((l16 & fx.Int32(1)) == fx.Int32(1))
            pcol = primitives.pair_col(l16)

            def pair_pk(a, b, w):
                pk = fx.Vector.from_elements(
                    [fx.Float32(a) * w, fx.Float32(b) * w], fx.Float32
                ).to(fx.BFloat16)
                return primitives.pair_cols_bf16x2(
                    fx.Vector(pk).bitcast(fx.Int32)[0], psel
                )

            r_w = primitives.rsrc(rw_ptr, fx.Int64(n_rows) * fx.Int64(4))
            # 64-bit per-tile base (no 32-bit offset overflow for any R). The record count
            # spans all remaining rows: a tile-sized count measured ~30% slower.
            r_o = primitives.rsrc(
                fx.Int64(o_ptr) + fx.Int64(row_start) * fx.Int64(N * 2),
                fx.Int64(n_rows - row_start) * fx.Int64(N * 2),
            )
            col0 = nblk * BN + wn * 64 + pcol
            if const_expr(S2W):
                gpu.barrier()  # every wave is done reading the LDS ring
                wb = wave * (ROWS_W * 128)
                for rb in range_constexpr(MBW):
                    vs = [fx.Vector(acc[rb][j]) for j in range_constexpr(4)]
                    r4 = rb0 * 16 + fx.Int32(rb * 16) + lg * 4
                    wv = fx.Vector(
                        primitives.bload(r_w, (row_start + r4) * 4, T.vec(4, T.f32))
                    )
                    for v in range_constexpr(4):
                        rl = fx.Int32(rb * 16) + lg * 4 + v  # wave-local row
                        w = fx.Float32(wv[v])
                        for p in range_constexpr(2):
                            pk = pair_pk(vs[2 * p][v], vs[2 * p + 1][v], w)
                            ch = (fx.Int32(p * 4) + pcol // 8) ^ (rl & 7)
                            primitives.lds_store(
                                pk,
                                lds_base,
                                wb + rl * 128 + ch * 16 + (pcol % 8) * 2,
                                align=4,
                            )
                cm_o = primitives.NT if const_expr(S2NT) else 0
                if const_expr(epi == "fused"):
                    # Shared-expert tile: out[t] = own row + the token's routed rows, summed in fp32;
                    # replaces the combine pass. Rows past nrows are clamped and not stored.
                    r_inv = primitives.rsrc(aux_ptr, fx.Int64(n_rows) * fx.Int64(4))
                    r_out = primitives.rsrc(os_ptr, fx.Int64(n_out) * fx.Int64(N * 2))
                    NIT = ROWS_W // 8
                    ch = lane % 8
                    colb = (nblk * BN + wn * 64) * 2 + ch * 16
                    rls = [fx.Int32(it * 8) + lane // 8 for it in range_constexpr(NIT)]
                    oks = [rb0 * 16 + rl < nrows for rl in rls]
                    grows = [
                        row_start + ok.select(rb0 * 16 + rl, fx.Int32(0))
                        for ok, rl in zip(oks, rls)
                    ]
                    ts = [
                        fx.Int32(primitives.bload(r_tok, g * 4, T.i32)) for g in grows
                    ]
                    srcs = []
                    for it in range_constexpr(NIT):
                        ss = []
                        for sl in range_constexpr(KTOP):
                            rs = fx.Int32(
                                primitives.bload(r_inv, (ts[it] * KTOP + sl) * 4, T.i32)
                            )
                            rs = fx.max(fx.min(rs, n_rows - 1), fx.Int32(0))
                            ss.append(rs)
                        srcs.append(ss)

                    def fc_rows(it):
                        if const_expr("fcsk" in DG):
                            # the own slot is never written: load another slot of the token (an L2 hit) instead
                            src = [
                                (srcs[it][sl] != grows[it]).select(
                                    srcs[it][sl], srcs[it][(sl + 1) % KTOP]
                                )
                                for sl in range_constexpr(KTOP)
                            ]
                        else:
                            src = srcs[it]
                        return [
                            fx.Vector(
                                primitives.gload(
                                    fx.Int64(o_ptr)
                                    + fx.Int64(src[sl]) * fx.Int64(N * 2)
                                    + fx.Int64(colb),
                                    T.vec(8, T.bf16),
                                )
                            )
                            for sl in range_constexpr(KTOP)
                        ]

                    ys = {0: fc_rows(0)}
                    z8 = fx.Vector.filled(8, 0.0, fx.Float32)
                    for it in range_constexpr(NIT):
                        if const_expr(it + 1 < NIT):
                            ys[it + 1] = fc_rows(it + 1)
                        rl = rls[it]
                        v16 = primitives.lds_load(
                            lds_base, wb + rl * 128 + ((ch ^ (rl & 7)) * 16), T.i32x4
                        )
                        accv = fx.Vector(v16).bitcast(fx.BFloat16).to(fx.Float32)
                        yl = ys.pop(it)
                        for sl in range_constexpr(KTOP):
                            other = srcs[it][sl] != grows[it]
                            accv = accv + other.select(yl[sl].to(fx.Float32), z8)
                        primitives.bstore(
                            accv.to(fx.BFloat16),
                            r_out,
                            oks[it].select(
                                ts[it] * (N * 2) + colb, fx.Int32(0x7FFFFFF0)
                            ),
                        )
                else:
                    for it in range_constexpr(ROWS_W // 8):
                        rl = fx.Int32(it * 8) + lane // 8
                        ch = lane % 8
                        v16 = primitives.lds_load(
                            lds_base, wb + rl * 128 + ((ch ^ (rl & 7)) * 16), T.i32x4
                        )
                        row = rb0 * 16 + rl
                        colb = (nblk * BN + wn * 64) * 2 + ch * 16
                        off = row * (N * 2) + colb
                        primitives.bstore(
                            v16,
                            r_o,
                            (row < nrows).select(off, fx.Int32(0x7FFFFF00)),
                            cm=cm_o,
                        )
            for rb in range_constexpr(MBW if not S2W else 0):
                vs = [fx.Vector(acc[rb][j]) for j in range_constexpr(4)]
                for v in range_constexpr(4):
                    row = rb0 * 16 + fx.Int32(rb * 16) + lg * 4 + v
                    valid = row < nrows
                    grow = row_start + row
                    w = fx.Float32(primitives.bload(r_w, grow * 4, T.f32))
                    for p in range_constexpr(2):
                        pk = pair_pk(vs[2 * p][v], vs[2 * p + 1][v], w)
                        col = col0 + p * 32
                        off = valid.select((row * N + col) * 2, fx.Int32(0x7FFFFF00))
                        primitives.bstore(pk, r_o, off)

        # CTAs are dealt round-robin over the 8 XCDs; each XCD gets a contiguous range of
        # work so an expert's m-tiles share one L2.
        xq = bound // 8
        xr = bound % 8
        xc = bid % 8
        xstart = xc * xq + fx.min(xc, xr)
        if const_expr(PERS):
            # Persistent il4: tile w+1's prologue DMAs overlap tile w's epilogue. The next
            # work index is clamped (the last tile re-fetches itself), so nothing branches.
            gx = fx.Int32(gpu.grid_dim.x) // 8
            wend = xstart + xq + (xc < xr).select(fx.Int32(1), fx.Int32(0))
            first = xstart + bid // 8
            st = _tile_body(fx.max(fx.min(first, bound - 1), fx.Int32(0)), "pro")
            for w in range(first, wend, gx):
                st = _tile_body(
                    fx.Int32(w), "main", fx.min(fx.Int32(w) + gx, wend - 1), st
                )
            rocdl.wait_asyncmark(0)
        else:
            work0 = xstart + bid // 8
            if bid < bound:
                _tile_body(work0)

    _kbody = ASTRewriter.transform(_kbody)

    @flyc.kernel(name=name, known_block_size=[THREADS, 1, 1])
    def kern(
        a_ptr: fx.Int64,
        as_ptr: fx.Int64,
        b_ptr: fx.Int64,
        bs_ptr: fx.Int64,
        tiles_ptr: fx.Int64,
        ntiles_ptr: fx.Int64,
        rtok_ptr: fx.Int64,
        rw_ptr: fx.Int64,
        o_ptr: fx.Int64,
        os_ptr: fx.Int64,
        aux_ptr: fx.Int64,
        n_a_rows: fx.Int32,
        n_rows: fx.Int32,
        n_out: fx.Int32,
    ):
        if const_expr(
            name == ""
        ):  # name in the JIT cache key (it encodes every build param)
            pass
        lds_base = fx.Int32(
            fx.ptrtoint(fx.SharedAllocator().allocate(SharedStorage).peek().raw.ptr)
        )
        _kbody(
            a_ptr,
            as_ptr,
            b_ptr,
            bs_ptr,
            tiles_ptr,
            ntiles_ptr,
            rtok_ptr,
            rw_ptr,
            o_ptr,
            os_ptr,
            aux_ptr,
            n_a_rows,
            n_rows,
            n_out,
            lds_base,
        )

    @flyc.jit
    def launch(
        a_ptr: fx.Int64,
        as_ptr: fx.Int64,
        b_ptr: fx.Int64,
        bs_ptr: fx.Int64,
        tiles_ptr: fx.Int64,
        ntiles_ptr: fx.Int64,
        rtok_ptr: fx.Int64,
        rw_ptr: fx.Int64,
        o_ptr: fx.Int64,
        os_ptr: fx.Int64,
        aux_ptr: fx.Int64,
        n_a_rows: fx.Int32,
        n_rows: fx.Int32,
        n_out: fx.Int32,
        grid: fx.Int32,
        stream: fx.Stream = fx.Stream(None),  # noqa: B008
    ):
        kern(
            a_ptr,
            as_ptr,
            b_ptr,
            bs_ptr,
            tiles_ptr,
            ntiles_ptr,
            rtok_ptr,
            rw_ptr,
            o_ptr,
            os_ptr,
            aux_ptr,
            n_a_rows,
            n_rows,
            n_out,
        ).launch(grid=(grid, 1, 1), block=(THREADS, 1, 1), stream=stream)

    hints = {}
    if MV:
        # MV=1: no AGPRs (amdgpu-agpr-alloc=0): MFMA accumulators in arch VGPRs, no copies.
        hints["fn_attrs"] = {"amdgpu-agpr-alloc": "0"}
    wpe = [int(t[3:]) for t in DG if t.startswith("wpe") and t[3:].isdigit()]
    if wpe:
        # wpeN: cap registers so N waves/SIMD fit (e.g. wpe3: <= 168 VGPRs).
        hints["waves_per_eu"] = wpe[0]
    ag = [t[2:] for t in DG if t.startswith("ag") and t[2:].isdigit()]
    ag += [
        t[3:] for t in DG if epi == "fused" and t.startswith("agf") and t[3:].isdigit()
    ]
    if ag:
        # agN: AGPR share of the unified budget (LLVM default: half when AGPRs are used).
        hints["fn_attrs"] = dict(
            hints.get("fn_attrs", {}), **{"amdgpu-agpr-alloc": ag[0]}
        )
    if hints:
        launch.compile_hints = hints
    return launch, NB


class _Runner:
    def __init__(self, launch):
        self.launch = launch
        self.cf = None

    def __call__(self, *args):
        if self.cf is None:
            with primitives.fn_attr_hook(self.launch):
                self.cf = flyc.compile(self.launch, *args)
        else:
            self.cf(*args)


_runners = {}


def run_gemm(
    stage,
    K,
    N,
    BM,
    args,
    max_tiles,
    D=3,
    epi="rows",
    pipe="async",
    NW=4,
    diag="",
    WM=1,
    EF=False,
    MV=0,
    AST=False,
    HT=False,
    KTOP=5,
    PERS=0,
    stream=None,
):
    import torch

    key = (stage, K, N, BM, D, epi, pipe, NW, diag, WM, EF, MV, AST, HT, KTOP, PERS)
    if key not in _runners:
        launch, nb = build_gemm(
            stage, K, N, BM, D, epi, pipe, NW, diag, WM, EF, MV, AST, HT, KTOP, PERS
        )
        _runners[key] = (_Runner(launch), nb)
    r, nb = _runners[key]
    stream = torch.cuda.current_stream() if stream is None else stream
    grid = _num_cus() * PERS if PERS else max_tiles * nb
    r(*args, grid, stream)
