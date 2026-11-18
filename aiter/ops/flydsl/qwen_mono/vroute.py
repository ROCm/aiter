"""Routing only, to iterate the handoff with full stamps: the router CTAs (0..15) of vmoe2 run the router GEMM, rank and
publish exactly as vmoe2 does (its functions, its flags: MB, LWS, EWS ...); every other CTA only waits for the selection
and derives its unit assignment (assign + the token's top-8 rows). No experts, no shared expert, no combine.
Stamps: 0 entry, 20 router x in LDS, 21 router weights in, 22 GEMM + reduce, 6 logits out, 16 logit in hand, 17 after the
rank's CTA barrier, 18 rank known, 7 published, 19 collector done, 10 selection held by the wave."""
import os
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu
from flydsl.expr.typing import Int64
import hxdev as hx
from flydsl._mlir.dialects import llvm
import vmoe2 as V
from vmoe2 import (F32, I32, BF16, i64, vec, bfr, PW, LDS, TAB, NSTAMP, MB, A4, BANK, BANKB, NRANKED, TWS, _ufl)
from post import wmma_run, reduce8, st_f32x8
from vmoe import f1_weights
RV = os.environ.get("RV", "0")                    # variant tag for experiments in this file


@ASTRewriter.transform
def flow(c, P, cnt, stamp):
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    if b < 16:
        f1_weights(c, P)
        V.router_x(c)
        stamp(20)
        hx.wait_asynccnt0()
        stamp(21)
        tot = reduce8(c, wmma_run(c, 8, c["wbuf"], fx.Vector.filled(8, 0.0, F32)))
        stamp(22)
        if fx.const_expr(V.RT3):
            V.router_out3(c, P, tot)
        elif fx.const_expr(MB):
            V.router_out_mb(c, P, tot)
        else:
            if w == 0:
                st_f32x8(P["lg"] + i64((tok * 256 + b * 16 + (l // 16) * 8) * 4), [bfr(F32(tot[e])) for e in range(8)])
            hx.wait_storecnt0()
            gpu.barrier()
            hx.fence_release("cluster")
            hx.cluster_barrier()
            hx.fence_acquire("cluster")
        stamp(6)
        if fx.const_expr(V.RT3):
            V.rank3(c, P, cnt)
        else:
            V.rank_publish(c, P, cnt)
        stamp(7)
    if fx.const_expr(V.RT3):
        if w == 0:
            hx.wait_count(cnt + i64(NRANKED), 128, 24, ws=TWS)
            stamp(28)
            V.collect3(c, P)
            stamp(19)
        gpu.barrier()
    elif fx.const_expr(MB):
        if w == 0:
            V.mb_collect(c, P)
            stamp(19)
        gpu.barrier()
    else:
        if w == 0:
            hx.wait_count(cnt + i64(NRANKED), 16, 24, ws=TWS)
        gpu.barrier()
    stamp(27)
    uw = V.uw_load(c, P)
    A0 = V.assign(c, cnt)
    if c["t"] % 32 == 0:                                  # what the wave works with: U, bitmap word 2, sel[0] of its token
        hx.gstore(c["st"] + i64(b * 8 + w) * (NSTAMP * 8) + i64(24 * 8), i64(A0["U"]))
        hx.gstore(c["st"] + i64(b * 8 + w) * (NSTAMP * 8) + i64(25 * 8), i64(A0["W"][2]))
        hx.gstore(c["st"] + i64(b * 8 + w) * (NSTAMP * 8) + i64(26 * 8), i64(I32(uw[0][0])))
    stamp(10)


def build():
    keyed = ("vroute", 4, V.RT3, V.RT3F, V.MB32, V.D32, RV, MB, V.LWS, V.EWS, V.MBS, BANK, A4, TWS, hx.WSLEEP, V.XDEV, V.DLOADS)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(tab: Int64, cnt: Int64, st: Int64, xin: Int64, out: Int64, nrows: fx.Int32):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = _ufl(t // 32)
        P = {k: hx.sload64(tab + i64(8 * i)) for i, k in enumerate(TAB)}
        G = hx.gload(P["gen"], I32)
        cnt0 = cnt
        if fx.const_expr(BANK):
            cnt = cnt + i64((G & 1) * BANKB)
        c = dict(st=st, cnt=cnt, t=t, b=b, w=w, l=t % 32, sm=sm, wbuf=sm + w * PW, xbuf=sm + w * PW + 16384, xin=xin, out=out, nrows=_ufl(nrows), G=G)

        def stamp(k):
            if t % 32 == 0:
                hx.gstore(st + i64(b * 8 + w) * (NSTAMP * 8) + i64(k * 8), hx.clock())

        c["stamp"] = stamp
        hx.TOFLAG[0] = cnt + i64(2048)
        stamp(0)
        flow(c, P, cnt, stamp)
        if fx.const_expr(BANK):
            V.bank_zero(c, cnt0 + i64((1 - (G & 1)) * BANKB))
        V.mb_next(c, P)
        hx.wait_storecnt0()
        llvm.inline_asm(None, [], "s_wait_loadcnt 0x0", "", has_side_effects=True)
        stamp(5)
        _ = keyed

    @flyc.jit
    def launch(tab: Int64, cnt: Int64, st: Int64, xin: Int64, out: Int64, nrows: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(tab, cnt, st, xin, out, nrows).launch(grid=(256,), block=(256,), smem=LDS, cluster=(16, 1, 1), stream=stream)

    return launch
