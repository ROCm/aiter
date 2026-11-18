"""Verifier (target) MoE block, 1-16 rows, one launch, with the target's formats: router bf16, experts MXFP4 1x32,
shared expert MXFP8 1x32, activations MXFP8 1x32. Entry state as the stage before it leaves it: h1 [16][2048] bf16
(the residual stream), xg1 = h1 (1 + w) in x layout and plain, per-token partial sums of squares.
 F1  every CTA: rstd, x -> MXFP8 in LDS. Cluster 0 (CTAs 0..15): router tile b, cluster barrier, then CTA b ranks token
     b's 256 logits (two-level: in the wave, then among the 64 wave-local top-8) -> sel / wt (renormalised).
     CTAs 16..31: shared expert gate + up tile pairs -> silu(gate) up as MXFP8 (1x32). CTA 32: the shared gate row.
 F2  the union's experts gate/up (MXFP4), weight(tok, e) silu(gate) up -> MXFP8 (1x32); passes pipelined.
 F3  experts down (MXFP4) + shared down (MXFP8, times the shared gate) + h1 -> h2 (bf16)."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
import hxdev as hx
import post as Q
from post import (F32, I32, BF16, i64, vec, bfr, exp, sigmoid, burst, wmma_run, reduce8, reduce8b, rstd_reduce, st_f32x8, quant_rows,
                  fp8_code, fp8_pack, wmma_fp4, silu_mul, union, PW, RED, TMP, TMP2, ULIST, RS, RSTMP, XQ, XQS, LDS, HID)
from head import _ufl

TAB = "Wg Wsg SW SWS EW ES ED EDS SD SDS h1 xg1 xg1p part1 lg sgate sel wt GE GS GSU GSS h2".split()
NSTAMP = 16


def amax8(v):
    am = fx.max(v[0], F32(0.0) - v[0])
    for q in range(1, len(v)):
        am = fx.max(am, fx.max(v[q], F32(0.0) - v[q]))
    return am


@ASTRewriter.transform
def f1_weights(c, P):
    """This CTA's F1 weights -> its LDS slices (issued before the stage's handoff)."""
    b, w, l = c["b"], c["w"], c["l"]
    if b < 16:
        burst(c, P["Wg"] + i64(b) * 65536 + i64(w * 8192), 8, c["wbuf"], 2)
    else:
        if b < 32:
            for i in range_constexpr(4):                 # gate tiles 2j, 2j+1, up tiles 2j, 2j+1: this wave's 2 K128 steps
                tile = (i // 2) * 32 + (b - 16) * 2 + i % 2
                burst(c, P["SW"] + i64(tile * 16 + w * 2) * 2048, 4, c["wbuf"] + i * 4096, 2)
                if l < 8:
                    hx.async_to_lds_b128(P["SWS"] + i64(tile * 16 + w * 2) * 64 + i64(l * 16), c["xbuf"] + 8192 + i * 128 + l * 16, 2)
        else:
            if b == 32:
                burst(c, P["Wsg"] + i64(w * 8192), 8, c["wbuf"], 2)


@ASTRewriter.transform
def topk2(c, P):
    """CTA b = token b, thread t = expert t (its wave holds 32 consecutive logits). Local rank in the wave by 31 lane
    rotations; the wave's top 8 go to LDS; the 64 candidates rank each other (ties: lower id). Softmax over the top 8."""
    b, t, l, w, sm = c["b"], c["t"], c["l"], c["w"], c["sm"]
    my = hx.gload(P["lg"] + i64((b * 256 + t) * 4), F32, scope="agent")
    bits = my.bitcast(I32)
    lr = I32(0)
    for r in range(1, 32):
        p = (l + r) & 31
        o = hx.lane_gather(bits, p).bitcast(F32)
        lr = lr + ((o > my) | ((o == my) & (p < l))).select(I32(1), I32(0))
    if lr < 8:
        hx.lstore(sm + TMP + (w * 8 + lr) * 4, my)
        hx.lstore(sm + TMP + 256 + (w * 8 + lr) * 4, t)
    gpu.barrier()
    cv = [hx.lload(sm + TMP + i * 64, F32, 16) for i in range(4)]
    ci = [hx.lload(sm + TMP + 256 + i * 64, I32, 16) for i in range(4)]
    rank = I32(0)
    for i in range_constexpr(64):
        o = F32(cv[i // 16][i % 16])
        rank = rank + ((o > my) | ((o == my) & (I32(ci[i // 16][i % 16]) < t))).select(I32(1), I32(0))
    pick = (lr < 8) & (rank < 8)
    if pick & (rank == 0):
        hx.lstore(sm + RSTMP, my)
    gpu.barrier()
    e = exp(my - hx.lload(sm + RSTMP, F32))
    if pick:
        hx.lstore(sm + RSTMP + 64 + rank * 4, e)
    gpu.barrier()
    v = hx.lload(sm + RSTMP + 64, F32, 8)
    s = F32(v[0])
    for q in range_constexpr(1, 8):
        s = s + F32(v[q])
    if pick:
        hx.gstore(P["sel"] + i64((b * 8 + rank) * 4), t, scope="agent")
        hx.gstore(P["wt"] + i64((b * 8 + rank) * 4), e / s, scope="agent")


@ASTRewriter.transform
def stage_f1(c, P):
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    rstd_reduce(c, P["part1"], 256)
    quant_rows(c, P["xg1p"], sm + XQ, sm + XQS, False)
    rs = hx.lload(sm + RS + tok * 4, F32)
    if b < 16:
        burst(c, P["xg1"] + i64(w * 8192), 8, c["xbuf"], 0)
        hx.wait_asynccnt0()
        tot = reduce8(c, wmma_run(c, 8, c["wbuf"], fx.Vector.filled(8, 0.0, F32)))
        if w == 0:
            st_f32x8(P["lg"] + i64((tok * 256 + b * 16 + (l // 16) * 8) * 4), [F32(tot[e]) * rs for e in range(8)])
        hx.wait_storecnt0()
        gpu.barrier()
        hx.fence_release("cluster")
        hx.cluster_barrier()
        hx.fence_acquire("cluster")
        topk2(c, P)
    else:
        if b < 32:
            hx.wait_asynccnt0()
            gpu.barrier()                                 # (XQ written by every thread)
            acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
            for k in range_constexpr(2):
                stp = w * 2 + k
                bv = hx.lload(sm + XQ + stp * 2048 + l * 64, I32, 16)
                sb = hx.lload(sm + XQS + (stp * 16 + tok) * 4, I32)
                for i in range_constexpr(4):
                    acc[i] = hx.wmma_mx(hx.lload(c["wbuf"] + i * 4096 + k * 2048 + l * 64, I32, 16),
                                        hx.lload(c["xbuf"] + 8192 + i * 128 + (k * 16 + tok) * 4, I32), bv, sb, acc[i])
            for i in range_constexpr(4):
                hx.lstore(c["xbuf"] + i * 1024 + l * 32, acc[i])
            gpu.barrier()
            if w == 0:
                tot = []
                for i in range_constexpr(4):
                    s_ = hx.lload(sm + 16384 + i * 1024 + l * 32, F32, 8)
                    for q in range_constexpr(1, 8):
                        s_ = s_ + hx.lload(sm + q * PW + 16384 + i * 1024 + l * 32, F32, 8)
                    tot.append(s_)
                act = [silu_mul([bfr(tot[i][e]) for e in range(8)], [bfr(tot[2 + i][e]) for e in range(8)], F32(1.0)) for i in range(2)]
                am = amax8(act[0] + act[1])
                am = fx.max(am, hx.lane_gather(am.bitcast(I32), l ^ 16).bitcast(F32))
                code, inv = fp8_code(am)
                j = b - 16
                for i in range_constexpr(2):
                    lane2 = i * 16 + tok
                    hx.stx_dev(P["GSU"] + i64((j // 4) * 2048 + lane2 * 64 + (j % 4) * 16 + (l // 16) * 8), vec(fp8_pack(act[i], inv), I32))
                    if l < 16:
                        hx.gstore(P["GSS"] + i64(((j // 4) * 32 + lane2) * 16 + (j % 4) * 4), code, scope="agent")
            gpu.barrier()
        else:
            if b == 32:
                burst(c, P["xg1"] + i64(w * 8192), 8, c["xbuf"], 0)
                hx.wait_asynccnt0()
                tot = reduce8(c, wmma_run(c, 8, c["wbuf"], fx.Vector.filled(8, 0.0, F32)))
                if (w == 0) & (l < 16):
                    hx.gstore(P["sgate"] + i64(tok * 4), sigmoid(F32(tot[0]) * rs), scope="agent")
                gpu.barrier()


@ASTRewriter.transform
def stage_f2(c, P, U):
    """As post.stage_experts (unit = (union expert, tile), gate + up tile on a wave pair), the passes pipelined: the
    next pass's weights are in flight in the other 16 KB half while this pass runs; its scales follow the WMMAs."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    half = w % 2
    sl = hx.gload(P["sel"] + i64(tok * 32), I32, 8)
    wl = hx.gload(P["wt"] + i64(tok * 32), F32, 8)
    NP = (U * 32 + 1023) // 1024

    def unit(p):
        gc = fx.min(b * 4 + w // 2 + p * 1024, U * 32 - 1)
        e = _ufl(hx.lload(sm + ULIST + (gc // 32) * 4, I32))
        return gc, e, i64((e * 64 + gc % 32) * 16 + half * 8)

    def issue_w(p):
        _, _, t0 = unit(p)
        buf = c["wbuf"] + (p % 2) * 16384 + (p % 2) * 1024           # [0, 16K) or [17K, 33K)
        burst(c, P["EW"] + t0 * 1024, 8, buf, 2)
        burst(c, P["EW"] + (t0 + i64(32 * 16)) * 1024, 8, buf + 8192, 2)

    def issue_s(p):
        _, _, t0 = unit(p)
        hx.async_to_lds_b128(P["ES"] + t0 * 64 + i64(l * 16), c["wbuf"] + 16384 + l * 16, 2)
        hx.async_to_lds_b128(P["ES"] + (t0 + i64(32 * 16)) * 64 + i64(l * 16), c["wbuf"] + 16384 + 512 + l * 16, 2)

    if b * 4 < U * 32:
        issue_w(I32(0))
        issue_s(I32(0))
    for p in range(I32(0), NP, I32(1)):
        if b * 4 + p * 1024 < U * 32:
            gu = b * 4 + w // 2 + p * 1024
            live = gu < U * 32
            gc, e, _ = unit(p)
            i = gc % 32
            more = b * 4 + (p + 1) * 1024 < U * 32
            if more:
                issue_w(p + 1)
                rocdl.s_wait_asynccnt(32)
            else:
                hx.wait_asynccnt0()
            buf = c["wbuf"] + (p % 2) * 16384 + (p % 2) * 1024
            acc = fx.Vector.filled(8, 0.0, F32)
            acc2 = fx.Vector.filled(8, 0.0, F32)
            for k in range(8):
                stp = half * 8 + k
                bv = hx.lload(sm + XQ + stp * 2048 + l * 64, I32, 16)
                sb = hx.lload(sm + XQS + (stp * 16 + tok) * 4, I32)
                acc = wmma_fp4(hx.lload(buf + k * 1024 + l * 32, I32, 8), hx.lload(c["wbuf"] + 16384 + (k * 16 + tok) * 4, I32), bv, sb, acc)
                acc2 = wmma_fp4(hx.lload(buf + 8192 + k * 1024 + l * 32, I32, 8), hx.lload(c["wbuf"] + 16384 + 512 + (k * 16 + tok) * 4, I32), bv, sb, acc2)
            if more:
                issue_s(p + 1)
            hx.lstore(sm + RED + (w * 32 + l) * 32, acc)
            hx.lstore(sm + RED + 8192 + (w * 32 + l) * 32, acc2)
            gpu.barrier()
            act = [F32(0.0) for _ in range(8)]
            if (half == 0) & live:
                g = acc + hx.lload(sm + RED + ((w + 1) * 32 + l) * 32, F32, 8)
                u = acc2 + hx.lload(sm + RED + 8192 + ((w + 1) * 32 + l) * 32, F32, 8)
                wv = F32(0.0)
                for q in range_constexpr(8):
                    wv = wv + (I32(sl[q]) == e).select(F32(wl[q]), F32(0.0))
                act = silu_mul([bfr(g[q]) for q in range(8)], [bfr(u[q]) for q in range(8)], wv)
                am = amax8(act)
                am = fx.max(am, hx.lane_gather(am.bitcast(I32), l ^ 16).bitcast(F32))
                if l < 16:
                    hx.lstore(sm + TMP + ((w // 2) * 16 + tok) * 4, am)
            gpu.barrier()
            if (half == 0) & live:
                am = fx.max(hx.lload(sm + TMP + ((w // 2) * 16 + tok) * 4, F32), hx.lload(sm + TMP + (((w // 2) ^ 1) * 16 + tok) * 4, F32))
                code, inv = fp8_code(am)
                lane2 = (i % 2) * 16 + tok
                hx.stx_dev(P["GE"] + i64(gc // 32) * 8192 + i64((i // 8) * 2048 + lane2 * 64 + ((i % 8) // 2) * 16 + (l // 16) * 8), vec(fp8_pack(act, inv), I32))
                if l < 16:
                    hx.gstore(P["GS"] + i64((((gc // 32) * 4 + i // 8) * 32 + lane2) * 16 + ((i % 8) // 2) * 4), code, scope="agent")


@ASTRewriter.transform
def stage_f3(c, P, U):
    """b < 128: tile b of the experts' down (MXFP4, a slot double-buffered a wave) + the shared expert's down (MXFP8,
    waves 0..3 a K128 step each, its own accumulator: scaled by the shared gate) + h1 -> h2 bf16."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    NU = (U - w + 7) // 8

    def issue(j):
        e = _ufl(hx.lload(sm + ULIST + (w + j * 8) * 4, I32))
        buf = c["wbuf"] + (j % 2) * 16384
        t0 = i64((e * 128 + b) * 4)
        burst(c, P["ED"] + t0 * 1024, 4, buf, 2)
        hx.async_to_lds_b128(P["EDS"] + t0 * 64 + i64(l * 16), buf + 4096 + l * 16, 2)
        burst(c, P["GE"] + i64(w + j * 8) * 8192, 8, buf + 4608, 0)
        burst(c, P["GS"] + i64(w + j * 8) * 2048, 2, buf + 12800, 0)

    def xscale(base, k):
        sv = hx.lload(base + (k * 32 + l) * 16, I32, 4)
        return I32(sv[0]) | (I32(sv[1]) << 8) | (I32(sv[2]) << 16) | (I32(sv[3]) << 24)

    if b < 128:
        acc = fx.Vector.filled(8, 0.0, F32)
        accs = fx.Vector.filled(8, 0.0, F32)
        if NU > 0:
            issue(I32(0))
        for j in range(I32(0), NU, I32(1)):
            if j + 1 < NU:
                issue(j + 1)
                rocdl.s_wait_asynccnt(29)
            else:
                hx.wait_asynccnt0()
            buf = c["wbuf"] + (j % 2) * 16384
            for k in range(4):
                acc = wmma_fp4(hx.lload(buf + k * 1024 + l * 32, I32, 8), hx.lload(buf + 4096 + (k * 16 + tok) * 4, I32),
                               hx.lload(buf + 4608 + k * 2048 + l * 64, I32, 16), xscale(buf + 12800, k), acc)
        if w < 4:                                   # the shared expert: K128 step w
            burst(c, P["SD"] + i64(b * 4 + w) * 2048, 2, c["wbuf"], 2)
            if l < 4:
                hx.async_to_lds_b128(P["SDS"] + i64(b * 4 + w) * 64 + i64(l * 16), c["wbuf"] + 4096 + l * 16, 2)
            burst(c, P["GSU"] + i64(w * 2048), 2, c["wbuf"] + 8192, 0)
            hx.async_to_lds_b128(P["GSS"] + i64(w * 512 + l * 16), c["wbuf"] + 12288 + l * 16, 0)
            hx.wait_asynccnt0()
            accs = hx.wmma_mx(hx.lload(c["wbuf"] + l * 64, I32, 16), hx.lload(c["wbuf"] + 4096 + tok * 4, I32),
                              hx.lload(c["wbuf"] + 8192 + l * 64, I32, 16), xscale(c["wbuf"] + 12288, 0), accs)
        tot = reduce8(c, acc)
        tots = reduce8b(c, accs, 1)
        if w == 0:
            col = b * 16 + (l // 16) * 8
            sg = hx.gload(P["sgate"] + i64(tok * 4), F32)
            h1 = hx.gload(P["h1"] + i64((tok * HID + col) * 2), BF16, 8).to(F32)
            hx.stx_dev(P["h2"] + i64((tok * HID + col) * 2), vec([bfr(F32(h1[e]) + F32(tot[e]) + sg * F32(tots[e])) for e in range(8)]).to(BF16))
        gpu.barrier()


def build():
    keyed = ("vmoe", 2)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(tab: Int64, cnt: Int64, st: Int64):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = t // 32
        c = dict(t=t, b=b, w=w, l=t % 32, sm=sm, wbuf=sm + w * PW, xbuf=sm + w * PW + 16384)
        P = {k: hx.sload64(tab + i64(8 * i)) for i, k in enumerate(TAB)}

        def stamp(k):
            if t == 0:
                hx.gstore(st + i64(b) * (NSTAMP * 8) + i64(k * 8), hx.clock())

        stamp(0)
        f1_weights(c, P)
        hx.handoff(cnt, t, True)
        stamp(1)
        stage_f1(c, P)
        hx.handoff(cnt + i64(64), t, True)
        stamp(2)
        U = union(c, P, 16)
        stage_f2(c, P, U)
        hx.handoff(cnt + i64(128), t, True)
        stamp(3)
        stage_f3(c, P, U)
        hx.handoff(cnt + i64(192), t, True)
        stamp(4)
        _ = keyed

    @flyc.jit
    def launch(tab: Int64, cnt: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(tab, cnt, st).launch(grid=(256,), block=(256,), smem=LDS, cluster=(16, 1, 1), stream=stream)

    return launch
