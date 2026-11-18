"""Drafter LM head, MXFP8 1x32 weights (e4m3 + e8m0 a 32-col block) x MXFP8 x, argmax in the epilogue. T <= 16 rows.
Same structure as head.py (LDS FIFO stream, runtime loops, equal work a wave), K128 scaled WMMA steps of 2 KB a tile.
Layouts (pack_*): W [grp][step][tile][lane = h*16+row][4 c][16] (lane holds K 128 s + 32 c + 16 h + 0..15),
W scales [grp][step][tile][row][4]; x [step][lane = h*16+tok][4 c][16], x scales [step][tok][4]."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
import hxdev as hx
from head import key64, _ufl, i64, IMIN, WTOT, THREADS

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
latomic_max64, gatomic_max64 = hx.latomic_max64, hx.gatomic_max64


KH = __import__("os").environ.get("HEAD_KH", "0") == "1"   # x operand of a K half held in registers (see head_argmax8)


def pack_w(w8, ws, TW, GW=None):
    """GW given (and KH): inside a wave's region the order is [K half][tile group][8 steps] instead of [tile group][16]."""
    n, k = w8.shape
    a = w8.view(n // (16 * TW), TW, 16, k // 128, 4, 2, 16).permute(0, 3, 1, 5, 2, 4, 6).contiguous()
    s = ws.view(n // (16 * TW), TW, 16, k // 128, 4).permute(0, 3, 1, 2, 4).contiguous()
    if KH and GW is not None:
        ns = k // 128
        a = a.view(-1, GW, 2, ns // 2, *a.shape[2:]).transpose(1, 2).contiguous()
        s = s.view(-1, GW, 2, ns // 2, *s.shape[2:]).transpose(1, 2).contiguous()
    return a, s


def pack_x(x8, xs):
    t, k = x8.shape
    return x8.view(t, k // 128, 4, 2, 16).permute(1, 3, 0, 2, 4).contiguous(), xs.view(t, k // 128, 4).permute(1, 0, 2).contiguous()


def geom(N, K, TW, GW, B):
    NS = K // 128
    S = GW * NS
    SB = TW * 2048
    XB = NS * 2048                       # x operands
    XSC = XB + NS * 64                   # x scales end
    RED = XSC
    WSC = RED + 256                      # W scales: 8 waves x S * TW * 64
    PART = WSC + 8 * S * TW * 64         # K-half-0 partial sums: 8 waves x GW * TW x 1 KB
    FIFO = PART + (8 * GW * TW * 1024 if KH else 0)
    return dict(NS=NS, S=S, SB=SB, XB=XB, RED=RED, WSC=WSC, PART=PART, FIFO=FIFO, LDS=FIFO + 8 * B * SB, NW=N // (16 * TW * GW))


@ASTRewriter.transform
def head_argmax8(x8, xs, wb, wsb, out, t, b, sm, N, K, B, TW, GW, st=None, nt=True):
    G = geom(N, K, TW, GW, B)
    NS, S, SB = G["NS"], G["S"], G["SB"]
    l = t % 32
    w = t // 32
    row = l % 16
    half = l // 16
    g = _ufl(b * 8 + w)
    base = wb + i64(g) * (S * SB)
    fifo = sm + G["FIFO"] + w * (B * SB)
    wsc = sm + G["WSC"] + w * (S * TW * 64)
    cpol = 2 if nt else 0

    def issue(s):
        for q in range_constexpr(SB // 512):
            hx.async_to_lds_b128(base + i64(s) * SB + i64(q * 512 + l * 16), fifo + (s % B) * SB + q * 512 + l * 16, cpol)

    if t < 16:
        hx.lstore(sm + G["RED"] + t * 8, i64(IMIN))
    if fx.const_expr(x8 is not None):                   # (None: x operands and scales are already in LDS)
        for it in range(NS * 2048 // 16 // THREADS):    # x operands -> LDS
            hx.async_to_lds_b128(x8 + i64(t + it * THREADS) * 16, sm + (t + it * THREADS) * 16, 0)
        if t < NS * 4:                                  # x scales (NS * 64 B)
            hx.async_to_lds_b128(xs + i64(t) * 16, sm + G["XB"] + t * 16, 0)
    bestv = F32(-3.0e38)
    besti = I32(0)
    if g < G["NW"]:
        for q in range(S * TW * 64 // 512):             # this wave's weight scales, one burst
            hx.async_to_lds_b128(wsb + i64(g) * (S * TW * 64) + i64(q * 512 + l * 16), wsc + q * 512 + l * 16, cpol)
        for s0 in range_constexpr(B - 1):
            issue(I32(s0))
        hx.wait_asynccnt0()
        gpu.barrier()
        if fx.const_expr(st is not None):
            if t == 0:
                hx.gstore(st + i64(b) * 32 + i64(16), hx.clock())
        if fx.const_expr(KH):
            # K half hf of every tile group, then the other half: the half's x operands (8 K128 steps, 128 registers)
            # and scales are read from LDS once, not once a WMMA. Partial sums of half 0 wait in LDS.
            part = sm + G["PART"] + w * (GW * TW * 1024)
            for hf in range_constexpr(2):
                xr = [hx.lload(sm + (hf * 8 + k) * 2048 + l * 64, I32, 16) for k in range(8)]
                xsc = [hx.lload(sm + G["XB"] + ((hf * 8 + k) * 16 + row) * 4, I32) for k in range(8)]
                for tg in range(GW):
                    acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(TW)]
                    if fx.const_expr(hf == 1):
                        acc = [hx.lload(part + (tg * TW + i) * 1024 + l * 32, F32, 8) for i in range(TW)]
                    for k in range_constexpr(8):
                        s = (tg + hf * GW) * 8 + k
                        issue(fx.min(s + (B - 1), I32(S - 1)))
                        rocdl.s_wait_asynccnt((B - 1) * (SB // 512))
                        a = [hx.lload(fifo + (s % B) * SB + i * 2048 + l * 64, I32, 16) for i in range(TW)]
                        sa = [hx.lload(wsc + ((s * TW + i) * 16 + row) * 4, I32) for i in range(TW)]
                        for i in range_constexpr(TW):
                            acc[i] = hx.wmma_mx(a[i], sa[i], xr[k], xsc[k], acc[i])
                    if fx.const_expr(hf == 0):
                        for i in range_constexpr(TW):
                            hx.lstore(part + (tg * TW + i) * 1024 + l * 32, acc[i])
                    else:
                        for i in range_constexpr(TW):
                            tile = (g * GW + tg) * TW + i
                            for e in range_constexpr(8):
                                v = F32(acc[i][e]).to(BF16).to(F32)
                                gt = v > bestv
                                bestv = gt.select(v, bestv)
                                besti = gt.select(tile * 16 + half * 8 + e, besti)
        else:
            acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(TW)]
            for s in range(S):
                issue(fx.min(s + (B - 1), I32(S - 1)))
                rocdl.s_wait_asynccnt((B - 1) * (SB // 512))
                ks = s % NS
                bv = hx.lload(sm + ks * 2048 + l * 64, I32, 16)
                sb = hx.lload(sm + G["XB"] + (ks * 16 + row) * 4, I32)
                a = [hx.lload(fifo + (s % B) * SB + i * 2048 + l * 64, I32, 16) for i in range(TW)]
                sa = [hx.lload(wsc + ((s * TW + i) * 16 + row) * 4, I32) for i in range(TW)]
                for i in range_constexpr(TW):
                    acc[i] = hx.wmma_mx(a[i], sa[i], bv, sb, acc[i])
                if (s + 1) % NS == 0:
                    for i in range_constexpr(TW):
                        tile = (g * GW + s // NS) * TW + i
                        for e in range_constexpr(8):
                            v = F32(acc[i][e]).to(BF16).to(F32)
                            gt = v > bestv
                            bestv = gt.select(v, bestv)
                            besti = gt.select(tile * 16 + half * 8 + e, besti)
                        acc[i] = fx.Vector.filled(8, 0.0, F32)
        hx.wait_asynccnt0()
        if fx.const_expr(st is not None):
            if t == 0:
                hx.gstore(st + i64(b) * 32 + i64(24), hx.clock())
        latomic_max64(sm + G["RED"] + row * 8, key64(bestv, besti))
    else:
        hx.wait_asynccnt0()
        gpu.barrier()
    gpu.barrier()
    if t < 16:
        gatomic_max64(out + i64(t) * 8, hx.lload(sm + G["RED"] + t * 8, fx.Int64))


def build(N=248320, K=2048, B=3, TW=4, GW=2, nt=True):
    keyed = (N, K, B, TW, GW, nt, "mx8", KH)
    G = geom(N, K, TW, GW, B)
    assert N % (16 * TW * GW) == 0 and G["NW"] <= WTOT and G["LDS"] <= 320 * 1024, G

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def body(x8: Int64, xs: Int64, wb: Int64, wsb: Int64, out: Int64, st: Int64):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        if t == 0:
            hx.gstore(st + i64(b) * 32, hx.clock())
        head_argmax8(x8, xs, wb, wsb, out, t, b, sm, N, K, B, TW, GW, st, nt)
        if t == 0:
            hx.gstore(st + i64(b) * 32 + i64(8), hx.clock())
        _ = keyed

    @flyc.jit
    def launch(x8: Int64, xs: Int64, wb: Int64, wsb: Int64, out: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(x8, xs, wb, wsb, out, st).launch(grid=(256,), block=(THREADS,), smem=G["LDS"], cluster=(16, 1, 1), stream=stream)

    return launch, G
