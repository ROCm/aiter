"""Drafter post-attention block with the model's math, one launch, R <= 16 rows: gated attention output -> o_proj +
residual -> post-attention norm -> router / shared expert gate-up / shared gate -> top-8 -> the union's experts
gate-up -> down (experts and shared expert as one K-concatenated GEMM) + residual -> final norm -> MXFP8 LM head + argmax.
Every loop is a runtime loop. Norms: the producer stores x (1 + w) and per-token partial sums of squares; consumers
rebuild rstd and scale their (linear) outputs by it, so a norm needs no stage of its own.
tab (i64): see TAB. Layouts as stage.py (W [tile][kstep][lane][16], x [kstep][lane = h*16+tok][16])."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
import hxdev as hx
import head8 as H8
from head import _ufl

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
TAB = "Wo Wr EW ES ED EDS SD x0 resid h1 xg1 xg1p part1 lg gsu sgate sel wt GE GS xf part2 g1 gf w8 ws8 out".split()
PW = 33 * 1024
RED, TMP, TMP2, ULIST, RS, RSTMP = 8 * PW, 8 * PW + 16384, 8 * PW + 17408, 8 * PW + 18432, 8 * PW + 19456, 8 * PW + 19520
XQ = 8 * PW + 20480          # the experts' input, MXFP8: 16 K128 steps x 2 KB, then its scales [step][tok] i32
XQS = XQ + 32768
LDS = XQS + 1024
EPS, HID = 1e-6, 2048
NSTAMP = 16
HEAD = dict(TW=2, GW=4, B=3)


def i64(x):
    return fx.Int64(x)


def rsq(v):
    return F32(hx.intrin(hx.T.f32, "llvm.amdgcn.rsq.f32", [F32(v)]))


def exp(v):
    return F32(hx.intrin(hx.T.f32, "llvm.amdgcn.exp2.f32", [F32(v) * 1.4426950408889634]))


def sigmoid(v):
    return F32(1.0) / (F32(1.0) + exp(F32(0.0) - v))


def bfr(v):
    return F32(v).to(BF16).to(F32)


def vec(vals, kind=F32):
    return fx.Vector.from_elements(list(vals), kind)


def st_f32x8(addr, vals):
    hx.stx_dev(addr, vec(vals[:4]))
    hx.stx_dev(addr + i64(16), vec(vals[4:]))


@ASTRewriter.transform
def burst(c, src, kb, dst, cpol):
    """kb KB at src -> LDS dst, 16 B a lane an instruction."""
    for q in range(kb * 2):
        hx.async_to_lds_b128(src + i64(q * 512 + c["l"] * 16), dst + q * 512 + c["l"] * 16, cpol)


@ASTRewriter.transform
def wmma_run(c, ck, wbuf, acc):
    for k in range(ck):
        a = hx.lload(wbuf + k * 1024 + c["l"] * 32, BF16, 16)
        bv = hx.lload(c["xbuf"] + k * 1024 + c["l"] * 32, BF16, 16)
        acc = hx.wmma_bf16(a, bv, acc)
    return acc


def reduce8(c, acc):
    """The CTA's 8 partial accumulators summed in wave order (every wave gets the sum)."""
    hx.lstore(c["sm"] + RED + (c["w"] * 32 + c["l"]) * 32, acc)
    gpu.barrier()
    tot = hx.lload(c["sm"] + RED + c["l"] * 32, F32, 8)
    for q in range(1, 8):
        tot = tot + hx.lload(c["sm"] + RED + (q * 32 + c["l"]) * 32, F32, 8)
    return tot


@ASTRewriter.transform
def rstd_reduce(c, part, nparts):
    """rstd[tok] = rsqrt(sum of the partials / HID + eps) -> LDS RS (every CTA, same order: deterministic).
    part [16 tok][nparts] f32: wave w sums its nparts / 8 (8 loads of 16 B issued together), then the 8 wave sums."""
    sm, w, l = c["sm"], c["w"], c["l"]
    tok = l % 16
    pw = nparts // 8
    vs = [hx.gload(part + i64((tok * nparts + w * pw + q * 4) * 4), F32, 4) for q in range(pw // 4)]
    s = F32(0.0)
    for v in vs:
        s = s + ((F32(v[0]) + F32(v[1])) + (F32(v[2]) + F32(v[3])))
    if l < 16:
        hx.lstore(sm + RSTMP + (w * 16 + tok) * 4, s)
    gpu.barrier()
    tot = hx.lload(sm + RSTMP + tok * 4, F32)
    for q in range_constexpr(1, 8):
        tot = tot + hx.lload(sm + RSTMP + (q * 16 + tok) * 4, F32)
    if (w == 0) & (l < 16):
        hx.lstore(sm + RS + tok * 4, rsq(tot * (1.0 / HID) + EPS))
    gpu.barrier()


def st_x(addr, tile, c, vals):
    """8 values (rows (l // 16) * 8 + e of `tile`, token l % 16) bf16 -> an x-layout buffer."""
    l = c["l"]
    hx.stx_dev(addr + i64(tile // 2) * 1024 + i64(((tile % 2) * 16 + l % 16) * 32 + (l // 16) * 16), vec(vals).to(BF16))


def silu_mul(g, u, wv):
    return [wv * (F32(g[q]) * sigmoid(F32(g[q]))) * F32(u[q]) for q in range(8)]


def reduce8b(c, acc, slot):
    """reduce8 on the second LDS partials region (two accumulators a pass)."""
    off = c["sm"] + RED + slot * 8192
    hx.lstore(off + (c["w"] * 32 + c["l"]) * 32, acc)
    gpu.barrier()
    tot = hx.lload(off + c["l"] * 32, F32, 8)
    for q in range(1, 8):
        tot = tot + hx.lload(off + (q * 32 + c["l"]) * 32, F32, 8)
    return tot


@ASTRewriter.transform
def router_w(c, P):
    """This CTA's weights of the router stage -> wbuf: b < 16 router tile b; 16 <= b < 48 shared gate tile b - 16 and
    up tile b - 16 (two 8 KB bursts); b == 48 the shared-expert gate row's tile."""
    b, w = c["b"], c["w"]
    if b < 16:
        burst(c, P["Wr"] + i64(b) * 65536 + i64(w * 8192), 8, c["wbuf"], 2)
    else:
        if b < 48:
            burst(c, P["Wr"] + i64(b) * 65536 + i64(w * 8192), 8, c["wbuf"], 2)
            burst(c, P["Wr"] + i64(b + 32) * 65536 + i64(w * 8192), 8, c["wbuf"] + 8192, 2)
        else:
            if b == 48:
                burst(c, P["Wr"] + i64(80) * 65536 + i64(w * 8192), 8, c["wbuf"], 2)


@ASTRewriter.transform
def stage_oproj(c, P):
    """b < 128: tile b of o_proj; + residual, h1 (bf16) stored, h1 (1 + w) -> xg1, partial sums of squares."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    acc = fx.Vector.filled(8, 0.0, F32)
    if b < 128:
        burst(c, P["x0"] + i64(w * 16384), 16, c["xbuf"], 0)
        hx.wait_asynccnt0()
        acc = wmma_run(c, 16, c["wbuf"], acc)
    router_w(c, P)                                       # the next stage's weights, once wbuf is free
    if b < 128:
        tot = reduce8(c, acc)
        if w == 0:
            tok = l % 16
            col = b * 16 + (l // 16) * 8
            r = hx.gload(P["resid"] + i64((tok * HID + col) * 2), BF16, 8).to(F32)
            g = hx.gload(P["g1"] + i64(col * 4), F32, 8)
            h = [bfr(bfr(tot[e]) + F32(r[e])) for e in range(8)]
            hx.stx_dev(P["h1"] + i64((tok * HID + col) * 2), vec(h).to(BF16))
            hg = [h[e] * F32(g[e]) for e in range(8)]
            st_x(P["xg1"], b, c, hg)
            hx.stx_dev(P["xg1p"] + i64((tok * HID + col) * 2), vec(hg).to(BF16))
            ss = h[0] * h[0]
            for e in range_constexpr(1, 8):
                ss = ss + h[e] * h[e]
            hx.gstore(P["part1"] + i64((tok * 256 + b * 2 + l // 16) * 4), ss, scope="agent")
        gpu.barrier()


@ASTRewriter.transform
def stage_router(c, P):
    """Router logits (f32), the shared expert's silu(gate) up (bf16 x layout, K 512, unweighted), the shared gate."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    if b < 49:
        burst(c, P["xg1"] + i64(w * 8192), 8, c["xbuf"], 0)
        hx.wait_asynccnt0()
        tok = l % 16
        rs = hx.lload(sm + RS + tok * 4, F32)
        acc = wmma_run(c, 8, c["wbuf"], fx.Vector.filled(8, 0.0, F32))
        if (b >= 16) & (b < 48):
            acc2 = wmma_run(c, 8, c["wbuf"] + 8192, fx.Vector.filled(8, 0.0, F32))
            tot = reduce8(c, acc)
            tot2 = reduce8b(c, acc2, 1)
            if w == 0:
                st_x(P["gsu"], b - 16, c, silu_mul([bfr(F32(tot[e]) * rs) for e in range(8)], [bfr(F32(tot2[e]) * rs) for e in range(8)], F32(1.0)))
        else:
            tot = reduce8(c, acc)
            if w == 0:
                v = [F32(tot[e]) * rs for e in range(8)]
                if b < 16:
                    st_f32x8(P["lg"] + i64((tok * 256 + b * 16 + (l // 16) * 8) * 4), v)
                else:
                    if l < 16:
                        hx.gstore(P["sgate"] + i64(tok * 4), sigmoid(v[0]), scope="agent")
        gpu.barrier()


@ASTRewriter.transform
def stage_topk(c, P):
    """CTA b < 16 = token b, thread t = expert t: softmax over the 256 logits, the top 8 by rank (ties: lower id)."""
    b, t, sm = c["b"], c["t"], c["sm"]
    if b < 16:
        my = hx.gload(P["lg"] + i64((b * 256 + t) * 4), F32)
        hx.lstore(sm + TMP + t * 4, my)
        gpu.barrier()
        if fx.const_expr(c.get('st') is not None):
            if t == 0:
                hx.gstore(c['st'] + i64(b) * (NSTAMP * 8) + i64(9 * 8), hx.clock())
        rank = I32(0)
        mx = my
        for j in range(16):
            v = hx.lload(sm + TMP + j * 64, F32, 16)
            for q in range_constexpr(16):
                o = F32(v[q])
                rank = rank + ((o > my) | ((o == my) & (j * 16 + q < t))).select(I32(1), I32(0))
                mx = fx.max(mx, o)
        if fx.const_expr(c.get('st') is not None):
            if t == 0:
                hx.gstore(c['st'] + i64(b) * (NSTAMP * 8) + i64(10 * 8), hx.clock())
        e = exp(my - mx)
        hx.lstore(sm + TMP2 + t * 4, e)
        gpu.barrier()
        if fx.const_expr(c.get('st') is not None):
            if t == 0:
                hx.gstore(c['st'] + i64(b) * (NSTAMP * 8) + i64(11 * 8), hx.clock())
        if t < 16:
            v = hx.lload(sm + TMP2 + t * 64, F32, 16)
            rsum = F32(v[0])
            for q in range_constexpr(1, 16):
                rsum = rsum + F32(v[q])
            hx.lstore(sm + TMP + t * 4, rsum)
        gpu.barrier()
        if fx.const_expr(c.get('st') is not None):
            if t == 0:
                hx.gstore(c['st'] + i64(b) * (NSTAMP * 8) + i64(12 * 8), hx.clock())
        v = hx.lload(sm + TMP, F32, 16)
        s = F32(v[0])
        for q in range_constexpr(1, 16):
            s = s + F32(v[q])
        if rank < 8:
            hx.gstore(P["sel"] + i64((b * 8 + rank) * 4), t, scope="agent")
            hx.gstore(P["wt"] + i64((b * 8 + rank) * 4), e / s, scope="agent")
        gpu.barrier()
        if fx.const_expr(c.get('st') is not None):
            if t == 0:
                hx.gstore(c['st'] + i64(b) * (NSTAMP * 8) + i64(13 * 8), hx.clock())


@ASTRewriter.transform
def union(c, P, R):
    """Every CTA: the sorted union of the R tokens' experts -> LDS ULIST; returns its size (wave-uniform)."""
    t, sm = c["t"], c["sm"]
    if t < 32:
        hx.async_to_lds_b128(P["sel"] + i64(t * 16), sm + TMP2 + t * 16, 0)
    hx.wait_asynccnt0()
    gpu.barrier()
    vs = [hx.lload(sm + TMP2 + i * 64, I32, 16) for i in range((R + 1) // 2)]
    flag = I32(0)
    for i in range_constexpr(R * 8):
        flag = flag | (I32(vs[i // 16][i % 16]) == t).select(I32(1), I32(0))
    hx.lstore(sm + TMP + t * 4, flag)
    gpu.barrier()
    pos = I32(0)
    tot = I32(0)
    for j in range(16):
        v = hx.lload(sm + TMP + j * 64, I32, 16)
        for q in range_constexpr(16):
            pos = pos + (j * 16 + q < t).select(I32(v[q]), I32(0))
            tot = tot + I32(v[q])
    if flag != 0:
        hx.lstore(sm + ULIST + pos * 4, t)
    gpu.barrier()
    return _ufl(tot)


def wmma_fp4(a8, sa, b, sb, acc):
    """Scaled WMMA, A = MXFP4 (a lane: K blocks h and h + 2 of its row, 16 B each), B = MXFP8."""
    a16 = vec([a8[i] if i < 8 else I32(0) for i in range(16)], I32)
    r = rocdl.wmma_scale_f32_16x16x128_f8f6f4(hx.vty(8, hx.T.f32), hx._v(a16), hx._v(b), hx._v(acc), hx._v(I32(sa)), hx._v(I32(sb)), fmtA=4, fmtB=0)
    return fx.Vector(r, (8,), F32)


def fp8_code(am):
    """E8M0 code of an MXFP8 block with max |x| = am, and 1 / scale."""
    code = fx.max(fx.min(((am.bitcast(I32) >> 23) & 255) - 8, I32(254)), I32(1))
    return code, ((I32(254) - code) << 23).bitcast(F32)


def fp8_pack(vals, inv):
    """4 k f32 -> k dwords of e4m3 (clamped)."""
    x = [fx.max(fx.min(v * inv, F32(448.0)), F32(-448.0)) for v in vals]
    return [hx.cvt_pk_fp8_w(hx.cvt_pk_fp8_w(I32(0), x[4 * d], x[4 * d + 1], False), x[4 * d + 2], x[4 * d + 3], True) for d in range(len(vals) // 4)]


@ASTRewriter.transform
def stage_experts(c, P, U):
    """MXFP4 experts (as ATOM runs them), x = the normed input in MXFP8 (LDS XQ). A unit = (union expert u, tile i < 32):
    its gate tile and up tile, K split over a wave pair (8 K128 steps = 8 KB a tile a wave). Unit gu = 4 b + w // 2 +
    1024 p < 32 U. The epilogue stores weight(tok, expert) silu(gate) up as FP8 with a scale a 16-col block (a lane's
    own scale in the down stage's B operand) -> GE [u][4 steps][lane][64 B], GS [u][4][lane][4 pieces] i32."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    half = w % 2
    sl = hx.gload(P["sel"] + i64(tok * 32), I32, 8)
    wl = hx.gload(P["wt"] + i64(tok * 32), F32, 8)
    for p in range(I32(0), (U * 32 + 1023) // 1024, I32(1)):
        gu = b * 4 + w // 2 + p * 1024
        if b * 4 + p * 1024 < U * 32:                      # (CTA-uniform: its barriers)
            live = gu < U * 32
            gc = fx.min(gu, U * 32 - 1)
            e = _ufl(hx.lload(sm + ULIST + (gc // 32) * 4, I32))
            i = gc % 32
            t0 = i64((e * 64 + i) * 16 + half * 8)
            burst(c, P["EW"] + t0 * 1024, 8, c["wbuf"], 2)
            burst(c, P["EW"] + (t0 + i64(32 * 16)) * 1024, 8, c["wbuf"] + 8192, 2)
            hx.async_to_lds_b128(P["ES"] + t0 * 64 + i64(l * 16), c["xbuf"] + l * 16, 2)
            hx.async_to_lds_b128(P["ES"] + (t0 + i64(32 * 16)) * 64 + i64(l * 16), c["xbuf"] + 512 + l * 16, 2)
            hx.wait_asynccnt0()
            acc = fx.Vector.filled(8, 0.0, F32)
            acc2 = fx.Vector.filled(8, 0.0, F32)
            for k in range(8):
                st = half * 8 + k
                bv = hx.lload(sm + XQ + st * 2048 + l * 64, I32, 16)
                sb = hx.lload(sm + XQS + (st * 16 + tok) * 4, I32)
                acc = wmma_fp4(hx.lload(c["wbuf"] + k * 1024 + l * 32, I32, 8), hx.lload(c["xbuf"] + (k * 16 + tok) * 4, I32), bv, sb, acc)
                acc2 = wmma_fp4(hx.lload(c["wbuf"] + 8192 + k * 1024 + l * 32, I32, 8), hx.lload(c["xbuf"] + 512 + (k * 16 + tok) * 4, I32), bv, sb, acc2)
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
                am = fx.max(act[0], F32(0.0) - act[0])
                for q in range_constexpr(1, 8):
                    am = fx.max(am, fx.max(act[q], F32(0.0) - act[q]))
                am = fx.max(am, hx.lane_gather(am.bitcast(I32), l ^ 16).bitcast(F32))
                if l < 16:
                    hx.lstore(sm + TMP + ((w // 2) * 16 + tok) * 4, am)
            gpu.barrier()
            if (half == 0) & live:
                # a 32-col block = this tile and its sibling (tiles 2 j, 2 j + 1: adjacent wave pairs of this CTA)
                am = fx.max(hx.lload(sm + TMP + ((w // 2) * 16 + tok) * 4, F32), hx.lload(sm + TMP + (((w // 2) ^ 1) * 16 + tok) * 4, F32))
                code, inv = fp8_code(am)
                lane2 = (i % 2) * 16 + tok
                dst = P["GE"] + i64(gc // 32) * 8192 + i64((i // 8) * 2048 + lane2 * 64 + ((i % 8) // 2) * 16 + (l // 16) * 8)
                hx.stx_dev(dst, vec(fp8_pack(act, inv), I32))
                if l < 16:
                    hx.gstore(P["GS"] + i64((((gc // 32) * 4 + i // 8) * 32 + lane2) * 16 + ((i % 8) // 2) * 4), code, scope="agent")
            gpu.barrier()


@ASTRewriter.transform
def stage_down(c, P, U):
    """b < 128: tile b of out = sum over the union's experts of Down_s . x_s (MXFP4 x FP8; a slot = 4 K128 steps: 4 KB
    of weights + 256 B scales cold, 8 KB x + 2 KB scales from L2, double-buffered a wave) + the shared expert's
    down (bf16, K 512 split over the 8 waves, x scaled by the shared gate). + h1 -> h2, h2 (1 + w_f) -> xf, partials."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    NU = (U - w + 7) // 8                          # this wave's slots: w, w + 8, ...

    def issue(j):
        e = _ufl(hx.lload(sm + ULIST + (w + j * 8) * 4, I32))
        buf = c["wbuf"] + (j % 2) * 16384
        t0 = i64((e * 128 + b) * 4)
        burst(c, P["ED"] + t0 * 1024, 4, buf, 2)
        hx.async_to_lds_b128(P["EDS"] + t0 * 64 + i64(l * 16), buf + 4096 + l * 16, 2)
        burst(c, P["GE"] + i64(w + j * 8) * 8192, 8, buf + 4608, 0)
        burst(c, P["GS"] + i64(w + j * 8) * 2048, 2, buf + 12800, 0)

    if b < 128:
        acc = fx.Vector.filled(8, 0.0, F32)
        sg = hx.gload(P["sgate"] + i64(tok * 4), F32)
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
                sv = hx.lload(buf + 12800 + (k * 32 + l) * 16, I32, 4)
                sb = I32(sv[0]) | (I32(sv[1]) << 8) | (I32(sv[2]) << 16) | (I32(sv[3]) << 24)
                acc = wmma_fp4(hx.lload(buf + k * 1024 + l * 32, I32, 8), hx.lload(buf + 4096 + (k * 16 + tok) * 4, I32),
                               hx.lload(buf + 4608 + k * 2048 + l * 64, I32, 16), sb, acc)
        # the shared expert: bf16, this wave's 2 k-steps
        burst(c, P["SD"] + i64(b) * 16384 + i64(w * 2048), 2, c["wbuf"], 2)
        burst(c, P["gsu"] + i64(w * 2048), 2, c["wbuf"] + 4096, 0)
        hx.wait_asynccnt0()
        for k in range_constexpr(2):
            v = hx.lload(c["wbuf"] + 4096 + k * 1024 + l * 32, BF16, 16).to(F32)
            acc = hx.wmma_bf16(hx.lload(c["wbuf"] + k * 1024 + l * 32, BF16, 16), vec([F32(v[q]) * sg for q in range(16)]).to(BF16), acc)
        tot = reduce8(c, acc)
        if w == 0:
            col = b * 16 + (l // 16) * 8
            h1 = hx.gload(P["h1"] + i64((tok * HID + col) * 2), BF16, 8).to(F32)
            g = hx.gload(P["gf"] + i64(col * 4), F32, 8)
            h = [bfr(F32(h1[e]) + F32(tot[e])) for e in range(8)]
            hx.stx_dev(P["xf"] + i64((tok * HID + col) * 2), vec([h[e] * F32(g[e]) for e in range(8)]).to(BF16))
            ss = h[0] * h[0]
            for e in range_constexpr(1, 8):
                ss = ss + h[e] * h[e]
            hx.gstore(P["part2"] + i64((tok * 256 + b * 2 + l // 16) * 4), ss, scope="agent")
        gpu.barrier()


@ASTRewriter.transform
def quant_rows(c, src, dst, dsc, sync):
    """src [16][2048] bf16 times rstd -> MXFP8 1x32 in WMMA B layout at LDS dst (16 steps x 2 KB), scales [step][tok]
    i32 at LDS dsc. Thread t: token t % 16, K128 step t // 16."""
    t, sm = c["t"], c["sm"]
    tok = t % 16
    s = t // 16
    rs = hx.lload(sm + RS + tok * 4, F32)
    raw = [hx.gload(src + i64((tok * HID + s * 128 + q * 8) * 2), BF16, 8).to(F32) for q in range(16)]
    if fx.const_expr(sync):
        gpu.barrier()                                # (every thread has read RS before the LDS is rewritten)
    sc = I32(0)
    for cb in range_constexpr(4):
        v = [F32(raw[cb * 4 + q // 8][q % 8]) * rs for q in range(32)]
        am = fx.max(v[0], F32(0.0) - v[0])
        for q in range_constexpr(1, 32):
            am = fx.max(am, fx.max(v[q], F32(0.0) - v[q]))
        code, inv = fp8_code(am)
        sc = sc | (code << (8 * cb))
        for h in range_constexpr(2):
            hx.lstore(dst + s * 2048 + (h * 16 + tok) * 64 + cb * 16, vec(fp8_pack(v[h * 16:h * 16 + 16], inv), I32))
    hx.lstore(dsc + (s * 16 + tok) * 4, sc)


def build(R=16, head=True):
    keyed = (R, head, 6)
    N = 248320

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(tab: Int64, cnt: Int64, st: Int64):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = t // 32
        c = dict(t=t, b=b, w=w, l=t % 32, sm=sm, wbuf=sm + w * PW, xbuf=sm + w * PW + 16384, st=st)
        P = {k: hx.sload64(tab + i64(8 * i)) for i, k in enumerate(TAB)}

        def stamp(k):
            if t == 0:
                hx.gstore(st + i64(b) * (NSTAMP * 8) + i64(k * 8), hx.clock())

        stamp(0)
        if b < 128:
            burst(c, P["Wo"] + i64(b) * 131072 + i64(w * 16384), 16, c["wbuf"], 2)
        hx.handoff(cnt, t, True)
        stamp(1)
        stage_oproj(c, P)
        hx.handoff(cnt + i64(64), t, True)
        stamp(2)
        rstd_reduce(c, P["part1"], 256)
        stage_router(c, P)
        quant_rows(c, P["xg1p"], sm + XQ, sm + XQS, False)
        hx.handoff(cnt + i64(128), t, True)
        stamp(3)
        stage_topk(c, P)
        hx.handoff(cnt + i64(192), t, True)
        stamp(4)
        U = union(c, P, R)
        stage_experts(c, P, U)
        hx.handoff(cnt + i64(256), t, True)
        stamp(5)
        stage_down(c, P, U)
        hx.handoff(cnt + i64(320), t, True)
        stamp(6)
        if fx.const_expr(head):
            rstd_reduce(c, P["part2"], 256)
            quant_rows(c, P["xf"], sm, sm + H8.geom(N, HID, HEAD["TW"], HEAD["GW"], HEAD["B"])["XB"], True)
            stamp(7)
            H8.head_argmax8(None, None, P["w8"], P["ws8"], P["out"], t, b, sm, N, HID, HEAD["B"], HEAD["TW"], HEAD["GW"])
        stamp(8)
        _ = keyed

    @flyc.jit
    def launch(tab: Int64, cnt: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(tab, cnt, st).launch(grid=(256,), block=(256,), smem=LDS, cluster=(16, 1, 1), stream=stream)

    return launch
