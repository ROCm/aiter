"""GDN mixer of a verifier layer as dataflow stages of one launch (step 1: the two projections, a stand-in for the verify).
Formats as ATOM serves them (ATOM_MXFP8_1X32): x, the fused in_proj [q|k|v|z|b|a] (N 12352, K 2048) and out_proj
(N 2048, K 4096) are MXFP8 1x32 on both operands.

 CTA 0..96     in_proj: wave 8 b + w = output tile (16 rows, all of K: 32 KB of weights by one TDM at entry); x from the
               CTA's LDS copy (XQ / XQS); bf16 rows -> proj [16][12352]; qdone += 1 a CTA.
 CTA 97..112   stand-in verify of token b - 97 after qdone == 97: o = v * sigmoid(z) -> MXFP8 1x32 at oq8 / oqs8 (the
               out_proj's B-operand layout); odone += 1.
 CTA 113..176  out_proj: wave = (tile, K quarter); weights by TDM at entry (fixed: a prefetch), the quarter's x by TDM
               after odone == 16; partial sums joined in LDS; h1 = h0 + out (bf16) -> hdone += 1.
"""
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64
import hxdev as hx
from flydsl._mlir.dialects import llvm
from post import F32, I32, BF16, i64, vec, bfr, sigmoid, fp8_pack, PW, RED, XQ, XQS, LDS, HID
from head import _ufl
from vmoe2 import fp8_code, xload

TAB = "Win WinS Wout WoutS xq8 xqs8 proj oq8 oqs8 h0 h1".split()
NSTAMP = 16
NPROJ = 12352
NT = NPROJ // 16                 # 772 in_proj tiles
QC = NT // 4                      # 193 in_proj CTAs (4 tiles each, on wave pairs)
VC0, VCN = QC, 16                 # stand-in verify CTAs (one a token)
OC0, OCN = QC + VCN, 43           # out_proj CTAs (3 tiles each)
QDONE, ODONE, HDONE = 0, 64, 128
STAMPS = os.environ.get('STAMPS', '1') == '1'
CPOL = int(os.environ.get('CPOL', 2))
XTDM = os.environ.get('XTDM', '1') == '1'          # in_proj's x copy by one TDM a CTA (else per-lane async copies)
WS = tuple(int(v) for v in os.environ.get('GWS', '1,15').split(','))


@ASTRewriter.transform
def wg4(ws_, sc_, xr, xs, k0, acc):
    """4 scaled WMMAs on operands already in registers (the block boundary keeps the LDS loads batched)."""
    if hx.opaque_s(I32(1)) != 0:
        for k in range_constexpr(4):
            acc = hx.wmma_mx(ws_[k], sc_[k], xr[k0 + k], xs[k0 + k], acc)
    return acc


def half_gemm(wbase, wsbase, xbase, xsbase, l, acc):
    """8 K128 steps: x operand (and scales) in registers once, weights 4 steps a group. Bases already at the half."""
    tok = l % 16
    xr = [hx.lload(xbase + k * 2048 + l * 64, I32, 16) for k in range(8)]
    xs = [hx.lload(xsbase + (k * 16 + tok) * 4, I32) for k in range(8)]
    for g in range(2):
        ws_ = [hx.lload(wbase + (g * 4 + k) * 2048 + l * 64, I32, 16) for k in range(4)]
        sc_ = [hx.lload(wsbase + ((g * 4 + k) * 16 + tok) * 4, I32) for k in range(4)]
        acc = wg4(ws_, sc_, xr, xs, g * 4, acc)
    return acc


@ASTRewriter.transform
def poll(c, addr, n, site):
    """One wave a CTA polls the counter; the others park at the CTA barrier (the poll rule)."""
    if c["w"] == 0:
        hx.wait_count(addr, n, site, ws=WS)
    gpu.barrier()


@ASTRewriter.transform
def stage_inproj(c, P, cnt, stamp):
    """CTA b: tiles 4 b .. 4 b + 3; wave w: K half w // 4 of tile 4 b + w % 4 (16 KB of weights, one TDM at entry)."""
    b, w, l, t, sm = c["b"], c["w"], c["l"], c["t"], c["sm"]
    tok = l % 16
    tile = _ufl(b * 4 + w % 4)
    hf = _ufl(w // 4)
    wb = _ufl(c["wbuf"])
    hx.tdm_load(P["Win"] + i64(tile) * 32768 + i64(hf) * 16384, wb, 512, 32, CPOL)
    hx.tdm_load(P["WinS"] + i64(tile) * 1024 + i64(hf) * 512, wb + 16384, 512, 1, CPOL)
    if fx.const_expr(XTDM):
        if w == 0:
            hx.tdm_load(P["xq8"], _ufl(sm + XQ), 512, 64, 0)
            hx.tdm_load(P["xqs8"], _ufl(sm + XQS), 512, 2, 0)
    else:
        for it in range_constexpr(8):
            hx.async_to_lds_b128(P["xq8"] + i64((t + it * 256) * 16), sm + XQ + (t + it * 256) * 16, 0)
        if t < 64:
            hx.async_to_lds_b128(P["xqs8"] + i64(t * 16), sm + XQS + t * 16, 0)
        hx.wait_asynccnt0()
    hx.wait_tensorcnt(0)
    stamp(2)
    gpu.barrier()
    stamp(1)
    buf = hx.opaque_v(c["wbuf"])
    xb = hx.opaque_v(sm + XQ + hf * 16384)
    xsb = hx.opaque_v(sm + XQS + hf * 512)
    acc = half_gemm(buf, buf + 16384, xb, xsb, l, fx.Vector.filled(8, 0.0, F32))
    hx.lstore(sm + RED + (w * 32 + l) * 32, acc)
    gpu.barrier()
    if w < 4:
        tot = acc + hx.lload(sm + RED + ((w + 4) * 32 + l) * 32, F32, 8)
        hx.stx_dev(P["proj"] + i64((tok * NPROJ + tile * 16 + (l // 16) * 8) * 2), vec([bfr(F32(tot[e])) for e in range(8)]).to(BF16))
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(cnt + i64(QDONE), 1)
    stamp(3)


@ASTRewriter.transform
def stage_vstub(c, P, cnt, stamp):
    """Stand-in for the verify: token b - VC0, thread t = columns 16 t .. 16 t + 15 of the 4096."""
    b, w, l, t = c["b"], c["w"], c["l"], c["t"]
    tokp = b - VC0
    poll(c, cnt + i64(QDONE), QC, 30)
    stamp(4)
    v = xload(P["proj"] + i64((tokp * NPROJ + 4096 + t * 16) * 2), 16).to(F32)
    z = xload(P["proj"] + i64((tokp * NPROJ + 8192 + t * 16) * 2), 16).to(F32)
    o = [bfr(F32(v[e]) * sigmoid(F32(z[e]))) for e in range(16)]
    am = fx.max(o[0], F32(0.0) - o[0])
    for e in range_constexpr(1, 16):
        am = fx.max(am, fx.max(o[e], F32(0.0) - o[e]))
    am = fx.max(am, hx.lane_gather(am.bitcast(I32), l ^ 1).bitcast(F32))
    code, inv = fp8_code(am)
    pk = fp8_pack(o, inv)
    step, g, half = t // 8, (t % 8) // 2, t % 2
    hx.stx_dev(P["oq8"] + i64(step * 2048 + (half * 16 + tokp) * 64 + g * 16), vec(pk, I32))
    sc = code
    for d in range_constexpr(1, 4):
        sc = sc | (hx.lane_gather(code, (l & 24) | (2 * d)) << (8 * d))
    if l % 8 == 0:
        hx.gstore(P["oqs8"] + i64((step * 16 + tokp) * 4), sc, scope="agent")
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(cnt + i64(ODONE), 1)
    stamp(5)


OX = 6 * PW                       # out_proj CTAs: waves 6, 7 idle; their 66 KB of LDS hold the x operand (64 KB + 2 KB of scales)


@ASTRewriter.transform
def stage_outproj(c, P, cnt, stamp):
    """CTA: 3 tiles on wave pairs (w < 6): wave w = K half w % 2 of tile 3 (b - OC0) + w // 2 (32 KB of weights by TDM at
    entry: fixed, a prefetch); x (64 KB) by one TDM a CTA after odone."""
    b, w, l, t, sm = c["b"], c["w"], c["l"], c["t"], c["sm"]
    tok = l % 16
    t3 = (b - OC0) * 3 + w // 2
    live = ((w < 6) & (t3 < 128)).select(I32(1), I32(0))
    tile = _ufl((t3 < 128).select(t3, I32(127)))
    hf = _ufl(w % 2)
    wb = _ufl(c["wbuf"])
    if w < 6:
        hx.tdm_load(P["Wout"] + i64(tile * 32 + hf * 16) * 2048, wb, 512, 64, CPOL)
        hx.tdm_load(P["WoutS"] + i64(tile * 32 + hf * 16) * 64, wb + 32768, 512, 2, CPOL)
    res = hx.gload(P["h0"] + i64((tok * HID + tile * 16 + (l // 16) * 8) * 2), BF16, 8).to(F32)
    poll(c, cnt + i64(ODONE), VCN, 31)
    stamp(6)
    if w == 0:
        hx.tdm_load(P["oq8"], _ufl(sm + OX), 512, 128, 0)
        hx.tdm_load(P["oqs8"], _ufl(sm + OX + 65536), 512, 4, 0)
    hx.wait_tensorcnt(0)
    gpu.barrier()
    stamp(7)
    acc = fx.Vector.filled(8, 0.0, F32)
    if w < 6:
        buf = hx.opaque_v(c["wbuf"])
        xb = hx.opaque_v(sm + OX + hf * 32768)
        xsb = hx.opaque_v(sm + OX + 65536 + hf * 1024)
        acc = half_gemm(buf, buf + 32768, xb, xsb, l, acc)
        acc = half_gemm(buf + 16384, buf + 32768 + 512, xb + 16384, xsb + 512, l, acc)
    hx.lstore(sm + RED + (w * 32 + l) * 32, acc)
    gpu.barrier()
    if (live != 0) & (w % 2 == 0):
        tot = acc + hx.lload(sm + RED + ((w + 1) * 32 + l) * 32, F32, 8)
        hx.stx_dev(P["h1"] + i64((tok * HID + tile * 16 + (l // 16) * 8) * 2),
                   vec([bfr(F32(res[e]) + bfr(F32(tot[e]))) for e in range(8)]).to(BF16))
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(cnt + i64(HDONE), 1)
    stamp(8)


@ASTRewriter.transform
def flow(c, P, cnt, stamp):
    b = c["b"]
    if b < QC:
        stage_inproj(c, P, cnt, stamp)
    else:
        if b < OC0:
            stage_vstub(c, P, cnt, stamp)
        else:
            if b < OC0 + OCN:
                stage_outproj(c, P, cnt, stamp)


def build():
    keyed = ("vgdn1", 2, XTDM, STAMPS, CPOL, WS, hx.TDM_ELEM)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(tab: Int64, cnt: Int64, st: Int64):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = _ufl(t // 32)
        c = dict(st=st, cnt=cnt, t=t, b=b, w=w, l=t % 32, sm=sm, wbuf=sm + w * PW)
        P = {k: hx.sload64(tab + i64(8 * i)) for i, k in enumerate(TAB)}

        def stamp(k):
            if fx.const_expr(STAMPS):
                if t % 32 == 0:
                    hx.gstore(st + i64(b * 8 + w) * (NSTAMP * 8) + i64(k * 8), hx.clock())

        stamp(0)
        flow(c, P, cnt, stamp)
        hx.wait_tensorcnt(0)
        hx.wait_asynccnt0()
        hx.wait_storecnt0()
        llvm.inline_asm(None, [], "s_wait_loadcnt 0x0", "", has_side_effects=True)
        stamp(15)
        _ = keyed

    @flyc.jit
    def launch(tab: Int64, cnt: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(tab, cnt, st).launch(grid=(256,), block=(256,), smem=LDS, cluster=(16, 1, 1), stream=stream)

    return launch
