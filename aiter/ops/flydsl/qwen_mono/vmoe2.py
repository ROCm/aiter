"""Verifier MoE block as a dataflow, 1-16 rows, one launch. Formats and tables as vmoe.py. After the prologue no
expert wave takes a CTA barrier: a wave runs whole units on its own and waits only on the counters of what it reads.
 CTA b owns expert b. Unit (E, j < 16) = rows 32 j .. 32 j + 31 of E's gate and up (4 tiles, 64 KB), streamed as 8
 chunks of 8 KB through a 3-slot LDS ring a wave -> MXFP8 1x32 activation of those 32 columns (GE / GS, by expert).
 b < 16   router tile b (unscaled logits: the norm scale cannot change the top 8) -> cluster barrier -> rank token b ->
          sel / topv / fl[E] = GEN, nranked += 1 (the tag). Then the prologue; a selected one runs its own units 0..7.
 16..31   shared expert gate/up -> cluster barrier -> wave (b - 16) * 8 + w: the shared down of that tile -> sdn.
 32       the shared gate.
 b >= 33  wave w starts unit (b, w) before the tag (first two chunks issued before the entry handoff). At a chunk
          boundary after the tag an unselected owner drops the unit and helps.
 helpers  (unselected, not 16..32): units 8..15 of every selected expert and 0..7 of those owned by 16..32, by index
          (bitmap of fl in registers: no barrier).
 down     unit (u-th selected expert, g < 32) = 4 output tiles (16 KB) on wave 8 b + w (+ 2048 a round), after
          actdone[E] == 16; a lane stores its token's 8 values to slot (token, rank).
 combine  wave 0 of CTA b < 128, after ddone == 32 U + 129: slots in rank order + shared gate * sdn + h1 -> h2."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
import hxdev as hx
from flydsl._mlir.dialects import llvm
from post import (F32, I32, BF16, i64, vec, bfr, exp, sigmoid, burst, wmma_run, reduce8, rstd_reduce, st_f32x8, quant_rows,
                  fp8_code, fp8_pack, wmma_fp4, silu_mul, PW, TMP, RS, XQ, XQS, LDS, HID)
from head import _ufl
from vmoe import amax8, f1_weights

TAB = "Wg Wsg SW SWS EW ES ED EDS SD SDS h1 xg1 xq8 xqs8 part1 lg sgate sel topv GE GS GSU GSS sdn slot h2 lgm bm selm topm gen".split()
NSTAMP = 32
GEN = 1
import os
REP = int(os.environ.get('REP', 1))      # diagnostic: the unit's WMMA loop REP times (same bytes)
NOX = os.environ.get('NOX', '0') == '1'   # diagnostic: x operand and scales not read from LDS in the loop
MIRROR = 3 * 8704
MIRROR_TAG = os.environ.get('MIRROR_TAG', '0') == '1'   # tag check from an LDS mirror riding with the chunks
POLL1 = os.environ.get('POLL1', '1') == '1'
HORD = os.environ.get('HORD', '1') == '1'
ROWN = os.environ.get('ROWN', '0') == '1'          # 1: a selected router CTA runs its own expert's units (needs FASTTAG=0 paths for SPEC)
QFIRST = os.environ.get('QFIRST', '0') == '1'
DLOADS = os.environ.get('DLOADS', '1') == '1'        # device-scope loads for every read of another CTA's / kernel's data (see dload)
INFENCE = os.environ.get('INFENCE', '0') == '1'   # global_inv from every wave: 140-1000 us a launch (diagnostic only)
XDEV = os.environ.get('XDEV', '1') == '1'            # x is read with device-scope loads (see xload)
HWQ = os.environ.get('HWQ', '1') == '1'            # x -> MXFP4 with the hardware pack ATOM uses (else an RNE emulation)
SHB = os.environ.get('SHB', '1') == '1' and os.environ.get('ATOMIO', '0') == '1'   # the shared expert in bf16 on the bf16 x, as ATOM serves it (ATOM mode only)
A4 = os.environ.get('A4', '1') == '1' and os.environ.get('ATOMIO', '0') == '1'   # routed experts with ATOM's MXFP4 activations (ATOM mode only)
ATOMIO = os.environ.get('ATOMIO', '0') == '1'      # input: ATOM's normed x [rows][2048] bf16 (quantized in the kernel); output: the MoE sum, no residual
STAMPS = os.environ.get('STAMPS', '1') == '1'
ENTRYH = os.environ.get('ENTRYH', '1') == '1'      # device-wide handoff at entry (aligns the stamps; not needed for the math)
RT3 = os.environ.get('RT3', '0') == '1' and os.environ.get('MB', '0') != '1'   # routing v3: packed stores, sleeping waits, one collector wave a CTA -> LDS
RT3F = os.environ.get('RT3F', '0') == '1'           # (keep the cluster fences around the cluster barrier)
MB32 = os.environ.get('MB32', '1') == '1'
D32 = os.environ.get('D32', '0') == '1'
BANK = os.environ.get('BANK', '1') == '1'            # two counter banks by launch-tag parity; this launch zeroes the other one: no counter-zero kernel before the launch
BANKB = 16384
S128 = os.environ.get('S128', '1') == '1'            # down / shared-down outputs as plain b128 stores (2 a tile a lane, not 4 b64)
G128 = os.environ.get('G128', '0') == '1'            # the activations (GE / GSU) as plain b128 stores (1 a lane a half, not 2 b64)
MB = os.environ.get('MB', '0') == '1'                # tagged mailbox for the routing handoffs: 64-bit words (launch tag << 32 | data), no store waits, no barriers, no counter
STX = tuple(int(v) for v in os.environ.get('STX', '').split(',') if v)   # diagnostic: which router stamps are emitted
LWS = int(os.environ.get('LWS', 2))                    # the rankers' poll of their logit word
MBS = os.environ.get('MBS', '1') == '1'              # the collector polls one word a lane, then reads the rest
MBX = 8 * PW + 17408                                # LDS: the poller wave's copy of the bitmap partials [32] i32, then sel [16][8] i32 and topv [16][8] f32
EARLY = os.environ.get('EARLY', '0') == '1'          # an owner CTA starts its expert's units 0..7 as soon as ANY token's top-8 names it (fl[b]), not at the all-tokens tag
EWS = int(os.environ.get('EWS', 15))
ALLSPLIT = os.environ.get('ALLSPLIT', '1') == '1' and not EARLY  # <= 59 experts: every gate/up unit on two waves, no full owner units
CH4 = os.environ.get('CH4', '0') == '1'            # gate/up units in 4 KB chunks (needs TDM, WPRE, SPEC=0)
IPF = int(os.environ.get('IPF', 0))
TWS = tuple(int(v) for v in os.environ.get('TWS', '1,15').split(','))   # the tag poll of a CTA's one polling wave
XB2 = os.environ.get('XB2', '1') == '1'            # split units: the partner's sums are read before a second CTA barrier
DBG = os.environ.get('DBG', '0') == '1'            # diagnostic: junk operands / sums of gate/up units recorded at st
WC0 = os.environ.get('WC0', '0') == '1'            # diagnostic: every TDM wait drains to 0 (no reliance on in-order completion)
EXITW = os.environ.get('EXITW', '1') == '1'        # every wave drains its tensor / async / store / load counters before it ends
FASTTAG = os.environ.get('FASTTAG', '1') == '1'    # one round of global loads after the tag, not four in series
TDM = os.environ.get('TDM', '1') == '1'            # expert weights by TDM (one instruction a chunk), not per-lane async copies
BATCH = os.environ.get('BATCH', '1') == '1'        # a chunk's LDS operand loads before its WMMAs, behind a block boundary
KSPLIT = os.environ.get('KSPLIT', '1') == '1'
CPRE = os.environ.get('CPRE', '1') == '1'
ENDH = os.environ.get('ENDH', '0') == '1'          # device-wide handoff before the exit (only made the end stamp uniform)
CWS = tuple(int(v) for v in os.environ.get('CWS', '1,1').split(','))   # the 128 combine waves poll hot
DN1 = os.environ.get('DN1', '1') == '1'           # one down unit a wave (4..8 tiles by U), no second round
WPRE = os.environ.get('WPRE', '1') == '1'
if 'WSLEEP' not in os.environ:
    hx.WSLEEP = (1, 15)                                  # hot polls from ~1800 waves stall other CTAs' loads (11-15)
SPEC = os.environ.get('SPEC', '0') == '1'               # 1: owners start their units before the tag (off: load only what top-8 selected)
CPOL = int(os.environ.get('CPOL', 2))
WAITS = os.environ.get('WAITS', '1') == '1'   # diagnostic: 0 = the unit never waits for its chunks (wrong results)
PROBE = os.environ.get('PROBE', '1') == '1'   # diagnostic: 0 = owners never look for the tag inside a unit
STAMPC = os.environ.get('STAMPC', '0') == '1'
SLOT = 8704                       # ring slot: 8 KB of weights + 512 B of scales
NRANKED, DDONE, ACTDONE, FL = 256, 320, 4096, 8192     # byte offsets in cnt (zeroed a launch)


def dload(addr, kind, n=8):
    """n (even) 4-byte elements at addr, read at device scope (8 bytes a load): data another CTA or another kernel
    wrote, at an address this CTA may have read in an earlier launch (an ordinary load can return that earlier content)."""
    if not DLOADS:
        return hx.gload(addr, kind, n)
    if D32:                                           # diagnostic: 4-byte device-scope loads instead of 8-byte ones
        return fx.Vector.from_elements([hx.gload(addr + i64(4 * i), I32, scope="agent") for i in range(n)], I32).bitcast(kind)
    ws = [hx.gload(addr + i64(8 * i), fx.Int64, scope="agent") for i in range(n // 2)]
    return fx.Vector.from_elements(ws, fx.Int64).bitcast(kind)


def stg(addr, val):
    """A lane's 16 bytes of activation. G128: one plain b128 store, not two device-scope b64."""
    if G128:
        hx.gstore(addr, val)
    else:
        hx.stx_dev(addr, val)


def st8(addr, vals):
    """8 f32 of a lane (a down tile's outputs) to global memory. S128: two plain 16-byte stores (global_store_b128)
    instead of four device-scope 8-byte ones: a store instruction costs the wave ~0.1 us to issue (stcost.py) and a down
    unit issued 20 a lane."""
    if S128:
        hx.gstore(addr, vec(vals[:4]))
        hx.gstore(addr + i64(16), vec(vals[4:]))
    else:
        st_f32x8(addr, vals)


def tagw(G, v):
    """The mailbox word: launch tag << 32 | the 32 bits of v."""
    return (i64(G) << 32) | (i64(I32(v)) & i64(0xFFFFFFFF))


def tagged(v, G):
    return (I32(v >> 32) == G).select(I32(1), I32(0))


@ASTRewriter.transform
def mb_logit(c, P):
    """Thread t of token CTA b: its expert's logit, as soon as the word carries this launch's tag."""
    b, t = c["b"], c["t"]
    a = P["lgm"] + i64((b * 256 + t) * 8)
    t0 = hx.clock()
    v = hx.gload(a, fx.Int64, scope="agent")
    while (tagged(v, c["G"]) == 0) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
        hx.s_sleep(LWS)
        v = hx.gload(a, fx.Int64, scope="agent")
    hx.note_timeout(t0, 50)
    return I32(v).bitcast(F32)


@ASTRewriter.transform
def mb_collect(c, P):
    """The poller wave. Poll: ONE word a CTA (the last bitmap word of token b % 16), at the old counter poll's rate: ~240
    CTAs polling 32 words each starved the router CTAs' loads (launches of 0.2-2 s). Then one round: lane l reads the
    bitmap words of tokens 4 (l // 8) .. + 3, wave slot l % 8, and its token's (l % 16) sel (l < 16) or topv row, re-read
    only while a word lacks the tag; their OR / rows go to LDS for the CTA's other waves (no global load after the
    signal)."""
    l, sm, G = c["l"], c["sm"], c["G"]
    t0 = hx.clock()
    sa = P["bm"] + i64(((c["b"] % 16) * 8 + 7) * 8)
    if fx.const_expr(MB32):                               # poll the tag half alone (a 4-byte load, as the counter poll is)
        while (hx.gload(sa + i64(4), I32, scope="agent") != G) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
            hx.s_sleep(EWS)
    else:
        while (tagged(hx.gload(sa, fx.Int64, scope="agent"), G) == 0) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
            hx.s_sleep(EWS)
    hx.note_timeout(t0, 51)
    ab = P["bm"] + i64(((l // 8) * 32 + l % 8) * 8)
    ar = (l < 16).select(P["selm"], P["topm"]) + i64((l % 16) * 64)
    wb = [hx.gload(ab + i64(64 * i), fx.Int64, scope="agent") for i in range(4)]
    wr = [hx.gload(ar + i64(8 * i), fx.Int64, scope="agent") for i in range(8)]
    ok = tagged(wb[0], G)
    for i in range_constexpr(1, 4):
        ok = ok & tagged(wb[i], G)
    for i in range_constexpr(8):
        ok = ok & tagged(wr[i], G)
    while (ok == 0) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
        hx.s_sleep(EWS)
        wb = [hx.gload(ab + i64(64 * i), fx.Int64, scope="agent") for i in range(4)]
        wr = [hx.gload(ar + i64(8 * i), fx.Int64, scope="agent") for i in range(8)]
        ok = tagged(wb[0], G)
        for i in range_constexpr(1, 4):
            ok = ok & tagged(wb[i], G)
        for i in range_constexpr(8):
            ok = ok & tagged(wr[i], G)
    hx.note_timeout(t0, 52)
    hx.lstore(sm + MBX + l * 4, I32(wb[0]) | I32(wb[1]) | I32(wb[2]) | I32(wb[3]))
    hx.lstore(sm + MBX + 128 + l * 32, vec([I32(wr[i]) for i in range(8)], I32))


@ASTRewriter.transform
def bank_zero(c, other):
    """One wave (the last of CTA 255) zeroes the 16 KB counter bank of the next launch: 32 plain 16-byte stores a lane."""
    if (c["b"] == 255) & (c["w"] == 7):
        zv = fx.Vector.filled(4, 0, I32)
        for i in range_constexpr(BANKB // 512):
            hx.gstore(other + i64(i * 512 + c["l"] * 16), zv)
        hx.wait_storecnt0()


@ASTRewriter.transform
def router_out3(c, P, tot):
    """Router CTA b (tile b): a wave-0 lane's 8 logits (token l % 16, experts 16 b + 8 (l // 16) ..) as bf16 in ONE plain
    16-byte store at lgm [tok][tile][half] (the logits are bf16 values: ATOM ranks its bf16 router output); then the
    cluster barrier (the 16 router CTAs are one cluster: a sleeping wait, no polling, no CTA barrier)."""
    b, l = c["b"], c["l"]
    if c["w"] == 0:
        hx.gstore(P["lgm"] + i64((((l % 16) * 16 + b) * 2 + l // 16) * 16), vec([F32(tot[e]) for e in range(8)]).to(BF16))
    hx.wait_storecnt0()
    if fx.const_expr(RT3F):
        hx.fence_release("cluster")
    hx.cluster_barrier()
    if fx.const_expr(RT3F):
        hx.fence_acquire("cluster")


@ASTRewriter.transform
def rank3(c, P, cnt):
    """rank_publish with one device-scope load a thread (its bf16 logit), ONE store a ranked thread (the record
    expert << 16 | bf16 logit at selm [tok][rank], 64-bit) and one counter add a wave: nranked reaches 128."""
    b, t, l, w, sm = c["b"], c["t"], c["l"], c["w"], c["sm"]
    wd = hx.gload(P["lgm"] + i64((((b * 16 + t // 16) * 2 + (t % 16) // 8) * 16) + ((t % 8) // 2) * 4), I32, scope="agent")
    bits16 = (t % 2 == 0).select(wd & 0xFFFF, (wd >> 16) & 0xFFFF)
    my = (bits16 << 16).bitcast(F32)
    c["stamp"](16)
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
    c["stamp"](17)
    cv = [hx.lload(sm + TMP + i * 64, F32, 16) for i in range(4)]
    ci = [hx.lload(sm + TMP + 256 + i * 64, I32, 16) for i in range(4)]
    rank = I32(0)
    for i in range_constexpr(64):
        o = F32(cv[i // 16][i % 16])
        rank = rank + ((o > my) | ((o == my) & (I32(ci[i // 16][i % 16]) < t))).select(I32(1), I32(0))
    c["stamp"](18)
    if (lr < 8) & (rank < 8):
        hx.gstore(P["selm"] + i64((b * 8 + rank) * 8), i64((t << 16) | bits16), scope="agent", align=8)
    hx.wait_storecnt0()
    if l == 0:
        hx.atomic_add(cnt + i64(NRANKED), 1)


@ASTRewriter.transform
def collect3(c, P):
    """The collector wave, once nranked == 128: lane l reads records 4 (l % 2) .. + 3 of token l // 2 (4 loads), leaves
    the token rows (sel i32, topv f32) and the selected-expert bitmap (8 words, OR over the wave) in LDS."""
    l, sm = c["l"], c["sm"]
    ar = P["selm"] + i64(((l // 2) * 8 + (l % 2) * 4) * 8)
    wr = [hx.gload(ar + i64(8 * i), fx.Int64, scope="agent") for i in range(4)]
    ids = [(I32(wr[i]) >> 16) & 255 for i in range(4)]
    hx.lstore(sm + MBX + 128 + (l // 2) * 32 + (l % 2) * 16, vec(ids, I32))
    hx.lstore(sm + MBX + 128 + 512 + (l // 2) * 32 + (l % 2) * 16, vec([((I32(wr[i]) & 0xFFFF) << 16).bitcast(F32) for i in range(4)], F32))
    for k in range_constexpr(8):
        m = I32(0)
        for i in range_constexpr(4):
            m = m | ((ids[i] >> 5) == k).select(I32(1) << (ids[i] & 31), I32(0))
        m = m | hx.lane_gather(m, l ^ 1)
        m = m | hx.lane_gather(m, l ^ 2)
        m = m | hx.lane_gather(m, l ^ 4)
        m = m | hx.lane_gather(m, l ^ 8)
        m = m | hx.lane_gather(m, l ^ 16)
        if l == 0:
            hx.lstore(sm + MBX + k * 4, m)


@ASTRewriter.transform
def router_out_mb(c, P, tot):
    """Router CTA b (tile b), wave 0: the lane's 8 logits (token l % 16) as tagged words."""
    b, l = c["b"], c["l"]
    tok = l % 16
    if c["w"] == 0:
        for e in range_constexpr(8):
            hx.gstore(P["lgm"] + i64((tok * 256 + b * 16 + (l // 16) * 8 + e) * 8), tagw(c["G"], bfr(F32(tot[e])).bitcast(I32)), scope="agent", align=8)


@ASTRewriter.transform
def mb_next(c, P):
    if (c["b"] == 0) & (c["t"] == 0):
        hx.gstore(P["gen"], c["G"] + 1)
        hx.wait_storecnt0()


def ctpop(x):
    return I32(hx.intrin(hx.i32(), "llvm.ctpop.i32", [I32(x)]))


def assign(c, cnt):
    """The selected-expert bitmap from fl (lane l reads experts 8 l .. 8 l + 7), then this wave's indices."""
    b, w, l = c["b"], c["w"], c["l"]
    W = []
    if RT3:                                           # the collector wave's bitmap
        for k in range(8):
            W.append(_ufl(hx.lload(c["sm"] + MBX + k * 4, I32)))
    elif MB:                                          # the poller wave's partial ORs (lanes k, k + 8, k + 16, k + 24: slot k)
        for k in range(8):
            v = hx.lload(c["sm"] + MBX + k * 4, I32)
            for i in range(1, 4):
                v = v | hx.lload(c["sm"] + MBX + (k + 8 * i) * 4, I32)
            W.append(_ufl(v))
    else:
        f = dload(cnt + i64(FL) + i64(l * 32), I32)
        bits = I32(0)
        for q in range(8):
            bits = bits | (I32(f[q]) == GEN).select(I32(1 << q), I32(0))
        for k in range(8):
            v = I32(0)
            for m in range(4):
                v = v | (hx.lane_gather(bits, I32(4 * k + m)) << (8 * m))
            W.append(_ufl(v))
    pc = [ctpop(v) for v in W]
    U = pc[0]
    for k in range(1, 8):
        U = U + pc[k]
    c16 = ctpop(W[0] & 0xFFFF)
    X = pc[0] + ctpop(W[1] & 1) - c16
    posb, selb = I32(0), I32(0)
    for k in range(8):
        posb = posb + (I32(k) < (b >> 5)).select(pc[k], I32(0))
        here = I32(k) == (b >> 5)
        posb = posb + here.select(ctpop(W[k] & ((I32(1) << (b & 31)) - 1)), I32(0))
        selb = selb | here.select((W[k] >> (b & 31)) & 1, I32(0))
    if not ROWN:                                      # the router CTAs own no units (they come out of their roles late):
        hr = (b < 16).select(223 - (U - c16 - X) + b, (b - 33) - (posb - c16 - X))    # all 16 are the last helpers
        return dict(W=W, pc=pc, U=U, X=X, c16=c16, sel=selb, hidx=hr * 8 + w, H=(239 - (U - c16 - X)) * 8, XN=X + c16, x0=I32(0))
    if HORD:                                          # helpers b >= 33 first, the router CTAs (late, busy) last
        hr = (b < 16).select(223 - (U - c16 - X) + b - posb, (b - 33) - (posb - c16 - X))
    else:
        hr = (b < 16).select(b - posb, b - posb - (17 - X))
    return dict(W=W, pc=pc, U=U, X=X, c16=c16, sel=selb, hidx=hr * 8 + w, H=(256 - U - (17 - X)) * 8, XN=X, x0=c16)


def nth(A, n):
    """The n-th selected expert (n < U)."""
    word, base, found, rem = I32(0), I32(0), I32(0), I32(n)
    for k in range(8):
        hit = (found == 0) & (rem < A["pc"][k])
        word = hit.select(A["W"][k], word)
        base = hit.select(I32(32 * k), base)
        found = found | hit.select(I32(1), I32(0))
        rem = (found == 0).select(rem - A["pc"][k], rem)
    pos = I32(0)
    for sh in (16, 8, 4, 2, 1):
        n_ = ctpop((word >> pos) & ((1 << sh) - 1))
        ge = rem >= n_
        rem = ge.select(rem - n_, rem)
        pos = ge.select(pos + sh, pos)
    return _ufl(base + pos)


def tload(c, src, kb, dst, cpol):
    """kb KB at src -> LDS dst: one TDM instruction (src, dst wave-uniform), or the per-lane async copies."""
    if TDM:
        hx.tdm_load(src, dst, 512, kb * 2, cpol)
    else:
        burst(c, src, kb, dst, cpol)


def wchunks(n):
    """Wait until at most n issued chunks are outstanding."""
    if TDM:
        hx.wait_tensorcnt(0 if WC0 else 2 * n)
    else:
        rocdl.s_wait_asynccnt(17 * n)


def drain():
    hx.wait_asynccnt0()
    if TDM:
        hx.wait_tensorcnt(0)


def hib(x):
    """1 where a byte of the i32 x is an E8M0 code above 199 (2^72: no weight / x / activation block has it)."""
    r = I32(0)
    for sh in (0, 8, 16, 24):
        r = r | (((x >> sh) & 255) > 199).select(I32(1), I32(0))
    return r


@ASTRewriter.transform
def dbg_mark(c, cond, kind, v0, v1):
    if cond != 0:
        a = c["st"] + i64(kind * 32)
        hx.gstore(a, I32(1))
        hx.gstore(a + i64(4), I32(v0))
        hx.gstore(a + i64(8), I32(v1))
        hx.gstore(a + i64(12), c["b"] * 8 + c["w"])
        hx.atomic_add(a + i64(16), 1)


@ASTRewriter.transform
def wgroup(ws_, sc_, xr, xs, acc):
    """8 WMMAs on operands already loaded. The block boundary keeps LLVM from sinking each LDS load next to its WMMA
    (one load + wait a WMMA is ~59 ns, batched ~37)."""
    if hx.opaque_s(I32(1)) != 0:
        for k in range_constexpr(8):
            acc = wmma_fp4(ws_[k], sc_[k], xr[k], xs[k], acc)
    return acc


def chunk_wmma(c, ch, xr, xs, acc):
    l = c["l"]
    tok = l % 16
    buf = c["wbuf"] + (ch % 3) * SLOT
    if TDM:
        buf = hx.opaque_v(buf)                        # LLVM does not see the TDM load write LDS
    if BATCH:
        ws_ = [hx.lload(buf + k * 1024 + l * 32, I32, 8) for k in range(8)]
        sc_ = [hx.lload(buf + 8192 + (k * 16 + tok) * 4, I32) for k in range(8)]
        if DBG:
            js, jx = hib(I32(sc_[0])), hib(I32(xs[0]))
            for k in range(1, 8):
                js, jx = js | hib(I32(sc_[k])), jx | hib(I32(xs[k]))
            dbg_mark(c, js, 1, ch, I32(sc_[0]))           # a weight scale code is junk
            dbg_mark(c, jx, 2, ch, I32(xs[0]))            # an x scale code is junk
            acc = wgroup(ws_, sc_, xr, xs, acc)
            big = (fx.max(F32(acc[0]), F32(0.0) - F32(acc[0])) > F32(1e6)).select(I32(1), I32(0))
            dbg_mark(c, big & (I32(1) - js) & (I32(1) - jx), 3, ch, I32(0))   # a huge sum with sane scale codes
            return acc
        return wgroup(ws_, sc_, xr, xs, acc)
    for k in range(8):
        acc = wmma_fp4(hx.lload(buf + k * 1024 + l * 32, I32, 8), hx.lload(buf + 8192 + (k * 16 + tok) * 4, I32), xr[k], xs[k], acc)
    return acc


def issue(c, P, E, j, ch):
    """Chunk ch < 8 (a Python int) of unit (E, j): K half ch // 4 of tile [gate 2 j, gate 2 j + 1, up 2 j, up 2 j + 1][ch % 4]."""
    ti = ch % 4
    if TDM:                                           # one TDM instruction for the 8 KB, one for its scales
        u = _ufl((E * 64 + (ti // 2) * 32 + 2 * j + ti % 2) * 16 + (ch // 4) * 8)
        dstu = _ufl(c["wbuf"]) + (ch % 3) * SLOT
        hx.tdm_load(P["EW"] + i64(u) * 1024, dstu, 512, 16, CPOL)
        hx.tdm_load(P["ES"] + i64(u) * 64, dstu + 8192, 512, 1, CPOL)
        return
    t0 = i64((E * 64 + (ti // 2) * 32 + 2 * j + ti % 2) * 16 + (ch // 4) * 8)
    dst = c["wbuf"] + (ch % 3) * SLOT
    for q in range(16):
        hx.async_to_lds_b128(P["EW"] + t0 * 1024 + i64(q * 512 + c["l"] * 16), dst + q * 512 + c["l"] * 16, CPOL)
    hx.async_to_lds_b128(P["ES"] + t0 * 64 + i64(c["l"] * 16), dst + 8192 + c["l"] * 16, CPOL)


@ASTRewriter.transform
def f1w(c, P):
    """vmoe.f1_weights, with the bf16 shared expert's gate / up weights (SHB): this wave's 256 K columns (8 K32 steps,
    8 KB) of gate tiles 2 j, 2 j + 1 and up tiles 2 j, 2 j + 1 -> its 32 KB of LDS."""
    b, w = c["b"], c["w"]
    if fx.const_expr(SHB):
        if (b >= 16) & (b < 32):
            for i in range_constexpr(4):
                tile = (i // 2) * 32 + (b - 16) * 2 + i % 2
                burst(c, P["SW"] + i64(tile * 64 + w * 8) * 1024, 8, c["wbuf"] + i * 8192, 2)
        else:
            f1_weights(c, P)
    else:
        f1_weights(c, P)


@ASTRewriter.transform
def shared_bf16(c, P, cnt):
    """The shared expert as ATOM serves it: bf16 weights on the bf16 x (no quantization anywhere). CTAs 16..31.
    gate / up: CTA j's 4 tiles, K split over the 8 waves (x of the wave's 256 columns straight into registers), partial
    sums joined in LDS; silu(gate) up -> bf16 at GSU in the down's operand layout; cluster barrier; down: one output
    tile a wave (16 KB of weights, the 16 KB activation)."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    row = fx.min(tok, c["nrows"] - 1)
    xr = [xload(c["xin"] + i64((row * HID + (w * 8 + k) * 32 + (l // 16) * 16) * 2), 16).to(F32) for k in range(8)]
    xs = [vec([finite0(F32(xr[k][e])) for e in range(16)]).to(BF16) for k in range(8)]
    acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    for i in range_constexpr(4):
        for k in range_constexpr(8):
            acc[i] = hx.wmma_bf16(hx.lload(c["wbuf"] + i * 8192 + k * 1024 + l * 32, BF16, 16), xs[k], acc[i])
    for i in range_constexpr(4):
        hx.lstore(c["wbuf"] + i * 1024 + l * 32, acc[i])
    gpu.barrier()
    if w == 0:
        tot = []
        for i in range_constexpr(4):
            s_ = hx.lload(sm + i * 1024 + l * 32, F32, 8)
            for q in range_constexpr(1, 8):
                s_ = s_ + hx.lload(sm + q * PW + i * 1024 + l * 32, F32, 8)
            tot.append(s_)
        act = [silu_mul([F32(tot[i][e]) for e in range(8)], [F32(tot[2 + i][e]) for e in range(8)], F32(1.0)) for i in range(2)]
        j = b - 16
        for i in range_constexpr(2):
            stg(P["GSU"] + i64(j * 1024 + (i * 16 + tok) * 32 + (l // 16) * 16), vec(act[i]).to(BF16))
    hx.wait_storecnt0()
    gpu.barrier()
    hx.fence_release("cluster")
    hx.cluster_barrier()
    hx.fence_acquire("cluster")
    sw = (b - 16) * 8 + w
    burst(c, P["SD"] + i64(sw) * 16384, 16, c["wbuf"], 2)
    burst(c, P["GSU"], 16, c["xbuf"], 0)
    hx.wait_asynccnt0()
    a = fx.Vector.filled(8, 0.0, F32)
    for k in range_constexpr(16):
        a = hx.wmma_bf16(hx.lload(c["wbuf"] + k * 1024 + l * 32, BF16, 16), hx.lload(c["xbuf"] + k * 1024 + l * 32, BF16, 16), a)
    st8(P["sdn"] + i64((tok * HID + sw * 16 + (l // 16) * 8) * 4), [F32(a[e]) for e in range(8)])
    hx.wait_storecnt0()
    if l == 0:
        hx.atomic_add(cnt + i64(DDONE), 1)


@ASTRewriter.transform
def prestage(c, P):
    b, w = c["b"], c["w"]
    if b >= 33:
        if fx.const_expr(SPEC):
            issue(c, P, b, w, 0)
            issue(c, P, b, w, 1)
    else:
        f1w(c, P)


@ASTRewriter.transform
def unit_start(c, P, E, j, pre):
    if pre == 0:
        drain()
        issue(c, P, E, j, 0)
        issue(c, P, E, j, 1)


@ASTRewriter.transform
def unit_chunk(c, P, cnt, E, j, ch, xr, xs, alive, spec, probe, acc):
    """probe: nranked as read (not waited on) at the previous chunk; a blocking read only once it shows the tag."""
    l = c["l"]
    tok = l % 16
    if alive != 0:
        if spec != 0:
            if probe >= 16:
                spec = I32(0)
                if hx.gload(cnt + i64(FL) + i64(E * 4), I32, scope="agent") != GEN:
                    alive = I32(0)
    if fx.const_expr(PROBE):
        if spec != 0:
            probe = hx.gload(cnt + i64(NRANKED), I32, scope="agent")
    if alive != 0:
        if fx.const_expr(ch + 2 < 8):
            issue(c, P, E, j, ch + 2)
            wchunks(2)
        else:
            if fx.const_expr(ch == 6):
                wchunks(1)
            else:
                drain()
        acc = chunk_wmma(c, ch, xr, xs, acc)
    return [alive, spec, probe, acc]


def unit_half(c, P, E, j, hf):
    """K half hf (a Python int) of unit (E, j) on this wave: 4 chunks, partial sums of the 4 tiles. Two waves of a helper
    CTA share a unit this way (KSPLIT); the partner's sums are added after a CTA barrier."""
    l, sm = c["l"], c["sm"]
    tok = l % 16
    drain()
    issue(c, P, E, j, hf * 4)
    issue(c, P, E, j, hf * 4 + 1)
    xr = [hx.lload(sm + XQ + (hf * 8 + k) * 2048 + l * 64, I32, 16) for k in range(8)]
    xs = [hx.lload(sm + XQS + ((hf * 8 + k) * 16 + tok) * 4, I32) for k in range(8)]
    accs = []
    for i in range(4):
        ch = hf * 4 + i
        if i + 2 < 4:
            issue(c, P, E, j, ch + 2)
            wchunks(2)
        elif i == 2:
            wchunks(1)
        else:
            drain()
        accs.append(chunk_wmma(c, ch, xr, xs, fx.Vector.filled(8, 0.0, F32)))
    return accs


@ASTRewriter.transform
def wgroup4(ws_, sc_, xr, xs, acc):
    if hx.opaque_s(I32(1)) != 0:
        for k in range_constexpr(4):
            acc = wmma_fp4(ws_[k], sc_[k], xr[k], xs[k], acc)
    return acc


SLOT4 = 4608                      # ring slot of the 4 KB chunks: 4 KB of weights + 256 B of scales


def gu_piece(c, P, E, j, quarters):
    """K quarters `quarters` (Python ints of 0..3) of unit (E, j): chunks of 4 KB = (quarter, tile), by TDM into a
    3-slot ring, two ahead; a chunk's 4 operand loads batched before its WMMAs. Returns the 4 tiles' sums (partial
    unless quarters is all four). The measured best unit shape (pull.py: 4.2 us a 64 KB unit under load, 8 KB chunks 4.6)."""
    l, sm = c["l"], c["sm"]
    tok = l % 16
    Eu, ju, wb = _ufl(E), _ufl(j), _ufl(c["wbuf"])
    seq = [(q, ti) for q in quarters for ti in range(4)]

    def issue4(s_):
        q, ti = seq[s_]
        u = (Eu * 64 + (ti // 2) * 32 + 2 * ju + ti % 2) * 16 + q * 4
        hx.tdm_load(P["EW"] + i64(u) * 1024, wb + (s_ % 3) * SLOT4, 512, 8, CPOL)
        hx.tdm_load(P["ES"] + i64(u) * 64, wb + (s_ % 3) * SLOT4 + 4096, 256, 1, CPOL)
    drain()
    issue4(0)
    issue4(1)
    accs = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    hf_cur = -1
    for s_, (q, ti) in enumerate(seq):
        if q // 2 != hf_cur:
            hf_cur = q // 2
            xr = [hx.lload(sm + XQ + (hf_cur * 8 + k) * 2048 + l * 64, I32, 16) for k in range(8)]
            xs = [hx.lload(sm + XQS + ((hf_cur * 8 + k) * 16 + tok) * 4, I32) for k in range(8)]
        if s_ + 2 < len(seq):
            issue4(s_ + 2)
            hx.wait_tensorcnt(4)
        elif s_ + 2 == len(seq):
            hx.wait_tensorcnt(2)
        else:
            hx.wait_tensorcnt(0)
        buf = hx.opaque_v(c["wbuf"] + (s_ % 3) * SLOT4)
        ws_ = [hx.lload(buf + k * 1024 + l * 32, I32, 8) for k in range(4)]
        sc_ = [hx.lload(buf + 4096 + (k * 16 + tok) * 4, I32) for k in range(4)]
        k0 = (q % 2) * 4
        accs[ti] = wgroup4(ws_, sc_, xr[k0:k0 + 4], xs[k0:k0 + 4], accs[ti])
    return accs


def unit_run(c, P, cnt, E, j, pre, spec):
    """Gate/up of unit (E, j) on this wave: K half 0 of the 4 tiles, then K half 1; the x operand of a half (8 K128
    steps, 128 registers) is read from LDS once (re-reading it for every WMMA was the bound). spec: an owner before the
    tag; it drops the unit (alive = 0) at the first chunk boundary where the tag is in and fl[E] is not set.
    Returns [alive, gate 2j, gate 2j+1, up 2j, up 2j+1, spec]."""
    l, sm = c["l"], c["sm"]
    tok = l % 16
    unit_start(c, P, E, j, pre)
    alive = I32(1)
    probe = I32(0)
    accs = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
    for hf in range(2):
        xr = [hx.lload(sm + XQ + (hf * 8 + k) * 2048 + l * 64, I32, 16) for k in range(8)]
        xs = [hx.lload(sm + XQS + ((hf * 8 + k) * 16 + tok) * 4, I32) for k in range(8)]
        for ti in range(4):
            alive, spec, probe, accs[ti] = unit_chunk(c, P, cnt, E, j, hf * 4 + ti, xr, xs, alive, spec, probe, accs[ti])
    return [alive] + accs + [spec]


@ASTRewriter.transform
def uw_load(c, P):
    """The lane's token's top-8 experts and logits (two global loads, independent of the expert)."""
    tok = c["l"] % 16
    if RT3:
        return (hx.lload(c["sm"] + MBX + 128 + tok * 32, I32, 8), hx.lload(c["sm"] + MBX + 128 + 512 + tok * 32, F32, 8))
    if MB:
        return (hx.lload(c["sm"] + MBX + 128 + tok * 32, I32, 8), hx.lload(c["sm"] + MBX + 128 + (16 + tok) * 32, F32, 8))
    return (dload(P["sel"] + i64(tok * 32), I32), dload(P["topv"] + i64(tok * 32), F32))


def uw_calc(c, uw, E):
    """weight(tok, E) from uw_load's rows."""
    sl, tv = uw
    rs = hx.lload(c["sm"] + RS + (c["l"] % 16) * 4, F32)
    ex = [exp((F32(tv[q]) - F32(tv[0])) * rs) for q in range(8)]
    s = ex[0]
    wv = (I32(sl[0]) == E).select(ex[0], F32(0.0))
    for q in range(1, 8):
        s = s + ex[q]
        wv = wv + (I32(sl[q]) == E).select(ex[q], F32(0.0))
    return wv / s


def unit_weight(c, P, E):
    """weight(tok, E) from the token's top-8 logits (unscaled) and its rstd; issued before the unit (WPRE) so that its
    two global loads are not on the path between the unit's last WMMA and actdone."""
    l, sm = c["l"], c["sm"]
    tok = l % 16
    sl = dload(P["sel"] + i64(tok * 32), I32)
    tv = dload(P["topv"] + i64(tok * 32), F32)
    rs = hx.lload(sm + RS + tok * 4, F32)
    ex = [exp((F32(tv[q]) - F32(tv[0])) * rs) for q in range(8)]
    s = ex[0]
    wv = (I32(sl[0]) == E).select(ex[0], F32(0.0))
    for q in range_constexpr(1, 8):
        s = s + ex[q]
        wv = wv + (I32(sl[q]) == E).select(ex[q], F32(0.0))
    return wv / s


@ASTRewriter.transform
def unit_epilogue_w(c, P, cnt, E, j, r, wv):
    """As unit_epilogue, the weight given."""
    l, sm = c["l"], c["sm"]
    tok = l % 16
    rs = hx.lload(sm + RS + tok * 4, F32)
    if fx.const_expr(A4):
        # ATOM's A4W4 experts: silu(gate) up in f32 (no bf16 rounding, no route weight: that multiplies the down's
        # output), quantized to MXFP4 a 32-column group
        act = [silu_mul([F32(r[1 + i][e]) for e in range(8)], [F32(r[3 + i][e]) for e in range(8)], F32(1.0)) for i in range(2)]
    else:
        act = [silu_mul([bfr(F32(r[1 + i][e]) * rs) for e in range(8)], [bfr(F32(r[3 + i][e]) * rs) for e in range(8)], wv) for i in range(2)]
    am = amax8(act[0] + act[1])
    am = fx.max(am, hx.lane_gather(am.bitcast(I32), l ^ 16).bitcast(F32))
    if fx.const_expr(A4):
        pk4, code = mx4_pack(act[0] + act[1], am)
        packed = [pk4[0:2], pk4[2:4]]
    else:
        code, inv = fp8_code(am)
        packed = [fp8_pack(act[0], inv), fp8_pack(act[1], inv)]
    if fx.const_expr(DBG):
        dbg_mark(c, (code > 199).select(I32(1), I32(0)), 4, E, j)      # a junk activation scale leaves this unit
    for i in range_constexpr(2):
        lane2 = i * 16 + tok
        stg(P["GE"] + i64(E) * 8192 + i64((j // 4) * 2048 + lane2 * 64 + (j % 4) * 16 + (l // 16) * 8), vec(packed[i], I32))
        if l < 16:
            hx.gstore(P["GS"] + i64(((E * 4 + j // 4) * 32 + lane2) * 16 + (j % 4) * 4), code, scope="agent")
    hx.wait_storecnt0()
    if l == 0:
        hx.atomic_add(cnt + i64(ACTDONE) + i64(E * 4), 1)


@ASTRewriter.transform
def unit_epilogue(c, P, cnt, E, j, r):
    """weight(tok, E) silu(gate) up of the unit's 32 columns -> MXFP8 (one scale a token) at GE / GS [E]; actdone[E] += 1.
    The weight is computed here from the token's top-8 logits (unscaled) and its rstd."""
    l, sm = c["l"], c["sm"]
    tok = l % 16
    sl = dload(P["sel"] + i64(tok * 32), I32)
    tv = dload(P["topv"] + i64(tok * 32), F32)
    rs = hx.lload(sm + RS + tok * 4, F32)
    ex = [exp((F32(tv[q]) - F32(tv[0])) * rs) for q in range(8)]
    s = ex[0]
    wv = (I32(sl[0]) == E).select(ex[0], F32(0.0))
    for q in range_constexpr(1, 8):
        s = s + ex[q]
        wv = wv + (I32(sl[q]) == E).select(ex[q], F32(0.0))
    wv = wv / s
    act = [silu_mul([bfr(F32(r[1 + i][e]) * rs) for e in range(8)], [bfr(F32(r[3 + i][e]) * rs) for e in range(8)], wv) for i in range(2)]
    am = amax8(act[0] + act[1])
    am = fx.max(am, hx.lane_gather(am.bitcast(I32), l ^ 16).bitcast(F32))
    code, inv = fp8_code(am)
    for i in range_constexpr(2):
        lane2 = i * 16 + tok
        stg(P["GE"] + i64(E) * 8192 + i64((j // 4) * 2048 + lane2 * 64 + (j % 4) * 16 + (l // 16) * 8), vec(fp8_pack(act[i], inv), I32))
        if l < 16:
            hx.gstore(P["GS"] + i64(((E * 4 + j // 4) * 32 + lane2) * 16 + (j % 4) * 4), code, scope="agent")
    hx.wait_storecnt0()
    if l == 0:
        hx.atomic_add(cnt + i64(ACTDONE) + i64(E * 4), 1)


@ASTRewriter.transform
def rank_publish(c, P, cnt):
    """CTA b = token b, thread t = expert t: two-level rank of the 256 logits; a top-8 thread publishes itself."""
    b, t, l, w, sm = c["b"], c["t"], c["l"], c["w"], c["sm"]
    if fx.const_expr(MB):
        my = mb_logit(c, P)
    else:
        my = hx.gload(P["lg"] + i64((b * 256 + t) * 4), F32, scope="agent")
    c["stamp"](16)
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
    c["stamp"](17)
    cv = [hx.lload(sm + TMP + i * 64, F32, 16) for i in range(4)]
    ci = [hx.lload(sm + TMP + 256 + i * 64, I32, 16) for i in range(4)]
    rank = I32(0)
    for i in range_constexpr(64):
        o = F32(cv[i // 16][i % 16])
        rank = rank + ((o > my) | ((o == my) & (I32(ci[i // 16][i % 16]) < t))).select(I32(1), I32(0))
    c["stamp"](18)
    if fx.const_expr(MB):
        top = ((lr < 8) & (rank < 8)).select(I32(1), I32(0))
        if top != 0:                                      # tagged rows first, each word valid on its own
            hx.gstore(P["selm"] + i64((b * 8 + rank) * 8), tagw(c["G"], t), scope="agent", align=8)
            hx.gstore(P["topm"] + i64((b * 8 + rank) * 8), tagw(c["G"], my.bitcast(I32)), scope="agent", align=8)
            hx.gstore(P["sel"] + i64((b * 8 + rank) * 4), t, scope="agent")       # (plain copies for the host-side tools)
            hx.gstore(P["topv"] + i64((b * 8 + rank) * 4), my, scope="agent")
        m = hx.lane_gather(top, I32(0))                  # this wave's 32 experts as one mask
        for p in range_constexpr(1, 32):
            m = m | (hx.lane_gather(top, I32(p)) << p)
        if l == 0:
            hx.gstore(P["bm"] + i64((b * 8 + w) * 8), tagw(c["G"], m), scope="agent", align=8)
    else:
        if (lr < 8) & (rank < 8):
            hx.gstore(P["sel"] + i64((b * 8 + rank) * 4), t, scope="agent")
            hx.gstore(P["topv"] + i64((b * 8 + rank) * 4), my, scope="agent")
            hx.gstore(cnt + i64(FL) + i64(t * 4), I32(GEN), scope="agent")
        hx.wait_storecnt0()
        gpu.barrier()
        if t == 0:
            hx.atomic_add(cnt + i64(NRANKED), 1)


def xscale(base, k, l):
    sv = hx.lload(base + (k * 32 + l) * 16, I32, 4)
    return I32(sv[0]) | (I32(sv[1]) << 8) | (I32(sv[2]) << 16) | (I32(sv[3]) << 24)


def fp8_code(am):
    """E8M0 byte of an MXFP8 group with max |x| = am, and 1 / scale, by ATOM's rule (mxfp8_1x32.mx32_e8m0, RoundUp):
    the fp32 exponent field of am / 448, plus one when its mantissa is non-zero, so that am / scale <= 448 and nothing
    is clamped. (post.fp8_code floors log2(am) - 8: a group whose max has a mantissa above 1.75 loses up to 12.5% of
    its largest elements; on served hidden states that was a 20% error of the routed experts' sum.)"""
    a_ = fx.min(fx.max(am, F32(1.0e-10)), F32(448.0 * 6.646139978924579e35))
    u = (a_ * F32(1.0 / 448.0)).bitcast(I32)
    e = ((u >> 23) & 255) + ((u & 0x7FFFFF) != 0).select(I32(1), I32(0))
    e = fx.min(e, I32(254))
    return e, ((I32(254) - e) << 23).bitcast(F32)


def mx_code(am, max_norm):
    """E8M0 byte and 1 / scale of a group with max |x| = am for an element format whose largest value is max_norm
    (448: fp8 e4m3, 6: fp4 e2m1), RoundUp as aiter's FlyDSL kernels (_e8m0_roundup): am / scale <= max_norm."""
    a_ = fx.min(fx.max(am, F32(1.0e-10)), F32(max_norm * 6.646139978924579e35))
    u = (a_ * F32(1.0 / max_norm)).bitcast(I32)
    e = ((u >> 23) & 255) + ((u & 0x7FFFFF) != 0).select(I32(1), I32(0))
    e = fx.min(e, I32(254))
    return e, ((I32(254) - e) << 23).bitcast(F32)


def fp4_grid(y):
    """y (already divided by the group scale, |y| <= 6) -> the nearest fp4 e2m1 value {0, .5, 1, 1.5, 2, 3, 4, 6},
    ties to the even code (round to one mantissa bit, step 0.5 below 1)."""
    ay = fx.max(y, F32(0.0) - y)
    lo, mid = ay < F32(2.0), ay < F32(4.0)
    inv_step = lo.select(F32(2.0), mid.select(F32(1.0), F32(0.5)))
    step = lo.select(F32(0.5), mid.select(F32(1.0), F32(2.0)))
    q = F32(hx.intrin(hx.T.f32, "llvm.rint.f32", [ay * inv_step])) * step
    q = fx.min(q, F32(6.0))
    return (y < F32(0.0)).select(F32(0.0) - q, q)


def mx4_pack(vals, am):
    """ATOM's A4 activation format in an fp8 container: vals (4 k f32) rounded to MXFP4 values of the group with max
    |x| = am, stored as e4m3 bytes under the group's fp4 scale (every fp4 value is exact in e4m3), so the FP4 x FP8
    WMMA computes what ATOM's FP4 x FP4 one does. -> (k dwords, e8m0 byte)."""
    code, inv = mx_code(am, 6.0)
    return fp8_pack([fp4_grid(F32(v) * inv) for v in vals], F32(1.0)), code


def finite0(v):
    """v, or 0 where it is NaN / inf. ATOM's pad rows can hold anything: a NaN row would rank every expert first (256
    experts selected, more down units than waves, the combine waits out its 2 s poll) and poison the launch."""
    return (fx.max(v, F32(0.0) - v) < F32(3.0e38)).select(v, F32(0.0))


def xload(addr, n):
    """n bf16 of ATOM's x at addr. x was written by the previous kernel on the stream with ordinary stores, at an
    address this kernel also read one layer earlier; an ordinary load here can return that earlier content (whole rows of
    the block's output were then computed from a stale x, 11-16). Device-scope loads (8 bytes each) read the current data."""
    if XDEV:
        ws = [hx.gload(addr + i64(8 * i), fx.Int64, scope="agent") for i in range(n // 4)]
        return fx.Vector.from_elements(ws, fx.Int64).bitcast(BF16)
    return hx.gload(addr, BF16, n)


def quant_x(c, fp4=False):
    """ATOMIO prologue: the normed input x [nrows][2048] bf16 (row-major, as ATOM holds it) -> MXFP8 1x32 in the WMMA B
    layout at LDS XQ / XQS (post.quant_rows without the rstd). Thread t: token t % 16 (rows past nrows repeat the last
    row, so they select no further experts), K128 step t // 16."""
    t, sm = c["t"], c["sm"]
    tok = t % 16
    row = fx.min(tok, c["nrows"] - 1)
    s_ = t // 16
    raw = [xload(c["xin"] + i64((row * HID + s_ * 128 + q * 8) * 2), 8).to(F32) for q in range(16)]
    sc = I32(0)
    for cb in range(4):
        v = [finite0(F32(raw[cb * 4 + q // 8][q % 8])) for q in range(32)]
        am = fx.max(v[0], F32(0.0) - v[0])
        for q in range(1, 32):
            am = fx.max(am, fx.max(v[q], F32(0.0) - v[q]))
        if fp4:
            pk, code = mx4_pack(v, am)
            sc = sc | (code << (8 * cb))
            for h in range(2):
                hx.lstore(sm + XQ + s_ * 2048 + (h * 16 + tok) * 64 + cb * 16, vec(pk[h * 4:h * 4 + 4], I32))
        else:
            code, inv = fp8_code(am)
            sc = sc | (code << (8 * cb))
            for h in range(2):
                hx.lstore(sm + XQ + s_ * 2048 + (h * 16 + tok) * 64 + cb * 16, vec(fp8_pack(v[h * 16:h * 16 + 16], inv), I32))
    hx.lstore(sm + XQS + (s_ * 16 + tok) * 4, sc)


def hw_fp4_bytes(v, code):
    """8 f32 (bf16 values) / 2^(code - 127) -> 8 fp4 nibbles with the instruction ATOM's quant kernel uses
    (v_cvt_scalef32_pk8_fp4_bf16), each nibble then written as the e4m3 byte of the same value: 2 dwords."""
    nib = I32(llvm.inline_asm(hx.i32(), [hx._v(vec(v).to(BF16)), hx._v((code << 23).bitcast(F32))],
                              "v_cvt_scalef32_pk8_fp4_bf16 $0, $1, $2", "=v,v,v", has_side_effects=False))
    by = []
    for e in range(8):
        c4 = (nib >> (4 * e)) & 15
        m = c4 & 7
        by.append((m == 0).select(I32(0), (m == 1).select(I32(0x30), (((m >> 1) + 6) << 3) | ((m & 1) << 2))) | ((c4 & 8) << 4))
    return [by[0] | (by[1] << 8) | (by[2] << 16) | (by[3] << 24), by[4] | (by[5] << 8) | (by[6] << 16) | (by[7] << 24)]


XQDONE = 384                                          # cnt: the quantizing CTAs that have published their token row


@ASTRewriter.transform
def quant_row_a4(c, P, cnt, tokq):
    """ATOM mode, A4: this CTA quantizes token row tokq of x to MXFP4 (in the fp8 container, see mx4_pack) and publishes
    it in the XQ / XQS layouts at the global xq8 / xqs8, for every expert CTA to copy. Thread t: columns 8 t .. 8 t + 7;
    a 32-column group is 4 neighbouring lanes. (Every CTA quantizing all of x itself cost 12 us on the router CTAs.)"""
    t, l = c["t"], c["l"]
    row = fx.min(tokq, c["nrows"] - 1)
    raw = xload(c["xin"] + i64((row * HID + t * 8) * 2), 8).to(F32)
    v = [finite0(F32(raw[e])) for e in range(8)]
    am = fx.max(v[0], F32(0.0) - v[0])
    for e in range_constexpr(1, 8):
        am = fx.max(am, fx.max(v[e], F32(0.0) - v[e]))
    amb = am.bitcast(I32)
    for d in range_constexpr(1, 4):
        am = fx.max(am, hx.lane_gather(amb, (l & 28) | ((l + d) & 3)).bitcast(F32))
    code, inv = mx_code(am, 6.0)
    if fx.const_expr(HWQ):
        pk = hw_fp4_bytes(v, code)
    else:
        pk = fp8_pack([fp4_grid(v[e] * inv) for e in range(8)], F32(1.0))
    s_ = t // 16
    hx.stx_dev(P["xq8"] + i64(s_ * 2048 + (((t % 4) // 2) * 16 + tokq) * 64 + ((t // 4) % 4) * 16 + (t % 2) * 8), vec(pk, I32))
    sc = code
    for d in range_constexpr(1, 4):
        sc = sc | (hx.lane_gather(code, (l & 16) | (4 * d)) << (8 * d))
    if l % 16 == 0:
        hx.gstore(P["xqs8"] + i64((s_ * 16 + tokq) * 4), sc, scope="agent")
    hx.wait_storecnt0()
    gpu.barrier()
    if t == 0:
        hx.atomic_add(cnt + i64(XQDONE), 1)


def router_x(c):
    """ATOMIO: this wave's 256 columns of x (bf16) into its xbuf in the router GEMM's operand layout (stage.pack_x)."""
    l, w = c["l"], c["w"]
    row = fx.min(l % 16, c["nrows"] - 1)
    vs = [xload(c["xin"] + i64((row * HID + (w * 8 + k) * 32 + (l // 16) * 16) * 2), 16).to(F32) for k in range(8)]
    for k in range(8):
        hx.lstore(c["xbuf"] + k * 1024 + l * 32, vec([finite0(F32(vs[k][e])) for e in range(16)]).to(BF16))


@ASTRewriter.transform
def roles(c, P, cnt, stamp):
    """The early roles of CTAs 0..32, and the prologue (rstd, x -> MXFP8 in LDS) of every CTA."""
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    if fx.const_expr(ATOMIO):
        if fx.const_expr(QFIRST):
            quant_x(c, A4)
        else:
            if fx.const_expr(not SHB):
                if (b >= 16) & (b < 32):              # the MXFP8 shared expert's CTAs: MXFP8 x
                    quant_x(c, False)
            if fx.const_expr(A4):
                if (b >= 33) & (b < 49):              # 16 CTAs that are idle until the tag: one token row each -> global
                    quant_row_a4(c, P, cnt, b - 33)
            else:
                if b >= 33:
                    quant_x(c, False)
    else:
        # x arrives quantized by its producer (MXFP8 1x32 of x (1 + w), not yet scaled by rstd: the linear outputs are)
        for it in range_constexpr(8):
            hx.async_to_lds_b128(P["xq8"] + i64((c["t"] + it * 256) * 16), sm + XQ + (c["t"] + it * 256) * 16, 0)
        if c["t"] < 64:
            hx.async_to_lds_b128(P["xqs8"] + i64(c["t"] * 16), sm + XQS + c["t"] * 16, 0)
    if b < 16:
        if fx.const_expr(23 in STX):
            stamp(23)
        if fx.const_expr(ATOMIO):
            router_x(c)
        else:
            burst(c, P["xg1"] + i64(w * 8192), 8, c["xbuf"], 0)
        if fx.const_expr(20 in STX):
            stamp(20)
        hx.wait_asynccnt0()
        if fx.const_expr(21 in STX):
            stamp(21)
        tot = reduce8(c, wmma_run(c, 8, c["wbuf"], fx.Vector.filled(8, 0.0, F32)))
        if fx.const_expr(22 in STX):
            stamp(22)
        if w == 0:
            if fx.const_expr(A4):                         # ATOM ranks its bf16 router output: the same ties and weights
                st_f32x8(P["lg"] + i64((tok * 256 + b * 16 + (l // 16) * 8) * 4), [bfr(F32(tot[e])) for e in range(8)])
            else:
                st_f32x8(P["lg"] + i64((tok * 256 + b * 16 + (l // 16) * 8) * 4), [F32(tot[e]) for e in range(8)])
        if fx.const_expr(MB):                             # the logits as tagged words: no store wait, no barriers
            router_out_mb(c, P, tot)
        else:
            hx.wait_storecnt0()
            gpu.barrier()
            hx.fence_release("cluster")
            hx.cluster_barrier()
            hx.fence_acquire("cluster")
        stamp(6)
        rank_publish(c, P, cnt)
        stamp(7)
    if fx.const_expr(ATOMIO):
        if fx.const_expr(not QFIRST):
            if fx.const_expr(A4):
                if (b < 16) | (b >= 33):                  # the routed experts' x: copy the published MXFP4 rows
                    if w == 0:
                        hx.wait_count(cnt + i64(XQDONE), 16, 25)
                    gpu.barrier()
                    for it in range_constexpr(8):
                        hx.async_to_lds_b128(P["xq8"] + i64((c["t"] + it * 256) * 16), sm + XQ + (c["t"] + it * 256) * 16, 0)
                    if c["t"] < 64:
                        hx.async_to_lds_b128(P["xqs8"] + i64(c["t"] * 16), sm + XQS + c["t"] * 16, 0)
            else:
                if b < 16:                                # (the router CTAs quantize after their GEMM: it is their critical path)
                    quant_x(c, False)
        if (w == 0) & (l < 16):                       # x is already normed: rstd = 1
            hx.lstore(sm + RS + tok * 4, F32(1.0))
    else:
        rstd_reduce(c, P["part1"], 256)
    hx.wait_asynccnt0()
    gpu.barrier()
    rs = hx.lload(sm + RS + tok * 4, F32)
    if fx.const_expr(SHB):
        if (b >= 16) & (b < 32):
            shared_bf16(c, P, cnt)
    if (b >= 16) & (b < 32) & (not SHB):
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
            act = [silu_mul([bfr(F32(tot[i][e]) * rs) for e in range(8)], [bfr(F32(tot[2 + i][e]) * rs) for e in range(8)], F32(1.0)) for i in range(2)]
            am = amax8(act[0] + act[1])
            am = fx.max(am, hx.lane_gather(am.bitcast(I32), l ^ 16).bitcast(F32))
            code, inv = fp8_code(am)
            j = b - 16
            for i in range_constexpr(2):
                lane2 = i * 16 + tok
                stg(P["GSU"] + i64((j // 4) * 2048 + lane2 * 64 + (j % 4) * 16 + (l // 16) * 8), vec(fp8_pack(act[i], inv), I32))
                if l < 16:
                    hx.gstore(P["GSS"] + i64(((j // 4) * 32 + lane2) * 16 + (j % 4) * 4), code, scope="agent")
        hx.wait_storecnt0()
        gpu.barrier()
        hx.fence_release("cluster")
        hx.cluster_barrier()
        hx.fence_acquire("cluster")
        # the shared expert's down, one tile a wave: 8 KB weights + 256 B scales, x 8 KB + 2 KB scales
        sw = (b - 16) * 8 + w
        burst(c, P["SD"] + i64(sw * 4) * 2048, 8, c["wbuf"], 2)
        if l < 16:
            hx.async_to_lds_b128(P["SDS"] + i64(sw * 256 + l * 16), c["wbuf"] + 8192 + l * 16, 2)
        burst(c, P["GSU"], 8, c["wbuf"] + 8704, 0)
        burst(c, P["GSS"], 2, c["wbuf"] + 16896, 0)
        hx.wait_asynccnt0()
        a = fx.Vector.filled(8, 0.0, F32)
        for k in range_constexpr(4):
            a = hx.wmma_mx(hx.lload(c["wbuf"] + k * 2048 + l * 64, I32, 16), hx.lload(c["wbuf"] + 8192 + (k * 16 + tok) * 4, I32),
                           hx.lload(c["wbuf"] + 8704 + k * 2048 + l * 64, I32, 16), xscale(c["wbuf"] + 16896, k, l), a)
        st8(P["sdn"] + i64((tok * HID + sw * 16 + (l // 16) * 8) * 4), [F32(a[e]) for e in range(8)])
        hx.wait_storecnt0()
        if l == 0:
            hx.atomic_add(cnt + i64(DDONE), 1)
    if b == 32:
        if fx.const_expr(ATOMIO):
            router_x(c)
        else:
            burst(c, P["xg1"] + i64(w * 8192), 8, c["xbuf"], 0)
        hx.wait_asynccnt0()
        tot = reduce8(c, wmma_run(c, 8, c["wbuf"], fx.Vector.filled(8, 0.0, F32)))
        if (w == 0) & (l < 16):
            hx.gstore(P["sgate"] + i64(tok * 4), sigmoid(F32(tot[0]) * hx.lload(sm + RS + tok * 4, F32)), scope="agent")
        hx.wait_storecnt0()
        gpu.barrier()
        if c["t"] == 0:
            hx.atomic_add(cnt + i64(DDONE), 1)


@ASTRewriter.transform
def down_unit(c, P, cnt, E, g):
    """Output tiles 4 g .. 4 g + 3 of expert E's down: 16 KB + 1 KB scales cold, issued before the wait on E's
    activation (8 KB + 2 KB scales). A lane stores to the slot (its token, the rank E has for it)."""
    l = c["l"]
    tok = l % 16
    buf = c["wbuf"]
    hx.wait_asynccnt0()
    t0 = i64((E * 128 + 4 * g) * 4)
    burst(c, P["ED"] + t0 * 1024, 16, buf, 2)
    burst(c, P["EDS"] + t0 * 64, 1, buf + 16384, 2)
    hx.wait_count(cnt + i64(ACTDONE) + i64(E * 4), 16, 21)
    burst(c, P["GE"] + i64(E) * 8192, 8, buf + 17408, 0)
    burst(c, P["GS"] + i64(E) * 2048, 2, buf + 25600, 0)
    sl = dload(P["sel"] + i64(tok * 32), I32)
    rk, has = I32(0), I32(0)
    for q in range_constexpr(8):
        hit = I32(sl[q]) == E
        rk = hit.select(I32(q), rk)
        has = has | hit.select(I32(1), I32(0))
    hx.wait_asynccnt0()
    xr = [hx.lload(buf + 17408 + k * 2048 + l * 64, I32, 16) for k in range(4)]
    xs = [xscale(buf + 25600, k, l) for k in range(4)]
    for tl in range(I32(0), I32(4), I32(1)):
        acc = fx.Vector.filled(8, 0.0, F32)
        for k in range_constexpr(4):
            acc = wmma_fp4(hx.lload(buf + (tl * 4 + k) * 1024 + l * 32, I32, 8), hx.lload(buf + 16384 + ((tl * 4 + k) * 16 + tok) * 4, I32), xr[k], xs[k], acc)
        if has != 0:
            st8(P["slot"] + i64(((tok * 8 + rk) * HID + (4 * g + tl) * 16 + (l // 16) * 8) * 4), [F32(acc[e]) for e in range(8)])
    hx.wait_storecnt0()
    if l == 0:
        hx.atomic_add(cnt + i64(DDONE), 1)


@ASTRewriter.transform
def down_unit_n(c, P, cnt, E, tile0, n):
    """n (<= 8) output tiles of expert E's down from tile0, so that every wave runs one down unit (no second round).
    Tiles 0..3 as down_unit; tiles 4.. are issued once x is in registers, into the LDS the activation used, their scales
    into the first tile's slot once that tile is done. Loads are clamped to the expert's last four tiles."""
    l = c["l"]
    tok = l % 16
    buf = c["wbuf"]
    drain()
    if TDM:
        E = _ufl(E)
        tile0 = _ufl(tile0)
        buf = _ufl(buf)
    ba = (tile0 < 124).select(tile0, I32(124))
    da = tile0 - ba
    bb = (tile0 + 4 < 124).select(tile0 + 4, I32(124))
    db = tile0 + 4 - bb
    ta = i64((E * 128 + ba) * 4)
    tb = i64((E * 128 + bb) * 4)
    tload(c, P["ED"] + ta * 1024, 16, buf, 2)
    tload(c, P["EDS"] + ta * 64, 1, buf + 16384, 2)
    hx.wait_count(cnt + i64(ACTDONE) + i64(E * 4), 16, 21)
    c["stamp"](12)
    tload(c, P["GE"] + i64(E) * 8192, 8, buf + 17408, 0)
    tload(c, P["GS"] + i64(E) * 2048, 2, buf + 25600, 0)
    if fx.const_expr("slv" in c):                     # already loaded with the tag
        sl = c["slv"]
    else:
        sl = dload(P["sel"] + i64(tok * 32), I32)
    wdn = F32(1.0)
    if fx.const_expr(A4 and "uw" in c):
        wdn = uw_calc(c, c["uw"], E)
    rk, has = I32(0), I32(0)
    for q in range_constexpr(8):
        hit = I32(sl[q]) == E
        rk = hit.select(I32(q), rk)
        has = has | hit.select(I32(1), I32(0))
    drain()
    xb = hx.opaque_v(buf + 17408) if TDM else buf + 17408
    xr = [hx.lload(xb + k * 2048 + l * 64, I32, 16) for k in range(4)]
    xs = [xscale(xb + 8192, k, l) for k in range(4)]
    if n > 4:
        tload(c, P["ED"] + tb * 1024, 16, buf + 17408, 2)
    for tl in range(I32(0), n, I32(1)):
        ina = tl < 4
        ti = ina.select(da + tl, db + tl - 4)
        wb = ina.select(buf, buf + 17408) + ti * 4096
        sb = ina.select(buf + 16384, buf) + ti * 256
        if tl == 4:
            drain()
        acc = fx.Vector.filled(8, 0.0, F32)
        if fx.const_expr(TDM):
            wb = hx.opaque_v(wb)
            sb = hx.opaque_v(sb)
        if fx.const_expr(BATCH):
            wd_ = [hx.lload(wb + k * 1024 + l * 32, I32, 8) for k in range(4)]
            sd_ = [hx.lload(sb + (k * 16 + tok) * 4, I32) for k in range(4)]
            if hx.opaque_s(I32(1)) != 0:
                for k in range_constexpr(4):
                    acc = wmma_fp4(wd_[k], sd_[k], xr[k], xs[k], acc)
        else:
            for k in range_constexpr(4):
                acc = wmma_fp4(hx.lload(wb + k * 1024 + l * 32, I32, 8), hx.lload(sb + (k * 16 + tok) * 4, I32), xr[k], xs[k], acc)
        if has != 0:
            if fx.const_expr(A4 and "uw" in c):           # A4: the route weight multiplies the down's output (ATOM's gather-reduce)
                st8(P["slot"] + i64(((tok * 8 + rk) * HID + (tile0 + tl) * 16 + (l // 16) * 8) * 4), [F32(acc[e]) * wdn for e in range(8)])
            else:
                st8(P["slot"] + i64(((tok * 8 + rk) * HID + (tile0 + tl) * 16 + (l // 16) * 8) * 4), [F32(acc[e]) for e in range(8)])
        if (tl == 0) & (n > 4):
            tload(c, P["EDS"] + tb * 64, 1, buf, 2)
    hx.wait_storecnt0()
    if l == 0:
        hx.atomic_add(cnt + i64(DDONE), 1)


@ASTRewriter.transform
def early_poll(cnt, b):
    """Until this CTA's expert is named by some token's top-8 (certain: no speculation), or every token has published."""
    t0 = hx.clock()
    while (hx.gload(cnt + i64(FL) + i64(b * 4), I32, scope="agent") != GEN) & (hx.gload(cnt + i64(NRANKED), I32, scope="agent") < 16) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
        hx.s_sleep(EWS)


@ASTRewriter.transform
def flow(c, P, cnt, stamp):
    b, w, l, sm = c["b"], c["w"], c["l"], c["sm"]
    tok = l % 16
    roles(c, P, cnt, stamp)
    stamp(1)
    early = I32(0)
    if fx.const_expr(EARLY and A4 and not SPEC):
        if b >= 33:
            if w == 0:
                early_poll(cnt, b)
            gpu.barrier()
            early = (hx.gload(cnt + i64(FL) + i64(b * 4), I32, scope="agent") == GEN).select(I32(1), I32(0))
            if early != 0:
                stamp(13)
                re = unit_run(c, P, cnt, b, w, I32(0), I32(0))
                stamp(8)
                unit_epilogue_w(c, P, cnt, b, w, re, F32(0.0))
    # own units: owners b >= 33 before the tag; a selected router CTA after it
    own = I32(0)
    if fx.const_expr(SPEC):
        if b >= 33:
            own = I32(1)
        else:
            if b < 16:
                own = (hx.gload(cnt + i64(FL) + i64(b * 4), I32, scope="agent") == GEN).select(I32(1), I32(0))
        spec = (b >= 33).select(I32(1), I32(0))
    else:
        if fx.const_expr(MB):                             # one wave collects the tagged bitmap and top-8 rows -> LDS
            if w == 0:
                mb_collect(c, P)
                stamp(19)
            gpu.barrier()
            stamp(10)
        else:
            if (b < 16) | (b >= 33):
                if fx.const_expr(POLL1):                      # one wave polls, the CTA's other waves park at the barrier
                    if w == 0:
                        hx.wait_count(cnt + i64(NRANKED), 16, 24, ws=TWS)
                    gpu.barrier()
                else:
                    hx.wait_count(cnt + i64(NRANKED), 16, 24)
                stamp(10)
                if fx.const_expr(not FASTTAG):
                    own = (hx.gload(cnt + i64(FL) + i64(b * 4), I32, scope="agent") == GEN).select(I32(1), I32(0))
            else:
                if fx.const_expr(FASTTAG):
                    hx.wait_count(cnt + i64(NRANKED), 16, 22)
        spec = I32(0)
        if fx.const_expr(FASTTAG):                        # every load the tag unlocks goes out together: the selected-expert
            uw = uw_load(c, P)                            # bitmap (own flag, indices, U) and the tokens' top-8 rows
            A0 = assign(c, cnt)
            if fx.const_expr(STAMPS):                     # diagnostic: the U and bitmap word 2 this wave works with
                if c["t"] % 32 == 0:
                    hx.gstore(c["st"] + i64(b * 8 + w) * (NSTAMP * 8) + i64(24 * 8), i64(A0["U"]))
                    hx.gstore(c["st"] + i64(b * 8 + w) * (NSTAMP * 8) + i64(25 * 8), i64(A0["W"][2]))
            c["slv"] = uw[0]
            c["uw"] = uw
            if fx.const_expr(ROWN):
                own = (((b < 16) | (b >= 33)) & (A0["sel"] != 0)).select(I32(1), I32(0))
            else:
                own = ((b >= 33) & (A0["sel"] != 0)).select(I32(1), I32(0))
            if fx.const_expr(ALLSPLIT):                   # every gate/up unit fits as a two-wave unit: no owner units
                asp = (A0["U"] * 16 <= 956).select(I32(1), I32(0))
                own = (asp != 0).select(I32(0), own)
    done = I32(0)
    r = [I32(0)] + [fx.Vector.filled(8, 0.0, F32) for _ in range(4)] + [I32(0)]
    if (own != 0) & (early == 0):
        if fx.const_expr((not SPEC) and WPRE):
            if fx.const_expr(FASTTAG):
                wv0 = uw_calc(c, uw, b)
            else:
                wv0 = unit_weight(c, P, b)
            if fx.const_expr(CH4):
                r = [I32(1)] + gu_piece(c, P, b, w, [0, 1, 2, 3]) + [I32(0)]
            else:
                r = unit_run(c, P, cnt, b, w, spec, spec)
            stamp(8)
            unit_epilogue_w(c, P, cnt, b, w, r, wv0)
        else:
            r = unit_run(c, P, cnt, b, w, spec, spec)
            stamp(8)
            done = r[0]
    if fx.const_expr(not (FASTTAG and not SPEC)):
        if (b >= 16) & ((own == 0) | (r[5] != 0)):     # (the tag was not seen yet: no unit, or not inside it)
            hx.wait_count(cnt + i64(NRANKED), 16, 22)
    stamp(11)
    if fx.const_expr(FASTTAG and not SPEC):
        A = A0
    else:
        A = assign(c, cnt)
    U = A["U"]
    stamp(2)
    if (done != 0) & (A["sel"] != 0):
        unit_epilogue(c, P, cnt, b, w, r)
    asp2 = I32(0)
    if fx.const_expr(ALLSPLIT and FASTTAG and not SPEC):
        asp2 = asp
    if ((A["sel"] == 0) | (asp2 != 0) | (own == 0)) & ((b < 16) | (b >= 33)):
        NR = (U + A["XN"]) * 8
        split = I32(0)
        hcr = A["hidx"] // 8                          # this CTA's helper rank, of Hc
        Hc = A["H"] // 8
        # mixed split: the first Fh helper CTAs run 8 whole units, the others 4 units on wave pairs (K halves), so that
        # one round covers the NR helper units (Fh = 0: every helper unit is split)
        Fh = fx.max((NR - 4 * Hc + 3) // 4, I32(0))
        if fx.const_expr(KSPLIT):
            split = ((NR <= 8 * Hc) & (hcr >= Fh)).select(I32(1), I32(0))
        split = (asp2 != 0).select(I32(1), split)
        if split != 0:
            rs_ = 8 * Fh + (hcr - Fh) * 4 + w % 4
            val = (rs_ < NR).select(I32(1), I32(0))
            # all-split: CTAs >= 33 first, then the routers; units = every (expert, j < 16)
            ra_ = (b >= 33).select(b - 33, 223 + b) * 4 + w % 4
            rs_ = (asp2 != 0).select(ra_, rs_)
            val = (asp2 != 0).select((ra_ < U * 16).select(I32(1), I32(0)), val)
            ph = [fx.Vector.filled(8, 0.0, F32) for _ in range(4)]
            Es, js, wvs = I32(0), I32(0), F32(0.0)
            if val != 0:
                firsts = rs_ < U * 8
                Es = nth(A, (asp2 != 0).select(rs_ // 16, firsts.select(rs_ // 8, A["x0"] + (rs_ - U * 8) // 8)))
                js = (asp2 != 0).select(rs_ % 16, firsts.select(8 + rs_ % 8, rs_ % 8))
                if fx.const_expr(FASTTAG and not SPEC):
                    wvs = uw_calc(c, uw, Es)
                else:
                    wvs = unit_weight(c, P, Es)
                if fx.const_expr(CH4):
                    if w < 4:
                        ph = gu_piece(c, P, Es, js, [0, 1])
                    else:
                        ph = gu_piece(c, P, Es, js, [2, 3])
                else:
                    if w < 4:
                        ph = unit_half(c, P, Es, js, 0)
                    else:
                        ph = unit_half(c, P, Es, js, 1)
                for i in range_constexpr(4):
                    hx.lstore(c["wbuf"] + i * 1024 + l * 32, ph[i])
                stamp(14)
            gpu.barrier()
            if fx.const_expr(XB2):
                # the partner's sums are read BEFORE a second barrier: after the first one the partner goes on to its
                # down unit, whose weight load lands in the same LDS (junk sums -> a junk activation scale, 11-16)
                pb = c["wbuf"] + (w < 4).select(I32(4 * PW), I32(0))
                tot = [ph[i] + hx.lload(pb + i * 1024 + l * 32, F32, 8) for i in range(4)]
                gpu.barrier()
                if (val != 0) & (w < 4):
                    unit_epilogue_w(c, P, cnt, Es, js, [I32(1)] + tot + [I32(0)], wvs)
            else:
                if (val != 0) & (w < 4):
                    tot = [ph[i] + hx.lload(c["wbuf"] + 4 * PW + i * 1024 + l * 32, F32, 8) for i in range(4)]
                    unit_epilogue_w(c, P, cnt, Es, js, [I32(1)] + tot + [I32(0)], wvs)
        else:
            for k in range(I32(0), I32(2), I32(1)):
                rr = A["hidx"] + k * A["H"]
                if rr < NR:
                    first = rr < U * 8
                    E = nth(A, first.select(rr // 8, A["x0"] + (rr - U * 8) // 8))
                    j = first.select(8 + rr % 8, rr % 8)
                    if fx.const_expr(WPRE):
                        if fx.const_expr(FASTTAG and not SPEC):
                            wvh = uw_calc(c, uw, E)
                        else:
                            wvh = unit_weight(c, P, E)
                        if fx.const_expr(CH4):
                            r = [I32(1)] + gu_piece(c, P, E, j, [0, 1, 2, 3]) + [I32(0)]
                        else:
                            r = unit_run(c, P, cnt, E, j, I32(0), I32(0))
                        stamp(14)
                        unit_epilogue_w(c, P, cnt, E, j, r, wvh)
                    else:
                        r = unit_run(c, P, cnt, E, j, I32(0), I32(0))
                        stamp(14)
                        unit_epilogue(c, P, cnt, E, j, r)
    stamp(3)
    if fx.const_expr(DN1):
        nt = (U <= 64).select(I32(4), (U <= 78).select(I32(5), (U <= 93).select(I32(6), (U <= 107).select(I32(7), I32(8)))))
        upe = (U <= 64).select(I32(32), (U <= 78).select(I32(26), (U <= 93).select(I32(22), (U <= 107).select(I32(19), I32(16)))))
        rr = b * 8 + w
        if rr < U * upe:
            tile0 = (rr % upe) * nt
            down_unit_n(c, P, cnt, nth(A, rr // upe), tile0, (128 - tile0 < nt).select(128 - tile0, nt))
    else:
        upe = I32(32)
        for k in range(I32(0), I32(2), I32(1)):
            rr = b * 8 + w + k * 2048
            if rr < U * 32:
                down_unit(c, P, cnt, nth(A, rr // 32), rr % 32)
    stamp(4)
    if (b < 128) & (w == 0):
        if fx.const_expr(CPRE and not ATOMIO):            # the inputs that are final long before the last down unit
            col0 = b * 16 + (l // 16) * 8
            h1p = hx.gload(P["h1"] + i64((tok * HID + col0) * 2), BF16, 8).to(F32)
        hx.wait_count(cnt + i64(DDONE), U * upe + 129, 23, ws=CWS)
        stamp(15)
        col = b * 16 + (l // 16) * 8
        sg = hx.gload(P["sgate"] + i64(tok * 4), F32, scope="agent") if DLOADS else hx.gload(P["sgate"] + i64(tok * 4), F32)
        sd = dload(P["sdn"] + i64((tok * HID + col) * 4), F32)
        tot = dload(P["slot"] + i64((tok * 8 * HID + col) * 4), F32)
        for q in range_constexpr(1, 8):
            tot = tot + dload(P["slot"] + i64(((tok * 8 + q) * HID + col) * 4), F32)
        if fx.const_expr(ATOMIO):                         # the MoE output alone (ATOM's norm adds the residual)
            # a plain store, waited for: the next kernel on the stream reads these rows as soon as this one exits (a
            # device-scope exchange store left in flight at the exit showed up as a few stale rows in ATOM, 11-16)
            hx.gstore(c["out"] + i64((tok * HID + col) * 2), vec([bfr(F32(tot[e]) + sg * F32(sd[e])) for e in range(8)]).to(BF16))
            hx.wait_storecnt0()
        else:
            if fx.const_expr(CPRE):
                h1 = h1p
            else:
                h1 = hx.gload(P["h1"] + i64((tok * HID + col) * 2), BF16, 8).to(F32)
            hx.stx_dev(P["h2"] + i64((tok * HID + col) * 2), vec([bfr(F32(h1[e]) + F32(tot[e]) + sg * F32(sd[e])) for e in range(8)]).to(BF16))
    if fx.const_expr(EXITW):                              # nothing of this wave may be in flight when it ends: the LDS and
        hx.wait_tensorcnt(0)                              # the memory it targets belong to the next kernel by then
        hx.wait_asynccnt0()
        hx.wait_storecnt0()
        llvm.inline_asm(None, [], "s_wait_loadcnt 0x0", "", has_side_effects=True)


def build():
    keyed = ("vmoe2", 56, RT3, RT3F, MB32, D32, BANK, S128, G128, LWS, STX, MBS, MB, EARLY, EWS, XB2, DBG, WC0, EXITW, DLOADS, XDEV, INFENCE, HWQ, SHB, A4, ROWN, QFIRST, ATOMIO, STAMPS, ENTRYH, ALLSPLIT, CH4, IPF, TWS, FASTTAG, TDM, BATCH, hx.TDM_ELEM, KSPLIT, CPRE, ENDH, CWS, DN1, HORD, WPRE, POLL1, hx.WSLEEP, PROBE, WAITS, CPOL, REP, NOX, STAMPC, MIRROR_TAG, SPEC)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(tab: Int64, cnt: Int64, st: Int64, xin: Int64, out: Int64, nrows: fx.Int32):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = _ufl(t // 32)
        P = {k: hx.sload64(tab + i64(8 * i)) for i, k in enumerate(TAB)}
        if fx.const_expr(BANK or MB):                     # the launch tag: read at entry, long before its first use
            G = hx.gload(P["gen"], I32)
        if fx.const_expr(BANK):
            cnt0 = cnt
            cnt = cnt + i64((G & 1) * BANKB)
        c = dict(st=st, cnt=cnt, t=t, b=b, w=w, l=t % 32, sm=sm, wbuf=sm + w * PW, xbuf=sm + w * PW + 16384, xin=xin, out=out, nrows=_ufl(nrows))
        if fx.const_expr(BANK or MB):
            c["G"] = G

        def stamp(k):
            if fx.const_expr(STAMPS):
                if t % 32 == 0:
                    hx.gstore(st + i64(b * 8 + w) * (NSTAMP * 8) + i64(k * 8), hx.clock())

        c['stamp'] = stamp
        if fx.const_expr(MB):
            hx.TOFLAG[0] = cnt + i64(2048)                # poll timeouts: count at cnt[512], last site at cnt[513]
        if fx.const_expr(IPF):                            # instruction prefetch of the next IPF KB of code (DSv4.1 monokernel)
            for off in range_constexpr(0, IPF * 1024, 4096):
                llvm.inline_asm(None, [], "s_prefetch_inst_pc_rel %d, null, 31" % off, "", has_side_effects=True)
        stamp(0)
        if fx.const_expr(INFENCE):
            # x was written by the previous kernel on the stream with ordinary stores, at an address this kernel also read
            # one layer earlier: an acquire fence at device scope before the first load, so no stale line is served
            hx.fence_acquire("agent")
        prestage(c, P)
        if fx.const_expr(ENTRYH):
            hx.handoff(cnt, t, True)
        stamp(9)
        flow(c, P, cnt, stamp)
        if fx.const_expr(ENDH):
            hx.handoff(cnt + i64(192), t, True)
        if fx.const_expr(BANK):                           # the next launch's counters: the bank this launch did not use
            bank_zero(c, cnt0 + i64((1 - (G & 1)) * BANKB))
        if fx.const_expr(BANK or MB):                     # the next launch's tag (every word of this launch is stale to it)
            mb_next(c, P)
        stamp(5)
        _ = keyed

    @flyc.jit
    def launch(tab: Int64, cnt: Int64, st: Int64, xin: Int64, out: Int64, nrows: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(tab, cnt, st, xin, out, nrows).launch(grid=(256,), block=(256,), smem=LDS, cluster=(16, 1, 1), stream=stream)

    return launch
