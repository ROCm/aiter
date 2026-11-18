"""Measurement: how fast one wave pulls 64 KB of cold weights into a 3-slot LDS ring (8 chunks of 8 KB), no math.
NA CTAs active. MODE 0: issue ahead by 2 chunks, wait for the oldest. MODE 1: issue a chunk, wait for all (serial).
MODE 2: as 0 plus one LDS read of every 512 B of the chunk (a consumer). Stamps: st[wave][0] start, [1] end."""
import os
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
import hxdev as hx
from head import _ufl
from post import wmma_fp4
I32, i64 = fx.Int32, fx.Int64
PW, SLOT = 33 * 1024, 8192


def build(MODE, NCH=8):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = _ufl(t // 32)
        l = t % 32
        buf = sm + w * PW
        base = src + i64(b * 8 + w) * i64(stride)

        def issue(ch):
            for q in range_constexpr(16):
                hx.async_to_lds_b128(base + i64(ch) * 8192 + i64(q * 512 + l * 16), buf + (ch % 3) * SLOT + q * 512 + l * 16, 2)

        gpu.barrier()
        if b < na:
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32, hx.clock())
            sink = I32(0)
            acc = fx.Vector.filled(8, 0.0, fx.Float32)
            xr = [hx.lload(sm + 8 * PW + k * 2048 + l * 64, I32, 16) for k in range(8)]
            xs = hx.lload(sm + 8 * PW + 32768 + (l % 16) * 32, I32, 8)
            if fx.const_expr(MODE == 1):
                for ch in range(I32(0), I32(NCH), I32(1)):
                    issue(ch)
                    hx.wait_asynccnt0()
            else:
                NC = {6: 2, 7: 8, 9: 8, 10: 4}.get(MODE, 1)          # independent accumulator chains
                accs = [fx.Vector.filled(8, 0.0, fx.Float32) for _ in range(NC)]
                if fx.const_expr(MODE < 8):
                    issue(I32(0))
                    issue(I32(1))
                for ch in range(I32(0), I32(NCH), I32(1)):
                    if fx.const_expr(MODE >= 8):
                        pass
                    elif ch + 2 < NCH:
                        issue(ch + 2)
                        rocdl.s_wait_asynccnt(32)
                    else:
                        if ch + 1 < NCH:
                            rocdl.s_wait_asynccnt(16)
                        else:
                            hx.wait_asynccnt0()
                    if fx.const_expr(MODE >= 6):
                        for k in range_constexpr(8):
                            accs[k % NC] = wmma_fp4(hx.lload(buf + (ch % 3) * SLOT + k * 1024 + l * 32, I32, 8), hx.lload(buf + 3 * SLOT + (k * 16 + l % 16) * 4, I32),
                                                    xr[k], I32(xs[k]), accs[k % NC])
                    if fx.const_expr(MODE == 5):          # x operand and its scales held in registers for the unit
                        for k in range_constexpr(8):
                            acc = wmma_fp4(hx.lload(buf + (ch % 3) * SLOT + k * 1024 + l * 32, I32, 8), hx.lload(buf + 3 * SLOT + (k * 16 + l % 16) * 4, I32),
                                           xr[k], I32(xs[k]), acc)
                    if fx.const_expr(MODE == 3 or MODE == 4):
                        for k in range(8 * (MODE - 2)):
                            kk = k % 8
                            acc = wmma_fp4(hx.lload(buf + (ch % 3) * SLOT + kk * 1024 + l * 32, I32, 8), hx.lload(buf + 3 * SLOT + (kk * 16 + l % 16) * 4, I32),
                                           hx.lload(sm + 8 * PW + ((ch % 2) * 8 + kk) * 2048 + l * 64, I32, 16), hx.lload(sm + 8 * PW + 32768 + (kk * 16 + l % 16) * 4, I32), acc)
                    if fx.const_expr(MODE == 2):
                        for q in range(16):
                            sink = sink + hx.lload(buf + (ch % 3) * SLOT + q * 512 + l * 16, I32)
            if fx.const_expr(MODE >= 6):
                for q in range_constexpr(NC):
                    acc = acc + accs[q]
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32 + i64(8), hx.clock())
            hx.gstore(st + i64(b * 8 + w) * 32 + i64(16) + i64((l % 4) * 4), sink + fx.Float32(acc[l % 8]).bitcast(I32))

    @flyc.jit
    def launch(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        body(src, st, na, stride).launch(grid=(256,), block=(256,), smem=9 * PW + 1024, cluster=(16, 1, 1), stream=stream)
    return launch


def build2(CK, NS, LATE, COMPUTE, WAITALL=0, REPC=1, GRP=0, TDM=0, CHECK=0, NOCOPY=0, ZERO=0, NW_=64, SPREAD=0, SBAR=0):
    """64 KB a wave in chunks of CK KB through an NS-slot ring (NS * CK <= 32). LATE: the next chunk is issued after the
    compute of the chunk whose slot it reuses (NS in flight), else before it (NS - 1 ahead, one slot spare).
    WAITALL: every wait is for all outstanding copies."""
    NCH, IPC, CB = 64 // CK, CK * 2, CK * 1024

    def opnd(a, l):
        if SPREAD:
            v0, v1 = hx.lload(a + l * 16, I32, 4), hx.lload(a + 512 + l * 16, I32, 4)
            return fx.Vector.from_elements([v0[e] for e in range(4)] + [v1[e] for e in range(4)], I32)
        return hx.lload(a + l * 32, I32, 8)
    AH = NS if LATE else NS - 1

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = _ufl(t // 32)
        l = t % 32
        buf = sm + w * PW
        base = src + i64(b * 8 + w) * i64(stride)

        ubase = src + i64(_ufl(b * 8 + w)) * i64(_ufl(stride))
        ubuf = _ufl(buf)

        def issue(ch):
            if NOCOPY:
                return
            if TDM:                                   # one instruction a chunk: TDM rows of TDM bytes
                hx.tdm_load(ubase + i64(ch * CB), ubuf + (ch % NS) * CB, TDM, CB // TDM, 2)
            else:
                for q in range_constexpr(IPC):
                    hx.async_to_lds_b128(base + i64(ch * CB + q * 512) + i64(l * 16), buf + (ch % NS) * CB + q * 512 + l * 16, 2)

        if fx.const_expr(ZERO):                       # the wave's ring, x and scales as zeros (ZERO=1) or small values (2)
            zv = I32(0) if ZERO == 1 else I32(0x38383838)
            for q in range_constexpr(NS * CB // 512):
                hx.lstore(buf + q * 512 + l * 16, fx.Vector.filled(4, zv, I32))
            for q in range_constexpr(34):
                hx.lstore(sm + 8 * PW + q * 1024 + t * 4, zv)
        gpu.barrier()
        if b < na:
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32, hx.clock())
            acc = fx.Vector.filled(8, 0.0, fx.Float32)
            xr = [hx.lload(sm + 8 * PW + k * 2048 + l * 64, I32, 16) for k in range(8)]
            xs = hx.lload(sm + 8 * PW + 32768 + (l % 16) * 32, I32, 8)
            chk = I32(0)
            one = _ufl(na * 0 + 1)
            for ch in range_constexpr(min(AH, NCH)):
                issue(ch)
            for ch in range_constexpr(NCH):
                if fx.const_expr((not LATE) and ch + AH < NCH):
                    issue(ch + AH)
                out = (min(ch + AH + (0 if LATE else 1), NCH) - ch - 1) * (1 if TDM else IPC)
                if NOCOPY:
                    pass
                elif TDM:
                    hx.wait_tensorcnt(0 if WAITALL else out)
                else:
                    rocdl.s_wait_asynccnt(0 if WAITALL else out)
                # LLVM does not see a TDM load write LDS: without an opaque address it reuses the slot's earlier reads
                cb_ = hx.opaque_v(buf + (ch % NS) * CB) if (TDM or NOCOPY) else buf + (ch % NS) * CB
                if fx.const_expr(CHECK):                    # a checksum of what arrived (first word of each 16 B)
                    for q in range_constexpr(IPC):
                        chk = chk + hx.lload(cb_ + q * 512 + l * 16, I32)
                if fx.const_expr(COMPUTE and GRP):          # the LDS operands of GRP WMMAs are loaded before the first of them
                    for k0 in range_constexpr(0, CK, GRP):
                        ws_ = [opnd(cb_ + k * 1024, l) for k in range(k0, k0 + GRP)]
                        sc_ = [hx.lload(sm + 8 * PW + 32768 + 512 + ((k % 8) * 16 + l % 16) * 4, I32) for k in range(k0, k0 + GRP)]
                        if fx.const_expr(SBAR):             # the DSv4.1 monokernel's way: a scheduler barrier
                            sched_barrier()
                            for k in range_constexpr(k0, k0 + GRP):
                                acc = wmma_fp4(ws_[k - k0], sc_[k - k0], xr[k % 8], I32(xs[k % 8]), acc)
                            sched_barrier()
                        else:
                            if hx.opaque_s(one) != 0:       # a block boundary: the scheduler cannot sink the loads past it
                                for k in range_constexpr(k0, k0 + GRP):
                                    acc = wmma_fp4(ws_[k - k0], sc_[k - k0], xr[k % 8], I32(xs[k % 8]), acc)
                if fx.const_expr(COMPUTE and not GRP):
                  for _rp in range_constexpr(REPC):
                    for k in range_constexpr(CK):
                        acc = wmma_fp4(opnd(cb_ + k * 1024, l), hx.lload(sm + 8 * PW + 32768 + 512 + ((k % 8) * 16 + l % 16) * 4, I32),
                                       xr[k % 8], I32(xs[k % 8]), acc)
                if fx.const_expr(LATE and ch + AH < NCH):
                    issue(ch + AH)
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32 + i64(8), hx.clock())
            if fx.const_expr(CHECK):
                if l < 4:
                    hx.gstore(st + i64(b * 8 + w) * 32 + i64(16) + i64(l * 4), chk)
            else:
                hx.gstore(st + i64(b * 8 + w) * 32 + i64(16) + i64((l % 4) * 4), fx.Float32(acc[l % 8]).bitcast(I32))

    @flyc.jit
    def launch(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        body(src, st, na, stride).launch(grid=(256,), block=(256,), smem=9 * PW + 1024, cluster=(16, 1, 1), stream=stream)
    return launch


def build3(AHEAD, NWM=64, SC=1):
    """64 KB a wave straight from global memory into registers, no LDS: WMMA k's 1 KB of weights (32 B a lane) and its
    scales are loaded AHEAD WMMAs before use, straight-line code. NWM WMMAs. SC: scales loaded too."""
    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = _ufl(t // 32)
        l = t % 32
        base = src + i64(b * 8 + w) * i64(stride)
        gpu.barrier()
        if b < na:
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32, hx.clock())
            acc = fx.Vector.filled(8, 0.0, fx.Float32)
            xr = [hx.lload(sm + 8 * PW + k * 2048 + l * 64, I32, 16) for k in range(8)]
            xs = hx.lload(sm + 8 * PW + 32768 + (l % 16) * 32, I32, 8)
            ld = lambda k: (hx.gload(base + i64(k * 1024) + i64(l * 32), I32, 8),
                            hx.gload(base + i64(65536 + k * 64) + i64((l % 16) * 4), I32) if SC else I32(xs[k % 8]))
            q = [ld(k) for k in range(min(AHEAD, NWM))]
            for k in range_constexpr(NWM):
                if fx.const_expr(k + AHEAD < NWM):
                    q.append(ld(k + AHEAD))
                wv, wsc = q[k]
                acc = wmma_fp4(wv, wsc, xr[k % 8], I32(xs[k % 8]), acc)
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32 + i64(8), hx.clock())
            hx.gstore(st + i64(b * 8 + w) * 32 + i64(16) + i64((l % 4) * 4), fx.Float32(acc[l % 8]).bitcast(I32))

    @flyc.jit
    def launch(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        body(src, st, na, stride).launch(grid=(256,), block=(256,), smem=9 * PW + 1024, cluster=(16, 1, 1), stream=stream)
    return launch


def build4(GRP, NWM=64, SC=1, NT=0, DEPTH=1):
    """As build3 with the loads in groups of GRP WMMAs and a block boundary after each group's loads (the scheduler
    cannot sink them): group g + DEPTH is issued before group g computes."""
    NG = NWM // GRP

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = _ufl(t // 32)
        l = t % 32
        base = src + i64(b * 8 + w) * i64(stride)
        gpu.barrier()
        if b < na:
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32, hx.clock())
            one = _ufl(na * 0 + 1)
            acc = fx.Vector.filled(8, 0.0, fx.Float32)
            xr = [hx.lload(sm + 8 * PW + k * 2048 + l * 64, I32, 16) for k in range(8)]
            xs = hx.lload(sm + 8 * PW + 32768 + (l % 16) * 32, I32, 8)

            def ldg(g):
                return [(hx.gload(base + i64(k * 1024) + i64(l * 32), I32, 8, nt=bool(NT)),
                         hx.gload(base + i64(65536 + k * 64) + i64((l % 16) * 4), I32) if SC else I32(xs[k % 8])) for k in range(g * GRP, (g + 1) * GRP)]
            q = [ldg(g) for g in range(min(DEPTH, NG))]
            for g in range_constexpr(NG):
                if fx.const_expr(g + DEPTH < NG):
                    q.append(ldg(g + DEPTH))
                if hx.opaque_s(one) != 0:
                    for k in range_constexpr(GRP):
                        acc = wmma_fp4(q[g][k][0], q[g][k][1], xr[(g * GRP + k) % 8], I32(xs[(g * GRP + k) % 8]), acc)
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32 + i64(8), hx.clock())
            hx.gstore(st + i64(b * 8 + w) * 32 + i64(16) + i64((l % 4) * 4), fx.Float32(acc[l % 8]).bitcast(I32))

    @flyc.jit
    def launch(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        body(src, st, na, stride).launch(grid=(256,), block=(256,), smem=9 * PW + 1024, cluster=(16, 1, 1), stream=stream)
    return launch


def sched_barrier():
    hx.intrin(None, "llvm.amdgcn.sched.barrier", [I32(0)])


def build5(DEPTH, NT=1, SB=1, PF=0, NWM=64, STEP=1):
    """The DSv4.1 monokernel's register ring: WMMA k's 1 KB (two 16 B a lane loads) + scales go global -> registers,
    DEPTH units in flight, a sched barrier after every STEP WMMAs + their refills so the loads stay where written.
    NT: non-temporal loads. PF: global_prefetch of the unit's 64 KB (128 B lines, strided over the lanes) first."""
    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = _ufl(t // 32)
        l = t % 32
        base = src + i64(b * 8 + w) * i64(stride)
        gpu.barrier()
        if b < na:
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32, hx.clock())
            if fx.const_expr(PF):
                for q in range_constexpr(16):
                    hx.prefetch(base + i64(q * 4096) + i64(l * 128))
            acc = fx.Vector.filled(8, 0.0, fx.Float32)
            xr = [hx.lload(sm + 8 * PW + k * 2048 + l * 64, I32, 16) for k in range(8)]
            xs = hx.lload(sm + 8 * PW + 32768 + (l % 16) * 32, I32, 8)
            ld = lambda k: (hx.gload(base + i64(k * 1024) + i64(l * 16), I32, 4, nt=bool(NT)), hx.gload(base + i64(k * 1024 + 512) + i64(l * 16), I32, 4, nt=bool(NT)),
                            hx.gload(base + i64(65536 + k * 64) + i64((l % 16) * 4), I32, nt=bool(NT)))
            q = [ld(k) for k in range(min(DEPTH, NWM))]
            if fx.const_expr(SB):
                sched_barrier()
            for k in range_constexpr(NWM):
                a = fx.Vector.from_elements([q[k][h][e] for h in range(2) for e in range(4)], I32)
                acc = wmma_fp4(a, q[k][2], xr[k % 8], I32(xs[k % 8]), acc)
                if fx.const_expr(k + DEPTH < NWM):
                    q.append(ld(k + DEPTH))
                if fx.const_expr(SB and k % STEP == STEP - 1):
                    sched_barrier()
            if l == 0:
                hx.gstore(st + i64(b * 8 + w) * 32 + i64(8), hx.clock())
            hx.gstore(st + i64(b * 8 + w) * 32 + i64(16) + i64((l % 4) * 4), fx.Float32(acc[l % 8]).bitcast(I32))

    @flyc.jit
    def launch(src: Int64, st: Int64, na: fx.Int32, stride: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        body(src, st, na, stride).launch(grid=(256,), block=(256,), smem=9 * PW + 1024, cluster=(16, 1, 1), stream=stream)
    return launch
