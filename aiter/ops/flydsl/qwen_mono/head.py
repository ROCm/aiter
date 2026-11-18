"""Drafter LM head with argmax in the epilogue (S5). T <= 16 token rows (one WMMA B tile), bf16 row-major weights
[N][K] read in place (no repack), no logits written.

256 CTAs x 8 waves. Wave g = 8 b + w owns tile groups g, g + 2048, ... (TW vocab tiles of 16 rows a group). One flat
runtime loop over (group, k-step): D k-steps of weights in flight a wave, across group boundaries, so the stream never
drains. An x fragment read from LDS serves the group's TW tiles. Each lane keeps the best (value, index) of its token
column; lanes -> LDS atomic max64 -> one global atomic max64 a token.
out[t] (i64, host sets INT64_MIN): (ordered bits of the bf16-rounded logit) << 32 | (0x7fffffff - index)."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
from flydsl._mlir.dialects import llvm
import hxdev as hx

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
latomic_max64, gatomic_max64 = hx.latomic_max64, hx.gatomic_max64
THREADS, WTOT = 256, 2048
IMIN = -(1 << 63)


def i64(x):
    return fx.Int64(x)


def key64(v, idx):
    """(f32 v, i32 idx) -> i64 ordered by v, then by lower idx."""
    bits = v.bitcast(I32)
    o = (bits < 0).select(bits ^ I32(0x7FFFFFFF), bits)
    return (i64(o) << 32) | i64(I32(0x7FFFFFFF) - idx)


def geom(N, K, TW):
    NT, NS = N // 16, K // 32
    NG = -(-NT // TW)
    return NT, NS, NG, -(-NG // WTOT)


def pack(w, TW):
    """[N][K] bf16 -> [N/(16 TW)][K/32][TW][2][16][16]: a wave's k-step of a tile group is TW KB contiguous (tile, then
    lane = half * 16 + row)."""
    n, k = w.shape
    return w.view(n // (16 * TW), TW, 16, k // 32, 2, 16).permute(0, 3, 1, 4, 2, 5).contiguous()


def _ufl(x):
    return fx.Int32(hx.intrin(hx.i32(), "llvm.amdgcn.readfirstlane.i32", [fx.Int32(x)]))


def lds_bytes(K, TW, B):
    return 16 * (K * 2 + 16) + 256 + 8 * B * TW * 1024


@ASTRewriter.transform
def head_argmax(x, wb, out, t, b, sm, N, K, B, math=True, nt=True, TW=4, GW=2, U=1, st=None):
    """Wave g < NG / GW owns tile groups [g GW, (g + 1) GW) (TW tiles a group, k-major packed): one contiguous byte
    range. The weights stream through a FIFO of B k-steps (TW KB each) in the wave's own LDS slice, filled by async
    copies: step s + B - 1 is issued, then s_wait_asynccnt leaves the B - 1 younger steps in flight while step s is
    used. The wait is ours (the compiler tracks no async copies), so the stream stays in flight across the runtime
    loop's back edge: no unrolling. An x fragment from LDS serves the step's TW tiles."""
    NT, NS = N // 16, K // 32
    assert NT % (TW * GW) == 0 and NS % U == 0
    NW = NT // (TW * GW)            # active waves
    assert NW <= WTOT
    XS = K * 2 + 16
    RED = 16 * XS
    S = GW * NS
    SB = TW * 1024                  # bytes of a step
    l = t % 32
    row = l % 16
    half = l // 16
    g = _ufl(b * 8 + t // 32)
    base = wb + i64(g) * (S * SB)
    fifo = sm + RED + 256 + (t // 32) * (B * SB)
    cpol = 2 if nt else 0

    def issue(s):
        for q in range_constexpr(2 * TW):
            hx.async_to_lds_b128(base + i64(s) * SB + i64(q * 512 + l * 16), fifo + (s % B) * SB + q * 512 + l * 16, cpol)

    if t < 16:
        hx.lstore(sm + RED + t * 8, i64(IMIN))
    for it in range(16 * K // 8 // THREADS):   # x [16][K] bf16 -> LDS, 16 B a thread a round
        i = t + it * THREADS
        hx.async_to_lds_b128(x + i64(i) * 16, sm + (i // (K // 8)) * XS + (i % (K // 8)) * 16)
    bestv = F32(-3.0e38)
    besti = I32(0)
    if g < NW:
        for s0 in range_constexpr(B - 1):
            issue(I32(s0))
        hx.wait_asynccnt0()
        gpu.barrier()
        if fx.const_expr(st is not None):
            if t == 0:
                hx.gstore(st + i64(b) * 32 + i64(16), hx.clock())
        acc = [fx.Vector.filled(8, 0.0, F32) for _ in range(TW)]
        for su in range(S // U):
            for u in range_constexpr(U):
                s = su * U + u
                issue(fx.min(s + (B - 1), I32(S - 1)))
                rocdl.s_wait_asynccnt((B - 1) * 2 * TW)
                bv = hx.lload(sm + row * XS + ((s % NS) * 32 + half * 16) * 2, BF16, 16)
                a = [hx.lload(fifo + (s % B) * SB + i * 1024 + l * 32, BF16, 16) for i in range(TW)]
                for i in range_constexpr(TW):
                    if fx.const_expr(math):
                        acc[i] = hx.wmma_bf16(a[i], bv, acc[i])
                    else:
                        acc[i] = acc[i] + a[i].bitcast(F32)
            if (su * U + U) % NS == 0:
                for i in range_constexpr(TW):
                    tile = (g * GW + su * U // NS) * TW + i
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
        latomic_max64(sm + RED + row * 8, key64(bestv, besti))
    else:
        hx.wait_asynccnt0()
        gpu.barrier()
    gpu.barrier()
    if t < 16:
        gatomic_max64(out + i64(t) * 8, hx.lload(sm + RED + t * 8, fx.Int64))


def build(N=248320, K=2048, B=5, math=True, nt=True, TW=4, GW=2, U=1):
    keyed = (N, K, B, math, nt, TW, GW, U, 'fifo')
    LDS = lds_bytes(K, TW, B)

    @flyc.kernel(known_block_size=[THREADS, 1, 1])
    def body(x: Int64, wb: Int64, out: Int64, st: Int64):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        if t == 0:
            hx.gstore(st + i64(b) * 32, hx.clock())
        head_argmax(x, wb, out, t, b, sm, N, K, B, math, nt, TW, GW, U, st)
        if t == 0:
            hx.gstore(st + i64(b) * 32 + i64(8), hx.clock())
        _ = keyed

    @flyc.jit
    def launch(x: Int64, wb: Int64, out: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(x, wb, out, st).launch(grid=(256,), block=(THREADS,), smem=LDS, cluster=(16, 1, 1), stream=stream)

    return launch
