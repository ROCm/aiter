"""Small-GEMM stage chain for T <= 16 token rows (S6 and S1-S4 of the drafter), compact: every loop is a runtime loop.

A stage is out[N][16 tok] = W[N][K] . x[K][16 tok], bf16. CTA b owns tiles b, b + 256, ... (16 output rows a tile); its
8 waves split K in 8 parts (CK = K / 256 k-steps each). Per wave: one async burst of its weights (CK KB, cold, issued
while the previous stage finishes) and one of its x part (CK KB, L2-resident) into its own LDS slice, CK WMMAs from LDS,
partials summed through LDS after a CTA barrier, the result stored bf16 at device scope in the next stage's x layout,
then a fence-free device handoff.
Layouts: W [N/16][K/32][32 lanes][16] (lane = half * 16 + row), x [K/32][32 lanes][16] (lane = half * 16 + token)."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
import hxdev as hx

F32, I32, BF16 = fx.Float32, fx.Int32, fx.BFloat16
PW = 33 * 1024            # LDS a wave: 16 KB weights, 16 KB x, 1 KB partials
NSTAMP = 16


def i64(x):
    return fx.Int64(x)


def pack_w(w):
    n, k = w.shape
    return w.view(n // 16, 16, k // 32, 2, 16).permute(0, 2, 3, 1, 4).contiguous()


def pack_x(x):
    t, k = x.shape
    return x.view(t, k // 32, 2, 16).permute(1, 2, 0, 3).contiguous()


def unpack_x(xp, k):
    return xp.view(k // 32, 2, 16, 16).permute(2, 0, 1, 3).reshape(16, k)


@ASTRewriter.transform
def issue_w(c, W, NS, tile):
    """This wave's weights of `tile` (its K part) -> its LDS slice, non-temporal."""
    CK = NS // 8
    src = W + i64(tile) * (NS * 1024) + i64(c["w"] * (CK * 1024) + c["l"] * 16)
    for q in range(CK * 2):
        hx.async_to_lds_b128(src + i64(q * 512), c["wbuf"] + q * 512 + c["l"] * 16, 2)


@ASTRewriter.transform
def gemm_stage(c, W, NT, NS, xin, xout, nxt):
    """One stage. Its first pass's weights are already on their way (issue_w by the caller / the previous stage);
    nxt = (W, NT, NS) of the next stage, whose weights are issued once this stage's WMMAs are done."""
    t, b, w, l = c["t"], c["b"], c["w"], c["l"]
    CK = NS // 8
    assert NS % 8 == 0 and CK <= 16
    xsrc = xin + i64(w * (CK * 1024) + l * 16)
    for q in range(CK * 2):
        hx.async_to_lds_b128(xsrc + i64(q * 512), c["xbuf"] + q * 512 + l * 16, 0)
    if fx.const_expr(nxt is not None and NT < 256):
        if (b >= NT) & (b < nxt[1]):       # a CTA with no tile here: the next stage's weights right away
            issue_w(c, nxt[0], nxt[2], b)
    for p in range((NT + 255) // 256):
        tile = b + p * 256
        if tile < NT:
            hx.wait_asynccnt0()
            acc = fx.Vector.filled(8, 0.0, F32)
            for k in range(CK):
                a = hx.lload(c["wbuf"] + k * 1024 + l * 32, BF16, 16)
                bv = hx.lload(c["xbuf"] + k * 1024 + l * 32, BF16, 16)
                acc = hx.wmma_bf16(a, bv, acc)
            if tile + 256 < NT:
                issue_w(c, W, NS, tile + 256)
            else:
                if fx.const_expr(nxt is not None):
                    if b < nxt[1]:
                        issue_w(c, nxt[0], nxt[2], b)
            hx.lstore(c["red"] + (w * 32 + l) * 32, acc)
            gpu.barrier()
            if w == 0:
                tot = hx.lload(c["red"] + l * 32, F32, 8)
                for q in range_constexpr(1, 8):
                    tot = tot + hx.lload(c["red"] + (q * 32 + l) * 32, F32, 8)
                # rows (l // 16) * 8 + e of the tile, token l % 16 -> x layout of the next stage: 16 B a lane
                hx.stx_dev(xout + i64(tile // 2) * 1024 + i64(((tile % 2) * 16 + l % 16) * 32 + (l // 16) * 16), tot.to(BF16))
            gpu.barrier()


def build(stages):
    """stages: [(N, K)]; stage i reads x buffer i and writes x buffer i + 1. tab: i64 [W0.., x0..] pointers."""
    keyed = (tuple(stages), 3)
    ns = len(stages)
    geo = [(n // 16, k // 32) for n, k in stages]

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(tab: Int64, cnt: Int64, st: Int64):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        w = t // 32
        c = dict(t=t, b=b, w=w, l=t % 32, wbuf=sm + w * PW, xbuf=sm + w * PW + 16384, red=sm + 8 * PW)
        Wp = [hx.sload64(tab + i64(8 * i)) for i in range(ns)]
        Xp = [hx.sload64(tab + i64(8 * (ns + i))) for i in range(ns + 1)]
        if t == 0:
            hx.gstore(st + i64(b) * (NSTAMP * 8), hx.clock())
        if b < geo[0][0]:
            issue_w(c, Wp[0], geo[0][1], b)
        hx.handoff(cnt, t, True)
        if t == 0:
            hx.gstore(st + i64(b) * (NSTAMP * 8) + i64(8), hx.clock())
        for i in range_constexpr(ns):
            nxt = (Wp[i + 1], geo[i + 1][0], geo[i + 1][1]) if i + 1 < ns else None
            gemm_stage(c, Wp[i], geo[i][0], geo[i][1], Xp[i], Xp[i + 1], nxt)
            hx.handoff(cnt + i64(64 * (i + 1)), t, True)
            if t == 0:
                hx.gstore(st + i64(b) * (NSTAMP * 8) + i64(8 * (i + 2)), hx.clock())
        _ = keyed

    @flyc.jit
    def launch(tab: Int64, cnt: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(tab, cnt, st).launch(grid=(256,), block=(256,), smem=8 * PW + 8192, cluster=(16, 1, 1), stream=stream)

    return launch
