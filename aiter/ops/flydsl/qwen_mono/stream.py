"""Stream floor on the fixed grid: every wave reads its share of `nbytes` contiguously (CH bytes a lane a load, D loads
in flight), xor-accumulates, stores one word. mode: 'reg' (vector loads), 'lds' (async to LDS, 16 B a lane)."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr import gpu, range_constexpr
from flydsl.expr.typing import Int64
from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx

I32 = fx.Int32


def i64(x):
    return fx.Int64(x)


def build(nbytes, CH=32, D=4, nt=True, mode="reg", waves=8, ctas=256):
    keyed = (nbytes, CH, D, nt, mode, waves, ctas)
    per_wave = nbytes // (ctas * waves)
    steps = per_wave // (32 * CH)
    steps -= steps % D

    SMEM = waves * 2 * D * 512 + 4096 if mode == 'lds2' else 65536

    @flyc.kernel(known_block_size=[waves * 32, 1, 1])
    def body(wb: Int64, out: Int64, st: Int64):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        l = t % 32
        base = wb + i64(b * waves + t // 32) * per_wave + i64(l * CH)
        if t == 0:
            hx.gstore(st + i64(b) * 16, hx.clock())
        acc = I32(0)
        if fx.const_expr(mode == "reg"):
            A = [hx.gload(base + i64(d) * (32 * CH), I32, CH // 4, nt=nt) for d in range(D)]
            for sb in range(steps // D):
                for d in range_constexpr(D):
                    a = A[d]
                    sp = fx.min(sb * D + d + D, I32(steps - 1))
                    A[d] = hx.gload(base + i64(sp) * (32 * CH), I32, CH // 4, nt=nt)
                    acc = acc ^ a[0] ^ a[CH // 4 - 1]
                    hx.intrin(None, "llvm.amdgcn.sched.barrier", [fx.Int32(0)])
        elif fx.const_expr(mode == "pf"):
            # software prefetch D * 4 KB ahead (a 128-byte line a lane), then cached loads of the current 4 KB
            wbase = wb + i64(b * waves + t // 32) * per_wave
            for d in range_constexpr(D):
                hx.prefetch(wbase + i64(d * 4096 + l * 128))
            for sb in range(per_wave // 4096):
                hx.prefetch(wbase + i64(fx.min(sb + D, I32(per_wave // 4096 - 1))) * 4096 + i64(l * 128))
                for q in range_constexpr(8):
                    a = hx.gload(wbase + i64(sb) * 4096 + i64(l * 128 + q * 16), I32, 4, nt=nt)
                    acc = acc ^ a[0] ^ a[3]
        elif fx.const_expr(mode == "lds2"):
            # double-buffered async copies: burst n + 1 in flight while burst n is read from LDS (D x 512 B a wave a burst)
            BB = D * 512
            wo = sm + (t // 32) * (2 * BB) + l * 16
            nb = per_wave // BB
            for d in range_constexpr(D):
                hx.async_to_lds_b128(base + i64(d * 512), wo + d * 512, 2 if nt else 0)
            hx.wait_asynccnt0()
            for sb in range(nb):
                nx = fx.min(sb + 1, I32(nb - 1))
                for d in range_constexpr(D):
                    hx.async_to_lds_b128(base + i64(nx) * BB + i64(d * 512), wo + ((sb + 1) % 2) * BB + d * 512, 2 if nt else 0)
                for d in range_constexpr(D):
                    a = hx.lload(wo + (sb % 2) * BB + d * 512, I32, 4)
                    acc = acc ^ a[0] ^ a[3]
                hx.wait_asynccnt0()
        else:
            for sb in range(steps * (CH // 16) // D):
                for d in range_constexpr(D):
                    hx.async_to_lds_b128(base + i64(sb * D + d) * (32 * 16), sm + t * 16 + d * 4096, 2 if nt else 0)
                hx.wait_asynccnt0()
                acc = acc ^ hx.lload(sm + t * 16, I32)
        hx.gstore(out + i64(b * 256 + t) * 4, acc)
        if t == 0:
            hx.gstore(st + i64(b) * 16 + i64(8), hx.clock())
        _ = keyed

    @flyc.jit
    def launch(wb: Int64, out: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(wb, out, st).launch(grid=(ctas,), block=(waves * 32,), smem=SMEM, cluster=(16, 1, 1), stream=stream)

    return launch, (per_wave // 4096 * 4096 if mode == 'pf' else per_wave // (D * 512) * D * 512 if mode == 'lds2' else steps * 32 * CH) * ctas * waves
