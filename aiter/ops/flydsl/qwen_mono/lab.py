"""Latency lab on the fixed grid (256 CTAs x 8 waves, clusters of 16): what each building block of a stage costs.
Stamps (100 MHz clock) by thread 0 of every CTA into st[b][k]."""
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr.typing import Int64
from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx

I32 = fx.Int32
NST = 16


def i64(x):
    return fx.Int64(x)


def build(KB=8, cached=False):
    keyed = (KB, cached, 3)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(wb: Int64, xg: Int64, cnt: Int64, st: Int64):
        t = hx.opaque_v(fx.thread_idx.x)
        b = hx.opaque_s(fx.block_idx.x)
        sm = hx.lds_base()
        l = t % 32
        w = t // 32
        g = b * 8 + w

        def stamp(k):
            if t == 0:
                hx.gstore(st + i64(b) * (NST * 8) + i64(k * 8), hx.clock())

        stamp(0)
        gpu.barrier()
        stamp(1)                                   # 0->1: a CTA barrier
        hx.handoff(cnt, t, True)
        stamp(2)                                   # 1->2: device handoff (fence-free), includes start skew
        hx.handoff(cnt + i64(64), t, True)
        stamp(3)                                   # 2->3: device handoff, grid aligned
        hx.handoff(cnt + i64(128), t, False)
        stamp(4)                                   # 3->4: device handoff with fences
        hx.fence_release("cluster"); hx.cluster_barrier(); hx.fence_acquire("cluster")
        stamp(5)                                   # 4->5: cluster barrier with fences
        # a cold burst of KB per wave from HBM into LDS (async), then read it
        for q in range_constexpr(KB * 2):
            hx.async_to_lds_b128(wb + i64(g) * (KB * 1024) + i64(q * 512 + l * 16), sm + w * (KB * 1024) + q * 512 + l * 16, 0 if cached else 2)
        hx.wait_asynccnt0()
        stamp(6)                                   # 5->6: cold burst latency
        acc = hx.lload(sm + t * 16, I32)
        # device-scope store by every thread, handoff, then L2-resident loads of another CTA's data (regs)
        hx.stx_dev(xg + i64(b * 256 + t) * 16, fx.Vector.filled(4, 1, I32) + acc)
        stamp(7)
        hx.handoff(cnt + i64(192), t, True)
        stamp(8)                                   # 7->8: store + handoff
        v = [hx.gload(xg + i64(((b + 37) % 256) * 256 + t) * 16 + i64(0), I32, 4) for q in range(1)]
        acc = acc + v[0][0]
        hx.gstore(xg + i64(65536 * 16) + i64(b * 256 + t) * 4, acc)
        hx.wait_storecnt0()
        stamp(9)                                   # 8->9: L2-resident 16 B/lane load (another CTA's fresh data) + a store
        # x [16][2048] bf16 = 64 KB from L2 into LDS (async, cached), as a stage's B operand staging
        for it in range_constexpr(16):
            hx.async_to_lds_b128(xg + i64(t + it * 256) * 16, sm + (t + it * 256) * 16, 0)
        hx.wait_asynccnt0()
        gpu.barrier()
        stamp(10)                                  # 9->10: 64 KB x staging from L2 + barrier
        _ = keyed

    @flyc.jit
    def launch(wb: Int64, xg: Int64, cnt: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(wb, xg, cnt, st).launch(grid=(256,), block=(256,), smem=max(8 * KB * 1024, 65536) + 4096, cluster=(16, 1, 1), stream=stream)

    return launch
