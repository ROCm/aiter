"""Cost of issuing stores from a wave: every lane of NWV waves of every CTA issues N stores of KIND, then waits for them.
st[0] / [1] / [2] of a wave: clock before, after issuing, after s_wait_storecnt 0."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.typing import Int64
import hxdev as hx
I32, i64 = fx.Int32, lambda v: fx.Int64(v)
KIND = os.environ.get("KIND", "dev8"); N = int(os.environ.get("N", 8)); NC = int(os.environ.get("NC", 16)); NWV = int(os.environ.get("NWV", 1))


@ASTRewriter.transform
def flow(buf, st, t, b):
    w = t // 32
    if (b < NC) & (w < NWV):
        a = buf + i64((b * 256 + t) * 64)
        if t % 32 == 0:
            hx.gstore(st + i64(b * 8 + w) * 32, hx.clock())
        for i in range(N):
            if fx.const_expr(KIND == "dev8"):
                hx.gstore(a + i64(8 * i), (i64(t) << 32) | i64(i), scope="agent", align=8)
            if fx.const_expr(KIND == "plain8"):
                hx.gstore(a + i64(8 * i), (i64(t) << 32) | i64(i))
            if fx.const_expr(KIND == "plain16"):
                hx.gstore(a + i64(8 * i), fx.Vector.from_elements([I32(t), I32(i), I32(t), I32(i)], I32)) if i % 2 == 0 else None
        if t % 32 == 0:
            hx.gstore(st + i64(b * 8 + w) * 32 + i64(8), hx.clock())
        hx.wait_storecnt0()
        if t % 32 == 0:
            hx.gstore(st + i64(b * 8 + w) * 32 + i64(16), hx.clock())


def build():
    keyed = ("stcost", 1, KIND, N, NC, NWV)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(buf: Int64, st: Int64):
        flow(buf, st, hx.opaque_v(fx.thread_idx.x), hx.opaque_s(fx.block_idx.x))
        _ = keyed

    @flyc.jit
    def launch(buf: Int64, st: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(buf, st).launch(grid=(256,), block=(256,), cluster=(16, 1, 1), stream=stream)

    return launch


if __name__ == "__main__":
    dev = "cuda"; ln = build()
    buf = torch.zeros(256 * 256 * 8, dtype=torch.int64, device=dev); st = torch.zeros(2048, 4, dtype=torch.int64, device=dev)
    iss, dn = [], []
    for rep in range(200):
        st.zero_(); ln(buf.data_ptr(), st.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize()
        if rep < 20: continue
        s = st.double(); s = s[s[:, 0] > 0]
        iss.append(((s[:, 1] - s[:, 0]) / 100).cpu()); dn.append(((s[:, 2] - s[:, 0]) / 100).cpu())
    a = torch.cat(iss).sort().values; d = torch.cat(dn).sort().values
    print("%-7s x %d a lane, %3d CTAs x %d wave(s): issued after %.2f us (p90 %.2f) | all acknowledged after %.2f us (p90 %.2f, p99 %.2f)" % (
        KIND, N, NC, NWV, float(a[len(a) // 2]), float(a[9 * len(a) // 10]), float(d[len(d) // 2]), float(d[9 * len(d) // 10]), float(d[99 * len(d) // 100])))
