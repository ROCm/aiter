"""Handoff micro-bench. CTA 0 produces one 32-bit value ~3 us after entry; one wave of every other CTA waits for it.
 MODE=cnt : data store (device scope), s_wait_storecnt 0, atomic counter += 1; readers poll the counter, then load the data.
 MODE=mb  : ONE 64-bit store of (tag << 32 | data), no wait, no atomic; readers poll that word until its tag is the launch's.
Stamps: st[0] = producer's clock just before its store; st[b] = reader b's clock once it holds the data."""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.compiler.ast_rewriter import ASTRewriter
from flydsl.expr.typing import Int64
import hxdev as hx
I32, i64 = fx.Int32, lambda v: fx.Int64(v)
MODE = os.environ.get("MODE", "mb"); SL = tuple(int(v) for v in os.environ.get("SL", "1,1").split(",")); NW = int(os.environ.get("NW", 1))


@ASTRewriter.transform
def mb_wait(addr, tag):
    t0 = hx.clock()
    if fx.const_expr(os.environ.get("CARRY", "1") == "1"):        # the polled word itself is the result: no second round trip
        v = hx.gload(addr, fx.Int64, scope="agent")
        while (I32(v >> 32) != tag) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
            for _ in range(SL[0]):
                hx.s_sleep(SL[1])
            v = hx.gload(addr, fx.Int64, scope="agent")
        return I32(v)
    while (I32(hx.gload(addr, fx.Int64, scope="agent") >> 32) != tag) & (hx.clock() - t0 < fx.Int64(hx.POLL_TICKS)):
        for _ in range(SL[0]):
            hx.s_sleep(SL[1])
    return I32(hx.gload(addr, fx.Int64, scope="agent"))


@ASTRewriter.transform
def spin(ticks):
    t0 = hx.clock()
    while hx.clock() - t0 < fx.Int64(ticks):
        hx.s_sleep(1)


@ASTRewriter.transform
def flow(box, cnt, st, tag, t, b):
    if b == 0:
        if t == 0:
            spin(300)
            hx.gstore(st, hx.clock())
            if fx.const_expr(MODE == "mb"):
                hx.gstore(box, (i64(tag) << 32) | i64(1234), scope="agent")
            elif fx.const_expr(MODE == "p16"):            # ONE plain 16-byte store (global_store_b128): [1234, tag, 77, tag]
                hx.gstore(box, fx.Vector.from_elements([I32(1234), I32(tag), I32(77), I32(tag)], I32))
            else:
                hx.gstore(box, i64(1234), scope="agent")
                hx.wait_storecnt0()
                hx.atomic_add(cnt, 1)
    else:
        if (t % 32 == 0) & (t // 32 < NW):
            if fx.const_expr(MODE == "mb"):
                v = mb_wait(box, tag)
            elif fx.const_expr(MODE == "p16"):
                v = mb_wait(box, tag)
                v2 = hx.gload(box + i64(8), fx.Int64, scope="agent")      # the other half of the same store
                hx.gstore(st + i64(2048 + b * 8 + t // 32) * 8, v2)
            else:
                hx.wait_count(cnt, 1, 40, ws=SL)
                v = I32(hx.gload(box, fx.Int64, scope="agent"))
            hx.gstore(st + i64(b * 8 + t // 32) * 8, hx.clock() + i64(v - 1234))


def build():
    keyed = ("mbox", 3, os.environ.get("CARRY", "1"), MODE, SL, NW)

    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(box: Int64, cnt: Int64, st: Int64, tag: fx.Int32):
        flow(box, cnt, st, tag, hx.opaque_v(fx.thread_idx.x), hx.opaque_s(fx.block_idx.x))
        _ = keyed

    @flyc.jit
    def launch(box: Int64, cnt: Int64, st: Int64, tag: fx.Int32, stream: fx.Stream = fx.Stream(None)):
        _ = keyed
        body(box, cnt, st, tag).launch(grid=(256,), block=(256,), cluster=(16, 1, 1), stream=stream)

    return launch


if __name__ == "__main__":
    dev = "cuda"; ln = build()
    box = torch.zeros(8, dtype=torch.int64, device=dev); cnt = torch.zeros(64, dtype=torch.int32, device=dev); st = torch.zeros(4096, dtype=torch.int64, device=dev)
    lat = []; BAD = [0, 0]
    for rep in range(300):
        cnt.zero_(); st.zero_()
        ln(box.data_ptr(), cnt.data_ptr(), st.data_ptr(), rep + 1, stream=torch.cuda.current_stream()); torch.cuda.synchronize()
        if rep < 20: continue
        if MODE == "p16":
            h2 = st[2048 + 8:4096]; h2 = h2[st[8:2048] > 0]
            BAD[0] += int((h2 != (((rep + 1) << 32) | 77)).sum()); BAD[1] += h2.numel()
        s = st.double(); r = s[8:2048]; r = r[r > 0]
        lat.append(((r - s[0]) / 100).cpu())
    L = torch.cat(lat).sort().values; per = torch.stack([x.max() for x in lat]).sort().values
    print("MODE %-3s sleep %s, %d polling wave(s) a CTA: store -> reader holds the data: median %.2f us, p90 %.2f, p99 %.2f | slowest reader of a launch: median %.2f us" % (
        MODE, SL, NW, float(L[len(L) // 2]), float(L[9 * len(L) // 10]), float(L[99 * len(L) // 100]), float(per[len(per) // 2])))
    if MODE == "p16": print("   second half of the 16-byte store wrong when the first half's tag was seen: %d of %d reads" % (BAD[0], BAD[1]))
