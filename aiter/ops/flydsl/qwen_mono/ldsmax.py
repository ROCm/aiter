import sys, torch, flydsl.compiler as flyc, flydsl.expr as fx
from flydsl.expr.typing import Int64
from atom.models.deepseek_v41.mono.gfx1250.kernels import device as hx
def mk(kb):
    @flyc.kernel(known_block_size=[256, 1, 1])
    def body(o: Int64):
        t = fx.thread_idx.x
        hx.lstore(hx.lds_base() + (kb * 1024 - 1024) + t * 4, fx.Int32(t))
        hx.gstore(o + fx.Int64(t) * 4, hx.lload(hx.lds_base() + (kb * 1024 - 1024) + t * 4, fx.Int32))
    @flyc.jit
    def launch(o: Int64, stream: fx.Stream = fx.Stream(None)):
        _ = kb
        body(o).launch(grid=(256,), block=(256,), smem=kb * 1024, cluster=(16, 1, 1), stream=stream)
    return launch
o = torch.zeros(256, dtype=torch.int32, device="cuda")
kb = int(sys.argv[1])
try:
    mk(kb)(o.data_ptr(), stream=torch.cuda.current_stream()); torch.cuda.synchronize(); print(kb, "KB ok", int(o[255]))
except Exception as e:
    print(kb, "KB FAIL", str(e)[:80])
