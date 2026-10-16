"""Which gfx950 cache-policy bits each Gluon buffer cache modifier emits."""
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

L = gl.constexpr(gl.BlockedLayout([1], [64], [1], [0]))


@gluon.jit
def k(x, y, LC: gl.constexpr, SC: gl.constexpr):
    i = gl.arange(0, 64, layout=L)
    v = cdna4.buffer_load(x, i, cache=LC)
    cdna4.buffer_store(v + 1, y, i, cache=SC)


x = torch.zeros(64, dtype=torch.int32, device="cuda")
for lc, sc in [("", ""), (".ca", ".wb"), (".cg", ".cg"), (".cv", ".cs"), (".cv", ".wt")]:
    try:
        cc = k.warmup(x, x, lc, sc, grid=(1,), num_warps=1)
        a = cc.asm["amdgcn"]
        ld = [l.strip() for l in a.split("\n") if "buffer_load" in l]
        st = [l.strip() for l in a.split("\n") if "buffer_store" in l]
        print(f"load {lc or 'default':8s} -> {ld[0] if ld else '?'}\nstore {sc or 'default':7s} -> {st[0] if st else '?'}")
    except Exception as e:
        print(lc, sc, "FAIL", str(e).splitlines()[-1][:120])
