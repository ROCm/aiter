"""Which Gluon cache modifier gives which gfx950 cache bits (buffer/global load/store)."""
import re
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from triton.experimental.gluon.language.amd import cdna4

L = gl.constexpr(gl.BlockedLayout([1], [64], [1], [0]))


@gluon.jit
def k(src, dst, LC: gl.constexpr, SC: gl.constexpr):
    i = gl.arange(0, 64, layout=L)
    v = cdna4.buffer_load(src, i, cache=LC)
    cdna4.buffer_store(v, dst, i, cache=SC)
    w = gl.load(src + 64 + i, cache_modifier=LC)
    gl.store(dst + 64 + i, w, cache_modifier=SC)


x = torch.zeros(128, dtype=torch.int64, device="cuda")
for lc, sc in [("", ""), (".ca", ".wb"), (".cg", ".cg"), (".cv", ".wt"), (".cs", ".cs")]:
    c = k.warmup(x, x, lc, sc, grid=(1,), num_warps=1)
    ops = [l.strip() for l in c.asm["amdgcn"].splitlines() if re.match(r"\s*(buffer|global)_(load|store)", l)]
    print(f"load {lc or 'none':4s} store {sc or 'none':4s}:", " | ".join(ops))
