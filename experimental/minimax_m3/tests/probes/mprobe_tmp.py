import sys
sys.path.insert(0, "/home/vigokhal/m3_monokernel/tests/probes")
import torch
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import vllm.models.minimax_m3.amd.mono.gluon.kernel as K
from vllm.models.minimax_m3.amd.mono.gluon.prims import T1, THREADS

@gluon.jit
def mk(keys_ptr, out_ptr, n):
    bid = gl.program_id(0)
    t = K.ar(THREADS, T1)
    lctr = gl.allocate_shared_memory(gl.int32, [THREADS], K.FLAT1)
    key = gl.load(keys_ptr + bid * 1024 + t)
    live = t < n
    lctr.store(t * 0)
    gl.barrier()
    least = t * 0 - 2147483647 - 1
    top = gl.where(live, key, least)
    gmax = gl.max(gl.convert_layout(gl.reshape(top, [32, 16]), K.G16), axis=0)
    floor = gl.min(gmax, axis=0)
    ce = live & (key >= floor)
    slot = lctr.atomic_scatter_add(t * 0 + 1, t * 0, 0, mask=ce)
    gl.barrier()
    m = gl.max(lctr.load(T1), axis=0)
    gl.store(out_ptr + bid * 4 + t * 0, m + t * 0, mask=t == 0)
    gl.store(out_ptr + bid * 4 + 1 + t * 0, floor + t * 0, mask=t == 0)
    gl.store(out_ptr + bid * 4 + 2 + t * 0, gl.max(slot, axis=0) + t * 0, mask=t == 0)

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)
keys = (torch.randn(32, 1024, generator=g, device=dev) * 1e4).to(torch.int32)
for n in (321, 512):
    out = torch.zeros(32, 4, dtype=torch.int32, device=dev)
    mk[(32,)](keys, out, n, num_warps=8)
    torch.cuda.synchronize()
    kk = keys[:, :512].clone(); kk[:, n:] = -2**31
    gm = kk.view(32, 32, 16).max(1).values.min(1).values
    ref_m = ((kk >= gm[:, None]) & (torch.arange(512, device=dev) < n)).sum(1)
    print(n, "m", out[:4, 0].tolist(), "ref", ref_m[:4].tolist(), "floor", out[:4, 1].tolist(), gm[:4].tolist(), "maxslot", out[:4, 2].tolist())
