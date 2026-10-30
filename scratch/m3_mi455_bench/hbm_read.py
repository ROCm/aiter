"""HBM read bandwidth of one GPU: a streaming sum over a large buffer (no writes)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _rd(src, out, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    acc = tl.zeros([BLOCK], dtype=tl.int32)
    for start in range(pid * BLOCK, n, tl.num_programs(0) * BLOCK):
        offs = start + tl.arange(0, BLOCK)
        acc += tl.load(src + offs, mask=offs < n, other=0, cache_modifier=".cg")
    tl.store(out + pid * BLOCK + tl.arange(0, BLOCK), acc)


cu = torch.cuda.get_device_properties(0).multi_processor_count
for mb in (64, 256, 1024, 4096):
    n = (mb << 20) // 4
    src = torch.randint(0, 7, (n,), dtype=torch.int32, device="cuda")
    for grid_mul, block in ((4, 2048), (8, 2048), (16, 1024)):
        grid = cu * grid_mul
        out = torch.empty(grid * block, dtype=torch.int32, device="cuda")
        f = lambda: _rd[(grid,)](src, out, n, BLOCK=block, num_warps=8)
        for _ in range(3):
            f()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(10):
            f()
        e.record()
        torch.cuda.synchronize()
        us = s.elapsed_time(e) * 100
        print(f"{mb:5d} MiB grid {grid:5d} x {block}: {mb * 1.048576 / us:6.2f} TB/s read ({us:8.1f} us)")
