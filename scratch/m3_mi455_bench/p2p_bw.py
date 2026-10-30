"""Single-process HBM and peer-to-peer bandwidth on the visible GPUs (kernel loads/stores)."""

import torch
import triton
import triton.language as tl


@triton.jit
def _copy(dst, src0, src1, src2, n, NSRC: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n, stride):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        acc = tl.load(src0 + offs, mask=m, other=0.0).to(tl.float32)
        if NSRC >= 2:
            acc += tl.load(src1 + offs, mask=m, other=0.0).to(tl.float32)
        if NSRC >= 3:
            acc += tl.load(src2 + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(dst + offs, acc.to(dst.dtype.element_ty), mask=m)


def timed(fn, iters=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3  # us


def run(dst, srcs, grid):
    n = dst.numel()
    s = srcs + [srcs[0]] * (3 - len(srcs))
    _copy[(grid,)](dst, s[0], s[1], s[2], n, NSRC=len(srcs), BLOCK=4096, num_warps=8)


ng = torch.cuda.device_count()
print("GPUs:", ng, torch.cuda.get_device_name(0), "CUs:", torch.cuda.get_device_properties(0).multi_processor_count)
for i in range(ng):
    for j in range(ng):
        if i != j:
            assert torch.cuda.can_device_access_peer(i, j), (i, j)
# enable peer access both ways (torch enables it on the first cross-device copy)
for i in range(ng):
    for j in range(ng):
        if i != j:
            torch.zeros(1, device=f"cuda:{j}").copy_(torch.zeros(1, device=f"cuda:{i}"))

cu = torch.cuda.get_device_properties(0).multi_processor_count
for nbytes in (1 << 20, 16 << 20, 256 << 20):
    n = nbytes // 2
    a = [torch.randn(n, device=f"cuda:{i}", dtype=torch.bfloat16) for i in range(ng)]
    b = [torch.empty(n, device=f"cuda:{i}", dtype=torch.bfloat16) for i in range(ng)]
    torch.cuda.set_device(0)
    res = {}
    for grid in (cu, 4 * cu):
        # HBM: read local, write local (bytes moved = 2x)
        t = timed(lambda: run(b[0], [a[0]], grid))
        res.setdefault("hbm read+write (GB/s, 2x bytes)", []).append(2 * nbytes / t / 1e3)
        # pull: GPU0 reads GPU1, writes local
        t = timed(lambda: run(b[0], [a[1]], grid))
        res.setdefault("pull 1 peer -> local (GB/s of peer reads)", []).append(nbytes / t / 1e3)
        # push: GPU0 reads local, writes into GPU1
        t = timed(lambda: run(b[1], [a[0]], grid))
        res.setdefault("push local -> 1 peer (GB/s of peer writes)", []).append(nbytes / t / 1e3)
        # pull from 3 peers at once (all-reduce read pattern), bytes = 3x
        if ng >= 4:
            t = timed(lambda: run(b[0], [a[1], a[2], a[3]], grid))
            res.setdefault("pull 3 peers -> local (GB/s total peer reads)", []).append(3 * nbytes / t / 1e3)
    print(f"--- {nbytes >> 20} MiB per buffer (best of grid={cu},{4 * cu})")
    for k, v in res.items():
        print(f"  {k:48s} {max(v):8.1f}")
    del a, b
    torch.cuda.empty_cache()
