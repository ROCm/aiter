"""All-to-all push bandwidth on 4 GPUs: the reduce-scatter (and all-gather) traffic pattern.

Every GPU pushes a distinct chunk into each of its 3 peers at the same time, so each
GPU sends 3 chunks and receives 3 chunks. Reports per-GPU ingress (= egress) GB/s.
Also: one GPU pushing to 3 idle peers (egress cap), and 3 GPUs pushing into one (ingress cap).
"""

import time

import torch
import triton
import triton.language as tl


@triton.jit
def _push3(src, d0, d1, d2, n, BLOCK: tl.constexpr):
    # program ids are split in 3 equal ranges, one per destination
    pid = tl.program_id(0)
    per = tl.num_programs(0) // 3
    part = pid // per
    lpid = pid - part * per
    stride = per * BLOCK
    base = src + part * n
    for start in range(lpid * BLOCK, n, stride):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        v = tl.load(base + offs, mask=m)
        if part == 0:
            tl.store(d0 + offs, v, mask=m)
        elif part == 1:
            tl.store(d1 + offs, v, mask=m)
        else:
            tl.store(d2 + offs, v, mask=m)


@triton.jit
def _push1(src, dst, n, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    stride = tl.num_programs(0) * BLOCK
    for start in range(pid * BLOCK, n, stride):
        offs = start + tl.arange(0, BLOCK)
        m = offs < n
        tl.store(dst + offs, tl.load(src + offs, mask=m), mask=m)


ng = torch.cuda.device_count()
assert ng >= 4
for i in range(ng):
    for j in range(ng):
        if i != j:
            torch.zeros(1, device=f"cuda:{j}").copy_(torch.zeros(1, device=f"cuda:{i}"))
cu = torch.cuda.get_device_properties(0).multi_processor_count


def dev_time(launch_one, devs, iters=10):
    """Max over GPUs of the median kernel time, each timed by events on its own stream."""
    for _ in range(2):
        for d in devs:
            launch_one(d)
    for d in devs:
        torch.cuda.synchronize(d)
    per = {d: [] for d in devs}
    for _ in range(iters):
        evs = {}
        for d in devs:
            with torch.cuda.device(d):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                launch_one(d)
                e.record()
                evs[d] = (s, e)
        for d in devs:
            torch.cuda.synchronize(d)
            per[d].append(evs[d][0].elapsed_time(evs[d][1]) * 1e3)
    return max(sorted(v)[len(v) // 2] for v in per.values())


def wall(launch, devs, iters=10):
    for _ in range(2):
        launch()
    for d in devs:
        torch.cuda.synchronize(d)
    ts = []
    for _ in range(iters):
        t0 = time.perf_counter()
        launch()
        for d in devs:
            torch.cuda.synchronize(d)
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2] * 1e6  # us


for total_mb in (4, 12, 32, 100, 200):
    chunk = (total_mb << 20) // 4 // 2  # bf16 elements per (sender, receiver) chunk
    src = [torch.randn(3 * chunk, device=f"cuda:{i}", dtype=torch.bfloat16) for i in range(4)]
    # rx[j][i]: buffer on GPU j that receives sender i's chunk
    rx = [[torch.empty(chunk, device=f"cuda:{j}", dtype=torch.bfloat16) for i in range(4)] for j in range(4)]
    peers = lambda i: [(i + k) % 4 for k in (1, 2, 3)]
    chunk_bytes = chunk * 2
    print(f"--- {total_mb} MB per rank (chunk {chunk_bytes / 1e6:.1f} MB per sender->receiver)")
    for grid_mul in (1, 2):
        grid = 3 * cu * grid_mul // 3 * 3

        def a2a():
            for i in range(4):
                with torch.cuda.device(i):
                    p = peers(i)
                    _push3[(grid,)](src[i], rx[p[0]][i], rx[p[1]][i], rx[p[2]][i], chunk,
                                    BLOCK=4096, num_warps=8)

        def one_to_three():
            with torch.cuda.device(0):
                p = peers(0)
                _push3[(grid,)](src[0], rx[p[0]][0], rx[p[1]][0], rx[p[2]][0], chunk,
                                BLOCK=4096, num_warps=8)

        def three_to_one():
            for i in (1, 2, 3):
                with torch.cuda.device(i):
                    _push1[(grid // 3,)](src[i], rx[0][i], chunk, BLOCK=4096, num_warps=8)

        def a2a_one(i):
            with torch.cuda.device(i):
                p = peers(i)
                _push3[(grid,)](src[i], rx[p[0]][i], rx[p[1]][i], rx[p[2]][i], chunk, BLOCK=4096, num_warps=8)

        t = dev_time(a2a_one, list(range(4)))
        t1 = wall(one_to_three, [0])
        t3 = wall(three_to_one, [1, 2, 3])
        print(f"  grid={grid:5d}: all-to-all {3 * chunk_bytes / t / 1e3:7.1f} GB/s per GPU in ({t:7.1f} us) | "
              f"1->3 egress {3 * chunk_bytes / t1 / 1e3:7.1f} | 3->1 ingress {3 * chunk_bytes / t3 / 1e3:7.1f}")
    del src, rx
    torch.cuda.empty_cache()
