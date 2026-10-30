"""Kernel durations and gaps on rank 0 for one site at one M, fused vs unfused.

Usage: python prof_ll.py <M> <fused|unfused|ar_only> [post_attn|post_moe]
"""

import statistics
import sys
from multiprocessing import set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)
HIDDEN, EPS, K = 6144, 1e-6, 16


def worker(rank, world, init, m, variant, site):
    from aiter.dist.communication_op import (
        tensor_model_parallel_all_reduce,
        tensor_model_parallel_fused_allreduce_rmsnorm,
        tensor_model_parallel_fused_allreduce_rmsnorm_quant,
    )
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=world, rank=rank, distributed_init_method=init)
    ensure_model_parallel_initialized(world, 1)
    dist.all_reduce(torch.zeros(1).cuda(), group=get_tp_group().device_group)
    ca = get_tp_group().device_communicator.ca_comm
    if variant == "unfused":
        ca.should_fuse_ar_rms = lambda *a, **k: False
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
    x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05
    r = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")

    def call():
        if variant == "ar_only":
            return tensor_model_parallel_all_reduce(x)
        if site == "post_attn":
            return tensor_model_parallel_fused_allreduce_rmsnorm(x, r, w, EPS, gemma_norm=True)
        return tensor_model_parallel_fused_allreduce_rmsnorm_quant(
            x, r, w, EPS, quant_type="per_token", gemma_norm=True)

    g = torch.cuda.CUDAGraph()
    with graph_capture() as gc, torch.cuda.graph(g, stream=gc.stream):
        for _ in range(K):
            call()
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    dist.barrier(group=get_tp_group().cpu_group)
    if rank == 0:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                g.replay()
            torch.cuda.synchronize()
    else:
        for _ in range(5):
            g.replay()
        torch.cuda.synchronize()
    dist.barrier(group=get_tp_group().cpu_group)
    if rank == 0:
        evs = sorted([e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA],
                     key=lambda e: e.time_range.start)
        by = {}
        for e in evs:
            by.setdefault(e.name.split("<")[0].split("(")[0][:40], []).append(e.time_range.elapsed_us())
        gaps = [b.time_range.start - a.time_range.end for a, b in zip(evs, evs[1:])]
        span = (evs[-1].time_range.end - evs[0].time_range.start) / (5 * K)
        print(f"M={m} {variant} {site}: {len(evs) / (5 * K):.1f} kernels/call, span {span:.2f} us/call, "
              f"gap median {statistics.median(gaps):.2f} us")
        for n, v in by.items():
            print(f"   {n:40s} median {statistics.median(v):7.2f} us  p10 {sorted(v)[len(v) // 10]:7.2f}  max {max(v):7.2f}")
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    m, variant = int(sys.argv[1]), sys.argv[2]
    site = sys.argv[3] if len(sys.argv) > 3 else "post_moe"
    init = get_distributed_init_method(get_ip(), get_open_port())
    torch.multiprocessing.spawn(worker, args=(4, init, m, variant, site), nprocs=4, join=True)
