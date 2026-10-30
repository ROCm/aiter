"""Read the debug phase timestamps the instrumented ar_ll_rmsnorm_gfx1250 writes into residual_out."""

import statistics
import sys
from multiprocessing import set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)
HIDDEN, EPS, K = 6144, 1e-6, 16


def worker(rank, world, init, m, result_q):
    from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm
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
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
    x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05
    r = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    g = torch.cuda.CUDAGraph()
    outs = []
    with graph_capture() as gc, torch.cuda.graph(g, stream=gc.stream):
        for _ in range(K):
            outs.append(tensor_model_parallel_fused_allreduce_rmsnorm(x, r, w, EPS, gemma_norm=True))
    for _ in range(10):
        g.replay()
    torch.cuda.synchronize()
    ts = outs[-1][1].view(torch.int64)[:, :4].cpu()  # [m, 4] ticks of 10 ns
    d = (ts[:, 1:] - ts[:, :3]).float()  # ticks
    tot = (ts[:, 3] - ts[:, 0]).float()
    result_q.put((rank, [statistics.median(d[:, i].tolist()) for i in range(3)] + [statistics.median(tot.tolist())]))
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    m = int(sys.argv[1])
    init = get_distributed_init_method(get_ip(), get_open_port())
    q = torch.multiprocessing.get_context("spawn").SimpleQueue()
    torch.multiprocessing.spawn(worker, args=(4, init, m, q), nprocs=4, join=True)
    rows = sorted(q.get() for _ in range(4))
    print(f"M={m}: per owner block, median over rows, wall_clock64 ticks (share of total): publish | wait for peers | epilogue | total")
    for rank, v in rows:
        t = v[3] or 1
        print(f"  rank {rank}: {v[0]:6.0f} ({v[0]/t:4.0%}) | {v[1]:6.0f} ({v[1]/t:4.0%}) | {v[2]:6.0f} ({v[2]/t:4.0%}) | {v[3]:6.0f}")
