"""Read the debug phase stamps the instrumented ar_push_rmsnorm_gfx1250 writes into residual_out."""
import sys
from multiprocessing import set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)
HIDDEN, EPS = 6144, 1e-6


def worker(rank, world, init, m, result_q):
    from aiter.dist.communication_op import tensor_model_parallel_fused_allreduce_rmsnorm_quant
    from aiter.dist.parallel_state import (destroy_distributed_environment, destroy_model_parallel,
                                           ensure_model_parallel_initialized, get_tp_group,
                                           init_distributed_environment, set_custom_all_reduce)
    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=world, rank=rank, distributed_init_method=init)
    ensure_model_parallel_initialized(world, 1)
    dist.all_reduce(torch.zeros(1).cuda(), group=get_tp_group().device_group)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1
    x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05
    r = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
    for _ in range(6):
        q, ro, s = tensor_model_parallel_fused_allreduce_rmsnorm_quant(x, r, w, EPS, quant_type="per_token", gemma_norm=True)
    torch.cuda.synchronize()
    waves = min(256, (m + 31) // 32) * 32
    rows = ro[: min(m, waves)].contiguous().view(torch.int64)[:, :6].cpu().double()
    d = rows - rows[:, :1]
    ph = torch.stack([d[:, 1] - d[:, 0], d[:, 2] - d[:, 1], d[:, 3] - d[:, 2], d[:, 4] - d[:, 3], d[:, 5] - d[:, 4]], 1)
    result_q.put((rank, d[:, 0].max().item(), ph.median(0).values.tolist(), ph.max(0).values.tolist(), d[:, 5].max().item()))
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
    print(f"M={m}: us (10 ns ticks /100): start skew | A push | wait RS | B reduce+push | wait AG | C norm (per wave) | block end (max)")
    for rank, skew, med, mx, end in sorted(q.get() for _ in range(4)):
        f = lambda v: " | ".join(f"{x / 100:6.1f}" for x in v)
        print(f"  rank {rank} median: {skew / 100:6.1f} | {f(med)} | {end / 100:6.1f}")
        print(f"  rank {rank} max   :        | {f(mx)}")
