"""Per-wave stamps of the instrumented role-split ar_push_rmsnorm_gfx1250 (4 x u64 a wave in residual_out)."""
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
    st = ro.view(torch.int64).flatten()[: 256 * 32 * 4].view(256, 32, 4).cpu().double()
    d = (st - st[:, :, :1]) / 100.0  # us since the wave's start
    mov, nor = d[:, :16], d[:, 16:]
    out = {}
    out["mover A done"] = mov[..., 1].flatten()
    out["mover 1st RS seen"] = mov[..., 2].flatten()
    out["mover end"] = mov[..., 3].flatten()
    out["norm 1st AG seen"] = nor[..., 1].flatten()
    out["norm end"] = nor[..., 3].flatten()
    res = {k: (v[v > 0].median().item() if (v > 0).any() else 0, v.max().item()) for k, v in out.items()}
    result_q.put((rank, res))
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
    for rank, res in sorted(q.get() for _ in range(4)):
        if rank == 0:
            print(f"M={m} rank {rank}: us since the wave's own start, median / max")
            for k, (med, mx) in res.items():
                print(f"   {k:20s} {med:7.1f} / {mx:7.1f}")
