"""Dump the barrier one-stage fused kernel (ar_rmsnorm_gfx1250) outputs for the LL test shapes.

Run with AITER_CUSTOM_AR_DISABLE_LL=1 so the C++ dispatch takes ar_rmsnorm_gfx1250;
the communicator gate is bypassed by calling ca_comm directly. Same inputs as test_ll_fused.py.
"""

import os
import sys
from multiprocessing import set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)
HIDDEN, EPS = 6144, 1e-6
MS = [1, 4, 12, 16, 32, 64, 128, 256, 341]
DUMP = os.environ["LL_TEST_DUMP"]


def worker(rank, world, init):
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=world, rank=rank, distributed_init_method=init)
    ensure_model_parallel_initialized(world, 1)
    dist.all_reduce(torch.zeros(1).cuda(), group=get_tp_group().device_group)
    ca = get_tp_group().device_communicator.ca_comm
    gen = torch.Generator(device="cuda").manual_seed(1234)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda", generator=gen) * 0.1
    for m in MS:
        torch.manual_seed(100 + rank * 7 + m)
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.5
        torch.manual_seed(999 + m)
        r = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        o, ro = ca.custom_fused_ar_rms(x, r, w, EPS, use_1stage=True, gemma_norm=True)
        q, rq, s = ca.custom_fused_ar_rms_quant(x, r, w, EPS, use_1stage=True, gemma_norm=True)
        torch.cuda.synchronize()
        torch.save({"o": o.cpu(), "ro": ro.cpu(), "q": q.view(torch.uint8).cpu(), "s": s.cpu()},
                   f"{DUMP}/r{rank}_m{m}.pt")
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    init = get_distributed_init_method(get_ip(), get_open_port())
    torch.multiprocessing.spawn(worker, args=(4, init), nprocs=4, join=True)
    print("dumped", DUMP)
