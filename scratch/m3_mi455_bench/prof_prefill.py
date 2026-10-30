"""Per-kernel device time of the M3 all-reduce sites at prefill sizes (eager), rank 0.

Usage: prof_prefill.py M[,M...] [post_attn|post_moe|bare_ar]
Every rank runs the calls; only rank 0 records, so collectives stay matched.
"""

import statistics
import sys
from collections import defaultdict
from multiprocessing import set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)
HIDDEN, EPS, K = 6144, 1e-6, 8


def worker(rank, world, init, ms, site_name):
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
        init_distributed_environment,
        set_custom_all_reduce,
    )

    torch.cuda.set_device(rank)
    set_custom_all_reduce(True)
    init_distributed_environment(world_size=world, rank=rank, distributed_init_method=init)
    ensure_model_parallel_initialized(world, 1)
    dist.all_reduce(torch.zeros(1).cuda(), group=get_tp_group().device_group)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1

    def site(x, r):
        if site_name == "bare_ar":
            return tensor_model_parallel_all_reduce(x)
        if site_name == "post_attn":
            return tensor_model_parallel_fused_allreduce_rmsnorm(x, r, w, EPS, gemma_norm=True)
        return tensor_model_parallel_fused_allreduce_rmsnorm_quant(
            x, r, w, EPS, quant_type="per_token", gemma_norm=True)

    for m in ms:
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05
        r = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for _ in range(3):
            site(x, r)
        torch.cuda.synchronize()
        dist.barrier(group=get_tp_group().cpu_group)
        times = []
        for _ in range(5):
            st, en = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            st.record()
            for _ in range(K):
                site(x, r)
            en.record()
            torch.cuda.synchronize()
            times.append(st.elapsed_time(en) * 1e3 / K)
        dist.barrier(group=get_tp_group().cpu_group)
        if rank != 0:
            for _ in range(K):
                site(x, r)
            torch.cuda.synchronize()
        else:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(K):
                    site(x, r)
                torch.cuda.synchronize()
            per = defaultdict(list)
            evs = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
            for e in evs:
                per[e.name.split("(")[0][:70]].append(e.device_time)
            mb = m * HIDDEN * 2 / 1e6
            span = statistics.median(times)
            bound = 2 * 0.75 * mb / 1.48  # two-shot at the measured ~1.48 TB/s ingress, us
            print(f"M={m} {site_name}: {mb:.1f} MB/rank, {span:.1f} us/call (events), "
                  f"two-shot ingress bound {bound:.1f} us", flush=True)
            for name, ts in sorted(per.items(), key=lambda kv: -sum(kv[1])):
                print(f"   {statistics.median(ts):8.1f} us x{len(ts) // K}  {name}", flush=True)
        dist.barrier(group=get_tp_group().cpu_group)
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    ms = [int(v) for v in sys.argv[1].split(",")]
    site_name = sys.argv[2] if len(sys.argv) > 2 else "post_moe"
    init = get_distributed_init_method(get_ip(), get_open_port())
    torch.multiprocessing.spawn(worker, args=(4, init, ms, site_name), nprocs=4, join=True)
