"""Per-call cost of the two MiniMax-M3 TP4 all-reduce sites, as ATOM calls them.

post_attn: fused all-reduce + residual add + Gemma RMSNorm, BF16 out.
post_moe : the same plus per-token FP8 quant (feeds the next layer's FP8 QKV GEMM).
Decode-sized M runs inside one CUDA graph of K back-to-back calls (gaps included);
prefill-sized M runs eagerly. Prints us/call and kernels/call on rank 0.
"""

import os
import statistics
import sys
from multiprocessing import set_start_method

import torch
import torch.distributed as dist

set_start_method("spawn", force=True)

HIDDEN, EPS, K = 6144, 1e-6, 16
DECODE_M = [1, 4, 12, 16, 32, 64, 128, 512]
PREFILL_M = [2517, 8192, 16384]
if os.environ.get("BENCH_M"):
    DECODE_M = [int(v) for v in os.environ["BENCH_M"].split(",")]
    PREFILL_M = [int(v) for v in os.environ.get("BENCH_PREFILL_M", "").split(",") if v]
SITES = os.environ.get("BENCH_SITES", "post_attn,post_moe").split(",")


def worker(rank, world, init, out_file):
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
    torch.manual_seed(rank)
    w = torch.randn(HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.1

    def site(name, x, r):
        if name == "bare_ar":
            return tensor_model_parallel_all_reduce(x)
        if name == "post_attn":
            return tensor_model_parallel_fused_allreduce_rmsnorm(x, r, w, EPS, gemma_norm=True)
        return tensor_model_parallel_fused_allreduce_rmsnorm_quant(
            x, r, w, EPS, quant_type="per_token", gemma_norm=True)

    rows = []
    for m in DECODE_M + PREFILL_M:
        x = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda") * 0.05
        r = torch.randn(m, HIDDEN, dtype=torch.bfloat16, device="cuda")
        for name in SITES:
            graph = m in DECODE_M
            if graph:
                g = torch.cuda.CUDAGraph()
                with graph_capture() as gc, torch.cuda.graph(g, stream=gc.stream):
                    for _ in range(K):
                        site(name, x, r)
                fn = g.replay
            else:
                def fn():
                    for _ in range(K):
                        site(name, x, r)
            for _ in range(5):
                fn()
            torch.cuda.synchronize()
            dist.barrier(group=get_tp_group().cpu_group)
            times = []
            for _ in range(20):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                fn()
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e) * 1e3 / K)
            kernels = ""
            dist.barrier(group=get_tp_group().cpu_group)
            if rank != 0:
                fn()
                torch.cuda.synchronize()
            else:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
                    fn()
                    torch.cuda.synchronize()
                names = [ev.name for ev in prof.events() if ev.device_type == torch.autograd.DeviceType.CUDA]
                from collections import Counter
                c = Counter(n.split("<")[0].split("(")[0][:48] for n in names)
                kernels = f"{len(names) / K:.1f}|" + ";".join(f"{k}x{v // K}" for k, v in c.items())
            dist.barrier(group=get_tp_group().cpu_group)
            rows.append((m, name, "graph" if graph else "eager", statistics.median(times), kernels))
            if rank == 0:
                print(f"M={m:6d} {name:9s} {rows[-1][2]:5s} {rows[-1][3]:9.2f} us/call  kernels/call={kernels}", flush=True)
    if rank == 0 and out_file:
        with open(out_file, "w") as f:
            f.write("M,site,mode,us_per_call,kernels\n")
            for row in rows:
                f.write(",".join(str(v) for v in row) + "\n")
    torch.cuda.synchronize()
    get_tp_group().barrier()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port

    world = 4
    out_file = sys.argv[1] if len(sys.argv) > 1 else ""
    init = get_distributed_init_method(get_ip(), get_open_port())
    torch.multiprocessing.spawn(worker, args=(world, init, out_file), nprocs=world, join=True)
