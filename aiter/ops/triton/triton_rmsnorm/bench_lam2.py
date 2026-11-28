"""
bench_lam2.py -- all-reduce + residual add + RMSNorm latency: lam2
(TlLamport2RMSNorm) vs aiter CAR + rmsnorm2d_fwd_with_add vs aiter's fused
kernel vs RCCL + the same norm. Graph replay: CALLS calls per graph, REPLAYS
replays, best of TRIALS, slowest rank.

    # lam2: torchrun + gloo
    PROVIDERS=lam2 torchrun --nproc_per_node=4 bench_lam2.py
    # aiter: its own distributed setup. Do not mix with lam2 in one process:
    # our IPC buffers next to aiter's communicators hung (2026-11-15).
    PROVIDERS=car python bench_lam2.py

Providers:
  lam2       TlLamport2RMSNorm (all-reduce + norm, 2 kernels)
  lam2_ar    its all-reduce kernel alone;  lam2_norm  its norm kernel alone
  car        aiter custom all-reduce + rmsnorm2d_fwd_with_add (+ static quant)
  car_ar     aiter custom all-reduce alone;  aiter_norm  the aiter norm alone
  car_fused  aiter tensor_model_parallel_fused_allreduce_rmsnorm[_quant]
             (its fp8 is per-token, not static per-tensor)
  rccl       RCCL all-reduce + the aiter norm (+ static quant)

Env: ROWS (1,4,16,64,128,256,512,1024,2048), H (8192), FP8 (0 | 1: static
per-tensor fp8 out), CALLS (10), REPLAYS (50), TRIALS (3). On some ROCm
stacks a process looping over many sizes hung or crashed: run one provider,
or one row count, per process if that happens.
"""
import contextlib
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

W = int(os.environ.get("WORLD", "4"))
H = int(os.environ.get("H", "8192"))
ROWS = [int(r) for r in os.environ.get("ROWS", "1,4,16,64,128,256,512,1024,2048").split(",")]
PROVIDERS = os.environ.get("PROVIDERS", "lam2").split(",")
FP8 = os.environ.get("FP8", "0") == "1"
CALLS = int(os.environ.get("CALLS", "10"))
REPLAYS = int(os.environ.get("REPLAYS", "50"))
TRIALS = int(os.environ.get("TRIALS", "3"))
EPS, SCALE = 1e-6, 0.05
AITER = {"car", "car_ar", "aiter_norm", "car_fused", "rccl"}


def _worker(rank, init):
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    use_aiter = init is not None
    graph_capture = None
    if use_aiter:
        import aiter
        from aiter import rmsnorm2d_fwd_with_add
        from aiter.dist.communication_op import (
            tensor_model_parallel_all_reduce,
            tensor_model_parallel_fused_allreduce_rmsnorm,
            tensor_model_parallel_fused_allreduce_rmsnorm_quant,
        )
        from aiter.dist.parallel_state import get_tp_group, graph_capture

        aiter.init_dist_env(W, rank, distributed_init_method=init)
        g = get_tp_group().cpu_group
        nccl_group = get_tp_group().device_group
    else:
        from aiter.ops.triton.triton_rmsnorm import TlLamport2RMSNorm

        dist.init_process_group("gloo")
        g = dist.group.WORLD
        lam = TlLamport2RMSNorm(rank, W, H, max(ROWS))
        lam.register(g)
    w = (torch.rand(H, generator=torch.Generator(device=dev).manual_seed(5),
                    dtype=torch.float32, device=dev) + 0.5).to(torch.bfloat16)
    sc = torch.tensor([SCALE], dtype=torch.float32, device=dev) if FP8 else None
    res = {}
    try:
        keep = {}  # inputs stay alive for the whole run
        for rows in ROWS:
            gen = torch.Generator(device=dev).manual_seed(rows + rank)
            keep[rows] = (
                torch.randn(rows, H, generator=gen, dtype=torch.bfloat16, device=dev),
                torch.randn(rows, H, generator=gen, dtype=torch.bfloat16, device=dev))
        for rows in ROWS:
            x, r = keep[rows]
            out, ro = torch.empty_like(x), torch.empty_like(x)
            q = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=dev)
            xa = x.clone()  # rccl reduces in place; values grow, timing does not change
            fns = {}
            if use_aiter:
                def car():
                    y = tensor_model_parallel_all_reduce(x)
                    rmsnorm2d_fwd_with_add(out, y, r, ro, w, EPS)
                    if FP8:
                        aiter.static_per_tensor_quant(q, out, sc)

                def rccl():
                    dist.all_reduce(xa, group=nccl_group)
                    rmsnorm2d_fwd_with_add(out, xa, r, ro, w, EPS)
                    if FP8:
                        aiter.static_per_tensor_quant(q, out, sc)

                fused = ((lambda: tensor_model_parallel_fused_allreduce_rmsnorm_quant(
                              x, r, w, EPS, prefill_support=True, quant_type="per_token"))
                         if FP8 else
                         (lambda: tensor_model_parallel_fused_allreduce_rmsnorm(
                              x, r, w, EPS, prefill_support=True)))
                fns = {
                    "car": car,
                    "rccl": rccl,
                    "car_ar": lambda: tensor_model_parallel_all_reduce(x),
                    "aiter_norm": lambda: rmsnorm2d_fwd_with_add(out, x, r, ro, w, EPS),
                    "car_fused": fused,
                }
            else:
                parts = {"lam2": ("ar", "norm"), "lam2_ar": ("ar",), "lam2_norm": ("norm",)}
                fns = {p: (lambda pp=pp: lam(x, r, w, EPS, sc, parts=pp)) for p, pp in parts.items()}
            for p in PROVIDERS:
                fn = fns[p]
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                dist.barrier(g)
                gr = torch.cuda.CUDAGraph()
                with (graph_capture() if use_aiter else contextlib.nullcontext()) as c:
                    stream = c.stream if use_aiter and p.startswith("car") else None
                    with torch.cuda.graph(gr, stream=stream):
                        for _ in range(CALLS):
                            fn()
                torch.cuda.synchronize()
                best = 1e30
                for _ in range(TRIALS):
                    dist.barrier(g)
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(REPLAYS):
                        gr.replay()
                    e.record()
                    torch.cuda.synchronize()
                    t = torch.tensor([s.elapsed_time(e) * 1e3 / (REPLAYS * CALLS)])
                    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=g)
                    best = min(best, t.item())
                res[(rows, p)] = best
                del gr
    finally:
        if use_aiter:
            aiter.destroy_dist_env()
    return res


def _print(res):
    print(f"H={H}, output {'fp8' if FP8 else 'bf16'}, us/call")
    print("rows " + " ".join(f"{p:>10}" for p in PROVIDERS))
    for rows in ROWS:
        print(f"{rows:>4} " + " ".join(f"{res[(rows, p)]:>10.2f}" for p in PROVIDERS))


def main():
    if "RANK" in os.environ:  # torchrun: lam2
        assert not AITER & set(PROVIDERS), "aiter providers run with plain python"
        res = _worker(int(os.environ["RANK"]), None)
        if int(os.environ["RANK"]) == 0:
            _print(res)
        dist.barrier()
        return
    assert set(PROVIDERS) <= AITER, "lam2 providers run under torchrun"
    from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
    from aiter.test_common import ensure_spawn_method

    ensure_spawn_method()
    init = get_distributed_init_method(get_ip(), get_open_port())
    with mp.Pool(W) as pool:
        out = [pool.apply_async(_worker, (r, init)) for r in range(W)]
        res = out[0].get(timeout=3600)
        for o in out[1:]:
            o.get(timeout=3600)
    _print(res)


if __name__ == "__main__":
    main()
