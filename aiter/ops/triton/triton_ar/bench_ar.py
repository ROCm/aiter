"""
bench_ar.py -- all-reduce latency: triton (TritonAllReduce) vs RCCL vs aiter
CAR, bf16. Graph replay: CALLS calls per graph, REPLAYS replays, best of
TRIALS, slowest rank.

    # triton: torchrun + gloo
    PROVIDERS=triton torchrun --nproc_per_node=4 bench_ar.py
    # rccl / car: aiter's distributed setup (do not mix with our IPC buffers
    # in one process: that hung, 2026-11-15)
    PROVIDERS=rccl,car python bench_ar.py

Env: SIZES (bytes, comma list), CALLS (10), REPLAYS (50), TRIALS (3),
ONE_SHOT_BYTES (triton's one-shot / two-shot switch), DTYPE (bf16 | fp16).

fp8 all-gather providers (lossy; DTYPE=fp16, the only dtype aiter quantises):
  triton_fp8  TritonAllReduce(fp8=True), ar(x, fp8=True) above one_shot_bytes
  car_fp8     tensor_model_parallel_all_reduce(x, open_fp8_quant=True); aiter
              quantises only fp16 with numel >= 262144, else plain CAR
"""
import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

_HERE = os.path.dirname(os.path.abspath(__file__))
W = int(os.environ.get("WORLD", "4"))
PROVIDERS = os.environ.get("PROVIDERS", "triton").split(",")
SIZES = [int(v) for v in os.environ.get(
    "SIZES", ",".join(str(1 << k) for k in range(13, 27))).split(",")]  # 8 KB .. 64 MB
CALLS = int(os.environ.get("CALLS", "10"))
REPLAYS = int(os.environ.get("REPLAYS", "50"))
TRIALS = int(os.environ.get("TRIALS", "3"))
DTYPE = {"bf16": torch.bfloat16, "fp16": torch.float16}[os.environ.get("DTYPE", "bf16")]


def _worker(rank, init):
    import contextlib
    dev = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(dev)
    use_aiter = init is not None
    if use_aiter:
        import aiter
        from aiter.dist.communication_op import tensor_model_parallel_all_reduce
        from aiter.dist.parallel_state import get_tp_group, graph_capture
        aiter.init_dist_env(W, rank, distributed_init_method=init)
        g = get_tp_group().cpu_group
        nccl_group = get_tp_group().device_group
    else:
        from aiter.ops.triton.triton_ar import TritonAllReduce
        dist.init_process_group("gloo")
        g = dist.group.WORLD
        kw = {}
        if os.environ.get("ONE_SHOT_BYTES"):
            kw["one_shot_bytes"] = int(os.environ["ONE_SHOT_BYTES"])
        ar = TritonAllReduce(rank, W, max_bytes=max(SIZES), fp8="triton_fp8" in PROVIDERS, **kw)
        ar.register(g)
    res = {}
    try:
        keep = {}  # inputs stay alive for the whole run
        for nb in SIZES:
            gen = torch.Generator(device=dev).manual_seed(nb + rank)
            keep[nb] = torch.randn(nb // 2, generator=gen, dtype=DTYPE, device=dev)
        for nb in SIZES:
            x = keep[nb]
            out = torch.empty_like(x)
            xa = x.clone()  # rccl reduces in place; values grow, timing does not change
            fns = {}
            if use_aiter:
                fns["car"] = ((lambda: tensor_model_parallel_all_reduce(x)), graph_capture)
                fns["car_fp8"] = ((lambda: tensor_model_parallel_all_reduce(x, open_fp8_quant=True)),
                                  graph_capture)
                fns["rccl"] = ((lambda: dist.all_reduce(xa, group=nccl_group)), graph_capture)
            else:
                fns["triton"] = ((lambda: ar(x, out)), contextlib.nullcontext)
                fns["triton_fp8"] = ((lambda: ar(x, out, fp8=True)), contextlib.nullcontext)
            for p in PROVIDERS:
                fn, ctx = fns[p]
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                dist.barrier(g)
                gr = torch.cuda.CUDAGraph()
                with ctx() as c:
                    stream = c.stream if p.startswith("car") else None
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
                res[(nb, p)] = best
                del gr
    finally:
        if use_aiter:
            aiter.destroy_dist_env()
    return res


def _fmt(nb):
    return f"{nb >> 20}M" if nb >= 1 << 20 else f"{nb >> 10}K"


def _print(res):
    print(f"{os.environ.get('DTYPE', 'bf16')}, us/call (algbw GB/s = bytes / time)")
    print("size  " + " ".join(f"{p:>16}" for p in PROVIDERS))
    for nb in SIZES:
        print(f"{_fmt(nb):>5} " + " ".join(
            f"{res[(nb, p)]:>8.2f} ({nb / res[(nb, p)] / 1e3:>5.1f})" for p in PROVIDERS))


def main():
    if "RANK" in os.environ:
        assert not {"rccl", "car", "car_fp8"} & set(PROVIDERS), "rccl / car need the aiter path"
        res = _worker(int(os.environ["RANK"]), None)
        if int(os.environ["RANK"]) == 0:
            _print(res)
        dist.barrier()
        return
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
