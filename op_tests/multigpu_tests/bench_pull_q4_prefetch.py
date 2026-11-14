# SPDX-License-Identifier: MIT
"""TP4 PullQ4 A/B benchmark, run in separate processes for each env value.

AITER_PULL_Q4_PREFETCH=0 python bench_pull_q4_prefetch.py baseline 19301
AITER_PULL_Q4_PREFETCH=1 python bench_pull_q4_prefetch.py candidate 19302 --compare baseline.json

Compare each output JSON's full-tensor hashes before considering timings.
Timings include quantize, publish, and reduce, not just the optimized kernel.
"""

import argparse
import hashlib
import json
import os
import pathlib
import statistics

os.environ["AITER_QUICK_REDUCE_QUANTIZATION"] = "INT4"
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from aiter.dist.device_communicators.quick_all_reduce import QuickAllReduce


def worker(rank, port, tag):
    torch.cuda.set_device(rank)
    torch.manual_seed(188 + rank)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=4
    )
    ar = QuickAllReduce(dist.group.WORLD, torch.device(f"cuda:{rank}"))
    assert ar.uses_pull_q4
    rows = []
    for n in [
        1,
        7,
        1023,
        16391,
        8 * 1024**2 // 2,
        32 * 1024**2 // 2,
        1216 * 16384,
        64 * 1024**2 // 2,
    ]:
        x = torch.randn(n, device="cuda", dtype=torch.bfloat16) * (rank + 1)
        y = ar.quick_all_reduce(x)
        torch.cuda.synchronize()
        h = hashlib.sha256(y.cpu().view(torch.uint16).numpy().tobytes()).hexdigest()
        hashes = [None] * 4
        dist.all_gather_object(hashes, h)
        assert len(set(hashes)) == 1, hashes
        row = dict(n=n, hash=h)
        if n >= 8 * 1024**2 // 2:
            for _ in range(3):
                ar.quick_all_reduce(x)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                for _ in range(10):
                    ar.quick_all_reduce(x)
            torch.cuda.synchronize()
            dist.barrier()
            values = []
            for _ in range(5):
                a, b = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                a.record()
                g.replay()
                b.record()
                b.synchronize()
                values.append(a.elapsed_time(b) * 1000 / 10)
            alltimes = [None] * 4
            dist.all_gather_object(alltimes, values)
            row["us_per_rank"] = [statistics.median(v) for v in alltimes]
            row["us"] = max(row["us_per_rank"])
        rows.append(row)
        if rank == 0:
            print(json.dumps(row), flush=True)
        del x, y
    dist.barrier()
    ar.close()
    dist.destroy_process_group()
    if rank == 0:
        pathlib.Path(f"{tag}.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag")
    parser.add_argument("port", type=int)
    parser.add_argument("--compare", type=pathlib.Path)
    args = parser.parse_args()
    mp.spawn(worker, args=(args.port, args.tag), nprocs=4, join=True)
    if args.compare is not None:
        baseline = json.loads(args.compare.read_text())
        candidate = json.loads(pathlib.Path(f"{args.tag}.json").read_text())
        assert len(baseline) == len(candidate)
        for old, new in zip(baseline, candidate):
            assert old["n"] == new["n"]
            assert old["hash"] == new["hash"], f"output differs at {old['n']} elements"
            if "us" in old:
                print(
                    f"{old['n'] * 2 / 2**20:g} MiB: "
                    f"{old['us']:.2f} -> {new['us']:.2f} us, "
                    f"{old['us'] / new['us']:.3f}x; bitwise equal"
                )
