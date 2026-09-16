# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Perf + correctness bench for the two gfx1250 CAS 2-shot all-reduce variants.
#
#   inplace  — ar_cas_2shot_gfx1250. Reduce-scatters into the caller's registered
#              input buffer, i.e. it destroys the input. This is the variant on
#              the routed path.
#   scratch  — ar_cas_2shot_gfx1250_scratch. Reduce-scatters into a separate
#              peer-visible buffer, leaving the input read-only. Not routed;
#              reachable only through its standalone entry point.
#
# Both are called directly here, bypassing size-based routing, so a single shape
# can be compared across variants.
#
# Usage:
#   python bench_gfx1250_cas_2shot.py -t 4
#   python bench_gfx1250_cas_2shot.py -t 4 --variant scratch
#   python bench_gfx1250_cas_2shot.py -t 2 -m eager -s 2048,8192

import argparse
import os

os.environ.setdefault("ENABLE_CK", "0")
# The scratch staging buffer is allocated at communicator init, which happens in
# the worker processes; set this before they are spawned.
os.environ["AITER_GFX1250_CAS_SCRATCH"] = "1"

from multiprocessing import Pool, freeze_support, set_start_method

import pandas as pd
import torch
import torch.distributed as dist

from aiter import dtypes, logger
from aiter.dist.utils import get_distributed_init_method, get_ip, get_open_port
from aiter.test_common import perftest

set_start_method("spawn", force=True)

_VARIANT_OPS = {
    "inplace": "all_reduce_cas_2shot_gfx1250",
    "scratch": "all_reduce_cas_2shot_scratch_gfx1250",
}

# CAS is the tail of the routing table: > 16 MiB at tp4, > 64 MiB at tp2. The
# sweep starts below that so the comparison also covers sizes the router would
# hand to LL128, which is useful when deciding whether the band boundary is right.
SHAPES = [
    (256, 8192),
    (512, 8192),
    (1024, 8192),
    (1536, 8192),
    (2048, 8192),
    (3072, 8192),
    (4096, 8192),
]


def _worker(tp_size, rank, x_cpu, init_method, variant, with_graph, block_size, unroll):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    import aiter as ops
    from aiter.dist.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        get_tp_group,
        graph_capture,
        init_distributed_environment,
        set_custom_all_reduce,
    )

    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=tp_size, rank=rank, distributed_init_method=init_method
    )
    ensure_model_parallel_initialized(tp_size, 1)

    x = x_cpu.to(device)
    tp_group = get_tp_group()
    ca = tp_group.device_communicator.ca_comm
    assert ca is not None and not ca.disabled, "custom allreduce is disabled"

    ca_ptr = ca._ptr
    op_fn = getattr(ops, _VARIANT_OPS[variant])
    reg_ptr = ca._pool["input"].data_ptr
    reg_bytes = ca._pool["input"].max_size

    def do_all_reduce(inp):
        out = torch.empty_like(inp)
        op_fn(ca_ptr, inp, out, reg_ptr, reg_bytes, block_size, unroll)
        return out

    dist.all_reduce(torch.zeros(1, device=device), group=tp_group.device_group)
    torch.cuda.synchronize()

    if with_graph:
        graph = torch.cuda.CUDAGraph()
        with graph_capture() as gc:
            with torch.cuda.graph(graph, stream=gc.stream):
                out = do_all_reduce(x)
        out.fill_(0)

        @perftest()
        def run():
            graph.replay()

        _, us = run()
        result = out
    else:

        @perftest()
        def run(inp):
            return do_all_reduce(inp)

        result, us = run(x)

    # Whether the kernel left the input alone. inplace is expected to fail this;
    # that difference is the whole reason the scratch variant exists.
    input_intact = torch.equal(x.cpu(), x_cpu)

    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
    return result.cpu(), us, input_intact


def run_one(tp_size, shape, variant, with_graph, init_method, block_size, unroll):
    dtype = dtypes.bf16
    torch.manual_seed(0)
    xs = [torch.randn(shape, dtype=dtype) for _ in range(tp_size)]
    ref = torch.stack([v.float() for v in xs]).sum(0).to(dtype)

    with Pool(tp_size) as pool:
        outs = [
            pool.apply_async(
                _worker,
                args=(
                    tp_size, i, xs[i], init_method, variant, with_graph,
                    block_size, unroll,
                ),
            )
            for i in range(tp_size)
        ]
        results = [o.get(timeout=900) for o in outs]

    err = max((r.float() - ref.float()).abs().max().item() for r, _, _ in results)
    us = max(us for _, us, _ in results)
    intact = all(i for _, _, i in results)
    return us, err, intact


def main():
    parser = argparse.ArgumentParser(description="gfx1250 CAS 2-shot variant bench")
    parser.add_argument("-t", "--tp-size", type=int, choices=[2, 4], default=4)
    parser.add_argument(
        "--variant", type=str, default="inplace,scratch",
        help="comma-separated: inplace, scratch",
    )
    parser.add_argument(
        "-m", "--mode", type=str, choices=["graph", "eager"], default="graph"
    )
    parser.add_argument(
        "-s", "--shape", type=dtypes.str2tuple, default=None,
        help="single shape override, e.g. -s 2048,8192",
    )
    parser.add_argument("-b", "--block-size", type=int, default=1024)
    parser.add_argument("-u", "--unroll", type=int, default=1)
    args = parser.parse_args()

    shapes = [args.shape] if args.shape else SHAPES
    variants = [v.strip() for v in args.variant.split(",")]
    init_method = get_distributed_init_method(get_ip(), get_open_port())

    rows = []
    for shape in shapes:
        mib = shape[0] * shape[1] * 2 / 1024 / 1024
        for variant in variants:
            us, err, intact = run_one(
                args.tp_size, shape, variant, args.mode == "graph", init_method,
                args.block_size, args.unroll,
            )
            rows.append(
                {
                    "shape": str(shape),
                    "MiB": round(mib, 2),
                    "variant": variant,
                    "us": round(us, 2),
                    "err": err,
                    "input_intact": intact,
                }
            )
            logger.info("%s", rows[-1])

    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))

    if set(variants) == set(_VARIANT_OPS):
        pivot = df.pivot(index=["shape", "MiB"], columns="variant", values="us")
        pivot["scratch/inplace"] = (pivot["scratch"] / pivot["inplace"]).round(3)
        print()
        print(pivot.to_string())


if __name__ == "__main__":
    freeze_support()
    main()
