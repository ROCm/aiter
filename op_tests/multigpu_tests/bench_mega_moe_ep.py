# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Fused EP MegaMoE layer perf (DeepSeek-V4-Pro shape, MXFP4 weights) over token
counts and routings, against the two-stage ``MegaMoEV2`` on the same shape::

    torchrun --nproc_per_node=8 op_tests/multigpu_tests/bench_mega_moe_ep.py \\
        --engine ep -q a8w4 --mtpr 8192 -t 1024 2048 4096 8192 -r uniform zipf:1 spike

One engine per run (``--engine ep | v2``): EP uses a hipIpc symmetric arena, V2 mori
shmem.  ``v2`` takes whichever ``MegaMoEV2`` is on ``PYTHONPATH``; if it has a small-m
path (``small_m_mtpr``) the token counts it covers go through ``forward_small_m``.

The time is the mean over ``--iters`` CUDA-graph replays of one forward (``--eager``:
eager calls), wall clock, taken as the worst rank.  The table also lists, per case,
``bal`` = mean / max routed rows per rank and ``hot`` = max / mean routed rows per
expert.  Routings: see ``test_mega_moe_ep.route_ids``.  The weights are random, so the
outputs are not checked here (``test_mega_moe_ep.py`` does that).
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_mega_moe_ep import EPR, SWIGLU_LIMIT, H, I, K, route_ids


def _weights(rank, device):
    qz = aiter.get_torch_quant(aiter.QuantType.per_1x32)
    g = torch.Generator(device=device).manual_seed(100 + rank)
    out = []
    for rows, cols in ((2 * I, H), (H, I)):
        w = torch.randn(
            EPR, rows, cols, dtype=torch.bfloat16, device=device, generator=g
        )
        q, s = qz(w * cols**-0.25, quant_dtype=dtypes.fp4x2)
        out += [q.view(EPR, rows, cols // 2).contiguous(), s]
    return out


def build(engine, quant, mtpr, rank, world, device):
    w1q, w1s, w2q, w2s = _weights(rank, device)
    common = {
        "rank": rank,
        "world_size": world,
        "model_dim": H,
        "inter_dim": I,
        "experts": EPR * world,
        "topk": K,
        "max_tok_per_rank": mtpr,
        "quant": quant,
        "swiglu_limit": SWIGLU_LIMIT,
    }
    if engine == "ep":
        from aiter.ops.flydsl.mega_moe_ep import MegaMoeEP, MegaMoeEPConfig

        layer = MegaMoeEP(
            MegaMoeEPConfig(**common),
            w1=w1q,
            w1_scale=w1s.view(EPR, 2 * I, H // 32).contiguous(),
            w2=w2q,
            w2_scale=w2s.view(EPR, H, I // 32).contiguous(),
            device=device,
        )
        layer.prepare()
        return layer
    from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    return MegaMoEV2(
        **common,
        w1=shuffle_weight_a16w4(w1q, 16, True).contiguous(),
        w1_scale=shuffle_scale_a16w4(w1s, EPR, True).contiguous(),
        w2=shuffle_weight_a16w4(w2q, 16, False).contiguous(),
        w2_scale=shuffle_scale_a16w4(w2s, EPR, False).contiguous(),
    )


def balance(ids, world):
    rk = torch.bincount(ids.flatten().long() // EPR, minlength=world).float()
    ex = torch.bincount(ids.flatten().long(), minlength=EPR * world).float()
    dist.all_reduce(rk)
    dist.all_reduce(ex)
    return (rk.mean() / rk.max()).item(), (ex.max() / ex.mean()).item()


def bench(layer, x, wts, ids, args, world, device):
    tokens = x.shape[0]
    if tokens <= getattr(layer, "small_m_mtpr", 0):
        call = lambda: layer.forward_small_m(x, wts, ids, max_tokens=tokens)
    else:
        call = lambda: layer(x, wts, ids)
    for _ in range(args.warmup):
        call()
    torch.cuda.synchronize()
    dist.barrier()
    step = call
    if not args.eager:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            call()
        for _ in range(args.warmup):
            graph.replay()
        step = graph.replay
    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()
    ms = torch.tensor([(time.perf_counter() - t0) / args.iters * 1e3], device=device)
    dist.all_reduce(ms, op=dist.ReduceOp.MAX)
    return ms.item()


def main() -> int:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter, description=__doc__
    )
    parser.add_argument("--engine", default="ep", choices=["ep", "v2"])
    parser.add_argument("-q", "--quant", default="a8w4", choices=["a8w4", "a4w4"])
    parser.add_argument("--mtpr", type=int, default=8192, help="max_tok_per_rank")
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 64, 512, 1024, 2048, 4096, 8192],
        help="Tokens per rank (cases above mtpr are skipped).",
    )
    parser.add_argument(
        "-r",
        "--routes",
        nargs="*",
        default=["uniform", "zipf:0.5", "zipf:1", "zipf:2", "spike"],
    )
    parser.add_argument("--eager", action="store_true", help="Time eager calls.")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    args = parser.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    device = torch.device("cuda", lr)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=device)
    if args.engine == "v2":
        import mori.shmem as mori_shmem
        import torch._C._distributed_c10d as c10d

        c10d._register_process_group("default", dist.group.WORLD)
        mori_shmem.shmem_torch_process_group_init("default")

    layer = build(args.engine, args.quant, args.mtpr, rank, world, device)
    rows = []
    for route in args.routes:
        for tokens in (t for t in args.tokens if t <= args.mtpr):
            x, wts, ids = route_ids(route, tokens, EPR * world, world, rank, 0, device)
            bal, hot = balance(ids, world)
            ms = bench(layer, x, wts, ids, args, world, device)
            rows.append(
                {
                    "engine": args.engine,
                    "quant": args.quant,
                    "mtpr": args.mtpr,
                    "route": route,
                    "tokens": tokens,
                    "bal": f"{bal:.3f}",
                    "hot": f"{hot:.1f}",
                    "ms": f"{ms:.3f}",
                }
            )
            if rank == 0:
                aiter.logger.info("%s", rows[-1])
            dist.barrier()
    if rank == 0:
        mode = "eager" if args.eager else "graph"
        aiter.logger.info(
            "MegaMoE EP bench ep%d (%s, worst rank):\n%s",
            world,
            mode,
            pd.DataFrame(rows).to_markdown(index=False),
        )
    del layer
    if args.engine == "v2":
        mori_shmem.shmem_finalize()
    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
