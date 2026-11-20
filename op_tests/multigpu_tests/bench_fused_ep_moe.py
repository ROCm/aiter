# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""FusedEPMoE against MegaMoEV2 on 8 ranks, DeepSeek-V4.1-Flash EP8 layer (384 experts, 48 per rank, top 6,
hidden 5120, inter 2304, MXFP4 weights, bf16 tokens in, bf16 out).

    MORI_SHMEM_HEAP_SIZE=64G torchrun --nproc-per-node 8 op_tests/multigpu_tests/bench_fused_ep_moe.py

Each (implementation, routing, tokens per rank) captures ``--ninputs`` calls on distinct tokens and routes, one
CUDA graph per call, and replays them back to back ``--reps`` times; us per call is the max over the ranks. The
samples run in A B ... B A order over implementations and routings. Routing: ``uniform`` (random top 6 of 384),
``zipf`` (Zipf 1.2 expert popularity over a fixed random order), ``hot`` (every route to rank 0's experts).

MegaMoEV2 is set up as vLLM's DeepseekV4AiterMegaMoEExperts does: a fixed-slot instance for up to
``prev_power_of_2(FIXED_SLOT_MAX_MTPR)`` tokens and one of ``--megamoe-max-tokens`` capacity, the smallest that
holds the call. When MegaMoEV2 accepts ``routed=True`` (fused small- and large-input routes), the instances are
built with it and every call passes ``max_tokens`` (the tokens per rank of the call).
"""

import argparse
import inspect
import json
import os

import torch
import torch.distributed as dist

H, I, EPR, TOPK, WORLD = 5120, 2304, 48, 6, 8
E = EPR * WORLD


def make_weights(rank, dev):
    """Random MXFP4 bytes and e8m0 exponents 120..124 in the MegaMoEV2 / FusedEPMoE layouts."""
    from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4

    g = torch.Generator(device=dev).manual_seed(9000 + rank)

    def rnd(*shape):
        return torch.randint(0, 256, shape, dtype=torch.uint8, device=dev, generator=g)

    def scl(*shape):
        return torch.randint(
            120, 125, shape, dtype=torch.uint8, device=dev, generator=g
        )

    w1 = shuffle_weight_a16w4(rnd(EPR, 2 * I, H // 2), 16, True).contiguous()
    w1s = shuffle_scale_a16w4(
        scl(EPR, 2 * I, H // 32).view(-1, H // 32), EPR, True
    ).contiguous()
    w2 = shuffle_weight_a16w4(rnd(EPR, H, I // 2), 16, False).contiguous()
    w2s = shuffle_scale_a16w4(
        scl(EPR, H, I // 32).view(-1, I // 32), EPR, False
    ).contiguous()
    return w1, w1s, w2, w2s


def make_inputs(n_tokens, route, rank, dev, n, seed=5000):
    xs, ws, ids = [], [], []
    for i in range(n):
        g = torch.Generator(device=dev).manual_seed(seed + 1000 * rank + i)
        x = torch.randn(n_tokens, H, dtype=torch.bfloat16, device=dev, generator=g)
        s = torch.randn(n_tokens, E, dtype=torch.float32, device=dev, generator=g)
        if route == "zipf":
            perm = torch.randperm(
                E, device=dev, generator=torch.Generator(device=dev).manual_seed(4242)
            )
            logp = torch.empty(E, device=dev)
            logp[perm] = -1.2 * torch.log(torch.arange(1, E + 1, device=dev).float())
            u = torch.rand(n_tokens, E, device=dev, generator=g).clamp_(1e-9, 1 - 1e-9)
            s = logp - torch.log(-torch.log(u))
        elif route == "hot":
            s[:, EPR:] -= 1e4
        v, idx = torch.topk(s, TOPK, dim=-1)
        xs.append(x)
        ws.append(torch.softmax(v, -1).contiguous())
        ids.append(idx.to(torch.int32).contiguous())
    return xs, ws, ids


class MegaMoE:
    """MegaMoEV2 instances as vLLM builds them."""

    def __init__(self, rank, w, max_tokens):
        from aiter.ops.flydsl.kernels.mega_moe import MegaMoEV2
        from aiter.ops.flydsl.kernels.mega_moe.mega_moe_config import (
            FIXED_SLOT_MAX_MTPR,
            build_mega_moe_bundle_plan,
        )

        self.routed = "routed" in inspect.signature(MegaMoEV2.__init__).parameters
        fixed = 1 << (FIXED_SLOT_MAX_MTPR.bit_length() - 1)
        caps = [max_tokens]
        plan = build_mega_moe_bundle_plan(
            fixed, experts_per_rank=EPR, model_dim=H, inter_dim=I, world_size=WORLD
        )
        if max_tokens > fixed and plan.fixed_slot_dispatch:
            caps.insert(0, fixed)
        extra = {"routed": True} if self.routed else {}
        self.instances = [
            (
                c,
                MegaMoEV2(
                    rank=rank,
                    world_size=WORLD,
                    model_dim=H,
                    inter_dim=I,
                    experts=E,
                    topk=TOPK,
                    quant="a8w4",
                    w1=w[0],
                    w1_scale=w[1],
                    w2=w[2],
                    w2_scale=w[3],
                    max_tok_per_rank=c,
                    swiglu_limit=10.0,
                    **extra,
                ),
            )
            for c in caps
        ]

    def __call__(self, x, tw, ids, max_tokens):
        rt = next(r for c, r in self.instances if max_tokens <= c)
        kw = {"max_tokens": max_tokens} if self.routed else {}
        return rt.forward(x, tw, ids, **kw)


def timed(graphs, reps, dev):
    torch.cuda.synchronize()
    dist.barrier()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(reps):
        for g in graphs:
            g.replay()
    end.record()
    torch.cuda.synchronize()
    us = torch.tensor(
        start.elapsed_time(end) * 1e3 / (reps * len(graphs)),
        dtype=torch.float64,
        device=dev,
    )
    per_rank = [torch.zeros_like(us) for _ in range(WORLD)]
    dist.all_gather(per_rank, us)
    return max(t.item() for t in per_rank)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "-t",
        "--tokens",
        type=int,
        nargs="*",
        default=[1, 16, 32, 64, 128, 192, 384, 512, 768],
    )
    parser.add_argument("-r", "--routes", nargs="*", default=["uniform", "zipf", "hot"])
    parser.add_argument(
        "-i",
        "--impls",
        nargs="*",
        default=["megamoe", "fused"],
        choices=["megamoe", "fused"],
    )
    parser.add_argument("--megamoe-max-tokens", type=int, default=8192)
    parser.add_argument("--ninputs", type=int, default=100)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    rank, world = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    assert world == WORLD
    local = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(local)
    dev = torch.device("cuda", local)
    dist.init_process_group("cpu:gloo,cuda:nccl", device_id=dev)
    w = make_weights(rank, dev)
    impls = {}
    if "megamoe" in args.impls:
        import mori.shmem as ms
        import torch._C._distributed_c10d as c10d

        c10d._register_process_group("default", dist.group.WORLD)
        ms.shmem_torch_process_group_init("default")
        impls["megamoe"] = MegaMoE(rank, w, args.megamoe_max_tokens)
    if "fused" in args.impls:
        from aiter.ops.flydsl.fused_ep_moe import FusedEPMoE, FusedEPMoEWeights

        fw = FusedEPMoEWeights(*w)
        fused = FusedEPMoE(
            rank=rank, world_size=WORLD, group=dist.group.WORLD, device=dev
        )
        fused.prepare(fw)
        impls["fused"] = lambda x, tw, ids, max_tokens: fused(x, None, fw, tw, ids)
    if rank == 0 and "megamoe" in impls:
        caps = [c for c, _ in impls["megamoe"].instances]
        print(
            f"MegaMoEV2 instances {caps}, routed={impls['megamoe'].routed}", flush=True
        )

    results = []
    for t in args.tokens:
        graphs = {}
        for route in args.routes:
            xs, ws, ids = make_inputs(t, route, rank, dev, args.ninputs)
            for name, fn in impls.items():
                torch.cuda.synchronize()
                dist.barrier()
                fn(xs[0], ws[0], ids[0], t)
                torch.cuda.synchronize()
                dist.barrier()
                gs = []
                for i in range(args.ninputs):
                    g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(g, stream=torch.cuda.Stream()):
                        fn(xs[i], ws[i], ids[i], t)
                    gs.append(g)
                torch.cuda.synchronize()
                dist.barrier()
                for g in gs[:10]:
                    g.replay()
                graphs[(name, route)] = (gs, (xs, ws, ids))
        keys = list(graphs)
        samples = {k: [] for k in keys}
        for k in keys + keys[::-1]:
            samples[k].append(timed(graphs[k][0], args.reps, dev))
        for name, route in keys:
            s = sorted(samples[(name, route)])
            rec = {"impl": name, "route": route, "tokens": t, "us_max": s, "us": s[0]}
            results.append(rec)
            if rank == 0:
                print(json.dumps(rec), flush=True)
        del graphs
        torch.cuda.synchronize()
        dist.barrier()

    if rank == 0:
        for route in args.routes:
            print(
                f"\n{route}: us per call (best of the A B B A samples, max over ranks)"
            )
            print(
                "| tokens/rank | "
                + " | ".join(impls)
                + (" | speedup |" if len(impls) == 2 else " |")
            )
            print("|---" * (len(impls) + 1 + (len(impls) == 2)) + "|")
            for t in args.tokens:
                us = {
                    r["impl"]: r["us"]
                    for r in results
                    if r["route"] == route and r["tokens"] == t
                }
                row = f"| {t} | " + " | ".join(f"{us[n]:.1f}" for n in impls)
                if len(impls) == 2:
                    row += f" | {us['megamoe'] / us['fused']:.2f}x"
                print(row + " |")
        if args.out:
            with open(args.out, "w") as f:
                json.dump(results, f, indent=1)
    torch.cuda.synchronize()
    dist.barrier()
    dist.destroy_process_group()
    os._exit(0)  # MegaMoEV2 / mori teardown is not needed for a benchmark


if __name__ == "__main__":
    main()
