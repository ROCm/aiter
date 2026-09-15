# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time K3 latent FHMoE against production and reference split paths.

Answers one question: does the FHMoE advantage widen as the decode token count
grows?  conc-1 gives M=8 (DSpark K=7), conc-2 M=16, conc-4 M=32.

FlyDSL specializes runtime integer arguments in a process-local cache, so each M
must run in its own process.  Invoke as ``python3 k3_latent_fhmoe_m_sweep.py M``.

The production routed arm calls ``aiter.fused_moe`` with the same A8W4 SiTU
contract used by vLLM. The shared branch uses ``silu_and_mul`` as a same-shape
stand-in for vLLM's ``+situ_and_mul`` custom op, which lives in vLLM and not in
aiter. This is a timing harness only; correctness is covered by
``test_latent_fhmoe_gpu.py``.
"""

from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

import aiter
from aiter.fused_moe import fused_moe as aiter_fused_moe
from aiter.latent_fhmoe import latent_fhmoe
from aiter.ops.quant import per_1x32_f4_quant
from aiter.ops.shuffle import (
    shuffle_scale_a16w4,
    shuffle_weight,
    shuffle_weight_a16w4,
)

from test_latent_fhmoe_gpu import _routed_aiter_reference

EXPERTS = 896
TOPK = 16
ITERS = 20
WARMUP = 5


def _build(m: int, device: torch.device):
    torch.manual_seed(7 + m)

    routed_input = torch.randn((m, 3584), device=device, dtype=torch.bfloat16) * 0.02
    shared_input = torch.randn((m, 7168), device=device, dtype=torch.bfloat16) * 0.02

    raw_routed_w1 = (
        torch.randn((EXPERTS, 768, 3584), device=device, dtype=torch.bfloat16) * 0.01
    )
    raw_routed_w2 = (
        torch.randn((EXPERTS, 3584, 384), device=device, dtype=torch.bfloat16) * 0.01
    )
    routed_w1, routed_s1 = per_1x32_f4_quant(raw_routed_w1)
    routed_w2, routed_s2 = per_1x32_f4_quant(raw_routed_w2)
    routed_w1 = shuffle_weight_a16w4(routed_w1, 16, True)
    routed_w2 = shuffle_weight_a16w4(routed_w2, 16, False)
    routed_s1 = shuffle_scale_a16w4(
        routed_s1.view(-1, routed_s1.shape[-1]), EXPERTS, True
    ).reshape(EXPERTS, 768, 112)
    routed_s2 = shuffle_scale_a16w4(
        routed_s2.view(-1, routed_s2.shape[-1]), EXPERTS, False
    ).reshape(EXPERTS, 3584, 16)

    raw_shared_w1 = (
        torch.randn((1, 1536, 7168), device=device, dtype=torch.bfloat16) * 0.01
    )
    raw_shared_w2 = (
        torch.randn((1, 7168, 768), device=device, dtype=torch.bfloat16) * 0.01
    )
    shared_w1 = shuffle_weight(raw_shared_w1, layout=(16, 16))
    shared_w2 = shuffle_weight(raw_shared_w2, layout=(16, 16))

    # Spread the routes so every M sees a comparable expert-activation pattern
    # instead of M=8 hitting 128 experts and M=64 hitting all 896.
    if EXPERTS >= m * TOPK:
        topk_ids = torch.randperm(EXPERTS, dtype=torch.int32, device=device)[
            : m * TOPK
        ].reshape(m, TOPK)
    else:
        topk_ids = torch.randint(
            EXPERTS, (m, TOPK), dtype=torch.int32, device=device
        )
    topk_weight = torch.full(
        (m, TOPK), 1.0 / TOPK, dtype=torch.float32, device=device
    )

    return {
        "routed_input": routed_input,
        "shared_input": shared_input,
        "routed_w1": routed_w1,
        "routed_w2": routed_w2,
        "routed_s1": routed_s1,
        "routed_s2": routed_s2,
        "shared_w1": shared_w1,
        "shared_w2": shared_w2,
        "raw_shared_w1": raw_shared_w1,
        "raw_shared_w2": raw_shared_w2,
        "topk_ids": topk_ids,
        "topk_weight": topk_weight,
    }


def _time_eager(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / ITERS * 1000.0  # us


def _time_graph(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    graph.replay()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(ITERS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / ITERS * 1000.0  # us


def main() -> None:
    global ITERS
    parser = argparse.ArgumentParser()
    parser.add_argument("m", type=int)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument(
        "--arm",
        choices=(
            "all",
            "fhmoe",
            "hybrid",
            "conventional",
            "prod-routed",
            "shared",
        ),
        default="all",
    )
    parser.add_argument("--iters", type=int, default=ITERS)
    args = parser.parse_args()
    m = args.m
    device = torch.device("cuda")
    t = _build(m, device)

    def fhmoe():
        latent_fhmoe(
            t["routed_input"],
            t["routed_w1"],
            t["routed_w2"],
            t["routed_s1"],
            t["routed_s2"],
            t["topk_weight"],
            t["topk_ids"],
            t["shared_input"],
            t["shared_w1"],
            t["shared_w2"],
        )

    def conv_routed():
        _routed_aiter_reference(
            t["routed_input"],
            t["routed_w1"],
            t["routed_w2"],
            t["routed_s1"],
            t["routed_s2"],
            t["topk_weight"],
            t["topk_ids"],
        )

    def prod_routed():
        aiter_fused_moe(
            t["routed_input"],
            t["routed_w1"],
            t["routed_w2"],
            t["topk_weight"],
            t["topk_ids"],
            activation=aiter.ActivationType.Situv2,
            quant_type=aiter.QuantType.per_1x32,
            doweight_stage1=False,
            w1_scale=t["routed_s1"].view(-1, t["routed_s1"].shape[-1]),
            w2_scale=t["routed_s2"].view(-1, t["routed_s2"].shape[-1]),
            dtype=torch.bfloat16,
            beta=4.0,
            linear_beta=25.0,
            gate_mode="interleave",
        )

    inter = torch.empty((m, 768), device=device, dtype=torch.bfloat16)

    def conv_shared():
        gate_up = F.linear(t["shared_input"], t["raw_shared_w1"][0])
        aiter.silu_and_mul(inter, gate_up)
        F.linear(inter, t["raw_shared_w2"][0])

    def conventional():
        prod_routed()
        conv_shared()

    shared_stream = torch.cuda.Stream(device=device)
    shared_start = torch.cuda.Event()
    shared_done = torch.cuda.Event()

    def hybrid():
        main_stream = torch.cuda.current_stream()
        shared_start.record(main_stream)
        shared_stream.wait_event(shared_start)
        with torch.cuda.stream(shared_stream):
            conv_shared()
            shared_done.record(shared_stream)
        prod_routed()
        main_stream.wait_event(shared_done)

    ITERS = args.iters
    timer = _time_graph if args.graph else _time_eager
    arms = {
        "fhmoe": fhmoe,
        "hybrid": hybrid,
        "conventional": conventional,
        "prod-routed": prod_routed,
        "shared": conv_shared,
    }
    if args.arm != "all":
        elapsed_us = timer(arms[args.arm])
        print(
            json.dumps(
                {
                    "m": m,
                    "mode": "graph" if args.graph else "eager",
                    "arm": args.arm,
                    "us": round(elapsed_us, 2),
                }
            )
        )
        return

    fhmoe_us = timer(fhmoe)
    conv_us = timer(conventional)
    prod_routed_us = timer(prod_routed)
    ref_routed_us = timer(conv_routed)
    shared_us = timer(conv_shared)

    print(
        json.dumps(
            {
                "m": m,
                "mode": "graph" if args.graph else "eager",
                "fhmoe_us": round(fhmoe_us, 2),
                "conventional_us": round(conv_us, 2),
                "prod_routed_us": round(prod_routed_us, 2),
                "ref_routed_us": round(ref_routed_us, 2),
                "conv_shared_us": round(shared_us, 2),
                "speedup": round(conv_us / fhmoe_us, 4),
            }
        )
    )


if __name__ == "__main__":
    main()
