# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Per-stage cost breakdown of the FlyDSL MoonEP planner.

Answers the question the three-launch split raises: how much of the planner's
wall time is the kernels themselves and how much is the two extra launches.
``stage_only`` re-runs a single stage in a loop, so the difference between the
sum of the stages and the fused three-launch loop is the launch/ordering cost.
"""

import os
import time

import torch

from aiter.ops.flydsl.moonep import (
    MoonEPGpuPlanner,
    MoonEPPlanConfig,
    build_reference_plan,
)

R = 8
S = 8192
K = 8
E = 384
TOKEN_PADDING = 128
ITERS = 200
WARMUP = 20


def _routing(device: torch.device, seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    topk_all = [
        torch.rand(S, E, generator=generator).topk(K, dim=1).indices.to(torch.int32).to(device)
        for _ in range(R)
    ]
    tpe = torch.stack(
        [
            torch.bincount(t.reshape(-1).to(torch.int64), minlength=E).to(torch.int32)
            for t in topk_all
        ]
    ).to(device)
    return topk_all, tpe


def _time_ms(fn, iters: int = ITERS, warmup: int = WARMUP) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1e3 / iters


def _bench_one(config, topk, tpe, device, num_vblocks: int, host_ms: float, tag: str = "") -> None:
    planner = MoonEPGpuPlanner(config, device, num_vblocks=num_vblocks)
    planner.build(topk, tpe)  # compile all three stages

    total_ms = _time_ms(lambda: planner.build(topk, tpe))

    stage_ms = []
    for slot, name in enumerate(MoonEPGpuPlanner.STAGES):
        # Every stage is idempotent given the previous stages' outputs, except
        # stage 1, which turns local_hist counts into their exclusive prefix in
        # place.  Re-seed stage 0 before timing stage 1 so the loop is honest.
        if slot == 1:
            def run_meta(slot=slot):
                planner.run_stage(0, topk, tpe)
                planner.run_stage(1, topk, tpe)

            with_prefix = _time_ms(run_meta)
            stage_ms.append(with_prefix - stage_ms[0])
        else:
            stage_ms.append(_time_ms(lambda slot=slot: planner.run_stage(slot, topk, tpe)))

    geo = planner.geo
    label = tag or f"NV={geo.NV:<4} EPV={geo.EPV:<5} hist_blocks={geo.hist_blocks:<4}"
    print(
        f"  {label:<38}"
        f"| order_hist {stage_ms[0] * 1e3:7.1f}  meta {stage_ms[1] * 1e3:7.1f}  "
        f"dst {stage_ms[2] * 1e3:7.1f}  | total {total_ms * 1e3:7.1f} us  "
        f"({host_ms / total_ms:.0f}x vs host)"
    )


def main() -> int:
    device = torch.device("cuda")
    topk_all, tpe = _routing(device)
    config = MoonEPPlanConfig(
        rank=0,
        world_size=R,
        num_tokens=S,
        top_k=K,
        num_experts=E,
        token_padding=TOKEN_PADDING,
    )
    topk = topk_all[0]

    begin = time.perf_counter()
    build_reference_plan(config, topk, tpe)
    host_ms = (time.perf_counter() - begin) * 1e3

    print(f"shape S={S} K={K} E={E} R={R} B={config.prefetch_slots} tp={TOKEN_PADDING}")
    print(f"host build_reference_plan {host_ms:.1f} ms")
    print("-- num_vblocks sweep (isolates the histogram / vblock-prefix work) --")
    sweep = os.environ.get("MOONEP_PLAN_NV_SWEEP", "16,32,64,128,256")
    for nv in (int(x) for x in sweep.split(",")):
        _bench_one(config, topk, tpe, device, nv, host_ms)

    # prefetch_slots only feeds the top-B selection loop and the E+B layout
    # scan, so sweeping it attributes meta's NV-independent floor.
    print("-- prefetch_slots sweep at NV=128 (isolates top-B + layout) --")
    for slots in (int(x) for x in os.environ.get("MOONEP_PLAN_B_SWEEP", "4,12,24,48").split(",")):
        cfg = MoonEPPlanConfig(
            rank=0,
            world_size=R,
            num_tokens=S,
            top_k=K,
            num_experts=E,
            prefetch_slots=slots,
            token_padding=TOKEN_PADDING,
        )
        _bench_one(cfg, topk, tpe, device, 128, host_ms, tag=f"B={slots:<4} G={E + slots:<5}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
