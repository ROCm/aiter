# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Fused vs three-launch MoonEP planner, plus knob sweeps for the fused one.

The fused kernel has no stage boundaries to time, so its phases are attributed
by sweeping a knob that only feeds one of them:

* ``prefetch_slots`` only feeds the top-B selection and the E+B layout scan.
* ``num_vblocks`` only feeds the histogram and the vblock prefix.
* ``blocks`` changes every grid-parallel phase and the grid-barrier width.

Routing is swept too: the balancing greedy exits early, so a skewed workload
does strictly more rounds than a uniform one.
"""

import os
import time

import torch

from aiter.ops.flydsl.moonep import (
    MoonEPFusedPlanner,
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


def _routing(device, *, skew, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    topk_all = []
    for _ in range(R):
        scores = torch.rand(S, E, generator=g)
        if skew:
            scores[:, : E // 4] += 4.0
        topk_all.append(scores.topk(K, dim=1).indices.to(torch.int32).to(device))
    tpe = torch.stack(
        [
            torch.bincount(t.reshape(-1).to(torch.int64), minlength=E).to(torch.int32)
            for t in topk_all
        ]
    ).to(device)
    return topk_all[0], tpe


def _time_us(fn):
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(ITERS):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - begin) * 1e6 / ITERS


def _config(slots=None):
    return MoonEPPlanConfig(
        rank=0,
        world_size=R,
        num_tokens=S,
        top_k=K,
        num_experts=E,
        prefetch_slots=slots,
        token_padding=TOKEN_PADDING,
    )


def main() -> int:
    device = torch.device("cuda")
    rows = []

    for skew in (False, True):
        topk, tpe = _routing(device, skew=skew)
        cfg = _config()
        tag = "skew" if skew else "uniform"

        multi = MoonEPGpuPlanner(cfg, device)
        multi.build(topk, tpe)
        rows.append((f"{tag:8s} multi (3 launches)", _time_us(lambda: multi.build(topk, tpe))))

        for blocks in (int(x) for x in os.environ.get("MOONEP_BLOCKS", "16,32,64,128,256").split(",")):
            fused = MoonEPFusedPlanner(cfg, device, blocks=blocks)
            fused.build(topk, tpe)
            rows.append(
                (f"{tag:8s} fused blocks={blocks}", _time_us(lambda: fused.build(topk, tpe)))
            )

        for nv in (int(x) for x in os.environ.get("MOONEP_NV", "32,64,128,256").split(",")):
            fused = MoonEPFusedPlanner(cfg, device, num_vblocks=nv)
            fused.build(topk, tpe)
            rows.append(
                (f"{tag:8s} fused NV={nv}", _time_us(lambda: fused.build(topk, tpe)))
            )

        for slots in (int(x) for x in os.environ.get("MOONEP_B", "4,12,24,48").split(",")):
            cfg_b = _config(slots)
            fused = MoonEPFusedPlanner(cfg_b, device)
            fused.build(topk, tpe)
            rows.append(
                (f"{tag:8s} fused B={slots}", _time_us(lambda: fused.build(topk, tpe)))
            )

    begin = time.perf_counter()
    build_reference_plan(_config(), *_routing(device, skew=False))
    host_ms = (time.perf_counter() - begin) * 1e3

    print(f"shape S={S} K={K} E={E} R={R} B=48 tp={TOKEN_PADDING}")
    print(f"host build_reference_plan {host_ms:.1f} ms")
    for label, us in rows:
        print(f"  {label:<28} {us:8.1f} us   ({host_ms * 1e3 / us:.0f}x vs host)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
