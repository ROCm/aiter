# SPDX-License-Identifier: MIT
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Bit-exact check of the FlyDSL MoonEP planner against the host reference.

MoonEP planning needs no cross-rank traffic once ``tokens_per_expert`` has been
all-gathered, so every rank can be validated in a single process: the loop below
walks all ``world_size`` ranks over the same routing tables.

Run directly::

    python op_tests/test_moonep_planning_gpu.py            # small + real shape
    MOONEP_PLAN_SMALL_ONLY=1 python op_tests/test_moonep_planning_gpu.py
"""

from __future__ import annotations

import os
import time

import pytest
import torch

from aiter.ops.flydsl.moonep import (
    MoonEPFusedPlanner,
    MoonEPGpuPlanner,
    MoonEPPlanConfig,
    build_reference_plan,
)

# Both planners are validated against the same golden.  "multi" is the
# three-launch baseline; "fused" is the single-kernel latency-tuned one.
PLANNERS = {"multi": MoonEPGpuPlanner, "fused": MoonEPFusedPlanner}

PLAN_FIELDS = (
    "dst",
    "cu_seqlens",
    "experts_to_copy",
    "zero_fill_ranges",
    "remote_stats",
    "alloc",
    "group_expert_ids",
)

# The production shape the 8k forward profile uses.
REAL_SHAPE = dict(world_size=8, num_tokens=8192, top_k=8, num_experts=384, token_padding=128)
SMALL_SHAPE = dict(world_size=4, num_tokens=256, top_k=4, num_experts=32, token_padding=16)


def _routing(shape: dict, device: torch.device, *, skew: bool, seed: int = 0):
    """Build ``topk_experts`` per rank plus the all-gathered histogram.

    Real top-k routing never picks the same expert twice for one token, so each
    token's K entries are drawn without replacement.  ``skew`` concentrates the
    draw on a quarter of the experts to force heavy migration in the balancer.
    """

    R = shape["world_size"]
    S = shape["num_tokens"]
    K = shape["top_k"]
    E = shape["num_experts"]
    generator = torch.Generator(device="cpu").manual_seed(seed)

    topk_all = []
    for rank in range(R):
        if skew:
            hot = max(K, E // 4)
            scores = torch.rand(S, E, generator=generator)
            scores[:, :hot] += 4.0
        else:
            scores = torch.rand(S, E, generator=generator)
        picks = scores.topk(K, dim=1).indices.to(torch.int32)
        topk_all.append(picks.to(device))

    tpe = torch.stack(
        [
            torch.bincount(t.reshape(-1).to(torch.int64), minlength=E).to(torch.int32)
            for t in topk_all
        ]
    ).to(device)
    return topk_all, tpe


def _compare(reference, actual, label: str) -> list[str]:
    errors = []
    for field in PLAN_FIELDS:
        want = getattr(reference, field).to(torch.int32).cpu()
        got = getattr(actual, field).to(torch.int32).cpu()
        if want.shape != got.shape:
            errors.append(f"{label}: {field} shape {tuple(got.shape)} != {tuple(want.shape)}")
            continue
        if torch.equal(want, got):
            continue
        bad = (want != got).nonzero()
        first = tuple(bad[0].tolist())
        errors.append(
            f"{label}: {field} differs at {bad.shape[0]}/{want.numel()} slots; "
            f"first {first} want={want[first].item()} got={got[first].item()}"
        )
    return errors


def _run_shape(shape: dict, *, skew: bool, device: torch.device, timed: bool,
               variant: str = "multi"):
    topk_all, tpe = _routing(shape, device, skew=skew)
    label = (f"{variant} {'skew' if skew else 'uniform'} "
             f"S={shape['num_tokens']} E={shape['num_experts']}")
    errors: list[str] = []

    for rank in range(shape["world_size"]):
        config = MoonEPPlanConfig(rank=rank, **shape)
        reference = build_reference_plan(config, topk_all[rank], tpe)
        planner = PLANNERS[variant](config, device)
        actual = planner.build(topk_all[rank], tpe).clone()
        errors.extend(_compare(reference, actual, f"{label} rank={rank}"))

        if timed and rank == 0:
            _report_timing(config, topk_all[rank], tpe, planner, label)

    return errors


def _report_timing(config, topk, tpe, planner, label: str) -> None:
    for _ in range(3):
        planner.build(topk, tpe)
    torch.cuda.synchronize()

    iters = 20
    begin = time.perf_counter()
    for _ in range(iters):
        planner.build(topk, tpe)
    torch.cuda.synchronize()
    gpu_ms = (time.perf_counter() - begin) * 1e3 / iters

    begin = time.perf_counter()
    build_reference_plan(config, topk, tpe)
    host_ms = (time.perf_counter() - begin) * 1e3

    print(
        f"[timing] {label}: gpu={gpu_ms:.3f} ms  host_reference={host_ms:.1f} ms  "
        f"speedup={host_ms / max(gpu_ms, 1e-9):.0f}x"
    )


@pytest.mark.parametrize("variant", ["multi", "fused"])
@pytest.mark.parametrize("skew", [False, True])
def test_plan_gpu_matches_reference_small(skew: bool, variant: str):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    errors = _run_shape(
        SMALL_SHAPE, skew=skew, device=torch.device("cuda"), timed=False,
        variant=variant,
    )
    assert not errors, "\n".join(errors)


@pytest.mark.parametrize("variant", ["multi", "fused"])
@pytest.mark.parametrize("skew", [False, True])
def test_plan_gpu_matches_reference_real_shape(skew: bool, variant: str):
    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    errors = _run_shape(REAL_SHAPE, skew=skew, device=torch.device("cuda"),
                        timed=False, variant=variant)
    assert not errors, "\n".join(errors)


def main() -> int:
    device = torch.device("cuda")
    failures = 0
    shapes = [("small", SMALL_SHAPE)]
    if not int(os.environ.get("MOONEP_PLAN_SMALL_ONLY", "0")):
        shapes.append(("real", REAL_SHAPE))

    for name, shape in shapes:
        for variant in ("multi", "fused"):
          for skew in (False, True):
            errors = _run_shape(shape, skew=skew, device=device,
                                timed=(name == "real"), variant=variant)
            tag = f"{name}/{variant}/{'skew' if skew else 'uniform'}"
            if errors:
                failures += 1
                print(f"FAIL {tag}")
                for line in errors[:12]:
                    print("  " + line)
                if len(errors) > 12:
                    print(f"  ... {len(errors) - 12} more")
            else:
                print(f"PASS {tag}")

    print("MOONEP_PLAN_GPU " + ("FAIL" if failures else "PASS"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
