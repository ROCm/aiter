#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Performance comparison for FlyDSL biased grouped_topk.

Standalone routing-kernel latency (FlyDSL vs HIP) across a range of token counts.

Usage:
    python3 op_tests/perf_flydsl_biased_grouped_topk.py
"""

from __future__ import annotations

import argparse

import torch

from aiter import dtypes
from aiter.test_common import run_perftest
from aiter.ops.topk import biased_grouped_topk_hip
from aiter.ops.flydsl import flydsl_biased_grouped_topk

# (num_tokens, num_experts, num_expert_group, topk_group, topk)
ROUTING_CASES = [
    {"token": 1, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 32, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 64, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 256, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 1024, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 4096, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
    {"token": 8192, "expert": 256, "group": 8, "topk_group": 4, "topk": 8},
]


def _alloc_out(token, topk):
    w = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.fp32, device="cuda")
    i = torch.empty_strided((token, topk), (topk + 10, 1), dtype=dtypes.i32, device="cuda")
    return w, i


def bench_routing(dtype, need_renorm=True, scale=1.0):
    print(
        f"\n=== biased routing latency  dtype={str(dtype).split('.')[-1]} "
        f"renorm={need_renorm} ==="
    )
    print(f"{'shape':<34}{'HIP (us)':>12}{'FlyDSL (us)':>14}{'speedup':>10}")
    for c in ROUTING_CASES:
        token, E, G, TG, K = c["token"], c["expert"], c["group"], c["topk_group"], c["topk"]
        gating = torch.randn((token, E), dtype=dtype, device="cuda")
        bias = torch.randn((E,), dtype=dtype, device="cuda") * 0.3

        w_hip, id_hip = _alloc_out(token, K)
        _, us_hip = run_perftest(
            biased_grouped_topk_hip,
            gating, bias, w_hip, id_hip, G, TG, need_renorm, scale,
            num_iters=100, num_warmup=10,
        )

        w_fly, id_fly = _alloc_out(token, K)
        _, us_fly = run_perftest(
            flydsl_biased_grouped_topk,
            gating, bias, w_fly, id_fly, G, TG, need_renorm, scale,
            num_iters=100, num_warmup=10,
        )
        speedup = us_hip / us_fly if us_fly > 0 else float("nan")
        label = f"t{token}_e{E}_g{G}_tg{TG}_k{K}"
        print(f"{label:<34}{us_hip:>12.3f}{us_fly:>14.3f}{speedup:>9.2f}x")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    args = ap.parse_args()
    dmap = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dmap[args.dtype]

    if not torch.cuda.is_available():
        print("CUDA not available; abort.")
        return

    bench_routing(dtype, need_renorm=True)
    bench_routing(dtype, need_renorm=False)


if __name__ == "__main__":
    main()
