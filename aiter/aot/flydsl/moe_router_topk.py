#!/usr/bin/env python3

# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""AOT for the gfx1250 MoE router top-k: each fused row of the tuned table,
replayed through ``flydsl_moe_router_topk`` (``replay.py``) with and without
expert bias and renorm, which are compiled in."""

from __future__ import annotations

import csv
import itertools
import os

from aiter.aot.flydsl import replay as _replay
from aiter.jit.core import AITER_CONFIGS

LABEL = "MOE_ROUTER_TOPK"
DEFAULT_CSVS = [AITER_CONFIGS.AITER_CONFIG_MOE_ROUTER_TOPK_FILE]
AOT_ARCH_DEFAULT = "gfx1250"
_SHAPE = ("cu_num", "M", "N", "K", "topk")


def parse_csv(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["fused"] == "1"]
    return [
        {
            "kernel_name": "moe_router_topk",
            "gfx": r["gfx"],
            **{c: int(r[c]) for c in _SHAPE},
            "has_bias": has_bias,
            "need_renorm": need_renorm,
        }
        for r in rows
        for has_bias, need_renorm in itertools.product((False, True), repeat=2)
    ]


def replay(job: dict) -> None:
    import torch

    from aiter.ops.flydsl import moe_front_gfx1250 as front

    # Rows of several cu_num can share a worker.
    front.router_topk_config.cache_clear()
    m, n, k, topk = job["M"], job["N"], job["K"], job["topk"]

    def empty(*shape, dtype=torch.bfloat16):
        return torch.empty(shape, dtype=dtype, device="cuda")

    args = (
        empty(m, k),
        empty(n, k),
        topk,
        empty(n, dtype=torch.float32) if job["has_bias"] else None,
        job["need_renorm"],
    )
    nvr = empty(1, dtype=torch.int32)
    if front.flydsl_moe_router_topk_supported(*args, num_valid_rows=nvr):
        front.flydsl_moe_router_topk(*args, num_valid_rows=nvr)


def describe(job: dict) -> str:
    shape = " ".join(f"{c}={job[c]}" for c in _SHAPE[1:])
    return f"moe_router_topk {shape} bias={job['has_bias']} renorm={job['need_renorm']}"


def config_env(csv_paths: list[str]) -> dict[str, str]:
    return {"AITER_CONFIG_MOE_ROUTER_TOPK": os.pathsep.join(csv_paths)}


def main() -> None:
    _replay.main("aiter.aot.flydsl.moe_router_topk", __doc__)


if __name__ == "__main__":
    main()
