# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Isolated gfx942 FP8 unified-attention A/B baseline for Gemma-4-31B.

Tier A uses main's shipped JSON unchanged. Tier B uses the isolated config table
from PR #5650 head 184aae9498c1de3d6f60c0a2ba8b9338907ffc78, plus only the
D>=512 attn_3d decode waves_per_eu=1 from PR #5664 head
0743fc1f5f7ca1a50e6c36df43ed2a25d03818f9 (no #5664 prefill changes).
The #5650 shuffled-page >128 guard is active in both tiers; all measured pages
are <=128, so the guard does not change Tier A behavior for measured shapes.
This script runs one tier per process; it changes only the config loader's root
before any config lookup, leaving the production DEFAULT.json intact.
"""

import argparse
import csv
import itertools
import math
from pathlib import Path

import torch

from aiter.ops.triton.attention.unified_attention import unified_attention
from aiter.ops.triton.utils import config_utils
from aiter.ops.triton.utils._triton import arch_info
from aiter.test_common import run_perftest

ROOT = Path(__file__).resolve().parent
CACHE_TARGET_BYTES = 320 * 1024 * 1024
SMOKE = False


def measure(kind, batch, context, layout, mode, tier, iterations):
    dim, heads_kv, page_size, window = (
        (256, 16, 32, 1024) if kind == "sliding" else (512, 4, 64, 0)
    )
    q_len = context if mode == "prefill" else 1
    max_pages = math.ceil(context / page_size)
    # Plain and shuffled KV pages have identical logical shape and element count.
    q = torch.randn(batch * q_len, 32, dim, device="cuda").to(torch.float8_e4m3fnuz)
    k = torch.randn(batch * max_pages, page_size, heads_kv, dim, device="cuda").to(
        torch.float8_e4m3fnuz
    )
    v = torch.randn_like(k.float()).to(torch.float8_e4m3fnuz)
    if layout == "shuffled":
        width = 16
        k = k.view(-1, page_size, heads_kv, dim // width, width).permute(
            0, 2, 3, 1, 4
        ).contiguous()
        v = v.view(-1, page_size // width, width, heads_kv, dim).permute(
            0, 3, 1, 4, 2
        ).contiguous()
    out = torch.empty(batch * q_len, 32, dim, dtype=torch.bfloat16, device="cuda")
    cu_q = torch.arange(batch + 1, device="cuda", dtype=torch.int32) * q_len
    used_k = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    table = torch.arange(batch * max_pages, device="cuda", dtype=torch.int32).view(
        batch, max_pages
    )
    scales = tuple(torch.ones(1, device="cuda", dtype=torch.float32) for _ in range(3))

    def call(q, k, v, out, cu_q, used_k, table, q_scale, k_scale, v_scale):
        return unified_attention(
            q, k, v, out, cu_q, q_len, used_k, context, dim ** -0.5,
            True, (window - 1, 0) if window else (-1, -1), table, 0.0,
            q_scale, k_scale, v_scale,
            shuffled_kv_cache=(layout == "shuffled"), backend="triton",
        )

    positional = (q, k, v, out, cu_q, used_k, table, *scales)
    footprint = sum(t.numel() * t.element_size() for t in positional)
    copies = max(2, math.ceil(CACHE_TARGET_BYTES / footprint))
    if copies > iterations:
        raise ValueError(f"--iterations must be >= {copies} for cache defeat")
    _, latency_us = run_perftest(
        call, *positional, num_iters=iterations, num_rotate_args=copies
    )
    if not math.isfinite(latency_us) or latency_us <= 0:
        raise RuntimeError(f"invalid timing: {latency_us}")
    torch.cuda.synchronize()
    if not out.abs().max().item() > 0:
        raise RuntimeError("attention output is zero")
    if SMOKE:
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            call(*positional)
            torch.cuda.synchronize()
        kernels = [event.name for event in prof.events() if "unified_attention" in event.name]
        if not kernels:
            raise RuntimeError("no unified_attention Triton kernel dispatched")
        print(f"Triton dispatch: {kernels[0]}", flush=True)
    return {
        "tier": tier, "attention": kind, "mode": mode, "layout": layout,
        "batch": batch, "query_length": q_len, "context": context,
        "num_q_heads": 32, "num_kv_heads": heads_kv, "head_dim": dim,
        "window": window, "page_size": page_size, "rotations": copies,
        "footprint_bytes": footprint, "us": latency_us,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("A", "B"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", required=True, help="output CSV path")
    parser.add_argument("--iterations", type=int, default=101)
    args = parser.parse_args()
    global SMOKE
    SMOKE = args.smoke
    if arch_info.get_arch() != "gfx942":
        raise RuntimeError("gfx942 GPU required")
    if args.tier == "B":
        config_utils.AITER_TRITON_CONFIGS_PATH = str(ROOT / "benchmark_configs")
        config_utils.load_config_json.cache_clear()
    cases = (
        [("sliding", "prefill", 1, 1024), ("full", "decode", 4, 1024)]
        if args.smoke else
        [(kind, "prefill", 1, n) for kind, n in itertools.product(
            ("sliding", "full"), (1024, 4096, 16384, 32768)
        )] + [(kind, "decode", b, n) for kind, b, n in itertools.product(
            ("sliding", "full"), (1, 4, 16, 64), (1024, 4096, 16384, 32768)
        )]
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as file:
        writer = None
        for kind, mode, batch, context in cases:
            for layout in ("plain", "shuffled"):
                row = measure(kind, batch, context, layout, mode, args.tier, args.iterations)
                if writer is None:
                    writer = csv.DictWriter(file, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                file.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
