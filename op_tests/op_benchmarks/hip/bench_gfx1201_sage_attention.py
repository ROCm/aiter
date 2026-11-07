# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark gfx1201_sage_attention against BF16 SDPA and Triton fav3_sage on gfx1201.

Providers run in interleaved rounds on the same inputs; the reported time is the
median over rounds of the per-round mean. With --compare_to_ref every provider is
also checked against an FP32 reference (computed in query chunks).
"""

import argparse
import csv
import math
import statistics

import torch
import torch.nn.functional as F

import aiter
from aiter.jit.utils.chip_info import get_gfx

DEFAULT_SHAPES = (
    "1,113102,28",  # MiniMax-H3 video DiT (768x1344), 28 local heads
    "1,75600,5",  # shapes from the Triton fav3_sage PR benchmark (#1818)
    "1,16452,24",
    "1,118808,3",
    "2,29760,2",
    "1,32760,24",
    "1,4096,24",
)


def parse_shape(text):
    batch, rows, heads = (int(x) for x in text.split(","))
    return batch, rows, heads


def make_providers(names):
    providers = {}
    if "sage" in names:
        providers["sage"] = lambda q, k, v: torch.ops.aiter.gfx1201_sage_attention(
            q, k, v
        )
    if "sage_core" in names:
        cache = {}

        def core(q, k, v):
            key = (q.data_ptr(), q.shape)
            if key not in cache:
                cache.clear()
                cache[key] = aiter.gfx1201_sage_prepare(q, k, v)
            return aiter.gfx1201_sage_attention_fwd(*cache[key], q.shape[1])[
                :, : q.shape[1]
            ]

        providers["sage_core"] = core
    if "sdpa" in names:
        providers["sdpa"] = lambda q, k, v: F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)
    if "triton_sage" in names:
        from aiter.ops.triton.attention.fav3_sage import fav3_sage_wrapper_func

        def triton_sage(q, k, v):
            out = fav3_sage_wrapper_func(
                q,
                k,
                v,
                q.shape[-1] ** -0.5,
                causal=False,
                return_lse=False,
                layout="bshd",
            )
            return out[0] if isinstance(out, (tuple, list)) else out

        providers["triton_sage"] = triton_sage
    return providers


def time_ms(fn, args, warmup, iters):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    start.record()
    for _ in range(iters):
        fn(*args)
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / iters


@torch.no_grad()
def reference(q, k, v, chunk=4096):
    out = torch.empty(q.shape, device=q.device, dtype=torch.float32)
    scale = q.shape[-1] ** -0.5
    for b in range(q.shape[0]):
        for h in range(q.shape[2]):
            kh, vh = k[b, :, h].float(), v[b, :, h].float()
            for i in range(0, q.shape[1], chunk):
                s = (q[b, i : i + chunk, h].float() @ kh.T) * scale
                out[b, i : i + chunk, h] = torch.softmax(s, dim=-1) @ vh
    return out


def accuracy(out, ref):
    out = out.float()
    abs_diff = (out - ref).abs()
    rel_diff = abs_diff / ref.clamp(min=1e-6).abs()
    row_cos = F.cosine_similarity(out, ref, dim=-1).flatten()
    return {
        "fail%": ((abs_diff > 0.3) & (rel_diff > 0.25)).float().mean().item() * 100,
        "mismatch@1e-2%": (~torch.isclose(out, ref, rtol=1e-2, atol=1e-2))
        .float()
        .mean()
        .item()
        * 100,
        "max_abs": abs_diff.max().item(),
        "cos": F.cosine_similarity(out.flatten(), ref.flatten(), dim=0).item(),
        "row_cos_median": row_cos.median().item(),
        "rel_l2": ((out - ref).norm() / ref.norm()).item(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--shapes", nargs="+", default=DEFAULT_SHAPES, help="B,S,H (head dim is 128)"
    )
    parser.add_argument(
        "--providers", nargs="+", default=["sage", "sage_core", "sdpa", "triton_sage"]
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--scale", type=float, default=1.0, help="std of the random Q/K/V"
    )
    parser.add_argument("--compare_to_ref", action="store_true")
    parser.add_argument("--csv", default=None)
    args = parser.parse_args()
    if get_gfx() != "gfx1201":
        print(f"skip: requires gfx1201, got {get_gfx()}")
        return

    providers = make_providers(args.providers)
    rows = []
    for shape in args.shapes:
        batch, seq, heads = parse_shape(shape)
        torch.manual_seed(0)
        q, k, v = (
            torch.randn(batch, seq, heads, 128, device="cuda") * args.scale
            for _ in range(3)
        )
        q, k, v = q.bfloat16(), k.bfloat16(), v.bfloat16()
        flops = 4 * batch * heads * seq * seq * 128
        times = {name: [] for name in providers}
        for _ in range(args.rounds):
            for name, fn in providers.items():
                if times[name] is None:
                    continue
                try:
                    times[name].append(time_ms(fn, (q, k, v), args.warmup, args.iters))
                except (RuntimeError, ValueError, AssertionError) as error:
                    print(f"{shape} {name}: {type(error).__name__}: {str(error)[:120]}")
                    times[name] = None
                    torch.cuda.empty_cache()
        ref = reference(q, k, v) if args.compare_to_ref else None
        for name, fn in providers.items():
            row = {"B": batch, "S": seq, "H": heads, "provider": name}
            if times[name]:
                ms = statistics.median(times[name])
                row.update({"ms": round(ms, 3), "TFLOPS": round(flops / ms / 1e9, 1)})
                if ref is not None:
                    row.update(
                        {
                            k_: round(v_, 5)
                            for k_, v_ in accuracy(fn(q, k, v), ref).items()
                        }
                    )
            rows.append(row)
            print(", ".join(f"{k_}={v_}" for k_, v_ in row.items()), flush=True)
        del ref
        torch.cuda.empty_cache()
    if args.csv:
        keys = sorted(
            {k_ for row in rows for k_ in row},
            key=lambda x: list(rows[0]).index(x) if x in rows[0] else math.inf,
        )
        with open(args.csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
