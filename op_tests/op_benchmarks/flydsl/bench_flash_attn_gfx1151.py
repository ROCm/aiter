# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Benchmark FlyDSL FlashAttention against SDPA (AOTriton) on gfx1151.

Shapes are Qwen-Image 2.1 cross-attention (seq_len_k = image + text tokens).
Layouts: ``contiguous`` BSHD, or ``packed`` views of one ``[B, S, 3, H, D]``
buffer. SDPA is timed twice: ``sdpa_view`` (zero-copy transpose) and
``sdpa_bhsd`` (pre-copied contiguous, copy not timed; the speedup baseline).
Accuracy is relative RMSE against an FP32 SDPA reference.

    TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1 \
        python -m op_tests.op_benchmarks.flydsl.bench_flash_attn_gfx1151
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from aiter.ops.flydsl import (
    flydsl_flash_attn_func_gfx1151,
    flydsl_flash_attn_gfx1151_supported,
)

# Qwen-Image 2.1 at 512/1024/1280 px.
DEFAULT_SHAPES = ((1024, 1152), (3136, 3267), (4096, 4607))
DEFAULT_HEADS = 32
DEFAULT_HEAD_DIM = 128


def make_inputs(batch, seq_len_q, seq_len_k, num_heads, head_dim, layout, seed=0):
    """Build one BSHD triple, either separately packed or as packed-QKV views."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if layout == "packed":
        qkv = torch.randn(
            (batch, max(seq_len_q, seq_len_k), 3, num_heads, head_dim),
            generator=generator,
            dtype=torch.bfloat16,
            device="cuda",
        )
        return qkv[:, :seq_len_q, 0], qkv[:, :seq_len_k, 1], qkv[:, :seq_len_k, 2]
    shape_q = (batch, seq_len_q, num_heads, head_dim)
    shape_kv = (batch, seq_len_k, num_heads, head_dim)
    return (
        torch.randn(shape_q, generator=generator, dtype=torch.bfloat16, device="cuda"),
        torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16, device="cuda"),
        torch.randn(shape_kv, generator=generator, dtype=torch.bfloat16, device="cuda"),
    )


def to_bhsd(tensor):
    return tensor.transpose(1, 2).contiguous()


def sdpa_bshd(q, k, v):
    """SDPA with BSHD in and out, layout conversion included."""
    out = F.scaled_dot_product_attention(to_bhsd(q), to_bhsd(k), to_bhsd(v))
    return out.transpose(1, 2).contiguous()


def make_candidates(q, k, v, out):
    """Timed callables, in their own scope so closures bind this shape."""
    q_view, k_view, v_view = (
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
    )
    q_bhsd, k_bhsd, v_bhsd = to_bhsd(q), to_bhsd(k), to_bhsd(v)
    return {
        "flydsl": lambda: flydsl_flash_attn_func_gfx1151(q, k, v, out=out),
        "sdpa_view": lambda: F.scaled_dot_product_attention(
            q_view, k_view, v_view
        ).transpose(1, 2),
        "sdpa_bhsd": lambda: F.scaled_dot_product_attention(
            q_bhsd, k_bhsd, v_bhsd
        ).transpose(1, 2),
    }


def rel_rmse(got, reference):
    got = got.float()
    return (
        (got - reference).pow(2).mean().sqrt() / reference.pow(2).mean().sqrt()
    ).item()


def time_us(fn, iters, trials, warmup):
    """Per-iteration GPU latency samples, in microseconds."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1e3 / iters)
    return np.array(samples)


def attention_flops(num_heads, seq_len_q, seq_len_k, head_dim, batch):
    """QK^T and PV, two flops per multiply-accumulate."""
    return 4.0 * batch * num_heads * seq_len_q * seq_len_k * head_dim


def parse_args():
    parser = argparse.ArgumentParser(
        description="FlyDSL vs SDPA/AOTriton FlashAttention forward on gfx1151"
    )
    parser.add_argument(
        "--shapes",
        default=",".join(f"{sq}:{sk}" for sq, sk in DEFAULT_SHAPES),
        help="comma-separated seq_len_q:seq_len_k pairs",
    )
    parser.add_argument("-b", "--batch", type=int, default=1)
    parser.add_argument("-nh", "--num-heads", type=int, default=DEFAULT_HEADS)
    parser.add_argument("-d", "--head-dim", type=int, default=DEFAULT_HEAD_DIM)
    parser.add_argument(
        "--layout",
        choices=("contiguous", "packed", "both"),
        default="both",
    )
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--trials", type=int, default=11)
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="untimed iterations; also absorbs the FlyDSL JIT compile",
    )
    parser.add_argument(
        "--no-check", action="store_true", help="skip the fp32 SDPA reference"
    )
    parser.add_argument("-o", "--output", help="write the table to this CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    if not flydsl_flash_attn_gfx1151_supported(torch.device("cuda"), args.head_dim):
        raise SystemExit(
            "this benchmark needs a gfx1151 wave32 device and a head_dim that is "
            "a multiple of 16"
        )

    shapes = [
        tuple(int(part) for part in shape.split(":"))
        for shape in args.shapes.split(",")
    ]
    layouts = ("contiguous", "packed") if args.layout == "both" else (args.layout,)

    rows = []
    for seq_len_q, seq_len_k in shapes:
        flops = attention_flops(
            args.num_heads, seq_len_q, seq_len_k, args.head_dim, args.batch
        )
        for layout in layouts:
            q, k, v = make_inputs(
                args.batch,
                seq_len_q,
                seq_len_k,
                args.num_heads,
                args.head_dim,
                layout,
            )
            out = torch.empty(
                (args.batch, seq_len_q, args.num_heads, args.head_dim),
                dtype=torch.bfloat16,
                device="cuda",
            )

            reference = None
            if not args.no_check:
                reference = sdpa_bshd(q.float(), k.float(), v.float())

            candidates = make_candidates(q, k, v, out)
            baseline_us = None
            for name, fn in candidates.items():
                try:
                    produced = fn()
                except RuntimeError as error:
                    rows.append(
                        {
                            "shape": f"{seq_len_q}:{seq_len_k}",
                            "layout": layout,
                            "backend": name,
                            "note": str(error).splitlines()[0][:60],
                        }
                    )
                    continue
                torch.cuda.synchronize()
                samples = time_us(fn, args.iters, args.trials, args.warmup)
                median = float(np.median(samples))
                if name == "sdpa_bhsd":
                    baseline_us = median
                rows.append(
                    {
                        "shape": f"{seq_len_q}:{seq_len_k}",
                        "layout": layout,
                        "backend": name,
                        "p20_us": round(float(np.percentile(samples, 20)), 1),
                        "median_us": round(median, 1),
                        "p80_us": round(float(np.percentile(samples, 80)), 1),
                        "tflops": round(flops / median / 1e6, 2),
                        "rel_rmse": (
                            None
                            if reference is None
                            else float(f"{rel_rmse(produced, reference):.3g}")
                        ),
                    }
                )

            if baseline_us is not None:
                for row in rows:
                    if (
                        row["shape"] == f"{seq_len_q}:{seq_len_k}"
                        and row["layout"] == layout
                        and "median_us" in row
                    ):
                        row["vs_sdpa_bhsd"] = round(baseline_us / row["median_us"], 3)

    table = pd.DataFrame(rows)
    print(table.to_string(index=False, na_rep="-"))
    if args.output:
        table.to_csv(args.output, index=False)
        print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
