#!/usr/bin/env python3
"""Measure gfx1250 native MXFP8 through AITER's production entry point.

Torch is used only to build an FP32 correctness reference. Timing, logical
bytes, FLOPs and roofline efficiency come from the AITER kernel invocation.
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.shuffle import shuffle_mxfp8fp4_scale, shuffle_weight


MI455_MXFP8_TFLOPS = 20133.0
MI455_HBM_TB_S = 23.3


def make_data(m, n, k, seed):
    torch.manual_seed(seed)
    x = (torch.randn((m, k), device="cuda") * 2).to(dtypes.fp8)
    weight = (torch.randn((n, k), device="cuda") * 2).to(dtypes.fp8)
    xs_raw = torch.randint(124, 131, (m, k // 32), dtype=torch.uint8, device="cuda")
    ws_raw = torch.randint(124, 131, (n, k // 32), dtype=torch.uint8, device="cuda")
    xs = xs_raw.view(dtypes.fp8_e8m0)
    ws = ws_raw.view(dtypes.fp8_e8m0)
    ref = F.linear(
        x.float() * torch.exp2(xs_raw.float() - 127).repeat_interleave(32, 1),
        weight.float() * torch.exp2(ws_raw.float() - 127).repeat_interleave(32, 1),
    ).to(torch.bfloat16)
    return {
        "x": x,
        "w": shuffle_weight(weight, layout=(16, 16)),
        "xs": xs,
        "ws": shuffle_mxfp8fp4_scale(ws),
        "out": torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
        "ref": ref,
    }


def run(data):
    return aiter.gemm_a8w8_blockscale_bpreshuffle(
        data["x"],
        data["w"],
        data["xs"],
        data["ws"],
        dtypes.bf16,
        out=data["out"],
    )


def error_ratio(output, reference):
    return 1.0 - float(
        torch.isclose(
            output.float(), reference.float(), atol=0.01, rtol=0.01
        ).float().mean()
    )


def samples_us(data, warmup, reps, inner):
    for _ in range(warmup):
        run(data)
    torch.cuda.synchronize()
    values = []
    for _ in range(reps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(inner):
            run(data)
        end.record()
        end.synchronize()
        values.append(start.elapsed_time(end) * 1000.0 / inner)
    values.sort()
    return values


def logical_metrics(m, n, k, us):
    flops = 2 * m * n * k
    # E4M3 A/B + one E8M0 byte per 32 values + BF16 output.
    logical_bytes = (m * k + n * k) * (1 + 1 / 32) + 2 * m * n
    intensity = flops / logical_bytes
    tflops = flops / us / 1e6
    tb_s = logical_bytes / us / 1e6
    roof_tflops = min(MI455_MXFP8_TFLOPS, intensity * MI455_HBM_TB_S)
    return {
        "logical_flops": flops,
        "logical_bytes": logical_bytes,
        "arithmetic_intensity_flop_per_byte": intensity,
        "tflops": tflops,
        "logical_tb_s": tb_s,
        "roof_tflops": roof_tflops,
        "roofline_efficiency": tflops / roof_tflops,
        "compute_peak_efficiency": tflops / MI455_MXFP8_TFLOPS,
        "hbm_peak_efficiency": tb_s / MI455_HBM_TB_S,
    }


def load_shapes(config, families, ms):
    family_set = set(families)
    m_set = set(ms)
    rows = []
    with open(config, newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["N"]), int(row["K"]))
            m = int(row["M"])
            if row["gfx"] == "gfx1250" and key in family_set and m in m_set:
                rows.append((m, *key))
    expected = len(family_set) * len(m_set)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} exact config shapes, found {len(rows)}")
    return sorted(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--families", required=True)
    parser.add_argument("--ms", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--reps", type=int, default=15)
    parser.add_argument("--inner", type=int, default=50)
    args = parser.parse_args()
    families = [tuple(map(int, item.split(","))) for item in args.families.split(";")]
    ms = list(map(int, args.ms.split(",")))
    results = []
    for m, n, k in load_shapes(args.config, families, ms):
        data = make_data(m, n, k, 113000 + m + n + k)
        eager = run(data)
        torch.cuda.synchronize()
        err = error_ratio(eager, data["ref"])
        timing = samples_us(data, args.warmup, args.reps, args.inner)
        median_us = timing[len(timing) // 2]
        results.append(
            {
                "M": m,
                "N": n,
                "K": k,
                "median_us": median_us,
                "min_us": timing[0],
                "p90_us": timing[math.ceil(0.9 * len(timing)) - 1],
                "samples_us": timing,
                "err_ratio": err,
                **logical_metrics(m, n, k, median_us),
            }
        )
        print(json.dumps(results[-1]), flush=True)
        del data, eager
        torch.cuda.empty_cache()
    payload = {
        "hardware_roof": {
            "mx_fp8_tflops": MI455_MXFP8_TFLOPS,
            "hbm_tb_s": MI455_HBM_TB_S,
        },
        "timing": {
            "warmup": args.warmup,
            "reps": args.reps,
            "inner": args.inner,
            "statistic": "median GPU-event time per production AITER call",
        },
        "results": results,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
