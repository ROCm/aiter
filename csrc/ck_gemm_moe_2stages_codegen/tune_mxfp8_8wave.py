#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Sweep eight-wave MXFP8 prefill tiles/swizzles and update selected CSV rows.

Example (gfx950):
  python csrc/ck_gemm_moe_2stages_codegen/tune_mxfp8_8wave.py \
    --csv aiter/configs/model_configs/minimax_m3_mxfp8_tuned_fmoe.csv \
    --tokens 16384 32768 --output /tmp/mxfp8_sweep.json --write

The input CSV supplies the baseline. Every candidate uses the same seeded
weights, activations and routing. Timings are sums of GPU kernel durations;
stage wrappers include quantization/reduction, and total includes sorting.
"""

import argparse
import csv
import functools
import importlib
import json
import os
from pathlib import Path

import torch

# AITER resolves model CSVs at import time. Select the requested baseline first.
if __name__ == "__main__":
    _early_parser = argparse.ArgumentParser(add_help=False)
    _early_parser.add_argument("--csv", type=Path)
    _early, _ = _early_parser.parse_known_args()
    if _early.csv is not None:
        os.environ["AITER_CONFIG_FMOE"] = str(_early.csv.resolve())

import aiter
from aiter import dtypes
from aiter.ops.flydsl.mxfp8_moe_8wave import kernel_name, stage1, stage2
from aiter.ops.quant import per_1x32_f8_scale_f8_quant
from aiter.ops.shuffle import shuffle_scale_a16w4, shuffle_weight_a16w4
from aiter.test_common import run_perftest
from aiter.utility import fp4_utils

fm = importlib.import_module("aiter.fused_moe")


def inputs(row, seed, balanced=False):
    torch.manual_seed(seed)
    t, h, i, e, k = (
        int(row[x]) for x in ("token", "model_dim", "inter_dim", "expert", "topk")
    )
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16)
    w1 = torch.randn(e, 2 * i, h, device="cuda", dtype=torch.bfloat16)
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16)
    score = torch.randn(t, e, device="cuda", dtype=torch.bfloat16)
    selected, ids = score.float().topk(k, dim=-1)
    weights = selected.softmax(-1)
    if balanced:
        ids = torch.arange(t * k, device="cuda").reshape(t, k) % e
        weights = torch.full((t, k), 1.0 / k, device="cuda")
    q1, s1 = per_1x32_f8_scale_f8_quant(w1, scale_type=dtypes.fp8_e8m0)
    q2, s2 = per_1x32_f8_scale_f8_quant(w2, scale_type=dtypes.fp8_e8m0)
    q1, q2 = shuffle_weight_a16w4(q1, 16, True), shuffle_weight_a16w4(q2, 16, False)
    s1, s2 = shuffle_scale_a16w4(s1, e, True), fp4_utils.e8m0_shuffle(s2)
    return (x, q1, q2, weights, ids.int()), {
        "w1_scale": s1,
        "w2_scale": s2,
        "quant_type": aiter.QuantType.per_1x32.value,
        "activation": aiter.ActivationType.Swiglu.value,
        "gate_mode": "interleave",
    }


def metadata(name1, name2):
    return fm.MOEMetadata(
        functools.partial(stage1, kernelName=name1),
        functools.partial(stage2, kernelName=name2),
        256,
        0,
        prequant=False,
        fuse_quant="fp8",
        skip_inter_quant=True,
    )


def capture(call):
    calls = []
    fm.kernel_bench_callable = calls
    try:
        output = call().clone()
    finally:
        fm.kernel_bench_callable = None
    return output, dict(calls)


def time_us(call):
    return run_perftest(call, num_iters=20, num_warmup=3)[1]


def diff(a, b):
    a, b = a.float(), b.float()
    return ((a - b).square().sum() / (a.square() + b.square()).sum()).item()


def tune(row, args):
    data, kwargs = inputs(row, args.seed, args.balanced)
    common = kwargs
    baseline = functools.partial(fm._fused_moe_impl, *data, **common)
    reference, old_calls = capture(baseline)
    original = dict(
        us=time_us(baseline), **{s + "_us": time_us(c) for s, c in old_calls.items()}
    )
    selected = [kernel_name(1), kernel_name(2, swizzle=3)]
    candidates = []
    for stage in (1, 2):
        tiles = ((256, 256), (128, 512)) if stage == 1 else ((256, 256),)
        trials = []
        for tm, tn in tiles:
            for sw in args.swizzles:
                name = kernel_name(stage, tm, tn, sw)
                names = selected.copy()
                names[stage - 1] = name
                meta = metadata(*names)
                forward = functools.partial(
                    fm._fused_moe_impl,
                    *data,
                    **common,
                    _metadata_transform=lambda _, m=meta: m,
                )
                output, calls = capture(forward)
                error = diff(reference, output)
                assert output.isfinite().all() and error < 2e-4, (name, error)
                us = time_us(calls[f"stage{stage}"])
                trial = {"stage": stage, "kernel": name, "us": us, "logits_diff": error}
                print(
                    json.dumps(
                        dict(token=row["token"], inter_dim=row["inter_dim"], **trial)
                    ),
                    flush=True,
                )
                candidates.append(trial)
                trials.append(trial)
        best = min(trials, key=lambda x: x["us"])
        selected[stage - 1] = best["kernel"]
    meta = metadata(*selected)
    forward = functools.partial(
        fm._fused_moe_impl, *data, **common, _metadata_transform=lambda _, m=meta: m
    )
    output, calls = capture(forward)
    for _ in range(3):
        assert torch.equal(forward(), output), "Non-deterministic eight-wave MoE output"
    measured = dict(
        us=time_us(forward), **{s + "_us": time_us(c) for s, c in calls.items()}
    )
    error = diff(reference, output)
    assert error < 2e-4
    tuned = dict(
        row,
        block_m=256,
        ksplit=0,
        kernelName1=selected[0],
        kernelName2=selected[1],
        us1=round(measured["stage1_us"], 4),
        us2=round(measured["stage2_us"], 4),
        us=round(measured["us"], 4),
        err1="0.0%",
        err2="0.0%",
    )
    t, h, i, k = (int(row[x]) for x in ("token", "model_dim", "inter_dim", "topk"))
    tuned["tflops"] = round(6 * t * h * i * k / measured["us"] / 1e6, 2)
    # Old CSV bandwidth estimates have a different traffic accounting convention.
    tuned["bw"] = ""
    result = {
        "token": t,
        "inter_dim": i,
        "seed": args.seed,
        "balanced": args.balanced,
        "baseline": original,
        "selected": measured,
        "kernels": selected,
        "speedup": original["us"] / measured["us"],
        "logits_diff": error,
        "repeatability": "full output, three identical launches",
        "candidates": candidates,
    }
    return tuned, result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--tokens", nargs="+", type=int, default=[16384, 32768])
    p.add_argument("--inter-dims", nargs="+", type=int, default=[384, 768])
    p.add_argument("--swizzles", nargs="+", type=int, default=[0, 1, 2, 3, 4, 8])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balanced", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    with args.csv.open() as f:
        reader = csv.DictReader(f)
        fields, rows = reader.fieldnames, list(reader)
    results = []
    for index, row in enumerate(rows):
        if (
            int(row["token"]) not in args.tokens
            or int(row["inter_dim"]) not in args.inter_dims
        ):
            continue
        if (
            row["q_dtype_w"] != "torch.float8_e4m3fn"
            or row["act_type"] != "ActivationType.Swiglu"
        ):
            continue
        rows[index], result = tune(row, args)
        results.append(result)
        args.output.write_text(json.dumps(results, indent=2) + "\n")
    if not results:
        raise ValueError("No matching MXFP8 prefill rows")
    if args.write:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
