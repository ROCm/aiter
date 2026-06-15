# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Run the FlyDSL fused MoE (a4w4) stage1 + stage2 path over ``config.json``.

The FlyDSL MoE kernels operate on quantized inputs (mxfp4 activations and
weights, ``per_1x32`` scales) with pre-shuffled weights/scales and a sorted
token dispatch. This script builds those inputs from seeded random weights,
runs stage1 (gate/up GEMM + gated activation), re-quantizes the intermediate,
runs stage2 (down GEMM + weighted combine), and prints output stats.

Standalone runnable::

    python flydsl_run.py
    python flydsl_run.py --json flydsl_out.json --case a4w4_t16_e256
    python -m aiter.ops.flydsl.examples.moe.flydsl_run

Skips with a clear message and exits 0 when ROCm/CUDA or the optional
``flydsl`` package is unavailable.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make ``aiter`` importable when this file is executed directly (not via -m).
try:
    from aiter.ops.flydsl.examples import _common
except ModuleNotFoundError:  # pragma: no cover - direct-execution fallback
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
    from aiter.ops.flydsl.examples import _common

DEFAULT_CONFIG = Path(__file__).resolve().parent / "config.json"


def _build_inputs(case: dict, params: dict, torch_dtype, device: str) -> dict:
    import torch
    import aiter
    from aiter import QuantType, dtypes
    from aiter.fused_moe import fused_topk, moe_sorting
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from aiter.utility.fp4_utils import e8m0_shuffle, moe_mxfp4_sort

    token, model_dim, inter_dim = case["tokens"], case["model_dim"], case["inter_dim"]
    experts, topk = case["experts"], case["topk"]
    block_m = params["block_m"]
    q_type = QuantType.per_1x32
    q_dtype = dtypes.fp4x2
    torch_quant = aiter.get_torch_quant(q_type)

    gen = torch.Generator(device=device)
    gen.manual_seed(_common.DEFAULT_INPUT_SEED)

    def randn(*shape):
        return torch.randn(shape, generator=gen, device=device, dtype=torch_dtype) / 10.0

    inp = randn(token, model_dim)
    w1 = randn(experts, inter_dim * 2, model_dim)
    w2 = randn(experts, model_dim, inter_dim)
    score = torch.randn((token, experts), generator=gen, device=device, dtype=torch_dtype)
    topk_weights, topk_ids = fused_topk(inp, score, topk, True)

    w1_qt, w1_scale = torch_quant(w1, quant_dtype=q_dtype)
    w2_qt, w2_scale = torch_quant(w2, quant_dtype=q_dtype)
    w1_qt = w1_qt.view(experts, inter_dim * 2, model_dim // 2)
    w2_qt = w2_qt.view(experts, model_dim, inter_dim // 2)
    a1_qt, a1_scale = torch_quant(inp, quant_dtype=q_dtype)

    sorted_ids, sorted_weights, sorted_expert_ids, num_valid_ids, _ = moe_sorting(
        topk_ids, topk_weights, experts, model_dim, torch_dtype, block_m
    )

    w1_qt_shuf = shuffle_weight(w1_qt, (16, 16))
    w2_qt_shuf = shuffle_weight_a16w4(w2_qt, 16, False)
    w1_scale_shuf = e8m0_shuffle(w1_scale)
    w2_scale_shuf = shuffle_scale_a16w4(w2_scale, experts, False)
    a1_scale_sort = moe_mxfp4_sort(
        a1_scale[:token, :].view(token, 1, -1),
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token,
        block_size=block_m,
    )

    return {
        "a1_qt": a1_qt,
        "a1_scale_sort": a1_scale_sort,
        "w1_qt_shuf": w1_qt_shuf,
        "w1_scale_shuf": w1_scale_shuf,
        "w2_qt_shuf": w2_qt_shuf,
        "w2_scale_shuf": w2_scale_shuf,
        "sorted_ids": sorted_ids,
        "sorted_weights": sorted_weights,
        "sorted_expert_ids": sorted_expert_ids,
        "num_valid_ids": num_valid_ids,
    }


def run_case(case: dict, *, params: dict, dtype: str, device: str) -> dict:
    import torch
    import aiter
    from aiter import QuantType, dtypes
    from aiter.utility.fp4_utils import moe_mxfp4_sort
    from aiter.ops.flydsl.moe_kernels import flydsl_moe_stage1, flydsl_moe_stage2

    token, model_dim, inter_dim = case["tokens"], case["model_dim"], case["inter_dim"]
    topk = case["topk"]
    block_m = params["block_m"]
    tile_n = params["tile_n"]
    tile_k = params["tile_k"]
    mode = params["mode"]
    torch_dtype = _common.parse_dtype(dtype)
    out_dtype = "bf16" if torch_dtype == torch.bfloat16 else "f16"

    data = _build_inputs(case, params, torch_dtype, device)

    stage1_out = flydsl_moe_stage1(
        a=data["a1_qt"],
        w1=data["w1_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype=out_dtype,
        w1_scale=data["w1_scale_shuf"],
        a1_scale=data["a1_scale_sort"],
        sorted_weights=None,
    )
    torch.cuda.synchronize()

    torch_quant = aiter.get_torch_quant(QuantType.per_1x32)
    a2_qt, a2_scale = torch_quant(stage1_out.view(-1, inter_dim), quant_dtype=dtypes.fp4x2)
    a2_qt = a2_qt.view(token, topk, -1)
    a2_scale_sort = moe_mxfp4_sort(
        a2_scale[: token * topk, :].view(token, topk, -1),
        sorted_ids=data["sorted_ids"],
        num_valid_ids=data["num_valid_ids"],
        token_num=token,
        block_size=block_m,
    )

    out = flydsl_moe_stage2(
        inter_states=a2_qt,
        w2=data["w2_qt_shuf"],
        sorted_token_ids=data["sorted_ids"],
        sorted_expert_ids=data["sorted_expert_ids"],
        num_valid_ids=data["num_valid_ids"],
        topk=topk,
        tile_m=block_m,
        tile_n=tile_n,
        tile_k=tile_k,
        a_dtype="fp4",
        b_dtype="fp4",
        out_dtype=out_dtype,
        mode=mode,
        w2_scale=data["w2_scale_shuf"],
        a2_scale=a2_scale_sort,
        sorted_weights=data["sorted_weights"],
    )
    torch.cuda.synchronize()

    return {
        "name": case["name"],
        "shape": {
            "tokens": token,
            "model_dim": model_dim,
            "inter_dim": inter_dim,
            "experts": case["experts"],
            "topk": topk,
        },
        "params": params,
        "output": _common.tensor_stats(out),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FlyDSL fused MoE kernels.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="path to config.json")
    parser.add_argument("--json", default=None, help="write run outputs here")
    parser.add_argument(
        "--case", action="append", default=None, help="only run case(s) by name (repeatable)"
    )
    args = parser.parse_args(argv)

    if not _common.require_backends():
        return 0

    config = _common.load_config(args.config)
    dtype = config.get("dtype", "bf16")
    params = config.get("default_params", {})
    cases = config.get("cases", [])
    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c["name"] in wanted]
    if not cases:
        print("[examples] no matching cases in config; nothing to do.")
        return 0

    device = "cuda"
    print(f"[examples] moe FlyDSL a4w4 stage1+stage2  dtype={dtype}")
    results = []
    for case in cases:
        case_params = {**params, **{k: case[k] for k in params if k in case}}
        print("=" * 70)
        print(
            f"[moe] case={case['name']} tokens={case['tokens']} "
            f"D={case['model_dim']} I={case['inter_dim']} "
            f"E={case['experts']} topk={case['topk']}"
        )
        results.append(run_case(case, params=case_params, dtype=dtype, device=device))

    rows = []
    for r in results:
        s, o = r["shape"], r["output"]
        rows.append(
            {
                "name": r["name"],
                "tokens": s["tokens"],
                "dims": f"D{s['model_dim']}/I{s['inter_dim']}/E{s['experts']}/k{s['topk']}",
                "out_shape": "x".join(str(x) for x in o["shape"]),
                "min": o["min"],
                "max": o["max"],
                "mean": o["mean"],
                "std": o["std"],
            }
        )
    print(f"\n{'=' * 70}")
    _common.print_table(
        rows,
        [
            ("name", "CASE"),
            ("tokens", "TOKENS"),
            ("dims", "DIMS"),
            ("out_shape", "OUT"),
            ("min", "MIN"),
            ("max", "MAX"),
            ("mean", "MEAN"),
            ("std", "STD"),
        ],
    )

    if args.json:
        _common.dump_json(
            args.json,
            {
                "op": config.get("op", "moe"),
                "kind": "flydsl_run",
                "dtype": dtype,
                "environment": _common.environment_info(),
                "results": results,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
