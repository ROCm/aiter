# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare HipKittens with current AITER; save raw samples and correctness checks."""

import argparse
import gc
import json
import os
import subprocess
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]

D, INTER, ROUTED_EXPERTS = 4096, 1024, 288


def prepare_weights():
    from aiter.ops.shuffle import (
        shuffle_scale_a16w4,
        shuffle_weight,
        shuffle_weight_a16w4,
    )
    from op_tests.hk_fp8_moe_utils import dequant, make_weights

    def pack(q, scale, gate_up):
        _, n, k = q.shape
        reference = dequant(q, scale)
        packed = shuffle_weight(q, (16, 16))
        # Repeat each power-of-two 128x128 scale over its 1x32 sub-blocks.
        # The E4M3 payload and every dequantized weight remain exactly equal.
        block_scale = scale.view(ROUTED_EXPERTS + 1, n // 128, k // 128)
        exponent = ((block_scale.view(torch.int32) >> 23) & 255).to(torch.uint8)
        assert torch.equal((exponent.int() << 23).view(torch.float32), block_scale)
        mx_scale = exponent.repeat_interleave(128, 1).repeat_interleave(4, 2)
        mx_scale = mx_scale.flatten(0, 1).view(torch.float8_e8m0fnu)
        mx_weight = shuffle_weight_a16w4(q, 16, gate_up)
        mx_scale = shuffle_scale_a16w4(mx_scale, ROUTED_EXPERTS + 1, gate_up)
        return (
            packed,
            scale.view(ROUTED_EXPERTS + 1, n // 128, k // 128).contiguous(),
            reference,
            mx_weight,
            mx_scale,
        )

    q13, scale13, q2, scale2 = make_weights(ROUTED_EXPERTS + 1)
    w13, s13, reference13, mx13, mxs13 = pack(q13, scale13, True)
    w2, s2, reference2, mx2, mxs2 = pack(q2, scale2, False)
    return w13, s13, w2, s2, reference13, reference2, mx13, mxs13, mx2, mxs2


def capture_repeated(call, repetitions):
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repetitions):
            output = call()
    for _ in range(5):
        graph.replay()
    torch.cuda.synchronize()
    return graph, output


def measure_pair(graphs, repetitions, rounds):
    import statistics

    samples = {name: [] for name in graphs}
    names = list(graphs)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    for iteration in range(rounds):
        order = names if iteration % 2 == 0 else names[::-1]
        for name in order:
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000 / repetitions)
    result = {}
    for name, values in samples.items():
        ordered = sorted(values)
        result[name] = {
            "median_us": statistics.median(values),
            "min_us": min(values),
            "p10_us": ordered[int((len(values) - 1) * 0.1)],
            "p90_us": ordered[int((len(values) - 1) * 0.9)],
            "samples_us": values,
        }
    return result


def dispatch_metadata(fm, tokens, topk, experts, native=False):
    from aiter import ActivationType, QuantType

    metadata = fm.get_2stage_cfgs(
        fm.get_padded_M(tokens),
        D,
        INTER,
        experts,
        topk,
        torch.bfloat16,
        torch.float8_e4m3fn,
        torch.float8_e4m3fn,
        QuantType.per_1x32 if native else QuantType.per_1x128,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        opus_weights_shuffled=True,
        input_dtype=torch.bfloat16,
        gate_mode="interleave" if native else "separated",
        gate_up_interleave=True if native else None,
    )
    return {
        name: (
            str(getattr(metadata, name))
            if name in ("stage1", "stage2", "full_impl")
            else getattr(metadata, name)
        )
        for name in (
            "stage1",
            "stage2",
            "block_m",
            "ksplit",
            "run_1stage",
            "flat",
            "full_impl",
        )
    }


def accuracy(out, ref, rows, tolerance):
    error = ((out[rows].float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    finite = torch.isfinite(out).all().item()
    assert finite and error < tolerance, (error, finite, tolerance)
    return {"relative_l2": error, "all_finite": finite}


def moe():
    import csv

    sources = []
    fields = []
    config_rows = []
    for name in (
        "a8w8_blockscale_tuned_fmoe_glm5_3_flash_gfx950.csv",
        "a8w8_blockscale_tuned_fmoe_glm5_3_flash_shared_gfx950.csv",
    ):
        p = ROOT / "aiter/configs/model_configs" / name
        with p.open() as f:
            reader = csv.DictReader(f)
            fields.extend(k for k in reader.fieldnames if k not in fields)
            config_rows.extend(reader)
        sources.append(str(p.relative_to(ROOT)))
    native_recipes = {}
    for model in ("glm5", "minimax_m3"):
        path = ROOT / "aiter/configs/model_configs" / f"{model}_mxfp8_tuned_fmoe.csv"
        with path.open() as source:
            reader = csv.DictReader(source)
            fields.extend(k for k in reader.fieldnames if k not in fields)
            for recipe in reader:
                name = f"aiter_{model}_recipe_i{recipe['inter_dim']}"
                native_recipes.setdefault(int(recipe["token"]), []).append(
                    (name, recipe)
                )
        sources.append(str(path.relative_to(ROOT)))
    cfg = OUT.with_suffix(".csv")
    with cfg.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(config_rows)
    os.environ["AITER_CONFIG_FMOE"] = str(cfg)
    import importlib

    import aiter
    from aiter.ops.hk_fp8_moe import hk_fp8_moe
    from op_tests.hk_fp8_moe_utils import make_routing, reference

    fm = importlib.import_module("aiter.fused_moe")
    w13, s13, w2, s2, ref13, ref2, mx13, mxs13, mx2, mxs2 = prepare_weights()
    result = {
        "mode": "moe",
        "base": os.environ["REVIEW_BASE"],
        "config_sources": sources,
        "benchmark_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "native_recipes": "Retarget the published GLM-5 and MiniMax-M3 MXFP8 kernel choices to D4096/I1024 and the actual expert/top-k counts. Validate each whole operator against the same FP32 reference before ranking; these are tuning candidates, not prevalidated configs for this geometry.",
        "native_mxfp8": "Identical FP8 weight payloads; power-of-two 128x128 scales repeated losslessly into E8M0 1x32 scales. Activation quantization changes to per-32. Weight repacking excluded; runtime quantization included.",
        "protocol": "whole operator, identical BF16 input and shuffled block FP8 weights; checked-in GLM tuning; warm HIP graphs; 21 alternating samples; unclamped Silu",
        "cases": [],
    }
    for topk in (8, 9):
        E = 288 + (topk == 9)
        a, b = w13[:E], w2[:E]
        a.is_shuffled = b.is_shuffled = True
        ma, mb = mx13[:E], mx2[:E]
        ma.is_shuffled = mb.is_shuffled = True
        ma.gate_up_interleave = True
        mb.gate_up_interleave = False
        ms1, ms2 = mxs13[: E * 2 * INTER], mxs2[: E * D]
        shared_a, shared_b = mx13[-1:], mx2[-1:]
        shared_a.is_shuffled = shared_b.is_shuffled = True
        shared_a.gate_up_interleave = True
        shared_b.gate_up_interleave = False
        for T in (1, 128, 1024, 2048, 4096, 8192, 16384):
            x, ids, tw = make_routing(T, E)
            rows = torch.linspace(0, T - 1, min(T, 128), device="cuda").long()
            expected = reference(x[rows], ref13, ref2, tw[rows], ids[rows], 0.0)
            out = torch.empty_like(x)
            calls = {
                "hk": partial(hk_fp8_moe, x, ids, tw, a, s13[:E], b, s2[:E]),
                "aiter_tuned": partial(
                    fm.fused_moe,
                    x,
                    a,
                    b,
                    tw,
                    ids,
                    activation=aiter.ActivationType.Silu,
                    quant_type=aiter.QuantType.per_1x128,
                    w1_scale=s13[:E],
                    w2_scale=s2[:E],
                    dtype=torch.bfloat16,
                    output=out,
                ),
            }
            graphs = {}
            outputs = {}
            for name, call in calls.items():
                os.environ["AITER_BYPASS_TUNE_CONFIG"] = "0"
                fm.get_2stage_cfgs.cache_clear()
                graphs[name], outputs[name] = capture_repeated(call, 10)
            tuned_dispatch = dispatch_metadata(fm, T, topk, E)
            os.environ["AITER_BYPASS_TUNE_CONFIG"] = "1"
            fm.get_2stage_cfgs.cache_clear()
            out = torch.empty_like(x)
            graphs["aiter_heuristic"], outputs["aiter_heuristic"] = capture_repeated(
                partial(calls["aiter_tuned"], output=out), 10
            )
            heuristic_dispatch = dispatch_metadata(fm, T, topk, E)
            os.environ["AITER_BYPASS_TUNE_CONFIG"] = "0"
            fm.get_2stage_cfgs.cache_clear()
            checks = {
                name: accuracy(output, expected, rows, 0.06)
                for name, output in outputs.items()
            }
            native_calls = {
                "aiter_mxfp8": partial(
                    fm.fused_moe,
                    x,
                    ma,
                    mb,
                    tw,
                    ids,
                    activation=aiter.ActivationType.Silu,
                    quant_type=aiter.QuantType.per_1x32,
                    w1_scale=ms1,
                    w2_scale=ms2,
                    gate_mode="interleave",
                    dtype=torch.bfloat16,
                    output=torch.empty_like(x),
                )
            }
            if topk == 9:
                native_calls["aiter_fhmoe_mxfp8"] = partial(
                    native_calls["aiter_mxfp8"],
                    shared_w1=shared_a,
                    shared_w2=shared_b,
                    shared_w1_scale=mxs13[-2 * INTER :],
                    shared_w2_scale=mxs2[-D:],
                    shared_expert_id=E - 1,
                    clamp_shared=False,
                    output=torch.empty_like(x),
                )
            for name, call in list(native_calls.items()):
                native_calls[name + "_reduce"] = partial(
                    call, output=torch.empty_like(x)
                )
            failures, native_dispatch = {}, {}
            for name, call in native_calls.items():
                os.environ["AITER_FLYDSL_FORCE_REDUCE"] = (
                    "1" if name.endswith("_reduce") else "0"
                )
                fm.get_2stage_cfgs.cache_clear()
                try:
                    graph, output = capture_repeated(call, 10)
                    checks[name] = accuracy(output, expected, rows, 0.06)
                    graphs[name], outputs[name] = graph, output
                    if name == "aiter_mxfp8":
                        native_dispatch = dispatch_metadata(fm, T, topk, E, native=True)
                except Exception as exc:  # noqa: BLE001
                    # Retain failed candidates for review instead of hiding them.
                    failures[name] = f"{type(exc).__name__}: {exc}"
                    print("CANDIDATE_FAILED", T, topk, name, failures[name], flush=True)
            os.environ["AITER_FLYDSL_FORCE_REDUCE"] = "0"
            recipe_dispatch = {}
            default_config = fm.cfg_2stages
            for name, original in native_recipes.get(T, []):
                recipe = dict(
                    original,
                    model_dim=str(D),
                    inter_dim=str(INTER),
                    expert=str(E),
                    topk=str(topk),
                )
                with cfg.open("w") as source:
                    writer = csv.DictWriter(source, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(config_rows + [recipe])
                fm.cfg_2stages = None
                fm.get_2stage_cfgs.cache_clear()
                try:
                    call = partial(
                        native_calls["aiter_mxfp8"], output=torch.empty_like(x)
                    )
                    graph, output = capture_repeated(call, 10)
                    check = accuracy(output, expected, rows, 0.06)
                    selected_metadata = dispatch_metadata(fm, T, topk, E, native=True)
                    selected_text = json.dumps(selected_metadata)
                    assert (
                        recipe["kernelName1"] in selected_text
                        and recipe["kernelName2"] in selected_text
                    ), selected_metadata
                    checks[name] = check
                    graphs[name], outputs[name] = graph, output
                    recipe_dispatch[name] = selected_metadata
                except Exception as exc:  # noqa: BLE001
                    # Retargeted recipes must compile and pass the numerical reference.
                    failures[name] = f"{type(exc).__name__}: {exc}"
                    print("CANDIDATE_FAILED", T, topk, name, failures[name], flush=True)
            with cfg.open("w") as source:
                writer = csv.DictWriter(source, fieldnames=fields)
                writer.writeheader()
                writer.writerows(config_rows)
            fm.cfg_2stages = default_config
            fm.get_2stage_cfgs.cache_clear()
            timing = measure_pair(graphs, 10, 21)
            winner = min(
                (n for n in timing if n != "hk"), key=lambda n: timing[n]["median_us"]
            )
            row = {
                "tokens": T,
                "topk": topk,
                "correctness": checks,
                "timing": timing,
                "hk_speedup": timing[winner]["median_us"] / timing["hk"]["median_us"],
                "strongest_baseline": winner,
                "candidate_failures": failures,
                "dispatch": {
                    "tuned": tuned_dispatch,
                    "heuristic": heuristic_dispatch,
                    "native": native_dispatch,
                    "retargeted_recipes": recipe_dispatch,
                },
            }
            result["cases"].append(row)
            OUT.write_text(json.dumps(result, indent=2) + "\n")
            print(
                "MOE_RESULT",
                T,
                topk,
                row["hk_speedup"],
                timing["hk"]["median_us"],
                winner,
                timing[winner]["median_us"],
                flush=True,
            )
            del graphs, outputs, calls, native_calls, x, ids, tw, expected, out
            gc.collect()
            torch.cuda.empty_cache()
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("hk_moe_benchmark.json"))
    args = parser.parse_args()
    OUT = args.output
    OUT.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "REVIEW_BASE",
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
    )
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result = moe()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
