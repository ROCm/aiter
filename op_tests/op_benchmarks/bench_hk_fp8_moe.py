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


def make_weights():
    import torch

    from aiter.ops.shuffle import shuffle_weight

    generator = torch.Generator(device="cuda").manual_seed(0)

    def quantize(n, k):
        w = (
            torch.randn(ROUTED_EXPERTS + 1, n, k, device="cuda", generator=generator)
            * 0.02
        )
        blocks = w.view(ROUTED_EXPERTS + 1, n // 128, 128, k // 128, 128)
        scale = torch.exp2(
            torch.ceil(
                torch.log2(
                    blocks.abs().amax(dim=(2, 4), keepdim=True).clamp_min(1e-12) / 448.0
                )
            )
        )
        q = (blocks / scale).to(torch.float8_e4m3fn).view_as(w)
        reference = (q.float().view_as(blocks) * scale).view_as(w)
        packed = shuffle_weight(q, (16, 16))
        return (
            packed,
            scale.view(ROUTED_EXPERTS + 1, n // 128, k // 128).contiguous(),
            reference,
        )

    w13, s13, reference13 = quantize(2 * INTER, D)
    w2, s2, reference2 = quantize(D, INTER)
    return w13, s13, w2, s2, reference13, reference2


def make_inputs(tokens, topk):
    import torch

    generator = torch.Generator(device="cuda").manual_seed(tokens)
    x = torch.randn(tokens, D, dtype=torch.bfloat16, device="cuda", generator=generator)
    logits = torch.randn(tokens, ROUTED_EXPERTS, device="cuda", generator=generator)
    weights, ids = torch.topk(logits.softmax(-1), 8, dim=-1)
    if topk == 9:
        ids = torch.cat((ids, torch.full_like(ids[:, :1], ROUTED_EXPERTS)), dim=1)
        weights = torch.cat((weights, torch.ones_like(weights[:, :1])), dim=1)
    return x, ids.to(torch.int32).contiguous(), weights.float().contiguous()


def reference(x, ids, weights, w13, w2, limit=0.0):
    import torch

    rows = torch.linspace(0, x.shape[0] - 1, min(x.shape[0], 128), device="cuda").long()
    xs, es, ws = x[rows].float(), ids[rows], weights[rows]
    out = torch.zeros(len(rows), D, dtype=torch.float32, device="cuda")
    for expert in torch.unique(es).tolist():
        token, slot = (es == expert).nonzero(as_tuple=True)
        gate, up = (xs[token] @ w13[expert].T).chunk(2, dim=-1)
        if limit > 0:
            gate, up = gate.clamp(max=limit), up.clamp(-limit, limit)
        hidden = torch.nn.functional.silu(gate) * up
        out.index_add_(0, token, (hidden @ w2[expert].T) * ws[token, slot, None])
    return rows, out


def capture_repeated(call, repetitions):
    import torch

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

    import torch

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


def dispatch_metadata(fm, tokens, topk, experts):
    import torch

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
        QuantType.per_1x128,
        True,
        ActivationType.Silu,
        False,
        0,
        0,
        True,
        opus_weights_shuffled=True,
        input_dtype=torch.bfloat16,
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
    rows = []
    for name in (
        "a8w8_blockscale_tuned_fmoe_glm5_3_flash_gfx950.csv",
        "a8w8_blockscale_tuned_fmoe_glm5_3_flash_shared_gfx950.csv",
    ):
        p = ROOT / "aiter/configs/model_configs" / name
        with p.open() as f:
            reader = csv.DictReader(f)
            fields.extend(k for k in reader.fieldnames if k not in fields)
            rows.extend(reader)
        sources.append(str(p.relative_to(ROOT)))
    cfg = OUT.with_suffix(".csv")
    with cfg.open("w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.environ["AITER_CONFIG_FMOE"] = str(cfg)
    import importlib

    import aiter
    from aiter.ops.hk_fp8_moe import hk_fp8_moe

    fm = importlib.import_module("aiter.fused_moe")
    w13, s13, w2, s2, ref13, ref2 = make_weights()
    result = {
        "mode": "moe",
        "base": os.environ["REVIEW_BASE"],
        "config_sources": sources,
        "protocol": "whole operator, identical BF16 input and shuffled block FP8 weights; checked-in GLM tuning; warm HIP graphs; 21 alternating samples; unclamped Silu",
        "cases": [],
    }
    for topk in (8, 9):
        E = 288 + (topk == 9)
        a, b = w13[:E], w2[:E]
        a.is_shuffled = b.is_shuffled = True
        for T in (1, 128, 1024, 2048, 4096, 8192, 16384):
            x, ids, tw = make_inputs(T, topk)
            rows, expected = reference(x, ids, tw, ref13, ref2)
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
            timing = measure_pair(graphs, 10, 21)
            row = {
                "tokens": T,
                "topk": topk,
                "correctness": checks,
                "timing": timing,
                "hk_speedup": min(
                    timing["aiter_tuned"]["median_us"],
                    timing["aiter_heuristic"]["median_us"],
                )
                / timing["hk"]["median_us"],
                "dispatch": {"tuned": tuned_dispatch, "heuristic": heuristic_dispatch},
            }
            result["cases"].append(row)
            OUT.write_text(json.dumps(result, indent=2) + "\n")
            print(
                "MOE_RESULT",
                T,
                topk,
                row["hk_speedup"],
                timing["hk"]["median_us"],
                min(
                    timing["aiter_tuned"]["median_us"],
                    timing["aiter_heuristic"]["median_us"],
                ),
                flush=True,
            )
            del graphs, outputs, calls, x, ids, tw, expected, out
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
