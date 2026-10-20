# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare HipKittens with current AITER; save raw samples and correctness checks."""

import argparse
import gc
import importlib.util
import json
import os
import statistics
import subprocess
from functools import partial
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def capture(fn):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = fn()
    graph.replay()
    torch.cuda.synchronize()
    return graph, out


def measure(graphs, rounds, flush=None):
    samples = {name: [] for name in graphs}
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
        enable_timing=True
    )
    names = list(graphs)
    for r in range(rounds):
        for name in names if r % 2 == 0 else names[::-1]:
            if flush is not None:
                flush.fill_(r % 251)
            start.record()
            graphs[name].replay()
            end.record()
            end.synchronize()
            samples[name].append(start.elapsed_time(end) * 1000)
    return {
        name: {"median_us": statistics.median(values), "samples_us": values}
        for name, values in samples.items()
    }


def accuracy(out, ref, rows, tolerance):
    error = ((out[rows].float() - ref).norm() / ref.norm().clamp_min(1e-12)).item()
    finite = torch.isfinite(out).all().item()
    assert finite and error < tolerance, (error, finite, tolerance)
    return {"relative_l2": error, "all_finite": finite}


def sparse():
    from aiter.ops.hk_sparse_mla import hk_sparse_mla_fwd
    from aiter.ops.pa_sparse_prefill_opus import (
        pa_sparse_prefill_fp8_opus,
        pa_sparse_prefill_opus,
    )
    from aiter.ops.triton.attention.sparse_mla import sparse_mla_fwd

    tests = module(ROOT / "op_tests/test_hk_sparse_mla.py", "hk_tests")
    one = torch.ones(1, dtype=torch.float32, device="cuda")
    flush = torch.empty(512 << 20, dtype=torch.uint8, device="cuda")
    result = {
        "mode": "sparse",
        "benchmark_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "base": os.environ["REVIEW_BASE"],
        "torch": str(torch.__version__),
        "hip": str(torch.version.hip),
        "protocol": {
            "heads": 32,
            "latent": 512,
            "q_kv_dtype": "float8_e4m3fn",
            "output": "bfloat16",
            "scale": 512**-0.5,
            "cache": "512 MiB eviction before each timed one-call HIP graph replay; eviction excluded",
            "selection": "all valid uniform random physical slots; duplicate selections retained",
            "tuning": "FP8 and BF16 dots; kv_splits default,1,2,4,8,16,32; nine samples",
            "opus": "Lossless repack of identical values to native OPUS caches: BF16 (2x bytes) or 448 FP8 + unit E8M0 scales + 64 BF16 (1.25x bytes). Q/cache repack excluded; no attention sink. Reported separately from matching-layout Gluon.",
            "confirmation": "top three candidates plus both defaults, two fresh seeds, 21 samples each, alternating order",
            "reference": "FP32 attention on identical quantized inputs, sampled rows; relative L2 < 0.04",
        },
        "cases": [],
    }
    for slots, K in ((16384, 2048), ((5 << 30) // 512, 2051)):
        torch.manual_seed(slots)
        kv = torch.empty(slots, 512, dtype=torch.float8_e4m3fn, device="cuda")
        for first in range(0, slots, 65536):
            n = min(65536, slots - first)
            kv[first : first + n] = (torch.randn(n, 512, device="cuda") * 0.4).to(
                kv.dtype
            )
        kv_bf16 = kv.to(torch.bfloat16)
        kv_nope = torch.zeros_like(kv)
        kv_rope = kv[:, 448:].to(torch.bfloat16).contiguous()
        kv_nope[:, :448].copy_(kv[:, :448])
        kv_nope.view(torch.uint8)[:, 448:462].fill_(127)
        sink = torch.full((32,), float("-inf"), dtype=torch.float32, device="cuda")
        for T in (1, 8, 32, 128, 512, 2048, 8192, 16384):
            torch.manual_seed(T + slots)
            q = (torch.randn(T, 32, 512, device="cuda") * 0.4).to(kv.dtype)
            ids = torch.randint(slots, (T, K), device="cuda", dtype=torch.int32)
            indptr = torch.arange(T + 1, device="cuda", dtype=torch.int32) * K
            rows = list(range(0, T, max(1, T // 16)))[:16]
            expected = tests.reference(q, kv[:, None, :], ids, 512**-0.5, rows)
            calls = {"hk": partial(hk_sparse_mla_fwd, q, kv, ids, 512**-0.5)}
            for precision in ("fp8", "bf16"):
                for split in (None, 1, 2, 4, 8, 16, 32):
                    name = f"aiter_{precision}_" + (
                        "default" if split is None else str(split)
                    )

                    def invoke(precision, split, q, kv, indptr, ids):
                        return sparse_mla_fwd(
                            q,
                            kv,
                            indptr,
                            ids.flatten(),
                            512**-0.5,
                            kv_scale=one,
                            q_scale=one,
                            kv_lora_rank=512,
                            qk_rope_head_dim=0,
                            dot_precision=precision,
                            kv_splits=split,
                            has_invalid=False,
                        )[0]

                    calls[name] = partial(invoke, precision, split, q, kv, indptr, ids)
            # Give OPUS its native layouts with exactly the same q/k/v values.
            # Cache and query conversion are deliberately excluded from timing.
            q_bf16 = q.to(torch.bfloat16)
            q_nope = torch.zeros_like(q)
            q_nope[:, :, :448].copy_(q[:, :, :448])
            q_nope.view(torch.uint8)[:, :, 448:462].fill_(127)
            q_rope = q[:, :, 448:].to(torch.bfloat16).contiguous()
            empty_ids = ids.flatten()[:0]
            empty_ptr = torch.zeros_like(indptr)
            for split in (None, 1, 2, 4, 8):
                name = "aiter_opus_bf16_" + ("default" if split is None else str(split))
                calls[name] = partial(
                    pa_sparse_prefill_opus,
                    q_bf16,
                    kv_bf16,
                    ids.flatten(),
                    indptr,
                    kv_bf16[:0],
                    empty_ids,
                    empty_ptr,
                    sink,
                    512**-0.5,
                    num_kv_splits=split,
                )
            calls["aiter_opus_fp8"] = partial(
                pa_sparse_prefill_fp8_opus,
                q_nope,
                q_rope,
                kv_nope,
                kv_rope,
                ids.flatten(),
                indptr,
                kv_nope[:0],
                kv_rope[:0],
                empty_ids,
                empty_ptr,
                sink,
                512**-0.5,
            )
            candidates, failures = {}, {}
            for name, call in calls.items():
                try:
                    graph, out = capture(call)
                    check = accuracy(out, expected, rows, 0.04)
                    timing = measure({name: graph}, 9, flush)[name]
                    candidates[name] = {"correctness": check, **timing}
                    del graph, out
                except Exception as exc:  # noqa: BLE001
                    # Record unsupported compiler/configuration candidates.
                    failures[name] = str(exc)
                    print("CANDIDATE_FAILED", T, K, name, str(exc)[:300], flush=True)
                gc.collect()
            assert "hk" in candidates, failures
            ranked = sorted(
                (n for n in candidates if n != "hk"),
                key=lambda n: candidates[n]["median_us"],
            )
            assert ranked, failures
            selected = list(
                dict.fromkeys(
                    ["hk"]
                    + ranked[:3]
                    + [
                        n for n in ranked if n.startswith(("aiter_fp8_", "aiter_bf16_"))
                    ][:1]
                    + [n for n in ranked if n.startswith("aiter_opus_")][:1]
                    + [
                        n
                        for n in ("aiter_fp8_default", "aiter_bf16_default")
                        if n in candidates
                    ]
                )
            )
            graphs, outputs = {}, {}
            for name in selected:
                graphs[name], outputs[name] = capture(calls[name])
            phases = []
            for seed in (1701, 2701):
                torch.manual_seed(seed + T)
                q.copy_((torch.randn(T, 32, 512, device="cuda") * 0.4).to(kv.dtype))
                ids.copy_(
                    torch.randint(slots, (T, K), device="cuda", dtype=torch.int32)
                )
                q_bf16.copy_(q)
                q_nope[:, :, :448].copy_(q[:, :, :448])
                q_rope.copy_(q[:, :, 448:])
                expected = tests.reference(q, kv[:, None, :], ids, 512**-0.5, rows)
                for graph in graphs.values():
                    graph.replay()
                torch.cuda.synchronize()
                checks = {
                    name: accuracy(out, expected, rows, 0.04)
                    for name, out in outputs.items()
                }
                phases.append(
                    {
                        "seed": seed,
                        "correctness": checks,
                        "timing": measure(graphs, 21, flush),
                    }
                )
            medians = {
                name: statistics.median(p["timing"][name]["median_us"] for p in phases)
                for name in selected
            }
            winner = min((n for n in selected if n != "hk"), key=medians.get)
            matching = min(
                (n for n in selected if n.startswith(("aiter_fp8_", "aiter_bf16_"))),
                key=medians.get,
            )
            row = {
                "tokens": T,
                "topk": K,
                "pool_bytes": kv.numel(),
                "candidates": candidates,
                "candidate_failures": failures,
                "phases": phases,
                "median_us": medians,
                "strongest_baseline": winner,
                "strongest_matching_layout": matching,
                "hk_speedup_matching_layout": medians[matching] / medians["hk"],
                "hk_speedup": medians[winner] / medians["hk"],
            }
            result["cases"].append(row)
            OUT.write_text(json.dumps(result, indent=2) + "\n")
            print(
                "SPARSE_RESULT",
                json.dumps(
                    {k: v for k, v in row.items() if k not in ("candidates", "phases")}
                ),
                flush=True,
            )
            del calls, graphs, outputs, q, q_bf16, q_nope, q_rope, ids, expected
            gc.collect()
            torch.cuda.empty_cache()
        del kv, kv_bf16, kv_nope, kv_rope
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("hk_sparse_benchmark.json"))
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
    result = sparse()
    OUT.write_text(json.dumps(result, indent=2) + "\n")
