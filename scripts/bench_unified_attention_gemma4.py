# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Isolated gfx942 FP8 unified-attention A/B benchmark for Gemma-4-31B.

Tier A uses main's shipped JSON unchanged. Tier B uses the isolated config table
from PR #5650 head 184aae9498c1de3d6f60c0a2ba8b9338907ffc78, plus only the
D>=512 attn_3d decode waves_per_eu=1 from PR #5664 head
0743fc1f5f7ca1a50e6c36df43ed2a25d03818f9 (no #5664 prefill changes).
The #5650 shuffled-page >128 guard is active in both tiers; all measured pages
are <=128, so the guard does not change Tier A behavior for measured shapes.
This script changes only the config loader's root before any config lookup,
leaving the production DEFAULT.json intact.
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
TRITON_DEFAULT_CONFIGS_PATH = config_utils.AITER_TRITON_CONFIGS_PATH


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


def sliding_specs(smoke=False, page_sizes=(32, 64)):
    specs = []
    prefill_contexts = (1024,) if smoke else (1024, 4096, 16384, 32768)
    decode_batches = (1,) if smoke else (1, 4, 16, 64)
    decode_contexts = (1024,) if smoke else (1024, 4096, 16384, 32768)
    for page_size in page_sizes:
        for context in prefill_contexts:
            specs.append({
                "name": f"prefill-{context}", "mode": "prefill", "batch": 1,
                "query_lens": [context], "kv_lens": [context],
                "context": context, "page_size": page_size,
            })
        for batch, context in itertools.product(decode_batches, decode_contexts):
            specs.append({
                "name": f"decode-b{batch}-context{context}", "mode": "decode",
                "batch": batch, "query_lens": [1] * batch,
                "kv_lens": [context] * batch, "context": context,
                "page_size": page_size,
            })
        specs.append({
            "name": "mixed", "mode": "mixed", "batch": 6,
            "query_lens": [1, 17, 65, 1, 33, 1],
            "kv_lens": [8192, 1041, 4097, 31, 97, 2049],
            "context": 8192, "page_size": page_size,
        })
    return specs


def _legacy_flydsl_specs(smoke=False):
    cases = (
        [("sliding", "prefill", 1, 1024), ("full", "decode", 4, 1024)]
        if smoke else
        [(kind, "prefill", 1, n) for kind, n in itertools.product(
            ("sliding", "full"), (1024, 4096, 16384, 32768)
        )] + [(kind, "decode", b, n) for kind, b, n in itertools.product(
            ("sliding", "full"), (1, 4, 16, 64), (1024, 4096, 16384, 32768)
        )]
    )
    specs = []
    for kind, mode, batch, context in cases:
        for layout in ("plain", "shuffled"):
            if kind != "sliding" or layout != "plain":
                print(
                    f"SKIP unsupported FlyDSL row: {kind}/{mode}/batch{batch}/"
                    f"context{context}/{layout}", flush=True
                )
                continue
            q_lens = [context] * batch if mode == "prefill" else [1] * batch
            specs.append({
                "name": f"{mode}-b{batch}-context{context}", "mode": mode,
                "batch": batch, "query_lens": q_lens,
                "kv_lens": [context] * batch, "context": context,
                "page_size": 32,
            })
    specs.extend(sliding_specs(smoke=smoke, page_sizes=(64,))[-1:])
    return specs


def _make_case(spec):
    page = spec["page_size"]
    query_lens, kv_lens = spec["query_lens"], spec["kv_lens"]
    batch, dim, heads_kv = len(kv_lens), 256, 16
    page_counts = [math.ceil(length / page) for length in kv_lens]
    n_pages = sum(page_counts)
    max_pages = max(page_counts)
    seed = 1709 + page * 100003 + sum(query_lens) * 17 + sum(kv_lens)
    generator = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(
        sum(query_lens), 32, dim, device="cuda", generator=generator
    ).to(torch.float8_e4m3fnuz)
    k = torch.randn(
        n_pages, page, heads_kv, dim, device="cuda", generator=generator
    ).to(torch.float8_e4m3fnuz)
    v = torch.randn(
        n_pages, page, heads_kv, dim, device="cuda", generator=generator
    ).to(torch.float8_e4m3fnuz)
    block_table = torch.zeros(batch, max_pages, device="cuda", dtype=torch.int32)
    offset = 0
    for row, count in enumerate(page_counts):
        block_table[row, :count] = torch.arange(
            offset, offset + count, device="cuda", dtype=torch.int32
        )
        offset += count
    cu_q = torch.tensor(
        [0] + list(itertools.accumulate(query_lens)), device="cuda", dtype=torch.int32
    )
    used_k = torch.tensor(kv_lens, device="cuda", dtype=torch.int32)
    scales = tuple(torch.ones(1, device="cuda", dtype=torch.float32) for _ in range(3))
    return {
        "q": q, "k": k, "v": v,
        "cu_q": cu_q, "used_k": used_k, "block_table": block_table,
        "scales": scales, "max_q": max(query_lens),
        "max_k": max(kv_lens), "query_lens": query_lens,
        "kv_lens": kv_lens, "spec": spec,
    }


def _positional(case, out):
    return (
        case["q"], case["k"], case["v"], out, case["cu_q"], case["used_k"],
        case["block_table"], *case["scales"],
    )


def _triton_call(case, out):
    dim = 256
    q_len, context = case["max_q"], case["max_k"]

    def call(q, k, v, output, cu_q, used_k, table, q_scale, k_scale, v_scale):
        return unified_attention(
            q, k, v, output, cu_q, q_len, used_k, context, dim ** -0.5,
            True, (1023, 0), table, 0.0, q_scale, k_scale, v_scale,
            shuffled_kv_cache=False, backend="triton",
        )

    return call, _positional(case, out)


def _flydsl_call(case, out, force_splits=None):
    from aiter.ops.flydsl.kernels.flash_attn_fp8_gfx942 import (
        build_flash_attn_fp8_gfx942,
    )

    launch = build_flash_attn_fp8_gfx942(case["spec"]["page_size"])
    q_len, context = case["max_q"], case["max_k"]

    def call(q, k, v, output, cu_q, used_k, table, q_scale, k_scale, v_scale):
        return launch(
            q, k, v, output,
            cu_seqlens_q=cu_q,
            seqused_k=used_k,
            max_seqlen_q=q_len,
            max_seqlen_k=context,
            block_table=table,
            softmax_scale=256 ** -0.5,
            q_descale=q_scale,
            k_descale=k_scale,
            v_descale=v_scale,
            window_size=(1023, 0),
            causal=True,
            softcap=0,
            _force_splits=force_splits,
        )

    return call, _positional(case, out)


def _timing_row(case, backend, tier, iterations, force_splits=None):
    out = torch.empty(
        case["q"].shape, device="cuda", dtype=torch.bfloat16
    )
    call, positional = (
        _flydsl_call(case, out, force_splits) if backend == "flydsl" else _triton_call(case, out)
    )
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
        raise RuntimeError(f"{backend} attention output is zero for {case['spec']['name']}")
    spec = case["spec"]
    return {
        "tier": tier, "attention": "sliding", "mode": spec["mode"],
        "layout": "plain", "batch": spec["batch"],
        "query_length": case["max_q"], "context": spec["context"],
        "num_q_heads": 32, "num_kv_heads": 16, "head_dim": 256,
        "window": 1024, "page_size": spec["page_size"],
        "rotations": copies, "footprint_bytes": footprint, "us": latency_us,
    }


def _check_flydsl(case):
    ref = torch.empty(case["q"].shape, device="cuda", dtype=torch.bfloat16)
    got = torch.empty_like(ref)
    triton, triton_args = _triton_call(case, ref)
    flydsl, flydsl_args = _flydsl_call(case, got)
    triton(*triton_args)
    flydsl(*flydsl_args)
    torch.cuda.synchronize()
    max_ref = ref.float().abs().max().item()
    max_abs = (got.float() - ref.float()).abs().max().item()
    tolerance = 0.08 * max_ref
    passed = max_abs <= tolerance
    print(
        f"correctness {case['spec']['name']} page={case['spec']['page_size']}: "
        f"{'PASS' if passed else 'FAIL'} max_abs={max_abs:.8g} "
        f"tolerance={tolerance:.8g} max_ref={max_ref:.8g}",
        flush=True,
    )
    if not passed:
        raise AssertionError(
            f"FlyDSL failed Triton tier A comparison for {case['spec']['name']} "
            f"page={case['spec']['page_size']}: {max_abs} > {tolerance}"
        )


def _select_triton_tier(tier):
    if tier == "B":
        config_utils.AITER_TRITON_CONFIGS_PATH = str(ROOT / "benchmark_configs")
    else:
        config_utils.AITER_TRITON_CONFIGS_PATH = TRITON_DEFAULT_CONFIGS_PATH
    config_utils.load_config_json.cache_clear()


def run_sliding_suite(args, specs, outputs):
    if args.backend in ("flydsl", "all"):
        _select_triton_tier("A")
        for spec in specs:
            case = _make_case(spec)
            _check_flydsl(case)
            del case
        torch.cuda.empty_cache()

    writers = {}
    files = {}
    for name, path in outputs.items():
        file = Path(path)
        file.parent.mkdir(parents=True, exist_ok=True)
        handle = file.open("w", newline="")
        files[name] = handle
        writers[name] = None

    try:
        for index, spec in enumerate(specs):
            case = _make_case(spec)
            if args.backend == "all":
                order = ("A", "B", "flydsl") if index % 2 == 0 else ("flydsl", "B", "A")
                for backend in order:
                    if backend == "flydsl":
                        _select_triton_tier("A")
                        tier_name, file_key = "flydsl", "flydsl"
                    else:
                        _select_triton_tier(backend)
                        tier_name, file_key = backend, f"triton-{backend.lower()}"
                    row = _timing_row(case, "flydsl" if backend == "flydsl" else "triton", tier_name, args.iterations)
                    if writers[file_key] is None:
                        writers[file_key] = csv.DictWriter(files[file_key], fieldnames=list(row))
                        writers[file_key].writeheader()
                    writers[file_key].writerow(row)
                    files[file_key].flush()
                    print(row, flush=True)
            else:
                backend = args.backend
                _select_triton_tier(args.tier)
                tier_name = "flydsl" if backend == "flydsl" else args.tier
                row = _timing_row(case, backend, tier_name, args.iterations, args.force_splits)
                key = backend
                if writers[key] is None:
                    writers[key] = csv.DictWriter(files[key], fieldnames=list(row))
                    writers[key].writeheader()
                writers[key].writerow(row)
                files[key].flush()
                print(row, flush=True)
            del case
            torch.cuda.empty_cache()
    finally:
        for handle in files.values():
            handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("A", "B"), default="A")
    parser.add_argument("--backend", choices=("triton", "flydsl", "all"), default="triton")
    parser.add_argument("--suite", choices=("legacy", "sliding-d256"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", help="output CSV path for one backend")
    parser.add_argument("--output-dir", help="output directory when --backend all")
    parser.add_argument("--iterations", type=int, default=101)
    parser.add_argument("--case", help="run one named sliding-d256 case")
    parser.add_argument("--force-splits", type=int, choices=range(1, 17), default=None,
                        help="test-only FlyDSL decode split override")
    parser.add_argument("--page-size", type=int, choices=(32, 64), default=None)
    parser.add_argument(
        "--profile-once", action="store_true",
        help="launch one selected FlyDSL case without correctness or timing (after validation)",
    )
    args = parser.parse_args()
    args.suite = args.suite or ("legacy" if args.backend == "triton" else "sliding-d256")
    global SMOKE
    SMOKE = args.smoke
    if arch_info.get_arch() != "gfx942":
        raise RuntimeError("gfx942 GPU required")

    if args.profile_once:
        if args.backend != "flydsl" or args.suite != "sliding-d256":
            parser.error("--profile-once requires --backend flydsl --suite sliding-d256")
        if not args.case or args.page_size is None:
            parser.error("--profile-once requires --case and --page-size")
        specs = [
            spec for spec in sliding_specs(args.smoke, (args.page_size,))
            if spec["name"] == args.case
        ]
        if len(specs) != 1:
            parser.error(f"selected case not found: {args.case} page={args.page_size}")
        case = _make_case(specs[0])
        out = torch.empty(case["q"].shape, device="cuda", dtype=torch.bfloat16)
        call, positional = _flydsl_call(case, out, args.force_splits)
        call(*positional)
        torch.cuda.synchronize()
        print(f"profile dispatch: FlyDSL {args.case} page={args.page_size}", flush=True)
        return

    if args.backend == "all":
        if args.suite != "sliding-d256":
            parser.error("--backend all requires --suite sliding-d256")
        if not args.output_dir or args.output:
            parser.error("--backend all requires --output-dir and does not accept --output")
        outputs = {
            "triton-a": str(Path(args.output_dir) / "gemma4-triton-a-d256.csv"),
            "triton-b": str(Path(args.output_dir) / "gemma4-triton-b-d256.csv"),
            "flydsl": "/tmp/gemma4-flydsl942-d256.csv",
        }
        run_sliding_suite(args, sliding_specs(args.smoke), outputs)
        return

    if not args.output:
        parser.error("--output is required for a single backend")
    if args.suite == "sliding-d256":
        specs = sliding_specs(args.smoke, (args.page_size,) if args.page_size else (32, 64))
        if args.case:
            specs = [spec for spec in specs if spec["name"] == args.case]
            if not specs:
                parser.error(f"case not found: {args.case}")
        run_sliding_suite(args, specs, {args.backend: args.output})
        return

    if args.backend == "flydsl":
        parser.error("FlyDSL supports the sliding-d256 suite, not the legacy suite")

    _select_triton_tier(args.tier)
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
