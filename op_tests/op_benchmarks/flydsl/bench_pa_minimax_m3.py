# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""MiniMax-M3 TP4 paged attention benchmark; see README_pa_minimax_m3.md."""

import argparse
import csv
import fnmatch
import hashlib
import importlib
import inspect
import json
import math
import os
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HEADS, KV_HEADS, DIM = 16, 1, 128
TOPK_TOKENS, SPARSE_POOL_PAGES, CAPTURE_TOKENS = 2048, 4096, 256 * 1024
SUITES = (
    "dense-decode",
    "dense-verify",
    "sparse-decode",
    "sparse-verify",
    "sparse-prefill",
)


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def case_matrix():
    cases = []

    def add(suite, batch, qlen, context, requests=None):
        cases.append(
            {
                "id": f"{suite}-b{batch}-q{qlen}-ctx{context}",
                "suite": suite,
                "batch": batch,
                "query_length": qlen,
                "context": context,
                "requests": requests or batch,
                "page": 128 if suite.startswith("dense") else 16,
            }
        )

    for batch, context in (
        (1, 8192),
        (4, 8192),
        (8, 8192),
        (32, 8192),
        (4, 32768),
        (3, 220 * 1024),
    ):
        add("dense-decode", batch, 1, context)
    for batch, context in ((3, 8192), (3, 32768), (8, 32768), (3, 220 * 1024)):
        add("dense-verify", batch, 4, context)
    for rows in (4, 64, 128, 256):
        add("sparse-decode", rows, 1, TOPK_TOKENS)
    # Each sparse verify token has independently selected KV: 3 requests x 4
    # tokens become 12 PA sequences, each with query_length=1.
    add("sparse-verify", 12, 1, TOPK_TOKENS, requests=3)
    for rows in (1024, 4096, 8192):
        add("sparse-prefill", rows, 1, TOPK_TOKENS, requests=1)
    for rows in (239, 240, 241):
        add("cluster-boundary", rows, 1, TOPK_TOKENS)
    return cases


def git_info(source):
    def git(*args):
        return subprocess.run(
            ["git", "-C", str(source), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def logical_traffic(case, lengths, dtype):
    """Count logical tensor IO once per sequence, including reuse across rows."""
    tokens = sum(lengths)
    kv_bytes = 2 * tokens * KV_HEADS * DIM * (1 if dtype == "fp8" else 2)
    scale_bytes = 2 * tokens * KV_HEADS * 4 if dtype == "fp8" else 0
    # BF16 Q read and BF16 output write. Dense MTP queries share their KV;
    # count that KV once per request rather than once per query token.
    query_output_bytes = 2 * case["batch"] * case["query_length"] * HEADS * DIM * 2
    return {
        "logical_kv_tokens": tokens,
        "kv_payload_bytes": kv_bytes,
        "kv_scale_bytes": scale_bytes,
        "query_output_bytes": query_output_bytes,
        "logical_io_bytes": kv_bytes + scale_bytes + query_output_bytes,
    }


def build_inputs(case, dtype, seed, torch):
    torch.manual_seed(seed)
    batch, qlen, page = case["batch"], case["query_length"], case["page"]
    if case["suite"] == "sparse-prefill":
        lengths = [min(row + 1, TOPK_TOKENS) for row in range(batch)]
    else:
        lengths = [case["context"]] * batch
    width = math.ceil(max(lengths) / page)
    sparse = not case["suite"].startswith("dense")
    if sparse:
        width = TOPK_TOKENS // page
        pages = SPARSE_POOL_PAGES
        # Select 16 logical 128-token blocks without replacement, each backed
        # by eight consecutive physical page16 entries, from a shared KV pool.
        blocks = torch.rand(batch, pages // 8, device="cuda").argsort(dim=1)[:, :16]
        table = (
            (blocks[:, :, None] * 8 + torch.arange(8, device="cuda"))
            .reshape(batch, width)
            .to(torch.int32)
        )
    else:
        pages = batch * width
        table = torch.randperm(pages, device="cuda", dtype=torch.int32).reshape(
            batch, width
        )
    query = torch.empty((batch * qlen, HEADS, DIM), dtype=torch.bfloat16, device="cuda")
    query.uniform_(-0.5, 0.5)
    key = torch.empty((pages, KV_HEADS, page, DIM), dtype=torch.bfloat16, device="cuda")
    key.uniform_(-0.5, 0.5)
    value = torch.empty_like(key).uniform_(-0.5, 0.5)
    # Distinct per-token magnitudes make misplaced FP8 scale indices observable.
    key.mul_(torch.randn_like(key[..., :1]).mul_(0.4).exp_())
    value.mul_(torch.randn_like(value[..., :1]).mul_(0.4).exp_())
    ks = vs = None
    if dtype == "fp8":
        from aiter import pertoken_quant

        key, ks = pertoken_quant(key, quant_dtype=torch.float8_e4m3fn)
        value, vs = pertoken_quant(value, quant_dtype=torch.float8_e4m3fn)
    vector = 16 if dtype == "fp8" else 8
    kc = key.reshape(pages, KV_HEADS, page, DIM // vector, vector)
    kc = kc.permute(0, 1, 3, 2, 4).contiguous()
    vc = value.reshape(pages, KV_HEADS, page // vector, vector, DIM)
    vc = vc.permute(0, 1, 2, 4, 3).contiguous()
    return {
        "query": query,
        "key": key,
        "value": value,
        "key_cache": kc,
        "value_cache": vc,
        "key_scale": ks,
        "value_scale": vs,
        "table": table,
        "context": torch.tensor(lengths, device="cuda", dtype=torch.int32),
        "lengths": lengths,
    }


def reference(case, inputs, torch):
    """FP32 attention over dequantized logical KV, with independent causal masks."""
    qlen, page = case["query_length"], case["page"]
    queries = inputs["query"].float().reshape(case["batch"], qlen, HEADS, DIM)
    result = torch.empty_like(queries)
    # Bound reference memory, including long dense contexts and 8192 sparse rows.
    chunk = max(1, min(64, 2**20 // case["context"]))
    for begin in range(0, case["batch"], chunk):
        end = min(case["batch"], begin + chunk)
        lengths = inputs["context"][begin:end]
        tokens = torch.arange(max(inputs["lengths"][begin:end]), device="cuda")
        pages = inputs["table"][begin:end, tokens // page].long()
        offsets = tokens % page
        key = inputs["key"][pages, 0, offsets].float()
        value = inputs["value"][pages, 0, offsets].float()
        if inputs["key_scale"] is not None:
            key *= inputs["key_scale"][pages, 0, offsets]
            value *= inputs["value_scale"][pages, 0, offsets]
        scores = torch.einsum("bqhd,bkd->bqhk", queries[begin:end], key) * DIM**-0.5
        visible = lengths[:, None] - qlen + 1 + torch.arange(qlen, device="cuda")
        scores.masked_fill_(
            tokens[None, None, None, :] >= visible[:, :, None, None], -math.inf
        )
        result[begin:end] = torch.einsum(
            "bqhk,bkd->bqhd", scores.softmax(dim=-1), value
        )
    return result.reshape_as(inputs["query"])


def check_output(output, expected, torch, dtype):
    actual = output.float()
    if not torch.isfinite(actual).all():
        raise AssertionError("attention output contains NaN or infinity")
    delta = actual - expected
    row_error = delta.flatten(1).norm(dim=1) / expected.flatten(1).norm(
        dim=1
    ).clamp_min(1e-12)
    if dtype == "fp8":
        # FP8 Q/P arithmetic has ~2-3% error even against dequantized KV on
        # main. Short prefill rows amplify absolute error; check every row.
        if row_error.max().item() > 0.05:
            raise AssertionError(
                f"FP8 row relative L2 error {row_error.max().item():.5f} exceeds 0.05"
            )
    else:
        torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-3)
    return {
        "max_abs_error": delta.abs().max().item(),
        "relative_l2_error": (delta.norm() / expected.norm().clamp_min(1e-12)).item(),
        "max_row_relative_l2_error": row_error.max().item(),
    }


def measure(case, dtype, profile, args, pa, torch):
    inputs = build_inputs(case, dtype, args.seed, torch)
    table = inputs["table"]
    sparse = not case["suite"].startswith("dense")
    cap, bound = None, max(inputs["lengths"])
    if profile == "atom":
        if sparse:
            from aiter.ops.triton.gluon.pa_decode_gluon import get_recommended_splits

            cap = get_recommended_splits(case["batch"], KV_HEADS)
            if case["suite"] == "sparse-prefill":
                cap = 1
            bound = TOPK_TOKENS
        else:
            table = torch.nn.functional.pad(
                table, (0, CAPTURE_TOKENS // case["page"] - table.shape[1])
            )
            bound = None
    if case["suite"] == "cluster-boundary":
        # Pin two partitions: the legacy ATOM heuristic returns three at
        # 239-241 rows, which already exceeds the CTA residency budget.
        cap = 2
    # PR profile uses default plan cap/budget and a tight live context bound.
    plan_options = {}
    if (
        case["batch"] > 4096
        and "max_batch_size" in inspect.signature(pa.plan_pa_decode).parameters
    ):
        plan_options["max_batch_size"] = case["batch"]
    plan = pa.plan_pa_decode(
        inputs["context"],
        KV_HEADS,
        query_length=case["query_length"],
        max_partitions=cap,
        workgroup_budget=args.workgroup_budget,
        **plan_options,
    )
    shape = (KV_HEADS, plan.capacity, case["query_length"] * HEADS)
    workspace = {
        "exp_sums": torch.empty(shape, device="cuda", dtype=torch.float32),
        "max_logits": torch.empty(shape, device="cuda", dtype=torch.float32),
        "temporary_output": torch.empty(
            (*shape, DIM), device="cuda", dtype=torch.bfloat16
        ),
    }
    output = torch.full_like(inputs["query"], math.nan)

    def decode():
        pa.pa_decode(
            output,
            inputs["query"],
            inputs["key_cache"],
            inputs["value_cache"],
            inputs["context"],
            table,
            DIM**-0.5,
            case["query_length"],
            compute_type=inputs["key"].dtype,
            key_scale=inputs["key_scale"],
            value_scale=inputs["value_scale"],
            work_plan=plan,
            max_context_length=bound,
            **workspace,
        )

    policies = []
    original = pa.compile_pa_decode_tile

    def spy(**kwargs):
        policies.append(kwargs)
        return original(**kwargs)

    pa.compile_pa_decode_tile = spy
    try:
        decode()
    finally:
        pa.compile_pa_decode_tile = original
    expected = reference(case, inputs, torch) if args.check else None
    errors = check_output(output, expected, torch, dtype) if args.check else {}
    for _ in range(args.warmup):
        decode()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.graph_calls):
            decode()
    for _ in range(args.warmup):
        graph.replay()
    torch.cuda.synchronize()
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    rounds = []
    all_samples = []
    for _ in range(args.rounds):
        samples = []
        for _ in range(args.samples):
            start.record()
            for _ in range(args.replays):
                graph.replay()
            stop.record()
            stop.synchronize()
            samples.append(
                start.elapsed_time(stop) * 1000 / args.replays / args.graph_calls
            )
        rounds.append(min(samples))
        all_samples.append(samples)
    if args.check:
        check_output(output, expected, torch, dtype)
        # Replay must read current Q and write every output row, including the
        # short causal prefill rows; a stale captured result must fail the check.
        inputs["query"].neg_()
        output.fill_(math.nan)
        graph.replay()
        check_output(output, reference(case, inputs, torch), torch, dtype)
    policy = policies[-1]
    cluster = bool(policy.get("cluster_merge", False))
    direct = bool(policy.get("direct_output", False))
    mode = "cluster" if cluster else "direct" if direct else "reducer"
    latency_us = statistics.median(rounds)
    traffic = logical_traffic(case, inputs["lengths"], dtype)
    return dict(
        case=case["id"],
        suite=case["suite"],
        profile=profile,
        kv_dtype=dtype,
        query_heads=HEADS,
        kv_heads=KV_HEADS,
        head_dim=DIM,
        query_dtype="bf16",
        query_shape=list(output.shape),
        batch=case["batch"],
        requests=case["requests"],
        query_length=case["query_length"],
        page=case["page"],
        context_min=min(inputs["lengths"]),
        context_max=max(inputs["lengths"]),
        table_width=table.shape[1],
        kv_pool_pages=inputs["key"].shape[0],
        max_context_length=bound,
        plan_cap=plan.max_partitions,
        plan_capacity=plan.capacity,
        workgroup_budget=args.workgroup_budget,
        output_mode=mode,
        scratch_bytes=sum(t.numel() * t.element_size() for t in workspace.values()),
        median_us=latency_us,
        effective_bandwidth_tb_s=traffic["logical_io_bytes"] / latency_us / 1e6,
        **traffic,
        round_min_us=rounds,
        samples_us=all_samples,
        checked=args.check,
        reference_check=(
            "max row relative L2 <= 0.05" if dtype == "fp8" else "rtol=atol=0.005"
        ),
        graph_changed_query_checked=args.check,
        **errors,
    )


def write_results(path, metadata, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2, allow_nan=False)
        + "\n"
    )
    temporary.replace(path)
    fields = [
        key
        for key in rows[0]
        if key not in ("samples_us", "round_min_us", "query_shape")
    ]
    with path.with_suffix(".csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compare(path, metadata, rows):
    baseline = json.loads(path.read_text())
    if baseline["metadata"].get("status") != "complete":
        raise ValueError("baseline run is incomplete")
    for field in (
        "gpu",
        "torch",
        "flydsl",
        "hip",
        "seed",
        "timing",
        "workgroup_budget",
        "environment",
        "harness_sha256",
    ):
        if baseline["metadata"][field] != metadata[field]:
            raise ValueError(
                f"baseline {field} differs; use the same machine, harness and settings"
            )
    indexed = {(r["case"], r["profile"], r["kv_dtype"]): r for r in baseline["results"]}
    for row in rows:
        key = (row["case"], row["profile"], row["kv_dtype"])
        if key not in indexed:
            raise ValueError(f"baseline has no result for {key}")
        old = indexed[key]
        for field in (
            "table_width",
            "kv_pool_pages",
            "max_context_length",
            "query_shape",
            "checked",
        ):
            if old[field] != row[field]:
                raise ValueError(f"baseline {field} differs for {key}")
        row["baseline_us"] = old["median_us"]
        row["speedup"] = old["median_us"] / row["median_us"]
        print(
            f"{row['profile']:4} {row['kv_dtype']:4} {row['case']:42} {row['speedup']:.3f}x"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("atom", "pr", "both"), default="atom")
    parser.add_argument("--dtype", choices=("bf16", "fp8", "both"), default="both")
    parser.add_argument(
        "--suite", nargs="+", choices=(*SUITES, "cluster-boundary"), default=SUITES
    )
    parser.add_argument(
        "--case", action="append", help="case ID/glob; repeat to select several"
    )
    parser.add_argument(
        "--list", action="store_true", help="list cases without GPU imports"
    )
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument("--label", default="minimax-m3")
    parser.add_argument("--output", type=Path, default=Path("pa_minimax_m3.json"))
    parser.add_argument(
        "--compare-to", type=Path, help="baseline JSON with identical settings"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="visible PyTorch device index"
    )
    parser.add_argument("--seed", type=int, default=37)
    parser.add_argument("--workgroup-budget", type=positive_int, default=None)
    parser.add_argument("--graph-calls", type=positive_int, default=10)
    parser.add_argument("--replays", type=positive_int, default=5)
    parser.add_argument("--samples", type=positive_int, default=5)
    parser.add_argument("--rounds", type=positive_int, default=3)
    parser.add_argument("--warmup", type=positive_int, default=5)
    parser.add_argument(
        "--no-check",
        dest="check",
        action="store_false",
        help="skip FP32 and graph correctness checks",
    )
    args = parser.parse_args()
    if args.output.suffix != ".json":
        parser.error(
            "--output must have a .json suffix; a matching CSV is also written"
        )
    if args.compare_to and args.compare_to.resolve() == args.output.resolve():
        parser.error("--output must differ from --compare-to")
    cases = [c for c in case_matrix() if c["suite"] in args.suite]
    if args.case:
        for pattern in args.case:
            if not any(fnmatch.fnmatchcase(c["id"], pattern) for c in cases):
                parser.error(f"case pattern matched no selected case: {pattern}")
        cases = [
            c for c in cases if any(fnmatch.fnmatchcase(c["id"], p) for p in args.case)
        ]
    profiles = ("atom", "pr") if args.profile == "both" else (args.profile,)
    dtypes = ("bf16", "fp8") if args.dtype == "both" else (args.dtype,)
    if args.list:
        for profile in profiles:
            for dtype in dtypes:
                for case in cases:
                    print(f"{profile:4} {dtype:4} {case['id']}")
        return
    if args.device < 0:
        parser.error("--device must be nonnegative")
    source = args.source.resolve()
    if not (source / "aiter/ops/flydsl/pa_decode.py").is_file():
        parser.error(f"--source is not an AITER checkout: {source}")
    sys.path.insert(0, str(source))
    import flydsl
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("an available ROCm GPU is required")
    torch.cuda.set_device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = False
    props = torch.cuda.get_device_properties(args.device)
    if props.gcnArchName.split(":")[0] != "gfx1250":
        raise RuntimeError("this PR 7079 benchmark targets gfx1250")
    pa = importlib.import_module("aiter.ops.flydsl.pa_decode")
    if not Path(pa.__file__).resolve().is_relative_to(source):
        raise RuntimeError(
            f"AITER was imported from {pa.__file__}, not --source {source}"
        )
    # All allocations/reference/tuning/compilation stay outside timed graphs.
    metadata = dict(
        label=args.label,
        source=str(source),
        **git_info(source),
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        gpu={
            "name": props.name,
            "arch": props.gcnArchName,
            "cus": props.multi_processor_count,
            "uuid": str(props.uuid),
            "total_memory": props.total_memory,
        },
        torch=torch.__version__,
        flydsl=flydsl.__version__,
        hip=torch.version.hip,
        seed=args.seed,
        workgroup_budget=args.workgroup_budget,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        timing={
            "graph_calls": args.graph_calls,
            "replays": args.replays,
            "samples": args.samples,
            "rounds": args.rounds,
            "warmup": args.warmup,
            "statistic": "median of round minima; GPU events; microseconds per complete decode",
        },
        environment={
            key: os.environ.get(key)
            for key in (
                "AITER_FLYDSL_KERNARG_PRELOAD",
                "FLYDSL_AUTOTUNE",
                "FLYDSL_RUNTIME_CACHE_DIR",
            )
        },
        checked=args.check,
        expected_results=len(cases) * len(profiles) * len(dtypes),
        bandwidth_definition=(
            "logical K/V payload + FP32 per-token K/V scales for FP8 + BF16 Q read "
            "+ BF16 output write; KV once per sequence; excludes metadata/scratch; "
            "decimal TB/s, not measured HBM traffic"
        ),
        status="running",
    )
    rows = []
    with torch.inference_mode():
        for profile in profiles:
            for dtype in dtypes:
                for case in cases:
                    row = measure(case, dtype, profile, args, pa, torch)
                    rows.append(row)
                    write_results(args.output, metadata, rows)
                    print(
                        f"{profile:4} {dtype:4} {case['id']:42} "
                        f"{row['median_us']:9.3f} us  "
                        f"{row['effective_bandwidth_tb_s']:7.3f} TB/s  {row['output_mode']}",
                        flush=True,
                    )
    if args.compare_to:
        compare(args.compare_to, metadata, rows)
        metadata["baseline"] = str(args.compare_to.resolve())
        metadata["baseline_commit"] = json.loads(args.compare_to.read_text())[
            "metadata"
        ]["commit"]
    metadata["status"] = "complete"
    write_results(args.output, metadata, rows)
    print(
        f"Saved {len(rows)} results to {args.output} and {args.output.with_suffix('.csv')}"
    )


if __name__ == "__main__":
    main()
