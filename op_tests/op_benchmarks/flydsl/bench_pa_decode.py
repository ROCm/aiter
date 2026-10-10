# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark FlyDSL PA decode with random data and variable context lengths.

Run from the repository root with:
    python -m op_tests.op_benchmarks.flydsl.bench_pa_decode --help
"""

import argparse
import csv
import itertools
import json
import math
import random
import statistics
from importlib.metadata import version
from pathlib import Path

import torch


def context_lengths(batch, minimum, maximum, distribution, empty_fraction, seed):
    """Sample token counts without rounding them to pages or compute tiles."""
    rng = random.Random(seed)
    if distribution == "uniform":
        lengths = [rng.randint(minimum, maximum) for _ in range(batch)]
    elif distribution == "log-uniform":
        low, high = math.log1p(minimum), math.log1p(maximum)
        lengths = [round(math.expm1(rng.uniform(low, high))) for _ in range(batch)]
    else:
        width = (maximum - minimum) // 8
        lengths = [
            (
                rng.randint(minimum, minimum + width)
                if rng.random() < 0.5
                else rng.randint(maximum - width, maximum)
            )
            for _ in range(batch)
        ]
    if batch > 1:
        lengths[:2] = [minimum, maximum]
    for index in rng.sample(range(batch), round(batch * empty_fraction)):
        lengths[index] = 0
    rng.shuffle(lengths)
    return lengths


def random_data(shape, generator, args, device):
    result = torch.empty(shape, dtype=torch.float32, device=device)
    if args.data_init == "uniform":
        return result.uniform_(-0.5, 0.5, generator=generator)
    return result.normal_(0, 0.5, generator=generator)


def make_inputs(lengths, args, kv_dtype, device):
    page, heads, dim = args.page_size, args.kv_heads, args.head_dim
    counts = [(length + page - 1) // page for length in lengths]
    num_pages = max(1, sum(counts))
    columns = max(1, max(counts))
    chunk = 16 // torch.empty((), dtype=kv_dtype).element_size()
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)
    permutation = torch.randperm(num_pages, generator=generator, device=device)
    table = torch.zeros((len(lengths), columns), dtype=torch.int32, device=device)
    offset = 0
    for seq, count in enumerate(counts):
        table[seq, :count] = permutation[offset : offset + count].to(torch.int32)
        offset += count

    query_generator = torch.Generator(device=device).manual_seed(args.seed)
    query_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.query_dtype]
    query = random_data(
        (len(lengths) * args.query_length, args.query_heads, dim),
        query_generator,
        args,
        device,
    ).to(query_dtype)
    key = torch.empty(
        (num_pages, heads, dim // chunk, page, chunk), dtype=kv_dtype, device=device
    )
    vshape = (
        (num_pages, heads, page // chunk, dim, chunk)
        if args.v_layout == "transposed"
        else (num_pages, heads, dim, page)
    )
    value = torch.empty(vshape, dtype=kv_dtype, device=device)
    # Bound FP32 initialization staging to about 32 MiB, regardless of cache size.
    pages_per_chunk = max(1, (8 * 1024**2) // (heads * page * dim))
    fp8 = kv_dtype != torch.bfloat16

    def initialize(cache, is_key):
        scale = None
        if fp8 and args.kv_scale == "per-tensor":
            # A first pass finds one scale for the entire cache. Restore the RNG
            # to regenerate exactly the same data without retaining a FP32 cache.
            state = generator.get_state()
            maximum = torch.zeros((), device=device)
            for start in range(0, num_pages, pages_per_chunk):
                count = min(pages_per_chunk, num_pages - start)
                raw = random_data((count, heads, page, dim), generator, args, device)
                maximum = torch.maximum(maximum, raw.abs().amax())
                del raw
            scale = (maximum / torch.finfo(kv_dtype).max).reshape(1).clamp_min(1e-12)
            generator.set_state(state)
        elif fp8:
            scale = torch.empty(
                (num_pages, heads, page, 1), dtype=torch.float32, device=device
            )
        for start in range(0, num_pages, pages_per_chunk):
            count = min(pages_per_chunk, num_pages - start)
            raw = random_data((count, heads, page, dim), generator, args, device)
            if fp8:
                local_scale = scale
                if args.kv_scale == "per-token":
                    local_scale = (
                        raw.abs().amax(-1, keepdim=True) / torch.finfo(kv_dtype).max
                    ).clamp_min(1e-12)
                    scale[start : start + count] = local_scale
                raw = (raw / local_scale).to(kv_dtype)
            else:
                raw = raw.to(kv_dtype)
            packed = (
                raw.reshape(count, heads, page, dim // chunk, chunk).permute(
                    0, 1, 3, 2, 4
                )
                if is_key
                else (
                    raw.reshape(count, heads, page // chunk, chunk, dim).permute(
                        0, 1, 2, 4, 3
                    )
                    if args.v_layout == "transposed"
                    else raw.permute(0, 1, 3, 2)
                )
            )
            cache[start : start + count].copy_(packed)
        return scale

    key_scale = initialize(key, True)
    value_scale = initialize(value, False)
    return {
        "query": query,
        "key_cache": key,
        "value_cache": value,
        "key_scale": key_scale,
        "value_scale": value_scale,
        "context_lengths": torch.tensor(lengths, dtype=torch.int32, device=device),
        "block_tables": table,
    }


def reference_sequence(inputs, seq, length, args):
    """FP32 dense causal attention over the actual stored/dequantized caches."""
    query = inputs["query"][
        seq * args.query_length : (seq + 1) * args.query_length
    ].float()
    if length == 0:
        return torch.zeros_like(query)
    page, dim, heads = args.page_size, args.head_dim, args.kv_heads
    pages = inputs["block_tables"][seq, : (length + page - 1) // page].long()
    key = (
        inputs["key_cache"][pages]
        .permute(0, 1, 3, 2, 4)
        .reshape(-1, heads, page, dim)
        .float()
    )
    packed_v = inputs["value_cache"][pages]
    value = (
        packed_v.permute(0, 1, 2, 4, 3).reshape(-1, heads, page, dim)
        if args.v_layout == "transposed"
        else packed_v.permute(0, 1, 3, 2)
    ).float()
    for tensor, name in ((key, "key_scale"), (value, "value_scale")):
        scale = inputs[name]
        if scale is not None:
            tensor.mul_(scale if scale.numel() == 1 else scale[pages])
    key = key.permute(0, 2, 1, 3).reshape(-1, heads, dim)[:length]
    value = value.permute(0, 2, 1, 3).reshape(-1, heads, dim)[:length]
    group = args.query_heads // heads
    query = query.reshape(args.query_length, heads, group, dim)
    scores = torch.einsum("qhgd,khd->qhgk", query, key) * dim**-0.5
    positions = torch.arange(length, device=query.device)
    visible = (
        length
        - args.query_length
        + 1
        + torch.arange(args.query_length, device=query.device)
    )
    masked = positions[None, :] >= visible[:, None]
    if args.window:
        masked |= positions[None, :] < (visible - args.window)[:, None]
    scores.masked_fill_(masked[:, None, None, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    probabilities.masked_fill_(visible[:, None, None, None] <= 0, 0)
    return torch.einsum("qhgk,khd->qhgd", probabilities, value).reshape(
        -1, args.query_heads, dim
    )


def check_output(inputs, output, lengths, args, kv_dtype):
    if not bool(torch.isfinite(output).all()):
        raise ArithmeticError("PA decode produced nonfinite output")
    count = min(args.check_sequences, len(lengths))
    order = sorted(range(len(lengths)), key=lengths.__getitem__)
    selected = list(dict.fromkeys([order[0], order[-1]]))[:count]
    rng = random.Random(args.seed)
    selected += rng.sample(
        [seq for seq in order if seq not in selected], count - len(selected)
    )
    maximum_error = 0.0
    atol, rtol = (0.002, 0.01) if kv_dtype == torch.bfloat16 else (0.005, 0.005)
    atol = atol if args.atol is None else args.atol
    rtol = rtol if args.rtol is None else args.rtol
    for seq in selected:
        expected = reference_sequence(inputs, seq, lengths[seq], args)
        actual = output[seq * args.query_length : (seq + 1) * args.query_length].float()
        torch.testing.assert_close(
            actual,
            expected,
            atol=atol,
            rtol=rtol,
            msg=lambda message, seq=seq: (
                f"Sequence {seq}, context length {lengths[seq]}: {message}"
            ),
        )
        maximum_error = max(maximum_error, (actual - expected).abs().max().item())
    return {
        "checked_sequences": selected,
        "max_abs_error": maximum_error,
        "reference_checked": bool(selected),
        "atol": atol,
        "rtol": rtol,
    }


def benchmark(inputs, lengths, args, budget, kv_dtype):
    from aiter.ops.flydsl.pa_decode import pa_decode, plan_pa_decode

    plan = plan_pa_decode(
        inputs["context_lengths"],
        args.kv_heads,
        max_partitions=args.max_partitions,
        workgroup_budget=budget,
        query_length=args.query_length,
        sliding_window=args.window,
    )
    output = torch.empty_like(inputs["query"])
    shape = (
        args.kv_heads,
        plan.capacity,
        args.query_length * args.query_heads // args.kv_heads,
    )
    sums = torch.empty(shape, dtype=torch.float32, device=output.device)
    maxima = torch.empty_like(sums)
    partial = torch.empty(
        (*shape, args.head_dim), dtype=output.dtype, device=output.device
    )

    def launch():
        if args.include_plan:
            plan_pa_decode(
                inputs["context_lengths"],
                args.kv_heads,
                plan=plan,
                query_length=args.query_length,
                sliding_window=args.window,
            )
        pa_decode(
            output,
            **inputs,
            softmax_scale=args.head_dim**-0.5,
            query_length=args.query_length,
            work_plan=plan,
            sliding_window=args.window,
            max_context_length=max(lengths),
            exp_sums=sums,
            max_logits=maxima,
            temporary_output=partial,
        )

    launch()
    torch.cuda.synchronize()
    accuracy = check_output(inputs, output, lengths, args, kv_dtype)
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(args.graph_iterations):
            launch()
    graph.replay()
    torch.cuda.synchronize()
    samples = []
    for _ in range(args.rounds):
        begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        begin.record()
        for _ in range(args.replays):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(
            begin.elapsed_time(end) * 1000 / (args.graph_iterations * args.replays)
        )
    check_output(inputs, output, lengths, args, kv_dtype)
    latency = statistics.median(samples)
    if not math.isfinite(latency) or latency <= 0:
        raise ArithmeticError(f"Invalid GPU timing: {latency}")
    tokens = sum(
        min(length, args.window + args.query_length - 1) if args.window else length
        for length in lengths
    )
    kv_bytes = (
        2
        * tokens
        * args.kv_heads
        * args.head_dim
        * torch.empty((), dtype=kv_dtype).element_size()
    )
    return {
        "workgroup_budget": budget,
        "plan_capacity": plan.capacity,
        "active_tasks": int(plan.reduce_info[:, 1].sum().item()),
        "latency_us": latency,
        "min_us": min(samples),
        "max_us": max(samples),
        "samples_us": samples,
        "useful_kv_bytes": kv_bytes,
        "useful_kv_tb_s": kv_bytes / latency / 1e6,
        **accuracy,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 64])
    parser.add_argument("--min-context-len", type=int, default=1)
    parser.add_argument("--max-context-lens", type=int, nargs="+", default=[1024, 4096])
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        help="Exact lengths; overrides batch sizes and context ranges",
    )
    parser.add_argument(
        "--distributions",
        nargs="+",
        choices=["uniform", "log-uniform", "bimodal"],
        default=["uniform"],
    )
    parser.add_argument("--empty-fraction", type=float, default=0.0)
    parser.add_argument("--query-heads", type=int, default=64)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument(
        "--head-dim",
        type=int,
        choices=[64, 128, 256, 384, 512, 640, 768, 896, 1024],
        default=128,
    )
    parser.add_argument(
        "--query-length",
        type=int,
        default=1,
        help="MTP rows per sequence; context lengths include these tokens",
    )
    parser.add_argument("--query-dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--page-size", type=int, choices=[16, 64, 128], default=128)
    parser.add_argument(
        "--kv-dtypes", nargs="+", choices=["fp8", "bf16"], default=["fp8"]
    )
    parser.add_argument(
        "--kv-scale", choices=["per-token", "per-tensor"], default="per-token"
    )
    parser.add_argument(
        "--v-layout", choices=["plain", "transposed"], default="transposed"
    )
    parser.add_argument(
        "--data-init",
        choices=["uniform", "normal"],
        default="uniform",
        help="U(-0.5,0.5) or N(0,0.5)",
    )
    parser.add_argument(
        "--window", type=int, default=0, help="Sliding window; zero means full context"
    )
    parser.add_argument("--workgroup-budgets", type=int, nargs="+", default=[512])
    parser.add_argument("--max-partitions", type=int, default=None)
    parser.add_argument(
        "--include-plan",
        action="store_true",
        help="Also time an in-place GPU work-plan refresh",
    )
    parser.add_argument(
        "--check-sequences",
        type=int,
        default=4,
        help="Reference-check shortest/longest and random sequences; 0 disables reference checking",
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--atol",
        type=float,
        default=None,
        help="Override the dtype-specific absolute reference tolerance",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=None,
        help="Override the dtype-specific relative reference tolerance",
    )
    parser.add_argument("--graph-iterations", type=int, default=32)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=None, help="Write JSON and a sibling CSV file"
    )
    args = parser.parse_args()
    positive = (
        args.batch_sizes
        + args.workgroup_budgets
        + [
            args.query_heads,
            args.kv_heads,
            args.query_length,
            args.graph_iterations,
            args.replays,
            args.rounds,
        ]
    )
    if any(value < 1 for value in positive) or max(args.batch_sizes) > 4096:
        parser.error(
            "Sizes, budgets and timing counts must be positive; batch sizes are limited to 4096"
        )
    if args.query_heads % args.kv_heads:
        parser.error("query-heads must be divisible by kv-heads")
    if (
        not 0 <= args.empty_fraction <= 1
        or min(
            args.window,
            args.min_context_len,
            args.check_sequences,
            args.warmup,
            args.device,
        )
        < 0
    ):
        parser.error("Invalid negative parameter or empty-fraction outside [0,1]")
    if args.context_lengths is not None:
        if len(args.context_lengths) > 4096 or min(args.context_lengths) < 0:
            parser.error(
                "Exact lengths must be nonnegative and have at most 4096 entries"
            )
    elif any(length < args.min_context_len for length in args.max_context_lens):
        parser.error("max-context-lens must be at least min-context-len")
    bounds = (
        args.context_lengths
        if args.context_lengths is not None
        else args.max_context_lens
    )
    if max(bounds) > 2**31 - 1:
        parser.error("Context lengths must fit int32")
    if args.output is not None and args.output.suffix != ".json":
        parser.error("--output must use a .json extension; CSV is written alongside it")
    if any(
        value is not None and (not math.isfinite(value) or value < 0)
        for value in (args.atol, args.rtol)
    ):
        parser.error("Reference tolerances must be finite and nonnegative")
    return args


def write_results(path, metadata, rows, status="RUNNING", error=None):
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {"status": status, "error": error, "metadata": metadata, "results": rows},
            indent=2,
        )
        + "\n"
    )
    if not rows:
        return
    fields = [name for name, value in rows[0].items() if not isinstance(value, list)]
    with path.with_suffix(".csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


@torch.inference_mode()
def main():
    args = parse_args()
    if not torch.version.hip or not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires ROCm PyTorch and an AMD GPU")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    props = torch.cuda.get_device_properties(device)
    arch = props.gcnArchName.split(":", 1)[0]
    if arch not in ("gfx942", "gfx950", "gfx1250"):
        raise RuntimeError(f"Unsupported architecture: {arch}")
    if "bf16" in args.kv_dtypes and arch != "gfx1250":
        raise RuntimeError("Native BF16 caches require gfx1250")
    if (
        args.max_partitions is not None
        and not 1 <= args.max_partitions <= props.multi_processor_count
    ):
        raise ValueError("max-partitions must be between 1 and the device CU count")
    torch.backends.cuda.matmul.allow_tf32 = False
    metadata = {
        "gpu": props.name,
        "architecture": arch,
        "compute_units": props.multi_processor_count,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "flydsl": version("flydsl"),
        "timed_operations": (
            "plan+decode+reduce" if args.include_plan else "decode+reduce"
        ),
        "cache_policy": "repeated graph replays on the same inputs",
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
    }
    print(json.dumps(metadata), flush=True)
    print(
        " B   distribution   ctx min/mean/max         KV     budget tasks  median us  KV TB/s  check error",
        flush=True,
    )
    cases = (
        [(len(args.context_lengths), max(args.context_lengths), "explicit")]
        if args.context_lengths is not None
        else list(
            itertools.product(
                args.batch_sizes, args.max_context_lens, args.distributions
            )
        )
    )
    rows = []
    write_results(args.output, metadata, rows)
    for batch, maximum, distribution in cases:
        lengths = args.context_lengths or context_lengths(
            batch,
            args.min_context_len,
            maximum,
            distribution,
            args.empty_fraction,
            args.seed,
        )
        for dtype_name in args.kv_dtypes:
            dtype = (
                torch.bfloat16
                if dtype_name == "bf16"
                else torch.float8_e4m3fnuz if arch == "gfx942" else torch.float8_e4m3fn
            )
            pages = max(
                1,
                sum(
                    (length + args.page_size - 1) // args.page_size
                    for length in lengths
                ),
            )
            # Include caches, per-token scales, initialization staging and scratch.
            capacity = min(
                batch * (args.max_partitions or props.multi_processor_count),
                max(batch, math.ceil(max(args.workgroup_budgets) / args.kv_heads)),
            )
            estimate = (
                2
                * pages
                * args.kv_heads
                * args.page_size
                * args.head_dim
                * torch.empty((), dtype=dtype).element_size()
                + 8 * pages * args.kv_heads * args.page_size
                + capacity
                * args.query_length
                * args.query_heads
                * (8 + 2 * args.head_dim)
                + batch * args.query_length * args.query_heads * args.head_dim * 8
                + 256 * 1024**2
            )
            if estimate > 0.9 * torch.cuda.mem_get_info(device)[0]:
                raise RuntimeError(
                    f"Estimated allocation {estimate / 2**30:.2f} GiB exceeds available GPU memory"
                )
            inputs = make_inputs(lengths, args, dtype, device)
            budgets = list(dict.fromkeys(args.workgroup_budgets))
            random.Random(args.seed).shuffle(budgets)
            for budget in budgets:
                try:
                    result = benchmark(inputs, lengths, args, budget, dtype)
                except Exception as exc:
                    write_results(
                        args.output,
                        metadata,
                        rows,
                        "FAILED",
                        f"{type(exc).__name__}: {exc}",
                    )
                    raise
                row = {
                    "batch_size": batch,
                    "distribution": distribution,
                    "context_min": min(lengths),
                    "context_mean": statistics.mean(lengths),
                    "context_max": max(lengths),
                    "total_context_tokens": sum(lengths),
                    "context_lengths": lengths,
                    "physical_pages": pages,
                    "kv_dtype": dtype_name,
                    "kv_scale": args.kv_scale if dtype_name == "fp8" else "none",
                    "query_dtype": args.query_dtype,
                    "query_heads": args.query_heads,
                    "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim,
                    "query_length": args.query_length,
                    "page_size": args.page_size,
                    "v_layout": args.v_layout,
                    "window": args.window,
                    "seed": args.seed,
                    "timed_operations": metadata["timed_operations"],
                    **result,
                }
                rows.append(row)
                print(
                    f"{batch:3d}  {distribution:11s}  {min(lengths):6d}/{statistics.mean(lengths):7.1f}/{max(lengths):6d}  "
                    f"{dtype_name:5s}  {budget:6d} {result['active_tasks']:5d}  "
                    f"{result['latency_us']:10.3f}  {result['useful_kv_tb_s']:7.3f}  {result['max_abs_error']:.5f}",
                    flush=True,
                )
                write_results(args.output, metadata, rows)
            del inputs
    write_results(args.output, metadata, rows, "COMPLETE")
    return rows


if __name__ == "__main__":
    main()
