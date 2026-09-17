# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

r"""FlyDSL PA decode: MTP/TP sweep with uniform and one-long KV lengths.

Run from the repository root, for example::

    python3 -m op_tests.benchmark_flydsl_pa_decode_mtp \
        --batch-sizes 1 2 4 8 16 200 --context-length 200000 \
        --short-context-length 1024 --mtp 4 --tp-size 4 \
        --num-query-heads 64 --num-kv-heads 4 --head-dim 128 \
        --block-size 16 128 --trans-v 0 1 \
        --modes static planned --output pa_decode_mtp_tp4.csv

Add --dry-run to print the matrix without importing torch or initializing a GPU.

TP describes head sharding on ONE rank: global Hq=64/Hkv=4 at TP=4 gives
local Hq=16/Hkv=1. Batch size and context length are not divided by TP. This
measures one rank's attention operator; it does not launch distributed ranks
or include tensor-parallel collectives. Defaults use BF16 Q/O, FP8 K/V and
per-token FP32 K/V scales. Context lengths include the MTP query positions.

Both patterns use a 200000-token maximum: uniform=[200000]*B and
one_long=[200000]+[short]*(B-1). At B=1 the two inputs are identical.
Pages are allocated only for each sequence's actual length.

The timer is aiter.test_common.run_perftest, as in test_flydsl_pa_decode.py,
with allocation rotation. time_us is the per-call sum of GPU kernel times:
attention + partition reduction, plus a plan refresh on EVERY planned call.
Input generation, quantization, FP32 reference, JIT warmup, scratch allocation
and result validation are outside timing. This is not CPU wall-clock latency.

Effective bytes = Q + O + valid K/V tokens + valid K/V scales + referenced page
table entries + context lengths, counted once per rank. KV bytes are not
multiplied by MTP or TP. Padding, repeated loads, scratch and planner metadata
traffic are excluded, so effective bandwidth is not measured HBM bus traffic.
effective_GB_s = effective_bytes / time_us / 1e3 (decimal GB/s).
effective_TB_s = effective_bytes / time_us / 1e6 (decimal TB/s).
"""

import argparse
import csv
import itertools
import json
import math
from pathlib import Path


def _positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


def _nonnegative_int(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--batch-sizes", nargs="+", type=_positive_int, default=[1, 2, 4, 8, 16, 200]
    )
    parser.add_argument("--context-length", type=_positive_int, default=200000)
    parser.add_argument("--short-context-length", type=_positive_int, default=1024)
    parser.add_argument(
        "--patterns",
        nargs="+",
        choices=["uniform", "one_long"],
        default=["uniform", "one_long"],
    )
    parser.add_argument("--mtp", type=_positive_int, default=4)
    parser.add_argument("--tp-size", type=_positive_int, default=4)
    parser.add_argument(
        "--num-query-heads",
        type=_positive_int,
        default=64,
        help="Global Q head count before TP sharding (default: 64).",
    )
    parser.add_argument(
        "--num-kv-heads",
        type=_positive_int,
        default=4,
        help="Global KV head count before TP sharding (default: 4).",
    )
    parser.add_argument(
        "--head-dim", type=int, choices=[64, *range(128, 1025, 128)], default=128
    )
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--per-token", type=int, choices=[0, 1], default=1)
    parser.add_argument(
        "--block-size", nargs="+", type=int, choices=[16, 64, 128], default=[16, 128]
    )
    parser.add_argument(
        "--trans-v", nargs="+", type=int, choices=[0, 1], default=[0, 1]
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["static", "planned"],
        default=["static", "planned"],
        help="Static auto splits and/or GPU-planned splits with timed plan refresh.",
    )
    parser.add_argument("--max-partitions", type=int, default=256)
    parser.add_argument("--num-iters", type=_positive_int, default=101)
    parser.add_argument("--num-warmup", type=_positive_int, default=5)
    parser.add_argument(
        "--num-rotate-args",
        type=_nonnegative_int,
        default=0,
        help="0: aiter automatic allocation rotation; 1: reuse the same buffers.",
    )
    parser.add_argument("--device", type=_nonnegative_int, default=0)
    parser.add_argument("--seed", type=_nonnegative_int, default=0)
    parser.add_argument("--output", type=Path, default=Path("pa_decode_mtp_tp4.csv"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.short_context_length >= args.context_length:
        parser.error("--short-context-length must be less than --context-length")
    if args.context_length >= 2**31:
        parser.error("context lengths must fit in int32")
    if args.num_query_heads % args.tp_size or args.num_kv_heads % args.tp_size:
        parser.error("global Q and KV head counts must be divisible by --tp-size")
    if args.num_query_heads % args.num_kv_heads:
        parser.error("Q head count must be divisible by KV head count")
    if not 4 <= args.max_partitions <= 256:
        parser.error("--max-partitions must be in [4, 256]")
    if args.num_iters < 2:
        parser.error("--num-iters must be at least 2 for aiter's profiler timer")
    if "planned" in args.modes and max(args.batch_sizes) > 4096:
        parser.error("the GPU planner supports batch sizes up to 4096")
    return args


def _lengths(args, batch_size, pattern):
    if pattern == "uniform":
        return [args.context_length] * batch_size
    return [args.context_length] + [args.short_context_length] * (batch_size - 1)


def _traffic_bytes(args, lengths, block_size):
    """Useful traffic, independent of cache padding and scheduling strategy."""
    q_heads = args.num_query_heads // args.tp_size
    kv_heads = args.num_kv_heads // args.tp_size
    tokens = sum(lengths)
    components = {
        "query_output_bytes": 2 * len(lengths) * args.mtp * q_heads * args.head_dim * 2,
        "kv_bytes": 2 * tokens * kv_heads * args.head_dim,
        "scale_bytes": 8 * (tokens * kv_heads if args.per_token else 1),
        "page_table_bytes": 4
        * sum((length + block_size - 1) // block_size for length in lengths),
        "context_length_bytes": 4 * len(lengths),
    }
    return {**components, "effective_bytes": sum(components.values())}


def _build_inputs(args, lengths, block_size, trans_v, device):
    import torch

    from aiter import per_tensor_quant, pertoken_quant
    from op_tests.test_flydsl_pa_decode import _quant_dtype, run_torch

    torch.manual_seed(args.seed)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16
    quant_dtype = _quant_dtype()
    q_heads = args.num_query_heads // args.tp_size
    kv_heads = args.num_kv_heads // args.tp_size
    page_counts = [(length + block_size - 1) // block_size for length in lengths]
    num_pages = sum(page_counts)
    query = torch.empty(
        (len(lengths) * args.mtp, q_heads, args.head_dim), dtype=dtype, device=device
    ).uniform_(-0.5, 0.5)

    # Quantize K and V separately so both full BF16 caches need not coexist.
    def make_quantized_cache():
        tensor = torch.empty(
            (num_pages, kv_heads, block_size, args.head_dim), dtype=dtype, device=device
        ).uniform_(-0.5, 0.5)
        quantize = pertoken_quant if args.per_token else per_tensor_quant
        return quantize(tensor, quant_dtype=quant_dtype)

    key_quant, key_scale = make_quantized_cache()
    value_tokens, value_scale = make_quantized_cache()
    value_plain = value_tokens.permute(0, 1, 3, 2).contiguous()

    # Short sequences get only their own pages; the unused rectangular table
    # tail points to a valid page and is ignored by the kernel's length mask.
    tables_host = torch.zeros(
        (len(lengths), max(page_counts)), dtype=torch.int32, device="cpu"
    )
    next_page = 0
    for seq, page_count in enumerate(page_counts):
        tables_host[seq, :page_count] = torch.arange(
            next_page, next_page + page_count, dtype=torch.int32, device="cpu"
        )
        next_page += page_count
    tables = tables_host.to(device)
    contexts = torch.tensor(lengths, dtype=torch.int32, device=device)

    # Reuse aiter's dequantized FP32 causal reference for BOTH length patterns.
    reference = run_torch(
        query,
        key_quant,
        value_plain,
        tables,
        contexts,
        key_scale,
        value_scale,
        query_length=args.mtp,
    )
    key_cache = (
        key_quant.view(num_pages, kv_heads, block_size, args.head_dim // 16, 16)
        .permute(0, 1, 3, 2, 4)
        .contiguous()
    )
    if trans_v:
        value_cache = (
            value_tokens.view(num_pages, kv_heads, block_size // 16, 16, args.head_dim)
            .permute(0, 1, 2, 4, 3)
            .contiguous()
        )
    else:
        value_cache = value_plain
    inputs = (
        torch.empty_like(query),
        query,
        key_cache,
        value_cache,
        tables,
        contexts,
        key_scale,
        value_scale,
    )
    return inputs, reference


def _benchmark_case(args, batch_size, pattern, block_size, trans_v, device):
    import torch

    from aiter.jit.utils.chip_info import get_gfx_runtime
    from aiter.ops.flydsl.pa_decode import (
        get_recommended_splits,
        pa_decode,
        plan_pa_decode,
    )
    from aiter.test_common import run_perftest

    lengths = _lengths(args, batch_size, pattern)
    inputs, reference = _build_inputs(args, lengths, block_size, trans_v, device)
    query, contexts = inputs[1], inputs[5]
    kv_heads = args.num_kv_heads // args.tp_size
    query_rows = args.mtp * query.shape[1] // kv_heads
    static_parts = get_recommended_splits(
        batch_size,
        kv_heads,
        256 // block_size,
        max_partitions=args.max_partitions,
        max_context_length=max(lengths),
        query_length=args.mtp,
    )
    traffic = _traffic_bytes(args, lengths, block_size)

    for mode in args.modes:
        plan = None
        partial_shape = (batch_size, kv_heads, static_parts, query_rows)
        if mode == "planned":
            plan = plan_pa_decode(
                contexts,
                kv_heads,
                max_partitions=args.max_partitions,
                total_context_length=sum(lengths),
                query_length=args.mtp,
            )
            partial_shape = (kv_heads, plan.capacity, query_rows)
        partition_kwargs = (
            {"max_context_partition_num": static_parts}
            if plan is None
            else {"work_plan": plan}
        )
        pmax = torch.empty(partial_shape, dtype=torch.float32, device=device)
        psum = torch.empty_like(pmax)
        pout = torch.empty(
            (*partial_shape, args.head_dim), dtype=query.dtype, device=device
        )

        def run(out, q, k, v, bt, ctx, ks, vs, pm, ps, po):
            if plan is not None:
                plan_pa_decode(ctx, kv_heads, plan=plan)
            pa_decode(
                out,
                q,
                k,
                v,
                ctx,
                bt,
                args.head_dim**-0.5,
                args.mtp,
                compute_type=k.dtype,
                key_scale=ks,
                value_scale=vs,
                max_logits=pm,
                exp_sums=ps,
                temporary_output=po,
                **partition_kwargs,
            )
            return out

        # All large attention buffers are explicit arguments for aiter's
        # rotation. Plan metadata is preallocated and refreshed inside run().
        out, us = run_perftest(
            run,
            *inputs,
            pmax,
            psum,
            pout,
            num_iters=args.num_iters,
            num_warmup=args.num_warmup,
            num_rotate_args=args.num_rotate_args,
        )
        torch.testing.assert_close(
            out.float(), reference.float(), rtol=0.005, atol=0.005
        )
        if not math.isfinite(us) or us <= 0:
            raise RuntimeError(f"invalid GPU timing: {us} us")
        # Metadata readback is outside timing, after the accuracy check.
        partition_counts = (
            [static_parts] * batch_size
            if plan is None
            else plan.num_partitions.tolist()
        )
        yield {
            "batch_size": batch_size,
            "pattern": pattern,
            "mode": mode,
            "context_min": min(lengths),
            "context_max": max(lengths),
            "total_kv_tokens": sum(lengths),
            "mtp": args.mtp,
            "tp_size": args.tp_size,
            "scope": "single_rank",
            "global_q_heads": args.num_query_heads,
            "global_kv_heads": args.num_kv_heads,
            "local_q_heads": query.shape[1],
            "local_kv_heads": kv_heads,
            "head_dim": args.head_dim,
            "query_dtype": args.dtype,
            "kv_dtype": str(inputs[2].dtype),
            "per_token": args.per_token,
            "block_size": block_size,
            "trans_v": trans_v,
            "partition_capacity": (
                batch_size * static_parts if plan is None else plan.capacity
            ),
            "partitions_per_sequence": json.dumps(partition_counts),
            "time_us": us,
            "time_ms": us / 1e3,
            "us_per_output_token": us / (batch_size * args.mtp),
            "effective_GB_s": traffic["effective_bytes"] / us / 1e3,
            "effective_TB_s": traffic["effective_bytes"] / us / 1e6,
            **traffic,
            "correctness": "pass",
            "num_iters": args.num_iters,
            "num_warmup": args.num_warmup,
            "num_rotate_args": args.num_rotate_args,
            "seed": args.seed,
            "device": args.device,
            "gpu": torch.cuda.get_device_name(device),
            "gfx": get_gfx_runtime(),
        }
        del out, pmax, psum, pout


def main():
    args = _parse_args()
    cases = list(
        itertools.product(
            args.batch_sizes, args.patterns, args.block_size, args.trans_v
        )
    )
    print(
        f"Single-rank PA: TP={args.tp_size}, "
        f"local Hq/Hkv={args.num_query_heads // args.tp_size}/"
        f"{args.num_kv_heads // args.tp_size}, D={args.head_dim}, MTP={args.mtp}; "
        f"{len(cases) * len(args.modes)} result rows."
    )
    print(
        f"Q/O={args.dtype}, KV=FP8, per_token={args.per_token}; "
        f"uniform=[{args.context_length}]*B, "
        f"one_long=[{args.context_length}]+[{args.short_context_length}]*(B-1)."
    )
    print(
        "At batch_size=1 both patterns are identical. Bandwidth is per-rank effective traffic."
    )
    if args.dry_run:
        for batch, pattern, block, trans in cases:
            lengths = _lengths(args, batch, pattern)
            traffic = _traffic_bytes(args, lengths, block)
            print(
                f"B={batch:<3} {pattern:<8} block={block:<3} trans_v={trans} "
                f"tokens={sum(lengths):<9} effective_bytes={traffic['effective_bytes']} "
                f"modes={','.join(args.modes)}"
            )
        print(
            f"Output when run: {args.output} (time_us, effective_GB_s, effective_TB_s, etc.)"
        )
        return

    import torch

    from aiter.jit.utils.chip_info import get_gfx_runtime

    if not torch.cuda.is_available():
        raise SystemExit("A ROCm GPU is required")
    if args.device >= torch.cuda.device_count():
        raise SystemExit(f"GPU index {args.device} is not available")
    device = torch.device("cuda", args.device)
    with torch.cuda.device(device), torch.no_grad():
        if get_gfx_runtime() not in ("gfx942", "gfx950"):
            raise SystemExit(f"Unsupported PA architecture: {get_gfx_runtime()}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", newline="") as output:
            writer = None
            print(
                f"{'B':>4} {'pattern':>8} {'block':>5} {'Vt':>2} {'mode':>7} "
                f"{'time_us':>12} {'GB/s':>12} {'TB/s':>10}",
                flush=True,
            )
            for batch, pattern, block, trans in cases:
                print(
                    f"Preparing B={batch} {pattern} block={block} trans_v={trans}",
                    flush=True,
                )
                for row in _benchmark_case(args, batch, pattern, block, trans, device):
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow(row)
                    output.flush()
                    print(
                        f"{batch:>4} {pattern:>8} {block:>5} {trans:>2} {row['mode']:>7} "
                        f"{row['time_us']:>12.3f} {row['effective_GB_s']:>12.3f} "
                        f"{row['effective_TB_s']:>10.4f}",
                        flush=True,
                    )
                torch.cuda.empty_cache()
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
