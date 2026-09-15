# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the full GDN prefill block (``chunk_gated_delta_rule_opt_vk``).

Times wall / device (K1..K6) / launch-only / host gap for flydsl vs hip (optional
triton). Shapes are built from ``--model`` + ``--tp`` + ``--t`` + ``--n`` for
Qwen3.5-35B (Hv=32) and Qwen3.5-397B (Hv=64) only.

Example::

    python bench_gated_delta_rule_block.py --model 35b --tp 1 --t 8192 --n 1
    python bench_gated_delta_rule_block.py --model 397b --tp 4 --t 8192 --n 8 \\
        --snapshot-dtype bf16 fp32
"""

from __future__ import annotations

import argparse
import importlib.util
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from aiter.ops.prefill_batch_metadata import (
    build_gated_delta_rule_prefill_metadata,
)
from aiter.ops.triton.gated_delta_net import chunk_gated_delta_rule_opt_vk

CHUNK_SIZE = 64

_K5_TEST_PATH = (
    Path(__file__).resolve().parents[2] / "test_flydsl_linear_attention_prefill.py"
)

MODELS = {
    "35b": {"label": "Qwen3.5-35B", "Hv": 32},
    "397b": {"label": "Qwen3.5-397B", "Hv": 64},
}

STAGE_RULES = [
    ("cumsum_scaled_dot_kkt", "K1+K2 cumsum/KKT"),
    ("merge_16x16_to_64x64_inverse", "K3 solve_tril"),
    ("solve_tril", "K3 solve_tril"),
    ("recompute_w_u", "K4 W/U"),
    ("chunk_gated_delta_rule_fwd_h_hip_kernel", "K5 state scan"),
    ("chunk_gdn_fwd_h_flydsl", "K5 state scan"),
    ("chunk_gated_delta_rule_fwd_kernel_h", "K5 state scan"),
    ("chunk_fwd_kernel_o", "K6 output"),
]
STAGES = [
    "K1+K2 cumsum/KKT",
    "K3 solve_tril",
    "K4 W/U",
    "K5 state scan",
    "K6 output",
    "other",
]
TOTALS = ["kernel total", "wall total", "launch only", "host gap"]

DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32}


def _load_prefill_args_cls():
    spec = importlib.util.spec_from_file_location(
        "_gdn_k5_prefill_cases", _K5_TEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.PrefillArgs


def _build_case(
    PrefillArgs,
    *,
    model: str,
    tp: int,
    t: int,
    n: int,
    dense: bool,
    snapshot_name: str,
):
    spec = MODELS[model]
    snapshot_dtype = DTYPES[snapshot_name] if snapshot_name == "fp32" else None
    if dense:
        return PrefillArgs(
            K=128,
            V=128,
            Hk=16,
            Hv=spec["Hv"],
            tp=tp,
            full_prompt_len=t,
            model_name=f"{spec['label']}-dense-bench",
            max_num_batched_tokens=t,
            is_varlen=False,
            output_final_state=False,
            snapshot_dtype=snapshot_dtype,
        )
    if n < 1:
        raise SystemExit("--n must be >= 1 for varlen")
    return PrefillArgs(
        K=128,
        V=128,
        Hk=16,
        Hv=spec["Hv"],
        tp=tp,
        full_prompt_len=t,
        model_name=f"{spec['label']}-varlen-bench",
        max_num_batched_tokens=n * t,
        is_varlen=True,
        output_final_state=True,
        snapshot_dtype=snapshot_dtype,
    )


def _stage_of(kernel_name: str) -> str:
    for needle, stage in STAGE_RULES:
        if needle in kernel_name:
            return stage
    return "other"


def _build_inputs(case, seed, with_state_pool=False):
    torch.manual_seed(seed)
    device = torch.device("cuda")
    context_lens = case.resolve_context_lens()
    total_tokens = sum(context_lens)
    num_heads, num_kv_heads = case.H, case.Hg

    if case.is_varlen:
        batch = 1
        num_states = len(context_lens)
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(context_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=device,
        )
        metadata = build_gated_delta_rule_prefill_metadata(
            context_lens, cu_seqlens=cu_seqlens, chunk_size=CHUNK_SIZE
        )
    else:
        batch = case.dense_batch
        num_states = batch
        cu_seqlens = None
        metadata = None

    q = F.normalize(
        torch.randn(batch, total_tokens, num_kv_heads, case.K, device=device), dim=-1
    ).to(case.dtype)
    k = F.normalize(
        torch.randn(batch, total_tokens, num_kv_heads, case.K, device=device), dim=-1
    ).to(case.dtype)
    v = (torch.randn(batch, total_tokens, num_heads, case.V, device=device) * 0.1).to(
        case.dtype
    )
    beta = torch.sigmoid(
        torch.randn(batch, total_tokens, num_heads, device=device, dtype=torch.float32)
    ).to(case.dtype)
    g = F.logsigmoid(
        torch.randn(batch, total_tokens, num_heads, device=device, dtype=torch.float32)
    )

    def make_states(count):
        return (
            torch.randn(
                count, num_heads, case.V, case.K, device=device, dtype=torch.float32
            )
            * 0.01
        ).to(case.ssm_state_dtype)

    initial_state = make_states(num_states)
    state_pool = make_states(2 * num_states) if with_state_pool else None
    state_indices = (
        torch.randperm(2 * num_states).to(device=device, dtype=torch.int32)[:num_states]
        if with_state_pool
        else None
    )

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "state_pool": state_pool,
        "state_indices": state_indices,
        "cu_seqlens": cu_seqlens,
        "metadata": metadata,
        "total_tokens": total_tokens,
        "num_seqs": num_states,
    }


def _make_callable(
    tensors,
    case,
    backend,
    snapshot_dtype,
    prefill_metadata,
    state_indices=None,
    *,
    cold_indices_check=False,
):
    initial_state = (
        tensors["state_pool"] if state_indices is not None else tensors["initial_state"]
    )
    pool_size = initial_state.shape[0] if state_indices is not None else 0

    def run():
        if cold_indices_check and state_indices is not None:
            new_slots = torch.randperm(pool_size, device=state_indices.device)[
                : state_indices.numel()
            ]
            state_indices.copy_(new_slots.to(dtype=state_indices.dtype))
        return chunk_gated_delta_rule_opt_vk(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            g=tensors["g"],
            beta=tensors["beta"],
            initial_state=initial_state,
            initial_state_indices=state_indices,
            output_final_state=case.output_final_state,
            cu_seqlens=tensors["cu_seqlens"],
            use_chunk_flydsl=backend == "flydsl",
            use_chunk_hip=backend == "hip",
            state_dtype=case.ssm_state_dtype,
            snapshot_dtype=snapshot_dtype,
            prefill_metadata=prefill_metadata,
        )

    return run


def _bench_wall_us(run, warmup_iters, bench_iters):
    for _ in range(warmup_iters):
        run()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(bench_iters)]
    for i in range(bench_iters):
        starts[i].record()
        run()
        ends[i].record()
    torch.cuda.synchronize()
    times = sorted(s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends))
    return times[len(times) // 2]


def _bench_launch_us(run, warmup_iters, bench_iters):
    for _ in range(warmup_iters):
        run()
    torch.cuda.synchronize()
    times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        run()
        times.append((time.perf_counter() - start) * 1e6)
    torch.cuda.synchronize()
    return statistics.median(times)


def _profile_kernels_us(run, warmup_iters, prof_iters):
    for _ in range(warmup_iters):
        run()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(prof_iters):
            run()
        torch.cuda.synchronize()
    per_kernel = {}
    for evt in prof.key_averages():
        if evt.device_type is None or "cuda" not in str(evt.device_type).lower():
            continue
        us = evt.self_device_time_total / prof_iters
        if us > 0.0:
            per_kernel[evt.key] = per_kernel.get(evt.key, 0.0) + us
    return per_kernel


def _measure(run, args):
    run()
    torch.cuda.synchronize()
    wall_us = _bench_wall_us(run, args.warmup_iters, args.bench_iters)
    launch_us = _bench_launch_us(run, args.warmup_iters, args.bench_iters)
    per_kernel = _profile_kernels_us(run, args.warmup_iters, args.prof_iters)
    row = {stage: 0.0 for stage in STAGES}
    for name, us in per_kernel.items():
        row[_stage_of(name)] += us
    row["kernel total"] = sum(per_kernel.values())
    row["wall total"] = wall_us
    row["launch only"] = launch_us
    row["host gap"] = wall_us - row["kernel total"]
    return row, per_kernel


def _print_table(labels, rows):
    width = max(16, max(len(label) for label in labels) + 2)
    header = f"{'stage (us)':<20}" + "".join(f"{label:>{width}}" for label in labels)
    print(header)
    print("-" * len(header))
    for stage in STAGES + TOTALS:
        if stage in STAGES and all(rows[label][stage] == 0.0 for label in labels):
            continue
        if stage == TOTALS[0]:
            print("-" * len(header))
        print(
            f"{stage:<20}"
            + "".join(f"{rows[label][stage]:{width}.1f}" for label in labels)
        )


def _print_summary(records, metrics):
    columns, row_keys, cells = [], [], {}
    for rec in records:
        if rec["column"] not in columns:
            columns.append(rec["column"])
        row_key = (rec["case"], rec["snap"])
        if row_key not in row_keys:
            row_keys.append(row_key)
        cells[(row_key, rec["column"])] = rec["row"]

    ratios = []
    for column in columns:
        if column == "hip" or column.startswith("hip "):
            continue
        backend, _, suffix = column.partition(" ")
        hip_column = "hip" if not suffix else f"hip {suffix}"
        if hip_column in columns:
            label = f"{backend[:3]}/hip"
            if suffix:
                label += f" {suffix}"
            ratios.append((label, hip_column, column))

    case_width = max(len(case_id) for case_id, _ in row_keys) + 2
    col_width = max(12, max(len(column) for column in columns) + 2)
    if ratios:
        col_width = max(col_width, max(len(column) for column, _, _ in ratios) + 2)
    for metric in metrics:
        header = f"{metric + ' (us)':<{case_width}}{'snap':>6}"
        header += "".join(f"{column:>{col_width}}" for column in columns)
        header += "".join(f"{column:>{col_width}}" for column, _, _ in ratios)
        print(f"\n{header}")
        print("-" * len(header))
        for row_key in row_keys:
            case_id, snap = row_key
            line = f"{case_id:<{case_width}}{snap:>6}"
            for column in columns:
                row = cells.get((row_key, column))
                line += f"{row[metric]:{col_width}.1f}" if row else "-".rjust(col_width)
            for _, hip_column, column in ratios:
                hip_row = cells.get((row_key, hip_column))
                row = cells.get((row_key, column))
                if hip_row is None or row is None or row[metric] <= 0.0:
                    line += "-".rjust(col_width)
                else:
                    line += f"{hip_row[metric] / row[metric]:{col_width - 1}.2f}x"
            print(line)


def _run_case(case_id, case, args):
    if not case.use_g:
        print(f"\n== {case_id}\nskipped: the block entry requires g")
        return []

    with_pool = args.with_state_pool and case.output_final_state
    tensors = _build_inputs(case, args.seed, with_state_pool=with_pool)
    state_variants = [("", None)]
    if with_pool:
        state_variants.append(("pool", tensors["state_indices"]))
        if args.with_state_pool_cold_check:
            state_variants.append(("pool-cold", tensors["state_indices"]))

    records = []
    labels, rows, per_kernel_rows = [], {}, {}
    for backend in args.backends:
        snapshot_dtype = case.snapshot_dtype or case.dtype
        snapshot_name = "fp32" if snapshot_dtype == torch.float32 else "bf16"
        for state_name, state_indices in state_variants:
            run = _make_callable(
                tensors,
                case,
                backend,
                snapshot_dtype,
                prefill_metadata=tensors["metadata"],
                state_indices=state_indices,
                cold_indices_check=state_name == "pool-cold",
            )
            parts = (backend, snapshot_name, state_name)
            label = " ".join(part for part in parts if part)
            labels.append(label)
            rows[label], per_kernel_rows[label] = _measure(run, args)
            records.append(
                {
                    "case": case_id,
                    "snap": snapshot_name,
                    "column": " ".join(part for part in (backend, state_name) if part),
                    "row": rows[label],
                }
            )

    if not args.summary_only:
        if with_pool:
            pool_note = f", pool={2 * tensors['num_seqs']} slots"
        elif args.with_state_pool:
            pool_note = ", pool skipped (needs final_state=on)"
        else:
            pool_note = ""
        layout = "dense" if not case.is_varlen else f"varlen N={tensors['num_seqs']}"
        print(f"\n== {case_id}")
        print(
            f"TP{case.tp} Hg={case.Hg} H={case.H} K={case.K} V={case.V} {layout}, "
            f"T={tensors['total_tokens']}, "
            f"final_state={'on' if case.output_final_state else 'off'}{pool_note}"
        )
        _print_table(labels, rows)

    if args.show_kernels:
        print("\n-- per-kernel device time (us)")
        for label in labels:
            print(f"   {label}")
            for name, us in sorted(per_kernel_rows[label].items(), key=lambda x: -x[1]):
                print(f"     {us:8.1f}  [{_stage_of(name)}]  {name[:100]}")

    return records


def _selected_cases(args, PrefillArgs):
    cases = []
    for snap in args.snapshot_dtype:
        case = _build_case(
            PrefillArgs,
            model=args.model,
            tp=args.tp,
            t=args.t,
            n=args.n,
            dense=args.dense,
            snapshot_name=snap,
        )
        cases.append((repr(case), case))
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark GDN prefill block for Qwen3.5-35B / 397B shapes."
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODELS),
        default="35b",
        help="Qwen3.5-35B (Hv=32) or Qwen3.5-397B (Hv=64).",
    )
    parser.add_argument(
        "--tp",
        type=int,
        default=1,
        choices=[1, 2, 4, 8],
        help="Tensor-parallel rank count (Hg=Hk/tp, H=Hv/tp).",
    )
    parser.add_argument(
        "--t",
        type=int,
        default=8192,
        help="Per-sequence token length (full_prompt_len).",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=1,
        help="Varlen batch size: num sequences with length --t (mnbt = n * t). Ignored with --dense.",
    )
    parser.add_argument(
        "--dense",
        action="store_true",
        help="Dense layout (single seq, no cu_seqlens) instead of varlen.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["flydsl", "hip", "triton"],
        default=["flydsl", "hip"],
    )
    parser.add_argument(
        "--snapshot-dtype",
        nargs="+",
        choices=["bf16", "fp32"],
        default=["bf16"],
        help="Per-chunk h snapshot dtype to benchmark.",
    )
    parser.add_argument(
        "--with-state-pool",
        action="store_true",
        help="Also benchmark indexed initial_state_indices / state pool.",
    )
    parser.add_argument(
        "--with-state-pool-cold-check",
        action="store_true",
        help="Rewrite state_indices each call (with --with-state-pool).",
    )
    parser.add_argument("--show-kernels", action="store_true")
    parser.add_argument(
        "--summary-metrics",
        nargs="+",
        choices=STAGES + TOTALS,
        default=["wall total", "K5 state scan", "host gap"],
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Skip per-case tables when multiple snapshot dtypes are requested.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--prof-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    PrefillArgs = _load_prefill_args_cls()
    selected = _selected_cases(args, PrefillArgs)

    props = torch.cuda.get_device_properties(0)
    spec = MODELS[args.model]
    layout = "dense" if args.dense else f"varlen n={args.n}"
    print(f"\ngfx={props.gcnArchName} CUs={props.multi_processor_count}")
    print(
        f"shape: {spec['label']} TP{args.tp} T={args.t} {layout} "
        f"snapshot={','.join(args.snapshot_dtype)}"
    )
    print(
        f"wall / launch = median of {args.bench_iters} timings; "
        f"device = {args.prof_iters}-iter profiler self time"
    )

    records = []
    for case_id, case in selected:
        records += _run_case(case_id, case, args)

    if records and (len(selected) > 1 or args.summary_only):
        _print_summary(records, args.summary_metrics)


if __name__ == "__main__":
    main()
