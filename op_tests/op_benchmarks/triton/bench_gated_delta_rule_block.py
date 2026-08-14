# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the whole GDN prefill block per K5 backend x snapshot dtype.

``bench_gated_delta_rule_snapshot_dtype.py`` times the snapshot producer (K5)
and consumer (K6) in isolation. This benchmark instead runs the full
``chunk_gated_delta_rule_opt_vk`` block a serving stack calls, and splits the
measurement three ways:

* wall time -- the host-visible block latency (median of CUDA-event timings),
* device time per stage -- profiler self time bucketed into K1..K6,
* launch only -- the python-side cost of one call, measured without syncing,
* host gap -- wall minus device total.

`launch only` is what reads the host gap: it is a floor on wall time, so a
shape whose device work sits below it is launch bound and the gap is that
shortfall. A gap above `launch only` instead means a wrapper stalled the
stream, which is what the metadata=None runs below show.

Shapes are the ``PrefillArgs`` cases from the K5 test suite
(``op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py``), selected by
their pytest ids, so a block measurement and the K5-only table in that suite
describe the same workload. ``--list-cases`` prints the available ids.

The block is timed with a prebuilt ``GatedDeltaRulePrefillMetadata``, matching
how a serving stack builds the chunk schedule once and reuses it across the
layer stack. Pass ``--without-metadata`` to also time the metadata=None path,
where each wrapper recovers the chunk counts with a blocking device-to-host
copy that shows up as host gap rather than device time.

States are per-sequence and dense by default. Pass ``--with-state-pool`` to
also time the indexed path a serving stack uses, where ``initial_state`` is a
pool wider than the batch and K5 gathers each sequence's slot from an index
array, writing the final state back into that same slot.

Usage examples
--------------
# Default: the varlen-qwen-ali-tp1 T=8192 single-sequence case, bf16 vs fp32
# snapshots, flydsl vs hip
python bench_gated_delta_rule_block.py

# List the case ids this benchmark can run
python bench_gated_delta_rule_block.py --list-cases

# Any subset of the suite, matched as regexes against the case ids. Case ids
# start with the PrefillGroup name, so a prefix runs that group's whole sweep,
# and several patterns run several groups; more than one case adds a
# cross-case summary table (--summary-only drops the per-case ones).
python bench_gated_delta_rule_block.py --summary-only \
    --case "^varlen-qwen-ali-tp1-" "^varlen-qwen3.5-397b-ptpc-ali-"

# Quantify what skipping the prebuilt chunk schedule costs
python bench_gated_delta_rule_block.py --without-metadata

# Quantify what the indexed state pool costs against the dense states
python bench_gated_delta_rule_block.py --with-state-pool

# Per-kernel breakdown behind the stage buckets
python bench_gated_delta_rule_block.py --show-kernels
"""

from __future__ import annotations

import argparse
import importlib.util
import re
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

# One sequence of 8192 on the TP1 shape: the smallest case that exposes the
# full fp32-snapshot store cost (BV=64 at H=32 keeps every CU busy).
DEFAULT_CASE_PATTERN = r"^varlen-qwen-ali-tp1-fp32snapshot.*_mnbt8192_"

_K5_TEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "flydsl_tests"
    / "test_flydsl_linear_attention_prefill.py"
)

# Kernel-name needle -> stage bucket. The K5 needles cover the three backends
# (hip / flydsl / triton); everything unmatched lands in "other".
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


def _load_k5_cases():
    """Return the K5 suite's ``[(case_id, PrefillArgs)]`` list.

    The suite owns the shape definitions (``PrefillGroup`` -> ``PrefillArgs``),
    so it is imported as a plain module rather than duplicated here. It skips
    itself at import time when ROCm or flydsl is missing, which surfaces as an
    exception outside pytest.
    """
    spec = importlib.util.spec_from_file_location(
        "_gdn_k5_prefill_cases", _K5_TEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return list(zip(module.PREFILL_TEST_IDS, module.PREFILL_PARAMS, strict=True))


def _stage_of(kernel_name: str) -> str:
    for needle, stage in STAGE_RULES:
        if needle in kernel_name:
            return stage
    return "other"


def _build_inputs(case, seed, with_state_pool=False):
    """Materialize the block-level tensors a ``PrefillArgs`` case describes.

    The K5 suite builds K5's own operands (k / w / u); the block entry takes the
    layer inputs instead, so q / v / beta are built here from the same shape
    fields. q and k are L2-normalized and v is scaled down to keep the
    recurrence in the numeric range the suite uses.

    ``with_state_pool`` additionally builds the operands of the indexed path:
    a state pool wider than the batch plus the slot indices into it.
    """
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
        # Without cu_seqlens the wrappers take the layout from the tensor
        # shapes, so there is no chunk schedule to prebuild.
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

    # A serving stack holds the states in a pool wider than the running batch
    # and hands K5 the slots this batch owns, so oversize the pool and scatter
    # the slots rather than passing an identity mapping the gather sees through.
    state_pool = make_states(2 * num_states) if with_state_pool else None
    state_indices = (
        torch.randperm(2 * num_states)[:num_states].to(device=device, dtype=torch.int32)
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
    tensors, case, backend, snapshot_dtype, prefill_metadata, state_indices=None
):
    # The indexed path reads and writes the pool, so the states it walks are
    # the ones the previous iteration wrote back. The gates decay, so the
    # recurrence stays in range across a benchmark loop.
    initial_state = (
        tensors["state_pool"] if state_indices is not None else tensors["initial_state"]
    )

    def run():
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
    """Median python-side cost of one block call, measured without syncing.

    The device stays behind the host here, so this is what the host needs to
    enqueue the block: dispatch, allocations and the kernel launches. It is a
    floor on wall time -- a shape whose device work is below it is launch
    bound, and the wall-minus-device gap is that shortfall rather than a stall
    inside a wrapper.
    """
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


def _select_cases(cases, patterns):
    selected = [
        (case_id, case)
        for case_id, case in cases
        if any(re.search(p, case_id) for p in patterns)
    ]
    if not selected:
        raise SystemExit(
            f"No K5 case id matches {patterns}; run --list-cases to see the ids."
        )
    return selected


def _print_summary(records, metrics):
    """Cross-case view: one table per metric, cases down, configs across.

    A case contributes one row per snapshot dtype, so the columns stay dense
    when a sweep mixes bf16 and fp32 groups (the case id already carries the
    shape, the snapshot policy does not fit in it). Each non-HIP backend also
    gets a speedup-vs-HIP column, >1 meaning that backend is the faster one.
    """
    columns, row_keys, cells = [], [], {}
    for rec in records:
        if rec["column"] not in columns:
            columns.append(rec["column"])
        row_key = (rec["case"], rec["snap"])
        if row_key not in row_keys:
            row_keys.append(row_key)
        cells[(row_key, rec["column"])] = rec["row"]

    # (ratio column, numerator column, denominator column) for every backend
    # timed against HIP under the same metadata variant.
    ratios = []
    for column in columns:
        backend, _, meta = column.partition(" ")
        hip_column = f"hip {meta}"
        if backend != "hip" and hip_column in columns:
            ratios.append((f"{backend[:3]}/hip {meta}", hip_column, column))

    case_width = max(len(case_id) for case_id, _ in row_keys) + 2
    col_width = max(12, max(len(column) for column, _, _ in ratios) + 2)
    col_width = max(col_width, max(len(column) for column in columns) + 2)
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
                # host gap goes to zero (and through it, on jitter), so a ratio
                # there would be noise rather than a speedup.
                if hip_row is None or row is None or row[metric] <= 0.0:
                    line += "-".rjust(col_width)
                else:
                    line += f"{hip_row[metric] / row[metric]:{col_width - 1}.2f}x"
            print(line)


def _run_case(case_id, case, args):
    if not case.use_g:
        # The suite's no-g cases exercise K5's padding masking directly; the
        # block entry always cumsums a gate, so there is no g=None path here.
        print(f"\n== {case_id}\nskipped: the block entry requires g")
        return []

    # The indexed path writes the final state back into its pool slot, so it
    # only applies to cases that ask for a final state at all.
    with_pool = args.with_state_pool and case.output_final_state
    tensors = _build_inputs(case, args.seed, with_state_pool=with_pool)
    metadata_variants = [("meta", tensors["metadata"])]
    if args.without_metadata and tensors["metadata"] is not None:
        metadata_variants.append(("no-meta", None))
    # The dense variant carries no suffix, keeping the labels of a run that
    # does not ask for the pool unchanged.
    state_variants = [("", None)]
    if with_pool:
        state_variants.append(("pool", tensors["state_indices"]))

    records = []
    labels, rows, per_kernel_rows = [], {}, {}
    for backend in args.backends:
        for dtype_name in args.snapshot_dtype:
            snapshot_dtype = (
                case.snapshot_dtype if dtype_name == "case" else DTYPES[dtype_name]
            )
            snapshot_name = (
                "fp32" if snapshot_dtype == torch.float32 else "bf16"
            )  # None resolves to k.dtype, i.e. bf16 here
            for meta_name, metadata in metadata_variants:
                for state_name, state_indices in state_variants:
                    run = _make_callable(
                        tensors,
                        case,
                        backend,
                        snapshot_dtype,
                        prefill_metadata=metadata,
                        state_indices=state_indices,
                    )
                    parts = (backend, snapshot_name, meta_name, state_name)
                    label = " ".join(part for part in parts if part)
                    labels.append(label)
                    rows[label], per_kernel_rows[label] = _measure(run, args)
                    records.append(
                        {
                            "case": case_id,
                            "snap": snapshot_name,
                            "column": " ".join(
                                part
                                for part in (backend, meta_name, state_name)
                                if part
                            ),
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
        print(f"\n== {case_id}")
        print(
            f"TP{case.tp} Hg={case.Hg} H={case.H} K={case.K} V={case.V} "
            f"{'varlen' if case.is_varlen else f'dense B={case.dense_batch}'}, "
            f"T={tensors['total_tokens']} ({tensors['num_seqs']} seqs), "
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark the GDN prefill block per K5 backend and snapshot dtype."
    )
    parser.add_argument(
        "--case",
        nargs="+",
        default=[DEFAULT_CASE_PATTERN],
        help="Regexes matched against the K5 suite's case ids (see --list-cases).",
    )
    parser.add_argument(
        "--list-cases",
        action="store_true",
        help="Print the K5 suite case ids this benchmark can run, then exit.",
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
        choices=["case", "bf16", "fp32"],
        default=["case"],
        help="'case' keeps the snapshot policy the case id encodes.",
    )
    parser.add_argument(
        "--without-metadata",
        action="store_true",
        help="Also time each config with prefill_metadata=None, exposing the "
        "chunk-schedule device-to-host copy as host gap.",
    )
    parser.add_argument(
        "--with-state-pool",
        action="store_true",
        help="Also time each config against an indexed state pool, where K5 "
        "gathers each sequence's slot from an index array and writes the "
        "final state back in place. Cases without a final state keep the "
        "dense states only.",
    )
    parser.add_argument(
        "--show-kernels",
        action="store_true",
        help="Print the per-kernel device times behind the stage buckets.",
    )
    parser.add_argument(
        "--summary-metrics",
        nargs="+",
        choices=STAGES + TOTALS,
        default=["wall total", "K5 state scan", "host gap"],
        help="Metrics tabulated across cases when more than one case runs.",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print only the cross-case summary, not the per-case stage tables.",
    )
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--bench-iters", type=int, default=50)
    parser.add_argument("--prof-iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260811)
    args = parser.parse_args()

    cases = _load_k5_cases()
    if args.list_cases:
        for case_id, _ in cases:
            print(case_id)
        return

    props = torch.cuda.get_device_properties(0)
    print(f"\ngfx={props.gcnArchName} CUs={props.multi_processor_count}")
    print(
        f"wall / launch = median of {args.bench_iters} CUDA-event / unsynced "
        f"host timings; device = {args.prof_iters}-iter profiler self time"
    )
    selected = _select_cases(cases, args.case)
    records = []
    for case_id, case in selected:
        records += _run_case(case_id, case, args)

    if records and (len(selected) > 1 or args.summary_only):
        _print_summary(records, args.summary_metrics)


if __name__ == "__main__":
    main()
