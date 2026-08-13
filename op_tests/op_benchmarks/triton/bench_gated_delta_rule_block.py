# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Benchmark the whole GDN prefill block per K5 backend x snapshot dtype.

``bench_gated_delta_rule_snapshot_dtype.py`` times the snapshot producer (K5)
and consumer (K6) in isolation. This benchmark instead runs the full
``chunk_gated_delta_rule_opt_vk`` block a serving stack calls, and splits the
measurement three ways:

* wall time -- the host-visible block latency (median of CUDA-event timings),
* device time per stage -- profiler self time bucketed into K1..K6,
* host gap -- wall minus device total, i.e. launch overhead and any host
  stall a wrapper introduces.

Shapes are the ``PrefillArgs`` cases from the K5 test suite
(``op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py``), selected by
their pytest ids, so a block measurement and the K5-only table in that suite
describe the same workload. ``--list-cases`` prints the available ids.

The block is timed with a prebuilt ``GatedDeltaRulePrefillMetadata``, matching
how a serving stack builds the chunk schedule once and reuses it across the
layer stack. Pass ``--without-metadata`` to also time the metadata=None path,
where each wrapper recovers the chunk counts with a blocking device-to-host
copy that shows up as host gap rather than device time.

Usage examples
--------------
# Default: the varlen-qwen-ali-tp1 T=8192 single-sequence case, bf16 vs fp32
# snapshots, flydsl vs hip
python bench_gated_delta_rule_block.py

# List the case ids this benchmark can run
python bench_gated_delta_rule_block.py --list-cases

# Any subset of the suite, matched as regexes against the case ids
python bench_gated_delta_rule_block.py --case "varlen-qwen3.5-397b.*mnbt8192_"

# Quantify what skipping the prebuilt chunk schedule costs
python bench_gated_delta_rule_block.py --without-metadata

# Per-kernel breakdown behind the stage buckets
python bench_gated_delta_rule_block.py --show-kernels
"""

from __future__ import annotations

import argparse
import importlib.util
import re
import sys
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
TOTALS = ["kernel total", "wall total", "host gap"]

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


def _build_inputs(case, seed):
    """Materialize the block-level tensors a ``PrefillArgs`` case describes.

    The K5 suite builds K5's own operands (k / w / u); the block entry takes the
    layer inputs instead, so q / v / beta are built here from the same shape
    fields. q and k are L2-normalized and v is scaled down to keep the
    recurrence in the numeric range the suite uses.
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
    initial_state = (
        torch.randn(
            num_states, num_heads, case.V, case.K, device=device, dtype=torch.float32
        )
        * 0.01
    ).to(case.ssm_state_dtype)

    return {
        "q": q,
        "k": k,
        "v": v,
        "g": g,
        "beta": beta,
        "initial_state": initial_state,
        "cu_seqlens": cu_seqlens,
        "metadata": metadata,
        "total_tokens": total_tokens,
        "num_seqs": num_states,
    }


def _make_callable(tensors, case, backend, snapshot_dtype, prefill_metadata):
    def run():
        return chunk_gated_delta_rule_opt_vk(
            q=tensors["q"],
            k=tensors["k"],
            v=tensors["v"],
            g=tensors["g"],
            beta=tensors["beta"],
            initial_state=tensors["initial_state"],
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
    per_kernel = _profile_kernels_us(run, args.warmup_iters, args.prof_iters)
    row = {stage: 0.0 for stage in STAGES}
    for name, us in per_kernel.items():
        row[_stage_of(name)] += us
    row["kernel total"] = sum(per_kernel.values())
    row["wall total"] = wall_us
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


def _run_case(case_id, case, args):
    if not case.use_g:
        # The suite's no-g cases exercise K5's padding masking directly; the
        # block entry always cumsums a gate, so there is no g=None path here.
        print(f"\n== {case_id}\nskipped: the block entry requires g")
        return

    tensors = _build_inputs(case, args.seed)
    metadata_variants = [("meta", tensors["metadata"])]
    if args.without_metadata and tensors["metadata"] is not None:
        metadata_variants.append(("no-meta", None))

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
                run = _make_callable(
                    tensors, case, backend, snapshot_dtype, prefill_metadata=metadata
                )
                label = f"{backend} {snapshot_name} {meta_name}"
                labels.append(label)
                rows[label], per_kernel_rows[label] = _measure(run, args)

    print(f"\n== {case_id}")
    print(
        f"TP{case.tp} Hg={case.Hg} H={case.H} K={case.K} V={case.V} "
        f"{'varlen' if case.is_varlen else f'dense B={case.dense_batch}'}, "
        f"T={tensors['total_tokens']} ({tensors['num_seqs']} seqs), "
        f"final_state={'on' if case.output_final_state else 'off'}"
    )
    _print_table(labels, rows)

    if args.show_kernels:
        print("\n-- per-kernel device time (us)")
        for label in labels:
            print(f"   {label}")
            for name, us in sorted(per_kernel_rows[label].items(), key=lambda x: -x[1]):
                print(f"     {us:8.1f}  [{_stage_of(name)}]  {name[:100]}")


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
        "--show-kernels",
        action="store_true",
        help="Print the per-kernel device times behind the stage buckets.",
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
        f"wall = median of {args.bench_iters} CUDA-event timings; "
        f"device = {args.prof_iters}-iter profiler self time"
    )
    for case_id, case in _select_cases(cases, args.case):
        _run_case(case_id, case, args)


if __name__ == "__main__":
    main()
