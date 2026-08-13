# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Sweep BV for the FlyDSL K5 mfma16_hip kernel and emit tuned-table rows.

The kernel picks its value tile at runtime: a measured row in
``aiter/ops/flydsl/chunk_gdn_h_mfma16_hip_tuned.csv`` wins, otherwise the
``_hipeq_select_bv`` CU/LDS rule decides. This script produces those rows. It
times K5 alone (BV changes nothing else in the block) across BV in {16,32,64}
via the ``FLYDSL_K5_MFMA16HIP_BV`` override, and prints both the winner and what
the rule would have chosen, so a row is only worth keeping when the two differ.

Shapes are the ``PrefillArgs`` cases of the K5 suite
(``op_tests/flydsl_tests/test_flydsl_linear_attention_prefill.py``), selected by
pytest id, so the table and the suite describe the same workloads. Gate layout
and log2 scaling follow the production dispatch in ``chunk.py`` (head-major,
pre-scaled) rather than a case's own gate flags, since the table serves that
path.

Usage examples
--------------
# The two families currently in the tuned table
python tune_gated_delta_rule_bv.py

# Any subset of the suite, matched as regexes against case ids
python tune_gated_delta_rule_bv.py --case "^varlen-32k-qwen-"

# Append the measured rows straight to the tuned table
python tune_gated_delta_rule_bv.py --append-csv

# Only emit rows where the measured winner beats the rule's choice
python tune_gated_delta_rule_bv.py --only-improvements
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import statistics
import sys
from pathlib import Path

import torch

from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    _GFX_ARCH,
    _hipeq_select_bv,
)
from aiter.ops.flydsl.linear_attention_prefill_kernels import (
    chunk_gated_delta_rule_fwd_h_flydsl_mfma16_hip as k5,
)

CHUNK_SIZE = 64
BV_CANDIDATES = (16, 32, 64)
DEFAULT_CASE_PATTERNS = [
    r"^varlen-qwen3\.5-397b-ptpc-ali-",
    r"^varlen-qwen-ali-tp1-",
]

_K5_TEST_PATH = (
    Path(__file__).resolve().parents[2]
    / "flydsl_tests"
    / "test_flydsl_linear_attention_prefill.py"
)
_TUNED_CSV = (
    Path(__file__).resolve().parents[3]
    / "aiter"
    / "ops"
    / "flydsl"
    / "chunk_gdn_h_mfma16_hip_tuned.csv"
)


def _load_k5_cases():
    spec = importlib.util.spec_from_file_location(
        "_gdn_k5_prefill_cases", _K5_TEST_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return list(zip(module.PREFILL_TEST_IDS, module.PREFILL_PARAMS, strict=True))


def _chunk_counts(context_lens, batch):
    """(total_chunks, max_seq_chunks) -- the two host-side quantities that key
    both the runtime lookup and the BV rule."""
    per_seq = [(n + CHUNK_SIZE - 1) // CHUNK_SIZE for n in context_lens]
    return sum(per_seq) * batch, max(per_seq)


def _build_k5_inputs(case, snapshot_dtype, seed=0):
    """K5's own operands for a case. Values are irrelevant to BV timing (the
    kernel has no data-dependent control flow), so they are random."""
    torch.manual_seed(seed)
    dev = torch.device("cuda")
    context_lens = case.resolve_context_lens()
    T_flat = sum(context_lens)
    H, Hg = case.H, case.Hg
    batch = 1 if case.is_varlen else case.dense_batch
    num_states = len(context_lens) if case.is_varlen else batch

    cu_seqlens = None
    if case.is_varlen:
        cu_seqlens = torch.tensor(
            [0] + torch.tensor(context_lens).cumsum(0).tolist(),
            dtype=torch.int32,
            device=dev,
        )

    args = {
        "k": torch.randn((batch, T_flat, Hg, case.K), device=dev, dtype=case.dtype),
        "w": torch.randn((batch, H, T_flat, case.K), device=dev, dtype=case.dtype),
        "u": torch.randn((batch, H, T_flat, case.V), device=dev, dtype=case.dtype),
        "g": torch.randn((batch, H, T_flat), device=dev, dtype=torch.float32) * -0.1,
        "initial_state": torch.zeros(
            (num_states, H, case.V, case.K), device=dev, dtype=case.ssm_state_dtype
        ),
        "output_final_state": case.output_final_state,
        "cu_seqlens": cu_seqlens,
        "state_dtype": case.ssm_state_dtype,
        "snapshot_dtype": snapshot_dtype,
        "g_head_major": True,
    }
    total_chunks, max_seq_chunks = _chunk_counts(context_lens, batch)
    return args, T_flat, num_states, total_chunks, max_seq_chunks


def _bench_us(args, warmup, iters):
    for _ in range(warmup):
        k5(**args)
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        starts[i].record()
        k5(**args)
        ends[i].record()
    torch.cuda.synchronize()
    return statistics.median(s.elapsed_time(e) * 1e3 for s, e in zip(starts, ends))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--case", nargs="+", default=DEFAULT_CASE_PATTERNS)
    parser.add_argument("--list-cases", action="store_true")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--append-csv",
        action="store_true",
        help=f"append the emitted rows to {_TUNED_CSV.name}",
    )
    parser.add_argument(
        "--only-improvements",
        action="store_true",
        help="emit a row only when the measured BV beats the rule's choice",
    )
    args = parser.parse_args()

    cases = _load_k5_cases()
    if args.list_cases:
        for case_id, _ in cases:
            print(case_id)
        return

    patterns = [re.compile(p) for p in args.case]
    selected = [(cid, c) for cid, c in cases if any(p.search(cid) for p in patterns)]
    if not selected:
        print(f"no case matches {args.case}")
        sys.exit(1)

    header = (
        f"{'case':58s} {'chunks':>7s} "
        + " ".join(f"BV{bv:<7d}" for bv in BV_CANDIDATES)
    ) + f" {'best':>5s} {'rule':>5s} {'gain%':>6s}"
    print(header)
    print("-" * len(header))

    emitted: dict[tuple, str] = {}
    for case_id, case in selected:
        if case.K != 128 or case.V != 128 or case.BT != 64:
            print(f"{case_id:58s} skipped (kernel supports K=V=128, BT=64 only)")
            continue
        snapshot_dtype = case.snapshot_dtype or case.dtype
        inputs, T_flat, N, total_chunks, max_seq_chunks = _build_k5_inputs(
            case, snapshot_dtype
        )

        times = {}
        for bv in BV_CANDIDATES:
            if case.V % bv:
                continue
            os.environ["FLYDSL_K5_MFMA16HIP_BV"] = str(bv)
            try:
                times[bv] = _bench_us(inputs, args.warmup, args.iters)
            finally:
                os.environ.pop("FLYDSL_K5_MFMA16HIP_BV", None)

        best = min(times, key=times.get)
        rule = _hipeq_select_bv(
            torch.device("cuda:0"), case.H, total_chunks, max_seq_chunks
        )
        gain = (times[rule] - times[best]) / times[rule] * 100 if rule in times else 0.0
        cells = " ".join(f"{times.get(bv, float('nan')):8.1f}" for bv in BV_CANDIDATES)
        print(
            f"{case_id:58s} {total_chunks:7d} {cells} {best:5d} {rule:5d} {gain:6.1f}"
        )

        if not args.only_improvements or best != rule:
            # ``_build_k5_inputs`` always passes an initial state, so use_h0 is
            # True for every measured row; store_fs follows the case.
            key = (
                _GFX_ARCH,
                case.H,
                case.Hg,
                case.is_varlen,
                case.output_final_state,
                snapshot_dtype is torch.bfloat16,
                case.ssm_state_dtype is torch.bfloat16,
                total_chunks,
                max_seq_chunks,
            )
            emitted[key] = (
                f"{case.model_name},{_GFX_ARCH},{case.dtype},{case.K},{case.V},"
                f"{case.BT},{case.H},{case.Hg},{case.is_varlen},True,"
                f"{case.output_final_state},"
                f"{snapshot_dtype is torch.bfloat16},"
                f"{case.ssm_state_dtype is torch.bfloat16},{T_flat},{N},"
                f"{total_chunks},{max_seq_chunks},{best},{times[best]:.1f}"
            )

        del inputs
        torch.cuda.empty_cache()

    print(f"\n{len(emitted)} tuned rows:")
    for line in emitted.values():
        print(line)

    if args.append_csv and emitted:
        with open(_TUNED_CSV, "a", encoding="utf-8") as f:
            for line in emitted.values():
                f.write(line + "\n")
        print(f"\nappended to {_TUNED_CSV}")


if __name__ == "__main__":
    main()
