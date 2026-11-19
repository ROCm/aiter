# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Check the retuned fused scorer, then time one p128k step against today's indexer.

Launch from the repo root, in flydsl_venv, through gpuq.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from aiter.ops.flydsl import qsa_paged as paged
from op_tests.bench_qsa_indexer_dispatch import _arms, _buffers, _max_seq_len
from op_tests.bench_qsa_indexer_select import (
    COLUMNS,
    _assert_idle,
    _build,
    _cases,
    _print_device,
    _wide_indexer_cache,
)
from op_tests.test_flydsl_qsa import (
    run_qsa_indexer_select_gate,
    run_qsa_indexer_select_oracle,
)


def _median_us(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--time-only", action="store_true")
    args = parser.parse_args()
    _print_device()
    _assert_idle()
    if not args.time_only:
        paged._fused_select_wins = lambda *_args, **_kwargs: True
        run_qsa_indexer_select_oracle()
        run_qsa_indexer_select_gate()
        print("FUSED_GATE_OK", flush=True)
    index_k = _wide_indexer_cache()
    case = next(item for item in _cases() if item.name == "p128k")
    built = _build(case, index_k)
    block_ids, token_ids, scores, scratch = _buffers(
        built[0], COLUMNS, _max_seq_len(case)
    )
    arms = _arms(case, built, index_k, scratch, scores, block_ids, token_ids)
    print(f"p128k existing {_median_us(arms['existing'], 2, 5):.1f}", flush=True)
    print(f"p128k fused {_median_us(arms['fused'], 2, 5):.1f}", flush=True)
    print("SCOUT_DONE", flush=True)


if __name__ == "__main__":
    main()
