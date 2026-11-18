# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Time fused span widths on the losing Phase 6 cases.

A span is one workgroup's column tiles (64 columns each). Launch from the
repo root, in flydsl_venv, through gpuq. ``--set-gate`` runs the Phase 2
oracle and set gate at ``--width`` without changing the module default.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from aiter.ops.flydsl import qsa_paged as paged
from aiter.ops.flydsl.kernels.qsa import indexer_fused_select as fused_mod
from op_tests.bench_qsa_indexer_dispatch import _fused, _max_seq_len
from op_tests.bench_qsa_indexer_select import (
    COLUMNS,
    PAGE,
    TOKEN_TOPK,
    _assert_idle,
    _build,
    _cases,
    _print_device,
    _wide_indexer_cache,
)

# 514, 128, 32, 16, 8 and 4 tiles of 64. 32928 is the Phase 6 gate width.
_WIDTHS = (32928, 8192, 2048, 1024, 512, 256)
_NAMES = ("p8k", "p128k", "p256k")


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
    parser.add_argument("--set-gate", action="store_true")
    parser.add_argument("--width", type=int, default=512)
    args = parser.parse_args()
    _print_device()
    _assert_idle()
    if args.set_gate:
        from op_tests.test_flydsl_qsa import (
            run_qsa_indexer_select_gate,
            run_qsa_indexer_select_oracle,
        )

        fused_mod.FUSED_SPLIT_COLUMNS = args.width
        paged._fused_select_wins = lambda *_args, **_kwargs: True
        run_qsa_indexer_select_oracle()
        run_qsa_indexer_select_gate()
        print(f"SPLIT_GATE_OK width={args.width}", flush=True)
        return
    index_k = _wide_indexer_cache()
    cases = {item.name: item for item in _cases()}
    built = {name: _build(cases[name], index_k) for name in _NAMES}
    print("TUNE_READY", flush=True)
    for width in _WIDTHS:
        fused_mod.FUSED_SPLIT_COLUMNS = width
        for name in _NAMES:
            case = cases[name]
            q = built[name][0]
            max_seq_len = _max_seq_len(case)
            spans = fused_mod.fused_split_spans(max_seq_len, COLUMNS, PAGE, paged._R)
            rows = int(q.shape[0])
            block_ids = torch.empty(rows, 512, dtype=torch.int32, device=q.device)
            token_ids = torch.empty(
                rows, TOKEN_TOPK + 3, dtype=torch.int32, device=q.device
            )
            scratch = torch.empty(
                len(spans) * rows * (512 + fused_mod.HIST_BINS),
                dtype=torch.float32,
                device=q.device,
            )
            table, token_to_req, positions, context = built[name][2:6]

            def once(
                q=q,
                table=table,
                token_to_req=token_to_req,
                positions=positions,
                context=context,
                block_ids=block_ids,
                token_ids=token_ids,
                scratch=scratch,
                max_seq_len=max_seq_len,
            ):
                _fused(
                    q,
                    index_k,
                    table,
                    token_to_req,
                    positions,
                    context,
                    block_ids,
                    token_ids,
                    scratch,
                    max_seq_len,
                )

            us = _median_us(once, 2, 5)
            tiles = (spans[0][1] - spans[0][0] + 63) // 64
            print(
                f"TUNE {name} width={width} spans={len(spans)} "
                f"tiles={tiles} us={us:.1f}",
                flush=True,
            )
            del scratch, block_ids, token_ids
    fused_mod.FUSED_SPLIT_COLUMNS = _WIDTHS[0]
    print("TUNE_DONE", flush=True)


if __name__ == "__main__":
    main()
