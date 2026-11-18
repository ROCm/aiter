# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Phase 5 dispatch measurement for ``qsa_indexer_select``.

Times today's indexer plus expand against the fused scorer and merge, on the
Phase 0 geometry. Mixed steps also time a split: the decode prefix on today's
indexer, and the prefill rows on the fused path. One scheduler step per case.

Launch from the repo root, in flydsl_venv, through gpuq. Compile first::

    gpuq --num-gpus=1 /workspaces/aiter/flydsl_venv/bin/python \\
        op_tests/bench_qsa_indexer_dispatch.py --compile-only

Then queue the timed run.
"""

from __future__ import annotations

import argparse
import statistics

import torch

from aiter.ops.flydsl import qsa_paged as paged
from aiter.ops.qsa import qsa_expand_block_indices
from op_tests.bench_qsa_indexer_select import (
    COLUMNS,
    COMPRESS,
    HEADS,
    PAGE,
    TOKEN_TOPK,
    Case,
    Request,
    _assert_idle,
    _build,
    _cases,
    _print_device,
    _wide_indexer_cache,
)

WARMUP = 5
ITERS = 10
# Decode rows sit at the front of these Phase 0 mixed steps.
_DECODE_ROWS = {"mix3": 3, "mix16": 16, "mix64": 64, "mtp": 4}


def _dispatch_cases() -> tuple[Case, ...]:
    """Serving steps, plus a 16-row tail at 8K and 128K."""
    named = {case.name: case for case in _cases()}
    keep = ("p8k", "p32k", "p128k", "p256k", "mix3", "mix16", "mix64", "mtp")
    small = (
        Case("m16_8k", (Request(16, 8192, 8192 - 16),)),
        Case("m16_128k", (Request(16, 131072, 131072 - 16),)),
    )
    return tuple(named[name] for name in keep) + small


def _max_seq_len(case: Case) -> int:
    return max(req.context for req in case.requests)


def _decode_rows(case: Case) -> int:
    return _DECODE_ROWS.get(case.name, 0)


def _buffers(q: torch.Tensor, n_columns: int, max_seq_len: int):
    rows = int(q.shape[0])
    block_ids = torch.empty(rows, 512, dtype=torch.int32, device=q.device)
    token_ids = torch.empty(
        rows, TOKEN_TOPK + COMPRESS - 1, dtype=torch.int32, device=q.device
    )
    scores = torch.empty(rows, n_columns, dtype=torch.float32, device=q.device)
    scratch_elems = paged.qsa_indexer_select_workspace_elems(rows, max_seq_len, PAGE)
    scratch = torch.empty(scratch_elems, dtype=torch.float32, device=q.device)
    return block_ids, token_ids, scores, scratch


def _existing(
    q, index_k, table, token_to_req, positions, context, block_ids, token_ids, scores
):
    paged.qsa_indexer_block_ids(
        q,
        index_k,
        table,
        token_to_req,
        positions,
        context,
        out=block_ids,
        heads=(HEADS,),
        scores=scores,
    )
    qsa_expand_block_indices(
        block_ids,
        positions,
        context,
        token_to_req,
        COMPRESS,
        TOKEN_TOPK,
        token_ids,
    )


def _fused(
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
):
    paged._write_fused_indexer_block_ids(
        q,
        index_k,
        table,
        token_to_req,
        positions,
        context,
        block_ids,
        token_ids,
        paged._SCORE_SCALE,
        scratch,
        max_seq_len,
    )


def _arms(case: Case, built, index_k, scratch_all, scores_all, block_ids, token_ids):
    """Callables for this case. Outputs are the preallocated full-step buffers."""
    q, _q_gqa, table, token_to_req, positions, context, *_rest = built
    max_seq_len = _max_seq_len(case)
    decode = _decode_rows(case)
    rows = int(q.shape[0])

    def existing():
        _existing(
            q,
            index_k,
            table,
            token_to_req,
            positions,
            context,
            block_ids,
            token_ids,
            scores_all,
        )

    def fused():
        _fused(
            q,
            index_k,
            table,
            token_to_req,
            positions,
            context,
            block_ids,
            token_ids,
            scratch_all,
            max_seq_len,
        )

    arms = {"existing": existing, "fused": fused}
    if 0 < decode < rows:

        def split():
            _existing(
                q[:decode],
                index_k,
                table,
                token_to_req[:decode],
                positions[:decode],
                context,
                block_ids[:decode],
                token_ids[:decode],
                scores_all[:decode],
            )
            _fused(
                q[decode:],
                index_k,
                table,
                token_to_req[decode:],
                positions[decode:],
                context,
                block_ids[decode:],
                token_ids[decode:],
                scratch_all,
                max_seq_len,
            )

        arms["split"] = split
    return arms


def _median_us(fn) -> float:
    for _ in range(WARMUP):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)
    return statistics.median(samples)


def _check_workspace_helper() -> None:
    """The helper matches one span through 128K and two spans at 256K."""
    per_row = 512 + 256
    assert paged.qsa_indexer_select_workspace_elems(16384, 8192) == 16384 * per_row
    assert paged.qsa_indexer_select_workspace_elems(16384, 32768) == 16384 * per_row
    assert paged.qsa_indexer_select_workspace_elems(16384, 131072) == 16384 * per_row
    assert (
        paged.qsa_indexer_select_workspace_elems(16384, 262144) == 2 * 16384 * per_row
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 indexer dispatch measurement")
    parser.add_argument(
        "--compile-only",
        action="store_true",
        help="Launch each arm once so the kernels are compiled, then exit.",
    )
    args = parser.parse_args()
    _check_workspace_helper()
    _print_device()
    _assert_idle()
    index_k = _wide_indexer_cache()
    print(
        f"indexer pool pages={index_k.shape[0]} "
        f"stride={tuple(int(s) for s in index_k.stride())} "
        f"span_bytes={paged.span_bytes(index_k)} columns={COLUMNS}",
        flush=True,
    )
    cases = _dispatch_cases()
    if args.compile_only:
        for case in cases:
            print(f"{case.name}: compile launch", flush=True)
            built = _build(case, index_k)
            q = built[0]
            block_ids, token_ids, scores, scratch = _buffers(
                q, COLUMNS, _max_seq_len(case)
            )
            for name, fn in _arms(
                case, built, index_k, scratch, scores, block_ids, token_ids
            ).items():
                print(f"{case.name}: compile {name}", flush=True)
                fn()
                torch.cuda.synchronize()
            del built
        print("COMPILE_DONE", flush=True)
        return

    # Two passes reverse the arm order so the first launch is not the only warm one.
    orders = (("existing", "fused", "split"), ("split", "fused", "existing"))
    print("case | arm | pass | us", flush=True)
    for case in cases:
        built = _build(case, index_k)
        q = built[0]
        block_ids, token_ids, scores, scratch = _buffers(q, COLUMNS, _max_seq_len(case))
        arms = _arms(case, built, index_k, scratch, scores, block_ids, token_ids)
        for pass_id, order in enumerate(orders, start=1):
            for name in order:
                if name not in arms:
                    continue
                us = _median_us(arms[name])
                print(f"{case.name} | {name} | {pass_id} | {us:.1f}", flush=True)
        del built
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
