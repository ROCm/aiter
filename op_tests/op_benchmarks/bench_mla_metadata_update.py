# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Compare MLA rebuild, whole-row update and legacy decode-update launch cost.

Run as ``python -m op_tests.op_benchmarks.bench_mla_metadata_update``.
Use this benchmark and its test helpers with base AITER and ``--legacy-only``
for the default-mode comparison. Allocations and seed planning are not timed.
"""

import argparse
import time

import torch

import aiter
from aiter.jit.utils.chip_info import get_gfx_runtime
from aiter.ops.attention import decode_update_mla_metadata_v1
from aiter.test_common import run_perftest
from op_tests.test_mla_decode_update_metadata import (
    PLAN_ORDER,
    _kv_indptr,
    _ones,
    _plan,
    _qo_indptr,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", type=int, nargs="+", default=[16, 4096])
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--legacy-only", action="store_true")
    args = parser.parse_args()
    for bs in args.batch:
        run_case(bs, args.length, args.iterations, args.legacy_only)


def run_case(bs, length, iterations, legacy_only):
    qd, n = torch.bfloat16, 16
    granularity = max(16, 1 << (length - 1).bit_length())
    qo, indptr, last = _qo_indptr(bs, 1), _kv_indptr([length] * bs), _ones(bs)
    meta = _plan([1] * bs, 1, n, qd, granularity=granularity)
    legacy = _plan([length] * bs, 1, n, qd, granularity=granularity)
    targets = _plan([length] * bs, 1, n, qd, granularity=granularity)

    def update(whole_row=False):
        selected = meta if whole_row else legacy
        mode = {"whole_row": True} if whole_row else {"num_reject_tokens": last}
        decode_update_mla_metadata_v1(
            *(qo, indptr, last, n, 1, True),
            *(selected[k] for k in PLAN_ORDER),
            kv_granularity=granularity,
            dtype_q=qd,
            dtype_kv=qd,
            **mode,
        )

    def rebuild():
        aiter.get_mla_metadata_v1(
            *(qo, indptr, last, n, 1, True),
            *(targets[k] for k in PLAN_ORDER),
            kv_granularity=granularity,
            max_seqlen_qo=1,
            uni_seqlen_qo=1,
            dtype_q_nope=qd,
            dtype_kv_nope=qd,
            max_split_per_batch=16,
        )

    calls = {"decode_update": update}
    if not legacy_only:
        calls.update(whole_row=lambda: update(True), rebuild=rebuild)
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        _, device_us = run_perftest(call, num_iters=iterations)
        started = time.perf_counter()
        for _ in range(iterations):
            call()
        host_us = (time.perf_counter() - started) * 1e6 / iterations
        torch.cuda.synchronize()
        print(
            f"{get_gfx_runtime()} {bs=} {length=} bf16 qh16 {name}: "
            f"device_us={device_us:.3f} host_enqueue_us={host_us:.3f}"
        )


if __name__ == "__main__":
    main()
