# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Regression test: get_meta_param's split indptr must survive graph replay.

`get_meta_param` used to build `num_kv_splits_indptr` inside its own
`@functools.lru_cache`, making the cache entry the tensor's only owner. A graph
that captured a cache hit recorded a bare pointer and no producing kernel, so
once the entry was evicted the storage was recycled and replay read garbage.
Eviction is routine here: `total_kv` is in the key and changes almost every
decode step, and captured keys are never looked up again.

Building the indptr per call fixes it: `torch.arange` is a fill kernel, so
capture records the kernel and every replay re-initialises the buffer.
"""

import argparse
import gc
import sys

import pytest
import torch

import aiter.mla


def _capture(pool, fn):
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, pool=pool):
        out = fn()
    return g, out


def _check_replayed_indptr(bs):
    pool = torch.cuda.graph_pool_handle()
    args = (None, bs, bs * 4096, 16, 1, torch.bfloat16, 1, 1)

    # Warm the cache so capture takes a hit: the condition that recorded a bare
    # pointer with no producing kernel.
    splits, _ = aiter.mla.get_meta_param(*args)
    expected = torch.arange(
        0, (bs + 1) * splits, splits, dtype=torch.int, device="cuda"
    )

    dst = torch.zeros(bs + 1, dtype=torch.int32, device="cuda")
    g, _ = _capture(pool, lambda: dst.copy_(aiter.mla.get_meta_param(*args)[1]))

    # Churn past the cache maxsize to evict, then reuse the pool so the freed
    # storage is handed out again.
    for i in range(1024):
        aiter.mla.get_meta_param(
            None, bs, bs * 4096 + i + 1, 16, 1, torch.bfloat16, 1, 1
        )
    scribble = torch.zeros(1 << 16, dtype=torch.int32, device="cuda")
    g2, _ = _capture(
        pool,
        lambda: scribble.copy_(
            torch.full((1 << 16,), -777, dtype=torch.int32, device="cuda")
        ),
    )
    g2.replay()
    torch.cuda.synchronize()

    dst.zero_()
    g.replay()
    torch.cuda.synchronize()
    assert torch.equal(dst, expected), (
        f"split indptr corrupted after replay (bs={bs}, splits={splits}): "
        f"{dst.tolist()[:16]} != {expected.tolist()[:16]}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_get_meta_param_indptr_survives_replay(l_bs=(8, 480)):
    """Both sizes, in this order, with the small pass released before the next.

    Only bs=480 reproduces, and only once bs=8's churn and freed graph pool have
    left blocks of the right size class in the allocator. Splitting these into
    separate tests would make the failure depend on test ordering.
    """
    for bs in l_bs:
        _check_replayed_indptr(bs)
        print(f"[PASS] split indptr survives graph replay (bs={bs})")
        gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bs",
        type=int,
        default=None,
        help="batch size to check (default: 8 then 480; only 480 reproduces, "
        "and only after the bs=8 pass has run and been released)",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("SKIPPED: needs a GPU")
        sys.exit(0)

    test_get_meta_param_indptr_survives_replay((args.bs,) if args.bs else (8, 480))
