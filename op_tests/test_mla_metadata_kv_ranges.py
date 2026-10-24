# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The v1_0 (intra-batch) and v1_2 (fast) MLA metadata must cover every key of every batch exactly once."""

import pytest
import torch

import aiter
from aiter import dtypes

NAMES = (
    "work_meta_data",
    "work_indptr",
    "work_info_set",
    "reduce_indptr",
    "reduce_final_map",
    "reduce_partial_map",
)


# (fast_mode, intra_batch_mode): v1_0 is the "fake non-persistent" planner, v1_2 the fast one.
PLANNERS = {"v1_0": (False, True), "v1_2": (True, False)}


def _covered_ranges(kv_lens, kv_start=0, kv_granularity=16, nhead=16, planner="v1_0"):
    fast_mode, intra_batch_mode = PLANNERS[planner]
    batch = len(kv_lens)
    kv_indptr = torch.full((batch + 1,), kv_start, dtype=torch.int32, device="cuda")
    kv_indptr[1:] += torch.tensor(kv_lens, device="cuda").cumsum(0).int()
    qo_indptr = torch.arange(batch + 1, dtype=torch.int32, device="cuda")
    last_page_lens = torch.ones(batch, dtype=torch.int32, device="cuda")
    info = aiter.get_mla_metadata_info_v1(
        batch,
        1,
        nhead,
        dtypes.bf16,
        dtypes.fp8,
        is_sparse=True,
        fast_mode=fast_mode,
        num_kv_splits=64,
        intra_batch_mode=intra_batch_mode,
    )
    meta = {
        name: torch.empty(shape, dtype=dtype, device="cuda")
        for name, (shape, dtype) in zip(NAMES, info)
    }
    aiter.get_mla_metadata_v1(
        qo_indptr,
        kv_indptr,
        last_page_lens,
        nhead,
        1,
        False,
        meta["work_meta_data"],
        meta["work_info_set"],
        meta["work_indptr"],
        meta["reduce_indptr"],
        meta["reduce_final_map"],
        meta["reduce_partial_map"],
        page_size=1,
        kv_granularity=kv_granularity,
        max_seqlen_qo=1,
        uni_seqlen_qo=1,
        fast_mode=fast_mode,
        topk=-1,
        max_split_per_batch=64,
        intra_batch_mode=intra_batch_mode,
        dtype_q=dtypes.bf16,
        dtype_kv=dtypes.fp8,
    )
    num_works = meta["work_indptr"][-1].item()
    works = meta["work_info_set"][:num_works].tolist()
    ranges = {b: [] for b in range(batch)}
    for batch_idx, _, _, _, kv_begin, kv_end, *_ in works:
        ranges[batch_idx].append((kv_begin, kv_end))
    return kv_indptr.tolist(), ranges


def _assert_exact_cover(kv_indptr, ranges):
    for b, spans in ranges.items():
        spans.sort()
        assert spans[0][0] == kv_indptr[b], (b, spans, kv_indptr[b])
        assert spans[-1][1] == kv_indptr[b + 1], (b, spans, kv_indptr[b + 1])
        for (_, end), (start, _) in zip(spans, spans[1:]):
            assert end == start, (b, spans)


@pytest.mark.parametrize("planner", list(PLANNERS))
@pytest.mark.parametrize("kv_granularity", [16, 32, 64])
def test_every_key_length_is_fully_covered(planner, kv_granularity):
    # v1_0 used to drop a 1-4 key tail folded into the only remaining split.
    _assert_exact_cover(
        *_covered_ranges(list(range(1, 401)), 0, kv_granularity, planner=planner)
    )


@pytest.mark.parametrize("planner", list(PLANNERS))
@pytest.mark.parametrize("batches", [1, 3, 10, 240, 1100])
def test_ranges_start_at_the_indptr_base(planner, batches):
    # v1_0 started batch 0 at key 0, v1_2 started every batch there, instead of at
    # kv_indptr[0]; the batch counts reach each of v1_2's code paths.
    lens = [(7 * i) % 300 + 1 for i in range(batches)]
    _assert_exact_cover(*_covered_ranges(lens, kv_start=1000, planner=planner))


if __name__ == "__main__":
    for planner in PLANNERS:
        for granularity in (16, 32, 64):
            test_every_key_length_is_fully_covered(planner, granularity)
        for batches in (1, 3, 10, 240, 1100):
            test_ranges_start_at_the_indptr_base(planner, batches)
    print("MLA metadata kv ranges: OK")
