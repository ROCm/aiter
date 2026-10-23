# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The v1_0 (intra-batch) MLA metadata must cover every key of every batch exactly once."""

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


def _covered_ranges(kv_lens, kv_start=0, kv_granularity=16, nhead=16):
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
        fast_mode=False,
        num_kv_splits=64,
        intra_batch_mode=True,
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
        fast_mode=False,
        topk=2048,
        max_split_per_batch=64,
        intra_batch_mode=True,
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


@pytest.mark.parametrize("kv_granularity", [16, 32, 64])
def test_every_key_length_is_fully_covered(kv_granularity):
    # A 1-4 key tail folded into the only remaining split used to be dropped.
    _assert_exact_cover(*_covered_ranges(list(range(1, 401)), 0, kv_granularity))


def test_first_batch_starts_at_its_indptr():
    # Batch 0 used to start at key 0 instead of kv_indptr[0].
    _assert_exact_cover(*_covered_ranges([1, 17, 300, 2048, 5], kv_start=1000))


if __name__ == "__main__":
    for granularity in (16, 32, 64):
        test_every_key_length_is_fully_covered(granularity)
    test_first_batch_starts_at_its_indptr()
    print("v1_0 MLA metadata kv ranges: OK")
