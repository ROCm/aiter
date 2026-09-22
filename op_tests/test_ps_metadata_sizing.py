# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Buffer-sizing invariants of ``aiter.get_ps_metadata_info_v1``.

``get_ps_metadata_info_v1`` is pure Python, but it has to bound what the C++
generator ``get_ps_metadata_v1`` (csrc/kernels/mla/metadata/v1_2_host.cuh)
actually emits. The bound assuming every batch carries ``max_qlen`` query
tokens is far looser than a serving engine ever produces, so the function takes
an optional ``total_qlen`` token budget; these tests pin both the loose default
and the safety of the budget-aware bound against the real generator.

Run:
    python3 -m pytest op_tests/test_ps_metadata_sizing.py -v
"""

import math

import pytest
import torch

import aiter

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="get_ps_metadata_info_v1 sizes against the current device",
)

BLOCK_SIZE = 16
KVLEN_GRANULARITY = 16
CONTEXT_LEN = 64

QLEN_GRANULARITIES = [16, 256]
BATCH_SIZES = [1, 4, 17]

WORKLOADS = {
    "one_long_rest_single_token": lambda b, g: [7 * g + 1] + [1] * (b - 1),
    "every_batch_one_token_over_a_tile": lambda b, g: [g + 1] * b,
    "every_batch_exactly_one_tile": lambda b, g: [g] * b,
    "ragged": lambda b, g: [(i % 5) * g + 1 for i in range(b)],
}


def _info(batch_size, num_head_k, max_qlen, qlen_granularity, total_qlen=None):
    return aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_head_k,
        max_qlen=max_qlen,
        qlen_granularity=qlen_granularity,
        total_qlen=total_qlen,
    )


def _legacy_info(batch_size, num_head_k, max_qlen, qlen_granularity):
    """The sizes get_ps_metadata_info_v1 returned before total_qlen existed."""
    cu_num = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    cus_per_cluster = cu_num // math.gcd(num_head_k, cu_num)
    max_qo_split_per_batch = math.ceil(max_qlen / qlen_granularity)
    qo_tile_cnt = batch_size * max_qo_split_per_batch
    max_works = (batch_size + cus_per_cluster - 1) * max_qo_split_per_batch * num_head_k
    return (
        (2, torch.uint64),
        (cu_num + 1, torch.int32),
        ((max_works, 8), torch.int32),
        (qo_tile_cnt + 1, torch.int32),
        ((qo_tile_cnt, 2), torch.int32),
        (qo_tile_cnt + cus_per_cluster - 1, torch.int32),
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("qlen_granularity", QLEN_GRANULARITIES)
@pytest.mark.parametrize("num_head_k", [1, 8])
def test_unset_total_qlen_keeps_the_legacy_sizes(
    batch_size, qlen_granularity, num_head_k
):
    max_qlen = 8 * qlen_granularity + 3
    info = _info(batch_size, num_head_k, max_qlen, qlen_granularity)
    assert info == _legacy_info(batch_size, num_head_k, max_qlen, qlen_granularity)
    assert (
        _info(batch_size, num_head_k, max_qlen, qlen_granularity, total_qlen=None)
        == info
    )


@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("qlen_granularity", QLEN_GRANULARITIES)
@pytest.mark.parametrize("divisor", [1, 2, 8, 64])
def test_budget_never_grows_the_buffers(batch_size, qlen_granularity, divisor):
    max_qlen = 8 * qlen_granularity + 3
    unbounded = _info(batch_size, 1, max_qlen, qlen_granularity)
    bounded = _info(
        batch_size,
        1,
        max_qlen,
        qlen_granularity,
        total_qlen=max(batch_size * max_qlen // divisor, 1),
    )
    for got, ref in zip(bounded[2:6], unbounded[2:6]):
        assert got[0] <= ref[0]
    # a single request is still allowed to be max_qlen long under any budget
    assert bounded[4][0] >= math.ceil(max_qlen / qlen_granularity)


@pytest.mark.parametrize("total_qlen", [0, -1])
def test_non_positive_total_qlen_is_rejected(total_qlen):
    # a bogus budget must not silently fall back to the unbounded sizes
    with pytest.raises(AssertionError):
        _info(4, 1, 8 * 256, 256, total_qlen=total_qlen)


def _generate(qlens, qlen_granularity, is_causal, need_lse):
    """Size buffers from the declared budget, then let the generator fill them."""
    batch_size = len(qlens)
    (
        (work_metadata_ptrs_size, work_metadata_ptrs_type),
        (work_indptr_size, work_indptr_type),
        (work_info_size, work_info_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = _info(
        batch_size,
        1,
        max(qlens),
        qlen_granularity,
        total_qlen=sum(qlens),
    )
    work_metadata_ptrs = torch.empty(
        work_metadata_ptrs_size, dtype=work_metadata_ptrs_type, device="cuda"
    )
    work_indptr = torch.zeros(work_indptr_size, dtype=work_indptr_type, device="cuda")
    work_info = torch.zeros(work_info_size, dtype=work_info_type, device="cuda")
    reduce_indptr = torch.zeros(
        reduce_indptr_size, dtype=reduce_indptr_type, device="cuda"
    )
    reduce_final_map = torch.zeros(
        reduce_final_map_size, dtype=reduce_final_map_type, device="cuda"
    )
    reduce_partial_map = torch.zeros(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device="cuda"
    )

    kv_lens = torch.tensor([q + CONTEXT_LEN for q in qlens], dtype=torch.int32)
    qo_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    qo_indptr[1:] = torch.tensor(qlens, dtype=torch.int32).cumsum(0)
    kv_indptr = torch.zeros(batch_size + 1, dtype=torch.int32)
    kv_indptr[1:] = ((kv_lens + BLOCK_SIZE - 1) // BLOCK_SIZE).cumsum(0)

    aiter.get_ps_metadata_v1(
        qo_indptr,
        kv_indptr,
        kv_lens,
        1,
        1,
        work_metadata_ptrs,
        work_indptr,
        work_info,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        qhead_granularity=1,
        qlen_granularity=qlen_granularity,
        kvlen_granularity=KVLEN_GRANULARITY,
        block_size=BLOCK_SIZE,
        is_causal=is_causal,
        need_lse=need_lse,
    )

    partial_rows = reduce_partial_map_size * qlen_granularity
    used_partials = reduce_indptr.cpu()[-1].item()
    assert used_partials <= reduce_partial_map_size
    assert work_indptr.cpu()[-1].item() <= work_info_size[0]
    if used_partials > 0:
        last_row = reduce_partial_map.cpu()[:used_partials].max().item()
        assert last_row + qlen_granularity <= partial_rows


@pytest.mark.parametrize("workload", sorted(WORKLOADS))
@pytest.mark.parametrize("batch_size", BATCH_SIZES)
@pytest.mark.parametrize("qlen_granularity", QLEN_GRANULARITIES)
@pytest.mark.parametrize("is_causal", [True, False])
@pytest.mark.parametrize("need_lse", [True, False])
def test_budget_sized_buffers_hold_the_generated_metadata(
    workload, batch_size, qlen_granularity, is_causal, need_lse
):
    qlens = WORKLOADS[workload](batch_size, qlen_granularity)
    _generate(qlens, qlen_granularity, is_causal, need_lse)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
