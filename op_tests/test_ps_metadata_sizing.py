# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Buffer-sizing invariants of ``aiter.get_ps_metadata_info_v1``.

``get_ps_metadata_info_v1`` is pure Python, but it has to bound what the C++
generator ``get_ps_metadata_v1`` (csrc/kernels/mla/metadata/v1_2_host.cuh)
actually emits. The bound assuming every batch carries ``max_qlen`` query
tokens is far looser than a serving engine ever produces, so the function takes
an optional ``total_qlen`` token budget; these tests pin the loose default, the
safety of the budget-aware bound against the real generator, and the size of
the partial pool (``max_partials * qlen_granularity`` rows) that a serving
engine has to reserve.

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


# Configs and budgets for regression testing to avoid metadata buffers not growing 
# unexpectedly. max_gib is current sizes rounded up. 
SERVING_CONFIGS = {
    "dsv3_tp8_8k_budget": dict(
        batch_size=256, total_qlen=8192, num_head_k=16, v_head_dim=512, max_gib=3.25
    ),
    "dsv3_tp8_16k_budget": dict(
        batch_size=256, total_qlen=16384, num_head_k=16, v_head_dim=512, max_gib=3.25
    ),
    "dsv3_tp8_16k_budget_1k_seqs": dict(
        batch_size=1024, total_qlen=16384, num_head_k=16, v_head_dim=512, max_gib=9.5
    ),
    "dsv3_tp4_16k_budget": dict(
        batch_size=128, total_qlen=16384, num_head_k=32, v_head_dim=512, max_gib=4.25
    ),
    "kimi_k25_tp8_8k_budget": dict(
        batch_size=256, total_qlen=8192, num_head_k=16, v_head_dim=128, max_gib=0.85
    ),
    "kimi_k25_tp8_16k_budget": dict(
        batch_size=256, total_qlen=16384, num_head_k=16, v_head_dim=128, max_gib=0.85
    ),
    "kimi_k25_tp8_16k_budget_1k_seqs": dict(
        batch_size=1024, total_qlen=16384, num_head_k=16, v_head_dim=128, max_gib=2.5
    ),
}
SERVING_QLEN_GRANULARITY = 256


def _info(batch_size, num_head_k, max_qlen, qlen_granularity, total_qlen=None):
    return aiter.get_ps_metadata_info_v1(
        batch_size=batch_size,
        num_head_k=num_head_k,
        max_qlen=max_qlen,
        qlen_granularity=qlen_granularity,
        total_qlen=total_qlen,
    )


def _rows(info_entry):
    shape = info_entry[0]
    return shape[0] if isinstance(shape, tuple) else shape


def _cu_num():
    return torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count


def _partial_pool_rows(info, qlen_granularity):
    return _rows(info[5]) * qlen_granularity


def _partial_pool_bytes(cfg, total_qlen):
    """Bytes vLLM reserves for the fp32 partial logits and their lse."""
    rows = _partial_pool_rows(
        _info(
            cfg["batch_size"],
            cfg["num_head_k"],
            cfg["total_qlen"],
            SERVING_QLEN_GRANULARITY,
            total_qlen=total_qlen,
        ),
        SERVING_QLEN_GRANULARITY,
    )
    itemsize = torch.finfo(torch.float32).bits // 8
    return rows * cfg["num_head_k"] * (cfg["v_head_dim"] + 1) * itemsize


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
        assert _rows(got) <= _rows(ref)
    # a single request is still allowed to be max_qlen long under any budget
    assert _rows(bounded[4]) >= math.ceil(max_qlen / qlen_granularity)


@pytest.mark.parametrize("batch_size", BATCH_SIZES + [256])
@pytest.mark.parametrize("qlen_granularity", QLEN_GRANULARITIES)
@pytest.mark.parametrize("total_qlen", [512, 8192, 16384])
def test_partial_pool_grows_with_the_budget_not_with_max_qlen(
    batch_size, qlen_granularity, total_qlen
):
    cu_num = _cu_num()
    for max_qlen in (qlen_granularity, total_qlen, 163840):
        rows = _partial_pool_rows(
            _info(batch_size, 1, max_qlen, qlen_granularity, total_qlen=total_qlen),
            qlen_granularity,
        )
        # linear in the budget (or in one max-length request, when the caller
        # declares a max_qlen above its own budget), never in their product
        assert rows <= max(total_qlen, max_qlen) + qlen_granularity * (
            batch_size + cu_num
        )


@pytest.mark.parametrize("config", sorted(SERVING_CONFIGS))
def test_serving_config_partial_pool_fits_in_the_memory_budget(config):
    """The sizing that OOM'd vLLM at startup (ROCm/aiter#5729) must stay bounded.

    Reserving ``max_partials * qlen_granularity`` rows of fp32 logits plus lse
    asked for 64 GiB at an 8k token budget and 128 GiB at 16k (513 GiB at 1k
    concurrent seqs); the budget-aware bound brings those to 2.4, 2.6 and
    8.7 GiB on DSv3 at TP8.
    """
    cfg = SERVING_CONFIGS[config]
    budget = cfg["max_gib"] * 1024**3
    bounded = _partial_pool_bytes(cfg, total_qlen=cfg["total_qlen"])
    assert bounded <= budget, (
        f"{config}: partial pool is {bounded / 1024 ** 3:.2f} GiB, over the "
        f"{cfg['max_gib']:.2f} GiB this config is expected to cost"
    )
    # without a budget the same config is the OOM that motivated total_qlen
    assert _partial_pool_bytes(cfg, total_qlen=None) > budget


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
