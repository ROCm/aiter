# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import pytest
import torch

import aiter


def ceil_div(x, y):
    return (x + y - 1) // y


def make_uniform_indptr(batch_size, seqlen, device="cuda"):
    return torch.arange(
        0,
        (batch_size + 1) * seqlen,
        seqlen,
        dtype=torch.int32,
        device=device,
    )


def alloc_metadata_tensors(
    batch_size,
    max_seqlen_qo,
    max_seqlen_kv,
    q_tile_size,
    split_k_size,
    is_causal,
    uni_seqlen_qo,
    device="cuda",
):
    (
        (work_metadata_ptrs_size, work_metadata_ptrs_type),
        (work_indptr_size, work_indptr_type),
        (work_info_set_size, work_info_set_type),
        (reduce_indptr_size, reduce_indptr_type),
        (reduce_final_map_size, reduce_final_map_type),
        (reduce_partial_map_size, reduce_partial_map_type),
    ) = aiter.get_mha_prefill_splitk_metadata_info_v1(
        batch_size=batch_size,
        max_seqlen_qo=max_seqlen_qo,
        max_seqlen_kv=max_seqlen_kv,
        q_tile_size=q_tile_size,
        split_k_size=split_k_size,
        is_causal=is_causal,
        uni_seqlen_qo=uni_seqlen_qo,
    )

    work_metadata_ptrs = torch.empty(
        work_metadata_ptrs_size, dtype=work_metadata_ptrs_type, device=device
    )
    work_indptr = torch.empty(work_indptr_size, dtype=work_indptr_type, device=device)
    work_info_set = torch.empty(
        work_info_set_size, dtype=work_info_set_type, device=device
    )
    reduce_indptr = torch.empty(
        reduce_indptr_size, dtype=reduce_indptr_type, device=device
    )
    reduce_final_map = torch.empty(
        reduce_final_map_size, dtype=reduce_final_map_type, device=device
    )
    reduce_partial_map = torch.empty(
        reduce_partial_map_size, dtype=reduce_partial_map_type, device=device
    )

    return (
        work_metadata_ptrs,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    )


def to_work_list(work_info_set, valid_work_num):
    rows = work_info_set[:valid_work_num].cpu().tolist()
    return [
        {
            "batch_idx": row[0],
            "partial_qo_loc": row[1],
            "qo_start": row[2],
            "qo_end": row[3],
            "kv_start": row[4],
            "kv_end": row[5],
            "kv_offset": row[6],
        }
        for row in rows
    ]


@pytest.mark.parametrize("q_len", [128, 192, 256])
@pytest.mark.parametrize("q_tile_size", [64, 128])
@pytest.mark.parametrize("split_k_size", [64, 128])
def test_mha_prefill_splitk_metadata_noncausal_uniform(
    q_len,
    q_tile_size,
    split_k_size,
):
    device = "cuda"
    batch_size = 2
    kv_len = 384

    qo_indptr = make_uniform_indptr(batch_size, q_len, device=device)
    kv_indptr = make_uniform_indptr(batch_size, kv_len, device=device)

    (
        work_metadata_ptrs,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = alloc_metadata_tensors(
        batch_size=batch_size,
        max_seqlen_qo=q_len,
        max_seqlen_kv=kv_len,
        q_tile_size=q_tile_size,
        split_k_size=split_k_size,
        is_causal=False,
        uni_seqlen_qo=q_len,
        device=device,
    )

    aiter.get_mha_prefill_splitk_metadata_v1(
        qo_indptr,
        kv_indptr,
        False,
        q_tile_size,
        split_k_size,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_qo=q_len,
        uni_seqlen_qo=q_len,
    )

    valid_work_num = int(work_indptr.cpu()[1].item())
    work_list = to_work_list(work_info_set, valid_work_num)
    num_q_tiles = ceil_div(q_len, q_tile_size)
    expected_total_work = batch_size * num_q_tiles * ceil_div(kv_len, split_k_size)

    assert valid_work_num == expected_total_work
    assert int(reduce_indptr.cpu()[batch_size * num_q_tiles].item()) == expected_total_work

    for item in work_list:
        assert 0 <= item["batch_idx"] < batch_size
        assert item["qo_end"] > item["qo_start"]
        assert item["qo_end"] - item["qo_start"] <= q_tile_size
        assert item["kv_end"] > item["kv_start"]
        assert item["kv_end"] - item["kv_start"] <= split_k_size
        assert item["kv_offset"] == 0

    reduce_final_map_cpu = reduce_final_map.cpu()
    for group_idx in range(batch_size * num_q_tiles):
        q_start = int(reduce_final_map_cpu[group_idx, 0].item())
        q_end = int(reduce_final_map_cpu[group_idx, 1].item())
        assert q_end > q_start
        assert q_end - q_start <= q_tile_size


@pytest.mark.parametrize("q_len", [128, 192, 256])
@pytest.mark.parametrize("q_tile_size", [64, 128])
@pytest.mark.parametrize("split_k_size", [64, 128])
def test_mha_prefill_splitk_metadata_causal_uniform(
    q_len,
    q_tile_size,
    split_k_size,
):
    device = "cuda"
    batch_size = 2
    kv_len = q_len

    qo_indptr = make_uniform_indptr(batch_size, q_len, device=device)
    kv_indptr = make_uniform_indptr(batch_size, kv_len, device=device)

    (
        work_metadata_ptrs,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = alloc_metadata_tensors(
        batch_size=batch_size,
        max_seqlen_qo=q_len,
        max_seqlen_kv=kv_len,
        q_tile_size=q_tile_size,
        split_k_size=split_k_size,
        is_causal=True,
        uni_seqlen_qo=q_len,
        device=device,
    )

    aiter.get_mha_prefill_splitk_metadata_v1(
        qo_indptr,
        kv_indptr,
        True,
        q_tile_size,
        split_k_size,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_qo=q_len,
        uni_seqlen_qo=q_len,
    )

    valid_work_num = int(work_indptr.cpu()[1].item())
    work_list = to_work_list(work_info_set, valid_work_num)

    expected_per_batch = 0
    for q_tile_idx in range(ceil_div(q_len, q_tile_size)):
        visible_kv_len = min((q_tile_idx + 1) * q_tile_size, q_len)
        expected_per_batch += ceil_div(visible_kv_len, split_k_size)
    assert valid_work_num == batch_size * expected_per_batch

    for batch_idx in range(batch_size):
        batch_q_begin = int(qo_indptr[batch_idx].item())
        batch_kv_begin = int(kv_indptr[batch_idx].item())
        batch_items = [item for item in work_list if item["batch_idx"] == batch_idx]
        for qo_start in sorted({item["qo_start"] for item in batch_items}):
            tile_items = [item for item in batch_items if item["qo_start"] == qo_start]
            qo_end = tile_items[0]["qo_end"]
            allowed_kv_end = batch_kv_begin + (qo_end - batch_q_begin)
            assert tile_items[0]["kv_start"] == batch_kv_begin
            assert max(item["kv_end"] for item in tile_items) == allowed_kv_end


def test_mha_prefill_splitk_metadata_variable_q_len_smoke():
    device = "cuda"

    qo_indptr = torch.tensor([0, 128, 320, 576], dtype=torch.int32, device=device)
    kv_indptr = torch.tensor([0, 256, 448, 832], dtype=torch.int32, device=device)

    (
        work_metadata_ptrs,
        work_indptr,
        work_info_set,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
    ) = alloc_metadata_tensors(
        batch_size=3,
        max_seqlen_qo=256,
        max_seqlen_kv=384,
        q_tile_size=64,
        split_k_size=128,
        is_causal=False,
        uni_seqlen_qo=-1,
        device=device,
    )

    aiter.get_mha_prefill_splitk_metadata_v1(
        qo_indptr,
        kv_indptr,
        False,
        64,
        128,
        work_metadata_ptrs,
        work_info_set,
        work_indptr,
        reduce_indptr,
        reduce_final_map,
        reduce_partial_map,
        max_seqlen_qo=256,
        uni_seqlen_qo=-1,
    )

    valid_work_num = int(work_indptr.cpu()[1].item())
    work_list = to_work_list(work_info_set, valid_work_num)

    assert valid_work_num > 0
    for item in work_list:
        assert item["qo_end"] > item["qo_start"]
        assert item["kv_end"] > item["kv_start"]
