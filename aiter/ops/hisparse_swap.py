# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""HiSparse swap-in ops for GLM-5.2 DSA decode.

Keeps the full KV in a pinned host cold pool and a fixed-size GPU hot buffer per
request. Each decode step, per layer, the fused kernel miss-detects the indexer
top-k against the resident hot set, evicts the least-recently-used slots for the
misses, gathers the missing tokens from the cold pool over PCIe/XGMI, and
translates the top-k into hot-buffer rows. All bookkeeping is on GPU (no host
sync) and the launch shape is fixed, so the path is CUDAGraph-capturable.

On an xnack- agent a GPU kernel faults on a raw host VA, so the cold-pool pointer
passed to the kernels must be the device-mapped pointer from
:func:`hisparse_host_get_device_pointer`; cache it once per cold-pool tensor.

See ``aiter/csrc/include/hisparse_swap.h`` for the C++ API.
"""

import torch

from ..jit.core import compile_ops

MD_NAME = "module_hisparse_swap"


@compile_ops("module_hisparse_swap")
def hisparse_host_get_device_pointer(pinned_host_tensor: torch.Tensor) -> int: ...


@compile_ops("module_hisparse_swap")
def hisparse_swap_in(
    cold_pool_dev_ptr: int,
    hot_buffer: torch.Tensor,
    src_locs: torch.Tensor,
    dst_locs: torch.Tensor,
    item_size_bytes: int,
) -> None: ...


@compile_ops("module_hisparse_swap")
def hisparse_swap_and_translate(
    cold_pool_dev_ptr: int,
    hot_buffer: torch.Tensor,
    topk_logical: torch.Tensor,
    indptr: torch.Tensor,
    req_slots: torch.Tensor,
    slot_token: torch.Tensor,
    last_used: torch.Tensor,
    token_to_slot: torch.Tensor,
    recency: torch.Tensor,
    out_translated: torch.Tensor,
    item_size_bytes: int,
    hot_slots: int,
    cold_depth: int,
    topk: int,
) -> None: ...


@compile_ops("module_hisparse_swap")
def hisparse_swap_and_translate_record(
    cold_pool_dev_ptr: int,
    hot_buffer: torch.Tensor,
    topk_logical: torch.Tensor,
    indptr: torch.Tensor,
    req_slots: torch.Tensor,
    slot_token: torch.Tensor,
    last_used: torch.Tensor,
    token_to_slot: torch.Tensor,
    recency: torch.Tensor,
    out_translated: torch.Tensor,
    plan_miss_tok: torch.Tensor,
    plan_miss_slot: torch.Tensor,
    plan_miss_count: torch.Tensor,
    item_size_bytes: int,
    hot_slots: int,
    cold_depth: int,
    topk: int,
) -> None: ...


@compile_ops("module_hisparse_swap")
def hisparse_copy_planned(
    cold_pool_dev_ptr: int,
    hot_buffer: torch.Tensor,
    req_slots: torch.Tensor,
    plan_miss_tok: torch.Tensor,
    plan_miss_slot: torch.Tensor,
    plan_miss_count: torch.Tensor,
    item_size_bytes: int,
    hot_slots: int,
    cold_depth: int,
    topk: int,
) -> None: ...


@compile_ops("module_hisparse_swap")
def hisparse_backup_into_assigned(
    cold_pool_dev_ptr: int,
    hot_buffer: torch.Tensor,
    layer_kv: torch.Tensor,
    src_slots: torch.Tensor,
    req_slots: torch.Tensor,
    logical_pos: torch.Tensor,
    token_to_slot: torch.Tensor,
    item_size_bytes: int,
    hot_slots: int,
    cold_depth: int,
) -> None: ...


@compile_ops("module_hisparse_swap")
def hisparse_backup_new_token(
    cold_pool_dev_ptr: int,
    hot_buffer: torch.Tensor,
    layer_kv: torch.Tensor,
    src_slots: torch.Tensor,
    req_slots: torch.Tensor,
    logical_pos: torch.Tensor,
    slot_token: torch.Tensor,
    last_used: torch.Tensor,
    token_to_slot: torch.Tensor,
    recency: torch.Tensor,
    item_size_bytes: int,
    hot_slots: int,
    cold_depth: int,
) -> None: ...


__all__ = [
    "hisparse_backup_into_assigned",
    "hisparse_backup_new_token",
    "hisparse_copy_planned",
    "hisparse_host_get_device_pointer",
    "hisparse_swap_and_translate",
    "hisparse_swap_and_translate_record",
    "hisparse_swap_in",
]
