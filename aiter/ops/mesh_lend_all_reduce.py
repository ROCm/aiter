# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.jit.core import compile_ops

MD_NAME = "module_mesh_lend_all_reduce"


@compile_ops(MD_NAME, develop=True)
def mesh_lend_create(
    device_id: int, rank: int, world_size: int, max_bytes: int, max_lenders: int
) -> int: ...


@compile_ops(MD_NAME, develop=True)
def mesh_lend_export_size() -> int: ...


@compile_ops(MD_NAME, develop=True)
def mesh_lend_export(fa: int, out_ptr: int) -> None: ...


@compile_ops(MD_NAME, develop=True)
def mesh_lend_connect(fa: int, export_ptrs: list[int]) -> None: ...


@compile_ops(MD_NAME, develop=True)
def mesh_lend_all_reduce(
    fa: int, inp: torch.Tensor, out: torch.Tensor, bits: int = 0
) -> None: ...


@compile_ops(MD_NAME, develop=True)
def mesh_lend_num_lenders(fa: int) -> int: ...


@compile_ops(MD_NAME, develop=True)
def mesh_lend_destroy(fa: int) -> None: ...
