# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor, Generator
from ..jit.core import compile_ops
from typing import Optional, List


@compile_ops("module_trtllm_all_reduce_fusion")
def init_trtllm_ar_fusion(
    rank: int,
    world_size: int,
    max_size_in_bytes: int,
) -> int: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def destroy_trtllm_ar_fusion(
    fptr_t: int,
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def get_trtllm_ar_fusion_handle(
    fptr_t: int,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def open_trtllm_ar_fusion_handles(
    fptr_t: int,
    handles: list[Tensor],
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def get_trtllm_ar_fusion_workspace(
    fptr_t: int,
    ref: Tensor,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_allreduce_rms(
    rank: int,
    nranks: int,
    allreduce_in: Tensor,
    residual_in: Tensor,
    rms_gamma: Tensor,
    residual_out: Tensor,
    norm_out: Tensor,
    eps: float,
    workspace: Tensor
) -> None: ...
