# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# mxfp4_moe — pure-HIP MoE backend for MXFP4 (a4w4) weights.
#
# Each function is a thin @compile_ops binding to a C++ host entry in
# csrc/py_itfs_cu/mxfp4_moe_{aux,gemm}.cu. The C++ side internally switches
# on (NE, TOPK, D_HIDDEN, D_INTER, MB) to pick the right template
# instantiation. Adding a new shape: edit the switch-case in C++ + add a
# row to aiter/configs/tuned_fmoe.csv.
#
# Shape-parameter glossary (uppercase params on the host-side wrappers):
#   NE       = num routed experts + 1 shared expert (e.g. 385 for Kimi-K2.5)
#   TOPK     = top_k + 1 shared
#   D_HIDDEN = model hidden_size
#   D_INTER  = per-shard MLP intermediate size = moe_intermediate_size / TP
#              (mirrors D_HIDDEN naming; the "INTER" matches aiter's main
#              `inter_dim` convention. Not the expert count.)
#   MB       = block_m (sort/gemm block size, ∈ {16, 32, 64, 128})

import torch
from torch import Tensor

from ..jit.core import compile_ops


@compile_ops("module_moe_mxfp4_aux")
def mxfp4_moe_sort_quant(
    a_input: Tensor,
    topk_ids: Tensor,
    topk_weight: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    cumsum_tensor: Tensor,
    reverse_sorted: Tensor,
    sorted_weights: Tensor,
    a_quant: Tensor,
    a_scale: Tensor,
    masked_m: Tensor,
    m_indices: Tensor,
    bf16_zero_out: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux")
def mxfp4_moe_sort(
    topk_ids: Tensor,
    topk_weight: Tensor,
    sorted_token_ids: Tensor,
    sorted_expert_ids: Tensor,
    cumsum_tensor: Tensor,
    reverse_sorted: Tensor,
    sorted_weights: Tensor,
    masked_m: Tensor,
    m_indices: Tensor,
    bf16_zero_out: Tensor,
    bf16_zero_workspace: Tensor,
    M_logical: int,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    D_INTER: int,
    MB: int,
    prologue: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux")
def mxfp4_moe_quant(
    a_input: Tensor,
    a_quant: Tensor,
    a_scale: Tensor,
    bf16_zero_out: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux")
def mxfp4_moe_sort_scales(
    a_scale: Tensor,
    sorted_token_ids: Tensor,
    cumsum_tensor: Tensor,
    a_scale_sorted_shuffled: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    D_INTER: int,
    MB: int,
    max_sorted: int,
) -> None: ...


@compile_ops("module_moe_mxfp4_aux")
def mxfp4_moe_scatter_reduce(
    flat_out: Tensor,
    reverse_sorted: Tensor,
    sorted_weights: Tensor,
    out: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
) -> None: ...
