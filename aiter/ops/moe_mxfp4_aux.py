# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# mxfp4_moe -- HIP aux kernels (sort / quant / sort_scales / scatter_reduce) for
# the MXFP4 (a4w4) MoE. The gemm1/gemm2 themselves run on the FlyDSL port
# (aiter/ops/flydsl/mxfp4_gemm{1,2}*.py); only these shared aux ops stay in HIP.
#
# Each function is a thin @compile_ops binding to a C++ host entry in
# csrc/kernels/mxfp4_moe/moe_mxfp4_aux.cu. The codegen'd dispatch switches on
# (NE, TOPK, D_HIDDEN, D_INTER, MB) to pick the right template instantiation.
#
# Shape-parameter glossary (uppercase params on the host-side wrappers):
#   NE       = num routed experts + 1 shared expert (e.g. 385 for Kimi-K2.5)
#   TOPK     = top_k + 1 shared
#   D_HIDDEN = model hidden_size
#   D_INTER  = per-shard MLP intermediate size = moe_intermediate_size / TP
#              (mirrors D_HIDDEN naming; the "INTER" matches aiter's main
#              `inter_dim` convention. Not the expert count.)
#   MB       = block_m (sort/gemm block size, ? {16, 32, 64, 128})

from torch import Tensor

from ..jit.core import compile_ops


# Keep synchronized with moe_aux/codegen/gen_instances.py::SHAPES. The
# replacement-dispatch test compares the two definitions.
MXFP4_MOE_SUPPORTED_SHAPES = frozenset(
    {
        (385, 7168, 512, 9),
        (385, 7168, 1024, 9),
        (257, 7168, 512, 9),
        (257, 7168, 256, 9),
        (384, 7168, 512, 8),
        (385, 7168, 256, 9),
        (32, 7168, 2048, 8),
        (256, 3072, 1536, 8),
        (256, 3072, 768, 8),
        (512, 4096, 256, 10),
        (48, 7168, 3072, 6),
        (384, 7168, 1536, 6),
        (384, 7168, 768, 6),
        (384, 7168, 512, 6),
        (256, 4096, 256, 6),
        (385, 7168, 1536, 7),
        (385, 7168, 768, 7),
        (385, 7168, 512, 7),
        (257, 6144, 2048, 9),
        (257, 6144, 1024, 9),
        (257, 6144, 512, 9),
        (257, 6144, 256, 9),
        (896, 3584, 512, 16),
    }
)


def is_mxfp4_moe_shape_supported(
    expert: int,
    model_dim: int,
    inter_dim: int,
    topk: int,
) -> bool:
    """Return whether the generated MXFP4 auxiliary kernels cover this shape."""
    padded_inter = ((int(inter_dim) + 255) // 256) * 256
    return (int(expert), int(model_dim), padded_inter, int(topk)) in (
        MXFP4_MOE_SUPPORTED_SHAPES
    )


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


@compile_ops("module_moe_mxfp4_aux")
def mxfp4_moe_scatter_reduce_q(
    flat_out_q: Tensor,
    flat_out_scale: Tensor,
    reverse_sorted: Tensor,
    sorted_weights: Tensor,
    out: Tensor,
    NE: int,
    TOPK: int,
    D_HIDDEN: int,
    MB: int,
) -> None: ...
