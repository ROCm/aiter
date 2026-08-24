# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE Stage1.

Each TP rank owns ALL experts and one 1/tp shard of the intermediate
dimension. The caller passes its own DP token shard; this operator
all-gathers across the TP group (DP group == TP group), runs grouping,
GEMM1, SwiGLU and per-1x32 FP8 output quantization, and returns the
six tensors the ordinary FlyDSL v2 FMoE GEMM2 consumes.
"""

from dataclasses import dataclass

import torch

from aiter.ops.flydsl.moe_kernels import get_flydsl_kernel_params

_SUPPORTED_TP = (4, 8)
_DEFAULT_STAGE1_KERNEL = "flydsl_moe1_afp8_wfp4_bf16_t32x64x256_w4_gui_xcd4_kw4_fp8"


@dataclass(frozen=True)
class TPMoEStage1Output:
    """Everything ordinary FlyDSL v2 FMoE GEMM2 needs, plus host metadata."""

    inter_sorted_quant: torch.Tensor
    inter_sorted_shuffled_scale: torch.Tensor
    sorted_token_ids: torch.Tensor
    sorted_weights: torch.Tensor
    sorted_expert_ids: torch.Tensor
    num_valid_ids: torch.Tensor

    m_logical: int
    max_sorted: int
    num_experts: int
    model_dim: int
    inter_dim: int
    topk: int
    sort_block_m: int


class TPMoEStage1:
    """Stateful TP4/TP8 MoE Stage1 operator.

    Preconditions (documented, not checked at runtime):
      * every rank in the group calls with the same ``m_local``
      * ``w1`` / ``w1_scale`` are already preshuffled for the a16w4
        gate/up-interleaved layout
      * the group used here is both the DP group and the TP group
    """

    def __init__(
        self,
        *,
        model_dim: int,
        inter_dim: int,
        experts: int,
        topk: int,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        group=None,
        tp_size: int | None = None,
        tp_rank: int | None = None,
        device: torch.device | None = None,
        sort_block_m: int = 32,
        swiglu_limit: float = 0.0,
        stage1_kernel_name: str = _DEFAULT_STAGE1_KERNEL,
        transport: str = "allgather_bf16",
    ):
        self.group = group
        if tp_size is None or tp_rank is None:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise ValueError(
                    "TPMoEStage1 needs an initialized process group, or explicit "
                    "tp_size/tp_rank"
                )
            tp_size = dist.get_world_size(group)
            tp_rank = dist.get_rank(group)
        if int(tp_size) not in _SUPPORTED_TP:
            raise ValueError(f"tp_size={tp_size} unsupported; expected one of {_SUPPORTED_TP}")

        params = get_flydsl_kernel_params(stage1_kernel_name)
        if params is None:
            raise ValueError(f"unknown stage1 kernel name: {stage1_kernel_name}")
        if int(sort_block_m) != int(params["tile_m"]):
            raise ValueError(
                f"sort_block_m={sort_block_m} must equal the stage1 kernel tile_m="
                f"{params['tile_m']} ({stage1_kernel_name})"
            )
        if inter_dim % int(params["tile_n"]) != 0:
            raise ValueError(
                f"inter_dim={inter_dim} must be divisible by tile_n={params['tile_n']}"
            )
        if float(swiglu_limit) < 0:
            raise ValueError("swiglu_limit must be non-negative")
        if transport != "allgather_bf16":
            raise NotImplementedError(
                f"transport={transport!r} is not implemented yet; phase 1 only "
                "supports 'allgather_bf16'"
            )

        self.tp_size = int(tp_size)
        self.tp_rank = int(tp_rank)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.topk = int(topk)
        self.sort_block_m = int(sort_block_m)
        self.swiglu_limit = float(swiglu_limit)
        self.stage1_kernel_name = stage1_kernel_name
        self.stage1_params = params
        self.transport = transport
        self.device = device or torch.device("cuda", torch.cuda.current_device())
        self.w1 = w1
        self.w1_scale = w1_scale

    def m_logical(self, m_local: int) -> int:
        return self.tp_size * int(m_local)

    def max_sorted(self, m_local: int) -> int:
        """Mirror of moe_sorting's max_num_tokens_padded."""
        return self.m_logical(m_local) * self.topk + self.experts * self.sort_block_m - self.topk
