# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL bridge for the Kimi-K3 latent FHMoE contract."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentFHMoELayout:
    """Compile-time geometry shared by both integrated stage launches."""

    routed_model_dim: int = 3584
    routed_inter_dim: int = 384
    shared_model_dim: int = 7168
    shared_inter_dim: int = 768

    @property
    def max_model_dim(self) -> int:
        return max(self.routed_model_dim, self.shared_model_dim)

    @property
    def max_inter_dim(self) -> int:
        return max(self.routed_inter_dim, self.shared_inter_dim)


def integrated_route_domain(
    topk_ids: torch.Tensor,
    topk_weight: torch.Tensor,
    routed_experts: int,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Append one shared task per token to the routed sorting domain.

    The returned synthetic expert id is used only as the format/task selector.
    Both stage kernels must consume the resulting single sorted domain.
    """
    m = topk_ids.shape[0]
    shared_ids = torch.full(
        (m, 1), routed_experts, dtype=topk_ids.dtype, device=topk_ids.device
    )
    shared_weights = torch.ones(
        (m, 1), dtype=topk_weight.dtype, device=topk_weight.device
    )
    return (
        torch.cat((topk_ids, shared_ids), dim=1),
        torch.cat((topk_weight, shared_weights), dim=1),
        routed_experts,
    )


def run_latent_fhmoe(
    routed_input: torch.Tensor,
    routed_w1: torch.Tensor,
    routed_w2: torch.Tensor,
    routed_w1_scale: torch.Tensor,
    routed_w2_scale: torch.Tensor,
    topk_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    shared_input: torch.Tensor,
    shared_w1: torch.Tensor,
    shared_w2: torch.Tensor,
    *,
    beta: float,
    linear_beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Launch the two-stage integrated kernel.

    The public contract and scheduling-domain construction are complete. The
    launch remains disabled until the common kernel has a BF16 shared MFMA
    operand loader. Falling back to standalone shared GEMMs here would violate
    the required one-launch-per-stage scheduling contract.
    """
    del (
        routed_input,
        routed_w1,
        routed_w2,
        routed_w1_scale,
        routed_w2_scale,
        topk_weight,
        topk_ids,
        shared_input,
        shared_w1,
        shared_w2,
        beta,
        linear_beta,
    )
    raise NotImplementedError(
        "K3 latent FHMoE launch is blocked on the common FlyDSL kernel's BF16 "
        "shared MFMA loader and separate shared stage2 output epilogue; no "
        "FP8 or padded homogeneous fallback is used"
    )
