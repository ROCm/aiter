# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer as one kernel (MegaMoE TP).

Experts are replicated over the TP group and ``inter_dim`` is sharded; tokens
enter sequence-parallel. One launch quantizes and all-gathers the tokens,
runs GEMM1 + activation + GEMM2 for this rank's inter slice (the intermediate
stays in LDS), sums each token's top-k routes and reduce-scatters the result::

    moe = MegaMoeTP(MegaMoeTPConfig(...), w1=..., w1_scale=..., w2=..., w2_scale=...)
    y_local = moe(x_local, topk_weights, topk_ids)     # [m, model_dim] bf16

Weights are MXFP4 (``shuffle_weight(16, 16)`` + ``e8m0_shuffle`` scales), the
layout the flydsl MoE kernels take. See ``kernels/mega_moe_tp/fused_tp.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter import ActivationType

from .kernels.mega_moe_tp.fused_tp_engine import FusedTpMegaMoe, fused_tp_supported

__all__ = ["MegaMoeTP", "MegaMoeTPConfig", "mega_moe_tp_supported"]


def mega_moe_tp_supported(gfx: str | None = None) -> bool:
    """Whether this device has the kernel (gfx950)."""
    if gfx is None:
        from aiter.jit.utils.chip_info import get_gfx

        gfx = get_gfx()
    return gfx == "gfx950"


@dataclass(frozen=True)
class MegaMoeTPConfig:
    rank: int
    world_size: int
    model_dim: int
    inter_dim: int  # this rank's shard
    experts: int
    topk: int
    max_local_tokens: int
    activation: ActivationType = ActivationType.Silu
    beta: float | None = None  # Situv2 only
    linear_beta: float | None = None  # Situv2 only


class MegaMoeTP:
    """AllGather + GEMM1 + act + GEMM2 + ReduceScatter for one TP rank."""

    def __init__(
        self,
        cfg: MegaMoeTPConfig,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
        group=None,
        device: torch.device | None = None,
    ):
        if not fused_tp_supported(cfg.model_dim, cfg.inter_dim, cfg.world_size):
            raise ValueError(
                f"MegaMoeTP does not tile model_dim={cfg.model_dim} "
                f"inter_dim={cfg.inter_dim} tp={cfg.world_size}"
            )
        situ = cfg.activation == ActivationType.Situv2
        self.cfg = cfg
        self.engine = FusedTpMegaMoe(
            rank=cfg.rank,
            world_size=cfg.world_size,
            model_dim=cfg.model_dim,
            inter_dim=cfg.inter_dim,
            experts=cfg.experts,
            topk=cfg.topk,
            max_local_tokens=cfg.max_local_tokens,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            activation="situv2" if situ else "silu",
            situ_beta=cfg.beta if situ and cfg.beta is not None else 1.0,
            situ_linear_beta=(
                cfg.linear_beta if situ and cfg.linear_beta is not None else 1.0
            ),
            group=group,
            device=device,
        )

    def forward(
        self,
        x_local: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """x_local [m, model_dim] bf16 (this rank's tokens), topk_* [m, topk]."""
        return self.engine(x_local, topk_weights, topk_ids)

    __call__ = forward
