# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer as one kernel (MegaMoE TP).

Experts are replicated over the TP group and ``inter_dim`` is sharded. One
launch runs the collectives around GEMM1 + activation + GEMM2 of this rank's
inter slice (the intermediate stays in LDS), dispatched on ``comm_mode``:

* ``"ag_rs"`` (default): tokens enter sequence-parallel; the layer quantizes
  and all-gathers them, sums each token's top-k routes and reduce-scatters::

    y_local = moe(x_local, topk_weights, topk_ids)   # [m, H] -> [m, H]

* ``"ar_ar"``: every rank holds a bf16 partial of all ``M = tp * m`` tokens
  (e.g. a row-parallel projection's output) and the routing of all of them;
  the layer all-reduces the input, and all-reduces the output::

    y = moe(x_partial, topk_weights, topk_ids)       # [M, H] -> [M, H]

Weights are MXFP4 (``shuffle_weight(16, 16)`` + ``e8m0_shuffle`` scales), the
layout the flydsl MoE kernels take. See ``kernels/mega_moe_tp/fused_tp.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter import ActivationType

from .kernels.mega_moe_tp.fused_tp_engine import FusedTpMegaMoe, fused_tp_supported

__all__ = ["COMM_MODES", "MegaMoeTP", "MegaMoeTPConfig", "mega_moe_tp_supported"]

COMM_MODES = ("ag_rs", "ar_ar")


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
    comm_mode: str = "ag_rs"  # "ag_rs" | "ar_ar"


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
        if cfg.comm_mode not in COMM_MODES:
            raise ValueError(f"comm_mode must be one of {COMM_MODES}, got {cfg.comm_mode!r}")
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
            comm_mode=cfg.comm_mode,
            group=group,
            device=device,
        )

    def forward(
        self,
        x_local: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> torch.Tensor:
        """ag_rs: x_local [m, H] bf16 (this rank's tokens), topk_* [m, topk].
        ar_ar: x [M, H] bf16 (this rank's partial of every token), topk_* [M, topk]."""
        return self.engine(x_local, topk_weights, topk_ids)

    __call__ = forward

    def poll_errors(self) -> int:
        """Nonzero if a wait inside the kernel gave up (a peer never arrived)."""
        return self.engine.poll_errors()
