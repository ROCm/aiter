# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer as one kernel (MegaMoE TP), gfx950.

Experts are replicated over the TP group and ``inter_dim`` is sharded; one launch
runs the collectives around GEMM1 + activation + GEMM2 of this rank's inter slice.
``comm_mode``: ``"ag_rs"`` (sequence-parallel in/out), ``"rs"`` (replicated in,
reduce-scattered out), ``"ar"`` (replicated in, all-reduced out), ``"ar_ar"``
(partial in, all-reduced out). Weights are MXFP4 (``shuffle_weight(16, 16)`` +
``e8m0_shuffle``). ``forward`` is a collective; ``prepare`` before graph capture.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter import ActivationType

from .kernels.mega_moe_tp.mega_moe_tp import COMM_MODES, MegaMoeTPEngine

__all__ = ["COMM_MODES", "MegaMoeTP", "MegaMoeTPConfig", "mega_moe_tp_supported"]

_ACTS = (ActivationType.Silu, ActivationType.Swiglu, ActivationType.Situv2)


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
    inter_dim: int
    experts: int
    topk: int
    max_local_tokens: int
    activation: ActivationType = ActivationType.Silu
    beta: float | None = None
    linear_beta: float | None = None
    swiglu_limit: float | None = None
    comm_mode: str = "ag_rs"
    comm_dtype: str = "fp8"
    ar_gather: str = "auto"
    schedule: str = "tuned"
    act_dtype: str = "fp4"


class MegaMoeTP:
    """One fused TP MoE layer (see the module docstring)."""

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
        if cfg.activation not in _ACTS:
            raise ValueError(f"MegaMoeTP: unsupported activation {cfg.activation}")
        if not mega_moe_tp_supported():
            raise ValueError("MegaMoeTP: needs gfx950")
        act = cfg.activation.name.lower()
        if cfg.comm_mode not in COMM_MODES:
            raise ValueError(f"MegaMoeTP: unknown comm_mode {cfg.comm_mode!r}")
        situ = act == "situv2"
        self.cfg = cfg
        self.engine = MegaMoeTPEngine(
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
            activation=act,
            situ_beta=cfg.beta if situ and cfg.beta is not None else 1.0,
            situ_linear_beta=(
                cfg.linear_beta if situ and cfg.linear_beta is not None else 1.0
            ),
            swiglu_limit=cfg.swiglu_limit,
            comm_mode=cfg.comm_mode,
            comm_dtype=cfg.comm_dtype,
            ar_gather=cfg.ar_gather,
            schedule=cfg.schedule,
            act_dtype=cfg.act_dtype,
            group=group,
            device=device,
        )

    def forward(
        self,
        x_local: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor | None = None,
        tail=None,
        bf16: bool = False,
    ):
        """ag_rs: x [m, H] (own tokens); rs / ar: x [M, H] replicated; ar_ar: x [M, H] partial."""
        return self.engine(x_local, topk_weights, topk_ids, out, tail=tail, bf16=bf16)

    __call__ = forward

    def set_weights(
        self,
        *,
        w1: torch.Tensor,
        w1_scale: torch.Tensor,
        w2: torch.Tensor,
        w2_scale: torch.Tensor,
    ) -> None:
        """Run the next forwards on another layer's weights of the same shape."""
        self.engine.set_weights(w1, w1_scale, w2, w2_scale)

    def prepare(
        self, local_tokens, tail: bool = False, tail_bf16: bool = False
    ) -> None:
        """Collective: compile and arm the launch configs of these local token counts."""
        self.engine.prepare(local_tokens, tail, tail_bf16)

    def poll_errors(self) -> int:
        """Nonzero if a wait inside the kernel gave up (a peer never arrived)."""
        return self.engine.poll_errors()

    def check_errors(self) -> None:
        """Raise RuntimeError (and clear) if a wait gave up since the last check."""
        self.engine.check_errors()

    def clear_errors(self) -> None:
        self.engine.clear_errors()

    def reset(self) -> None:
        """Collective: restart the cross-rank state (after a timeout / desync)."""
        self.engine.reset()
