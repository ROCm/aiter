# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer as one kernel (MegaMoE TP).

Experts are replicated over the TP group and ``inter_dim`` is sharded. One
launch runs the collectives around GEMM1 + activation + GEMM2 of this rank's
inter slice (the intermediate stays in LDS), dispatched on ``comm_mode``:

* ``"ag_rs"`` (default): tokens enter sequence-parallel (each rank its own
  ``m`` tokens and their routing); the layer all-gathers them, sums each
  token's top-k routes and reduce-scatters back to the owner::

    y_local = moe(x_local, topk_weights, topk_ids)   # [m, H] -> [m, H]

* ``"rs"``: every rank holds the same ``M = tp * m`` tokens and routing (all
  gathered outside); the output is reduce-scattered (rank r gets rows
  ``r*m : (r+1)*m``)::

    y_local = moe(x, topk_weights, topk_ids)[r*m:(r+1)*m]   # [M, H] -> [m, H]

* ``"ar"``: the standard TP MoE layer: every rank holds the same full hidden
  state ``x`` and routing of all ``M`` tokens; only the output is
  all-reduced (identical on every rank)::

    y = moe(x, topk_weights, topk_ids)               # [M, H] -> [M, H]

* ``"ar_ar"``: as ``"ar"``, but each rank holds a bf16 *partial* of ``x``
  (e.g. the output of a row-parallel projection before its all-reduce); the
  layer all-reduces the input too. Routing must be the same on every rank.

Weights are MXFP4 (``shuffle_weight(16, 16)`` + ``e8m0_shuffle`` scales), the
layout the flydsl MoE kernels take. See ``kernels/mega_moe_tp/mega_moe_tp_kernel.py``.

Contract (the kernel synchronizes the ranks through flags in each other's memory):

* ``forward`` is a collective: every rank of the group calls it the same number
  of times, in the same order, with the same number of local tokens ``m``
  (ag_rs: pad to a common ``m``); in ``rs`` / ``ar`` / ``ar_ar`` every rank
  passes the same routing (and, ``rs`` / ``ar``, the same ``x``).
  ``AITER_MEGAMOE_TP_CHECK=1`` verifies both per call (one collective). ``m == 0`` returns an empty tensor. A rank
  that falls out of step makes the waits time out (2 s each); ``poll_errors`` /
  ``check_errors`` (or ``AITER_MEGAMOE_TP_CHECK=1``: check after every eager
  forward) report it, and ``reset()`` (collective) restarts the layer.
* ``topk_ids`` outside ``[0, experts)`` are masked (the route adds nothing);
  a token must not repeat an expert.
* The result is a view of an internal buffer, valid until the next forward of
  this layer (ar / ar_ar: peers write into it during that forward); pass
  ``out=`` to get it in a buffer of your own (ag_rs / rs: written in place).
* The kernel keeps every CU busy and spins on peers: do not overlap it with
  another persistent / communication kernel (e.g. another layer's, a custom
  all-reduce on a side stream).
* The first forward with a new launch config compiles and synchronizes the
  ranks; ``prepare(local_token_counts)`` does that ahead of CUDA graph capture.
* ``comm_dtype="fp8"`` (default): the reduce-scatter / all-reduce partials and
  the per-route GEMM2 rows are MXFP8 (E4M3 + E8M0 per 32): rel L2 ~0.038
  against a bf16-math torch reference (glm5 / m3), vs ~0.003 with ``"bf16"``
  (bf16 partials and route rows, no LL packets: ~3-14% slower, more at large M).
* Launch epochs are int32: ``reset()`` at least every 2**31 forwards per layer.
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


class MegaMoeTP:
    """Per layer: scratch of about ``max_local_tokens * world_size * topk *
    model_dim * 2`` bytes plus the symmetric arena."""

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
            group=group,
            device=device,
        )

    def forward(
        self,
        x_local: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """ag_rs: x_local [m, H] bf16 (this rank's tokens), topk_* [m, topk].
        rs / ar: x [M, H] bf16 (all tokens, same on every rank), topk_* [M, topk].
        ar_ar: x [M, H] bf16 (this rank's partial of every token), topk_* [M, topk]
        (the same routing on every rank). topk_ids int32, topk_weights float32
        (others are converted per call)."""
        return self.engine(x_local, topk_weights, topk_ids, out)

    __call__ = forward

    def prepare(self, local_tokens) -> None:
        """Collective: compile + arm the launch configs of these local token counts."""
        self.engine.prepare(local_tokens)

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
