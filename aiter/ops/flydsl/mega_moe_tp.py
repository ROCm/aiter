# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Tensor-parallel MoE layer as one kernel (MegaMoE TP), gfx950.

Experts are replicated over the TP group and ``inter_dim`` is sharded; one launch
runs the collectives around GEMM1 + activation + GEMM2 of this rank's inter slice.
``comm_mode``: ``"ag_rs"`` (sequence-parallel in/out), ``"rs"`` (replicated in,
reduce-scattered out), ``"ar"`` (replicated in, all-reduced out), ``"ar_ar"``
(partial in, all-reduced out). Weights are MXFP4 (``shuffle_weight(16, 16)`` +
``e8m0_shuffle``).

Contract:

* ``forward`` is a collective: every rank calls it the same number of times, in the
  same order, with the same local token count (and, rs / ar / ar_ar, the same routing).
  Layers of one shape may share an instance (``set_weights``): their forwards form one
  collective sequence.
* ``prepare(local_tokens)`` before CUDA graph capture: it compiles every launch config,
  runs each once and checks every rank's watchdog (reset + retry on a timeout).
* The kernel is persistent (one CTA per CU, CTAs and ranks spin on each other): it
  needs the whole GPU. Do not run it concurrently with other kernels (other streams,
  other processes, CU-masked / partitioned GPUs).
* A wait that times out (2 s) sets a sticky bit and the launch's output is invalid.
  ``error_flag()`` is that bit as a device tensor (copy it asynchronously, e.g. every N
  steps); ``check_errors()`` synchronizes and raises; ``reset()`` (collective) restarts
  the cross-rank state, which a timed-out launch leaves inconsistent.
* The returned tensors (``out=None``, ar outputs, the fused tail's rows) are views of
  internal buffers, valid until the next forward of this instance; pass ``out=`` to keep
  the output.
* Launch epochs are int32, compared wrap-safe; a flag slot unused for 2**31 launches can
  read stale: ``reset()`` at least that often for very long-lived instances.
* The symmetric memory uses hipIpc handles; in containers on the host network ROCm 7.1
  needs ``HSA_ENABLE_IPC_MODE_LEGACY=1``.

Large-batch (LB) schedule (``schedule="dynamic"``): from ``lb_min`` global tokens up; its
row tile is 16 * ``lb_mt`` rows (``"T1:N1,...,N"``: up to Ti tokens Ni), 16 * ``lb_mt_small``
up to ``lb_small_max`` tokens; ``lb_npp`` GEMM1 column blocks per A pass; ``lb_q`` output
column chunks per GEMM2 unit. Unset fields fall back to ``AITER_MEGAMOE_TP_LB_*``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter import ActivationType

from .kernels.mega_moe_tp.mega_moe_tp import (
    COMM_MODES,
    LaunchCfg,
    MegaMoeTPEngine,
    MegaMoeTPLDSError,
)

__all__ = [
    "COMM_MODES",
    "MegaMoeTP",
    "MegaMoeTPConfig",
    "MegaMoeTPLDSError",
    "mega_moe_tp_supported",
]

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
    lb_min: int | None = None
    lb_mt: int | str | None = None
    lb_mt_small: int | None = None
    lb_small_max: int | None = None
    lb_npp: int | None = None
    lb_q: int | None = None
    # the fused tail's norm: GemmaRMSNorm (scale 1 + w) or RMSNorm (w), and its eps
    tail_eps: float = 1e-6
    tail_gemma: bool = True


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
            lb={
                "min": cfg.lb_min,
                "mt": cfg.lb_mt,
                "mt_small": cfg.lb_mt_small,
                "small_max": cfg.lb_small_max,
                "npp": cfg.lb_npp,
                "q": cfg.lb_q,
            },
            tail_eps=cfg.tail_eps,
            tail_gemma=cfg.tail_gemma,
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
        """ag_rs: x [m, H] (own tokens); rs / ar: x [M, H] replicated; ar_ar: x [M, H]
        partial. tail (ag_rs, see tail_ok): (res_in, res_out, norm_w) -> also
        res_out = y + res_in and every rank's FP8 rows (+ fp32 scales; bf16: + the bf16
        rows) of the norm of res_out: returns (y, q, scale[, rows])."""
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
        self,
        local_tokens,
        tail: bool = False,
        tail_bf16: bool = False,
        warmup: bool = True,
    ) -> None:
        """Collective: compile, arm and (warmup) run once and check the launch configs
        of these local token counts (tail / tail_bf16: their fused-tail variants)."""
        self.engine.prepare(local_tokens, tail, tail_bf16, warmup)

    @property
    def max_local_tokens(self) -> int:
        return self.engine.mmax

    def tail_ok(self, local_tokens: int) -> bool:
        """forward(tail=...) runs for this local token count."""
        return self.engine.tail_ok(local_tokens)

    def launch_config(self, local_tokens: int) -> LaunchCfg:
        return self.engine.config(local_tokens)

    def error_flag(self) -> torch.Tensor:
        """Sticky watchdog bits (nonzero: a launch timed out) as a device tensor."""
        return self.engine.error_flag()

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
