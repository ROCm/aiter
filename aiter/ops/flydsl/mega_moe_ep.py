# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Expert-parallel MoE as one kernel (MegaMoE EP), gfx950.

Experts are partitioned over the EP group (``experts_per_rank = experts // world_size``);
one persistent launch runs the all-to-all dispatch, the local experts' GEMM1 +
activation + GEMM2 and the all-to-all combine (top-k weighted sum), replacing the
multi-launch EP pipeline (``mega_moe/mega_moe_v2.py``). Weights are MXFP4.

Mirrors the :mod:`mega_moe_tp` public surface and keeps MegaMoEV2's constructor /
forward shapes so the op-test can validate it against MegaMoEV2 and ``ep_reference``.

* The kernel is persistent (one CTA per CU, CTAs and ranks spin on each other): it
  needs the whole GPU; do not run it next to other kernels / processes.
* The symmetric memory uses hipIpc handles; in containers on the host network ROCm 7.1
  needs ``HSA_ENABLE_IPC_MODE_LEGACY=1``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .kernels.mega_moe_ep.mega_moe_ep import MegaMoeEPEngine, ep_reference

__all__ = ["MegaMoeEP", "MegaMoeEPConfig", "ep_reference", "mega_moe_ep_supported"]


def mega_moe_ep_supported(gfx: str | None = None) -> bool:
    """Whether this device has the kernel (gfx950)."""
    if gfx is None:
        from aiter.jit.utils.chip_info import get_gfx

        gfx = get_gfx()
    return gfx == "gfx950"


@dataclass(frozen=True)
class MegaMoeEPConfig:
    rank: int
    world_size: int
    model_dim: int
    inter_dim: int
    experts: int
    topk: int
    max_tok_per_rank: int
    quant: str = "a4w4"
    act: str = "silu"
    situ_beta: float = 1.0
    situ_linear_beta: float = 1.0
    swiglu_limit: float = 0.0


class MegaMoeEP:
    """One fused EP MoE layer (see the module docstring)."""

    # forward(mask_invalid_slots=...) is accepted: combine can skip -1 top-k slots.
    supports_combine_mask = True

    def __init__(
        self,
        cfg: MegaMoeEPConfig,
        *,
        w1,
        w1_scale,
        w2,
        w2_scale,
        group=None,
        device=None,
        weights_preshuffled: bool = False,
    ):
        """``weights_preshuffled``: w1 / w1_scale / w2 / w2_scale are already in MegaMoEV2's
        layout (``shuffle_weight_a16w4(., 16, gate_up)`` / ``shuffle_scale_a16w4``) and
        are bound by pointer (kept alive by the caller); else natural MXFP4, shuffled
        into copies."""
        if not mega_moe_ep_supported():
            raise ValueError("MegaMoeEP: needs gfx950")
        self.cfg = cfg
        self.engine = MegaMoeEPEngine(
            rank=cfg.rank,
            world_size=cfg.world_size,
            model_dim=cfg.model_dim,
            inter_dim=cfg.inter_dim,
            experts=cfg.experts,
            topk=cfg.topk,
            max_tok_per_rank=cfg.max_tok_per_rank,
            w1=w1,
            w1_scale=w1_scale,
            w2=w2,
            w2_scale=w2_scale,
            quant=cfg.quant,
            act=cfg.act,
            situ_beta=cfg.situ_beta,
            situ_linear_beta=cfg.situ_linear_beta,
            swiglu_limit=cfg.swiglu_limit,
            group=group,
            device=device,
            weights_preshuffled=weights_preshuffled,
        )

    def forward(
        self,
        x_bf16,
        wts,
        topk_ids,
        *,
        stream=None,
        slice_output=True,
        mask_invalid_slots=None,
    ):
        return self.engine(
            x_bf16,
            wts,
            topk_ids,
            stream=stream,
            slice_output=slice_output,
            mask_invalid_slots=mask_invalid_slots,
        )

    __call__ = forward

    def prepare(self) -> None:
        """Collective: compile every launch variant on every rank (forward's first call
        does it otherwise; call it before capturing a CUDA graph)."""
        self.engine.prepare()

    def reference_forward(self, x_bf16, wts, topk_ids, ref_weights):
        """Pure-torch EP MoE oracle for these inputs; ``ref_weights`` are the
        unshuffled quantized weights ``(w1_q, w1_scale, w2_q, w2_scale)``."""
        return self.engine.reference_forward(x_bf16, wts, topk_ids, ref_weights)

    def set_weights(
        self, *, w1, w1_scale, w2, w2_scale, preshuffled: bool = False
    ) -> None:
        """Run the next forwards on another layer's weights of the same shape
        (``preshuffled``: as in the constructor)."""
        self.engine.set_weights(w1, w1_scale, w2, w2_scale, preshuffled)

    def error_flag(self):
        return self.engine.error_flag()

    def poll_errors(self) -> int:
        return self.engine.poll_errors()

    def reset(self) -> None:
        self.engine.reset()
