# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
"""Public MegaMoEV2 facade selecting an intra- or inter-node backend."""

import torch

from .backend import MegaMoEBackend

__all__ = ["MegaMoEV2"]


class MegaMoEV2:
    """Fused MoE facade with one stable API across intra- and inter-node paths."""

    # fmt: off
    def __init__(self, *, rank: int, world_size: int, model_dim: int, inter_dim: int, experts: int, topk: int,
        quant: str, w1: torch.Tensor, w1_scale: torch.Tensor, w2: torch.Tensor, w2_scale: torch.Tensor,
        max_tok_per_rank: int, mega_scheme: str = "fixedslot", swiglu_limit: float = 0.0):
    # fmt: on
        if quant not in ("a8w4", "a4w4"):
            raise ValueError("MegaMoEV2 supports quant='a8w4' or quant='a4w4'")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank={rank} must be in [0, world_size={world_size})")
        if experts % world_size != 0:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0:
            raise ValueError("max_tok_per_rank must be positive")
        if quant == "a8w4" and max_tok_per_rank & (max_tok_per_rank - 1):
            raise ValueError(f"max_tok_per_rank={max_tok_per_rank} must be a power of two for A8W4")
        if swiglu_limit < 0:
            raise ValueError("swiglu_limit must be non-negative")

        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = int(experts // world_size)
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.quant = quant
        self.swiglu_limit = float(swiglu_limit)
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.w1 = w1 if w1.is_contiguous() else w1.contiguous()
        self.w1_scale = w1_scale if w1_scale.is_contiguous() else w1_scale.contiguous()
        self.w2 = w2 if w2.is_contiguous() else w2.contiguous()
        self.w2_scale = w2_scale if w2_scale.is_contiguous() else w2_scale.contiguous()

        if self.world_size == 16:
            from aiter.jit.utils.chip_info import get_gfx_runtime

            if get_gfx_runtime() != "gfx950" or quant != "a4w4":
                raise ValueError("EP16 is only supported with quant='a4w4' on gfx950")
            from .inter_node import MegaMoEInterNodeBackend

            self._backend: MegaMoEBackend = MegaMoEInterNodeBackend(
                rank=self.rank, world_size=self.world_size, model_dim=self.model_dim,
                experts=self.experts, topk=self.topk, w1=self.w1, w1_scale=self.w1_scale,
                w2=self.w2, w2_scale=self.w2_scale, max_tok_per_rank=self.mtpr,
            )
        else:
            if quant == "a4w4":
                raise ValueError("quant='a4w4' is supported only for EP16 on gfx950")
            from .intra_node import MegaMoEIntraNodeBackend

            self._backend = MegaMoEIntraNodeBackend(
                rank=self.rank, world_size=self.world_size, model_dim=self.model_dim,
                inter_dim=self.inter_dim, experts=self.experts, topk=self.topk, quant=self.quant,
                w1=self.w1, w1_scale=self.w1_scale, w2=self.w2, w2_scale=self.w2_scale,
                max_tok_per_rank=self.mtpr, mega_scheme=mega_scheme,
                swiglu_limit=self.swiglu_limit,
            )

    def quantize(self, x_bf16):
        return self._backend.quantize(x_bf16)

    def forward(self, x_bf16, wts, topk_ids, *, stream=None, slice_output=True):
        return self._backend.forward(
            x_bf16, wts, topk_ids, stream=stream, slice_output=slice_output
        )

    def forward_prequant(self, x_q, scales, wts, topk_ids, *, stream=None, slice_output=True):
        return self._backend.forward_prequant(
            x_q, scales, wts, topk_ids, stream=stream, slice_output=slice_output
        )

    forward_bf16 = forward
    __call__ = forward
