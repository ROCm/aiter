# SPDX-License-Identifier: Apache-2.0
"""MORI inter-node backend for the gfx950 EP16 A4W4 MegaMoEV2 path."""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class MegaMoEInterNodeContext:
    """Dispatch payload plus the private source-side state needed by combine."""

    tokens: torch.Tensor
    weights: torch.Tensor
    scales: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens: torch.Tensor
    _source_topk_ids: torch.Tensor
    _source_tokens: int
    _owner_id: int
    _generation: int
    _consumed: bool = False


class MegaMoEInterNodeBackend:
    """EP16 packed-FP4 dispatch/compute/combine implementation using MORI."""

    def __init__(self, owner, *, gpu_per_node: int = 8):
        import mori
        from aiter import dtypes

        if owner.world_size != 16 or gpu_per_node != 8:
            raise ValueError("A4W4 inter-node MegaMoEV2 requires EP16 (2 nodes x 8 GPUs)")
        if owner.model_dim % 32:
            raise ValueError("A4W4 inter-node model_dim must be divisible by 32")
        # SiTUv2 otherwise defaults to the BF16-activation A16W4 path. This
        # backend promises packed-FP4 activations, so select A4W4 explicitly.
        os.environ["AITER_SITUV2_A8W4"] = "0"
        os.environ["AITER_SITUV2_A4W4"] = "1"
        cfg = mori.ops.EpDispatchCombineConfig(
            data_type=dtypes.fp4x2,
            rank=owner.rank,
            world_size=owner.world_size,
            hidden_dim=owner.model_dim,
            scale_dim=owner.model_dim // 32,
            scale_type_size=1,
            max_num_inp_token_per_rank=owner.mtpr,
            num_experts_per_rank=owner.epr,
            num_experts_per_token=owner.topk,
            max_token_type_size=2,
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
            gpu_per_node=gpu_per_node,
            num_qp_per_pe=2,
            rdma_block_num=64,
            block_num=96,
            warp_num_per_block=8,
        )
        self.owner = owner
        self.op = mori.ops.EpDispatchCombineOp(cfg)
        self._owner_id = id(self)
        self._generation = 0
        self._active_dispatch = None

    def _validate_active_dispatch(self, dispatched):
        if not isinstance(dispatched, MegaMoEInterNodeContext):
            raise TypeError("dispatched must be MegaMoEInterNodeContext")
        if dispatched._owner_id != self._owner_id:
            raise ValueError("dispatch result belongs to a different MegaMoEV2 instance")
        if dispatched._generation != self._generation or dispatched is not self._active_dispatch:
            raise RuntimeError("dispatch result is stale; only one dispatch may be in flight")
        if dispatched._consumed:
            raise RuntimeError("dispatch result has already been consumed by combine")

    def _validate_routing(self, weights, topk_ids):
        o = self.owner
        tokens = int(topk_ids.shape[0])
        if tokens > o.mtpr:
            raise ValueError(f"run_tokens={tokens} > max_tok_per_rank={o.mtpr}")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if weights.dtype != torch.float32 or not weights.is_contiguous():
            raise ValueError("weights must be contiguous float32")
        expected = (tokens, o.topk)
        if tuple(topk_ids.shape) != expected or tuple(weights.shape) != expected:
            raise ValueError(f"weights and topk_ids must have shape {expected}")
        dev = torch.device("cuda", torch.cuda.current_device())
        if weights.device != dev or topk_ids.device != dev:
            raise ValueError(f"weights and topk_ids must be on current device {dev}")
        return tokens

    def dispatch_prequant(self, x_fp4, x_scale, weights, topk_ids):
        from aiter import dtypes

        if self._active_dispatch is not None and not self._active_dispatch._consumed:
            raise RuntimeError("complete the in-flight dispatch with combine before dispatching again")
        tokens = self._validate_routing(weights, topk_ids)
        o = self.owner
        if x_fp4.dtype != dtypes.fp4x2 or not x_fp4.is_contiguous():
            raise ValueError("x_fp4 must be contiguous fp4x2")
        if tuple(x_fp4.shape) != (tokens, o.model_dim // 2):
            raise ValueError(f"x_fp4 must have shape ({tokens}, {o.model_dim // 2})")
        if not x_scale.is_contiguous() or tuple(x_scale.shape) != (tokens, o.model_dim // 32):
            raise ValueError(f"x_scale must be contiguous with shape ({tokens}, {o.model_dim // 32})")
        # PyTorch/ROCm exposes E8M0 as a dedicated 1-byte dtype on newer
        # builds and as uint8 storage on older ones. MORI accepts both forms.
        if x_scale.element_size() != 1:
            raise ValueError("x_scale must use 1-byte E8M0 storage")
        if x_fp4.device != o.dev or x_scale.device != o.dev:
            raise ValueError(f"x_fp4 and x_scale must be on current device {o.dev}")
        recv = self.op.dispatch(x_fp4, weights, x_scale, topk_ids)
        self._generation += 1
        dispatched = MegaMoEInterNodeContext(
            tokens=recv[0], weights=recv[1], scales=recv[2], expert_ids=recv[3],
            num_tokens=recv[4], _source_topk_ids=topk_ids, _source_tokens=tokens,
            _owner_id=self._owner_id, _generation=self._generation,
        )
        self._active_dispatch = dispatched
        return dispatched

    def dispatch(self, x_bf16, weights, topk_ids):
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if tuple(x_bf16.shape) != (int(x_bf16.shape[0]), self.owner.model_dim):
            raise ValueError(f"x_bf16 must have shape (tokens, {self.owner.model_dim})")
        if x_bf16.device != self.owner.dev:
            raise ValueError(f"x_bf16 must be on current device {self.owner.dev}")
        x_fp4, x_scale = self.owner.quantize(x_bf16)
        return self.dispatch_prequant(x_fp4, x_scale, weights, topk_ids)

    def fused_moe(self, dispatched: MegaMoEInterNodeContext):
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_moe as run_fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        self._validate_active_dispatch(dispatched)
        o = self.owner
        for name in ("tokens", "weights", "scales", "expert_ids", "num_tokens"):
            value = getattr(dispatched, name)
            if value.device != o.dev:
                raise ValueError(f"dispatched.{name} must be on current device {o.dev}")
        return run_fused_moe(
            dispatched.tokens, o.w1, o.w2, dispatched.weights, dispatched.expert_ids,
            expert_mask=o.expert_mask, activation=ActivationType.Situv2,
            gate_mode=GateMode.SEPARATED.value, quant_type=QuantType.per_1x32,
            w1_scale=o.w1_scale, w2_scale=o.w2_scale, a1_scale=dispatched.scales,
            num_local_tokens=dispatched.num_tokens[:1].to(dtypes.i32), dtype=torch.bfloat16,
        )

    def combine(self, local_output, dispatched: MegaMoEInterNodeContext):
        self._validate_active_dispatch(dispatched)
        if dispatched._source_topk_ids.device != self.owner.dev or tuple(
            dispatched._source_topk_ids.shape
        ) != (
            dispatched._source_tokens, self.owner.topk
        ):
            raise ValueError("dispatch result source topk_ids has invalid device or shape")
        if local_output.device != self.owner.dev or local_output.dtype != torch.bfloat16:
            raise ValueError("local_output must be bfloat16 on the current CUDA device")
        output, output_weights = self.op.combine(local_output, None, dispatched._source_topk_ids)
        dispatched._consumed = True
        return output[: dispatched._source_tokens], output_weights

    def forward(self, x_bf16, weights, topk_ids):
        dispatched = self.dispatch(x_bf16, weights, topk_ids)
        local_output = self.fused_moe(dispatched)
        return self.combine(local_output, dispatched)[0]
