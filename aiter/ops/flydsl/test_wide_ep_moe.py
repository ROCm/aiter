# SPDX-License-Identifier: Apache-2.0
"""Standalone MORI inter-node EP16 A4W4 MoE operator.

This FlyDSL operator is intentionally independent from MegaMoEV2 and its intranode
implementation.  It only mirrors MegaMoEV2's public calling convention.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch


@dataclass
class TestWideEpMoeContext:
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


class TestWideEpMoe:
    """EP16 packed-FP4 dispatch/compute/combine implementation using MORI."""

    def __init__(
        self, *, rank, world_size, model_dim, inter_dim, experts, topk, quant,
        w1, w1_scale, w2, w2_scale, max_tok_per_rank, gpu_per_node: int = 8,
        mega_scheme: str = "fixedslot", swiglu_limit: float = 0.0,
    ):
        import mori
        from aiter import dtypes
        from aiter.jit.utils.chip_info import get_gfx_runtime

        if get_gfx_runtime() != "gfx950":
            raise ValueError("TestWideEpMoe is supported only on gfx950")
        if quant != "a4w4":
            raise ValueError("TestWideEpMoe supports quant='a4w4' only")
        if world_size != 16 or gpu_per_node != 8:
            raise ValueError("TestWideEpMoe requires EP16 (2 nodes x 8 GPUs)")
        if experts % world_size:
            raise ValueError(f"experts={experts} must be divisible by world_size={world_size}")
        if max_tok_per_rank <= 0:
            raise ValueError("max_tok_per_rank must be positive")
        if model_dim % 32:
            raise ValueError("A4W4 inter-node model_dim must be divisible by 32")
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.model_dim = int(model_dim)
        self.inter_dim = int(inter_dim)
        self.experts = int(experts)
        self.epr = self.experts // self.world_size
        self.topk = int(topk)
        self.mtpr = int(max_tok_per_rank)
        self.quant = quant
        self.mega_scheme = mega_scheme
        self.swiglu_limit = float(swiglu_limit)
        self.capacity_mtpr = 1 << (self.mtpr - 1).bit_length()
        self.dev = torch.device("cuda", torch.cuda.current_device())
        self.w1 = w1
        self.w1_scale = w1_scale
        self.w2 = w2
        self.w2_scale = w2_scale
        local_start = self.rank * self.epr
        self.expert_mask = torch.zeros(self.experts + 1, dtype=torch.int32, device=self.dev)
        self.expert_mask[local_start : local_start + self.epr] = 1
        # SiTUv2 otherwise defaults to the BF16-activation A16W4 path. This
        # backend promises packed-FP4 activations, so select A4W4 explicitly.
        os.environ["AITER_SITUV2_A8W4"] = "0"
        os.environ["AITER_SITUV2_A4W4"] = "1"
        cfg = mori.ops.EpDispatchCombineConfig(
            data_type=dtypes.fp4x2,
            rank=self.rank,
            world_size=self.world_size,
            hidden_dim=self.model_dim,
            scale_dim=self.model_dim // 32,
            scale_type_size=1,
            max_num_inp_token_per_rank=self.capacity_mtpr,
            num_experts_per_rank=self.epr,
            num_experts_per_token=self.topk,
            max_token_type_size=2,
            kernel_type=mori.ops.EpDispatchCombineKernelType.InterNodeV1LL,
            gpu_per_node=gpu_per_node,
            num_qp_per_pe=2,
            rdma_block_num=int(os.environ.get("MORI_EP_RDMA_BLOCK_NUM", "64")),
            block_num=int(os.environ.get("MORI_EP_BLOCK_NUM", "96")),
            warp_num_per_block=int(os.environ.get("MORI_EP_WARP_PER_BLOCK", "8")),
        )
        self.op = mori.ops.EpDispatchCombineOp(cfg)
        self._owner_id = id(self)
        self._generation = 0
        self._active_dispatch = None

    def _validate_active_dispatch(self, dispatched):
        if not isinstance(dispatched, TestWideEpMoeContext):
            raise TypeError("dispatched must be TestWideEpMoeContext")
        if dispatched._owner_id != self._owner_id:
            raise ValueError("dispatch result belongs to a different TestWideEpMoe instance")
        if dispatched._generation != self._generation or dispatched is not self._active_dispatch:
            raise RuntimeError("dispatch result is stale; only one dispatch may be in flight")
        if dispatched._consumed:
            raise RuntimeError("dispatch result has already been consumed by combine")

    def _validate_routing(self, weights, topk_ids):
        tokens = int(topk_ids.shape[0])
        if tokens > self.mtpr:
            raise ValueError(f"run_tokens={tokens} > max_tok_per_rank={self.mtpr}")
        if topk_ids.dtype != torch.int32 or not topk_ids.is_contiguous():
            raise ValueError("topk_ids must be contiguous int32")
        if weights.dtype != torch.float32 or not weights.is_contiguous():
            raise ValueError("weights must be contiguous float32")
        expected = (tokens, self.topk)
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
        if x_fp4.dtype != dtypes.fp4x2 or not x_fp4.is_contiguous():
            raise ValueError("x_fp4 must be contiguous fp4x2")
        if tuple(x_fp4.shape) != (tokens, self.model_dim // 2):
            raise ValueError(f"x_fp4 must have shape ({tokens}, {self.model_dim // 2})")
        if not x_scale.is_contiguous() or tuple(x_scale.shape) != (tokens, self.model_dim // 32):
            raise ValueError(f"x_scale must be contiguous with shape ({tokens}, {self.model_dim // 32})")
        # PyTorch/ROCm exposes E8M0 as a dedicated 1-byte dtype on newer
        # builds and as uint8 storage on older ones. MORI accepts both forms.
        if x_scale.element_size() != 1:
            raise ValueError("x_scale must use 1-byte E8M0 storage")
        if x_fp4.device != self.dev or x_scale.device != self.dev:
            raise ValueError(f"x_fp4 and x_scale must be on current device {self.dev}")
        recv = self.op.dispatch(x_fp4, weights, x_scale, topk_ids)
        self._generation += 1
        dispatched = TestWideEpMoeContext(
            tokens=recv[0], weights=recv[1], scales=recv[2], expert_ids=recv[3],
            num_tokens=recv[4], _source_topk_ids=topk_ids, _source_tokens=tokens,
            _owner_id=self._owner_id, _generation=self._generation,
        )
        self._active_dispatch = dispatched
        return dispatched

    def dispatch(self, x_bf16, weights, topk_ids):
        if x_bf16.dtype != torch.bfloat16 or not x_bf16.is_contiguous():
            raise ValueError("x_bf16 must be contiguous bfloat16")
        if tuple(x_bf16.shape) != (int(x_bf16.shape[0]), self.model_dim):
            raise ValueError(f"x_bf16 must have shape (tokens, {self.model_dim})")
        if x_bf16.device != self.dev:
            raise ValueError(f"x_bf16 must be on current device {self.dev}")
        x_fp4, x_scale = self.quantize(x_bf16)
        return self.dispatch_prequant(x_fp4, x_scale, weights, topk_ids)

    def quantize(self, x_bf16):
        from .kernels.mega_moe.quant import per_1x32_mx_quant

        return per_1x32_mx_quant(x_bf16, quant_mode="fp4")

    def fused_moe(self, dispatched: TestWideEpMoeContext):
        from aiter import ActivationType, QuantType, dtypes
        from aiter.fused_moe import fused_moe as run_fused_moe
        from aiter.ops.flydsl.moe_common import GateMode

        self._validate_active_dispatch(dispatched)
        for name in ("tokens", "weights", "scales", "expert_ids", "num_tokens"):
            value = getattr(dispatched, name)
            if value.device != self.dev:
                raise ValueError(f"dispatched.{name} must be on current device {self.dev}")
        return run_fused_moe(
            dispatched.tokens, self.w1, self.w2, dispatched.weights, dispatched.expert_ids,
            expert_mask=self.expert_mask, activation=ActivationType.Situv2,
            gate_mode=GateMode.SEPARATED.value, quant_type=QuantType.per_1x32,
            w1_scale=self.w1_scale, w2_scale=self.w2_scale, a1_scale=dispatched.scales,
            num_local_tokens=dispatched.num_tokens[:1].to(dtypes.i32), dtype=torch.bfloat16,
        )

    def combine(self, local_output, dispatched: TestWideEpMoeContext):
        self._validate_active_dispatch(dispatched)
        if dispatched._source_topk_ids.device != self.dev or tuple(
            dispatched._source_topk_ids.shape
        ) != (
            dispatched._source_tokens, self.topk
        ):
            raise ValueError("dispatch result source topk_ids has invalid device or shape")
        if local_output.device != self.dev or local_output.dtype != torch.bfloat16:
            raise ValueError("local_output must be bfloat16 on the current CUDA device")
        output, output_weights = self.op.combine(local_output, None, dispatched._source_topk_ids)
        dispatched._consumed = True
        return output[: dispatched._source_tokens], output_weights

    @staticmethod
    def _validate_public_options(stream, slice_output):
        if stream is not None or not slice_output:
            raise ValueError("EP16 A4W4 currently supports current stream and slice_output=True only")

    def forward(self, x_bf16, weights, topk_ids, *, stream=None, slice_output=True):
        self._validate_public_options(stream, slice_output)
        dispatched = self.dispatch(x_bf16, weights, topk_ids)
        local_output = self.fused_moe(dispatched)
        return self.combine(local_output, dispatched)[0]

    def forward_prequant(
        self, x_fp4, x_scale, weights, topk_ids, *, stream=None, slice_output=True
    ):
        self._validate_public_options(stream, slice_output)
        dispatched = self.dispatch_prequant(x_fp4, x_scale, weights, topk_ids)
        if os.environ.get("AITER_DEBUG_WIDE_EP", "0") == "1":
            print(f"[TestWideEpMoe rank={self.rank}] dispatch complete", flush=True)
        local_output = self.fused_moe(dispatched)
        if os.environ.get("AITER_DEBUG_WIDE_EP", "0") == "1":
            print(f"[TestWideEpMoe rank={self.rank}] fused_moe complete", flush=True)
        output = self.combine(local_output, dispatched)[0]
        if os.environ.get("AITER_DEBUG_WIDE_EP", "0") == "1":
            print(f"[TestWideEpMoe rank={self.rank}] combine complete", flush=True)
        return output

    forward_bf16 = forward
    __call__ = forward
