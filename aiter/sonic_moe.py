# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fused_moe import fused_moe
from .ops.enum import ActivationType as AiterActivationType
from .ops.enum import QuantType


class KernelBackendMoE(Enum):
    torch = "torch"
    aiter = "aiter"


class ActivationType(Enum):
    SWIGLU = "swiglu"
    GEGLU = "geglu"
    REGLU = "reglu"
    RELU_SQ = "relu_sq"
    RELU = "relu"
    GELU = "gelu"
    SILU = "silu"


def is_glu(activation_type: ActivationType) -> bool:
    return activation_type in {
        ActivationType.SWIGLU,
        ActivationType.GEGLU,
        ActivationType.REGLU,
    }


def _coerce_activation(activation_type: ActivationType | str) -> ActivationType:
    if isinstance(activation_type, ActivationType):
        return activation_type
    return ActivationType(str(activation_type).lower())


def _coerce_backend(backend: KernelBackendMoE | str) -> KernelBackendMoE:
    if isinstance(backend, KernelBackendMoE):
        return backend
    return KernelBackendMoE(str(backend).lower())


def _apply_interleaved_activation(
    x: torch.Tensor, activation_type: ActivationType
) -> torch.Tensor:
    if activation_type == ActivationType.SWIGLU:
        gate, up = x[..., ::2], x[..., 1::2]
        return up * F.silu(gate)
    if activation_type == ActivationType.GEGLU:
        gate, up = x[..., ::2], x[..., 1::2]
        return (F.gelu(gate.float()) * up).to(dtype=x.dtype)
    if activation_type == ActivationType.REGLU:
        gate, up = x[..., ::2], x[..., 1::2]
        return (F.relu(gate) * up).to(dtype=x.dtype)
    if activation_type == ActivationType.GELU:
        return F.gelu(x.float()).to(dtype=x.dtype)
    if activation_type == ActivationType.RELU:
        return F.relu(x)
    if activation_type == ActivationType.SILU:
        return F.silu(x)
    if activation_type == ActivationType.RELU_SQ:
        return F.relu(x) ** 2
    raise ValueError(f"unsupported activation_type {activation_type}")


def _to_aiter_activation(activation_type: ActivationType) -> AiterActivationType:
    if activation_type in (ActivationType.SWIGLU, ActivationType.SILU):
        return AiterActivationType.Silu
    if activation_type in (ActivationType.GEGLU, ActivationType.GELU):
        return AiterActivationType.Gelu
    raise NotImplementedError(
        f"AITER fused_moe backend does not support SonicMoE activation "
        f"{activation_type.value!r} in this wrapper"
    )


class Experts(nn.Module):
    def __init__(
        self,
        num_experts: int,
        in_features: int,
        out_features: int,
        add_bias: bool = True,
        std: float = 0.02,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_experts, out_features, in_features))
        self.bias = (
            nn.Parameter(torch.empty(num_experts, out_features)) if add_bias else None
        )
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.std = std
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, mean=0.0, std=self.std)
        if self.bias is not None:
            self.bias.zero_()

    def extra_repr(self) -> str:
        return (
            f"num_experts={self.num_experts}, in_features={self.in_features}, "
            f"out_features={self.out_features}"
        )


class MoE(nn.Module):
    """SonicMoE-compatible module backed by AITER fused_moe on ROCm.

    The module keeps SonicMoE's parameter layout and routing semantics:
    GLU up-projection weights are stored interleaved as
    ``[gate_0, up_0, gate_1, up_1, ...]`` and routing computes
    ``softmax(topk(router_logits))``.

    The AITER backend is intended as a forward/inference path. The reference
    ``torch`` backend supports all activations and bias, and is useful for
    validation.
    """

    def __init__(
        self,
        num_experts: int,
        num_experts_per_tok: int,
        hidden_size: int,
        intermediate_size: int,
        activation_function: ActivationType | str,
        add_bias: bool,
        std: float,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.activation_function = _coerce_activation(activation_function)

        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        self.c_fc = Experts(
            num_experts=num_experts,
            in_features=hidden_size,
            out_features=(
                2 * intermediate_size
                if is_glu(self.activation_function)
                else intermediate_size
            ),
            add_bias=add_bias,
            std=std,
        )
        self.c_proj = Experts(
            num_experts=num_experts,
            in_features=intermediate_size,
            out_features=hidden_size,
            add_bias=add_bias,
            std=std,
        )
        self._aiter_w1_cache: torch.Tensor | None = None
        self._aiter_w1_cache_version = -1

    def forward(
        self,
        hidden_states: torch.Tensor,
        kernel_backend_moe: KernelBackendMoE | str = KernelBackendMoE.aiter,
        is_inference_mode: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        original_shape = hidden_states.shape
        hidden_states = hidden_states.view(-1, self.hidden_size)

        router_logits, router_weights, selected_experts = (
            self._compute_routing_weights(hidden_states)
        )
        backend = _coerce_backend(kernel_backend_moe)

        if backend == KernelBackendMoE.aiter:
            out = self._aiter_forward(hidden_states, router_weights, selected_experts)
        elif backend == KernelBackendMoE.torch:
            out = self._torch_forward(hidden_states, router_weights, selected_experts)
        else:
            raise ValueError(f"unexpected kernel_backend_moe ({kernel_backend_moe})")

        out = out.view(original_shape)
        if is_inference_mode:
            aux_loss = None
        else:
            expert_frequency = selected_experts.reshape(-1).bincount(
                minlength=self.num_experts
            )
            aux_loss = self._compute_switch_loss(
                logits=router_logits,
                probs=F.softmax(router_logits, dim=-1, dtype=torch.float32),
                expert_frequency=expert_frequency,
            )
        return out, aux_loss

    def _compute_routing_weights(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = self.router(hidden_states)
        topk_logits, selected_experts = router_logits.topk(self.top_k, dim=-1)
        router_weights = F.softmax(topk_logits.float(), dim=-1)
        return router_logits, router_weights, selected_experts

    def _aiter_forward(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        if not hidden_states.is_cuda:
            raise NotImplementedError("AITER fused_moe backend requires a CUDA/HIP device")
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            raise NotImplementedError(
                f"AITER fused_moe backend expects fp16 or bf16 input, got {hidden_states.dtype}"
            )
        if self.c_fc.bias is not None or self.c_proj.bias is not None:
            raise NotImplementedError(
                "AITER fused_moe backend in this wrapper currently supports add_bias=False"
            )

        return fused_moe(
            hidden_states,
            self._aiter_w1(),
            self.c_proj.weight.contiguous(),
            router_weights.contiguous(),
            selected_experts.to(torch.int32).contiguous(),
            activation=_to_aiter_activation(self.activation_function),
            quant_type=QuantType.No,
        )

    def _aiter_w1(self) -> torch.Tensor:
        w1 = self.c_fc.weight
        if not is_glu(self.activation_function):
            return w1.contiguous()
        if (
            self._aiter_w1_cache is not None
            and self._aiter_w1_cache_version == w1._version
            and self._aiter_w1_cache.device == w1.device
            and self._aiter_w1_cache.dtype == w1.dtype
        ):
            return self._aiter_w1_cache
        gate = w1[:, ::2, :]
        up = w1[:, 1::2, :]
        w1_aiter = torch.cat((gate, up), dim=1).contiguous()
        if not torch.is_grad_enabled():
            self._aiter_w1_cache = w1_aiter
            self._aiter_w1_cache_version = w1._version
        return w1_aiter

    def _torch_forward(
        self,
        hidden_states: torch.Tensor,
        router_weights: torch.Tensor,
        selected_experts: torch.Tensor,
    ) -> torch.Tensor:
        output = torch.zeros(
            hidden_states.size(0),
            self.hidden_size,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        router_weights = router_weights.to(dtype=hidden_states.dtype)

        for expert_idx in range(self.num_experts):
            token_idx, slot_idx = torch.where(selected_experts == expert_idx)
            if token_idx.numel() == 0:
                continue

            h = F.linear(
                hidden_states[token_idx],
                self.c_fc.weight[expert_idx],
                None if self.c_fc.bias is None else self.c_fc.bias[expert_idx],
            )
            h = _apply_interleaved_activation(h, self.activation_function)
            h = F.linear(
                h,
                self.c_proj.weight[expert_idx],
                None if self.c_proj.bias is None else self.c_proj.bias[expert_idx],
            )
            h = h * router_weights[token_idx, slot_idx, None]
            output = output.index_add(0, token_idx, h.float())

        return output.to(dtype=hidden_states.dtype)

    def _compute_switch_loss(
        self,
        logits: torch.Tensor,
        probs: torch.Tensor,
        expert_frequency: torch.Tensor,
    ) -> torch.Tensor:
        logits = logits.view(-1, logits.size(-1))
        probs = probs.view(-1, probs.size(-1))
        acc_probs = probs.sum(0)
        expert_frequency = expert_frequency.float()
        return self.num_experts * (
            F.normalize(acc_probs, p=1, dim=0)
            * F.normalize(expert_frequency, p=1, dim=0)
        ).sum()


SonicMoE = MoE

__all__ = [
    "ActivationType",
    "Experts",
    "KernelBackendMoE",
    "MoE",
    "SonicMoE",
    "is_glu",
]
