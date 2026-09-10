# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Whole-graph fused MoE implementations selected by ``kernelName1``.

Implementations receive a normalized request and an opaque config string. Code
registers a stable name; tuned CSV rows select it with the convention
``impl__<name>__<config>``.
"""

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class FusedMoeRequest:
    hidden_states: torch.Tensor
    w1: torch.Tensor
    w2: torch.Tensor
    topk_weight: torch.Tensor
    topk_ids: torch.Tensor
    expert_mask: torch.Tensor | None = None
    activation: Any = None
    quant_type: Any = None
    doweight_stage1: bool = False
    w1_scale: torch.Tensor | None = None
    w2_scale: torch.Tensor | None = None
    a1_scale: torch.Tensor | None = None
    a2_scale: torch.Tensor | None = None
    block_size_m: int | None = None
    ksplit: int = 0
    num_local_tokens: torch.Tensor | None = None
    moe_sorting_dispatch_policy: int = 0
    dtype: torch.dtype | None = None
    hidden_pad: int = 0
    intermediate_pad: int = 0
    bias1: torch.Tensor | None = None
    bias2: torch.Tensor | None = None
    swiglu_limit: float | None = None
    beta: float | None = None
    linear_beta: float | None = None
    gate_mode: Any = None
    q_dtype_a: torch.dtype | None = None
    q_dtype_w: torch.dtype | None = None


FusedMoeImpl = Callable[[FusedMoeRequest, str], torch.Tensor]
BoundFusedMoeImpl = Callable[[FusedMoeRequest], torch.Tensor]
_IMPLEMENTATIONS: dict[str, FusedMoeImpl | str] = {}

# ``kernelName1`` protocol: ``impl__<registered-name>__<opaque-config>``.
# The name must be non-empty and cannot contain ``__``. Everything after the
# second separator is passed to the implementation unchanged, so the config may
# itself contain ``__``. Names without this prefix remain normal stage1 kernels.
# Adding a new extension:
# 1. In the backend module, implement ``run(request, config) -> Tensor``; see
#    ``aiter/ops/flydsl/fused_moe_gfx942.py::run_flydsl_moe_gfx942_impl``.
# 2. In the backend package initializer, call ``register_fused_moe_impl`` with a
#    stable name and the function/import path; see ``aiter/ops/flydsl/__init__.py``.
# 3. In the tuner/config writer, call ``make_fused_moe_impl_kernel_name`` and
#    store its result in ``kernelName1``. ``aiter/fused_moe.py`` then resolves
#    and runs the registered whole-graph implementation automatically.
_KERNEL_NAME_PREFIX = "impl__"


class FusedMoeImplResolutionError(ValueError):
    pass


def register_fused_moe_impl(name: str, impl: FusedMoeImpl | str) -> None:
    """Register a whole-graph fused MoE implementation or lazy import path."""
    previous = _IMPLEMENTATIONS.get(name)
    if previous is not None and previous != impl:
        raise ValueError(f"Fused MoE implementation already registered: {name}")
    _IMPLEMENTATIONS[name] = impl


def make_fused_moe_impl_kernel_name(name: str, config: str) -> str:
    if not name or "__" in name:
        raise ValueError(f"Invalid fused MoE implementation name: {name!r}")
    return f"{_KERNEL_NAME_PREFIX}{name}__{config}"


def resolve_fused_moe_impl(kernel_name: str) -> BoundFusedMoeImpl | None:
    """Resolve ``impl__<name>__<config>`` and bind its opaque config."""
    if not kernel_name.startswith(_KERNEL_NAME_PREFIX):
        return None

    name, separator, config = kernel_name[len(_KERNEL_NAME_PREFIX) :].partition("__")
    if not name or not separator:
        raise FusedMoeImplResolutionError(
            f"Invalid fused MoE implementation kernel name: {kernel_name!r}"
        )
    candidate = _IMPLEMENTATIONS.get(name)
    if candidate is None:
        raise FusedMoeImplResolutionError(f"Unknown fused MoE implementation: {name!r}")
    if callable(candidate):
        implementation = candidate
    else:
        try:
            module_name, attribute = candidate.rsplit(":", 1)
            implementation = getattr(importlib.import_module(module_name), attribute)
            if not callable(implementation):
                raise TypeError(
                    f"Fused MoE implementation is not callable: {candidate}"
                )
        except (ImportError, AttributeError, TypeError, ValueError) as error:
            raise FusedMoeImplResolutionError(
                f"Failed to load fused MoE implementation {name!r}: {error}"
            ) from error

    return lambda request: implementation(request, config)
