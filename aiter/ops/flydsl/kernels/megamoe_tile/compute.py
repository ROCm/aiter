# SPDX-License-Identifier: MIT
from __future__ import annotations

import contextlib
import os

import torch

from .activation import (
    aiter_activation_type,
    normalize_activation,
    validate_activation_parameters,
)
from .markers import roctx_range


@contextlib.contextmanager
def _temporary_env(name: str, value: str):
    old = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if old is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = old


def a4w4_local(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    *,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
    expert_mask: torch.Tensor | None = None,
    activation: str = "silu",
    swiglu_limit: float | None = None,
    beta: float = 4.0,
    linear_beta: float = 25.0,
) -> torch.Tensor:
    """Call AITER's legacy local fused-MoE seam.

    Inputs/weights must already follow AITER's native MXFP4 layouts.  This is the
    original seam retained for SiLU/SiTUv2 compatibility.  Use
    :func:`compute_v2.run_local_ep_a4w4` for the explicit A4W4 path, including
    SwiGLU: top-level ``fused_moe`` may otherwise select A16W4 for small-batch
    SwiGLU and silently change the quantization contract.
    """

    activation = normalize_activation(activation)
    if activation == "swiglu":
        raise NotImplementedError(
            "legacy a4w4_local cannot guarantee A4W4 for SwiGLU; "
            "use compute_v2.run_local_ep_a4w4"
        )
    validate_activation_parameters(
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=beta,
        situ_linear_beta=linear_beta,
    )

    from aiter.fused_moe import fused_moe

    import aiter

    act = aiter_activation_type(activation)
    env = _temporary_env("AITER_SITUV2_A4W4", "1") if activation == "situv2" else contextlib.nullcontext()
    with env:
        with roctx_range(f"megamoeTile.compute.a4w4_{activation}"):
            return fused_moe(
                hidden_states,
                w1,
                w2,
                topk_weights,
                topk_ids,
                expert_mask=expert_mask,
                activation=act,
                quant_type=aiter.QuantType.per_1x32,
                doweight_stage1=False,
                w1_scale=w1_scale,
                w2_scale=w2_scale,
                beta=float(beta),
                linear_beta=float(linear_beta),
                swiglu_limit=(
                    None if swiglu_limit is None else float(swiglu_limit)
                ),
            )


def a4w4_situv2_local(*args, **kwargs) -> torch.Tensor:
    """Compatibility wrapper; the new operator defaults to SiLU."""

    kwargs["activation"] = "situv2"
    return a4w4_local(*args, **kwargs)
