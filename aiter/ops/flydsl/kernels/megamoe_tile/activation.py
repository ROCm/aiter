# SPDX-License-Identifier: MIT
"""Activation contract shared by the hierarchical A4W4 pipeline."""

from __future__ import annotations

import math
from typing import Final

import torch


SUPPORTED_ACTIVATIONS: Final[tuple[str, ...]] = ("silu", "swiglu", "situv2")
DEFAULT_SWIGLU_LIMIT: Final[float] = 7.0
DEFAULT_SITUV2_BETA: Final[float] = 4.0
DEFAULT_SITUV2_LINEAR_BETA: Final[float] = 25.0


def normalize_activation(activation: str) -> str:
    """Return the compile-time activation spelling used by AITER/FlyDSL.

    The public field is case-insensitive and tolerates ``-``/``_`` separators.
    ``situ`` is accepted because it is the Kimi-K3 model-config spelling for
    AITER's ``situv2`` implementation.  ``siluv2`` is deliberately rejected:
    it is not an AITER activation and is not an alias for SiTUv2.
    """

    if not isinstance(activation, str):
        raise TypeError("activation must be a string")
    key = activation.strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "silu": "silu",
        "swiglu": "swiglu",
        "situ": "situv2",
        "situv2": "situv2",
    }
    try:
        return aliases[key]
    except KeyError as error:
        supported = ", ".join(SUPPORTED_ACTIVATIONS)
        raise ValueError(
            f"unsupported activation {activation!r}; expected one of: {supported}"
        ) from error


def validate_activation_parameters(
    *,
    activation: str | None = None,
    swiglu_limit: float | None = None,
    situ_beta: float = DEFAULT_SITUV2_BETA,
    situ_linear_beta: float = DEFAULT_SITUV2_LINEAR_BETA,
) -> None:
    if (
        activation is not None
        and normalize_activation(activation) == "situv2"
        and swiglu_limit is not None
    ):
        raise ValueError("swiglu_limit does not apply to situv2; use None")
    values = {
        "situ_beta": float(situ_beta),
        "situ_linear_beta": float(situ_linear_beta),
    }
    if swiglu_limit is not None:
        limit = float(swiglu_limit)
        if math.isnan(limit) or limit <= 0.0:
            raise ValueError(
                f"swiglu_limit must be positive (or +inf), got {limit!r}"
            )
    for name, value in values.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive, got {value!r}")


def apply_gate_up(
    gate: torch.Tensor,
    up: torch.Tensor,
    activation: str,
    *,
    swiglu_limit: float | None = None,
    situ_beta: float = DEFAULT_SITUV2_BETA,
    situ_linear_beta: float = DEFAULT_SITUV2_LINEAR_BETA,
) -> torch.Tensor:
    """Torch oracle matching AITER's stage-1 activation epilogues."""

    activation = normalize_activation(activation)
    validate_activation_parameters(
        activation=activation,
        swiglu_limit=swiglu_limit,
        situ_beta=situ_beta,
        situ_linear_beta=situ_linear_beta,
    )
    if activation == "silu":
        if swiglu_limit is not None:
            limit = float(swiglu_limit)
            gate = gate.clamp(max=limit)
            up = up.clamp(min=-limit, max=limit)
        return torch.nn.functional.silu(gate) * up
    if activation == "swiglu":
        limit = (
            DEFAULT_SWIGLU_LIMIT
            if swiglu_limit is None
            else float(swiglu_limit)
        )
        gate = gate.clamp(max=limit)
        up = up.clamp(min=-limit, max=limit)
        return gate * torch.sigmoid(1.702 * gate) * (up + 1.0)

    beta = float(situ_beta)
    linear_beta = float(situ_linear_beta)
    situ_gate = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
    situ_up = linear_beta * torch.tanh(up / linear_beta)
    return situ_gate * situ_up


def aiter_activation_type(activation: str):
    """Map a public activation string to AITER's native enum lazily."""

    import aiter

    activation = normalize_activation(activation)
    return {
        "silu": aiter.ActivationType.Silu,
        "swiglu": aiter.ActivationType.Swiglu,
        "situv2": aiter.ActivationType.Situv2,
    }[activation]
