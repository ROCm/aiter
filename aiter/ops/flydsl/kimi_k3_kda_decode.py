# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level API for the fused Kimi-K3 KDA decode specialization."""

from __future__ import annotations

import functools
from collections.abc import Iterable

import torch

from .kernels.kimi_k3_kda_decode import (
    create_kimi_k3_kda_decode_kernel,
)
from .kernels.tensor_shim import _run_compiled

_HEADS = 12
_DIM = 128
_CONV_CHANNELS = 3 * _HEADS * _DIM
_CONV_WIDTH = 4


@functools.lru_cache(maxsize=None)
def _rocm_arch(device: torch.device) -> str | None:
    properties = torch.cuda.get_device_properties(device)
    arch = getattr(properties, "gcnArchName", None)
    return arch.split(":", 1)[0] if arch is not None else None


def is_flydsl_kimi_k3_kda_decode_supported(
    device: torch.device | str | int | None = None,
) -> bool:
    """Return whether ``device`` can run this gfx950-only specialization."""
    if not torch.cuda.is_available():
        return False
    try:
        resolved = torch.device(
            "cuda",
            torch.cuda.current_device(),
        )
        if device is not None:
            resolved = (
                torch.device("cuda", device)
                if isinstance(device, int)
                else torch.device(device)
            )
            if resolved.type != "cuda":
                return False
            if resolved.index is None:
                resolved = torch.device(
                    "cuda",
                    torch.cuda.current_device(),
                )
        return _rocm_arch(resolved) == "gfx950"
    except (AssertionError, RuntimeError, TypeError, ValueError):
        return False


def _check_tensor(
    name: str,
    tensor: torch.Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    inner_strides: tuple[int, ...] = (),
) -> None:
    if tensor.shape != shape:
        raise ValueError(
            f"`{name}` must have shape {list(shape)}, " f"got {list(tensor.shape)}."
        )
    if tensor.dtype != dtype:
        raise ValueError(f"`{name}` must have dtype {dtype}, got {tensor.dtype}.")
    if tensor.device != device:
        raise ValueError(f"`{name}` must be on {device}, got {tensor.device}.")
    if inner_strides and tensor.stride()[-len(inner_strides) :] != inner_strides:
        raise ValueError(
            f"`{name}` must have inner strides {inner_strides}, "
            f"got {tensor.stride()}."
        )


def _check_same_device(
    tensors: Iterable[tuple[str, torch.Tensor]],
    device: torch.device,
) -> None:
    for name, tensor in tensors:
        if not tensor.is_cuda:
            raise ValueError(f"`{name}` must be a CUDA tensor.")
        if tensor.device != device:
            raise ValueError(f"`{name}` must be on {device}, got {tensor.device}.")


def flydsl_kimi_k3_kda_decode(
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    raw_g: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    norm_eps: float,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fused Kimi-K3 KDA decode on MI350-series GPUs.

    This pure-decode specialization fuses the packed width-4 Q/K/V causal
    convolution, the FP32 recurrent-state update, and the BF16
    RMSNorm/sigmoid output gate. Slot zero is reserved: non-positive
    ``state_indices`` produce zero output without modifying either cache.

    The layout is fixed to Kimi-K3 TP8: 12 local heads and 128-dimensional
    key/value state. Call
    :func:`is_flydsl_kimi_k3_kda_decode_supported` before dispatching from a
    model implementation.
    """
    if not x.is_cuda:
        raise ValueError("`x` must be a CUDA tensor.")
    device = x.device
    if not is_flydsl_kimi_k3_kda_decode_supported(device):
        raise RuntimeError("`flydsl_kimi_k3_kda_decode` requires a gfx950 GPU.")
    batch = x.shape[0] if x.ndim == 2 else -1
    if batch <= 0:
        raise ValueError("`x` must have a non-empty batch dimension.")
    if conv_bias is not None:
        raise ValueError("This specialization requires `conv_bias=None`.")
    if lower_bound is None:
        raise ValueError("This specialization requires the KDA lower-bound gate.")

    tensors = (
        ("conv_weight", conv_weight),
        ("conv_state", conv_state),
        ("raw_g", raw_g),
        ("raw_beta", raw_beta),
        ("A_log", A_log),
        ("dt_bias", dt_bias),
        ("state", state),
        ("state_indices", state_indices),
        ("output_gate", output_gate),
        ("norm_weight", norm_weight),
    )
    _check_same_device(tensors, device)

    _check_tensor(
        "x",
        x,
        shape=(batch, _CONV_CHANNELS),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "conv_weight",
        conv_weight,
        shape=(_CONV_CHANNELS, _CONV_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    if conv_state.ndim != 3 or conv_state.shape[1:] != (
        _CONV_CHANNELS,
        _CONV_WIDTH - 1,
    ):
        raise ValueError(
            "`conv_state` must have shape [cache, 4608, 3], "
            f"got {list(conv_state.shape)}."
        )
    if conv_state.dtype != torch.bfloat16:
        raise ValueError("`conv_state` must have dtype torch.bfloat16.")
    if state.ndim != 4 or state.shape[1:] != (
        _HEADS,
        _DIM,
        _DIM,
    ):
        raise ValueError(
            "`state` must have shape [cache, 12, 128, 128], "
            f"got {list(state.shape)}."
        )
    if state.dtype != torch.float32:
        raise ValueError("`state` must have dtype torch.float32.")
    if state.stride()[-3:] != (_DIM * _DIM, _DIM, 1):
        raise ValueError("`state` must be contiguous within each cache slot.")
    _check_tensor(
        "raw_g",
        raw_g,
        shape=(1, batch, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(_DIM, 1),
    )
    _check_tensor(
        "raw_beta",
        raw_beta,
        shape=(1, batch, _HEADS),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "A_log",
        A_log,
        shape=(_HEADS,),
        dtype=torch.float32,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "dt_bias",
        dt_bias,
        shape=(_HEADS * _DIM,),
        dtype=torch.float32,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "state_indices",
        state_indices,
        shape=(batch,),
        dtype=torch.int32,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "output_gate",
        output_gate,
        shape=(batch, _HEADS, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "norm_weight",
        norm_weight,
        shape=(_DIM,),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )

    if out is None:
        out = torch.empty(
            (1, batch, _HEADS, _DIM),
            dtype=torch.bfloat16,
            device=device,
        )
    else:
        _check_same_device((("out", out),), device)
        _check_tensor(
            "out",
            out,
            shape=(1, batch, _HEADS, _DIM),
            dtype=torch.bfloat16,
            device=device,
            inner_strides=(1,),
        )

    executable = create_kimi_k3_kda_decode_kernel(
        float(norm_eps),
        float(lower_bound),
    )
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        _run_compiled(
            executable,
            x,
            conv_weight,
            conv_state,
            raw_g,
            raw_beta,
            A_log,
            dt_bias,
            state,
            state_indices,
            output_gate,
            norm_weight,
            out,
            batch,
            x.stride(0),
            conv_weight.stride(0),
            conv_weight.stride(1),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
            raw_g.stride(1),
            raw_beta.stride(1),
            state.stride(0),
            output_gate.stride(0),
            output_gate.stride(1),
            out.stride(1),
            out.stride(2),
            stream,
        )
    return out


__all__ = [
    "flydsl_kimi_k3_kda_decode",
    "is_flydsl_kimi_k3_kda_decode_supported",
]
