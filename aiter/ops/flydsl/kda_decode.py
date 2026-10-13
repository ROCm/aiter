# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""High-level API for fused KDA decode on gfx950 with supported head counts."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from aiter.jit.utils.chip_info import get_gfx_runtime

from .kernels.kda_decode_gfx950 import create_kda_decode_kernel
from .kernels.tensor_shim import _run_compiled

_SUPPORTED_HEADS = (8, 12, 16)
_DIM = 128
_CONV_WIDTH = 4
_INT32_MAX = 2**31 - 1


def is_flydsl_kda_decode_supported(
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
        if not 0 <= resolved.index < torch.cuda.device_count():
            return False
        return get_gfx_runtime() == "gfx950"
    except (AssertionError, KeyError, RuntimeError, TypeError, ValueError):
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
            f"`{name}` must have shape {list(shape)}, got {list(tensor.shape)}."
        )
    if tensor.dtype != dtype:
        raise ValueError(f"`{name}` must have dtype {dtype}, got {tensor.dtype}.")
    if tensor.device != device:
        raise ValueError(f"`{name}` must be on {device}, got {tensor.device}.")
    if inner_strides and tensor.stride()[-len(inner_strides) :] != inner_strides:
        raise ValueError(
            f"`{name}` must have inner strides {inner_strides}, got {tensor.stride()}."
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


def _num_cache_slots(state: torch.Tensor, conv_state: torch.Tensor) -> int:
    """Return the slot count shared by both caches, clamped to the int32 index range."""
    return min(state.shape[0], conv_state.shape[0], _INT32_MAX)


def _check_dense_cache_slots(
    name: str,
    tensor: torch.Tensor,
    layouts: tuple[tuple[int, ...], ...],
) -> None:
    """Require disjoint, dense cache slots while allowing padding between slots.

    ``layouts`` lists the accepted inner strides; the kernel takes every stride
    as an argument, so any dense ordering of a slot's elements works.
    """
    if (
        tensor.stride()[1:] not in layouts
        or tensor.stride(0) < tensor.shape[1:].numel()
    ):
        raise ValueError(f"`{name}` must have dense, non-overlapping cache slots.")


def _check_conv_cache_slots(tensor: torch.Tensor) -> None:
    """Accept channel-major and transposed width-major convolution envelopes."""
    channels = tensor.shape[1]
    channel_stride, width_stride = tensor.stride()[1:]
    if width_stride == 1 and channel_stride >= _CONV_WIDTH - 1:
        slot_span = (channels - 1) * channel_stride + _CONV_WIDTH - 1
    elif channel_stride == 1 and width_stride >= channels:
        slot_span = (_CONV_WIDTH - 2) * width_stride + channels
    else:
        raise ValueError("`conv_state` must have non-overlapping channel/width rows.")
    if tensor.stride(0) < slot_span:
        raise ValueError("`conv_state` must have non-overlapping cache slots.")


def _check_i32_addressing(
    name: str,
    tensor: torch.Tensor,
    *,
    first_dim: int = 0,
) -> None:
    """Bound strides and the full byte span relative to the buffer base.

    Cache buffers are rebased per slot in i64, so only their inner dimensions
    contribute. The leading sequence dimension of raw gates and output is
    unused by the kernel and is also excluded.
    """
    strides = tensor.stride()[first_dim:]
    if any(stride < 0 or stride > _INT32_MAX for stride in strides):
        raise ValueError(f"`{name}` strides must fit non-negative int32 addressing.")
    # Include the last element's full width, including vector load/store tails.
    span_bytes = (
        1
        + sum(
            (size - 1) * stride
            for size, stride in zip(tensor.shape[first_dim:], strides)
        )
    ) * tensor.element_size()
    if span_bytes > _INT32_MAX:
        raise ValueError(f"`{name}` byte span must fit int32 buffer addressing.")


def _validate_kda_inputs(
    *,
    api_name: str,
    batch_source: str,
    device: torch.device,
    batch: int,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
    raw_beta: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    lower_bound: float | None,
    state: torch.Tensor,
    state_indices: torch.Tensor,
    output_gate: torch.Tensor,
    norm_weight: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[torch.Tensor, int]:
    """Validate operands shared by both explicit KDA specializations."""
    if not is_flydsl_kda_decode_supported(device):
        raise RuntimeError(f"`{api_name}` requires a gfx950 GPU.")
    if batch <= 0:
        raise ValueError(f"`{batch_source}` must have a non-empty batch dimension.")
    if state.ndim != 4:
        raise ValueError(f"`state` must have rank 4, got rank {state.ndim}.")
    num_heads = state.shape[1]
    if num_heads not in _SUPPORTED_HEADS:
        raise ValueError(
            f"`state` head count must be one of {_SUPPORTED_HEADS}, got {num_heads}."
        )
    conv_channels = 3 * num_heads * _DIM
    if batch * num_heads > _INT32_MAX:
        raise ValueError("The KDA launch grid must fit int32 addressing.")
    if lower_bound is None:
        raise ValueError("This specialization requires the KDA lower-bound gate.")

    _check_same_device(
        (
            ("x", x),
            ("conv_weight", conv_weight),
            ("conv_state", conv_state),
            ("raw_beta", raw_beta),
            ("A_log", A_log),
            ("dt_bias", dt_bias),
            ("state", state),
            ("state_indices", state_indices),
            ("output_gate", output_gate),
            ("norm_weight", norm_weight),
        ),
        device,
    )
    if conv_bias is not None:
        _check_same_device((("conv_bias", conv_bias),), device)
    _check_tensor(
        "x",
        x,
        shape=(batch, conv_channels),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "conv_weight",
        conv_weight,
        shape=(conv_channels, _CONV_WIDTH),
        dtype=torch.float32,
        device=device,
    )
    if conv_bias is not None:
        _check_tensor(
            "conv_bias",
            conv_bias,
            shape=(conv_channels,),
            dtype=torch.float32,
            device=device,
            inner_strides=(1,),
        )
    if conv_state.ndim != 3 or conv_state.shape[1:] != (
        conv_channels,
        _CONV_WIDTH - 1,
    ):
        raise ValueError(
            f"`conv_state` must have shape [cache, {conv_channels}, 3], "
            f"got {list(conv_state.shape)}."
        )
    if conv_state.dtype != torch.bfloat16:
        raise ValueError("`conv_state` must have dtype torch.bfloat16.")
    _check_conv_cache_slots(conv_state)
    if state.shape[1:] != (
        num_heads,
        _DIM,
        _DIM,
    ):
        raise ValueError(
            f"`state` must have shape [cache, {num_heads}, 128, 128], "
            f"got {list(state.shape)}."
        )
    if state.dtype != torch.float32:
        raise ValueError("`state` must have dtype torch.float32.")
    _check_dense_cache_slots("state", state, ((_DIM * _DIM, _DIM, 1),))
    _check_tensor(
        "raw_beta",
        raw_beta,
        shape=(1, batch, num_heads),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "A_log",
        A_log,
        shape=(num_heads,),
        dtype=torch.float32,
        device=device,
        inner_strides=(1,),
    )
    _check_tensor(
        "dt_bias",
        dt_bias,
        shape=(num_heads * _DIM,),
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
        shape=(batch, num_heads, _DIM),
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

    for name, tensor, first_dim in (
        ("x", x, 0),
        ("conv_weight", conv_weight, 0),
        ("conv_state", conv_state, 1),
        ("raw_beta", raw_beta, 1),
        ("A_log", A_log, 0),
        ("dt_bias", dt_bias, 0),
        ("state", state, 1),
        ("state_indices", state_indices, 0),
        ("output_gate", output_gate, 0),
        ("norm_weight", norm_weight, 0),
    ):
        _check_i32_addressing(name, tensor, first_dim=first_dim)
    if conv_bias is not None:
        _check_i32_addressing("conv_bias", conv_bias)

    if out is None:
        out = torch.empty(
            (1, batch, num_heads, _DIM),
            dtype=torch.bfloat16,
            device=device,
        )
    _check_same_device((("out", out),), device)
    _check_tensor(
        "out",
        out,
        shape=(1, batch, num_heads, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_i32_addressing("out", out, first_dim=1)
    # The final head need not include trailing padding in the batch stride.
    head_span = (num_heads - 1) * out.stride(2) + _DIM
    if out.stride(2) < _DIM or (batch > 1 and out.stride(1) < head_span):
        raise ValueError("`out` must have non-overlapping batch and head rows.")
    return out, num_heads


def flydsl_kda_decode(
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
    """Fuse width-4 Q/K/V convolution, FP32 KDA updates, and BF16 output gating.

    Requires gfx950, a supported head count, and 128-dim state. Check
    :func:`is_flydsl_kda_decode_supported` before dispatch.
    Strides and byte spans must fit non-negative int32 buffer addressing,
    except cache slot strides, which are rebased in int64.
    Both caches require dense inner dimensions and non-overlapping slots;
    padding between slots is supported. Output batch and head rows must not
    overlap, but may also be padded.
    Uses RMSNorm/sigmoid gating. Cache slot zero is reserved: non-positive
    ``state_indices`` produce zero output and leave both caches unchanged.
    Indices at or above ``min(state.shape[0], conv_state.shape[0])`` are
    treated the same way, so out-of-range slots are never accessed.

    Every positive in-range entry in ``state_indices`` must be unique within
    the batch. Duplicate live slots cause unsynchronized concurrent updates of
    the same recurrent and convolution cache rows and have undefined behavior.
    Padding entries may repeat.
    """
    if x.ndim != 2:
        raise ValueError(f"`x` must have rank 2, got rank {x.ndim}.")
    if not x.is_cuda:
        raise ValueError("`x` must be a CUDA tensor.")
    device = x.device
    batch = x.shape[0]
    out, num_heads = _validate_kda_inputs(
        api_name="flydsl_kda_decode",
        batch_source="x",
        device=device,
        batch=batch,
        x=x,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        conv_state=conv_state,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        state=state,
        state_indices=state_indices,
        output_gate=output_gate,
        norm_weight=norm_weight,
        out=out,
    )
    _check_same_device((("raw_g", raw_g),), device)
    _check_tensor(
        "raw_g",
        raw_g,
        shape=(1, batch, num_heads, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(_DIM, 1),
    )
    _check_i32_addressing("raw_g", raw_g, first_dim=1)

    executable = create_kda_decode_kernel(
        num_heads,
        float(norm_eps),
        float(lower_bound),
        fuse_f_b=False,
        has_conv_bias=conv_bias is not None,
    )
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        _run_compiled(
            executable,
            x,
            conv_weight,
            conv_bias if conv_bias is not None else A_log,
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
            _num_cache_slots(state, conv_state),
            x.stride(0),
            conv_weight.stride(0),
            conv_weight.stride(1),
            0 if conv_bias is None else conv_bias.stride(0),
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


def flydsl_kda_decode_with_f_b(
    f_a: torch.Tensor,
    f_b_weight: torch.Tensor,
    x: torch.Tensor,
    conv_weight: torch.Tensor,
    conv_bias: torch.Tensor | None,
    conv_state: torch.Tensor,
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
    """Run :func:`flydsl_kda_decode` with a fused head-local ``f_b`` projection.

    Projects ``f_a`` with ``f_b_weight`` in FP32, then rounds once to BF16
    before the lower-bound decay gate, without storing raw-g in global memory.
    Uses the same int32 stride/span limits as :func:`flydsl_kda_decode`.
    ``state_indices`` follows the same bounded, unique-live-slot contract as
    :func:`flydsl_kda_decode`.
    """
    if f_a.ndim != 2:
        raise ValueError(f"`f_a` must have rank 2, got rank {f_a.ndim}.")
    if not f_a.is_cuda:
        raise ValueError("`f_a` must be a CUDA tensor.")
    device = f_a.device
    batch = f_a.shape[0]
    _check_same_device((("f_b_weight", f_b_weight),), device)
    _check_tensor(
        "f_a",
        f_a,
        shape=(batch, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(1,),
    )
    _check_i32_addressing("f_a", f_a)
    out, num_heads = _validate_kda_inputs(
        api_name="flydsl_kda_decode_with_f_b",
        batch_source="f_a",
        device=device,
        batch=batch,
        x=x,
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        conv_state=conv_state,
        raw_beta=raw_beta,
        A_log=A_log,
        dt_bias=dt_bias,
        lower_bound=lower_bound,
        state=state,
        state_indices=state_indices,
        output_gate=output_gate,
        norm_weight=norm_weight,
        out=out,
    )
    _check_tensor(
        "f_b_weight",
        f_b_weight,
        shape=(num_heads, _DIM, _DIM),
        dtype=torch.bfloat16,
        device=device,
        inner_strides=(_DIM, 1),
    )
    _check_i32_addressing("f_b_weight", f_b_weight)

    executable = create_kda_decode_kernel(
        num_heads,
        float(norm_eps),
        float(lower_bound),
        fuse_f_b=True,
        has_conv_bias=conv_bias is not None,
    )
    with torch.cuda.device(device):
        stream = torch.cuda.current_stream(device)
        _run_compiled(
            executable,
            f_a,
            f_b_weight,
            x,
            conv_weight,
            conv_bias if conv_bias is not None else A_log,
            conv_state,
            raw_beta,
            A_log,
            dt_bias,
            state,
            state_indices,
            output_gate,
            norm_weight,
            out,
            batch,
            _num_cache_slots(state, conv_state),
            f_a.stride(0),
            f_b_weight.stride(0),
            f_b_weight.stride(1),
            x.stride(0),
            conv_weight.stride(0),
            conv_weight.stride(1),
            0 if conv_bias is None else conv_bias.stride(0),
            conv_state.stride(0),
            conv_state.stride(1),
            conv_state.stride(2),
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
    "flydsl_kda_decode",
    "flydsl_kda_decode_with_f_b",
    "is_flydsl_kda_decode_supported",
]
