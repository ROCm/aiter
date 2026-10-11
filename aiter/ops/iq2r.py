# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compiled gfx950 operations for native-basis IQ2R."""

from __future__ import annotations

import torch
from torch import Tensor

from ..jit.core import compile_ops
from .iq2r_format import IQ2RMetadata, iq2r_validate_expert_weights


@compile_ops("module_iq2r_moe", fc_name="iq2r_encode_out", develop=True)
def _iq2r_encode_out(
    weight: Tensor,
    importance: Tensor,
    codebook: Tensor,
    indices: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    scale_delta_overflow: Tensor,
    valid_k: int,
    exponent_radius: int,
    codebook_max: float,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_materialize_out", develop=True)
def _iq2r_materialize_out(
    data: Tensor,
    auxiliary: Tensor,
    output: Tensor,
    logical_n: int,
    logical_k: int,
) -> None: ...


# GLM-5.3 packed-layout MoE stages (TP4/TP8, 257 experts incl. the fused
# shared expert, top-9, hidden 6144).  Shapes are validated natively; see
# ``aiter.iq2r_glm53`` for the orchestration and tuned dispatch.


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_m1_route_quant_out", develop=True)
def iq2r_glm53_m1_route_quant_out(
    input: Tensor,
    topk_ids: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_sort_quant_out", develop=True)
def iq2r_glm53_sort_quant_out(
    input: Tensor,
    topk_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    gate_tasks: Tensor,
    gate_task_count: Tensor,
) -> None: ...


@compile_ops(
    "module_iq2r_moe", fc_name="iq2r_glm53_prefill_sort_quant_out", develop=True
)
def iq2r_glm53_prefill_sort_quant_out(
    input: Tensor,
    topk_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    scratch: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_route_reduce_out", develop=True)
def iq2r_glm53_route_reduce_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    output: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_gate_m1_out", develop=True)
def iq2r_glm53_gate_m1_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    output_scales: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_gate_out", develop=True)
def iq2r_glm53_gate_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    gather: Tensor,
    output: Tensor,
    output_scales: Tensor,
    kernel: int,
    grid_multiplier: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_down_out", develop=True)
def iq2r_glm53_down_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    kernel: int,
    grid_multiplier: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_down_route9_out", develop=True)
def iq2r_glm53_down_route9_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    expert_ids: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_down_reduce_out", develop=True)
def iq2r_glm53_down_reduce_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    route_output: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
    chunks: int,
) -> None: ...


def _validate_gpu_weights(
    data: Tensor, auxiliary: Tensor, metadata: IQ2RMetadata
) -> None:
    # Reserved-zero verification belongs at checkpoint-load time.  Reading a
    # device tensor with ``.item()`` here would add a synchronization and make
    # an otherwise graph-safe out-op impossible to capture.
    iq2r_validate_expert_weights(data, auxiliary, metadata, verify_reserved_zero=False)
    if data.device.type != "cuda":
        raise ValueError("IQ2R compiled operations require GPU tensors")


@torch.no_grad()
def iq2r_encode_device(
    weight: Tensor,
    importance: Tensor,
    codebook: Tensor,
    *,
    exponent_radius: int = 0,
) -> tuple[Tensor, Tensor]:
    """Encode one matrix with the gfx950 implementation."""

    if weight.device.type != "cuda":
        raise ValueError("IQ2R device encoding requires a GPU weight tensor")
    if weight.dtype != torch.float32 or weight.ndim != 2 or not weight.is_contiguous():
        raise ValueError("weight must be contiguous GPU float32 [N,K]")
    n, valid_k = weight.shape
    if n % 16 or valid_k % 32:
        raise ValueError("IQ2R requires N%16==0 and K%32==0")
    if importance.dtype != torch.float32 or tuple(importance.shape) != (valid_k,):
        raise ValueError(f"importance must be float32 [{valid_k}]")
    if tuple(codebook.shape) != (512, 8) or codebook.dtype != torch.float32:
        raise ValueError("codebook must be float32 [512,8]")
    if importance.device != weight.device or codebook.device != weight.device:
        raise ValueError("weight, importance, and codebook must share a GPU")
    if not importance.is_contiguous() or not codebook.is_contiguous():
        raise ValueError("importance and codebook must be contiguous")
    if not 0 <= exponent_radius <= 16:
        raise ValueError("exponent_radius must be in [0,16]")

    from .iq2r_encoder import iq2r_reserve_zero_codeword
    from .iq2r_format import (
        IQ2R_CODEBOOK_BYTES,
        iq2r_packed_sizes,
        iq2r_physical_n_blocks,
    )

    codebook = iq2r_reserve_zero_codeword(codebook).contiguous()
    storage_k = ((valid_k + 127) // 128) * 128
    if storage_k != valid_k:
        weight = torch.nn.functional.pad(weight, (0, storage_k - valid_k)).contiguous()
        importance = torch.nn.functional.pad(
            importance, (0, storage_k - valid_k)
        ).contiguous()
    data_bytes, auxiliary_bytes = iq2r_packed_sizes(n, valid_k)
    physical_n_blocks = iq2r_physical_n_blocks(n)
    tiles = physical_n_blocks * (storage_k // 128)
    indices = torch.zeros(tiles * 64 * 4, dtype=torch.int16, device=weight.device)
    scales = torch.full((tiles * 64,), 127, dtype=torch.uint8, device=weight.device)
    data = torch.zeros(data_bytes, dtype=torch.uint8, device=weight.device)
    auxiliary = torch.full(
        (auxiliary_bytes,), 127, dtype=torch.uint8, device=weight.device
    )
    auxiliary[:IQ2R_CODEBOOK_BYTES].copy_(
        codebook.to(torch.float8_e4m3fn).view(torch.uint8).reshape(-1)
    )
    overflow = torch.zeros(1, dtype=torch.int32, device=weight.device)
    _iq2r_encode_out(
        weight,
        importance,
        codebook,
        indices,
        scales,
        data,
        auxiliary,
        overflow,
        valid_k,
        exponent_radius,
        float(codebook.max().item()),
    )
    if overflow.item() != 0:
        raise ValueError("IQ2R scale exponent range exceeds the 4-bit delta format")
    return data, auxiliary


def iq2r_materialize_out(
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
    output: Tensor,
) -> None:
    """Materialize expert 0 into caller-owned FP32 ``[N,K]`` storage."""

    _validate_gpu_weights(data, auxiliary, metadata)
    if output.dtype != torch.float32 or tuple(output.shape) != (
        metadata.logical_n,
        metadata.logical_k,
    ):
        raise ValueError(
            f"output must be float32 [{metadata.logical_n},{metadata.logical_k}]"
        )
    if output.device != data.device or not output.is_contiguous():
        raise ValueError("output must be contiguous and on the IQ2R weight device")
    _iq2r_materialize_out(
        data, auxiliary, output, metadata.logical_n, metadata.logical_k
    )


def iq2r_materialize_device(
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
) -> Tensor:
    output = torch.empty(
        (metadata.logical_n, metadata.logical_k),
        dtype=torch.float32,
        device=data.device,
    )
    iq2r_materialize_out(data, auxiliary, metadata, output)
    return output


__all__ = [
    "iq2r_encode_device",
    "iq2r_glm53_down_out",
    "iq2r_glm53_down_reduce_out",
    "iq2r_glm53_down_route9_out",
    "iq2r_glm53_gate_m1_out",
    "iq2r_glm53_gate_out",
    "iq2r_glm53_m1_route_quant_out",
    "iq2r_glm53_prefill_sort_quant_out",
    "iq2r_glm53_route_reduce_out",
    "iq2r_glm53_sort_quant_out",
    "iq2r_materialize_device",
    "iq2r_materialize_out",
]
