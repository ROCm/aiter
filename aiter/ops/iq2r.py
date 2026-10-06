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
    expert_index: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_route_gather_quant_out", develop=True)
def _iq2r_route_gather_quant_out(
    input: Tensor,
    gather_indices: Tensor,
    output: Tensor,
    scales: Tensor,
    topk: int,
) -> None: ...


@compile_ops(
    "module_iq2r_moe", fc_name="iq2r_route_direct_gather_quant_out", develop=True
)
def _iq2r_route_direct_gather_quant_out(
    input: Tensor,
    expert_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    topk: int,
    expert_count: int,
) -> None: ...


# GLM-5.3 packed-layout MoE stages (TP4/TP8, 257 experts incl. the fused
# shared expert, top-9, hidden 6144).  Shapes are validated natively; see
# ``aiter.iq2r_glm53`` for the orchestration and tuned dispatch.


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_sort_quant_out", develop=True)
def iq2r_glm53_sort_quant_out(
    input: Tensor,
    topk_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    gate_tasks: Tensor,
    gate_task_count: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_sort_out", develop=True)
def iq2r_glm53_sort_out(
    topk_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
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


def _valid_activation_scale_shape(
    scales: Tensor, rows: int, groups_per_row: int
) -> bool:
    row_major = tuple(scales.shape) == (rows, groups_per_row)
    tile16 = (
        scales.ndim == 4
        and scales.shape[0] == (groups_per_row + 3) // 4
        and scales.shape[1] >= (rows + 15) // 16
        and scales.shape[2:] == (4, 16)
    )
    return scales.dtype == torch.uint8 and (row_major or tile16)


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
    *,
    expert_index: int = 0,
) -> None:
    """Materialize one expert into caller-owned FP32 ``[N,K]`` storage."""

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
    if not 0 <= expert_index < data.shape[0]:
        raise ValueError("expert_index is out of range")
    _iq2r_materialize_out(
        data,
        auxiliary,
        output,
        metadata.logical_n,
        metadata.logical_k,
        expert_index,
    )


def iq2r_materialize_device(
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
    *,
    expert_index: int = 0,
) -> Tensor:
    output = torch.empty(
        (metadata.logical_n, metadata.logical_k),
        dtype=torch.float32,
        device=data.device,
    )
    iq2r_materialize_out(data, auxiliary, metadata, output, expert_index=expert_index)
    return output


def iq2r_route_gather_quant_out(
    input: Tensor,
    gather_indices: Tensor,
    output: Tensor,
    scales: Tensor,
    *,
    topk: int,
) -> None:
    """Gather routes and emit row-major MXFP8/E8M0 blocks in one pass."""

    if input.dtype != torch.bfloat16 or input.ndim != 2:
        raise ValueError("input must be BF16 [tokens,hidden]")
    if gather_indices.dtype != torch.int32 or gather_indices.ndim != 1:
        raise ValueError("gather_indices must be int32 [routes]")
    if topk <= 0 or gather_indices.numel() != input.shape[0] * topk:
        raise ValueError("gather_indices length must equal tokens*topk")
    if input.shape[1] % 32:
        raise ValueError("IQ2R MXFP8 quantization requires hidden divisible by 32")
    expected_output = (gather_indices.numel(), input.shape[1])
    scale_rows = gather_indices.numel()
    scale_groups = input.shape[1] // 32
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != expected_output:
        raise ValueError(f"output must be float8_e4m3fn {expected_output}")
    if not _valid_activation_scale_shape(scales, scale_rows, scale_groups):
        raise ValueError(
            f"scales must be row-major uint8 [{scale_rows},{scale_groups}] "
            "or tile16 uint8 "
            f"[{(scale_groups + 3) // 4},{(scale_rows + 15) // 16},4,16]"
        )
    tensors = (input, gather_indices, output, scales)
    if any(t.device != input.device for t in tensors):
        raise ValueError("IQ2R fused gather/quant tensors must share a device")
    if input.device.type != "cuda":
        raise ValueError("IQ2R fused gather/quant tensors must be on one GPU")
    if input.stride(-1) != 1 or input.stride(0) < input.shape[1]:
        raise ValueError(
            "IQ2R fused gather/quant input must have contiguous columns and "
            "non-overlapping rows"
        )
    if any(not t.is_contiguous() for t in (gather_indices, output, scales)):
        raise ValueError(
            "IQ2R fused gather/quant indices and outputs must be contiguous"
        )
    _iq2r_route_gather_quant_out(input, gather_indices, output, scales, topk)


def iq2r_route_direct_gather_quant_out(
    input: Tensor,
    expert_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    *,
    topk: int,
    expert_count: int,
) -> None:
    """Fuse unsorted one-row task construction with low-M gather/quantization."""

    routes = expert_ids.numel()
    if not 0 < routes <= 16 or routes % topk:
        raise ValueError("direct IQ2R routing requires 1..16 complete top-k rows")
    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 1:
        raise ValueError("expert_ids must be int32 [routes]")
    vectors = (sorted_expert_ids, gather_indices, scatter_indices)
    if any(t.dtype != torch.int32 or tuple(t.shape) != (routes,) for t in vectors):
        raise ValueError("sorted/gather/scatter tensors must be int32 [routes]")
    if tasks.dtype != torch.int32 or tasks.ndim != 2 or tasks.shape[1] != 3:
        raise ValueError("tasks must be int32 [capacity,3]")
    if tasks.shape[0] < routes:
        raise ValueError("direct IQ2R routing requires at least one task per route")
    if task_count.dtype != torch.int32 or tuple(task_count.shape) != (1,):
        raise ValueError("task_count must be int32 [1]")
    if input.dtype != torch.bfloat16 or input.ndim != 2:
        raise ValueError("input must be BF16 [tokens,hidden]")
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != (
        routes,
        input.shape[1],
    ):
        raise ValueError("output must be FP8 [routes,hidden]")
    scale_groups = input.shape[1] // 32
    if not _valid_activation_scale_shape(scales, routes, scale_groups):
        raise ValueError(
            "scales must be row-major uint8 [routes,hidden/32] or "
            "tile16 uint8 [ceil(hidden/128),ceil(routes/16),4,16]"
        )
    tensors = (
        expert_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        output,
        scales,
    )
    if input.device.type != "cuda" or any(t.device != input.device for t in tensors):
        raise ValueError("all direct IQ2R routing tensors must share one GPU")
    if input.stride(-1) != 1 or input.stride(0) < input.shape[1]:
        raise ValueError("direct IQ2R input must have contiguous non-overlapping rows")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("direct IQ2R routing outputs must be contiguous")
    _iq2r_route_direct_gather_quant_out(
        input,
        expert_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        output,
        scales,
        topk,
        expert_count,
    )


__all__ = [
    "iq2r_encode_device",
    "iq2r_glm53_down_out",
    "iq2r_glm53_down_reduce_out",
    "iq2r_glm53_down_route9_out",
    "iq2r_glm53_gate_m1_out",
    "iq2r_glm53_gate_out",
    "iq2r_glm53_route_reduce_out",
    "iq2r_glm53_sort_out",
    "iq2r_glm53_sort_quant_out",
    "iq2r_materialize_device",
    "iq2r_materialize_out",
    "iq2r_route_direct_gather_quant_out",
    "iq2r_route_gather_quant_out",
]
