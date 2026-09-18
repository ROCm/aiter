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


@compile_ops("module_iq2r_moe", fc_name="iq2r_gemm_out", develop=True)
def _iq2r_gemm_out(
    activations: Tensor,
    activation_scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    bias: Tensor | None,
    output: Tensor,
    logical_n: int,
    logical_k: int,
    tile_n: int,
    expert_index: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_task_gemm_out", develop=True)
def _iq2r_task_gemm_out(
    activations: Tensor,
    activation_scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    bias: Tensor | None,
    output: Tensor,
    logical_n: int,
    logical_k: int,
    tile_n: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_route_sort_tasks_out", develop=True)
def _iq2r_route_sort_tasks_out(
    expert_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    expert_count: int,
    task_rows: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_route_gather_indexed_out", develop=True)
def _iq2r_route_gather_indexed_out(
    input: Tensor,
    gather_indices: Tensor,
    output: Tensor,
    topk: int,
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


@compile_ops("module_iq2r_moe", fc_name="iq2r_swiglu_out", develop=True)
def _iq2r_swiglu_out(gate_up: Tensor, output: Tensor) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_swiglu_quant_out", develop=True)
def _iq2r_swiglu_quant_out(
    gate_up: Tensor,
    output: Tensor,
    scales: Tensor,
    activated: Tensor | None,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_route_reduce_indexed_out", develop=True)
def _iq2r_route_reduce_indexed_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    output: Tensor,
    topk: int,
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


def iq2r_gemm_out(
    activations: Tensor,
    activation_scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
    output: Tensor,
    *,
    tile_n: int,
    expert_index: int = 0,
    bias: Tensor | None = None,
) -> None:
    """Run the native scaled-FP8 IQ2R GEMM into caller-owned BF16 output."""

    _validate_gpu_weights(data, auxiliary, metadata)
    if activations.dtype != torch.float8_e4m3fn:
        raise TypeError("activations must be float8_e4m3fn")
    if activations.ndim != 2 or activations.shape[1] != metadata.logical_k:
        raise ValueError(f"activations must have shape [M,{metadata.logical_k}]")
    expected_scales = (activations.shape[0], metadata.logical_k // 32)
    if (
        activation_scales.dtype != torch.uint8
        or tuple(activation_scales.shape) != expected_scales
    ):
        raise ValueError(f"activation_scales must be uint8 {expected_scales}")
    expected_output = (activations.shape[0], metadata.logical_n)
    if output.dtype != torch.bfloat16 or tuple(output.shape) != expected_output:
        raise ValueError(f"output must be bfloat16 {expected_output}")
    tensors = (activations, activation_scales, data, auxiliary, output)
    if any(tensor.device != data.device for tensor in tensors):
        raise ValueError("all IQ2R GEMM tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all IQ2R GEMM tensors must be contiguous")
    if tile_n not in (64, 128) or metadata.logical_n % tile_n:
        raise ValueError("tile_n must be 64 or 128 and divide logical_n")
    if bias is not None:
        if bias.dtype != torch.bfloat16 or tuple(bias.shape) != (
            data.shape[0],
            metadata.logical_n,
        ):
            raise ValueError(
                f"bias must be bfloat16 [{data.shape[0]},{metadata.logical_n}]"
            )
        if bias.device != data.device or not bias.is_contiguous():
            raise ValueError("bias must be contiguous and on the IQ2R weight device")
    _iq2r_gemm_out(
        activations,
        activation_scales,
        data,
        auxiliary,
        bias,
        output,
        metadata.logical_n,
        metadata.logical_k,
        tile_n,
        expert_index,
    )


def iq2r_gemm(
    activations: Tensor,
    activation_scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    metadata: IQ2RMetadata,
    *,
    tile_n: int,
    expert_index: int = 0,
    bias: Tensor | None = None,
) -> Tensor:
    output = torch.empty(
        (activations.shape[0], metadata.logical_n),
        dtype=torch.bfloat16,
        device=activations.device,
    )
    iq2r_gemm_out(
        activations,
        activation_scales,
        data,
        auxiliary,
        metadata,
        output,
        tile_n=tile_n,
        expert_index=expert_index,
        bias=bias,
    )
    return output


def iq2r_task_gemm_out(
    activations: Tensor,
    activation_scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    metadata: IQ2RMetadata,
    output: Tensor,
    *,
    tile_n: int,
    bias: Tensor | None = None,
) -> None:
    """Execute expert-homogeneous ``[start,height,expert]`` routed tasks."""

    _validate_gpu_weights(data, auxiliary, metadata)
    if activations.dtype != torch.float8_e4m3fn or activations.ndim != 2:
        raise ValueError("activations must be a two-dimensional float8_e4m3fn tensor")
    if activations.shape[1] != metadata.logical_k:
        raise ValueError(f"activations must have K={metadata.logical_k}")
    expected_scales = (activations.shape[0], metadata.logical_k // 32)
    if (
        activation_scales.dtype != torch.uint8
        or tuple(activation_scales.shape) != expected_scales
    ):
        raise ValueError(f"activation_scales must be uint8 {expected_scales}")
    if tasks.dtype != torch.int32 or tasks.ndim != 2 or tasks.shape[1] != 3:
        raise ValueError("tasks must be int32 [capacity,3]")
    if task_count.dtype != torch.int32 or tuple(task_count.shape) != (1,):
        raise ValueError("task_count must be int32 [1]")
    if output.dtype != torch.bfloat16 or tuple(output.shape) != (
        activations.shape[0],
        metadata.logical_n,
    ):
        raise ValueError(
            f"output must be bfloat16 [{activations.shape[0]},{metadata.logical_n}]"
        )
    tensors = (
        activations,
        activation_scales,
        data,
        auxiliary,
        tasks,
        task_count,
        output,
    )
    if any(tensor.device != data.device for tensor in tensors):
        raise ValueError("all IQ2R task GEMM tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all IQ2R task GEMM tensors must be contiguous")
    if tile_n not in (64, 128) or metadata.logical_n % tile_n:
        raise ValueError("tile_n must be 64 or 128 and divide logical_n")
    if bias is not None:
        if bias.dtype != torch.bfloat16 or tuple(bias.shape) != (
            data.shape[0],
            metadata.logical_n,
        ):
            raise ValueError(
                f"bias must be bfloat16 [{data.shape[0]},{metadata.logical_n}]"
            )
        if bias.device != data.device or not bias.is_contiguous():
            raise ValueError("bias must be contiguous and on the IQ2R weight device")
    _iq2r_task_gemm_out(
        activations,
        activation_scales,
        data,
        auxiliary,
        tasks,
        task_count,
        bias,
        output,
        metadata.logical_n,
        metadata.logical_k,
        tile_n,
    )


def iq2r_task_gemm(
    activations: Tensor,
    activation_scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    metadata: IQ2RMetadata,
    *,
    tile_n: int,
    bias: Tensor | None = None,
) -> Tensor:
    output = torch.empty(
        (activations.shape[0], metadata.logical_n),
        dtype=torch.bfloat16,
        device=activations.device,
    )
    iq2r_task_gemm_out(
        activations,
        activation_scales,
        data,
        auxiliary,
        tasks,
        task_count,
        metadata,
        output,
        tile_n=tile_n,
        bias=bias,
    )
    return output


def iq2r_task_capacity(routes: int, expert_count: int, task_rows: int) -> int:
    """Worst-case task capacity for stable expert runs plus invalid IDs."""

    for name, value in (("routes", routes), ("expert_count", expert_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if expert_count > 128:
        raise ValueError("initial IQ2R routing supports at most 128 experts")
    if task_rows not in (16, 32, 64):
        raise ValueError("task_rows must be 16, 32, or 64")
    return (routes + task_rows - 1) // task_rows + min(routes, expert_count + 1)


def iq2r_route_sort_tasks_out(
    expert_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    *,
    expert_count: int,
    task_rows: int,
) -> None:
    """Stable-sort routes by expert and build bounded GEMM tasks."""

    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 1:
        raise ValueError("expert_ids must be int32 [routes]")
    routes = expert_ids.numel()
    if not 0 < routes <= 4096:
        raise ValueError("IQ2R route sorting supports 1..4096 routed rows")
    expected_vectors = (sorted_expert_ids, gather_indices, scatter_indices)
    if any(
        t.dtype != torch.int32 or tuple(t.shape) != (routes,) for t in expected_vectors
    ):
        raise ValueError("sorted/gather/scatter tensors must be int32 [routes]")
    if tasks.dtype != torch.int32 or tasks.ndim != 2 or tasks.shape[1] != 3:
        raise ValueError("tasks must be int32 [capacity,3]")
    if task_count.dtype != torch.int32 or tuple(task_count.shape) != (1,):
        raise ValueError("task_count must be int32 [1]")
    required = iq2r_task_capacity(routes, expert_count, task_rows)
    if tasks.shape[0] < required:
        raise ValueError(
            f"tasks capacity {tasks.shape[0]} is smaller than required {required}"
        )
    tensors = (
        expert_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
    )
    if any(t.device != expert_ids.device for t in tensors):
        raise ValueError("all IQ2R routing tensors must share a device")
    if expert_ids.device.type != "cuda":
        raise ValueError("IQ2R route sorting requires GPU tensors")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all IQ2R routing tensors must be contiguous")
    _iq2r_route_sort_tasks_out(
        expert_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        expert_count,
        task_rows,
    )


def iq2r_route_gather_indexed_out(
    input: Tensor,
    gather_indices: Tensor,
    output: Tensor,
    *,
    topk: int,
) -> None:
    """Gather token rows into stable expert-sorted route order."""

    if input.dtype != torch.bfloat16 or input.ndim != 2:
        raise ValueError("input must be BF16 [tokens,hidden]")
    if gather_indices.dtype != torch.int32 or gather_indices.ndim != 1:
        raise ValueError("gather_indices must be int32 [routes]")
    if topk <= 0 or gather_indices.numel() != input.shape[0] * topk:
        raise ValueError("gather_indices length must equal tokens*topk")
    expected = (gather_indices.numel(), input.shape[1])
    if output.dtype != torch.bfloat16 or tuple(output.shape) != expected:
        raise ValueError(f"output must be BF16 {expected}")
    if any(t.device != input.device for t in (gather_indices, output)):
        raise ValueError("IQ2R route gather tensors must share a device")
    if input.stride(-1) != 1 or input.stride(0) < input.shape[1]:
        raise ValueError(
            "IQ2R route gather input must have contiguous columns and "
            "non-overlapping rows"
        )
    if not gather_indices.is_contiguous() or not output.is_contiguous():
        raise ValueError("IQ2R route gather indices/output must be contiguous")
    _iq2r_route_gather_indexed_out(input, gather_indices, output, topk)


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
    expected_scales = (gather_indices.numel(), input.shape[1] // 32)
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != expected_output:
        raise ValueError(f"output must be float8_e4m3fn {expected_output}")
    if scales.dtype != torch.uint8 or tuple(scales.shape) != expected_scales:
        raise ValueError(f"scales must be uint8 {expected_scales}")
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
    if scales.dtype != torch.uint8 or tuple(scales.shape) != (
        routes,
        input.shape[1] // 32,
    ):
        raise ValueError("scales must be uint8 [routes,hidden/32]")
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


def iq2r_swiglu_out(gate_up: Tensor, output: Tensor) -> None:
    """Apply GPT-OSS clipped SwiGLU to interleaved gate/up columns."""

    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2:
        raise ValueError("gate_up must be BF16 [rows,2*intermediate]")
    expected = (gate_up.shape[0], gate_up.shape[1] // 2)
    if (
        gate_up.shape[1] % 2
        or output.dtype != torch.bfloat16
        or tuple(output.shape) != expected
    ):
        raise ValueError(f"output must be BF16 {expected}")
    if (
        output.device != gate_up.device
        or not gate_up.is_contiguous()
        or not output.is_contiguous()
    ):
        raise ValueError("IQ2R SwiGLU tensors must be contiguous on one device")
    _iq2r_swiglu_out(gate_up, output)


def iq2r_swiglu_quant_out(
    gate_up: Tensor,
    output: Tensor,
    scales: Tensor,
    *,
    activated: Tensor | None = None,
) -> None:
    """Apply GPT-OSS clipped SwiGLU and emit MXFP8/E8M0 in one pass."""

    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2:
        raise ValueError("gate_up must be BF16 [rows,2*intermediate]")
    if gate_up.shape[1] % 64:
        raise ValueError("gate_up width must be divisible by 64")
    expected_output = (gate_up.shape[0], gate_up.shape[1] // 2)
    expected_scales = (gate_up.shape[0], gate_up.shape[1] // 64)
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != expected_output:
        raise ValueError(f"output must be float8_e4m3fn {expected_output}")
    if scales.dtype != torch.uint8 or tuple(scales.shape) != expected_scales:
        raise ValueError(f"scales must be uint8 {expected_scales}")
    tensors = [gate_up, output, scales]
    if activated is not None:
        if (
            activated.dtype != torch.bfloat16
            or tuple(activated.shape) != expected_output
        ):
            raise ValueError(f"activated must be BF16 {expected_output}")
        tensors.append(activated)
    if any(t.device != gate_up.device for t in tensors):
        raise ValueError("IQ2R fused SwiGLU/quant tensors must share a device")
    if gate_up.device.type != "cuda" or any(not t.is_contiguous() for t in tensors):
        raise ValueError(
            "IQ2R fused SwiGLU/quant tensors must be contiguous on one GPU"
        )
    _iq2r_swiglu_quant_out(gate_up, output, scales, activated)


def iq2r_route_reduce_indexed_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    output: Tensor,
    *,
    topk: int,
) -> None:
    """Apply FP32 route weights and reduce sorted routes to BF16 tokens."""

    if route_output.dtype != torch.bfloat16 or route_output.ndim != 2:
        raise ValueError("route_output must be BF16 [routes,hidden]")
    if route_weights.dtype != torch.float32 or route_weights.ndim != 2:
        raise ValueError("route_weights must be FP32 [tokens,topk]")
    if route_weights.shape[1] != topk:
        raise ValueError("route_weights top-k dimension does not match topk")
    routes = route_weights.shape[0] * topk
    if route_output.shape[0] != routes:
        raise ValueError("route_output row count must equal tokens*topk")
    if scatter_indices.dtype != torch.int32 or tuple(scatter_indices.shape) != (
        routes,
    ):
        raise ValueError("scatter_indices must be int32 [routes]")
    expected = (route_weights.shape[0], route_output.shape[1])
    if output.dtype != torch.bfloat16 or tuple(output.shape) != expected:
        raise ValueError(f"output must be BF16 {expected}")
    tensors = (route_output, route_weights, scatter_indices, output)
    if any(t.device != route_output.device for t in tensors):
        raise ValueError("IQ2R route reduction tensors must share a device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("IQ2R route reduction tensors must be contiguous")
    _iq2r_route_reduce_indexed_out(
        route_output, route_weights, scatter_indices, output, topk
    )


__all__ = [
    "iq2r_encode_device",
    "iq2r_gemm",
    "iq2r_gemm_out",
    "iq2r_materialize_device",
    "iq2r_materialize_out",
    "iq2r_route_gather_indexed_out",
    "iq2r_route_gather_quant_out",
    "iq2r_route_direct_gather_quant_out",
    "iq2r_route_reduce_indexed_out",
    "iq2r_route_sort_tasks_out",
    "iq2r_swiglu_out",
    "iq2r_swiglu_quant_out",
    "iq2r_task_capacity",
    "iq2r_task_gemm",
    "iq2r_task_gemm_out",
]
