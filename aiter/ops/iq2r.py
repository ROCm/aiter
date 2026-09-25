# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Compiled gfx950 operations for native-basis IQ2R."""

from __future__ import annotations

import torch
from torch import Tensor

from ..jit.core import compile_ops
from .iq2r_format import IQ2RMetadata, iq2r_validate_expert_weights

# GLM-5.3 top-k=8 with ATOM's production 16K-token prefill budget. The route
# sorter is grid-stride and its caller owns the buffers, so this is a workspace
# memory contract rather than a launch-shape limit.
IQ2R_MAX_ROUTES = 131072


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


@compile_ops("module_iq2r_moe", fc_name="iq2r_task_gemm_indexed_out", develop=True)
def _iq2r_task_gemm_indexed_out(
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
    gather_indices: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_task_gemm_swiglu_quant_out", develop=True)
def _iq2r_task_gemm_swiglu_quant_out(
    activations: Tensor,
    activation_scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    bias: Tensor | None,
    output: Tensor,
    output_scales: Tensor,
    logical_n: int,
    logical_k: int,
    limit: float,
    alpha: float,
    up_offset: float,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_route_sort_tasks_out", develop=True)
def _iq2r_route_sort_tasks_out(
    expert_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    expert_map: Tensor | None,
    expert_count: int,
    expert_start: int,
    expert_stride: int,
    task_rows: int,
    drop_nonlocal_tasks: bool,
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


@compile_ops("module_iq2r_moe", fc_name="iq2r_route_scatter_quant_out", develop=True)
def _iq2r_route_scatter_quant_out(
    input: Tensor,
    scatter_indices: Tensor,
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
    expert_map: Tensor | None,
    topk: int,
    expert_count: int,
    expert_start: int,
    expert_stride: int,
    drop_nonlocal_routes: bool,
) -> None: ...


@compile_ops(
    "module_iq2r_moe",
    fc_name="iq2r_route_topk_direct_gather_quant_out",
    develop=True,
)
def _iq2r_route_topk_direct_gather_quant_out(
    input: Tensor,
    router_logits: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    renormalize: bool,
    router_bias: Tensor | None,
    biased_sigmoid: bool,
    routed_scaling_factor: float,
    expert_map: Tensor | None,
    expert_count: int,
    expert_start: int,
    expert_stride: int,
    drop_nonlocal_routes: bool,
) -> None: ...


@compile_ops(
    "module_iq2r_moe",
    fc_name="iq2r_route_topk_sort_gather_quant_out",
    develop=True,
)
def _iq2r_route_topk_sort_gather_quant_out(
    input: Tensor,
    router_logits: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    task_rows: int,
    renormalize: bool,
    router_bias: Tensor | None,
    biased_sigmoid: bool,
    routed_scaling_factor: float,
    expert_map: Tensor | None,
    expert_count: int,
    expert_start: int,
    expert_stride: int,
    drop_nonlocal_routes: bool,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_swiglu_out", develop=True)
def _iq2r_swiglu_out(
    gate_up: Tensor,
    output: Tensor,
    limit: float,
    alpha: float,
    up_offset: float,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_swiglu_quant_out", develop=True)
def _iq2r_swiglu_quant_out(
    gate_up: Tensor,
    output: Tensor,
    scales: Tensor,
    activated: Tensor | None,
    limit: float,
    alpha: float,
    up_offset: float,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_swiglu_quant_scatter_out", develop=True)
def _iq2r_swiglu_quant_scatter_out(
    gate_up: Tensor,
    scatter_indices: Tensor,
    output: Tensor,
    scales: Tensor,
    topk: int,
    activated: Tensor | None,
    limit: float,
    alpha: float,
    up_offset: float,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_route_reduce_indexed_out", develop=True)
def _iq2r_route_reduce_indexed_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    output: Tensor,
    topk: int,
) -> None: ...


@compile_ops(
    "module_iq2r_moe", fc_name="iq2r_route_reduce_add_indexed_out", develop=True
)
def _iq2r_route_reduce_add_indexed_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    shared_output: Tensor,
    output: Tensor,
    topk: int,
) -> None: ...


@compile_ops(
    "module_iq2r_moe",
    fc_name="iq2r_route_reduce_add_rmsnorm_indexed_out",
    develop=True,
)
def _iq2r_route_reduce_add_rmsnorm_indexed_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    output: Tensor,
    residual_out: Tensor,
    topk: int,
    epsilon: float,
    block_size: int,
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
    scale_rows = activations.shape[0]
    scale_groups = metadata.logical_k // 32
    if not _valid_activation_scale_shape(activation_scales, scale_rows, scale_groups):
        raise ValueError(
            "activation_scales must be row-major uint8 "
            f"[{scale_rows},{scale_groups}] or tile16 uint8 "
            f"[{(scale_groups + 3) // 4},{(scale_rows + 15) // 16},4,16]"
        )
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


def iq2r_task_gemm_indexed_out(
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
    gather_indices: Tensor,
    bias: Tensor | None = None,
) -> None:
    """E063 GLM TP8 gate with token-major FP8 and trusted sorter permutation."""

    _validate_gpu_weights(data, auxiliary, metadata)
    if activations.dtype != torch.float8_e4m3fn or activations.ndim != 2:
        raise ValueError("activations must be a two-dimensional float8_e4m3fn tensor")
    if activations.shape[1] != metadata.logical_k:
        raise ValueError(f"activations must have K={metadata.logical_k}")
    scale_rows = activations.shape[0]
    scale_groups = metadata.logical_k // 32
    if not _valid_activation_scale_shape(activation_scales, scale_rows, scale_groups):
        raise ValueError(
            "activation_scales must be row-major uint8 "
            f"[{scale_rows},{scale_groups}] or tile16 uint8 "
            f"[{(scale_groups + 3) // 4},{(scale_rows + 15) // 16},4,16]"
        )
    if tasks.dtype != torch.int32 or tasks.ndim != 2 or tasks.shape[1] != 3:
        raise ValueError("tasks must be int32 [capacity,3]")
    if task_count.dtype != torch.int32 or tuple(task_count.shape) != (1,):
        raise ValueError("task_count must be int32 [1]")
    if output.dtype != torch.bfloat16 or tuple(output.shape) != (
        activations.shape[0] * 9,
        metadata.logical_n,
    ):
        raise ValueError(
            f"output must be bfloat16 [{activations.shape[0] * 9},{metadata.logical_n}]"
        )
    if (
        gather_indices.dtype != torch.int32
        or gather_indices.ndim != 1
        or gather_indices.numel() != output.shape[0]
    ):
        raise ValueError("gather_indices must be int32 [routes]")
    tensors = (
        activations,
        activation_scales,
        data,
        auxiliary,
        tasks,
        task_count,
        output,
        gather_indices,
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
    _iq2r_task_gemm_indexed_out(
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
        gather_indices,
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


def iq2r_task_gemm_swiglu_quant_out(
    activations: Tensor,
    activation_scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    metadata: IQ2RMetadata,
    output: Tensor,
    output_scales: Tensor,
    *,
    bias: Tensor | None = None,
    limit: float = 0.0,
    alpha: float = 1.0,
    up_offset: float = 0.0,
) -> None:
    """Fuse routed IQ2R gate/up, standard SwiGLU, and row-major MXFP8 output."""

    _validate_gpu_weights(data, auxiliary, metadata)
    if metadata.logical_n % 64:
        raise ValueError("fused IQ2R gate/up requires logical_n divisible by 64")
    if activations.dtype != torch.float8_e4m3fn or activations.ndim != 2:
        raise ValueError("activations must be a two-dimensional float8_e4m3fn tensor")
    if activations.shape[1] != metadata.logical_k:
        raise ValueError(f"activations must have K={metadata.logical_k}")
    rows = activations.shape[0]
    input_scale_groups = metadata.logical_k // 32
    if not _valid_activation_scale_shape(activation_scales, rows, input_scale_groups):
        raise ValueError(
            "activation_scales must be row-major uint8 "
            f"[{rows},{input_scale_groups}] or tile16 uint8 "
            f"[{(input_scale_groups + 3) // 4},{(rows + 15) // 16},4,16]"
        )
    if tasks.dtype != torch.int32 or tasks.ndim != 2 or tasks.shape[1] != 3:
        raise ValueError("tasks must be int32 [capacity,3]")
    if task_count.dtype != torch.int32 or tuple(task_count.shape) != (1,):
        raise ValueError("task_count must be int32 [1]")
    intermediate = metadata.logical_n // 2
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != (
        rows,
        intermediate,
    ):
        raise ValueError(f"output must be float8_e4m3fn [{rows},{intermediate}]")
    if output_scales.dtype != torch.uint8 or tuple(output_scales.shape) != (
        rows,
        intermediate // 32,
    ):
        raise ValueError(
            f"output_scales must be row-major uint8 [{rows},{intermediate // 32}]"
        )
    tensors = (
        activations,
        activation_scales,
        data,
        auxiliary,
        tasks,
        task_count,
        output,
        output_scales,
    )
    if any(tensor.device != data.device for tensor in tensors):
        raise ValueError("all fused IQ2R gate/up tensors must be on the same device")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("all fused IQ2R gate/up tensors must be contiguous")
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
    _validate_swiglu_parameters(limit, alpha, up_offset)
    _iq2r_task_gemm_swiglu_quant_out(
        activations,
        activation_scales,
        data,
        auxiliary,
        tasks,
        task_count,
        bias,
        output,
        output_scales,
        metadata.logical_n,
        metadata.logical_k,
        limit,
        alpha,
        up_offset,
    )


def iq2r_task_capacity(routes: int, expert_count: int, task_rows: int) -> int:
    """Worst-case task capacity for stable expert runs plus invalid IDs."""

    for name, value in (("routes", routes), ("expert_count", expert_count)):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive int, got {value!r}")
    if expert_count > 512:
        raise ValueError("IQ2R routing supports at most 512 experts")
    if task_rows not in (16, 32, 64, 128, 256):
        raise ValueError("task_rows must be 16, 32, 64, 128, or 256")
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
    expert_map: Tensor | None = None,
    expert_start: int = 0,
    expert_stride: int = 1,
    task_rows: int,
    drop_nonlocal_tasks: bool = False,
) -> None:
    """Map global expert IDs, group local routes, and build bounded GEMM tasks."""

    if expert_ids.dtype != torch.int32 or expert_ids.ndim != 1:
        raise ValueError("expert_ids must be int32 [routes]")
    routes = expert_ids.numel()
    if not 0 < routes <= IQ2R_MAX_ROUTES:
        raise ValueError(
            f"IQ2R route sorting supports 1..{IQ2R_MAX_ROUTES} routed rows"
        )
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
    tensors = [
        expert_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
    ]
    if expert_map is not None:
        if expert_map.dtype != torch.int32 or expert_map.ndim != 1:
            raise ValueError("expert_map must be int32 [global_experts]")
        if not 0 < expert_map.numel() <= 512:
            raise ValueError("expert_map must cover 1..512 global experts")
        tensors.append(expert_map)
    if any(t.device != expert_ids.device for t in tensors):
        raise ValueError("all IQ2R routing tensors must share a device")
    if expert_ids.device.type != "cuda":
        raise ValueError("IQ2R route sorting requires GPU tensors")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all IQ2R routing tensors must be contiguous")
    if (
        isinstance(expert_start, bool)
        or not isinstance(expert_start, int)
        or expert_start < 0
        or isinstance(expert_stride, bool)
        or not isinstance(expert_stride, int)
        or expert_stride <= 0
        or expert_start + (expert_count - 1) * expert_stride >= 512
    ):
        raise ValueError(
            "expert_start and expert_stride must define a positive strided local "
            "expert range within 512"
        )
    _iq2r_route_sort_tasks_out(
        expert_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        expert_map,
        expert_count,
        expert_start,
        expert_stride,
        task_rows,
        drop_nonlocal_tasks,
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


def iq2r_route_scatter_quant_out(
    input: Tensor,
    scatter_indices: Tensor,
    output: Tensor,
    scales: Tensor,
    *,
    topk: int,
) -> None:
    """Quantize each token once and scatter it to its local sorted routes."""

    if input.dtype != torch.bfloat16 or input.ndim != 2:
        raise ValueError("input must be BF16 [tokens,hidden]")
    if scatter_indices.dtype != torch.int32 or scatter_indices.ndim != 1:
        raise ValueError("scatter_indices must be int32 [routes]")
    if topk not in (4, 8, 9) or scatter_indices.numel() != input.shape[0] * topk:
        raise ValueError(
            "scatter_indices length must equal tokens*topk for top-k 4, 8, or 9"
        )
    if input.shape[1] % 32:
        raise ValueError("IQ2R MXFP8 quantization requires hidden divisible by 32")
    expected_output = (scatter_indices.numel(), input.shape[1])
    scale_rows = scatter_indices.numel()
    scale_groups = input.shape[1] // 32
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != expected_output:
        raise ValueError(f"output must be float8_e4m3fn {expected_output}")
    if not _valid_activation_scale_shape(scales, scale_rows, scale_groups):
        raise ValueError(
            f"scales must be row-major uint8 [{scale_rows},{scale_groups}] "
            "or tile16 uint8 "
            f"[{(scale_groups + 3) // 4},{(scale_rows + 15) // 16},4,16]"
        )
    tensors = (input, scatter_indices, output, scales)
    if any(t.device != input.device for t in tensors):
        raise ValueError("IQ2R fused scatter/quant tensors must share a device")
    if input.device.type != "cuda":
        raise ValueError("IQ2R fused scatter/quant tensors must be on one GPU")
    if input.stride(-1) != 1 or input.stride(0) < input.shape[1]:
        raise ValueError(
            "IQ2R fused scatter/quant input must have contiguous columns and "
            "non-overlapping rows"
        )
    if any(not t.is_contiguous() for t in (scatter_indices, output, scales)):
        raise ValueError("IQ2R fused scatter/quant outputs must be contiguous")
    _iq2r_route_scatter_quant_out(input, scatter_indices, output, scales, topk)


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
    expert_map: Tensor | None = None,
    expert_start: int = 0,
    expert_stride: int = 1,
    drop_nonlocal_routes: bool = False,
) -> None:
    """Fuse low-M task construction with gather/quantization."""

    routes = expert_ids.numel()
    glm53_grouped_decode = (
        topk == 9
        and expert_count == 257
        and 16 < routes <= 144
        and input.ndim == 2
        and input.shape[1] == 6144
        and expert_map is None
        and expert_start == 0
        and expert_stride == 1
        and not drop_nonlocal_routes
    )
    if not ((0 < routes <= 16 or glm53_grouped_decode) and routes % topk == 0):
        raise ValueError(
            "direct IQ2R routing requires 1..16 complete top-k rows or the "
            "fused GLM-5.3 grouped decode shape"
        )
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
    tensors = [
        expert_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        output,
        scales,
    ]
    if expert_map is not None:
        if expert_map.dtype != torch.int32 or expert_map.ndim != 1:
            raise ValueError("expert_map must be int32 [global_experts]")
        if not 0 < expert_map.numel() <= 512:
            raise ValueError("expert_map must cover 1..512 global experts")
        tensors.append(expert_map)
    if input.device.type != "cuda" or any(t.device != input.device for t in tensors):
        raise ValueError("all direct IQ2R routing tensors must share one GPU")
    if input.stride(-1) != 1 or input.stride(0) < input.shape[1]:
        raise ValueError("direct IQ2R input must have contiguous non-overlapping rows")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("direct IQ2R routing outputs must be contiguous")
    if (
        isinstance(expert_start, bool)
        or not isinstance(expert_start, int)
        or expert_start < 0
        or isinstance(expert_stride, bool)
        or not isinstance(expert_stride, int)
        or expert_stride <= 0
        or expert_start + (expert_count - 1) * expert_stride >= 512
    ):
        raise ValueError(
            "expert_start and expert_stride must define a positive strided local "
            "expert range within 512"
        )
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
        expert_map,
        topk,
        expert_count,
        expert_start,
        expert_stride,
        drop_nonlocal_routes,
    )


def iq2r_route_topk_direct_gather_quant_out(
    input: Tensor,
    router_logits: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    *,
    renormalize: bool,
    router_bias: Tensor | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    expert_map: Tensor | None = None,
    expert_count: int | None = None,
    expert_start: int = 0,
    expert_stride: int = 1,
    drop_nonlocal_routes: bool = False,
) -> None:
    """Fuse supported low-M top-k routing, direct tasks, and MXFP8 input quantization."""

    if input.dtype != torch.bfloat16 or input.ndim != 2:
        raise ValueError("input must be BF16 [tokens,hidden]")
    tokens, hidden = input.shape
    experts = router_logits.shape[1] if router_logits.ndim == 2 else 0
    topk = topk_ids.shape[1] if topk_ids.ndim == 2 else 0
    biased_sigmoid = scoring_func == "sigmoid"
    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError("fused IQ2R routing supports softmax or sigmoid")
    if (experts, topk) not in ((128, 4), (256, 8)):
        raise ValueError(
            "fused IQ2R routing supports (experts,topk)=(128,4) or (256,8)"
        )
    if biased_sigmoid != ((experts, topk) == (256, 8)):
        raise ValueError("softmax requires 128/top-4 and sigmoid requires 256/top-8")
    max_tokens = 16 // topk
    if not 0 < tokens <= max_tokens:
        raise ValueError(f"fused direct IQ2R routing requires 1..{max_tokens} tokens")
    if hidden % 32:
        raise ValueError("IQ2R MXFP8 quantization requires hidden divisible by 32")
    if router_logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("router_logits must be BF16 or FP32")
    if tuple(router_logits.shape) != (tokens, experts):
        raise ValueError(f"router_logits must be [tokens,{experts}]")
    if router_bias is not None:
        bias_dtype = router_logits.dtype if biased_sigmoid else torch.bfloat16
        if router_bias.dtype != bias_dtype or tuple(router_bias.shape) != (experts,):
            raise ValueError(f"router_bias must be {bias_dtype} [{experts}]")
        if router_bias.device != input.device or not router_bias.is_contiguous():
            raise ValueError("router_bias must be contiguous on the input GPU")
    if biased_sigmoid and router_bias is None:
        raise ValueError("biased sigmoid routing requires router_bias")
    if topk_weights.dtype != torch.float32 or tuple(topk_weights.shape) != (
        tokens,
        topk,
    ):
        raise ValueError(f"topk_weights must be FP32 [tokens,{topk}]")
    if topk_ids.dtype != torch.int32 or tuple(topk_ids.shape) != (tokens, topk):
        raise ValueError(f"topk_ids must be int32 [tokens,{topk}]")
    routes = tokens * topk
    vectors = (sorted_expert_ids, gather_indices, scatter_indices)
    if any(t.dtype != torch.int32 or tuple(t.shape) != (routes,) for t in vectors):
        raise ValueError("sorted/gather/scatter tensors must be int32 [routes]")
    if (
        tasks.dtype != torch.int32
        or tasks.ndim != 2
        or tasks.shape[1] != 3
        or tasks.shape[0] < routes
    ):
        raise ValueError("tasks must be int32 [capacity>=routes,3]")
    if task_count.dtype != torch.int32 or tuple(task_count.shape) != (1,):
        raise ValueError("task_count must be int32 [1]")
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != (
        routes,
        hidden,
    ):
        raise ValueError("output must be FP8 [routes,hidden]")
    if not _valid_activation_scale_shape(scales, routes, hidden // 32):
        raise ValueError(
            "scales must be row-major uint8 [routes,hidden/32] or "
            "tile16 uint8 [ceil(hidden/128),ceil(routes/16),4,16]"
        )
    if expert_count is None:
        expert_count = experts
    if not 0 < expert_count <= experts:
        raise ValueError("expert_count must be in [1, global_experts]")
    tensors = [
        router_logits,
        topk_weights,
        topk_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        output,
        scales,
    ]
    if expert_map is not None:
        if (
            expert_map.dtype != torch.int32
            or expert_map.ndim != 1
            or expert_map.numel() < experts
        ):
            raise ValueError("expert_map must be int32 [global_experts]")
        tensors.append(expert_map)
    if input.device.type != "cuda" or any(t.device != input.device for t in tensors):
        raise ValueError("all fused IQ2R routing tensors must share one GPU")
    if input.stride(-1) != 1 or input.stride(0) < hidden:
        raise ValueError("input must have contiguous non-overlapping rows")
    if router_logits.stride(-1) != 1 or router_logits.stride(0) < experts:
        raise ValueError("router_logits must have contiguous non-overlapping rows")
    if any(not t.is_contiguous() for t in tensors[1:]):
        raise ValueError("all fused IQ2R routing outputs must be contiguous")
    if (
        not isinstance(expert_start, int)
        or isinstance(expert_start, bool)
        or expert_start < 0
        or not isinstance(expert_stride, int)
        or isinstance(expert_stride, bool)
        or expert_stride <= 0
        or expert_start + (expert_count - 1) * expert_stride >= experts
    ):
        raise ValueError("expert_start/expert_stride do not define the local experts")
    if scoring_func == "softmax" and (
        routed_scaling_factor != 1.0
        or expert_map is not None
        or expert_start != 0
        or expert_stride != 1
        or expert_count != experts
        or drop_nonlocal_routes
    ):
        raise ValueError(
            "the softmax fused router requires unsharded experts and scale 1"
        )
    _iq2r_route_topk_direct_gather_quant_out(
        input,
        router_logits,
        topk_weights,
        topk_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        output,
        scales,
        renormalize,
        router_bias,
        biased_sigmoid,
        routed_scaling_factor,
        expert_map,
        expert_count,
        expert_start,
        expert_stride,
        drop_nonlocal_routes,
    )


def iq2r_route_topk_sort_gather_quant_out(
    input: Tensor,
    router_logits: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    sorted_expert_ids: Tensor,
    gather_indices: Tensor,
    scatter_indices: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    scales: Tensor,
    *,
    task_rows: int,
    renormalize: bool,
    router_bias: Tensor | None = None,
    scoring_func: str = "softmax",
    routed_scaling_factor: float = 1.0,
    expert_map: Tensor | None = None,
    expert_count: int | None = None,
    expert_start: int = 0,
    expert_stride: int = 1,
    drop_nonlocal_routes: bool = False,
) -> None:
    """Fuse supported top-k routing, grouped tasks, gather, and input quantization."""

    if input.dtype != torch.bfloat16 or input.ndim != 2:
        raise ValueError("input must be BF16 [tokens,hidden]")
    tokens, hidden = input.shape
    if not 0 < tokens <= 16:
        raise ValueError("fused sorted IQ2R routing requires 1..16 tokens")
    if hidden % 32:
        raise ValueError("IQ2R MXFP8 quantization requires hidden divisible by 32")
    if router_logits.dtype not in (torch.bfloat16, torch.float32):
        raise TypeError("router_logits must be BF16 or FP32")
    experts = router_logits.shape[1] if router_logits.ndim == 2 else 0
    topk = topk_ids.shape[1] if topk_ids.ndim == 2 else 0
    biased_sigmoid = scoring_func == "sigmoid"
    if scoring_func not in ("softmax", "sigmoid"):
        raise ValueError("fused IQ2R routing supports softmax or sigmoid")
    if (experts, topk) not in ((128, 4), (256, 8)):
        raise ValueError(
            "fused IQ2R routing supports (experts,topk)=(128,4) or (256,8)"
        )
    if biased_sigmoid != ((experts, topk) == (256, 8)):
        raise ValueError("softmax requires 128/top-4 and sigmoid requires 256/top-8")
    if tuple(router_logits.shape) != (tokens, experts):
        raise ValueError(f"router_logits must be [tokens,{experts}]")
    if router_bias is not None:
        bias_dtype = router_logits.dtype if biased_sigmoid else torch.bfloat16
        if router_bias.dtype != bias_dtype or tuple(router_bias.shape) != (experts,):
            raise ValueError(f"router_bias must be {bias_dtype} [{experts}]")
        if router_bias.device != input.device or not router_bias.is_contiguous():
            raise ValueError("router_bias must be contiguous on the input GPU")
    if biased_sigmoid and router_bias is None:
        raise ValueError("biased sigmoid routing requires router_bias")
    if topk_weights.dtype != torch.float32 or tuple(topk_weights.shape) != (
        tokens,
        topk,
    ):
        raise ValueError(f"topk_weights must be FP32 [tokens,{topk}]")
    if topk_ids.dtype != torch.int32 or tuple(topk_ids.shape) != (tokens, topk):
        raise ValueError(f"topk_ids must be int32 [tokens,{topk}]")
    routes = tokens * topk
    vectors = (sorted_expert_ids, gather_indices, scatter_indices)
    if any(t.dtype != torch.int32 or tuple(t.shape) != (routes,) for t in vectors):
        raise ValueError("sorted/gather/scatter tensors must be int32 [routes]")
    if expert_count is None:
        expert_count = experts
    if not 0 < expert_count <= experts:
        raise ValueError("expert_count must be in [1, global_experts]")
    required = iq2r_task_capacity(routes, expert_count, task_rows)
    if (
        tasks.dtype != torch.int32
        or tasks.ndim != 2
        or tasks.shape[1] != 3
        or tasks.shape[0] < required
    ):
        raise ValueError(f"tasks must be int32 [capacity>={required},3]")
    if task_count.dtype != torch.int32 or tuple(task_count.shape) != (1,):
        raise ValueError("task_count must be int32 [1]")
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != (
        routes,
        hidden,
    ):
        raise ValueError("output must be FP8 [routes,hidden]")
    if not _valid_activation_scale_shape(scales, routes, hidden // 32):
        raise ValueError(
            "scales must be row-major uint8 [routes,hidden/32] or "
            "tile16 uint8 [ceil(hidden/128),ceil(routes/16),4,16]"
        )
    tensors = [
        router_logits,
        topk_weights,
        topk_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        output,
        scales,
    ]
    if expert_map is not None:
        if (
            expert_map.dtype != torch.int32
            or expert_map.ndim != 1
            or expert_map.numel() < experts
        ):
            raise ValueError("expert_map must be int32 [global_experts]")
        tensors.append(expert_map)
    if input.device.type != "cuda" or any(t.device != input.device for t in tensors):
        raise ValueError("all fused sorted IQ2R tensors must share one GPU")
    if input.stride(-1) != 1 or input.stride(0) < hidden:
        raise ValueError("input must have contiguous non-overlapping rows")
    if router_logits.stride(-1) != 1 or router_logits.stride(0) < experts:
        raise ValueError("router_logits must have contiguous non-overlapping rows")
    if any(not t.is_contiguous() for t in tensors[1:]):
        raise ValueError("all fused sorted IQ2R outputs must be contiguous")
    if (
        not isinstance(expert_start, int)
        or isinstance(expert_start, bool)
        or expert_start < 0
        or not isinstance(expert_stride, int)
        or isinstance(expert_stride, bool)
        or expert_stride <= 0
        or expert_start + (expert_count - 1) * expert_stride >= experts
    ):
        raise ValueError("expert_start/expert_stride do not define the local experts")
    if scoring_func == "softmax" and (
        routed_scaling_factor != 1.0
        or expert_map is not None
        or expert_start != 0
        or expert_stride != 1
        or expert_count != experts
        or drop_nonlocal_routes
    ):
        raise ValueError(
            "the softmax fused router requires unsharded experts and scale 1"
        )
    _iq2r_route_topk_sort_gather_quant_out(
        input,
        router_logits,
        topk_weights,
        topk_ids,
        sorted_expert_ids,
        gather_indices,
        scatter_indices,
        tasks,
        task_count,
        output,
        scales,
        task_rows,
        renormalize,
        router_bias,
        biased_sigmoid,
        routed_scaling_factor,
        expert_map,
        expert_count,
        expert_start,
        expert_stride,
        drop_nonlocal_routes,
    )


def _validate_swiglu_parameters(limit: float, alpha: float, up_offset: float) -> None:
    import math

    if not math.isfinite(limit) or limit < 0:
        raise ValueError("SwiGLU limit must be finite and non-negative")
    if not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("SwiGLU alpha must be finite and positive")
    if not math.isfinite(up_offset):
        raise ValueError("SwiGLU up_offset must be finite")


def iq2r_swiglu_out(
    gate_up: Tensor,
    output: Tensor,
    *,
    limit: float = 7.0,
    alpha: float = 1.702,
    up_offset: float = 1.0,
) -> None:
    """Apply SwiGLU to interleaved gate/up columns; limit 0 disables clamping."""

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
    _validate_swiglu_parameters(limit, alpha, up_offset)
    _iq2r_swiglu_out(gate_up, output, limit, alpha, up_offset)


def iq2r_swiglu_quant_out(
    gate_up: Tensor,
    output: Tensor,
    scales: Tensor,
    *,
    activated: Tensor | None = None,
    limit: float = 7.0,
    alpha: float = 1.702,
    up_offset: float = 1.0,
) -> None:
    """Apply SwiGLU and emit MXFP8/E8M0; limit 0 disables clamping."""

    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2:
        raise ValueError("gate_up must be BF16 [rows,2*intermediate]")
    if gate_up.shape[1] % 64:
        raise ValueError("gate_up width must be divisible by 64")
    expected_output = (gate_up.shape[0], gate_up.shape[1] // 2)
    scale_rows = gate_up.shape[0]
    scale_groups = gate_up.shape[1] // 64
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != expected_output:
        raise ValueError(f"output must be float8_e4m3fn {expected_output}")
    if not _valid_activation_scale_shape(scales, scale_rows, scale_groups):
        raise ValueError(
            f"scales must be row-major uint8 [{scale_rows},{scale_groups}] "
            "or tile16 uint8 "
            f"[{(scale_groups + 3) // 4},{(scale_rows + 15) // 16},4,16]"
        )
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
    _validate_swiglu_parameters(limit, alpha, up_offset)
    _iq2r_swiglu_quant_out(gate_up, output, scales, activated, limit, alpha, up_offset)


def iq2r_swiglu_quant_scatter_out(
    gate_up: Tensor,
    scatter_indices: Tensor,
    output: Tensor,
    scales: Tensor,
    *,
    topk: int,
    activated: Tensor | None = None,
    limit: float = 7.0,
    alpha: float = 1.702,
    up_offset: float = 1.0,
) -> None:
    """Apply and quantize SwiGLU only for routes local to this EP rank."""

    if gate_up.dtype != torch.bfloat16 or gate_up.ndim != 2:
        raise ValueError("gate_up must be BF16 [routes,2*intermediate]")
    if gate_up.shape[1] % 64:
        raise ValueError("gate_up width must be divisible by 64")
    if scatter_indices.dtype != torch.int32 or tuple(scatter_indices.shape) != (
        gate_up.shape[0],
    ):
        raise ValueError("scatter_indices must be int32 [routes]")
    if topk not in (4, 8) or gate_up.shape[0] % topk:
        raise ValueError("indexed IQ2R SwiGLU requires top-k 4 or 8")
    expected_output = (gate_up.shape[0], gate_up.shape[1] // 2)
    scale_rows = gate_up.shape[0]
    scale_groups = gate_up.shape[1] // 64
    if output.dtype != torch.float8_e4m3fn or tuple(output.shape) != expected_output:
        raise ValueError(f"output must be float8_e4m3fn {expected_output}")
    if not _valid_activation_scale_shape(scales, scale_rows, scale_groups):
        raise ValueError(
            f"scales must be row-major uint8 [{scale_rows},{scale_groups}] "
            "or tile16 uint8 "
            f"[{(scale_groups + 3) // 4},{(scale_rows + 15) // 16},4,16]"
        )
    tensors = [gate_up, scatter_indices, output, scales]
    if activated is not None:
        if (
            activated.dtype != torch.bfloat16
            or tuple(activated.shape) != expected_output
        ):
            raise ValueError(f"activated must be BF16 {expected_output}")
        tensors.append(activated)
    if any(t.device != gate_up.device for t in tensors):
        raise ValueError("IQ2R indexed SwiGLU/quant tensors must share a device")
    if gate_up.device.type != "cuda" or any(not t.is_contiguous() for t in tensors):
        raise ValueError(
            "IQ2R indexed SwiGLU/quant tensors must be contiguous on one GPU"
        )
    _validate_swiglu_parameters(limit, alpha, up_offset)
    _iq2r_swiglu_quant_scatter_out(
        gate_up,
        scatter_indices,
        output,
        scales,
        topk,
        activated,
        limit,
        alpha,
        up_offset,
    )


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


def iq2r_route_reduce_add_indexed_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    shared_output: Tensor,
    output: Tensor,
    *,
    topk: int,
) -> None:
    """Reduce sorted IQ2R routes and add a BF16 shared-expert output."""

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
    for name, tensor in (("shared_output", shared_output), ("output", output)):
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must be BF16 {expected}")
    tensors = (route_output, route_weights, scatter_indices, shared_output, output)
    if any(t.device != route_output.device for t in tensors):
        raise ValueError("IQ2R route reduction/add tensors must share a device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("IQ2R route reduction/add tensors must be contiguous")
    _iq2r_route_reduce_add_indexed_out(
        route_output,
        route_weights,
        scatter_indices,
        shared_output,
        output,
        topk,
    )


def iq2r_route_reduce_add_rmsnorm_indexed_out(
    route_output: Tensor,
    route_weights: Tensor,
    scatter_indices: Tensor,
    residual: Tensor,
    norm_weight: Tensor,
    output: Tensor,
    residual_out: Tensor,
    *,
    topk: int,
    epsilon: float,
    block_size: int = 256,
) -> None:
    """Reduce IQ2R routes, update the residual, and RMS-normalize in one launch.

    The route sum is rounded to BF16 before the residual add, matching the
    unfused ``route_reduce`` followed by ``rmsnorm2d_fwd_with_add`` sequence.
    ``block_size`` is exposed while the gfx950 launch geometry is qualified.
    """

    if route_output.dtype != torch.bfloat16 or route_output.ndim != 2:
        raise ValueError("route_output must be BF16 [routes,hidden]")
    if route_weights.dtype != torch.float32 or route_weights.ndim != 2:
        raise ValueError("route_weights must be FP32 [tokens,topk]")
    if route_weights.shape[1] != topk:
        raise ValueError("route_weights top-k dimension does not match topk")
    tokens = route_weights.shape[0]
    routes = tokens * topk
    hidden = route_output.shape[1]
    if route_output.shape[0] != routes:
        raise ValueError("route_output row count must equal tokens*topk")
    if scatter_indices.dtype != torch.int32 or tuple(scatter_indices.shape) != (
        routes,
    ):
        raise ValueError("scatter_indices must be int32 [routes]")
    expected = (tokens, hidden)
    for name, tensor in (
        ("residual", residual),
        ("output", output),
        ("residual_out", residual_out),
    ):
        if tensor.dtype != torch.bfloat16 or tuple(tensor.shape) != expected:
            raise ValueError(f"{name} must be BF16 {expected}")
    if norm_weight.dtype != torch.bfloat16 or tuple(norm_weight.shape) != (hidden,):
        raise ValueError(f"norm_weight must be BF16 ({hidden},)")
    tensors = (
        route_output,
        route_weights,
        scatter_indices,
        residual,
        norm_weight,
        output,
        residual_out,
    )
    if any(t.device != route_output.device for t in tensors):
        raise ValueError("fused route reduction/RMSNorm tensors must share a device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("fused route reduction/RMSNorm tensors must be contiguous")
    if block_size not in (256, 512, 1024):
        raise ValueError("block_size must be 256, 512, or 1024")
    _iq2r_route_reduce_add_rmsnorm_indexed_out(
        route_output,
        route_weights,
        scatter_indices,
        residual,
        norm_weight,
        output,
        residual_out,
        topk,
        epsilon,
        block_size,
    )


__all__ = [
    "IQ2R_MAX_ROUTES",
    "iq2r_encode_device",
    "iq2r_gemm",
    "iq2r_gemm_out",
    "iq2r_materialize_device",
    "iq2r_materialize_out",
    "iq2r_route_direct_gather_quant_out",
    "iq2r_route_gather_indexed_out",
    "iq2r_route_gather_quant_out",
    "iq2r_route_reduce_add_indexed_out",
    "iq2r_route_reduce_add_rmsnorm_indexed_out",
    "iq2r_route_reduce_indexed_out",
    "iq2r_route_scatter_quant_out",
    "iq2r_route_sort_tasks_out",
    "iq2r_route_topk_direct_gather_quant_out",
    "iq2r_route_topk_sort_gather_quant_out",
    "iq2r_swiglu_out",
    "iq2r_swiglu_quant_out",
    "iq2r_swiglu_quant_scatter_out",
    "iq2r_task_capacity",
    "iq2r_task_gemm",
    "iq2r_task_gemm_out",
    "iq2r_task_gemm_swiglu_quant_out",
]


@compile_ops("module_iq2r_moe", fc_name="iq2r_gate_aligned_fused_out", develop=True)
def iq2r_gate_aligned_fused_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    gather: Tensor,
    output: Tensor,
    output_scales: Tensor,
    rows_per_cta: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_gate_quad_fused_out", develop=True)
def iq2r_gate_quad_fused_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    gather: Tensor,
    output: Tensor,
    output_scales: Tensor,
    rows_per_cta: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_gate_quad_sparse_out", develop=True)
def iq2r_gate_quad_sparse_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    gather: Tensor,
    output: Tensor,
    output_scales: Tensor,
    rows_per_cta: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_gate_quad_splitk_out", develop=True)
def iq2r_gate_quad_splitk_out(
    input: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    partials: Tensor,
    output: Tensor,
    output_scales: Tensor,
    physical_waves: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_gate_quad_route_fused_out", develop=True)
def iq2r_gate_quad_route_fused_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    output_scales: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_down_sparse_large32_out", develop=True)
def iq2r_down_sparse_large32_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    grid_multiplier: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_down_sparse_scheduled_out", develop=True)
def iq2r_down_sparse_scheduled_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    grid_multiplier: int,
    variant: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_down_shortk_out", develop=True)
def iq2r_down_shortk_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    grid_multiplier: int,
    variant: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_gate_quad_scheduled_out", develop=True)
def iq2r_gate_quad_scheduled_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    output_scales: Tensor,
    variant: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_down_token_fused48_out", develop=True)
def iq2r_down_token_fused48_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    expert_ids: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_down_token_route9_out", develop=True)
def iq2r_down_token_route9_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    expert_ids: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_tp4_gate_out", develop=True)
def iq2r_glm53_tp4_gate_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    output_scales: Tensor,
    variant: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_tp4_down_out", develop=True)
def iq2r_glm53_tp4_down_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    grid_multiplier: int,
    variant: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_tp4_route9_out", develop=True)
def iq2r_glm53_tp4_route9_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    expert_ids: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_tp4_indexed_gate_out", develop=True)
def iq2r_glm53_tp4_indexed_gate_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    gather: Tensor,
    output: Tensor,
    output_scales: Tensor,
    rows_per_cta: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_tp4_large_down_out", develop=True)
def iq2r_glm53_tp4_large_down_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    output: Tensor,
    grid_multiplier: int,
    variant: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_down_token_pair9_out", develop=True)
def iq2r_down_token_pair9_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    expert_ids: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
    group_tokens: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_glm53_dense_gate_out", develop=True)
def iq2r_glm53_dense_gate_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    tasks: Tensor,
    task_count: Tensor,
    gather: Tensor,
    output: Tensor,
    output_scales: Tensor,
    rows_per_cta: int,
    variant: int,
) -> None: ...


@compile_ops("module_iq2r_moe", fc_name="iq2r_down_token_adaptive9_out", develop=True)
def iq2r_down_token_adaptive9_out(
    activations: Tensor,
    scales: Tensor,
    data: Tensor,
    auxiliary: Tensor,
    expert_ids: Tensor,
    scatter: Tensor,
    route_weights: Tensor,
    output: Tensor,
    task_count: Tensor,
    task_table: Tensor,
) -> None: ...
