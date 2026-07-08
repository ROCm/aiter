"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2026, The vLLM team.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*      http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
"""

from typing import Any, Dict, Optional, Union

import torch
import torch.distributed

from aiter.ops.enum import QuantType

from .parallel_state import (
    get_tp_group,
    get_pp_group,
    get_dp_group,
    get_ep_group,
    get_custom_group,
    has_custom_group,
)

# quant_type aliases -> fused AR+RMSNorm+quant family. Single source of truth,
# also imported by communicator_cuda.
_FUSED_AR_RMS_QUANT_ALIASES = {
    "fp8": "per_token",
    "fp8_per_token": "per_token",
    "per-token": "per_token",
    "per_token": "per_token",
    "per_token_fp8": "per_token",
    "fp8_per_group": "per_group",
    "per-group": "per_group",
    "per_group": "per_group",
    "per_group_fp8": "per_group",
    "per_1x128": "per_group",
    "fp4": "mxfp4",
    "fp4_e2m1": "mxfp4",
    "mx_fp4": "mxfp4",
    "mxfp4": "mxfp4",
    "per_1x32": "mxfp4",
}


def _normalize_fused_ar_rms_quant_type(quant_type):
    if isinstance(quant_type, str):
        normalized = _FUSED_AR_RMS_QUANT_ALIASES.get(quant_type.lower())
        if normalized is not None:
            return normalized
    else:
        if quant_type == QuantType.per_Token:
            return "per_token"
        if quant_type in (QuantType.per_1x128, getattr(QuantType, "per_128x128", None)):
            return "per_group"
        if quant_type == QuantType.per_1x32:
            return "mxfp4"
        try:
            return _normalize_fused_ar_rms_quant_type(QuantType(quant_type))
        except Exception:
            pass
    raise ValueError(
        "unsupported fused AR+RMSNorm quant_type="
        f"{quant_type!r}; expected per_token, per_group/per_1x128, or mxfp4/per_1x32"
    )


# Lookup sets for trace-safe dispatch: `enum == enum` graph-breaks Dynamo, so
# match on int `.value` / string membership instead.
_PER_GROUP_QUANT_ALIASES = frozenset(
    k for k, v in _FUSED_AR_RMS_QUANT_ALIASES.items() if v == "per_group"
)
_PER_GROUP_QUANT_VALUES = frozenset(
    q.value
    for q in (getattr(QuantType, "per_1x128", None), getattr(QuantType, "per_128x128", None))
    if q is not None
)
_PER_TOKEN_QUANT_VALUES = frozenset(
    q.value for q in (getattr(QuantType, "per_Token", None),) if q is not None
)
_MXFP4_QUANT_VALUES = frozenset(
    q.value for q in (getattr(QuantType, "per_1x32", None),) if q is not None
)


def _is_per_group_quant(quant_type):
    if isinstance(quant_type, str):
        return quant_type.lower() in _PER_GROUP_QUANT_ALIASES
    return getattr(quant_type, "value", quant_type) in _PER_GROUP_QUANT_VALUES


def _canonical_fused_ar_rms_quant_type(quant_type):
    """Trace-safe canonical string ("per_token"/"per_group"/"mxfp4").

    Reduces the pybind enum via int ``.value`` (never ``enum == enum``, which
    graph-breaks Dynamo) so downstream string guards match under torch.compile.
    """
    if isinstance(quant_type, str):
        return _normalize_fused_ar_rms_quant_type(quant_type)
    v = getattr(quant_type, "value", quant_type)
    if v in _PER_GROUP_QUANT_VALUES:
        return "per_group"
    if v in _PER_TOKEN_QUANT_VALUES:
        return "per_token"
    if v in _MXFP4_QUANT_VALUES:
        return "mxfp4"
    raise ValueError(
        "unsupported fused AR+RMSNorm quant_type="
        f"{quant_type!r}; expected per_token, per_group/per_1x128, or mxfp4/per_1x32"
    )


def _assert_no_custom_group(op_name: str):
    assert not has_custom_group(), (
        f"custom_group_config is set — use custom_all_reduce() instead of "
        f"{op_name}()"
    )


def _assert_has_custom_group():
    assert has_custom_group(), (
        "custom_group_config is not set — use tensor_model_parallel_all_reduce() "
        "or other standard parallel group operations instead of custom_all_reduce()"
    )


# ============================================================
# Tensor Model Parallel (TP) communication operations
# ============================================================


def tensor_model_parallel_all_reduce(
    input_: torch.Tensor,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    prefill_support: bool = False,
) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    _assert_no_custom_group("tensor_model_parallel_all_reduce")
    return get_tp_group().all_reduce(input_, use_new, open_fp8_quant, prefill_support)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    prefill_support: bool = False,
    x_pad_to_multiple: int = 0,
    gemma_norm: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    _assert_no_custom_group("tensor_model_parallel_fused_allreduce_rmsnorm")
    return get_tp_group().fused_allreduce_rmsnorm(
        input_,
        residual_inp_,
        weight_,
        eps,
        prefill_support,
        x_pad_to_multiple=x_pad_to_multiple,
        gemma_norm=gemma_norm,
    )


def tensor_model_parallel_fused_allreduce_rmsnorm_quant(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    prefill_support: bool = False,
    quant_type: Any = "per_token",
    group_size: int = 128,
    emit_bf16: bool = False,
    transpose_scale: bool = False,
    gemma_norm: bool = False,
):
    """Fused tensor-parallel all-reduce + RMSNorm + quantization.

    ``quant_type`` selects the quantization epilogue:
    ``"per_token"`` for existing FP8 per-token quantization,
    ``"per_group"`` / ``"per_1x128"`` for FP8 per-group quantization, and
    ``"mxfp4"`` / ``"per_1x32"`` for MXFP4 quantization.

    ``group_size`` is the per-group quantization block size (per-group only).

    ``transpose_scale`` (per-group only) writes the per-group scale in
    column-major layout ``(num_groups, M)`` viewed as ``(M, num_groups)``,
    matching what ``gemm_a8w8_blockscale_preshuffle`` expects.
    """
    _assert_no_custom_group("tensor_model_parallel_fused_allreduce_rmsnorm_quant")

    # Enum -> string up front (trace-safe) so downstream string guards match
    # under torch.compile instead of falling to the untraced pybind path.
    quant_type = _canonical_fused_ar_rms_quant_type(quant_type)

    # Per-group has its own traceable torch.library method; the generic path would
    # hit raw pybind and graph-break.
    if _is_per_group_quant(quant_type):
        return get_tp_group().fused_allreduce_rmsnorm_quant_per_group(
            input_,
            residual_inp_,
            weight_,
            eps,
            group_size=group_size,
            prefill_support=prefill_support,
            emit_bf16=emit_bf16,
            transpose_scale=transpose_scale,
        )

    return get_tp_group().fused_allreduce_rmsnorm_quant(
        input_,
        residual_inp_,
        weight_,
        eps,
        prefill_support,
        quant_type=quant_type,
        group_size=group_size,
        emit_bf16=emit_bf16,
        transpose_scale=transpose_scale,
        gemma_norm=gemma_norm,
    )


def tensor_model_parallel_fused_allreduce_rmsnorm_mxfp4_quant(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    prefill_support: bool = False,
    emit_bf16: bool = False,
):
    return tensor_model_parallel_fused_allreduce_rmsnorm_quant(
        input_,
        residual_inp_,
        weight_,
        eps,
        prefill_support,
        quant_type="mxfp4",
        emit_bf16=emit_bf16,
    )


def tensor_model_parallel_fused_qknorm_allreduce(
    qkv_in: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    eps: float,
):
    return get_tp_group().fused_qknorm_allreduce(
        qkv_in,
        q_w,
        k_w,
        eps,
    )


def tensor_model_parallel_fused_qknorm_allreduce_rope(
    qkv_in: torch.Tensor,
    q_w: torch.Tensor,
    k_w: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    position_ids: torch.Tensor,
    head_dim: int,
    rotary_dim: int,
    eps: float,
):
    return get_tp_group().fused_qknorm_allreduce_rope(
        qkv_in,
        q_w,
        k_w,
        cos_sin_cache,
        position_ids,
        head_dim,
        rotary_dim,
        eps,
    )


def tensor_model_parallel_custom_all_gather(input_: torch.Tensor) -> torch.Tensor:
    _assert_no_custom_group("tensor_model_parallel_custom_all_gather")
    return get_tp_group().custom_all_gather(input_)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor,
    use_custom: bool = True,
    dim: int = 0,
) -> torch.Tensor:
    _assert_no_custom_group("tensor_model_parallel_reduce_scatter")
    return get_tp_group().reduce_scatter_tensor(input_, use_custom, dim)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor,
    use_custom: bool = False,
    dim: int = -1,
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    _assert_no_custom_group("tensor_model_parallel_all_gather")
    return get_tp_group().all_gather(input_, use_custom, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    _assert_no_custom_group("tensor_model_parallel_gather")
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    _assert_no_custom_group("broadcast_tensor_dict")
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


# ============================================================
# Expert Parallel (EP) communication operations
# ============================================================


def expert_parallel_all_reduce(
    input_: torch.Tensor, use_new: bool = True, open_fp8_quant: bool = False
) -> torch.Tensor:
    """All-reduce the input tensor across expert parallel group."""
    _assert_no_custom_group("expert_parallel_all_reduce")
    return get_ep_group().all_reduce(input_, use_new, open_fp8_quant)


def expert_parallel_all_gather(
    input_: torch.Tensor, use_custom: bool = False, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across expert parallel group."""
    _assert_no_custom_group("expert_parallel_all_gather")
    return get_ep_group().all_gather(input_, use_custom, dim)


def expert_parallel_reduce_scatter(
    input_: torch.Tensor, use_custom: bool = True, dim: int = 0
) -> torch.Tensor:
    """Reduce-scatter the input tensor across expert parallel group."""
    _assert_no_custom_group("expert_parallel_reduce_scatter")
    return get_ep_group().reduce_scatter_tensor(input_, use_custom, dim)


def expert_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across expert parallel group."""
    _assert_no_custom_group("expert_parallel_gather")
    return get_ep_group().gather(input_, dst, dim)


def expert_parallel_broadcast(input_: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast the input tensor across expert parallel group."""
    _assert_no_custom_group("expert_parallel_broadcast")
    return get_ep_group().broadcast(input_, src)


def expert_parallel_broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    """Broadcast a tensor dict across expert parallel group."""
    _assert_no_custom_group("expert_parallel_broadcast_tensor_dict")
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_ep_group().broadcast_tensor_dict(tensor_dict, src)


# ============================================================
# Data Parallel (DP) communication operations
# ============================================================


def data_parallel_all_reduce(
    input_: torch.Tensor, use_new: bool = True, open_fp8_quant: bool = False
) -> torch.Tensor:
    """All-reduce the input tensor across data parallel group."""
    _assert_no_custom_group("data_parallel_all_reduce")
    return get_dp_group().all_reduce(input_, use_new, open_fp8_quant)


def data_parallel_all_gather(
    input_: torch.Tensor, use_custom: bool = False, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across data parallel group."""
    _assert_no_custom_group("data_parallel_all_gather")
    return get_dp_group().all_gather(input_, use_custom, dim)


def data_parallel_reduce_scatter(
    input_: torch.Tensor, use_custom: bool = True, dim: int = 0
) -> torch.Tensor:
    """Reduce-scatter the input tensor across data parallel group."""
    _assert_no_custom_group("data_parallel_reduce_scatter")
    return get_dp_group().reduce_scatter_tensor(input_, use_custom, dim)


def data_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across data parallel group."""
    _assert_no_custom_group("data_parallel_gather")
    return get_dp_group().gather(input_, dst, dim)


def data_parallel_broadcast(input_: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast the input tensor across data parallel group."""
    _assert_no_custom_group("data_parallel_broadcast")
    return get_dp_group().broadcast(input_, src)


def data_parallel_broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    """Broadcast a tensor dict across data parallel group."""
    _assert_no_custom_group("data_parallel_broadcast_tensor_dict")
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_dp_group().broadcast_tensor_dict(tensor_dict, src)


# ============================================================
# Pipeline Model Parallel (PP) communication operations
# ============================================================


def pipeline_model_parallel_all_reduce(
    input_: torch.Tensor, use_new: bool = True, open_fp8_quant: bool = False
) -> torch.Tensor:
    """All-reduce the input tensor across pipeline parallel group."""
    _assert_no_custom_group("pipeline_model_parallel_all_reduce")
    return get_pp_group().all_reduce(input_, use_new, open_fp8_quant)


def pipeline_model_parallel_all_gather(
    input_: torch.Tensor, use_custom: bool = False, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across pipeline parallel group."""
    _assert_no_custom_group("pipeline_model_parallel_all_gather")
    return get_pp_group().all_gather(input_, use_custom, dim)


def pipeline_model_parallel_broadcast(
    input_: torch.Tensor, src: int = 0
) -> torch.Tensor:
    """Broadcast the input tensor across pipeline parallel group."""
    _assert_no_custom_group("pipeline_model_parallel_broadcast")
    return get_pp_group().broadcast(input_, src)


def pipeline_model_parallel_send(
    input_: torch.Tensor, dst: Optional[int] = None
) -> None:
    """Send a tensor to the next stage in the pipeline."""
    _assert_no_custom_group("pipeline_model_parallel_send")
    get_pp_group().send(input_, dst)


def pipeline_model_parallel_recv(
    size: torch.Size, dtype: torch.dtype, src: Optional[int] = None
) -> torch.Tensor:
    """Receive a tensor from the previous stage in the pipeline."""
    _assert_no_custom_group("pipeline_model_parallel_recv")
    return get_pp_group().recv(size, dtype, src)


def pipeline_model_parallel_broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    """Broadcast a tensor dict across pipeline parallel group."""
    _assert_no_custom_group("pipeline_model_parallel_broadcast_tensor_dict")
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_pp_group().broadcast_tensor_dict(tensor_dict, src)


# ============================================================
# Custom group communication operations
# ============================================================


def custom_all_reduce(
    input_: torch.Tensor,
    use_new: bool = True,
    open_fp8_quant: bool = False,
    group: Optional[str] = None,
) -> torch.Tensor:
    """All-reduce the input tensor across the user-specified custom group.

    Args:
        group: Name of the custom group. When only one custom group is
            initialized this can be omitted. When multiple groups exist,
            pass the group name to select which one to use.
    """
    _assert_has_custom_group()
    return get_custom_group(group).all_reduce(input_, use_new, open_fp8_quant)


def custom_all_gather(
    input_: torch.Tensor,
    use_custom: bool = True,
    dim: int = 0,
    group: Optional[str] = None,
) -> torch.Tensor:
    """All-gather the input tensor across the user-specified custom group.

    Args:
        group: Name of the custom group. When only one custom group is
            initialized this can be omitted. When multiple groups exist,
            pass the group name to select which one to use.
    """
    _assert_has_custom_group()
    return get_custom_group(group).all_gather(input_, use_custom, dim)


def custom_reduce_scatter(
    input_: torch.Tensor,
    use_custom: bool = True,
    dim: int = 0,
    group: Optional[str] = None,
) -> torch.Tensor:
    """Reduce-scatter the input tensor across the user-specified custom group.

    Args:
        group: Name of the custom group. When only one custom group is
            initialized this can be omitted. When multiple groups exist,
            pass the group name to select which one to use.
    """
    _assert_has_custom_group()
    return get_custom_group(group).reduce_scatter_tensor(input_, use_custom, dim)
