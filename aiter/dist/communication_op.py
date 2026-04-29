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

from .parallel_state import (
    get_tp_group,
    get_pp_group,
    get_dp_group,
    get_ep_group,
    get_custom_group,
    has_custom_group,
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
) -> tuple[torch.Tensor, torch.Tensor]:
    _assert_no_custom_group("tensor_model_parallel_fused_allreduce_rmsnorm")
    return get_tp_group().fused_allreduce_rmsnorm(
        input_, residual_inp_, weight_, eps, prefill_support
    )


def tensor_model_parallel_fused_allreduce_rmsnorm_quant(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    prefill_support: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _assert_no_custom_group("tensor_model_parallel_fused_allreduce_rmsnorm_quant")
    return get_tp_group().fused_allreduce_rmsnorm_quant(
        input_,
        residual_inp_,
        weight_,
        eps,
        prefill_support,
    )


def tensor_model_parallel_fused_allreduce_rmsnorm_quant_per_group(
    input_: torch.Tensor,
    residual_inp_: torch.Tensor,
    weight_: torch.Tensor,
    eps: float,
    group_size: int = 128,
    prefill_support: bool = False,
    emit_bf16: bool = False,
):
    return get_tp_group().fused_allreduce_rmsnorm_quant_per_group(
        input_,
        residual_inp_,
        weight_,
        eps,
        group_size,
        prefill_support,
        emit_bf16=emit_bf16,
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
