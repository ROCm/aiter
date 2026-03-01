"""
* Copyright (C) Advanced Micro Devices, Inc. All rights reserved.
* Copyright (C) 2024-2025, The vLLM team.
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

from .parallel_state import get_tp_group, get_pp_group, get_dp_group, get_ep_group

# ============================================================
# Tensor Model Parallel (TP) communication operations
# ============================================================


def tensor_model_parallel_all_reduce(
    input_: torch.Tensor, use_new: bool = True, open_fp8_quant: bool = False
) -> torch.Tensor:
    """All-reduce the input tensor across model parallel group."""
    return get_tp_group().all_reduce(input_, use_new, open_fp8_quant)


def tensor_model_parallel_fused_allreduce_rmsnorm(
    input_: torch.Tensor, residual_inp_: torch.Tensor, weight_: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    return get_tp_group().fused_allreduce_rmsnorm(input_, residual_inp_, weight_, eps)


def tensor_model_parallel_fused_allreduce_rmsnorm_quant(
    input_: torch.Tensor, residual_inp_: torch.Tensor, weight_: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return get_tp_group().fused_allreduce_rmsnorm_quant(
        input_, residual_inp_, weight_, eps
    )


def tensor_model_parallel_custom_all_gather(input_: torch.Tensor) -> torch.Tensor:
    return get_tp_group().custom_all_gather(input_)


def tensor_model_parallel_reduce_scatter(
    input_: torch.Tensor, use_custom: bool = True, dim: int = 0
) -> torch.Tensor:
    return get_tp_group().reduce_scatter_tensor(input_, use_custom, dim)


def tensor_model_parallel_all_gather(
    input_: torch.Tensor, use_custom: bool = False, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across model parallel group."""
    return get_tp_group().all_gather(input_, use_custom, dim)


def tensor_model_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across model parallel group."""
    return get_tp_group().gather(input_, dst, dim)


def broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
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
    return get_ep_group().all_reduce(input_, use_new, open_fp8_quant)


def expert_parallel_all_gather(
    input_: torch.Tensor, use_custom: bool = False, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across expert parallel group."""
    return get_ep_group().all_gather(input_, use_custom, dim)


def expert_parallel_reduce_scatter(
    input_: torch.Tensor, use_custom: bool = True, dim: int = 0
) -> torch.Tensor:
    """Reduce-scatter the input tensor across expert parallel group."""
    return get_ep_group().reduce_scatter_tensor(input_, use_custom, dim)


def expert_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across expert parallel group."""
    return get_ep_group().gather(input_, dst, dim)


def expert_parallel_broadcast(input_: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast the input tensor across expert parallel group."""
    return get_ep_group().broadcast(input_, src)


def expert_parallel_broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    """Broadcast a tensor dict across expert parallel group."""
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
    return get_dp_group().all_reduce(input_, use_new, open_fp8_quant)


def data_parallel_all_gather(
    input_: torch.Tensor, use_custom: bool = False, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across data parallel group."""
    return get_dp_group().all_gather(input_, use_custom, dim)


def data_parallel_reduce_scatter(
    input_: torch.Tensor, use_custom: bool = True, dim: int = 0
) -> torch.Tensor:
    """Reduce-scatter the input tensor across data parallel group."""
    return get_dp_group().reduce_scatter_tensor(input_, use_custom, dim)


def data_parallel_gather(
    input_: torch.Tensor, dst: int = 0, dim: int = -1
) -> Optional[torch.Tensor]:
    """Gather the input tensor across data parallel group."""
    return get_dp_group().gather(input_, dst, dim)


def data_parallel_broadcast(input_: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast the input tensor across data parallel group."""
    return get_dp_group().broadcast(input_, src)


def data_parallel_broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    """Broadcast a tensor dict across data parallel group."""
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
    return get_pp_group().all_reduce(input_, use_new, open_fp8_quant)


def pipeline_model_parallel_all_gather(
    input_: torch.Tensor, use_custom: bool = False, dim: int = -1
) -> torch.Tensor:
    """All-gather the input tensor across pipeline parallel group."""
    return get_pp_group().all_gather(input_, use_custom, dim)


def pipeline_model_parallel_broadcast(
    input_: torch.Tensor, src: int = 0
) -> torch.Tensor:
    """Broadcast the input tensor across pipeline parallel group."""
    return get_pp_group().broadcast(input_, src)


def pipeline_model_parallel_send(
    input_: torch.Tensor, dst: Optional[int] = None
) -> None:
    """Send a tensor to the next stage in the pipeline."""
    get_pp_group().send(input_, dst)


def pipeline_model_parallel_recv(
    size: torch.Size, dtype: torch.dtype, src: Optional[int] = None
) -> torch.Tensor:
    """Receive a tensor from the previous stage in the pipeline."""
    return get_pp_group().recv(size, dtype, src)


def pipeline_model_parallel_broadcast_tensor_dict(
    tensor_dict: Optional[Dict[Any, Union[torch.Tensor, Any]]] = None, src: int = 0
):
    """Broadcast a tensor dict across pipeline parallel group."""
    if not torch.distributed.is_initialized():
        return tensor_dict
    return get_pp_group().broadcast_tensor_dict(tensor_dict, src)
