# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
import torch.distributed as dist
from ..dist.parallel_state import (
    ensure_model_parallel_initialized,
    init_distributed_environment,
    set_custom_all_reduce,
    get_tp_group,
    destroy_model_parallel,
    destroy_distributed_environment,
)
from ..dist.utils import get_open_port, get_distributed_init_method, get_ip
import aiter
import logging

logger = logging.getLogger("aiter")


def init_dist_env(world_size, rankID):
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
        backend="cpu:gloo,cuda:nccl",
    )
    ensure_model_parallel_initialized(world_size, 1)

    # hack custom_allreduce
    tp_grp = get_tp_group()
    ca_comm = tp_grp.ca_comm

    # signal
    signal = torch.zeros(world_size * 64, dtype=torch.int64, device=rankID)

    ca_comm.signal = signal
    ca_comm.register_buffer(signal)
    logger.debug(f"RANK: {rankID}/{world_size} init_dist_env...")


def destroy_dist_env():
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()


def all_reduce_asm(inp: torch.Tensor):
    tp_grp = get_tp_group()
    ca = tp_grp.ca_comm

    if ca._IS_CAPTURING:
        if torch.cuda.is_current_stream_capturing():
            return aiter.all_reduce_asm_(
                inp, ca._ptr, ca.signal, ca.buffer, ca._IS_CAPTURING
            )
        else:
            # if warm up, mimic the allocation pattern
            # since custom allreduce is out-of-place
            return torch.empty_like(inp)
    else:
        # note: outside of cuda graph context,
        # custom allreduce incurs a cost of cudaMemcpy, which should
        # be small(<=1% of overall latency) compared to the performance
        # gains of using custom kernels
        return aiter.all_reduce_asm_(
            inp, ca._ptr, ca.signal, ca.buffer, ca._IS_CAPTURING
        )


def all_reduce_rmsnorm(
    input: Tensor, residual_in: Tensor, weight: Tensor, bias: Tensor, epsilon: float
):
    tp_grp = get_tp_group()
    ca = tp_grp.ca_comm

    return aiter.all_reduce_rmsnorm_(
        input,
        residual_in,
        weight,
        bias,
        epsilon,
        ca._ptr,
        ca.signal,
        ca.buffer,
        ca._IS_CAPTURING,
    )


def all_reduce_rmsnorm_quant(
    input: Tensor,
    residual_in: Tensor,
    xscale: Tensor,
    weight: Tensor,
    bias: Tensor,
    epsilon: float,
):
    tp_grp = get_tp_group()
    ca = tp_grp.ca_comm

    return aiter.all_reduce_rmsnorm_quant_(
        input,
        residual_in,
        xscale,
        weight,
        bias,
        epsilon,
        ca._ptr,
        ca.signal,
        ca.buffer,
        ca._IS_CAPTURING,
    )
