# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import logging
from typing import Optional

# from ..dist.utils import get_open_port, get_distributed_init_method, get_ip
import torch
import torch.distributed as dist

from ..dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,
    init_distributed_environment,
    set_custom_all_reduce,
)

logger = logging.getLogger("aiter")


def init_dist_env(
    tensor_model_parallel_size: int,
    rankID: int,
    backend: str = "cpu:gloo,cuda:nccl",
    distributed_init_method: Optional[str] = "env://",
    local_rank: int = -1,
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
):
    pipeline_model_parallel_size = 1
    # world_size is TPxPP
    world_size = pipeline_model_parallel_size * tensor_model_parallel_size
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rankID,
        distributed_init_method=distributed_init_method,
        # distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()),
        backend=backend,
        local_rank=local_rank,
        data_parallel_size=data_parallel_size,
        data_parallel_rank=data_parallel_rank,
    )
    ensure_model_parallel_initialized(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        data_parallel_size=data_parallel_size,
    )

    if tensor_model_parallel_size > 1:
        # hack custom_allreduce
        tp_grp = get_tp_group()
        ca_comm = tp_grp.device_communicator.ca_comm
        # signal
        signal = torch.zeros(
            tensor_model_parallel_size * 64, dtype=torch.int64, device=rankID
        )
        ca_comm.signal = signal
        ca_comm.register_input_buffer(signal)
        ca_comm.buffer = ca_comm._pool["input"].tensor
    logger.debug(f"RANK: {rankID}/{tensor_model_parallel_size} init_dist_env...")


def destroy_dist_env():
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
