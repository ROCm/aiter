# Copyright (c) 2024 Advanced Micro Devices, Inc.  All rights reserved.
#
# -*- coding:utf-8 -*-
# @Script: communication.py
# @Author: valarLip
# @Email: lingpeng.jin@amd.com
# @Create At: 2024-12-14 15:47:26
# @Last Modified By: valarLip
# @Last Modified At: 2024-12-19 15:33:35
# @Description: This is description.

import torch
import torch.distributed as dist
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, ATER_CSRC_DIR, ATER_ROOT_DIR
from ..dist.parallel_state import (ensure_model_parallel_initialized,
                                   init_distributed_environment,
                                   set_custom_all_reduce,
                                   get_tp_group,
                                   graph_capture,
                                   destroy_model_parallel,
                                   destroy_distributed_environment)
from ..dist.utils import (get_open_port,
                          get_distributed_init_method,
                          get_ip)
import ater
import logging
logger = logging.getLogger("ater")


def call_all_reduce_asm(input: torch.Tensor):
    tp_grp = get_tp_group()
    ca_comm = tp_grp.ca_comm
    # TODO check input size <= pre allocated buffer
    ca_comm.signal.fill_(0)
    torch.cuda.synchronize()
    dist.barrier()
    return ca_comm.all_reduce_asm(input)


def init_dist_env_asm(world_size, rankID):
    logger.info(
        f"RANK: {rankID}/{world_size} init_process_group...")
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world_size,
        rank=rankID,
        distributed_init_method=get_distributed_init_method(get_ip(), get_open_port()))
    ensure_model_parallel_initialized(world_size, 1)

    # hack custom_allreduce
    tp_grp = get_tp_group()
    ca_comm = tp_grp.ca_comm

    # signal
    signal = torch.zeros(world_size*2,
                         dtype=torch.int64,
                         device=rankID)

    ca_comm.signal = signal
    ca_comm.register_buffer(signal)


def destroy_dist_env():
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
