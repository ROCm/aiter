# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import logging

# from ..dist.utils import get_open_port, get_distributed_init_method, get_ip
import torch
import torch.distributed as dist

from ..dist.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
    ensure_model_parallel_initialized,
    get_tp_group,  # noqa: F401 -- re-exported; downstreams import via this module
    init_distributed_environment,
    set_custom_all_reduce,
)

logger = logging.getLogger("aiter")


def init_dist_env(
    tensor_model_parallel_size: int,
    rankID: int,
    backend: str = "cpu:gloo,cuda:nccl",
    distributed_init_method: str | None = "env://",
    local_rank: int = -1,
    data_parallel_size: int = 1,
    data_parallel_rank: int = 0,
    decode_context_parallel_size: int = 1,
    prefill_context_model_parallel_size: int = 1,
):
    pipeline_model_parallel_size = 1
    # world_size is TP x PP x PCP (PCP is an independent dimension that grows
    # world_size; see initialize_model_parallel in dist/parallel_state.py).
    world_size = (
        pipeline_model_parallel_size
        * tensor_model_parallel_size
        * prefill_context_model_parallel_size
    )
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
        decode_context_model_parallel_size=decode_context_parallel_size,
        data_parallel_size=data_parallel_size,
        prefill_context_model_parallel_size=prefill_context_model_parallel_size,
    )

    # No per-rank signal/input-buffer registration here. An earlier version
    # registered a torch.zeros "signal" tensor with CustomAllreduce and mirrored
    # the input pool as `ca_comm.buffer`. All of it was vestigial:
    #   * `ca_comm.signal` / `ca_comm.buffer` are never read anywhere;
    #   * register_input_buffer only inserts a pointer-translation entry keyed
    #     by the registered tensor's own address, consulted when an allreduce
    #     is invoked with that exact tensor as input -- which never happens for
    #     the signal tensor;
    #   * gfx1250 has skipped the whole block (deadlock in its vmm_exchange
    #     rendezvous) and works without it.
    # It was also what made raw IPC input pools unusable (#4921):
    #   * under PYTORCH_HIP_ALLOC_CONF=expandable_segments:True the torch.zeros
    #     signal is VMM-backed, so registering it dies in hipIpcGetMemHandle
    #     (custom_all_reduce.cu:417, "invalid argument");
    #   * the raw_cached input pool has no backing torch.Tensor, so
    #     `ca_comm.buffer = ca_comm._pool["input"].tensor` raised by design.
    # CustomAllreduce.__init__ already sets up its own meta/input pools and the
    # copy-in path for expandable segments (#4174); nothing further is needed.
    logger.debug(f"RANK: {rankID}/{tensor_model_parallel_size} init_dist_env...")


def destroy_dist_env():
    if dist.is_initialized():
        destroy_model_parallel()
        destroy_distributed_environment()
        torch.cuda.empty_cache()
