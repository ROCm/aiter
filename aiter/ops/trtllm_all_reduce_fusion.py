# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
import torch.distributed as dist
from torch.distributed import ProcessGroup
from ..jit.core import compile_ops


@compile_ops("module_trtllm_all_reduce_fusion")
def init_trtllm_ar_fusion(
    rank: int,
    world_size: int,
    max_size_in_bytes: int,
) -> int: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def destroy_trtllm_ar_fusion(
    fptr_t: int,
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def get_trtllm_ar_fusion_handle(
    fptr_t: int,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def open_trtllm_ar_fusion_handles(
    fptr_t: int,
    handles: list[Tensor],
) -> None: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def get_trtllm_ar_fusion_workspace(
    fptr_t: int,
    ref: Tensor,
) -> Tensor: ...


@compile_ops("module_trtllm_all_reduce_fusion")
def trtllm_allreduce_rms(
    rank: int,
    nranks: int,
    allreduce_in: Tensor,
    residual_in: Tensor,
    rms_gamma: Tensor,
    residual_out: Tensor,
    norm_out: Tensor,
    eps: float,
    fp8_out: bool,
    workspace: Tensor,
) -> None: ...


class TRTLLMAllreduceFusion:
    _SUPPORTED_WORLD_SIZES = [2, 4, 8]

    # max_size: max supported allreduce size
    def __init__(
        self,
        group: ProcessGroup = None,
        max_size_in_bytes=8192 * 16384,
    ) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self.group = group
        rank = dist.get_rank(group=self.group)
        torch.cuda.set_device(rank)
        self.rank = rank
        self.fptr = None
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            return

        if world_size not in TRTLLMAllreduceFusion._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "TRTLLMAllreduceFusion is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s.",
                world_size,
                str(TRTLLMAllreduceFusion._SUPPORTED_WORLD_SIZES),
            )
            return

        torch.cuda.set_device(rank)
        self.fptr = init_trtllm_ar_fusion(rank, world_size, max_size_in_bytes)
        handle = get_trtllm_ar_fusion_handle(self.fptr)
        handle_list = [None] * world_size
        dist.all_gather_object(handle_list, handle, group=group)
        open_trtllm_ar_fusion_handles(self.fptr, handle_list)
        torch.cuda.synchronize(rank)
        dist.barrier(group=group)
        # print(f"init TRTLLMAllreduceFusion at rank:{rank}", flush=True)

    def get_workspace(self, ref: torch.Tensor):
        return get_trtllm_ar_fusion_workspace(self.fptr, ref)

    def __del__(self):
        if self.fptr:
            destroy_trtllm_ar_fusion(self.fptr)
