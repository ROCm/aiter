'''
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
 '''

from contextlib import contextmanager
from typing import Any, List, Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

# import vllm.envs as envs
# from vllm import _custom_ops as ops
import aiter as ops
import os
from .custom_all_reduce_utils import (
    gpu_p2p_access_check)
from .parallel_state import in_the_same_node_as
from aiter import logger

try:
    ops.meta_size()
    custom_ar = True
except Exception:
    # For CPUs
    custom_ar = False


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if not gpu_p2p_access_check(rank, i):
            return False
    return True


def is_weak_contiguous(inp: torch.Tensor):
    return inp.is_contiguous() or (inp.storage().nbytes() -
                                   inp.storage_offset() * inp.element_size()
                                   == inp.numel() * inp.element_size())


class CustomAllreduce:

    _SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]

    # max_size: max supported allreduce size
    def __init__(self,
                 group: ProcessGroup,
                 device: Union[int, str, torch.device],
                 max_size=8192 * 1024 * 8) -> None:
        """
        Args:
            group: the process group to work on. If None, it will use the
                default process group.
            device: the device to bind the CustomAllreduce to. If None,
                it will be bind to f"cuda:{local_rank}".
        It is the caller's responsibility to make sure each communicator
        is bind to a unique device, and all communicators in this group
        are in the same node.
        """
        self._IS_CAPTURING = False
        self.disabled = True

        if not custom_ar:
            # disable because of missing custom allreduce library
            # e.g. in a non-cuda environment
            return

        self.group = group

        assert dist.get_backend(group) != dist.Backend.NCCL, (
            "CustomAllreduce should be attached to a non-NCCL group.")

        if not all(in_the_same_node_as(group, source_rank=0)):
            # No need to initialize custom allreduce for multi-node case.
            logger.warning(
                "Custom allreduce is disabled because this process group"
                " spans across nodes.")
            return

        rank = dist.get_rank(group=self.group)
        world_size = dist.get_world_size(group=self.group)
        if world_size == 1:
            # No need to initialize custom allreduce for single GPU case.
            return

        if world_size not in CustomAllreduce._SUPPORTED_WORLD_SIZES:
            logger.warning(
                "Custom allreduce is disabled due to an unsupported world"
                " size: %d. Supported world sizes: %s. To silence this "
                "warning, specify disable_custom_all_reduce=True explicitly.",
                world_size, str(CustomAllreduce._SUPPORTED_WORLD_SIZES))
            return

        if isinstance(device, int):
            device = torch.device(f"cuda:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device

        cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "-1")
        # cuda_visible_devices = envs.CUDA_VISIBLE_DEVICES
        if cuda_visible_devices:
            device_ids = list(map(int, cuda_visible_devices.split(",")))
        else:
            from vllm.utils import cuda_device_count_stateless
            device_ids = list(range(cuda_device_count_stateless()))

        physical_device_id = device_ids[device.index]
        tensor = torch.tensor([physical_device_id],
                              dtype=torch.int,
                              device="cpu")
        gather_list = [
            torch.tensor([0], dtype=torch.int, device="cpu")
            for _ in range(world_size)
        ]
        dist.all_gather(gather_list, tensor, group=self.group)
        physical_device_ids = [t.item() for t in gather_list]

        # test nvlink first, this will filter out most of the cases
        # where custom allreduce is not supported
        # this checks hardware and driver support for NVLink
        # assert current_platform.is_cuda() or current_platform.is_rocm()
        # full_nvlink = current_platform.is_full_nvlink(physical_device_ids)
        full_nvlink = True
        if world_size > 2 and not full_nvlink:
            logger.warning(
                "Custom allreduce is disabled because it's not supported on"
                " more than two PCIe-only GPUs. To silence this warning, "
                "specify disable_custom_all_reduce=True explicitly.")
            return
        # test P2P capability, this checks software/cudaruntime support
        # this is expensive to compute at the first time
        # then we cache the result
        # On AMD GPU, p2p is always enabled between XGMI connected GPUs
        # if not current_platform.is_rocm() and not _can_p2p(rank, world_size):
        #     logger.warning(
        #         "Custom allreduce is disabled because your platform lacks "
        #         "GPU P2P capability or P2P test failed. To silence this "
        #         "warning, specify disable_custom_all_reduce=True explicitly.")
        #     return

        self.disabled = False
        # buffers memory are owned by this Python class and passed to C++
        # meta data composes of two parts: meta data for synchronization
        # (256 bytes) and a temporary buffer for storing intermediate
        # allreduce results.
        # if current_platform.is_rocm():
        if 1:
            # meta data buffers need to be "uncached" for signal on MI200
            self.meta = ops.allocate_meta_buffer(ops.meta_size() + max_size)
        else:
            self.meta = torch.zeros(ops.meta_size() + max_size,
                                    dtype=torch.uint8,
                                    device=self.device)
        # This is a pre-registered IPC buffer. In eager mode, input tensors
        # are first copied into this buffer before allreduce is performed
        self.buffer = torch.empty(max_size,
                                  dtype=torch.uint8,
                                  device=self.device)
        # This is a buffer for storing the tuples of pointers pointing to
        # IPC buffers from all ranks. Each registered tuple has size of
        # 8*world_size bytes where world_size is at most 8. Allocating 8MB
        # is enough for 131072 such tuples. The largest model I've seen only
        # needs less than 10000 of registered tuples.
        self.rank_data = torch.empty(8 * 1024 * 1024,
                                     dtype=torch.uint8,
                                     device=self.device)
        self.max_size = max_size
        self.rank = rank
        self.world_size = world_size
        # if current_platform.is_rocm():
        if 1:
            # _share_cuda_() doesn't accept meta buffer not allocated from
            # PyTorch cache allocator, use direct HIP call to get IPC handle
            handle = ops.get_meta_buffer_ipc_handle(self.meta)
            shard_data = (
                bytes(handle),  # ipc handle to base ptr
                0,  # offset of base ptr
            )
            handles, offsets = self._gather_ipc_meta(shard_data)
        else:
            handles, offsets = self._get_ipc_meta(self.meta)
        self.full_nvlink = full_nvlink
        self._ptr = ops.init_custom_ar(self.meta, self.rank_data, handles,
                                       offsets, rank, self.full_nvlink)
        self.register_buffer(self.buffer)

    @contextmanager
    def capture(self):
        """
        The main responsibility of this context manager is the
        `register_graph_buffers` call at the end of the context.
        It records all the buffer addresses used in the CUDA graph.
        """
        try:
            self._IS_CAPTURING = True
            yield
        finally:
            self._IS_CAPTURING = False
            if not self.disabled:
                self.register_graph_buffers()

    def _get_ipc_meta(self, inp: torch.Tensor):
        # if current_platform.is_rocm():
        if 1:
            # _share_cuda_() doesn't accept meta buffer not allocated from
            # PyTorch cache allocator, use direct HIP call to get IPC handle
            handle = ops.get_meta_buffer_ipc_handle(inp)
            shard_data = (
                bytes(handle),  # ipc handle to base ptr
                0,  # offset of base ptr
            )
        else:
            data = inp.untyped_storage()._share_cuda_()
            shard_data = (
                data[1],  # ipc handle to base ptr
                data[3],  # offset of base ptr
            )
        return self._gather_ipc_meta(shard_data)

    def _gather_ipc_meta(self, shard_data):
        # Note: don't use `[[None]] * self.world_size` here
        # because it will create a list of the same reference
        all_data: List[Optional[Any]] = [[None]
                                         for i in range(self.world_size)]
        all_data[self.rank][0] = shard_data

        ranks = dist.get_process_group_ranks(group=self.group)
        ranks.sort()
        for i, rank in enumerate(ranks):
            dist.broadcast_object_list(all_data[i],
                                       src=rank,
                                       group=self.group,
                                       device="cpu")

        # we cannot directly use `dist.all_gather_object` here
        # because it is incompatible with `gloo` backend under inference mode.
        # see https://github.com/pytorch/pytorch/issues/126032 for details.

        handles = []
        offsets = []
        for i in range(len(all_data)):
            handles.append(all_data[i][0][0])  # type: ignore
            offsets.append(all_data[i][0][1])  # type: ignore
        return handles, offsets

    def register_buffer(self, inp: torch.Tensor):
        handles, offsets = self._get_ipc_meta(inp)
        ops.register_buffer(self._ptr, inp, handles, offsets)

    def register_graph_buffers(self):
        handle, offset = ops.get_graph_buffer_ipc_meta(self._ptr)
        handles, offsets = self._gather_ipc_meta((bytes(handle), offset))
        logger.info("Registering %d cuda graph addresses", len(offset))
        ops.register_graph_buffers(self._ptr, handles, offsets)

    def should_custom_ar(self, inp: torch.Tensor):
        if self.disabled:
            return False
        inp_size = inp.numel() * inp.element_size()
        # custom allreduce requires input byte size to be multiples of 16
        if inp_size % 16 != 0:
            return False
        if not is_weak_contiguous(inp):
            return False
        # for 4 or more non NVLink-capable GPUs, custom allreduce provides
        # little performance improvement over NCCL.
        if self.world_size == 2 or self.full_nvlink:
            return inp_size < self.max_size
        return False

    # all reduce, assuming inp tensor is IPC registered with register_buffer,
    # or, in the context of cuda graphs, register_graph_buffers
    def all_reduce_reg(self, inp: torch.Tensor, out: torch.Tensor = None, open_fp8_quant: bool = False):
        if out is None:
            out = torch.empty_like(inp)
        ops.all_reduce_reg(self._ptr, inp, out, open_fp8_quant)
        return out

    # all reduce, assuming inp tensor is NOT IPC registered
    def all_reduce_unreg(self, inp: torch.Tensor, out: torch.Tensor = None):
        if out is None:
            out = torch.empty_like(inp)
        ops.all_reduce_unreg(self._ptr, inp, self.buffer, out)
        return out

    def custom_all_reduce(self, input: torch.Tensor, open_fp8_quant: bool) -> Optional[torch.Tensor]:
        # when custom allreduce is disabled, this will be None
        if self.disabled or not self.should_custom_ar(input):
            return None
        if self._IS_CAPTURING:
            if torch.cuda.is_current_stream_capturing():
                return self.all_reduce_reg(input, open_fp8_quant = open_fp8_quant)
            else:
                # if warm up, mimic the allocation pattern
                # since custom allreduce is out-of-place
                return torch.empty_like(input)
        else:
            # note: outside of cuda graph context,
            # custom allreduce incurs a cost of cudaMemcpy, which should
            # be small(<=1% of overall latency) compared to the performance
            # gains of using custom kernels
            return self.all_reduce_unreg(input)

        return None

    def close(self):
        if not self.disabled and self._ptr:
            ops.dispose(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.close()
