# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL single-tensor intranode push all-to-all."""

from __future__ import annotations

import functools

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.expr.typing import Stream
from mori.shmem import mori_shmem_create_tensor

from .fused_a2a_intranode_kernel import make_fused_a2a_jit

_DEFAULT_BLOCK_NUM = 128
_DEFAULT_WARP_NUM_PER_BLOCK = 8
_MAX_INTRANODE_NPES = 8


@functools.cache
def _device_cu_count(device_index):
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def _build_p2p_table(tensor, rank, world_size, device):
    table = torch.zeros(world_size, dtype=torch.int64, device=device)
    for peer in range(world_size):
        table[peer] = ms.shmem_ptr_p2p(tensor.data_ptr(), rank, peer)
    return table


class FusedA2AIntraNodeOp:
    """Own the symmetric receive and handshake buffers for one tensor shape."""

    def __init__(
        self,
        *,
        rank,
        world_size,
        shape,
        dtype=torch.bfloat16,
        block_num=_DEFAULT_BLOCK_NUM,
        warp_num_per_block=_DEFAULT_WARP_NUM_PER_BLOCK,
    ):
        if dtype != torch.bfloat16:
            raise ValueError(f"only torch.bfloat16 is supported, got {dtype}")
        if world_size <= 0 or world_size > _MAX_INTRANODE_NPES:
            raise ValueError(f"world_size must be in [1, 8], got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 128:
            raise ValueError(f"expected input shape [1, H, S, 128], got {tuple(shape)}")
        if shape[1] % world_size != 0:
            raise ValueError(
                f"head count {shape[1]} must be divisible by world_size {world_size}"
            )
        if block_num <= 0 or warp_num_per_block <= 0:
            raise ValueError("launch geometry must be positive")
        cu_count = _device_cu_count(torch.cuda.current_device())
        if block_num > cu_count:
            raise ValueError(
                f"block_num={block_num} exceeds device CU count ({cu_count}); "
                "the grid-wide barrier requires all blocks to be resident"
            )

        numel = 1
        for dim in shape:
            numel *= dim
        if numel % world_size != 0:
            raise ValueError("input numel must divide evenly across ranks")
        peer_nbytes = numel * torch.tensor([], dtype=dtype).element_size() // world_size
        if peer_nbytes % 16 != 0:
            raise ValueError(
                f"per-peer chunk must be 16-byte aligned, got {peer_nbytes}"
            )

        self.rank = rank
        self.world_size = world_size
        self.shape = tuple(shape)
        self.dtype = dtype
        self.peer_numel = numel // world_size
        self.output = mori_shmem_create_tensor((numel,), dtype)
        self.xdb_mem = mori_shmem_create_tensor((world_size,), torch.int64)
        self.output.zero_()
        self.xdb_mem.zero_()
        self.xdb_flag = torch.ones(1, dtype=torch.int64, device=self.output.device)
        self.grid_barrier = torch.zeros(1, dtype=torch.int32, device=self.output.device)

        ms.shmem_barrier_all()
        self.p2p_output = _build_p2p_table(
            self.output, rank, world_size, self.output.device
        )
        self.p2p_xdb_mem = _build_p2p_table(
            self.xdb_mem, rank, world_size, self.output.device
        )
        ms.shmem_barrier_all()

        self._launch = make_fused_a2a_jit(
            rank=rank,
            npes=world_size,
            peer_nbytes=peer_nbytes,
            block_num=block_num,
            warp_num_per_block=warp_num_per_block,
        )
        self._compiled = None

    def __call__(self, input, stream=None):
        if input.dtype != self.dtype or tuple(input.shape) != self.shape:
            raise ValueError(
                f"expected contiguous {self.dtype} tensor with shape {self.shape}, "
                f"got {input.dtype} {tuple(input.shape)}"
            )
        if not input.is_cuda or not input.is_contiguous():
            raise ValueError("input must be a contiguous CUDA tensor")

        stream = Stream(torch.cuda.current_stream() if stream is None else stream)
        args = (
            input.data_ptr(),
            self.output.data_ptr(),
            self.p2p_output.data_ptr(),
            self.xdb_mem.data_ptr(),
            self.p2p_xdb_mem.data_ptr(),
            self.xdb_flag.data_ptr(),
            self.grid_barrier.data_ptr(),
            stream,
        )
        if self._compiled is None:
            self._compiled = flyc.compile(
                self._launch,
                *(fx.Int64(arg) for arg in args[:-1]),
                args[-1],
            )
        else:
            self._compiled(*args)
        return self.output
