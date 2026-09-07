# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

"""Host wrapper for the FlyDSL single-tensor intranode push all-to-all."""

from __future__ import annotations

import functools
import os

import flydsl.compiler as flyc
import flydsl.expr as fx
import mori.shmem as ms
import torch
from flydsl.expr.typing import Stream
from mori.shmem import mori_shmem_create_tensor

from .fused_a2a_intranode_kernel import (
    _TRANSPORT_CHUNK_BYTES,
    make_fused_a2a_jit,
    make_fused_a2a_out_jit,
)

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
    """Own symmetric receive and handshake buffers for one tensor shape.

    Set split=True or FUSED_A2A_SPLIT=1 for three ordered per-tensor launches.
    Set quant=True or FUSED_A2A_QUANT=1 to allocate byte payloads and E8M0 scales;
    quantized execution is not implemented yet.
    All ranks must use the same mode and serialize calls on one stream.
    """

    def __init__(
        self,
        *,
        rank,
        world_size,
        shape,
        dtype=torch.bfloat16,
        block_num=_DEFAULT_BLOCK_NUM,
        warp_num_per_block=_DEFAULT_WARP_NUM_PER_BLOCK,
        fuse_norm_rope=True,
        split=False,
        quant=False,
    ):
        self.quant = quant or os.environ.get("FUSED_A2A_QUANT", "0") == "1"
        if dtype != torch.bfloat16 and not (self.quant and dtype == torch.uint8):
            raise ValueError(
                f"expected torch.bfloat16 or torch.uint8 with quant enabled, got {dtype}"
            )
        if world_size <= 0 or world_size > _MAX_INTRANODE_NPES:
            raise ValueError(f"world_size must be in [1, 8], got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 128:
            raise ValueError(f"expected input shape [1, S, H, 128], got {tuple(shape)}")
        if shape[2] % world_size != 0:
            raise ValueError(
                f"head count {shape[2]} must be divisible by world_size {world_size}"
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
        payload_dtype = torch.uint8 if self.quant else dtype
        element_size = torch.tensor([], dtype=payload_dtype).element_size()
        row_nbytes = shape[-1] * element_size
        if row_nbytes % _TRANSPORT_CHUNK_BYTES != 0:
            raise ValueError(
                f"head row must be {_TRANSPORT_CHUNK_BYTES}-byte aligned, got {row_nbytes}"
            )

        self.rank = rank
        self.world_size = world_size
        self.shape = tuple(shape)
        self.dtype = dtype
        self.peer_numel = numel // world_size
        self.outputs_sets = tuple(
            tuple(mori_shmem_create_tensor((numel,), payload_dtype) for _ in range(3))
            for _ in range(2)
        )
        # One E8M0 byte per block of 32 values; left unwritten until quantization.
        self.scales_sets = tuple(
            tuple(
                mori_shmem_create_tensor((numel // 32,), torch.uint8) for _ in range(3)
            )
            for _ in range(2)
        )
        self.output = self.outputs_sets[0][0]
        self.xdb_mem = mori_shmem_create_tensor((world_size,), torch.int64)
        for outputs in self.outputs_sets:
            for output in outputs:
                output.zero_()
        self.xdb_mem.zero_()
        self.xdb_flag = torch.ones(1, dtype=torch.int64, device=self.output.device)
        self.grid_barrier = torch.zeros(1, dtype=torch.int32, device=self.output.device)

        ms.shmem_barrier_all()
        self.p2p_outputs_sets = tuple(
            tuple(
                _build_p2p_table(output, rank, world_size, self.output.device)
                for output in outputs
            )
            for outputs in self.outputs_sets
        )
        self.p2p_scales_sets = tuple(
            tuple(
                _build_p2p_table(scale, rank, world_size, self.output.device)
                for scale in scales
            )
            for scales in self.scales_sets
        )
        self.p2p_xdb_mem = _build_p2p_table(
            self.xdb_mem, rank, world_size, self.output.device
        )
        ms.shmem_barrier_all()
        self._epoch = 0

        self.split = split or os.environ.get("FUSED_A2A_SPLIT", "0") == "1"
        self.fuse_norm_rope = fuse_norm_rope
        roles = (
            (fuse_norm_rope, fuse_norm_rope, False) if self.split else (fuse_norm_rope,)
        )
        self._launches = tuple(
            make_fused_a2a_jit(
                rank=rank,
                npes=world_size,
                heads=shape[2],
                seq_len=shape[1],
                head_dim=shape[3],
                block_num=block_num,
                warp_num_per_block=warp_num_per_block,
                fuse_norm_rope=role,
                split=self.split,
                quant=self.quant,
                element_size=element_size,
            )
            for role in roles
        )
        self._compiled = [None] * len(self._launches)

    def __call__(
        self, q, k, v, norm_q=None, norm_k=None, cos=None, sin=None, stream=None
    ):
        if self.quant:
            raise NotImplementedError(
                "quantized transport is allocation-only scaffolding"
            )
        inputs = (q, k, v)
        for input in inputs:
            if input.dtype != self.dtype or tuple(input.shape) != self.shape:
                raise ValueError(
                    f"expected contiguous {self.dtype} tensor with shape {self.shape}, "
                    f"got {input.dtype} {tuple(input.shape)}"
                )
            if not input.is_cuda or not input.is_contiguous():
                raise ValueError("input must be a contiguous CUDA tensor")
        if self.fuse_norm_rope:
            hd = self.shape[2] * self.shape[3]
            for name, weight in (("norm_q", norm_q), ("norm_k", norm_k)):
                if (
                    weight is None
                    or weight.dtype != self.dtype
                    or tuple(weight.shape) != (hd,)
                ):
                    raise ValueError(
                        f"{name} must be contiguous bf16 with shape ({hd},)"
                    )
                if not weight.is_cuda or not weight.is_contiguous():
                    raise ValueError(f"{name} must be a contiguous CUDA tensor")
            expected_freq_shape = (1, self.shape[1], 1, self.shape[3])
            for name, table in (("cos", cos), ("sin", sin)):
                if (
                    table is None
                    or table.dtype != torch.float32
                    or tuple(table.shape) != expected_freq_shape
                ):
                    raise ValueError(
                        f"{name} must be contiguous fp32 with shape {expected_freq_shape}"
                    )
                if not table.is_cuda or not table.is_contiguous():
                    raise ValueError(f"{name} must be a contiguous CUDA tensor")
        else:
            norm_q = norm_k = cos = sin = q

        parity = self._epoch % 2
        outputs = self.outputs_sets[parity]
        stream = Stream(torch.cuda.current_stream() if stream is None else stream)
        sync_args = (
            self.xdb_mem.data_ptr(),
            self.p2p_xdb_mem.data_ptr(),
            self.xdb_flag.data_ptr(),
            self.grid_barrier.data_ptr(),
            stream,
        )
        tables = self.p2p_outputs_sets[parity]
        if self.split:
            launch_args = tuple(
                (
                    input.data_ptr(),
                    norm.data_ptr(),
                    cos.data_ptr(),
                    sin.data_ptr(),
                    table.data_ptr(),
                    *sync_args,
                )
                for input, norm, table in zip(inputs, (norm_q, norm_k, v), tables)
            )
        else:
            launch_args = (
                (
                    *(input.data_ptr() for input in inputs),
                    norm_q.data_ptr(),
                    norm_k.data_ptr(),
                    cos.data_ptr(),
                    sin.data_ptr(),
                    *(table.data_ptr() for table in tables),
                    *sync_args,
                ),
            )
        for i, args in enumerate(launch_args):
            if self._compiled[i] is None:
                self._compiled[i] = flyc.compile(
                    self._launches[i],
                    *(fx.Int64(arg) for arg in args[:-1]),
                    args[-1],
                )
            else:
                self._compiled[i](*args)
        self._epoch += 1
        return outputs


class FusedA2AOutIntraNodeOp:
    """Own symmetric buffers for the inverse Ulysses out-hop."""

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
        if len(shape) != 4 or shape[0] != 1 or shape[-1] != 128:
            raise ValueError(f"expected input shape [1, H, S, 128], got {tuple(shape)}")
        if shape[2] % world_size != 0:
            raise ValueError(
                f"sequence length {shape[2]} must divide by world_size {world_size}"
            )
        cu_count = _device_cu_count(torch.cuda.current_device())
        if block_num > cu_count:
            raise ValueError(
                f"block_num={block_num} exceeds device CU count ({cu_count}); "
                "the grid-wide barrier requires all blocks to be resident"
            )

        numel = 1
        for dim in shape:
            numel *= dim
        self.shape = tuple(shape)
        self.dtype = dtype
        self.outputs = tuple(
            mori_shmem_create_tensor((numel,), dtype) for _ in range(2)
        )
        self.output = self.outputs[0]
        self.xdb_mem = mori_shmem_create_tensor((world_size,), torch.int64)
        for output in self.outputs:
            output.zero_()
        self.xdb_mem.zero_()
        self.xdb_flag = torch.ones(1, dtype=torch.int64, device=self.output.device)
        self.grid_barrier = torch.zeros(1, dtype=torch.int32, device=self.output.device)

        ms.shmem_barrier_all()
        self.p2p_outputs = tuple(
            _build_p2p_table(output, rank, world_size, self.output.device)
            for output in self.outputs
        )
        self.p2p_xdb_mem = _build_p2p_table(
            self.xdb_mem, rank, world_size, self.output.device
        )
        ms.shmem_barrier_all()
        self._epoch = 0

        self._launch = make_fused_a2a_out_jit(
            rank=rank,
            npes=world_size,
            heads_local=shape[1],
            seq_local=shape[2] // world_size,
            head_dim=shape[3],
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

        parity = self._epoch % 2
        output = self.outputs[parity]
        stream = Stream(torch.cuda.current_stream() if stream is None else stream)
        args = (
            input.data_ptr(),
            self.p2p_outputs[parity].data_ptr(),
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
        self._epoch += 1
        return output
