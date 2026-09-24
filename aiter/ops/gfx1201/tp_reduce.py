# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm
import triton
import triton.language as tl


@triton.jit
def _sum_half(local, peer, scratch, first, count, RANK: tl.constexpr, BLOCK: tl.constexpr):
    half = (count + 1) // 2
    begin = RANK * half
    length = tl.minimum(half, count - begin)
    for tile in range(tl.program_id(0), tl.cdiv(length, BLOCK), tl.num_programs(0)):
        position = tile * BLOCK + tl.arange(0, BLOCK)
        offsets = first + begin + position
        left = tl.load(local + offsets, position < length, other=0).to(tl.float32)
        right = tl.load(peer + offsets, position < length, other=0).to(tl.float32)
        total = (left + right).to(tl.bfloat16, fp_downcast_rounding='rtne')
        tl.store(scratch + offsets, total, position < length)


@triton.jit
def _gather_halves(scratch, peer, output, residual, gate, indices, dimension: tl.constexpr, gate_stride: tl.constexpr, first, count, RANK: tl.constexpr, BLOCK: tl.constexpr):
    half = (count + 1) // 2
    for tile in range(tl.program_id(0), tl.cdiv(count, BLOCK), tl.num_programs(0)):
        position = tile * BLOCK + tl.arange(0, BLOCK)
        local_half = (position < half) if RANK == 0 else (position >= half)
        offsets = first + position
        local_value = tl.load(scratch + offsets, position < count, other=0).to(tl.float32)
        peer_value = tl.load(peer + offsets, position < count, other=0).to(tl.float32)
        valid = position < count
        reduced = (local_value + peer_value).to(tl.bfloat16, fp_downcast_rounding='rtne').to(tl.float32)
        group = tl.load(indices + offsets // dimension, valid, other=0)
        scale = tl.load(gate + group * gate_stride + offsets % dimension, valid, other=0).to(tl.float32)
        product = (reduced * scale).to(tl.bfloat16).to(tl.float32)
        original = tl.load(residual + offsets, valid, other=0).to(tl.float32)
        tl.store(output + offsets, original + product, valid)


class FusedReducer:
    def __init__(self, shape, group, blocks=8):
        if dist.get_world_size(group) != 2 or blocks < 1:
            raise ValueError('TP2 and positive block count required')
        self.rank = dist.get_rank(group)
        self.blocks = blocks
        self.input = symm.empty(shape, device=torch.device('cuda', torch.cuda.current_device()), dtype=torch.bfloat16)
        self.output = torch.empty_like(self.input)
        self.input_handle = symm.rendezvous(self.input, group)
        self.peer_input = self.input_handle.get_buffer(1 - self.rank, tuple(shape), torch.bfloat16)

    def reduce(self, residual, gate, indices, first=0, count=None):
        if count is None:
            count = self.input.numel() - first
        if first < 0 or count < 1 or first + count > self.input.numel():
            raise ValueError('Invalid flat reduction range')
        with torch.cuda.device(self.input.device):
            self.input_handle.barrier(timeout_ms=30000)
            _gather_halves[(32,)](self.input, self.peer_input, self.output, residual, gate, indices, residual.shape[1], gate.stride(0),
                                           first, count, self.rank, BLOCK=4096, num_warps=4, enable_fp_fusion=False)
            self.input_handle.barrier(timeout_ms=30000)
        return self.output
