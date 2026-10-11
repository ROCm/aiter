# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import triton
from .tp_reduce import FusedReducer
from .gated_residual import _gated_residual

from .ffn_up import fused_up_out
from .ffn_down import down_out


class BufferedFFN:
    def __init__(self, rows, token_rows, device, group=None):
        if rows < 1 or token_rows < 1 or rows * 7168 >= 2**31:
            raise ValueError('Positive dimensions within int32 indexing required')
        self.rows = rows
        self.width = min(rows, token_rows)
        self.group = group
        self.device = torch.device(device)
        if group is not None and dist.get_world_size(group) != 2:
            raise ValueError('TP2 only')
        with torch.cuda.device(self.device):
            self.activation = torch.empty((self.width, 7168), device=self.device, dtype=torch.bfloat16)
            self.reducer = FusedReducer((rows, 5376), group)
            self.output = self.reducer.output
            self.communication = torch.cuda.Stream() if group is not None else None
            self.last = torch.cuda.Event()
            self.last.record(torch.cuda.current_stream())

    def run(self, inputs, up_weights, down_weights, residual, gate, indices, custom=True):
        overlap = True
        if inputs.shape != (self.rows, 5376) or up_weights.shape != (14336, 5376):
            raise ValueError('Expected TP2 [S,5376] input and paired [14336,5376] upper weight')
        if down_weights.shape != (7168, 5376) or down_weights.stride() != (1, 7168):
            raise ValueError('Expected transposed [7168,5376] lower weight')
        if any(tensor.device != self.device or tensor.dtype != torch.bfloat16 or tensor.requires_grad
               for tensor in (inputs, up_weights, down_weights)):
            raise ValueError('Same-device BF16 inference only')
        if not inputs.is_contiguous() or not up_weights.is_contiguous():
            raise ValueError('Contiguous input and upper weight required')
        if overlap and self.group is None:
            raise ValueError('Overlap requires TP2 group')
        with torch.cuda.device(self.device):
            compute = torch.cuda.current_stream()
            compute.wait_event(self.last)
            works, events = [], []
            for first in range(0, self.rows, self.width):
                last = min(self.rows, first + self.width)
                activation = self.activation[:last - first]
                target = self.reducer.input[first:last]
                fused_up_out(inputs[first:last], up_weights, activation)
                down_out(activation, down_weights, target)
                if self.group is not None:
                    if overlap:
                        ready = torch.cuda.Event()
                        ready.record(compute)
                        events.append(ready)
                        with torch.cuda.stream(self.communication):
                            self.communication.wait_event(ready)
                            if custom:
                                self.reducer.reduce(residual, gate, indices, first * 5376, (last - first) * 5376)
                            else:
                                work = dist.all_reduce(target, group=self.group, async_op=True)
                                work.wait()
                                works.append(work)
                                _gated_residual[(triton.cdiv(target.numel(), 1024),)](
                                    residual[first:last], target, gate, indices[first:last], self.output[first:last],
                                    target.numel(), 5376, gate.stride(0), BLOCK=1024,
                                    num_warps=4, enable_fp_fusion=False)
                    else:
                        dist.all_reduce(target, group=self.group)
            if overlap:
                complete = torch.cuda.Event()
                complete.record(self.communication)
                compute.wait_event(complete)
            self.last = torch.cuda.Event()
            self.last.record(compute)
        return self.output
