# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.distributed as dist
import triton
from .tp_reduce import FusedReducer
from .gated_residual import _gated_residual


class BufferedProjection:
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
            self.reducer = FusedReducer((rows, 5376), group)
            self.output = self.reducer.output
            self.communication = torch.cuda.Stream() if group is not None else None
            self.last = torch.cuda.Event()
            self.last.record(torch.cuda.current_stream())

    def run(self, inputs, down_weights, residual, gate, indices, custom=True):
        overlap = True
        if inputs.shape != (self.rows, 3584):
            raise ValueError('Expected TP2 [S,3584] attention input')
        if down_weights.shape != (3584, 5376) or down_weights.stride() != (1, 3584):
            raise ValueError('Expected transposed [3584,5376] lower weight')
        if any(tensor.device != self.device or tensor.dtype != torch.bfloat16 or tensor.requires_grad
               for tensor in (inputs, down_weights)):
            raise ValueError('Same-device BF16 inference only')
        if not inputs.is_contiguous():
            raise ValueError('Contiguous input and upper weight required')
        if overlap and self.group is None:
            raise ValueError('Overlap requires TP2 group')
        with torch.cuda.device(self.device):
            compute = torch.cuda.current_stream()
            compute.wait_event(self.last)
            works, events = [], []
            for first in range(0, self.rows, self.width):
                last = min(self.rows, first + self.width)
                target = self.reducer.input[first:last]
                # hipBLASLt beats the Triton out GEMM (out_projection.project_out) at TP2 chunk shapes.
                torch.mm(inputs[first:last], down_weights, out=target)
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
