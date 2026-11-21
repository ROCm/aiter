# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
from .ffn_pipeline import BufferedFFN
from .projection_pipeline import BufferedProjection
from .modulation import modulate


class H3TP2Workspace:
    def __init__(self, rows, device, group):
        self.rows = rows
        self.device = torch.device(device)
        self.ffn = BufferedFFN(rows, 8192, device, group)
        self.projection = BufferedProjection(rows, triton.cdiv(rows, 1024) * 128, device, group)

    def feed_forward(self, inputs, up_weights, down_weights, residual, gate, indices):
        return self.ffn.run(inputs, up_weights, down_weights, residual, gate, indices)

    def attention_output(self, inputs, weights, residual, gate, indices):
        return self.projection.run(inputs, weights, residual, gate, indices)
