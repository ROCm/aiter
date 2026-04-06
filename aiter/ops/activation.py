# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import torch.nn.functional as F
from torch import Tensor


def silu_and_mul(out: Tensor, input: Tensor) -> None:
    d = input.shape[-1] // 2
    gate = input[..., :d]
    up = input[..., d:]
    out.copy_(F.silu(gate) * up)


def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor) -> None:
    d = input.shape[-1] // 2
    gate = input[..., :d]
    up = input[..., d:]
    out.copy_(F.silu(gate) * up * scale)


def gelu_and_mul(out: Tensor, input: Tensor) -> None:
    d = input.shape[-1] // 2
    gate = input[..., :d]
    up = input[..., d:]
    out.copy_(F.gelu(gate) * up)


def gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None:
    d = input.shape[-1] // 2
    gate = input[..., :d]
    up = input[..., d:]
    out.copy_(F.gelu(gate, approximate="tanh") * up)


def gelu_fast(out: Tensor, input: Tensor) -> None:
    out.copy_(F.gelu(input, approximate="tanh"))
