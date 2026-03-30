# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional

import torch
from torch import Generator, Tensor

from ..jit.core import compile_ops


@compile_ops("module_sample")
def greedy_sample(
    out: Tensor,
    input: Tensor,
) -> None: ...


@compile_ops("module_sample")
def random_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperatures: Tensor,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample")
def random_sample(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample")
def _mixed_sample_outer_exponential_hip(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperature: Tensor,
    eps: float = 1e-10,
) -> None: ...


def mixed_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperature: Tensor,
    eps: float = 1e-10,
) -> None:
    from aiter.jit.utils.chip_info import get_gfx as _get_gfx
    if _get_gfx().startswith("gfx125"):
        # PyTorch fallback: Gumbel-max trick with mixed greedy/random sampling
        # temperature=0 -> greedy (argmax), temperature>0 -> random (Gumbel)
        import torch as _torch
        temp = temperature.unsqueeze(-1) if temperature.dim() < input.dim() else temperature
        # For greedy rows (temp==0), just argmax
        greedy_mask = (temp.squeeze(-1) == 0)
        # For random rows: logits / temp, then Gumbel-max with pre-sampled exponentials
        safe_temp = temp.clamp(min=eps)
        scaled_logits = input.float() / safe_temp.float()
        # Gumbel-max: argmax(logits/temp - log(exponentials))
        gumbel = scaled_logits - _torch.log(exponentials.float() + eps)
        sampled = gumbel.argmax(dim=-1).to(out.dtype)
        if greedy_mask.any():
            greedy_tokens = input.argmax(dim=-1).to(out.dtype)
            sampled[greedy_mask] = greedy_tokens[greedy_mask]
        out.copy_(sampled)
    else:
        _mixed_sample_outer_exponential_hip(out, input, exponentials, temperature, eps)


@compile_ops("module_sample")
def mixed_sample(
    out: Tensor,
    input: Tensor,
    temperature: Tensor,
    lambd: float = 1.0,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample")
def exponential(
    out: Tensor,
    lambd: float = 1,
    generator: Optional[Generator] = None,
    eps: float = 1e-10,
) -> None: ...
