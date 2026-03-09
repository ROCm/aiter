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
def mixed_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperature: Tensor,
    eps: float = 1e-10,
) -> None: ...


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
