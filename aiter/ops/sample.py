# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.


import torch
from torch import Generator, Tensor

from ..jit.core import compile_ops


def _philox_seed_offset(tensor: Tensor, generator: Generator | None = None):
    # The compiled kernels are torch-free and take a plain (seed, offset) pair
    # instead of an at::Generator. Mirror what the old C++ host code did with
    # gen->philox_cuda_state(counter_offset): read (seed, offset) from the torch
    # generator, then advance its offset by the same per-launch increment so that
    # back-to-back calls don't reuse the same random numbers.
    if generator is None:
        generator = torch.cuda.default_generators[tensor.device.index]
    numel = tensor.numel()
    M = tensor.size(0)
    # block_size=1024, unroll_factor=4, grid.x=M (see sample_kernels.cu)
    counter_offset = ((numel - 1) // (1024 * M * 4) + 1) * 4
    seed = generator.initial_seed()
    offset = generator.get_offset()
    generator.set_offset(offset + counter_offset)
    return int(seed), int(offset)


@compile_ops("module_sample", develop=True)
def greedy_sample(
    out: Tensor,
    input: Tensor,
) -> None: ...


@compile_ops("module_sample", develop=True)
def random_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperatures: Tensor,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample", fc_name="random_sample", develop=True)
def _random_sample(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    eps: float = 1e-10,
) -> None: ...


def random_sample(
    out: Tensor,
    input: Tensor,
    temperatures: Tensor,
    lambd: float = 1.0,
    generator: Generator | None = None,
    eps: float = 1e-10,
) -> None:
    philox_seed, philox_offset = _philox_seed_offset(input, generator)
    _random_sample(out, input, temperatures, lambd, philox_seed, philox_offset, eps)


@compile_ops("module_sample", develop=True)
def mixed_sample_outer_exponential(
    out: Tensor,
    input: Tensor,
    exponentials: Tensor,
    temperature: Tensor,
    eps: float = 1e-10,
) -> None: ...


@compile_ops("module_sample", fc_name="mixed_sample", develop=True)
def _mixed_sample(
    out: Tensor,
    input: Tensor,
    temperature: Tensor,
    lambd: float = 1.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    eps: float = 1e-10,
) -> None: ...


def mixed_sample(
    out: Tensor,
    input: Tensor,
    temperature: Tensor,
    lambd: float = 1.0,
    generator: Generator | None = None,
    eps: float = 1e-10,
) -> None:
    philox_seed, philox_offset = _philox_seed_offset(input, generator)
    _mixed_sample(out, input, temperature, lambd, philox_seed, philox_offset, eps)


@compile_ops("module_sample", fc_name="exponential", develop=True)
def _exponential(
    out: Tensor,
    lambd: float = 1.0,
    philox_seed: int = 0,
    philox_offset: int = 0,
    eps: float = 1e-10,
) -> None: ...


def exponential(
    out: Tensor,
    lambd: float = 1.0,
    generator: Generator | None = None,
    eps: float = 1e-10,
) -> None:
    philox_seed, philox_offset = _philox_seed_offset(out, generator)
    _exponential(out, lambd, philox_seed, philox_offset, eps)
