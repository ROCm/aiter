# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_aiter_operator"

@compile_ops("module_aiter_operator")
def add(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator")
def sub(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator")
def mul(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator")
def div(input: Tensor, other: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator")
def sigmoid(input: Tensor) -> Tensor: ...


@compile_ops("module_aiter_operator")
def tanh(input: Tensor) -> Tensor: ...
