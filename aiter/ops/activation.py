# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from typing import List, Optional
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR
import torch.nn.functional as F

MD_NAME = "module_activation"

@compile_ops("module_activation")
def silu_and_mul(out: Tensor, input: Tensor): ...

@compile_ops("module_activation")
def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor): ...

@compile_ops("module_activation")
def gelu_and_mul(out: Tensor, input: Tensor): ...

@compile_ops("module_activation")
def gelu_tanh_and_mul(out: Tensor, input: Tensor): ...
