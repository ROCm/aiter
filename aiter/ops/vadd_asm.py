# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor

from ..jit.core import compile_ops


@compile_ops("module_vadd_asm")
def vadd_asm(a: Tensor, b: Tensor, c: Tensor) -> None: ...
