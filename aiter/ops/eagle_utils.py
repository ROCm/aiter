# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops, CK_DIR, AITER_CSRC_DIR

MD_NAME = "module_eagle_utils"

@compile_ops(MD_NAME)
def build_tree_kernel(*args, **kwargs) -> Tensor: ...


@compile_ops(MD_NAME)
def build_tree_kernel_efficient(*args, **kwargs) -> Tensor: ...

