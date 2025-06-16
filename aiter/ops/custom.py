# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

from torch import Tensor
from ..jit.core import compile_ops

MD_NAME = "module_custom"


@compile_ops("module_custom")
def wvSpltK(in_a: Tensor, in_b: Tensor, out_c: Tensor, N_in: int, CuCount: int): ...


@compile_ops("module_custom")
def wv_splitk_small_fp16_bf16(
    in_a: Tensor, in_b: Tensor, out_c: Tensor, N_in: int, CuCount: int
): ...


@compile_ops("module_custom")
def LLMM1(in_a: Tensor, in_b: Tensor, out_c: Tensor, rows_per_block: int): ...


@compile_ops("module_custom")
def wvSplitKQ(
    in_a: Tensor,
    in_b: Tensor,
    out_c: Tensor,
    scale_a: Tensor,
    scale_b: Tensor,
    CuCount: int,
): ...
