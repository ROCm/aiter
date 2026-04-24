# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""
Opus DeepGEMM (FP8 grouped + optional group_layout) Python binding.

Original home was aiter.ops.deepgemm.deepgemm_opus; moved here as part
of the opus module refactor. Semantics and pybind binding are unchanged:
C++ entry `opus_gemm` via JIT module `module_deepgemm_opus`.
"""

from typing import Optional

from torch import Tensor

from ...jit.core import compile_ops


@compile_ops("module_deepgemm_opus", fc_name="opus_gemm")
def deepgemm_opus(
    XQ: Tensor,
    WQ: Tensor,
    Y: Tensor,
    group_layout: Optional[Tensor] = None,
    x_scale: Optional[Tensor] = None,
    w_scale: Optional[Tensor] = None,
) -> Tensor: ...


__all__ = ["deepgemm_opus"]
