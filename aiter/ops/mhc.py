# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from typing import Optional
from ..jit.core import compile_ops


@compile_ops("module_mhc")
def mhc_pre_gemm_sqrsum(
    out: Tensor,
    sqrsum: Tensor,
    x: Tensor,
    fn: Tensor,
    tile_k: int = 128,  # 64 or 128
) -> None: ...


@compile_ops("module_mhc")
def mhc_pre_big_fuse(
    post_mix: Tensor,
    comb_mix: Tensor,
    layer_input: Tensor,
    gemm_out_mul: Tensor,
    gemm_out_sqrsum: Tensor,
    hc_scale: Tensor,
    hc_base: Tensor,
    residual: Tensor,
    rms_eps: float = 1e-6,
    hc_pre_eps: float = 1e-6,
    hc_sinkhorn_eps: float = 1e-6,
    hc_post_mult_value: float = 1.0,
    sinkhorn_repeat: int = 20,
) -> None: ...
