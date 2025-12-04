# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

from typing import Optional
import torch
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.fused_indices_gather import (
    _fused_indices_and_gather_kernel,
    _get_config,
)
from aiter.ops.triton.utils.logger import AiterTritonLogger

_LOGGER = AiterTritonLogger()


def fuse_indices_and_gather(x, E, a, D, config: Optional[dict] = None, num_ctas=1):
    (B, T, D) = x.shape
    x2d = x.view(-1, D)
    idx1d = torch.empty(E * a, dtype=torch.long, device=x.device)
    gather_out = torch.empty([E * a, D], device=x.device, dtype=x.dtype)

    # compute strides for accessing elems in Triton
    sx2d0 = x2d.stride(0)
    sx2d1 = x2d.stride(1)
    sidx = idx1d.stride(0)
    sgatherout0 = gather_out.stride(0)
    sgatherout1 = gather_out.stride(1)

    if config is None:
        _get_config.cache_clear()
        if hasattr(_get_config, "_config_dict"):
            delattr(_get_config, "_config_dict")
        config = _get_config(B, E)

    grid = (triton.cdiv(E * a, config["BLOCK_M"]), triton.cdiv(D, config["BLOCK_N"]))

    _fused_indices_and_gather_kernel[grid](
        x2d,
        gather_out,
        idx1d,
        E,
        a,
        D,
        sx2d0,
        sx2d1,
        sgatherout0,
        sgatherout1,
        sidx,
        config["BLOCK_M"],
        config["BLOCK_N"],
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
        num_ctas=num_ctas,
    )

    return gather_out, idx1d
