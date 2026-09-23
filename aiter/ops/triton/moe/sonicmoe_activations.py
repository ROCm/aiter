# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton

from aiter.ops.triton._triton_kernels.moe.activations import (
    _glu_bwd_kernel,
    _glu_fwd_kernel,
    _pointwise_act_bwd_kernel,
    _pointwise_act_fwd_kernel,
)
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_sonicmoe_kernel_config,
    split_launch_config,
)

_GLU_ACT_MAP = {"swiglu": 0, "geglu": 1, "reglu": 2}
_POINTWISE_ACT_MAP = {"gelu_tanh_approx": 3, "relu": 4, "silu": 5, "relu_sq": 6}


def _launch_config(TK, I):
    config = get_sonicmoe_kernel_config("activation_kernel")
    block_i = min(triton.next_power_of_2(I), config.pop("BLOCK_I_MAX"))
    block_m = config["BLOCK_M"]
    grid = (triton.cdiv(TK, block_m), triton.cdiv(I, block_i))
    constexprs, launch = split_launch_config(config)
    constexprs["BLOCK_I"] = block_i
    return grid, constexprs, launch


def activation_fwd(
    h: torch.Tensor, I: int, activation_type: str, concat_layout: bool = False
) -> torch.Tensor:
    TK = h.shape[0]

    if activation_type in _GLU_ACT_MAP:
        a = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        grid, constexprs, launch = _launch_config(TK, I)
        _glu_fwd_kernel[grid](
            h,
            a,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            a.stride(0),
            a.stride(1),
            CONCAT_LAYOUT=concat_layout,
            ACT_TYPE=_GLU_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return a
    elif activation_type in _POINTWISE_ACT_MAP:
        a = torch.empty(TK, I, dtype=h.dtype, device=h.device)
        grid, constexprs, launch = _launch_config(TK, I)
        _pointwise_act_fwd_kernel[grid](
            h,
            a,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            a.stride(0),
            a.stride(1),
            ACT_TYPE=_POINTWISE_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return a
    else:
        raise NotImplementedError(f"activation_type={activation_type}")


def activation_bwd(
    h: torch.Tensor,
    da: torch.Tensor,
    I: int,
    activation_type: str,
    concat_layout: bool = False,
) -> torch.Tensor:
    TK = h.shape[0]

    if activation_type in _GLU_ACT_MAP:
        dh = torch.empty_like(h)
        grid, constexprs, launch = _launch_config(TK, I)
        _glu_bwd_kernel[grid](
            h,
            dh,
            da,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            dh.stride(0),
            dh.stride(1),
            da.stride(0),
            da.stride(1),
            CONCAT_LAYOUT=concat_layout,
            ACT_TYPE=_GLU_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return dh
    elif activation_type in _POINTWISE_ACT_MAP:
        dh = torch.empty_like(h)
        grid, constexprs, launch = _launch_config(TK, I)
        _pointwise_act_bwd_kernel[grid](
            h,
            dh,
            da,
            TK,
            I,
            h.stride(0),
            h.stride(1),
            dh.stride(0),
            dh.stride(1),
            da.stride(0),
            da.stride(1),
            ACT_TYPE=_POINTWISE_ACT_MAP[activation_type],
            **constexprs,
            **launch,
        )
        return dh
    else:
        raise NotImplementedError(f"activation_type={activation_type}")
