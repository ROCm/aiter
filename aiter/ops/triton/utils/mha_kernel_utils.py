# SPDX-License-Identifier: MIT
# Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.

import torch
import triton
import triton.language as tl


def get_arch():
    return triton.runtime.driver.active.get_current_target().arch


def is_hip():
    return triton.runtime.driver.active.get_current_target().backend == "hip"


def arch_supports_fp8():
    return is_hip() and get_arch() in ("gfx942")


@triton.jit
def compute_fp8_scaling_factors(x, fp8_max: tl.constexpr):
    # compute fp8 scaling and descaling factor for a block
    x_amax = tl.max(tl.abs(x))  # NOTE: abs deals with negative values
    x_amax = tl.where(x_amax <= 1e-9, 1e-9, x_amax)
    scale_x = fp8_max / x_amax
    descale_x = x_amax / fp8_max
    return scale_x, descale_x


def is_fp8(x):
    if x.dtype in {
        torch.float8_e4m3fnuz,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.float8_e5m2fnuz,
    }:
        if arch_supports_fp8():
            return True
        else:
            raise RuntimeError("This device does not support fp8")
    else:
        return False
