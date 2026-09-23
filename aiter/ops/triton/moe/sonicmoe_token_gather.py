# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch

from aiter.ops.triton._triton_kernels.moe.sonicmoe.token_gather import (
    token_gather_sum_kernel,
)
from aiter.ops.triton.utils.sonicmoe_config_utils import (
    get_token_gather_config,
    split_launch_config,
)


def token_gather_and_sum_varlen_K_triton(
    x: torch.Tensor,  # (Mtotal, H)
    w: torch.Tensor | None,  # (Mtotal,)
    out: torch.Tensor,  # (T, H)
    M_perm: torch.Tensor,  # (Mtotal,) int32
    M_offset: torch.Tensor,  # (T+1,)   int32, variable K per token
    T: int,
    MAX_K: int,  # maximum K across all tokens
    H: int,
    is_varlen_K: bool,
):
    """Gather and reduce a variable number of weighted rows per token."""
    common = (x, w, M_perm, M_offset, out)
    kwargs = {
        "T": T,
        "H": H,
        "MAX_K": MAX_K,
        "stride_xM": x.stride(0),
        "stride_xH": x.stride(1),
        "stride_outT": out.stride(0),
        "stride_outH": out.stride(1),
        "w_is_None": (w is None),
        "is_varlen_K": is_varlen_K,
    }
    constexprs, launch = split_launch_config(get_token_gather_config(H))
    token_gather_sum_kernel[(T,)](*common, **kwargs, **constexprs, **launch)
