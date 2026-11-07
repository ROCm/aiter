# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import (
    load_config_json,
    resolve_config_dir,
    select_leq_config,
)

_fused_nhwc_to_nchw_add_repr = make_kernel_repr(
    "_fused_nhwc_to_nchw_add_kernel",
    ["BLOCK_P", "BLOCK_C", "HAS_RESIDUAL", "num_warps"],
)


def _get_config(C: int) -> dict:
    """Launch config for C channels."""
    config_dir = resolve_config_dir("fusions", "FUSED_NHWC_TO_NCHW_ADD")
    configs = load_config_json(f"{config_dir}/DEFAULT.json", required=True)
    return select_leq_config(configs, C, prefix="C_LEQ_")


@triton.jit(repr=_fused_nhwc_to_nchw_add_repr)
def _fused_nhwc_to_nchw_add_kernel(
    x_ptr,
    res_ptr,
    out_ptr,
    P,
    C,
    stride_xb,
    stride_xp,
    stride_rb,
    stride_rc,
    stride_ob,
    stride_oc,
    BLOCK_P: tl.constexpr,
    BLOCK_C: tl.constexpr,
    HAS_RESIDUAL: tl.constexpr,
):
    """out[b, c, p] = x[b, p, c] (+ res[b, c, p]) for a BLOCK_P x BLOCK_C tile.

    Offsets inside one batch are int32 (the wrapper checks that they fit).
    """
    pid = tl.program_id(0)
    b = tl.program_id(1).to(tl.int64)
    tiles_c = tl.cdiv(C, BLOCK_C)
    offs_p = (pid // tiles_c) * BLOCK_P + tl.arange(0, BLOCK_P)
    offs_c = (pid % tiles_c) * BLOCK_C + tl.arange(0, BLOCK_C)
    mask = (offs_p[:, None] < P) & (offs_c[None, :] < C)

    v = tl.load(
        x_ptr + b * stride_xb + offs_p[:, None] * stride_xp + offs_c[None, :],
        mask=mask,
    )
    if HAS_RESIDUAL:
        r = tl.load(
            res_ptr + b * stride_rb + offs_c[None, :] * stride_rc + offs_p[:, None],
            mask=mask,
        )
        # As eager: the add is in fp32 and rounds once.
        v = (v.to(tl.float32) + r.to(tl.float32)).to(v.dtype)
    tl.store(
        out_ptr + b * stride_ob + offs_c[None, :] * stride_oc + offs_p[:, None],
        v,
        mask=mask,
    )
