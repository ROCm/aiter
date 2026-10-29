# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import triton
import triton.language as tl

from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
from aiter.ops.triton.utils.config_utils import load_config_json, resolve_config_dir

_fused_situ_mul_repr = make_kernel_repr(
    "_fused_situ_mul_kernel",
    ["BLOCK_SIZE_M", "BLOCK_SIZE_N", "NEED_MASK", "CLIP_UP", "num_warps"],
)

_LOG2_E = tl.constexpr(1.4426950408889634)


def _get_config(key: str = "any") -> dict:
    config_dir = resolve_config_dir("fusions", "FUSED_SITU_MUL")
    return dict(load_config_json(f"{config_dir}/DEFAULT.json", required=True)[key])


@triton.jit
def _rcp(x):
    return tl.inline_asm_elementwise(
        "v_rcp_f32 $0, $1", "=v,v", [x], dtype=tl.float32, is_pure=True, pack=1
    )


@triton.jit(repr=_fused_situ_mul_repr, do_not_specialize=["M", "num_tiles"])
def _fused_situ_mul_kernel(
    x_ptr,
    out_ptr,
    M,
    N,
    stride_x_m,
    stride_out_m,
    gate_scale,
    up_scale,
    beta,
    linear_beta,
    num_tiles,
    tiles_n,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    NEED_MASK: tl.constexpr,
    CLIP_UP: tl.constexpr,
):
    """out = beta * tanh(gate / beta) * sigmoid(gate) * up, where x = [gate | up].
    With CLIP_UP, up is first clipped to linear_beta * tanh(up / linear_beta).

    Computed in fp32 using exp2 and a hardware reciprocal. Each program loops over
    tiles pid, pid + grid, ...
    """
    for tile in range(tl.program_id(0), num_tiles, tl.num_programs(0)):
        tile_n = tile % tiles_n
        tile_m = tile // tiles_n
        cols = tile_n.to(tl.int64) * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        rows = (tile_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)
        mask = None
        if NEED_MASK:
            mask = (rows[:, None] < M) & (cols[None, :] < N)

        gate_ptrs = x_ptr + rows[:, None] * stride_x_m + cols[None, :]
        g = tl.load(gate_ptrs, mask=mask).to(tl.float32)
        u = tl.load(gate_ptrs + N, mask=mask).to(tl.float32)

        if CLIP_UP:
            e0 = tl.exp2(tl.minimum(g * -_LOG2_E, 42.0, propagate_nan=tl.PropagateNan.ALL))
            e1 = tl.exp2(tl.minimum(g * gate_scale, 42.0, propagate_nan=tl.PropagateNan.ALL))
            e2 = tl.exp2(tl.minimum(u * up_scale, 42.0, propagate_nan=tl.PropagateNan.ALL))
            num = (1.0 - e1) * (1.0 - e2) * (beta * linear_beta)
            den = (1.0 + e1) * (1.0 + e0) * (1.0 + e2)
            res = num * _rcp(den)
        else:
            e0 = tl.exp2(tl.minimum(g * -_LOG2_E, 63.0, propagate_nan=tl.PropagateNan.ALL))
            e1 = tl.exp2(tl.minimum(g * gate_scale, 63.0, propagate_nan=tl.PropagateNan.ALL))
            act = (1.0 - e1) * beta * _rcp((1.0 + e1) * (1.0 + e0))
            res = act * u

        tl.store(
            out_ptr + rows[:, None] * stride_out_m + cols[None, :],
            res.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
