# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""HipKittens block-FP8 grouped MoE for gfx950 (csrc/kernels/hk_fp8_moe).

A whole MoE layer forward (routing, both expert GEMMs, SwiGLU, weighted combine) specialized to hidden 4096 /
intermediate 1024, top-8/9, with power-of-two (ue8M0) 128x128 block scales.
"""

import torch

from ..jit.core import compile_ops
from ..jit.utils.chip_info import get_gfx

HIDDEN, INTER, MAX_EXPERTS = 4096, 1024, 1024


@compile_ops(
    "module_hk_fp8_moe", fc_name="aiter_hk_fp8_moe_workspace_units", ffi_type="ctypes"
)
def _hk_fp8_moe_workspace_units(
    num_tokens: int, topk: int, num_experts: int
) -> int: ...


@compile_ops("module_hk_fp8_moe", fc_name="aiter_hk_fp8_moe_fwd", ffi_type="ctypes")
def _hk_fp8_moe_fwd_raw(
    x: int,
    out: int,
    num_tokens: int,
    topk_ids: int,
    topk_weights: int,
    topk: int,
    w13: int,
    w13_scale: int,
    w2: int,
    w2_scale: int,
    num_experts: int,
    swiglu_limit: float,
    workspace: int,
    workspace_bytes: int,
) -> None: ...


def hk_fp8_moe(
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
    swiglu_limit: float = 0.0,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Block-FP8 MoE layer forward on gfx950.

    x [T, 4096] bf16; topk_ids [T, topk] (topk 8 or 9; out-of-range ids contribute nothing) and topk_weights
    [T, topk]; w13 [E, 2048, 4096] (gate rows then up rows) and w2 [E, 4096, 1024] fp8 e4m3 in
    shuffle_weight(w, (16, 16)) layout; w13_scale [E, 16, 32] / w2_scale [E, 32, 8] fp32 power-of-two block
    scales. Experts compute silu(min(gate, L)) * clamp(up, -L, L) for swiglu_limit L > 0, else silu(gate) * up.
    Returns bf16 [T, 4096]. The workspace comes from torch's caching allocator, so the call is graph-capturable.
    """
    if get_gfx() != "gfx950":
        raise RuntimeError(f"hk_fp8_moe requires gfx950, running on {get_gfx()}")
    if x.dim() != 2 or x.shape[1] != HIDDEN or x.dtype != torch.bfloat16:
        raise ValueError(
            f"hk_fp8_moe: x must be bf16 [T, {HIDDEN}], got {x.dtype} {tuple(x.shape)}"
        )
    T, E, topk = x.shape[0], w13.shape[0], topk_ids.shape[-1]
    if (
        topk_ids.dim() != 2
        or topk not in (8, 9)
        or topk_ids.shape[0] != T
        or topk_weights.shape != topk_ids.shape
    ):
        raise ValueError(
            f"hk_fp8_moe: topk_ids / topk_weights must be [T, 8 or 9], got {tuple(topk_ids.shape)}"
        )
    if E > MAX_EXPERTS:
        raise ValueError(f"hk_fp8_moe: at most {MAX_EXPERTS} experts, got {E}")
    for name, w, shape in (
        ("w13", w13, (E, 2 * INTER, HIDDEN)),
        ("w2", w2, (E, HIDDEN, INTER)),
    ):
        if (
            tuple(w.shape) != shape
            or w.dtype != torch.float8_e4m3fn
            or not w.is_contiguous()
        ):
            raise ValueError(
                f"hk_fp8_moe: {name} must be contiguous fp8 e4m3 {shape}, got {w.dtype} {tuple(w.shape)}"
            )
    for name, s, shape in (
        ("w13_scale", w13_scale, (E, 2 * INTER // 128, HIDDEN // 128)),
        ("w2_scale", w2_scale, (E, HIDDEN // 128, INTER // 128)),
    ):
        if tuple(s.shape) != shape or s.dtype != torch.float32 or not s.is_contiguous():
            raise ValueError(
                f"hk_fp8_moe: {name} must be contiguous fp32 {shape}, got {s.dtype} {tuple(s.shape)}"
            )
    x = x.contiguous()
    if out is None:
        out = torch.empty_like(x)
    elif out.shape != x.shape or out.dtype != torch.bfloat16 or not out.is_contiguous():
        raise ValueError(
            "hk_fp8_moe: out must be a contiguous bf16 tensor shaped like x"
        )
    if T == 0:
        return out
    ids = topk_ids.to(torch.int32).contiguous()
    wts = topk_weights.to(torch.float32).contiguous()
    with torch.cuda.device(x.device):
        ws = torch.empty(
            _hk_fp8_moe_workspace_units(T, topk, E) * 256,
            dtype=torch.uint8,
            device=x.device,
        )
        _hk_fp8_moe_fwd_raw(
            x.data_ptr(),
            out.data_ptr(),
            T,
            ids.data_ptr(),
            wts.data_ptr(),
            topk,
            w13.data_ptr(),
            w13_scale.data_ptr(),
            w2.data_ptr(),
            w2_scale.data_ptr(),
            E,
            float(swiglu_limit),
            ws.data_ptr(),
            ws.numel(),
        )
    return out
