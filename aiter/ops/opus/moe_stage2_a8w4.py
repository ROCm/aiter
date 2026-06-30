# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from ...jit.core import compile_ops
from .moe_stage2_a8w4_meta import (
    OPUS_A8W4_OUT_MODE_BF16,
    OPUS_A8W4_OUT_MODE_FP8,
    opus_a8w4_best_atomic_kid,
    opus_a8w4_decode_kid,
    opus_a8w4_kid_block_m,
    opus_a8w4_kid_is_fp8,
    opus_a8w4_kid_name,
    opus_a8w4_kid_uses_route,
    opus_a8w4_shape_family_for_shape,
)

_OPUS_MOE_STAGE2_ROUTE_REDUCE_AUTO_BLOCK_N = -1


def _contiguous(tensor: Tensor) -> Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _optional_contiguous(tensor: Optional[Tensor]) -> Optional[Tensor]:
    return None if tensor is None else _contiguous(tensor)


def _route_out_mode_from_dtype(route_out_dtype: Optional[str]) -> int:
    if route_out_dtype is None:
        return OPUS_A8W4_OUT_MODE_FP8
    route_out_dtype = str(route_out_dtype).strip().lower()
    if route_out_dtype in ("fp8", "mxfp8", "uint8"):
        return OPUS_A8W4_OUT_MODE_FP8
    if route_out_dtype in ("bf16", "bfloat16", "torch.bfloat16"):
        return OPUS_A8W4_OUT_MODE_BF16
    raise ValueError(
        "route_out_dtype must be one of "
        f"('fp8', 'mxfp8', 'uint8', 'bf16', 'bfloat16'), got {route_out_dtype!r}"
    )


def _check_reduce_out(
    out: Tensor,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tuple(out.shape) != shape:
        raise ValueError(f"out must be {shape}, got {tuple(out.shape)}")
    if out.dtype != dtype:
        raise ValueError(f"out must be {dtype}, got {out.dtype}")
    if out.device != device:
        raise ValueError(f"out must be on {device}, got {out.device}")
    if out.dim() == 0 or out.stride(-1) != 1:
        raise ValueError("out last dimension must be contiguous")


def _gen_opus_moe_stage2_a8w4_decode_fake_tensors(
    inter_states: Tensor,
    w2: Tensor,
    a2_scale: Tensor,
    w2_scale: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
) -> Tensor:
    return out


def _gen_opus_moe_stage2_reduce_fake_tensors(
    route_out: Tensor,
    out: Tensor,
    topk: int,
    block_n: int,
) -> Tensor:
    return out


@compile_ops(
    "module_moe_opus",
    fc_name="opus_moe_stage2_a8w4_decode_fwd",
    gen_fake=_gen_opus_moe_stage2_a8w4_decode_fake_tensors,
    develop=True,
)
def _opus_moe_stage2_a8w4_decode_fwd_raw(
    inter_states: Tensor,
    w2: Tensor,
    a2_scale: Tensor,
    w2_scale: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    block_m: int,
    kernel_id: int,
    inter_dim_pad: int,
) -> Tensor: ...


@compile_ops(
    "module_moe_opus",
    fc_name="opus_moe_stage2_reduce_token_slot_route_output_fwd",
    gen_fake=_gen_opus_moe_stage2_reduce_fake_tensors,
    develop=True,
)
def _opus_moe_stage2_reduce_token_slot_route_output_fwd_raw(
    route_out: Tensor,
    out: Tensor,
    topk: int,
    block_n: int,
) -> Tensor: ...


def opus_moe_stage2_a8w4_decode_fwd(
    inter_states: Tensor,
    w2: Tensor,
    a2_scale: Tensor,
    w2_scale: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Optional[Tensor],
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    block_m: int,
    inter_dim_pad: int,
    out: Optional[Tensor] = None,
    kernel_id: int = -1,
    return_per_slot: bool = False,
    route_out_dtype: Optional[str] = None,
) -> Tensor:
    # Output mode is encoded in the kid. Route-out kernels must be requested
    # explicitly after they have been added to the metadata table.
    shape_family = opus_a8w4_shape_family_for_shape(
        model_dim=w2.shape[1],
        inter_dim=inter_states.shape[2],
        expert=w2.shape[0],
        topk=inter_states.shape[1],
    )
    shape_family_name = None if shape_family is None else shape_family.name
    if kernel_id == -1 and shape_family_name is None:
        raise ValueError(
            "unsupported Opus A8W4 shape family for auto kid selection: "
            f"inter_states={tuple(inter_states.shape)}, w2={tuple(w2.shape)}"
        )
    if route_out_dtype is not None and not return_per_slot:
        raise ValueError("route_out_dtype requires return_per_slot=True")
    if return_per_slot and kernel_id == -1:
        kernel_id = opus_a8w4_decode_kid(
            _route_out_mode_from_dtype(route_out_dtype),
            block_m,
            shape_family=shape_family_name,
        )
    elif not return_per_slot and kernel_id == -1 and block_m == 32:
        kernel_id = opus_a8w4_best_atomic_kid(
            inter_states.shape[0],
            shape_family=shape_family_name,
        )
        block_m = opus_a8w4_kid_block_m(kernel_id)
    route_out = bool(return_per_slot)
    route_out_fp8 = False
    if kernel_id != -1:
        kid_route_out = opus_a8w4_kid_uses_route(kernel_id)
        if return_per_slot and not kid_route_out:
            raise ValueError(
                "return_per_slot=True requires a route-output Opus A8W4 stage2 "
                f"kid, got kernel_id={kernel_id} ({opus_a8w4_kid_name(kernel_id)})"
            )
        route_out = kid_route_out
        route_out_fp8 = opus_a8w4_kid_is_fp8(kernel_id)
    md = w2.shape[1]
    if out is None:
        if route_out_fp8:
            # MXFP8 route_out: uint8 [rows, md fp8 | md/8 e8m0 scale].
            rows = inter_states.shape[0] * inter_states.shape[1]
            out = torch.empty((rows, md + md // 8), dtype=torch.uint8, device=w2.device)
        else:
            shape = (
                (inter_states.shape[0], inter_states.shape[1], w2.shape[1])
                if route_out
                else (inter_states.shape[0], w2.shape[1])
            )
            alloc = torch.empty if route_out else torch.zeros
            out = alloc(shape, dtype=torch.bfloat16, device=w2.device)

    kernel_out = (
        out if route_out_fp8 else (out.view(-1, w2.shape[1]) if route_out else out)
    )

    _opus_moe_stage2_a8w4_decode_fwd_raw(
        _contiguous(inter_states),
        _contiguous(w2),
        _contiguous(a2_scale),
        _contiguous(w2_scale),
        _contiguous(sorted_token_ids),
        _optional_contiguous(sorted_weights),
        _contiguous(sorted_expert_ids),
        _contiguous(num_valid_ids),
        kernel_out,
        int(block_m),
        int(kernel_id),
        int(inter_dim_pad),
    )
    return out


def opus_moe_stage2_reduce_token_slot_route_output_fwd(
    route_out: Tensor,
    out: Optional[Tensor] = None,
    *,
    topk: int | None = None,
    block_n: int | None = None,
) -> Tensor:
    if route_out.dtype == torch.uint8:
        # MXFP8 route_out: uint8 [rows, md + md/8]; topk required, out [rows/topk, md].
        # fp8 is inferred from the uint8 dtype (no OPUS_ROUTE_FP8 env); C++ matches.
        if topk is None:
            raise ValueError("fp8 route_out reduce requires topk")
        topk = int(topk)
        if topk <= 0:
            raise ValueError(f"fp8 route_out reduce requires positive topk, got {topk}")
        if route_out.dim() != 2:
            raise ValueError(
                "fp8 route_out must be [token * topk, hidden + hidden / 8], "
                f"got {tuple(route_out.shape)}"
            )
        if route_out.shape[0] % topk != 0:
            raise ValueError(
                f"fp8 route_out rows must be divisible by topk={topk}, "
                f"got rows={route_out.shape[0]}"
            )
        if route_out.shape[1] % 9 != 0:
            raise ValueError(
                "fp8 route_out columns must be hidden + hidden / 8 "
                f"(a multiple of 9), got {route_out.shape[1]}"
            )
        md = route_out.shape[1] * 8 // 9
        out_shape = (route_out.shape[0] // topk, md)
        if out is None:
            out = torch.empty(
                out_shape,
                dtype=torch.bfloat16,
                device=route_out.device,
            )
        else:
            _check_reduce_out(
                out,
                shape=out_shape,
                dtype=torch.bfloat16,
                device=route_out.device,
            )
        bn = _OPUS_MOE_STAGE2_ROUTE_REDUCE_AUTO_BLOCK_N if block_n is None else block_n
        _opus_moe_stage2_reduce_token_slot_route_output_fwd_raw(
            _contiguous(route_out), out, int(topk), int(bn)
        )
        return out
    if route_out.dtype != torch.bfloat16:
        raise ValueError(
            f"route_out must be uint8 MXFP8 or bfloat16, got {route_out.dtype}"
        )
    if route_out.dim() != 3:
        raise ValueError(
            f"route_out must be [token, topk, hidden], got {tuple(route_out.shape)}"
        )
    if topk is None:
        topk = int(route_out.shape[1])
    else:
        topk = int(topk)
    if topk <= 0:
        raise ValueError(f"route_out reduce requires positive topk, got {topk}")
    if route_out.shape[1] != topk:
        raise ValueError(
            f"route_out topk dimension must match topk={topk}, "
            f"got {route_out.shape[1]}"
        )
    out_shape = (route_out.shape[0], route_out.shape[2])
    if out is None:
        out = torch.empty(
            out_shape,
            dtype=route_out.dtype,
            device=route_out.device,
        )
    else:
        _check_reduce_out(
            out,
            shape=out_shape,
            dtype=route_out.dtype,
            device=route_out.device,
        )
    if block_n is None:
        block_n = _OPUS_MOE_STAGE2_ROUTE_REDUCE_AUTO_BLOCK_N
    _opus_moe_stage2_reduce_token_slot_route_output_fwd_raw(
        _contiguous(route_out),
        out,
        int(topk),
        int(block_n),
    )
    return out


__all__ = [
    "opus_moe_stage2_a8w4_decode_fwd",
    "opus_moe_stage2_reduce_token_slot_route_output_fwd",
]
