# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from typing import Optional

import torch

OPUS_A8W4_STAGE2_KERNEL = "opus_moe_stage2_a8w4_decode"
_DEFAULT_SORT_BLOCK_M = 32


def _value_is_empty(value) -> bool:
    return value is None or value != value or str(value).strip() in ("", "nan", "None")


def _cfg_first(cfg: dict, *names: str):
    for name in names:
        if name in cfg and not _value_is_empty(cfg[name]):
            return cfg[name]
    return None


def _cfg_int(value, default: int = 0) -> int:
    if _value_is_empty(value):
        return default
    return int(float(value))


def _cfg_optional_int(value) -> Optional[int]:
    if _value_is_empty(value):
        return None
    return int(float(value))


def _cfg_bool(value, default: bool = False) -> bool:
    if _value_is_empty(value):
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in ("true", "yes", "y"):
        return True
    if text in ("false", "no", "n"):
        return False
    return bool(int(float(value)))


def _cfg_str(value, default: str = "") -> str:
    return default if _value_is_empty(value) else str(value).strip()


def is_opus_a8w4_stage2_kernel(kernel_name) -> bool:
    from .moe_stage2_a8w4_meta import opus_a8w4_kid_from_name

    name = _cfg_str(kernel_name)
    # Explicit per-kid name (opus_moe2_afp8_wfp4_...) or legacy single-name form.
    return name == OPUS_A8W4_STAGE2_KERNEL or opus_a8w4_kid_from_name(name) is not None


def route_bucket_metadata(cfg: dict) -> dict[str, object]:
    return {
        "route_bucket": _cfg_str(
            _cfg_first(cfg, "route_bucket", "route_bucket_name"),
        ),
        "expected_sorted_blocks": _cfg_optional_int(
            _cfg_first(cfg, "expected_sorted_blocks", "expected_route_blocks")
        ),
        "min_sorted_blocks": _cfg_optional_int(
            _cfg_first(cfg, "min_sorted_blocks", "min_route_blocks")
        ),
        "max_sorted_blocks": _cfg_optional_int(
            _cfg_first(cfg, "max_sorted_blocks", "max_route_blocks")
        ),
    }


def stage2_cfg_values(cfg: dict, block_m) -> dict[str, object]:
    from .moe_stage2_a8w4_meta import (
        opus_a8w4_kid_block_m,
        opus_a8w4_kid_from_name,
        opus_a8w4_kid_reduce_block_n,
        opus_a8w4_reduce_block_n_from_name,
        opus_a8w4_kid_uses_route,
    )

    sort_block_m = _cfg_int(block_m, _DEFAULT_SORT_BLOCK_M)
    # Preferred path: explicit per-kid name encodes mode (atomic/bf16/fp8) + block_m,
    # so a single kernelName2 column fully selects the kernel (FlyDSL-style).
    name = _cfg_str(_cfg_first(cfg, "kernelName2", "kernel_name2", "stage2_kernel"))
    kid = opus_a8w4_kid_from_name(name)
    if kid is not None:
        reduce_block_n = opus_a8w4_reduce_block_n_from_name(name)
        if reduce_block_n is None:
            reduce_block_n = opus_a8w4_kid_reduce_block_n(kid)
        return {
            "kernel_id": kid,
            "stage2_block_m": opus_a8w4_kid_block_m(kid),
            "route_out": opus_a8w4_kid_uses_route(kid),
            "stage2_reduce_block_n": reduce_block_n,
        }
    # Legacy fallback: explicit numeric columns.
    kernel_id = _cfg_int(
        _cfg_first(cfg, "stage2_kernel_id", "opus_kernel_id", "kernel_id2"),
        -1,
    )
    reduce_block_n = _cfg_optional_int(
        _cfg_first(cfg, "stage2_reduce_block_n", "reduce_block_n")
    )
    if kernel_id != -1:
        if reduce_block_n is None:
            reduce_block_n = opus_a8w4_kid_reduce_block_n(kernel_id)
        return {
            "kernel_id": kernel_id,
            "stage2_block_m": opus_a8w4_kid_block_m(kernel_id),
            "route_out": opus_a8w4_kid_uses_route(kernel_id),
            "stage2_reduce_block_n": reduce_block_n,
        }
    return {
        "kernel_id": kernel_id,
        "stage2_block_m": _cfg_int(
            _cfg_first(cfg, "stage2_block_m", "opus_block_m", "kernel_block_m"),
            sort_block_m,
        ),
        "route_out": _cfg_bool(
            _cfg_first(cfg, "stage2_route_out", "route_out", "return_per_slot"),
            False,
        ),
        "stage2_reduce_block_n": reduce_block_n,
    }


def cfg_is_supported(
    kernel_name,
    *,
    cfg: dict,
    gfx: str,
    block_m,
    is_ep: bool,
    has_stage2_bias: bool = False,
) -> tuple[bool, str]:
    if not is_opus_a8w4_stage2_kernel(kernel_name):
        return False, f"unknown Opus A8W4 stage2 kernelName2={kernel_name!r}"
    if gfx != "gfx950":
        return False, f"requires gfx950, got {gfx}"
    if is_ep:
        return False, "EP expert_mask/topk_ids are not supported"
    if has_stage2_bias:
        return False, "stage2 bias is not supported"

    sort_block_m = _cfg_int(block_m, _DEFAULT_SORT_BLOCK_M)
    if sort_block_m <= 0:
        return False, f"requires positive sort block_m, got {sort_block_m}"
    try:
        values = stage2_cfg_values(cfg, block_m)
    except ValueError as exc:
        return False, str(exc)
    kernel_block_m = int(values["stage2_block_m"])
    from .moe_stage2_a8w4_meta import opus_a8w4_supported_block_ms

    supported_block_ms = opus_a8w4_supported_block_ms()
    if kernel_block_m not in supported_block_ms:
        supported = "/".join(str(v) for v in supported_block_ms)
        return False, f"requires stage2_block_m {supported}, got {kernel_block_m}"
    if int(values["kernel_id"]) == -1:
        route_out = bool(values["route_out"])
        if route_out and kernel_block_m != 64:
            return False, "auto route-out requires an explicit route-output kid"
        if not route_out and kernel_block_m == 64:
            return False, "auto direct-atomic only supports bm16/bm32 kernels"
    return True, ""


def stage2_uses_route_reduce(stage2) -> bool:
    return _cfg_bool(getattr(stage2, "keywords", {}).get("route_out"), False)


def check_route_bucket_metadata(metadata, sorted_expert_ids, logger) -> None:
    if (
        not metadata.route_bucket
        and metadata.expected_sorted_blocks is None
        and metadata.min_sorted_blocks is None
        and metadata.max_sorted_blocks is None
    ):
        return

    actual = int(sorted_expert_ids.numel())
    errors = []
    if (
        metadata.expected_sorted_blocks is not None
        and actual != metadata.expected_sorted_blocks
    ):
        errors.append(f"expected sorted_blocks={metadata.expected_sorted_blocks}")
    if metadata.min_sorted_blocks is not None and actual < metadata.min_sorted_blocks:
        errors.append(f"min sorted_blocks={metadata.min_sorted_blocks}")
    if metadata.max_sorted_blocks is not None and actual > metadata.max_sorted_blocks:
        errors.append(f"max sorted_blocks={metadata.max_sorted_blocks}")
    if not errors:
        return

    bucket = f" route_bucket={metadata.route_bucket!r}" if metadata.route_bucket else ""
    logger.warning(
        f"[fused_moe] tuned route bucket mismatch{bucket}: actual sorted_blocks={actual}; "
        + ", ".join(errors)
    )


def opus_a8w4_stage2_wrapper(
    inter_states,
    w1,
    w2,
    sorted_token_ids,
    sorted_expert_ids,
    num_valid_ids,
    out,
    topk,
    kernelName="",
    w2_scale=None,
    a2_scale=None,
    sorted_weights=None,
    bias2=None,
    inter_dim_pad: int = 0,
    model_dim_pad: int = 0,
    expert_mask=None,
    topk_ids=None,
    block_m: int = _DEFAULT_SORT_BLOCK_M,
    kernel_id: int = -1,
    stage2_block_m: Optional[int] = None,
    stage2_reduce_block_n: Optional[int] = None,
    route_out: bool = False,
    **_kwargs,
):
    del w1, model_dim_pad, _kwargs
    if not is_opus_a8w4_stage2_kernel(kernelName):
        raise ValueError(f"Invalid Opus A8W4 stage2 kernel name: {kernelName}")
    route_out_mode = bool(route_out)
    kernel_block_m = int(stage2_block_m or block_m)
    if bias2 is not None:
        raise ValueError("Opus A8W4 stage2 does not support bias2")
    if expert_mask is not None or topk_ids is not None:
        raise ValueError("Opus A8W4 stage2 does not support EP expert_mask/topk_ids")
    if a2_scale is None or w2_scale is None:
        raise ValueError("Opus A8W4 stage2 requires a2_scale and w2_scale")
    if inter_states.dim() != 3:
        raise ValueError(
            "Opus A8W4 stage2 expects inter_states=[token, topk, inter_dim], "
            f"got {tuple(inter_states.shape)}"
        )
    from .moe_stage2_a8w4_meta import (
        OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT,
    )

    actual_inter_dim_pad = int(inter_dim_pad)
    kernel_contract = OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT
    expected_w2 = (
        w2.shape[0],
        w2.shape[1],
        inter_states.shape[2] // kernel_contract.fp4_values_per_byte,
    )
    if tuple(w2.shape) != expected_w2:
        raise ValueError(
            f"Opus A8W4 stage2 expects w2={list(expected_w2)}, got {tuple(w2.shape)}"
        )
    expected_out = (inter_states.shape[0], w2.shape[1])
    if tuple(out.shape) != expected_out:
        raise ValueError(
            f"Opus A8W4 stage2 expects out={list(expected_out)}, "
            f"got {tuple(out.shape)}"
        )

    from aiter.ops.opus.moe_stage2_a8w4 import (
        opus_moe_stage2_a8w4_decode_fwd,
        opus_moe_stage2_reduce_token_slot_route_output_fwd,
    )

    if route_out_mode:
        route_out = opus_moe_stage2_a8w4_decode_fwd(
            inter_states,
            w2,
            a2_scale,
            w2_scale,
            sorted_token_ids,
            sorted_weights,
            sorted_expert_ids,
            num_valid_ids,
            block_m=kernel_block_m,
            kernel_id=int(kernel_id),
            inter_dim_pad=actual_inter_dim_pad,
            return_per_slot=True,
        )
        if route_out.dtype == torch.uint8:  # MXFP8 route_out
            return opus_moe_stage2_reduce_token_slot_route_output_fwd(
                route_out, out=out, topk=int(topk), block_n=stage2_reduce_block_n
            )
        return opus_moe_stage2_reduce_token_slot_route_output_fwd(
            route_out.view(out.shape[0], int(topk), out.shape[1]),
            out=out,
            topk=int(topk),
            block_n=stage2_reduce_block_n,
        )

    return opus_moe_stage2_a8w4_decode_fwd(
        inter_states,
        w2,
        a2_scale,
        w2_scale,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        out=out,
        block_m=kernel_block_m,
        kernel_id=int(kernel_id),
        inter_dim_pad=actual_inter_dim_pad,
    )


__all__ = [
    "OPUS_A8W4_STAGE2_KERNEL",
    "cfg_is_supported",
    "check_route_bucket_metadata",
    "is_opus_a8w4_stage2_kernel",
    "opus_a8w4_stage2_wrapper",
    "route_bucket_metadata",
    "stage2_cfg_values",
    "stage2_uses_route_reduce",
]
