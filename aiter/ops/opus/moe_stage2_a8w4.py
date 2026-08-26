# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from csrc.opus_moe.opus_moe_common import (
    OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT,
    OPUS_A8W4_OUT_MODE_BF16,
    OPUS_A8W4_OUT_MODE_FP8,
    OpusA8W4Stage2Instance,
    opus_a8w4_best_atomic_kid,
    opus_a8w4_decode_kid,
    opus_a8w4_effective_inter_dim,
    opus_a8w4_scale_cols_for_effective_inter_dim,
    opus_a8w4_stage2_instance_from_name,
    require_opus_a8w4_stage2_instance,
)

from ...jit.core import compile_ops

_DEFAULT_SORT_BLOCK_M = 32
_OPUS_MOE_STAGE2_ROUTE_REDUCE_AUTO_BLOCK_N = -1


@dataclass(frozen=True)
class OpusA8W4LaunchConfig:
    """Resolved runtime plan for one tuned Opus A8W4 Stage2 selection."""

    instance: OpusA8W4Stage2Instance

    @property
    def kernel_id(self) -> int:
        return self.instance.kid

    @property
    def sort_block_m(self) -> int:
        return self.instance.sort_block_m

    @property
    def reduce_block_n(self) -> int | None:
        return self.instance.reduce_block_n

    @property
    def route_out(self) -> bool:
        return self.instance.route_out


def _contiguous(tensor: Tensor) -> Tensor:
    return tensor if tensor.is_contiguous() else tensor.contiguous()


def _optional_contiguous(tensor: Tensor | None) -> Tensor | None:
    return None if tensor is None else _contiguous(tensor)


def _pad_scale_cols(tensor: Tensor, cols: int) -> Tensor:
    if tensor.shape[1] >= cols:
        return tensor
    padded = torch.empty(
        (*tensor.shape[:-1], cols), dtype=tensor.dtype, device=tensor.device
    )
    padded[..., : tensor.shape[-1]] = tensor
    padded[..., tensor.shape[-1] :] = tensor[..., -1:]
    return padded


def _pad_scale_rows(tensor: Tensor, rows: int) -> Tensor:
    if tensor.shape[0] >= rows:
        return tensor
    padded = torch.empty(
        (rows, tensor.shape[1]), dtype=tensor.dtype, device=tensor.device
    )
    padded[: tensor.shape[0], :] = tensor
    padded[tensor.shape[0] :, :] = tensor[-1:, :]
    return padded


def _route_out_mode_from_dtype(route_out_dtype: str | None) -> int:
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
    sorted_weights: Tensor | None,
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
    sorted_weights: Tensor | None,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    out: Tensor,
    token_num: int,
    topk: int,
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
    sorted_weights: Tensor | None,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    block_m: int,
    inter_dim_pad: int,
    out: Tensor | None = None,
    kernel_id: int = -1,
    return_per_slot: bool = False,
    route_out_dtype: str | None = None,
    token_num: int | None = None,
    topk: int | None = None,
) -> Tensor:
    if inter_states.dim() == 3:
        token_num = int(inter_states.shape[0])
        topk = int(inter_states.shape[1])
    elif inter_states.dim() == 2:
        if token_num is None or topk is None:
            raise ValueError(
                "2D sorted inter_states requires explicit token_num and topk"
            )
        token_num = int(token_num)
        topk = int(topk)
    else:
        raise ValueError(
            "Opus A8W4 stage2 expects inter_states=[token, topk, inter_dim] or "
            f"[sorted_row, inter_dim], got {tuple(inter_states.shape)}"
        )
    effective_inter_dim = opus_a8w4_effective_inter_dim(
        inter_states.shape[-1], inter_dim_pad
    )
    if effective_inter_dim is None:
        raise ValueError(
            "Opus A8W4 stage2 requires 0 <= inter_dim_pad < logical inter_dim, "
            f"got inter_states={tuple(inter_states.shape)}, inter_dim_pad={inter_dim_pad}"
        )
    if route_out_dtype is not None and not return_per_slot:
        raise ValueError("route_out_dtype requires return_per_slot=True")
    if return_per_slot and kernel_id == -1:
        kernel_id = opus_a8w4_decode_kid(
            _route_out_mode_from_dtype(route_out_dtype),
            block_m,
        )
    elif not return_per_slot and kernel_id == -1 and block_m == 32:
        kernel_id = opus_a8w4_best_atomic_kid(
            token_num,
        )
        block_m = require_opus_a8w4_stage2_instance(kernel_id).block_m
    route_out = bool(return_per_slot)
    route_out_fp8 = False
    if kernel_id != -1:
        instance = require_opus_a8w4_stage2_instance(kernel_id)
        if return_per_slot and not instance.route_out:
            raise ValueError(
                "return_per_slot=True requires a route-output Opus A8W4 stage2 "
                f"kid, got kernel_id={kernel_id} ({instance.name})"
            )
        route_out = instance.route_out
        route_out_fp8 = instance.route_out_fp8
    scale_cols = opus_a8w4_scale_cols_for_effective_inter_dim(effective_inter_dim)
    scale_row_pack = 2 * OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT.mfma_m
    scale_rows = (
        (int(sorted_token_ids.shape[0]) + scale_row_pack - 1)
        // scale_row_pack
        * scale_row_pack
    )
    a2_scale = _pad_scale_rows(a2_scale, scale_rows)
    a2_scale = _pad_scale_cols(a2_scale, scale_cols)
    w2_scale = _pad_scale_cols(w2_scale, scale_cols)
    md = w2.shape[1]
    if out is None:
        if route_out_fp8:
            # MXFP8 route_out: uint8 [rows, md fp8 | md/8 e8m0 scale].
            rows = token_num * topk
            out = torch.empty((rows, md + md // 8), dtype=torch.uint8, device=w2.device)
        else:
            shape = (
                (token_num, topk, w2.shape[1])
                if route_out
                else (token_num, w2.shape[1])
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
        int(token_num),
        int(topk),
        int(block_m),
        int(kernel_id),
        int(inter_dim_pad),
    )
    return out


def opus_moe_stage2_reduce_token_slot_route_output_fwd(
    route_out: Tensor,
    out: Tensor | None = None,
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


# ---- Tuned-config adapter -------------------------------------------------


def _value_is_empty(value) -> bool:
    return (
        value is None
        or value != value  # noqa: PLR0124
        or str(value).strip() in ("", "nan", "None")
    )


def _cfg_first(cfg: dict, *names: str):
    for name in names:
        if name in cfg and not _value_is_empty(cfg[name]):
            return cfg[name]
    return None


def _cfg_int(value, default: int = 0) -> int:
    return default if _value_is_empty(value) else int(float(value))


def _cfg_optional_int(value) -> int | None:
    return None if _value_is_empty(value) else int(float(value))


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
    name = _cfg_str(kernel_name)
    return opus_a8w4_stage2_instance_from_name(name) is not None


def route_bucket_metadata(cfg: dict) -> dict[str, object]:
    return {
        "route_bucket": _cfg_str(_cfg_first(cfg, "route_bucket", "route_bucket_name")),
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


def stage2_launch_config(kernel_id: int) -> OpusA8W4LaunchConfig:
    instance = require_opus_a8w4_stage2_instance(kernel_id)
    return OpusA8W4LaunchConfig(instance=instance)


def parse_stage2_config(cfg: dict, block_m) -> OpusA8W4LaunchConfig:
    """Resolve a tuned CSV row into one typed Stage2 launch plan."""

    sort_block_m = _cfg_int(block_m, _DEFAULT_SORT_BLOCK_M)
    name = _cfg_str(_cfg_first(cfg, "kernelName2", "kernel_name2", "stage2_kernel"))
    instance = opus_a8w4_stage2_instance_from_name(name)
    if instance is None:
        raise ValueError(f"unknown Opus A8W4 stage2 kernelName2={name!r}")
    launch = stage2_launch_config(instance.kid)
    if sort_block_m != launch.sort_block_m:
        raise ValueError(
            f"requires sort block_m={launch.sort_block_m}, "
            f"got tuned block_m={sort_block_m}"
        )
    return launch


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
        parse_stage2_config(cfg, block_m)
    except ValueError as exc:
        return False, str(exc)
    return True, ""


def stage2_uses_route_reduce(stage2) -> bool:
    keywords = getattr(stage2, "keywords", {})
    launch = keywords.get("launch")
    if isinstance(launch, OpusA8W4LaunchConfig):
        return launch.route_out
    return _cfg_bool(keywords.get("route_out"), False)


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
        f"[fused_moe] tuned route bucket mismatch{bucket}: "
        f"actual sorted_blocks={actual}; " + ", ".join(errors)
    )


# ---- Unified high-level Stage2 execution ----------------------------------


def opus_moe_stage2_a8w4_fwd(
    inter_states: Tensor,
    w2: Tensor,
    a2_scale: Tensor,
    w2_scale: Tensor,
    sorted_token_ids: Tensor,
    sorted_weights: Tensor | None,
    sorted_expert_ids: Tensor,
    num_valid_ids: Tensor,
    *,
    launch: OpusA8W4LaunchConfig,
    inter_dim_pad: int,
    out: Tensor | None = None,
    token_num: int | None = None,
    topk: int | None = None,
) -> Tensor:
    """Run Stage2 and fold route-output reduction into one shared path."""

    if not launch.route_out:
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
            block_m=launch.sort_block_m,
            kernel_id=launch.kernel_id,
            inter_dim_pad=inter_dim_pad,
            token_num=token_num,
            topk=topk,
        )

    if token_num is None:
        if inter_states.dim() == 3:
            token_num = int(inter_states.shape[0])
        elif out is not None:
            token_num = int(out.shape[0])
        else:
            raise ValueError("route-output Stage2 requires explicit token_num")
    if topk is None:
        if inter_states.dim() != 3:
            raise ValueError("route-output Stage2 requires explicit topk")
        topk = int(inter_states.shape[1])
    route_output = opus_moe_stage2_a8w4_decode_fwd(
        inter_states,
        w2,
        a2_scale,
        w2_scale,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        block_m=launch.sort_block_m,
        kernel_id=launch.kernel_id,
        inter_dim_pad=inter_dim_pad,
        return_per_slot=True,
        token_num=token_num,
        topk=topk,
    )
    return opus_moe_stage2_reduce_token_slot_route_output_fwd(
        route_output,
        out=out,
        topk=int(topk),
        block_n=launch.reduce_block_n,
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
    *,
    launch: OpusA8W4LaunchConfig,
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
    **_kwargs,
):
    del w1, model_dim_pad, block_m, _kwargs
    named_instance = opus_a8w4_stage2_instance_from_name(kernelName)
    if named_instance is None:
        raise ValueError(f"Invalid Opus A8W4 stage2 kernel name: {kernelName}")
    if named_instance != launch.instance:
        raise ValueError(
            "Opus A8W4 stage2 kernel name/launch mismatch: "
            f"kernelName={kernelName!r}, launch={launch.instance.name!r}"
        )
    if bias2 is not None:
        raise ValueError("Opus A8W4 stage2 does not support bias2")
    if expert_mask is not None or topk_ids is not None:
        raise ValueError("Opus A8W4 stage2 does not support EP expert_mask/topk_ids")
    if a2_scale is None or w2_scale is None:
        raise ValueError("Opus A8W4 stage2 requires a2_scale and w2_scale")
    if inter_states.dim() not in (2, 3):
        raise ValueError(
            "Opus A8W4 stage2 expects inter_states=[token, topk, inter_dim] or "
            "[sorted_row, inter_dim], "
            f"got {tuple(inter_states.shape)}"
        )

    contract = OPUS_A8W4_GFX950_DECODE_KERNEL_CONTRACT
    expected_w2 = (
        w2.shape[0],
        w2.shape[1],
        inter_states.shape[-1] // contract.fp4_values_per_byte,
    )
    if tuple(w2.shape) != expected_w2:
        raise ValueError(
            f"Opus A8W4 stage2 expects w2={list(expected_w2)}, got {tuple(w2.shape)}"
        )
    token_num = inter_states.shape[0] if inter_states.dim() == 3 else out.shape[0]
    expected_out = (token_num, w2.shape[1])
    if tuple(out.shape) != expected_out:
        raise ValueError(
            f"Opus A8W4 stage2 expects out={list(expected_out)}, "
            f"got {tuple(out.shape)}"
        )

    return opus_moe_stage2_a8w4_fwd(
        inter_states,
        w2,
        a2_scale,
        w2_scale,
        sorted_token_ids,
        sorted_weights,
        sorted_expert_ids,
        num_valid_ids,
        launch=launch,
        inter_dim_pad=int(inter_dim_pad),
        out=out,
        token_num=int(token_num),
        topk=int(topk),
    )


__all__ = [
    "OpusA8W4LaunchConfig",
    "cfg_is_supported",
    "check_route_bucket_metadata",
    "is_opus_a8w4_stage2_kernel",
    "opus_a8w4_stage2_wrapper",
    "opus_moe_stage2_a8w4_decode_fwd",
    "opus_moe_stage2_a8w4_fwd",
    "opus_moe_stage2_reduce_token_slot_route_output_fwd",
    "parse_stage2_config",
    "route_bucket_metadata",
    "stage2_launch_config",
    "stage2_uses_route_reduce",
]
