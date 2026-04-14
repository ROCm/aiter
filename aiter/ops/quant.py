# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from aiter.jit.utils.torch_guard import torch_compile_guard

from ..jit.core import compile_ops
from ..utility import dtypes, fp4_utils
from . import triton
from .enum import ActivationType, QuantType
from ..jit.utils.chip_info import get_cu_num


@compile_ops("module_smoothquant")
def smoothquant_fwd(
    out: Tensor, input: Tensor, x_scale: Tensor, y_scale: Tensor
) -> None: ...


@compile_ops("module_smoothquant")
def moe_smoothquant_fwd(
    out: Tensor, input: Tensor, x_scale: Tensor, topk_ids: Tensor, y_scale: Tensor
) -> None: ...


# following are pure torch implement
@functools.lru_cache()
def get_dtype_max(dtype):
    try:
        dtypeMax = torch.finfo(dtype).max
    except:
        dtypeMax = torch.iinfo(dtype).max
    return dtypeMax


def pertoken_quant(
    x,
    scale=None,
    x_scale=None,  # smooth_scale
    scale_dtype=dtypes.fp32,
    quant_dtype=dtypes.i8,
    dtypeMax=None,
):
    x = x.to(dtypes.fp32)
    if x_scale is None:
        hidden_states = x
    else:
        # smooth quant
        hidden_states = x * x_scale

    if dtypeMax is None:
        dtypeMax = get_dtype_max(quant_dtype)

    per_token_scale = scale
    if scale is None:
        # [m, 1]
        per_token_amax, _ = torch.max(
            input=torch.abs(hidden_states), dim=-1, keepdim=True
        )
        per_token_scale = per_token_amax / dtypeMax
        per_token_scale[per_token_scale == 0] = 1

    # quant hidden_states
    y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
    y_scale = per_token_scale.to(scale_dtype)
    return y, y_scale


def per_1x32_f4_quant(
    x, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False, pack_dim=-1
):
    """Quantize a tensor to MXFP4 (e2m1) format with per-1x32 block scaling.

    By default, packing is along the last dimension (dim=-1), which produces
    output suitable for ``tl.dot_scaled`` **LHS** operand:
        A(M, K) -> fp4=(M, K//2), scale=(M, K//32)

    For ``tl.dot_scaled`` **RHS** operand, set ``pack_dim=0`` so the packing
    is along the first dimension (the K / contraction dimension):
        B(K, N) -> fp4=(K//2, N), scale=(K//32, N)

    Args:
        x: Input tensor of shape (..., N) or (M, N).
        scale: Pre-computed scale tensor (optional, usually None).
        quant_dtype: Target quantized dtype, must be ``dtypes.fp4x2``.
        shuffle: Whether to apply e8m0 scale shuffling for hardware.
        pack_dim: Dimension along which to pack two FP4 values into one byte.
            -1 (default): pack along the last dimension (for dot_scaled LHS).
             0: pack along the first dimension (for dot_scaled RHS).

    Returns:
        Tuple of (quantized_tensor, scale_tensor).
    """
    assert quant_dtype == dtypes.fp4x2
    block_size = 32
    F8E8M0_EXP_BIAS = 127  # noqa:F841
    F4E2M1_MAX = 6.0
    MAX_POW2 = int(torch.log2(torch.tensor(F4E2M1_MAX, dtype=torch.float32)).item())
    # dtypeMax = F4E2M1_MAX
    dtypeMax = 2.0**MAX_POW2

    # For pack_dim=0, transpose so packing always happens along last dim internally
    transposed = False
    if pack_dim == 0:
        assert x.dim() == 2, "pack_dim=0 requires a 2D input tensor (K, N)"
        x = x.T.contiguous()
        transposed = True

    shape_original = x.shape
    x = x.view(-1, shape_original[-1])

    m, n = x.shape
    x = x.view(-1, block_size)
    max_abs = torch.amax(torch.abs(x.float()), 1)
    # max_abs = max_abs.view(torch.int32)
    # max_abs = ((max_abs + 0x200000) & 0xFF800000).view(torch.float32)

    # fp8e8m0fnu_from_fp32_value
    scale_e8m0_biased = fp4_utils.f32_to_e8m0(max_abs / dtypeMax)

    # Float8_e8m0fnu to float
    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0_biased)

    y = x.float() / scale_f32.view(-1, 1)
    y = fp4_utils.f32_to_mxfp4(y)
    y = y.view(*shape_original[:-1], -1)
    scale = scale_e8m0_biased.view(m, -1).view(torch.uint8)
    if shuffle:
        scale = fp4_utils.e8m0_shuffle(scale)
    scale = scale.view(dtypes.fp8_e8m0)

    # For pack_dim=0, transpose results back: (N, K//2) -> (K//2, N)
    if transposed:
        y = y.T.contiguous()
        scale = scale.view(torch.uint8).T.contiguous().view(dtypes.fp8_e8m0)

    return y, scale


def per_1x32_f4_quant_for_dot_scaled(lhs, rhs, quant_dtype=dtypes.fp4x2,shuffle=False):
    """Convenience function: quantize both LHS and RHS for ``tl.dot_scaled``.

    Handles the packing dimension automatically:
    - LHS A(M, K): packed along K (dim=-1) -> fp4=(M, K//2), scale=(M, K//32)
    - RHS B(K, N): packed along K (dim=0)  -> fp4=(K//2, N), scale=(K//32, N)

    Note: Triton 3.6+ expects rhs_scale in transposed form (N, K//32). Users
    should transpose the returned rhs_scale accordingly if using Triton >= 3.6.

    Args:
        lhs: LHS tensor of shape (M, K).
        rhs: RHS tensor of shape (K, N).

    Returns:
        Tuple of (lhs_fp4, lhs_scale, rhs_fp4, rhs_scale).
    """
    lhs_fp4, lhs_scale = per_1x32_f4_quant(
        lhs, quant_dtype=quant_dtype, shuffle=shuffle, pack_dim=-1
    )
    rhs_fp4, rhs_scale = per_1x32_f4_quant(
        rhs, quant_dtype=quant_dtype, shuffle=shuffle, pack_dim=0
    )
    return lhs_fp4, lhs_scale, rhs_fp4, rhs_scale


def per_1x32_f8_scale_f8_quant(
    x, scale=None, quant_dtype=dtypes.fp8, scale_type=dtypes.fp32, shuffle=False
):
    assert quant_dtype == dtypes.fp8
    block_size = 32
    dtypeMax = 448.0
    MAX_POW2 = int(torch.log2(torch.tensor(dtypeMax, dtype=torch.float32)).item())
    dtypeMax = 2.0**MAX_POW2

    shape_original = x.shape
    x = x.view(-1, shape_original[-1])

    m, n = x.shape
    x = x.view(-1, block_size)
    max_abs = torch.amax(torch.abs(x.float()), 1)

    # fp8e8m0fnu_from_fp32_value
    if scale_type == dtypes.fp32:
        scale_f32 = max_abs / dtypeMax
        scale_e8m0_biased = None
    else:
        scale_e8m0_biased = fp4_utils.f32_to_e8m0(max_abs / dtypeMax)
        scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0_biased)
        # scale_f32 = max_abs / dtypeMax

    y = x.float() / scale_f32.view(-1, 1)
    y = y.view(*shape_original[:-1], -1)
    if scale_type == dtypes.fp32:
        scale = scale_f32.view(m, -1)
    else:
        scale = scale_e8m0_biased.view(m, -1)  # .view(torch.uint8)
        if shuffle:
            scale = fp4_utils.e8m0_shuffle(scale)
    return y.to(quant_dtype), scale


def per_tensor_quant(
    x, scale=None, scale_dtype=dtypes.fp32, quant_dtype=dtypes.i8, dtypeMax=None
):
    x = x.to(dtypes.fp32)
    if scale is None:
        if dtypeMax is None:
            dtypeMax = get_dtype_max(quant_dtype)
        scale = torch.abs(x).max() / dtypeMax
    y = x / scale

    return y.to(quant_dtype), scale.view(1).to(scale_dtype)


def per_block_quant_wrapper(block_shape=(1, 128)):
    def decorator(per_token_quant_func):
        def wrapper(x, scale=None, quant_dtype=dtypes.i8):
            blk_m, blk_n = block_shape
            assert (
                x.shape[-1] % blk_n == 0
            ), f"block size {blk_n} not match {x.shape[-1]}"
            assert blk_m == 1, "only support 1xN block, TODO: support MxN"
            m, n = x.shape
            x = x.view(-1, blk_n)
            y, scale = per_token_quant_func(x, scale=scale, quant_dtype=quant_dtype)
            return y.view(m, n), scale.view(m, n // blk_n)

        return wrapper

    return decorator


@functools.lru_cache()
def get_torch_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant,
        QuantType.per_Token: pertoken_quant,
        QuantType.per_1x32: per_1x32_f4_quant,
        QuantType.per_1x128: per_block_quant_wrapper((1, 128))(pertoken_quant),
        QuantType.per_256x128: per_block_quant_wrapper((256, 128))(pertoken_quant),
        QuantType.per_1024x128: per_block_quant_wrapper((1024, 128))(pertoken_quant),
    }

    def raise_NotImplementedError(*a, **k):
        raise NotImplementedError(f"unsupported quant type {qType=}")

    return tmp.get(qType, raise_NotImplementedError)


@functools.lru_cache()
def get_hip_quant(qType):
    tmp = {
        QuantType.No.value: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor.value: per_tensor_quant_hip,
        QuantType.per_Token.value: per_token_quant_hip,
        QuantType.per_1x32.value: per_1x32_f4_quant_hip,
        QuantType.per_1x128.value: functools.partial(
            per_group_quant_hip, group_size=128
        ),
    }

    def raise_NotImplementedError(*a, **k):
        raise NotImplementedError(f"unsupported quant type {qType=}")

    return tmp.get(qType.value, raise_NotImplementedError)


@functools.lru_cache()
def get_triton_quant(qType):
    tmp = {
        QuantType.No: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor: per_tensor_quant_triton,
        QuantType.per_Token: per_token_quant_triton,
        QuantType.per_1x32: per_1x32_f4_quant_triton,
        QuantType.per_1x128: per_block_quant_wrapper((1, 128))(per_token_quant_triton),
    }

    def raise_NotImplementedError(*a, **k):
        raise NotImplementedError(f"unsupported quant type {qType=}")

    return tmp.get(qType, raise_NotImplementedError)


@torch_compile_guard()
def per_token_quant_hip(
    x: Tensor,
    scale: Optional[Tensor] = None,
    quant_dtype: torch.dtype = dtypes.i8,
    num_rows: Optional[Tensor] = None,
    num_rows_factor: int = 1,
) -> Tuple[Tensor, Tensor]:
    shape = x.shape
    device = x.device
    if scale is None:
        scale = torch.empty((*shape[:-1], 1), dtype=dtypes.fp32, device=device)
    else:
        raise ValueError("unsupported: static per token quant")

    if 1:
        y = torch.empty(shape, dtype=quant_dtype, device=device)
        dynamic_per_token_scaled_quant(
            y, x, scale, num_rows=num_rows, num_rows_factor=num_rows_factor
        )
    elif quant_dtype == dtypes.i8:
        M, N = x.view(-1, shape[-1]).shape
        y = torch.empty((M, N), dtype=dtypes.i8, device=device)
        scale = torch.empty(M, dtype=dtypes.fp32, device=device)
        smooth_scale = torch.ones(N, dtype=dtypes.fp32, device=device)
        smoothquant_fwd(y, x, smooth_scale, scale)
        y = y.view(shape)
    else:
        raise ValueError(f"unsupported: {quant_dtype=}")
    # print("finished per token quant hip")
    return y, scale


@torch_compile_guard()
def per_group_quant_hip(
    x: Tensor,
    scale: Optional[Tensor] = None,
    quant_dtype: torch.dtype = dtypes.i8,
    group_size: int = 128,
    transpose_scale: bool = False,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
) -> Tuple[Tensor, Tensor]:
    shape = x.shape
    device = x.device
    if scale is None:
        scale = torch.empty(
            (*shape[:-1], shape[-1] // group_size), dtype=dtypes.fp32, device=device
        )
    else:
        raise ValueError("unsupported: static per token quant")
    assert group_size in [
        32,
        64,
        128,
    ], f"unsupported group size {group_size=}, only support [32, 64, 128]"
    y = torch.empty(shape, dtype=quant_dtype, device=device)
    dynamic_per_token_scaled_quant(
        y,
        x.view(-1, group_size),
        scale,
        shuffle_scale=transpose_scale,
        num_rows=num_rows,
        num_rows_factor=num_rows_factor,
    )
    return y, scale


def per_1x32_f4_quant_hip(
    x,
    scale=None,
    quant_dtype=dtypes.fp4x2,
    shuffle=False,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor=1,
):
    m, n = x.shape
    assert quant_dtype == dtypes.fp4x2
    assert n % 2 == 0
    device = x.device
    if scale is None:
        if shuffle:
            scale = (
                torch.empty(
                    (
                        (m + 255) // 256 * 256,
                        ((n + 31) // 32 + 7) // 8 * 8,
                    ),
                    dtype=torch.uint8,
                    device=device,
                )
                # .fill_(0x7F)
                .view(dtypes.fp8_e8m0)
            )
        else:
            scale = (
                torch.empty(
                    (m, (n + 31) // 32),
                    dtype=torch.uint8,
                    device=device,
                )
                # .fill_(0x7F)
                .view(dtypes.fp8_e8m0)
            )
    else:
        raise ValueError("unsupported: static per token quant")
    y = torch.empty(m, n // 2, dtype=quant_dtype, device=device)
    dynamic_per_group_scaled_quant_fp4(
        y,
        x,
        scale,
        32,
        shuffle_scale=shuffle,
        num_rows=num_rows,
        num_rows_factor=num_rows_factor,
    )
    return y, scale


def per_tensor_quant_hip(
    x,
    scale=None,
    quant_dtype=dtypes.i8,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor=1,
):
    assert num_rows is None, "num_rows is not supported for per_tensor_quant_hip"
    y = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    if quant_dtype in [dtypes.fp8, dtypes.i8]:
        if scale is None:
            scale = torch.empty(1, dtype=dtypes.fp32, device=x.device)
            dynamic_per_tensor_quant(y, x, scale)
        else:
            static_per_tensor_quant(y, x, scale)
    else:
        raise ValueError(f"unsupported: {quant_dtype=}")
    return y, scale.view(1)


def per_token_quant_triton(x, scale=None, quant_dtype=dtypes.i8):
    shape = x.shape
    device = x.device
    y = torch.empty(shape, dtype=quant_dtype, device=device)
    if scale is None:
        scale = torch.empty((*shape[:-1], 1), dtype=dtypes.fp32, device=device)
        triton.quant.dynamic_per_token_quant_fp8_i8(y, x.view(-1, x.shape[-1]), scale)
    else:
        raise ValueError("unsupported: static per token quant")

    return y, scale


def per_1x32_f4_quant_triton(x, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False):
    assert quant_dtype == dtypes.fp4x2
    # y, scale = triton.quant.dynamic_mxfp4_quant(x)
    y, scale = fp4_utils.dynamic_mxfp4_quant(x, shuffle=shuffle)
    return y.view(quant_dtype), scale


def per_tensor_quant_triton(x, scale=None, quant_dtype=dtypes.i8):
    y = torch.empty(x.shape, dtype=quant_dtype, device=x.device)
    x = x.view(-1, x.shape[-1])
    if scale is None:
        scale = torch.zeros(1, dtype=dtypes.fp32, device=x.device)
        triton.quant.dynamic_per_tensor_quant_fp8_i8(y, x, scale)
    else:
        triton.quant.static_per_tensor_quant_fp8_i8(y, x, scale)
    return y, scale


@functools.lru_cache()
def get_torch_act(aType):
    tmp = {
        ActivationType.No: lambda *a, **k: a[0],
        ActivationType.Silu: F.silu,
        ActivationType.Gelu: F.gelu,
    }
    return tmp.get(aType, NotImplementedError)


def moe_smooth_per_token_scaled_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    smooth_scale: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    block_m: int,
    local_expert_hash: Optional[torch.Tensor] = None,
    shuffle_scale: bool = False,
    transpose_out: bool = False,
    is_balanced: bool = False,
) -> None:
    cu_num = get_cu_num()
    is_moe_stage1 = input.numel() != out.numel()
    M = input.shape[0]
    if is_moe_stage1 and local_expert_hash is not None and M < cu_num * 8:
        if is_balanced:
            moe_smooth_per_token_scaled_quant_v1(
                out,
                input,
                scales,
                smooth_scale,
                topk_ids,
                shuffle_scale,
                local_expert_hash,
                transpose_out,
            )
        else:
            topk = topk_ids.shape[1]
            model_dim = input.shape[-1]
            smooth_per_token_scaled_quant(
                out.view(topk, M, model_dim).transpose(0, 1),
                input.view(M, 1, model_dim).expand(-1, topk, -1),
                scales,
                smooth_scale,
                topk_ids,
                smooth_scale_map_hash=local_expert_hash,
                enable_ps=True,
            )
    else:
        moe_smooth_per_token_scaled_quant_v2(
            out,
            input,
            scales,
            smooth_scale,
            sorted_token_ids,
            sorted_expert_ids,
            num_valid_ids,
            block_m,
            shuffle_scale,
            transpose_out,
        )


@compile_ops("module_quant")
def static_per_tensor_quant(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_quant")
def dynamic_per_tensor_quant(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_quant")
def dynamic_per_token_scaled_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    shuffle_scale: bool = False,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
) -> None: ...


@compile_ops("module_quant")
def dynamic_per_group_scaled_quant_fp4(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
    shuffle_scale: bool = True,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
) -> None:
    """
    Only support group_size in [32, 64, 128]
    """
    ...


@compile_ops("module_quant")
def smooth_per_token_scaled_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    smooth_scale: torch.Tensor,
    smooth_scale_map: Optional[torch.Tensor] = None,
    shuffle_scale: bool = False,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
    smooth_scale_map_hash: Optional[torch.Tensor] = None,
    enable_ps: bool = True,
) -> None: ...


@compile_ops("module_quant")
def moe_smooth_per_token_scaled_quant_v1(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    smooth_scale: torch.Tensor,
    smooth_scale_map: torch.Tensor,
    shuffle_scale: bool = False,
    smooth_scale_map_hash: Optional[torch.Tensor] = None,
    transpose_out: bool = False,
) -> None:
    """
    v1: token loops along topk experts. Only supports moe stage1.
    """
    ...


@compile_ops("module_quant")
def moe_smooth_per_token_scaled_quant_v2(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    smooth_scale: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    sorted_expert_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    block_m: int,
    shuffle_scale: bool = False,
    transpose_out: bool = False,
) -> None:
    """
    v2: expert loops along sorted_token_ids. Supports both moe stage1 and stage2.
    """
    ...


@compile_ops("module_quant")
def mxfp4_moe_sort_hip(
    out_scale: torch.Tensor,
    scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    cols: int,
) -> None:
    """
    MoE scale sorting with MXFP4 shuffle layout.
    """
    ...


def mxfp4_moe_sort_fwd(
    scale: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    cols: int,
):
    out_scale = torch.empty(
        (sorted_ids.shape[0] + 31) // 32 * 32,
        (cols + 31) // 32,
        dtype=dtypes.fp8_e8m0,
        device=scale.device,
    )
    mxfp4_moe_sort_hip(out_scale, scale, sorted_ids, num_valid_ids, token_num, cols)
    return out_scale


@compile_ops("module_quant")
def fused_dynamic_mxfp4_quant_moe_sort_hip(
    out: torch.Tensor,
    scales: torch.Tensor,
    input: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    block_m: int,
    group_size: int = 32,
) -> None:
    """
    HIP path for fused dynamic MXFP4 quantization and MoE scale sorting.
    """
    ...


def fused_dynamic_mxfp4_quant_moe_sort(
    input: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,  # stage1 and stage2: same topk value
    block_size: int,
    num_rows: Optional[torch.Tensor] = None,
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    token_num_quant_moe_sort_switch = [
        8 * 64 / topk,  # stage1
        8 * 1024 / topk,  # stage2
    ]
    M, N = input.view(-1, input.shape[-1]).shape
    is_stage1 = M == token_num
    topk = 1 if is_stage1 else topk
    scale = torch.empty(
        (sorted_ids.shape[0] + 31) // 32 * 32,
        (N + 31) // 32,
        dtype=dtypes.fp8_e8m0,
        device=input.device,
    )
    if (
        (is_stage1 and M <= token_num_quant_moe_sort_switch[0])
        or (not is_stage1 and M <= token_num_quant_moe_sort_switch[1] * topk)
        or group_size != 32
    ):
        out = torch.empty(M, N // 2, dtype=dtypes.fp4x2, device=input.device)
        fused_dynamic_mxfp4_quant_moe_sort_hip(
            out,
            scale,
            input,
            sorted_ids,
            num_valid_ids,
            token_num,
            block_size,
            group_size,
        )
    else:
        out, scale_ = per_1x32_f4_quant_hip(
            input, None, dtypes.fp4x2, num_rows=num_rows, num_rows_factor=topk
        )
        mxfp4_moe_sort_hip(scale, scale_, sorted_ids, num_valid_ids, token_num, N)
    return out, scale


@compile_ops("module_quant")
def partial_transpose(
    out: Tensor,
    input: Tensor,
    num_rows: Tensor,
) -> None: ...
