# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import functools
from typing import TYPE_CHECKING, Optional, Tuple, Union

if TYPE_CHECKING:
    from ..utility.mx_types import MxScaleRoundMode

import torch
import torch.nn.functional as F
from torch import Tensor

from aiter.jit.utils.torch_guard import torch_compile_guard

from ..jit.core import compile_ops
from ..utility import dtypes, fp4_utils
from ..utility import mx_types as _mx_types
from ..utility.mx_types import (
    MX_DEFAULT_ROUND_MODE,
    MxDtypeInt,
    MxScaleRoundModeInt,
)
from . import triton
from .enum import ActivationType, QuantType
from ..jit.utils.chip_info import get_cu_num, get_gfx

# Type alias for round-mode parameters; Union keeps int interop without
# triggering the JIT build that loading MxScaleRoundMode would cause.
RoundModeLike = Union[int, "MxScaleRoundMode"]


def __getattr__(name):
    if name in ("MxScaleRoundMode", "MxFp4RoundMode"):
        cls = _mx_types.MxScaleRoundMode
        globals()["MxScaleRoundMode"] = cls
        globals()["MxFp4RoundMode"] = cls
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
    except TypeError:
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
    x,
    scale=None,
    quant_dtype=dtypes.fp4x2,
    shuffle=False,
    pack_dim=-1,
    round_mode: RoundModeLike = MX_DEFAULT_ROUND_MODE,
):
    """Torch reference for MXFP4 (E2M1) per-1x32 block-scale quantization.

    Mirrors the HIP path (:func:`per_1x32_f4_quant_hip` /
    :func:`quant_mxfp4_hip`) and the Triton path
    (:func:`per_1x32_f4_quant_triton`); all share the E8M0 block layout and
    the :class:`MxScaleRoundMode` enum. PyTorch equivalent:
    ``torchao/prototype/mx_formats/{config,mx_tensor}.py``.

    Args:
        x: Input of shape ``(..., N)`` or ``(M, N)``.
        scale: Pre-computed scale (optional, usually ``None``).
        quant_dtype: Must be ``dtypes.fp4x2``.
        shuffle: Apply e8m0 scale shuffling for hardware.
        pack_dim: ``-1`` (default) packs the last dim -> ``tl.dot_scaled``
            **LHS** ``A(M,K) -> fp4=(M,K//2), scale=(M,K//32)``;
            ``0`` packs the first dim -> **RHS**
            ``B(K,N) -> fp4=(K//2,N), scale=(K//32,N)``.
        round_mode: :class:`MxScaleRoundMode` or bare ``int`` 0..3
            (interchangeable). Default ``RoundUp`` (industry default:
            NV ROUND_UP / DSv4 Pro / FlashInfer / torchao RCEIL).
            See :class:`MxScaleRoundMode` for the four formulas and
            cross-stack mapping.

    Returns:
        ``(quantized_tensor, scale_tensor)``.
    """
    assert quant_dtype == dtypes.fp4x2
    block_size = 32

    # Internally we always pack along the last dim. For RHS layout
    # (pack_dim=0) transpose the input here and the outputs at the end.
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

    # E8M0 block-scale dispatch via the generic helper; ``mode`` selects the
    # formula (RoundDown / RoundUp / Even / Ceil) and ``dtype`` selects the
    # ``max_pos`` / ``target_max_pow2`` / ``mbits`` constants -- see
    # MxScaleRoundMode docstring for cross-stack equivalences.
    try:
        scale_e8m0_biased = fp4_utils.f32_to_mx_e8m0_scale(
            max_abs, mode=round_mode, dtype=MxDtypeInt.FP4_E2M1
        )
    except ValueError as e:
        raise ValueError(
            "per_1x32_f4_quant: invalid "
            f"round_mode={round_mode!r} (type={type(round_mode).__name__}); "
            "expected 0 (RoundDown/FLOOR), 1 (RoundUp/RCEIL), "
            "2 (Even), 3 (Ceil), or any MxScaleRoundMode value."
        ) from e

    scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0_biased)

    y = x.float() / scale_f32.view(-1, 1)
    y = fp4_utils.f32_to_mxfp4(y)
    y = y.view(*shape_original[:-1], -1)
    scale = scale_e8m0_biased.view(m, -1).view(torch.uint8)
    if shuffle:
        scale = fp4_utils.e8m0_shuffle(scale)
    scale = scale.view(dtypes.fp8_e8m0)

    if transposed:
        y = y.T.contiguous()
        scale = scale.view(torch.uint8).T.contiguous().view(dtypes.fp8_e8m0)

    return y, scale


def per_1x32_i4_quant(weight, group_size=32):
    """Groupwise int4 [-7, 7] symmetric quantization along the last dim.

    Input  : weight       [..., N, K]
    Output : weight_qt    int8 container, same shape [..., N, K], values in [-7, 7]
             weight_scale bf16, shape [..., K // group_size, N] (G/N transposed)
    """
    *batch_dims, N, K = weight.shape
    G = K // group_size
    w_groups = weight.view(*batch_dims, N, G, group_size)
    w_group_max = w_groups.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
    w_scale_raw = (w_group_max / 7.0).squeeze(-1)
    weight_qt = (
        (w_groups / w_scale_raw.unsqueeze(-1))
        .round()
        .clamp(-7, 7)
        .to(dtypes.i8)
        .view(*batch_dims, N, K)
    )
    weight_scale = w_scale_raw.transpose(-1, -2).contiguous().to(dtypes.bf16)
    return weight_qt, weight_scale


def per_1x32_f4_quant_for_dot_scaled(
    lhs,
    rhs,
    quant_dtype=dtypes.fp4x2,
    shuffle=False,
    round_mode: RoundModeLike = MX_DEFAULT_ROUND_MODE,
):
    """Convenience function: quantize both LHS and RHS for ``tl.dot_scaled``.

    Handles the packing dimension automatically:
    - LHS A(M, K): packed along K (dim=-1) -> fp4=(M, K//2), scale=(M, K//32)
    - RHS B(K, N): packed along K (dim=0)  -> fp4=(K//2, N), scale=(K//32, N)

    Note: Triton 3.6+ expects rhs_scale in transposed form (N, K//32). Users
    should transpose the returned rhs_scale accordingly if using Triton >= 3.6.

    Args:
        lhs: LHS tensor of shape (M, K).
        rhs: RHS tensor of shape (K, N).
        round_mode: Scale rounding strategy, forwarded to
            :func:`per_1x32_f4_quant`. Default 1 (RoundUp / torchao RCEIL).

    Returns:
        Tuple of (lhs_fp4, lhs_scale, rhs_fp4, rhs_scale).
    """
    lhs_fp4, lhs_scale = per_1x32_f4_quant(
        lhs,
        quant_dtype=quant_dtype,
        shuffle=shuffle,
        pack_dim=-1,
        round_mode=round_mode,
    )
    rhs_fp4, rhs_scale = per_1x32_f4_quant(
        rhs,
        quant_dtype=quant_dtype,
        shuffle=shuffle,
        pack_dim=0,
        round_mode=round_mode,
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
        # Route the e8m0 path through the centralized MX scale API so this
        # reference honors MX_DEFAULT_ROUND_MODE (RoundUp / RCEIL =
        # ceil_pow2(amax / max_pos)), matching the HIP kernel's
        # fp_f32_to_e8m0_scale<kDefaultMxScaleRoundMode, FP8_E4M3{,_FNUZ}>.
        # The legacy f32_to_e8m0(amax / floor_pow2(max)) bypassed the default
        # and diverged by one exponent on ~1/8 of groups vs the RoundUp kernel.
        fp8_mx_dtype = (
            MxDtypeInt.FP8_E4M3_FNUZ if get_gfx() == "gfx942" else MxDtypeInt.FP8_E4M3
        )
        scale_e8m0_biased = fp4_utils.f32_to_mx_e8m0_scale(
            max_abs, mode=MX_DEFAULT_ROUND_MODE, dtype=fp8_mx_dtype
        )
        scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0_biased)

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
    # `per_1x32` points to the dtype-aware MX entry so callers can do
    # `get_hip_quant(QuantType.per_1x32)(x, quant_dtype=dtypes.fp4x2)` for
    # MXFP4 or `quant_dtype=dtypes.fp8` for MXFP8. The legacy
    # `per_1x32_f4_quant_hip` thin wrapper is preserved separately for
    # external callers that imported it by name.
    tmp = {
        QuantType.No.value: lambda *a, **k: (a[0], None),
        QuantType.per_Tensor.value: per_tensor_quant_hip,
        QuantType.per_Token.value: per_token_quant_hip,
        QuantType.per_1x32.value: per_1x32_mx_quant_hip,
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


def per_1x32_mx_quant_hip(
    x,
    scale=None,
    quant_dtype=dtypes.fp4x2,
    shuffle=False,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor=1,
    scale_type=None,
):
    """1x32 per-group MX dynamic quant (HIP).

    Dispatches by ``quant_dtype``:
      * ``dtypes.fp4x2``: MXFP4, packed fp4x2 output ``(m, n // 2)``,
        e8m0-byte scale ``(m, ceil(n/32))`` (or padded to
        ``(pad256(m), pad8(ceil(n/32)))`` when ``shuffle=True``).
        ``scale_type`` is ignored -- fp4 always produces an e8m0 byte scale.
      * ``dtypes.fp8``: MXFP8, fp8 output ``(m, n)``. The scale layout is
        chosen by ``scale_type``:
            - ``dtypes.fp32`` (default for backward compat): continuous
              fp32 per-group scale ``(m, ceil(n/32))``.
            - ``dtypes.fp8_e8m0``: e8m0 byte scale ``(m, ceil(n/32))``
              (or padded ``(pad256(m), pad8(ceil(n/32)))`` when
              ``shuffle=True``). This is the byte layout consumed by
              ``mxfp4_moe_sort_hip`` and MXFP8 GEMM kernels, enabling the
              MXFP8 "split" path  (per_1x32 quant -> sorted-scale shuffle)
              that mirrors the existing MXFP4 split path.

    The legacy ``per_1x32_f4_quant_hip`` is kept as a thin wrapper for
    backward compatibility.
    """
    m, n = x.shape
    assert n % 32 == 0, f"n={n} must be divisible by 32"
    device = x.device

    # Per-dtype defaults / validation.
    if quant_dtype == dtypes.fp4x2:
        assert n % 2 == 0
        out_cols = n // 2
        # fp4 is always e8m0; ignore caller-supplied scale_type (or assert
        # it agrees) so the public surface stays simple.
        effective_scale_type = dtypes.fp8_e8m0
        if scale_type is not None and scale_type not in (dtypes.fp8_e8m0,):
            raise ValueError(
                f"per_1x32_mx_quant_hip: fp4x2 output requires e8m0 scale; "
                f"got scale_type={scale_type}"
            )
    elif quant_dtype == dtypes.fp8:
        out_cols = n
        effective_scale_type = scale_type if scale_type is not None else dtypes.fp32
        if effective_scale_type not in (dtypes.fp32, dtypes.fp8_e8m0):
            raise ValueError(
                f"per_1x32_mx_quant_hip: fp8 output expects scale_type in "
                f"{{fp32, fp8_e8m0}}, got {effective_scale_type}"
            )
        if shuffle and effective_scale_type == dtypes.fp32:
            raise NotImplementedError(
                "per_1x32_mx_quant_hip(quant_dtype=fp8, scale_type=fp32, "
                "shuffle=True): the fp32-scale path uses a transposed "
                "(scaleN, M) layout and is not supported through this "
                "wrapper. Pass scale_type=dtypes.fp8_e8m0 for the swizzled "
                "byte-scale layout, or use fused_dynamic_mxfp8_quant_moe_sort "
                "for the MoE-sort fused path."
            )
    else:
        raise ValueError(
            f"per_1x32_mx_quant_hip: unsupported quant_dtype {quant_dtype}, "
            f"expected one of (dtypes.fp4x2, dtypes.fp8)"
        )

    # Allocate scale buffer matching the requested layout.
    is_e8m0 = effective_scale_type == dtypes.fp8_e8m0
    if scale is None:
        if is_e8m0:
            if shuffle:
                scale = torch.empty(
                    (
                        (m + 255) // 256 * 256,
                        ((n + 31) // 32 + 7) // 8 * 8,
                    ),
                    dtype=torch.uint8,
                    device=device,
                ).view(dtypes.fp8_e8m0)
            else:
                scale = torch.empty(
                    (m, (n + 31) // 32),
                    dtype=torch.uint8,
                    device=device,
                ).view(dtypes.fp8_e8m0)
        else:
            scale = torch.empty(
                (m, (n + 31) // 32),
                dtype=dtypes.fp32,
                device=device,
            )
    else:
        raise ValueError("unsupported: static per token quant")
    y = torch.empty(m, out_cols, dtype=quant_dtype, device=device)
    # `dynamic_per_group_scaled_quant` is the canonical dtype-aware C entry;
    # the C++ host dispatches by `out.dtype()` (fp4/fp8/i8) and by
    # `scales.dtype()` (e8m0 vs fp32). The legacy
    # `dynamic_per_group_scaled_quant_fp4` symbol is retained as a
    # backward-compat forwarder for callers that import it directly.
    dynamic_per_group_scaled_quant(
        y,
        x,
        scale,
        32,
        shuffle_scale=shuffle,
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
    """Backward-compat fp4-only wrapper around :func:`per_1x32_mx_quant_hip`.

    New code should call ``per_1x32_mx_quant_hip`` directly with the
    desired ``quant_dtype``; this wrapper exists so existing callers (and
    the ``get_hip_quant(QuantType.per_1x32)`` lookup before MXFP8 support
    was added) keep working unchanged.
    """
    assert (
        quant_dtype == dtypes.fp4x2
    ), "per_1x32_f4_quant_hip is fp4-only; use per_1x32_mx_quant_hip for fp8"
    return per_1x32_mx_quant_hip(
        x,
        scale=scale,
        quant_dtype=quant_dtype,
        shuffle=shuffle,
        num_rows=num_rows,
        num_rows_factor=num_rows_factor,
    )


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


@compile_ops("module_quant", develop=True)
def static_per_tensor_quant(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_quant", develop=True)
def dynamic_per_tensor_quant(out: Tensor, input: Tensor, scale: Tensor) -> None: ...


@compile_ops("module_quant", develop=True)
def dynamic_per_token_scaled_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    scale_ub: Optional[torch.Tensor] = None,
    shuffle_scale: bool = False,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
) -> None: ...


@compile_ops("module_quant", develop=True)
def dynamic_per_group_scaled_quant(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
    shuffle_scale: bool = True,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
) -> None:
    """Dtype-aware per-group dynamic quant.

    ``out.dtype`` selects the element format:
      * ``dtypes.fp4x2`` / ``torch.uint8`` -> MXFP4 (e8m0 byte scale)
      * ``dtypes.fp8``                      -> MXFP8 (fp32 per-group scale today)
      * ``dtypes.i8``                       -> int8 per-group (fp32 scale)

    Only ``group_size`` in {32, 64, 128} is supported.
    """
    ...


@compile_ops("module_quant", develop=True)
def dynamic_per_group_scaled_quant_fp4(
    out: torch.Tensor,
    input: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 32,
    shuffle_scale: bool = True,
    num_rows: Optional[torch.Tensor] = None,
    num_rows_factor: int = 1,
) -> None:
    """Backward-compat fp4x2-only forwarder; delegates to
    ``dynamic_per_group_scaled_quant``.

    Only support group_size in [32, 64, 128].
    """
    ...


@compile_ops("module_quant", develop=True)
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


@compile_ops("module_quant", develop=True)
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


@compile_ops("module_quant", develop=True)
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


@compile_ops("module_quant", develop=True)
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
    # Pad cols to multiple of 8 to match `mx_scale_shuffle_idx`'s
    # `scaleN_pad = pad8(scaleN)`. See note in `fused_dynamic_mx_quant_moe_sort`.
    scaleN_pad = ((cols + 31) // 32 + 7) // 8 * 8
    out_scale = torch.empty(
        (sorted_ids.shape[0] + 31) // 32 * 32,
        scaleN_pad,
        dtype=dtypes.fp8_e8m0,
        device=scale.device,
    )
    mxfp4_moe_sort_hip(out_scale, scale, sorted_ids, num_valid_ids, token_num, cols)
    return out_scale


@compile_ops("module_quant", develop=True)
def fused_dynamic_mx_quant_moe_sort_hip(
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
    HIP path for fused dynamic MX (fp4 or fp8) quantization and MoE scale
    sorting. The output dtype of ``out`` selects the quant target: fp4x2/uint8
    for MXFP4, fp8 for MXFP8.
    """
    ...


@compile_ops("module_quant", develop=True)
def quant_mxfp4(
    inp: torch.Tensor,
    out_packed: torch.Tensor,
    out_scale: torch.Tensor,
    group_size: int = 32,
    round_mode: int = 0,
    e8m0_shuffle: bool = False,
    a16w4_shuffle: bool = False,
    gate_up: bool = False,
    shuffle_weight: bool = False,
) -> None: ...


def quant_mxfp4_hip(
    x: torch.Tensor,
    group_size: int = 32,
    round_mode: RoundModeLike = MX_DEFAULT_ROUND_MODE,
    e8m0_shuffle: bool = False,
    a16w4_shuffle: bool = False,
    gate_up: bool = False,
    shuffle_weight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HIP MXFP4 (E2M1) per-1x32 block-scale quantization.

    Production fast path. Numerical reference is :func:`per_1x32_f4_quant`;
    Triton variant is :func:`per_1x32_f4_quant_triton`. The C++ kernel lives
    in ``csrc/kernels/quant_mxfp4.cu`` and dispatches on the same
    :class:`MxScaleRoundMode` enum. PyTorch equivalent:
    ``torchao/prototype/mx_formats/{config,mx_tensor}.py``.

    When ``round_mode == RoundUp`` and no shuffle/gate_up flags are set, this
    routes to :func:`per_1x32_f4_quant_hip` (the default FP4 path using
    ``fp_f32_to_e8m0_scale<RoundUp, FP4_E2M1> = ceil_pow2(amax/6)``).

    Args:
        x: Input tensor, 2D, contiguous, dtype ``float16`` or ``bfloat16``.
        group_size: Block size along the last dim (must divide ``cols``;
            spec = 32).
        round_mode: :class:`MxScaleRoundMode` or bare ``int`` 0..3
            (interchangeable). Default ``RoundUp`` (industry default:
            NV ROUND_UP / DSv4 Pro / FlashInfer / torchao RCEIL).
            See :class:`MxScaleRoundMode` for the four formulas and
            cross-stack mapping. (Note: ``Even`` mode uses
            HW builtin RNE on gfx950 vs SW round-half-away on gfx942.)
        e8m0_shuffle: Apply HW e8m0 scale shuffling layout.
        a16w4_shuffle: Apply A16W4 weight shuffling.
        gate_up: Pack gate / up activations together (MoE).
        shuffle_weight: Apply weight shuffling.

    Returns:
        ``(out_packed, out_scale)`` -- fp4 (``float4_e2m1fn_x2``) and e8m0
        (``float8_e8m0fnu``) tensors. With ``e8m0_shuffle=True`` the scale
        tensor is row/col-padded for the HW shuffle layout.
    """
    assert x.is_contiguous() and x.dim() == 2
    assert x.dtype in (torch.float16, torch.bfloat16)
    rows, cols = x.shape
    assert cols % group_size == 0

    # Normalise to the raw int the C++ ``static_cast<MxScaleRoundMode>(int)``
    # binding expects. ``MxScaleRoundMode`` is an ``IntEnum`` so this is also
    # a no-op (and a cheap one) for callers that already pass a bare int.
    try:
        round_mode_int = int(round_mode)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "quant_mxfp4_hip: invalid "
            f"round_mode={round_mode!r} (type={type(round_mode).__name__}); "
            "expected int in {0,1,2,3} or MxScaleRoundMode."
        ) from e

    # Default-mode fast path: per_1x32_f4_quant_hip ->
    # dynamic_per_group_scaled_quant_fp4 derives its scale via
    # fp_f32_to_e8m0_scale<kDefaultMxScaleRoundMode, FP4_E2M1>, so it is only
    # valid when the *requested* round_mode equals the project-wide default
    # (MX_DEFAULT_ROUND_MODE). Comparing against MX_DEFAULT_ROUND_MODE (not a
    # hard-coded RoundUp) keeps this routing correct if the default is ever
    # changed in lockstep with C++ kDefaultMxScaleRoundMode. Any non-default
    # mode falls through to the quant_mxfp4 kernel's explicit 4-way dispatch.
    # Skip on gfx942: dynamic_per_group_scaled_quant_fp4 requires
    # __Float4_e2m1fn_x2 which is not available on gfx942; fall through
    # to the quant_mxfp4 kernel instead.
    if (
        round_mode_int == MxScaleRoundModeInt.RoundUp
        and not a16w4_shuffle
        and not gate_up
        and not shuffle_weight
        and get_gfx() != "gfx942"
    ):
        return per_1x32_f4_quant_hip(x, shuffle=e8m0_shuffle)

    fp4x2 = getattr(torch, "float4_e2m1fn_x2", torch.uint8)
    fp8_e8m0 = getattr(torch, "float8_e8m0fnu", torch.uint8)

    out_packed = torch.empty(rows, cols // 2, dtype=fp4x2, device=x.device)

    scaleN = cols // group_size
    if e8m0_shuffle:
        scaleN_pad = ((scaleN + 7) // 8) * 8
        rows_pad = ((rows + 255) // 256) * 256
        out_scale = torch.empty(
            rows_pad, scaleN_pad, dtype=torch.uint8, device=x.device
        ).view(fp8_e8m0)
    else:
        out_scale = torch.empty(rows, scaleN, dtype=torch.uint8, device=x.device).view(
            fp8_e8m0
        )

    quant_mxfp4(
        x,
        out_packed,
        out_scale,
        group_size,
        round_mode_int,
        e8m0_shuffle,
        a16w4_shuffle,
        gate_up,
        shuffle_weight,
    )
    return out_packed, out_scale


def fused_dynamic_mx_quant_moe_sort(
    input: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,  # stage1 and stage2: same topk value
    block_size: int,
    quant_dtype: torch.dtype = dtypes.fp4x2,
    num_rows: Optional[torch.Tensor] = None,
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Unified fused dynamic MX quant + MoE-sort entry (MXFP4 / MXFP8).

    Returns ``(out, scale)`` where:
      * ``out``:
          - ``dtypes.fp4x2`` -> ``(M, N // 2)`` packed fp4 (MXFP4)
          - ``dtypes.fp8``   -> ``(M, N)`` fp8 e4m3 (MXFP8)
      * ``scale``: ``(pad32(sorted_ids.shape[0]), N // group_size)`` e8m0
                   byte, swizzled to the GEMM tile layout consumed by MXFP4
                   / MXFP8 MoE GEMM kernels. The byte memory layout is
                   identical between MXFP4 and MXFP8 (see
                   ``mx_scale_shuffle_idx`` -- the swizzle formula is
                   dtype-agnostic), only the per-byte exponent value
                   differs based on the dtype's ``max_pow2`` constant.

    Dispatch:
      * Small ``M`` (or non-default ``group_size``): single fused HIP kernel
        ``fused_dynamic_mx_quant_moe_sort_hip`` which does quant + sort +
        swizzle in one pass. Saves a kernel launch but re-quantises each
        input row up to ``topk`` times.
      * Large ``M``: split path - one pass of ``per_1x32_mx_quant_hip`` to
        produce per-token unswizzled fp8_e8m0 scale, then ``mxfp4_moe_sort_hip``
        to sort + swizzle the byte tensor. The split kernel itself is dtype-
        agnostic for the byte shuffle step (same kernel handles both MXFP4
        and MXFP8 scale bytes). Wins at large ``M`` because the quant pass
        reads each input row exactly once instead of ``topk`` times.

    Threshold: stage1 cuts over at ``M = 8*256/topk``; stage2 cuts over at
    ``M = 8*1024/topk * topk = 8*1024``. The previous ``fused_dynamic_mxfp4_*``
    /``fused_dynamic_mxfp8_*`` entries are retained as thin wrappers for
    backward compatibility.
    """
    if quant_dtype not in (dtypes.fp4x2, dtypes.fp8):
        raise ValueError(
            f"fused_dynamic_mx_quant_moe_sort: unsupported quant_dtype "
            f"{quant_dtype}, expected one of (dtypes.fp4x2, dtypes.fp8)"
        )

    token_num_quant_moe_sort_switch = [
        8 * 256 / topk,  # stage1
        8 * 1024 / topk,  # stage2
    ]
    M, N = input.view(-1, input.shape[-1]).shape
    # Packed fp4x2 stores 2 elements per byte; fp8 is 1 elem per byte.
    out_cols = N // 2 if quant_dtype == dtypes.fp4x2 else N
    is_stage1 = M == token_num
    # `eff_topk` mirrors what the HIP launcher would infer from
    # `input.numel() / (cols * token_num)` (= 1 for stage1, original topk
    # for stage2). Used as `num_rows_factor` in the split path.
    eff_topk = 1 if is_stage1 else topk
    # Scale cols pad to multiple of 8 to match `mx_scale_shuffle_idx`'s
    # `scaleN_pad = pad8(scaleN)`. Without this, MX shapes with
    # `N/group_size` not divisible by 8 (e.g. inter_dim=384 -> scaleN=12)
    # trigger OOB writes in the kernel that uses the padded stride.
    # Padding columns `[scaleN_valid, scaleN_pad)` are NOT written by the
    # kernel (guarded by `scale_k < scaleN_valid`) and contain allocator
    # garbage; production GEMM consumers don't read them so this is safe.
    # Tests comparing byte-level equality must mask out those positions.
    scaleN_pad = ((N + group_size - 1) // group_size + 7) // 8 * 8
    scale = torch.empty(
        (sorted_ids.shape[0] + 31) // 32 * 32,
        scaleN_pad,
        dtype=dtypes.fp8_e8m0,
        device=input.device,
    )
    use_fused = (
        (is_stage1 and M <= token_num_quant_moe_sort_switch[0])
        or (not is_stage1 and M <= token_num_quant_moe_sort_switch[1] * eff_topk)
        or group_size != 32
    )
    if use_fused:
        out = torch.empty(M, out_cols, dtype=quant_dtype, device=input.device)
        fused_dynamic_mx_quant_moe_sort_hip(
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
        # Split path: per-token quant produces unswizzled e8m0 byte scale,
        # then `mxfp4_moe_sort_hip` (dtype-agnostic byte shuffle) sorts +
        # swizzles it into the GEMM-consumed tile layout.
        # `per_1x32_mx_quant_hip` handles both fp4 (always e8m0 scale) and
        # fp8 (with `scale_type=fp8_e8m0` for the byte-scale split path).
        scale_type = dtypes.fp8_e8m0 if quant_dtype == dtypes.fp8 else None
        out, scale_per_token = per_1x32_mx_quant_hip(
            input,
            scale=None,
            quant_dtype=quant_dtype,
            scale_type=scale_type,
            shuffle=False,
            num_rows=num_rows,
            num_rows_factor=eff_topk,
        )
        mxfp4_moe_sort_hip(
            scale, scale_per_token, sorted_ids, num_valid_ids, token_num, N
        )
    return out, scale


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
    """Backward-compat wrapper around :func:`fused_dynamic_mx_quant_moe_sort`.

    Forces ``quant_dtype=dtypes.fp4x2``; same dispatch + behaviour as the
    unified entry. New code should call ``fused_dynamic_mx_quant_moe_sort``
    directly with the desired ``quant_dtype``.
    """
    return fused_dynamic_mx_quant_moe_sort(
        input=input,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        topk=topk,
        block_size=block_size,
        quant_dtype=dtypes.fp4x2,
        num_rows=num_rows,
        group_size=group_size,
    )


def fused_dynamic_mxfp8_quant_moe_sort(
    input: torch.Tensor,
    sorted_ids: torch.Tensor,
    num_valid_ids: torch.Tensor,
    token_num: int,
    topk: int,
    block_size: int,
    num_rows: Optional[torch.Tensor] = None,
    group_size: int = 32,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Backward-compat wrapper around :func:`fused_dynamic_mx_quant_moe_sort`.

    Forces ``quant_dtype=dtypes.fp8``; same dispatch + behaviour as the
    unified entry. New code should call ``fused_dynamic_mx_quant_moe_sort``
    directly with ``quant_dtype=dtypes.fp8``.

    Returns (fp8_out, e8m0_scale) with the scale tensor laid out as
    (pad32(M_o), N_o) fp8_e8m0 -- the same byte layout the FlyDSL stage1/stage2
    GEMM consumes. ``topk`` is accepted for parity with the fp4 wrapper but
    is inferred inside the HIP launcher from ``input.numel() / (cols * token_num)``.

    Note: ``num_rows`` was not part of the original signature but is exposed
    here so callers benefit from the new split-path dispatch on large ``M``
    without a separate update. Set to ``None`` to match the previous
    behaviour exactly.
    """
    return fused_dynamic_mx_quant_moe_sort(
        input=input,
        sorted_ids=sorted_ids,
        num_valid_ids=num_valid_ids,
        token_num=token_num,
        topk=topk,
        block_size=block_size,
        quant_dtype=dtypes.fp8,
        num_rows=num_rows,
        group_size=group_size,
    )


@compile_ops("module_quant", develop=True)
def partial_transpose(
    out: Tensor,
    input: Tensor,
    num_rows: Tensor,
) -> None: ...


@compile_ops("module_dsv4_rotate_quant", develop=True)
def rotate_activation_fp4quant_inplace(
    out: torch.Tensor,
    input: torch.Tensor,
    group_size: int = 32,
) -> None:
    """Hadamard-rotate activation, FP4-quantize, then dequantize back to BF16 in-place."""
    ...


@compile_ops("module_dsv4_rotate_quant", develop=True)
def rotate_activation(
    out: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """Apply Walsh-Hadamard transform along last dim with 1/sqrt(N) scaling."""
    ...


@compile_ops("module_dsv4_rotate_quant", develop=True)
def rope_rotate_activation_fp4quant_inplace(
    out: torch.Tensor,
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
    group_size: int = 32,
) -> None:
    """Apply interleaved RoPE to trailing ``rope_dim``, Hadamard-rotate,
    FP4-quantize, then dequantize back to BF16."""
    ...


@compile_ops("module_dsv4_rotate_quant", develop=True)
def rope_rotate_activation(
    out: torch.Tensor,
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    positions: torch.Tensor,
    rope_dim: int,
) -> None:
    """Apply interleaved RoPE to trailing ``rope_dim``, then Hadamard-rotate."""
    ...
