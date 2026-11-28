# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import operator
from collections.abc import Sequence

import torch
import triton

# Channel-block and padding granularity for prepacked inputs and weights.
# Per-architecture JSON configurations were tuned with this value; changing
# it requires validating and retuning the convolution configurations.
BLOCK_K = 64

_PADDING_MODES = ("zeros", "reflect", "replicate", "circular")


def _is_amd_wave32():
    """Return whether the active Triton target is an AMD wave32 device."""
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "hip" and target.warp_size == 32


def _out_hw(H, W, R, S, stride, padding, dilation):
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    return P, Q


def _validate_conv_devices(x, weight, bias=None, *, op: str):
    """Require input, weight, and optional bias on one CUDA/ROCm device."""
    if not x.is_cuda:
        message = f"{op} input must be a CUDA/ROCm tensor"
    elif weight.device != x.device or (bias is not None and bias.device != x.device):
        message = f"{op} input, weight, and bias must be on the same device"
    else:
        return
    raise ValueError(
        f"{message}, got input={x.device}, weight={weight.device}, "
        f"bias={bias.device if bias is not None else None}"
    )


def _conv_dims(x, w_oihw, stride, padding, dilation):
    """Shared wrapper preamble: validate inputs and return the conv dimensions."""
    N, C, H, W_in = x.shape
    K_out, Cw, R, S = w_oihw.shape
    assert Cw == C
    P, Q = _out_hw(H, W_in, R, S, stride, padding, dilation)
    return N, C, H, W_in, K_out, R, S, P, Q


def _validate_conv_groups(groups, in_channels, out_channels, weight_channels, bias):
    """Check grouped weight and bias dimensions and return the integer group count."""
    try:
        count = operator.index(groups)
    except TypeError:
        raise ValueError(f"groups must be a positive integer, got {groups!r}") from None
    if isinstance(groups, bool) or count <= 0:
        raise ValueError(f"groups must be a positive integer, got {groups!r}")
    if in_channels % count:
        raise ValueError(
            f"input channels {in_channels} must be divisible by groups={count}"
        )
    if out_channels < count or out_channels % count:
        raise ValueError(
            f"output channels {out_channels} must be a positive multiple of groups={count}"
        )
    if weight_channels != in_channels // count:
        raise ValueError(
            f"weight has {weight_channels} input channels per group, "
            f"expected {in_channels // count} for groups={count}"
        )
    if bias is not None and bias.shape != (out_channels,):
        raise ValueError(
            f"bias must have shape ({out_channels},), got {tuple(bias.shape)}"
        )
    return count


def _alloc_output(N, K_out, P, Q, x, layout):
    """Allocate the output tensor, channels_last for nhwc else contiguous."""
    return torch.empty(
        (N, K_out, P, Q),
        device=x.device,
        dtype=x.dtype,
        memory_format=(
            torch.channels_last if layout == "nhwc" else torch.contiguous_format
        ),
    )


def _prep_bias(bias):
    """Return a unit-stride bias without launching a dtype-conversion kernel.

    The Triton epilogues promote fp16/bf16 bias values to fp32 while loading.
    """
    if bias is None:
        return None
    return bias if bias.stride(0) == 1 else bias.contiguous()


def _ensure_layout(x, layout):
    """Materialize the physical layout expected by a convolution kernel."""
    if layout == "nhwc":
        return x.contiguous(memory_format=torch.channels_last)
    if layout == "ndhwc":
        return x.contiguous(memory_format=torch.channels_last_3d)
    if layout in ("nchw", "ncdhw"):
        return x.contiguous()
    raise ValueError(
        f"layout must be one of 'nchw', 'nhwc', 'ncdhw', or 'ndhwc', got {layout!r}"
    )


def _is_1x1_conv(R, S, dilation):
    """Check if this is a 1x1 convolution (no spatial reduction in kernel)."""
    return R == 1 and S == 1 and dilation == (1, 1)


def _is_3x3_conv(R, S):
    """Check if this is a 3x3 convolution."""
    return R == 3 and S == 3


def _is_winograd_2d_eligible(R, S, stride, dilation, C=None):
    if not (R == 3 and S == 3 and stride == (1, 1) and dilation == (1, 1)):
        return False
    # F(4,3) output transform amplifies bf16 rounding by up to 361x (AT row3 L1=19).
    # With very few input channels the tolerance budget is too small to absorb this.
    return not (C is not None and C < 4)


def _require_winograd_2d_eligible(name, R, S, stride, dilation, C):
    """Raise a uniform ValueError if this shape isn't Winograd F(4,3)-eligible."""
    if not _is_winograd_2d_eligible(R, S, stride, dilation, C):
        raise ValueError(
            f"{name} requires 3x3 kernel with stride=1, dilation=1, "
            f"and C >= 4 (F(4,3) output transform amplifies rounding by up to "
            f"361x; C<4 has too few reduction terms to absorb it), "
            f"got {R}x{S} stride={stride} dilation={dilation} C={C}"
        )


def _normalize_tuple(
    value, dimensions: int, name: str, *, allow_zero: bool = False
) -> tuple[int, ...]:
    """Normalize an integer or fixed-length sequence to a spatial tuple."""
    if isinstance(value, int):
        result = (value,) * dimensions
    elif isinstance(value, Sequence) and len(value) == dimensions:
        result = tuple(int(v) for v in value)
    else:
        raise ValueError(
            f"{name} must be an int or a length-{dimensions} sequence, got {value!r}"
        )

    minimum = 0 if allow_zero else 1
    if any(v < minimum for v in result):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} values must be {relation}, got {result}")
    return result


def _normalize_conv2d_params(stride, padding, dilation):
    return (
        _normalize_tuple(stride, 2, "stride"),
        _normalize_tuple(padding, 2, "padding", allow_zero=True),
        _normalize_tuple(dilation, 2, "dilation"),
    )


def _validate_padding_mode(
    padding_mode: str, spatial: tuple[int, ...], padding: tuple[int, ...]
) -> None:
    """Validate the mode and spatial limits after non-negative normalization."""
    if padding_mode not in _PADDING_MODES:
        raise ValueError(
            f"padding_mode must be one of {_PADDING_MODES}, got {padding_mode!r}"
        )
    if any(size <= 0 for size in spatial):
        raise ValueError(f"input spatial dimensions must be positive, got {spatial}")
    if padding_mode == "reflect" and any(
        amount >= size for amount, size in zip(padding, spatial, strict=True)
    ):
        raise ValueError(
            "reflect padding must be less than each input spatial dimension, "
            f"got padding={padding} for input {spatial}"
        )
    if padding_mode == "circular" and any(
        amount > size for amount, size in zip(padding, spatial, strict=True)
    ):
        raise ValueError(
            "circular padding must not exceed each input spatial dimension, "
            f"got padding={padding} for input {spatial}"
        )


def _padding_geometry(
    spatial: tuple[int, ...], padding: tuple[int, ...], padding_mode: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return effective dimensions and kernel padding for normalized, valid inputs."""
    if padding_mode == "zeros" or not any(padding):
        return tuple(spatial), padding
    return (
        tuple(size + 2 * amount for size, amount in zip(spatial, padding, strict=True)),
        (0,) * len(padding),
    )


# ---------------------------------------------------------------------------
# Conv3D helpers
# ---------------------------------------------------------------------------


def _normalize_conv3d_params(stride, padding, dilation):
    return (
        _normalize_tuple(stride, 3, "stride"),
        _normalize_tuple(padding, 3, "padding", allow_zero=True),
        _normalize_tuple(dilation, 3, "dilation"),
    )


def _out_dhw(D, H, W, T, R, S, stride, padding, dilation):
    """Return the standard Conv3D output depth, height, and width."""
    sd, sh, sw = stride
    pd, ph, pw = padding
    dd, dh, dw = dilation
    OD = (D + 2 * pd - dd * (T - 1) - 1) // sd + 1
    P = (H + 2 * ph - dh * (R - 1) - 1) // sh + 1
    Q = (W + 2 * pw - dw * (S - 1) - 1) // sw + 1
    return OD, P, Q


def _conv3d_dims(x, w_oidhw, stride, padding, dilation):
    """Validate NCDHW/OIDHW tensors and return all convolution dimensions."""
    if x.ndim != 5:
        raise ValueError(f"conv3d input must be 5-D NCDHW, got {x.ndim}-D")
    if w_oidhw.ndim != 5:
        raise ValueError(f"conv3d weight must be 5-D OIDHW, got {w_oidhw.ndim}-D")
    N, C, D, H, W_in = x.shape
    K_out, Cw, T, R, S = w_oidhw.shape
    if Cw != C:
        raise ValueError(f"weight in-channels {Cw} != input channels {C}")
    OD, P, Q = _out_dhw(D, H, W_in, T, R, S, stride, padding, dilation)
    if OD <= 0 or P <= 0 or Q <= 0:
        raise ValueError(
            "calculated conv3d output is non-positive: "
            f"({OD}, {P}, {Q}) for input {(D, H, W_in)} and kernel {(T, R, S)}"
        )
    return N, C, D, H, W_in, K_out, T, R, S, OD, P, Q


def _alloc_output_3d(N, K_out, OD, P, Q, x, layout):
    """Allocate logical NCDHW output in contiguous or channels-last-3d form."""
    return torch.empty(
        (N, K_out, OD, P, Q),
        device=x.device,
        dtype=x.dtype,
        memory_format=(
            torch.channels_last_3d if layout == "ndhwc" else torch.contiguous_format
        ),
    )


def _is_1x1x1_conv(T, R, S, dilation):
    return T == 1 and R == 1 and S == 1 and dilation == (1, 1, 1)


def _is_3x3x3_conv(T, R, S):
    return T == 3 and R == 3 and S == 3


def _is_winograd_hw_3d_eligible(T, R, S, stride, dilation, C=None):
    if not _is_3x3x3_conv(T, R, S):
        return False
    if stride != (1, 1, 1) or dilation != (1, 1, 1):
        return False
    return not (C is not None and C < 4)


def _require_winograd_hw_3d_eligible(name, T, R, S, stride, dilation, C):
    """Raise if a shape cannot use Conv3D's H/W-only Winograd F(4,3)."""
    if not _is_winograd_hw_3d_eligible(T, R, S, stride, dilation, C):
        raise ValueError(
            f"{name} requires a 3x3x3 kernel with stride=1, dilation=1, "
            f"and C >= 4, got {T}x{R}x{S} stride={stride} "
            f"dilation={dilation} C={C}"
        )


def _winograd_transform_storage_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use fp16 Winograd operands for bf16 input to reduce transform error."""
    return torch.float16 if dtype == torch.bfloat16 else dtype
