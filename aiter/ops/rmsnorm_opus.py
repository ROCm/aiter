# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor

from ..jit.core import compile_ops

MD_NAME = "module_rmsnorm_opus"

_DTYPE_CODE = {torch.float16: 0, torch.bfloat16: 1}


# Raw C ABI (ctypes): pointers/dims/stream travel as int64, so the C++ side needs
# no torch / HIP-runtime / aiter_tensor.h and compiles in ~0.2s. Validation and
# pointer/stream extraction happen in the Python wrappers below.
@compile_ops("module_rmsnorm_opus", fc_name="rms_norm_opus", ffi_type="ctypes")
def _rms_norm_opus_raw(
    out: int,
    input: int,
    weight: int,
    epsilon: float,
    rows: int,
    hidden: int,
    is_bf16: int,
    model_sensitive: int,
    stream: int,
) -> None: ...


@compile_ops(
    "module_rmsnorm_opus", fc_name="fused_add_rms_norm_opus", ffi_type="ctypes"
)
def _fused_add_rms_norm_opus_raw(
    input: int,
    residual: int,
    weight: int,
    epsilon: float,
    rows: int,
    hidden: int,
    is_bf16: int,
    model_sensitive: int,
    stream: int,
) -> None: ...


def _check(input: Tensor, weight: Tensor):
    assert (
        input.dtype in _DTYPE_CODE
    ), f"rms_norm_opus: bf16/fp16 only, got {input.dtype}"
    assert weight.dtype == input.dtype, "rms_norm_opus: weight dtype must match input"
    assert (
        input.is_contiguous() and weight.is_contiguous()
    ), "rms_norm_opus: contiguous only"
    assert weight.shape[-1] == input.shape[-1], "rms_norm_opus: weight length != hidden"


def rms_norm_opus(
    out: Tensor, input: Tensor, weight: Tensor, epsilon: float, model_sensitive: int = 0
) -> None:
    """out = rmsnorm(input) * weight (bf16/fp16, fp32 accumulate)."""
    _check(input, weight)
    assert out.dtype == input.dtype and out.is_contiguous(), "rms_norm_opus: bad out"
    hidden = input.shape[-1]
    rows = input.numel() // hidden
    _rms_norm_opus_raw(
        out.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        float(epsilon),
        rows,
        hidden,
        _DTYPE_CODE[input.dtype],
        int(model_sensitive),
        torch.cuda.current_stream().cuda_stream,
    )


def fused_add_rms_norm_opus(
    input: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    model_sensitive: int = 0,
) -> None:
    """In place: x = input + residual; residual = x; input = rmsnorm(x) * weight."""
    _check(input, weight)
    assert residual.dtype == input.dtype and residual.is_contiguous(), "bad residual"
    assert residual.numel() == input.numel(), "residual shape != input"
    hidden = input.shape[-1]
    rows = input.numel() // hidden
    _fused_add_rms_norm_opus_raw(
        input.data_ptr(),
        residual.data_ptr(),
        weight.data_ptr(),
        float(epsilon),
        rows,
        hidden,
        _DTYPE_CODE[input.dtype],
        int(model_sensitive),
        torch.cuda.current_stream().cuda_stream,
    )


def rms_norm(input: Tensor, weight: Tensor, epsilon: float) -> Tensor:
    """Functional wrapper mirroring aiter.ops.rmsnorm.rms_norm (allocates out)."""
    out = torch.empty_like(input)
    rms_norm_opus(out, input, weight, epsilon)
    return out


# opus mirrors of the CK entrypoints (same signatures as the *_ck functions).
def rmsnorm2d_fwd_opus(
    input: Tensor, weight: Tensor, epsilon: float, use_model_sensitive_rmsnorm: int = 0
) -> Tensor:
    out = torch.empty_like(input)
    rms_norm_opus(out, input, weight, epsilon, use_model_sensitive_rmsnorm)
    return out


def rmsnorm2d_fwd_with_add_opus(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    # opus fused kernel is in-place on (io, res); stage into out/residual_out so
    # input/residual_in are left untouched.
    out.copy_(input)
    residual_out.copy_(residual_in)
    fused_add_rms_norm_opus(
        out, residual_out, weight, epsilon, use_model_sensitive_rmsnorm
    )


# ---------------------------------------------------------------------------
# Fused rmsnorm + dynamic/smooth quant (int8/fp8 out). residual/xscale/unquant
# pointers are 0 when unused. out_code: 0=int8, 1=fp8.
# ---------------------------------------------------------------------------
@compile_ops("module_rmsnorm_opus", fc_name="rms_norm_quant_opus", ffi_type="ctypes")
def _rms_norm_quant_opus_raw(
    out: int,
    yscale: int,
    unquant: int,
    input: int,
    weight: int,
    residual: int,
    xscale: int,
    epsilon: float,
    rows: int,
    hidden: int,
    qmax: float,
    in_code: int,
    out_code: int,
    model_sensitive: int,
    stream: int,
) -> None: ...


def _qmax_outcode(out_dtype):
    if out_dtype == torch.int8:
        return 127.0, 0
    if out_dtype == torch.float8_e4m3fn:
        return 448.0, 1
    if out_dtype == torch.float8_e4m3fnuz:
        return 240.0, 1
    raise AssertionError(f"rms_norm_quant_opus: unsupported out dtype {out_dtype}")


def _quant(
    out, input, weight, yscale, xscale, residual, unquant, epsilon, model_sensitive
):
    _check(input, weight)
    qmax, out_code = _qmax_outcode(out.dtype)
    hidden = input.shape[-1]
    rows = input.numel() // hidden
    _rms_norm_quant_opus_raw(
        out.data_ptr(),
        yscale.data_ptr(),
        unquant.data_ptr() if unquant is not None else 0,
        input.data_ptr(),
        weight.data_ptr(),
        residual.data_ptr() if residual is not None else 0,
        xscale.data_ptr() if xscale is not None else 0,
        float(epsilon),
        rows,
        hidden,
        qmax,
        _DTYPE_CODE[input.dtype],
        out_code,
        int(model_sensitive),
        torch.cuda.current_stream().cuda_stream,
    )


def rmsnorm2d_fwd_with_dynamicquant_opus(
    out, input, yscale, weight, epsilon, use_model_sensitive_rmsnorm=0
) -> None:
    _quant(
        out,
        input,
        weight,
        yscale,
        None,
        None,
        None,
        epsilon,
        use_model_sensitive_rmsnorm,
    )


def rmsnorm2d_fwd_with_smoothquant_opus(
    out, input, xscale, yscale, weight, epsilon, use_model_sensitive_rmsnorm=0
) -> None:
    _quant(
        out,
        input,
        weight,
        yscale,
        xscale,
        None,
        None,
        epsilon,
        use_model_sensitive_rmsnorm,
    )


def rmsnorm2d_fwd_with_add_dynamicquant_opus(
    out,
    input,
    residual_in,
    residual_out,
    yscale,
    weight,
    epsilon,
    use_model_sensitive_rmsnorm=0,
) -> None:
    residual_out.copy_(residual_in)  # opus adds in place on the residual buffer
    _quant(
        out,
        input,
        weight,
        yscale,
        None,
        residual_out,
        None,
        epsilon,
        use_model_sensitive_rmsnorm,
    )


def rmsnorm2d_fwd_with_add_smoothquant_opus(
    out,
    input,
    residual_in,
    residual_out,
    xscale,
    yscale,
    weight,
    epsilon,
    out_before_quant=None,
    use_model_sensitive_rmsnorm=0,
) -> None:
    residual_out.copy_(residual_in)
    _quant(
        out,
        input,
        weight,
        yscale,
        xscale,
        residual_out,
        out_before_quant,
        epsilon,
        use_model_sensitive_rmsnorm,
    )
