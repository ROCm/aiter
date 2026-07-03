# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops
from typing import Optional


def _use_opus(
    input, use_model_sensitive_rmsnorm: int = 0, gemma_norm: bool = False
) -> bool:
    """True when the opus backend can serve this call.

    Opus is the sole rmsnorm backend and covers fp16/bf16/fp32 for the plain,
    fused-add, dynamic/smooth-quant and T5 paths (any hidden). gemma_norm is not
    supported and falls back to the (non-CK) module_rmsnorm_quant kernels.
    """
    return not gemma_norm and input.dtype in (
        torch.float16,
        torch.bfloat16,
        torch.float32,
    )


# ==========================================================================
# OPUS backend (self-contained: no CK/torch/HIP-runtime in the C++ TU).
# module_rmsnorm is a single ctypes TU and the sole rmsnorm backend
# (fp16/bf16/fp32). Only gemma_norm/group_size/shuffle_scale fall back to the
# separate module_rmsnorm_quant kernels.
# ==========================================================================
_DTYPE_CODE = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}


# Raw C ABI (ctypes): pointers/dims/stream travel as int64, so the C++ side needs
# no torch / HIP-runtime / aiter_tensor.h and compiles in ~0.2s. Validation and
# pointer/stream extraction happen in the Python wrappers below.
@compile_ops("module_rmsnorm", fc_name="rms_norm_opus", ffi_type="ctypes")
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


@compile_ops("module_rmsnorm", fc_name="fused_add_rms_norm_opus", ffi_type="ctypes")
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
    ), f"rms_norm_opus: fp16/bf16/fp32 only, got {input.dtype}"
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


# opus mirrors of the public rmsnorm entrypoints (same signatures).
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
@compile_ops("module_rmsnorm", fc_name="rms_norm_quant_opus", ffi_type="ctypes")
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


def rms_norm_cu(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """out = rmsnorm(input) * weight (opus; fp16/bf16/fp32)."""
    rms_norm_opus(out, input, weight, epsilon)


def fused_add_rms_norm_cu(
    input: Tensor,  # input/out
    residual_in: Tensor,  # residual_in/out
    weight: Tensor,
    epsilon: float,
) -> None:
    """In-place fused add + rmsnorm (opus; fp16/bf16/fp32)."""
    fused_add_rms_norm_opus(input, residual_in, weight, epsilon)


def rms_norm(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    """rmsnorm (opus; fp16/bf16/fp32)."""
    if _use_opus(input, use_model_sensitive_rmsnorm):
        return rmsnorm2d_fwd_opus(input, weight, epsilon, use_model_sensitive_rmsnorm)
    out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    rmsnorm(out, input, weight, epsilon)
    return out


def rmsnorm2d_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    if _use_opus(input, use_model_sensitive_rmsnorm):
        return rmsnorm2d_fwd_opus(input, weight, epsilon, use_model_sensitive_rmsnorm)
    out = torch.empty_like(input, dtype=input.dtype, device=input.device)
    rmsnorm(out, input, weight, epsilon)
    return out


def rmsnorm2d_fwd_with_add(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    gemma_norm: bool = False,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    if _use_opus(input, use_model_sensitive_rmsnorm, gemma_norm):
        rmsnorm2d_fwd_with_add_opus(
            out,
            input,
            residual_in,
            residual_out,
            weight,
            epsilon,
            use_model_sensitive_rmsnorm,
        )
        return
    add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon, gemma_norm)


def rmsnorm2d_fwd_with_smoothquant(
    out: Tensor,
    input: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    rmsnorm2d_fwd_with_smoothquant_opus(
        out, input, xscale, yscale, weight, epsilon, use_model_sensitive_rmsnorm
    )


def rmsnorm2d_fwd_with_add_smoothquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    xscale: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    out_before_quant: Optional[Tensor] = None,
    use_model_sensitive_rmsnorm: int = 0,
) -> None:
    rmsnorm2d_fwd_with_add_smoothquant_opus(
        out,
        input,
        residual_in,
        residual_out,
        xscale,
        yscale,
        weight,
        epsilon,
        out_before_quant,
        use_model_sensitive_rmsnorm,
    )


def rmsnorm2d_fwd_with_dynamicquant(
    out: Tensor,
    input: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None:
    if _use_opus(input) and group_size == 0 and not shuffle_scale:
        rmsnorm2d_fwd_with_dynamicquant_opus(
            out, input, yscale, weight, epsilon, use_model_sensitive_rmsnorm
        )
    else:
        rmsnorm_quant(out, input, yscale, weight, epsilon, group_size, shuffle_scale)


def rmsnorm2d_fwd_with_add_dynamicquant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    yscale: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
    group_size: int = 0,
    shuffle_scale: bool = False,
) -> None:
    if _use_opus(input) and group_size == 0 and not shuffle_scale:
        rmsnorm2d_fwd_with_add_dynamicquant_opus(
            out,
            input,
            residual_in,
            residual_out,
            yscale,
            weight,
            epsilon,
            use_model_sensitive_rmsnorm,
        )
    else:
        add_rmsnorm_quant(
            out,
            input,
            residual_in,
            residual_out,
            yscale,
            weight,
            epsilon,
            group_size,
            shuffle_scale,
        )


@compile_ops("module_rmsnorm_quant")
def add_rmsnorm_quant(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    scale: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = 0,
    shuffle_scale: bool = False,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops("module_rmsnorm_quant")
def add_rmsnorm(
    out: Tensor,
    input: Tensor,
    residual_in: Tensor,
    residual_out: Tensor,
    weight: Tensor,
    epsilon: float,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops("module_rmsnorm_quant")
def rmsnorm_quant(
    out: Tensor,
    input: Tensor,
    scale: Tensor,
    weight: Tensor,
    epsilon: float,
    group_size: int = 0,
    shuffle_scale: bool = False,
    gemma_norm: bool = False,
) -> None: ...


@compile_ops("module_rmsnorm_quant")
def rmsnorm(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    gemma_norm: bool = False,
) -> None: ...
