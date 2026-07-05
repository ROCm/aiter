# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import torch
from torch import Tensor
from ..jit.core import compile_ops
from ..jit.utils.torch_guard import torch_compile_guard
from .quant import get_dtype_max
from typing import Optional

# opus is the sole rmsnorm backend (fp16/bf16/fp32, any hidden). Only group_size/
# shuffle_scale quant and exotic dtypes fall back to module_rmsnorm_quant.
_DTYPE_CODE = {torch.float16: 0, torch.bfloat16: 1, torch.float32: 2}


# Raw C ABI (ctypes): pointers/dims/stream as int64; validated in the wrappers below.
@compile_ops("module_rmsnorm", fc_name="rms_norm_opus", ffi_type="ctypes")
def _rms_norm_opus_raw(
    out: int,
    input: int,
    weight: int,
    epsilon: float,
    rows: int,
    hidden: int,
    in_s: int,
    is_bf16: int,
    model_sensitive: int,
    gemma: int,
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
    gemma: int,
    stream: int,
) -> None: ...


@compile_ops("module_rmsnorm", fc_name="add_rms_norm_opus", ffi_type="ctypes")
def _add_rms_norm_opus_raw(
    out: int,
    input: int,
    residual_in: int,
    residual_out: int,
    weight: int,
    epsilon: float,
    rows: int,
    hidden: int,
    in_s: int,
    is_bf16: int,
    model_sensitive: int,
    gemma: int,
    stream: int,
) -> None: ...


def _check(input: Tensor, weight: Tensor, allow_row_stride: bool = False):
    assert (
        input.dtype in _DTYPE_CODE
    ), f"rms_norm_opus: fp16/bf16/fp32 only, got {input.dtype}"
    assert weight.dtype == input.dtype, "rms_norm_opus: weight dtype must match input"
    # allow_row_stride: row-contiguous is enough (kernel takes a row stride); else fully contiguous.
    if allow_row_stride:
        assert input.stride(-1) == 1, "rms_norm_opus: last dim must be contiguous"
    else:
        assert input.is_contiguous(), "rms_norm_opus: contiguous only"
    assert weight.is_contiguous(), "rms_norm_opus: weight must be contiguous"
    assert weight.shape[-1] == input.shape[-1], "rms_norm_opus: weight length != hidden"


def rms_norm_opus(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    model_sensitive: int = 0,
    gemma_norm: bool = False,
) -> None:
    """out = rmsnorm(input) * (weight [+ 1 if gemma_norm]) (fp32 accumulate)."""
    hidden = input.shape[-1]
    # The kernel takes a row stride, so a 2-D row-strided view (e.g. a torch.split slice)
    # needs no copy; anything else non-contiguous is materialized.
    row_strided_2d = input.dim() == 2 and input.stride(-1) == 1
    if not (input.is_contiguous() or row_strided_2d):
        input = input.contiguous()
    in_s = input.stride(-2) if input.dim() >= 2 else hidden
    _check(input, weight, allow_row_stride=True)
    assert out.dtype == input.dtype and out.is_contiguous(), "rms_norm_opus: bad out"
    rows = input.numel() // hidden
    _rms_norm_opus_raw(
        out.data_ptr(),
        input.data_ptr(),
        weight.data_ptr(),
        float(epsilon),
        rows,
        hidden,
        int(in_s),
        _DTYPE_CODE[input.dtype],
        int(model_sensitive),
        int(gemma_norm),
        torch.cuda.current_stream().cuda_stream,
    )


def fused_add_rms_norm_opus(
    input: Tensor,
    residual: Tensor,
    weight: Tensor,
    epsilon: float,
    model_sensitive: int = 0,
    gemma_norm: bool = False,
) -> None:
    """In place: x = input + residual; residual = x; input = rmsnorm(x) * (weight [+1])."""
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
        int(gemma_norm),
        torch.cuda.current_stream().cuda_stream,
    )


# opus mirrors of the public rmsnorm entrypoints (same signatures).
def rmsnorm2d_fwd_opus(
    input: Tensor, weight: Tensor, epsilon: float, use_model_sensitive_rmsnorm: int = 0
) -> Tensor:
    out = torch.empty_like(input)
    # Common case (fp16/bf16, non-T5, hidden <= 8192, 2-D): opus arq kernel -- the bit-exact
    # port of the HIP add_rmsnorm_quant_kernel main dispatched here. Generic kernel otherwise.
    if (
        use_model_sensitive_rmsnorm == 0
        and input.dim() == 2
        and input.element_size() == 2
        and input.shape[-1] <= 8192
    ):
        rmsnorm(out, input, weight, epsilon)
    else:
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
    gemma_norm: bool = False,
) -> None:
    """out = rmsnorm(input + residual_in) * weight; residual_out = input + residual_in.

    Single out-of-place kernel pass (no host staging copies) -- the per-layer residual-add
    path vLLM/SGLang/ATOM call, so staging copies were a ~2x regression.
    """
    hidden = input.shape[-1]
    # Common case (fp16/bf16, non-T5, hidden <= 8192, 2-D): opus arq kernel -- the bit-exact
    # port of the HIP add_rmsnorm_quant_kernel main dispatched here. Generic kernel otherwise.
    if (
        use_model_sensitive_rmsnorm == 0
        and input.dim() == 2
        and input.element_size() == 2
        and hidden <= 8192
    ):
        add_rmsnorm(out, input, residual_in, residual_out, weight, epsilon, gemma_norm)
        return
    # kernel takes an input row stride; only a non-unit last-dim stride needs materializing.
    if input.stride(-1) != 1:
        input = input.contiguous()
    in_s = input.stride(-2) if input.dim() >= 2 else hidden
    _check(input, weight, allow_row_stride=True)
    assert (
        out.dtype == input.dtype and out.is_contiguous()
    ), "add_rms_norm_opus: bad out"
    assert (
        residual_in.dtype == input.dtype and residual_out.dtype == input.dtype
    ), "add_rms_norm_opus: residual dtype mismatch"
    assert (
        residual_in.is_contiguous() and residual_out.is_contiguous()
    ), "add_rms_norm_opus: residual must be contiguous"
    rows = input.numel() // hidden
    _add_rms_norm_opus_raw(
        out.data_ptr(),
        input.data_ptr(),
        residual_in.data_ptr(),
        residual_out.data_ptr(),
        weight.data_ptr(),
        float(epsilon),
        rows,
        hidden,
        int(in_s),
        _DTYPE_CODE[input.dtype],
        int(use_model_sensitive_rmsnorm),
        int(gemma_norm),
        torch.cuda.current_stream().cuda_stream,
    )


# Fused rmsnorm + dynamic/smooth quant (int8/fp8). Unused pointers pass 0.
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
    # qmax from torch (127 / 448 / 240); out_code: 0=int8, 1=fp8.
    assert out_dtype in (
        torch.int8,
        torch.float8_e4m3fn,
        torch.float8_e4m3fnuz,
    ), f"rms_norm_quant_opus: unsupported out dtype {out_dtype}"
    return float(get_dtype_max(out_dtype)), (0 if out_dtype == torch.int8 else 1)


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


@torch_compile_guard(mutates_args=["out"], gen_fake=lambda *a, **k: None)
def rms_norm_cu(
    out: Tensor,
    input: Tensor,
    weight: Tensor,
    epsilon: float,
) -> None:
    """out = rmsnorm(input) * weight (opus; fp16/bf16/fp32)."""
    rms_norm_opus(out, input, weight, epsilon)


@torch_compile_guard(
    mutates_args=["input", "residual_in"], gen_fake=lambda *a, **k: None
)
def fused_add_rms_norm_cu(
    input: Tensor,  # input/out
    residual_in: Tensor,  # residual_in/out
    weight: Tensor,
    epsilon: float,
) -> None:
    """In-place fused add + rmsnorm (opus; fp16/bf16/fp32)."""
    fused_add_rms_norm_opus(input, residual_in, weight, epsilon)


# opus is ctypes (reads .data_ptr()), so torch.compile must not trace in -- wrap each
# entrypoint as an opaque aiter custom op with a fake impl (as the CK ops were).
def _rms_norm_fwd_fake(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    return torch.empty_like(input)


@torch_compile_guard(mutates_args=[], gen_fake=_rms_norm_fwd_fake)
def rms_norm(
    input: Tensor,
    weight: Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    """rmsnorm (opus; fp16/bf16/fp32)."""
    return rmsnorm2d_fwd_opus(input, weight, epsilon, use_model_sensitive_rmsnorm)


@torch_compile_guard(mutates_args=[], gen_fake=_rms_norm_fwd_fake)
def rmsnorm2d_fwd(
    input: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    use_model_sensitive_rmsnorm: int = 0,
) -> Tensor:
    return rmsnorm2d_fwd_opus(input, weight, epsilon, use_model_sensitive_rmsnorm)


@torch_compile_guard(
    mutates_args=["out", "residual_out"], gen_fake=lambda *a, **k: None
)
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
    rmsnorm2d_fwd_with_add_opus(
        out,
        input,
        residual_in,
        residual_out,
        weight,
        epsilon,
        use_model_sensitive_rmsnorm,
        gemma_norm,
    )


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
    if group_size == 0 and not shuffle_scale:
        rmsnorm2d_fwd_with_dynamicquant_opus(
            out, input, yscale, weight, epsilon, use_model_sensitive_rmsnorm
        )
    else:
        # grouped / shuffle quant lives in the shared module_rmsnorm_quant (n<=8192).
        assert (
            input.shape[-1] <= 8192
        ), "grouped/shuffle rmsnorm dynamicquant supports hidden<=8192"
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
    if group_size == 0 and not shuffle_scale:
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
        # grouped / shuffle quant lives in the shared module_rmsnorm_quant (n<=8192).
        assert (
            input.shape[-1] <= 8192
        ), "grouped/shuffle rmsnorm add_dynamicquant supports hidden<=8192"
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


# ---------------------------------------------------------------------------
# module_rmsnorm_quant surface, now served by opus (add_rmsnorm_quant_opus).
# out_code: -1 no-quant, 0 int8, 1 fp8, 2 fp4x2. Grouped/shuffle/fp4 + strided.
# ---------------------------------------------------------------------------
@compile_ops("module_rmsnorm", fc_name="add_rmsnorm_quant_opus_raw", ffi_type="ctypes")
def _add_rmsnorm_quant_opus_raw(
    out: int,
    rout: int,
    scale: int,
    input: int,
    rin: int,
    weight: int,
    xscale: int,
    epsilon: float,
    m: int,
    n: int,
    qmax: float,
    in_code: int,
    out_code: int,
    in_s: int,
    rin_s: int,
    rout_s: int,
    out_s: int,
    group_size: int,
    shuffle: int,
    gemma: int,
    cu_num: int,
    stream: int,
) -> None: ...


def _out_code_qmax(out_dtype):
    if out_dtype == torch.int8:
        return 0, 127.0
    if out_dtype in (torch.float8_e4m3fn, torch.float8_e4m3fnuz):
        return 1, float(get_dtype_max(out_dtype))
    return 2, 6.0  # fp4x2 (MXFP4, e8m0 block scale)


# Cache the CU count per device: get_device_properties() is an expensive host query
# that would otherwise dominate the small-n kernels (few-us) on every call.
_CU_NUM: dict = {}


def _cu_num(device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    cu = _CU_NUM.get(idx)
    if cu is None:
        cu = torch.cuda.get_device_properties(idx).multi_processor_count
        _CU_NUM[idx] = cu
    return cu


def _arq(
    out,
    input,
    weight,
    epsilon,
    out_code,
    qmax,
    scale=None,
    residual_in=None,
    residual_out=None,
    group_size=0,
    shuffle_scale=False,
    gemma_norm=False,
):
    assert input.dtype in _DTYPE_CODE
    # kernel takes a row stride (in_s); only materialize when the last dim is strided.
    if input.stride(-1) != 1:
        input = input.contiguous()
    n = input.shape[-1]
    m = input.numel() // n
    add = residual_in is not None
    cu = _cu_num(input.device)
    # pass the EXACT reciprocal of qmax: the kernel scales by it (max*(1/qmax)); an
    # in-kernel approximate reciprocal would flip int8 roundings at boundaries.
    inv_qmax = 1.0 / qmax if qmax > 0 else 0.0
    _add_rmsnorm_quant_opus_raw(
        out.data_ptr(),
        residual_out.data_ptr() if add else 0,
        scale.data_ptr() if scale is not None else 0,
        input.data_ptr(),
        residual_in.data_ptr() if add else 0,
        weight.data_ptr(),
        0,
        float(epsilon),
        m,
        n,
        float(inv_qmax),
        _DTYPE_CODE[input.dtype],
        out_code,
        input.stride(0),
        residual_in.stride(0) if add else 0,
        residual_out.stride(0) if add else 0,
        out.stride(0),
        int(group_size),
        int(shuffle_scale),
        int(gemma_norm),
        int(cu),
        torch.cuda.current_stream().cuda_stream,
    )


def rmsnorm(out, input, weight, epsilon, gemma_norm=False):
    _arq(out, input, weight, epsilon, -1, 0.0, gemma_norm=gemma_norm)


def add_rmsnorm(
    out, input, residual_in, residual_out, weight, epsilon, gemma_norm=False
):
    _arq(
        out,
        input,
        weight,
        epsilon,
        -1,
        0.0,
        residual_in=residual_in,
        residual_out=residual_out,
        gemma_norm=gemma_norm,
    )


def rmsnorm_quant(
    out,
    input,
    scale,
    weight,
    epsilon,
    group_size=0,
    shuffle_scale=False,
    gemma_norm=False,
):
    oc, qmax = _out_code_qmax(out.dtype)
    _arq(
        out,
        input,
        weight,
        epsilon,
        oc,
        qmax,
        scale=scale,
        group_size=group_size,
        shuffle_scale=shuffle_scale,
        gemma_norm=gemma_norm,
    )


def add_rmsnorm_quant(
    out,
    input,
    residual_in,
    residual_out,
    scale,
    weight,
    epsilon,
    group_size=0,
    shuffle_scale=False,
    gemma_norm=False,
):
    oc, qmax = _out_code_qmax(out.dtype)
    _arq(
        out,
        input,
        weight,
        epsilon,
        oc,
        qmax,
        scale=scale,
        residual_in=residual_in,
        residual_out=residual_out,
        group_size=group_size,
        shuffle_scale=shuffle_scale,
        gemma_norm=gemma_norm,
    )
