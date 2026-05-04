#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
from aiter import dtypes
from aiter.ops.triton.quant.fused_fp8_quant import fused_rms_fp8_group_quant
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_rms_mxfp4_quant
from aiter.test_common import benchmark, checkAllclose, perftest
from aiter.utility import fp4_utils

MI308_BW_MAX_TBPS = 5.3


def _rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return F.rms_norm(x.float(), (x.shape[-1],), w.float(), eps).to(x.dtype)


def _gemma_rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    """Gemma-style RMSNorm: x * rsqrt(mean(x^2) + eps) * (1 + w)."""
    xf = x.float()
    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    normed = xf * torch.rsqrt(variance + eps)
    return (normed * (1.0 + w.float())).to(x.dtype)


def _fp4x2_hip_supported_gfx() -> set[str]:
    # Opus fp4 cast path is only implemented for architectures with fp4 builtins.
    return {"gfx950", "gfx1250"}


def _per_token_group_fp8_quant_ref(
    x: torch.Tensor,
    group_size: int,
    dtype_quant: torch.dtype = dtypes.fp8,
) -> tuple[torch.Tensor, torch.Tensor]:
    m, n = x.shape
    xg = x.view(m, n // group_size, group_size).float()
    dmax = torch.finfo(dtype_quant).max
    x_max = torch.amax(torch.abs(xg), dim=-1, keepdim=True)
    x_max = torch.where(x_max < 1e-10, torch.full_like(x_max, 1e-10), x_max)
    scale = x_max / dmax
    q = torch.clamp(xg / scale, -dmax, dmax).to(dtype_quant).view(m, n)
    return q, scale.squeeze(-1)


def _per_token_group_fp4x2_quant_ref(
    x: torch.Tensor,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Mirror HIP fp4x2 path: power-of-two scale and e8m0-style exponent storage.
    m, n = x.shape
    xg = x.view(m, n // group_size, group_size).float()
    x_max = torch.amax(torch.abs(xg), dim=-1, keepdim=True)
    x_max = torch.where(x_max < 1e-10, torch.full_like(x_max, 1e-10), x_max)

    x_max_bits = x_max.view(torch.int32)
    x_max_rounded = ((x_max_bits + (1 << 22)) & 0xFF800000).view(torch.float32)
    quant_scale = x_max_rounded * 0.25
    x_scaled = (xg / quant_scale).view(m, n)

    q_fp4x2 = fp4_utils.f32_to_mxfp4(x_scaled)
    scale_exp = ((quant_scale.view(torch.int32) >> 23) & 0xFF).to(torch.uint8)
    return q_fp4x2, scale_exp.squeeze(-1)


def _to_transposed_scale_layout(scale: torch.Tensor) -> torch.Tensor:
    m, g = scale.shape
    return scale.transpose(0, 1).contiguous().view(m, g)


def _recover_row_major_scale(
    scale: torch.Tensor, transpose_scale: bool
) -> torch.Tensor:
    if not transpose_scale:
        return scale
    m, g = scale.shape
    return scale.view(g, m).transpose(0, 1).contiguous()


def _upcast_group_fp8(
    x_q: torch.Tensor, x_s: torch.Tensor, group_size: int
) -> torch.Tensor:
    m, n = x_q.shape
    return (
        x_q.float().view(m, n // group_size, group_size) * x_s.float().view(m, -1, 1)
    ).view(m, n)


def _upcast_group_fp4x2(
    x_q: torch.Tensor, x_s: torch.Tensor, group_size: int
) -> torch.Tensor:
    m = x_q.shape[0]
    n = x_s.shape[1] * group_size
    packed_cols = n // 2
    x_q_u8 = x_q.view(torch.uint8) if x_q.dtype == dtypes.fp4x2 else x_q
    if x_q_u8.shape[1] < packed_cols:
        raise ValueError(
            f"fp4x2 packed columns insufficient: got {x_q_u8.shape[1]}, need {packed_cols}"
        )
    if x_q_u8.shape[1] != packed_cols:
        x_q_u8 = x_q_u8[:, :packed_cols]
    q_f32 = fp4_utils.mxfp4_to_f32(x_q_u8).view(m, n // group_size, group_size)
    s_f32 = fp4_utils.e8m0_to_f32(x_s).float().view(m, -1, 1)
    return (q_f32 * s_f32).view(m, n)


def run_torch_ref(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: torch.Tensor | None,
    x2_weight: torch.Tensor | None,
    x2_epsilon: float | None,
    group_size: int,
    res1: torch.Tensor | None,
    transpose_scale: bool,
    quant_out_dtype: torch.dtype = dtypes.fp8,
    gemma_norm: bool = False,
):
    x1_in = x1 if res1 is None else x1 + res1
    norm_fn = _gemma_rmsnorm_ref if gemma_norm else _rmsnorm_ref
    x1_norm = norm_fn(x1_in, x1_weight, x1_epsilon)
    if quant_out_dtype == dtypes.fp8:
        x1_q, x1_s = _per_token_group_fp8_quant_ref(x1_norm, group_size, dtypes.fp8)
    elif quant_out_dtype == dtypes.fp4x2:
        if transpose_scale:
            raise ValueError(
                "fp4x2 path currently does not support transpose_scale=True"
            )
        x1_q, x1_s = _per_token_group_fp4x2_quant_ref(x1_norm, group_size)
    else:
        raise ValueError(f"Unsupported quant_out_dtype={quant_out_dtype}")

    if transpose_scale and quant_out_dtype == dtypes.fp8:
        x1_s = _to_transposed_scale_layout(x1_s)

    x2_norm = None
    if x2 is not None:
        assert x2_weight is not None
        x2_norm = norm_fn(
            x2, x2_weight, x2_epsilon if x2_epsilon is not None else x1_epsilon
        )
    return (x1_q, x1_s), x1_norm, x2_norm, x1_in


def _tensor_bytes(x: torch.Tensor | None) -> int:
    if x is None:
        return 0
    return x.numel() * x.element_size()


def _calc_io_bytes(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x2: torch.Tensor | None,
    x2_weight: torch.Tensor | None,
    res1: torch.Tensor | None,
    x1_q: torch.Tensor,
    x1_s: torch.Tensor,
    x1_unquantized: torch.Tensor | None,
    x2_out: torch.Tensor | None,
    res_out: torch.Tensor | None,
) -> int:
    return (
        _tensor_bytes(x1)
        + _tensor_bytes(x1_weight)
        + _tensor_bytes(x2)
        + _tensor_bytes(x2_weight)
        + _tensor_bytes(res1)
        + _tensor_bytes(x1_q)
        + _tensor_bytes(x1_s)
        + _tensor_bytes(x1_unquantized)
        + _tensor_bytes(x2_out)
        + _tensor_bytes(res_out)
    )


def _focus_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    # Keep the summary focused on shape + time + bandwidth + error.
    focus_cols = [
        "dtype",
        "quant_type",
        "gfx",
        "token",
        "num_head1",
        "num_head2",
        "head_dim",
        "residual",
        "triton_us",
        "hip_us",
        "uplift",
        "triton_bw_TBps",
        "hip_bw_TBps",
        "triton_error_rate",
        "hip_error_rate",
        "note",
    ]
    return df[[c for c in focus_cols if c in df.columns]]


@perftest()
def run_triton(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: torch.Tensor | None,
    x2_weight: torch.Tensor | None,
    x2_epsilon: float | None,
    group_size: int,
    res1: torch.Tensor | None,
    output_unquantized_inp1: bool,
    transpose_scale: bool,
):
    return fused_rms_fp8_group_quant(
        x1,
        x1_weight,
        x1_epsilon,
        x2,
        x2_weight,
        x2_epsilon,
        group_size,
        dtypes.fp8,
        res1,
        output_unquantized_inp1,
        transpose_scale,
    )


@perftest()
def run_triton_fp4(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: torch.Tensor | None,
    x2_weight: torch.Tensor | None,
    x2_epsilon: float | None,
    res1: torch.Tensor | None,
    output_unquantized_inp1: bool,
):
    # Triton MXFP4 RMS path uses fixed quant block size 32 and e8m0-like scales.
    return fused_rms_mxfp4_quant(
        x1,
        x1_weight,
        x1_epsilon,
        x2=x2,
        x2_weight=x2_weight,
        x2_epsilon=(x2_epsilon if x2_epsilon is not None else x1_epsilon),
        res1=res1,
        shuffle=False,
        scale_shuffle_padding=False,
        output_unquantized_inp1=output_unquantized_inp1,
    )


@perftest()
def run_hip(
    x1: torch.Tensor,
    x1_weight: torch.Tensor,
    x1_epsilon: float,
    x2: torch.Tensor | None,
    x2_weight: torch.Tensor | None,
    x2_epsilon: float | None,
    group_size: int,
    res1: torch.Tensor | None,
    output_unquantized_inp1: bool,
    transpose_scale: bool,
    quant_out_dtype: torch.dtype,
    gemma_norm: bool = False,
    no_quant: bool = False,
):
    m, n1 = x1.shape
    num_scale_cols = n1 // group_size
    if no_quant:
        x1_q = None
        x1_s = None
    elif quant_out_dtype == dtypes.fp8:
        x1_q = torch.empty((m, n1), dtype=dtypes.fp8, device=x1.device)
        if transpose_scale:
            # Match Triton's transposed-storage convention while keeping the public shape [m, g].
            x1_s = torch.empty(
                (num_scale_cols, m), dtype=torch.float32, device=x1.device
            ).view(m, num_scale_cols)
        else:
            x1_s = torch.empty(
                (m, num_scale_cols), dtype=torch.float32, device=x1.device
            )
    elif quant_out_dtype == dtypes.fp4x2:
        x1_q = torch.empty((m, n1 // 2), dtype=dtypes.fp4x2, device=x1.device)
        x1_s = torch.empty((m, num_scale_cols), dtype=torch.uint8, device=x1.device)
    else:
        raise ValueError(f"Unsupported quant_out_dtype={quant_out_dtype}")
    # In no-quant mode q_out_unquantized is the only x1 output; force-allocate it.
    x1_u = torch.empty_like(x1) if (output_unquantized_inp1 or no_quant) else None
    x2_out = torch.empty_like(x2) if x2 is not None else None
    res_out = torch.empty_like(x1) if res1 is not None else None

    aiter.fused_qk_rmsnorm_group_quant(
        x1_q,
        x1_s,
        x1,
        x1_weight,
        x1_epsilon,
        x1_u,
        x2_out,
        res_out,
        x2,
        x2_weight,
        x2_epsilon,
        res1,
        group_size,
        transpose_scale,
        gemma_norm,
    )
    return (x1_q, x1_s), x1_u, x2_out, res_out


@benchmark()
def test_fused_qk_rmsnorm_group_quant(
    dtype: torch.dtype,
    token: int,
    num_head1: int,
    num_head2: int,
    add_residual: bool,
    head_dim: int = 128,
    group_size: int = 128,
    output_unquantized_inp1: bool = False,
    transpose_scale: bool = True,
    quant_out_dtype: torch.dtype = dtypes.fp8,
    gemma_norm: bool = False,
    no_quant: bool = False,
):
    # No-quant mode only writes q_out_unquantized; force unquantized output and skip Triton.
    if no_quant:
        output_unquantized_inp1 = True
    assert token > 0
    assert num_head1 > 0
    assert num_head2 >= 0
    assert head_dim > 0
    if quant_out_dtype == dtypes.fp4x2:
        if (
            getattr(torch, "float4_e2m1fn_x2", None) is None
            or dtypes.fp4x2 == torch.uint8
        ):
            raise RuntimeError(
                "fp4x2 quant_out_dtype was requested but torch.float4_e2m1fn_x2 is unavailable"
            )
        if transpose_scale:
            raise ValueError(
                "fp4x2 path currently does not support transpose_scale=True"
            )
    quant_out_dtype_name = (
        "fp8"
        if quant_out_dtype == dtypes.fp8
        else ("fp4x2" if quant_out_dtype == dtypes.fp4x2 else str(quant_out_dtype))
    )
    gfx = aiter.get_gfx()

    if quant_out_dtype == dtypes.fp4x2 and gfx not in _fp4x2_hip_supported_gfx():
        note = (
            f"skip: HIP fp4x2 is unsupported on {gfx}; "
            f"supported architectures: {sorted(_fp4x2_hip_supported_gfx())}"
        )
        aiter.logger.warning(
            "[skip] dtype=%s quant_type=%s token=%d num_head1=%d num_head2=%d head_dim=%d "
            "group_size=%d residual=%s | %s",
            dtype,
            quant_out_dtype_name,
            token,
            num_head1,
            num_head2,
            head_dim,
            group_size,
            add_residual,
            note,
        )
        m = token
        n1 = num_head1 * head_dim
        n2 = num_head2 * head_dim
        return {
            "dtype": str(dtype),
            "quant_type": quant_out_dtype_name,
            "quant_out_dtype": quant_out_dtype_name,
            "gfx": gfx,
            "token": token,
            "num_head1": num_head1,
            "num_head2": num_head2,
            "head_dim": head_dim,
            "heads1": num_head1,
            "heads2": num_head2,
            "M": m,
            "N1": n1,
            "N2": n2,
            "residual": add_residual,
            "triton_us": None,
            "hip_us": None,
            "uplift": "N/A",
            "triton_bw_TBps": None,
            "hip_bw_TBps": None,
            "hip_bw_peak_ratio": "N/A",
            "triton_error_rate": None,
            "hip_error_rate": None,
            "triton_mae": None,
            "hip_mae": None,
            "triton_max_abs_err": None,
            "hip_max_abs_err": None,
            "x1_deq_err_rate_triton": None,
            "x1_deq_err_rate_hip": None,
            "x2_err_rate_triton": None,
            "x2_err_rate_hip": None,
            "res_err_rate_triton": None,
            "res_err_rate_hip": None,
            "x1_unq_err_rate_triton": None,
            "x1_unq_err_rate_hip": None,
            "note": note,
        }

    m = token
    n1 = num_head1 * head_dim
    n2 = num_head2 * head_dim

    assert n1 % group_size == 0
    assert n1 % head_dim == 0
    if n2 > 0:
        assert n2 % group_size == 0
        assert n2 % head_dim == 0

    # Build strided x1/x2 by splitting one full tensor, mirroring model flow:
    # q_c, kv_c, _ = torch.split(full, [n1, n2, tail], dim=-1)
    split_tail_dim = head_dim
    if n2 > 0:
        full_qk = (
            torch.randn((m, n1 + n2 + split_tail_dim), dtype=dtype, device="cuda") / 10
        )
        x1, x2, _ = torch.split(full_qk, [n1, n2, split_tail_dim], dim=1)
    else:
        full_q = torch.randn((m, n1 + split_tail_dim), dtype=dtype, device="cuda") / 10
        x1, _ = torch.split(full_q, [n1, split_tail_dim], dim=1)
        x2 = None
    assert x1.stride(1) == 1 and x1.stride(0) >= n1 and not x1.is_contiguous()
    if n2 > 0:
        assert x2 is not None
        assert x2.stride(1) == 1 and x2.stride(0) >= n2 and not x2.is_contiguous()

    x1_weight = (
        torch.randn((num_head1, head_dim), dtype=dtype, device="cuda")
        .reshape(n1)
        .contiguous()
    )
    x2_weight = (
        torch.randn((num_head2, head_dim), dtype=dtype, device="cuda")
        .reshape(n2)
        .contiguous()
        if n2 > 0
        else None
    )
    if add_residual:
        full_res = (
            torch.randn((m, n1 + split_tail_dim), dtype=dtype, device="cuda") / 10
        )
        res1, _ = torch.split(full_res, [n1, split_tail_dim], dim=1)
        assert res1.stride(1) == 1 and res1.stride(0) >= n1 and not res1.is_contiguous()
    else:
        res1 = None

    torch_out = run_torch_ref(
        x1,
        x1_weight,
        1e-6,
        x2,
        x2_weight,
        1e-6 if n2 > 0 else None,
        group_size,
        res1,
        transpose_scale,
        quant_out_dtype,
        gemma_norm=gemma_norm,
    )
    has_triton_fp8 = quant_out_dtype == dtypes.fp8 and not gemma_norm and not no_quant
    has_triton_fp4 = (
        quant_out_dtype == dtypes.fp4x2
        and group_size == 32
        and not transpose_scale
        and not gemma_norm
        and not no_quant
    )
    has_triton = has_triton_fp8 or has_triton_fp4
    if has_triton_fp8:
        triton_out, triton_us = run_triton(
            x1,
            x1_weight,
            1e-6,
            x2,
            x2_weight,
            1e-6 if n2 > 0 else None,
            group_size,
            res1,
            output_unquantized_inp1,
            transpose_scale,
        )
    elif has_triton_fp4:
        triton_out, triton_us = run_triton_fp4(
            x1,
            x1_weight,
            1e-6,
            x2,
            x2_weight,
            1e-6 if n2 > 0 else None,
            res1,
            output_unquantized_inp1,
        )
    else:
        triton_out, triton_us = None, None
        if quant_out_dtype == dtypes.fp4x2:
            aiter.logger.info(
                "[note] fp4x2 Triton baseline skipped (requires group_size=32 and transpose_scale=False)"
            )
    hip_out, hip_us = run_hip(
        x1,
        x1_weight,
        1e-6,
        x2,
        x2_weight,
        1e-6 if n2 > 0 else None,
        group_size,
        res1,
        output_unquantized_inp1,
        transpose_scale,
        quant_out_dtype,
        gemma_norm=gemma_norm,
        no_quant=no_quant,
    )

    (x1_q_torch, x1_s_torch), x1_torch, x2_torch, res_torch = torch_out
    if has_triton:
        (x1_q_triton, x1_s_triton), x1_triton, x2_triton, res_triton = triton_out
    else:
        x1_q_triton = None
        x1_s_triton = None
        x1_triton = None
        x2_triton = None
        res_triton = None
    (x1_q_hip, x1_s_hip), x1_hip, x2_hip, res_hip = hip_out

    triton_error_rate = None
    hip_error_rate = None
    if no_quant:
        # Quant outputs are not produced; only x1_unquantized vs torch reference is meaningful.
        # checkAllclose against x1_torch is performed below in the output_unquantized_inp1 block.
        pass
    else:
        if quant_out_dtype == dtypes.fp8:
            q_atol = 0.05
            q_rtol = 0.05
            x1_deq_torch = _upcast_group_fp8(
                x1_q_torch,
                _recover_row_major_scale(x1_s_torch, transpose_scale),
                group_size,
            )
            x1_deq_hip = _upcast_group_fp8(
                x1_q_hip,
                _recover_row_major_scale(x1_s_hip, transpose_scale),
                group_size,
            )
            x1_deq_triton = (
                _upcast_group_fp8(
                    x1_q_triton,
                    _recover_row_major_scale(x1_s_triton, transpose_scale),
                    group_size,
                )
                if has_triton
                else None
            )
        elif quant_out_dtype == dtypes.fp4x2:
            q_atol = 0.5
            q_rtol = 0.5
            x1_deq_torch = _upcast_group_fp4x2(x1_q_torch, x1_s_torch, group_size)
            x1_deq_hip = _upcast_group_fp4x2(x1_q_hip, x1_s_hip, group_size)
            x1_deq_triton = (
                _upcast_group_fp4x2(x1_q_triton, x1_s_triton, group_size)
                if has_triton
                else None
            )
        else:
            raise ValueError(f"Unsupported quant_out_dtype={quant_out_dtype}")

        if has_triton:
            checkAllclose(
                x1_deq_torch,
                x1_deq_triton,
                rtol=q_rtol,
                atol=q_atol,
                msg=f"check dequantized x1 torch vs triton, m={m}, n1={n1}, n2={n2}",
            )
        checkAllclose(
            x1_deq_torch,
            x1_deq_hip,
            rtol=q_rtol,
            atol=q_atol,
            msg=f"check dequantized x1 torch vs hip, m={m}, n1={n1}, n2={n2}",
        )
        if has_triton:
            checkAllclose(
                x1_deq_triton,
                x1_deq_hip,
                rtol=q_rtol,
                atol=q_atol,
                msg=f"check dequantized x1, m={m}, n1={n1}, n2={n2}",
            )

        hip_error_rate = checkAllclose(
            x1_deq_torch, x1_deq_hip, rtol=q_rtol, atol=q_atol, printLog=False
        )
        if has_triton:
            triton_error_rate = checkAllclose(
                x1_deq_torch, x1_deq_triton, rtol=q_rtol, atol=q_atol, printLog=False
            )

    if x2 is not None:
        if has_triton:
            checkAllclose(
                x2_torch,
                x2_triton,
                rtol=0.02,
                atol=0.02,
                msg=f"check x2 torch vs triton, m={m}, n2={n2}",
            )
        checkAllclose(
            x2_torch,
            x2_hip,
            rtol=0.02,
            atol=0.02,
            msg=f"check x2 torch vs hip, m={m}, n2={n2}",
        )
        if has_triton:
            checkAllclose(
                x2_triton,
                x2_hip,
                rtol=0.02,
                atol=0.02,
                msg=f"check x2, m={m}, n2={n2}",
            )

    if res1 is not None:
        if has_triton:
            checkAllclose(
                res_torch,
                res_triton,
                rtol=0.02,
                atol=0.02,
                msg=f"check residual torch vs triton, m={m}, n1={n1}",
            )
        checkAllclose(
            res_torch,
            res_hip,
            rtol=0.02,
            atol=0.02,
            msg=f"check residual torch vs hip, m={m}, n1={n1}",
        )
        if has_triton:
            checkAllclose(
                res_triton,
                res_hip,
                rtol=0.02,
                atol=0.02,
                msg=f"check residual, m={m}, n1={n1}",
            )

    if output_unquantized_inp1:
        if has_triton:
            checkAllclose(
                x1_torch,
                x1_triton,
                rtol=0.02,
                atol=0.02,
                msg=f"check unquantized x1 torch vs triton, m={m}, n1={n1}",
            )
        checkAllclose(
            x1_torch,
            x1_hip,
            rtol=0.02,
            atol=0.02,
            msg=f"check unquantized x1 torch vs hip, m={m}, n1={n1}",
        )
        if has_triton:
            checkAllclose(
                x1_triton,
                x1_hip,
                rtol=0.02,
                atol=0.02,
                msg=f"check unquantized x1, m={m}, n1={n1}",
            )

    io_bytes = _calc_io_bytes(
        x1=x1,
        x1_weight=x1_weight,
        x2=x2,
        x2_weight=x2_weight,
        res1=res1,
        x1_q=x1_q_hip,
        x1_s=x1_s_hip,
        x1_unquantized=x1_hip,
        x2_out=x2_hip,
        res_out=res_hip,
    )
    hip_bw_tbps = io_bytes / (hip_us * 1e-6) / 1e12
    triton_bw_tbps = io_bytes / (triton_us * 1e-6) / 1e12 if has_triton else None
    uplift = (triton_us / hip_us - 1) if has_triton else None

    info = (
        f"dtype={dtype}, quant_out_dtype={quant_out_dtype_name}, token={token}, "
        f"num_head1={num_head1}, num_head2={num_head2}, "
        f"head_dim={head_dim}, m={m}, n1={n1}, n2={n2}, residual={add_residual}, "
        f"group_size={group_size}, transpose_scale={transpose_scale}"
    )
    if has_triton:
        aiter.logger.info(
            "[result] %s | time(us): triton=%.2f hip=%.2f uplift=%.1f%% | "
            "bw(TB/s): triton=%.3f hip=%.3f hip/mi308_peak=%.1f%% | "
            "err: triton_rate=%.6f hip_rate=%.6f",
            info,
            triton_us,
            hip_us,
            uplift * 100.0,
            triton_bw_tbps,
            hip_bw_tbps,
            (hip_bw_tbps / MI308_BW_MAX_TBPS) * 100.0,
            triton_error_rate,
            hip_error_rate,
        )
    else:
        hip_err_str = "N/A" if hip_error_rate is None else ("%.6f" % hip_error_rate)
        aiter.logger.info(
            "[result] %s | time(us): hip=%.2f | "
            "bw(TB/s): hip=%.3f hip/mi308_peak=%.1f%% | "
            "err: hip_rate=%s",
            info,
            hip_us,
            hip_bw_tbps,
            (hip_bw_tbps / MI308_BW_MAX_TBPS) * 100.0,
            hip_err_str,
        )

    return {
        "dtype": str(dtype),
        "quant_type": quant_out_dtype_name,
        "gfx": gfx,
        "token": token,
        "num_head1": num_head1,
        "num_head2": num_head2,
        "head_dim": head_dim,
        "residual": add_residual,
        "triton_us": triton_us,
        "hip_us": hip_us,
        "uplift": f"{uplift:.1%}" if uplift is not None else "N/A",
        "triton_bw_TBps": triton_bw_tbps,
        "hip_bw_TBps": hip_bw_tbps,
        "triton_error_rate": triton_error_rate,
        "hip_error_rate": hip_error_rate,
        "note": "",
    }


if __name__ == "__main__":
    l_dtype = ["bf16"]
    l_quant_type = ["fp8"]
    # Auto-include fp4x2 on architectures that support it (e.g. gfx950 / mi355).
    _gfx = aiter.get_gfx()
    _fp4x2_available = (
        _gfx in _fp4x2_hip_supported_gfx()
        and getattr(torch, "float4_e2m1fn_x2", None) is not None
        and dtypes.fp4x2 != torch.uint8
    )
    if _fp4x2_available:
        l_quant_type = ["fp8", "fp4x2"]
    # DeepSeekV2 MLA realistic default:
    # q_lora_rank ~= 1536 (12 * 128), kv_lora_rank ~= 512 (4 * 128), usually n1 > n2.
    l_token = [32, 256, 8192, 16384]
    l_num_head1 = [12]
    l_num_head2 = [4]
    l_head_dim = [128]
    l_residual = [0, 1]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Compare HIP fused_qk_rmsnorm_group_quant against Triton baselines.\n"
            "On gfx950/gfx1250 both fp8 and fp4x2 are tested by default.\n"
            "Use --quant_out_dtype to override."
        ),
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        nargs="*",
        default=None,
        help="Data type(s). e.g. -d bf16 or -d bf16 fp16",
    )
    parser.add_argument(
        "--quant_out_dtype",
        type=str,
        choices=["fp8", "fp4x2"],
        nargs="*",
        default=None,
        help=(
            "Quant output dtype(s). Overrides auto-detected defaults. "
            "fp8 enables Triton fused_rms_fp8_group_quant baseline; "
            "fp4x2 enables Triton fused_rms_mxfp4_quant baseline only when "
            "group_size=32 and transpose_scale=False."
        ),
    )
    parser.add_argument(
        "-t",
        "--token",
        type=int,
        nargs="*",
        default=None,
        help="Token count(s), equivalent to M.",
    )
    parser.add_argument(
        "--num_head1",
        type=int,
        nargs="*",
        default=None,
        help="Head count(s) for x1.",
    )
    parser.add_argument(
        "--num_head2",
        type=int,
        nargs="*",
        default=None,
        help="Head count(s) for x2 (0 means no second input).",
    )
    parser.add_argument(
        "--head_dim",
        type=int,
        nargs="*",
        default=None,
        help="Head dimension(s). Final hidden size will be num_head * head_dim.",
    )
    parser.add_argument(
        "--broad_sweep",
        action="store_true",
        help="Expand the default head/residual test matrix for broader stress/perf sweep "
        "(num_head1=[1,12,56], num_head2=[0,1,4], residual=[0,1]).",
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=None,
        help="[legacy] Alias of --token.",
    )
    parser.add_argument(
        "-n1",
        "--n1",
        type=int,
        nargs="*",
        default=None,
        help="[legacy] x1 hidden size. Will be converted to num_head1 by head_dim.",
    )
    parser.add_argument(
        "-n2",
        "--n2",
        type=int,
        nargs="*",
        default=None,
        help="[legacy] x2 hidden size (0 means no second input). Will be converted by head_dim.",
    )
    parser.add_argument(
        "--residual",
        type=int,
        nargs="*",
        default=None,
        choices=[0, 1],
        help="Whether to include residual input, 0 or 1",
    )
    parser.add_argument("--group_size", type=int, default=128)
    parser.add_argument("--output_unquantized_inp1", action="store_true")
    parser.add_argument("--transpose_scale", action="store_true")
    parser.add_argument(
        "--gemma_norm",
        action="store_true",
        help="Test gemma-style RMSNorm: x * rsqrt(mean(x^2)+eps) * (1+w) instead of standard * w.",
    )
    parser.add_argument(
        "--no_quant",
        action="store_true",
        help=(
            "Also exercise the no-quant path (q_out_scale=None): kernel only does RMSNorm "
            "and writes q_out_unquantized. Adds the no_quant=True axis alongside no_quant=False."
        ),
    )
    args = parser.parse_args()

    if args.dtype is not None:
        l_dtype = args.dtype
    if args.quant_out_dtype is not None:
        l_quant_type = args.quant_out_dtype
    if args.broad_sweep:
        l_num_head1 = [1, 12, 56]
        l_num_head2 = [0, 1, 4]
        l_residual = [0, 1]
    if args.head_dim is not None:
        l_head_dim = args.head_dim
    token_override = args.token if args.token is not None else args.m
    if token_override is not None:
        l_token = token_override
    has_legacy_hidden = args.n1 is not None or args.n2 is not None
    has_head_count = args.num_head1 is not None or args.num_head2 is not None
    if has_legacy_hidden and has_head_count:
        raise ValueError(
            "Use either --num_head1/--num_head2 or legacy -n1/-n2, not both."
        )
    if has_legacy_hidden:
        if args.n1 is None or args.n2 is None:
            raise ValueError("Legacy shape mode requires both -n1 and -n2.")
        if len(l_head_dim) != 1:
            raise ValueError("Legacy shape mode requires exactly one --head_dim value.")
        hd = l_head_dim[0]
        if hd <= 0:
            raise ValueError("--head_dim must be > 0")
        for n in args.n1:
            if n <= 0 or n % hd != 0:
                raise ValueError(
                    f"n1={n} must be positive and divisible by head_dim={hd}"
                )
        for n in args.n2:
            if n < 0 or (n > 0 and n % hd != 0):
                raise ValueError(f"n2={n} must be 0 or divisible by head_dim={hd}")
        l_num_head1 = [n // hd for n in args.n1]
        l_num_head2 = [n // hd for n in args.n2]
    else:
        if args.num_head1 is not None:
            l_num_head1 = args.num_head1
        if args.num_head2 is not None:
            l_num_head2 = args.num_head2

    if any(t <= 0 for t in l_token):
        raise ValueError("token must be > 0")
    if any(hd <= 0 for hd in l_head_dim):
        raise ValueError("head_dim must be > 0")
    if any(nh <= 0 for nh in l_num_head1):
        raise ValueError("num_head1 must be > 0")
    if any(nh < 0 for nh in l_num_head2):
        raise ValueError("num_head2 must be >= 0")
    if args.transpose_scale and "fp4x2" in l_quant_type:
        raise ValueError("fp4x2 path currently does not support --transpose_scale")
    if "fp4x2" in l_quant_type and (
        getattr(torch, "float4_e2m1fn_x2", None) is None or dtypes.fp4x2 == torch.uint8
    ):
        raise RuntimeError(
            "Requested --quant_out_dtype fp4x2 but torch.float4_e2m1fn_x2 is unavailable"
        )

    if args.residual is not None:
        l_residual = args.residual

    # When --no_quant, exercise both the quant and the no-quant code paths.
    l_no_quant = [False, True] if args.no_quant else [False]

    df = []
    for dtype in [dtypes.d_dtypes[k] for k in l_dtype]:
        for quant_out_dtype_name in l_quant_type:
            quant_out_dtype = (
                dtypes.fp8 if quant_out_dtype_name == "fp8" else dtypes.fp4x2
            )
            for head_dim in l_head_dim:
                for token in l_token:
                    for num_head1 in l_num_head1:
                        for num_head2 in l_num_head2:
                            for add_residual in l_residual:
                                for no_quant in l_no_quant:
                                    # Skip redundant fp4x2/no_quant pairs: no-quant ignores quant dtype.
                                    if no_quant and quant_out_dtype_name == "fp4x2":
                                        continue
                                    row = test_fused_qk_rmsnorm_group_quant(
                                        dtype=dtype,
                                        token=token,
                                        num_head1=num_head1,
                                        num_head2=num_head2,
                                        add_residual=bool(add_residual),
                                        head_dim=head_dim,
                                        group_size=args.group_size,
                                        output_unquantized_inp1=args.output_unquantized_inp1,
                                        transpose_scale=args.transpose_scale,
                                        quant_out_dtype=quant_out_dtype,
                                        gemma_norm=args.gemma_norm,
                                        no_quant=no_quant,
                                    )
                                    df.append(row)

    df = pd.DataFrame(df)
    focus_df = _focus_summary_df(df)
    aiter.logger.info(
        "fused_qk_rmsnorm_group_quant summary (time/err/bw, markdown):\n%s",
        focus_df.to_markdown(index=False),
    )
