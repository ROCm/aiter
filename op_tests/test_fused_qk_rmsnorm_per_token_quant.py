#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import argparse

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest

MI308_BW_MAX_TBPS = 5.3


def _aiter_rmsnorm_baseline(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> torch.Tensor:
    out = torch.empty_like(x)
    aiter.rmsnorm(out, x, weight, eps)
    return out


def _aiter_add_rmsnorm_baseline(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    out = torch.empty_like(x)
    residual_out = torch.empty_like(x)
    aiter.add_rmsnorm(out, x, residual, residual_out, weight, eps)
    return out, residual_out


def _aiter_rmsnorm_quant_baseline(
    x: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    quant = torch.empty_like(x, dtype=dtypes.fp8)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    aiter.rmsnorm_quant(quant, x, scale, weight, eps, 0, False)
    return quant, scale


def _aiter_add_rmsnorm_quant_baseline(
    x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    quant = torch.empty_like(x, dtype=dtypes.fp8)
    scale = torch.empty((x.shape[0], 1), dtype=torch.float32, device=x.device)
    residual_out = torch.empty_like(x)
    aiter.add_rmsnorm_quant(
        quant, x, residual, residual_out, scale, weight, eps, 0, False
    )
    return quant, scale, residual_out


def _make_split_view(
    m: int, n1: int, n2: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor | None]:
    tail = 128
    if n2 > 0:
        full = torch.randn((m, n1 + n2 + tail), dtype=dtype, device="cuda") / 10
        q, k, _ = torch.split(full, [n1, n2, tail], dim=1)
        return q, k
    full = torch.randn((m, n1 + tail), dtype=dtype, device="cuda") / 10
    q, _ = torch.split(full, [n1, tail], dim=1)
    return q, None


def _tensor_bytes(x: torch.Tensor | None) -> int:
    if x is None:
        return 0
    return x.numel() * x.element_size()


def _calc_io_bytes(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    q_residual: torch.Tensor | None,
    q_quant: torch.Tensor,
    q_scale: torch.Tensor,
    q_unquant: torch.Tensor | None,
    k_out: torch.Tensor | None,
    q_res_out: torch.Tensor | None,
) -> int:
    return (
        _tensor_bytes(q)
        + _tensor_bytes(q_weight)
        + _tensor_bytes(k)
        + _tensor_bytes(k_weight)
        + _tensor_bytes(q_residual)
        + _tensor_bytes(q_quant)
        + _tensor_bytes(q_scale)
        + _tensor_bytes(q_unquant)
        + _tensor_bytes(k_out)
        + _tensor_bytes(q_res_out)
    )


def _focus_summary_df(df: pd.DataFrame) -> pd.DataFrame:
    focus_cols = [
        "dtype",
        "gfx",
        "m",
        "n1",
        "n2",
        "residual",
        "gemma_norm",
        "baseline_us",
        "hip_us",
        "uplift",
        "baseline_bw_TBps",
        "hip_bw_TBps",
        "q_scale_error_rate",
        "q_deq_error_rate",
        "q_unquant_error_rate",
        "k_error_rate",
        "res_error_rate",
        "q_scale_max_abs",
        "q_deq_max_abs",
        "q_deq_mean_abs",
        "q_unquant_max_abs",
        "k_max_abs",
        "res_max_abs",
        "note",
    ]
    return df[[c for c in focus_cols if c in df.columns]]


def _run_hip_impl(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    q_residual: torch.Tensor | None,
    gemma_norm: bool,
):
    m, n1 = q.shape
    q_quant = torch.empty((m, n1), dtype=dtypes.fp8, device=q.device)
    q_scale = torch.empty((m, 1), dtype=torch.float32, device=q.device)
    q_unquant = torch.empty_like(q)
    k_out = torch.empty_like(k) if k is not None else None
    q_res_out = torch.empty_like(q) if q_residual is not None else None

    aiter.fused_qk_rmsnorm_per_token_quant(
        q_quant,
        q_scale,
        q,
        q_weight,
        1e-6,
        q_unquant,
        k_out,
        q_res_out,
        k,
        k_weight,
        1e-6 if k is not None else None,
        q_residual,
        gemma_norm,
    )
    return (q_quant, q_scale), q_unquant, k_out, q_res_out


def _run_baseline_impl(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    q_residual: torch.Tensor | None,
    gemma_norm: bool,
):
    if gemma_norm:
        assert (
            q_residual is None
        ), "Gemma residual baseline is not covered in this smoke test"
        q_weight = (q_weight + 1).contiguous()
        if k_weight is not None:
            k_weight = (k_weight + 1).contiguous()

    if q_residual is not None:
        q_quant, q_scale, q_res_out = _aiter_add_rmsnorm_quant_baseline(
            q, q_residual, q_weight, 1e-6
        )
        q_unquant, _ = _aiter_add_rmsnorm_baseline(q, q_residual, q_weight, 1e-6)
    else:
        q_quant, q_scale = _aiter_rmsnorm_quant_baseline(q, q_weight, 1e-6)
        q_unquant = _aiter_rmsnorm_baseline(q, q_weight, 1e-6)
        q_res_out = None

    k_out = None
    if k is not None:
        assert k_weight is not None
        k_out = _aiter_rmsnorm_baseline(k, k_weight, 1e-6)

    return (q_quant, q_scale), q_unquant, k_out, q_res_out


@perftest()
def run_baseline(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    q_residual: torch.Tensor | None,
    gemma_norm: bool,
):
    return _run_baseline_impl(q, q_weight, k, k_weight, q_residual, gemma_norm)


@perftest()
def run_hip(
    q: torch.Tensor,
    q_weight: torch.Tensor,
    k: torch.Tensor | None,
    k_weight: torch.Tensor | None,
    q_residual: torch.Tensor | None,
    gemma_norm: bool,
):
    return _run_hip_impl(q, q_weight, k, k_weight, q_residual, gemma_norm)


def _validate_fused_qk_rmsnorm_per_token_quant_case(
    dtype: torch.dtype,
    m: int,
    n1: int,
    n2: int,
    add_residual: bool,
    gemma_norm: bool,
    collect_perf: bool = False,
) -> dict:
    q, k = _make_split_view(m, n1, n2, dtype)
    q_weight = torch.randn((n1,), dtype=dtype, device="cuda").contiguous()
    k_weight = (
        torch.randn((n2,), dtype=dtype, device="cuda").contiguous()
        if k is not None
        else None
    )
    q_residual = None
    if add_residual:
        full_res = torch.randn((m, n1 + 128), dtype=dtype, device="cuda") / 10
        q_residual, _ = torch.split(full_res, [n1, 128], dim=1)

    if collect_perf:
        baseline_out, baseline_us = run_baseline(
            q, q_weight, k, k_weight, q_residual, gemma_norm
        )
        hip_out, hip_us = run_hip(q, q_weight, k, k_weight, q_residual, gemma_norm)
    else:
        baseline_out = _run_baseline_impl(
            q, q_weight, k, k_weight, q_residual, gemma_norm
        )
        baseline_us = None
        hip_out = _run_hip_impl(q, q_weight, k, k_weight, q_residual, gemma_norm)
        hip_us = None

    (q_quant_ref, q_scale_ref), q_unquant_ref, k_out_ref, q_res_out_ref = baseline_out
    (q_quant, q_scale), q_unquant, k_out, q_res_out = hip_out

    q_dequant_ref = q_quant_ref.float() * q_scale_ref.float()
    q_dequant = q_quant.float() * q_scale.float()

    if gemma_norm:
        q_scale_error_rate = checkAllclose(
            q_scale_ref,
            q_scale,
            rtol=5e-4,
            atol=2e-5,
            tol_err_ratio=0.0,
            msg=f"check q_scale baseline vs hip, m={m}, n1={n1}, n2={n2}: ",
        )

        q_deq_error_rate = checkAllclose(
            q_dequant_ref,
            q_dequant,
            rtol=0.1,
            atol=1.0,
            tol_err_ratio=0.0,
            msg=f"check q_dequant baseline vs hip, m={m}, n1={n1}, n2={n2}: ",
        )

        q_unquant_error_rate = checkAllclose(
            q_unquant_ref,
            q_unquant,
            rtol=1e-2,
            atol=1e-2,
            tol_err_ratio=0.0,
            msg=f"check q_unquant baseline vs hip, m={m}, n1={n1}, n2={n2}: ",
        )
    else:
        q_scale_error_rate = checkAllclose(
            q_scale_ref,
            q_scale,
            rtol=0.0,
            atol=0.0,
            tol_err_ratio=0.0,
            msg=f"check q_scale baseline vs hip, m={m}, n1={n1}, n2={n2}: ",
        )

        q_deq_error_rate = checkAllclose(
            q_dequant_ref,
            q_dequant,
            rtol=0.0,
            atol=0.0,
            tol_err_ratio=0.0,
            msg=f"check q_dequant baseline vs hip, m={m}, n1={n1}, n2={n2}: ",
        )

        q_unquant_error_rate = checkAllclose(
            q_unquant_ref,
            q_unquant,
            rtol=0.0,
            atol=0.0,
            tol_err_ratio=0.0,
            msg=f"check q_unquant baseline vs hip, m={m}, n1={n1}, n2={n2}: ",
        )

    if k_out_ref is not None:
        assert k_out is not None
        checkAllclose(
            k_out_ref,
            k_out,
            rtol=(1e-2 if gemma_norm else 0.0),
            atol=(1e-2 if gemma_norm else 0.0),
            tol_err_ratio=0.0,
            msg=f"check k_out baseline vs hip, m={m}, n1={n1}, n2={n2}: ",
        )

    if q_res_out_ref is not None:
        assert q_res_out is not None
        checkAllclose(
            q_res_out_ref,
            q_res_out,
            rtol=0.0,
            atol=0.0,
            tol_err_ratio=0.0,
            msg=f"check q_res_out baseline vs hip, m={m}, n1={n1}, n2={n2}: ",
        )

    io_bytes = _calc_io_bytes(
        q=q,
        q_weight=q_weight,
        k=k,
        k_weight=k_weight,
        q_residual=q_residual,
        q_quant=q_quant,
        q_scale=q_scale,
        q_unquant=q_unquant,
        k_out=k_out,
        q_res_out=q_res_out,
    )
    baseline_bw_tbps = (
        io_bytes / (baseline_us * 1e-6) / 1e12 if baseline_us is not None else None
    )
    hip_bw_tbps = io_bytes / (hip_us * 1e-6) / 1e12 if hip_us is not None else None
    uplift = (baseline_us / hip_us - 1) if hip_us is not None else None

    if hip_us is not None:
        aiter.logger.info(
            "[result] dtype=%s m=%d n1=%d n2=%d residual=%s gemma_norm=%s | "
            "time(us): baseline=%.2f hip=%.2f uplift=%.1f%% | "
            "bw(TB/s): baseline=%.3f hip=%.3f hip/mi308_peak=%.1f%% | "
            "err: q_scale=%.6f q_deq=%.6f q_unquant=%.6f",
            dtype,
            m,
            n1,
            n2,
            add_residual,
            gemma_norm,
            baseline_us,
            hip_us,
            uplift * 100.0,
            baseline_bw_tbps,
            hip_bw_tbps,
            (hip_bw_tbps / MI308_BW_MAX_TBPS) * 100.0,
            q_scale_error_rate,
            q_deq_error_rate,
            q_unquant_error_rate,
        )

    return {
        "dtype": str(dtype),
        "gfx": aiter.get_gfx(),
        "m": m,
        "n1": n1,
        "n2": n2,
        "residual": add_residual,
        "gemma_norm": gemma_norm,
        "baseline_us": baseline_us,
        "hip_us": hip_us,
        "uplift": f"{uplift:.1%}" if uplift is not None else "N/A",
        "baseline_bw_TBps": baseline_bw_tbps,
        "hip_bw_TBps": hip_bw_tbps,
    }


@pytest.mark.parametrize(
    "dtype,m,n1,n2,add_residual,gemma_norm",
    [
        (torch.bfloat16, 8, 1536, 512, True, False),
        (torch.float16, 4, 6144, 0, False, True),
    ],
)
def test_fused_qk_rmsnorm_per_token_quant(
    dtype: torch.dtype,
    m: int,
    n1: int,
    n2: int,
    add_residual: bool,
    gemma_norm: bool,
) -> None:
    _validate_fused_qk_rmsnorm_per_token_quant_case(
        dtype=dtype,
        m=m,
        n1=n1,
        n2=n2,
        add_residual=add_residual,
        gemma_norm=gemma_norm,
        collect_perf=False,
    )


@benchmark()
def run_fused_qk_rmsnorm_per_token_quant_case(
    dtype: torch.dtype,
    m: int,
    n1: int,
    n2: int,
    add_residual: bool,
    gemma_norm: bool,
):
    return _validate_fused_qk_rmsnorm_per_token_quant_case(
        dtype=dtype,
        m=m,
        n1=n1,
        n2=n2,
        add_residual=add_residual,
        gemma_norm=gemma_norm,
        collect_perf=True,
    )


if __name__ == "__main__":
    l_dtype = ["bf16"]
    l_m = [8]
    l_n1 = [1536]
    l_n2 = [512]
    l_residual = [1]

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Validate HIP fused_qk_rmsnorm_per_token_quant.\n"
            "Compare the fused kernel against sequential AITER baselines.\n"
            "Use --gemma_norm to switch defaults to the Gemma smoke case."
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
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=None,
        help="Row count(s).",
    )
    parser.add_argument(
        "-n1",
        "--n1",
        type=int,
        nargs="*",
        default=None,
        help="q hidden size(s).",
    )
    parser.add_argument(
        "-n2",
        "--n2",
        type=int,
        nargs="*",
        default=None,
        help="k hidden size(s), 0 means no second input.",
    )
    parser.add_argument(
        "--residual",
        type=int,
        nargs="*",
        default=None,
        choices=[0, 1],
        help="Whether to include residual input, 0 or 1",
    )
    parser.add_argument(
        "--gemma_norm",
        action="store_true",
        help="Test gemma-style RMSNorm: x * rsqrt(mean(x^2)+eps) * (1+w).",
    )
    args = parser.parse_args()

    if args.gemma_norm:
        l_dtype = ["fp16"]
        l_m = [4]
        l_n1 = [6144]
        l_n2 = [0]
        l_residual = [0]

    if args.dtype is not None:
        l_dtype = args.dtype
    if args.m is not None:
        l_m = args.m
    if args.n1 is not None:
        l_n1 = args.n1
    if args.n2 is not None:
        l_n2 = args.n2
    if args.residual is not None:
        l_residual = args.residual

    if any(m <= 0 for m in l_m):
        raise ValueError("m must be > 0")
    if any(n1 <= 0 for n1 in l_n1):
        raise ValueError("n1 must be > 0")
    if any(n2 < 0 for n2 in l_n2):
        raise ValueError("n2 must be >= 0")
    if args.gemma_norm and any(r != 0 for r in l_residual):
        raise ValueError(
            "Gemma per-token CLI currently requires --residual 0 in this test."
        )

    df = []
    for dtype in [dtypes.d_dtypes[k] for k in l_dtype]:
        for m in l_m:
            for n1 in l_n1:
                for n2 in l_n2:
                    for add_residual in l_residual:
                        row = run_fused_qk_rmsnorm_per_token_quant_case(
                            dtype=dtype,
                            m=m,
                            n1=n1,
                            n2=n2,
                            add_residual=bool(add_residual),
                            gemma_norm=args.gemma_norm,
                        )
                        df.append(row)

    df = pd.DataFrame(df)
    focus_df = _focus_summary_df(df)
    aiter.logger.info(
        "fused_qk_rmsnorm_per_token_quant summary (time/err/bw, markdown):\n%s",
        focus_df.to_markdown(index=False),
    )
