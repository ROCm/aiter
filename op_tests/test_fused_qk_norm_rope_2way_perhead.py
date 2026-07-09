#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate fused_qk_norm_rope_2way_fp8_perhead_quant and v_2way_per_head_fp8_quant.

Compares HIP kernels against:
  - Q/K: fused_qk_norm_rope_2way + per_tensor_quant (baseline)
  - V:   per_tensor_quant on torch.cat([v0, v1])
"""

import argparse

import pandas as pd
import pytest
import torch
from torch import Tensor

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest

FP8_DTYPE = dtypes.fp8


def _torch_per_head_fp8_quant(x: Tensor) -> tuple[Tensor, Tensor]:
    """Reference per-(batch, head) fp8 quant (amax/240 descale)."""
    batch_size, num_tokens, num_heads, head_size = x.shape
    x32 = x.float()
    amax = x32.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-8)
    descale = amax / 240.0
    q = (x32 / descale).to(FP8_DTYPE)
    scale = descale.squeeze(-1).squeeze(1)
    return q, scale


def _dequant_per_head(fp8: Tensor, descale: Tensor) -> Tensor:
    b, t, h, d = fp8.shape
    return fp8.float() * descale.view(b, 1, h, 1).float()


@perftest()
def _run_qk_baseline(
    q0: Tensor,
    k0: Tensor,
    q1: Tensor,
    k1: Tensor,
    w_q0: Tensor,
    w_k0: Tensor,
    w_q1: Tensor,
    w_k1: Tensor,
    cos_sin0: Tensor,
    cos_sin1: Tensor,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
):
    q01 = torch.empty(
        (batch_size, num_tokens0 + num_tokens1, num_heads_q, head_size),
        dtype=q0.dtype,
        device=q0.device,
    )
    k01 = torch.empty(
        (batch_size, num_tokens0 + num_tokens1, num_heads_k, head_size),
        dtype=k0.dtype,
        device=k0.device,
    )
    aiter.fused_qk_norm_rope_2way(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        q01,
        k01,
    )
    q_fp8, q_scale = aiter.per_tensor_quant(q01, quant_dtype=FP8_DTYPE)
    k_fp8, k_scale = aiter.per_tensor_quant(k01, quant_dtype=FP8_DTYPE)
    return q_fp8, k_fp8, q_scale, k_scale, q01, k01


@perftest()
def _run_qk_perhead(
    q0: Tensor,
    k0: Tensor,
    q1: Tensor,
    k1: Tensor,
    w_q0: Tensor,
    w_k0: Tensor,
    w_q1: Tensor,
    w_k1: Tensor,
    cos_sin0: Tensor,
    cos_sin1: Tensor,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    out_q01: Tensor,
    out_k01: Tensor,
):
    return aiter.fused_qk_norm_rope_2way_fp8_perhead_quant(
        q0,
        k0,
        q1,
        k1,
        w_q0,
        w_k0,
        w_q1,
        w_k1,
        cos_sin0,
        cos_sin1,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        out_q01,
        out_k01,
    )


@perftest()
def _run_v_baseline(v0: Tensor, v1: Tensor):
    v = torch.cat([v0, v1], dim=1).contiguous()
    return aiter.per_tensor_quant(v, quant_dtype=FP8_DTYPE)


@perftest()
def _run_v_perhead(v0: Tensor, v1: Tensor):
    return aiter.v_2way_per_head_fp8_quant(v0, v1)


def _make_qk_inputs(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    seed: int,
):
    torch.manual_seed(seed)
    dev = "cuda"

    def rn(*shape):
        return torch.randn(*shape, dtype=dtype, device=dev)

    q0 = rn(batch_size, num_tokens0, num_heads_q, head_size)
    k0 = rn(batch_size, num_tokens0, num_heads_k, head_size)
    q1 = rn(batch_size, num_tokens1, num_heads_q, head_size)
    k1 = rn(batch_size, num_tokens1, num_heads_k, head_size)
    w_q0 = rn(head_size)
    w_k0 = rn(head_size)
    w_q1 = rn(head_size)
    w_k1 = rn(head_size)
    cos_sin0 = rn(num_tokens0, head_size)
    cos_sin1 = rn(num_tokens1, head_size)
    return q0, k0, q1, k1, w_q0, w_k0, w_q1, w_k1, cos_sin0, cos_sin1


def _validate_qk_perhead_case(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    collect_perf: bool = False,
) -> dict:
    q0, k0, q1, k1, w_q0, w_k0, w_q1, w_k1, cos_sin0, cos_sin1 = _make_qk_inputs(
        dtype,
        batch_size,
        num_tokens0,
        num_tokens1,
        num_heads_q,
        num_heads_k,
        head_size,
        seed=0,
    )
    out_q01 = torch.empty(
        (batch_size, num_tokens0 + num_tokens1, num_heads_q, head_size),
        dtype=dtype,
        device="cuda",
    )
    out_k01 = torch.empty(
        (batch_size, num_tokens0 + num_tokens1, num_heads_k, head_size),
        dtype=dtype,
        device="cuda",
    )

    if collect_perf:
        baseline_out, baseline_us = _run_qk_baseline(
            q0,
            k0,
            q1,
            k1,
            w_q0,
            w_k0,
            w_q1,
            w_k1,
            cos_sin0,
            cos_sin1,
            batch_size,
            num_tokens0,
            num_tokens1,
            num_heads_q,
            num_heads_k,
            head_size,
            is_interleaved,
            eps,
        )
        hip_out, hip_us = _run_qk_perhead(
            q0,
            k0,
            q1,
            k1,
            w_q0,
            w_k0,
            w_q1,
            w_k1,
            cos_sin0,
            cos_sin1,
            batch_size,
            num_tokens0,
            num_tokens1,
            num_heads_q,
            num_heads_k,
            head_size,
            is_interleaved,
            eps,
            out_q01,
            out_k01,
        )
    else:
        baseline_out = _run_qk_baseline(
            q0,
            k0,
            q1,
            k1,
            w_q0,
            w_k0,
            w_q1,
            w_k1,
            cos_sin0,
            cos_sin1,
            batch_size,
            num_tokens0,
            num_tokens1,
            num_heads_q,
            num_heads_k,
            head_size,
            is_interleaved,
            eps,
        )[0]
        baseline_us = None
        hip_out = _run_qk_perhead(
            q0,
            k0,
            q1,
            k1,
            w_q0,
            w_k0,
            w_q1,
            w_k1,
            cos_sin0,
            cos_sin1,
            batch_size,
            num_tokens0,
            num_tokens1,
            num_heads_q,
            num_heads_k,
            head_size,
            is_interleaved,
            eps,
            out_q01,
            out_k01,
        )[0]
        hip_us = None

    _, _, _, _, q01_ref, k01_ref = baseline_out
    q_fp8, k_fp8, q_descale, k_descale, q01, k01 = hip_out

    q_bf16_err = checkAllclose(
        q01_ref,
        q01,
        rtol=1e-2,
        atol=0.05,
        tol_err_ratio=0.0,
        msg=f"check q_bf16 baseline vs perhead, B={batch_size}, T0={num_tokens0}, "
        f"T1={num_tokens1}, Hq={num_heads_q}: ",
    )
    k_bf16_err = checkAllclose(
        k01_ref,
        k01,
        rtol=1e-2,
        atol=0.05,
        tol_err_ratio=0.0,
        msg=f"check k_bf16 baseline vs perhead, B={batch_size}, T0={num_tokens0}, "
        f"T1={num_tokens1}, Hk={num_heads_k}: ",
    )

    q_ref = q01_ref.float()
    k_ref = k01_ref.float()
    q_deq = _dequant_per_head(q_fp8, q_descale)
    k_deq = _dequant_per_head(k_fp8, k_descale)
    q_torch_fp8, q_torch_scale = _torch_per_head_fp8_quant(q01_ref)
    k_torch_fp8, k_torch_scale = _torch_per_head_fp8_quant(k01_ref)

    q_deq_err = checkAllclose(
        q_ref,
        q_deq,
        rtol=0.15,
        atol=1.0,
        tol_err_ratio=0.01,
        msg=f"check q_dequant vs bf16 ref, head_size={head_size}: ",
    )
    k_deq_err = checkAllclose(
        k_ref,
        k_deq,
        rtol=0.15,
        atol=1.0,
        tol_err_ratio=0.01,
        msg=f"check k_dequant vs bf16 ref, head_size={head_size}: ",
    )
    q_deq_torch = _dequant_per_head(q_torch_fp8, q_torch_scale)
    k_deq_torch = _dequant_per_head(k_torch_fp8, k_torch_scale)
    q_fp8_err = checkAllclose(
        q_deq_torch,
        q_deq,
        rtol=0.1,
        atol=1.0,
        tol_err_ratio=0.01,
        msg="check q_dequant hip vs torch per-head ref: ",
    )
    k_fp8_err = checkAllclose(
        k_deq_torch,
        k_deq,
        rtol=0.1,
        atol=1.0,
        tol_err_ratio=0.01,
        msg="check k_dequant hip vs torch per-head ref: ",
    )
    q_scale_err = checkAllclose(
        q_torch_scale,
        q_descale,
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0.0,
        msg="check q_descale vs torch per-head ref: ",
    )
    k_scale_err = checkAllclose(
        k_torch_scale,
        k_descale,
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0.0,
        msg="check k_descale vs torch per-head ref: ",
    )

    uplift = (baseline_us / hip_us - 1) if baseline_us and hip_us else None
    info = (
        f"dtype:{dtype}, batch_size:{batch_size}, num_tokens0:{num_tokens0}, "
        f"num_tokens1:{num_tokens1}, num_heads_q:{num_heads_q}, "
        f"num_heads_k:{num_heads_k}, head_size:{head_size}, "
        f"is_interleaved:{is_interleaved}"
    )
    if hip_us is not None:
        msg = (
            f"[perf][qk_perhead] === {info} === "
            f"baseline avg: {baseline_us:<8.2f} us, hip avg: {hip_us:<8.2f} us, "
            f"uplift: {uplift:<5.1%}"
        )
        print(msg, flush=True)

    return {
        "op": "qk_perhead",
        "dtype": str(dtype),
        "gfx": aiter.get_gfx(),
        "batch_size": batch_size,
        "num_tokens0": num_tokens0,
        "num_tokens1": num_tokens1,
        "num_heads_q": num_heads_q,
        "num_heads_k": num_heads_k,
        "head_size": head_size,
        "is_interleaved": is_interleaved,
        "baseline_us": baseline_us,
        "hip_us": hip_us,
        "uplift": f"{uplift:.1%}" if uplift is not None else "N/A",
        "q_bf16_err": q_bf16_err,
        "k_bf16_err": k_bf16_err,
        "q_deq_err": q_deq_err,
        "k_deq_err": k_deq_err,
        "q_fp8_err": q_fp8_err,
        "k_fp8_err": k_fp8_err,
        "q_scale_err": q_scale_err,
        "k_scale_err": k_scale_err,
    }


def _validate_v_perhead_case(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads: int,
    head_size: int,
    collect_perf: bool = False,
) -> dict:
    torch.manual_seed(1)
    v0 = torch.randn(
        batch_size, num_tokens0, num_heads, head_size, dtype=dtype, device="cuda"
    ).contiguous()
    v1 = torch.randn(
        batch_size, num_tokens1, num_heads, head_size, dtype=dtype, device="cuda"
    ).contiguous()
    v_ref = torch.cat([v0, v1], dim=1).contiguous()

    if collect_perf:
        (v_pt, v_pt_s), baseline_us = _run_v_baseline(v0, v1)
        (v_ph, v_ph_s), hip_us = _run_v_perhead(v0, v1)
    else:
        v_pt, v_pt_s = _run_v_baseline(v0, v1)[0]
        baseline_us = None
        v_ph, v_ph_s = _run_v_perhead(v0, v1)[0]
        hip_us = None

    v_torch_fp8, v_torch_scale = _torch_per_head_fp8_quant(v_ref)
    v_deq = _dequant_per_head(v_ph, v_ph_s)

    v_deq_torch = _dequant_per_head(v_torch_fp8, v_torch_scale)
    v_fp8_err = checkAllclose(
        v_deq_torch,
        v_deq,
        rtol=0.1,
        atol=1.0,
        tol_err_ratio=0.01,
        msg=f"check v_dequant hip vs torch per-head ref, B={batch_size}: ",
    )
    v_scale_err = checkAllclose(
        v_torch_scale,
        v_ph_s,
        rtol=0.15,
        atol=0.5,
        tol_err_ratio=0.25,
        msg="check v_descale vs torch per-head ref: ",
    )
    v_deq_err = checkAllclose(
        v_ref.float(),
        v_deq,
        rtol=0.15,
        atol=1.0,
        tol_err_ratio=0.01,
        msg="check v_dequant vs bf16 ref: ",
    )

    uplift = (baseline_us / hip_us - 1) if baseline_us and hip_us else None
    info = (
        f"dtype:{dtype}, batch_size:{batch_size}, num_tokens0:{num_tokens0}, "
        f"num_tokens1:{num_tokens1}, num_heads:{num_heads}, head_size:{head_size}"
    )
    if hip_us is not None:
        msg = (
            f"[perf][v_perhead] === {info} === "
            f"baseline avg: {baseline_us:<8.2f} us, hip avg: {hip_us:<8.2f} us, "
            f"uplift: {uplift:<5.1%}"
        )
        print(msg, flush=True)

    return {
        "op": "v_perhead",
        "dtype": str(dtype),
        "gfx": aiter.get_gfx(),
        "batch_size": batch_size,
        "num_tokens0": num_tokens0,
        "num_tokens1": num_tokens1,
        "num_heads": num_heads,
        "head_size": head_size,
        "baseline_us": baseline_us,
        "hip_us": hip_us,
        "uplift": f"{uplift:.1%}" if uplift is not None else "N/A",
        "v_fp8_err": v_fp8_err,
        "v_scale_err": v_scale_err,
        "v_deq_err": v_deq_err,
    }


@pytest.mark.parametrize(
    "dtype,batch_size,num_tokens0,num_tokens1,num_heads_q,num_heads_k,head_size,is_interleaved",
    [
        (torch.bfloat16, 1, 512, 4096, 24, 24, 128, False),
        (torch.bfloat16, 1, 64, 128, 8, 8, 128, False),
        (torch.float16, 2, 32, 64, 8, 8, 128, False),
    ],
)
def test_fused_qk_norm_rope_2way_fp8_perhead(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
) -> None:
    _validate_qk_perhead_case(
        dtype=dtype,
        batch_size=batch_size,
        num_tokens0=num_tokens0,
        num_tokens1=num_tokens1,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_size=head_size,
        is_interleaved=is_interleaved,
        eps=1e-6,
        collect_perf=False,
    )


@pytest.mark.parametrize(
    "dtype,batch_size,num_tokens0,num_tokens1,num_heads,head_size",
    [
        (torch.bfloat16, 1, 512, 4096, 24, 128),
        (torch.bfloat16, 1, 64, 128, 8, 128),
        (torch.float16, 2, 32, 64, 8, 128),
    ],
)
def test_v_2way_per_head_fp8_quant(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads: int,
    head_size: int,
) -> None:
    _validate_v_perhead_case(
        dtype=dtype,
        batch_size=batch_size,
        num_tokens0=num_tokens0,
        num_tokens1=num_tokens1,
        num_heads=num_heads,
        head_size=head_size,
        collect_perf=False,
    )


@benchmark()
def run_qk_perhead_case(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
):
    return _validate_qk_perhead_case(
        dtype=dtype,
        batch_size=batch_size,
        num_tokens0=num_tokens0,
        num_tokens1=num_tokens1,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_size=head_size,
        is_interleaved=is_interleaved,
        eps=1e-6,
        collect_perf=True,
    )


@benchmark()
def run_v_perhead_case(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens0: int,
    num_tokens1: int,
    num_heads: int,
    head_size: int,
):
    return _validate_v_perhead_case(
        dtype=dtype,
        batch_size=batch_size,
        num_tokens0=num_tokens0,
        num_tokens1=num_tokens1,
        num_heads=num_heads,
        head_size=head_size,
        collect_perf=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Validate per-(batch, head) FP8 quant for fused QK norm/rope (2way) and V.\n"
            "Use --bench to collect perf; default runs correctness sweeps."
        ),
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        choices=["fp16", "bf16"],
        nargs="*",
        default=["bf16"],
        help="Data type(s). e.g. -d bf16 fp16",
    )
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Collect perf via @benchmark (slower).",
    )
    args = parser.parse_args()

    qk_cases = [
        # Qwen-Image-Edit style
        (1, 512, 4096, 24, 24, 128, False),
        # smoke
        (1, 64, 128, 8, 8, 128, False),
        (2, 32, 64, 8, 8, 128, False),
    ]
    v_cases = [
        (1, 512, 4096, 24, 128),
        (1, 64, 128, 8, 128),
        (2, 32, 64, 8, 128),
    ]

    rows = []
    for key in args.dtype:
        dtype = dtypes.d_dtypes[key]
        for batch_size, t0, t1, hq, hk, hs, interleaved in qk_cases:
            if args.bench:
                row = run_qk_perhead_case(
                    dtype, batch_size, t0, t1, hq, hk, hs, interleaved
                )
            else:
                row = _validate_qk_perhead_case(
                    dtype=dtype,
                    batch_size=batch_size,
                    num_tokens0=t0,
                    num_tokens1=t1,
                    num_heads_q=hq,
                    num_heads_k=hk,
                    head_size=hs,
                    is_interleaved=interleaved,
                    eps=1e-6,
                    collect_perf=False,
                )
            rows.append(row)
        for batch_size, t0, t1, h, hs in v_cases:
            if args.bench:
                row = run_v_perhead_case(dtype, batch_size, t0, t1, h, hs)
            else:
                row = _validate_v_perhead_case(
                    dtype=dtype,
                    batch_size=batch_size,
                    num_tokens0=t0,
                    num_tokens1=t1,
                    num_heads=h,
                    head_size=hs,
                    collect_perf=False,
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "fused_qk_norm_rope_2way_perhead summary:\n%s",
        df.to_string(index=False),
    )
