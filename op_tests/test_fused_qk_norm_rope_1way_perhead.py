#!/usr/bin/env python
# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Validate fused_qk_norm_rope_1way_fp8_perhead_quant and v_1way_per_head_fp8_quant.

Compares HIP kernels against:
  - Q/K: fused_qk_norm_rope_1way + torch per-head FP8 reference
  - V:   torch per-head FP8 reference
"""

import argparse
from typing import Optional

import pandas as pd
import pytest
import torch
from torch import Tensor

import aiter
from aiter import dtypes
from aiter.test_common import benchmark, checkAllclose, perftest

FP8_DTYPE = dtypes.fp8
FP8_MAX_GFX942 = 240.0

pytestmark = pytest.mark.skipif(
    aiter.get_gfx() != "gfx942",
    reason="Z-Image 1way per-head FP8 tests are validated only on MI308/gfx942 (fp8_max=240)",
)


def _torch_per_head_fp8_quant(x: Tensor) -> tuple[Tensor, Tensor]:
    """Reference per-(batch, head) fp8 quant for gfx942 fp8_e4m3fnuz."""
    batch_size, _, _, _ = x.shape
    x32 = x.float()
    amax = x32.abs().amax(dim=(1, 3), keepdim=True).clamp(min=1e-8)
    descale = amax / FP8_MAX_GFX942
    q = (x32 / descale).to(FP8_DTYPE)
    scale = descale.view(batch_size, 1, -1, 1).squeeze(-1).squeeze(1)
    return q, scale


def _dequant_per_head(fp8: Tensor, descale: Tensor) -> Tensor:
    b, _, h, _ = fp8.shape
    return fp8.float() * descale.view(b, 1, h, 1).float()


@perftest()
def _run_qk_baseline(
    q: Tensor,
    k: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    cos_sin: Tensor,
    batch_size: int,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
):
    q_ref = torch.empty(
        (batch_size, num_tokens, num_heads_q, head_size),
        dtype=q.dtype,
        device=q.device,
    )
    k_ref = torch.empty(
        (batch_size, num_tokens, num_heads_k, head_size),
        dtype=k.dtype,
        device=k.device,
    )
    aiter.fused_qk_norm_rope_1way(
        q,
        k,
        w_q,
        w_k,
        cos_sin,
        batch_size,
        num_tokens,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        q_ref,
        k_ref,
    )
    return q_ref, k_ref


@perftest()
def _run_qk_perhead(
    q: Tensor,
    k: Tensor,
    w_q: Tensor,
    w_k: Tensor,
    cos_sin: Tensor,
    batch_size: int,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    out_q: Optional[Tensor] = None,
    out_k: Optional[Tensor] = None,
):
    return aiter.fused_qk_norm_rope_1way_fp8_perhead_quant(
        q,
        k,
        w_q,
        w_k,
        cos_sin,
        batch_size,
        num_tokens,
        num_heads_q,
        num_heads_k,
        head_size,
        is_interleaved,
        eps,
        out_q,
        out_k,
    )


@perftest()
def _run_v_baseline(v: Tensor):
    return _torch_per_head_fp8_quant(v)


@perftest()
def _run_v_perhead(v: Tensor):
    return aiter.v_1way_per_head_fp8_quant(v)


def _make_qk_inputs(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    seed: int,
):
    torch.manual_seed(seed)
    dev = "cuda"

    def rn(*shape, out_dtype: torch.dtype = dtype):
        return torch.randn(*shape, dtype=out_dtype, device=dev)

    q = rn(batch_size, num_tokens, num_heads_q, head_size)
    k = rn(batch_size, num_tokens, num_heads_k, head_size)
    w_q = rn(head_size)
    w_k = rn(head_size)
    cos_sin = rn(num_tokens, head_size, out_dtype=torch.float32)
    return q, k, w_q, w_k, cos_sin


def _validate_qk_perhead_case(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    eps: float,
    provide_outputs: bool,
    collect_perf: bool = False,
) -> dict:
    q, k, w_q, w_k, cos_sin = _make_qk_inputs(
        dtype,
        batch_size,
        num_tokens,
        num_heads_q,
        num_heads_k,
        head_size,
        seed=0,
    )

    out_q = None
    out_k = None
    if provide_outputs:
        out_q = torch.empty(
            (batch_size, num_tokens, num_heads_q, head_size),
            dtype=dtype,
            device="cuda",
        )
        out_k = torch.empty(
            (batch_size, num_tokens, num_heads_k, head_size),
            dtype=dtype,
            device="cuda",
        )

    if collect_perf:
        baseline_out, baseline_us = _run_qk_baseline(
            q,
            k,
            w_q,
            w_k,
            cos_sin,
            batch_size,
            num_tokens,
            num_heads_q,
            num_heads_k,
            head_size,
            is_interleaved,
            eps,
        )
        hip_out, hip_us = _run_qk_perhead(
            q,
            k,
            w_q,
            w_k,
            cos_sin,
            batch_size,
            num_tokens,
            num_heads_q,
            num_heads_k,
            head_size,
            is_interleaved,
            eps,
            out_q,
            out_k,
        )
    else:
        baseline_out = _run_qk_baseline(
            q,
            k,
            w_q,
            w_k,
            cos_sin,
            batch_size,
            num_tokens,
            num_heads_q,
            num_heads_k,
            head_size,
            is_interleaved,
            eps,
        )[0]
        baseline_us = None
        hip_out = _run_qk_perhead(
            q,
            k,
            w_q,
            w_k,
            cos_sin,
            batch_size,
            num_tokens,
            num_heads_q,
            num_heads_k,
            head_size,
            is_interleaved,
            eps,
            out_q,
            out_k,
        )[0]
        hip_us = None

    q_ref, k_ref = baseline_out
    q_fp8, k_fp8, q_descale, k_descale, q_out, k_out = hip_out

    if provide_outputs:
        q_bf16_err = checkAllclose(
            q_ref,
            q_out,
            rtol=1e-2,
            atol=0.05,
            tol_err_ratio=0.0,
            msg=f"check q_bf16 baseline vs perhead, B={batch_size}, T={num_tokens}, "
            f"Hq={num_heads_q}: ",
        )
        k_bf16_err = checkAllclose(
            k_ref,
            k_out,
            rtol=1e-2,
            atol=0.05,
            tol_err_ratio=0.0,
            msg=f"check k_bf16 baseline vs perhead, B={batch_size}, T={num_tokens}, "
            f"Hk={num_heads_k}: ",
        )
    else:
        assert q_out.numel() == 0
        assert k_out.numel() == 0
        assert q_out.dtype == dtype
        assert k_out.dtype == dtype
        q_bf16_err = 0.0
        k_bf16_err = 0.0

    q_deq = _dequant_per_head(q_fp8, q_descale)
    k_deq = _dequant_per_head(k_fp8, k_descale)
    q_torch_fp8, q_torch_scale = _torch_per_head_fp8_quant(q_ref)
    k_torch_fp8, k_torch_scale = _torch_per_head_fp8_quant(k_ref)

    q_deq_err = checkAllclose(
        q_ref.float(),
        q_deq,
        rtol=0.15,
        atol=1.0,
        tol_err_ratio=0.01,
        msg=f"check q_dequant vs bf16 ref, head_size={head_size}: ",
    )
    k_deq_err = checkAllclose(
        k_ref.float(),
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
        f"dtype:{dtype}, batch_size:{batch_size}, num_tokens:{num_tokens}, "
        f"num_heads_q:{num_heads_q}, num_heads_k:{num_heads_k}, "
        f"head_size:{head_size}, is_interleaved:{is_interleaved}, "
        f"provide_outputs:{provide_outputs}"
    )
    if hip_us is not None:
        msg = (
            f"[perf][qk_1way_perhead] === {info} === "
            f"baseline avg: {baseline_us:<8.2f} us, hip avg: {hip_us:<8.2f} us, "
            f"uplift: {uplift:<5.1%}"
        )
        print(msg, flush=True)

    return {
        "op": "qk_1way_perhead",
        "dtype": str(dtype),
        "gfx": aiter.get_gfx(),
        "batch_size": batch_size,
        "num_tokens": num_tokens,
        "num_heads_q": num_heads_q,
        "num_heads_k": num_heads_k,
        "head_size": head_size,
        "is_interleaved": is_interleaved,
        "provide_outputs": provide_outputs,
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
    num_tokens: int,
    num_heads: int,
    head_size: int,
    collect_perf: bool = False,
) -> dict:
    torch.manual_seed(1)
    v = torch.randn(
        batch_size, num_tokens, num_heads, head_size, dtype=dtype, device="cuda"
    ).contiguous()

    if collect_perf:
        (v_torch_fp8, v_torch_scale), baseline_us = _run_v_baseline(v)
        (v_ph, v_ph_s), hip_us = _run_v_perhead(v)
    else:
        v_torch_fp8, v_torch_scale = _run_v_baseline(v)[0]
        baseline_us = None
        v_ph, v_ph_s = _run_v_perhead(v)[0]
        hip_us = None

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
        v.float(),
        v_deq,
        rtol=0.15,
        atol=1.0,
        tol_err_ratio=0.01,
        msg="check v_dequant vs bf16 ref: ",
    )

    uplift = (baseline_us / hip_us - 1) if baseline_us and hip_us else None
    info = (
        f"dtype:{dtype}, batch_size:{batch_size}, num_tokens:{num_tokens}, "
        f"num_heads:{num_heads}, head_size:{head_size}"
    )
    if hip_us is not None:
        msg = (
            f"[perf][v_1way_perhead] === {info} === "
            f"baseline avg: {baseline_us:<8.2f} us, hip avg: {hip_us:<8.2f} us, "
            f"uplift: {uplift:<5.1%}"
        )
        print(msg, flush=True)

    return {
        "op": "v_1way_perhead",
        "dtype": str(dtype),
        "gfx": aiter.get_gfx(),
        "batch_size": batch_size,
        "num_tokens": num_tokens,
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
    "dtype,batch_size,num_tokens,num_heads_q,num_heads_k,head_size,is_interleaved,provide_outputs",
    [
        (torch.bfloat16, 1, 4096, 24, 24, 128, False, True),
        (torch.bfloat16, 1, 512, 24, 8, 128, False, True),
        (torch.bfloat16, 1, 64, 8, 8, 128, False, False),
        (torch.float16, 2, 64, 8, 8, 128, False, True),
        (torch.float16, 2, 512, 16, 8, 128, False, False),
    ],
)
def test_fused_qk_norm_rope_1way_fp8_perhead(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    provide_outputs: bool,
) -> None:
    _validate_qk_perhead_case(
        dtype=dtype,
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_size=head_size,
        is_interleaved=is_interleaved,
        eps=1e-6,
        provide_outputs=provide_outputs,
        collect_perf=False,
    )


@pytest.mark.parametrize(
    "dtype,batch_size,num_tokens,num_heads,head_size",
    [
        (torch.bfloat16, 1, 4096, 24, 128),
        (torch.bfloat16, 1, 512, 24, 128),
        (torch.bfloat16, 1, 64, 8, 128),
        (torch.float16, 2, 64, 8, 128),
    ],
)
def test_v_1way_per_head_fp8_quant(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
) -> None:
    _validate_v_perhead_case(
        dtype=dtype,
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_heads=num_heads,
        head_size=head_size,
        collect_perf=False,
    )


@benchmark()
def run_qk_perhead_case(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_heads_q: int,
    num_heads_k: int,
    head_size: int,
    is_interleaved: bool,
    provide_outputs: bool,
):
    return _validate_qk_perhead_case(
        dtype=dtype,
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_heads_q=num_heads_q,
        num_heads_k=num_heads_k,
        head_size=head_size,
        is_interleaved=is_interleaved,
        eps=1e-6,
        provide_outputs=provide_outputs,
        collect_perf=True,
    )


@benchmark()
def run_v_perhead_case(
    dtype: torch.dtype,
    batch_size: int,
    num_tokens: int,
    num_heads: int,
    head_size: int,
):
    return _validate_v_perhead_case(
        dtype=dtype,
        batch_size=batch_size,
        num_tokens=num_tokens,
        num_heads=num_heads,
        head_size=head_size,
        collect_perf=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description=(
            "Validate per-(batch, head) FP8 quant for fused QK norm/rope (1way) and V.\n"
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

    if aiter.get_gfx() != "gfx942":
        print(
            "Skipping: Z-Image 1way per-head FP8 tests require MI308/gfx942 (fp8_max=240)",
            flush=True,
        )
        raise SystemExit(0)

    qk_cases = [
        # Z-Image style
        (1, 4096, 24, 24, 128, False, True),
        # GQA/MQA shape
        (1, 512, 24, 8, 128, False, True),
        # smoke plus no out_q/out_k coverage
        (1, 64, 8, 8, 128, False, False),
        (2, 64, 8, 8, 128, False, True),
    ]
    v_cases = [
        (1, 4096, 24, 128),
        (1, 512, 24, 128),
        (1, 64, 8, 128),
        (2, 64, 8, 128),
    ]

    rows = []
    for key in args.dtype:
        dtype = dtypes.d_dtypes[key]
        for batch_size, t, hq, hk, hs, interleaved, provide_outputs in qk_cases:
            if args.bench:
                row = run_qk_perhead_case(
                    dtype, batch_size, t, hq, hk, hs, interleaved, provide_outputs
                )
            else:
                row = _validate_qk_perhead_case(
                    dtype=dtype,
                    batch_size=batch_size,
                    num_tokens=t,
                    num_heads_q=hq,
                    num_heads_k=hk,
                    head_size=hs,
                    is_interleaved=interleaved,
                    eps=1e-6,
                    provide_outputs=provide_outputs,
                    collect_perf=False,
                )
            rows.append(row)
        for batch_size, t, h, hs in v_cases:
            if args.bench:
                row = run_v_perhead_case(dtype, batch_size, t, h, hs)
            else:
                row = _validate_v_perhead_case(
                    dtype=dtype,
                    batch_size=batch_size,
                    num_tokens=t,
                    num_heads=h,
                    head_size=hs,
                    collect_perf=False,
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    aiter.logger.info(
        "fused_qk_norm_rope_1way_perhead summary:\n%s",
        df.to_string(index=False),
    )
