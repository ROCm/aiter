# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Correctness + perf test for ``fused_clamp_act_mul`` (SwiGLU clamp + act*up +
optional weights + optional FP8 group quant).

Drives the public ``fused_clamp_act_mul`` wrapper, which dispatches to the
gfx1250 Gluon port on gfx1250 and the portable Triton kernel elsewhere — so this
single test exercises whichever backend the running card uses. torch is the
untimed reference.
"""

import argparse
import itertools

import aiter
import pandas as pd
import torch
import torch.nn.functional as F
from aiter import dtypes
from aiter.ops.triton.fusions.fused_clamp_act_mul import (
    fused_clamp_act_mul,
    _is_gluon_available,
)
from aiter.test_common import benchmark, checkAllclose, run_perftest
from aiter.jit.utils.chip_info import get_gfx

torch.set_default_device("cuda")

# Triton path is arch-portable; gfx1250 additionally has the Gluon port.
SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]

# quant mode -> (dtype_quant, quant_block_size, scale_dtype_fmt)
_QUANT_MODES = {
    "none": (None, 128, "fp32"),
    "fp8": (dtypes.fp8, 128, "fp32"),
    "ue8m0": (dtypes.fp8, 32, "ue8m0"),
}


def _apply_act(gate, activation):
    if activation == "silu":
        return F.silu(gate)
    if activation == "gelu":
        return F.gelu(gate)
    if activation == "gelu_tanh":
        return F.gelu(gate, approximate="tanh")
    return gate


def run_torch(
    inp,
    swiglu_limit,
    activation,
    weights,
    dtype_quant,
    quant_block_size,
    scale_dtype_fmt,
    out_dtype,
):
    """Reference only (fp32 math): returns (out, scale). Not timed / not tabled."""
    M, D = inp.shape
    n = D // 2
    gate = inp[:, :n].to(dtypes.fp32)
    up = inp[:, n:].to(dtypes.fp32)

    if swiglu_limit > 0:
        up = up.clamp(-swiglu_limit, swiglu_limit)
        gate = gate.clamp(max=swiglu_limit)

    out = _apply_act(gate, activation) * up
    if weights is not None:
        out = out * weights.to(dtypes.fp32)

    if dtype_quant is None:
        return out.to(out_dtype), None

    # Quant path: the value reference is the fp32 pre-quant ``out`` (the kernel's
    # dequantized output is compared against it, absorbing fp8 rounding). We also
    # return the scale in the exact format the kernel emits (fp32 dequant scale
    # for the standard path, uint8 biased-exponent byte for ue8m0).
    DTYPE_MAX = torch.finfo(dtype_quant).max
    NQB = n // quant_block_size
    x = out.reshape(M, NQB, quant_block_size)
    amax = x.abs().amax(dim=-1, keepdim=True)

    if scale_dtype_fmt == "ue8m0":
        dequant = (amax / DTYPE_MAX).to(dtypes.fp32)
        # ROUND_UP to a power of two via the exponent field (matches the kernel).
        exp_bits = (dequant.view(torch.int32) + 0x007FFFFF) & 0x7F800000
        scale_store = (exp_bits >> 23).to(torch.uint8).reshape(M, NQB)
    else:
        amax = amax.clamp_min(1e-10)
        scale_store = (amax / DTYPE_MAX).to(dtypes.fp32).reshape(M, NQB)  # dequant scale
    return out, scale_store


@benchmark()
def test_fused_clamp_act_mul(
    m, n, dtype, quant, activation, swiglu_limit, weighted, backend
):
    dtype_quant, quant_block_size, scale_dtype_fmt = _QUANT_MODES[quant]
    HAS_QUANT = dtype_quant is not None
    out_dtype = dtype_quant if HAS_QUANT else dtype

    inp = torch.randn((m, 2 * n), dtype=dtype) * 3.0
    weights = torch.randn((m, 1), dtype=dtype) if weighted else None

    ref_out, ref_scale = run_torch(
        inp,
        swiglu_limit,
        activation,
        weights,
        dtype_quant,
        quant_block_size,
        scale_dtype_fmt,
        out_dtype,
    )

    def _run():
        return fused_clamp_act_mul(
            inp,
            swiglu_limit=swiglu_limit,
            activation=activation,
            weights=weights,
            dtype_quant=dtype_quant,
            quant_block_size=quant_block_size,
            scale_dtype_fmt=scale_dtype_fmt,
            backend=backend,
        )

    # Elementwise + group-quant: read gate+up, write out (+ small scale buffer).
    # Memory-bound, so TB/s is the meaningful roofline metric.
    flops = 10 * m * n  # ~clamp+act+mul+weights+quant per element (approx)
    nbytes = 2 * m * n * inp.element_size() + m * n * (
        torch.tensor([], dtype=out_dtype).element_size()
    )

    ret = {"gfx": get_gfx()}
    out, us = run_perftest(_run)
    if HAS_QUANT:
        out_q, scale = out
        NQB = n // quant_block_size
        # Dequantize the KERNEL output with the KERNEL's emitted scale and compare
        # to the fp32 reference (fp8 rounding => allow a fraction of element
        # mismatches via tol_err_ratio).
        if scale_dtype_fmt == "ue8m0":
            mult = (2.0 ** (scale.to(dtypes.fp32) - 127.0)).reshape(m, NQB, 1)
        else:
            mult = scale.to(dtypes.fp32).reshape(m, NQB, 1)
        deq = out_q.to(dtypes.fp32).reshape(m, NQB, quant_block_size) * mult
        err = checkAllclose(
            ref_out.reshape(m, NQB, quant_block_size).to(dtypes.fp32),
            deq,
            rtol=0.2,
            atol=0.2,
            tol_err_ratio=0.1,
            msg=f"{quant}: dequant value",
        )
        # scales must match the reference formula (ue8m0 exponent byte is exact;
        # fp32 scales match to reduction-order rounding).
        s_atol = 0.0 if scale_dtype_fmt == "ue8m0" else 1e-6
        checkAllclose(
            ref_scale.to(dtypes.fp32),
            scale.to(dtypes.fp32),
            rtol=1e-3,
            atol=s_atol,
            msg=f"{quant}: scales",
        )
    else:
        err = checkAllclose(
            ref_out.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{quant}: value",
        )

    ret[f"aiter us"] = us
    ret[f"aiter TFLOPS"] = flops / us / 1e6
    ret[f"aiter TB/s"] = nbytes / us / 1e6
    ret[f"aiter err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            f"fused_clamp_act_mul unsupported on {get_gfx()}; skipping"
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d", "--dtype", type=dtypes.str2Dtype, nargs="*", default=[dtypes.bf16],
        help="input/output dtype for the non-quant path (e.g. bf16)",
    )
    parser.add_argument(
        "-s", "--mn", type=dtypes.str2tuple, nargs="*",
        default=[(1024, 2048), (8192, 2048), (16384, 4096)],
        help="(M, N=n_half) pairs; N must be a multiple of 128",
    )
    parser.add_argument(
        "--quant", type=str, nargs="*", default=["none", "fp8", "ue8m0"],
        help="quant modes to sweep: none | fp8 | ue8m0",
    )
    parser.add_argument(
        "--activation", type=str, nargs="*", default=["silu"],
        help="activation applied to gate: silu | gelu | gelu_tanh",
    )
    parser.add_argument(
        "--swiglu-limit", type=float, nargs="*", default=[7.0],
        help="SwiGLU clamp limit (<=0 disables the clamp)",
    )
    parser.add_argument(
        "--weighted", type=int, nargs="*", default=[1],
        help="1 to multiply broadcast [M,1] row weights, 0 to skip",
    )
    parser.add_argument(
        "--backend", type=str, nargs="*", default=["auto"],
        help="backend(s) to sweep: auto | triton | gluon\n"
        "'auto' picks gluon on gfx1250, triton elsewhere",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = []
        for (m, n), quant, activation, swiglu_limit, weighted, backend in (
            itertools.product(
                args.mn,
                args.quant,
                args.activation,
                args.swiglu_limit,
                args.weighted,
                args.backend,
            )
        ):
            # Resolve "auto" the same way the wrapper does, so the table shows the
            # backend that actually ran; skip an explicit gluon request on a card
            # without a Gluon port rather than aborting the whole sweep.
            if backend == "auto":
                backend = "gluon" if _is_gluon_available() else "triton"
            elif backend == "gluon" and not _is_gluon_available():
                aiter.logger.warning(
                    f"gluon backend unavailable on {get_gfx()}; skipping"
                )
                continue
            df.append(
                test_fused_clamp_act_mul(
                    m, n, dtype, quant, activation, swiglu_limit, bool(weighted), backend
                )
            )
        df = pd.DataFrame(df)
        aiter.logger.info(
            f"fused_clamp_act_mul summary (dtype={dtype}, markdown):\n"
            f"{df.to_markdown(index=False)}"
        )


if __name__ == "__main__":
    main()
