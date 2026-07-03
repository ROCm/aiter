# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Parity + performance test for the opus RMSNorm backend (aiter issue #4055).
#
# opus is the sole rmsnorm backend (CK removed). Verifies the public entrypoints
# match torch F.rms_norm and that dynamic/smooth quant matches a torch reference,
# across bf16/fp16/fp32, model-sensitive on/off, and typical/edge shapes. Also
# has a --build-wall probe for the cold JIT build (~1s opus vs ~225s legacy CK).
#
# Run:
#   python op_tests/test_rmsnorm_opus.py            # parity vs torch
#   python op_tests/test_rmsnorm_opus.py --perf     # + latency table
#   python op_tests/test_rmsnorm_opus.py --build-wall

import argparse
import os

import torch
import torch.nn.functional as F

import aiter
from aiter.test_common import checkAllclose, perftest

# Shape matrix: typical LLM hidden sizes, >8192, non-power-of-2, small/latency-bound.
SHAPES = [
    (4096, 8192),
    (2048, 8192),
    (8192, 4096),
    (32, 16384),  # > 8192
    (2048, 5120),
    (17, 257),  # non-pow2, small
    (300, 1024),
]
DTYPES = [torch.bfloat16, torch.float16, torch.float32]


def _tol(dtype):
    if dtype == torch.bfloat16:
        return (1e-2, 1e-2)
    if dtype == torch.float16:
        return (3e-3, 3e-3)
    return (1e-5, 1e-5)


# ---- torch reference ----
def torch_rms(x, w, eps, residual=None):
    if residual is None:
        return F.rms_norm(x, (x.shape[-1],), weight=w, eps=eps), None
    resid = x + residual
    return F.rms_norm(resid, (x.shape[-1],), weight=w, eps=eps), resid


# ---- timed aiter (opus) entrypoints ----
@perftest()
def run_rms_norm(x, w, eps):
    return aiter.rms_norm(x, w, eps)


@perftest()
def run_rmsnorm2d_fwd(x, w, eps):
    return aiter.rmsnorm2d_fwd(x, w, eps)


@perftest()
def run_rms_norm_cu(x, w, eps):
    out = torch.empty_like(x)
    aiter.rms_norm_cu(out, x, w, eps)
    return out


@perftest()
def run_fused_add_cu(x, residual, w, eps):
    xi, ri = x.clone(), residual.clone()
    aiter.fused_add_rms_norm_cu(xi, ri, w, eps)
    return xi, ri


@perftest()
def run_rmsnorm2d_fwd_with_add(x, residual, w, eps):
    out = torch.empty_like(x)
    resid_out = torch.empty_like(x)
    aiter.rmsnorm2d_fwd_with_add(out, x, residual, resid_out, w, eps)
    return out, resid_out


def run_case(dtype, m, n, do_perf):
    rtol, atol = _tol(dtype)
    x = torch.randn((m, n), dtype=dtype, device="cuda")
    w = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn((m, n), dtype=dtype, device="cuda")

    ref, _ = torch_rms(x, w, 1e-6)
    ref_out, ref_resid = torch_rms(x, w, 1e-6, residual=res)

    a, ta = run_rms_norm(x, w, 1e-6)
    checkAllclose(ref, a, rtol=rtol, atol=atol, msg=f"rms_norm {dtype}")

    b, tb = run_rmsnorm2d_fwd(x, w, 1e-6)
    checkAllclose(ref, b, rtol=rtol, atol=atol, msg=f"rmsnorm2d_fwd {dtype}")

    c, tc = run_rms_norm_cu(x, w, 1e-6)
    checkAllclose(ref, c, rtol=rtol, atol=atol, msg=f"rms_norm_cu {dtype}")

    (fo, fr), tf = run_fused_add_cu(x, res, w, 1e-6)
    checkAllclose(ref_out, fo, rtol=rtol, atol=atol, msg="fused_add_cu out")
    checkAllclose(ref_resid, fr, rtol=rtol, atol=atol, msg="fused_add_cu resid")

    (go, gr), tg = run_rmsnorm2d_fwd_with_add(x, res, w, 1e-6)
    checkAllclose(ref_out, go, rtol=rtol, atol=atol, msg="fwd_with_add out")
    checkAllclose(ref_resid, gr, rtol=rtol, atol=atol, msg="fwd_with_add resid")

    if do_perf:
        print(
            f"-- {dtype} [{m:5d},{n:5d}] --  "
            f"rms_norm={ta*1e3:6.1f}us  fwd={tb*1e3:6.1f}us  "
            f"cu={tc*1e3:6.1f}us  fused={tf*1e3:6.1f}us  fwd_add={tg*1e3:6.1f}us"
        )


def report_build_wall():
    # Cold JIT build wall of module_rmsnorm_opus vs the ~225s CK module_rmsnorm.
    import shutil
    import time

    from aiter.jit.core import get_user_jit_dir

    jit = get_user_jit_dir()
    for p in (
        os.path.join(jit, "module_rmsnorm_opus.so"),
        os.path.join(jit, "build", "module_rmsnorm_opus"),
    ):
        if os.path.isfile(p):
            os.remove(p)
        elif os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
    x = torch.randn((16, 8192), dtype=torch.bfloat16, device="cuda")
    w = torch.ones(8192, dtype=torch.bfloat16, device="cuda")
    t0 = time.perf_counter()
    aiter.rms_norm(x, w, 1e-6)  # blocks on cold build
    torch.cuda.synchronize()
    print(
        f"module_rmsnorm_opus cold build + first call: {time.perf_counter() - t0:.1f}s"
    )


def _fp8_dtype():
    from aiter import dtypes

    return getattr(dtypes, "fp8", torch.float8_e4m3fn)


def _quant_ref(n_val, out_dtype):
    """torch reference: per-row yscale = max|n|/qmax, out = round-to-nearest(n/yscale)."""
    qmax = 127.0 if out_dtype == torch.int8 else torch.finfo(out_dtype).max
    rowmax = n_val.abs().amax(-1, keepdim=True)
    yscale = rowmax / qmax
    q = n_val / yscale
    if out_dtype == torch.int8:
        out = torch.clamp(torch.round(q), -128, 127)
    else:
        out = q.to(out_dtype).float()
    return out, yscale


def run_quant_case(dtype, m, n):
    """opus dynamic/smooth quant (int8 + fp8, with fused-add) vs torch reference."""
    x = torch.randn((m, n), dtype=dtype, device="cuda")
    w = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn((m, n), dtype=dtype, device="cuda")
    xscale = (torch.rand(n, device="cuda") * 0.3 + 1).float()

    for out_dtype in (torch.int8, _fp8_dtype()):
        # int8: within 1 quantization level; fp8: within a few percent (1 ULP ~ 12.5%)
        tol = 1.5 if out_dtype == torch.int8 else 0.15

        # dynamic quant
        out = torch.empty((m, n), dtype=out_dtype, device="cuda")
        ys = torch.empty((m, 1), dtype=torch.float32, device="cuda")
        aiter.rmsnorm2d_fwd_with_dynamicquant(out, x, ys, w, 1e-6)
        nrm = torch_rms(x, w, 1e-6)[0].float()
        ref_out, ref_ys = _quant_ref(nrm, out_dtype)
        checkAllclose(ref_ys, ys, rtol=5e-3, atol=5e-3, msg=f"dynq yscale {out_dtype}")
        checkAllclose(
            ref_out,
            out.float(),
            rtol=tol,
            atol=tol,
            msg=f"dynq out {out_dtype} [{m},{n}]",
        )

        # smooth-quant + fused-add + save-unquant
        out = torch.empty((m, n), dtype=out_dtype, device="cuda")
        ys = torch.empty((m, 1), dtype=torch.float32, device="cuda")
        rout = torch.empty_like(x)
        uq = torch.empty_like(x)
        aiter.rmsnorm2d_fwd_with_add_smoothquant(
            out, x, res, rout, xscale, ys, w, 1e-6, out_before_quant=uq
        )
        ref_norm, ref_resid = torch_rms(x, w, 1e-6, residual=res)
        nrm = ref_norm.float() * xscale
        ref_out, ref_ys = _quant_ref(nrm, out_dtype)
        checkAllclose(ref_ys, ys, rtol=5e-3, atol=5e-3, msg="smoothq yscale")
        checkAllclose(
            ref_resid.to(dtype).float(),
            rout.float(),
            rtol=1e-2,
            atol=1e-2,
            msg="smoothq residual",
        )
        checkAllclose(
            ref_out, out.float(), rtol=tol, atol=tol, msg=f"smoothq out {out_dtype}"
        )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf", action="store_true", help="print opus latency table")
    ap.add_argument("--build-wall", action="store_true", help="measure cold JIT build")
    args = ap.parse_args()

    if args.build_wall:
        report_build_wall()
    for dtype in DTYPES:
        for m, n in SHAPES:
            run_case(dtype, m, n, args.perf)
        for m, n in ((2048, 8192), (4096, 4096)):
            run_quant_case(dtype, m, n)
    print("\nrmsnorm_opus: parity vs torch passed for all cases")
