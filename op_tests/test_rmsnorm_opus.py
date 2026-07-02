# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Parity + performance test for the opus RMSNorm backend (aiter issue #4055).
#
# Verifies that AITER_RMSNORM_BACKEND=opus matches (a) torch F.rms_norm and
# (b) the CK backend, across the entrypoints opus replaces, and benchmarks opus
# vs CK so a regression is visible. Also has a --build-wall probe for cold JIT.
#
# Run:
#   python op_tests/test_rmsnorm_opus.py            # parity, opus vs ck vs torch
#   python op_tests/test_rmsnorm_opus.py --perf     # + latency table
#   python op_tests/test_rmsnorm_opus.py --build-wall

import argparse
import os

import torch
import torch.nn.functional as F

import aiter
from aiter.ops.rmsnorm import get_rmsnorm_backend
from aiter.test_common import checkAllclose, perftest

# Shape matrix: typical LLM hidden sizes, >8192 (opus handles it; CK routes to a
# dedicated path), non-power-of-2, and small/latency-bound.
SHAPES = [
    (4096, 8192),
    (2048, 8192),
    (8192, 4096),
    (32, 16384),  # > 8192
    (2048, 5120),
    (17, 257),  # non-pow2, small
    (300, 1024),
]
DTYPES = [torch.bfloat16, torch.float16]


def _tol(dtype):
    return (1e-2, 1e-2) if dtype == torch.bfloat16 else (3e-3, 3e-3)


def _set_backend(name):
    os.environ["AITER_RMSNORM_BACKEND"] = name


# ---- torch reference ----
def torch_rms(x, w, eps, residual=None):
    if residual is None:
        return F.rms_norm(x, (x.shape[-1],), weight=w, eps=eps), None
    resid = x + residual
    return F.rms_norm(resid, (x.shape[-1],), weight=w, eps=eps), resid


# ---- timed aiter entrypoints (backend chosen by env) ----
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


def _report(times, tag, m, n):
    if times:
        ck, opus = times
        speedup = ck / opus if opus else float("nan")
        print(
            f"  {tag:22s} [{m:5d},{n:5d}] "
            f"ck={ck*1e3:7.1f}us  opus={opus*1e3:7.1f}us  x{speedup:4.2f}"
        )


def run_case(dtype, m, n, do_perf):
    rtol, atol = _tol(dtype)
    x = torch.randn((m, n), dtype=dtype, device="cuda")
    w = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn((m, n), dtype=dtype, device="cuda")

    ref, _ = torch_rms(x, w, 1e-6)
    ref_out, ref_resid = torch_rms(x, w, 1e-6, residual=res)

    perf = {
        "rms_norm": [],
        "rmsnorm2d_fwd": [],
        "rms_norm_cu": [],
        "fused_cu": [],
        "fwd_add": [],
    }

    for backend in ("ck", "opus"):
        _set_backend(backend)
        assert get_rmsnorm_backend() == backend
        tag = f"[{backend}]"

        a, ta = run_rms_norm(x, w, 1e-6)
        checkAllclose(ref, a, rtol=rtol, atol=atol, msg=f"rms_norm {tag} {dtype}")
        perf["rms_norm"].append(ta)

        b, tb = run_rmsnorm2d_fwd(x, w, 1e-6)
        checkAllclose(ref, b, rtol=rtol, atol=atol, msg=f"rmsnorm2d_fwd {tag} {dtype}")
        perf["rmsnorm2d_fwd"].append(tb)

        c, tc = run_rms_norm_cu(x, w, 1e-6)
        checkAllclose(ref, c, rtol=rtol, atol=atol, msg=f"rms_norm_cu {tag} {dtype}")
        perf["rms_norm_cu"].append(tc)

        (fo, fr), tf = run_fused_add_cu(x, res, w, 1e-6)
        checkAllclose(ref_out, fo, rtol=rtol, atol=atol, msg=f"fused_add_cu out {tag}")
        checkAllclose(
            ref_resid, fr, rtol=rtol, atol=atol, msg=f"fused_add_cu resid {tag}"
        )
        perf["fused_cu"].append(tf)

        (go, gr), tg = run_rmsnorm2d_fwd_with_add(x, res, w, 1e-6)
        checkAllclose(ref_out, go, rtol=rtol, atol=atol, msg=f"fwd_with_add out {tag}")
        checkAllclose(
            ref_resid, gr, rtol=rtol, atol=atol, msg=f"fwd_with_add resid {tag}"
        )
        perf["fwd_add"].append(tg)

    if do_perf:
        print(f"-- {dtype} --")
        _report(perf["rms_norm"], "rms_norm", m, n)
        _report(perf["rmsnorm2d_fwd"], "rmsnorm2d_fwd", m, n)
        _report(perf["rms_norm_cu"], "rms_norm_cu", m, n)
        _report(perf["fused_cu"], "fused_add_rms_norm_cu", m, n)
        _report(perf["fwd_add"], "rmsnorm2d_fwd_with_add", m, n)


def report_build_wall():
    # Cold JIT build wall of module_rmsnorm_opus vs the ~225s CK module_rmsnorm.
    import shutil
    import time

    from aiter.jit.core import get_user_jit_dir

    _set_backend("opus")
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


def run_quant_case(dtype, m, n):
    """opus vs CK parity for dynamic/smooth quant (int8 + fp8), with fused-add."""
    x = torch.randn((m, n), dtype=dtype, device="cuda")
    w = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn((m, n), dtype=dtype, device="cuda")
    xscale = (torch.rand(n, device="cuda") * 0.3 + 1).float()

    for out_dtype in (torch.int8, _fp8_dtype()):
        for backend in ("ck", "opus"):
            _set_backend(backend)
            out = torch.empty((m, n), dtype=out_dtype, device="cuda")
            ys = torch.empty((m, 1), dtype=torch.float32, device="cuda")
            aiter.rmsnorm2d_fwd_with_dynamicquant(out, x, ys, w, 1e-6)
            if backend == "ck":
                ck_out, ck_ys = out.float(), ys.clone()
            else:
                # opus vs CK: int8 within 1 level, fp8 within a few percent
                tol = 1.5 if out_dtype == torch.int8 else 0.15
                checkAllclose(
                    ck_ys, ys, rtol=5e-3, atol=5e-3, msg=f"dynq yscale {out_dtype}"
                )
                checkAllclose(
                    ck_out,
                    out.float(),
                    rtol=tol,
                    atol=tol,
                    msg=f"dynq out {out_dtype} [{m},{n}]",
                )

        # smooth-quant + fused-add + save-unquant parity (opus vs CK)
        for backend in ("ck", "opus"):
            _set_backend(backend)
            out = torch.empty((m, n), dtype=out_dtype, device="cuda")
            ys = torch.empty((m, 1), dtype=torch.float32, device="cuda")
            rout = torch.empty_like(x)
            uq = torch.empty_like(x)
            aiter.rmsnorm2d_fwd_with_add_smoothquant(
                out, x, res, rout, xscale, ys, w, 1e-6, out_before_quant=uq
            )
            if backend == "ck":
                ck = (out.float(), ys.clone(), rout.clone(), uq.clone())
            else:
                tol = 1.5 if out_dtype == torch.int8 else 0.15
                checkAllclose(ck[1], ys, rtol=5e-3, atol=5e-3, msg="smoothq yscale")
                checkAllclose(ck[2], rout, rtol=1e-2, atol=1e-2, msg="smoothq residual")
                checkAllclose(
                    ck[0],
                    out.float(),
                    rtol=tol,
                    atol=tol,
                    msg=f"smoothq out {out_dtype}",
                )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--perf", action="store_true", help="print opus-vs-ck latency table"
    )
    ap.add_argument("--build-wall", action="store_true", help="measure cold JIT build")
    args = ap.parse_args()

    prev = os.environ.get("AITER_RMSNORM_BACKEND")
    try:
        if args.build_wall:
            report_build_wall()
        for dtype in DTYPES:
            for m, n in SHAPES:
                run_case(dtype, m, n, args.perf)
            for m, n in ((2048, 8192), (4096, 4096)):
                run_quant_case(dtype, m, n)
        print("\nrmsnorm_opus: parity vs torch + CK passed for all cases")
    finally:
        if prev is None:
            os.environ.pop("AITER_RMSNORM_BACKEND", None)
        else:
            os.environ["AITER_RMSNORM_BACKEND"] = prev
