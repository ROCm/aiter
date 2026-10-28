# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import itertools
import os
from unittest import mock

import pandas as pd
import torch
import torch.nn.functional as F

import aiter
import aiter.ops.rmsnorm as rmsnorm_ops
from aiter import dtypes
from aiter.jit.utils import chip_info
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.fused_qk_rmsnorm_group_quant import fused_qk_rmsnorm_group_quant
from aiter.ops.rmsnorm import add_rmsnorm, rmsnorm
from aiter.test_common import benchmark, checkAllclose, perftest, run_perftest


@perftest()
def run_torch(input, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = F.rms_norm(
            input=input, normalized_shape=(input.shape[-1],), weight=weight, eps=eps
        )
    else:
        residual_out = input + residual
        output = F.rms_norm(
            input=residual_out,
            normalized_shape=(input.shape[-1],),
            weight=weight,
            eps=eps,
        )
    return output, residual_out


@perftest()
def run_ck(input, weight, eps, residual=None, use_model_sensitive_rmsnorm=0):
    if residual is None:
        residual_out = None
        output = aiter.rms_norm(input, weight, eps, use_model_sensitive_rmsnorm)
    else:
        residual_out = torch.empty_like(input)
        output = torch.empty_like(input)
        aiter.rmsnorm2d_fwd_with_add(
            output,
            input,
            residual,
            residual_out,
            weight,
            eps,
            use_model_sensitive_rmsnorm=use_model_sensitive_rmsnorm,
        )
    return output, residual_out


@perftest()
def run_cu(input, weight, eps, residual=None):
    if residual is None:
        residual_out = None
        output = torch.empty_like(input)
        aiter.rms_norm_cu(output, input, weight, eps)
    else:
        aiter.fused_add_rms_norm_cu(input, residual, weight, eps)
        output = input
        residual_out = residual
    return output, residual_out


def test_rmsnorm2d(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    # q, k, v = torch.split(hidden_stats, [6*n, n, n], dim=1)
    # input = k
    (a, *_), avg_a = run_torch(input, weight, 1e-5)
    (b, *_), avg_b = run_ck(input, weight, 1e-5)
    (c, *_), avg_c = run_cu(input, weight, 1e-5)
    msg = f"[perf] dim: {dim!s:<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, cu avg: {avg_c:<8.2f} us, uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, msg=msg)
    checkAllclose(a, c, msg="cu")


def test_rmsnorm2d_fuseAdd(dtype, m, n):
    dim = (m, n)
    input = torch.randn(dim, dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    res = torch.randn(dim, dtype=dtype, device="cuda")
    # q, k, v = torch.split(hidden_stats, [6*n, n, n], dim=1)
    # input = k
    (a, res_a, *_), avg_a = run_torch(input, weight, 1e-5, residual=res)
    (b, res_b, *_), avg_b = run_ck(input, weight, 1e-5, residual=res)
    (c, res_c, *_), avg_c = run_ck(
        input, weight, 1e-5, residual=res, use_model_sensitive_rmsnorm=1
    )
    (_d, _res_d, *_), _avg_d = run_cu(input, weight, 1e-5, residual=res)

    msg = f"[perf] dim: {dim!s:<20}, dtype: {dtype}, torch avg: {avg_a:<8.2f} us, ck avg: {avg_b:<8.2f} us, cu avg: {avg_c:<8.2f} us,uplift: {avg_a/avg_b-1:<5.1%}"
    checkAllclose(a, b, atol=0.03, msg=msg)
    checkAllclose(res_a, res_b, msg="ck res check (NO_SPECIFIC_MODEL)")

    checkAllclose(a, c, atol=0.03, msg=msg)
    checkAllclose(res_a, res_c, msg="ck res check (T5_MODEL_LIKE)")
    # checkAllclose(a, d, atol=0.03, msg='cu')
    # checkAllclose(res_a, res_d, atol=0.01, msg='cu res check')

    # gemma_norm: opus folds (weight + 1); reference uses weight (w + 1). Fresh tensors
    # because run_cu above rewrites input/res in place.
    gx = torch.randn(dim, dtype=dtype, device="cuda")
    gr = torch.randn(dim, dtype=dtype, device="cuda")
    gout = torch.empty_like(gx)
    gres = torch.empty_like(gx)
    aiter.rmsnorm2d_fwd_with_add(gout, gx, gr, gres, weight, 1e-5, gemma_norm=True)
    (g, gres_ref, *_), _ = run_torch(gx, (weight + 1).to(dtype), 1e-5, residual=gr)
    checkAllclose(g, gout, atol=0.03, msg="gemma out")
    checkAllclose(gres_ref, gres, msg="gemma res check")


def test_rmsnorm2d_row_tail(dtype, m, n):
    """Every element of rows whose length is not a multiple of 8 is written."""
    x = torch.randn((m, n), dtype=dtype, device="cuda")
    res = torch.randn((m, n), dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    out = torch.full_like(x, float("nan"))
    res_out = torch.full_like(x, float("nan"))
    aiter.rmsnorm2d_fwd_with_add(out, x, res, res_out, weight, 1e-5)
    y = aiter.rms_norm(x, weight, 1e-5)
    added = x.float() + res.float()
    ref = F.rms_norm(added, (n,), weight.float(), 1e-5)
    ref_y = F.rms_norm(x.float(), (n,), weight.float(), 1e-5)
    msg = f"[row tail] dim: {(m, n)!s:<12}, dtype: {dtype}, "
    assert torch.isfinite(out).all() and torch.isfinite(res_out).all(), (
        msg + "unwritten"
    )
    tail = slice(n - 8, n)
    for name, a, b in (
        ("out", out, ref),
        ("residual_out", res_out, added),
        ("y", y, ref_y),
    ):
        err = checkAllclose(
            b[:, tail], a[:, tail].float(), atol=0.03, tol_err_ratio=0, msg=msg + name
        )
        assert err == 0, msg + name


def test_rmsnorm2d_fuseAdd_deterministic(dtype, m, n, runs=5):
    """The block reduction does not race: repeated calls on the same inputs agree bit for bit."""
    x = torch.randn((m, n), dtype=dtype, device="cuda")
    res = torch.randn((m, n), dtype=dtype, device="cuda")
    weight = torch.randn(n, dtype=dtype, device="cuda")
    first = None
    for _ in range(runs):
        out = torch.empty_like(x)
        res_out = torch.empty_like(x)
        aiter.rmsnorm2d_fwd_with_add(out, x, res, res_out, weight, 1e-5)
        if first is None:
            first = out
        else:
            bad_rows = (out != first).any(dim=1).sum().item()
            assert (
                bad_rows == 0
            ), f"[deterministic] dim: {(m, n)}, {bad_rows} rows differ"


def test_rmsnorm2d_tail_dispatch_follows_device():
    """GPU_ARCHS names the build target. A gfx1250 device keeps rows with N % 8 != 0
    off the HIP kernel whatever GPU_ARCHS says."""
    x = torch.empty((1, 1002), dtype=dtypes.bf16)
    old_archs = os.environ.get("GPU_ARCHS")
    os.environ["GPU_ARCHS"] = "gfx942"
    chip_info.get_gfx.cache_clear()
    chip_info.get_gfx_custom_op_core.cache_clear()
    try:
        for device, hip in (("gfx1250", False), ("gfx942", True)):
            with mock.patch.object(
                rmsnorm_ops, "get_gfx_runtime", return_value=device, create=True
            ):
                assert rmsnorm_ops._use_hip_common(x, 0) == hip, (
                    f"[tail dispatch] device {device}, GPU_ARCHS=gfx942: "
                    f"expected {'HIP' if hip else 'opus'}"
                )
    finally:
        if old_archs is None:
            os.environ.pop("GPU_ARCHS", None)
        else:
            os.environ["GPU_ARCHS"] = old_archs
        chip_info.get_gfx.cache_clear()
        chip_info.get_gfx_custom_op_core.cache_clear()


SUPPORTED_GFX = ["gfx942", "gfx950", "gfx1250"]


def run_torch_no_quant(x, weight, epsilon, residual=None, gemma_norm=False):
    """No-quant C++ semantics: normalize the FP32 sum, then cast the output."""
    value = x.to(dtypes.fp32)
    if residual is not None:
        value = value + residual.to(dtypes.fp32)
    factor = weight.to(dtypes.fp32) + (1.0 if gemma_norm else 0.0)
    out = F.rms_norm(value, (x.shape[-1],), factor, epsilon)
    return out.to(x.dtype), value.to(x.dtype) if residual is not None else None


@benchmark()
def test_rmsnorm2d_cpp(m, n, dtype, mode, gemma_norm, layout, second_n=0):
    """Public C++ candidates, including row-strided and dual-input fallbacks.

    For MiniMax-M3, N=6144 is the hidden width; N=128 is a per-head norm,
    not the concatenated heads. M counts rows/tokens (or token*local_heads),
    not context length. This is an independent no-quant baseline, not the
    allreduce/quant/RoPE/cache-insert fused path.
    """
    assert dtype in (dtypes.fp16, dtypes.bf16)
    assert mode in ("norm", "add_norm")
    assert layout in ("contiguous", "strided")
    assert m > 0 and 0 < n <= 8192 and n % 8 == 0
    assert 0 <= second_n <= 8192 and second_n % 8 == 0
    eps = 1e-6
    if layout == "strided":
        # Views of one packed allocation: no copy/contiguous conversion.
        storage = torch.randn((m, n + second_n + 16), dtype=dtype, device="cuda")
        x = storage[:, :n]
        second = storage[:, n : n + second_n] if second_n else None
    else:
        x = torch.randn((m, n), dtype=dtype, device="cuda")
        second = (
            torch.randn((m, second_n), dtype=dtype, device="cuda") if second_n else None
        )
    weight = torch.randn(n, dtype=dtype, device="cuda")
    second_weight = (
        torch.randn(second_n, dtype=dtype, device="cuda") if second_n else None
    )
    residual = None
    if mode == "add_norm":
        residual_storage = torch.randn(
            (m, n + (16 if layout == "strided" else 0)), dtype=dtype, device="cuda"
        )
        residual = residual_storage[:, :n]

    ref, res_ref = run_torch_no_quant(x, weight, eps, residual, gemma_norm)
    second_ref = None
    if second is not None:
        second_ref, _ = run_torch_no_quant(
            second, second_weight, eps, gemma_norm=gemma_norm
        )

    def allocate_output(width):
        if not width:
            return None, None
        backing = torch.full(
            (m, width + (16 if layout == "strided" else 0)),
            7,
            dtype=dtype,
            device="cuda",
        )
        view = backing[:, :width]
        view.fill_(float("nan"))
        return backing, view

    out_storage, out = allocate_output(n)
    res_storage, res_out = allocate_output(n if residual is not None else 0)
    second_storage, second_out = allocate_output(second_n)

    def run_fused():
        fused_qk_rmsnorm_group_quant(
            q=x,
            q_weight=weight,
            q_epsilon=eps,
            q_out_scale=None,
            q_out_unquantized=out,
            q_residual=residual,
            q_res_out=res_out,
            k=second,
            k_weight=second_weight,
            k_out=second_out,
            k_epsilon=eps if second is not None else None,
            gemma_norm=gemma_norm,
        )
        return out, res_out, second_out

    def run_generic():
        if residual is None:
            rmsnorm(out, x, weight, eps, gemma_norm)
        else:
            add_rmsnorm(out, x, residual, res_out, weight, eps, gemma_norm)
        if second is not None:
            rmsnorm(second_out, second, second_weight, eps, gemma_norm)
        return out, res_out, second_out

    candidates = {"fused_cpp": run_fused, "generic_cpp": run_generic}
    # Logical math counts: N squares + N-1 sum additions + mean/epsilon/rsqrt
    # (three scalar operations) + 2N output multiplies. Count rsqrt as one
    # operation. Gemma adds N weight offsets, residual adds N elements per row.
    # These are op-level numerators, not hardware instruction/HBM counters.
    flops = m * ((4 + int(gemma_norm)) * n + 2)
    nbytes = (2 * m * n + n) * x.element_size()
    if residual is not None:
        flops += m * n
        nbytes += 2 * m * n * x.element_size()
    if second is not None:
        flops += m * ((4 + int(gemma_norm)) * second_n + 2)
        nbytes += (2 * m * second_n + second_n) * second.element_size()

    ret = {"gfx": get_gfx()}
    for name, fn in candidates.items():
        # Both candidates reuse MODEL-style caller-provided output buffers.
        # Poison between candidates to catch unwritten elements; no allocations
        # or poisoning are performed in the timed callable.
        for value in (out, res_out, second_out):
            if value is not None:
                value.fill_(float("nan"))
        (actual, res_actual, second_actual), us = run_perftest(fn, num_rotate_args=1)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            actual.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            tol_err_ratio=0,
            msg=f"{name}: rmsnorm",
        )
        assert err == 0, f"{name}: norm mismatch"
        if residual is not None:
            res_err = checkAllclose(
                res_ref.to(dtypes.fp32),
                res_actual.to(dtypes.fp32),
                rtol=0,
                atol=0,
                tol_err_ratio=0,
                msg=f"{name}: residual",
            )
            assert res_err == 0, f"{name}: residual mismatch"
        if second is not None:
            second_err = checkAllclose(
                second_ref.to(dtypes.fp32),
                second_actual.to(dtypes.fp32),
                rtol=1e-2,
                atol=1e-2,
                tol_err_ratio=0,
                msg=f"{name}: second norm",
            )
            assert second_err == 0, f"{name}: second norm mismatch"
            err = max(err, second_err)
        if layout == "strided":
            for backing, width in (
                (out_storage, n),
                (res_storage, n),
                (second_storage, second_n),
            ):
                if backing is not None:
                    assert torch.all(
                        backing[:, width:] == 7
                    ), f"{name}: padding overwritten"
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("C++ RMSNorm unsupported on %s; skipping", get_gfx())
        return
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="C++ no-quant RMSNorm correctness/performance shape sweep",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.fp16, dtypes.bf16],
    )
    parser.add_argument(
        "-m",
        "--m",
        type=int,
        nargs="*",
        default=[1, 7, 128, 255, 256, 257, 1024, 4096, 16384, 32768],
        help="Number of rows, not sequence/context length",
    )
    parser.add_argument(
        "-n",
        "--n",
        type=int,
        nargs="*",
        default=[128, 6144],
        help="Per-row normalization width",
    )
    parser.add_argument(
        "-s",
        "--mn",
        type=dtypes.str2tuple,
        nargs="*",
        help="Explicit (M,N) pairs; otherwise use the -m/-n product",
    )
    parser.add_argument(
        "--modes",
        nargs="*",
        choices=["norm", "add_norm"],
        default=["norm", "add_norm"],
    )
    parser.add_argument("--gemma", type=int, choices=[0, 1], nargs="*", default=[0, 1])
    parser.add_argument(
        "--layout",
        nargs="*",
        choices=["contiguous", "strided"],
        default=["contiguous"],
    )
    parser.add_argument(
        "--second-n",
        type=int,
        nargs="*",
        default=[0],
        help="Optional second input width (0 disables it); its rows match M",
    )
    args = parser.parse_args()
    shapes = args.mn if args.mn is not None else list(itertools.product(args.m, args.n))
    if any(
        len(shape) != 2
        or shape[0] <= 0
        or not 0 < shape[1] <= 8192
        or shape[1] % 8 != 0
        for shape in shapes
    ):
        parser.error("C++ candidates need M>0 and N divisible by 8, 0<N<=8192")
    if any(n < 0 or n > 8192 or n % 8 for n in args.second_n):
        parser.error("--second-n must be 0 or divisible by 8, <=8192")
    torch.manual_seed(2026)
    for dtype in args.dtype:
        if dtype not in (dtypes.fp16, dtypes.bf16):
            aiter.logger.warning(
                "C++ no-quant candidates do not support %s; skipping", dtype
            )
            continue
        rows = []
        for (m, n), mode, gemma, layout, second_n in itertools.product(
            shapes, args.modes, args.gemma, args.layout, args.second_n
        ):
            rows.append(
                test_rmsnorm2d_cpp(m, n, dtype, mode, bool(gemma), layout, second_n)
            )
        df = pd.DataFrame(rows)
        aiter.logger.info(
            "C++ RMSNorm summary (markdown), dtype=%s:\n%s",
            dtype,
            df.to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
