# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import argparse
import os
from unittest import mock

import torch
import torch.nn.functional as F

import aiter
import aiter.ops.rmsnorm as rmsnorm_ops
from aiter import dtypes
from aiter.jit.utils import chip_info
from aiter.test_common import checkAllclose, perftest


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


_GUARD_PAD = 64


def _guarded(m, n, dtype, guard):
    """An ``(m, n)`` tensor inside a larger buffer; every other element holds ``guard``."""
    numel = m * n
    if dtype == dtypes.fp8:
        buf = torch.full(
            (2 * _GUARD_PAD + numel,), guard, dtype=torch.uint8, device="cuda"
        )
        return buf, buf[_GUARD_PAD : _GUARD_PAD + numel].view(dtype).view(m, n)
    buf = torch.full((2 * _GUARD_PAD + numel,), guard, dtype=dtype, device="cuda")
    return buf, buf[_GUARD_PAD : _GUARD_PAD + numel].view(m, n)


def _guards_changed(buf, numel, guard):
    """Guard elements of ``_guarded``'s buffer that no longer hold ``guard``."""
    g = torch.full((1,), guard, dtype=buf.dtype, device=buf.device)
    return int((buf[:_GUARD_PAD] != g).sum() + (buf[_GUARD_PAD + numel :] != g).sum())


def test_rmsnorm2d_row_bound(dtype, m, n):
    """Nothing is read or written past a row that is not a whole number of 4-byte words.

    That is an odd N for the 16-bit input and output, and N % 4 != 0 for the 8-bit
    output of the fused quant. Outputs sit inside sentinel-filled buffers, so a write
    past a row lands on a sentinel or on the next row; inputs sit inside buffers whose
    guards are large, so a read past the last row changes that row."""
    eps = 1e-5
    _, x = _guarded(m, n, dtype, 100.0)
    _, res = _guarded(m, n, dtype, 100.0)
    w = _guarded(1, n, dtype, 100.0)[1].view(n)
    x.copy_(torch.randn((m, n), dtype=dtype, device="cuda"))
    res.copy_(torch.randn((m, n), dtype=dtype, device="cuda"))
    w.copy_(torch.rand(n, device="cuda").add(0.5).to(dtype))
    added = x.float() + res.float()
    ref_add = F.rms_norm(added, (n,), w.float(), eps)
    ref = F.rms_norm(x.float(), (n,), w.float(), eps)
    msg = f"[row bound] dim: {(m, n)!s:<12}, dtype: {dtype}, "

    out_buf, out = _guarded(m, n, dtype, 12345.0)
    res_buf, res_out = _guarded(m, n, dtype, 12345.0)
    aiter.rmsnorm2d_fwd_with_add(out, x, res, res_out, w, eps)
    y = aiter.rms_norm(x, w, eps)
    for name, buf in (("out", out_buf), ("residual_out", res_buf)):
        changed = _guards_changed(buf, m * n, 12345.0)
        assert changed == 0, msg + f"{name}: {changed} elements written past the tensor"
    for name, a, b in (
        ("out", out, ref_add),
        ("residual_out", res_out, added),
        ("y", y, ref),
    ):
        err = checkAllclose(b, a.float(), atol=0.03, tol_err_ratio=0, msg=msg + name)
        assert err == 0, msg + name

    for name, with_add, q_ref in (("quant", False, ref), ("add quant", True, ref_add)):
        q_buf, q = _guarded(m, n, dtypes.fp8, 0xA5)
        s_buf = torch.full(
            (2 * _GUARD_PAD + m,), 12345.0, dtype=dtypes.fp32, device="cuda"
        )
        scale = s_buf[_GUARD_PAD : _GUARD_PAD + m].view(m, 1)
        written = [("out", q_buf, m * n, 0xA5), ("scale", s_buf, m, 12345.0)]
        if with_add:
            r_buf, r_out = _guarded(m, n, dtype, 12345.0)
            aiter.rmsnorm2d_fwd_with_add_dynamicquant(q, x, res, r_out, scale, w, eps)
            written.append(("residual_out", r_buf, m * n, 12345.0))
            err = checkAllclose(
                added,
                r_out.float(),
                atol=0.03,
                tol_err_ratio=0,
                msg=msg + f"{name} residual_out",
            )
            assert err == 0, msg + f"{name} residual_out"
        else:
            aiter.rmsnorm2d_fwd_with_dynamicquant(q, x, scale, w, eps)
        for what, buf, numel, guard in written:
            changed = _guards_changed(buf, numel, guard)
            assert changed == 0, (
                msg + f"{name} {what}: {changed} elements written past the tensor"
            )
        # fp8 (e4m3) rounds a value by up to 1/16 of itself; a byte that another row
        # wrote, or that was computed from an element past the row, is further off.
        tol = 0.07 * q_ref.abs() + q_ref.abs().amax(dim=1, keepdim=True) / 448
        bad = ((q.float() * scale - q_ref).abs() > tol).sum().item()
        assert bad == 0, msg + f"{name} out: {bad} elements off the reference"


def test_rmsnorm_hip_refuses_partial_words():
    """Each of the four HIP entry points refuses a row that is not a whole number of
    4-byte words (an odd N, and N % 4 != 0 with the 8-bit output of the two quant
    entries) and writes nothing."""
    for name, n in (
        ("rmsnorm", 1003),
        ("add_rmsnorm", 1003),
        ("rmsnorm_quant", 1002),
        ("rmsnorm_quant", 1003),
        ("add_rmsnorm_quant", 1002),
        ("add_rmsnorm_quant", 1003),
    ):
        quant = name.endswith("_quant")
        x = torch.randn((2, n), dtype=dtypes.bf16, device="cuda")
        w = torch.ones(n, dtype=dtypes.bf16, device="cuda")
        out_buf, out = _guarded(2, n, dtypes.fp8 if quant else dtypes.bf16, 0x5A)
        args, written = [out, x], [("out", out_buf, 0x5A)]
        if name.startswith("add_"):
            res_buf, res_out = _guarded(2, n, dtypes.bf16, 12345.0)
            args += [torch.randn_like(x), res_out]
            written.append(("residual_out", res_buf, 12345.0))
        if quant:
            scale = torch.full((2, 1), 12345.0, dtype=dtypes.fp32, device="cuda")
            args.append(scale)
            written.append(("scale", scale, 12345.0))
        try:
            getattr(rmsnorm_ops, name)(*args, w, 1e-5)
        except RuntimeError as exc:
            assert "whole 4-byte words" in str(exc), exc
        else:
            raise AssertionError(f"[row bound] {name} accepted n = {n}")
        torch.cuda.synchronize()
        for what, buf, guard in written:
            g = torch.full((1,), guard, dtype=buf.dtype, device="cuda")
            assert bool((buf == g).all()), f"[row bound] {name}, n = {n}: wrote {what}"


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
    x_odd = torch.empty((1, 1003), dtype=dtypes.bf16)
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
                # No device takes a row that is not a whole number of 4-byte words:
                # an odd N, or N % 4 != 0 with the 8-bit output of the fused quant.
                assert not rmsnorm_ops._use_hip_common(
                    x_odd, 0
                ), f"[tail dispatch] device {device}: odd N expected on opus"
                assert not rmsnorm_ops._use_hip_common(x, 0, 1), (
                    f"[tail dispatch] device {device}: N % 4 != 0 with an 8-bit "
                    "output expected on opus"
                )
    finally:
        if old_archs is None:
            os.environ.pop("GPU_ARCHS", None)
        else:
            os.environ["GPU_ARCHS"] = old_archs
        chip_info.get_gfx.cache_clear()
        chip_info.get_gfx_custom_op_core.cache_clear()


# for dtype in [dtypes.fp16, dtypes.bf16]:
#     for m in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
#         for n in [4096, 8192, 16384, 32768, 65536]:
#             test_rmsnorm2d(dtype, m, n)

l_dtype = ["fp16", "bf16", "fp32"]
l_m = [1, 2, 4, 8, 16, 32, 64, 128, 256]
l_n = [4096, 8192, 16384, 32768, 65536]
parser = argparse.ArgumentParser(
    formatter_class=argparse.RawTextHelpFormatter,
    description="config input of test",
)
parser.add_argument(
    "-d",
    "--dtype",
    type=str,
    choices=l_dtype,
    nargs="?",
    const=None,
    default=None,
    help="""Data type.
    e.g.: -d bf16""",
)
parser.add_argument(
    "-m",
    "--m",
    type=int,
    nargs="?",
    default=None,
    help="""M of mnk.
    e.g.: -m 32""",
)
parser.add_argument(
    "-n",
    "--n",
    type=int,
    nargs="?",
    default=None,
    help="""N of mnk.
    e.g.: -n 1024""",
)

args = parser.parse_args()
if args.dtype is None:
    l_dtype = [dtypes.d_dtypes[key] for key in l_dtype]
else:
    l_dtype = [dtypes.d_dtypes[args.dtype]]
if args.m is not None:
    l_m = [args.m]
if args.n is not None:
    l_n = [args.n]

print("\nstart fuse add test")
for dtype in l_dtype:
    for m in l_m:
        for n in l_n:
            test_rmsnorm2d_fuseAdd(dtype, m, n)

print("\nstart row tail and determinism tests")
test_rmsnorm2d_tail_dispatch_follows_device()
for dtype in [d for d in l_dtype if d in (dtypes.fp16, dtypes.bf16)]:
    for m in [1, 64]:
        for n in [1002, 4094, 8190]:
            test_rmsnorm2d_row_tail(dtype, m, n)
    for m in [1, 4, 64]:
        for n in [255, 257, 1001, 1002, 1003, 4095]:
            test_rmsnorm2d_row_bound(dtype, m, n)
    test_rmsnorm2d_row_bound(dtype, 7, 769)  # the shape reported in #5044
    test_rmsnorm2d_fuseAdd_deterministic(dtype, 8192, 65536)
test_rmsnorm_hip_refuses_partial_words()
