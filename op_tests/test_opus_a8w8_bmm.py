# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression + perf sweep for the opus fp8 e8m0 mxscale flatmm split-K BMM.

Covers the mmajor DeepSeek-V4 wo_a path: O/Y are [M, G, *] (transposed views of
batch-major [G, M, *]); wo_a + w_scale stay batch-major. Activation scale is
per-token e8m0 (GROUP_M=1), weight scale is 128x128-block e8m0. Candidates are
kid 8000 (always-runnable baseline) and the public dispatch path; the reference is
a dequantized fp32 einsum. Per-kid perf comparison / winner selection lives in
``csrc/opus_gemm/opus_bmm_mxscale_tune.py``.

``--check-m-align`` runs a different check instead of the sweep: an every-kid
guard that OpusGemmInstance.m_align still matches launcher behaviour (see
``check_m_align``). It is kept out of the sweep because it deliberately provokes
launch failures and needs no timing.

Usage:
    python3 op_tests/test_opus_a8w8_bmm.py
    python3 op_tests/test_opus_a8w8_bmm.py -s 512,1024,4096 -g 2 -d bf16
    python3 op_tests/test_opus_a8w8_bmm.py --check-m-align
"""

import argparse
import itertools
import sys

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.batched_gemm_op_a8w8 import lookup_mxscale_bmm_config
from aiter.ops.opus import opus_gemm
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]  # fp8 e8m0 mxscale flatmm is gfx950-only
GROUP = 128  # GROUP_N == GROUP_K == 128; GROUP_M == 1 (per-token)
_DT = {"fp32": dtypes.fp32, "bf16": dtypes.bf16}


def _run_opus(
    x,
    weight,
    out,
    x_scale,
    w_scale,
    kid,
    split_k=1,
    workspace=None,
):
    return opus_gemm(
        x,
        weight,
        out,
        kid=int(kid),
        layout="mxscale_bmm",
        x_scale=x_scale,
        w_scale=w_scale,
        split_k=int(split_k),
        workspace=workspace,
    )


def _to_e8m0_scale(scale):
    # Round scale up to a power of two so quantized fp8 values stay in range.
    e = torch.ceil(torch.log2(scale.to(dtypes.fp32))).to(torch.int32) + 127
    e = torch.clamp(e, 0, 255).to(torch.uint8)
    scale_pow2 = torch.exp2(e.to(dtypes.fp32) - 127.0)
    return e, scale_pow2


def _quant_per_token_e8m0(x_bf16):
    """[G,M,K] bf16 -> fp8 + e8m0 x_scale [G,M,K/128] + fp32 scale."""
    G, M, K = x_bf16.shape
    xb = x_bf16.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    raw = xb.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (xb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return q.view(G, M, K), e8m0.squeeze(-1), scale.squeeze(-1)


def _quant_block_e8m0(w_bf16):
    """[G,N,K] bf16 -> fp8 + e8m0 w_scale [G,N/128,K/128] + fp32 scale."""
    G, N, K = w_bf16.shape
    wb = w_bf16.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    raw = wb.abs().amax(dim=(2, 4), keepdim=True).clamp(min=1e-8) / 448.0
    e8m0, scale = _to_e8m0_scale(raw)
    q = (wb / scale).clamp(-448.0, 448.0).to(dtypes.fp8)
    return (
        q.view(G, N, K),
        e8m0.view(G, N // GROUP, K // GROUP),
        scale.view(G, N // GROUP, K // GROUP),
    )


def run_torch(O_fp8, W_fp8, x_scale, w_scale):
    """Reference: dequant fp8 -> fp32 einsum -> [G,M,N]. Not timed."""
    G, M, K = O_fp8.shape
    N = W_fp8.shape[1]
    act = O_fp8.to(dtypes.fp32).view(G, M, K // GROUP, GROUP)
    act = (act * x_scale.unsqueeze(-1)).view(G, M, K)
    W = W_fp8.to(dtypes.fp32).view(G, N // GROUP, GROUP, K // GROUP, GROUP)
    W = (W * w_scale.view(G, N // GROUP, 1, K // GROUP, 1)).view(G, N, K)
    return torch.einsum("gmk,gnk->gmn", act, W).to(dtypes.fp32)


def _block_varied(shape, k):
    """Signed random tensor whose per-128-K-block magnitude spans several powers
    of two, so the e8m0 128-block scales cover many exponents.

    ``rand()/10`` (non-negative, near-uniform) is what let the shipped
    global kids 8312/8313
    tileN COM_REP_N>1 kernels pass this test at ~0.007 rel while silently
    transposing output column groups: a pure column permutation over symmetric
    positive columns barely moves any element, and the collapsed single block
    scale hides scale-application bugs. Signed data makes swapped columns
    uncorrelated (~100% element mismatch), and the varied amplitude exercises
    real per-block scales -- together they turn this test into a real guard."""
    x = torch.randn(shape, dtype=dtypes.fp32)
    amp = torch.exp2(torch.randint(-4, 4, (k // GROUP,), device=x.device).float())
    x = x * amp.repeat_interleave(GROUP)
    return x.to(dtypes.bf16)


@benchmark()
def test_mxscale_bmm(g, m, n, k, dtype):
    ydt = _DT[dtype]
    # Canonical batch-major tensors, then feed the kernel transposed (mmajor)
    # views exactly like the DSV4 wo_a call does (zero-copy, no contiguous copy).
    O_bf16 = _block_varied((g, m, k), k)
    W_bf16 = _block_varied((g, n, k), k)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)

    O_in = O_mx.transpose(0, 1)  # [m,g,k] view
    xs_in = xs_mx.transpose(0, 1)  # [m,g,k/128] view
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)  # [m,g,n]
    y_shape = (m, g, n)

    def _call(kid):
        Y = torch.empty(y_shape, dtype=ydt)
        _run_opus(O_in, W_mx, Y, xs_in, ws_mx, kid)
        return Y

    # Correctness-focused: global kid 8000 is a fixed baseline with no
    # tile-alignment requirement (always runnable), plus the public dispatch path
    # end to end. Per-kid perf comparison / winner selection lives in
    # csrc/opus_gemm/opus_bmm_mxscale_tune.py, not here.
    candidates = {"kid8000_k32_fused": (lambda: _call(8000), ref)}

    # Public backend-neutral entry: per-(g,m,n,k) tuned-CSV lookup + heuristic
    # final-kid resolution + libtype backend routing. Exercises the whole
    # aiter.batched_gemm_a8w8_mxscale -> unified opus_gemm path end to end
    # (not the raw binding).
    candidates["auto (batched_gemm_a8w8_mxscale)"] = (
        lambda: aiter.batched_gemm_a8w8_mxscale(O_in, W_mx, xs_in, ws_mx, dtype=ydt),
        ref,
    )

    flops = 2.0 * g * m * n * k
    # fp8 A + fp8 W + e8m0 scales (uint8) + output.
    nbytes = (
        g * m * k
        + g * n * k
        + g * m * (k // GROUP)
        + g * (n // GROUP) * (k // GROUP)
        + m * g * n * torch.empty((), dtype=ydt).element_size()
    )

    ret = {"gfx": get_gfx()}
    for name, (fn, fn_ref) in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            fn_ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"mxscale_bmm {name} g={g} m={m} n={n} k={k}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


@benchmark()
def test_mxscale_bmm_batch_first(g, m, n, k, dtype):
    """Batch-leading (batch-major) round trip.

    The caller's natural DSV4 buffers are batch-major: batch is the *first*
    (outermost-in-memory) dimension -- activation/output are [G, M, *], weight
    is [G, N, K]. They are handed to the kernel as zero-copy [M, G, *]
    transposed views (dim0=M, dim1=batch), and the result is written straight
    back into a batch-major [G, M, N] buffer through its [M, G, N] view.

    This is the stride path the dropped ``_mmajor`` suffix used to over-claim:
    the batch axis sits at an arbitrary (here outermost) memory position while
    only K (inputs) and N (output) stay contiguous. Same tuned CSV / heuristic
    entries must serve it. Correctness is checked in the caller's native
    [G, M, N] order.
    """
    ydt = _DT[dtype]
    O_bf16 = _block_varied((g, m, k), k)
    W_bf16 = _block_varied((g, n, k), k)
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(O_bf16)
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(W_bf16)

    O_in = O_mx.transpose(0, 1)  # [m, g, k] view (K contiguous)
    xs_in = xs_mx.transpose(0, 1)  # [m, g, k/128] view
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32)  # [g, m, n] batch-major

    def _call_raw(kid):
        # Batch-major output buffer; hand the kernel its [m, g, n] view so the
        # store lands at Y.stride(1) (batch) = m*n (outermost), N contiguous.
        Yb = torch.empty((g, m, n), dtype=ydt)
        _run_opus(O_in, W_mx, Yb.transpose(0, 1), xs_in, ws_mx, kid)
        return Yb  # [g, m, n]

    def _call_auto():
        # Resolve the final global kid here, then use the same unified OPUS
        # entry with a caller-owned batch-major output view.
        Yb = torch.empty((g, m, n), dtype=ydt)
        cfg = lookup_mxscale_bmm_config(g, m, n, k)
        _run_opus(
            O_in,
            W_mx,
            Yb.transpose(0, 1),
            xs_in,
            ws_mx,
            int(cfg["kernelId"]) if cfg is not None else 8000,
            int(cfg["splitK"]) if cfg is not None else 1,
        )
        return Yb

    # Correctness-focused: kid 8000 as the batch-major baseline,
    # plus the backend dispatch path writing into the batch-major buffer via
    # out=. Per-kid perf sweep lives in csrc/opus_gemm/opus_bmm_mxscale_tune.py.
    candidates = {"kid8000_k32_fused": (lambda: _call_raw(8000), ref)}
    candidates["auto (unified opus_gemm)"] = (_call_auto, ref)

    flops = 2.0 * g * m * n * k
    # fp8 A + fp8 W + e8m0 scales (uint8) + output.
    nbytes = (
        g * m * k
        + g * n * k
        + g * m * (k // GROUP)
        + g * (n // GROUP) * (k // GROUP)
        + m * g * n * torch.empty((), dtype=ydt).element_size()
    )

    ret = {"gfx": get_gfx()}
    for name, (fn, fn_ref) in candidates.items():
        out, us = run_perftest(fn)
        err = checkAllclose(
            fn_ref.to(dtypes.fp32),
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"mxscale_bmm_batch_first {name} g={g} m={m} n={n} k={k}",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


# These two functions are CLI benchmarks whose required arguments come from
# ``main()`` rather than pytest fixtures.  Keep their historical names for the
# benchmark report while preventing pytest from invoking them with no shape.
test_mxscale_bmm.__test__ = False
test_mxscale_bmm_batch_first.__test__ = False


# --- tileN column-map regression guard ------------------------------------
# These COM_REP_N>1 kernels previously transposed output column groups. Keep
# them out of the narrow perf table, but always exercise both output layouts
# with signed, varied-block-scale data so the bug cannot silently return.
_TILEN_REGRESSION_KIDS = (8312, 8313)
_TILEN_REGRESSION_SHAPE = (2, 16, 128, 1024)  # G, M, N, K; accepts both kids
_TILEN_REGRESSION_ERR_TOL = 0.003


def check_tilen_column_map():
    """Check global kids 8312/8313 for both output memory layouts."""
    g, m, n, k = _TILEN_REGRESSION_SHAPE
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(_block_varied((g, m, k), k))
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(_block_varied((g, n, k), k))
    O_in = O_mx.transpose(0, 1)
    xs_in = xs_mx.transpose(0, 1)
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)
    failures = []

    for kid in _TILEN_REGRESSION_KIDS:
        for layout in ("token-major", "batch-major"):
            if layout == "token-major":
                out = torch.full((m, g, n), float("nan"), dtype=dtypes.bf16)
            else:
                out = torch.full((g, m, n), float("nan"), dtype=dtypes.bf16).transpose(
                    0, 1
                )
            _run_opus(O_in, W_mx, out, xs_in, ws_mx, kid)
            torch.cuda.synchronize()
            delta = (out.to(dtypes.fp32) - ref).abs()
            rows = delta.flatten(1).mean(1) / (ref.abs().flatten(1).mean(1) + 1e-9)
            err = rows.max().item()
            if not (err <= _TILEN_REGRESSION_ERR_TOL):
                failures.append(
                    f"kid {kid} {layout}: worst row rel err {err:.4f} "
                    f"> {_TILEN_REGRESSION_ERR_TOL}"
                )

    assert not failures, "tileN column-map regression:\n  " + "\n  ".join(failures)
    return len(_TILEN_REGRESSION_KIDS) * 2


def check_workspace_and_graph():
    """Exercise auto/caller workspace, validation and graph replay."""
    g, m, n, k = 2, 32, 128, 2048
    O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(_block_varied((g, m, k), k))
    W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(_block_varied((g, n, k), k))
    O_in = O_mx.transpose(0, 1)
    xs_in = xs_mx.transpose(0, 1)
    ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)
    required_numel = 2 * g * m * n

    auto_out = torch.empty((m, g, n), dtype=dtypes.bf16)
    _run_opus(O_in, W_mx, auto_out, xs_in, ws_mx, 8000, split_k=2)

    workspace = torch.empty(required_numel, dtype=torch.float32)
    caller_out = torch.empty_like(auto_out)
    _run_opus(
        O_in,
        W_mx,
        caller_out,
        xs_in,
        ws_mx,
        8000,
        split_k=2,
        workspace=workspace,
    )
    torch.cuda.synchronize()
    for label, out in (("auto", auto_out), ("caller", caller_out)):
        checkAllclose(
            ref,
            out.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"MXFP8 BMM {label} workspace",
        )

    for bad_workspace in (
        torch.empty(required_numel - 1, dtype=torch.float32),
        torch.empty(required_numel, dtype=torch.bfloat16),
    ):
        try:
            _run_opus(
                O_in,
                W_mx,
                caller_out,
                xs_in,
                ws_mx,
                8000,
                split_k=2,
                workspace=bad_workspace,
            )
        except ValueError:
            pass
        else:
            raise AssertionError("invalid MXFP8 BMM workspace was accepted")

    try:
        _run_opus(
            O_in,
            W_mx,
            caller_out,
            xs_in,
            ws_mx,
            8000,
            split_k=1,
            workspace=workspace,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("split_k=1 unexpectedly accepted workspace")

    graph_out = torch.empty_like(auto_out)
    _run_opus(
        O_in,
        W_mx,
        graph_out,
        xs_in,
        ws_mx,
        8000,
        split_k=2,
        workspace=workspace,
    )
    graph = torch.cuda.CUDAGraph()
    capture_stream = torch.cuda.Stream()
    with torch.cuda.graph(graph, stream=capture_stream):
        _run_opus(
            O_in,
            W_mx,
            graph_out,
            xs_in,
            ws_mx,
            8000,
            split_k=2,
            workspace=workspace,
        )
    graph.replay()
    torch.cuda.synchronize()
    checkAllclose(
        ref,
        graph_out.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="MXFP8 BMM graph workspace",
    )
    return 3


# --- m_align guard ---------------------------------------------------------
# Straddles every tile boundary in the family (B_M is 16/32/64/128/256) and every
# declared m_align (1 / B_M / 2*B_M), with aligned and unaligned M on both sides.
_ALIGN_MS = [1, 17, 48, 64, 96, 127, 128, 129, 200, 255, 256, 512]
_ALIGN_G = 2
# (N, K) candidates: the second entry serves the k1024-only pipeline kids.
_ALIGN_SHAPES = [(1024, 4096), (1024, 1024)]
_ALIGN_ERR_TOL = 0.003  # e8m0 quant floor is ~0.0014; same gate the tuner uses


def _align_kids():
    # Imported here so the sweep and the many scripts reusing the helpers above
    # never need the codegen package on sys.path.
    from csrc.opus_gemm.opus_gemm_common import a8w8_mxscale_bmm_kernel_lists

    return {
        int(kid): inst
        for fam in a8w8_mxscale_bmm_kernel_lists
        for kid, inst in fam.items()
    }


_ALIGN_INPUTS = {}


def _align_inputs(m, n, k):
    """Quantized inputs + fp32 reference for one shape, shared across kids."""
    key = (m, n, k)
    if key not in _ALIGN_INPUTS:
        g = _ALIGN_G
        O_mx, xs_mx, xs_fp32 = _quant_per_token_e8m0(_block_varied((g, m, k), k))
        W_mx, ws_mx, ws_fp32 = _quant_block_e8m0(_block_varied((g, n, k), k))
        ref = run_torch(O_mx, W_mx, xs_fp32, ws_fp32).transpose(0, 1)
        _ALIGN_INPUTS[key] = (
            O_mx.transpose(0, 1),
            W_mx,
            xs_mx.transpose(0, 1),
            ws_mx,
            ref,
        )
    return _ALIGN_INPUTS[key]


def _align_run(kid, m, n, k):
    """Return (ok, rel_err). ok False means the launcher refused the shape."""
    O_in, W_mx, xs_in, ws_mx, ref = _align_inputs(m, n, k)
    # NaN-filled so a row the kernel never writes shows up as nan, not as a
    # plausible value that a mean error would dilute.
    Y = torch.full((m, _ALIGN_G, n), float("nan"), dtype=dtypes.bf16)
    try:
        _run_opus(O_in, W_mx, Y, xs_in, ws_mx, kid)
        torch.cuda.synchronize()
    except (RuntimeError, ValueError):
        # The launcher's AITER_CHECK on M surfaces here. Deliberately not a
        # blanket except: a harness bug must fail loudly, not read as a refusal.
        return False, 0.0
    d = (Y.to(dtypes.fp32) - ref).abs()
    # Per-row, not global: one wrong row out of a long M barely moves the mean.
    rows = d.flatten(1).mean(1) / (ref.abs().flatten(1).mean(1) + 1e-9)
    return True, rows.max().item()


def _align_pick_shape(kid, inst):
    """First (N, K) this kid accepts at an aligned M, or None if it accepts none."""
    for n, k in _ALIGN_SHAPES:
        if n % inst.B_N or k % inst.B_K:
            continue
        if _align_run(kid, max(inst.m_align, inst.B_M), n, k)[0]:
            return n, k
    return None


def check_m_align():
    """Assert OpusGemmInstance.m_align matches what each mxscale BMM kid does.

    m_align says which M values a kid's launcher accepts (1 == it masks a partial
    M tile). Both the tuned caller's padded-M lookup and a
    tuner's candidate filter act on it, so a wrong value is not merely cosmetic:
    too strict hides the fastest kernel from tuning (kid326 lost ~9% at the DSV4
    wo_a decode shapes that way, while the runtime dispatched it at those very
    M), too loose makes both propose a kid whose launcher throws.

    For every kid this checks the declaration against observed behaviour: at an M
    the declaration accepts, the launch must succeed and match the dequantized
    fp32 reference; at an M it rejects, the launch must raise. Kids are never
    silently skipped -- an unrunnable kid is reported.
    """
    kids = _align_kids()
    failures, unrunnable = [], []

    for kid, inst in sorted(kids.items()):
        shape = _align_pick_shape(kid, inst)
        if shape is None:
            unrunnable.append(kid)
            continue
        n, k = shape
        align = inst.m_align
        for m in _ALIGN_MS:
            ok, err = _align_run(kid, m, n, k)
            if m % align == 0:
                if not ok:
                    failures.append(f"kid {kid}: m_align={align} but M={m} rejected")
                elif not (err <= _ALIGN_ERR_TOL):
                    failures.append(
                        f"kid {kid}: M={m} accepted but worst row rel err "
                        f"{err:.4f} > {_ALIGN_ERR_TOL}"
                    )
            elif ok:
                failures.append(
                    f"kid {kid}: m_align={align} claims M={m} unusable, "
                    f"but it ran (worst row rel err {err:.4f}) -- m_align too strict"
                )

    assert not unrunnable, (
        f"kids that ran on no test shape: {unrunnable}; extend _ALIGN_SHAPES so "
        f"the guard keeps covering them"
    )
    assert not failures, "m_align disagrees with the launcher:\n  " + "\n  ".join(
        failures
    )
    return len(kids)


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "opus mxscale flatmm BMM unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="opus fp8 e8m0 mxscale flatmm split-K BMM test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=str,
        nargs="*",
        default=["bf16"],
        choices=["bf16", "fp32"],
        help="output dtype(s) to sweep (default: bf16)",
    )
    parser.add_argument(
        "-g",
        "--groups",
        type=int,
        nargs="*",
        default=[2, 8],
        help="batch group counts to sweep (DSV4 wo_a G; default: 2)",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (1, 1024, 4096),
            (16, 1024, 4096),
            (128, 1024, 4096),
            (256, 1024, 4096),
            (512, 1024, 4096),
            (8192, 1024, 4096),
            (16384, 1024, 4096),
        ],
        help="(m,n,k) shapes to sweep",
    )
    parser.add_argument(
        "--check-m-align",
        action="store_true",
        help="run the every-kid m_align guard instead of the perf sweep",
    )
    args = parser.parse_args()

    n_tilen_checks = check_tilen_column_map()
    aiter.logger.info(
        "tileN column mapping passed for %d kid/layout combinations", n_tilen_checks
    )
    n_workspace_checks = check_workspace_and_graph()
    aiter.logger.info(
        "MXFP8 BMM workspace/graph checks passed (%d paths)", n_workspace_checks
    )

    if args.check_m_align:
        try:
            n_kids = check_m_align()
        except AssertionError as exc:
            aiter.logger.error("m_align guard FAILED: %s", exc)
            sys.exit(1)
        aiter.logger.info(
            "m_align matches launcher behaviour for all %d mxscale BMM kids", n_kids
        )
        return

    for dtype in args.dtype:
        df = []
        df_bf = []
        for g, (m, n, k) in itertools.product(args.groups, args.mnk):
            df.append(test_mxscale_bmm(g, m, n, k, dtype))
            df_bf.append(test_mxscale_bmm_batch_first(g, m, n, k, dtype))
        aiter.logger.info(
            "opus mxscale flatmm BMM summary (dtype=%s):\n%s",
            dtype,
            pd.DataFrame(df).to_markdown(index=False),
        )
        aiter.logger.info(
            "opus mxscale flatmm BMM batch-first (batch-major) summary "
            "(dtype=%s):\n%s",
            dtype,
            pd.DataFrame(df_bf).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
