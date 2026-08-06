# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Regression + perf sweep for the opus fp8 e8m0 mxscale flatmm split-K BMM.

Covers the mmajor DeepSeek-V4 wo_a path: O/Y are [M, G, *] (transposed views of
batch-major [G, M, *]); wo_a + w_scale stay batch-major. Activation scale is
per-token e8m0 (GROUP_M=1), weight scale is 128x128-block e8m0. Candidates are
the curated flatmm kernel IDs; the reference is a dequantized fp32 einsum.

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
from aiter.ops.opus.bmm_op import _opus_bmm_a8w8_mxscale_raw
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx950"]  # fp8 e8m0 mxscale flatmm is gfx950-only
GROUP = 128  # GROUP_N == GROUP_K == 128; GROUP_M == 1 (per-token)
_DT = {"fp32": dtypes.fp32, "bf16": dtypes.bf16}

# Curated flatmm kernel IDs (splitK == 1, direct store). Each requires
# m % B_M == 0, n % B_N == 0, k % 128 == 0 for its tile.
FLATMM_KIDS = {
    "m64n64k128": (650, 64, 64),
    "m64n64k128_scale_prefetch": (653, 64, 64),
    # M-tile interleaved 128x128 (MI=2 tiles/WG share B). Needs M % 256 == 0;
    # wins on large-M shapes where the 64x64 tiles are memory/pipeline bound.
    "m128n128k128_minterleave": (163, 256, 128),
    # tileN (T_N=2) COM_REP_N>1 regression guards: these are the configs that
    # transposed output column groups on signed / varied-block-scale data (the
    # DSV4 wo_a DP-attention G=16 small-M case). Kept in the sweep so the
    # correctness check keeps exercising the fixed consumer-store column map.
    "m16n64k256_tileN_crn2": (313, 16, 64),
    "m16n128k256_tileN_crn4": (312, 16, 128),
}


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

    ``rand()/10`` (non-negative, near-uniform) is what let the shipped kid312/313
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
        _opus_bmm_a8w8_mxscale_raw(O_in, W_mx, Y, xs_in, ws_mx, 1, kid)
        return Y

    candidates = {}
    for name, (kid, bm, bn) in FLATMM_KIDS.items():
        # Skip a tile whose block shape does not divide this (m, n).
        if m % bm == 0 and n % bn == 0 and k % GROUP == 0:
            candidates[name] = (lambda kid=kid: _call(kid), ref)

    # Public backend-neutral entry: no kernelId -> per-(g,m,n,k) tuned-CSV
    # lookup + heuristic fallback + libtype backend routing. Exercises the
    # whole aiter.batched_gemm_a8w8_mxscale -> bmm_a8w8_mxscale_opus path end
    # to end (not the raw binding). Always runnable: on a CSV/heuristic miss it
    # falls back to kid 0 (k32 fused), which has no tile-alignment requirement.
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
        _opus_bmm_a8w8_mxscale_raw(O_in, W_mx, Yb.transpose(0, 1), xs_in, ws_mx, 1, kid)
        return Yb  # [g, m, n]

    def _call_auto():
        Yb = torch.empty((g, m, n), dtype=ydt)
        aiter.batched_gemm_a8w8_mxscale(
            O_in, W_mx, xs_in, ws_mx, out=Yb.transpose(0, 1), dtype=ydt
        )
        return Yb

    # kid 0 (k32 fused) has no tile-alignment requirement -> always runnable.
    candidates = {"kid0_k32_fused": (lambda: _call_raw(0), ref)}
    for name, (kid, bm, bn) in FLATMM_KIDS.items():
        if m % bm == 0 and n % bn == 0 and k % GROUP == 0:
            candidates[name] = (lambda kid=kid: _call_raw(kid), ref)
    # Public dispatch path, writing into the batch-major buffer via out=.
    candidates["auto (batched_gemm_a8w8_mxscale)"] = (_call_auto, ref)

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
        _opus_bmm_a8w8_mxscale_raw(O_in, W_mx, Y, xs_in, ws_mx, 1, kid)
        torch.cuda.synchronize()
    except RuntimeError:
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
    M tile). Both the runtime's padded-M lookup (aiter/ops/opus/bmm_op.py) and a
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
