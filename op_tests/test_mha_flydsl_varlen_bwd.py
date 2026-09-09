# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL varlen FMHA backward (d_qk=192, d_v=128, causal, bf16, THD) on gfx942.

Validates the FlyDSL backward -- the kernel native to the unpadded d_v = 128
shape -- against an fp32 torch reference, and times it.  Cross-backend
performance against CK and the v-padded ASM v3 path lives in
op_tests/op_benchmarks/flydsl/bench_mha_bwd.py, which reuses this module's input
builder, reference and roofline helpers.

`out` and `softmax_lse` come from aiter's real varlen forward, so the LSE
convention under test is the one the model actually produces, not a synthesized
one.
"""

import argparse
import itertools
import math
from typing import NamedTuple

import pandas as pd
import pytest
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import is_flydsl_available
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_bwd
from aiter.ops.mha import flash_attn_varlen_func
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

DEVICE = "cuda"
SUPPORTED_GFX = ["gfx942"]

HEAD_DIM_QK = 192
HEAD_DIM_V = 128

# Accuracy gate.  bf16 gradients over multi-thousand-token causal sequences span
# a wide dynamic range with many near-zero elements, so a few elements always
# miss an elementwise 2e-2 isclose -- TOL_ERR_RATIO bounds that fraction.
# mean_abs_diff / mean_abs_ref is the aggregate metric the kernel was tuned on
# and the one that actually moves when the maths is wrong; both are asserted.
RTOL = ATOL = 2e-2
TOL_ERR_RATIO = 0.05
TOL_MEAN_REL = 2e-2

# Ragged sequence-length patterns, each summing to its total token count. `main`
# is the target workload shape (13 ragged sequences over 32K tokens), `uniform`
# an equal-split variant of the same size, `small` a case whose whole grid is
# co-resident on 304 CUs -- that last one is what the kernel's split-K path
# exists for, so it must stay in the sweep.
SEQLEN_CASES = {
    "main": [
        4096,
        3840,
        3584,
        3072,
        2816,
        2560,
        2560,
        2048,
        2048,
        1792,
        1536,
        1536,
        1280,
    ],
    "uniform": [4096] * 8,
    "small": [900, 1200, 1700, 2200, 2192],
}


class VarlenInputs(NamedTuple):
    """One varlen THD batch plus the forward's `out` / `lse`."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    lse: torch.Tensor
    dout: torch.Tensor
    cu_seqlens: torch.Tensor
    seqlens: list[int]
    max_seqlen: int
    scale: float
    d_qk: int
    d_v: int


def make_varlen_inputs(
    seqlens, nheads, dtype, seed=0, d_qk=HEAD_DIM_QK, d_v=HEAD_DIM_V
) -> VarlenInputs:
    """Build the backward's inputs for one THD batch of `seqlens`.

    The real forward runs here, so `out` and `lse` carry aiter's own
    conventions rather than a synthesized LSE the kernel was never handed.

    `d_qk` / `d_v` default to the shape under test; the benchmark overrides them
    to build the square d=128 batch it uses as a reference point.
    """
    total = sum(seqlens)
    max_seqlen = max(seqlens)
    scale = 1.0 / math.sqrt(d_qk)

    torch.manual_seed(seed)
    cu_seqlens = torch.tensor(
        [0] + list(itertools.accumulate(seqlens)), dtype=dtypes.i32, device=DEVICE
    )
    q = torch.randn((total, nheads, d_qk), dtype=dtype, device=DEVICE)
    k = torch.randn((total, nheads, d_qk), dtype=dtype, device=DEVICE)
    v = torch.randn((total, nheads, d_v), dtype=dtype, device=DEVICE)

    with torch.no_grad():
        out, lse = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            softmax_scale=scale,
            causal=True,
            return_lse=True,
        )

    return VarlenInputs(
        q=q,
        k=k,
        v=v,
        out=out,
        lse=lse,
        dout=torch.randn_like(out),
        cu_seqlens=cu_seqlens,
        seqlens=list(seqlens),
        max_seqlen=max_seqlen,
        scale=scale,
        d_qk=d_qk,
        d_v=d_v,
    )


def flops_bytes(
    seqlens, nheads, esz, d_qk=HEAD_DIM_QK, d_v=HEAD_DIM_V
) -> tuple[int, int]:
    """Causal FLOPs and minimum HBM traffic for one backward.

    Only the j <= i half of each sequence's score matrix is computed.  Five
    GEMMs contract over d: S, dQ and dK over d_qk; dP and dV over d_v.
    """
    total = sum(seqlens)
    pairs = sum(n * (n + 1) // 2 for n in seqlens) * nheads
    flops = 2 * pairs * (3 * d_qk + 2 * d_v)
    # q, k, dq, dk at d_qk; v, out, dout, dv at d_v; lse fp32.
    nbytes = total * nheads * (4 * d_qk + 4 * d_v) * esz + total * nheads * 4
    return flops, nbytes


def run_torch(dout, q, k, v, out, lse, cu_seqlens, softmax_scale):
    """fp32 reference for the causal varlen backward.

    Reference only: not timed, not in the table.  Loops (sequence, head) so the
    [n, n] score matrices stay small enough for a 4K max_seqlen.  Uses the
    kernel's own `lse`, since the backward's contract is conditional on the LSE
    it is handed: P = exp(scale*Q@K^T - LSE).
    """
    t, h, dqk = q.shape
    d_v = v.shape[-1]
    dq = torch.zeros((t, h, dqk), dtype=dtypes.fp32, device=q.device)
    dk = torch.zeros((t, h, dqk), dtype=dtypes.fp32, device=q.device)
    dv = torch.zeros((t, h, d_v), dtype=dtypes.fp32, device=q.device)
    for lo, hi in itertools.pairwise(cu_seqlens.tolist()):
        n = hi - lo
        if n == 0:
            continue
        mask = torch.triu(torch.ones(n, n, dtype=torch.bool, device=q.device), 1)
        for hd in range(h):
            qs = q[lo:hi, hd].to(dtypes.fp32)
            ks = k[lo:hi, hd].to(dtypes.fp32)
            vs = v[lo:hi, hd].to(dtypes.fp32)
            dos = dout[lo:hi, hd].to(dtypes.fp32)
            os_ = out[lo:hi, hd].to(dtypes.fp32)
            s = (qs @ ks.transpose(-1, -2)) * softmax_scale
            s = s.masked_fill(mask, float("-inf"))
            p = torch.exp(s - lse[hd, lo:hi].to(dtypes.fp32).unsqueeze(-1))
            dv[lo:hi, hd] = p.transpose(-1, -2) @ dos
            dp = dos @ vs.transpose(-1, -2)
            d = (dos * os_).sum(-1, keepdim=True)
            ds = p * (dp - d)
            dq[lo:hi, hd] = (ds @ ks) * softmax_scale
            dk[lo:hi, hd] = (ds.transpose(-1, -2) @ qs) * softmax_scale
    return dq, dk, dv


def mean_rel(ref, got):
    """mean_abs_diff / mean_abs_ref, the metric the accuracy gate was tuned on.

    Elementwise isclose is a poor fit for attention gradients (wide dynamic
    range, many near-zero elements), so this is reported alongside the
    checkAllclose mismatch ratio.
    """
    ref = ref.to(dtypes.fp32)
    got = got.to(dtypes.fp32)
    return ((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-12)).item()


@benchmark()
def run_fmha_varlen_bwd(case, nheads, dtype):
    x = make_varlen_inputs(SEQLEN_CASES[case], nheads, dtype)
    ref_dq, ref_dk, ref_dv = run_torch(
        x.dout, x.q, x.k, x.v, x.out, x.lse, x.cu_seqlens, x.scale
    )

    # Preallocated: autograd allocates these per backward, so reusing them
    # across the timed repeats keeps the measurement on the kernel rather than
    # on the caching allocator.
    dq = torch.empty_like(x.q)
    dk = torch.empty_like(x.k)
    dv = torch.empty_like(x.v)

    def _flydsl():
        flydsl_flash_attn_varlen_bwd(
            x.dout,
            x.q,
            x.k,
            x.v,
            x.out,
            x.lse,
            dq,
            dk,
            dv,
            x.cu_seqlens,
            x.max_seqlen,
            x.max_seqlen,
            x.scale,
        )
        return dq, dk, dv

    # Only the FlyDSL kernel is under test here; CK and the v-padded ASM v3 path
    # are timed against it in op_benchmarks/flydsl/bench_mha_bwd.py.
    candidates = {"flydsl": _flydsl}

    flops, nbytes = flops_bytes(x.seqlens, nheads, x.q.element_size())

    ret = {
        "gfx": get_gfx(),
        "total_tokens": sum(x.seqlens),
        "max_seqlen": x.max_seqlen,
    }
    for name, fn in candidates.items():
        (got_dq, got_dk, got_dv), us = run_perftest(fn)
        grads = (
            ("q", ref_dq, got_dq),
            ("k", ref_dk, got_dk),
            ("v", ref_dv, got_dv),
        )
        err = max(
            checkAllclose(
                ref.to(dtypes.fp32),
                got.to(dtypes.fp32),
                rtol=RTOL,
                atol=ATOL,
                tol_err_ratio=TOL_ERR_RATIO,
                msg=f"{name}: d{tag}",
            )
            for tag, ref, got in grads
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        for tag, ref, got in grads:
            ret[f"{name} d{tag} mean_rel"] = mean_rel(ref, got)
        ret[f"{name} err"] = err
    return ret


# The (case, nheads) pairs pytest collects.  nheads selects the dispatch regime as much as the
# sequence pattern does: at 2 heads `small` is co-resident on 304 CUs and takes the split-K
# path, while at 16 -- the 128-head MLA shape this kernel targets, at TP=8 -- the same tokens
# produce ~2.2k workgroups and take the untouched nsp == 1 path.  Both must stay covered.
_PYTEST_CASES = [(c, 2) for c in SEQLEN_CASES] + [("small", 16)]


@pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX or not is_flydsl_available(),
    reason="flydsl varlen fmha backward requires flydsl on gfx942",
)
@pytest.mark.parametrize("case, nheads", _PYTEST_CASES)
def test_fmha_varlen_bwd(case, nheads):
    """pytest entry point: the same row the CLI tabulates, gated on the accuracy
    thresholds.  ``checkAllclose`` only raises on a catastrophic delta, so the
    mismatch ratio and the aggregate relative error are asserted here."""
    ret = run_fmha_varlen_bwd(case, nheads, dtypes.bf16)
    where = f"{case}/nheads={nheads}"
    assert ret["flydsl err"] <= TOL_ERR_RATIO, (
        f"{where}: {ret['flydsl err']:.2%} of gradient elements outside "
        f"rtol={RTOL} atol={ATOL}"
    )
    for tag in "qkv":
        rel = ret[f"flydsl d{tag} mean_rel"]
        assert rel <= TOL_MEAN_REL, f"{where}: d{tag} mean_rel {rel:.3e}"


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl varlen fmha backward unsupported on %s; skipping", get_gfx()
        )
        return
    if not is_flydsl_available():
        aiter.logger.warning("flydsl is not installed; skipping")
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        choices=[dtypes.d_dtypes["bf16"]],
        nargs="*",
        default="bf16,",
        metavar="{bf16}",
        help="data type.\ne.g.: -d bf16",
    )
    parser.add_argument(
        "-c",
        "--case",
        type=str,
        nargs="*",
        choices=list(SEQLEN_CASES),
        default=list(SEQLEN_CASES),
        help=f"sequence-length pattern(s) from {list(SEQLEN_CASES)}."
        "\ne.g.: -c main small",
    )
    parser.add_argument(
        "-nh",
        "--nheads",
        type=int,
        nargs="*",
        default=[2],
        help="number of attention heads.\ne.g.: -nh 2 8",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        df = [
            run_fmha_varlen_bwd(case, nheads, dtype)
            for case, nheads in itertools.product(args.case, args.nheads)
        ]
        aiter.logger.info(
            "flydsl varlen fmha backward summary (markdown):\n%s",
            pd.DataFrame(df).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
