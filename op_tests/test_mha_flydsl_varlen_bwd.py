# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL varlen FMHA backward (d_qk=192, d_v=128, causal, bf16, THD) on gfx942.

Validates the FlyDSL backward -- the kernel native to the unpadded d_v = 128
shape -- against an fp32 torch reference.  Cross-backend performance against CK
and the v-padded ASM v3 path lives in
op_tests/op_benchmarks/flydsl/bench_mha_bwd.py.

`out` and `softmax_lse` come from aiter's real varlen forward, so the LSE
convention under test is the one the model actually produces, not a synthesized
one.
"""

import argparse
import itertools
import math

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl import is_flydsl_available
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_bwd
from aiter.ops.mha import flash_attn_varlen_func
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]

HEAD_DIM_QK = 192
HEAD_DIM_V = 128

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


def _mean_rel(ref, got):
    """mean_abs_diff / mean_abs_ref, the metric the accuracy gate was tuned on.

    Elementwise isclose is a poor fit for attention gradients (wide dynamic
    range, many near-zero elements), so this is reported alongside the
    checkAllclose mismatch ratio.
    """
    ref = ref.to(dtypes.fp32)
    got = got.to(dtypes.fp32)
    return ((got - ref).abs().mean() / ref.abs().mean().clamp_min(1e-12)).item()


@benchmark()
def test_fmha_varlen_bwd(case, nheads, dtype):
    seqlens = SEQLEN_CASES[case]
    total = sum(seqlens)
    max_seqlen = max(seqlens)
    scale = 1.0 / math.sqrt(HEAD_DIM_QK)

    torch.manual_seed(0)
    cu_seqlens = torch.tensor(
        [0] + list(itertools.accumulate(seqlens)), dtype=dtypes.i32
    )
    q = torch.randn((total, nheads, HEAD_DIM_QK), dtype=dtype)
    k = torch.randn((total, nheads, HEAD_DIM_QK), dtype=dtype)
    v = torch.randn((total, nheads, HEAD_DIM_V), dtype=dtype)

    # The real forward, so `out` and `lse` carry aiter's own conventions.
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
    dout = torch.randn_like(out)

    ref_dq, ref_dk, ref_dv = run_torch(dout, q, k, v, out, lse, cu_seqlens, scale)

    # Preallocated: autograd allocates these per backward, so reusing them
    # across the timed repeats keeps the measurement on the kernel rather than
    # on the caching allocator.
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)

    def _flydsl():
        flydsl_flash_attn_varlen_bwd(
            dout,
            q,
            k,
            v,
            out,
            lse,
            dq,
            dk,
            dv,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            scale,
        )
        return dq, dk, dv

    (got_dq, got_dk, got_dv), us = run_perftest(_flydsl)

    err = max(
        checkAllclose(
            ref.to(dtypes.fp32),
            got.to(dtypes.fp32),
            rtol=2e-2,
            atol=2e-2,
            msg=f"flydsl: d{tag}",
        )
        for tag, ref, got in (
            ("q", ref_dq, got_dq),
            ("k", ref_dk, got_dk),
            ("v", ref_dv, got_dv),
        )
    )

    # Causal: only the j <= i half of each sequence's score matrix is computed.
    # Five GEMMs contract over d: S and dQ and dK over d_qk, dP and dV over d_v.
    pairs = sum(n * (n + 1) // 2 for n in seqlens) * nheads
    flops = 2 * pairs * (3 * HEAD_DIM_QK + 2 * HEAD_DIM_V)
    esz = q.element_size()
    # q, k, dq, dk at d_qk; v, out, dout, dv at d_v; lse fp32.
    nbytes = (
        total * nheads * (4 * HEAD_DIM_QK + 4 * HEAD_DIM_V) * esz + total * nheads * 4
    )

    return {
        "gfx": get_gfx(),
        "total_tokens": total,
        "max_seqlen": max_seqlen,
        "us": us,
        "TFLOPS": flops / us / 1e6,
        "TB/s": nbytes / us / 1e6,
        "dq mean_rel": _mean_rel(ref_dq, got_dq),
        "dk mean_rel": _mean_rel(ref_dk, got_dk),
        "dv mean_rel": _mean_rel(ref_dv, got_dv),
        "err": err,
    }


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
            test_fmha_varlen_bwd(case, nheads, dtype)
            for case, nheads in itertools.product(args.case, args.nheads)
        ]
        aiter.logger.info(
            "flydsl varlen fmha backward summary (markdown):\n%s",
            pd.DataFrame(df).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
