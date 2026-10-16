# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""FlyDSL varlen FMHA backward (causal, bf16, THD) on gfx942.

Validates the FlyDSL backward over its head-dim range -- ``64 <= d_v <= d_qk <=
192``, both multiples of 8; the square shapes plus the MLA-style d_qk=192 /
d_v=128 -- against an fp32 torch reference, in both its fused (atomic dQ) and
deterministic modes, and times it.  Cross-backend
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
from aiter.ops.flydsl.fmha_bwd_gfx942 import _num_cu, _plan
from aiter.ops.flydsl.fmha_kernels import flydsl_flash_attn_varlen_bwd
from aiter.ops.flydsl.kernels.fmha_bwd_gfx942.fmha_bwd_core import (
    D_MAX,
    D_MIN,
    D_STEP,
    _geom,
)
from aiter.ops.mha import flash_attn_varlen_func
from aiter.test_common import benchmark, checkAllclose, run_perftest

torch.set_default_device("cuda")

DEVICE = "cuda"
SUPPORTED_GFX = ["gfx942"]

HEAD_DIM_QK = 192
HEAD_DIM_V = 128

# (d_qk, d_v) pairs the pytest collection covers: every square head dim the backward
# supports (each compiles its own kernel set, so a spill or miscompile at one d is not
# caught by its neighbours), plus the MLA shape.
HEAD_DIMS = [(d, d) for d in range(D_MIN, D_MAX + 1, D_STEP)] + [(192, 128)]

# Accuracy gate.
RTOL = ATOL = 2e-2
TOL_ERR_RATIO = 0.05
TOL_MEAN_REL = 2e-2

# Ragged sequence-length patterns, each summing to its total token count.
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
    """One varlen THD batch plus the forward's out / lse."""

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
def run_fmha_varlen_bwd(
    case, nheads, dtype, d_qk=HEAD_DIM_QK, d_v=HEAD_DIM_V, deterministic=False
):
    x = make_varlen_inputs(SEQLEN_CASES[case], nheads, dtype, d_qk=d_qk, d_v=d_v)
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
            deterministic=deterministic,
        )
        return dq, dk, dv

    # Only the FlyDSL kernel is under test here; CK and the v-padded ASM v3 path
    # are timed against it in op_benchmarks/flydsl/bench_mha_bwd.py.
    candidates = {"flydsl": _flydsl}

    flops, nbytes = flops_bytes(x.seqlens, nheads, x.q.element_size(), d_qk, d_v)

    ret = {
        "gfx": get_gfx(),
        "total_tokens": sum(x.seqlens),
        "max_seqlen": x.max_seqlen,
        "fused": _plan(
            sum(x.seqlens),
            nheads,
            len(x.seqlens),
            x.max_seqlen,
            x.max_seqlen,
            _num_cu(),
            d_qk,
            d_v,
            deterministic,
        )[6],
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
    if deterministic:
        again = [t.clone() for t in (got_dq, got_dk, got_dv)]
        _flydsl()
        ret["bitwise repeatable"] = all(
            torch.equal(a, b) for a, b in zip(again, (dq, dk, dv))
        )
    return ret


# The (case, nheads) pairs pytest collects.  "small" at 2 heads takes the co-resident
# split-K path; "main" at 2 heads and "small" at 16 heads take the n_split == 1 kernel,
# which is the fused single-role one for d_qk <= 128 unless `deterministic`.
_PYTEST_CASES = [
    (c, h, d_qk, d_v, det)
    for c, h in (("small", 2), ("main", 2), ("small", 16))
    for d_qk, d_v in HEAD_DIMS
    for det in (False, True)
]


@pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason="flydsl varlen fmha backward requires gfx942",
)
@pytest.mark.parametrize("case, nheads, d_qk, d_v, deterministic", _PYTEST_CASES)
def test_fmha_varlen_bwd(case, nheads, d_qk, d_v, deterministic):
    """pytest entry point: the same row the CLI tabulates, gated on the accuracy
    thresholds.  ``checkAllclose`` only raises on a catastrophic delta, so the
    mismatch ratio and the aggregate relative error are asserted here."""
    ret = run_fmha_varlen_bwd(case, nheads, dtypes.bf16, d_qk, d_v, deterministic)
    where = f"{case}/nheads={nheads}/d={d_qk}x{d_v}/det={deterministic}"
    assert ret["flydsl err"] <= TOL_ERR_RATIO, (
        f"{where}: {ret['flydsl err']:.2%} of gradient elements outside "
        f"rtol={RTOL} atol={ATOL}"
    )
    for tag in "qkv":
        rel = ret[f"flydsl d{tag} mean_rel"]
        assert rel <= TOL_MEAN_REL, f"{where}: d{tag} mean_rel {rel:.3e}"
    if deterministic:
        assert ret["bitwise repeatable"], f"{where}: two runs differ bitwise"
        assert not ret["fused"], f"{where}: atomic-dQ fused kernel selected"
    elif (case, nheads) == ("small", 16) and _geom(d_qk, d_v).fusable:
        # The case that covers the fused kernel: its grid (est_wgs = 2208) is never
        # co-resident on a gfx942 part, so the planner cannot route it to two-role split-K.
        assert ret["fused"], f"{where}: fused kernel not selected"


# Through the public entry point: `flash_attn_varlen_func` autograd ->
# `can_impl_fmha_bwd_flydsl` in aiter/ops/mha.py.  (case, nheads, d_qk, d_v,
# deterministic, whether the gate is expected to pick FlyDSL.)
_ROUTE_CASES = [
    ("small", 2, 192, 128, False, True),  # no v3 kernel for d_qk != d_v
    ("small", 2, 192, 128, True, True),
    # d_qk == d_v stays on the existing ASM / CK backwards.
    ("small", 2, 96, 96, False, False),
    ("small", 16, 160, 160, False, False),
    ("small", 2, 64, 64, True, False),
    ("uniform", 16, 128, 128, False, False),
]


@pytest.mark.skipif(
    get_gfx() not in SUPPORTED_GFX,
    reason="flydsl varlen fmha backward requires gfx942",
)
@pytest.mark.parametrize(
    "case, nheads, d_qk, d_v, deterministic, expect_flydsl", _ROUTE_CASES
)
def test_varlen_func_routing(
    monkeypatch, case, nheads, d_qk, d_v, deterministic, expect_flydsl
):
    import aiter.ops.flydsl.fmha_kernels as fmha_kernels

    calls = []
    real = fmha_kernels.flydsl_flash_attn_varlen_bwd

    def spy(*args, **kwargs):
        calls.append(kwargs.get("deterministic", args[13] if len(args) > 13 else None))
        return real(*args, **kwargs)

    monkeypatch.setattr(fmha_kernels, "flydsl_flash_attn_varlen_bwd", spy)

    seqlens = SEQLEN_CASES[case]
    max_seqlen = max(seqlens)
    scale = 1.0 / math.sqrt(d_qk)
    torch.manual_seed(0)
    cu_seqlens = torch.tensor(
        [0] + list(itertools.accumulate(seqlens)), dtype=dtypes.i32, device=DEVICE
    )
    total = sum(seqlens)
    q, k = (
        torch.randn((total, nheads, d_qk), dtype=dtypes.bf16).requires_grad_()
        for _ in range(2)
    )
    v = torch.randn((total, nheads, d_v), dtype=dtypes.bf16).requires_grad_()
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
        deterministic=deterministic,
        return_lse=True,
    )
    dout = torch.randn_like(out)
    out.backward(dout)

    where = f"{case}/nheads={nheads}/d={d_qk}x{d_v}/det={deterministic}"
    assert bool(calls) == expect_flydsl, f"{where}: FlyDSL routed={bool(calls)}"
    if not expect_flydsl:
        return
    assert calls == [deterministic], f"{where}: deterministic not forwarded"
    refs = run_torch(
        dout, q.detach(), k.detach(), v.detach(), out.detach(), lse, cu_seqlens, scale
    )
    for tag, ref, got in zip("qkv", refs, (q.grad, k.grad, v.grad)):
        rel = mean_rel(ref, got)
        assert rel <= TOL_MEAN_REL, f"{where}: d{tag} mean_rel {rel:.3e}"


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "flydsl varlen fmha backward unsupported on %s; skipping", get_gfx()
        )
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
    parser.add_argument(
        "-hd",
        "--head_dim",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(HEAD_DIM_QK, HEAD_DIM_V)],
        help="head dims as d (square) or d_qk,d_v.\ne.g.: -hd 64 128 192,128",
    )
    parser.add_argument(
        "-det",
        "--deterministic",
        type=dtypes.str2bool,
        nargs="*",
        default=[False, True],
        help="deterministic (two-role, no-atomic) kernel, default is [False, True]."
        "\ne.g.: -det true",
    )
    args = parser.parse_args()

    head_dims = [
        (hd, hd) if isinstance(hd, int) else (hd[0], hd[-1]) for hd in args.head_dim
    ]

    for dtype in args.dtype:
        df = [
            run_fmha_varlen_bwd(case, nheads, dtype, d_qk, d_v, det)
            for case, nheads, (d_qk, d_v), det in itertools.product(
                args.case, args.nheads, head_dims, args.deterministic
            )
        ]
        aiter.logger.info(
            "flydsl varlen fmha backward summary (markdown):\n%s",
            pd.DataFrame(df).to_markdown(index=False),
        )


if __name__ == "__main__":
    main()
