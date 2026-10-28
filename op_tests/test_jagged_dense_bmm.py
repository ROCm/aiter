# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import itertools

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.ops.flydsl.jagged_dense_bmm_dispatch import jagged_dense_bmm_dispatched
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
SEED = 1234

try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_fwd,
    )

    _HAS_TRITON = True
except ModuleNotFoundError as _exc:  # pragma: no cover
    _HAS_TRITON = False
    _TRITON_ERR = _exc


def _make_seq_offsets(B, Mi, regime, seed=SEED):
    if regime == "uniform":
        return torch.arange(0, (B + 1) * Mi, Mi, dtype=torch.int32, device="cuda")
    g = torch.Generator().manual_seed(seed)
    t = (Mi * (torch.rand(B, generator=g, device="cpu") ** 4)).floor().to(torch.int64)
    t[: max(1, B // 5)] = 0
    t[-1] = Mi
    if B > 1:
        t[-2] = int(0.9 * Mi)
    off = torch.zeros(B + 1, dtype=torch.int32, device="cpu")
    off[1:] = torch.cumsum(t, 0).to(torch.int32)
    return off.cuda()


def run_torch(jagged, dense, bias, seq_offsets, N, dtype=dtypes.bf16):
    L = jagged.shape[0]
    out = torch.zeros((L, N), dtype=dtype, device="cuda")
    so = seq_offsets.cpu().tolist()
    for b in range(dense.shape[0]):
        s, e = so[b], so[b + 1]
        if e > s:
            out[s:e] = (
                jagged[s:e].to(dtypes.fp32) @ dense[b].to(dtypes.fp32)
                + bias[b].to(dtypes.fp32)[None, :]
            ).to(dtype)
    return out


def _build_inputs(B, D, Kout, Mi, regime, dtype, seq_offsets=None):
    N, K = Kout, D
    if seq_offsets is None:
        seq_offsets = _make_seq_offsets(B, Mi, regime)
    L = int(seq_offsets[-1].item())
    torch.manual_seed(0)
    jagged = torch.randn(L, K, dtype=dtype, device="cuda")
    dense = torch.randn(B, K, N, dtype=dtype, device="cuda")
    bias = torch.randn(B, N, dtype=dtype, device="cuda")
    return {
        "jagged": jagged,
        "dense": dense,
        "bias": bias,
        "seq_offsets": seq_offsets,
        "dense_tall": dense.transpose(1, 2).reshape(B * N, K).contiguous(),
        "bias_flat": bias.reshape(B * N).contiguous(),
        "L": L,
        "N": N,
        "K": K,
    }


def _nan_out(L, N, dtype):
    # NaN exposes rows the kernel never writes and stores that land past L.
    return torch.full((L, N), float("nan"), dtype=dtype, device="cuda")


def _launch(d, out, B, Mi, uniform, **kwargs):
    jagged_dense_bmm_dispatched(
        out,
        d["jagged"],
        d["dense_tall"],
        d["bias_flat"],
        d["seq_offsets"],
        B,
        Mi,
        uniform_seqlen=uniform,
        **kwargs,
    )
    return out


def _check(d, got, dtype, msg):
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], d["N"], dtype)
    return checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=msg,
    )


def _roofline(d, B, us):
    L, N, K = d["L"], d["N"], d["K"]
    flops = 2 * L * K * N
    nbytes = (L * K + B * K * N + B * N + L * N) * d["jagged"].element_size()
    return flops / us / 1e6, nbytes / us / 1e6


@benchmark()
def test_jdbba(B, D, Kout, Mi, regime, dtype):
    d = _build_inputs(B, D, Kout, Mi, regime, dtype)
    L, N = d["L"], d["N"]
    uniform = regime == "uniform"
    out = _nan_out(L, N, dtype)

    candidates = {"flydsl": lambda: _launch(d, out, B, Mi, uniform)}
    if _HAS_TRITON:
        so64 = d["seq_offsets"].to(torch.int64)
        candidates["triton"] = lambda: triton_jagged_dense_bmm_add_fwd(
            Mi, so64, d["jagged"], d["dense"], d["bias"]
        )[0][:L]

    ret = {"gfx": get_gfx(), "L": L}
    for name, fn in candidates.items():
        out.fill_(float("nan"))
        got, us = run_perftest(fn)
        tflops, tbps = _roofline(d, B, us)
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = tflops
        ret[f"{name} TB/s"] = tbps
        ret[f"{name} err"] = _check(
            d, got, dtype, f"{name}: jagged_dense_bmm ({regime})"
        )
    return ret


@benchmark()
def test_jdbba_block_k(B, D, Kout, Mi, block_k, dtype):
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    out = _nan_out(d["L"], d["N"], dtype)

    got, us = run_perftest(lambda: _launch(d, out, B, Mi, True, block_k=block_k))
    tflops, tbps = _roofline(d, B, us)
    return {
        "gfx": get_gfx(),
        "L": d["L"],
        "flydsl us": us,
        "flydsl TFLOPS": tflops,
        "flydsl TB/s": tbps,
        "flydsl err": _check(d, got, dtype, f"block_k={block_k} uniform"),
    }


def _permute_group_lengths_(seq_offsets):
    lengths = seq_offsets[1:] - seq_offsets[:-1]
    permuted = torch.zeros_like(seq_offsets)
    permuted[1:] = torch.cumsum(lengths.flip(0), 0).to(seq_offsets.dtype)
    seq_offsets.copy_(permuted)


@benchmark()
def test_jdbba_seq_offsets_reuse(B, D, Kout, Mi, seed, dtype):
    def _skew_run(d, msg, **kwargs):
        out = _nan_out(d["L"], d["N"], dtype)
        _launch(d, out, B, Mi, False, **kwargs)
        return _check(d, out, dtype, msg)

    d = _build_inputs(
        B,
        D,
        Kout,
        Mi,
        "skew",
        dtype,
        seq_offsets=_make_seq_offsets(B, Mi, "skew", seed),
    )
    err_first = _skew_run(d, f"skew first offsets seed={seed}")
    first_L = d["L"]

    d = _build_inputs(
        B,
        D,
        Kout,
        Mi,
        "skew",
        dtype,
        seq_offsets=_make_seq_offsets(B, Mi, "skew", seed + 1),
    )
    assert d["L"] != first_L, "second offsets must have a different L"
    err_new_l = _skew_run(d, f"skew new offsets and L seed={seed + 1}")

    before = d["seq_offsets"].clone()
    _permute_group_lengths_(d["seq_offsets"])
    assert d["seq_offsets"][-1].item() == d["L"], "permutation must keep L"
    assert not torch.equal(before, d["seq_offsets"]), "permutation must change offsets"
    err_inplace = _skew_run(d, "skew in-place offsets update")

    side = torch.cuda.Stream()
    out = _nan_out(d["L"], d["N"], dtype)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        _launch(d, out, B, Mi, False, stream=side)
    torch.cuda.current_stream().wait_stream(side)
    err_side = _check(d, out, dtype, "skew explicit side stream")

    out = _nan_out(d["L"], d["N"], dtype)
    _, us = run_perftest(lambda: _launch(d, out, B, Mi, False))
    tflops, tbps = _roofline(d, B, us)
    return {
        "gfx": get_gfx(),
        "L": d["L"],
        "err new": err_first,
        "err new L": err_new_l,
        "err inplace": err_inplace,
        "err side stream": err_side,
        "flydsl us": us,
        "flydsl TFLOPS": tflops,
        "flydsl TB/s": tbps,
    }


def _summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def _expand_shapes(shapes, batch, parser):
    """Yield (B, D, Kout, Mi). --batch, when given, supplies/overrides B."""
    out = []
    for shape in shapes:
        if not isinstance(shape, tuple):
            shape = (shape,)
        if len(shape) == 4:
            head, rest = (shape[0],), shape[1:]
        elif len(shape) == 3:
            head, rest = (), shape
        else:
            parser.error(f"shape must be B,D,Kout,Mi or D,Kout,Mi; got {shape}")
        if batch:
            out.extend((b, *rest) for b in batch)
        elif head:
            out.append((*head, *rest))
        else:
            parser.error(f"shape {shape} omits B; pass -b/--batch")
    return list(dict.fromkeys(out))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("jagged_dense_bmm unsupported on %s; skipping", get_gfx())
        return
    if not _HAS_TRITON:
        aiter.logger.warning(
            "Triton baseline unavailable (%s); running flydsl-only",
            _TRITON_ERR,
        )

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.bf16],
        help="e.g.: -d bf16",
    )
    parser.add_argument(
        "-b",
        "--batch",
        type=int,
        nargs="*",
        default=None,
        help="B (n_groups). Supplies B for 3-tuple -s, or overrides B in 4-tuple -s.\n"
        "e.g.: -b 120 1024 -s 256,256,7680",
    )
    parser.add_argument(
        "-s",
        "--shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (120, 256, 256, 7680),
            (120, 512, 512, 7680),
            (1024, 256, 256, 7680),
            (1024, 512, 512, 7680),
            (64, 512, 1024, 8192),
        ],
        help="B,D,Kout,Mi or D,Kout,Mi (requires -b).\n"
        "e.g.: -s 64,512,1024,8192  or  -s 256,256,7680",
    )
    parser.add_argument(
        "-r",
        "--regime",
        type=str,
        choices=["uniform", "skew"],
        nargs="*",
        default=["uniform", "skew"],
        help="e.g.: -r uniform",
    )
    parser.add_argument(
        "-k",
        "--block-k",
        type=int,
        nargs="*",
        default=[64, 128],
        help="e.g.: -k 64 128",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=[111, 222],
        help="e.g.: --seeds 111 222",
    )
    parser.add_argument(
        "-rs",
        "--regression-shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[(64, 256, 256, 2048)],
        help="B,D,Kout,Mi for block_k and seq_offsets reuse. e.g.: -rs 64,256,256,2048",
    )
    args = parser.parse_args()
    shapes = _expand_shapes(args.shapes, args.batch, parser)
    for shape in args.regression_shapes:
        if not isinstance(shape, tuple) or len(shape) != 4:
            parser.error(f"regression shape must be B,D,Kout,Mi; got {shape}")

    for dtype in args.dtype:
        rows = []
        for regime, (B, D, Kout, Mi) in itertools.product(args.regime, shapes):
            rows.append(test_jdbba(B, D, Kout, Mi, regime, dtype))
        _summarize("jagged_dense_bmm", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), block_k in itertools.product(
        args.dtype, args.regression_shapes, args.block_k
    ):
        rows.append(test_jdbba_block_k(B, D, Kout, Mi, block_k, dtype))
    _summarize("jagged_dense_bmm block_k", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), seed in itertools.product(
        args.dtype, args.regression_shapes, args.seeds
    ):
        rows.append(test_jdbba_seq_offsets_reuse(B, D, Kout, Mi, seed, dtype))
    _summarize("jagged_dense_bmm seq_offsets reuse", rows)


if __name__ == "__main__":
    main()
