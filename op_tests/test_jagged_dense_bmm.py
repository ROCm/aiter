# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import itertools
import os
import tempfile
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import get_gfx, get_gfx_custom_op_core
from aiter.ops.flydsl import jagged_dense_bmm_dispatch as jdbba_dispatch
from aiter.ops.flydsl.jagged_dense_bmm_dispatch import jagged_dense_bmm_dispatched
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import _LAUNCHERS
from aiter.test_common import (
    assertAllclose,
    benchmark,
    checkAllclose,
    run_perftest,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
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
        d["dense"],
        d["bias"],
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


def _require(d, got, dtype, msg):
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], d["N"], dtype)
    return assertAllclose(ref.to(dtypes.fp32), got.to(dtypes.fp32), msg=msg)


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

    candidates = {
        "flydsl": lambda: _launch(d, out, B, Mi, uniform),
    }
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


@benchmark()
def test_jdbba_block_k_invalid(B, D, Kout, Mi, block_k, dtype):
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    out = _nan_out(d["L"], d["N"], dtype)
    try:
        _launch(d, out, B, Mi, True, block_k=block_k)
    except ValueError:
        raised = True
    else:
        raised = False
    untouched = bool(out.isnan().all())
    assert raised, f"block_k={block_k} with K={D} must raise ValueError"
    assert untouched, f"block_k={block_k}: rejected call must not write C"
    return {"gfx": get_gfx(), "raised ValueError": raised, "C untouched": untouched}


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


_SELECT_B, _SELECT_K, _SELECT_N, _SELECT_MI = 64, 256, 256, 2048
# Fixed so a new length changes the grid without changing the launcher key.
_REUSE_XCD = {"xcd_c": 32, "xcd_w": 8, "block_k": 64}


def _clear_tuned_cache():
    jdbba_dispatch._tuned_table.cache_clear()
    AITER_CONFIGS.get_config_file.cache_clear()
    get_gfx.cache_clear()
    get_gfx_custom_op_core.cache_clear()


def _selection_csv(path, tiles):
    rows = (
        f"gfx950,256,{_SELECT_K},{_SELECT_N},{tiles},48,24,64,120,16,1\n"
        f"gfx942,304,{_SELECT_K},{_SELECT_N},{tiles},32,16,64,64,8,1\n"
        f"gfx942,38,{_SELECT_K},{_SELECT_N},{tiles},9,9,64,9,9,1\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(
            "gfx,cu_num,K,N,tiles_log2,xcd_c,xcd_w,block_k,skew_xcd_c,skew_xcd_w,us\n"
        )
        fh.write(rows)


@benchmark()
def test_jdbba_config_selection(
    arch, cu_num, num_xcds, regime, gpu_archs, xcd_c, xcd_w, expect_c, expect_w
):
    """The live arch owns the tuned row. One XCD with an empty cell is identity.

    ``gpu_archs`` is written to ``GPU_ARCHS`` so a lookup that still calls
    ``get_gfx()`` selects the other arch and fails. A filled cell still wins
    over the one-XCD identity fallback. An explicit override still wins over both.
    """
    tiles = jdbba_dispatch.tiles_log2(
        n_groups=_SELECT_B, output_n=_SELECT_N, max_seq_len=_SELECT_MI
    )
    prev = {
        "AITER_CONFIG_JAGGED_DENSE_BMM": os.environ.get(
            "AITER_CONFIG_JAGGED_DENSE_BMM"
        ),
        "GPU_ARCHS": os.environ.get("GPU_ARCHS"),
        "FLYDSL_JAGGED_DENSE_BMM_ARCH": os.environ.get("FLYDSL_JAGGED_DENSE_BMM_ARCH"),
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "jagged_dense_bmm_tuned.csv")
            _selection_csv(path, tiles)
            os.environ["AITER_CONFIG_JAGGED_DENSE_BMM"] = path
            os.environ["GPU_ARCHS"] = gpu_archs
            # Retired per-op arch override. It must not choose a row.
            os.environ["FLYDSL_JAGGED_DENSE_BMM_ARCH"] = "gfx90a"
            _clear_tuned_cache()
            with (
                mock.patch.object(jdbba_dispatch, "get_gfx_runtime", return_value=arch),
                mock.patch.object(jdbba_dispatch, "get_cu_num", return_value=cu_num),
                mock.patch.object(
                    jdbba_dispatch, "get_num_xcds", return_value=num_xcds
                ),
            ):
                cfg = jdbba_dispatch.resolve_config(
                    n_groups=_SELECT_B,
                    reduction_k=_SELECT_K,
                    output_n=_SELECT_N,
                    max_seq_len=_SELECT_MI,
                    uniform_seqlen=regime == "uniform",
                    xcd_c=xcd_c,
                    xcd_w=xcd_w,
                )
    finally:
        for name, value in prev.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _clear_tuned_cache()
    assert (cfg["xcd_c"], cfg["xcd_w"]) == (expect_c, expect_w), (
        f"{arch} cu={cu_num} xcds={num_xcds} {regime} selected "
        f"{cfg['xcd_c']},{cfg['xcd_w']} instead of {expect_c},{expect_w}"
    )
    return {"xcd_c": cfg["xcd_c"], "xcd_w": cfg["xcd_w"], "block_k": cfg["block_k"]}


@benchmark()
def test_jdbba_launcher_reuse(B, D, Kout, Mi, regime, dtype):
    """A new length or batch reuses the compiled launcher and matches torch.

    The XCD and ``block_k`` overrides hold the launcher key fixed, so a new
    group count or packed length has to come from the launch grid.
    """
    uniform = regime == "uniform"
    if uniform:
        B2, Mi2 = B + 16, Mi + 64
    else:
        B2, Mi2 = B, Mi * 2
    first = _build_inputs(B, D, Kout, Mi, regime, dtype)
    second = _build_inputs(B2, D, Kout, Mi2, regime, dtype)
    if uniform:
        assert (B2, Mi2) != (B, Mi), "uniform reuse pair did not change the grid"
    else:
        assert second["L"] != first["L"], "skew reuse pair did not change L"

    out = _nan_out(first["L"], first["N"], dtype)
    _launch(first, out, B, Mi, uniform, **_REUSE_XCD)
    err_prime = _require(first, out, dtype, f"launcher reuse prime ({regime})")
    n_launchers = len(_LAUNCHERS)

    out = _nan_out(second["L"], second["N"], dtype)
    got, us = run_perftest(lambda: _launch(second, out, B2, Mi2, uniform, **_REUSE_XCD))
    new_launchers = len(_LAUNCHERS) - n_launchers
    assert new_launchers == 0, (
        f"launcher cache grew by {new_launchers} for a new {regime} length"
    )
    err = _require(second, got, dtype, f"launcher reuse ({regime})")
    tflops, tbps = _roofline(second, B2, us)
    return {
        "gfx": get_gfx(),
        "B2": B2,
        "Mi2": Mi2,
        "L": second["L"],
        "err prime": err_prime,
        "flydsl us": us,
        "flydsl TFLOPS": tflops,
        "flydsl TB/s": tbps,
        "flydsl err": err,
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
    rows = []
    # gpu_archs names the other arch. xcd_c/xcd_w None means no explicit override.
    for case in (
        ("gfx950", 256, 8, "uniform", "gfx942", None, None, 48, 24),
        ("gfx942", 304, 8, "uniform", "gfx950", None, None, 32, 16),
        ("gfx950", 256, 8, "skew", "gfx942", None, None, 120, 16),
        ("gfx950", 256, 1, "uniform", "gfx942", None, None, 48, 24),
        ("gfx950", 256, 1, "skew", "gfx942", None, None, 120, 16),
        ("gfx950", 38, 1, "uniform", "gfx942", None, None, 1, 1),
        ("gfx950", 38, 1, "skew", "gfx942", None, None, 1, 1),
        ("gfx942", 38, 1, "uniform", "gfx950", None, None, 9, 9),
        ("gfx950", 38, 1, "uniform", "gfx942", 32, 8, 32, 8),
    ):
        rows.append(test_jdbba_config_selection(*case))
    _summarize("jagged_dense_bmm config selection", rows)
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
    for dtype, (B, D, Kout, Mi) in itertools.product(
        args.dtype, args.regression_shapes
    ):
        for block_k in (0, 24, 3 * D // 8):
            rows.append(test_jdbba_block_k_invalid(B, D, Kout, Mi, block_k, dtype))
    _summarize("jagged_dense_bmm invalid block_k", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), seed in itertools.product(
        args.dtype, args.regression_shapes, args.seeds
    ):
        rows.append(test_jdbba_seq_offsets_reuse(B, D, Kout, Mi, seed, dtype))
    _summarize("jagged_dense_bmm seq_offsets reuse", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), regime in itertools.product(
        args.dtype, args.regression_shapes, args.regime
    ):
        rows.append(test_jdbba_launcher_reuse(B, D, Kout, Mi, regime, dtype))
    _summarize("jagged_dense_bmm launcher reuse", rows)


if __name__ == "__main__":
    main()
