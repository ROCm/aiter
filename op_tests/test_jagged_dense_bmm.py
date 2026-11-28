# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import contextlib
import functools
import itertools
import os
import tempfile
from unittest import mock

import pandas as pd
import torch

import aiter
from aiter import dtypes
from aiter.jit.core import AITER_CONFIGS
from aiter.jit.utils.chip_info import (
    get_cu_num,
    get_gfx,
    get_gfx_custom_op_core,
    get_gfx_runtime,
    get_num_xcds,
)
from aiter.ops.flydsl import jagged_dense_bmm_dispatch as jdbba_dispatch
from aiter.ops.flydsl.jagged_dense_bmm_dispatch import jagged_dense_bmm_dispatched
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import _LAUNCHERS, BLOCK_N
from aiter.test_common import (
    assertAllclose,
    benchmark,
    run_perftest,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942", "gfx950"]
SEED = 1234
INT32_MAX = 2**31 - 1
_SIDE_GATE_CYCLES = 10**9  # GPU cycles; must outlast the reader stream's copy of C.

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


# One missed group is 1/B of the rows, which the 5% default error ratio passes.
def _check(d, got, dtype, msg):
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], d["N"], dtype)
    return assertAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        tol_err_ratio=0,
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

    candidates = {
        "flydsl": lambda: _launch(d, out, B, Mi, uniform),
    }
    if _HAS_TRITON:
        so64 = d["seq_offsets"].to(torch.int64)
        candidates["triton"] = lambda: triton_jagged_dense_bmm_add_fwd(
            Mi, so64, d["jagged"], d["dense"], d["bias"]
        )[0][:L]

    ret = {"gfx": get_gfx_runtime(), "L": L}
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
        "gfx": get_gfx_runtime(),
        "L": d["L"],
        "flydsl us": us,
        "flydsl TFLOPS": tflops,
        "flydsl TB/s": tbps,
        "flydsl err": _check(d, got, dtype, f"block_k={block_k} uniform"),
    }


@benchmark()
def test_jdbba_config_invalid(B, D, Kout, Mi, arg, value, dtype):
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    out = _nan_out(d["L"], d["N"], dtype)
    try:
        _launch(d, out, B, Mi, True, **{arg: value})
    except ValueError:
        raised = True
    else:
        raised = False
    torch.cuda.synchronize()
    untouched = bool(out.isnan().all())
    assert raised, f"{arg}={value} with K={D} must raise ValueError"
    assert untouched, f"{arg}={value}: rejected call must not write C"
    return {
        "gfx": get_gfx_runtime(),
        "raised ValueError": raised,
        "C untouched": untouched,
    }


def _expect_rejected(launch, out, what, match):
    before = out.clone()
    try:
        launch()
    except ValueError as exc:
        message = str(exc)
    else:
        message = None
    torch.cuda.synchronize(out.device)
    untouched = torch.equal(out.view(torch.int16), before.view(torch.int16))
    assert (
        message and match in message
    ), f"{what} must raise ValueError naming {match!r}, got {message!r}"
    assert untouched, f"{what}: rejected call must not write C"
    return {"C untouched": untouched, "error": message}


@benchmark()
def test_jdbba_limit_invalid(limit, B, D, Kout, rows, max_seq_len, regime, dtype):
    # The call raises before launch, so A, B, and BIAS can stay uninitialized.
    d = {
        "jagged": torch.empty(B * rows, D, dtype=dtype),
        "dense": torch.empty(B, D, Kout, dtype=dtype),
        "bias": torch.empty(B, Kout, dtype=dtype),
        "seq_offsets": torch.arange(0, (B + 1) * rows, rows, dtype=torch.int32),
    }
    out = _nan_out(B * rows, Kout, dtype)
    ret = _expect_rejected(
        lambda: _launch(d, out, B, max_seq_len, regime == "uniform"),
        out,
        f"B={B} K={D} N={Kout} max_seq_len={max_seq_len} {regime}",
        limit,
    )
    return {"gfx": get_gfx_runtime(), **ret}


@benchmark()
def test_jdbba_k0(B, Kout, Mi, regime, dtype):
    d = _build_inputs(B, 0, Kout, Mi, regime, dtype)
    out = _nan_out(d["L"], d["N"], dtype)
    _launch(d, out, B, Mi, regime == "uniform")
    return {"gfx": get_gfx_runtime(), "err": _check(d, out, dtype, f"K=0 {regime}")}


@benchmark()
def test_jdbba_rows_past_c(regime, dtype):
    K = N = 256
    lengths = (200, 150, 100)
    L = 300
    d = _build_inputs(3, K, N, max(lengths), regime, dtype, _offsets(lengths))
    guard_rows = sum(lengths) - L
    pool = torch.full(((L + guard_rows) * N,), 7.0, dtype=dtype)
    out = pool[: L * N].view(L, N)
    jagged = d["jagged"][:L]
    jagged_dense_bmm_dispatched(
        out,
        jagged,
        d["dense"],
        d["bias"],
        d["seq_offsets"],
        3,
        max(lengths),
        uniform_seqlen=regime == "uniform",
    )
    torch.cuda.synchronize()
    guard_changed = int((pool[L * N :] != 7.0).sum())
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N, dtype)
    bad = _bad_count(out, ref[:L])
    assert guard_changed == 0, f"{regime}: {guard_changed} elements past C written"
    assert bad == 0, f"{regime}: {bad} wrong elements in C"
    return {"gfx": get_gfx_runtime(), "guard changed": guard_changed, "bad": bad}


def _offsets(lengths):
    off = [0]
    for n in lengths:
        off.append(off[-1] + n)
    return torch.tensor(off, dtype=torch.int32)


def _bad_count(got, ref):
    got, ref = got.float(), ref.float()
    return int(((got - ref).abs() > 1e-2 + 1e-2 * ref.abs()).sum() + got.isnan().sum())


@benchmark()
def test_jdbba_wide_group(K, N, M, regime, dtype):
    torch.manual_seed(0)
    jagged = torch.randn(M, K, dtype=dtype)
    dense = torch.randn(1, K, N, dtype=dtype)
    bias = torch.randn(1, N, dtype=dtype)
    out = _nan_out(M, N, dtype)
    jagged_dense_bmm_dispatched(
        out,
        jagged,
        dense,
        bias,
        _offsets((M,)),
        1,
        M,
        uniform_seqlen=regime == "uniform",
    )
    torch.cuda.synchronize()
    chunk = 2**26
    rows = max(1, min(M, chunk // max(1, K, N)))
    cols = max(1, min(N, chunk // max(1, K, rows)))
    bad = 0
    for r0 in range(0, M, rows):
        a = jagged[r0 : r0 + rows].float()
        for c0 in range(0, N, cols):
            ref = a @ dense[0, :, c0 : c0 + cols].float() + bias[0, c0 : c0 + cols]
            bad += _bad_count(out[r0 : r0 + rows, c0 : c0 + cols], ref.to(dtype))
    assert bad == 0, f"K={K} N={N} M={M} {regime}: {bad} wrong elements"
    return {"gfx": get_gfx_runtime(), "bad": bad}


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

    out = _nan_out(d["L"], d["N"], dtype)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    # Holds back a launch that ignores `stream`, so the snapshot sees NaN in C.
    torch.cuda._sleep(_SIDE_GATE_CYCLES)
    _launch(d, out, B, Mi, False, stream=side)
    reader = torch.cuda.Stream()
    reader.wait_stream(side)
    with torch.cuda.stream(reader):
        snap = out.clone()
    reader.synchronize()
    err_side = _check(d, snap, dtype, "skew explicit side stream")
    torch.cuda.synchronize()

    out = _nan_out(d["L"], d["N"], dtype)
    _, us = run_perftest(lambda: _launch(d, out, B, Mi, False))
    tflops, tbps = _roofline(d, B, us)
    return {
        "gfx": get_gfx_runtime(),
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
_PINNED_LAUNCHER_KEY = {"xcd_c": 32, "xcd_w": 8, "block_k": 64}
_CSV_COLS = (
    "gfx",
    "cu_num",
    "K",
    "N",
    "tiles_log2",
    "xcd_c",
    "xcd_w",
    "block_k",
    "skew_xcd_c",
    "skew_xcd_w",
    "us",
)


def _clear_tuned_cache():
    jdbba_dispatch._tuned_table.cache_clear()
    AITER_CONFIGS.get_config_file.cache_clear()
    get_gfx.cache_clear()
    get_gfx_custom_op_core.cache_clear()


@contextlib.contextmanager
def _tuned_csv(rows, gpu_archs=None):
    prev = {
        name: os.environ.get(name)
        for name in ("AITER_CONFIG_JAGGED_DENSE_BMM", "GPU_ARCHS")
    }
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "jagged_dense_bmm_tuned.csv")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(",".join(_CSV_COLS) + "\n")
                fh.write(rows)
            os.environ["AITER_CONFIG_JAGGED_DENSE_BMM"] = path
            if gpu_archs is not None:
                os.environ["GPU_ARCHS"] = gpu_archs
            _clear_tuned_cache()
            yield
    finally:
        for name, value in prev.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
        _clear_tuned_cache()


@benchmark()
def test_jdbba_config_selection(
    arch, cu_num, regime, decoy_gpu_archs, xcd_c, xcd_w, expect_c, expect_w
):
    tiles = jdbba_dispatch.tiles_log2(
        n_groups=_SELECT_B, output_n=_SELECT_N, max_seq_len=_SELECT_MI
    )
    rows = (
        f"gfx950,256,{_SELECT_K},{_SELECT_N},{tiles},48,24,64,120,16,1\n"
        f"gfx942,304,{_SELECT_K},{_SELECT_N},{tiles},32,16,64,64,8,1\n"
        f"gfx942,38,{_SELECT_K},{_SELECT_N},{tiles},9,9,64,9,9,1\n"
    )
    with (
        _tuned_csv(rows, gpu_archs=decoy_gpu_archs),
        mock.patch.object(jdbba_dispatch, "get_gfx_runtime", return_value=arch),
        mock.patch.object(jdbba_dispatch, "get_cu_num", return_value=cu_num),
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
    assert (cfg["xcd_c"], cfg["xcd_w"]) == (expect_c, expect_w), (
        f"{arch} cu={cu_num} {regime} selected "
        f"{cfg['xcd_c']},{cfg['xcd_w']} instead of {expect_c},{expect_w}"
    )
    return {"xcd_c": cfg["xcd_c"], "xcd_w": cfg["xcd_w"], "block_k": cfg["block_k"]}


@benchmark()
def test_jdbba_tuned_row(B, D, Kout, Mi, xcd_c, xcd_w, block_k, dropped, dtype):
    tiles = jdbba_dispatch.tiles_log2(n_groups=B, output_n=Kout, max_seq_len=Mi)
    key = (get_gfx_runtime(), get_cu_num(), D, Kout, tiles)
    resolve = functools.partial(
        jdbba_dispatch.resolve_config,
        n_groups=B,
        reduction_k=D,
        output_n=Kout,
        max_seq_len=Mi,
        uniform_seqlen=True,
    )
    with _tuned_csv(""):
        default = resolve()
    row = ",".join(map(str, (*key, xcd_c, xcd_w, block_k, "", "", 1)))
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    out = _nan_out(d["L"], d["N"], dtype)
    with (
        _tuned_csv(row + "\n"),
        mock.patch.object(
            aiter.logger, "warning", wraps=aiter.logger.warning
        ) as warning,
    ):
        cfg = resolve()
        _launch(d, out, B, Mi, True)
    err = _check(d, out, dtype, f"tuned row {row}")
    warned = any(call.args[1:2] == (key,) for call in warning.call_args_list)
    kept = {"xcd_c": xcd_c, "xcd_w": xcd_w, "block_k": block_k}
    expect = default if dropped else kept
    assert cfg == expect, f"tuned row {row}: selected {cfg}, expected {expect}"
    assert warned == dropped, f"tuned row {row}: warned={warned}, dropped={dropped}"
    return {
        "gfx": get_gfx_runtime(),
        **{f"used {k}": v for k, v in cfg.items()},
        "err": err,
    }


def _live_tuned_row(B, D, Kout, Mi, xcd_c, xcd_w, block_k):
    tiles = jdbba_dispatch.tiles_log2(n_groups=B, output_n=Kout, max_seq_len=Mi)
    key = (get_gfx_runtime(), get_cu_num(), D, Kout, tiles)
    return ",".join(map(str, (*key, xcd_c, xcd_w, block_k, "", "", 1)))


@benchmark()
def test_jdbba_tuned_csv_bad_cell(B, D, Kout, Mi, cell, dtype):
    good = _live_tuned_row(B, D, Kout, Mi, 16, 4, 64)
    bad = _live_tuned_row(B, 2 * D, Kout, Mi, 32, 8, 64).split(",")
    col, value = cell.split("=")
    bad[_CSV_COLS.index(col)] = value
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    out = _nan_out(d["L"], d["N"], dtype)
    with (
        _tuned_csv(",".join(bad) + "\n" + good + "\n"),
        mock.patch.object(
            aiter.logger, "warning", wraps=aiter.logger.warning
        ) as warning,
    ):
        cfg = jdbba_dispatch.resolve_config(
            n_groups=B,
            reduction_k=D,
            output_n=Kout,
            max_seq_len=Mi,
            uniform_seqlen=True,
        )
        _launch(d, out, B, Mi, True)
    err = _check(d, out, dtype, f"tuned CSV with a bad row ({cell})")
    drops = sum("dropping tuned row" in c.args[0] for c in warning.call_args_list)
    expect = {"xcd_c": 16, "xcd_w": 4, "block_k": 64}
    assert cfg == expect, f"bad row ({cell}): selected {cfg}, expected {expect}"
    assert drops == 1, f"bad row ({cell}): {drops} drop warnings, expected 1"
    return {"gfx": get_gfx_runtime(), "drop warnings": drops, "err": err}


@benchmark()
def test_jdbba_tuned_row_skew_fallback(B, D, Kout, Mi, dtype):
    row = _live_tuned_row(B, D, Kout, Mi, 16, 4, 64)
    d = _build_inputs(B, D, Kout, Mi, "skew", dtype)
    out = _nan_out(d["L"], d["N"], dtype)
    with _tuned_csv(row + "\n"):
        cfg = jdbba_dispatch.resolve_config(
            n_groups=B,
            reduction_k=D,
            output_n=Kout,
            max_seq_len=Mi,
            uniform_seqlen=False,
        )
        _launch(d, out, B, Mi, False)
    err = _check(d, out, dtype, f"skew fallback, tuned row {row}")
    expect = (
        jdbba_dispatch.SKEW_COMPACT_KN_XCD_C_SMALL,
        jdbba_dispatch.SKEW_COMPACT_KN_XCD_W_SMALL,
    )
    assert (cfg["xcd_c"], cfg["xcd_w"]) == expect, (
        f"tuned row {row} without skew cells: selected "
        f"{cfg['xcd_c']},{cfg['xcd_w']} instead of {expect[0]},{expect[1]}"
    )
    return {
        "gfx": get_gfx_runtime(),
        "used xcd_c": cfg["xcd_c"],
        "used xcd_w": cfg["xcd_w"],
        "err": err,
    }


@benchmark()
def test_jdbba_block_k_override(B, D, Kout, Mi, block_k, dtype):
    row = _live_tuned_row(B, D, Kout, Mi, 16, 4, 64)
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    out = _nan_out(d["L"], d["N"], dtype)
    with _tuned_csv(row + "\n"), mock.patch.dict(_LAUNCHERS, clear=True):
        cfg = jdbba_dispatch.resolve_config(
            n_groups=B,
            reduction_k=D,
            output_n=Kout,
            max_seq_len=Mi,
            uniform_seqlen=True,
            block_k=block_k,
        )
        _launch(d, out, B, Mi, True, block_k=block_k)
        launched = {launcher_key[4] for _, launcher_key in _LAUNCHERS}
    err = _check(d, out, dtype, f"block_k={block_k} override, tuned row {row}")
    expect = {"xcd_c": 16, "xcd_w": 4, "block_k": block_k}
    what = f"block_k={block_k} override"
    assert cfg == expect, f"{what}: selected {cfg}, expected {expect}"
    assert launched == {block_k}, f"{what}: launcher built for {sorted(launched)}"
    return {"gfx": get_gfx_runtime(), "used block_k": cfg["block_k"], "err": err}


@benchmark()
def test_jdbba_dense_2d_invalid(B, D, Kout, Mi, dtype):
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    out = _nan_out(d["L"], d["N"], dtype)
    dense_2d = d["dense"].reshape(B * D, Kout)
    try:
        _launch({**d, "dense": dense_2d}, out, B, Mi, True)
    except ValueError:
        raised = True
    else:
        raised = False
    torch.cuda.synchronize()
    untouched = bool(out.isnan().all())
    assert raised, "2-D dense B must raise ValueError"
    assert untouched, "2-D dense B: rejected call must not write C"
    return {
        "gfx": get_gfx_runtime(),
        "raised ValueError": raised,
        "C untouched": untouched,
    }


@benchmark()
def test_jdbba_stream_device_invalid(B, D, Kout, Mi, dtype):
    n_dev = torch.cuda.device_count()
    if n_dev < 2:
        reason = f"needs 2 visible GPUs, found {n_dev}"
        aiter.logger.warning("jagged_dense_bmm stream device test skipped: %s", reason)
        return {"skipped": reason}
    with torch.cuda.device(1):
        d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
        out = _nan_out(d["L"], d["N"], dtype)
    stream = torch.cuda.Stream(torch.device("cuda", 0))
    ret = _expect_rejected(
        lambda: _launch(d, out, B, Mi, True, stream=stream),
        out,
        "tensors on cuda:1 with a cuda:0 stream",
        "stream is on cuda:0",
    )
    return {"gfx": get_gfx_runtime(), **ret}


@benchmark()
def test_jdbba_output_overlap_invalid(B, D, Kout, Mi, overlaps, dtype):
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    c_bytes = d["L"] * d["N"] * dtype.itemsize
    if overlaps == "A":
        out = d["jagged"]
    else:
        key = {"B": "dense", "BIAS": "bias", "SEQ_OFFSETS": "seq_offsets"}[overlaps]
        src = d[key]
        buf = torch.empty(c_bytes - 4 + src.nbytes, dtype=torch.uint8)
        out = buf[:c_bytes].view(dtype).view(d["L"], d["N"])
        shared = buf[c_bytes - 4 :].view(src.dtype).view(src.shape)
        shared.copy_(src)
        d = {**d, key: shared}
    ret = _expect_rejected(
        lambda: _launch(d, out, B, Mi, True),
        out,
        f"C overlapping {overlaps}",
        f"C overlaps {overlaps}",
    )
    return {"gfx": get_gfx_runtime(), **ret}


@benchmark()
def test_jdbba_launcher_reuse(B, D, Kout, Mi, regime, dtype):
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
    _launch(first, out, B, Mi, uniform, **_PINNED_LAUNCHER_KEY)
    err_prime = _check(first, out, dtype, f"launcher reuse prime ({regime})")
    n_launchers = len(_LAUNCHERS)

    out = _nan_out(second["L"], second["N"], dtype)
    got, us = run_perftest(
        lambda: _launch(second, out, B2, Mi2, uniform, **_PINNED_LAUNCHER_KEY)
    )
    new_launchers = len(_LAUNCHERS) - n_launchers
    assert (
        new_launchers == 0
    ), f"launcher cache grew by {new_launchers} for a new {regime} length"
    err = _check(second, got, dtype, f"launcher reuse ({regime})")
    tflops, tbps = _roofline(second, B2, us)
    return {
        "gfx": get_gfx_runtime(),
        "B2": B2,
        "Mi2": Mi2,
        "L": second["L"],
        "err prime": err_prime,
        "flydsl us": us,
        "flydsl TFLOPS": tflops,
        "flydsl TB/s": tbps,
        "flydsl err": err,
    }


@benchmark()
def test_jdbba_two_devices(B, D, Kout, Mi, dtype):
    n_dev = torch.cuda.device_count()
    if n_dev < 2:
        reason = f"needs 2 visible GPUs, found {n_dev}"
        aiter.logger.warning("jagged_dense_bmm two-device test skipped: %s", reason)
        return {"skipped": reason}
    ret = {"gfx": get_gfx_runtime()}
    for dev in range(2):
        with torch.cuda.device(dev):
            d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
            out = _nan_out(d["L"], d["N"], dtype)
        _launch(d, out, B, Mi, True, **_PINNED_LAUNCHER_KEY)
        with torch.cuda.device(dev):
            ret[f"err cuda:{dev}"] = _check(d, out, dtype, f"two devices cuda:{dev}")
    return ret


@benchmark()
def test_jdbba_size1_dim_strides(B, D, Kout, Mi, odd, dtype):
    first = _build_inputs(1, D, Kout, 1, "uniform", dtype)
    out = _nan_out(1, Kout, dtype)
    if "A" in odd:
        first["jagged"] = first["jagged"].as_strided((1, D), (1, 1))
    if "C" in odd:
        out = out.as_strided((1, Kout), (1, 1))
    assert first["jagged"].is_contiguous() and out.is_contiguous()
    _launch(first, out, 1, 1, True, **_PINNED_LAUNCHER_KEY)
    err_first = _check(first, out, dtype, f"size-1 dim, odd {odd} strides")

    second = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    out = _nan_out(second["L"], second["N"], dtype)
    _launch(second, out, B, Mi, True, **_PINNED_LAUNCHER_KEY)
    err = _check(second, out, dtype, f"canonical strides after odd {odd}")
    return {"gfx": get_gfx_runtime(), "err odd": err_first, "err next": err}


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
    if get_gfx_runtime() not in SUPPORTED_GFX:
        aiter.logger.warning(
            "jagged_dense_bmm unsupported on %s; skipping", get_gfx_runtime()
        )
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
    fields = (
        "arch",
        "cu_num",
        "regime",
        "decoy_gpu_archs",
        "xcd_c",
        "xcd_w",
        "expect_c",
        "expect_w",
    )
    for case in (
        ("gfx950", 256, "uniform", "gfx942", None, None, 48, 24),
        ("gfx942", 304, "uniform", "gfx950", None, None, 32, 16),
        ("gfx950", 256, "skew", "gfx942", None, None, 120, 16),
        ("gfx942", 38, "uniform", "gfx950", None, None, 9, 9),
        ("gfx942", 38, "uniform", "gfx950", 16, 4, 16, 4),
    ):
        rows.append(test_jdbba_config_selection(**dict(zip(fields, case))))
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
        cases = [
            (D, "block_k", 0),
            (D, "block_k", 24),
            (D, "block_k", 3 * D // 8),
            (192, "block_k", 96),
        ]
        cases += [(D, arg, v) for arg in ("xcd_c", "xcd_w") for v in (0, -1)]
        cases += [
            (D, "xcd_c", INT32_MAX // get_num_xcds() + 1),
            (D, "xcd_w", INT32_MAX // (Kout // BLOCK_N) + 1),
        ]
        for K, arg, value in cases:
            rows.append(test_jdbba_config_invalid(B, K, Kout, Mi, arg, value, dtype))
    _summarize("jagged_dense_bmm invalid config", rows)

    rows = []
    for dtype, case in itertools.product(
        args.dtype,
        (
            ("skew path (uniform_seqlen=False)", 65537, 64, 128, 1, 1, "uniform"),
            ("grid x", 1, 64, 128, 5, INT32_MAX, "uniform"),
            ("workgroups", 65536, 64, 128, 1, 2**22, "uniform"),
            ("grid z = N / 128", 1, 64, 65537 * BLOCK_N, 1, 1, "skew"),
            ("BIAS", 2**23, 0, 256, 1, 1, "skew"),
            ("tile rows of A or C", 1, 0, 2**24 + BLOCK_N, 1, 1, "uniform"),
            ("one group of B", 1, 16448, 131072, 1, 1, "uniform"),
            ("N must be >= 1", 4, 64, 0, 5, 5, "uniform"),
        ),
    ):
        rows.append(test_jdbba_limit_invalid(*case, dtype))
    _summarize("jagged_dense_bmm launch limits", rows)

    rows = []
    for dtype, regime in itertools.product(args.dtype, args.regime):
        rows.append(test_jdbba_rows_past_c(regime, dtype))
    _summarize("jagged_dense_bmm rows past C", rows)

    rows = []
    for dtype, regime, (K, N, M) in itertools.product(
        args.dtype,
        args.regime,
        (
            (64, 3 * 2**21, 341),
            (64, 128, 2**24 + 1024),
            (4096, 128, 2**19 + 1024),
        ),
    ):
        rows.append(test_jdbba_wide_group(K, N, M, regime, dtype))
    _summarize("jagged_dense_bmm wide group", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), regime in itertools.product(
        args.dtype, args.regression_shapes, args.regime
    ):
        rows.append(test_jdbba_k0(B, Kout, Mi, regime, dtype))
    _summarize("jagged_dense_bmm K=0", rows)

    rows = []
    for dtype, (B, D, Kout, Mi) in itertools.product(
        args.dtype, args.regression_shapes
    ):
        rows.append(test_jdbba_dense_2d_invalid(B, D, Kout, Mi, dtype))
    _summarize("jagged_dense_bmm 2-D dense", rows)

    rows = []
    for dtype, (B, D, Kout, Mi) in itertools.product(
        args.dtype, args.regression_shapes
    ):
        for K, xcd_c, xcd_w, block_k, dropped in (
            (128, 16, 4, 128, False),
            (D, 0, 8, 64, True),
            (384, 32, 8, 96, True),
            (D, 32, INT32_MAX // (Kout // BLOCK_N) + 1, 64, True),
        ):
            rows.append(
                test_jdbba_tuned_row(
                    B, K, Kout, Mi, xcd_c, xcd_w, block_k, dropped, dtype
                )
            )
    _summarize("jagged_dense_bmm tuned row", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), cell in itertools.product(
        args.dtype, args.regression_shapes, ("xcd_c=abc", "xcd_w=32.0", "K=")
    ):
        rows.append(test_jdbba_tuned_csv_bad_cell(B, D, Kout, Mi, cell, dtype))
    _summarize("jagged_dense_bmm tuned CSV bad cell", rows)

    rows = []
    for dtype, (B, D, Kout, Mi) in itertools.product(
        args.dtype, args.regression_shapes
    ):
        rows.append(test_jdbba_tuned_row_skew_fallback(B, D, Kout, Mi, dtype))
    _summarize("jagged_dense_bmm tuned row skew fallback", rows)

    rows = []
    for dtype, (B, D, Kout, Mi) in itertools.product(
        args.dtype, args.regression_shapes
    ):
        rows.append(test_jdbba_block_k_override(B, D, Kout, Mi, 128, dtype))
    _summarize("jagged_dense_bmm block_k override", rows)

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

    rows = []
    for dtype, (B, D, Kout, Mi), odd in itertools.product(
        args.dtype, args.regression_shapes, ("A", "C", "AC")
    ):
        rows.append(test_jdbba_size1_dim_strides(B, D, Kout, Mi, odd, dtype))
    _summarize("jagged_dense_bmm size-1 dim strides", rows)

    rows = []
    for dtype, (B, D, Kout, Mi) in itertools.product(
        args.dtype, args.regression_shapes
    ):
        rows.append(test_jdbba_two_devices(B, D, Kout, Mi, dtype))
    _summarize("jagged_dense_bmm two devices", rows)

    rows = []
    for dtype, (B, D, Kout, Mi) in itertools.product(
        args.dtype, args.regression_shapes
    ):
        rows.append(test_jdbba_stream_device_invalid(B, D, Kout, Mi, dtype))
    _summarize("jagged_dense_bmm stream device", rows)

    rows = []
    for dtype, (B, D, Kout, Mi), overlaps in itertools.product(
        args.dtype, args.regression_shapes, ("A", "B", "BIAS", "SEQ_OFFSETS")
    ):
        if overlaps == "A" and D != Kout:
            continue
        rows.append(test_jdbba_output_overlap_invalid(B, D, Kout, Mi, overlaps, dtype))
    _summarize("jagged_dense_bmm output overlap", rows)


if __name__ == "__main__":
    main()
