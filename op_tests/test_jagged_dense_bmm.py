# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import argparse
import itertools
from unittest.mock import patch

import aiter
import pandas as pd
import torch
from aiter import dtypes
from aiter.test_common import (
    benchmark,
    checkAllclose,
    run_perftest,
)
from aiter.jit.utils.chip_info import get_gfx

import flydsl.compiler as flyc
from aiter.ops.flydsl.jagged_dense_bmm_dispatch import (
    _load_dispatch_table,
    clear_skew_tile_map_cache,
    jagged_dense_bmm_dispatched,
    resolve_config,
    shape_id,
)
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import BLOCK_M as _BLOCK_M
from aiter.ops.flydsl.kernels.jagged_dense_bmm_gen import (
    _COMPILED_CACHE,
    jagged_dense_bmm,
)

torch.set_default_device("cuda")

SUPPORTED_GFX = ["gfx942"]
SEED = 1234

try:
    from generative_recommenders.ops.triton.triton_jagged import (
        triton_jagged_dense_bmm_add_fwd,
    )

    _HAS_TRITON = True
except Exception as _exc:  # pragma: no cover
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
    jagged = torch.randn(max(L, 1), K, dtype=dtype, device="cuda")
    dense = torch.randn(B, K, N, dtype=dtype, device="cuda")
    bias = torch.randn(B, N, dtype=dtype, device="cuda")

    dense_tall = dense.transpose(1, 2).reshape(B * N, K).contiguous()
    bias_flat = bias.reshape(B * N).contiguous()
    out = torch.zeros(L + _BLOCK_M, N, dtype=dtype, device="cuda")
    tA = flyc.from_dlpack(jagged).mark_layout_dynamic(leading_dim=1, divisibility=8)
    tC = flyc.from_dlpack(out).mark_layout_dynamic(leading_dim=1, divisibility=8)
    return dict(
        jagged=jagged,
        dense=dense,
        bias=bias,
        seq_offsets=seq_offsets,
        dense_tall=dense_tall,
        bias_flat=bias_flat,
        out=out,
        tA=tA,
        tC=tC,
        L=L,
        N=N,
        K=K,
    )


@benchmark()
def test_jdbba(B, D, Kout, Mi, regime, dtype):
    d = _build_inputs(B, D, Kout, Mi, regime, dtype)
    L, N, K = d["L"], d["N"], d["K"]
    uniform = regime == "uniform"
    st = torch.cuda.current_stream()
    if not uniform:
        clear_skew_tile_map_cache()

    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N, dtype)

    def _flydsl():
        jagged_dense_bmm_dispatched(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            B,
            Mi,
            stream=st,
            uniform_seqlen=uniform,
        )
        return d["out"][:L]

    candidates = {"flydsl": _flydsl}
    if _HAS_TRITON:
        so64 = d["seq_offsets"].to(torch.int64)
        candidates["triton"] = lambda: triton_jagged_dense_bmm_add_fwd(
            Mi, so64, d["jagged"], d["dense"], d["bias"]
        )[0][:L]

    flops = 2 * L * K * N
    nbytes = (L * K + B * K * N + B * N + L * N) * d["jagged"].element_size()

    ret = {"gfx": get_gfx(), "L": L}
    for name, fn in candidates.items():
        got, us = run_perftest(fn)
        err = checkAllclose(
            ref.to(dtypes.fp32),
            got.to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"{name}: jagged_dense_bmm ({regime})",
        )
        ret[f"{name} us"] = us
        ret[f"{name} TFLOPS"] = flops / us / 1e6
        ret[f"{name} TB/s"] = nbytes / us / 1e6
        ret[f"{name} err"] = err
    return ret


def test_jdbba_compile_reraise():
    if get_gfx() not in SUPPORTED_GFX:
        return {"gfx": get_gfx(), "skipped": True}
    B, D, Kout, Mi = 8, 64, 64, 128
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtypes.bf16)
    st = torch.cuda.current_stream()
    n_before = len(_COMPILED_CACHE)
    with patch("flydsl.compiler.compile", side_effect=RuntimeError("boom")):
        try:
            jagged_dense_bmm(
                d["tC"],
                d["tA"],
                d["dense_tall"],
                d["bias_flat"],
                d["seq_offsets"],
                n_groups=B,
                max_seq_len=Mi,
                stream=st,
                uniform_seqlen=True,
                block_k=64,
            )
            raise AssertionError("expected RuntimeError from flyc.compile failure")
        except RuntimeError as exc:
            if str(exc) != "boom":
                raise
    cache_clean = len(_COMPILED_CACHE) == n_before
    assert cache_clean, "compile failure must not poison _COMPILED_CACHE"
    return {"gfx": get_gfx(), "reraise_ok": True, "cache_clean": cache_clean}


def test_jdbba_block_k(dtype):
    if get_gfx() not in SUPPORTED_GFX:
        return {"gfx": get_gfx(), "skipped": True}
    B, D, Kout, Mi = 64, 256, 256, 2048
    d = _build_inputs(B, D, Kout, Mi, "uniform", dtype)
    L, N = d["L"], d["N"]
    st = torch.cuda.current_stream()
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N, dtype)

    jagged_dense_bmm(
        d["tC"],
        d["tA"],
        d["dense_tall"],
        d["bias_flat"],
        d["seq_offsets"],
        n_groups=B,
        max_seq_len=Mi,
        stream=st,
        uniform_seqlen=True,
        block_k=128,
    )
    got = d["out"][:L]
    err = checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg="block_k=128 uniform",
    )
    try:
        jagged_dense_bmm(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            n_groups=B,
            max_seq_len=Mi,
            stream=st,
            uniform_seqlen=True,
            block_k=32,
        )
        raise AssertionError("block_k=32 should raise ValueError")
    except ValueError as exc:
        invalid_ok = "block_k" in str(exc)
    assert invalid_ok, "block_k=32 must be rejected before compile"
    try:
        jagged_dense_bmm(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            n_groups=B,
            max_seq_len=Mi,
            stream=st,
            uniform_seqlen=True,
            block_k=256,
        )
        raise AssertionError("block_k=256 should raise ValueError")
    except ValueError as exc:
        lds_ok = "LDS" in str(exc)
    assert lds_ok, "block_k=256 must be rejected for LDS overflow"
    return {
        "gfx": get_gfx(),
        "block_k": 128,
        "err": err,
        "invalid_block_k_ok": invalid_ok,
        "lds_block_k_ok": lds_ok,
    }


def test_jdbba_skew_varying_L(dtype):
    if get_gfx() not in SUPPORTED_GFX:
        return {"gfx": get_gfx(), "skipped": True}
    B, D, Kout, Mi = 64, 256, 256, 2048
    clear_skew_tile_map_cache()
    st = torch.cuda.current_stream()
    errs = []
    for seed in (111, 222):
        d = _build_inputs(
            B,
            D,
            Kout,
            Mi,
            "skew",
            dtype,
            seq_offsets=_make_seq_offsets(B, Mi, "skew", seed=seed),
        )
        L, N = d["L"], d["N"]
        ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N, dtype)
        jagged_dense_bmm_dispatched(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            B,
            Mi,
            stream=st,
            uniform_seqlen=False,
        )
        err = checkAllclose(
            ref.to(dtypes.fp32),
            d["out"][:L].to(dtypes.fp32),
            rtol=1e-2,
            atol=1e-2,
            msg=f"skew varying L seed={seed}",
        )
        errs.append(err)
    return {"gfx": get_gfx(), "err_max": max(errs), "L_seeds": (111, 222)}


@benchmark()
def test_jdbba_dispatch(B, D, Kout, Mi):
    winners = _load_dispatch_table().get("winners") or {}
    sid = shape_id(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=Mi)
    cfg = resolve_config(n_groups=B, reduction_k=D, output_n=Kout, max_seq_len=Mi)
    in_table = sid in winners
    if in_table:
        w = winners[sid]
        routing_ok = cfg["xcd_c"] == w["xcd_c"] and cfg["xcd_w"] == w["xcd_w"]
    else:
        routing_ok = True

    d = _build_inputs(B, D, Kout, Mi, "uniform", dtypes.bf16)
    L, N, K = d["L"], d["N"], d["K"]
    st = torch.cuda.current_stream()
    ref = run_torch(d["jagged"], d["dense"], d["bias"], d["seq_offsets"], N)

    def _flydsl():
        jagged_dense_bmm_dispatched(
            d["tC"],
            d["tA"],
            d["dense_tall"],
            d["bias_flat"],
            d["seq_offsets"],
            B,
            Mi,
            stream=st,
            uniform_seqlen=True,
        )
        return d["out"][:L]

    got, us = run_perftest(_flydsl)
    err = checkAllclose(
        ref.to(dtypes.fp32),
        got.to(dtypes.fp32),
        rtol=1e-2,
        atol=1e-2,
        msg=f"dispatch {sid}",
    )
    flops = 2 * L * K * N
    nbytes = (L * K + B * K * N + B * N + L * N) * d["jagged"].element_size()
    return {
        "gfx": get_gfx(),
        "in_table": in_table,
        "xcd_c": cfg["xcd_c"],
        "xcd_w": cfg["xcd_w"],
        "routing_ok": routing_ok,
        "flydsl us": us,
        "flydsl TFLOPS": flops / us / 1e6,
        "flydsl TB/s": nbytes / us / 1e6,
        "flydsl err": err,
    }


def _summarize(name, rows):
    df = pd.DataFrame(rows)
    aiter.logger.info("%s summary (markdown):\n%s", name, df.to_markdown(index=False))


def main():
    if get_gfx() not in SUPPORTED_GFX:
        aiter.logger.warning("jagged_dense_bmm unsupported on %s; skipping", get_gfx())
        return
    if not _HAS_TRITON:
        aiter.logger.warning(
            "generative-recommenders Triton baseline unavailable (%s); "
            "running flydsl-only",
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
        help="e.g.: -s 64,512,1024,8192",
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
        "-ds",
        "--dispatch-shapes",
        type=dtypes.str2tuple,
        nargs="*",
        default=[
            (120, 256, 256, 7680),
            (120, 512, 512, 7680),
            (1024, 256, 256, 7680),
            (1024, 512, 512, 7680),
            (120, 384, 384, 7680),
        ],
        help="e.g.: -ds 120,384,384,7680",
    )
    args = parser.parse_args()

    for dtype in args.dtype:
        rows = []
        for regime, (B, D, Kout, Mi) in itertools.product(args.regime, args.shapes):
            rows.append(test_jdbba(B, D, Kout, Mi, regime, dtype))
        _summarize("jagged_dense_bmm", rows)

    rows = [
        test_jdbba_dispatch(B, D, Kout, Mi) for (B, D, Kout, Mi) in args.dispatch_shapes
    ]
    _summarize("jagged_dense_bmm dispatch routing", rows)

    _summarize("jagged_dense_bmm compile re-raise", [test_jdbba_compile_reraise()])
    _summarize(
        "jagged_dense_bmm block_k", [test_jdbba_block_k(dtype) for dtype in args.dtype]
    )
    _summarize(
        "jagged_dense_bmm skew varying L",
        [test_jdbba_skew_varying_L(dtype) for dtype in args.dtype],
    )


if __name__ == "__main__":
    main()
